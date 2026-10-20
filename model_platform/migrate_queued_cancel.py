"""Explicit offline backed-up queue migration. Never interrupts running processes."""
import contextlib
import os
import sqlite3
import stat

from .common import Runtime
from .run_schema import (RUN_TABLE, VERSION, definition,
                         normalize, verify)


def _snapshot(db):
    tables = [r[0] for r in db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
    rows = {}
    for name in tables:
        quoted = '"' + name.replace('"', '""') + '"'
        rows[name] = [tuple(r) for r in db.execute(f'SELECT rowid,* FROM {quoted} ORDER BY rowid')]
    return rows


def _integrity(db):
    if db.execute('PRAGMA foreign_key_check').fetchall():
        raise RuntimeError('Training queue foreign key check failed')
    if [r[0] for r in db.execute('PRAGMA integrity_check')] != ['ok']:
        raise RuntimeError('Training queue database integrity check failed')


def migrate(runtime, fault=None):
    """Only supported imported-origin DBs; a failed transaction keeps its backup."""
    with contextlib.ExitStack() as stack:
        for name in ('api', 'worker', 'model-import'):
            stack.enter_context(runtime.lock(name))
        with runtime.connection() as db:
            # This also preserves actionable guidance for pre-import databases.
            from .migrate_imports import _verify_current
            if 'origin' not in {r['name'] for r in db.execute('PRAGMA table_info(models)')}:
                raise RuntimeError('Complete python -m model_platform.migrate_imports first, with services stopped')
            _verify_current(db)
            if normalize(definition(db) or '') == normalize(RUN_TABLE.format(name='runs')):
                verify(db)
                _integrity(db)
                return False
            verify(db, legacy=True)
            _integrity(db)
            for row in db.execute('SELECT pid FROM deployments WHERE pid IS NOT NULL'):
                pid = row['pid']
                if type(pid) is not int or pid <= 0:
                    raise RuntimeError('Ambiguous serving process record; stop services before migration')
                try:
                    os.kill(pid, 0)  # Liveness probe only. Never terminate a stored PID.
                except ProcessLookupError:
                    continue
                except (PermissionError, OverflowError):
                    raise RuntimeError('Cannot establish serving process absence; stop services before migration') from None
                raise RuntimeError('Stop all serving processes before migration')
            backup = runtime.home / 'before-queued-training-cancel.sqlite3'
            fd = os.open(backup, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            try:
                original = os.fstat(fd)
                with sqlite3.connect(backup) as dest:
                    current = os.stat(backup, follow_symlinks=False)
                    if (not stat.S_ISREG(current.st_mode)
                            or (current.st_dev, current.st_ino) != (original.st_dev, original.st_ino)):
                        raise RuntimeError('Backup identity changed; no migration performed')
                    db.backup(dest)
            finally:
                os.close(fd)
            db.execute('PRAGMA foreign_keys=OFF')
            db.execute('BEGIN IMMEDIATE')
            try:
                before = _snapshot(db)
                schema_before = {(r['type'], r['name']): r['sql'] for r in db.execute(
                    "SELECT type,name,sql FROM sqlite_master WHERE tbl_name<>'runs' AND sql IS NOT NULL")}
                objects = [r[0] for r in db.execute(
                    "SELECT sql FROM sqlite_master WHERE tbl_name='runs' AND type IN ('index','trigger') AND sql IS NOT NULL")]
                db.execute(RUN_TABLE.format(name='runs_new'))
                db.execute('INSERT INTO runs_new(rowid,id,tenant,idempotency_key,request_sha256,input_json,environment_json,status,created_at,updated_at,model_version_id) '
                           'SELECT rowid,id,tenant,idempotency_key,request_sha256,input_json,environment_json,status,created_at,updated_at,model_version_id FROM runs')
                db.execute('DROP TABLE runs')
                db.execute('ALTER TABLE runs_new RENAME TO runs')
                for sql in objects:
                    db.execute(sql)
                if before != _snapshot(db):
                    raise RuntimeError('Migration changed existing rows')
                for (kind, name), sql in schema_before.items():
                    row = db.execute('SELECT sql FROM sqlite_master WHERE type=? AND name=?', (kind, name)).fetchone()
                    if not row or row[0] != sql:
                        raise RuntimeError('Migration changed unrelated schema')
                if fault:
                    fault(db)
                db.execute('INSERT INTO schema_migrations VALUES(?)', (VERSION,))
                verify(db)
                _integrity(db)
                db.commit()
            except BaseException:
                db.rollback()
                raise
            finally:
                db.execute('PRAGMA foreign_keys=ON')
            return True


if __name__ == '__main__':
    print('migrated' if migrate(Runtime()) else 'already current')
