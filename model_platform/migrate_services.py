"""Explicit offline backed-up stable-service migration. Never kills PIDs."""
import contextlib
import os
import shutil
import sqlite3
import stat
import tempfile

from .common import Runtime
from .service_schema import SCHEMA, VERSION, TABLES, has_marker, verify

BACKUP_NAME = 'before-model-services.sqlite3'


def _snapshot(db):
    rows = {}
    for (name,) in db.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"):
        quoted = '"' + name.replace('"', '""') + '"'
        try:
            rows[name] = [tuple(row) for row in db.execute(f'SELECT rowid,* FROM {quoted} ORDER BY rowid')]
        except sqlite3.OperationalError as exc:
            # Preserve unrelated WITHOUT ROWID tables as well as ordinary rowids.
            if 'no such column: rowid' not in str(exc):
                raise
            columns = list(db.execute(f'PRAGMA table_info({quoted})'))
            order = ','.join(str(i + 1) for i in range(len(columns)))
            rows[name] = [tuple(row) for row in db.execute(f'SELECT * FROM {quoted} ORDER BY {order}')]
    return rows


def _integrity(db):
    if db.execute('PRAGMA foreign_key_check').fetchall():
        raise RuntimeError('Model services foreign key check failed')
    if [row[0] for row in db.execute('PRAGMA integrity_check')] != ['ok']:
        raise RuntimeError('Model services database integrity check failed')


def _serving_absent(db):
    for (pid,) in db.execute('SELECT pid FROM deployments WHERE pid IS NOT NULL'):
        if type(pid) is not int or pid <= 0:
            raise RuntimeError('Ambiguous serving process record; stop services before migration')
        try:
            os.kill(pid, 0)  # Liveness probe only. Never signal a stored PID to stop it.
        except ProcessLookupError:
            continue
        except (PermissionError, OverflowError):
            raise RuntimeError('Cannot establish serving process absence; stop services before migration') from None
        raise RuntimeError('Stop all serving processes before migration')


def _backup(runtime):
    """Write only through the exclusively created final fd, never reopen its path."""
    backup = runtime.home / BACKUP_NAME
    fd = os.open(backup, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        original = os.fstat(fd)
        # A private staging directory lets SQLite use its ordinary backup API;
        # copying via the held fd avoids a path-reopen race at the final backup.
        with tempfile.TemporaryDirectory(prefix='.model-services-backup-', dir=runtime.home) as stage:
            temporary = os.path.join(stage, 'snapshot.sqlite3')
            with runtime.connection() as source, sqlite3.connect(temporary) as target:
                source.backup(target)
                _integrity(target)
            with open(temporary, 'rb') as source, os.fdopen(os.dup(fd), 'wb') as target:
                shutil.copyfileobj(source, target)
                target.flush()
                os.fsync(target.fileno())
        current = os.stat(backup, follow_symlinks=False)
        if (not stat.S_ISREG(current.st_mode) or
                (current.st_dev, current.st_ino) != (original.st_dev, original.st_ino)):
            raise RuntimeError('Backup identity changed; no migration performed')
    finally:
        os.close(fd)


def _execute_schema(db):
    # executescript commits an existing transaction; execute complete statements.
    statement = ''
    for line in SCHEMA.splitlines(True):
        statement += line
        if sqlite3.complete_statement(statement):
            db.execute(statement)
            statement = ''
    if statement.strip():
        raise RuntimeError('Incomplete model services schema statement')


def migrate(runtime, fault=None):
    """Add service objects to a complete predecessor, with an exact safe backup."""
    with contextlib.ExitStack() as stack:
        for name in ('api', 'worker', 'model-import'):
            stack.enter_context(runtime.lock(name))
        with runtime.connection() as db:
            current = has_marker(db)
            verify(db, legacy=not current)
            _serving_absent(db)
            if current:
                return False
            db.execute('BEGIN IMMEDIATE')
            try:
                verify(db, legacy=True)
                before = _snapshot(db)
                schema_before = {(r[0], r[1]): r[2] for r in db.execute(
                    'SELECT type,name,sql FROM sqlite_master')}
                _backup(runtime)
                _execute_schema(db)
                if fault:
                    fault(db)
                after = _snapshot(db)
                if set(after) != set(before) | set(TABLES) or any(after[t] for t in TABLES) or any(
                        before[name] != after[name] for name in before):
                    raise RuntimeError('Migration changed existing rows')
                for (kind, name), sql in schema_before.items():
                    row = db.execute('SELECT sql FROM sqlite_master WHERE type=? AND name=?', (kind, name)).fetchone()
                    if not row or row[0] != sql:
                        raise RuntimeError('Migration changed existing schema: ' + name)
                db.execute('INSERT INTO schema_migrations VALUES(?)', (VERSION,))
                verify(db)
                db.commit()
            except BaseException:
                db.rollback()
                raise
            return True


if __name__ == '__main__':
    print('migrated' if migrate(Runtime()) else 'already current')
