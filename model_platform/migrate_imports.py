"""Offline backed-up migration: python -m model_platform.migrate_imports."""
import contextlib
import os
import sqlite3
from .common import Runtime
from .import_schema import IMPORT_SCHEMA, MODEL_TABLE

VERSION = 'imported-origin/v1'


def _normalize(sql):
    return ''.join(sql.split()).replace('IFNOTEXISTS', '').replace('"', '')


def _verify_current(db):
    marker = db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='schema_migrations'").fetchone()
    if not marker or not db.execute('SELECT 1 FROM schema_migrations WHERE version=?', (VERSION,)).fetchone():
        raise RuntimeError('Partial imported-origin schema; migration marker missing')
    expected = {'id','tenant','run_id','attempt_id','evaluation_id','artifact_sha256','manifest_json','created_at','origin','import_id','validation_id','catalog_revision'}
    if {r['name'] for r in db.execute('PRAGMA table_info(models)')} != expected:
        raise RuntimeError('Unexpected imported-origin schema columns')
    sql = db.execute("SELECT sql FROM sqlite_master WHERE name='models'").fetchone()[0]
    if _normalize(sql) != _normalize(MODEL_TABLE.format(name='models')):
        raise RuntimeError('Unexpected imported-origin schema constraints')
    reference = sqlite3.connect(':memory:')
    try:
        reference.executescript(IMPORT_SCHEMA)
        for name, definition in reference.execute('SELECT name,sql FROM sqlite_master WHERE sql IS NOT NULL'):
            actual = db.execute('SELECT sql FROM sqlite_master WHERE name=?', (name,)).fetchone()
            if not actual or _normalize(actual[0]) != _normalize(definition):
                raise RuntimeError('Unexpected import schema object: ' + name)
    finally:
        reference.close()
    for action in ('UPDATE', 'DELETE'):
        name = 'models_' + action.lower() + '_immutable'
        actual = db.execute('SELECT sql FROM sqlite_master WHERE name=?', (name,)).fetchone()
        expected = f"CREATE TRIGGER {name} BEFORE {action} ON models BEGIN SELECT RAISE(ABORT,'technical records are immutable'); END"
        if not actual or _normalize(actual[0]) != _normalize(expected):
            raise RuntimeError('Unexpected model immutability trigger')
    if db.execute('PRAGMA foreign_key_check').fetchall():
        raise RuntimeError('Invalid imported-origin foreign keys')


def migrate(runtime, fault=None):
    with contextlib.ExitStack() as stack:
        for name in ('api', 'worker', 'model-import'):
            stack.enter_context(runtime.lock(name))
        with runtime.connection() as db:
            if 'origin' in {r['name'] for r in db.execute('PRAGMA table_info(models)')}:
                _verify_current(db)
                return False
            for row in db.execute('SELECT pid FROM deployments WHERE pid IS NOT NULL'):
                try:
                    os.kill(row['pid'], 0)  # Liveness check only; never terminate a stored PID.
                except ProcessLookupError:
                    continue
                raise RuntimeError('Stop all serving processes before migration')
            backup = runtime.home / 'before-imported-origin.sqlite3'
            # Do not overwrite an old backup or follow a pre-existing symlink.
            fd = os.open(backup, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            os.close(fd)
            with sqlite3.connect(backup) as dest:
                db.backup(dest)
            db.execute('PRAGMA foreign_keys=OFF')
            db.execute('BEGIN IMMEDIATE')
            try:
                tables = [r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT IN ('models','sqlite_sequence')")]
                rows_before = {name: [tuple(r) for r in db.execute('SELECT * FROM "' + name.replace('"','""') + '"')] for name in tables}
                schema_before = {(r['type'],r['name']):r['sql'] for r in db.execute("SELECT type,name,sql FROM sqlite_master WHERE tbl_name<>'models' AND sql IS NOT NULL")}
                before = [tuple(r) for r in db.execute('SELECT rowid,* FROM models ORDER BY rowid')]
                objects = [r['sql'] for r in db.execute("SELECT sql FROM sqlite_master WHERE tbl_name='models' AND type IN ('index','trigger') AND sql IS NOT NULL")]
                statement = ''
                for line in IMPORT_SCHEMA.splitlines(True):
                    statement += line
                    if sqlite3.complete_statement(statement):
                        db.execute(statement)
                        statement = ''
                db.execute(MODEL_TABLE.format(name='models_new'))
                db.execute('INSERT INTO models_new(rowid,id,tenant,run_id,attempt_id,evaluation_id,artifact_sha256,manifest_json,created_at) SELECT rowid,id,tenant,run_id,attempt_id,evaluation_id,artifact_sha256,manifest_json,created_at FROM models')
                db.execute('DROP TABLE models')
                db.execute('ALTER TABLE models_new RENAME TO models')
                for sql in objects:
                    db.execute(sql)
                after = [tuple(r) for r in db.execute('SELECT rowid,id,tenant,run_id,attempt_id,evaluation_id,artifact_sha256,manifest_json,created_at FROM models ORDER BY rowid')]
                if before != after or db.execute('PRAGMA foreign_key_check').fetchall():
                    raise RuntimeError('Migration integrity check failed')
                for name, rows in rows_before.items():
                    if rows != [tuple(r) for r in db.execute('SELECT * FROM "' + name.replace('"','""') + '"')]:
                        raise RuntimeError('Migration changed unrelated table: ' + name)
                for (kind, name), definition in schema_before.items():
                    current = db.execute('SELECT sql FROM sqlite_master WHERE type=? AND name=?', (kind,name)).fetchone()
                    if not current or current[0] != definition:
                        raise RuntimeError('Migration changed unrelated schema: ' + name)
                if fault:
                    fault(db)
                db.execute('CREATE TABLE IF NOT EXISTS schema_migrations(version TEXT PRIMARY KEY)')
                db.execute('INSERT INTO schema_migrations VALUES(?)', (VERSION,))
                _verify_current(db)
                db.commit()
            except BaseException:
                db.rollback()
                raise
            finally:
                db.execute('PRAGMA foreign_keys=ON')
            return True


if __name__ == '__main__':
    print('migrated' if migrate(Runtime()) else 'already current')
