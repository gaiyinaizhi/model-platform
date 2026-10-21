"""Versioned durable running-cancellation intent; startup never migrates data."""
import sqlite3

from .run_schema import normalize

VERSION = 'running-training-cancel/v1'
ATTEMPT_TABLE = """CREATE TABLE {name} (
 id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(id), number INTEGER NOT NULL,
 status TEXT NOT NULL CHECK(status IN ('running','succeeded','failed','cancelled')),
 started_at REAL NOT NULL, finished_at REAL, error_code TEXT, UNIQUE(run_id,number)
)"""
LEGACY_ATTEMPT_TABLE = ATTEMPT_TABLE.replace(",'cancelled'", '')
SCHEMA = """
CREATE TABLE IF NOT EXISTS training_cancellations (
 attempt_id TEXT PRIMARY KEY REFERENCES attempts(id), run_id TEXT NOT NULL UNIQUE REFERENCES runs(id),
 requested_at REAL NOT NULL, confirmed_at REAL, confirmation_kind TEXT,
 CHECK ((confirmed_at IS NULL AND confirmation_kind IS NULL) OR
        (confirmed_at IS NOT NULL AND confirmation_kind IS NOT NULL AND
         confirmation_kind IN ('child_reaped','never_started','lease_recovered')))
);
CREATE INDEX IF NOT EXISTS training_cancellations_pending ON training_cancellations(confirmed_at,requested_at);
CREATE TRIGGER IF NOT EXISTS training_cancellations_insert_consistent BEFORE INSERT ON training_cancellations
WHEN NEW.confirmed_at IS NOT NULL OR NEW.confirmation_kind IS NOT NULL OR EXISTS (
 SELECT 1 FROM training_cancellations WHERE attempt_id=NEW.attempt_id OR run_id=NEW.run_id) OR NOT EXISTS (
 SELECT 1 FROM attempts a JOIN runs r ON r.id=a.run_id WHERE a.id=NEW.attempt_id AND r.id=NEW.run_id)
BEGIN SELECT RAISE(ABORT,'cancellation must bind its run and attempt as a pending request'); END;
CREATE TRIGGER IF NOT EXISTS training_cancellations_identity_immutable
BEFORE UPDATE OF attempt_id,run_id,requested_at ON training_cancellations
BEGIN SELECT RAISE(ABORT,'cancellation identity and request are immutable'); END;
CREATE TRIGGER IF NOT EXISTS training_cancellations_delete_immutable BEFORE DELETE ON training_cancellations
BEGIN SELECT RAISE(ABORT,'cancellations are immutable'); END;
CREATE TRIGGER IF NOT EXISTS training_cancellations_confirm_once
BEFORE UPDATE OF confirmed_at,confirmation_kind ON training_cancellations
WHEN OLD.confirmed_at IS NOT NULL OR OLD.confirmation_kind IS NOT NULL OR
 NEW.confirmed_at IS NULL OR NEW.confirmation_kind IS NULL
BEGIN SELECT RAISE(ABORT,'cancellation confirmation is one-way'); END;
"""
MIGRATION_MESSAGE = (
    'Running training cancellation migration required. Stop API, Worker and serving processes; '
    'complete python -m model_platform.migrate_queued_cancel first if needed, then run '
    'python -m model_platform.migrate_running_cancel with the existing MP_HOME.'
)


def definition(db):
    row = db.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='attempts'").fetchone()
    return row[0] if row else None


def has_marker(db):
    if not db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='schema_migrations'").fetchone():
        return False
    try:
        return bool(db.execute('SELECT 1 FROM schema_migrations WHERE version=?', (VERSION,)).fetchone())
    except sqlite3.DatabaseError as exc:
        raise RuntimeError('Unexpected schema migration marker table; no changes made') from exc


def required_objects():
    """Canonical definitions, including implicit unique indexes."""
    with sqlite3.connect(':memory:') as reference:
        reference.executescript(ATTEMPT_TABLE.format(name='attempts') + ';' + SCHEMA)
        return {(row[0], row[1]): row[2] for row in reference.execute(
            'SELECT type,name,sql FROM sqlite_master')}


def verify(db, *, legacy=False):
    if db.execute("SELECT 1 FROM sqlite_master WHERE name='attempts_running_cancel_new'").fetchone():
        raise RuntimeError('Partial running cancellation temporary schema; no changes made')
    expected = LEGACY_ATTEMPT_TABLE if legacy else ATTEMPT_TABLE
    if normalize(definition(db) or '') != normalize(expected.format(name='attempts')):
        raise RuntimeError('Unexpected running cancellation attempt schema; no changes made')
    if has_marker(db) == legacy:
        raise RuntimeError('Partial running cancellation migration marker; no changes made')
    objects = required_objects()
    if legacy:
        if db.execute("SELECT 1 FROM sqlite_master WHERE tbl_name='training_cancellations' "
                      "OR name LIKE 'training_cancellations%'").fetchone():
            raise RuntimeError('Partial running cancellation schema objects; no changes made')
        objects = {key: value for key, value in objects.items()
                   if key[1].startswith('sqlite_autoindex_attempts_')}
    for (kind, name), sql in objects.items():
        row = db.execute('SELECT sql FROM sqlite_master WHERE type=? AND name=?', (kind, name)).fetchone()
        if not row or (sql is None and row[0] is not None) or (sql is not None and
                normalize(row[0] or '') != normalize(sql)):
            raise RuntimeError('Unexpected running cancellation schema object: ' + name + '; no changes made')


def startup_check(db):
    """Only an empty schema may be initialized. Verify before any startup write."""
    if definition(db) is None:
        if db.execute("SELECT 1 FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'").fetchone():
            raise RuntimeError('Incomplete platform schema; no changes made')
        return True
    from .run_schema import verify as verify_queue
    verify_queue(db)
    if normalize(definition(db)) == normalize(LEGACY_ATTEMPT_TABLE.format(name='attempts')):
        verify(db, legacy=True)
        raise RuntimeError(MIGRATION_MESSAGE)
    verify(db)
    return False
