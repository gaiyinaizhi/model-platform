"""Versioned training queue schema; startup checks never migrate existing data."""

VERSION = 'queued-training-cancel/v1'
RUN_TABLE = """CREATE TABLE {name} (
 id TEXT PRIMARY KEY, tenant TEXT NOT NULL, idempotency_key TEXT NOT NULL,
 request_sha256 TEXT NOT NULL, input_json TEXT NOT NULL, environment_json TEXT NOT NULL,
 status TEXT NOT NULL CHECK(status IN ('queued','running','succeeded','failed','cancelled')),
 created_at REAL NOT NULL, updated_at REAL NOT NULL, model_version_id TEXT,
 UNIQUE(tenant,idempotency_key)
)"""
LEGACY_RUN_TABLE = RUN_TABLE.replace(",'cancelled'", '')
RUN_INDEX = 'CREATE INDEX runs_history ON runs(tenant,created_at DESC,id DESC)'
RUN_TRIGGER = """CREATE TRIGGER runs_input_immutable BEFORE UPDATE OF tenant,idempotency_key,request_sha256,input_json,environment_json ON runs
BEGIN SELECT RAISE(ABORT,'run inputs are immutable'); END"""
MIGRATION_MESSAGE = (
    'Training queue migration required. Stop API, Worker and serving processes; '
    'complete imported-origin migration if needed, then run '
    'python -m model_platform.migrate_queued_cancel with the existing MP_HOME.'
)


def normalize(sql):
    return ''.join(sql.split()).replace('IFNOTEXISTS', '').replace('"', '')


def definition(db):
    row = db.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='runs'").fetchone()
    return row[0] if row else None


def has_marker(db):
    if not db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='schema_migrations'").fetchone():
        return False
    return bool(db.execute('SELECT 1 FROM schema_migrations WHERE version=?', (VERSION,)).fetchone())


def verify(db, *, legacy=False):
    expected = LEGACY_RUN_TABLE if legacy else RUN_TABLE
    if normalize(definition(db) or '') != normalize(expected.format(name='runs')):
        raise RuntimeError('Unexpected training queue schema; no changes made')
    for name, sql in (('runs_history', RUN_INDEX), ('runs_input_immutable', RUN_TRIGGER)):
        row = db.execute('SELECT sql FROM sqlite_master WHERE name=?', (name,)).fetchone()
        if not row or normalize(row[0]) != normalize(sql):
            raise RuntimeError('Unexpected training queue schema object; no changes made')
    if has_marker(db) == legacy:
        raise RuntimeError('Partial training queue migration marker; no changes made')


def startup_check(db):
    """Return True only for an empty schema. Existing old/partial DBs fail closed."""
    sql = definition(db)
    if sql is None:
        if db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'").fetchone():
            raise RuntimeError('Incomplete platform schema; no changes made')
        return True
    if normalize(sql) == normalize(LEGACY_RUN_TABLE.format(name='runs')):
        verify(db, legacy=True)
        raise RuntimeError(MIGRATION_MESSAGE)
    verify(db)
    return False
