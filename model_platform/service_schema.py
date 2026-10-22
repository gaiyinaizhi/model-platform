"""Exact additive stable-service schema; opening an old runtime never migrates it."""
from functools import lru_cache
import sqlite3

from .run_schema import normalize

VERSION = 'model-services/v1'
TABLES = ('model_services', 'model_service_revisions', 'model_service_keys')
SCHEMA = """
CREATE TABLE IF NOT EXISTS model_services (
 id TEXT PRIMARY KEY NOT NULL, tenant TEXT NOT NULL,
 revision INTEGER NOT NULL CHECK(typeof(revision)='integer' AND revision BETWEEN 1 AND 9007199254740991),
 active_deployment_id TEXT NOT NULL REFERENCES deployments(id),
 previous_deployment_id TEXT REFERENCES deployments(id), identity_json TEXT NOT NULL,
 created_at REAL NOT NULL, updated_at REAL NOT NULL,
 UNIQUE(tenant,id), CHECK(previous_deployment_id IS NULL OR previous_deployment_id<>active_deployment_id)
);
CREATE TABLE IF NOT EXISTS model_service_revisions (
 service_id TEXT NOT NULL, revision INTEGER NOT NULL CHECK(typeof(revision)='integer' AND revision BETWEEN 1 AND 9007199254740991),
 tenant TEXT NOT NULL, snapshot_json TEXT NOT NULL, created_at REAL NOT NULL,
 PRIMARY KEY(service_id,revision), UNIQUE(tenant,service_id,revision),
 FOREIGN KEY(tenant,service_id) REFERENCES model_services(tenant,id)
);
CREATE TABLE IF NOT EXISTS model_service_keys (
 tenant TEXT NOT NULL, idempotency_key TEXT NOT NULL CHECK(length(idempotency_key) BETWEEN 1 AND 128),
 request_sha256 TEXT NOT NULL, service_id TEXT NOT NULL, revision INTEGER NOT NULL,
 created_at REAL NOT NULL, PRIMARY KEY(tenant,idempotency_key),
 FOREIGN KEY(tenant,service_id,revision) REFERENCES model_service_revisions(tenant,service_id,revision)
);
CREATE INDEX IF NOT EXISTS model_services_history ON model_services(tenant,created_at DESC,id DESC);
CREATE TRIGGER IF NOT EXISTS model_services_insert_consistent BEFORE INSERT ON model_services
WHEN NEW.revision<>1 OR NEW.previous_deployment_id IS NOT NULL OR NEW.updated_at<>NEW.created_at
 OR EXISTS(SELECT 1 FROM model_services WHERE id=NEW.id)
 OR NOT EXISTS(SELECT 1 FROM deployments WHERE id=NEW.active_deployment_id AND tenant=NEW.tenant)
BEGIN SELECT RAISE(ABORT,'service must start at revision one with an owned deployment'); END;
CREATE TRIGGER IF NOT EXISTS model_services_identity_immutable
BEFORE UPDATE OF id,tenant,identity_json,created_at ON model_services
BEGIN SELECT RAISE(ABORT,'service identity is immutable'); END;
CREATE TRIGGER IF NOT EXISTS model_services_transition BEFORE UPDATE ON model_services
WHEN NEW.revision<>OLD.revision+1 OR NEW.previous_deployment_id IS NOT OLD.active_deployment_id
 OR NEW.active_deployment_id=OLD.active_deployment_id OR NEW.updated_at<OLD.updated_at
 OR NOT EXISTS(SELECT 1 FROM deployments WHERE id=NEW.active_deployment_id AND tenant=NEW.tenant)
 OR NOT EXISTS(SELECT 1 FROM model_service_revisions WHERE service_id=OLD.id AND revision=OLD.revision AND tenant=OLD.tenant)
BEGIN SELECT RAISE(ABORT,'service switch requires the next revision and an owned deployment'); END;
CREATE TRIGGER IF NOT EXISTS model_services_delete_immutable BEFORE DELETE ON model_services
BEGIN SELECT RAISE(ABORT,'services cannot be deleted'); END;
CREATE TRIGGER IF NOT EXISTS model_service_revisions_insert_consistent BEFORE INSERT ON model_service_revisions
WHEN EXISTS(SELECT 1 FROM model_service_revisions WHERE service_id=NEW.service_id AND revision=NEW.revision)
 OR NOT EXISTS(SELECT 1 FROM model_services WHERE id=NEW.service_id AND tenant=NEW.tenant
 AND revision=NEW.revision AND updated_at=NEW.created_at)
BEGIN SELECT RAISE(ABORT,'revision must capture the current owned service'); END;
CREATE TRIGGER IF NOT EXISTS model_service_revisions_update_immutable BEFORE UPDATE ON model_service_revisions
BEGIN SELECT RAISE(ABORT,'service revisions are immutable'); END;
CREATE TRIGGER IF NOT EXISTS model_service_revisions_delete_immutable BEFORE DELETE ON model_service_revisions
BEGIN SELECT RAISE(ABORT,'service revisions are immutable'); END;
CREATE TRIGGER IF NOT EXISTS model_service_keys_insert_consistent BEFORE INSERT ON model_service_keys
WHEN EXISTS(SELECT 1 FROM model_service_keys WHERE tenant=NEW.tenant AND idempotency_key=NEW.idempotency_key)
 OR NOT EXISTS(SELECT 1 FROM model_service_revisions WHERE tenant=NEW.tenant AND service_id=NEW.service_id
 AND revision=NEW.revision AND created_at=NEW.created_at)
BEGIN SELECT RAISE(ABORT,'service key must bind an owned immutable revision'); END;
CREATE TRIGGER IF NOT EXISTS model_service_keys_update_immutable BEFORE UPDATE ON model_service_keys
BEGIN SELECT RAISE(ABORT,'service keys are immutable'); END;
CREATE TRIGGER IF NOT EXISTS model_service_keys_delete_immutable BEFORE DELETE ON model_service_keys
BEGIN SELECT RAISE(ABORT,'service keys are immutable'); END;
"""
MIGRATION_MESSAGE = ('Model services migration required. Stop API, Worker and serving processes; '
                     'run python -m model_platform.migrate_services with the existing MP_HOME.')


@lru_cache(maxsize=1)
def required_objects():
    with sqlite3.connect(':memory:') as reference:
        reference.executescript(SCHEMA)
        return {(row[0], row[1]): row[2] for row in reference.execute('SELECT type,name,sql FROM sqlite_master')}


@lru_cache(maxsize=1)
def predecessor_objects():
    # Lazy import avoids the Store -> schema startup-check import cycle.
    from .store import SCHEMA as PLATFORM_SCHEMA
    from .model_evaluations import SCHEMA as EVALUATION_SCHEMA
    with sqlite3.connect(':memory:') as reference:
        reference.executescript(PLATFORM_SCHEMA + EVALUATION_SCHEMA)
        reference.execute('CREATE TABLE IF NOT EXISTS schema_migrations(version TEXT PRIMARY KEY)')
        for table in ('models', 'evaluations'):
            for action in ('UPDATE', 'DELETE'):
                reference.execute(f"CREATE TRIGGER IF NOT EXISTS {table}_{action.lower()}_immutable "
                                  f"BEFORE {action} ON {table} BEGIN "
                                  "SELECT RAISE(ABORT,'technical records are immutable'); END")
        return {(kind, name): sql for kind, name, table, sql in reference.execute(
            'SELECT type,name,tbl_name,sql FROM sqlite_master') if table not in TABLES}


def _compare(db, objects):
    actual = {(r[0], r[1]): r[2] for r in db.execute('SELECT type,name,sql FROM sqlite_master')}
    for key, sql in objects.items():
        if key not in actual or ((sql is None) != (actual[key] is None)) or (
                sql is not None and normalize(actual[key]) != normalize(sql)):
            raise RuntimeError('Unexpected model services schema object: ' + key[1] + '; no changes made')


def integrity(db):
    if db.execute('PRAGMA foreign_key_check').fetchall():
        raise RuntimeError('Model services foreign key check failed')
    if [row[0] for row in db.execute('PRAGMA integrity_check')] != ['ok']:
        raise RuntimeError('Model services database integrity check failed')


def verify_predecessor(db):
    from .run_schema import verify as verify_queue
    from .running_cancel_schema import verify as verify_cancel
    verify_queue(db)
    verify_cancel(db)
    _compare(db, predecessor_objects())
    if not db.execute("SELECT 1 FROM schema_migrations WHERE version='imported-origin/v1'").fetchone():
        raise RuntimeError('Incomplete imported-origin migration marker')


def has_marker(db):
    if not db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='schema_migrations'").fetchone():
        return False
    try:
        return bool(db.execute('SELECT 1 FROM schema_migrations WHERE version=?', (VERSION,)).fetchone())
    except sqlite3.DatabaseError as exc:
        raise RuntimeError('Unexpected schema migration marker table; no changes made') from exc


def verify(db, *, legacy=False):
    verify_predecessor(db)
    if has_marker(db) == legacy:
        raise RuntimeError('Partial model services migration marker; no changes made')
    if legacy:
        if db.execute("SELECT 1 FROM sqlite_master WHERE name GLOB 'model_service*' OR tbl_name GLOB 'model_service*'").fetchone():
            raise RuntimeError('Partial model services schema objects; no changes made')
    else:
        _compare(db, required_objects())
    integrity(db)


def startup_check(db):
    if not db.execute("SELECT 1 FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'").fetchone():
        return True
    if not has_marker(db):
        verify(db, legacy=True)
        raise RuntimeError(MIGRATION_MESSAGE)
    verify(db)
    return False
