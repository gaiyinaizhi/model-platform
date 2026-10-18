"""Explicit imported-origin schema; old databases require an offline migration."""
IMPORT_SCHEMA = """
CREATE TABLE IF NOT EXISTS model_imports (
 id TEXT PRIMARY KEY, tenant TEXT NOT NULL, idempotency_key TEXT NOT NULL,
 request_sha256 TEXT NOT NULL, catalog_id TEXT NOT NULL, catalog_revision TEXT NOT NULL,
 rights_confirmed INTEGER NOT NULL CHECK(rights_confirmed=1), created_at REAL NOT NULL,
 status TEXT NOT NULL CHECK(status IN ('created','validating','validated','rejected')),
 artifact_sha256 TEXT, artifact_id TEXT UNIQUE, validation_id TEXT, reason TEXT,
 UNIQUE(tenant,idempotency_key), UNIQUE(id,tenant,catalog_revision,artifact_sha256)
);
CREATE TABLE IF NOT EXISTS import_validations (
 id TEXT PRIMARY KEY, import_id TEXT NOT NULL UNIQUE, tenant TEXT NOT NULL,
 catalog_revision TEXT NOT NULL, artifact_sha256 TEXT NOT NULL, report_json TEXT NOT NULL,
 created_at REAL NOT NULL,
 UNIQUE(id,import_id,tenant,catalog_revision,artifact_sha256),
 FOREIGN KEY(import_id,tenant,catalog_revision,artifact_sha256)
 REFERENCES model_imports(id,tenant,catalog_revision,artifact_sha256)
);
CREATE TRIGGER IF NOT EXISTS model_import_identity_immutable
 BEFORE UPDATE OF id,tenant,idempotency_key,request_sha256,catalog_id,catalog_revision,rights_confirmed,created_at ON model_imports
 BEGIN SELECT RAISE(ABORT,'import identity is immutable'); END;
CREATE TRIGGER IF NOT EXISTS model_import_binding_immutable
 BEFORE UPDATE OF artifact_sha256,artifact_id ON model_imports WHEN OLD.artifact_sha256 IS NOT NULL
 BEGIN SELECT RAISE(ABORT,'import artifact binding is immutable'); END;
CREATE TRIGGER IF NOT EXISTS model_import_transition BEFORE UPDATE OF status ON model_imports
 WHEN NOT ((OLD.status='created' AND NEW.status='validating') OR
           (OLD.status='validating' AND NEW.status IN ('validated','rejected')))
 BEGIN SELECT RAISE(ABORT,'invalid import transition'); END;
CREATE TRIGGER IF NOT EXISTS model_import_terminal_immutable BEFORE UPDATE ON model_imports
 WHEN OLD.status IN ('validated','rejected')
 BEGIN SELECT RAISE(ABORT,'terminal import is immutable'); END;
CREATE TRIGGER IF NOT EXISTS model_import_delete_immutable BEFORE DELETE ON model_imports
 BEGIN SELECT RAISE(ABORT,'imports are immutable'); END;
CREATE TRIGGER IF NOT EXISTS import_validation_update_immutable BEFORE UPDATE ON import_validations
 BEGIN SELECT RAISE(ABORT,'validations are immutable'); END;
CREATE TRIGGER IF NOT EXISTS import_validation_delete_immutable BEFORE DELETE ON import_validations
 BEGIN SELECT RAISE(ABORT,'validations are immutable'); END;
"""
MODEL_TABLE = """CREATE TABLE {name} (
 id TEXT PRIMARY KEY, tenant TEXT NOT NULL, run_id TEXT UNIQUE REFERENCES runs(id),
 attempt_id TEXT UNIQUE REFERENCES attempts(id), evaluation_id TEXT UNIQUE REFERENCES evaluations(id),
 artifact_sha256 TEXT NOT NULL, manifest_json TEXT NOT NULL, created_at REAL NOT NULL,
 origin TEXT NOT NULL DEFAULT 'trained' CHECK(origin IN ('trained','imported')),
 import_id TEXT UNIQUE REFERENCES model_imports(id), validation_id TEXT UNIQUE REFERENCES import_validations(id),
 catalog_revision TEXT,
 CHECK((origin='trained' AND run_id IS NOT NULL AND attempt_id IS NOT NULL AND evaluation_id IS NOT NULL
        AND import_id IS NULL AND validation_id IS NULL AND catalog_revision IS NULL)
    OR (origin='imported' AND run_id IS NULL AND attempt_id IS NULL AND evaluation_id IS NULL
        AND import_id IS NOT NULL AND validation_id IS NOT NULL AND catalog_revision IS NOT NULL)),
 FOREIGN KEY(validation_id,import_id,tenant,catalog_revision,artifact_sha256)
 REFERENCES import_validations(id,import_id,tenant,catalog_revision,artifact_sha256)
)"""
