"""SQLite is the sole writer authority for technical records and queue state."""
import time
import json
import hashlib
import uuid

from .pagination import cursor_key, pack_cursor, unpack_cursor

from .common import (MAX_ACTIVE_RUNS, MAX_ATTEMPTS, MAX_DEPLOYMENTS, Problem,
                     RECIPE, digest, encode, environment, now)

from .import_schema import IMPORT_SCHEMA, MODEL_TABLE
from .run_schema import RUN_TABLE, VERSION as QUEUE_SCHEMA_VERSION, startup_check, verify as verify_queue_schema

from .running_cancel_schema import (ATTEMPT_TABLE, SCHEMA as CANCEL_SCHEMA,
    VERSION as CANCEL_SCHEMA_VERSION, startup_check as cancel_startup_check, verify as verify_cancel_schema)

from .service_schema import (SCHEMA as SERVICE_SCHEMA, VERSION as SERVICE_SCHEMA_VERSION,
    startup_check as service_startup_check, verify as verify_service_schema)

SCHEMA = IMPORT_SCHEMA + """
PRAGMA journal_mode=WAL;
""" + RUN_TABLE.format(name="IF NOT EXISTS runs") + ";\n" + """
CREATE TABLE IF NOT EXISTS run_dataset_bindings (
 run_id TEXT PRIMARY KEY REFERENCES runs(id), dataset_version_id TEXT NOT NULL,
 content_sha256 TEXT NOT NULL, binding_json TEXT NOT NULL
);
CREATE TRIGGER IF NOT EXISTS run_dataset_binding_update_immutable BEFORE UPDATE ON run_dataset_bindings
BEGIN SELECT RAISE(ABORT,'run dataset bindings are immutable'); END;
CREATE TRIGGER IF NOT EXISTS run_dataset_binding_delete_immutable BEFORE DELETE ON run_dataset_bindings
BEGIN SELECT RAISE(ABORT,'run dataset bindings are immutable'); END;
""" + ATTEMPT_TABLE.format(name="IF NOT EXISTS attempts") + ";\n" + CANCEL_SCHEMA + "\n" + """
CREATE TABLE IF NOT EXISTS logs (
 id INTEGER PRIMARY KEY AUTOINCREMENT, attempt_id TEXT NOT NULL REFERENCES attempts(id),
 time REAL NOT NULL, event TEXT NOT NULL, details TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS evaluations (
 id TEXT PRIMARY KEY, attempt_id TEXT NOT NULL UNIQUE REFERENCES attempts(id),
 artifact_sha256 TEXT NOT NULL, evaluation_sha256 TEXT NOT NULL, result_json TEXT NOT NULL,
 created_at REAL NOT NULL
);
""" + MODEL_TABLE.format(name="IF NOT EXISTS models") + ";\n" + """
CREATE TABLE IF NOT EXISTS deployments (
 id TEXT PRIMARY KEY, tenant TEXT NOT NULL, model_version_id TEXT NOT NULL REFERENCES models(id),
 desired TEXT NOT NULL CHECK(desired IN ('running','stopped')),
 observed TEXT NOT NULL CHECK(observed IN ('starting','ready','unavailable','failed','stopped')),
 checked_at REAL, error_code TEXT, pid INTEGER, created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS deployment_keys (
 tenant TEXT NOT NULL, idempotency_key TEXT NOT NULL, request_sha256 TEXT NOT NULL,
 deployment_id TEXT NOT NULL REFERENCES deployments(id), PRIMARY KEY(tenant,idempotency_key)
);
CREATE INDEX IF NOT EXISTS runs_history ON runs(tenant,created_at DESC,id DESC);
CREATE INDEX IF NOT EXISTS models_history ON models(tenant,created_at DESC,id DESC);
CREATE INDEX IF NOT EXISTS deployments_history ON deployments(tenant,created_at DESC,id DESC);
CREATE TABLE IF NOT EXISTS audit (
 id INTEGER PRIMARY KEY AUTOINCREMENT, time REAL NOT NULL, tenant TEXT NOT NULL,
 action TEXT NOT NULL, object_id TEXT NOT NULL
);
CREATE TRIGGER IF NOT EXISTS runs_input_immutable BEFORE UPDATE OF tenant,idempotency_key,request_sha256,input_json,environment_json ON runs
BEGIN SELECT RAISE(ABORT,'run inputs are immutable'); END;
CREATE TRIGGER IF NOT EXISTS deployment_version_immutable BEFORE UPDATE OF tenant,model_version_id ON deployments
BEGIN SELECT RAISE(ABORT,'deployment identity is immutable'); END;
"""


def identifier(prefix):
    return prefix + "_" + uuid.uuid4().hex


class Store:
    def __init__(self, runtime):
        self.runtime = runtime
        # Fail before cursor material or schema scripts can mutate an old/partial DB.
        with runtime.connection() as db:
            fresh_runs = startup_check(db)
            cancel_startup_check(db)
            service_startup_check(db)
        self.cursor_key = cursor_key(runtime.home)
        with runtime.connection() as db:
            fresh = not db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='models'").fetchone()
            from .model_evaluations import SCHEMA as EVALUATION_SCHEMA
            db.executescript(SCHEMA + EVALUATION_SCHEMA + SERVICE_SCHEMA)
            if fresh:
                db.execute('CREATE TABLE IF NOT EXISTS schema_migrations(version TEXT PRIMARY KEY)')
                db.execute("INSERT OR IGNORE INTO schema_migrations VALUES('imported-origin/v1')")
            if fresh_runs:
                db.execute('INSERT OR IGNORE INTO schema_migrations VALUES(?)', (QUEUE_SCHEMA_VERSION,))
                db.execute('INSERT OR IGNORE INTO schema_migrations VALUES(?)', (CANCEL_SCHEMA_VERSION,))
                db.execute('INSERT OR IGNORE INTO schema_migrations VALUES(?)', (SERVICE_SCHEMA_VERSION,))
            for table in ("models", "evaluations"):
                for action in ("UPDATE", "DELETE"):
                    db.execute(f"CREATE TRIGGER IF NOT EXISTS {table}_{action.lower()}_immutable "
                               f"BEFORE {action} ON {table} BEGIN "
                               "SELECT RAISE(ABORT,'technical records are immutable'); END")
            verify_queue_schema(db)
            verify_cancel_schema(db)
            verify_service_schema(db)

    @staticmethod
    def audit(db, tenant, action, object_id):
        db.execute("INSERT INTO audit(time,tenant,action,object_id) VALUES(?,?,?,?)",
                   (now(), tenant, action, object_id))

    def submit(self, tenant, key, body):
        request_hash = digest(body)
        from . import dataset_detection
        datasets = None
        if body.get('recipe') == dataset_detection.RECIPE:
            from .datasets import DatasetStore
            datasets = DatasetStore(self.runtime)
        with self.runtime.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            old = db.execute("SELECT * FROM runs WHERE tenant=? AND idempotency_key=?", (tenant, key)).fetchone()
            if old:
                if old["request_sha256"] != request_hash:
                    raise Problem(409, "idempotency_conflict", "Key already bound to different input")
                return old["id"], False
            binding = dataset_detection.bind_version(datasets, tenant, body['dataset']['dataset_version_id']) if datasets else None
            from .model_evaluations import active_count
            count = active_count(db)
            if count >= MAX_ACTIVE_RUNS:
                raise Problem(429, "run_capacity", "Active run budget reached")
            run_id = identifier("run")
            stamp = now()
            db.execute("INSERT INTO runs VALUES(?,?,?,?,?,?,?,?,?,NULL)",
                       (run_id, tenant, key, request_hash, encode(body).decode(),
                        encode(environment()).decode(), "queued", stamp, stamp))
            if binding:
                db.execute("INSERT INTO run_dataset_bindings VALUES(?,?,?,?)",
                           (run_id, binding['dataset_version_id'], binding['sha256'], encode(binding).decode()))
            self.audit(db, tenant, "submit", run_id)
            return run_id, True

    def run(self, tenant, run_id):
        with self.runtime.connection() as db:
            db.execute("BEGIN")
            return self._run(db, tenant, run_id)

    def _run(self, db, tenant, run_id):
        row = self._owned(db, "runs", tenant, run_id)
        position = None
        if row['status'] == 'queued':
            position = db.execute(
                "SELECT count(*) FROM runs WHERE tenant=? AND status='queued' "
                "AND (created_at < ? OR (created_at = ? AND id <= ?))",
                (tenant, row['created_at'], row['created_at'], run_id)).fetchone()[0]
        body = json.loads(row["input_json"])
        result = {k: row[k] for k in ("id", "status", "created_at", "updated_at", "request_sha256", "model_version_id")}
        from .recipes import get_recipe
        from .dataset_detection import RECIPE as DATASET_RECIPE
        binding = db.execute("SELECT binding_json FROM run_dataset_bindings WHERE run_id=?", (run_id,)).fetchone()
        if body.get('recipe') == DATASET_RECIPE:
            if not binding:
                raise Problem(409, 'dataset_binding_missing', 'Run dataset binding is unavailable')
            dataset = json.loads(binding['binding_json'])
        else:
            train_count, evaluation_count = get_recipe(body.get("recipe", RECIPE)).counts(body["dataset"])
            dataset = {"source": body["dataset"]["source"], "version": body["dataset"]["version"],
                       "sha256": digest(body["dataset"]), "train_count": train_count, "evaluation_count": evaluation_count}
        result["queue"] = {"scope": "tenant", "position": position}
        result.update(recipe=body.get("recipe", RECIPE), environment=json.loads(row["environment_json"]),
                      dataset=dataset, config=body["config"],
                      business_ref=body.get("business_ref"), attempts=[])
        for attempt in db.execute("SELECT * FROM attempts WHERE run_id=? ORDER BY number", (run_id,)):
            item = dict(attempt)
            item["logs"] = [dict(log) for log in db.execute(
                "SELECT time,event,details FROM logs WHERE attempt_id=? ORDER BY id", (attempt["id"],))]
            for log in item["logs"]:
                log["details"] = json.loads(log["details"])
            result["attempts"].append(item)
        cancellation = db.execute("SELECT attempt_id,requested_at,confirmed_at FROM training_cancellations WHERE run_id=?", (run_id,)).fetchone()
        result['cancellation'] = (dict(cancellation, status='confirmed' if cancellation['confirmed_at'] is not None else 'requested') if cancellation else None)
        return result

    def dataset_binding(self, run, verify=True):
        from .dataset_detection import RECIPE as DATASET_RECIPE, bind_version
        body = json.loads(run['input_json'])
        if body.get('recipe') != DATASET_RECIPE:
            return None
        with self.runtime.connection() as db:
            # Read the tenant from the actual persisted run, never a request field.
            owned = self._owned(db, 'runs', run['tenant'], run['id'])
            row = db.execute("SELECT * FROM run_dataset_bindings WHERE run_id=?", (owned['id'],)).fetchone()
        if not row or row['dataset_version_id'] != body['dataset']['dataset_version_id']:
            raise Problem(409, 'dataset_binding_missing', 'Run dataset binding is unavailable')
        binding = json.loads(row['binding_json'])
        if binding['sha256'] != row['content_sha256']:
            raise Problem(409, 'dataset_integrity', 'Run dataset binding has changed')
        if verify:
            from .datasets import DatasetStore
            current = bind_version(DatasetStore(self.runtime), owned['tenant'], row['dataset_version_id'])
            if current != binding:
                raise Problem(409, 'dataset_integrity', 'Bound dataset content has changed')
        return binding

    def compare_models(self, tenant, model_ids):
        from .dataset_detection import RECIPE as DATASET_RECIPE, compatibility
        if (not isinstance(model_ids, list) or len(model_ids) != 2 or
                any(not isinstance(x, str) for x in model_ids) or model_ids[0] == model_ids[1]):
            raise Problem(400, 'invalid_input', 'Choose two distinct model version IDs')
        models = [self.model(tenant, item) for item in model_ids]
        if any(model.get('recipe') != DATASET_RECIPE for model in models):
            raise Problem(409, 'evaluation_incompatible', 'Cannot compare evaluations: protocol')
        evaluations = [model['evaluation'] for model in models]
        signatures = [compatibility(e) for e in evaluations]
        differences = [key for key in ('dataset', 'classes', 'protocol', 'split') if signatures[0][key] != signatures[1][key]]
        if differences:
            raise Problem(409, 'evaluation_incompatible', 'Cannot compare evaluations: ' + ', '.join(differences))
        metrics = [dict(model_version_id=mid, **{k: e[k] for k in ('map_50', 'map_50_95', 'per_class')})
                   for mid, e in zip(model_ids, evaluations)]
        deltas = {key: (evaluations[1][key] - evaluations[0][key]
                       if all(type(e[key]) in (int, float) for e in evaluations) else None)
                  for key in ('map_50', 'map_50_95')}
        return {'compatible': True, 'compatibility_key': digest(signatures[0]),
                'model_version_ids': model_ids, 'metrics': metrics, 'deltas': deltas}

    @staticmethod
    def _owned(db, table, tenant, object_id):
        row = db.execute(f"SELECT * FROM {table} WHERE id=? AND tenant=?", (object_id, tenant)).fetchone()
        if not row:
            raise Problem(404, "not_found", "Object not found")
        return row

    def list_ids(self, table, tenant):
        if table not in ("runs", "models", "deployments"):
            raise ValueError("Invalid collection")
        with self.runtime.connection() as db:
            return [row[0] for row in db.execute(f"SELECT id FROM {table} WHERE tenant=? ORDER BY created_at DESC,id DESC LIMIT 100", (tenant,))]

    def list_page(self, table, tenant, limit=None, cursor=None):
        if table not in ("runs", "models", "deployments"):
            raise ValueError("Invalid collection")
        state = unpack_cursor(self.cursor_key, cursor, tenant, table) if cursor is not None else None
        if limit is None:
            limit = state["limit"] if state else 100
        if type(limit) is not int or not 1 <= limit <= 100:
            raise Problem(400, "invalid_input", "Limit must be 1..100")
        if state and limit != state["limit"]:
            raise Problem(400, "invalid_cursor", "Cursor page size cannot change")
        with self.runtime.connection() as db:
            # Freeze membership, not mutable statuses. A read transaction makes
            # the first-page high-water mark and selection one SQLite snapshot.
            db.execute("BEGIN")
            high = state["high"] if state else db.execute(
                f"SELECT COALESCE(MAX(rowid),0) FROM {table}").fetchone()[0]
            query = f"SELECT id,created_at FROM {table} WHERE tenant=? AND rowid<=?"
            args = [tenant, high]
            if state:
                query += " AND (created_at,id)<(?,?)"
                args.extend(state["last"])
            query += " ORDER BY created_at DESC,id DESC LIMIT ?"
            rows = db.execute(query, (*args, limit + 1)).fetchall()
        next_cursor = None
        if len(rows) > limit:
            last = rows[limit - 1]
            next_cursor = pack_cursor(self.cursor_key, {
                "v": 1, "tenant": hashlib.sha256(tenant.encode()).hexdigest(), "collection": table, "limit": limit,
                "high": high, "last": [last["created_at"], last["id"]]})
        return {"ids": [row["id"] for row in rows[:limit]], "limit": limit, "next_cursor": next_cursor}

    def retry(self, tenant, run_id, expected_attempt_id=None):
        """Optionally bind enqueue to the exact failed execution the caller saw."""
        with self.runtime.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = self._owned(db, "runs", tenant, run_id)
            count = db.execute("SELECT count(*) FROM attempts WHERE run_id=?", (run_id,)).fetchone()[0]
            if row["status"] != "failed" or count >= MAX_ATTEMPTS:
                raise Problem(409, "retry_conflict", "Only failed runs below the attempt limit can retry")
            if expected_attempt_id is not None:
                attempts = db.execute(
                    "SELECT id,number,status FROM attempts WHERE run_id=? ORDER BY number", (run_id,)).fetchall()
                if (row["model_version_id"] is not None or not attempts
                        or attempts[-1]["id"] != expected_attempt_id
                        or any(type(attempt["number"]) is not int or attempt["number"] != number
                               or attempt["status"] != "failed"
                               for number, attempt in enumerate(attempts, 1))):
                    raise Problem(409, "retry_conflict", "The approved failed attempt is no longer retryable")
                recorded = json.loads(row["environment_json"])
                if not isinstance(recorded, dict) or recorded.get("code_sha256") != environment()["code_sha256"]:
                    raise Problem(409, "retry_code_changed",
                                  "Training source has changed; explicitly create a new run with the current code")
            from .model_evaluations import active_count
            active = active_count(db)
            if active >= MAX_ACTIVE_RUNS:
                raise Problem(429, "run_capacity", "Active run budget reached")
            db.execute("UPDATE runs SET status='queued',updated_at=? WHERE id=?", (now(), run_id))
            self.audit(db, tenant, "retry", run_id)

    def cancel(self, tenant, run_id):
        """Withdraw only unclaimed work. Serialize with claim/retry and keep history."""
        with self.runtime.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = self._owned(db, "runs", tenant, run_id)
            if row["status"] == "cancelled":
                return
            if row["status"] != "queued":
                message = ("Running training cannot be cancelled by this operation"
                           if row["status"] == "running" else "Only queued training can be cancelled")
                raise Problem(409, "cancel_conflict", message)
            db.execute("UPDATE runs SET status='cancelled',updated_at=? WHERE id=?", (now(), run_id))
            self.audit(db, tenant, "cancel_queued", run_id)

    @staticmethod
    def _owns_attempt(db, run_id, attempt_id, status='running'):
        row = db.execute("SELECT status,model_version_id FROM runs WHERE id=?", (run_id,)).fetchone()
        attempts = db.execute("SELECT id,number,status FROM attempts WHERE run_id=? ORDER BY number", (run_id,)).fetchall()
        return bool(row and row['status'] == status and row['model_version_id'] is None
                    and attempts and attempts[-1]['id'] == attempt_id
                    and attempts[-1]['status'] == status
                    and all(a['number'] == n and a['status'] == 'failed'
                            for n, a in enumerate(attempts[:-1], 1))
                    and attempts[-1]['number'] == len(attempts))

    def cancel_running(self, tenant, run_id, expected_attempt_id):
        with self.runtime.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            row = self._owned(db, 'runs', tenant, run_id)
            old = db.execute('SELECT * FROM training_cancellations WHERE run_id=?', (run_id,)).fetchone()
            if old and old['attempt_id'] == expected_attempt_id:
                if not self._owns_attempt(db, run_id, expected_attempt_id, 'cancelled' if old['confirmed_at'] is not None else 'running'):
                    raise Problem(409, 'cancel_conflict', 'Cancellation history no longer matches the run')
                return (200 if old['confirmed_at'] is not None else 202), self._run(db, tenant, run_id)
            if old or not self._owns_attempt(db, run_id, expected_attempt_id):
                raise Problem(409, 'cancel_conflict', 'The approved running attempt is no longer cancellable')
            db.execute('INSERT INTO training_cancellations VALUES(?,?,?,NULL,NULL)', (expected_attempt_id, run_id, now()))
            self.audit(db, tenant, 'request_running_cancel', run_id)
            return 202, self._run(db, tenant, run_id)

    def cancellation_requested(self, run_id, attempt_id):
        with self.runtime.connection() as db:
            return bool(db.execute('SELECT 1 FROM training_cancellations WHERE run_id=? AND attempt_id=? AND confirmed_at IS NULL', (run_id, attempt_id)).fetchone())

    def _confirm_cancel(self, db, run_id, attempt_id, proof):
        if proof not in ('never_started', 'child_reaped', 'lease_recovered'):
            raise RuntimeError('Execution termination proof required')
        pending = db.execute('SELECT requested_at FROM training_cancellations WHERE run_id=? AND attempt_id=? AND confirmed_at IS NULL', (run_id, attempt_id)).fetchone()
        if not pending or not self._owns_attempt(db, run_id, attempt_id):
            return False
        # Wall clock may move backwards; preserve public request/confirmation order.
        stamp = max(now(), pending['requested_at'])
        db.execute('UPDATE training_cancellations SET confirmed_at=?,confirmation_kind=? WHERE attempt_id=?', (stamp, proof, attempt_id))
        db.execute("UPDATE attempts SET status='cancelled',finished_at=?,error_code=NULL WHERE id=?", (stamp, attempt_id))
        db.execute("UPDATE runs SET status='cancelled',updated_at=? WHERE id=?", (stamp, run_id))
        tenant = db.execute('SELECT tenant FROM runs WHERE id=?', (run_id,)).fetchone()[0]
        self.audit(db, tenant, 'confirm_running_cancel', run_id)
        db.execute('INSERT INTO logs(attempt_id,time,event,details) VALUES(?,?,?,?)', (attempt_id, stamp, 'cancelled', '{}'))
        return True

    def confirm_cancel(self, run_id, attempt_id, proof):
        with self.runtime.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            return self._confirm_cancel(db, run_id, attempt_id, proof)

    def log(self, attempt_id, event, details=None):
        with self.runtime.connection() as db:
            db.execute("INSERT INTO logs(attempt_id,time,event,details) VALUES(?,?,?,?)",
                       (attempt_id, now(), event, encode(details or {}).decode()))

    def recover_worker(self):
        # Caller must hold worker.lock. No other worker can still own these attempts.
        with self.runtime.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            attempts = db.execute("SELECT id,run_id FROM attempts WHERE status='running'").fetchall()
            for attempt in attempts:
                if self._confirm_cancel(db, attempt['run_id'], attempt['id'], 'lease_recovered'):
                    continue
                if not self._owns_attempt(db, attempt['run_id'], attempt['id']):
                    continue
                db.execute("UPDATE attempts SET status='failed',error_code='interrupted',finished_at=? WHERE id=?",
                           (now(), attempt["id"]))
                db.execute("UPDATE runs SET status='failed',updated_at=? WHERE id=?", (now(), attempt["run_id"]))
                db.execute("INSERT INTO logs(attempt_id,time,event,details) VALUES(?,?,?,?)",
                           (attempt["id"], now(), "interrupted", "{}"))
            return len(attempts)

    def claim(self):
        with self.runtime.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            run = db.execute("SELECT * FROM runs WHERE status='queued' ORDER BY created_at,id LIMIT 1").fetchone()
            if not run:
                return None
            number = db.execute("SELECT count(*)+1 FROM attempts WHERE run_id=?", (run["id"],)).fetchone()[0]
            attempt = identifier("attempt")
            db.execute("INSERT INTO attempts VALUES(?,?,?,'running',?,NULL,NULL)", (attempt, run["id"], number, now()))
            db.execute("UPDATE runs SET status='running',updated_at=? WHERE id=?", (now(), run["id"]))
            return dict(run), attempt

    def fail(self, run_id, attempt_id, code, *, termination_proof=None):
        with self.runtime.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            if not self._owns_attempt(db, run_id, attempt_id):
                return
            pending = db.execute('SELECT 1 FROM training_cancellations WHERE attempt_id=?', (attempt_id,)).fetchone()
            if pending:
                if termination_proof is not None:
                    self._confirm_cancel(db, run_id, attempt_id, termination_proof)
                return
            db.execute("UPDATE attempts SET status='failed',error_code=?,finished_at=? WHERE id=?", (code, now(), attempt_id))
            db.execute("UPDATE runs SET status='failed',updated_at=? WHERE id=?", (now(), run_id))
            db.execute('INSERT INTO logs(attempt_id,time,event,details) VALUES(?,?,?,?)', (attempt_id, now(), 'failed', encode({'code': code}).decode()))

    def register(self, run, attempt_id, artifact_hash, evaluation, manifest, *, deadline=None):
        def check_deadline():
            if deadline is not None and time.monotonic() >= deadline:
                raise Problem(409, "training_timeout", "Training deadline exceeded before publication")
        check_deadline()
        binding = self.dataset_binding(run)
        if binding is not None:
            from .dataset_detection import compatibility, FIXED_CONFIG
            body = json.loads(run['input_json'])
            class_map = [{'category_id': c['id'], 'coco_category_id': c['id']+1, 'name': c['name']} for c in binding['classes']]
            training_config = dict(body['config'], **FIXED_CONFIG)
            if (manifest.get('dataset') != binding or evaluation.get('dataset_version_id') != binding['dataset_version_id']
                    or evaluation.get('artifact_sha256') != artifact_hash
                    or evaluation.get('dataset_sha256') != binding['evaluation_sha256']
                    or evaluation.get('evaluation_sha256') != binding['evaluation_sha256']
                    or evaluation.get('dataset_content_sha256') != binding['sha256']
                    or evaluation.get('split') != binding['evaluation_split']
                    or evaluation.get('role') != ('held-out-test' if binding['evaluation_split']=='test' else 'validation-only')
                    or evaluation.get('category_mapping') != class_map
                    or evaluation.get('compatibility_key') != digest(compatibility(evaluation))
                    or manifest.get('config') != body['config'] or manifest.get('training_config') != training_config
                    or manifest.get('config_sha256') != digest(training_config)
                    or manifest.get('class_names') != binding['class_names']):
                raise Problem(409, 'dataset_integrity', 'Model evaluation dataset linkage mismatch')
        model_id, eval_id = identifier("model"), identifier("eval")
        with self.runtime.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            check_deadline()
            if not self._owns_attempt(db, run["id"], attempt_id):
                raise RuntimeError("Attempt no longer owns publication")
            if db.execute('SELECT 1 FROM training_cancellations WHERE attempt_id=?', (attempt_id,)).fetchone():
                raise Problem(409, 'cancel_requested', 'Cancellation prevents publication')
            db.execute("INSERT INTO evaluations VALUES(?,?,?,?,?,?)",
                       (eval_id, attempt_id, artifact_hash, evaluation["dataset_sha256"], encode(evaluation).decode(), now()))
            manifest.update(model_version_id=model_id, run_id=run["id"], attempt_id=attempt_id, evaluation_id=eval_id,
                            artifact_sha256=artifact_hash, artifact_id=attempt_id)
            db.execute("INSERT INTO models(id,tenant,run_id,attempt_id,evaluation_id,artifact_sha256,manifest_json,created_at) VALUES(?,?,?,?,?,?,?,?)",
                       (model_id, run["tenant"], run["id"], attempt_id, eval_id, artifact_hash, encode(manifest).decode(), now()))
            db.execute("UPDATE attempts SET status='succeeded',finished_at=? WHERE id=?", (now(), attempt_id))
            db.execute("UPDATE runs SET status='succeeded',model_version_id=?,updated_at=? WHERE id=?", (model_id, now(), run["id"]))
            self.audit(db, run["tenant"], "register_model", model_id)
            check_deadline()
        self.log(attempt_id, "registered", {"model_version_id": model_id})
        return model_id

    def model(self, tenant, model_id):
        with self.runtime.connection() as db:
            row = self._owned(db, "models", tenant, model_id)
            result = json.loads(row["manifest_json"])
            evaluation = db.execute("SELECT * FROM evaluations WHERE id=?", (row["evaluation_id"],)).fetchone()
            result["evaluation"] = json.loads(evaluation["result_json"]) if evaluation else None
            from .model_imports import validate_manifest
            validate_manifest(db, row, result)
            result["created_at"] = row["created_at"]
            return result

    def create_deployment(self, tenant, model_id):
        # Preserve the internal legacy helper's string return contract as well.
        return self.submit_deployment(tenant, model_id)[0]

    def submit_deployment(self, tenant, model_id, key=None):
        with self.runtime.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            request_hash = digest({"model_version_id": model_id})
            if key is not None:
                old = db.execute("SELECT * FROM deployment_keys WHERE tenant=? AND idempotency_key=?",
                                 (tenant, key)).fetchone()
                if old:
                    if old["request_sha256"] != request_hash:
                        raise Problem(409, "idempotency_conflict", "Key already bound to different input")
                    return old["deployment_id"], False
            self._owned(db, "models", tenant, model_id)
            count = db.execute("SELECT count(*) FROM deployments WHERE desired='running'").fetchone()[0]
            if count >= MAX_DEPLOYMENTS:
                raise Problem(429, "deployment_capacity", "Deployment budget reached")
            deployment_id = identifier("deployment")
            db.execute("INSERT INTO deployments VALUES(?,?,?,'running','starting',NULL,NULL,NULL,?)",
                       (deployment_id, tenant, model_id, now()))
            if key is not None:
                db.execute("INSERT INTO deployment_keys VALUES(?,?,?,?)",
                           (tenant, key, request_hash, deployment_id))
            self.audit(db, tenant, "start_deployment", deployment_id)
            return deployment_id, True

    def deployment(self, tenant, deployment_id):
        with self.runtime.connection() as db:
            row = dict(self._owned(db, "deployments", tenant, deployment_id))
            model = self._owned(db, "models", tenant, row["model_version_id"])
            from .recipes import REGISTRY
            manifest = json.loads(model["manifest_json"])
            from .model_imports import validate_manifest
            validate_manifest(db, model, manifest)
            runtime = next((spec.runtime for spec in REGISTRY.values() if spec.format == manifest["format"]), "unknown")
            if manifest.get('origin') == 'imported':
                from .model_imports import public_identity
                row.update(public_identity(manifest))
                runtime = 'onnxruntime-cpu'
            row.update(contract=manifest["contract"], runtime=runtime, resources={"processes": 1, "gpu": 0,
                       "hard_cpu_memory_isolation": False}, ready=row["observed"] == "ready" and row["desired"] == "running",
                       endpoint=f"/deployments/{deployment_id}/predict")
            if manifest.get("recipe") == "yolox-nano-dataset/v1":
                from .dataset_serving import public_identity
                row.update(public_identity(manifest))
            return row

    def stop_desired(self, tenant, deployment_id):
        with self.runtime.connection() as db:
            self._owned(db, "deployments", tenant, deployment_id)
            db.execute("UPDATE deployments SET desired='stopped' WHERE id=?", (deployment_id,))
            self.audit(db, tenant, "stop_deployment", deployment_id)

    def observe(self, deployment_id, status, error=None, pid=None):
        with self.runtime.connection() as db:
            db.execute("UPDATE deployments SET observed=?,error_code=?,pid=?,checked_at=? WHERE id=?",
                       (status, error, pid, now(), deployment_id))

    def deployment_rows(self, pending_only=False):
        with self.runtime.connection() as db:
            query = "SELECT * FROM deployments"
            if pending_only:
                query += " WHERE desired='running' OR observed<>'stopped'"
            return [dict(row) for row in db.execute(query)]
