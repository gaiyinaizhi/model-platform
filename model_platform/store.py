"""SQLite is the sole writer authority for technical records and queue state."""
import json
import uuid

from .common import (MAX_ACTIVE_RUNS, MAX_ATTEMPTS, MAX_DEPLOYMENTS, Problem,
                     RECIPE, digest, encode, environment, now)

SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS runs (
 id TEXT PRIMARY KEY, tenant TEXT NOT NULL, idempotency_key TEXT NOT NULL,
 request_sha256 TEXT NOT NULL, input_json TEXT NOT NULL, environment_json TEXT NOT NULL,
 status TEXT NOT NULL CHECK(status IN ('queued','running','succeeded','failed')),
 created_at REAL NOT NULL, updated_at REAL NOT NULL, model_version_id TEXT,
 UNIQUE(tenant,idempotency_key)
);
CREATE TABLE IF NOT EXISTS attempts (
 id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(id), number INTEGER NOT NULL,
 status TEXT NOT NULL CHECK(status IN ('running','succeeded','failed')),
 started_at REAL NOT NULL, finished_at REAL, error_code TEXT, UNIQUE(run_id,number)
);
CREATE TABLE IF NOT EXISTS logs (
 id INTEGER PRIMARY KEY AUTOINCREMENT, attempt_id TEXT NOT NULL REFERENCES attempts(id),
 time REAL NOT NULL, event TEXT NOT NULL, details TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS evaluations (
 id TEXT PRIMARY KEY, attempt_id TEXT NOT NULL UNIQUE REFERENCES attempts(id),
 artifact_sha256 TEXT NOT NULL, evaluation_sha256 TEXT NOT NULL, result_json TEXT NOT NULL,
 created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS models (
 id TEXT PRIMARY KEY, tenant TEXT NOT NULL, run_id TEXT NOT NULL UNIQUE REFERENCES runs(id),
 attempt_id TEXT NOT NULL UNIQUE REFERENCES attempts(id),
 evaluation_id TEXT NOT NULL UNIQUE REFERENCES evaluations(id),
 artifact_sha256 TEXT NOT NULL, manifest_json TEXT NOT NULL, created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS deployments (
 id TEXT PRIMARY KEY, tenant TEXT NOT NULL, model_version_id TEXT NOT NULL REFERENCES models(id),
 desired TEXT NOT NULL CHECK(desired IN ('running','stopped')),
 observed TEXT NOT NULL CHECK(observed IN ('starting','ready','unavailable','failed','stopped')),
 checked_at REAL, error_code TEXT, pid INTEGER, created_at REAL NOT NULL
);
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
        with runtime.connection() as db:
            db.executescript(SCHEMA)
            for table in ("models", "evaluations"):
                for action in ("UPDATE", "DELETE"):
                    db.execute(f"CREATE TRIGGER IF NOT EXISTS {table}_{action.lower()}_immutable "
                               f"BEFORE {action} ON {table} BEGIN "
                               "SELECT RAISE(ABORT,'technical records are immutable'); END")

    @staticmethod
    def audit(db, tenant, action, object_id):
        db.execute("INSERT INTO audit(time,tenant,action,object_id) VALUES(?,?,?,?)",
                   (now(), tenant, action, object_id))

    def submit(self, tenant, key, body):
        request_hash = digest(body)
        with self.runtime.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            old = db.execute("SELECT * FROM runs WHERE tenant=? AND idempotency_key=?", (tenant, key)).fetchone()
            if old:
                if old["request_sha256"] != request_hash:
                    raise Problem(409, "idempotency_conflict", "Key already bound to different input")
                return old["id"], False
            count = db.execute("SELECT count(*) FROM runs WHERE status IN ('queued','running')").fetchone()[0]
            if count >= MAX_ACTIVE_RUNS:
                raise Problem(429, "run_capacity", "Active run budget reached")
            run_id = identifier("run")
            stamp = now()
            db.execute("INSERT INTO runs VALUES(?,?,?,?,?,?,?,?,?,NULL)",
                       (run_id, tenant, key, request_hash, encode(body).decode(),
                        encode(environment()).decode(), "queued", stamp, stamp))
            self.audit(db, tenant, "submit", run_id)
            return run_id, True

    def run(self, tenant, run_id):
        with self.runtime.connection() as db:
            row = self._owned(db, "runs", tenant, run_id)
            body = json.loads(row["input_json"])
            result = {k: row[k] for k in ("id", "status", "created_at", "updated_at", "request_sha256", "model_version_id")}
            result.update(recipe=RECIPE, environment=json.loads(row["environment_json"]),
                          dataset={"source": body["dataset"]["source"], "version": body["dataset"]["version"],
                                   "sha256": digest(body["dataset"]), "train_count": len(body["dataset"]["train"]),
                                   "evaluation_count": len(body["dataset"]["evaluation"])}, config=body["config"],
                          business_ref=body.get("business_ref"), attempts=[])
            for attempt in db.execute("SELECT * FROM attempts WHERE run_id=? ORDER BY number", (run_id,)):
                item = dict(attempt)
                item["logs"] = [dict(log) for log in db.execute(
                    "SELECT time,event,details FROM logs WHERE attempt_id=? ORDER BY id", (attempt["id"],))]
                for log in item["logs"]:
                    log["details"] = json.loads(log["details"])
                result["attempts"].append(item)
            return result

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
            return [row[0] for row in db.execute(f"SELECT id FROM {table} WHERE tenant=? ORDER BY created_at DESC LIMIT 100", (tenant,))]

    def retry(self, tenant, run_id):
        with self.runtime.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = self._owned(db, "runs", tenant, run_id)
            count = db.execute("SELECT count(*) FROM attempts WHERE run_id=?", (run_id,)).fetchone()[0]
            if row["status"] != "failed" or count >= MAX_ATTEMPTS:
                raise Problem(409, "retry_conflict", "Only failed runs below the attempt limit can retry")
            active = db.execute("SELECT count(*) FROM runs WHERE status IN ('queued','running')").fetchone()[0]
            if active >= MAX_ACTIVE_RUNS:
                raise Problem(429, "run_capacity", "Active run budget reached")
            db.execute("UPDATE runs SET status='queued',updated_at=? WHERE id=?", (now(), run_id))
            self.audit(db, tenant, "retry", run_id)

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
                db.execute("UPDATE attempts SET status='failed',error_code='interrupted',finished_at=? WHERE id=?",
                           (now(), attempt["id"]))
                db.execute("UPDATE runs SET status='failed',updated_at=? WHERE id=?", (now(), attempt["run_id"]))
                db.execute("INSERT INTO logs(attempt_id,time,event,details) VALUES(?,?,?,?)",
                           (attempt["id"], now(), "interrupted", "{}"))
            return len(attempts)

    def claim(self):
        with self.runtime.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            run = db.execute("SELECT * FROM runs WHERE status='queued' ORDER BY created_at LIMIT 1").fetchone()
            if not run:
                return None
            number = db.execute("SELECT count(*)+1 FROM attempts WHERE run_id=?", (run["id"],)).fetchone()[0]
            attempt = identifier("attempt")
            db.execute("INSERT INTO attempts VALUES(?,?,?,'running',?,NULL,NULL)", (attempt, run["id"], number, now()))
            db.execute("UPDATE runs SET status='running',updated_at=? WHERE id=?", (now(), run["id"]))
            return dict(run), attempt

    def fail(self, run_id, attempt_id, code):
        with self.runtime.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("UPDATE attempts SET status='failed',error_code=?,finished_at=? WHERE id=? AND status='running'",
                       (code, now(), attempt_id))
            db.execute("UPDATE runs SET status='failed',updated_at=? WHERE id=? AND status='running'", (now(), run_id))
        self.log(attempt_id, "failed", {"code": code})

    def register(self, run, attempt_id, artifact_hash, evaluation, manifest):
        model_id, eval_id = identifier("model"), identifier("eval")
        with self.runtime.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            owned = db.execute("SELECT status FROM attempts WHERE id=? AND run_id=?", (attempt_id, run["id"])).fetchone()
            if not owned or owned[0] != "running":
                raise RuntimeError("Attempt no longer owns publication")
            db.execute("INSERT INTO evaluations VALUES(?,?,?,?,?,?)",
                       (eval_id, attempt_id, artifact_hash, evaluation["dataset_sha256"], encode(evaluation).decode(), now()))
            manifest.update(model_version_id=model_id, run_id=run["id"], attempt_id=attempt_id, evaluation_id=eval_id,
                            artifact_sha256=artifact_hash, artifact_id=attempt_id)
            db.execute("INSERT INTO models VALUES(?,?,?,?,?,?,?,?)",
                       (model_id, run["tenant"], run["id"], attempt_id, eval_id, artifact_hash, encode(manifest).decode(), now()))
            db.execute("UPDATE attempts SET status='succeeded',finished_at=? WHERE id=?", (now(), attempt_id))
            db.execute("UPDATE runs SET status='succeeded',model_version_id=?,updated_at=? WHERE id=?", (model_id, now(), run["id"]))
            self.audit(db, run["tenant"], "register_model", model_id)
        self.log(attempt_id, "registered", {"model_version_id": model_id})
        return model_id

    def model(self, tenant, model_id):
        with self.runtime.connection() as db:
            row = self._owned(db, "models", tenant, model_id)
            result = json.loads(row["manifest_json"])
            evaluation = db.execute("SELECT * FROM evaluations WHERE id=?", (row["evaluation_id"],)).fetchone()
            result["evaluation"] = json.loads(evaluation["result_json"])
            result["created_at"] = row["created_at"]
            return result

    def create_deployment(self, tenant, model_id):
        with self.runtime.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            self._owned(db, "models", tenant, model_id)
            count = db.execute("SELECT count(*) FROM deployments WHERE desired='running'").fetchone()[0]
            if count >= MAX_DEPLOYMENTS:
                raise Problem(429, "deployment_capacity", "Deployment budget reached")
            deployment_id = identifier("deployment")
            db.execute("INSERT INTO deployments VALUES(?,?,?,'running','starting',NULL,NULL,NULL,?)",
                       (deployment_id, tenant, model_id, now()))
            self.audit(db, tenant, "start_deployment", deployment_id)
            return deployment_id

    def deployment(self, tenant, deployment_id):
        with self.runtime.connection() as db:
            row = dict(self._owned(db, "deployments", tenant, deployment_id))
            row.update(contract="binary-numeric/v1", runtime="python-cpu-json/v1", resources={"processes": 1, "gpu": 0,
                       "hard_cpu_memory_isolation": False}, ready=row["observed"] == "ready" and row["desired"] == "running",
                       endpoint=f"/deployments/{deployment_id}/predict")
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
