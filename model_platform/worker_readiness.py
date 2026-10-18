"""Worker-authored liveness and installed dependency facts, never data readiness.

One bounded singleton row per runtime. The worker holds the existing flock for
its entire lifetime; instance-qualified updates also fence late old writers.
HTTP readers never import optional training libraries or start a worker.
"""
import importlib
import importlib.metadata
import json
import sqlite3
import threading
import time
import uuid

from .recipes import REGISTRY

HEARTBEAT_SECONDS = 2.0
STALE_AFTER_SECONDS = 15.0
# Source-reviewed import targets only, no entry points or request-driven imports.
IMPORT_TARGETS = {
    'torch': 'torch', 'numpy': 'numpy', 'onnx': 'onnx',
    'onnxruntime': 'onnxruntime', 'onnxscript': 'onnxscript',
    'Pillow': 'PIL.Image', 'pycocotools': 'pycocotools.cocoeval', 'PyYAML': 'yaml',
}
SCHEMA = """CREATE TABLE IF NOT EXISTS worker_readiness (
 singleton INTEGER PRIMARY KEY CHECK(singleton=1), instance TEXT NOT NULL,
 state TEXT NOT NULL CHECK(state IN ('idle','busy','stopped')),
 checked_at REAL NOT NULL, dependencies_json TEXT NOT NULL
)"""


def probe_dependencies():
    """Run once in the actual worker interpreter, including binary import checks.

    Presence is not a training, data, compatibility or quality guarantee. Import
    failures are reduced to fixed codes; paths and raw exceptions never escape.
    """
    result = {}
    names = {name for spec in REGISTRY.values() for name in spec.optional_dependencies}
    for name in sorted(names):
        item = {'name': name, 'available': False, 'version': None, 'reason': None}
        try:
            target = IMPORT_TARGETS.get(name)
            if target is None:
                raise ImportError
            importlib.import_module(target)
            version = importlib.metadata.version(name)
            # Avoid exposing unexpected metadata contents as an environment path.
            if not version or len(version) > 100 or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.+!_-' for c in version):
                raise ValueError
            item.update(available=True, version=version)
        except (ImportError, importlib.metadata.PackageNotFoundError):
            item['reason'] = 'dependency_missing'
        except Exception:
            item['reason'] = 'dependency_unavailable'
        result[name] = item
    return result


class WorkerHeartbeat:
    def __init__(self, runtime, dependencies, interval=HEARTBEAT_SECONDS):
        self.runtime = runtime
        self.instance = uuid.uuid4().hex
        self.dependencies = dependencies
        self.interval = interval
        self.state = 'idle'
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        with self.runtime.connection() as db:
            db.execute(SCHEMA)
            db.execute('INSERT OR REPLACE INTO worker_readiness VALUES(1,?,?,?,?)',
                       (self.instance, self.state, time.time(), json.dumps(self.dependencies)))
        self._thread = threading.Thread(target=self._loop, name='worker-heartbeat', daemon=True)
        self._thread.start()
        return self

    def _write(self):
        with self.runtime.connection() as db:
            db.execute('UPDATE worker_readiness SET state=?,checked_at=? WHERE singleton=1 AND instance=?',
                       (self.state, time.time(), self.instance))

    def set_state(self, state):
        if state not in ('idle', 'busy'):
            raise ValueError('Invalid live worker state')
        with self._lock:
            if self._stop.is_set():
                return
            self.state = state
            try:
                self._write()
            except sqlite3.Error:
                pass

    def _loop(self):
        while not self._stop.wait(self.interval):
            with self._lock:
                if self._stop.is_set():
                    return
                try:
                    self._write()
                except sqlite3.Error:
                    # No false fresh report on storage failure. Retry next tick;
                    # readers naturally mark the last successful heartbeat stale.
                    pass

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        with self._lock:
            self.state = 'stopped'
            try:
                self._write()
            except sqlite3.Error:
                # On storage failure the last successful report expires.
                pass


def public_report(store, tenant):
    """Return only worker facts and caller-owned aggregate queue counts."""
    with store.runtime.connection() as db:
        db.execute('BEGIN')
        exists = db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='worker_readiness'").fetchone()
        row = db.execute('SELECT state,checked_at,dependencies_json FROM worker_readiness WHERE singleton=1').fetchone() if exists else None
        queue = {'queued': 0, 'running': 0}
        for item in db.execute("SELECT status,count(*) AS n FROM runs WHERE tenant=? AND status IN ('queued','running') GROUP BY status", (tenant,)):
            queue[item['status']] = item['n']
        for item in db.execute("SELECT status,count(*) AS n FROM model_evaluations WHERE tenant=? AND status IN ('queued','running') GROUP BY status", (tenant,)):
            queue[item['status']] += item['n']
    checked_at = row['checked_at'] if row else None
    age = time.time() - checked_at if row else None
    fresh = row is not None and 0 <= age <= STALE_AFTER_SECONDS
    state = 'absent' if row is None else 'stopped' if row['state'] == 'stopped' else row['state'] if fresh else 'stale'
    ready = fresh and state in ('idle', 'busy')
    unready_reasons = [] if ready else ['worker_' + state]
    reasons = ['worker_busy'] if ready and state == 'busy' else list(unready_reasons)
    deps = json.loads(row['dependencies_json']) if row else {}
    recipes = []
    for spec in REGISTRY.values():
        dependencies = [deps.get(name, {'name': name, 'available': False, 'version': None,
                                       'reason': 'dependency_not_checked'}) for name in spec.optional_dependencies]
        recipe_reasons = list(unready_reasons)
        if any(not item['available'] for item in dependencies):
            recipe_reasons.append('dependencies_unavailable')
        recipes.append({'id': spec.id, 'ready': ready and not recipe_reasons,
                        'dependencies': dependencies, 'reasons': recipe_reasons,
                        'data_status': 'not_checked'})
    return {'worker': {'state': state, 'fresh': fresh, 'ready': ready,
                       'accepting_immediately': ready and state == 'idle',
                       'checked_at': checked_at, 'stale_after_seconds': STALE_AFTER_SECONDS,
                       'reasons': reasons}, 'recipes': recipes, 'queue': queue}
