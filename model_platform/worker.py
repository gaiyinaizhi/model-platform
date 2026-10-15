"""Independent single-host worker. HTTP processes never train."""
import json
import time

from .algorithm import TrainingFailure
from .common import RECIPE, Problem, environment, training_request
from .store import Store
from .worker_readiness import WorkerHeartbeat, probe_dependencies

# Match provenance to this process's loaded code, not a later on-disk edit.
EXECUTION_ENVIRONMENT = environment()


def verify_execution_code(run):
    loaded = EXECUTION_ENVIRONMENT["code_sha256"]
    if loaded != json.loads(run["environment_json"])["code_sha256"] or loaded != environment()["code_sha256"]:
        raise TrainingFailure("execution_code_changed")


def execute(store, run, attempt_id):
    try:
        body = training_request(json.loads(run["input_json"]))
        verify_execution_code(run)
        from .recipes import get_recipe
        spec = get_recipe(body.get("recipe", RECIPE))
        binding = store.dataset_binding(run)
        if binding is not None:
            body = dict(body, _dataset_binding=binding, _dataset_tenant=run['tenant'])
        store.log(attempt_id, "started", {"recipe": spec.id})
        artifact_hash, evaluation, manifest = spec.execute(
            spec, store.runtime, body, store.runtime.artifact(attempt_id),
            lambda event, details: store.log(attempt_id, event, details),
            EXECUTION_ENVIRONMENT, json.loads(run["environment_json"]))
        verify_execution_code(run)
        store.register(run, attempt_id, artifact_hash, evaluation, manifest)
    except Problem as exc:
        store.fail(run['id'], attempt_id, exc.code)
    except TrainingFailure as exc:
        store.fail(run["id"], attempt_id, exc.code)
    except Exception:
        # No raw exceptions: they can contain paths, user data or credentials.
        store.fail(run["id"], attempt_id, "execution_failed")


def run_worker(runtime, once=False):
    with runtime.lock("worker"):
        store = Store(runtime)
        recovered = store.recover_worker()
        print(f"worker started; interrupted attempts: {recovered}", flush=True)
        heartbeat = WorkerHeartbeat(runtime, probe_dependencies()).start()
        try:
            while True:
                job = store.claim()
                if job:
                    heartbeat.set_state("busy")
                    execute(store, *job)
                    heartbeat.set_state("idle")
                if once:
                    return
                if not job:
                    time.sleep(0.25)
        finally:
            heartbeat.stop()
