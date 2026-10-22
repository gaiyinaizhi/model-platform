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


def execute(store, run, attempt_id, lease_fd=None):
    from .training_process import preflight, supervise, ExecutionLifecycle, CancellationObserved
    if lease_fd is None:
        # Direct internal callers retain the same ownership fence. Collision
        # occurs outside failure handling: never fail another owner's attempt.
        preflight()
        with store.runtime.lock("worker", inherited_lease=True) as lease:
            return execute(store, run, attempt_id, lease)
    lifecycle = ExecutionLifecycle()
    try:
        if store.cancellation_requested(run['id'], attempt_id):
            raise CancellationObserved()
        body = training_request(json.loads(run["input_json"]))
        verify_execution_code(run)
        from .recipes import get_recipe
        spec = get_recipe(body.get("recipe", RECIPE))
        binding = store.dataset_binding(run)
        if binding is not None:
            body = dict(body, _dataset_binding=binding, _dataset_tenant=run['tenant'])
        store.log(attempt_id, "started", {"recipe": spec.id})
        deadline = time.monotonic() + (spec.budgets.training_seconds +
            spec.budgets.export_seconds + spec.budgets.evaluation_seconds + 30)
        artifact_hash, evaluation, manifest = supervise(
            store.runtime, spec, body, attempt_id, json.loads(run["environment_json"]),
            EXECUTION_ENVIRONMENT, lease_fd,
            lambda event, details: store.log(attempt_id, event, details), deadline,
            cancellation_requested=lambda: store.cancellation_requested(run['id'], attempt_id),
            lifecycle=lifecycle)
        verify_execution_code(run)
        store.register(run, attempt_id, artifact_hash, evaluation, manifest, deadline=deadline)
    except CancellationObserved:
        store.confirm_cancel(run['id'], attempt_id, lifecycle.proof)
    except Problem as exc:
        store.fail(run['id'], attempt_id, exc.code, termination_proof=lifecycle.proof)
    except TrainingFailure as exc:
        store.fail(run["id"], attempt_id, exc.code, termination_proof=lifecycle.proof)
    except Exception:
        # No raw exceptions: they can contain paths, user data or credentials.
        store.fail(run["id"], attempt_id, "execution_failed", termination_proof=lifecycle.proof)


def run_worker(runtime, once=False):
    from .training_process import preflight
    preflight()
    with runtime.lock("worker", inherited_lease=True) as lease_fd:
        store = Store(runtime)
        from .model_evaluations import ModelEvaluationStore, execute as execute_evaluation
        evaluations = ModelEvaluationStore(store)
        recovered = store.recover_worker() + evaluations.recover()
        print(f"worker started; interrupted attempts: {recovered}", flush=True)
        heartbeat = WorkerHeartbeat(runtime, probe_dependencies()).start()
        try:
            while True:
                evaluation = evaluations.claim()
                if evaluation:
                    heartbeat.set_state("busy")
                    execute_evaluation(evaluations, evaluation, EXECUTION_ENVIRONMENT, lease_fd=lease_fd)
                    heartbeat.set_state("idle")
                    if once:
                        return
                    continue
                job = store.claim()
                if job:
                    heartbeat.set_state("busy")
                    execute(store, *job, lease_fd=lease_fd)
                    heartbeat.set_state("idle")
                if once:
                    return
                if not job:
                    time.sleep(0.25)
        finally:
            heartbeat.stop()
