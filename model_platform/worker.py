"""Independent single-host worker. HTTP processes never train."""
import json
import time

from .algorithm import TrainingFailure, train, write_model
from . import digits
from .common import CONTRACT, RECIPE, digest, environment, training_request
from .evaluation import evaluate
from .store import Store

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
        data = body["dataset"]
        recipe = body.get("recipe", RECIPE)
        store.log(attempt_id, "started", {"recipe": recipe})
        started = time.monotonic()
        trainer = digits.train if recipe == digits.RECIPE else train
        model = trainer(data["train"], body["config"], lambda event, details: store.log(attempt_id, event, details))
        training_seconds = time.monotonic() - started
        path = store.runtime.artifact(attempt_id)
        artifact_hash = write_model(path, model)
        store.log(attempt_id, "artifact_written", {"sha256": artifact_hash})
        evaluation = evaluate(path, artifact_hash, data["evaluation"], EXECUTION_ENVIRONMENT)
        store.log(attempt_id, "evaluated", {"accuracy": evaluation["accuracy"], "log_loss": evaluation["log_loss"]})
        manifest = {"recipe": recipe, "contract": model["contract"], "format": model["format"],
                    "training_seconds": training_seconds,
                    "dimensions": model["dimensions"], "dataset": {"source": data["source"], "version": data["version"],
                    "sha256": digest(data), "train_sha256": digest(data["train"]), "evaluation_sha256": digest(data["evaluation"])},
                    "config": body["config"], "environment": json.loads(run["environment_json"]),
                    "execution_environment": EXECUTION_ENVIRONMENT, "target_profile": "local-cpu-python/v1",
                    "capabilities": {"train": True, "evaluate": True, "serve_cpu": True, "device": False}}
        if recipe == digits.RECIPE:
            train_x = {tuple(row["x"]) for row in data["train"]}
            manifest["dataset"]["provenance"] = data["provenance"]
            manifest["dataset"]["feature_overlap_evaluation_rows"] = sum(tuple(row["x"]) in train_x for row in data["evaluation"])
            manifest["dataset"]["split_policy"] = "Caller-declared official writer split; all records retained; source hashes are provenance declarations, not independently authenticated by API"
        verify_execution_code(run)
        store.register(run, attempt_id, artifact_hash, evaluation, manifest)
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
        while True:
            job = store.claim()
            if job:
                execute(store, *job)
            if once:
                return
            if not job:
                time.sleep(0.25)
