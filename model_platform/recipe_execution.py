"""Built-in execution adapters; no lifecycle or runtime registration side effects."""
import time
from .algorithm import write_model, TrainingFailure
from .common import digest
from .evaluation import evaluate
from . import digits


def execute_numeric(spec, runtime, body, path, log, execution_environment, submission_environment):
    data = body["dataset"]
    recipe = spec.id
    started = time.monotonic()
    model = spec.train(data["train"], body["config"], log)
    training_seconds = time.monotonic() - started
    artifact_hash = write_model(path, model)
    log("artifact_written", {"sha256": artifact_hash})
    evaluation = evaluate(path, artifact_hash, data["evaluation"], execution_environment)
    log("evaluated", {"accuracy": evaluation["accuracy"], "log_loss": evaluation["log_loss"]})
    manifest = {"recipe": recipe, "contract": model["contract"], "format": model["format"],
                "training_seconds": training_seconds,
                "dimensions": model["dimensions"], "dataset": {"source": data["source"], "version": data["version"],
                "sha256": digest(data), "train_sha256": digest(data["train"]), "evaluation_sha256": digest(data["evaluation"])},
                "config": body["config"], "environment": submission_environment,
                "execution_environment": execution_environment, "target_profile": "local-cpu-python/v1",
                "capabilities": {"train": True, "evaluate": True, "serve_cpu": True, "device": False}}
    if recipe == digits.RECIPE:
        train_x = {tuple(row["x"]) for row in data["train"]}
        manifest["dataset"]["provenance"] = data["provenance"]
        manifest["dataset"]["feature_overlap_evaluation_rows"] = sum(tuple(row["x"]) in train_x for row in data["evaluation"])
        manifest["dataset"]["split_policy"] = "Caller-declared official writer split; all records retained; source hashes are provenance declarations, not independently authenticated by API"
    return artifact_hash, evaluation, manifest


def execute_fashion(spec, runtime, body, path, log, execution_environment, submission_environment):
    from .fashion_training import train_export
    try:
        result = train_export(runtime, body, path, log, execution_environment)
    except ImportError:
        raise TrainingFailure("optional_dependencies_missing") from None
    result[2]["environment"] = submission_environment
    return result
