"""Separate evaluation stage: reload the actual artifact on a held-out split."""
import math
import time

from .algorithm import load_model, predict
from .common import digest, environment


def evaluate(path, artifact_hash, rows, execution_environment=None):
    model = load_model(path, artifact_hash)
    from .recipes import get_recipe
    spec = get_recipe(model["recipe"])
    batch_size = spec.budgets.prediction_rows
    # Evaluation may use most of the 1,000-row dataset budget. Keep the
    # independent online inference limit by evaluating in bounded batches.
    started = time.monotonic()
    outputs = []
    for offset in range(0, len(rows), batch_size):
        outputs.extend(predict(model, [row["x"] for row in rows[offset:offset + batch_size]]))
    inference_seconds = time.monotonic() - started
    classes = spec.classes
    multiclass = classes > 2
    confusion = [[0] * classes for _ in range(classes)]
    loss = 0.0
    for row, output in zip(rows, outputs):
        target = row["y"]
        p = output["probabilities"][target] if multiclass else output["probability"]
        confusion[target][output["class"]] += 1
        p = min(max(p, 1e-15), 1 - 1e-15)
        loss -= math.log(p) if multiclass else target * math.log(p) + (1 - target) * math.log(1 - p)
    f1s = []
    for label in range(classes):
        denominator = sum(confusion[label]) + sum(row[label] for row in confusion)
        f1s.append(2 * confusion[label][label] / denominator if denominator else 0.0)
    return {"protocol": "held-out-multiclass/v1" if multiclass else "held-out-binary/v1",
            "macro_f1": sum(f1s) / classes, "class_order": list(range(classes)),
            **({"inference_seconds": inference_seconds, "timing_scope": "Reloaded artifact, batched Python prediction; local observation, not benchmark"} if multiclass else {}), "sample_count": len(rows), "artifact_sha256": artifact_hash,
            "dataset_sha256": digest(rows), "accuracy": sum(confusion[i][i] for i in range(classes)) / len(rows),
            "log_loss": loss / len(rows), "confusion": confusion,
            "environment": execution_environment if execution_environment is not None else environment(),
            "limitations": "Caller-supplied holdout; not external audit or a visual/device benchmark"}
