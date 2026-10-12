"""Separate evaluation stage: reload the actual artifact on a held-out split."""
import math

from .algorithm import load_model, predict
from .common import digest, environment


def evaluate(path, artifact_hash, rows, execution_environment=None):
    model = load_model(path, artifact_hash)
    # Evaluation may use most of the 1,000-row dataset budget. Keep the
    # independent online inference limit by evaluating in bounded batches.
    outputs = []
    for offset in range(0, len(rows), 128):
        outputs.extend(predict(model, [row["x"] for row in rows[offset:offset + 128]]))
    confusion = [[0, 0], [0, 0]]
    loss = 0.0
    for row, output in zip(rows, outputs):
        target, p = row["y"], output["probability"]
        confusion[target][output["class"]] += 1
        p = min(max(p, 1e-15), 1 - 1e-15)
        loss -= target * math.log(p) + (1 - target) * math.log(1 - p)
    return {"protocol": "held-out-binary/v1", "sample_count": len(rows), "artifact_sha256": artifact_hash,
            "dataset_sha256": digest(rows), "accuracy": (confusion[0][0] + confusion[1][1]) / len(rows),
            "log_loss": loss / len(rows), "confusion": confusion,
            "environment": execution_environment if execution_environment is not None else environment(),
            "limitations": "Caller-supplied holdout; not external audit or a visual/device benchmark"}
