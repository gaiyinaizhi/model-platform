"""Pure Python binary logistic regression; no arbitrary recipe execution."""
import hashlib
import json
import math
import time

from .common import CONTRACT, RECIPE, encode, features, number, object_fields


class TrainingFailure(Exception):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def sigmoid(z):
    if z >= 0:
        return 1 / (1 + math.exp(-z))
    e = math.exp(z)
    return e / (1 + e)


def train(rows, config, log):
    width, count = len(rows[0]["x"]), len(rows)
    means = [sum(row["x"][j] for row in rows) / count for j in range(width)]
    # Scaled Euclidean norm avoids underflow when valid nonconstant inputs
    # have tiny magnitudes (squaring 1e-200 directly rounds to zero).
    scales = [math.hypot(*(row["x"][j] - means[j] for row in rows)) / math.sqrt(count) for j in range(width)]
    if not any(scales):
        raise TrainingFailure("degenerate_features")
    scales = [s or 1.0 for s in scales]
    xs = [[(row["x"][j] - means[j]) / scales[j] for j in range(width)] for row in rows]
    weights, bias = [0.0] * width, 0.0
    start = time.monotonic()
    for epoch in range(config["epochs"]):
        gradient, bias_gradient = [0.0] * width, 0.0
        for row, x in zip(rows, xs):
            residual = sigmoid(sum(w * v for w, v in zip(weights, x)) + bias) - row["y"]
            bias_gradient += residual
            for j, v in enumerate(x):
                gradient[j] += residual * v
        rate = config["learning_rate"] / count
        weights = [w - rate * g for w, g in zip(weights, gradient)]
        bias -= rate * bias_gradient
        if (epoch + 1) % 20 == 0 or epoch + 1 == config["epochs"]:
            log("train_progress", {"epoch": epoch + 1, "total_epochs": config["epochs"]})
        if time.monotonic() - start > 60:
            raise TrainingFailure("training_timeout")
    return {"format": "logistic-json/v1", "recipe": RECIPE, "contract": CONTRACT,
            "dimensions": width, "weights": weights, "bias": bias, "means": means, "scales": scales}


def validate_model(model):
    object_fields(model, ("format", "recipe", "contract", "dimensions", "weights", "bias", "means", "scales"))
    if model["format"] != "logistic-json/v1" or model["recipe"] != RECIPE or model["contract"] != CONTRACT:
        raise ValueError("Unsupported model contract")
    width = model["dimensions"]
    if type(width) is not int or not 1 <= width <= 16:
        raise ValueError("Invalid model dimensions")
    for key in ("weights", "means", "scales"):
        if not isinstance(model[key], list) or len(model[key]) != width:
            raise ValueError("Invalid model vectors")
        for v in model[key]:
            number(v, 1e-300 if key == "scales" else -1e12, 1e12)
    number(model["bias"], -1e12, 1e12)
    return model


def load_model(path, expected_hash):
    if path.is_symlink():
        raise ValueError("Symlink artifacts are not supported")
    with path.open("rb") as stream:
        content = stream.read(16385)
    if len(content) > 16384 or hashlib.sha256(content).hexdigest() != expected_hash:
        raise ValueError("Artifact integrity check failed")
    return validate_model(json.loads(content))


def predict(model, xs):
    features(xs, model["dimensions"])
    outputs = []
    for x in xs:
        z = model["bias"] + sum(w * (v - mean) / scale for w, v, mean, scale in
                                zip(model["weights"], x, model["means"], model["scales"]))
        p = sigmoid(z)
        outputs.append({"probability": p, "class": int(p >= 0.5)})
    return outputs


def write_model(path, model):
    content = encode(validate_model(model))
    # Unique attempt ID; exclusive create never overwrites an existing artifact.
    with path.open("xb") as stream:
        stream.write(content)
        stream.flush()
        import os
        os.fsync(stream.fileno())
    path.chmod(0o400)
    return hashlib.sha256(content).hexdigest()
