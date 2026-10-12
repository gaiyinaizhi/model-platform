"""Bounded, deterministic stdlib softmax SGD for 8x8 integer digit features."""
import math
import random
import time

from .common import Problem, features, number, object_fields

RECIPE = "softmax-digits/v1"
CONTRACT = "digits-64x10/v1"
FORMAT = "softmax-json/v1"
SOURCE_URL = "https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits"


def pixels(xs, maximum=128):
    features(xs, 64, maximum, max_dimensions=64)
    for row in xs:
        for value in row:
            if type(value) is not int or not 0 <= value <= 16:
                raise Problem(400, "invalid_input", "Digits require 64 integer pixels in 0..16")


def validate_request(data, config):
    import re
    object_fields(data, ("source", "version", "rights_confirmed", "train", "evaluation", "provenance"))
    provenance = data["provenance"]
    object_fields(provenance, ("archive_sha256", "train_sha256", "evaluation_sha256", "official_split", "license", "source_url"))
    if provenance["official_split"] is not True or provenance["license"] != "CC-BY-4.0" or provenance["source_url"] != SOURCE_URL:
        raise Problem(400, "invalid_input", "Official split and source/license declaration required")
    for key in ("archive_sha256", "train_sha256", "evaluation_sha256"):
        if not isinstance(provenance[key], str) or not re.fullmatch("[0-9a-f]{64}", provenance[key]):
            raise Problem(400, "invalid_input", "Invalid source checksum")
    # No feature deduplication: official writer-disjoint records can quantize alike.
    for split, maximum in (("train", 3823), ("evaluation", 1797)):
        rows = data[split]
        if not isinstance(rows, list) or not 10 <= len(rows) <= maximum:
            raise Problem(400, "invalid_input", "Digits split exceeds recipe budget or lacks classes")
        for row in rows:
            object_fields(row, ("x", "y"))
            if type(row["y"]) is not int or not 0 <= row["y"] <= 9:
                raise Problem(400, "invalid_input", "Digits labels must be integer 0..9")
        pixels([r["x"] for r in rows], maximum)
        if {r["y"] for r in rows} != set(range(10)):
            raise Problem(400, "invalid_input", "Each split must contain all ten classes")
    object_fields(config, ("epochs", "learning_rate", "seed"))
    if type(config["epochs"]) is not int or not 1 <= config["epochs"] <= 60:
        raise Problem(400, "invalid_input", "Digits epochs must be integer 1..60")
    number(config["learning_rate"], 0.001, 0.1)
    if type(config["seed"]) is not int or not 0 <= config["seed"] <= 2**32 - 1:
        raise Problem(400, "invalid_input", "Invalid seed")


def probabilities(weights, biases, x):
    scores = [b + sum(w * v for w, v in zip(ws, x)) for ws, b in zip(weights, biases)]
    peak = max(scores)
    exps = [math.exp(score - peak) for score in scores]
    total = sum(exps)
    return [value / total for value in exps]


def train(rows, config, log):
    from .algorithm import TrainingFailure
    if all(row["x"] == rows[0]["x"] for row in rows):
        raise TrainingFailure("degenerate_features")
    xs = [[v / 16 for v in row["x"]] for row in rows]
    weights, biases = [[0.0] * 64 for _ in range(10)], [0.0] * 10
    order = list(range(len(rows)))
    rng = random.Random(config["seed"])
    start = time.monotonic()
    for epoch in range(config["epochs"]):
        rng.shuffle(order)
        rate = config["learning_rate"] / (1 + epoch * 0.05)
        for i in order:
            x = xs[i]
            ps = probabilities(weights, biases, x)
            for label in range(10):
                delta = rate * (ps[label] - int(rows[i]["y"] == label))
                biases[label] -= delta
                ws = weights[label]
                for j, v in enumerate(x):
                    ws[j] -= delta * v
            if time.monotonic() - start > 120:
                raise TrainingFailure("training_timeout")
        log("train_progress", {"epoch": epoch + 1, "total_epochs": config["epochs"]})
    return {"format": FORMAT, "recipe": RECIPE, "contract": CONTRACT, "dimensions": 64,
            "classes": list(range(10)), "pixel_scale": 16, "weights": weights, "biases": biases}


def validate_model(model):
    object_fields(model, ("format", "recipe", "contract", "dimensions", "classes", "pixel_scale", "weights", "biases"))
    if (model["format"] != FORMAT or model["recipe"] != RECIPE or model["contract"] != CONTRACT or
            type(model["dimensions"]) is not int or model["dimensions"] != 64 or model["classes"] != list(range(10)) or
            any(type(v) is not int for v in model["classes"]) or type(model["pixel_scale"]) is not int or model["pixel_scale"] != 16):
        raise ValueError("Unsupported digits contract")
    if not isinstance(model["weights"], list) or len(model["weights"]) != 10 or not isinstance(model["biases"], list) or len(model["biases"]) != 10:
        raise ValueError("Invalid digits model vectors")
    for row in model["weights"]:
        if not isinstance(row, list) or len(row) != 64:
            raise ValueError("Invalid digits weights")
        for value in row:
            number(value, -1e6, 1e6)
    for value in model["biases"]:
        number(value, -1e6, 1e6)
    return model


def predict(model, xs):
    pixels(xs)
    outputs = []
    for x in xs:
        ps = probabilities(model["weights"], model["biases"], [v / 16 for v in x])
        outputs.append({"probabilities": ps, "class": max(range(10), key=ps.__getitem__)})
    return outputs
