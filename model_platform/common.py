"""Bounded JSON contracts, local configuration and single-host ownership."""
import contextlib
import fcntl
import hashlib
import json
import math
import os
import platform
import re
import sqlite3
import time
from pathlib import Path

MAX_BODY = 1024 * 1024
MAX_ACTIVE_RUNS = 32
MAX_DEPLOYMENTS = 4
MAX_ATTEMPTS = 3
RECIPE = "logistic-regression/v1"
CONTRACT = "binary-numeric/v1"
ID = re.compile(r"^[a-z]+_[0-9a-f]{32}$")


class Problem(Exception):
    def __init__(self, status, code, message):
        self.status, self.code, self.message = status, code, message
        super().__init__(message)


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(encode(value)).hexdigest()


def object_fields(value, required, optional=()):
    if not isinstance(value, dict) or not set(required) <= value.keys() or value.keys() - set(required) - set(optional):
        raise Problem(400, "invalid_input", "Missing or unknown fields")


def number(value, low, high):
    if type(value) not in (int, float) or not low <= value <= high or not math.isfinite(value):
        raise Problem(400, "invalid_input", "Expected a finite bounded number")
    return value


def text_value(value, name, limit=160):
    if not isinstance(value, str) or not 1 <= len(value) <= limit or any(ord(c) < 32 for c in value):
        raise Problem(400, "invalid_input", f"Invalid {name}")
    return value


def features(value, dimensions=None, maximum=128):
    if not isinstance(value, list) or not 1 <= len(value) <= maximum:
        raise Problem(400, "invalid_input", "Invalid feature row count")
    width = dimensions
    for row in value:
        if not isinstance(row, list) or not 1 <= len(row) <= 16:
            raise Problem(400, "invalid_input", "Invalid feature dimensions")
        width = width or len(row)
        if len(row) != width:
            raise Problem(400, "invalid_input", "Inconsistent feature dimensions")
        for x in row:
            number(x, -1e6, 1e6)
    return width


def samples(value, dimensions=None):
    if not isinstance(value, list) or not 4 <= len(value) <= 1000:
        raise Problem(400, "invalid_input", "Each split requires 4..1000 samples")
    xs, ys = [], []
    for row in value:
        object_fields(row, ("x", "y"))
        if type(row["y"]) is not int or row["y"] not in (0, 1):
            raise Problem(400, "invalid_input", "Labels must be integer 0 or 1")
        xs.append(row["x"])
        ys.append(row["y"])
    width = features(xs, dimensions, 1000)
    if set(ys) != {0, 1}:
        raise Problem(400, "invalid_input", "Each split must include both classes")
    return width


def training_request(body):
    object_fields(body, ("dataset", "config"), ("business_ref",))
    data = body["dataset"]
    object_fields(data, ("source", "version", "rights_confirmed", "train", "evaluation"))
    text_value(data["source"], "dataset source")
    text_value(data["version"], "dataset version")
    if data["rights_confirmed"] is not True:
        raise Problem(400, "invalid_input", "Legal data usage confirmation required")
    width = samples(data["train"])
    samples(data["evaluation"], width)
    if len(data["train"]) + len(data["evaluation"]) > 1000:
        raise Problem(400, "invalid_input", "At most 1000 samples in total")
    train_x = {tuple(row["x"]) for row in data["train"]}
    if any(tuple(row["x"]) in train_x for row in data["evaluation"]):
        raise Problem(400, "invalid_input", "Training/evaluation features overlap")
    config = body["config"]
    object_fields(config, ("epochs", "learning_rate"))
    if type(config["epochs"]) is not int or not 1 <= config["epochs"] <= 200:
        raise Problem(400, "invalid_input", "epochs must be integer 1..200")
    number(config["learning_rate"], 0.001, 1.0)
    if "business_ref" in body:
        text_value(body["business_ref"], "business_ref")
    return body


def environment():
    root = Path(__file__).parent
    code = hashlib.sha256()
    for path in sorted(root.glob("*.py")):
        code.update(path.name.encode())
        code.update(path.read_bytes())
    return {"python": platform.python_version(), "platform": platform.platform(),
            "code_sha256": code.hexdigest(), "dependencies": "stdlib-only",
            "recipe": RECIPE, "contract": CONTRACT}


class Runtime:
    def __init__(self, home=None):
        self.home = Path(home or os.environ.get("MP_HOME", ".runtime")).resolve()
        self.home.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.artifacts = self.home / "artifacts"
        self.artifacts.mkdir(mode=0o700, exist_ok=True)
        self.db = self.home / "platform.sqlite3"

    @contextlib.contextmanager
    def connection(self):
        conn = sqlite3.connect(self.db, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    @contextlib.contextmanager
    def lock(self, name):
        with (self.home / f"{name}.lock").open("a") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError(f"Another {name} owns this runtime") from None
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def artifact(self, attempt_id):
        if not isinstance(attempt_id, str) or not ID.fullmatch(attempt_id):
            raise ValueError("Invalid internal artifact ID")
        return self.artifacts / f"{attempt_id}.json"


def now():
    return time.time()
