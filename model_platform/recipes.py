"""Internal, source-reviewed recipe protocol and immutable built-in allowlist.

Adding a recipe is a source change, never an HTTP/plugin/upload operation. All
callbacks below are explicit local imports; no user-controlled import paths.
Importing the catalog requires only the standard library.
"""
from dataclasses import dataclass, asdict
import re
from types import MappingProxyType
from typing import Callable

from . import algorithm, digits, fashion, detection
from .common import CONTRACT, RECIPE, MAX_BODY, Problem, features, validate_binary_request


@dataclass(frozen=True)
class Budgets:
    request_bytes: int
    train_rows: int
    evaluation_rows: int
    total_rows: int | None
    max_dimensions: int
    max_epochs: int
    prediction_rows: int
    training_seconds: int
    artifact_bytes: int
    export_seconds: int = 0
    evaluation_seconds: int = 0


@dataclass(frozen=True)
class RecipeSpec:
    id: str
    contract: str
    format: str
    runtime: str
    classes: int
    data_mode: str
    optional_dependencies: tuple[str, ...]
    budgets: Budgets
    validate_request: Callable
    execute: Callable
    load: Callable
    predict: Callable
    validate_features: Callable
    train: Callable | None = None
    validate_model: Callable | None = None
    probe_input: Callable | None = None
    probe_output: Callable | None = None

    def readiness_input(self, dimensions):
        return self.probe_input() if self.probe_input else [[0] * dimensions]

    def validate_readiness(self, predictions):
        if self.probe_output:
            return self.probe_output(predictions)
        if not isinstance(predictions, list) or len(predictions) != 1:
            raise ValueError("Invalid probe batch")
        output = predictions[0]
        if self.classes > 2:
            ps = output["probabilities"]
            if (type(output["class"]) is not int or output["class"] not in range(self.classes)
                    or len(ps) != self.classes or not all(0 <= p <= 1 for p in ps)
                    or abs(sum(ps) - 1) > 1e-8):
                raise ValueError("Invalid classification probe output")
        elif output["class"] not in (0, 1) or not 0 <= output["probability"] <= 1:
            raise ValueError("Invalid binary probe output")

    def counts(self, dataset):
        if self.data_mode == "fixed-reference":
            return self.budgets.train_rows, self.budgets.evaluation_rows
        return len(dataset["train"]), len(dataset["evaluation"])

    def describe(self):
        """Detached JSON-safe metadata, without code callbacks or local paths."""
        return {"id": self.id, "contract": self.contract, "format": self.format,
                "runtime": self.runtime, "classes": self.classes, "data_mode": self.data_mode,
                "optional_dependencies": list(self.optional_dependencies),
                "budgets": asdict(self.budgets),
                "capabilities": {"train": True, "evaluate": True, "serve_cpu": True,
                                 "cancel": False, "device": False}}


def _numeric(*args):
    from .recipe_execution import execute_numeric
    return execute_numeric(*args)


def _fashion(*args):
    from .recipe_execution import execute_fashion
    return execute_fashion(*args)


def _detection(*args):
    from .detection_training import train_export
    from .algorithm import TrainingFailure
    try:
        return train_export(*args)
    except ImportError:
        raise TrainingFailure("optional_dependencies_missing") from None


def _binary_features(xs, dimensions):
    return features(xs, dimensions, maximum=get_recipe(RECIPE).budgets.prediction_rows)


def _digits_features(xs, dimensions):
    return digits.pixels(xs)


def _fashion_features(xs, dimensions):
    return fashion.pixels(xs)


def build_registry(specs):
    """Validate a source-defined catalog; return a read-only mapping.

    This does not mutate the active registry. There is no public registration
    endpoint, environment hook, entry-point discovery, or dynamic import.
    """
    entries, formats, contracts = {}, set(), set()
    for spec in specs:
        if not isinstance(spec, RecipeSpec):
            raise ValueError("Expected a RecipeSpec")
        if not all(isinstance(value, str) and value for value in (spec.id, spec.contract, spec.format, spec.runtime)):
            raise ValueError("Recipe metadata must be nonempty strings")
        if spec.id in entries or spec.format in formats or spec.contract in contracts:
            raise ValueError("Duplicate recipe, format or contract")
        if not re.fullmatch(r"[a-z][a-z0-9-]*/v[1-9][0-9]*", spec.id):
            raise ValueError("Invalid versioned recipe ID")
        if type(spec.classes) is not int or spec.data_mode not in ("inline", "fixed-reference") or spec.classes < 1:
            raise ValueError("Invalid recipe metadata")
        if not isinstance(spec.budgets, Budgets) or any(
                type(value) is not int or value <= 0
                for name, value in asdict(spec.budgets).items()
                if name not in ("total_rows", "export_seconds", "evaluation_seconds")):
            raise ValueError("Invalid recipe budgets")
        if (spec.budgets.total_rows is not None and
                (type(spec.budgets.total_rows) is not int or spec.budgets.total_rows <= 0)):
            raise ValueError("Invalid total row budget")
        if any(type(value) is not int or value < 0 for value in (
                spec.budgets.export_seconds, spec.budgets.evaluation_seconds)):
            raise ValueError("Invalid phase budget")
        if (not isinstance(spec.optional_dependencies, tuple) or
                any(not isinstance(value, str) or not value for value in spec.optional_dependencies)):
            raise ValueError("Dependencies must be an immutable tuple of names")
        if any(not callable(getattr(spec, name)) for name in (
                "validate_request", "execute", "load", "predict", "validate_features")):
            raise ValueError("Missing recipe callback")
        if (spec.train is None) != (spec.validate_model is None):
            raise ValueError("JSON recipe requires training and model validation together")
        if any(value is not None and not callable(value) for value in (spec.train, spec.validate_model)):
            raise ValueError("Invalid recipe callback")
        if ((spec.probe_input is None) != (spec.probe_output is None)
                or any(value is not None and not callable(value)
                       for value in (spec.probe_input, spec.probe_output))):
            raise ValueError("Custom readiness requires both input and output callbacks")
        if spec.classes == 1 and spec.probe_output is None:
            raise ValueError("Single-class recipe requires a task-specific readiness probe")
        entries[spec.id] = spec
        formats.add(spec.format)
        contracts.add(spec.contract)
    return MappingProxyType(entries)


REGISTRY = build_registry((
    RecipeSpec(RECIPE, CONTRACT, "logistic-json/v1", "python-cpu-json/v1", 2, "inline", (),
               Budgets(1024 * 1024, 1000, 1000, 1000, 16, 200, 128, 60, 32768),
               validate_binary_request, _numeric, algorithm.load_model, algorithm.predict_binary,
               _binary_features, algorithm.train, algorithm.validate_binary_model),
    RecipeSpec(digits.RECIPE, digits.CONTRACT, digits.FORMAT, "python-cpu-json/v1", 10, "inline", (),
               Budgets(MAX_BODY, 3823, 1797, None, 64, 60, 128, 120, 32768),
               digits.validate_request, _numeric, algorithm.load_model, digits.predict,
               _digits_features, digits.train, digits.validate_model),
    RecipeSpec(fashion.RECIPE, fashion.CONTRACT, fashion.FORMAT, "onnxruntime-cpu/v1", 10,
               "fixed-reference", ("torch", "numpy", "onnx", "onnxruntime", "onnxscript"),
               Budgets(MAX_BODY, 6000, 10000, None, 784, 5, 32, 600, 2 * 1024 * 1024, 120, 120),
               fashion.validate_request, _fashion, fashion.load_model, fashion.predict, _fashion_features),
    RecipeSpec(detection.RECIPE, detection.CONTRACT, detection.FORMAT, "onnxruntime-cpu/v1", 1,
               "fixed-reference", ("torch", "numpy", "onnx", "onnxruntime", "Pillow", "pycocotools"),
               Budgets(MAX_BODY, 128, 32, None, 160 * 160 * 3, 20, 1, 600, detection.MAX_ARTIFACT, 120, 180),
               detection.validate_request, _detection, detection.load_model, detection.predict, detection.images,
               probe_input=detection.probe_input, probe_output=detection.probe_output),
))


def get_recipe(recipe_id):
    if not isinstance(recipe_id, str) or recipe_id not in REGISTRY:
        raise Problem(400, "invalid_input", "Unsupported recipe")
    return REGISTRY[recipe_id]


def recipe_ids():
    return list(REGISTRY)


def catalog():
    return [spec.describe() for spec in REGISTRY.values()]


def for_manifest(manifest):
    spec = get_recipe(manifest.get("recipe"))
    if manifest.get("contract") != spec.contract or manifest.get("format") != spec.format:
        raise ValueError("Model manifest contract mismatch")
    return spec
