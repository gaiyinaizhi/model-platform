"""Registry contracts and dispatch boundaries, without downloads or ML imports.

The existing binary/digits lifecycle suites exercise actual numeric training;
optional Fashion tests exercise synthetic ONNX serving, not recognition quality.
Here the missing-dependency and interrupted-attempt cases use real HTTP/SQLite
and worker subprocesses, with dependency imports deliberately blocked.
"""
import copy
from dataclasses import FrozenInstanceError, asdict, replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from model_platform import algorithm, digits, fashion, recipes, detection
from model_platform.common import CONTRACT, RECIPE, Problem, digest, encode, training_request
from scripts.demo import request_body
import test_digits
import test_platform


EXPECTED_BUDGETS = {
    RECIPE: recipes.Budgets(1024 * 1024, 1000, 1000, 1000, 16, 200, 128, 60, 32768),
    digits.RECIPE: recipes.Budgets(2 * 1024 * 1024, 3823, 1797, None, 64, 60, 128, 120, 32768),
    detection.RECIPE: recipes.Budgets(2 * 1024 * 1024, 128, 32, None, 76800, 20, 1, 600, 8 * 1024 * 1024, 120, 180),
    fashion.RECIPE: recipes.Budgets(2 * 1024 * 1024, 6000, 10000, None, 784, 5, 32, 600,
                                   2 * 1024 * 1024, 120, 120),
}


class RegistryTests(unittest.TestCase):
    def test_exact_builtins_metadata_and_nested_immutability(self):
        self.assertEqual(recipes.recipe_ids(), [RECIPE, digits.RECIPE, fashion.RECIPE, detection.RECIPE, "yolox-nano-dataset/v1"])
        expected_contracts = {RECIPE: (CONTRACT, "logistic-json/v1", 2, "inline"),
                              digits.RECIPE: (digits.CONTRACT, digits.FORMAT, 10, "inline"),
                              fashion.RECIPE: (fashion.CONTRACT, fashion.FORMAT, 10, "fixed-reference"),
                              detection.RECIPE: (detection.CONTRACT, detection.FORMAT, 1, "fixed-reference")}
        for recipe_id, expected in expected_contracts.items():
            spec = recipes.get_recipe(recipe_id)
            self.assertEqual((spec.contract, spec.format, spec.classes, spec.data_mode), expected)
            self.assertEqual(spec.budgets, EXPECTED_BUDGETS[recipe_id])
            with self.assertRaises(FrozenInstanceError):
                spec.id = "replacement/v1"
            with self.assertRaises(FrozenInstanceError):
                spec.budgets.prediction_rows = 999
            with self.assertRaises(TypeError):
                recipes.REGISTRY[recipe_id] = spec
        self.assertEqual(recipes.get_recipe(RECIPE).optional_dependencies, ())
        self.assertEqual(recipes.get_recipe(digits.RECIPE).optional_dependencies, ())
        self.assertEqual(recipes.get_recipe(fashion.RECIPE).optional_dependencies,
                         ("torch", "numpy", "onnx", "onnxruntime", "onnxscript"))

    def test_registry_builder_is_isolated_and_rejects_duplicate_names(self):
        base = recipes.get_recipe(RECIPE)
        candidate = replace(base, id="source-defined/v1", contract="source-contract/v1",
                            format="source-format/v1")
        source = [base, candidate]
        registry = recipes.build_registry(source)
        source.clear()
        self.assertEqual(list(registry), [RECIPE, candidate.id])
        self.assertIs(registry[candidate.id], candidate)
        self.assertNotIn(candidate.id, recipes.REGISTRY)
        with self.assertRaises(TypeError):
            registry[candidate.id] = base
        for duplicate in (base, replace(candidate, id=base.id),
                          replace(candidate, format=base.format),
                          replace(candidate, contract=base.contract)):
            with self.subTest(duplicate=duplicate.id):
                with self.assertRaises(ValueError):
                    recipes.build_registry((base, duplicate))
        self.assertEqual(recipes.recipe_ids(), [RECIPE, digits.RECIPE, fashion.RECIPE, detection.RECIPE, "yolox-nano-dataset/v1"])

    def test_registry_rejects_malformed_source_registration(self):
        base = recipes.get_recipe(RECIPE)
        bad = [None, {}, "module:callable"]
        for field in ("id", "contract", "format", "runtime"):
            for value in ("", None, 1, []):
                bad.append(replace(base, **{field: value}))
        for value in ("module:callable", "unversioned", "upperCase/v1", "recipe/v0", "../recipe/v1"):
            bad.append(replace(base, id=value))
        for value in (None, True, 1, 2.0, "2"):
            bad.append(replace(base, classes=value))
        for value in (None, "uploaded", "dynamic-import"):
            bad.append(replace(base, data_mode=value))
        for value in ([], ["numpy"], "numpy", (None,), ("",)):
            bad.append(replace(base, optional_dependencies=value))
        bad.append(replace(base, budgets={}))
        for field in asdict(base.budgets):
            values = (0, -1, True, 1.5, "1", None)
            if field == "total_rows":
                values = (0, -1, True, 1.5, "1")
            elif field in ("export_seconds", "evaluation_seconds"):
                values = (-1, True, 1.5, "1", None)
            for value in values:
                bad.append(replace(base, budgets=replace(base.budgets, **{field: value})))
        for field in ("validate_request", "execute", "load", "predict", "validate_features"):
            for value in (None, "module:function", 1):
                bad.append(replace(base, **{field: value}))
        bad.extend((replace(base, train=None), replace(base, validate_model=None),
                    replace(base, train="module:function", validate_model="module:validate")))
        for index, spec in enumerate(bad):
            with self.subTest(index=index):
                with self.assertRaises(ValueError):
                    recipes.build_registry((spec,))

    def test_catalog_is_json_safe_detached_and_does_not_import_optional_code(self):
        original = recipes.catalog()
        self.assertEqual(json.loads(json.dumps(original)), original)
        for item in original:
            self.assertEqual(set(item), {"id", "contract", "format", "runtime", "classes", "data_mode",
                                         "optional_dependencies", "budgets", "capabilities"})
            self.assertEqual(item["capabilities"], {"train": True, "evaluate": True, "serve_cpu": True,
                                                    "cancel": False, "device": False})
        changed = recipes.catalog()
        changed[0]["budgets"]["prediction_rows"] = 1
        changed[-1]["optional_dependencies"].append("not-installed")
        changed[0]["capabilities"]["device"] = True
        changed.pop()
        self.assertEqual(recipes.catalog(), original)
        ids = recipes.recipe_ids()
        ids.clear()
        self.assertEqual(len(recipes.recipe_ids()), 5)
        script = """
import builtins
import json
import sys
original_import = builtins.__import__
blocked = {'torch', 'numpy', 'onnx', 'onnxruntime', 'onnxscript', 'PIL', 'pycocotools'}
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in blocked:
        raise AssertionError('optional dependency imported: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded
from model_platform import recipes
from model_platform.common import environment, training_request
from model_platform import api, worker, fashion
json.dumps(recipes.catalog())
assert environment()['supported_recipes'] == recipes.recipe_ids()
training_request(fashion.request_body())
assert not blocked.intersection(sys.modules)
assert 'model_platform.fashion_training' not in sys.modules
assert 'model_platform.recipe_execution' not in sys.modules
"""
        result = subprocess.run([sys.executable, "-c", script], cwd=test_platform.ROOT,
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_unknown_and_nonstring_recipe_lookup_is_a_client_error(self):
        for value in ("unregistered/v1", "module:function", "", None, True, 1, 1.0, {}, [], [RECIPE]):
            with self.subTest(value=value):
                with self.assertRaises(Problem) as caught:
                    recipes.get_recipe(value)
                self.assertEqual((caught.exception.status, caught.exception.code), (400, "invalid_input"))
                self.assertEqual(caught.exception.message, "Unsupported recipe")

    def test_manifest_dispatch_requires_matching_recipe_contract_and_format(self):
        for spec in recipes.REGISTRY.values():
            manifest = {"recipe": spec.id, "contract": spec.contract, "format": spec.format}
            self.assertIs(recipes.for_manifest(manifest), spec)
            for field in ("contract", "format"):
                changed = dict(manifest, **{field: "different/v1"})
                with self.assertRaises(ValueError):
                    recipes.for_manifest(changed)
            for value in ("unknown/v1", None, []):
                with self.assertRaises(Problem):
                    recipes.for_manifest(dict(manifest, recipe=value))

    def test_legacy_default_request_is_not_normalized_or_digest_changed(self):
        body = request_body()
        original = copy.deepcopy(body)
        self.assertEqual(digest(body), "1d0cebd791a9d712f4927f22133b5e336198fbe9e86545b21f07f0d899c9d90f")
        self.assertIs(training_request(body), body)
        self.assertEqual(body, original)
        self.assertNotIn("recipe", body)
        self.assertEqual(digest(body), digest(original))
        explicit = dict(body, recipe=RECIPE)
        self.assertIs(training_request(explicit), explicit)
        self.assertNotEqual(digest(explicit), digest(body))
        self.assertEqual(recipes.get_recipe(RECIPE).counts(body["dataset"]), (8, 4))
        self.assertEqual(recipes.get_recipe(fashion.RECIPE).counts(fashion.request_body()["dataset"]),
                         (6000, 10000))

    def test_declared_online_and_epoch_budgets_are_enforced(self):
        factories = {RECIPE: request_body, digits.RECIPE: test_digits.synthetic_request,
                     fashion.RECIPE: fashion.request_body, detection.RECIPE: detection.request_body}
        for spec in recipes.REGISTRY.values():
            with self.subTest(recipe=spec.id):
                if spec.id in (detection.RECIPE, 'yolox-nano-dataset/v1'):
                    spec.validate_features(detection.probe_input(), spec.budgets.max_dimensions)
                    with self.assertRaises(Problem):
                        spec.validate_features(detection.probe_input() * 2, spec.budgets.max_dimensions)
                    if spec.id == detection.RECIPE:
                        body = detection.request_body()
                    else:
                        from model_platform.dataset_detection import request_body as dataset_request
                        body = dataset_request('dsv_'+'0'*32)
                        body['config']['epochs'] = spec.budgets.max_epochs
                    training_request(body)
                    body["config"]["epochs"] += 1
                    with self.assertRaises(Problem):
                        training_request(body)
                    continue
                width = 1 if spec.id == RECIPE else spec.budgets.max_dimensions
                spec.validate_features([[0] * width] * spec.budgets.prediction_rows, width)
                with self.assertRaises(Problem):
                    spec.validate_features([[0] * width] * (spec.budgets.prediction_rows + 1), width)
                with self.assertRaises(Problem):
                    spec.validate_features([[0] * (spec.budgets.max_dimensions + 1)], width)
                body = factories[spec.id]()
                body["config"]["epochs"] = spec.budgets.max_epochs
                training_request(body)
                body["config"]["epochs"] += 1
                with self.assertRaises(Problem):
                    training_request(body)

    def test_declared_split_budgets_are_enforced_without_training(self):
        binary = recipes.get_recipe(RECIPE).budgets
        body = request_body()
        body["dataset"]["train"] = [{"x": [i + 10], "y": i % 2}
                                      for i in range(binary.total_rows - 4)]
        training_request(body)
        body["dataset"]["train"].append({"x": [binary.total_rows + 20], "y": 0})
        with self.assertRaises(Problem):
            training_request(body)
        for split in ("train", "evaluation"):
            maximum = getattr(recipes.get_recipe(digits.RECIPE).budgets, split + "_rows")
            body = test_digits.synthetic_request()
            body["dataset"][split] = test_digits.synthetic_rows(maximum)
            training_request(body)
            body["dataset"][split].append(body["dataset"][split][0])
            with self.assertRaises(Problem):
                training_request(body)

    def test_request_size_validation_reads_registered_budget(self):
        # The historical binary request cap exceeds normal bounded fixtures.
        # A smaller source-defined budget proves enforcement uses the registry.
        body = request_body()
        size = len(encode(body))
        base = recipes.get_recipe(RECIPE)
        for limit, accepted in ((size, True), (size - 1, False)):
            spec = replace(base, budgets=replace(base.budgets, request_bytes=limit))
            registry = recipes.build_registry((spec,))
            with mock.patch.object(recipes, "REGISTRY", registry):
                if accepted:
                    self.assertIs(training_request(body), body)
                else:
                    with self.assertRaises(Problem) as caught:
                        training_request(body)
                    self.assertEqual((caught.exception.status, caught.exception.code), (413, "too_large"))

    def test_numeric_deadlines_use_declared_training_budgets(self):
        for module, factory in ((algorithm, request_body), (digits, test_digits.synthetic_request)):
            spec = recipes.get_recipe(RECIPE if module is algorithm else digits.RECIPE)
            body = factory()
            body["config"]["epochs"] = 1
            with self.subTest(recipe=spec.id), mock.patch.object(
                    module.time, "monotonic", side_effect=[0, spec.budgets.training_seconds + 1]):
                with self.assertRaises(algorithm.TrainingFailure) as caught:
                    spec.train(body["dataset"]["train"], body["config"], lambda *_: None)
                self.assertEqual(caught.exception.code, "training_timeout")

    def test_declared_artifact_limits_fail_before_parsing_or_optional_imports(self):
        local = test_platform.ROOT / ".runtime" / "tests"
        local.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=local) as directory:
            for spec in recipes.REGISTRY.values():
                with self.subTest(recipe=spec.id):
                    path = Path(directory) / "oversized-artifact"
                    content = b"x" * (spec.budgets.artifact_bytes + 1)
                    path.write_bytes(content)
                    with self.assertRaises(ValueError):
                        spec.load(path, hashlib.sha256(content).hexdigest())

    def test_fashion_phase_budgets_match_cooperative_enforcement(self):
        from model_platform import fashion_training
        budget = recipes.get_recipe(fashion.RECIPE).budgets
        self.assertEqual(fashion_training.TRAIN_COUNT, budget.train_rows)
        self.assertEqual(fashion_training.EPOCHS, budget.max_epochs)
        self.assertEqual(fashion_training.MAX_ARTIFACT_BYTES, budget.artifact_bytes)
        for constant, field, code in (("TRAINING_SECONDS", "training_seconds", "training_timeout"),
                                      ("EXPORT_SECONDS", "export_seconds", "fashion_export_timeout"),
                                      ("EVALUATION_SECONDS", "evaluation_seconds", "fashion_evaluation_timeout")):
            seconds = getattr(budget, field)
            self.assertEqual(getattr(fashion_training, constant), seconds)
            with mock.patch.object(fashion_training.time, "monotonic", return_value=seconds):
                fashion_training._deadline(0, seconds, code)
            with mock.patch.object(fashion_training.time, "monotonic", return_value=seconds + 1):
                with self.assertRaises(algorithm.TrainingFailure) as caught:
                    fashion_training._deadline(0, seconds, code)
                self.assertEqual(caught.exception.code, code)


class RegistryHttpTests(unittest.TestCase):
    # Reuse helpers only. Inheriting PlatformTests would rerun its entire suite.
    setUp = test_platform.PlatformTests.setUp
    spawn = test_platform.PlatformTests.spawn
    start_api = test_platform.PlatformTests.start_api
    cleanup = test_platform.PlatformTests.cleanup
    call = test_platform.PlatformTests.call
    submit = test_platform.PlatformTests.submit
    worker_once = test_platform.PlatformTests.worker_once

    def test_unknown_nonstring_and_code_path_recipes_return_http_400(self):
        for index, value in enumerate(("unregistered/v1", "module:function", "../recipe.py", "", None,
                                       True, 1, 1.5, {}, [], [RECIPE])):
            body = request_body()
            body["recipe"] = value
            with self.subTest(recipe=value):
                status, response = self.call("/runs", body, f"bad-{index}")
                self.assertEqual(status, 400, response)
                self.assertEqual(response, {"error": {"code": "invalid_input", "message": "Unsupported recipe"}})
        self.assertEqual(self.call("/runs")[1]["ids"], [])
        health = self.call("/health")[1]
        self.assertEqual(health["recipe"], RECIPE)
        self.assertEqual(health["contract"], CONTRACT)
        self.assertEqual(health["recipes"], recipes.recipe_ids())
        self.assertFalse(health["cancel_supported"])
        # The catalog is internal source data, not a registration/upload API.
        self.assertEqual(self.call("/recipes", {"id": "uploaded/v1"})[0], 404)

    def test_legacy_http_snapshot_and_idempotency_keep_original_digest(self):
        body = request_body()
        run = self.submit(body, "legacy")
        self.assertEqual(run["recipe"], RECIPE)
        self.assertEqual(run["request_sha256"], digest(body))
        self.assertEqual(run["dataset"]["train_count"], 8)
        self.assertEqual(run["dataset"]["evaluation_count"], 4)
        self.assertEqual(run["environment"]["supported_recipes"], recipes.recipe_ids())
        with self.runtime.connection() as db:
            stored = db.execute("SELECT input_json,request_sha256 FROM runs WHERE id=?", (run["id"],)).fetchone()
        self.assertEqual(stored["input_json"], encode(body).decode())
        self.assertNotIn("recipe", json.loads(stored["input_json"]))
        self.assertEqual(stored["request_sha256"], digest(body))
        status, replay = self.call("/runs", body, "legacy")
        self.assertEqual((status, replay["id"]), (200, run["id"]))
        self.assertEqual(self.call("/runs", dict(body, recipe=RECIPE), "legacy")[0], 409)

    def blocked_worker_once(self):
        # Even an optional-ML environment must exercise dependency absence.
        # No artificial training result is registered, and no dataset is read.
        script = """
import builtins
import socket
from model_platform import fashion_data

def forbidden(*args, **kwargs):
    raise AssertionError('Dataset access or network is forbidden in this regression')
fashion_data.load_dataset = forbidden
fashion_data.fetch_dataset = forbidden
socket.create_connection = forbidden
socket.socket.connect = forbidden
original_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'numpy', 'onnx', 'onnxruntime', 'onnxscript'}:
        raise ModuleNotFoundError('intentionally absent optional dependency')
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded
from model_platform.common import Runtime
from model_platform.worker import run_worker
run_worker(Runtime(), once=True)
"""
        result = subprocess.run([sys.executable, "-c", script], env=self.env, cwd=test_platform.ROOT,
                                capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_missing_ml_dependency_failure_retry_and_recovery_remain_recipe_scoped(self):
        body = fashion.request_body()
        run = self.submit(body, "missing-ml")
        self.blocked_worker_once()
        failed = self.call("/runs/" + run["id"])[1]
        self.assertEqual(failed["status"], "failed", failed)
        self.assertIsNone(failed["model_version_id"])
        first = copy.deepcopy(failed["attempts"][0])
        self.assertEqual(first["error_code"], "optional_dependencies_missing")
        self.assertEqual(first["number"], 1)
        self.assertEqual(first["logs"][-1]["details"], {"code": "optional_dependencies_missing"})
        self.assertIn({"recipe": fashion.RECIPE}, [entry["details"] for entry in first["logs"]])
        self.assertEqual(self.call("/models")[1]["ids"], [])
        self.assertEqual(list(self.runtime.artifacts.iterdir()), [])
        self.assertEqual(self.call("/runs/" + run["id"] + "/retry", {}, tenant="other")[0], 404)
        status, retried = self.call("/runs/" + run["id"] + "/retry", {})
        self.assertEqual(status, 202)
        self.assertEqual(retried["attempts"], [first])
        self.assertEqual(retried["request_sha256"], digest(body))
        self.assertEqual(retried["environment"], run["environment"])
        self.assertEqual(self.call("/runs/" + run["id"] + "/retry", {})[0], 409)
        # Deterministically simulate a worker exiting after claim. Restart uses
        # the real recovery path and must not automatically retry this attempt.
        script = """
from model_platform.common import Runtime
from model_platform.store import Store
runtime = Runtime()
with runtime.lock('worker'):
    assert Store(runtime).claim() is not None
"""
        result = subprocess.run([sys.executable, "-c", script], env=self.env, cwd=test_platform.ROOT,
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        binary = self.submit(request_body(), "base-after-ml-failure")
        self.blocked_worker_once()
        recovered = self.call("/runs/" + run["id"])[1]
        self.assertEqual(recovered["status"], "failed")
        self.assertEqual(recovered["attempts"][0], first)
        self.assertEqual([a["error_code"] for a in recovered["attempts"]],
                         ["optional_dependencies_missing", "interrupted"])
        self.assertEqual(self.call("/runs/" + binary["id"])[1]["status"], "succeeded")
        self.assertEqual(self.call("/runs/" + run["id"] + "/retry", {})[0], 202)
        self.blocked_worker_once()
        final = self.call("/runs/" + run["id"])[1]
        self.assertEqual([a["error_code"] for a in final["attempts"]],
                         ["optional_dependencies_missing", "interrupted", "optional_dependencies_missing"])
        self.assertEqual([a["number"] for a in final["attempts"]], [1, 2, 3])
        self.assertEqual(final["attempts"][:2], recovered["attempts"])
        self.assertEqual(final["request_sha256"], digest(body))
        self.assertIsNone(final["model_version_id"])
        self.assertEqual(self.call("/runs/" + run["id"] + "/retry", {})[0], 409)
        self.assertEqual(self.call("/runs/" + run["id"] + "/cancel", {})[0], 409)
        self.assertEqual(len(self.call("/models")[1]["ids"]), 1)
        self.assertEqual(len(list(self.runtime.artifacts.iterdir())), 1)


if __name__ == "__main__":
    unittest.main()
