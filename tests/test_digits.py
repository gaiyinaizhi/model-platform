"""Digits contract and real local lifecycle tests using synthetic fixtures only.

These small one-hot examples are deliberately NOT the official UCI dataset and
do not establish digit-recognition quality. Provenance fields below exercise
caller declarations; the API does not authenticate external source archives.
The separate official-data demo is responsible for checking actual downloads.
"""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import tempfile
import time
import unittest

from model_platform import algorithm, digits
from model_platform.common import CONTRACT, RECIPE, Problem, digest, encode, training_request
from model_platform.evaluation import evaluate
from scripts.demo import request_body
import test_platform


def synthetic_rows(count=10, *, evaluation=False):
    rows = []
    for i in range(count):
        label = i % 10
        xs = [0] * 64
        xs[label] = 16
        xs[10] = 2 if evaluation else 1
        xs[11] = (i // 10) % 17
        rows.append({"x": xs, "y": label})
    return rows


def synthetic_request(train_count=10, evaluation_count=10):
    return {
        "recipe": digits.RECIPE,
        "dataset": {
            "source": "synthetic-contract-test-only-not-official-data",
            "version": "synthetic-onehot-v1",
            "rights_confirmed": True,
            "train": synthetic_rows(train_count),
            "evaluation": synthetic_rows(evaluation_count, evaluation=True),
            "provenance": {
                "archive_sha256": hashlib.sha256(b"synthetic archive declaration").hexdigest(),
                "train_sha256": hashlib.sha256(b"synthetic train declaration").hexdigest(),
                "evaluation_sha256": hashlib.sha256(b"synthetic evaluation declaration").hexdigest(),
                "official_split": True,
                "license": "CC-BY-4.0",
                "source_url": digits.SOURCE_URL,
            },
        },
        "config": {"epochs": 8, "learning_rate": 0.05, "seed": 42},
    }


class DigitsContractTests(unittest.TestCase):
    def reject(self, body):
        with self.assertRaises(Problem) as caught:
            training_request(body)
        self.assertEqual(caught.exception.status, 400)
        self.assertEqual(caught.exception.code, "invalid_input")

    def test_recipe_is_explicit_and_legacy_default_still_valid(self):
        body = synthetic_request()
        self.assertIs(training_request(body), body)
        del body["recipe"]
        self.reject(body)
        for value in ("softmax-digits/v2", "python:anything", "", None, True, {}, []):
            with self.subTest(recipe=value):
                body = synthetic_request()
                body["recipe"] = value
                self.reject(body)
        legacy = request_body()
        self.assertIs(training_request(legacy), legacy)
        legacy["recipe"] = RECIPE
        self.assertIs(training_request(legacy), legacy)
        legacy["config"]["seed"] = 42
        self.reject(legacy)

    def test_exact_shape_and_integer_pixel_domain_in_each_split(self):
        for split in ("train", "evaluation"):
            for width in (0, 1, 63, 65):
                with self.subTest(split=split, width=width):
                    body = synthetic_request()
                    body["dataset"][split][0]["x"] = [0] * width
                    self.reject(body)
            for value in (-1, 17, 0.0, 16.0, True, "1", None, float("nan"), float("inf"), 10**1000):
                with self.subTest(split=split, pixel=repr(value)[:50]):
                    body = synthetic_request()
                    body["dataset"][split][0]["x"][0] = value
                    self.reject(body)
        body = synthetic_request()
        body["dataset"]["train"][0]["x"] = [0, 16] * 32
        self.assertIs(training_request(body), body)

    def test_integer_labels_and_all_ten_classes_in_both_splits(self):
        for split in ("train", "evaluation"):
            for value in (-1, 10, True, False, 1.0, "1", None):
                with self.subTest(split=split, label=value):
                    body = synthetic_request()
                    body["dataset"][split][0]["y"] = value
                    self.reject(body)
            body = synthetic_request()
            body["dataset"][split][-1]["y"] = 8
            self.reject(body)

    def test_split_budget_boundaries(self):
        for split, maximum in (("train", 3823), ("evaluation", 1797)):
            for count in (10, maximum):
                with self.subTest(split=split, accepted=count):
                    body = synthetic_request()
                    body["dataset"][split] = synthetic_rows(count)
                    self.assertIs(training_request(body), body)
            for count in (0, 9, maximum + 1):
                with self.subTest(split=split, rejected=count):
                    body = synthetic_request()
                    body["dataset"][split] = synthetic_rows(count)
                    self.reject(body)
            for value in (None, {}, "ten rows"):
                body = synthetic_request()
                body["dataset"][split] = value
                self.reject(body)
        maximum = synthetic_request(3823, 1797)
        self.assertIs(training_request(maximum), maximum)

    def test_config_budget_boundaries_and_types(self):
        invalid = {
            "epochs": (0, 61, True, 1.0, "8"),
            "learning_rate": (0.00099, 0.10001, True, "0.05", None, float("nan"), float("inf")),
            "seed": (-1, 2**32, True, 1.0, "42"),
        }
        for key, values in invalid.items():
            for value in values:
                with self.subTest(key=key, value=value):
                    body = synthetic_request()
                    body["config"][key] = value
                    self.reject(body)
        for config in ({"epochs": 1, "learning_rate": 0.001, "seed": 0},
                       {"epochs": 60, "learning_rate": 0.1, "seed": 2**32 - 1}):
            body = synthetic_request()
            body["config"] = config
            self.assertIs(training_request(body), body)

    def test_all_provenance_declarations_are_checked(self):
        for key in ("archive_sha256", "train_sha256", "evaluation_sha256"):
            for value in ("", "a" * 63, "a" * 65, "G" * 64, "A" * 64, 123, None):
                with self.subTest(key=key, value=value):
                    body = synthetic_request()
                    body["dataset"]["provenance"][key] = value
                    self.reject(body)
        for key, values in {
            "official_split": (False, 1, "true", None),
            "license": ("CC0", "CC-BY-4.0 ", None),
            "source_url": ("https://example.invalid", "file:///tmp/data", None),
        }.items():
            for value in values:
                with self.subTest(key=key, value=value):
                    body = synthetic_request()
                    body["dataset"]["provenance"][key] = value
                    self.reject(body)
        # This acceptance is intentional: well-formed caller declarations do
        # not establish that the submitted inline rows came from that source.
        body = synthetic_request()
        for key in ("archive_sha256", "train_sha256", "evaluation_sha256"):
            body["dataset"]["provenance"][key] = "0" * 64
        self.assertIs(training_request(body), body)

    def test_unknown_missing_fields_and_rights_source_version(self):
        selectors = ((), ("dataset",), ("config",), ("dataset", "provenance"), ("dataset", "train", 0))
        for selector in selectors:
            body = synthetic_request()
            target = body
            for key in selector:
                target = target[key]
            required = list(target)
            for omitted in required:
                with self.subTest(path=selector, omitted=omitted):
                    changed = copy.deepcopy(body)
                    destination = changed
                    for key in selector:
                        destination = destination[key]
                    del destination[omitted]
                    self.reject(changed)
            target["unexpected"] = "not permitted"
            self.reject(body)
        for key, values in {
            "rights_confirmed": (False, 1, "true", None),
            "source": ("", "x" * 161, "bad\nsource", None),
            "version": ("", "x" * 161, "bad\nversion", None),
        }.items():
            for value in values:
                with self.subTest(key=key, value=value):
                    body = synthetic_request()
                    body["dataset"][key] = value
                    self.reject(body)

    def test_duplicate_features_are_retained_only_for_digits(self):
        body = synthetic_request()
        body["dataset"]["evaluation"] = copy.deepcopy(body["dataset"]["train"])
        body["dataset"]["train"].append(copy.deepcopy(body["dataset"]["train"][0]))
        self.assertIs(training_request(body), body)
        self.assertEqual(len(body["dataset"]["train"]), 11)
        legacy = request_body()
        legacy["dataset"]["evaluation"][0] = copy.deepcopy(legacy["dataset"]["train"][0])
        self.reject(legacy)

    def test_training_and_predictions_are_deterministic_without_mutating_input(self):
        body = synthetic_request(30, 10)
        before = copy.deepcopy(body)
        events = []
        first = digits.train(body["dataset"]["train"], body["config"], lambda *event: events.append(event))
        second = digits.train(body["dataset"]["train"], body["config"], lambda *_: None)
        self.assertEqual(first, second)
        self.assertEqual(digest(first), digest(second))
        self.assertEqual(body, before)
        self.assertEqual([details["epoch"] for _, details in events], list(range(1, 9)))
        self.assertTrue(any(value != 0 for row in first["weights"] for value in row))
        self.assertEqual(first["classes"], list(range(10)))
        self.assertEqual(first["dimensions"], 64)
        changed_config = dict(body["config"], seed=43)
        self.assertNotEqual(first, digits.train(body["dataset"]["train"], changed_config, lambda *_: None))
        xs = [row["x"] for row in body["dataset"]["evaluation"]]
        outputs = algorithm.predict(first, xs)
        self.assertEqual(outputs, algorithm.predict(second, xs))
        for output in outputs:
            self.assertEqual(set(output), {"class", "probabilities"})
            self.assertEqual(len(output["probabilities"]), 10)
            self.assertTrue(all(math.isfinite(p) and 0 <= p <= 1 for p in output["probabilities"]))
            self.assertAlmostEqual(sum(output["probabilities"]), 1.0, places=12)
            self.assertEqual(output["class"], max(range(10), key=output["probabilities"].__getitem__))

    def test_online_prediction_boundaries_and_stable_softmax(self):
        body = synthetic_request()
        model = digits.train(body["dataset"]["train"], body["config"], lambda *_: None)
        self.assertEqual(len(algorithm.predict(model, [[0] * 64] * 128)), 128)
        for xs in ([], [[0] * 64] * 129, [[0] * 63], [[0.0] * 64], [[17] * 64], [[True] * 64]):
            with self.subTest(rows=len(xs)):
                with self.assertRaises(Problem):
                    algorithm.predict(model, xs)
        weights = [[1e6] * 64] + [[-1e6] * 64 for _ in range(9)]
        probabilities = digits.probabilities(weights, [1e6] * 10, [1] * 64)
        self.assertEqual(probabilities, [1.0] + [0.0] * 9)

    def test_loading_rejects_malformed_multiclass_artifacts(self):
        body = synthetic_request()
        model = digits.train(body["dataset"]["train"], body["config"], lambda *_: None)
        mutations = [
            (("format",), "pickle/v1"), (("recipe",), RECIPE), (("contract",), CONTRACT),
            (("dimensions",), 63), (("dimensions",), 64.0), (("dimensions",), True),
            (("classes",), list(range(9))), (("classes",), list(reversed(range(10)))),
            (("classes",), [False] + list(range(1, 10))), (("classes",), list(map(float, range(10)))),
            (("classes",), None), (("pixel_scale",), 255), (("pixel_scale",), 16.0),
            (("weights",), []), (("weights",), [[0] * 64] * 9), (("weights",), None),
            (("weights", 0), [0] * 63), (("weights", 0), [0] * 65), (("weights", 0), {}),
            (("weights", 0, 0), True), (("weights", 0, 0), float("nan")),
            (("weights", 0, 0), float("inf")), (("weights", 0, 0), 1e6 + 1),
            (("biases",), [0] * 9), (("biases",), [0] * 11), (("biases",), None),
            (("biases", 0), "0"), (("biases", 0), -1e6 - 1), (("biases", 0), False),
        ]
        invalid = []
        for route, value in mutations:
            changed = copy.deepcopy(model)
            destination = changed
            for key in route[:-1]:
                destination = destination[key]
            destination[route[-1]] = value
            invalid.append((str(route), changed))
        for key in model:
            changed = copy.deepcopy(model)
            del changed[key]
            invalid.append(("missing " + key, changed))
        invalid.append(("unknown", dict(model, untrusted="extra")))
        local = test_platform.ROOT / ".runtime" / "tests"
        local.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=local) as temporary:
            path = Path(temporary) / "model.json"
            for name, changed in invalid:
                with self.subTest(case=name):
                    content = json.dumps(changed).encode()
                    path.write_bytes(content)
                    with self.assertRaises((Problem, ValueError)):
                        algorithm.load_model(path, hashlib.sha256(content).hexdigest())
            path.write_bytes(encode(model))
            self.assertEqual(algorithm.load_model(path, hashlib.sha256(path.read_bytes()).hexdigest()), model)
            with self.assertRaises(ValueError):
                algorithm.load_model(path, "0" * 64)
            content = b" " * 32769
            path.write_bytes(content)
            with self.assertRaises(ValueError):
                algorithm.load_model(path, hashlib.sha256(content).hexdigest())

    def test_reloaded_evaluation_metrics_are_independently_recomputed(self):
        body = synthetic_request(30, 139)
        model = digits.train(body["dataset"]["train"], body["config"], lambda *_: None)
        local = test_platform.ROOT / ".runtime" / "tests"
        local.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=local) as temporary:
            path = Path(temporary) / "model.json"
            artifact_hash = algorithm.write_model(path, model)
            rows = body["dataset"]["evaluation"]
            result = evaluate(path, artifact_hash, rows)
            self.assertEqual(result["protocol"], "held-out-multiclass/v1")
            self.assertEqual(result["sample_count"], 139)
            self.assertEqual(result["artifact_sha256"], artifact_hash)
            self.assertEqual(result["dataset_sha256"], digest(rows))
            self.assertEqual(result["class_order"], list(range(10)))
            self.assertGreaterEqual(result["inference_seconds"], 0)
            confusion = [[0] * 10 for _ in range(10)]
            loss = 0.0
            for row in rows:
                scores = [model["biases"][label] + sum(w * x / 16 for w, x in zip(model["weights"][label], row["x"]))
                          for label in range(10)]
                exps = [math.exp(score - max(scores)) for score in scores]
                ps = [value / sum(exps) for value in exps]
                predicted = max(range(10), key=ps.__getitem__)
                confusion[row["y"]][predicted] += 1
                loss -= math.log(min(max(ps[row["y"]], 1e-15), 1 - 1e-15))
            self.assertEqual(result["confusion"], confusion)
            self.assertAlmostEqual(result["accuracy"], sum(confusion[i][i] for i in range(10)) / len(rows))
            self.assertAlmostEqual(result["log_loss"], loss / len(rows))
            f1s = []
            for label in range(10):
                denominator = sum(confusion[label]) + sum(row[label] for row in confusion)
                f1s.append(2 * confusion[label][label] / denominator if denominator else 0.0)
            self.assertAlmostEqual(result["macro_f1"], sum(f1s) / 10)

    def test_legacy_artifacts_and_binary_predictions_remain_loadable(self):
        body = request_body()
        model = algorithm.train(body["dataset"]["train"], body["config"], lambda *_: None)
        local = test_platform.ROOT / ".runtime" / "tests"
        local.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=local) as temporary:
            path = Path(temporary) / "legacy.json"
            artifact_hash = algorithm.write_model(path, model)
            reloaded = algorithm.load_model(path, artifact_hash)
            self.assertEqual(reloaded["format"], "logistic-json/v1")
            self.assertEqual(reloaded["contract"], CONTRACT)
            outputs = algorithm.predict(reloaded, [[-2.5], [2.5]])
            self.assertEqual([output["class"] for output in outputs], [0, 1])
            self.assertTrue(all(set(output) == {"class", "probability"} for output in outputs))
            result = evaluate(path, artifact_hash, body["dataset"]["evaluation"])
            self.assertEqual(result["protocol"], "held-out-binary/v1")
            self.assertEqual(result["accuracy"], 1.0)
            self.assertEqual(result, evaluate(path, artifact_hash, body["dataset"]["evaluation"]))


class DigitsLifecycleTests(unittest.TestCase):
    # Reuse only the real HTTP/process harness, without inheriting and rerunning
    # PlatformTests' full binary acceptance suite a second time.
    setUp = test_platform.PlatformTests.setUp
    spawn = test_platform.PlatformTests.spawn
    start_api = test_platform.PlatformTests.start_api
    cleanup = test_platform.PlatformTests.cleanup
    call = test_platform.PlatformTests.call
    worker_once = test_platform.PlatformTests.worker_once
    submit = test_platform.PlatformTests.submit
    train = test_platform.PlatformTests.train
    ready = test_platform.PlatformTests.ready
    deploy = test_platform.PlatformTests.deploy

    def test_real_multiclass_train_register_serve_stop_and_lineage(self):
        body = synthetic_request(30, 139)
        run = self.train(body)
        self.assertEqual(run["recipe"], digits.RECIPE)
        self.assertEqual(run["request_sha256"], digest(body))
        self.assertEqual(run["config"], body["config"])
        self.assertNotIn("train", run["dataset"])
        status, model = self.call("/models/" + run["model_version_id"])
        self.assertEqual(status, 200)
        self.assertEqual(model["run_id"], run["id"])
        self.assertEqual(model["attempt_id"], run["attempts"][0]["id"])
        self.assertEqual(model["recipe"], digits.RECIPE)
        self.assertEqual(model["contract"], digits.CONTRACT)
        self.assertEqual(model["format"], digits.FORMAT)
        self.assertEqual(model["dimensions"], 64)
        self.assertEqual(model["dataset"]["source"], body["dataset"]["source"])
        self.assertEqual(model["dataset"]["provenance"], body["dataset"]["provenance"])
        self.assertIn("not independently authenticated", model["dataset"]["split_policy"])
        self.assertEqual(model["dataset"]["feature_overlap_evaluation_rows"], 0)
        for split in ("train", "evaluation"):
            self.assertEqual(model["dataset"][split + "_sha256"], digest(body["dataset"][split]))
        self.assertEqual(model["config"], body["config"])
        self.assertEqual(model["environment"], model["execution_environment"])
        self.assertGreater(model["training_seconds"], 0)
        self.assertNotIn("weights", model)
        artifact = self.runtime.artifact(model["attempt_id"])
        self.assertEqual(hashlib.sha256(artifact.read_bytes()).hexdigest(), model["artifact_sha256"])
        loaded = algorithm.load_model(artifact, model["artifact_sha256"])
        self.assertTrue(any(value != 0 for row in loaded["weights"] for value in row))
        expected = evaluate(artifact, model["artifact_sha256"], body["dataset"]["evaluation"])
        actual = dict(model["evaluation"])
        for result in (expected, actual):
            self.assertGreaterEqual(result.pop("inference_seconds"), 0)
        self.assertEqual(actual, expected)
        self.assertEqual(sum(map(sum, actual["confusion"])), 139)
        deployment = self.deploy(model["model_version_id"])
        self.assertEqual(deployment["contract"], digits.CONTRACT)
        self.assertNotEqual(deployment["pid"], self.api.pid)
        route = "/deployments/" + deployment["id"]
        xs = [row["x"] for row in body["dataset"]["evaluation"][:10]]
        status, output = self.call(route + "/predict", {"features": xs})
        self.assertEqual(status, 200)
        self.assertEqual(output["model_version_id"], model["model_version_id"])
        self.assertEqual(output["artifact_sha256"], model["artifact_sha256"])
        self.assertEqual(output["contract"], digits.CONTRACT)
        self.assertEqual(output["predictions"], algorithm.predict(loaded, xs))
        self.assertEqual(self.call(route + "/predict", {"features": [xs[0]] * 128})[0], 200)
        for xs in ([[0] * 64] * 129, [[0] * 63], [[0.0] * 64], [[17] * 64], [[True] * 64]):
            self.assertEqual(self.call(route + "/predict", {"features": xs})[0], 400)
        self.assertEqual(self.call(route + "/predict", {"features": [[0] * 64]}, tenant="other")[0], 404)
        self.assertEqual(self.call("/models/" + model["model_version_id"], tenant="other")[0], 404)
        status, stopped = self.call(route + "/stop", {})
        self.assertEqual(status, 200)
        self.assertEqual(stopped["observed"], "stopped")
        self.assertFalse(stopped["ready"])
        self.assertEqual(self.call(route + "/predict", {"features": [[0] * 64]})[0], 503)
        with self.assertRaises(ProcessLookupError):
            os.kill(deployment["pid"], 0)

    def test_http_validation_health_and_idempotency_include_recipe_config(self):
        health = self.call("/health")[1]
        self.assertIn(digits.RECIPE, health["recipes"])
        self.assertIn(RECIPE, health["recipes"])
        body = synthetic_request()
        run = self.submit(body)
        status, duplicate = self.call("/runs", body, "test")
        self.assertEqual(status, 200)
        self.assertEqual(run["id"], duplicate["id"])
        changed = copy.deepcopy(body)
        changed["config"]["seed"] += 1
        self.assertEqual(self.call("/runs", changed, "test")[0], 409)
        self.assertEqual(self.call("/runs", request_body(), "test")[0], 409)
        for name in ("recipe", "config", "provenance", "pixels"):
            changed = copy.deepcopy(body)
            if name == "recipe":
                changed["recipe"] = "unregistered/v1"
            elif name == "config":
                changed["config"]["seed"] = True
            elif name == "provenance":
                changed["dataset"]["provenance"]["license"] = "unknown"
            else:
                changed["dataset"]["evaluation"][0]["x"][0] = 16.0
            self.assertEqual(self.call("/runs", changed, name)[0], 400)
        self.assertEqual(self.call("/runs")[1]["ids"], [run["id"]])

    def test_json_numeric_overflow_is_a_client_error_for_both_recipes(self):
        for recipe, factory in ((RECIPE, request_body), (digits.RECIPE, synthetic_request)):
            for literal in (b"1e999", b"-1e999"):
                with self.subTest(recipe=recipe, literal=literal):
                    body = factory()
                    body["dataset"]["train"][0]["x"][0] = "OVERFLOW_LITERAL"
                    raw = json.dumps(body).encode().replace(b'"OVERFLOW_LITERAL"', literal)
                    status, response = self.call("/runs", key=recipe + literal.decode(), raw=raw)
                    self.assertEqual(status, 400, response)
                    self.assertEqual(response["error"]["code"], "invalid_input")
        self.assertEqual(self.call("/runs")[1]["ids"], [])

    def test_maximum_inline_digits_request_fits_transport_budget(self):
        body = synthetic_request(3823, 1797)
        raw = json.dumps(body).encode()
        self.assertGreater(len(raw), 1024 * 1024)
        self.assertLessEqual(len(raw), 2 * 1024 * 1024)
        status, run = self.call("/runs", key="full-budget-synthetic", raw=raw)
        self.assertEqual(status, 202, run)
        self.assertEqual(run["dataset"]["train_count"], 3823)
        self.assertEqual(run["dataset"]["evaluation_count"], 1797)

    def test_repeated_training_is_deterministic_and_overlap_is_declared(self):
        body = synthetic_request()
        body["dataset"]["evaluation"] = copy.deepcopy(body["dataset"]["train"])
        first = self.train(body, key="first")
        second = self.train(body, key="second")
        one = self.call("/models/" + first["model_version_id"])[1]
        two = self.call("/models/" + second["model_version_id"])[1]
        self.assertNotEqual(one["model_version_id"], two["model_version_id"])
        self.assertNotEqual(one["attempt_id"], two["attempt_id"])
        self.assertEqual(one["artifact_sha256"], two["artifact_sha256"])
        self.assertEqual(one["dataset"]["feature_overlap_evaluation_rows"], 10)
        self.assertEqual(one["evaluation"]["sample_count"], 10)
        for key in ("accuracy", "macro_f1", "log_loss", "confusion"):
            self.assertEqual(one["evaluation"][key], two["evaluation"][key])

    def test_real_multiclass_failure_retry_preserves_all_attempts(self):
        body = synthetic_request()
        for row in body["dataset"]["train"]:
            row["x"] = [0] * 64
        run = self.submit(body)
        attempt_ids = set()
        for number in range(1, 4):
            self.worker_once()
            failed = self.call("/runs/" + run["id"])[1]
            self.assertEqual(failed["status"], "failed")
            self.assertIsNone(failed["model_version_id"])
            self.assertEqual(failed["request_sha256"], digest(body))
            self.assertEqual(len(failed["attempts"]), number)
            self.assertEqual([a["number"] for a in failed["attempts"]], list(range(1, number + 1)))
            for attempt in failed["attempts"]:
                self.assertEqual(attempt["error_code"], "degenerate_features")
                self.assertEqual(attempt["status"], "failed")
                self.assertTrue(attempt["logs"])
                attempt_ids.add(attempt["id"])
            self.assertEqual(self.call("/runs/" + run["id"] + "/retry", {})[0], 202 if number < 3 else 409)
        self.assertEqual(len(attempt_ids), 3)
        self.assertEqual(self.call("/models")[1]["ids"], [])

    def test_multiclass_api_crash_recovers_models_and_desired_state(self):
        body = synthetic_request()
        run = self.train(body)
        model = self.call("/models/" + run["model_version_id"])[1]
        running = self.deploy(run["model_version_id"])
        stopped = self.deploy(run["model_version_id"])
        self.assertEqual(self.call("/deployments/" + stopped["id"] + "/stop", {})[0], 200)
        queued = self.submit(body, key="queued-digits")
        xs = [body["dataset"]["evaluation"][0]["x"]]
        route = "/deployments/" + running["id"]
        before = self.call(route + "/predict", {"features": xs})[1]
        self.api.kill()
        self.api.wait(timeout=5)
        self.start_api()
        recovered = self.ready(running["id"])
        self.assertNotEqual(recovered["pid"], running["pid"])
        self.assertEqual(recovered["contract"], digits.CONTRACT)
        self.assertEqual(self.call(route + "/predict", {"features": xs})[1], before)
        self.assertEqual(self.call("/models/" + model["model_version_id"])[1], model)
        self.assertEqual(self.call("/deployments/" + stopped["id"])[1]["observed"], "stopped")
        self.assertEqual(self.call("/deployments/" + stopped["id"] + "/predict", {"features": xs})[0], 503)
        self.worker_once()
        self.assertEqual(self.call("/runs/" + queued["id"])[1]["status"], "succeeded")
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            try:
                os.kill(running["pid"], 0)
            except ProcessLookupError:
                break
            time.sleep(0.1)
        else:
            self.fail("Orphan multiclass inference process did not exit")

    def test_multiclass_worker_interruption_requires_explicit_retry(self):
        body = synthetic_request(200, 10)
        body["config"]["epochs"] = 60
        run = self.submit(body)
        worker, log = self.spawn("worker")
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            with self.runtime.connection() as db:
                state = db.execute("SELECT status FROM runs WHERE id=?", (run["id"],)).fetchone()[0]
            if state == "running":
                break
            self.assertIsNone(worker.poll(), log.read_text())
            time.sleep(0.005)
        else:
            self.fail("Multiclass worker did not claim job")
        worker.kill()
        worker.wait(timeout=5)
        self.worker_once()
        failed = self.call("/runs/" + run["id"])[1]
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["attempts"][0]["error_code"], "interrupted")
        self.assertEqual(self.call("/models")[1]["ids"], [])
        self.assertEqual(self.call("/runs/" + run["id"] + "/retry", {})[0], 202)
        self.worker_once()
        recovered = self.call("/runs/" + run["id"])[1]
        self.assertEqual(recovered["status"], "succeeded", recovered)
        self.assertEqual([a["status"] for a in recovered["attempts"]], ["failed", "succeeded"])

    def test_multiclass_serving_crash_and_corruption_never_keep_ready(self):
        run = self.train(synthetic_request())
        deployed = self.deploy(run["model_version_id"])
        os.kill(deployed["pid"], signal.SIGKILL)
        time.sleep(0.1)
        route = "/deployments/" + deployed["id"]
        state = self.call(route)[1]
        self.assertEqual(state["observed"], "failed")
        self.assertFalse(state["ready"])
        self.assertEqual(self.call(route + "/predict", {"features": [[0] * 64]})[0], 503)
        self.assertEqual(self.call(route + "/stop", {})[0], 200)
        model = self.call("/models/" + run["model_version_id"])[1]
        artifact = self.runtime.artifact(model["attempt_id"])
        artifact.chmod(0o600)
        artifact.write_text('{"format":"not-a-model"}')
        status, corrupt = self.call("/deployments", {"model_version_id": model["model_version_id"]})
        self.assertEqual(status, 202)
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            corrupt = self.call("/deployments/" + corrupt["id"])[1]
            if corrupt["observed"] == "failed":
                break
            time.sleep(0.1)
        self.assertEqual(corrupt["observed"], "failed")
        self.assertFalse(corrupt["ready"])
        self.assertEqual(self.call("/deployments/" + corrupt["id"] + "/predict", {"features": [[0] * 64]})[0], 503)


if __name__ == "__main__":
    unittest.main()
