"""Fashion contracts and real ONNX serving with explicitly synthetic fixtures.

No test trains a CNN or reads official Fashion-MNIST data. The tiny independently
constructed ONNX graph establishes parser/serving/lifecycle behavior only, never
recognition quality. scripts/demo_fashion.py performs the official-data training
and evaluation once, outside this repeatable regression suite.
"""
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

from model_platform import fashion
from model_platform.common import Problem, digest, encode, environment, training_request
import test_platform


OPTIONAL_ONNX = all(importlib.util.find_spec(name) is not None for name in ("numpy", "onnx", "onnxruntime"))


def synthetic_onnx():
    """Hand-build a small synthetic graph, independently of the CNN exporter."""
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    weights = np.zeros((784, 10), dtype=np.float32)
    for label in range(10):
        weights[label, label] = 1.0
    graph = helper.make_graph(
        [helper.make_node("Flatten", ["pixels"], ["flat"], axis=1),
         helper.make_node("Gemm", ["flat", "weights", "bias"], ["logits"])],
        "synthetic-serving-test-not-trained-not-quality-evidence",
        [helper.make_tensor_value_info("pixels", TensorProto.FLOAT, ["batch", 1, 28, 28])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", 10])],
        [numpy_helper.from_array(weights, "weights"),
         numpy_helper.from_array(np.arange(10, dtype=np.float32) / 100, "bias")])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10)
    helper.set_model_props(model, {"recipe": fashion.RECIPE, "contract": fashion.CONTRACT,
                                  "format": fashion.FORMAT, "dimensions": "784",
                                  "pixel_scale": "255", "classes": "0,1,2,3,4,5,6,7,8,9"})
    onnx.checker.check_model(model)
    return model


class FashionContractTests(unittest.TestCase):
    def test_import_and_request_contract_need_no_optional_packages(self):
        # A clean isolated interpreter fails if contract imports try to load an extra.
        script = """
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in ('torch', 'onnx', 'onnxruntime', 'numpy'):
        raise AssertionError('optional dependency imported: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from model_platform.common import training_request
from model_platform import fashion, fashion_training
assert training_request(fashion.request_body())['recipe'] == fashion.RECIPE
fashion.pixels([[0] * 784])
"""
        result = subprocess.run([sys.executable, "-c", script], cwd=test_platform.ROOT,
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        first, second = fashion.request_body(), fashion.request_body()
        first["config"]["epochs"] = 1
        self.assertEqual(second["config"]["epochs"], 5)

    def test_fixed_source_configuration_and_unknown_fields(self):
        body = fashion.request_body()
        self.assertEqual(training_request(body), body)
        cases = []
        for key, value in {"source": "UCI", "version": "other", "rights_confirmed": False,
                           "reference": "file:///etc/passwd", "url": "https://example.invalid",
                           "path": "/tmp/dataset", "train": [], "evaluation": []}.items():
            changed = copy.deepcopy(body)
            changed["dataset"][key] = value
            cases.append(changed)
        for key, value in {"epochs": 4, "learning_rate": 0.1, "seed": 1,
                           "train_count": 60000, "batch_size": 64, "device": "cuda"}.items():
            changed = copy.deepcopy(body)
            changed["config"][key] = value
            cases.append(changed)
        for key in body["config"]:
            changed = copy.deepcopy(body)
            changed["config"][key] = True
            cases.append(changed)
            changed = copy.deepcopy(body)
            del changed["config"][key]
            cases.append(changed)
        for key in body["dataset"]:
            changed = copy.deepcopy(body)
            del changed["dataset"][key]
            cases.append(changed)
        for value in (None, [], "official", 1):
            for key in ("dataset", "config"):
                changed = copy.deepcopy(body)
                changed[key] = value
                cases.append(changed)
        for index, changed in enumerate(cases):
            with self.subTest(index=index):
                with self.assertRaises(Problem) as failure:
                    training_request(changed)
                self.assertEqual(failure.exception.status, 400)

    def test_pixel_shape_range_type_and_batch_budget(self):
        for value in ([[0] * 784], [[255] * 784] * 32):
            fashion.pixels(value)
        for value in ([], [[0] * 784] * 33, [[0] * 783], [[0] * 785],
                      [[0.0] * 784], [[True] * 784], [[-1] * 784], [[256] * 784],
                      [[float("nan")] * 784], [[float("inf")] * 784],
                      [["0"] * 784], None, "pixels", [None], [tuple([0] * 784)]):
            with self.subTest(value_type=type(value).__name__):
                with self.assertRaises(Problem):
                    fashion.pixels(value)

    def test_hash_size_and_symlink_fail_before_optional_import(self):
        local = test_platform.ROOT / ".runtime" / "tests"
        local.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=local) as directory:
            path = Path(directory) / "artifact.onnx"
            path.write_bytes(b"not an ONNX graph")
            with self.assertRaisesRegex(ValueError, "integrity"):
                fashion.load_model(path, "0" * 64)
            path.write_bytes(b"x" * (fashion.MAX_ARTIFACT + 1))
            with self.assertRaisesRegex(ValueError, "size"):
                fashion.load_model(path, hashlib.sha256(path.read_bytes()).hexdigest())
            link = Path(directory) / "linked.onnx"
            link.symlink_to(path)
            with self.assertRaises((ValueError, OSError)):
                fashion.load_model(link, hashlib.sha256(path.read_bytes()).hexdigest())


class FashionTrainingHelperTests(unittest.TestCase):
    def test_stratified_training_indices_are_fixed_and_training_only(self):
        from model_platform.fashion_training import select_training_indices
        from model_platform.algorithm import TrainingFailure
        labels = [label for label in range(10) for _ in range(6000)]
        indices = select_training_indices(labels)
        self.assertEqual(len(indices), 6000)
        self.assertEqual(len(set(indices)), 6000)
        self.assertEqual(indices, sorted(indices))
        self.assertEqual(indices, select_training_indices(labels))
        self.assertTrue(all(0 <= index < 60000 for index in indices))
        self.assertEqual([sum(labels[index] == label for index in indices) for label in range(10)], [600] * 10)
        for malformed in (labels[:-1], [0] * 60000, [10] * 60000):
            with self.assertRaises(TrainingFailure) as failed:
                select_training_indices(malformed)
            self.assertEqual(failed.exception.code, "fashion_dataset_invalid")

    def test_artifact_creation_is_exclusive_bounded_and_readonly(self):
        from model_platform.fashion_training import _write_artifact
        from model_platform.algorithm import TrainingFailure
        local = test_platform.ROOT / ".runtime" / "tests"
        local.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=local) as directory:
            path = Path(directory) / "test-artifact"
            content = b"synthetic bytes for exclusive writer unit test, not a model"
            self.assertEqual(_write_artifact(path, content), hashlib.sha256(content).hexdigest())
            self.assertEqual(path.stat().st_mode & 0o777, 0o400)
            with self.assertRaises(FileExistsError):
                _write_artifact(path, b"replacement")
            self.assertEqual(path.read_bytes(), content)
            for content in (b"", b"x" * (fashion.MAX_ARTIFACT + 1)):
                with self.assertRaises(TrainingFailure):
                    _write_artifact(Path(directory) / "oversize", content)
            self.assertFalse((Path(directory) / "oversize").exists())


class FashionHttpContractTests(unittest.TestCase):
    setUp = test_platform.PlatformTests.setUp
    spawn = test_platform.PlatformTests.spawn
    start_api = test_platform.PlatformTests.start_api
    cleanup = test_platform.PlatformTests.cleanup
    call = test_platform.PlatformTests.call
    submit = test_platform.PlatformTests.submit

    def test_http_queue_idempotency_and_fixed_reference_without_training(self):
        self.assertIn(fashion.RECIPE, self.call("/health")[1]["recipes"])
        body = fashion.request_body()
        run = self.submit(body)
        self.assertEqual(run["status"], "queued")
        self.assertEqual(run["dataset"]["train_count"], 6000)
        self.assertEqual(run["dataset"]["evaluation_count"], 10000)
        self.assertEqual(run["request_sha256"], digest(body))
        status, same = self.call("/runs", body, "test")
        self.assertEqual((status, same["id"]), (200, run["id"]))
        changed = copy.deepcopy(body)
        changed["business_ref"] = "different-business-reference"
        self.assertEqual(self.call("/runs", changed, "test")[0], 409)
        for name, value in (("url", "https://example.invalid"), ("path", "/tmp/data"), ("train", [])):
            changed = copy.deepcopy(body)
            changed["dataset"][name] = value
            self.assertEqual(self.call("/runs", changed, name)[0], 400)
        self.assertEqual(self.call("/runs/" + run["id"], tenant="other")[0], 404)
        self.assertEqual(self.call("/models")[1]["ids"], [])
        self.assertEqual(self.call("/runs/" + run["id"])[1]["attempts"], [])


@unittest.skipUnless(OPTIONAL_ONNX, "optional Fashion ONNX dependencies are not installed")
class FashionOnnxTests(unittest.TestCase):
    def setUp(self):
        local = test_platform.ROOT / ".runtime" / "tests"
        local.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=local)
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "synthetic.onnx"

    def load(self, graph):
        content = graph if isinstance(graph, bytes) else graph.SerializeToString()
        self.path.write_bytes(content)
        return fashion.load_model(self.path, hashlib.sha256(content).hexdigest())

    def test_independent_synthetic_graph_predicts_exact_preprocessing(self):
        model = self.load(synthetic_onnx())
        xs = [[0] * 784, [255] + [0] * 783]
        result = fashion.predict(model, xs)
        self.assertEqual([row["class"] for row in result], [9, 0])
        for image, output in zip(xs, result):
            scores = [image[index] / 255 + float(index / 100) for index in range(10)]
            expected = [math.exp(value - max(scores)) for value in scores]
            expected = [value / sum(expected) for value in expected]
            self.assertEqual(len(output["probabilities"]), 10)
            self.assertAlmostEqual(sum(output["probabilities"]), 1.0)
            for actual, wanted in zip(output["probabilities"], expected):
                self.assertAlmostEqual(actual, wanted, places=7)
        self.assertEqual(len(fashion.predict(model, [xs[0]] * 32)), 32)
        with self.assertRaises(Problem):
            fashion.predict(model, [xs[0]] * 33)

    def test_parity_rejects_error_and_argmax_mismatch_without_training(self):
        import numpy as np
        from model_platform.fashion_training import _parity
        from model_platform.algorithm import TrainingFailure
        reference = np.zeros((2, 10), dtype=np.float32)
        reference[:, 0] = 1
        self.assertEqual(_parity(np, reference, reference.copy()), (0.0, 2))
        altered = reference.copy()
        altered[0, 0] += 0.001
        with self.assertRaises(TrainingFailure) as failed:
            _parity(np, reference, altered)
        self.assertEqual(failed.exception.code, "fashion_onnx_parity_failed")
        # Tiny absolute error is insufficient when a tie flips the top class.
        tied = np.zeros((1, 10), dtype=np.float32)
        changed = tied.copy()
        changed[0, 1] = 1e-6
        with self.assertRaises(TrainingFailure) as failed:
            _parity(np, tied, changed)
        self.assertEqual(failed.exception.code, "fashion_onnx_parity_failed")
        invalid = [reference.astype(np.float64), reference[:1], reference[:, :9],
                   np.full((2, 10), np.nan, dtype=np.float32),
                   np.full((2, 10), np.inf, dtype=np.float32)]
        for actual in invalid:
            with self.assertRaises(TrainingFailure) as failed:
                _parity(np, reference, actual)
            self.assertEqual(failed.exception.code, "fashion_logits_invalid")

    def test_streaming_metrics_match_independent_ten_class_calculation(self):
        import numpy as np
        from model_platform.fashion_training import _Metrics
        from model_platform.algorithm import TrainingFailure
        labels = np.asarray([0, 1, 1, 9, 5], dtype=np.uint8)
        logits = np.zeros((5, 10), dtype=np.float32)
        for index, label in enumerate([0, 1, 2, 9, 0]):
            logits[index, label] = 2.0
        accumulator = _Metrics(np)
        accumulator.update(labels[:2], logits[:2])
        accumulator.update(labels[2:], logits[2:])
        result = accumulator.result()
        matrix = [[0] * 10 for _ in range(10)]
        loss = 0.0
        for label, row in zip(labels, logits.tolist()):
            predicted = max(range(10), key=row.__getitem__)
            matrix[int(label)][predicted] += 1
            exp_scores = [math.exp(value - max(row)) for value in row]
            loss -= math.log(exp_scores[int(label)] / sum(exp_scores))
        f1 = []
        for label in range(10):
            denominator = sum(matrix[label]) + sum(row[label] for row in matrix)
            f1.append(2 * matrix[label][label] / denominator if denominator else 0.0)
        self.assertEqual(result["confusion"], matrix)
        self.assertEqual(result["sample_count"], 5)
        self.assertEqual(result["class_order"], list(range(10)))
        self.assertEqual(result["accuracy"], 0.6)
        self.assertAlmostEqual(result["macro_f1"], sum(f1) / 10)
        self.assertAlmostEqual(result["log_loss"], loss / 5)
        with self.assertRaises(TrainingFailure):
            _Metrics(np).result()

    def test_wrong_hash_malformed_graph_and_metadata_rejected(self):
        graph = synthetic_onnx()
        self.path.write_bytes(graph.SerializeToString())
        with mock.patch("onnx.load_model_from_string") as parser:
            with self.assertRaisesRegex(ValueError, "integrity"):
                fashion.load_model(self.path, "0" * 64)
            parser.assert_not_called()
        for content in (b"not ONNX", b"", b'{"format":"fashion-onnx/v1"}'):
            with self.subTest(content=content):
                with self.assertRaises(Exception):
                    self.load(content)
        for key in ("recipe", "contract", "format", "dimensions", "pixel_scale", "classes"):
            graph = synthetic_onnx()
            next(item for item in graph.metadata_props if item.key == key).value = "wrong"
            with self.subTest(metadata=key), self.assertRaisesRegex(ValueError, "metadata"):
                self.load(graph)
        graph = synthetic_onnx()
        graph.metadata_props.add(key="recipe", value=fashion.RECIPE)
        with self.assertRaisesRegex(ValueError, "metadata"):
            self.load(graph)

    def test_custom_unsupported_operators_and_nested_graphs_rejected_before_runtime(self):
        import onnx
        from onnx import helper
        variants = []
        graph = synthetic_onnx()
        graph.graph.node[0].domain = "untrusted.custom"
        variants.append(graph)
        graph = synthetic_onnx()
        graph.graph.node[0].op_type = "Identity"
        variants.append(graph)
        graph = synthetic_onnx()
        graph.graph.node[0].attribute.append(helper.make_attribute("subgraph", helper.make_graph([], "nested", [], [])))
        variants.append(graph)
        graph = synthetic_onnx()
        graph.functions.append(helper.make_function("custom", "Function", ["x"], ["y"],
                                                    [helper.make_node("Identity", ["x"], ["y"])],
                                                    [helper.make_opsetid("", 17)]))
        variants.append(graph)
        for index, graph in enumerate(variants):
            with self.subTest(index=index), mock.patch("onnxruntime.InferenceSession") as session:
                with self.assertRaises(ValueError):
                    self.load(graph)
                session.assert_not_called()

    def test_external_initializer_and_tensor_attribute_never_reach_runtime(self):
        import onnx
        from onnx import TensorProto, helper
        graph = synthetic_onnx()
        tensor = graph.graph.initializer[0]
        tensor.ClearField("raw_data")
        tensor.data_location = TensorProto.EXTERNAL
        tensor.external_data.add(key="location", value="outside-test-weights.bin")
        variants = [graph]
        graph = synthetic_onnx()
        external = TensorProto(name="external", data_type=TensorProto.FLOAT, dims=[1], data_location=TensorProto.EXTERNAL)
        external.external_data.add(key="location", value="outside-test-weights.bin")
        graph.graph.node.append(helper.make_node("Constant", [], ["unused"], value=external))
        variants.append(graph)
        for index, graph in enumerate(variants):
            with self.subTest(index=index), mock.patch("onnxruntime.InferenceSession") as session:
                with self.assertRaises(Exception):
                    self.load(graph)
                session.assert_not_called()

    def test_invalid_input_output_shapes_types_and_fixed_batch_rejected(self):
        from onnx import TensorProto, helper
        changes = [
            ("input", TensorProto.FLOAT, ["batch", 784]),
            ("input", TensorProto.FLOAT, ["batch", 1, 27, 28]),
            ("input", TensorProto.INT64, ["batch", 1, 28, 28]),
            ("input", TensorProto.FLOAT, [1, 1, 28, 28]),
            ("output", TensorProto.FLOAT, ["batch", 9]),
            ("output", TensorProto.INT64, ["batch", 10]),
            ("output", TensorProto.FLOAT, [1, 10]),
        ]
        for field, datatype, shape in changes:
            graph = synthetic_onnx()
            collection = getattr(graph.graph, field)
            name = collection[0].name
            collection[0].CopyFrom(helper.make_tensor_value_info(name, datatype, shape))
            with self.subTest(field=field, datatype=datatype, shape=shape), self.assertRaises(Exception):
                self.load(graph)


@unittest.skipUnless(OPTIONAL_ONNX, "optional Fashion ONNX dependencies are not installed")
class FashionServingLifecycleTests(unittest.TestCase):
    # Borrow only the harness, never inherit and repeat the binary acceptance suite.
    setUp = test_platform.PlatformTests.setUp
    spawn = test_platform.PlatformTests.spawn
    start_api = test_platform.PlatformTests.start_api
    cleanup = test_platform.PlatformTests.cleanup
    call = test_platform.PlatformTests.call
    ready = test_platform.PlatformTests.ready
    deploy = test_platform.PlatformTests.deploy

    def register_synthetic_fixture(self, **manifest_changes):
        """Inject a clearly marked test fixture through the internal registry only."""
        run_id, created = self.store.submit("demo", "synthetic-" + str(time.time_ns()), fashion.request_body())
        self.assertTrue(created)
        run, attempt_id = self.store.claim()
        self.assertEqual(run["id"], run_id)
        content = synthetic_onnx().SerializeToString()
        self.runtime.artifact(attempt_id).write_bytes(content)
        artifact_hash = hashlib.sha256(content).hexdigest()
        manifest = {"recipe": fashion.RECIPE, "contract": fashion.CONTRACT, "format": fashion.FORMAT,
                    "dimensions": 784, "dataset": {"source": "synthetic-serving-fixture-NOT-official",
                    "quality_evidence": False}, "config": {}, "training_seconds": 0,
                    "environment": environment(), "execution_environment": environment(),
                    "target_profile": "onnxruntime-cpu/v1"}
        manifest.update(manifest_changes)
        evaluation = {"protocol": "synthetic-serving-fixture-only/v1", "dataset_sha256": digest("synthetic-fixture"),
                      "artifact_sha256": artifact_hash, "sample_count": 0, "quality_evidence": False}
        model_id = self.store.register(run, attempt_id, artifact_hash, evaluation, manifest)
        return self.store.model("demo", model_id)

    def failed(self, deployment_id):
        deadline = time.monotonic() + 12
        while time.monotonic() < deadline:
            status, state = self.call("/deployments/" + deployment_id)
            self.assertEqual(status, 200)
            if state["observed"] == "failed":
                self.assertFalse(state["ready"])
                return state
            time.sleep(0.2)
        self.fail("Invalid or crashed ONNX deployment never became failed")

    def test_synthetic_onnx_continuous_service_stop_restart_and_version_binding(self):
        model = self.register_synthetic_fixture()
        deployment = self.deploy(model["model_version_id"])
        self.assertEqual(deployment["runtime"], "onnxruntime-cpu/v1")
        self.assertEqual(deployment["contract"], fashion.CONTRACT)
        self.assertNotEqual(deployment["pid"], self.api.pid)
        route = "/deployments/" + deployment["id"]
        xs = [[0] * 784, [255] + [0] * 783]
        first = None
        for _ in range(3):
            status, result = self.call(route + "/predict", {"features": xs})
            self.assertEqual(status, 200)
            self.assertEqual(result["artifact_sha256"], model["artifact_sha256"])
            self.assertEqual(result["model_version_id"], model["model_version_id"])
            self.assertEqual(result["contract"], fashion.CONTRACT)
            self.assertEqual([item["class"] for item in result["predictions"]], [9, 0])
            if first is not None:
                self.assertEqual(result, first)
            first = result
            time.sleep(0.1)
        self.assertEqual(self.call(route + "/predict", {"features": [xs[0]] * 32})[0], 200)
        for images in ([xs[0]] * 33, [[0.0] * 784], [[256] * 784], [[0] * 783]):
            self.assertEqual(self.call(route + "/predict", {"features": images})[0], 400)
        self.assertEqual(self.call(route + "/predict", {"features": xs}, tenant="other")[0], 404)
        self.assertEqual(self.call("/models/" + model["model_version_id"], tenant="other")[0], 404)
        self.api.kill()
        self.api.wait(timeout=5)
        self.start_api()
        recovered = self.ready(deployment["id"])
        self.assertNotEqual(recovered["pid"], deployment["pid"])
        self.assertEqual(self.call(route + "/predict", {"features": xs})[1], first)
        self.assertEqual(self.call("/models/" + model["model_version_id"])[1], model)
        status, stopped = self.call(route + "/stop", {})
        self.assertEqual(status, 200)
        self.assertEqual((stopped["desired"], stopped["observed"], stopped["ready"]), ("stopped", "stopped", False))
        self.assertEqual(self.call(route + "/predict", {"features": xs})[0], 503)
        with self.assertRaises(ProcessLookupError):
            os.kill(recovered["pid"], 0)
        self.api.terminate()
        self.api.wait(timeout=10)
        self.start_api()
        self.assertEqual(self.call(route)[1]["observed"], "stopped")
        self.assertEqual(self.call(route + "/predict", {"features": xs})[0], 503)

    def test_crashed_or_corrupt_synthetic_artifact_is_not_ready(self):
        model = self.register_synthetic_fixture()
        deployment = self.deploy(model["model_version_id"])
        os.kill(deployment["pid"], signal.SIGKILL)
        self.failed(deployment["id"])
        route = "/deployments/" + deployment["id"]
        self.assertEqual(self.call(route + "/predict", {"features": [[0] * 784]})[0], 503)
        self.assertEqual(self.call(route + "/stop", {})[0], 200)
        path = self.runtime.artifact(model["attempt_id"])
        path.write_bytes(b"corrupt ONNX")
        status, corrupt = self.call("/deployments", {"model_version_id": model["model_version_id"]})
        self.assertEqual(status, 202)
        self.failed(corrupt["id"])
        self.assertEqual(self.call("/deployments/" + corrupt["id"] + "/predict", {"features": [[0] * 784]})[0], 503)

    def test_inconsistent_bound_manifest_never_becomes_ready(self):
        for changes in ({"recipe": "softmax-digits/v1"}, {"contract": "digits-64x10/v1"}, {"dimensions": 64}):
            with self.subTest(changes=changes):
                model = self.register_synthetic_fixture(**changes)
                status, deployment = self.call("/deployments", {"model_version_id": model["model_version_id"]})
                self.assertEqual(status, 202)
                self.failed(deployment["id"])
                self.assertEqual(self.call("/deployments/" + deployment["id"] + "/predict", {"features": [[0] * 784]})[0], 503)
                self.assertEqual(self.call("/deployments/" + deployment["id"] + "/stop", {})[0], 200)


if __name__ == "__main__":
    unittest.main()
