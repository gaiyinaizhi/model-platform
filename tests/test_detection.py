"""Detection contracts and actual untrained YOLOX-Nano ONNX interoperability.

All pixels and weights here are synthetic. The optional tests export the real
YOLOX-Nano architecture with random, untrained parameters; they verify loading,
tensor/preprocessing contracts, and inference only, never recognition quality.
No official images, downloads, training, deployed services or external API calls
are used. The real dataset training/evaluation demo is a separate acceptance run.
"""

import builtins
import copy
from dataclasses import replace
import hashlib
import importlib.util
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from model_platform import common, detection, detection_training, fashion, recipes
from model_platform.algorithm import TrainingFailure
from model_platform.common import Problem, training_request


ROOT = Path(__file__).resolve().parents[1]
OPTIONAL_DETECTION = all(importlib.util.find_spec(name) is not None
                         for name in ("torch", "numpy", "onnx", "onnxruntime", "PIL"))


def image(width=4, height=3):
    return {"width": width, "height": height, "pixels": [12, 34, 56] * (width * height)}


def ready_output(detections=None):
    return [{"width": detection.INPUT_SIZE, "height": detection.INPUT_SIZE,
             "detections": [] if detections is None else detections}]


def valid_box():
    return {"class_id": 0, "score": 0.5, "box": [0.0, 1.0, 20.0, 40.0]}


def temporary_runtime(test):
    parent = ROOT / ".runtime" / "tests"
    parent.mkdir(parents=True, exist_ok=True)
    directory = tempfile.TemporaryDirectory(prefix="detection-", dir=parent)
    test.addCleanup(directory.cleanup)
    return Path(directory.name)


class DetectionContractTests(unittest.TestCase):
    def test_fixed_request_is_detached_and_registered(self):
        body = detection.request_body()
        self.assertIs(training_request(body), body)
        spec = recipes.get_recipe(detection.RECIPE)
        self.assertEqual(spec.counts(body["dataset"]), (128, 32))
        self.assertEqual(spec.budgets.prediction_rows, 1)
        self.assertEqual(spec.budgets.artifact_bytes, detection.MAX_ARTIFACT)
        self.assertEqual(spec.classes, 1)
        self.assertEqual(spec.data_mode, "fixed-reference")
        body["config"]["epochs"] = 1
        body["dataset"]["reference"] = "changed"
        self.assertEqual(detection.request_body()["config"], detection.CONFIG)
        self.assertEqual(detection.request_body()["dataset"]["reference"], detection.REFERENCE)

    def test_dataset_rights_reference_fields_and_external_paths_rejected(self):
        base = detection.request_body()
        cases = []
        for field, values in {
            "source": ("other", None), "version": ("other", 1),
            "rights_confirmed": (False, 1, "true", None),
            "reference": ("file:///tmp/data", "https://example.invalid/data", "/tmp/data", None),
            "path": ("/tmp/data",), "url": ("https://example.invalid/data",),
            "train": ([],), "validation": ([],), "evaluation": ([],),
        }.items():
            for value in values:
                body = copy.deepcopy(base)
                body["dataset"][field] = value
                cases.append(body)
        for field in base["dataset"]:
            body = copy.deepcopy(base)
            del body["dataset"][field]
            cases.append(body)
        for value in (None, [], "reference", 1):
            cases.append({**base, "dataset": value})
        for body in cases:
            with self.subTest(dataset=body["dataset"]):
                with self.assertRaises(Problem) as failed:
                    training_request(body)
                self.assertEqual((failed.exception.status, failed.exception.code), (400, "invalid_input"))

    def test_fixed_configuration_unknown_fields_and_type_substitutions_rejected(self):
        base = detection.request_body()
        cases = []
        for field, current in base["config"].items():
            if type(current) is bool:
                values = (False, 1, "true", None)
            elif type(current) is int:
                values = (current + 1, float(current), True, str(current), None)
            else:
                values = (current + 0.1, True, str(current), None)
            for value in values:
                body = copy.deepcopy(base)
                body["config"][field] = value
                cases.append(body)
            body = copy.deepcopy(base)
            del body["config"][field]
            cases.append(body)
        for field, value in (("device", "cuda"), ("weights", "/tmp/weights"),
                             ("url", "https://example.invalid/model"), ("classes", 80)):
            body = copy.deepcopy(base)
            body["config"][field] = value
            cases.append(body)
        for value in (None, [], "config", 1):
            cases.append({**base, "config": value})
        cases.append({**base, "model_path": "/tmp/weights"})
        for index, body in enumerate(cases):
            with self.subTest(index=index):
                with self.assertRaises(Problem):
                    training_request(body)

    def test_single_inline_rgb_image_bounds_and_extreme_aspect_ratio(self):
        spec = recipes.get_recipe(detection.RECIPE)
        for value in ([image()], [image(384, 384)], [image(3, 384)], [image(384, 3)]):
            spec.validate_features(value, spec.budgets.max_dimensions)
        invalid = ([], None, [image(), image()], [image(385, 3)], [image(3, 385)],
                   [image(1, 384)], [image(384, 1)], [{**image(), "path": "/tmp/image"}],
                   [{**image(), "url": "https://example.invalid/image"}],
                   [{**image(), "width": True}], [{**image(), "height": 3.0}],
                   [{**image(), "pixels": [0] * 35}], [{**image(), "pixels": [0.0] * 36}],
                   [{**image(), "pixels": [True] * 36}], [{**image(), "pixels": [256] * 36}])
        for value in invalid:
            with self.subTest(value_type=type(value).__name__):
                with self.assertRaises(Problem) as failed:
                    spec.validate_features(value, spec.budgets.max_dimensions)
                self.assertEqual((failed.exception.status, failed.exception.code), (400, "invalid_input"))

    def test_detection_manifest_binding_never_falls_back_to_classification(self):
        manifest = {"recipe": detection.RECIPE, "contract": detection.CONTRACT, "format": detection.FORMAT}
        self.assertIs(recipes.for_manifest(manifest), recipes.get_recipe(detection.RECIPE))
        for field, value in (("recipe", fashion.RECIPE), ("contract", fashion.CONTRACT),
                             ("format", fashion.FORMAT)):
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    recipes.for_manifest({**manifest, field: value})
        for field in ("recipe", "contract", "format"):
            changed = dict(manifest)
            del changed[field]
            with self.assertRaises((Problem, ValueError)):
                recipes.for_manifest(changed)
        with self.assertRaises(Problem):
            recipes.for_manifest({**manifest, "recipe": "module:function"})

    def test_api_contract_imports_and_missing_dependency_failure_are_lazy(self):
        script = """
import builtins
import sys
blocked = {'torch', 'numpy', 'onnx', 'onnxruntime', 'onnxscript', 'PIL', 'pycocotools'}
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in blocked:
        raise ImportError('Explicitly absent optional dependency: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from model_platform import api, worker, serving, supervisor, recipes, detection
from model_platform.common import training_request, environment
from model_platform.algorithm import TrainingFailure
assert training_request(detection.request_body())['recipe'] == detection.RECIPE
spec = recipes.get_recipe(detection.RECIPE)
spec.validate_features(spec.readiness_input(spec.budgets.max_dimensions), spec.budgets.max_dimensions)
spec.validate_readiness([{'width': detection.INPUT_SIZE, 'height': detection.INPUT_SIZE, 'detections': []}])
assert detection.RECIPE in environment()['supported_recipes']
assert not blocked.intersection(sys.modules)
assert 'model_platform.detection_training' not in sys.modules
assert 'model_platform.yolox_nano.yolox' not in sys.modules
try:
    spec.execute(None, None, None, None, None, None, None)
except TrainingFailure as failure:
    assert failure.code == 'optional_dependencies_missing'
else:
    raise AssertionError('Missing detection dependencies did not fail execution')
assert not blocked.intersection(sys.modules)
"""
        result = subprocess.run([sys.executable, "-c", script], cwd=ROOT,
                                capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_artifact_integrity_size_and_symlink_checked_before_optional_import(self):
        root = temporary_runtime(self)
        path = root / "artifact.onnx"
        original = builtins.__import__

        def guarded(name, *args, **kwargs):
            if name.split(".")[0] in {"onnx", "onnxruntime", "numpy", "torch"}:
                raise AssertionError("Optional import before artifact validation: " + name)
            return original(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=guarded):
            for content in (b"", b"x" * (detection.MAX_ARTIFACT + 1)):
                path.write_bytes(content)
                with self.assertRaisesRegex(ValueError, "type or size"):
                    detection.load_model(path, hashlib.sha256(content).hexdigest())
            path.write_bytes(b"synthetic invalid ONNX bytes")
            with self.assertRaisesRegex(ValueError, "integrity"):
                detection.load_model(path, "0" * 64)
            link = root / "link.onnx"
            link.symlink_to(path)
            with self.assertRaises(OSError):
                detection.load_model(link, hashlib.sha256(path.read_bytes()).hexdigest())
            with self.assertRaises(OSError):
                detection.load_model(root, "0" * 64)

    def test_artifact_writer_is_bounded_exclusive_and_readonly(self):
        root = temporary_runtime(self)
        content = b"synthetic bytes, not model weights"
        path = root / "exclusive-artifact"
        self.assertEqual(detection_training._save(path, content), hashlib.sha256(content).hexdigest())
        self.assertEqual(path.stat().st_mode & 0o777, 0o400)
        with self.assertRaises(FileExistsError):
            detection_training._save(path, b"replacement")
        self.assertEqual(path.read_bytes(), content)
        for index, content in enumerate((b"", b"x" * (detection.MAX_ARTIFACT + 1))):
            bad_path = root / f"invalid-{index}"
            with self.assertRaises(TrainingFailure) as failed:
                detection_training._save(bad_path, content)
            self.assertEqual(failed.exception.code, "detection_artifact_too_large")
            self.assertFalse(bad_path.exists())

    def test_recursive_code_hash_includes_nested_files_paths_and_changes(self):
        # Redirect only the path inspected by common.environment; do not mutate
        # the active package while real worker/demo runs may be using its hash.
        root = temporary_runtime(self)
        (root / "common.py").write_text("# synthetic top-level code\n")
        nested = root / "yolox_nano"
        nested.mkdir()
        source = nested / "utils.py"
        source.write_text("# synthetic nested implementation\n")
        (root / "utils.py").write_text("# distinct top-level same basename\n")
        (nested / "ignored.txt").write_text("not Python source")
        with mock.patch.object(common, "__file__", str(root / "common.py")):
            expected = hashlib.sha256()
            for path in sorted(root.rglob("*.py")):
                expected.update(path.relative_to(root).as_posix().encode())
                expected.update(path.read_bytes())
            initial = common.environment()["code_sha256"]
            self.assertEqual(initial, expected.hexdigest())
            self.assertEqual(common.environment()["code_sha256"], initial)
            (nested / "ignored.txt").write_text("ignored content changed")
            self.assertEqual(common.environment()["code_sha256"], initial)
            source.write_text("# changed nested implementation\n")
            changed = common.environment()["code_sha256"]
            self.assertNotEqual(changed, initial)
            moved = nested / "renamed.py"
            source.rename(moved)
            renamed = common.environment()["code_sha256"]
            self.assertNotEqual(renamed, changed)
            moved.unlink()
            self.assertNotEqual(common.environment()["code_sha256"], renamed)


class DetectionReadinessTests(unittest.TestCase):
    def setUp(self):
        self.spec = recipes.get_recipe(detection.RECIPE)

    def test_task_specific_rgb_probe_and_empty_detections_are_valid(self):
        pixels = self.spec.readiness_input(self.spec.budgets.max_dimensions)
        self.assertEqual(pixels, detection.probe_input())
        self.spec.validate_features(pixels, self.spec.budgets.max_dimensions)
        self.spec.validate_readiness(ready_output())
        self.spec.validate_readiness(ready_output([valid_box()]))
        self.spec.validate_readiness(ready_output([valid_box()] * 100))
        for score in (detection.SCORE_THRESHOLD, 1.0):
            self.spec.validate_readiness(ready_output([{
                "class_id": 0, "score": score, "box": [0, 0, detection.INPUT_SIZE, detection.INPUT_SIZE]}]))
        pixels[0]["pixels"][0] = 255
        self.assertEqual(detection.probe_input()[0]["pixels"][0], 0)

    def test_single_class_recipe_requires_paired_task_specific_probe(self):
        for changed in (replace(self.spec, probe_input=None), replace(self.spec, probe_output=None),
                        replace(self.spec, probe_input=None, probe_output=None),
                        replace(self.spec, probe_input="not callable")):
            with self.assertRaises(ValueError):
                recipes.build_registry((changed,))

    def test_probe_batch_dimensions_fields_and_types_are_strict(self):
        cases = [None, [], {}, ready_output() * 2, [None], [{}],
                 [{"class": 0, "probability": 0.5}], ready_output([valid_box()] * 101)]
        for field in ("width", "height", "detections"):
            value = ready_output()
            del value[0][field]
            cases.append(value)
        for field in ("width", "height"):
            for value in (0, 1, detection.INPUT_SIZE + 1, float(detection.INPUT_SIZE), True, "160", None):
                output = ready_output()
                output[0][field] = value
                cases.append(output)
        output = ready_output()
        output[0]["extra"] = 1
        cases.append(output)
        for value in (None, {}, "boxes", ()):
            output = ready_output()
            output[0]["detections"] = value
            cases.append(output)
        for index, value in enumerate(cases):
            with self.subTest(index=index):
                with self.assertRaises(ValueError):
                    self.spec.validate_readiness(value)

    def test_probe_rejects_malformed_classes_scores_and_boxes(self):
        invalid = [None, {}, {**valid_box(), "extra": 1}]
        for field in valid_box():
            value = valid_box()
            del value[field]
            invalid.append(value)
        for value in (-1, 1, False, 0.0, "0", None):
            invalid.append({**valid_box(), "class_id": value})
        for value in (-1, 0, detection.SCORE_THRESHOLD / 2, 1.1, True, "0.5", None,
                      float("nan"), float("inf"), 10 ** 400):
            invalid.append({**valid_box(), "score": value})
        for value in (None, [], [0, 0, 1], [0, 0, 1, 1, 1], [0, 0, 0, 1],
                      [1, 1, 0, 0], [-1, 0, 1, 1], [0, 0, detection.INPUT_SIZE + 1, 1],
                      [0, 0, 1, float("nan")], [0, 0, 1, float("inf")],
                      [False, False, True, True], [0, 0, 1, "1"], [0, 0, 1, 10 ** 400]):
            invalid.append({**valid_box(), "box": value})
        for index, value in enumerate(invalid):
            with self.subTest(index=index):
                with self.assertRaises(ValueError):
                    self.spec.validate_readiness(ready_output([value]))


@unittest.skipUnless(OPTIONAL_DETECTION, "optional torch/numpy/ONNX/Pillow detection stack not installed")
class DetectionOnnxTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        import onnx
        import torch
        parent = ROOT / ".runtime" / "tests"
        parent.mkdir(parents=True, exist_ok=True)
        cls.directory = tempfile.TemporaryDirectory(prefix="untrained-yolox-", dir=parent)
        cls.addClassCleanup(cls.directory.cleanup)
        cls.root = Path(cls.directory.name)
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        cls.addClassCleanup(torch.set_num_threads, previous_threads)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261012)
            cls.untrained_model = detection_training.build_model().eval()
        cls.input = np.zeros((1, 3, detection.INPUT_SIZE, detection.INPUT_SIZE), dtype=np.float32)
        with torch.inference_mode():
            cls.expected = cls.untrained_model(torch.from_numpy(cls.input)).numpy()
        buffer = io.BytesIO()
        with torch.no_grad():
            torch.onnx.export(cls.untrained_model, torch.from_numpy(cls.input), buffer,
                              dynamo=False, opset_version=17, input_names=["images"],
                              output_names=["detections"], external_data=False)
        proto = onnx.load_model_from_string(buffer.getvalue())
        onnx.helper.set_model_props(proto, detection.metadata())
        cls.content = proto.SerializeToString()
        cls.path = cls.root / "synthetic-untrained-yolox-nano.onnx"
        cls.artifact_hash = detection_training._save(cls.path, cls.content)
        cls.loaded = detection.load_model(cls.path, cls.artifact_hash)

    def graph(self):
        import onnx
        return onnx.load_model_from_string(self.content)

    def assert_rejected_graph(self, proto, message):
        content = proto.SerializeToString()
        path = self.root / (self.id().split(".")[-1] + ".onnx")
        path.write_bytes(content)
        with mock.patch("onnxruntime.InferenceSession", side_effect=AssertionError("Unsafe graph reached runtime")):
            with self.assertRaisesRegex(ValueError, message):
                detection.load_model(path, hashlib.sha256(content).hexdigest())

    def test_actual_untrained_architecture_export_load_and_raw_parity(self):
        import numpy as np
        self.assertEqual(self.loaded["recipe"], detection.RECIPE)
        self.assertEqual(self.loaded["contract"], detection.CONTRACT)
        self.assertEqual(self.loaded["format"], detection.FORMAT)
        self.assertEqual(self.loaded["dimensions"], 3 * detection.INPUT_SIZE ** 2)
        self.assertLessEqual(len(self.content), detection.MAX_ARTIFACT)
        self.assertEqual(self.expected.shape, (1, 525, 6))
        self.assertTrue(np.isfinite(self.expected).all())
        actual = self.loaded["session"].run(["detections"], {"images": self.input})[0]
        self.assertEqual(actual.shape, (1, 525, 6))
        np.testing.assert_allclose(actual, self.expected, rtol=1e-4, atol=1e-3)
        self.assertGreater(sum(parameter.numel() for parameter in self.untrained_model.parameters()), 500000)

    def test_actual_onnx_predict_and_readiness_allow_empty_results(self):
        spec = recipes.get_recipe(detection.RECIPE)
        output = detection.predict(self.loaded, spec.readiness_input(spec.budgets.max_dimensions))
        spec.validate_readiness(output)
        self.assertEqual(output, ready_output())
        result = detection.predict(self.loaded, [image(64, 32)])
        self.assertEqual(len(result), 1)
        self.assertEqual((result[0]["width"], result[0]["height"]), (64, 32))
        self.assertIsInstance(result[0]["detections"], list)
        self.assertLessEqual(len(result[0]["detections"]), 100)

    def test_preprocessing_rgb_to_bgr_float32_chw_padding_without_normalization(self):
        import numpy as np
        from PIL import Image
        array, transform = detection.preprocess(Image.new("RGB", (4, 2), (12, 34, 56)))
        self.assertEqual(array.shape, (3, detection.INPUT_SIZE, detection.INPUT_SIZE))
        self.assertEqual(array.dtype, np.float32)
        self.assertTrue(array.flags.c_contiguous)
        self.assertEqual((transform.resized_width, transform.resized_height), (160, 80))
        np.testing.assert_array_equal(array[:, 0, 0], [56, 34, 12])
        np.testing.assert_array_equal(array[:, 79, 159], [56, 34, 12])
        self.assertTrue(np.all(array[:, 80:, :] == 114))

    def test_predict_rejects_invalid_output_shapes_and_nonfinite_values(self):
        import numpy as np
        for output in (np.zeros((1, 525, 5), np.float32), np.zeros((2, 525, 6), np.float32),
                       np.full((1, 525, 6), np.nan, np.float32), np.full((1, 525, 6), np.inf, np.float32)):
            model = {"session": mock.Mock()}
            model["session"].run.return_value = [output]
            with self.assertRaisesRegex(ValueError, "Invalid detection outputs"):
                detection.predict(model, [image()])

    def test_predict_integrates_decoded_boxes_scores_and_original_dimensions(self):
        import numpy as np
        raw = np.zeros((1, 525, 6), dtype=np.float32)
        # A 64x32 image becomes 160x80 plus 80 rows of bottom padding.
        raw[0, 0] = [75, 37.5, 50, 25, 0.9, 0.8]
        raw[0, 1] = [75, 37.5, 50, 25, 0.8, 0.8]  # Suppressed same-class duplicate.
        raw[0, 2] = [75, 120, 10, 10, 1, 1]  # Entirely in bottom padding.
        session = mock.Mock()
        session.run.return_value = [raw]
        result = detection.predict({"session": session}, [image(64, 32)])
        self.assertEqual((result[0]["width"], result[0]["height"]), (64, 32))
        self.assertEqual(len(result[0]["detections"]), 1)
        box = result[0]["detections"][0]
        self.assertEqual(box["class_id"], 0)
        self.assertEqual(box["box"], [20.0, 10.0, 40.0, 20.0])
        self.assertAlmostEqual(box["score"], 0.72, places=6)
        names, inputs = session.run.call_args.args
        self.assertEqual(names, ["detections"])
        self.assertEqual(inputs["images"].shape, (1, 3, 160, 160))
        self.assertEqual(inputs["images"].dtype, np.float32)

    def test_bad_recipe_preprocessing_or_class_metadata_rejected(self):
        for key in ("recipe", "contract", "format", "preprocessing", "class_ids", "score_threshold"):
            graph = self.graph()
            next(item for item in graph.metadata_props if item.key == key).value = "wrong-contract"
            with self.subTest(key=key):
                self.assert_rejected_graph(graph, "metadata")
        graph = self.graph()
        duplicate = graph.metadata_props.add()
        duplicate.key, duplicate.value = graph.metadata_props[0].key, graph.metadata_props[0].value
        self.assert_rejected_graph(graph, "metadata")

    def test_unapproved_operator_or_custom_domain_rejected(self):
        import onnx
        graph = self.graph()
        graph.graph.node.append(onnx.helper.make_node("Relu", ["images"], ["unapproved"]))
        self.assert_rejected_graph(graph, "operator")
        graph = self.graph()
        graph.graph.node[0].domain = "unapproved.example"
        self.assert_rejected_graph(graph, "operator")

    def test_external_initializer_and_constant_tensor_rejected(self):
        import onnx
        for constant in (False, True):
            graph = self.graph()
            if constant:
                tensor = onnx.helper.make_tensor("external-constant", onnx.TensorProto.FLOAT, [1], [1.0])
            else:
                tensor = graph.graph.initializer[0]
            tensor.data_location = onnx.TensorProto.EXTERNAL
            item = tensor.external_data.add()
            item.key, item.value = "location", "../must-not-be-read.weights"
            if constant:
                graph.graph.node.append(onnx.helper.make_node("Constant", [], ["external"], value=tensor))
            with self.subTest(constant=constant):
                self.assert_rejected_graph(graph, "External detection tensors")

    def test_nested_graph_attributes_rejected(self):
        import onnx
        graph = self.graph()
        graph.graph.node[0].attribute.append(onnx.helper.make_attribute(
            "unapproved_nested", onnx.helper.make_graph([], "nested", [], [])))
        self.assert_rejected_graph(graph, "Nested detection graph")

    def test_dynamic_input_batch_rejected_even_for_valid_onnx(self):
        graph = self.graph()
        dimension = graph.graph.input[0].type.tensor_type.shape.dim[0]
        dimension.ClearField("dim_value")
        dimension.dim_param = "unbounded_batch"
        content = graph.SerializeToString()
        path = self.root / "dynamic-batch.onnx"
        path.write_bytes(content)
        with self.assertRaisesRegex(ValueError, "tensor contract"):
            detection.load_model(path, hashlib.sha256(content).hexdigest())


if __name__ == "__main__":
    unittest.main()
