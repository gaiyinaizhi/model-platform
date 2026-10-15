"""Synthetic fixtures verify metric arithmetic, never model recognition quality."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest

from model_platform.detection_evaluation import evaluate_detections


ROOT = Path(__file__).resolve().parents[1]
OPTIONAL_COCO = all(importlib.util.find_spec(name) is not None for name in ("numpy", "pycocotools"))


def fixture():
    return ([{"id": "positive", "width": 100, "height": 80}],
            [{"image_id": "positive", "category_id": 0, "bbox": [10, 10, 30, 30]}],
            [{"image_id": "positive", "category_id": 0, "bbox": [10, 10, 30, 30], "score": 0.9}],
            [{"id": 0, "name": "cat"}])


class DetectionEvaluationValidationTests(unittest.TestCase):
    def test_import_and_validation_do_not_load_optional_dependencies(self):
        script = """
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in ('numpy', 'pycocotools', 'torch', 'onnx'):
        raise AssertionError('optional dependency imported: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from model_platform.detection_evaluation import evaluate_detections
try:
    evaluate_detections([], [], [], [])
except ValueError:
    pass
else:
    raise AssertionError('empty categories accepted')
"""
        result = subprocess.run([sys.executable, "-c", script], cwd=ROOT,
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_optional_dependency_failure_is_explicit(self):
        script = """
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in ('numpy', 'pycocotools'):
        raise ImportError('not installed')
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from model_platform.detection_evaluation import evaluate_detections
try:
    evaluate_detections([], [], [], [{'id': 0, 'name': 'cat'}])
except RuntimeError as exc:
    assert 'requirements-detection.lock' in str(exc)
else:
    raise AssertionError('missing evaluator did not fail')
"""
        result = subprocess.run([sys.executable, "-c", script], cwd=ROOT,
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_container_shapes_and_unknown_or_missing_fields(self):
        for position in range(4):
            for invalid in (None, {}, "records", 1):
                data = list(fixture())
                data[position] = invalid
                with self.subTest(position=position, invalid=invalid), self.assertRaises(ValueError):
                    evaluate_detections(*data)
            data = list(fixture())
            data[position][0]["unexpected"] = True
            with self.subTest(position=position, field="unknown"), self.assertRaises(ValueError):
                evaluate_detections(*data)
            for key in fixture()[position][0]:
                data = list(fixture())
                del data[position][0][key]
                with self.subTest(position=position, key=key), self.assertRaises(ValueError):
                    evaluate_detections(*data)

    def test_duplicate_identifiers_and_ground_truth_records_rejected(self):
        for position in (0, 1, 3):
            data = list(fixture())
            data[position].append(copy.deepcopy(data[position][0]))
            with self.subTest(position=position), self.assertRaisesRegex(ValueError, "duplicate"):
                evaluate_detections(*data)
        for position in (1, 2):
            data = list(fixture())
            data[position][0]["id"] = "same-id"
            duplicate = copy.deepcopy(data[position][0])
            duplicate["bbox"] = [40, 40, 60, 60]
            data[position].append(duplicate)
            with self.subTest(position=position, duplicate_id=True), self.assertRaisesRegex(ValueError, "duplicate"):
                evaluate_detections(*data)

    def test_unknown_or_invalid_image_and_category_ids_rejected(self):
        for position in (1, 2):
            for key, invalid_values in (("image_id", ["missing", True, None, [], -1, "", " ", 1.0]),
                                        ("category_id", [1, -1, True, None, [], "0", 0.0])):
                for value in invalid_values:
                    data = list(fixture())
                    data[position][0][key] = value
                    with self.subTest(position=position, key=key, value=value), self.assertRaises(ValueError):
                        evaluate_detections(*data)
            for value in (True, None, [], -1, "", " ", 1.0):
                data = list(fixture())
                data[position][0]["id"] = value
                with self.subTest(position=position, optional_id=value), self.assertRaises(ValueError):
                    evaluate_detections(*data)

    def test_invalid_images_and_category_definitions(self):
        for dimension in ("width", "height"):
            for value in (0, -1, True, 100.0, float("inf"), None, 2 ** 31):
                data = list(fixture())
                data[0][0][dimension] = value
                with self.subTest(dimension=dimension, value=value), self.assertRaises(ValueError):
                    evaluate_detections(*data)
        for key, values in (("id", [-1, True, None, 1.0, [], "", " ", "x" * 257]),):
            for value in values:
                data = list(fixture())
                data[0][0][key] = value
                with self.subTest(image_id=value), self.assertRaises(ValueError):
                    evaluate_detections(*data)
        for categories in ([], [{"id": 1, "name": "cat"}], [{"id": True, "name": "cat"}],
                           [{"id": -1, "name": "cat"}], [{"id": "0", "name": "cat"}],
                           [{"id": 0, "name": ""}], [{"id": 0, "name": None}],
                           [{"id": 0, "name": "cat"}, {"id": 2, "name": "dog"}]):
            data = list(fixture())
            data[3] = categories
            with self.subTest(categories=categories), self.assertRaises(ValueError):
                evaluate_detections(*data)

    def test_invalid_boxes_and_scores_rejected_not_clipped(self):
        invalid_boxes = [None, "xyxy", [], [1, 2, 3], [1, 2, 3, 4, 5],
                         [10, 10, 10, 20], [30, 10, 10, 30], [10, 30, 30, 10],
                         [-1, 10, 30, 30], [10, -1, 30, 30], [10, 10, 101, 30],
                         [10, 10, 30, 81], [0, 0, 1e-300, 1e-300]]
        for index in range(4):
            for value in (float("nan"), float("inf"), -float("inf"), True, "10", None, 10 ** 1000):
                box = [10, 10, 30, 30]
                box[index] = value
                invalid_boxes.append(box)
        for position in (1, 2):
            for box in invalid_boxes:
                data = list(fixture())
                data[position][0]["bbox"] = box
                with self.subTest(position=position, box=box), self.assertRaises(ValueError):
                    evaluate_detections(*data)
        for score in (float("nan"), float("inf"), -float("inf"), -0.01, 1.01, True, "0.9", None, 10 ** 1000):
            data = list(fixture())
            data[2][0]["score"] = score
            with self.subTest(score=score), self.assertRaises(ValueError):
                evaluate_detections(*data)
        data = list(fixture())
        data[0][0].update(width=1000000, height=1000000)
        data[1][0]["bbox"] = [0, 0, 1000000, 1000000]
        with self.assertRaisesRegex(ValueError, "all-area"):
            evaluate_detections(*data)

    def test_crowd_ignore_group_annotations_fail_closed(self):
        for flag in ("iscrowd", "ignore", "group_of"):
            for value in (True, 1, -1, "0", None, [], 0.0):
                data = list(fixture())
                data[1][0][flag] = value
                with self.subTest(flag=flag, value=value), self.assertRaisesRegex(ValueError, flag):
                    evaluate_detections(*data)

    def test_nonboolean_per_class_option_rejected(self):
        for value in (0, 1, None, "false"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                evaluate_detections(*fixture(), include_per_class=value)


@unittest.skipUnless(OPTIONAL_COCO, "optional numpy/pycocotools dependencies not installed")
class DetectionEvaluationMetricTests(unittest.TestCase):
    def assert_ap(self, result, ap50, ap5095=None):
        self.assertAlmostEqual(result["map_50"], ap50, places=12)
        self.assertAlmostEqual(result["map_50_95"], ap50 if ap5095 is None else ap5095, places=12)

    def test_perfect_prediction_and_complete_protocol(self):
        result = evaluate_detections(*fixture())
        self.assert_ap(result, 1)
        self.assertEqual(result["status"], "evaluated")
        self.assertEqual(result["protocol"], "coco-style-bbox-subset/v1")
        self.assertEqual(result["evaluator"]["implementation"], "pycocotools.cocoeval.COCOeval")
        self.assertTrue(result["evaluator"]["version"])
        params = result["parameters"]
        self.assertEqual(params["iou_type"], "bbox")
        self.assertEqual([round(value, 2) for value in params["iou_thresholds"]],
                         [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])
        self.assertEqual(len(params["recall_thresholds"]), 101)
        self.assertEqual(params["recall_thresholds"][0], 0)
        self.assertEqual(params["recall_thresholds"][-1], 1)
        self.assertEqual(params["max_detections"], [1, 10, 100])
        self.assertEqual(params["area_range_labels"], ["all"])
        self.assertEqual(params["area_ranges"], [[0, 1e10]])
        self.assertIs(params["use_categories"], True)
        self.assertEqual(result["category_mapping"], [{"category_id": 0, "coco_category_id": 1, "name": "cat"}])
        self.assertEqual(result["counts"]["ground_truth_boxes"], 1)
        self.assertAlmostEqual(result["per_class"][0]["ap_50"], 1)
        json.dumps(result, allow_nan=False)

    def test_empty_predictions_and_nonoverlapping_prediction_have_zero_ap(self):
        data = list(fixture())
        data[2] = []
        result = evaluate_detections(*data)
        self.assert_ap(result, 0)
        self.assertEqual(result['status'], 'evaluated')
        self.assertEqual(result['counts']['predictions'], 0)
        self.assertEqual(result['counts']['ground_truth_boxes'], 1)
        self.assertEqual(result['per_class'][0]['status'], 'evaluated')
        data = list(fixture())
        data[2][0]["bbox"] = [50, 50, 70, 70]
        self.assert_ap(evaluate_detections(*data), 0)

    def test_missed_object_uses_101_point_interpolation(self):
        data = list(fixture())
        data[1].append({"image_id": "positive", "category_id": 0, "bbox": [40, 40, 60, 60]})
        self.assert_ap(evaluate_detections(*data), 51 / 101)

    def test_false_positive_before_true_positive_reduces_ap(self):
        data = list(fixture())
        data[2].append({"image_id": "positive", "category_id": 0, "bbox": [50, 50, 70, 70], "score": 0.99})
        self.assert_ap(evaluate_detections(*data), 0.5)

    def test_duplicate_detections_are_false_positives_not_deduplicated(self):
        data = list(fixture())
        data[1].append({"image_id": "positive", "category_id": 0, "bbox": [40, 40, 60, 60]})
        data[2].append({**data[2][0], "score": 0.8})
        data[2].append({"image_id": "positive", "category_id": 0, "bbox": [40, 40, 60, 60], "score": 0.7})
        result = evaluate_detections(*data)
        self.assert_ap(result, (51 + 50 * 2 / 3) / 101)
        self.assertEqual(result["counts"]["predictions"], 3)

    def test_negative_images_preserved_and_false_positives_counted(self):
        data = list(fixture())
        data[0].append({"id": "negative", "width": 100, "height": 80})
        result = evaluate_detections(*data)
        self.assert_ap(result, 1)
        self.assertEqual(result["counts"]["images_without_ground_truth"], 1)
        data[2].append({"image_id": "negative", "category_id": 0, "bbox": [10, 10, 30, 30], "score": 1})
        result = evaluate_detections(*data)
        self.assert_ap(result, 0.5)
        self.assertEqual(result["counts"]["images"], 2)
        self.assertEqual(result["counts"]["images_with_predictions"], 2)

    def test_multiclass_macro_ap_and_no_gt_class_unavailable(self):
        data = list(fixture())
        data[3] = [{"id": 2, "name": "absent"}, {"id": 1, "name": "dog"}, {"id": 0, "name": "cat"}]
        data[1].extend([{"image_id": "positive", "category_id": 0, "bbox": [40, 40, 60, 60]},
                        {"image_id": "positive", "category_id": 1, "bbox": [10, 10, 30, 30]}])
        data[2].extend([{"image_id": "positive", "category_id": 0, "bbox": [40, 40, 60, 60], "score": 0.7},
                        {"image_id": "positive", "category_id": 2, "bbox": [10, 10, 30, 30], "score": 1}])
        result = evaluate_detections(*data)
        self.assert_ap(result, 0.5)
        self.assertEqual([row["category_id"] for row in result["per_class"]], [0, 1, 2])
        self.assertAlmostEqual(result["per_class"][0]["ap_50"], 1)
        self.assertEqual(result["per_class"][1]["ap_50"], 0)
        self.assertIsNone(result["per_class"][2]["ap_50"])
        self.assertIsNone(result["per_class"][2]["ap_50_95"])
        self.assertEqual(result["per_class"][2]["status"], "no_ground_truth")
        self.assertEqual(result["counts"]["categories_with_ground_truth"], 2)

    def test_all_empty_ground_truth_has_no_ap_even_with_predictions(self):
        for with_images, with_predictions in ((False, False), (True, False), (True, True)):
            data = list(fixture())
            data[1] = []
            if not with_images:
                data[0] = []
            if not with_predictions:
                data[2] = []
            with self.subTest(images=with_images, predictions=with_predictions):
                result = evaluate_detections(*data)
                self.assertEqual(result["status"], "no_ground_truth")
                self.assertIsNone(result["map_50"])
                self.assertIsNone(result["map_50_95"])
                self.assertIsNone(result["per_class"][0]["ap_50"])
                json.dumps(result, allow_nan=False)

    def test_fractional_iou_passes_only_three_of_ten_thresholds(self):
        data = list(fixture())
        data[1][0]["bbox"] = [10, 10, 20, 20]
        data[2][0]["bbox"] = [10, 10, 18, 18]  # IoU = 64/100 = .64.
        self.assert_ap(evaluate_detections(*data), 1, 0.3)

    def test_exact_iou_threshold_is_inclusive(self):
        for width, average_ap in ((5, 0.1), (7.5, 0.6), (9.5, 1)):
            data = list(fixture())
            data[1][0]["bbox"] = [0, 0, 10, 10]
            data[2][0]["bbox"] = [0, 0, width, 10]
            with self.subTest(iou=width / 10):
                self.assert_ap(evaluate_detections(*data), 1, average_ap)

    def test_boundary_box_and_zero_score_are_not_filtered(self):
        data = list(fixture())
        data[1][0]["bbox"] = [0, 0, 100, 80]
        data[2][0].update(bbox=[0, 0, 100, 80], score=0)
        self.assert_ap(evaluate_detections(*data), 1)

    def test_continuous_pixels_no_inclusive_plus_one(self):
        data = list(fixture())
        data[1][0]["bbox"] = [0, 0, 1, 1]
        data[2][0]["bbox"] = [0, 0, 0.4, 1]  # IoU .4; a +1 bug would pass .5.
        self.assert_ap(evaluate_detections(*data), 0)

    def test_max_detections_cap_is_applied(self):
        data = list(fixture())
        data[2][0]["score"] = 0.1
        data[2].extend({"image_id": "positive", "category_id": 0, "bbox": [50, 50, 70, 70], "score": 0.9}
                       for _ in range(100))
        result = evaluate_detections(*data)
        self.assert_ap(result, 0)
        self.assertEqual(result["counts"]["predictions"], 101)
        self.assertEqual(result["counts"]["predictions_above_max_detections"], 1)

    def test_stable_score_ties_preserve_prediction_order(self):
        data = list(fixture())
        data[2].insert(0, {"image_id": "positive", "category_id": 0, "bbox": [50, 50, 70, 70], "score": 0.9})
        self.assert_ap(evaluate_detections(*data), 0.5)
        data[2].reverse()
        self.assert_ap(evaluate_detections(*data), 1)

    def test_score_ties_follow_image_input_order_before_prediction_order(self):
        data = list(fixture())
        data[0].insert(0, {"id": "negative", "width": 100, "height": 80})
        data[2].append({"image_id": "negative", "category_id": 0, "bbox": [50, 50, 70, 70], "score": 0.9})
        self.assert_ap(evaluate_detections(*data), 0.5)
        data[0].reverse()
        self.assert_ap(evaluate_detections(*data), 1)

    def test_typed_image_ids_and_optional_fields_remain_valid(self):
        data = list(fixture())
        data[0][0]["id"] = 0
        data[0].append({"id": "0", "width": 100, "height": 80})
        data[1][0].update(image_id=0, id=0, iscrowd=False, ignore=0, group_of=False)
        data[2][0].update(image_id=0, id="detection")
        data[2].append({"id": 0, "image_id": "0", "category_id": 0, "bbox": [10, 10, 30, 30], "score": 1})
        self.assert_ap(evaluate_detections(*data), 0.5)

    def test_input_not_mutated_and_per_class_can_be_omitted(self):
        data = fixture()
        original = copy.deepcopy(data)
        result = evaluate_detections(*data, include_per_class=False)
        self.assertNotIn("per_class", result)
        self.assertEqual(data, original)
        self.assert_ap(result, 1)


if __name__ == "__main__":
    unittest.main()
