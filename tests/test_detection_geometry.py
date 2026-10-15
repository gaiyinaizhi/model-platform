"""Synthetic contract/geometry tests; these do not establish detection quality."""

from dataclasses import FrozenInstanceError
import subprocess
import sys
import unittest

from model_platform.common import Problem
from model_platform.detection_geometry import (
    LetterboxTransform, box_iou, class_aware_nms, letterbox_transform,
    postprocess_predictions, restore_box, validate_images,
)


def image(width=2, height=3):
    return {"width": width, "height": height, "pixels": [0, 114, 255] * (width * height)}


def detection(box, score=0.9, class_id=0):
    return {"box": box, "score": score, "class_id": class_id}


class DetectionImageContractTests(unittest.TestCase):
    def test_stdlib_only_import_and_execution(self):
        script = """
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in ('torch', 'onnx', 'onnxruntime', 'numpy', 'cv2', 'PIL'):
        raise AssertionError('Optional import: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from model_platform.detection_geometry import validate_images, letterbox_transform, postprocess_predictions
validate_images([{'width': 1, 'height': 1, 'pixels': [0, 114, 255]}])
assert postprocess_predictions([], letterbox_transform(1, 1), score_threshold=0.25, iou_threshold=0.45) == []
"""
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_default_and_parameterized_bounds(self):
        validate_images([image()])
        validate_images([image(512, 1)])
        validate_images([image(512, 512)])
        validate_images([image(), image()], max_images=2)
        validate_images([image(4, 4)], min_side=4, max_side=4)
        for value, kwargs in (([image(), image()], {}), ([image(513, 1)], {}),
                              ([image()], {"min_side": 3}), ([image(4, 4)], {"max_side": 3})):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(Problem):
                    validate_images(value, **kwargs)

    def test_invalid_bounds_are_configuration_errors(self):
        for kwargs in ({"max_images": 0}, {"max_images": True}, {"max_side": 0},
                       {"max_side": 1.0}, {"min_side": False}, {"min_side": 4, "max_side": 3}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    validate_images([image()], **kwargs)

    def test_batch_shape_and_unknown_fields(self):
        for value in (None, [], {}, "https://example.invalid/image", (image(),), [None], ["/tmp/image"],
                      [{**image(), "path": "/tmp/image"}], [{**image(), "url": "https://example.invalid"}],
                      [{"width": 2, "height": 3}], [{"pixels": [0, 0, 0]}]):
            with self.subTest(value_type=type(value).__name__):
                with self.assertRaises(Problem) as failed:
                    validate_images(value)
                self.assertEqual(failed.exception.status, 400)
                self.assertEqual(failed.exception.code, "invalid_input")

    def test_dimensions_must_be_positive_bounded_integers(self):
        for side in (-1, 0, 513, True, 2.0, "2", None, float("nan"), float("inf")):
            for axis in ("width", "height"):
                with self.subTest(side=side, axis=axis):
                    value = image()
                    value[axis] = side
                    with self.assertRaises(Problem):
                        validate_images([value])

    def test_pixel_length_hwc_range_and_integer_type(self):
        invalid = (None, "000", (0, 0, 0), [], [0] * 17, [0] * 19,
                   [[0, 0, 0]] * 6, [True] * 18, [0.0] * 18, [-1] * 18,
                   [256] * 18, [float("nan")] * 18, [float("inf")] * 18, ["0"] * 18)
        for pixels in invalid:
            with self.subTest(pixel_type=type(pixels).__name__):
                with self.assertRaises(Problem):
                    validate_images([{**image(), "pixels": pixels}])


class LetterboxGeometryTests(unittest.TestCase):
    def test_landscape_top_left_padding_and_half_open_boundary(self):
        transform = letterbox_transform(200, 100, target_width=160, target_height=160)
        self.assertEqual((transform.resized_width, transform.resized_height), (160, 80))
        self.assertEqual((transform.scale_x, transform.scale_y), (0.8, 0.8))
        self.assertEqual((transform.pad_left, transform.pad_top, transform.pad_right, transform.pad_bottom),
                         (0, 0, 0, 80))
        self.assertEqual(transform.pad_value, 114)
        self.assertEqual(restore_box([80, 40, 160, 80], transform), [0.0, 0.0, 200.0, 100.0])

    def test_portrait_top_left_padding(self):
        transform = letterbox_transform(100, 200, target_width=160, target_height=160)
        self.assertEqual((transform.resized_width, transform.resized_height), (80, 160))
        self.assertEqual((transform.pad_right, transform.pad_bottom), (80, 0))
        self.assertEqual(restore_box([40, 80, 80, 160], transform), [0.0, 0.0, 100.0, 200.0])

    def test_floor_resize_uses_distinct_realized_scales(self):
        transform = letterbox_transform(7, 3, target_width=10, target_height=10)
        self.assertEqual((transform.resized_width, transform.resized_height), (10, 4))
        self.assertEqual(transform.scale_x, 10 / 7)
        self.assertEqual(transform.scale_y, 4 / 3)
        self.assertNotEqual(transform.scale_x, transform.scale_y)
        self.assertEqual(restore_box([5, 2, 10, 4], transform), [0.0, 0.0, 7.0, 3.0])
        self.assertEqual(restore_box([5, 2, 5, 2], transform), [1.75, 0.75, 5.25, 2.25])

    def test_non_square_target_and_upscaling(self):
        transform = letterbox_transform(3, 7, target_width=10, target_height=20)
        self.assertEqual((transform.resized_width, transform.resized_height), (8, 20))
        self.assertEqual((transform.pad_right, transform.pad_bottom), (2, 0))
        self.assertEqual(restore_box([4, 10, 8, 20], transform), [0.0, 0.0, 3.0, 7.0])

    def test_clipping_partial_and_full_padding_boxes(self):
        transform = letterbox_transform(200, 100, target_width=160, target_height=160)
        self.assertEqual(restore_box([0, 0, 40, 40], transform), [0.0, 0.0, 25.0, 25.0])
        self.assertEqual(restore_box([160, 80, 40, 40], transform), [175.0, 75.0, 200.0, 100.0])
        self.assertEqual(restore_box([80, 80, 320, 320], transform), [0.0, 0.0, 200.0, 100.0])
        for box in ([80, 120, 20, 20], [-20, 0, 20, 20], [0, 0, 0, 20],
                    [0, 0, 20, -1], [170, 40, 20, 20]):
            with self.subTest(box=box):
                self.assertIsNone(restore_box(box, transform))

    def test_invalid_dimensions_transform_and_nonfinite_boxes(self):
        for side in (0, -1, True, 1.0, float("inf")):
            with self.subTest(side=side):
                with self.assertRaises(ValueError):
                    letterbox_transform(side, 2)
                with self.assertRaises(ValueError):
                    letterbox_transform(2, 2, target_height=side)
        with self.assertRaisesRegex(ValueError, "empty resized"):
            letterbox_transform(1, 512, target_width=160, target_height=160)
        with self.assertRaises(ValueError):
            LetterboxTransform(100, 200, 160, 160, 160, 160)
        transform = letterbox_transform(10, 10)
        with self.assertRaises(FrozenInstanceError):
            transform.original_width = 20
        with self.assertRaises(ValueError):
            restore_box([1, 2, 3, 4], None)
        for box in (None, [0, 0, 1], [0, 0, True, 1], [0, 0, float("inf"), 1],
                    [float("nan"), 0, 1, 1], [10 ** 400, 0, 1, 1], [1e308, 0, 1.7e308, 1]):
            with self.subTest(box_type=type(box).__name__):
                with self.assertRaises(ValueError):
                    restore_box(box, transform)


class DetectionPostprocessingTests(unittest.TestCase):
    def setUp(self):
        self.transform = letterbox_transform(100, 100, target_width=100, target_height=100)

    def process(self, rows, **kwargs):
        options = {"num_classes": 2, "score_threshold": 0.25, "iou_threshold": 0.5}
        options.update(kwargs)
        return postprocess_predictions(rows, self.transform, **options)

    def test_objectness_product_threshold_and_zero_based_class_ids(self):
        rows = [[20, 20, 10, 10, 0.5, 0.4, 0.5],
                [40, 40, 10, 10, 0.499, 0.4, 0.5],
                [60, 60, 10, 10, 0.8, 0.9, 0.1]]
        result = self.process(rows)
        self.assertEqual([item["class_id"] for item in result], [0, 1])
        self.assertAlmostEqual(result[0]["score"], 0.72)
        self.assertEqual(result[1], detection([15.0, 15.0, 25.0, 25.0], 0.25, 1))

    def test_class_ties_choose_smallest_id_and_score_ties_keep_row_order(self):
        rows = [[20, 20, 10, 10, 1.0, 0.5, 0.5],
                [40, 40, 10, 10, 1.0, 0.1, 0.5],
                [60, 60, 10, 10, 1.0, 0.5, 0.1]]
        for _ in range(3):
            result = self.process(rows)
            self.assertEqual([item["class_id"] for item in result], [0, 1, 0])
            self.assertEqual([item["box"][0] for item in result], [15.0, 35.0, 55.0])

    def test_class_aware_suppression_and_fixed_output_cap(self):
        rows = [[20, 20, 10, 10, 1.0, 0.9, 0.1],
                [20, 20, 10, 10, 1.0, 0.8, 0.1],
                [20, 20, 10, 10, 1.0, 0.1, 0.7]]
        result = self.process(rows)
        self.assertEqual([item["class_id"] for item in result], [0, 1])
        self.assertEqual(self.process(rows, max_detections=1), result[:1])
        candidates = [detection([i * 2, 0, i * 2 + 1, 1]) for i in range(105)]
        self.assertEqual(len(class_aware_nms(candidates, iou_threshold=0.5)), 100)

    def test_iou_has_no_inclusive_plus_one_and_threshold_is_strict(self):
        first, adjacent, overlapping = [0, 0, 1, 1], [1, 0, 2, 1], [0.5, 0, 1.5, 1]
        self.assertEqual(box_iou(first, adjacent), 0.0)
        self.assertEqual(box_iou(first, first), 1.0)
        self.assertAlmostEqual(box_iou(first, overlapping), 1 / 3)
        candidates = [detection(first), detection(overlapping, 0.8)]
        self.assertEqual(len(class_aware_nms(candidates, iou_threshold=1 / 3)), 2)
        self.assertEqual(len(class_aware_nms(candidates, iou_threshold=0.333)), 1)
        self.assertEqual(len(class_aware_nms([detection(first), detection(adjacent)], iou_threshold=0)), 2)
        self.assertEqual(len(class_aware_nms([detection(first), detection(first)], iou_threshold=1)), 2)

    def test_empty_results_and_degenerate_boxes(self):
        self.assertEqual(self.process([]), [])
        self.assertEqual(self.process([[10, 10, 1, 1, 0, 1, 1]]), [])
        self.assertEqual(self.process([[10, 10, 0, 1, 1, 1, 1], [120, 10, 10, 10, 1, 1, 1]]), [])
        for box in ([0, 0, 0, 1], [2, 2, 1, 1]):
            self.assertEqual(box_iou(box, [0, 0, 10, 10]), 0.0)
            self.assertEqual(class_aware_nms([detection(box)], iou_threshold=0.5), [])

    def test_nms_does_not_mutate_or_alias_inputs(self):
        source = [detection([0, 0, 1, 1], 0.5), detection([2, 2, 3, 3], 0.9)]
        result = class_aware_nms(source, iou_threshold=0.5)
        self.assertEqual(source[0]["score"], 0.5)
        result[0]["box"][0] = 99
        result[0]["score"] = 0
        self.assertEqual(source[1], detection([2, 2, 3, 3], 0.9))

    def test_invalid_thresholds_counts_and_classes(self):
        for field in ("score_threshold", "iou_threshold"):
            for value in (-0.1, 1.1, True, float("nan"), float("inf"), "0.5"):
                with self.subTest(field=field, value=value):
                    with self.assertRaises(ValueError):
                        self.process([], **{field: value})
        for field, values in (("max_detections", (0, 101, True, 1.0)),
                              ("num_classes", (0, -1, True, 2.0))):
            for value in values:
                with self.subTest(field=field, value=value):
                    with self.assertRaises(ValueError):
                        self.process([], **{field: value})

    def test_every_model_value_is_finite_including_below_threshold(self):
        valid = [20, 20, 10, 10, 0.0, 0.0, 0.0]
        for index in range(len(valid)):
            for value in (float("nan"), float("inf"), -float("inf"), True, "0.0", 10 ** 400):
                with self.subTest(index=index, value_type=type(value).__name__):
                    row = valid[:]
                    row[index] = value
                    with self.assertRaises(ValueError):
                        self.process([row])
        for index in range(4, len(valid)):
            for value in (-0.1, 1.1):
                row = valid[:]
                row[index] = value
                with self.assertRaises(ValueError):
                    self.process([row])

    def test_malformed_rows_and_invalid_tail_after_output_limit(self):
        for rows in (None, {}, "rows", [None], [[1, 2, 3, 4, 0.5, 0.5]],
                     [[1, 2, 3, 4, 0.5, 0.5, 0.5, 0.5]]):
            with self.subTest(row_type=type(rows).__name__):
                with self.assertRaises(ValueError):
                    self.process(rows)
        with self.assertRaises(ValueError):
            self.process([[20, 20, 10, 10, 1, 1, 0], [30, 30, 10, 10, 1, 0, float("nan")]],
                         max_detections=1)

    def test_invalid_nms_objects_class_ids_scores_and_boxes(self):
        invalid = [None, {}, {**detection([0, 0, 1, 1]), "extra": 1}]
        for class_id in (-1, True, 0.0, "0"):
            invalid.append(detection([0, 0, 1, 1], class_id=class_id))
        for score in (-0.1, 1.1, True, float("nan")):
            invalid.append(detection([0, 0, 1, 1], score=score))
        for box in ([0, 0, 1], [0, 0, 1, float("inf")], [0, 0, 1e308, 1e308]):
            invalid.append(detection(box))
        for item in invalid:
            with self.subTest(item_type=type(item).__name__):
                with self.assertRaises(ValueError):
                    class_aware_nms([item], iou_threshold=0.5)
        with self.assertRaises(ValueError):
            class_aware_nms([detection([0, 0, 1, 1]), detection([0, 0, 1, float("nan")])],
                            iou_threshold=0.5, max_detections=1)


if __name__ == "__main__":
    unittest.main()
