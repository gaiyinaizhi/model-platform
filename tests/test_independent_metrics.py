"""Synthetic arithmetic fixtures, not recognition-quality evidence."""
import copy
import unittest
from test_detection_evaluation import fixture, OPTIONAL_COCO
from model_platform.detection_evaluation import evaluate_detections, match_detections


class MatchingTests(unittest.TestCase):
    def test_perfect_and_immutable_inputs(self):
        data = fixture()
        original = copy.deepcopy(data)
        result = match_detections(*data)
        self.assertEqual(result['totals'], {'tp': 1, 'fp': 0, 'fn': 0, 'precision': 1.0, 'recall': 1.0})
        self.assertEqual(result['wrong_examples'], [])
        self.assertEqual(data, original)

    def test_empty_predictions(self):
        images, gt, _, cats = fixture()
        result = match_detections(images, gt, [], cats)
        self.assertEqual(result['totals'], {'tp': 0, 'fp': 0, 'fn': 1, 'precision': None, 'recall': 0.0})
        self.assertEqual(result['wrong_examples'][0]['false_negative_annotation_indices'], [0])

    def test_duplicates_preserved_false_positive(self):
        images, gt, dt, cats = fixture()
        dt.append(copy.deepcopy(dt[0]))
        result = match_detections(images, gt, dt, cats)
        self.assertEqual(result['totals'], {'tp': 1, 'fp': 1, 'fn': 0, 'precision': .5, 'recall': 1.0})
        self.assertEqual(result['per_image'][0]['matches'][0]['prediction_index'], 0)
        self.assertEqual(result['per_image'][0]['false_positive_prediction_indices'], [1])

    def test_no_gt_and_confused_class(self):
        images, gt, dt, cats = fixture()
        result = match_detections(images, [], dt, cats)
        self.assertEqual(result['totals'], {'tp': 0, 'fp': 1, 'fn': 0, 'precision': 0.0, 'recall': None})
        cats.append({'id': 1, 'name': 'dog'})
        dt[0]['category_id'] = 1
        result = match_detections(images, gt, dt, cats)
        self.assertEqual((result['totals']['tp'], result['totals']['fp'], result['totals']['fn']), (0, 1, 1))
        self.assertIsNone(result['per_class'][1]['recall'])

    def test_crowd_rejected_and_examples_bounded(self):
        images, gt, dt, cats = fixture()
        gt[0]['iscrowd'] = 1
        with self.assertRaises(ValueError):
            match_detections(images, gt, dt, cats)
        gt[0]['iscrowd'] = 0
        for limit in (-1, 65, True, 1.5):
            with self.assertRaises(ValueError):
                match_detections(images, gt, dt, cats, max_examples=limit)
        result = match_detections(images, gt, [], cats, max_examples=0)
        self.assertEqual(result['wrong_examples'], [])
        self.assertEqual(result['wrong_examples_total'], 1)

    def test_iou_threshold_inclusive_and_high_score_first(self):
        images, gt, dt, cats = fixture()
        dt[0]['bbox'] = [10, 10, 20, 30]  # exactly .5
        self.assertEqual(match_detections(images, gt, dt, cats)['totals']['tp'], 1)
        dt.append({**dt[0], 'score': .95})
        result = match_detections(images, gt, dt, cats)
        self.assertEqual(result['per_image'][0]['matches'][0]['prediction_index'], 1)
        self.assertEqual(result['per_image'][0]['false_positive_prediction_indices'], [0])

    def test_negative_image_and_empty_universe(self):
        images, gt, dt, cats = fixture()
        images.append({'id': 'negative', 'width': 100, 'height': 80})
        dt.append({**dt[0], 'image_id': 'negative'})
        result = match_detections(images, gt, dt, cats)
        self.assertEqual(result['totals']['fp'], 1)
        self.assertEqual(result['wrong_examples'][0]['image_id'], 'negative')
        empty = match_detections([], [], [], cats)
        self.assertEqual(empty['totals'], {'tp': 0, 'fp': 0, 'fn': 0, 'precision': None, 'recall': None})

    def test_highest_iou_and_equal_score_order(self):
        images, gt, dt, cats = fixture()
        gt.append({'image_id': 'positive', 'category_id': 0, 'bbox': [11, 10, 31, 30]})
        dt.append({**dt[0], 'bbox': [11, 10, 31, 30]})
        result = match_detections(images, gt, dt, cats)
        self.assertEqual([(m['prediction_index'],m['annotation_index']) for m in result['per_image'][0]['matches']], [(0,0),(1,1)])
        self.assertEqual(result, match_detections(images, gt, dt, cats))


@unittest.skipUnless(OPTIONAL_COCO, 'optional COCO dependencies absent')
class ARTests(unittest.TestCase):
    def test_ar_perfect_empty_no_gt_confusion(self):
        images, gt, dt, cats = fixture()
        perfect = evaluate_detections(images, gt, dt, cats)
        self.assertAlmostEqual(perfect['ar_100'], 1.0)
        self.assertNotIn('ar_100_iou_thresholds',perfect['parameters'])
        self.assertNotIn('ar_100_area',perfect['parameters'])
        self.assertAlmostEqual(perfect['per_class'][0]['ar_100'], 1.0)
        self.assertEqual(evaluate_detections(images, gt, [], cats)['ar_100'], 0)
        self.assertIsNone(evaluate_detections(images, [], dt, cats)['ar_100'])
        cats.append({'id': 1, 'name': 'dog'})
        dt[0]['category_id'] = 1
        self.assertEqual(evaluate_detections(images, gt, dt, cats)['ar_100'], 0)

    def test_ar_is_average_over_ten_ious(self):
        images, gt, dt, cats = fixture()
        dt[0]['bbox'] = [10, 10, 25, 30]  # IoU .75: six thresholds .50 through .75
        self.assertAlmostEqual(evaluate_detections(images, gt, dt, cats)['ar_100'], .6)
