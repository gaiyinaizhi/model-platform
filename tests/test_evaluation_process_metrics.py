"""Frozen synthetic measurement equality through the real bounded child transport."""
import importlib.util
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import patch

from model_platform.common import environment, encode
from model_platform import evaluation_process
from model_platform.training_process import ExecutionLifecycle
from tests import evaluation_measurement_fixture as fixture
from tests import test_model_evaluations as queue_fixture

OPTIONAL = all(importlib.util.find_spec(name) for name in (
    'numpy', 'onnxruntime', 'onnx', 'cv2', 'PIL', 'pycocotools'))


@unittest.skipUnless(sys.platform.startswith('linux') and OPTIONAL,
                     'Linux and locked optional independent-evaluation dependencies')
class EvaluationMeasurementTransportTests(unittest.TestCase):
    def setUp(self):
        self.queue = queue_fixture.QueueTests()
        self.queue.setUp()
        self.addCleanup(self.queue.doCleanups)
        self.runtime = self.queue.runtime
        self.root = Path(__file__).resolve().parents[1]

    def compare_case(self, case):
        fixture.prepare(self.runtime.home, case)
        item = self.queue.create(case)
        self.queue.evaluations.claim()
        direct = fixture.measure(self.runtime.home)
        # The source hash has already been checked by the real bootstrap. The
        # test installs only deterministic artifact/prediction substitutions;
        # real child main, read-only DB, transport, resize and metrics remain.
        original = evaluation_process.BOOTSTRAP
        marker = 'from model_platform.evaluation_child import main'
        self.assertIn(marker, original)
        injection = (marker + '\n'
                     'sys.path.insert(0, str(root / "tests"))\n'
                     'import evaluation_measurement_fixture as fixture\n'
                     'fixture.install()')
        lifecycle = ExecutionLifecycle()
        with self.runtime.lock('worker', inherited_lease=True) as lease, \
                patch.object(evaluation_process, 'BOOTSTRAP', original.replace(marker, injection)):
            payload = evaluation_process.supervise(
                self.runtime, item['id'], environment(), lease,
                time.monotonic() + 30, lifecycle=lifecycle)
        self.assertEqual(lifecycle.proof, 'child_reaped')
        self.assertEqual(payload, direct)
        self.assertEqual(encode(payload), encode(direct))
        # Supervision and child execution cannot publish state or evidence.
        with self.runtime.connection() as db:
            self.assertEqual(db.execute('SELECT status FROM model_evaluations WHERE id=?',
                                        (item['id'],)).fetchone()[0], 'running')
            self.assertEqual(db.execute('SELECT count(*) FROM model_evaluation_results').fetchone()[0], 0)
        self.assertEqual(list(self.queue.evaluations.root.iterdir()), [])
        return payload

    def test_nonzero_metrics_are_byte_identical_and_match_hand_computable_case(self):
        payload = self.compare_case('nonzero')
        result = payload['result']
        metrics = result['metrics']
        # Two GT-present classes: class0 one perfect prediction, class1 missed.
        # A prediction on absent-GT class2 contributes FP, not macro AP support.
        self.assertEqual(metrics['status'], 'evaluated')
        self.assertAlmostEqual(metrics['map_50'], .5)
        self.assertAlmostEqual(metrics['map_50_95'], .5)
        self.assertAlmostEqual(metrics['ar_100'], .5)
        self.assertEqual(result['operating_point']['totals'],
                         {'tp': 1, 'fp': 1, 'fn': 1, 'precision': .5, 'recall': .5})
        self.assertEqual(len(metrics['per_class']), 80)
        self.assertAlmostEqual(metrics['per_class'][0]['ap_50'], 1)
        self.assertEqual(metrics['per_class'][1]['ap_50'], 0)
        self.assertIsNone(metrics['per_class'][2]['ap_50'])
        self.assertEqual(metrics['per_class'][2]['predictions'], 1)
        first = payload['predictions'][0]
        self.assertEqual((first['width'], first['height']), (640, 320))
        self.assertEqual((first['request_width'], first['request_height']), (384, 192))
        self.assertEqual(len(result['wrong_examples']['items']), 2)

    def test_no_ground_truth_remains_null_not_zero_through_transport(self):
        payload = self.compare_case('no_ground_truth')
        result = payload['result']
        metrics = result['metrics']
        self.assertEqual(metrics['status'], 'no_ground_truth')
        for key in ('map_50', 'map_50_95', 'ar_100'):
            self.assertIsNone(metrics[key])
        self.assertEqual(result['operating_point']['totals'],
                         {'tp': 0, 'fp': 1, 'fn': 0, 'precision': 0, 'recall': None})
        self.assertTrue(all(row['ap_50'] is None for row in metrics['per_class']))
        self.assertEqual(len(metrics['per_class']), 80)


if __name__ == '__main__':
    unittest.main()
