"""Offline boundaries and optional fresh asset/pinned upstream parity. No AP claims."""
import ast
import hashlib
import importlib.util
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock
from model_platform import pretrained_yolox as r

OPTIONAL = all(importlib.util.find_spec(p) for p in ('numpy', 'onnx', 'onnxruntime', 'cv2'))
ASSET = os.environ.get('MP_PRETRAINED_YOLOX_ASSET')
REFERENCE = os.environ.get('MP_YOLOX_REFERENCE_DIR')


class BoundaryTests(unittest.TestCase):
    def test_mapping(self):
        self.assertEqual(len(r.CLASS_NAMES), 80)
        self.assertEqual(len(set(r.COCO_CATEGORY_IDS)), 80)
        self.assertEqual((r.CLASS_NAMES[15], r.COCO_CATEGORY_IDS[15]), ('cat', 17))
        self.assertEqual(r.LABEL_TABLE_SHA256, '5655e1a2c432e4cbf3f7417f6d0aa90d42ea03ec0880951aeba2b5fcbb3afc9d')

    def test_hash_before_child_and_parse(self):
        with mock.patch.object(r.subprocess, 'run') as child:
            with self.assertRaises(ValueError):
                r._validate_bytes(b'bad')
            child.assert_not_called()
        with self.assertRaises(ValueError):
            r._static_check(b'bad')

    def test_secure_open(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.onnx'
            path.write_bytes(b'bad')
            with self.assertRaises(ValueError):
                r.validate_asset(path)
            with self.assertRaises(ValueError):
                r.validate_asset(directory)
            link = Path(directory) / 'link'
            link.symlink_to(path)
            with self.assertRaises(OSError):
                r.validate_asset(link)
            parent = Path(directory) / 'parent'
            parent.symlink_to(directory, target_is_directory=True)
            with self.assertRaises(OSError):
                r.validate_asset(parent / 'model.onnx')

    def test_wall_timeout(self):
        with mock.patch.object(r, '_check_bytes'), mock.patch.object(r.subprocess, 'run', side_effect=subprocess.TimeoutExpired('child', 30)):
            with self.assertRaisesRegex(ValueError, 'wall-time'):
                r._validate_bytes(b'')

    def test_probe_and_inputs(self):
        from model_platform.common import Problem
        r.validate_features(r.probe_input())
        r.probe_output([{'width': 32, 'height': 32, 'detections': []}])
        for features in ([{'width': 1, 'height': 1, 'pixels': [True, 0, 0]}],
                         [{'width': 385, 'height': 1, 'pixels': [0] * 1155}], []):
            with self.assertRaises(Problem):
                r.validate_features(features)
        with self.assertRaises(ValueError):
            r.probe_output([{'width': 32, 'height': 32, 'detections': [{'class_id': 80, 'score': .9, 'box': [0, 0, 1, 1]}]}])


@unittest.skipUnless(OPTIONAL, 'Optional CPU stack unavailable')
class ProtocolTests(unittest.TestCase):
    def test_nonsquare_preproc(self):
        import cv2
        import numpy as np
        for width, height in ((97, 53), (53, 97), (384, 1), (1, 384)):
            rgb = np.random.default_rng(7).integers(0, 256, (height, width, 3), dtype=np.uint8)
            actual, ratio = r.preprocess({'width': width, 'height': height, 'pixels': rgb.flatten().tolist()})
            expected_ratio = min(416 / height, 416 / width)
            padded = np.ones((416, 416, 3), dtype=np.uint8) * 114
            resized = cv2.resize(rgb[:, :, ::-1].copy(), (int(width * expected_ratio), int(height * expected_ratio)), interpolation=cv2.INTER_LINEAR).astype(np.uint8)
            padded[:resized.shape[0], :resized.shape[1]] = resized
            np.testing.assert_array_equal(actual, np.ascontiguousarray(padded.transpose(2, 0, 1), dtype=np.float32))
            self.assertEqual(ratio, expected_ratio)

    def test_decode_grid(self):
        import numpy as np
        raw = np.random.default_rng(9).uniform(0, 1, (1, 3549, 85)).astype(np.float32)
        expected = raw.copy()
        offset = 0
        for stride in (8, 16, 32):
            side = 416 // stride
            for y in range(side):
                for x in range(side):
                    i = offset + y * side + x
                    expected[0, i, :2] = (raw[0, i, :2] + [x, y]) * stride
                    expected[0, i, 2:4] = np.exp(raw[0, i, 2:4]) * stride
            offset += side * side
        np.testing.assert_array_equal(r.decode(raw), expected)

    def test_empty_and_nonfinite(self):
        import numpy as np
        raw = np.zeros((1, 3549, 85), np.float32)
        self.assertEqual(r.reference_detections(raw, 1.), [])
        for value in (float('inf'), float('nan'), 1000.):
            raw[0, 0, 2] = value
            with self.assertRaises(ValueError):
                r.decode(raw)

    def test_cross_class_nms(self):
        import numpy as np
        raw = np.zeros((1, 3549, 85), np.float32)
        raw[0, 0, :4] = [1, 1, 0, 0]
        raw[0, 1, :4] = [0, 1, 0, 0]
        raw[0, :2, 4] = 1
        raw[0, 0, 20] = .9
        raw[0, 1, 6] = .8
        result = r.reference_detections(raw, 1.)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['class_id'], 15)
        self.assertEqual(result[0]['box'], [4., 4., 12., 12.])

    def test_adapter_clips_drops_caps(self):
        import numpy as np
        session = mock.Mock()
        session.run.return_value = [np.zeros((1, 3549, 85), np.float32)]
        records = [{'class_id': 0, 'score': .9, 'box': [-5., -2., 40., 50.]},
                   {'class_id': 1, 'score': .8, 'box': [-5., -2., -1., 5.]}]
        records += [{'class_id': 2, 'score': .7, 'box': [1., 2., 3., 4.]}] * 150
        with mock.patch.object(r, 'reference_detections', return_value=records):
            result = r.predict({'session': session}, r.probe_input())
        self.assertEqual(len(result[0]['detections']), 100)
        self.assertEqual(result[0]['detections'][0]['box'], [0., 0., 32, 32])
        self.assertTrue(all(d['class_id'] != 1 for d in result[0]['detections']))
        r.probe_output(result)


@unittest.skipUnless(OPTIONAL and ASSET, 'Exact private asset required')
class ActualAssetTests(unittest.TestCase):
    def test_validation_load_predict_tamper(self):
        report = r.validate_asset(ASSET)
        self.assertEqual(report['status'], 'validated')
        self.assertTrue(report['pipeline_probe'])
        self.assertFalse(report['publisher_authenticated'])
        self.assertLess(report['peak_rss_bytes'], r.VALIDATION_ADDRESS_BYTES)
        model = r.load(ASSET, r.ASSET_SHA256)
        r.probe_output(r.predict(model, r.probe_input()))
        with self.assertRaises(ValueError):
            r.load(ASSET, '0' * 64)
        with tempfile.TemporaryDirectory() as directory:
            content = bytearray(Path(ASSET).read_bytes())
            content[-1] ^= 1
            path = Path(directory) / 'tampered.onnx'
            path.write_bytes(content)
            with mock.patch.object(r.subprocess, 'run') as child:
                with self.assertRaises(ValueError):
                    r.validate_asset(path)
                child.assert_not_called()


@unittest.skipUnless(OPTIONAL and ASSET and REFERENCE, 'Exact asset and pinned upstream references required')
class UpstreamParityTests(unittest.TestCase):
    def test_exact_upstream_helpers(self):
        import cv2
        import numpy as np
        namespace = {'np': np, 'cv2': cv2}
        sources = {'demo_utils.py': 'e1d6e1d263691ac15fdaa4f891efa83707da4af0fd895be122459105df3ee601',
                   'data_augment.py': 'f99b0b1568dafde0acd99f0a1e27a33644885349834d55a9dc632fe691ee352d',
                   'coco_classes.py': 'b38193c481a73f1f674cedab9e551b15b39b1a7aaed3e09e16505362cc54ad51'}
        selected = {'preproc', 'demo_postprocess', 'nms', 'multiclass_nms', 'multiclass_nms_class_agnostic', 'multiclass_nms_class_aware'}
        for name, digest in sources.items():
            content = (Path(REFERENCE) / name).read_bytes()
            self.assertEqual(hashlib.sha256(content).hexdigest(), digest)
            tree = ast.parse(content)
            if name == 'coco_classes.py':
                assignment = next(n for n in tree.body if isinstance(n, ast.Assign))
                self.assertEqual(ast.literal_eval(assignment.value), r.CLASS_NAMES)
            else:
                tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in selected]
                exec(compile(tree, name, 'exec'), namespace)
        session = r.load(ASSET, r.ASSET_SHA256)['session']
        samples = [np.random.default_rng(42).integers(0, 256, (h, w, 3), dtype=np.uint8) for w, h in ((97, 53), (53, 97), (32, 32))]
        if os.environ.get('MP_YOLOX_PHOTO_DIR'):
            photos = sorted(Path(os.environ['MP_YOLOX_PHOTO_DIR']).glob('*.jpg'))
            self.assertTrue(photos)
            for photo in photos:
                bgr = cv2.imread(str(photo))
                self.assertIsNotNone(bgr)
                scale = min(384 / bgr.shape[0], 384 / bgr.shape[1], 1.)
                bgr = cv2.resize(bgr, (int(bgr.shape[1] * scale), int(bgr.shape[0] * scale)), interpolation=cv2.INTER_LINEAR)
                samples.append(bgr[:, :, ::-1])
        photo_detections = 0
        for sample_index, rgb in enumerate(samples):
            h, w = rgb.shape[:2]
            actual, ratio = r.preprocess({'width': w, 'height': h, 'pixels': rgb.flatten().tolist()})
            reference, reference_ratio = namespace['preproc'](rgb[:, :, ::-1], (416, 416))
            np.testing.assert_array_equal(actual, reference)
            self.assertEqual(ratio, reference_ratio)
            raw = session.run(['output'], {'images': actual[None]})[0]
            np.testing.assert_array_equal(r.decode(raw), namespace['demo_postprocess'](raw.copy(), (416, 416)))
            if sample_index >= 3:
                photo_detections += len(r.reference_detections(raw, ratio))
            synthetic = raw.copy()
            synthetic[..., 4:] = 0
            synthetic[0, :100, 4] = .9
            for i in range(100):
                synthetic[0, i, 5 + i % 80] = .8
            for values in (raw, synthetic):
                decoded = namespace['demo_postprocess'](values.copy(), (416, 416))[0]
                corners = np.ones_like(decoded[:, :4])
                corners[:, :2] = decoded[:, :2] - decoded[:, 2:4] / 2.
                corners[:, 2:] = decoded[:, :2] + decoded[:, 2:4] / 2.
                corners /= ratio
                dets = namespace['multiclass_nms'](corners, decoded[:, 4:5] * decoded[:, 5:], .45, .3)
                expected = [] if dets is None else [{'box': row[:4].tolist(), 'score': float(row[4]), 'class_id': int(row[5])} for row in dets]
                self.assertEqual(r.reference_detections(values, ratio), expected)
        if os.environ.get('MP_YOLOX_PHOTO_DIR'):
            self.assertGreater(photo_detections, 0)
            print(f'Fresh upstream parity: {len(samples)-3} public photos, {photo_detections} reference detections')


if __name__ == '__main__':
    unittest.main()
