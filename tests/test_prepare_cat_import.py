import io
import unittest
try:
    from PIL import Image
except ImportError:
    Image = None
from scripts.prepare_cat_import import derivative
from model_platform.dataset_detection import labels


@unittest.skipIf(Image is None, 'Pillow optional')
class CatDerivativeTests(unittest.TestCase):
    def test_rounding_uses_independent_actual_axes(self):
        source = Image.new('RGB', (511, 317), (10, 30, 40))
        box = {'class_id': 0, 'xyxy': [10.5, 20.25, 490.75, 302.5]}
        raw, text, size = derivative(source, [box])
        self.assertEqual(size, (256, 158))
        recovered = labels(text, *size)[0]['xyxy']
        for actual, expected in zip(recovered, [10.5*256/511, 20.25*158/317, 490.75*256/511, 302.5*158/317]):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(Image.open(io.BytesIO(raw)).size, size)

    def test_no_upsampling_and_explicit_negative(self):
        source = Image.new('RGB', (63, 47), 'white')
        raw, text, size = derivative(source, [])
        self.assertEqual(size, (63, 47))
        self.assertEqual(text, b'')
        self.assertEqual(Image.open(io.BytesIO(raw)).tobytes(), source.tobytes())

    def test_deterministic_png_and_box_boundary(self):
        source = Image.new('RGB', (64, 32), 'white')
        box = {'class_id': 0, 'xyxy': [0, 0, 64, 32]}
        self.assertEqual(derivative(source, [box]), derivative(source, [box]))
        self.assertEqual(labels(derivative(source, [box])[1], 64, 32)[0]['xyxy'], [0, 0, 64, 32])

    def test_reject_wrong_class_or_degenerate(self):
        image = Image.new('RGB', (64, 32))
        for box in [{'class_id': 1, 'xyxy': [0, 0, 20, 20]}, {'class_id': 0, 'xyxy': [10, 1, 10, 20]}]:
            with self.assertRaises(ValueError):
                derivative(image, [box])
