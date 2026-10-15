import hashlib
import unittest
try:
    from PIL import Image
except ImportError:
    Image = None


@unittest.skipIf(Image is None, 'Pillow optional')
class LearnabilityFixtureTests(unittest.TestCase):
    def test_reproducible_disjoint_and_valid(self):
        from scripts.make_learnability_fixture import generate, CONFIG
        from model_platform.dataset_validation import Validation
        raw, evidence = generate()
        self.assertEqual(raw, generate()[0])
        self.assertEqual(hashlib.sha256(raw).hexdigest(), evidence['archive_sha256'])
        self.assertEqual(len({r['pixel_sha256'] for r in evidence['samples']}), 160)
        self.assertEqual(len({r['generation_seed'] for r in evidence['samples']}), 160)
        self.assertEqual(CONFIG, {'epochs': 20, 'learning_rate': .001, 'seed': 20261015})
        report, manifest, _ = Validation('reject').validate(raw)
        self.assertTrue(report['valid'], report)
        self.assertEqual([report['splits'][s]['images'] for s in ('train','val','test')], [128,16,16])
        self.assertEqual(report['counts']['boxes'], 320)
        self.assertEqual(len(report['classes']), 2)
        self.assertLess(sum(s['width']*s['height'] for s in manifest['samples']), 20_000_000)
