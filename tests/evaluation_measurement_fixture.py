"""Self-generated RGB/labels and deterministic detector outputs, never real assets.

The production measurement function and official metric implementation remain
unchanged. Only imported artifact loading and detector outputs are replaced.
This is metric/transport equivalence evidence, not ONNX or detector-quality QA.
"""
import hashlib
import importlib.metadata
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from model_platform import model_evaluations as evaluation
from model_platform import model_imports, pretrained_yolox
from model_platform.common import digest

ORIGINAL_MEASURE = evaluation.measure
DATASET_ID = 'dsv_' + '9' * 32


def versions():
    return {name: importlib.metadata.version(package) for name, package in (
        ('numpy', 'numpy'), ('onnxruntime', 'onnxruntime'), ('onnx', 'onnx'),
        ('opencv', 'opencv-python-headless'), ('Pillow', 'Pillow'),
        ('pycocotools', 'pycocotools'))}


def prepare(home, case):
    """Generate fresh synthetic inputs in the caller's test-owned directory."""
    import numpy as np
    from PIL import Image
    home = Path(home)
    root = home / 'datasets' / 'versions' / DATASET_ID
    (root / 'images' / 'val').mkdir(parents=True, exist_ok=True)
    (root / 'labels' / 'val').mkdir(parents=True, exist_ok=True)
    rows = [(1, 640, 320, '0 0.5 0.5 0.6 0.6\n'), (2, 20, 10, '1 0.5 0.5 0.6 0.6\n')]
    if case == 'no_ground_truth':
        rows = [(1, 640, 320, '')]
    samples = []
    for marker, width, height, label in rows:
        image_path = f'images/val/{marker}.png'
        label_path = f'labels/val/{marker}.txt'
        rgb = np.full((height, width, 3), marker, dtype=np.uint8)
        Image.fromarray(rgb).save(root / image_path)
        (root / label_path).write_text(label, encoding='utf-8')
        samples.append({'image': image_path, 'label': label_path,
                        'width': width, 'height': height,
                        'image_sha256': hashlib.sha256((root / image_path).read_bytes()).hexdigest(),
                        'label_sha256': hashlib.sha256(label.encode()).hexdigest(),
                        'pixel_sha256': evaluation._pixel_digest(rgb)})
    data = {'case': case, 'samples': samples, 'boxes': sum(bool(row[3]) for row in rows)}
    (home / 'measurement-case.json').write_text(json.dumps(data), encoding='utf-8')


def measure(home):
    home = Path(home)
    data = json.loads((home / 'measurement-case.json').read_text())
    categories = [{'class_id': index, 'name': name,
                   'source_category_id': pretrained_yolox.COCO_CATEGORY_IDS[index],
                   'evaluator_internal_category_id': index + 1}
                  for index, name in enumerate(pretrained_yolox.CLASS_NAMES)]
    with patch.object(evaluation, 'installed_versions', side_effect=versions):
        binding = {'model': {'model_version_id': 'model_' + '8' * 32,
                             'artifact_sha256': 'b' * 64},
                   'dataset': {'dataset_version_id': DATASET_ID,
                               'content_sha256': digest(data['samples']),
                               'purpose': 'evaluation', 'split': 'val',
                               'images': len(data['samples']), 'boxes': data['boxes'],
                               'samples': data['samples'], 'source': None},
                   'categories': categories, 'category_mapping_sha256': digest(categories),
                   'protocol': evaluation.protocol(), 'protocol_sha256': digest(evaluation.protocol()),
                   'submission_environment': evaluation._env()}
    runtime = SimpleNamespace(home=home)
    store = SimpleNamespace(model=lambda *_: {'artifact_id': 'synthetic-only'})
    evaluator = SimpleNamespace(runtime=runtime, store=store,
        verify_binding=lambda _row: binding,
        verify_code=lambda *_: None,
        expected_digests=evaluation.ModelEvaluationStore.expected_digests)
    row = {'tenant': 'demo', 'model_version_id': binding['model']['model_version_id'],
           'dataset_version_id': DATASET_ID, 'binding_sha256': digest(binding)}

    def predict(_model, images):
        item = images[0]
        w, h = item['width'], item['height']
        detections = []
        if item['pixels'][0] == 1:
            detections = [{'class_id': 0, 'score': .9,
                           'box': [.2 * w, .2 * h, .8 * w, .8 * h]}]
            if data['case'] != 'no_ground_truth':
                detections.append({'class_id': 2, 'score': .8,
                                   'box': [0., 0., .1 * w, .1 * h]})
        return [{'width': w, 'height': h, 'detections': detections}]

    with patch.object(evaluation, 'installed_versions', side_effect=versions), \
            patch.object(model_imports, 'verify_artifact', return_value=home / 'unused-synthetic-artifact'), \
            patch.object(pretrained_yolox, '_read', return_value=b'synthetic detector fixture, not ONNX'), \
            patch.object(pretrained_yolox, '_static_check'), \
            patch.object(pretrained_yolox, '_session', return_value=None), \
            patch.object(pretrained_yolox, 'predict', side_effect=predict):
        return ORIGINAL_MEASURE(evaluator, row)


def install():
    """Patch only the test child; product source contains no fixture entry point."""
    from model_platform import evaluation_child
    wrapper = lambda evaluations, _row: measure(evaluations.runtime.home)
    evaluation.measure = wrapper
    evaluation_child.measure = wrapper
