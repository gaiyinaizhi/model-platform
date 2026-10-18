"""Pure validation of comparison-relevant fixed-v1 evaluation semantics.

Call after ModelEvaluationStore.detail has verified the captured record, model,
dataset files, result hash, and private prediction evidence. This supplements
those integrity checks; it does not rerun inference or trust hash consistency as
proof that metrics are well formed. Diagnostic examples and warning text retain
their existing BFF detail validation, and are not compatibility prerequisites.
"""
import json
import math
import re

from .common import digest, encode
from .model_evaluations import MAX_BOXES, MAX_DETAIL_BYTES, MAX_IMAGES, PROTOCOL, protocol
from .pretrained_yolox import CLASS_NAMES, COCO_CATEGORY_IDS


_PACKAGES = {'numpy', 'onnxruntime', 'opencv', 'Pillow', 'pycocotools'}
_PARAMETERS = {
    'iou_type': 'bbox', 'reported_max_detections': 100,
    'max_detections_scope': 'per_image_per_category', 'use_categories': True,
    'box_format': 'xyxy_original_image_pixels_continuous',
    'category_averaging': 'macro_over_categories_with_ground_truth',
    'no_ground_truth_ap': None, 'ar_100_area': 'all',
    'crowd_ignore_group_policy': 'rejected', 'score_filtering': 'none', 'nms': 'none',
    'score_tie_order': 'stable_image_input_order_then_prediction_input_order',
    'area_ranges': [[0, 10000000000]], 'area_range_labels': ['all'],
    'max_detections': [1, 10, 100],
}
_OPERATING = {'protocol': 'class-aware-greedy-iou50/v1', 'iou_threshold': .5,
              'score_threshold': .3, 'score_operator': '>'}


def _check(condition):
    if not condition:
        raise ValueError('Invalid comparison evaluation evidence')


def _object(value, keys=None):
    _check(type(value) is dict)
    if keys is not None:
        _check(value.keys() == set(keys))
    return value


def _fixed(actual, expected):
    """Exact JSON constants, including bool/number distinction and array order."""
    if type(expected) is dict:
        _object(actual, expected)
        for key, value in expected.items():
            _fixed(actual[key], value)
    elif type(expected) is list:
        _check(type(actual) is list and len(actual) == len(expected))
        for a, b in zip(actual, expected):
            _fixed(a, b)
    elif type(expected) in (int, float):
        _check(type(actual) in (int, float) and actual == expected)
    else:
        _check(type(actual) is type(expected) and actual == expected)


def _integer(value, maximum, minimum=0):
    _check(type(value) is int and minimum <= value <= maximum)
    return value


def _rate(value):
    _check(value is None or (type(value) in (int, float)
                            and 0 <= value <= 1 and math.isfinite(value)))
    return value


def _text(value, maximum, pattern):
    _check(type(value) is str and 1 <= len(value) <= maximum
           and re.fullmatch(pattern, value) is not None)
    return value


def _sha(value):
    return _text(value, 64, r'[a-f0-9]{64}')


def _version(value):
    return _text(value, 160, r'[A-Za-z0-9.+_-]+')


def _rows(value):
    _check(type(value) is list and len(value) == 80)
    return value


def _parameters(value):
    arrays = {'iou_thresholds': (10, .5, .05),
              'recall_thresholds': (101, 0, .01),
              'ar_100_iou_thresholds': (10, .5, .05)}
    p = _object(value, _PARAMETERS.keys() | arrays.keys())
    for key, constant in _PARAMETERS.items():
        _fixed(p[key], constant)
    for key, (length, base, step) in arrays.items():
        values = p[key]
        _check(type(values) is list and len(values) == length)
        for index, item in enumerate(values):
            _check(_rate(item) is not None and abs(item - (base + step * index)) <= 1e-12)
    # The numerical tolerance above matches the existing detail validator for
    # generated linspace floats. Compatibility still compares recorded values
    # exactly, without normalizing or tolerating differences between records.


def _ratios(value):
    row = _object(value)
    tp, fp, fn = (_integer(row[key], maximum) for key, maximum in
                  (('tp', MAX_BOXES), ('fp', MAX_IMAGES * 100), ('fn', MAX_BOXES)))
    for key, denominator in (('precision', tp + fp), ('recall', tp + fn)):
        actual = _rate(row[key])
        expected = tp / denominator if denominator else None
        _check(actual is expected if expected is None else
               actual is not None and abs(actual - expected) < 1e-12)
    return tp, fp, fn


def _metrics(result, dataset):
    metrics = _object(result['metrics'])
    _check(metrics['protocol'] == 'coco-style-bbox-subset/v1')
    evaluator = _object(metrics['evaluator'], ('implementation', 'version'))
    _check(evaluator['implementation'] == 'pycocotools.cocoeval.COCOeval')
    _version(evaluator['version'])
    _parameters(metrics['parameters'])
    counts = _object(metrics['counts'])
    for key, maximum, minimum in (
            ('images', MAX_IMAGES, 1), ('images_with_ground_truth', MAX_IMAGES, 0),
            ('images_without_ground_truth', MAX_IMAGES, 0), ('images_with_predictions', MAX_IMAGES, 0),
            ('ground_truth_boxes', MAX_BOXES, 0), ('predictions', MAX_IMAGES * 100, 0),
            ('predictions_above_max_detections', 0, 0), ('categories', 80, 80),
            ('categories_with_ground_truth', 80, 0)):
        _integer(counts[key], maximum, minimum)
    _check(counts['images'] == dataset['images'] and counts['ground_truth_boxes'] == dataset['boxes']
           and counts['images_with_ground_truth'] + counts['images_without_ground_truth'] == counts['images']
           and counts['images_with_ground_truth'] <= counts['ground_truth_boxes']
           and counts['images_with_predictions'] <= counts['images']
           and counts['images_with_predictions'] <= counts['predictions']
           and counts['predictions'] <= counts['images_with_predictions'] * 100
           and (not counts['ground_truth_boxes'] or counts['images_with_ground_truth'] > 0))
    rows = _rows(metrics['per_class'])
    fields = (('map_50', 'ap_50'), ('map_50_95', 'ap_50_95'), ('ar_100', 'ar_100'))
    for index, row in enumerate(rows):
        _object(row)
        _check(type(row['category_id']) is int and row['category_id'] == index
               and row['name'] == CLASS_NAMES[index])
        gt = _integer(row['ground_truth_boxes'], MAX_BOXES)
        predictions = _integer(row['predictions'], MAX_IMAGES * 100)
        _check(row['status'] == ('evaluated' if gt else 'no_ground_truth'))
        for _, field in fields:
            value = _rate(row[field])
            _check((value is not None and (predictions > 0 or value == 0)) if gt else value is None)
    present = [row for row in rows if row['ground_truth_boxes'] > 0]
    _check(sum(row['ground_truth_boxes'] for row in rows) == counts['ground_truth_boxes']
           and sum(row['predictions'] for row in rows) == counts['predictions']
           and len(present) == counts['categories_with_ground_truth']
           and metrics['status'] == ('evaluated' if present else 'no_ground_truth'))
    for field, class_field in fields:
        value = _rate(metrics[field])
        _check((value is not None and (counts['predictions'] > 0 or value == 0)) if present else value is None)
        if present:
            _check(abs(value - sum(row[class_field] for row in present) / len(present)) <= 1e-10)
    operating = _object(result['operating_point'])
    for key, constant in _OPERATING.items():
        _fixed(operating[key], constant)
    op_rows = _rows(operating['per_class'])
    for index, row in enumerate(op_rows):
        tp, fp, fn = _ratios(row)
        _check(type(row['category_id']) is int and row['category_id'] == index
               and row['name'] == CLASS_NAMES[index]
               and tp + fn == rows[index]['ground_truth_boxes']
               and tp + fp == rows[index]['predictions'])
    totals = operating['totals']
    _ratios(totals)
    _check(all(sum(row[key] for row in op_rows) == totals[key] for key in ('tp', 'fp', 'fn')))


def _environment(result, binding):
    keys = ('python', 'platform', 'code_sha256', 'dependencies')
    submission = _object(binding['submission_environment'], keys)
    execution = _object(result['environment'], keys)
    for env in (submission, execution):
        _version(env['python'])
        _text(env['platform'], 240, r'[A-Za-z0-9 ._()+-]+')
        _sha(env['code_sha256'])
    encoded = submission['dependencies']
    _check(type(encoded) is str and 1 <= len(encoded) <= 4096)
    pinned = _object(json.loads(encoded), _PACKAGES | {'onnx'})
    _check(encode(pinned).decode() == encoded)
    dependencies = _object(execution['dependencies'], _PACKAGES)
    for value in pinned.values():
        _version(value)
    _check(all(execution[key] == submission[key] for key in ('python', 'platform', 'code_sha256'))
           and all(value == pinned[key] for key, value in dependencies.items())
           and result['metrics']['evaluator']['version'] == dependencies['pycocotools'])


def validate(detail):
    """Raise ValueError on invalid comparison evidence; leave input unchanged.

    Never compares recorded environments to the current process environment.
    All loops are bounded by the existing 512-KiB detail and fixed v1 budgets.
    """
    try:
        _object(detail)
        _check(len(encode(detail)) <= MAX_DETAIL_BYTES and detail['status'] == 'succeeded'
               and detail['split'] == 'val' and detail['protocol'] == PROTOCOL)
        binding, result = _object(detail['binding']), _object(detail['result'])
        dataset = _object(binding['dataset'])
        _integer(dataset['images'], MAX_IMAGES, 1)
        _integer(dataset['boxes'], MAX_BOXES)
        _check(dataset['split'] == 'val' and dataset['purpose'] == 'evaluation'
               and type(dataset['samples']) is list and len(dataset['samples']) == dataset['images']
               and dataset['dataset_version_id'] == detail['dataset_version_id']
               and binding['model']['model_version_id'] == detail['model_version_id'])
        _fixed(binding['protocol'], protocol())
        _fixed(binding['categories'], [
            {'class_id': i, 'name': name, 'source_category_id': COCO_CATEGORY_IDS[i],
             'evaluator_internal_category_id': i + 1} for i, name in enumerate(CLASS_NAMES)])
        _check(binding['protocol_sha256'] == digest(binding['protocol'])
               and binding['category_mapping_sha256'] == digest(binding['categories']))
        _metrics(result, dataset)
        _environment(result, binding)
        source = dataset['source']
        expected = {'binding_sha256': digest(binding),
                    'model_artifact_sha256': binding['model']['artifact_sha256'],
                    'dataset_content_sha256': dataset['content_sha256'],
                    'category_mapping_sha256': binding['category_mapping_sha256'],
                    'protocol_sha256': binding['protocol_sha256'],
                    'source_selection_sha256': source['selection_sha256'] if source is not None else None,
                    'execution_code_sha256': binding['submission_environment']['code_sha256']}
        digests = _object(result['digests'], expected.keys() | {'prediction_sha256'})
        for key, value in digests.items():
            if key != 'source_selection_sha256' or value is not None:
                _sha(value)
        _check(all(digests[key] == value for key, value in expected.items())
               and _sha(detail['result_sha256']) == digest(result))
        _fixed(result['scope'], {'images': dataset['images'], 'split': 'val',
                                'official_benchmark': False, 'external_audit': False,
                                'production_allowed': False, 'device_certified': False})
    except (KeyError, TypeError, OverflowError, RecursionError):
        raise ValueError('Invalid comparison evaluation evidence') from None
