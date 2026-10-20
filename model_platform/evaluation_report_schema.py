"""Portable, stdlib-only independent evaluation report v1 validation.

Semantic helpers mirror evaluation_comparison_validation; parity is tested.
No runtime, database, optional ML library, network, or authentication imports.
"""
import hashlib
import json
import math
import re

MAX_IMAGES = 64
MAX_BOXES = 10000
MAX_REPORT_BYTES = 512 * 1024
MAX_DEPTH = 24
MAX_NODES = 20000
MAX_STRING = 8192
MAX_NUMBER_TOKEN = 64
PROTOCOL = 'imported-coco-serving-evaluation/v1'
SCHEMA_VERSION = 'independent-evaluation-report/v1'
CANONICALIZATION = 'python-json-sort-keys-ascii-compact/v1'


def encode(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True,
                      separators=(',', ':'), allow_nan=False).encode('ascii')


def digest(value):
    return hashlib.sha256(encode(value)).hexdigest()

CLASS_NAMES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')
COCO_CATEGORY_IDS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90)
def protocol():
    return {'id': PROTOCOL, 'inference_protocol': 'yolox-nano-coco-onnx/v1',
            'max_input_side': 384, 'input_size': 416, 'score_threshold': .3, 'score_operator': '>',
            'nms_iou_threshold': .45, 'nms_class_agnostic': True, 'max_detections': 100,
            'resize': 'opencv-inter-linear-floor-max384-no-upscale/v1',
            'metric_protocol': 'coco-style-bbox-subset/v1', 'matching_protocol': 'class-aware-greedy-iou50/v1'}


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


WARNINGS = [
    'Fixed non-crowd subset at the serving operating point; not the official COCO benchmark.',
    'All80 classes retained; categories without ground truth excluded from AP/AR macro averages.',
    'Dataset source and label completeness are declared provenance, not an external audit.',
    'COCO pretraining overlap is not independently excluded; weight license remains unverified.',
    'Logical budgets and process separation are not production multi-tenant isolation.',
    'Uses unchanged server inference with declared OpenCV caller resize; browser JPEG decoding or resizing may differ.',
]
VERIFICATION = {'internal_consistency_only': True, 'trusted_authenticity': False,
                'signature_present': False, 'reevaluated': False,
                'source_result_bytes_included': False, 'source_binding_bytes_included': False,
                'prediction_bytes_included': False}
LIMITATIONS = {'production_allowed': False, 'official_coco_benchmark': False,
               'device_certified': False, 'weight_license_verified': False,
               'pretraining_overlap': 'unknown'}
EVALUATION_KEYS = ('id', 'status', 'model_version_id', 'dataset_version_id', 'split',
                   'protocol', 'created_at', 'started_at', 'finished_at')
MODEL_KEYS = ('model_version_id', 'origin', 'import_id', 'validation_id', 'catalog_id',
              'catalog_revision', 'artifact_sha256', 'artifact_bytes', 'manifest_sha256')
DATASET_KEYS = ('dataset_version_id', 'content_sha256', 'purpose', 'split', 'images',
                'boxes', 'source_selection_sha256')
RESULT_KEYS = ('metrics', 'operating_point', 'environment', 'scope', 'warnings')
PROVENANCE_KEYS = ('model', 'dataset', 'categories', 'category_mapping_sha256',
                   'protocol', 'protocol_sha256', 'submission_environment')
TOP_KEYS = ('schema_version', 'canonicalization', 'evaluation', 'provenance', 'exported_result',
            'source_result_sha256', 'source_binding_sha256', 'source_prediction_sha256',
            'exported_result_sha256', 'verification', 'limitations', 'report_sha256')


def _identity(value, prefix):
    return _text(value, 80, prefix + r'_[0-9a-f]{32}')


def _bounds(value):
    # Iterative traversal bounds also protect programmatic server callers.
    stack = [(value, 0)]
    nodes = 0
    while stack:
        item, depth = stack.pop()
        nodes += 1
        _check(nodes <= MAX_NODES and depth <= MAX_DEPTH)
        if type(item) is dict:
            _check(all(type(k) is str and len(k) <= MAX_STRING for k in item))
            stack.extend((v, depth + 1) for v in item.values())
        elif type(item) is list:
            stack.extend((v, depth + 1) for v in item)
        elif type(item) is str:
            _check(len(item) <= MAX_STRING)
        elif type(item) in (int, float):
            _check(math.isfinite(item) and len(str(item)) <= MAX_NUMBER_TOKEN)
        else:
            _check(item is None or type(item) is bool)


def _validate(value):
    _bounds(value)
    report = _object(value, TOP_KEYS)
    _fixed(report['schema_version'], SCHEMA_VERSION)
    _fixed(report['canonicalization'], CANONICALIZATION)
    evaluation = _object(report['evaluation'], EVALUATION_KEYS)
    _identity(evaluation['id'], 'mev')
    _identity(evaluation['model_version_id'], 'model')
    _identity(evaluation['dataset_version_id'], 'dsv')
    _fixed(evaluation['status'], 'succeeded')
    _fixed(evaluation['split'], 'val')
    _fixed(evaluation['protocol'], PROTOCOL)
    times = [evaluation[k] for k in ('created_at', 'started_at', 'finished_at')]
    _check(all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1e12 for v in times))
    _check(times[0] <= times[1] <= times[2])
    provenance = _object(report['provenance'], PROVENANCE_KEYS)
    model = _object(provenance['model'], MODEL_KEYS)
    _check(model['model_version_id'] == evaluation['model_version_id'])
    _identity(model['import_id'], 'import')
    _identity(model['validation_id'], 'validation')
    for key, expected in {'origin': 'imported', 'catalog_id': 'yolox-nano-coco-onnx/v1',
                          'catalog_revision': 'yolox-nano-coco-onnx/2026-10-17',
                          'artifact_sha256': 'c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d'}.items():
        _fixed(model[key], expected)
    _integer(model['artifact_bytes'], 3659407, 3659407)
    _sha(model['manifest_sha256'])
    dataset = _object(provenance['dataset'], DATASET_KEYS)
    _check(dataset['dataset_version_id'] == evaluation['dataset_version_id'])
    _sha(dataset['content_sha256'])
    if dataset['source_selection_sha256'] is not None:
        _sha(dataset['source_selection_sha256'])
    _fixed(dataset['purpose'], 'evaluation')
    _fixed(dataset['split'], 'val')
    _integer(dataset['images'], MAX_IMAGES, 1)
    _integer(dataset['boxes'], MAX_BOXES)
    _fixed(provenance['categories'], [
        {'class_id': i, 'name': name, 'source_category_id': COCO_CATEGORY_IDS[i],
         'evaluator_internal_category_id': i + 1} for i, name in enumerate(CLASS_NAMES)])
    # Category identifiers are integers, including within otherwise fixed objects.
    for row in provenance['categories']:
        for key in ('class_id', 'source_category_id', 'evaluator_internal_category_id'):
            _integer(row[key], 100)
    _fixed(provenance['protocol'], protocol())
    for key, source in (('protocol_sha256', 'protocol'), ('category_mapping_sha256', 'categories')):
        _check(_sha(provenance[key]) == digest(provenance[source]))
    result = _object(report['exported_result'], RESULT_KEYS)
    metrics = _object(result['metrics'], ('map_50', 'map_50_95', 'ar_100', 'status', 'protocol',
                                         'evaluator', 'parameters', 'counts', 'per_class'))
    _object(metrics['counts'], ('images', 'images_with_ground_truth', 'images_without_ground_truth',
                                'images_with_predictions', 'ground_truth_boxes', 'predictions',
                                'predictions_above_max_detections', 'categories', 'categories_with_ground_truth'))
    for row in _rows(metrics['per_class']):
        _object(row, ('category_id', 'name', 'ground_truth_boxes', 'predictions', 'status',
                      'ap_50', 'ap_50_95', 'ar_100'))
    operating = _object(result['operating_point'], ('protocol', 'iou_threshold', 'score_threshold',
                                                   'score_operator', 'totals', 'per_class'))
    _object(operating['totals'], ('precision', 'recall', 'tp', 'fp', 'fn'))
    for row in _rows(operating['per_class']):
        _object(row, ('category_id', 'name', 'precision', 'recall', 'tp', 'fp', 'fn'))
    _metrics(result, dataset)
    _environment(result, provenance)
    _fixed(result['scope'], {'images': dataset['images'], 'split': 'val', 'official_benchmark': False,
                             'external_audit': False, 'production_allowed': False, 'device_certified': False})
    _integer(result['scope']['images'], MAX_IMAGES, 1)
    _fixed(result['warnings'], WARNINGS)
    _fixed(report['verification'], VERIFICATION)
    _fixed(report['limitations'], LIMITATIONS)
    for key in ('source_result_sha256', 'source_binding_sha256', 'source_prediction_sha256'):
        _sha(report[key])  # References only: corresponding bytes are deliberately absent.
    _check(_sha(report['exported_result_sha256']) == digest(result))
    _check(_sha(report['report_sha256']) == digest({k: v for k, v in report.items() if k != 'report_sha256'}))
    _check(len(encode(report)) <= MAX_REPORT_BYTES)
    return report


def validate_report(value):
    """Check schema/semantics/digests, never trusted origin or source evidence."""
    try:
        return _validate(value)
    except (ValueError, TypeError, KeyError, OverflowError, RecursionError, AttributeError):
        raise ValueError('Invalid independent evaluation report') from None


def _unique(pairs):
    result = {}
    for key, value in pairs:
        _check(key not in result)
        result[key] = value
    return result


def _number(token, convert):
    _check(len(token) <= MAX_NUMBER_TOKEN)
    value = convert(token)
    _check(math.isfinite(value))
    return value


def parse_report(raw):
    """Bound before parsing; reject duplicate fields and noncanonical bytes."""
    try:
        _check(type(raw) is bytes and 0 < len(raw) <= MAX_REPORT_BYTES)
        text = raw.decode('utf-8', errors='strict')
        depth = 0
        quoted = escaped = False
        token_length = 0
        for char in text:
            if quoted:
                token_length += 1
                _check(token_length <= MAX_STRING * 6 + 2)
                if escaped:
                    escaped = False
                elif char == '\\':
                    escaped = True
                elif char == '"':
                    quoted = False
            elif char == '"':
                quoted, token_length = True, 0
            elif char in '[{':
                depth += 1
                _check(depth <= MAX_DEPTH)
            elif char in ']}':
                depth -= 1
                _check(depth >= 0)
        _check(depth == 0 and not quoted)
        value = json.loads(text, object_pairs_hook=_unique,
                           parse_int=lambda token: _number(token, int),
                           parse_float=lambda token: _number(token, float),
                           parse_constant=lambda token: _check(False))
        _bounds(value)
        _check(encode(value) == raw)
        return validate_report(value)
    except (ValueError, TypeError, OverflowError, RecursionError, UnicodeError):
        raise ValueError('Invalid independent evaluation report') from None
