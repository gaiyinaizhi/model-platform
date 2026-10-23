"""Portable stdlib-only trained independent evaluation report schema.

Imports only the existing portable report primitives, never ML, runtime or DB.
Source result/binding/prediction hashes remain references, not verified evidence.
"""
from . import evaluation_report_schema as v

SCHEMA_VERSION = 'dataset-trained-independent-evaluation-report/v1'
PROTOCOL = 'dataset-trained-serving-evaluation/v1'
MODEL_KEYS = ('model_version_id', 'origin', 'recipe', 'run_id', 'attempt_id',
              'evaluation_id', 'artifact_sha256', 'artifact_bytes', 'manifest_sha256')
TRAINING_KEYS = ('dataset_version_id', 'content_sha256', 'train_sha256', 'validation_sha256',
                 'test_sha256', 'evaluation_sha256', 'evaluation_split')
PROVENANCE_KEYS = v.PROVENANCE_KEYS + ('training_dataset',)
RESULT_KEYS = ('protocol',) + v.RESULT_KEYS
LIMITATIONS = {'production_allowed': False, 'official_coco_benchmark': False,
               'device_certified': False, 'cross_dataset_overlap': 'not_audited'}
WARNINGS = [
    'Independent means a separate evaluation task; cross-dataset image overlap and leakage are not audited.',
    'Serving-filtered detections use score>=0.01 and class-aware NMS0.65; not training-evaluation score0.001.',
    'Categories without ground truth are excluded from AP/AR macro averages; no-GT metrics are null.',
    'Dataset provenance and label completeness are not externally audited; not an official benchmark or production quality approval.',
    'Logical budgets and process separation are not production multi-tenant isolation.',
    'Uses unchanged server inference with declared OpenCV caller resize; browser JPEG decoding or resizing may differ.',
]


def protocol():
    return {'id': PROTOCOL, 'inference_protocol': 'yolox-nano-dataset-onnx/v1',
            'max_input_side': 384, 'input_size': 160, 'score_threshold': .01, 'score_operator': '>=',
            'nms_iou_threshold': .65, 'nms_class_agnostic': False, 'max_detections': 100,
            'resize': 'opencv-inter-linear-floor-max384-no-upscale/v1',
            'metric_protocol': 'coco-style-bbox-subset/v1', 'matching_protocol': 'class-aware-greedy-iou50/v1'}


def _name(value):
    v._check(type(value) is str and 1 <= len(value) <= 80 and value.strip()
             and not any(ord(c) < 32 or ord(c) == 127 for c in value))


def _metrics(result, provenance):
    dataset = provenance['dataset']; count = len(provenance['categories'])
    metrics = v._object(result['metrics'], ('protocol', 'status', 'map_50', 'map_50_95', 'ar_100',
                                          'evaluator', 'parameters', 'counts', 'per_class'))
    v._fixed(metrics['protocol'], 'coco-style-bbox-subset/v1')
    evaluator = v._object(metrics['evaluator'], ('implementation', 'version'))
    v._fixed(evaluator['implementation'], 'pycocotools.cocoeval.COCOeval')
    v._version(evaluator['version']); v._parameters(metrics['parameters'])
    counts = v._object(metrics['counts'], ('images', 'images_with_ground_truth', 'images_without_ground_truth',
                      'images_with_predictions', 'ground_truth_boxes', 'predictions',
                      'predictions_above_max_detections', 'categories', 'categories_with_ground_truth'))
    for key, maximum in (('images', 64), ('images_with_ground_truth', 64), ('images_without_ground_truth', 64),
                         ('images_with_predictions', 64), ('ground_truth_boxes', 10000), ('predictions', 6400),
                         ('predictions_above_max_detections', 0), ('categories', count), ('categories_with_ground_truth', count)):
        v._integer(counts[key], maximum)
    v._check(counts['images'] == dataset['images'] and counts['ground_truth_boxes'] == dataset['boxes']
             and counts['categories'] == count
             and counts['images_with_ground_truth'] + counts['images_without_ground_truth'] == counts['images']
             and counts['images_with_ground_truth'] <= counts['ground_truth_boxes']
             and counts['images_with_predictions'] <= counts['images']
             and counts['images_with_predictions'] <= counts['predictions']
             and counts['predictions'] <= counts['images_with_predictions'] * 100
             and (not counts['ground_truth_boxes'] or counts['images_with_ground_truth'] > 0))
    operating = v._object(result['operating_point'], ('protocol', 'iou_threshold', 'score_threshold', 'score_operator', 'totals', 'per_class'))
    for key, value in {'protocol': 'class-aware-greedy-iou50/v1', 'iou_threshold': .5,
                       'score_threshold': .01, 'score_operator': '>='}.items():
        v._fixed(operating[key], value)
    rows, op_rows = metrics['per_class'], operating['per_class']
    v._check(type(rows) is list and type(op_rows) is list and len(rows) == len(op_rows) == count)
    fields = (('map_50', 'ap_50'), ('map_50_95', 'ap_50_95'), ('ar_100', 'ar_100'))
    for index, (row, op) in enumerate(zip(rows, op_rows)):
        v._object(row, ('category_id', 'name', 'ground_truth_boxes', 'predictions', 'status', 'ap_50', 'ap_50_95', 'ar_100'))
        v._object(op, ('category_id', 'name', 'tp', 'fp', 'fn', 'precision', 'recall'))
        for item in (row, op):
            v._integer(item['category_id'], count - 1)
            v._fixed(item['category_id'], index); v._fixed(item['name'], provenance['categories'][index]['name'])
        gt = v._integer(row['ground_truth_boxes'], 10000); predictions = v._integer(row['predictions'], 6400)
        v._fixed(row['status'], 'evaluated' if gt else 'no_ground_truth')
        for _, field in fields:
            value = v._rate(row[field])
            v._check((value is not None and (predictions > 0 or value == 0)) if gt else value is None)
        tp, fp, fn = v._ratios(op)
        v._check(tp + fn == gt and tp + fp == predictions)
    present = [row for row in rows if row['ground_truth_boxes']]
    v._check(sum(r['ground_truth_boxes'] for r in rows) == counts['ground_truth_boxes']
             and sum(r['predictions'] for r in rows) == counts['predictions']
             and len(present) == counts['categories_with_ground_truth']
             and metrics['status'] == ('evaluated' if present else 'no_ground_truth'))
    for field, class_field in fields:
        value = v._rate(metrics[field])
        v._check(value is not None if present else value is None)
        if present:
            v._check(abs(value - sum(r[class_field] for r in present) / len(present)) <= 1e-10)
    totals = v._object(operating['totals'], ('tp', 'fp', 'fn', 'precision', 'recall'))
    v._ratios(totals)
    v._check(all(sum(r[k] for r in op_rows) == totals[k] for k in ('tp', 'fp', 'fn')))


def validate_report(value):
    """Called by the bounded common dispatcher; strict shapes before semantics."""
    v._bounds(value)
    report = v._object(value, v.TOP_KEYS)
    v._fixed(report['schema_version'], SCHEMA_VERSION)
    v._fixed(report['canonicalization'], v.CANONICALIZATION)
    evaluation = v._object(report['evaluation'], v.EVALUATION_KEYS)
    for key, prefix in (('id', 'mev'), ('model_version_id', 'model'), ('dataset_version_id', 'dsv')):
        v._identity(evaluation[key], prefix)
    v._fixed(evaluation['status'], 'succeeded'); v._fixed(evaluation['split'], 'val')
    v._fixed(evaluation['protocol'], PROTOCOL)
    times = [evaluation[k] for k in ('created_at', 'started_at', 'finished_at')]
    v._check(all(type(t) in (int, float) and v.math.isfinite(t) and 0 <= t <= 1e12 for t in times))
    v._check(times[0] <= times[1] <= times[2])
    p = v._object(report['provenance'], PROVENANCE_KEYS)
    model = v._object(p['model'], MODEL_KEYS)
    v._fixed(model['origin'], 'trained'); v._fixed(model['recipe'], 'yolox-nano-dataset/v1')
    for key, prefix in (('model_version_id', 'model'), ('run_id', 'run'), ('attempt_id', 'attempt'), ('evaluation_id', 'eval')):
        v._identity(model[key], prefix)
    v._check(model['model_version_id'] == evaluation['model_version_id'])
    v._sha(model['artifact_sha256']); v._sha(model['manifest_sha256'])
    v._integer(model['artifact_bytes'], 8 * 1024 * 1024, 1)
    training = v._object(p['training_dataset'], TRAINING_KEYS)
    v._identity(training['dataset_version_id'], 'dsv')
    for key in ('content_sha256', 'train_sha256', 'validation_sha256', 'evaluation_sha256'): v._sha(training[key])
    if training['test_sha256'] is None:
        v._check(training['evaluation_split'] == 'val' and training['evaluation_sha256'] == training['validation_sha256'])
    else:
        v._sha(training['test_sha256'])
        v._check(training['evaluation_split'] == 'test' and training['evaluation_sha256'] == training['test_sha256'])
    dataset = v._object(p['dataset'], v.DATASET_KEYS)
    v._check(dataset['dataset_version_id'] == evaluation['dataset_version_id']
             and dataset['dataset_version_id'] != training['dataset_version_id'])
    v._sha(dataset['content_sha256']); v._fixed(dataset['purpose'], 'evaluation')
    v._fixed(dataset['split'], 'val'); v._fixed(dataset['source_selection_sha256'], None)
    v._integer(dataset['images'], 64, 1); v._integer(dataset['boxes'], 10000)
    categories = p['categories']
    v._check(type(categories) is list and 1 <= len(categories) <= 20)
    names = []
    for index, category in enumerate(categories):
        v._object(category, ('class_id', 'name', 'evaluator_internal_category_id'))
        v._integer(category['class_id'], 19); v._integer(category['evaluator_internal_category_id'], 20, 1)
        v._fixed(category['class_id'], index); v._fixed(category['evaluator_internal_category_id'], index + 1)
        _name(category['name']); names.append(category['name'])
    v._check(len(set(names)) == len(names))
    v._fixed(p['protocol'], protocol())
    for key, source in (('protocol_sha256', 'protocol'), ('category_mapping_sha256', 'categories')):
        v._check(v._sha(p[key]) == v.digest(p[source]))
    result = v._object(report['exported_result'], RESULT_KEYS)
    v._fixed(result['protocol'], PROTOCOL)
    _metrics(result, p); v._environment(result, p)
    v._fixed(result['scope'], {'images': dataset['images'], 'split': 'val', 'official_benchmark': False,
                             'external_audit': False, 'production_allowed': False, 'device_certified': False})
    v._integer(result['scope']['images'], 64, 1)
    v._fixed(result['warnings'], WARNINGS)
    v._fixed(report['verification'], v.VERIFICATION); v._fixed(report['limitations'], LIMITATIONS)
    for key in ('source_result_sha256', 'source_binding_sha256', 'source_prediction_sha256'): v._sha(report[key])
    v._check(v._sha(report['exported_result_sha256']) == v.digest(result))
    v._check(v._sha(report['report_sha256']) == v.digest({k: x for k, x in report.items() if k != 'report_sha256'}))
    v._check(len(v.encode(report)) <= v.MAX_REPORT_BYTES)
    return report


def project(detail):
    """Whitelist projection after the caller verifies one captured source detail."""
    binding, result = detail['binding'], detail['result']
    dataset = binding['dataset']
    # Refuse unrecognized provenance rather than silently discard source semantics.
    v._fixed(dataset['source'], None)
    p = {key: binding[key] for key in PROVENANCE_KEYS if key != 'dataset'}
    p['dataset'] = {key: dataset[key] for key in v.DATASET_KEYS if key != 'source_selection_sha256'}
    p['dataset']['source_selection_sha256'] = None
    exported = {key: result[key] for key in RESULT_KEYS}
    report = {'schema_version': SCHEMA_VERSION, 'canonicalization': v.CANONICALIZATION,
              'evaluation': {key: detail[key] for key in v.EVALUATION_KEYS},
              'provenance': p, 'exported_result': exported,
              'source_result_sha256': detail['result_sha256'],
              'source_binding_sha256': result['digests']['binding_sha256'],
              'source_prediction_sha256': result['digests']['prediction_sha256'],
              'exported_result_sha256': v.digest(exported),
              'verification': v.VERIFICATION, 'limitations': LIMITATIONS}
    report['report_sha256'] = v.digest(report)
    validate_report(report)
    return v.encode(report)
