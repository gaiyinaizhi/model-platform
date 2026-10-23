"""Independent serving-point measurements of registered dataset-trained models.

No training, publication, deployment or original training-file reads occur here.
The existing independent-evaluation supervisor owns execution and publication.
"""
import hashlib
import io
import json
import signal

from .common import Problem, digest
from . import dataset_detection as recipe, detection
from .detection_geometry import letterbox_transform

PROTOCOL = 'dataset-trained-serving-evaluation/v1'
MODEL_KEYS = ('model_version_id', 'origin', 'recipe', 'run_id', 'attempt_id',
              'evaluation_id', 'artifact_sha256', 'artifact_bytes', 'manifest_sha256')
TRAINING_KEYS = ('dataset_version_id', 'content_sha256', 'train_sha256', 'validation_sha256',
                 'test_sha256', 'evaluation_sha256', 'evaluation_split')


def protocol():
    return {'id': PROTOCOL, 'inference_protocol': recipe.FORMAT,
            'max_input_side': 384, 'input_size': 160, 'score_threshold': .01, 'score_operator': '>=',
            'nms_iou_threshold': .65, 'nms_class_agnostic': False, 'max_detections': 100,
            'resize': 'opencv-inter-linear-floor-max384-no-upscale/v1',
            'metric_protocol': 'coco-style-bbox-subset/v1', 'matching_protocol': 'class-aware-greedy-iou50/v1'}


def validate_binding(binding):
    """Pure validation also used by history reads without reopening dataset files."""
    from . import training_report_schema as schema
    schema.obj(binding, ('model', 'training_dataset', 'dataset', 'categories',
                         'category_mapping_sha256', 'protocol', 'protocol_sha256', 'submission_environment'))
    model = schema.obj(binding['model'], MODEL_KEYS)
    schema.exact(model['origin'], 'trained'); schema.exact(model['recipe'], recipe.RECIPE)
    for field, prefix in (('model_version_id', 'model'), ('run_id', 'run'),
                          ('attempt_id', 'attempt'), ('evaluation_id', 'eval')):
        schema.ident(model[field], prefix)
    schema.sha(model['artifact_sha256']); schema.sha(model['manifest_sha256'])
    schema.integer(model['artifact_bytes'], 1, detection.MAX_ARTIFACT)
    training = schema.obj(binding['training_dataset'], TRAINING_KEYS)
    schema.ident(training['dataset_version_id'], 'dsv')
    for key in ('content_sha256', 'train_sha256', 'validation_sha256', 'evaluation_sha256'):
        schema.sha(training[key])
    if training['test_sha256'] is None:
        schema.check(training['evaluation_split'] == 'val'
                     and training['evaluation_sha256'] == training['validation_sha256'])
    else:
        schema.sha(training['test_sha256'])
        schema.check(training['evaluation_split'] == 'test'
                     and training['evaluation_sha256'] == training['test_sha256'])
    dataset = schema.obj(binding['dataset'], ('dataset_version_id', 'content_sha256', 'purpose', 'split',
                                                'images', 'boxes', 'samples', 'source'))
    schema.ident(dataset['dataset_version_id'], 'dsv'); schema.sha(dataset['content_sha256'])
    schema.check(dataset['dataset_version_id'] != training['dataset_version_id']
                 and dataset['purpose'] == 'evaluation' and dataset['split'] == 'val'
                 and dataset['source'] is None)
    schema.integer(dataset['images'], 1, 64); schema.integer(dataset['boxes'], 0, 10000)
    from .dataset_validation import safe_path, MAX_PIXELS
    samples = dataset['samples']
    schema.check(type(samples) is list and len(samples) == dataset['images'])
    seen_images, seen_labels = set(), set()
    for sample in samples:
        schema.obj(sample, ('image', 'label', 'width', 'height', 'image_sha256', 'label_sha256', 'pixel_sha256'))
        for key, prefix, seen in (('image', 'images/val/', seen_images), ('label', 'labels/val/', seen_labels)):
            name = sample[key]
            schema.check(type(name) is str and safe_path(name) and name.startswith(prefix)
                         and not name.endswith('/') and name not in seen)
            seen.add(name)
        schema.integer(sample['width'], 1, MAX_PIXELS); schema.integer(sample['height'], 1, MAX_PIXELS)
        schema.check(sample['width'] * sample['height'] <= MAX_PIXELS)
        for key in ('image_sha256', 'label_sha256', 'pixel_sha256'): schema.sha(sample[key])
    categories = binding['categories']
    schema.check(type(categories) is list and 1 <= len(categories) <= recipe.MAX_CLASSES)
    names = []
    for index, category in enumerate(categories):
        schema.obj(category, ('class_id', 'name', 'evaluator_internal_category_id'))
        schema.exact(category['class_id'], index)
        schema.exact(category['evaluator_internal_category_id'], index + 1)
        schema.name(category['name']); names.append(category['name'])
    schema.check(len(set(names)) == len(names))
    schema.exact(binding['protocol'], protocol())
    schema.check(binding['category_mapping_sha256'] == digest(categories)
                 and binding['protocol_sha256'] == digest(protocol()))


def binding_for(evaluations, tenant, body, submitted_environment=None):
    from .datasets import DatasetStore
    from .training_reports import snapshot_report, _json
    from .model_evaluations import _read_file, _env, MAX_IMAGES, MAX_BOXES
    # Resolve both owned resources before support/evidence checks. Capture the
    # registered model and all of its original training associations together.
    with evaluations.runtime.connection() as db:
        db.execute('BEGIN')
        row = evaluations.store._owned(db, 'models', tenant, body['model_version_id'])
        evaluations.store._owned(db, 'dataset_versions', tenant, body['dataset_version_id'])
        try:
            manifest = _json(row['manifest_json'])
            if row['origin'] != 'trained' or manifest.get('recipe') != recipe.RECIPE:
                raise Problem(409, 'evaluation_model_unsupported', 'This protocol requires a registered dataset-trained model')
            registered = snapshot_report(evaluations.store, db, tenant, row)
        except Problem as exc:
            if exc.code == 'evaluation_model_unsupported':
                raise
            raise Problem(409, 'evaluation_integrity', 'Registered training provenance is invalid') from None
        except (ValueError, KeyError, TypeError, AttributeError):
            raise Problem(409, 'evaluation_integrity', 'Registered model identity is invalid') from None
    try:
        raw = _read_file(evaluations.runtime.home / 'artifacts' / (manifest['attempt_id'] + '.json'), detection.MAX_ARTIFACT)
        if len(raw) != manifest['artifact_bytes'] or hashlib.sha256(raw).hexdigest() != manifest['artifact_sha256']:
            raise ValueError('artifact')
    except (OSError, ValueError, KeyError, TypeError):
        raise Problem(409, 'evaluation_integrity', 'Registered artifact bytes or identity changed') from None
    dataset = DatasetStore(evaluations.runtime, read_only=True).get_version(tenant, body['dataset_version_id'])
    dm = dataset['manifest']; samples = dm['samples']; names = manifest['class_names']
    original = registered['provenance']['dataset']
    if (dataset['id'] == original['dataset_version_id'] or dm.get('purpose', 'training') != 'evaluation'
            or dm.get('source') is not None or not 1 <= len(names) <= recipe.MAX_CLASSES
            or [{'id': c['id'], 'name': c['name']} for c in dm['classes']] != [{'id': i, 'name': name} for i, name in enumerate(names)]
            or not 1 <= len(samples) <= MAX_IMAGES or any(s['split'] != 'val' for s in samples)
            or not 0 <= dm['counts']['boxes'] <= MAX_BOXES
            or any(not s.get('pixel_sha256') for s in samples)):
        raise Problem(409, 'evaluation_dataset_incompatible', 'Evaluation requires a distinct val-only evaluation dataset, 1..64 images, at most10000 boxes and exact ordered1..20 classes')
    # Validate the actual caller-resized aspect ratio, not the larger original.
    try:
        for sample in samples:
            width, height = sample['width'], sample['height']
            if max(width, height) > 384:
                ratio = 384 / max(width, height)
                width, height = max(1, int(width * ratio)), max(1, int(height * ratio))
            letterbox_transform(width, height, target_width=160, target_height=160)
    except ValueError:
        raise Problem(409, 'evaluation_dataset_incompatible', 'Image aspect ratio is outside the fixed serving resize contract') from None
    categories = [{'class_id': i, 'name': name, 'evaluator_internal_category_id': i + 1}
                  for i, name in enumerate(names)]
    binding = {'model': {**{key: manifest[key] for key in MODEL_KEYS
                            if key not in ('origin', 'manifest_sha256')},
                         'origin': 'trained', 'manifest_sha256': digest(manifest)},
               'training_dataset': {key: original[key] for key in TRAINING_KEYS},
               'dataset': {'dataset_version_id': dataset['id'], 'content_sha256': dataset['content_sha256'],
                           'purpose': 'evaluation', 'split': 'val', 'images': len(samples), 'boxes': dm['counts']['boxes'],
                           'samples': [{key: s[key] for key in ('image', 'label', 'width', 'height', 'image_sha256',
                                                               'label_sha256', 'pixel_sha256')} for s in samples],
                           'source': None},
               'categories': categories, 'category_mapping_sha256': digest(categories),
               'protocol': protocol(), 'protocol_sha256': digest(protocol()),
               'submission_environment': _env(submitted_environment)}
    try:
        validate_binding(binding)
    except (ValueError, KeyError, TypeError, AttributeError):
        raise Problem(409, 'evaluation_integrity', 'Registered evaluation binding is invalid') from None
    return binding


def validate_result(binding, result):
    """Validate new result semantics without loosening the imported validator."""
    from . import evaluation_comparison_validation as v
    validate_binding(binding)
    v._object(result, ('protocol', 'metrics', 'operating_point', 'wrong_examples', 'digests', 'environment', 'scope', 'warnings'))
    v._fixed(result['protocol'], PROTOCOL)
    metrics = v._object(result['metrics'], ('protocol', 'status', 'map_50', 'map_50_95', 'ar_100',
                                          'evaluator', 'parameters', 'counts', 'per_class'))
    v._fixed(metrics['protocol'], 'coco-style-bbox-subset/v1')
    v._fixed(metrics['evaluator']['implementation'], 'pycocotools.cocoeval.COCOeval')
    v._parameters(metrics['parameters']); v._environment(result, binding)
    dataset = binding['dataset']; count = len(binding['categories']); counts = metrics['counts']
    v._object(counts, ('images', 'images_with_ground_truth', 'images_without_ground_truth',
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
            v._fixed(item['category_id'], index); v._fixed(item['name'], binding['categories'][index]['name'])
        gt = v._integer(row['ground_truth_boxes'], 10000); predictions = v._integer(row['predictions'], 6400)
        v._check(row['status'] == ('evaluated' if gt else 'no_ground_truth'))
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
    v._fixed(result['scope'], {'images': dataset['images'], 'split': 'val', 'official_benchmark': False,
                              'external_audit': False, 'production_allowed': False, 'device_certified': False})
    v._check(type(result['warnings']) is list and 1 <= len(result['warnings']) <= 10
             and all(type(w) is str and 1 <= len(w) <= 500 for w in result['warnings']))
    wrong = v._object(result['wrong_examples'], ('total', 'limit', 'truncated', 'items'))
    total = v._integer(wrong['total'], dataset['images'])
    v._fixed(wrong['limit'], 20); v._fixed(wrong['truncated'], total > 20)
    v._check(type(wrong['items']) is list and len(wrong['items']) == min(total, 20))
    samples = {sample['image']: sample for sample in dataset['samples']}
    positions = {name: i for i, name in enumerate(samples)}
    last = -1
    global_indices = {'false_positives': set(), 'false_negatives': set()}
    for item in wrong['items']:
        v._object(item, ('image_id', 'width', 'height', 'tp', 'fp', 'fn', 'false_positives',
                         'false_negatives', 'false_positives_truncated', 'false_negatives_truncated'))
        sample = samples[item['image_id']]
        v._check(positions[item['image_id']] > last)
        last = positions[item['image_id']]
        v._fixed(item['width'], sample['width']); v._fixed(item['height'], sample['height'])
        tp, fp, fn = (v._integer(item[key], maximum) for key, maximum in (('tp', 100), ('fp', 100), ('fn', 10000)))
        v._check(fp + fn > 0 and tp + fp <= 100)
        for key, count, index_key, maximum in (('false_positives', fp, 'prediction_index', 6399),
                                               ('false_negatives', fn, 'annotation_index', 9999)):
            entries = item[key]
            v._check(type(entries) is list and len(entries) == min(count, 20))
            v._fixed(item[key + '_truncated'], count > 20)
            seen = set()
            for entry in entries:
                v._object(entry, (index_key, 'category_id', 'bbox', 'score') if key == 'false_positives'
                          else (index_key, 'category_id', 'bbox'))
                index = v._integer(entry[index_key], maximum)
                v._check(index not in seen and index not in global_indices[key]
                         and index < counts['predictions' if key == 'false_positives' else 'ground_truth_boxes'])
                seen.add(index); global_indices[key].add(index)
                category = v._integer(entry['category_id'], len(binding['categories']) - 1)
                v._check(op_rows[category]['fp' if key == 'false_positives' else 'fn'] > 0)
                box = entry['bbox']
                v._check(type(box) is list and len(box) == 4
                         and all(type(n) in (int, float) for n in box)
                         and 0 <= box[0] < box[2] <= sample['width']
                         and 0 <= box[1] < box[3] <= sample['height'])
                if key == 'false_positives':
                    score = v._rate(entry['score']); v._check(score is not None and score >= .01)
    v._check(all(sum(item[key] for item in wrong['items']) <= totals[key] for key in ('fp', 'fn')))
    if not wrong['truncated']:
        v._check(all(sum(item[key] for item in wrong['items']) == totals[key] for key in ('fp', 'fn')))
    from .model_evaluations import ModelEvaluationStore
    expected = ModelEvaluationStore.expected_digests(binding, digest(binding))
    digests = v._object(result['digests'], expected.keys() | {'prediction_sha256'})
    v._check(all(digests[key] == value for key, value in expected.items()))
    v._sha(digests['prediction_sha256'])


def measure(evaluations, row):
    import cv2
    import numpy as np
    from PIL import Image
    from .model_evaluations import (_read_file, _pixel_digest, resize_request, _diagnostics,
                                    _env, installed_versions, MAX_IMAGES)
    from .runtime_dispatch import validate_loaded_identity
    from .detection_evaluation import evaluate_detections, match_detections
    binding = evaluations.verify_binding(row); evaluations.verify_code(binding)
    cv2.setNumThreads(1)
    manifest = evaluations.store.model(row['tenant'], row['model_version_id'])
    signal.alarm(30)
    model = recipe.load_model(evaluations.runtime.home / 'artifacts' / (manifest['attempt_id'] + '.json'),
                              binding['model']['artifact_sha256'])
    validate_loaded_identity(manifest, model)
    signal.alarm(0)
    images, annotations, predictions, evidence = [], [], [], []
    root = evaluations.runtime.home / 'datasets' / 'versions' / row['dataset_version_id']
    for sample in binding['dataset']['samples']:
        raw = _read_file(root / sample['image'], 8 * 1024 * 1024)
        labels = _read_file(root / sample['label'], 1024 * 1024)
        if hashlib.sha256(raw).hexdigest() != sample['image_sha256'] or hashlib.sha256(labels).hexdigest() != sample['label_sha256']:
            raise Problem(409, 'evaluation_integrity', 'Dataset file changed before inference')
        with Image.open(io.BytesIO(raw)) as image:
            image.load(); rgb = np.asarray(image.convert('RGB'))
        height, width = rgb.shape[:2]
        if (width, height) != (sample['width'], sample['height']) or _pixel_digest(rgb) != sample['pixel_sha256']:
            raise Problem(409, 'evaluation_integrity', 'Decoded dataset RGB changed')
        resized = resize_request(rgb); rh, rw = resized.shape[:2]; image_id = sample['image']
        images.append({'id': image_id, 'width': width, 'height': height})
        annotations.extend({'image_id': image_id, 'category_id': box['class_id'], 'bbox': box['xyxy']}
                           for box in recipe.labels(labels, width, height))
        signal.alarm(3)
        returned = detection.predict(model, [{'width': rw, 'height': rh, 'pixels': resized.reshape(-1).tolist()}])[0]
        signal.alarm(0)
        restored = []
        for found in returned['detections']:
            box = found['box']
            restored.append({'category_id': found['class_id'], 'score': found['score'],
                             'bbox': [box[0]*width/rw, box[1]*height/rh, box[2]*width/rw, box[3]*height/rh]})
        predictions.extend(dict(p, image_id=image_id) for p in restored)
        evidence.append({'image_id': image_id, 'width': width, 'height': height,
                         'request_width': rw, 'request_height': rh, 'original_rgb_sha256': sample['pixel_sha256'],
                         'request_rgb_sha256': _pixel_digest(resized), 'detections': restored})
    if len(annotations) != binding['dataset']['boxes'] or len(predictions) > MAX_IMAGES * 100:
        raise Problem(409, 'evaluation_integrity', 'Evaluation counts do not match the bound dataset')
    categories = [{'id': c['class_id'], 'name': c['name']} for c in binding['categories']]
    measured = evaluate_detections(images, annotations, predictions, categories)
    metrics = {key: measured[key] for key in ('protocol', 'status', 'map_50', 'map_50_95', 'ar_100',
                                             'evaluator', 'parameters', 'counts', 'per_class')}
    metrics['parameters'] = dict(metrics['parameters'],
        ar_100_iou_thresholds=list(metrics['parameters']['iou_thresholds']), ar_100_area='all')
    matching = match_detections(images, annotations, predictions, categories, max_examples=20)
    result = {'protocol': PROTOCOL, 'metrics': metrics,
              'operating_point': {'protocol': matching['protocol'], 'iou_threshold': .5,
                                  'score_threshold': .01, 'score_operator': '>=',
                                  'totals': matching['totals'], 'per_class': matching['per_class']},
              'wrong_examples': _diagnostics(matching),
              'digests': dict(evaluations.expected_digests(binding, row['binding_sha256']), prediction_sha256=digest(evidence)),
              'environment': dict(_env(), dependencies={k: v for k, v in installed_versions().items() if k != 'onnx'}),
              'scope': {'images': len(images), 'split': 'val', 'official_benchmark': False, 'external_audit': False,
                        'production_allowed': False, 'device_certified': False},
              'warnings': ['Independent means a separate evaluation task; cross-dataset image overlap and leakage are not audited.',
                           'Serving-filtered detections use score>=0.01 and class-aware NMS0.65; not training-evaluation score0.001.',
                           'Categories without ground truth are excluded from AP/AR macro averages; no-GT metrics are null.',
                           'Dataset provenance and label completeness are not externally audited; not an official benchmark or production quality approval.',
                           'Logical budgets and process separation are not production multi-tenant isolation.',
                           'Uses unchanged server inference with declared OpenCV caller resize; browser JPEG decoding or resizing may differ.']}
    validate_result(binding, result)
    evaluations.verify_binding(row); evaluations.verify_code(binding)
    return {'result': result, 'predictions': evidence}
