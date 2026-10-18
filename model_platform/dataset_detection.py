"""Source-reviewed YOLOX recipe bound to one tenant-owned immutable dataset.

Importing this contract does not import optional numerical libraries. Uploaded
YAML is inert dataset metadata only; training settings never come from it.
"""
import json
import re
from . import detection
from .common import Problem, digest, encode, number, object_fields
from .detection_geometry import letterbox_transform

RECIPE = 'yolox-nano-dataset/v1'
CONTRACT = 'rgb-dataset-detection/v1'
FORMAT = 'yolox-nano-dataset-onnx/v1'
MAX_CLASSES = 20
MAX_TRAIN = 128
MAX_EVALUATION = 64
MAX_BOXES = 100
MAX_TRAINING_PIXELS = 20_000_000
FIXED_CONFIG = {'batch_size': 4, 'input_size': 160, 'from_scratch': True,
                'optimizer': 'Adam', 'checkpoint_selection': 'final-epoch'}
DEFAULT_CONFIG = {'epochs': 2, 'learning_rate': 0.001, 'seed': 20261015}
CONFIG_SCHEMA = {'type': 'object', 'additionalProperties': False,
                 'required': ['epochs', 'learning_rate', 'seed'], 'properties': {
                     'epochs': {'type': 'integer', 'minimum': 1, 'maximum': 20},
                     'learning_rate': {'type': 'number', 'minimum': 0.0001, 'maximum': 0.01},
                     'seed': {'type': 'integer', 'minimum': 0, 'maximum': 2**32-1}}}
DATASET_SCHEMA = {'type': 'object', 'additionalProperties': False,
                  'required': ['dataset_version_id'], 'properties': {
                      'dataset_version_id': {'type': 'string', 'pattern': '^dsv_[0-9a-f]{32}$'}}}


def request_body(version_id):
    return {'recipe': RECIPE, 'dataset': {'dataset_version_id': version_id}, 'config': dict(DEFAULT_CONFIG)}


def validate_request(data, config):
    object_fields(data, ('dataset_version_id',))
    if not isinstance(data['dataset_version_id'], str) or not re.fullmatch(r'dsv_[0-9a-f]{32}', data['dataset_version_id']):
        raise Problem(400, 'invalid_input', 'Expected a published dataset version ID')
    object_fields(config, ('epochs', 'learning_rate', 'seed'))
    if type(config['epochs']) is not int or not 1 <= config['epochs'] <= 20:
        raise Problem(400, 'invalid_input', 'epochs must be integer 1..20')
    number(config['learning_rate'], .0001, .01)
    if type(config['seed']) is not int or not 0 <= config['seed'] <= 2**32-1:
        raise Problem(400, 'invalid_input', 'seed must be integer 0..4294967295')


def labels(raw, width, height):
    """Convert the already validated YOLO rows, clamping only import epsilon."""
    result = []
    for line in raw.decode('utf-8').splitlines():
        if not line.strip():
            continue
        c, x, y, w, h = line.split()
        x, y, w, h = map(float, (x, y, w, h))
        box = [max(0., (x-w/2)*width), max(0., (y-h/2)*height),
               min(float(width), (x+w/2)*width), min(float(height), (y+h/2)*height)]
        result.append({'class_id': int(c), 'xyxy': box})
    return result


def bind_version(datasets, tenant, version_id):
    version = datasets.get_version(tenant, version_id)  # Ownership precedes paths.
    manifest = version['manifest']
    if version.get('purpose', manifest.get('purpose', 'training')) != 'training':
        raise Problem(400, 'dataset_training_purpose', 'Evaluation-purpose datasets cannot be used for training')
    classes = [{'id': row['id'], 'name': row['name']} for row in version['classes']]
    counts = {split: version['splits'][split]['images'] for split in ('train', 'val', 'test')}
    if not 1 <= len(classes) <= MAX_CLASSES or not 1 <= counts['train'] <= MAX_TRAIN or not 1 <= counts['val'] <= MAX_EVALUATION or counts['test'] > MAX_EVALUATION:
        raise Problem(400, 'dataset_training_budget', 'Training requires 1..20 classes, 1..128 train, 1..64 val and 0..64 test images')
    samples = manifest['samples']
    if (sum(row['width']*row['height'] for row in samples) > MAX_TRAINING_PIXELS
            or any(row['boxes'] > MAX_BOXES for row in samples)):
        raise Problem(400, 'dataset_training_budget', 'Training permits 20 million total pixels and at most 100 boxes per image')
    try:
        for row in samples:
            letterbox_transform(row['width'], row['height'], target_width=160, target_height=160)
    except ValueError:
        raise Problem(400, 'dataset_training_budget', 'Image aspect ratio is outside the fixed resize contract') from None
    root = datasets.root / 'versions' / version_id
    present = set()
    for row in samples:
        boxes = labels((root/row['label']).read_bytes(), row['width'], row['height'])
        keys = [(box['class_id'], *box['xyxy']) for box in boxes]
        if len(set(keys)) != len(keys) or any(b['xyxy'][0] >= b['xyxy'][2] or b['xyxy'][1] >= b['xyxy'][3] for b in boxes):
            raise Problem(400, 'dataset_training_annotations', 'Training requires unique, positive-area annotations per image')
        if row['split'] == 'train':
            present.update(box['class_id'] for box in boxes)
    if present != set(range(len(classes))):
        raise Problem(400, 'dataset_train_classes', 'Every declared class must have a training annotation')
    # Hashes include the ordered sample records and label/image hashes; no raw data
    # or caller-selected paths become run input. Import rejects cross-split pixels.
    split_hashes = {s: digest([row for row in samples if row['split'] == s]) for s in counts}
    evaluation_split = 'test' if counts['test'] else 'val'
    return {'dataset_version_id': version_id, 'source': version['name'], 'name': version['name'], 'version': version_id,
            'sha256': version['content_sha256'], 'classes': classes,
            'class_names': [row['name'] for row in classes], 'train_count': counts['train'],
            'validation_count': counts['val'], 'test_count': counts['test'],
            'evaluation_count': counts[evaluation_split], 'evaluation_split': evaluation_split,
            'train_sha256': split_hashes['train'], 'validation_sha256': split_hashes['val'],
            'test_sha256': split_hashes['test'] if counts['test'] else None,
            'evaluation_sha256': split_hashes[evaluation_split],
            'negative_policy': version['negative_policy']}


def metadata(class_names, binding):
    props = detection.metadata()
    props.update(recipe=RECIPE, contract=CONTRACT, format=FORMAT,
                 class_names=encode(class_names).decode(), class_ids=encode(list(range(len(class_names)))).decode(),
                 dataset_version_id=binding['dataset_version_id'], dataset_sha256=binding['sha256'],
                 train_sha256=binding['train_sha256'], evaluation_sha256=binding['evaluation_sha256'],
                 evaluation_split=binding['evaluation_split'])
    return props


def _metadata_identity(props):
    try:
        names = json.loads(props['class_names'])
        lineage = {key: props[key] for key in ('dataset_version_id', 'dataset_sha256', 'train_sha256', 'evaluation_sha256', 'evaluation_split')}
        if (not re.fullmatch(r'dsv_[0-9a-f]{32}', lineage['dataset_version_id'])
                or any(not re.fullmatch(r'[0-9a-f]{64}', lineage[k]) for k in ('dataset_sha256', 'train_sha256', 'evaluation_sha256'))
                or lineage['evaluation_split'] not in ('val', 'test')):
            raise ValueError('Invalid dataset model lineage')
        binding = dict(lineage, sha256=lineage['dataset_sha256'])
        if (not isinstance(names, list) or not 1 <= len(names) <= MAX_CLASSES
                or any(not isinstance(n, str) or not n.strip() or len(n)>80 or any(ord(c)<32 or ord(c)==127 for c in n) for n in names)
                or len(set(names)) != len(names) or props != metadata(names, binding)):
            raise ValueError('Invalid dataset model metadata')
    except (KeyError, TypeError, json.JSONDecodeError):
        raise ValueError('Invalid dataset model metadata') from None
    return {'recipe': RECIPE, 'contract': CONTRACT, 'format': FORMAT, 'class_names': names, 'dataset_lineage': lineage}


def load_model(path, expected_hash):
    return detection._load_model(path, expected_hash, _metadata_identity)


def probe_output(predictions):
    detection.probe_output(predictions, num_classes=MAX_CLASSES)


def execute(*args):
    from .algorithm import TrainingFailure
    try:
        from .dataset_training import train_export
        return train_export(*args)
    except ImportError:
        raise TrainingFailure('optional_dependencies_missing') from None


def compatibility(evaluation):
    """All upstream filtering and official metric implementation are versioned."""
    return {'dataset': evaluation['evaluation_sha256'], 'split': evaluation['split'],
            'classes': evaluation['category_mapping'],
            'protocol': {k: evaluation[k] for k in ('contract', 'protocol', 'evaluator', 'parameters',
                        'score_threshold', 'nms_iou_threshold', 'max_detections', 'preprocessing')}}
