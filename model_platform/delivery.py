"""Bounded, authenticated-by-caller delivery of registered ONNX artifacts.

No conversion, checkpoint restoration or target-device approval is implied.
The private snapshot is verified before any HTTP success headers are emitted.
"""
from contextlib import contextmanager
import hashlib
import os
import re
import stat
import tempfile
import threading

from .common import Problem
from . import detection, dataset_detection, fashion
from .recipes import get_recipe, public_catalog

SCHEMA = 'model-delivery/v1'
CHUNK = 64 * 1024
MAX_DOWNLOADS = 2
SUPPORTED = (detection.RECIPE, dataset_detection.RECIPE, fashion.RECIPE)


class Delivery:
    def __init__(self, store):
        self.store = store
        self.slots = threading.BoundedSemaphore(MAX_DOWNLOADS)

    @contextmanager
    def snapshot(self, tenant, model_id):
        # Resolve ownership before capacity/format checks; don't reveal foreign records.
        model = self.store.model(tenant, model_id)
        if model.get('recipe') not in SUPPORTED:
            raise Problem(409, 'delivery_format_unsupported', 'Only registered ONNX deployment artifacts are supported')
        spec = get_recipe(model['recipe'])
        if model.get('format') != spec.format:
            raise Problem(503, 'artifact_integrity_failed', 'Registered artifact identity is inconsistent')
        expected, size = model.get('artifact_sha256'), model.get('artifact_bytes')
        artifact_id = model.get('artifact_id')
        if (not isinstance(expected, str) or not re.fullmatch('[0-9a-f]{64}', expected)
                or type(size) is not int or not 0 < size <= spec.budgets.artifact_bytes
                or not isinstance(artifact_id, str) or not re.fullmatch('attempt_[0-9a-f]{32}', artifact_id)
                or artifact_id != model.get('attempt_id')
                or not re.fullmatch('model_[0-9a-f]{32}', model_id)):
            raise Problem(503, 'artifact_integrity_failed', 'Registered artifact identity is inconsistent')
        if not self.slots.acquire(blocking=False):
            raise Problem(429, 'download_capacity', 'At most two artifact snapshots may be active')
        try:
            with tempfile.TemporaryFile(mode='w+b') as snapshot:
                try:
                    # Refuse a symlinked artifact directory and symlinked leaf. A
                    # same-user administrator can still alter the runtime/database.
                    directory = os.open(self.store.runtime.artifacts, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
                    try:
                        fd = os.open(artifact_id + '.json', os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
                    finally:
                        os.close(directory)
                    with os.fdopen(fd, 'rb') as source:
                        before = os.fstat(source.fileno())
                        if not stat.S_ISREG(before.st_mode) or before.st_size != size:
                            raise ValueError('size')
                        digest, total = hashlib.sha256(), 0
                        while True:
                            chunk = source.read(min(CHUNK, size + 1 - total))
                            if not chunk:
                                break
                            total += len(chunk)
                            if total > size:
                                raise ValueError('growth')
                            digest.update(chunk)
                            snapshot.write(chunk)
                        after = os.fstat(source.fileno())
                        if ((before.st_size, before.st_mtime_ns, before.st_ctime_ns) !=
                                (after.st_size, after.st_mtime_ns, after.st_ctime_ns)
                                or total != size or digest.hexdigest() != expected):
                            raise ValueError('integrity')
                except FileNotFoundError:
                    raise Problem(503, 'artifact_unavailable', 'Registered artifact is unavailable') from None
                except (OSError, ValueError):
                    raise Problem(503, 'artifact_integrity_failed', 'Registered artifact failed integrity verification') from None
                snapshot.seek(0)
                yield model, snapshot
        finally:
            self.slots.release()

    def manifest(self, tenant, model_id):
        with self.snapshot(tenant, model_id) as (model, _):
            return delivery_manifest(model)


def delivery_manifest(model):
    recipe = model['recipe']
    is_detection = recipe in (detection.RECIPE, dataset_detection.RECIPE)
    if is_detection:
        names = model.get('class_names', list(detection.CLASS_NAMES))
        tensor_input = {'name': 'images', 'dtype': 'float32', 'layout': 'NCHW', 'shape': [1, 3, 160, 160]}
        tensor_output = {'name': 'detections', 'dtype': 'float32', 'shape': [1, 525, 5 + len(names)],
                         'row': ['cx', 'cy', 'width', 'height', 'objectness', 'class_scores...'],
                         'coordinates': 'decoded padded-input pixels; grid decode and sigmoid already inside graph'}
        preprocess = {'version': detection.metadata()['preprocessing'], 'source': 'RGB uint8 HWC',
                      'source_size_limits': {'min_side': 1, 'max_side': detection.MAX_SIDE},
                      'orientation': 'raw-decoded-no-exif-transpose', 'resize': 'Pillow bilinear',
                      'letterbox': {'target': [160, 160], 'anchor': 'top-left', 'pad_value': 114,
                                    'resized_size': 'floor aspect-preserving resize; integer ratios; reject zero side',
                                    'padding': 'right and bottom only'},
                      'channel_order': 'BGR', 'normalization': 'none; float32 0..255'}
        postprocess = {'implementation': 'model_platform.detection_geometry.postprocess_predictions',
                       'class_selection': 'largest class score; ties use smallest class ID',
                       'score': 'objectness * selected class score', 'score_threshold': detection.SCORE_THRESHOLD,
                       'threshold_comparison': '>=', 'evaluation_score_threshold': detection.EVALUATION_SCORE_THRESHOLD,
                       'restore': 'cxcywh to xyxy; divide x/y by separately realized resized_width/original_width and resized_height/original_height',
                       'clipping': 'original [0,width] x [0,height]; discard nonpositive or degenerate boxes',
                       'box_format': 'continuous original-xyxy-half-open; no inclusive +1',
                       'nms': {'class_aware': True, 'iou_threshold': detection.NMS_IOU_THRESHOLD,
                               'suppress_when': 'IoU > threshold', 'tie_order': 'stable input order'},
                       'max_detections': 100, 'max_detections_scope': 'global-per-image',
                       'empty_detections_valid': True, 'reject_nonfinite_outputs': True}
        opset = 17
    else:
        names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        tensor_input = {'name': 'pixels', 'dtype': 'float32', 'layout': 'NCHW', 'shape': ['batch', 1, 28, 28], 'batch_bounds': [1, 32]}
        tensor_output = {'name': 'logits', 'dtype': 'float32', 'shape': ['batch', 10], 'semantics': 'unnormalized class logits'}
        preprocess = {'source': '28x28 grayscale row-major uint8', 'normalization': 'float32 pixels / float32(255)', 'resize': 'none'}
        postprocess = {'probabilities': 'float64 stable softmax subtracting row maximum', 'class': 'argmax; ties first class index'}
        opset = 18
    quality = next(item['quality'] for item in public_catalog() if item['id'] == recipe)
    return {'schema_version': SCHEMA, 'model_version_id': model['model_version_id'], 'recipe': recipe,
            'artifact': {'kind': 'deployment-model-not-training-checkpoint', 'format': model['format'],
                         'media_type': 'application/octet-stream', 'filename': model['model_version_id'] + '.onnx',
                         'download_url': '/models/' + model['model_version_id'] + '/artifact',
                         'sha256': model['artifact_sha256'], 'bytes': model['artifact_bytes'], 'onnx_opset': opset,
                         'external_data': False},
            'input': tensor_input, 'preprocessing': preprocess, 'output': tensor_output, 'postprocessing': postprocess,
            'classes': [{'id': i, 'name': name} for i, name in enumerate(names)],
            'runtime': {'target_profile': model['target_profile'], 'provider': 'CPUExecutionProvider',
                        'verified_dependency_versions': model.get('dependencies', {}), 'supported_version_range': None,
                        'device_certified': False, 'tensorrt': 'unsupported', 'rknn': 'unsupported'},
            'lineage': {key: model.get(key) for key in ('run_id', 'attempt_id', 'evaluation_id', 'dataset', 'config_sha256', 'environment', 'execution_environment')},
            'evaluation': model['evaluation'], 'quality': dict(quality, production_allowed=False),
            'scope': 'Existing locally verified CPU ONNX protocol; no target-device, business-quality or flight-safety approval'}
