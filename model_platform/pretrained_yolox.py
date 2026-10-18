"""Exact reviewed uploaded YOLOX-Nano bytes; separate, CPU-only protocol.

Reference: Megvii-BaseDetection/YOLOX commit
6ddff4824372906469a7fae2dc3206c7aa4bbaee (Apache-2.0).
Asset hash identifies reviewed user bytes, not publisher authentication.
"""
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import time

PROTOCOL = CONTRACT = FORMAT = 'yolox-nano-coco-onnx/v1'
ASSET_SHA256 = 'c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d'
ASSET_BYTES = 3659407
SOURCE_COMMIT = '6ddff4824372906469a7fae2dc3206c7aa4bbaee'
INPUT_SIZE = 416
MAX_SIDE = 384
SCORE_THRESHOLD = 0.3
NMS_IOU_THRESHOLD = 0.45
MAX_DETECTIONS = 100
VALIDATION_WALL_SECONDS = 30
VALIDATION_CPU_SECONDS = 20
VALIDATION_ADDRESS_BYTES = 2 * 1024 ** 3
CLASS_NAMES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')
COCO_CATEGORY_IDS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90)
LABEL_TABLE_SHA256 = hashlib.sha256(json.dumps(list(zip(CLASS_NAMES, COCO_CATEGORY_IDS)), separators=(',', ':')).encode()).hexdigest()


def _check_bytes(content, expected_hash=ASSET_SHA256):
    if expected_hash != ASSET_SHA256 or len(content) != ASSET_BYTES or hashlib.sha256(content).hexdigest() != ASSET_SHA256:
        raise ValueError('Unreviewed pretrained artifact bytes')


def _read(path, expected_hash=ASSET_SHA256):
    parts = Path(os.path.abspath(path)).parts
    directory = os.open(parts[0], os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in parts[1:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = child
        fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
    finally:
        os.close(directory)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size != ASSET_BYTES:
            raise ValueError('Invalid pretrained artifact type or size')
    except BaseException:
        os.close(fd)
        raise
    with os.fdopen(fd, 'rb') as stream:
        content = stream.read(ASSET_BYTES + 1)
        after = os.fstat(stream.fileno())
    if any(getattr(before, f) != getattr(after, f) for f in ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')):
        raise ValueError('Pretrained artifact changed during read')
    _check_bytes(content, expected_hash)
    return content


def _static_check(content):
    _check_bytes(content)  # Before optional imports or protobuf parsing.
    import onnx
    graph = onnx.load_model_from_string(content)
    if (graph.ir_version != 6 or [(o.domain, o.version) for o in graph.opset_import] != [('', 11)]
            or graph.functions or graph.training_info or graph.metadata_props
            or graph.graph.sparse_initializer or len(graph.graph.node) != 369 or len(graph.graph.initializer) != 236):
        raise ValueError('Unsupported pretrained graph')
    def signature(values):
        return [(v.name, v.type.tensor_type.elem_type, [d.dim_value if d.HasField('dim_value') else None
                 for d in v.type.tensor_type.shape.dim]) for v in values]
    if (signature(graph.graph.input) != [('images', 1, [1, 3, 416, 416])]
            or signature(graph.graph.output) != [('output', 1, [1, 3549, 85])]):
        raise ValueError('Invalid pretrained tensor contract')
    tensors = list(graph.graph.initializer)
    allowed = {'Slice', 'Concat', 'Conv', 'Sigmoid', 'Mul', 'Add', 'MaxPool', 'Resize', 'Reshape', 'Transpose'}
    for node in graph.graph.node:
        if node.domain != '' or node.op_type not in allowed:
            raise ValueError('Unsupported pretrained operator')
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.TENSOR:
                tensors.append(attr.t)
            elif attr.type == onnx.AttributeProto.TENSORS:
                tensors.extend(attr.tensors)
            elif attr.type in (5, 10, 11, 12):
                raise ValueError('Nested or sparse graph forbidden')
    total = 0
    for tensor in tensors:
        if tensor.external_data or tensor.data_location == onnx.TensorProto.EXTERNAL:
            raise ValueError('External tensor forbidden')
        elements = 1
        for dimension in tensor.dims:
            if dimension < 0 or dimension > 1048576:
                raise ValueError('Tensor dimension budget exceeded')
            elements *= dimension
        total += elements
        if elements > 2000000 or total > 2000000:
            raise ValueError('Tensor element budget exceeded')
    onnx.checker.check_model(graph)
    return {'ir_version': 6, 'opset': 11, 'nodes': 369, 'initializers': 236, 'initializer_elements': total}


def _session(content):
    _check_bytes(content)
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(content, sess_options=options, providers=['CPUExecutionProvider'])
    if session.get_providers() != ['CPUExecutionProvider']:
        raise ValueError('CPU provider required')
    ins, outs = session.get_inputs(), session.get_outputs()
    if (len(ins) != 1 or len(outs) != 1 or ins[0].name != 'images' or outs[0].name != 'output'
            or ins[0].type != 'tensor(float)' or outs[0].type != 'tensor(float)'
            or ins[0].shape != [1, 3, 416, 416] or outs[0].shape != [1, 3549, 85]):
        raise ValueError('Unexpected runtime tensor contract')
    return session


def _validate_bytes(content):
    _check_bytes(content)
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    started = time.monotonic()
    try:
        process = subprocess.run([sys.executable, '-m', 'model_platform.pretrained_yolox', '--validate'],
            input=content, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=VALIDATION_WALL_SECONDS,
            cwd=str(Path(__file__).resolve().parents[1]), env=env, check=False)
    except subprocess.TimeoutExpired:
        raise ValueError('Pretrained validation wall-time budget exceeded') from None
    if process.returncode != 0:
        raise ValueError('Pretrained isolated validation failed')
    try:
        report = json.loads(process.stdout)
        if report['status'] != 'validated' or report['sha256'] != ASSET_SHA256:
            raise ValueError()
    except (ValueError, KeyError, TypeError):
        raise ValueError('Invalid isolated validation report') from None
    report['wall_seconds'] = time.monotonic() - started
    return report


def validate_asset(path):
    return _validate_bytes(_read(path))


def load(path, expected_hash):
    content = _read(path, expected_hash)
    report = _validate_bytes(content)
    return {'session': _session(content), 'protocol': PROTOCOL, 'contract': CONTRACT, 'format': FORMAT,
            'class_names': list(CLASS_NAMES), 'dimensions': INPUT_SIZE * INPUT_SIZE * 3, 'validation': report}


def images(features, dimensions=None):
    from .detection_geometry import validate_images
    validate_images(features, max_images=1, max_side=MAX_SIDE)


validate_features = images


def probe_input():
    return [{'width': 32, 'height': 32, 'pixels': [0] * (32 * 32 * 3)}]


def preprocess(item):
    """Pinned upstream modern OpenCV preprocessing, explicit RGB->BGR adapter."""
    import cv2
    import numpy as np
    cv2.setNumThreads(1)
    bgr = np.asarray(item['pixels'], dtype=np.uint8).reshape(item['height'], item['width'], 3)[:, :, ::-1]
    ratio = min(INPUT_SIZE / item['height'], INPUT_SIZE / item['width'])
    resized = cv2.resize(bgr, (int(item['width'] * ratio), int(item['height'] * ratio)), interpolation=cv2.INTER_LINEAR).astype(np.uint8)
    padded = np.ones((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8) * 114
    padded[:resized.shape[0], :resized.shape[1]] = resized
    return np.ascontiguousarray(padded.transpose(2, 0, 1), dtype=np.float32), ratio


def decode(raw):
    import numpy as np
    if raw.shape != (1, 3549, 85) or raw.dtype != np.float32 or not np.isfinite(raw).all():
        raise ValueError('Invalid pretrained outputs')
    output = raw.copy()
    grids, expanded = [], []
    for stride in (8, 16, 32):
        size = INPUT_SIZE // stride
        xv, yv = np.meshgrid(np.arange(size), np.arange(size))
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        expanded.append(np.full((*grid.shape[:2], 1), stride))
    grid, strides = np.concatenate(grids, 1), np.concatenate(expanded, 1)
    output[..., :2] = (output[..., :2] + grid) * strides
    with np.errstate(over='ignore', invalid='ignore'):
        output[..., 2:4] = np.exp(output[..., 2:4]) * strides
    if not np.isfinite(output).all() or ((output[..., 4:] < 0) | (output[..., 4:] > 1)).any():
        raise ValueError('Invalid decoded pretrained outputs')
    return output


def reference_detections(raw, ratio):
    """Upstream argmax/class-agnostic NMS, inclusive+1, strict score>.3.

    Uses upstream NumPy reverse argsort tie behavior; no cross-version promise.
    Boxes are intentionally not clipped here.
    """
    import numpy as np
    predictions = decode(raw)[0]
    boxes = predictions[:, :4]
    corners = np.ones_like(boxes)
    corners[:, :2] = boxes[:, :2] - boxes[:, 2:] / 2.
    corners[:, 2:] = boxes[:, :2] + boxes[:, 2:] / 2.
    corners /= ratio
    scores = predictions[:, 4:5] * predictions[:, 5:]
    classes = scores.argmax(1)
    best = scores[np.arange(len(classes)), classes]
    mask = best > SCORE_THRESHOLD
    boxes, scores, classes = corners[mask], best[mask], classes[mask]
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        others = order[1:]
        w = np.maximum(0., np.minimum(x2[i], x2[others]) - np.maximum(x1[i], x1[others]) + 1)
        h = np.maximum(0., np.minimum(y2[i], y2[others]) - np.maximum(y1[i], y1[others]) + 1)
        intersection = w * h
        overlap = intersection / (areas[i] + areas[others] - intersection)
        order = others[np.where(overlap <= NMS_IOU_THRESHOLD)[0]]
    return [{'class_id': int(classes[i]), 'score': float(scores[i]), 'box': boxes[i].tolist()} for i in keep]


def predict(model, features):
    images(features)
    results = []
    for item in features:
        batch, ratio = preprocess(item)
        raw = model['session'].run(['output'], {'images': batch[None]})[0]
        detections = []
        for detection in reference_detections(raw, ratio):
            x1, y1, x2, y2 = detection['box']
            box = [max(0., min(item['width'], x1)), max(0., min(item['height'], y1)),
                   max(0., min(item['width'], x2)), max(0., min(item['height'], y2))]
            if box[2] > box[0] and box[3] > box[1]:
                detections.append(dict(detection, box=box))
            if len(detections) == MAX_DETECTIONS:
                break
        results.append({'width': item['width'], 'height': item['height'], 'detections': detections})
    return results


def probe_output(predictions):
    import math
    if not isinstance(predictions, list) or len(predictions) != 1:
        raise ValueError('Invalid pretrained probe batch')
    result = predictions[0]
    if not isinstance(result, dict) or set(result) != {'width', 'height', 'detections'} or result['width'] != 32 or result['height'] != 32:
        raise ValueError('Invalid pretrained probe dimensions')
    boxes = result['detections']
    if not isinstance(boxes, list) or len(boxes) > MAX_DETECTIONS:
        raise ValueError('Invalid pretrained probe boxes')
    for item in boxes:
        if not isinstance(item, dict) or set(item) != {'class_id', 'score', 'box'}:
            raise ValueError('Invalid pretrained probe record')
        box, score = item['box'], item['score']
        if (type(item['class_id']) is not int or not 0 <= item['class_id'] < 80
                or type(score) not in (int, float) or not SCORE_THRESHOLD < score <= 1
                or not isinstance(box, list) or len(box) != 4
                or any(type(x) not in (int, float) or not math.isfinite(x) for x in box)
                or not 0 <= box[0] < box[2] <= 32 or not 0 <= box[1] < box[3] <= 32):
            raise ValueError('Invalid pretrained probe detection')


def _child():
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (VALIDATION_ADDRESS_BYTES, VALIDATION_ADDRESS_BYTES))
    resource.setrlimit(resource.RLIMIT_CPU, (VALIDATION_CPU_SECONDS, VALIDATION_CPU_SECONDS))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    started = time.monotonic()
    content = sys.stdin.buffer.read(ASSET_BYTES + 1)
    report = _static_check(content)
    import numpy as np
    import onnx
    import onnxruntime as ort
    import cv2
    cv2.setNumThreads(1)
    session = _session(content)
    raw = session.run(['output'], {'images': np.zeros((1, 3, 416, 416), dtype=np.float32)})[0]
    decode(raw)
    probe_output(predict({'session': session}, probe_input()))
    report.update(status='validated', protocol=PROTOCOL, sha256=ASSET_SHA256, bytes=ASSET_BYTES,
                  provider='CPUExecutionProvider', raw_output_shape=list(raw.shape), finite_output=True,
                  numpy_version=np.__version__, onnx_version=onnx.__version__, onnxruntime_version=ort.__version__,
                  opencv_version=cv2.__version__, pipeline_probe=True, label_table_sha256=LABEL_TABLE_SHA256,
                  child_wall_seconds=time.monotonic() - started,
                  peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                  limits={'wall_seconds': VALIDATION_WALL_SECONDS, 'cpu_seconds': VALIDATION_CPU_SECONDS,
                          'address_space_bytes': VALIDATION_ADDRESS_BYTES},
                  publisher_authenticated=False, local_evaluation_status='not_evaluated')
    print(json.dumps(report, allow_nan=False))


if __name__ == '__main__':
    if sys.argv[1:] != ['--validate']:
        raise SystemExit(2)
    _child()
