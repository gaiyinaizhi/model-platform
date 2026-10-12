"""Static YOLOX-Nano detection recipe; optional packages are imported lazily."""
import hashlib
import os
import stat
from .common import Problem, object_fields

RECIPE = 'yolox-nano-cat/v1'
CONTRACT = 'rgb-cat-detection/v1'
FORMAT = 'yolox-nano-onnx/v1'
REFERENCE = 'openimages-cat-reviewed/v1'
INPUT_SIZE = 160
MAX_SIDE = 384
MAX_ARTIFACT = 8 * 1024 * 1024
CLASS_NAMES = ['Cat']
SCORE_THRESHOLD = 0.01
EVALUATION_SCORE_THRESHOLD = 0.001
NMS_IOU_THRESHOLD = 0.65
CONFIG = {'epochs': 20, 'learning_rate': 0.001, 'seed': 20261012,
          'batch_size': 4, 'input_size': INPUT_SIZE, 'from_scratch': True}


def request_body():
    return {'recipe': RECIPE, 'dataset': {'source': 'Open Images', 'version': 'v6-cat-reviewed-v1',
            'rights_confirmed': True, 'reference': REFERENCE}, 'config': dict(CONFIG)}


def validate_request(data, config):
    object_fields(data, ('source', 'version', 'rights_confirmed', 'reference'))
    object_fields(config, tuple(CONFIG))
    if data != request_body()['dataset'] or data['rights_confirmed'] is not True:
        raise Problem(400, 'invalid_input', 'Only the reviewed fixed Open Images Cat reference is supported')
    if any(type(config[k]) is not type(v) or config[k] != v for k, v in CONFIG.items()):
        raise Problem(400, 'invalid_input', 'Detection uses a fixed reproducible configuration')


def images(xs, dimensions=None):
    from .detection_geometry import validate_images, letterbox_transform
    validate_images(xs, max_images=1, max_side=MAX_SIDE)
    try:
        for image in xs:
            letterbox_transform(image['width'], image['height'], target_width=INPUT_SIZE, target_height=INPUT_SIZE)
    except ValueError:
        raise Problem(400, 'invalid_input', 'Image aspect ratio is outside the resize contract') from None


def probe_input():
    return [{'width': INPUT_SIZE, 'height': INPUT_SIZE, 'pixels': [0] * (INPUT_SIZE * INPUT_SIZE * 3)}]


def probe_output(predictions):
    import math
    if not isinstance(predictions, list) or len(predictions) != 1:
        raise ValueError('Invalid detection probe batch')
    output = predictions[0]
    if (not isinstance(output, dict) or set(output) != {'width', 'height', 'detections'}
            or type(output['width']) is not int or type(output['height']) is not int
            or output['width'] != INPUT_SIZE or output['height'] != INPUT_SIZE):
        raise ValueError('Invalid detection probe dimensions')
    boxes = output['detections']
    if not isinstance(boxes, list) or len(boxes) > 100:
        raise ValueError('Invalid detection probe boxes')
    for detection in boxes:
        if not isinstance(detection, dict) or set(detection) != {'class_id', 'box', 'score'}:
            raise ValueError('Invalid detection probe record')
        box, score = detection['box'], detection['score']
        if (type(detection['class_id']) is not int or detection['class_id'] != 0
                or not isinstance(box, list) or len(box) != 4
                or not all(type(v) in (int, float) and 0 <= v <= INPUT_SIZE for v in box)
                or type(score) not in (int, float) or not SCORE_THRESHOLD <= score <= 1
                or not 0 <= box[0] < box[2] <= INPUT_SIZE or not 0 <= box[1] < box[3] <= INPUT_SIZE
                or not SCORE_THRESHOLD <= score <= 1):
            raise ValueError('Invalid detection probe output')


def metadata():
    return {'recipe': RECIPE, 'contract': CONTRACT, 'format': FORMAT, 'dimensions': str(INPUT_SIZE * INPUT_SIZE * 3),
            'input_size': str(INPUT_SIZE), 'class_names': 'Cat', 'class_ids': '0',
            'preprocessing': 'rgb-to-bgr-pillow-bilinear-top-left-114-f32-0-255/v1',
            'box_format': 'original-xyxy-half-open', 'score_threshold': str(SCORE_THRESHOLD),
            'evaluation_score_threshold': str(EVALUATION_SCORE_THRESHOLD), 'nms_iou_threshold': str(NMS_IOU_THRESHOLD)}


def load_model(path, expected_hash):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= MAX_ARTIFACT:
            raise ValueError('Invalid detection artifact type or size')
        content = stream.read(MAX_ARTIFACT + 1)
    if len(content) > MAX_ARTIFACT or hashlib.sha256(content).hexdigest() != expected_hash:
        raise ValueError('Detection artifact integrity failed')
    import onnx
    import onnxruntime as ort
    graph = onnx.load_model_from_string(content)
    props = {p.key: p.value for p in graph.metadata_props}
    if props != metadata() or len(graph.metadata_props) != len(props):
        raise ValueError('Invalid detection metadata')
    if graph.functions or graph.training_info or graph.graph.sparse_initializer or len(graph.graph.node) > 1500:
        raise ValueError('Unsupported detection graph')
    allowed = {'Conv', 'Sigmoid', 'Mul', 'Add', 'Concat', 'MaxPool', 'Resize', 'Slice', 'Constant',
               'Reshape', 'Transpose', 'Exp', 'Shape', 'Gather', 'Unsqueeze', 'Cast', 'Identity', 'Split', 'Expand'}
    tensors = list(graph.graph.initializer)
    for node in graph.graph.node:
        if node.domain not in ('', 'ai.onnx') or node.op_type not in allowed:
            raise ValueError('Unsupported detection operator')
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.TENSOR:
                tensors.append(attr.t)
            elif attr.type == onnx.AttributeProto.TENSORS:
                tensors.extend(attr.tensors)
            elif attr.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS,
                              onnx.AttributeProto.SPARSE_TENSOR, onnx.AttributeProto.SPARSE_TENSORS):
                raise ValueError('Nested detection graph forbidden')
    if any(t.data_location == onnx.TensorProto.EXTERNAL or t.external_data for t in tensors):
        raise ValueError('External detection tensors forbidden')
    onnx.checker.check_model(graph)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(content, sess_options=options, providers=['CPUExecutionProvider'])
    ins, outs = session.get_inputs(), session.get_outputs()
    if (len(ins) != 1 or len(outs) != 1 or ins[0].name != 'images' or outs[0].name != 'detections'
            or ins[0].type != 'tensor(float)' or outs[0].type != 'tensor(float)'
            or ins[0].shape != [1, 3, INPUT_SIZE, INPUT_SIZE] or outs[0].shape != [1, 525, 6]):
        raise ValueError('Invalid detection tensor contract')
    return {'recipe': RECIPE, 'contract': CONTRACT, 'format': FORMAT,
            'dimensions': INPUT_SIZE * INPUT_SIZE * 3, 'session': session}


def preprocess(image):
    """PIL RGB image -> model BGR CHW float32; exact realized resize scales."""
    import numpy as np
    from PIL import Image
    from .detection_geometry import letterbox_transform
    transform = letterbox_transform(image.width, image.height, target_width=INPUT_SIZE, target_height=INPUT_SIZE)
    resized = image.convert('RGB').resize((transform.resized_width, transform.resized_height), Image.Resampling.BILINEAR)
    canvas = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, dtype=np.float32)
    canvas[:transform.resized_height, :transform.resized_width] = np.asarray(resized, dtype=np.float32)[:, :, ::-1]
    return np.ascontiguousarray(canvas.transpose(2, 0, 1)), transform


def predict(model, xs):
    images(xs)
    import numpy as np
    from PIL import Image
    from .detection_geometry import postprocess_predictions
    results = []
    for item in xs:
        image = Image.fromarray(np.asarray(item['pixels'], dtype=np.uint8).reshape(item['height'], item['width'], 3))
        batch, transform = preprocess(image)
        raw = model['session'].run(['detections'], {'images': batch[None]})[0]
        if raw.shape != (1, 525, 6) or not np.isfinite(raw).all():
            raise ValueError('Invalid detection outputs')
        detections = postprocess_predictions(raw[0].tolist(), transform, num_classes=1,
                    score_threshold=SCORE_THRESHOLD, iou_threshold=NMS_IOU_THRESHOLD, max_detections=100)
        results.append({'width': item['width'], 'height': item['height'], 'detections': detections})
    return results
