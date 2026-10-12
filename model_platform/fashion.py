"""Optional Fashion-MNIST contract. Importing this module needs only stdlib."""
import hashlib
import os
import stat
from .common import Problem, object_fields

RECIPE = 'fashion-cnn/v1'
CONTRACT = 'fashion-28x28x10/v1'
FORMAT = 'fashion-onnx/v1'
REFERENCE = 'fashion-mnist-official/v1'
CONFIG = {'epochs': 5, 'learning_rate': 0.001, 'seed': 20261012, 'train_count': 6000, 'batch_size': 128}
MAX_ARTIFACT = 2 * 1024 * 1024


def request_body():
    return {'recipe': RECIPE, 'dataset': {'source': 'Fashion-MNIST', 'version': 'official-v1',
            'rights_confirmed': True, 'reference': REFERENCE}, 'config': dict(CONFIG)}


def validate_request(data, config):
    object_fields(data, ('source', 'version', 'rights_confirmed', 'reference'))
    if data != request_body()['dataset'] or data['rights_confirmed'] is not True:
        raise Problem(400, 'invalid_input', 'Only the fixed official Fashion-MNIST reference is supported')
    object_fields(config, tuple(CONFIG))
    if any(type(config[k]) is not type(v) or config[k] != v for k, v in CONFIG.items()):
        raise Problem(400, 'invalid_input', 'Fashion training uses a fixed reproducible configuration')


def pixels(xs):
    if not isinstance(xs, list) or not 1 <= len(xs) <= 32:
        raise Problem(400, 'invalid_input', 'Fashion inference requires 1..32 images')
    for row in xs:
        if not isinstance(row, list) or len(row) != 784 or any(type(p) is not int or not 0 <= p <= 255 for p in row):
            raise Problem(400, 'invalid_input', 'Images must be 784 row-major integer pixels in 0..255')


def load_model(path, expected_hash):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_ARTIFACT:
            raise ValueError('Invalid artifact file type or size')
        content = stream.read(MAX_ARTIFACT + 1)
    if len(content) > MAX_ARTIFACT or hashlib.sha256(content).hexdigest() != expected_hash:
        raise ValueError('Artifact integrity check failed')
    # Only hash-verified, registered, self-generated ONNX reaches this parser.
    import onnx
    import onnxruntime as ort
    graph = onnx.load_model_from_string(content)
    expected_metadata = {'recipe': RECIPE, 'contract': CONTRACT, 'format': FORMAT,
                         'dimensions': '784', 'pixel_scale': '255', 'classes': '0,1,2,3,4,5,6,7,8,9'}
    metadata = {item.key: item.value for item in graph.metadata_props}
    if metadata != expected_metadata or len(graph.metadata_props) != len(expected_metadata):
        raise ValueError('Invalid ONNX model metadata')
    if graph.functions or graph.training_info or graph.graph.sparse_initializer or len(graph.graph.node) > 64:
        raise ValueError('Unsupported ONNX graph')
    allowed = {'Conv', 'Relu', 'MaxPool', 'Reshape', 'Flatten', 'Gemm', 'MatMul', 'Add', 'Shape', 'Squeeze', 'Concat', 'Constant', 'Cast', 'Unsqueeze'}
    if any(node.domain not in ('', 'ai.onnx') or node.op_type not in allowed for node in graph.graph.node):
        raise ValueError('Unsupported ONNX operator')
    tensors = list(graph.graph.initializer)
    for node in graph.graph.node:
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.TENSOR:
                tensors.append(attr.t)
            elif attr.type == onnx.AttributeProto.TENSORS:
                tensors.extend(attr.tensors)
    if any(t.data_location == onnx.TensorProto.EXTERNAL or t.external_data for t in tensors):
        raise ValueError('External tensors are forbidden')
    if any(attr.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS, onnx.AttributeProto.SPARSE_TENSOR, onnx.AttributeProto.SPARSE_TENSORS) for node in graph.graph.node for attr in node.attribute):
        raise ValueError('Nested graphs and sparse attributes are forbidden')
    onnx.checker.check_model(graph)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(content, sess_options=options, providers=['CPUExecutionProvider'])
    ins, outs = session.get_inputs(), session.get_outputs()
    if (len(ins) != 1 or len(outs) != 1 or ins[0].type != 'tensor(float)' or outs[0].type != 'tensor(float)'
            or ins[0].shape[1:] != [1, 28, 28] or outs[0].shape[1:] != [10]
            or len(ins[0].shape) != 4 or len(outs[0].shape) != 2
            or isinstance(ins[0].shape[0], int) or isinstance(outs[0].shape[0], int)):
        raise ValueError('Invalid ONNX input/output contract')
    return {'recipe': RECIPE, 'contract': CONTRACT, 'format': FORMAT, 'dimensions': 784, 'session': session}


def predict(model, xs):
    pixels(xs)
    import numpy as np
    session = model['session']
    batch = np.asarray(xs, dtype=np.float32).reshape(-1, 1, 28, 28) / np.float32(255)
    logits = session.run(None, {session.get_inputs()[0].name: batch})[0]
    if logits.shape != (len(xs), 10) or not np.isfinite(logits).all():
        raise ValueError('Invalid ONNX outputs')
    # Double precision softmax keeps JSON probabilities and readiness probes stable.
    scores = logits.astype(np.float64)
    scores -= scores.max(axis=1, keepdims=True)
    probabilities = np.exp(scores)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return [{'class': int(row.argmax()), 'probabilities': row.tolist()} for row in probabilities]
