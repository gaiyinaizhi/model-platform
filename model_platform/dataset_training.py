"""Bounded imported-dataset training; no tuning or selection uses held-out data."""
import hashlib
import importlib.metadata
import io
import time

from .algorithm import TrainingFailure
from .common import digest, Problem
from . import dataset_detection as recipe, detection
from .datasets import DatasetStore
from .detection_training import build_model, _deadline, _save, _evaluate


def train_export(spec, runtime, body, path, log, execution_environment, submission_environment):
    import numpy as np
    import torch
    import onnx
    from PIL import Image

    recipe.validate_request(body['dataset'], body['config'])
    binding = body['_dataset_binding']
    datasets = DatasetStore(runtime)

    def verified():
        current = recipe.bind_version(datasets, body['_dataset_tenant'], binding['dataset_version_id'])
        if current != binding:
            raise Problem(409, 'dataset_integrity', 'Bound dataset content has changed')
        return datasets.get_version(body['_dataset_tenant'], binding['dataset_version_id'])

    version = verified()
    config = dict(body['config'], **recipe.FIXED_CONFIG)
    names = binding['class_names']
    categories = binding['classes']
    root = datasets.root / 'versions' / binding['dataset_version_id']
    # Budget at admission is 20 million total decoded pixels. Hash bytes actually
    # consumed before decoding, and release source files after each read.
    splits = {'train': [], 'val': [], 'test': []}
    for index, sample in enumerate(version['manifest']['samples']):
        raw = (root/sample['image']).read_bytes()
        label = (root/sample['label']).read_bytes()
        if hashlib.sha256(raw).hexdigest() != sample['image_sha256'] or hashlib.sha256(label).hexdigest() != sample['label_sha256']:
            raise Problem(409, 'dataset_integrity', 'Dataset sample has changed')
        with Image.open(io.BytesIO(raw)) as source:
            image = source.convert('RGB')  # Same raw orientation as import; no EXIF transpose.
        boxes = recipe.labels(label, image.width, image.height)
        splits[sample['split']].append({'record': {'image_id': index+1, 'boxes': boxes}, 'image': image})
    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(2)
    except RuntimeError:
        if torch.get_num_interop_threads() != 2:
            raise TrainingFailure('detection_thread_budget_unavailable') from None
    torch.manual_seed(config['seed'])
    torch.use_deterministic_algorithms(True)
    model = build_model(len(names))
    def parameter_hash():
        return hashlib.sha256(b''.join(p.detach().numpy().tobytes() for p in model.parameters())).hexdigest()
    initial_hash = parameter_hash()
    arrays, targets = [], []
    for item in splits['train']:
        array, transform = detection.preprocess(item['image'])
        arrays.append(array)
        target = np.zeros((recipe.MAX_BOXES, 5), dtype=np.float32)
        for index, box in enumerate(item['record']['boxes']):
            x1, y1, x2, y2 = box['xyxy']
            target[index] = [box['class_id'], (x1+x2)/2*transform.scale_x, (y1+y2)/2*transform.scale_y,
                             (x2-x1)*transform.scale_x, (y2-y1)*transform.scale_y]
        targets.append(target)
    train_x, train_y = torch.from_numpy(np.stack(arrays)), torch.from_numpy(np.stack(targets))
    log('dataset_version_verified', {key: binding[key] for key in ('dataset_version_id', 'sha256', 'train_count', 'validation_count', 'test_count', 'evaluation_split')})
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    generator = torch.Generator(device='cpu').manual_seed(config['seed'])
    started = time.monotonic()
    losses = []
    model.train()
    for epoch in range(config['epochs']):
        order = torch.randperm(len(train_x), generator=generator)
        total_loss = 0.
        for offset in range(0, len(order), config['batch_size']):
            _deadline(started, spec.budgets.training_seconds, 'training_timeout')
            ids = order[offset:offset+config['batch_size']]
            optimizer.zero_grad(set_to_none=True)
            loss = model(train_x[ids], train_y[ids])['total_loss']
            if not bool(torch.isfinite(loss)):
                raise TrainingFailure('detection_training_nonfinite')
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(ids)
        average = total_loss / len(train_x)
        losses.append(average)
        log('train_progress', {'epoch': epoch+1, 'total_epochs': config['epochs'], 'loss': average})
    _deadline(started, spec.budgets.training_seconds, 'training_timeout')
    training_seconds = time.monotonic()-started
    final_hash = parameter_hash()
    if final_hash == initial_hash:
        raise TrainingFailure('detection_parameters_unchanged')
    model.eval()
    started = time.monotonic()
    buffer = io.BytesIO()
    torch.onnx.export(model, train_x[:1], buffer, dynamo=False, opset_version=17,
                      input_names=['images'], output_names=['detections'], external_data=False)
    proto = onnx.load_model_from_string(buffer.getvalue())
    onnx.helper.set_model_props(proto, recipe.metadata(names, binding))
    content = proto.SerializeToString()
    artifact_hash = _save(path, content)
    loaded = recipe.load_model(path, artifact_hash)
    _deadline(started, spec.budgets.export_seconds, 'detection_export_timeout')
    export_seconds = time.monotonic()-started
    log('artifact_written', {'sha256': artifact_hash, 'bytes': len(content), 'format': recipe.FORMAT})
    verified()  # Independent ONNX evaluation never starts against changed bytes.
    started = time.monotonic()
    validation = _evaluate(model, loaded['session'], splits['val'], started, spec.budgets.evaluation_seconds, categories)
    primary = (_evaluate(model, loaded['session'], splits['test'], started, spec.budgets.evaluation_seconds, categories)
               if splits['test'] else validation)
    evaluation = dict(primary['onnx'])
    evaluation.update({'contract': 'dataset-coco-style-bbox/v1', 'artifact_sha256': artifact_hash,
                       'dataset_version_id': binding['dataset_version_id'], 'dataset_content_sha256': binding['sha256'],
                       'dataset_sha256': binding['evaluation_sha256'], 'evaluation_sha256': binding['evaluation_sha256'],
                       'split': binding['evaluation_split'], 'role': 'held-out-test' if splits['test'] else 'validation-only',
                       'validation': validation, 'pytorch': primary['pytorch'], 'parity': primary['parity'],
                       'scope': 'Caller-imported immutable YOLO split; validation-only if test absent; no tuning, early stopping or checkpoint selection on val/test; no business-quality proof',
                       'score_threshold': detection.EVALUATION_SCORE_THRESHOLD, 'nms_iou_threshold': detection.NMS_IOU_THRESHOLD,
                       'max_detections': {'upstream': 100, 'scope': 'global-per-image-class-aware-nms'},
                       'preprocessing': {'version': detection.metadata()['preprocessing'], 'input_size': 160,
                                         'orientation': 'raw-decoded-no-exif-transpose', 'epsilon_box_clamp': 1e-9},
                       'execution_environment': execution_environment})
    evaluation['compatibility_key'] = digest(recipe.compatibility(evaluation))
    log('evaluated', {'split': evaluation['split'], 'role': evaluation['role'], 'map_50': evaluation['map_50'], 'map_50_95': evaluation['map_50_95']})
    manifest = {'recipe': recipe.RECIPE, 'contract': recipe.CONTRACT, 'format': recipe.FORMAT,
                'artifact_bytes': len(content), 'dimensions': 160*160*3, 'training_seconds': training_seconds,
                'export_seconds': export_seconds, 'evaluation_seconds': time.monotonic()-started,
                'dataset': binding, 'config': body['config'], 'training_config': config,
                'config_sha256': digest(config), 'environment': submission_environment,
                'execution_environment': execution_environment, 'target_profile': 'local-cpu-onnx/v1',
                'class_names': names, 'preprocessing': recipe.metadata(names, binding),
                'parameter_count': sum(p.numel() for p in model.parameters()),
                'initial_parameters_sha256': initial_hash, 'final_parameters_sha256': final_hash,
                'training_loss_history': losses,
                'dependencies': {n: importlib.metadata.version(n) for n in ('torch','numpy','onnx','onnxruntime','Pillow','pycocotools','PyYAML')},
                'upstream': {'repository': 'https://github.com/Megvii-BaseDetection/YOLOX',
                             'commit': '6ddff4824372906469a7fae2dc3206c7aa4bbaee', 'code_license': 'Apache-2.0', 'pretrained_weights': False},
                'quality': {'production_allowed': False, 'status': 'not-production-validated'},
                'capabilities': {'train': True, 'evaluate': True, 'serve_cpu': True, 'device': False}}
    verified()
    return artifact_hash, evaluation, manifest
