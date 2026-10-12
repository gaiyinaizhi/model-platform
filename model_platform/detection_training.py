"""Fixed CPU YOLOX-Nano training, immutable ONNX, independent COCO-style AP."""
import hashlib
import importlib.metadata
import io
import os
import time
from .algorithm import TrainingFailure
from .common import digest
from . import detection


def build_model():
    import torch
    from .yolox_nano.yolox import YOLOX
    from .yolox_nano.yolo_pafpn import YOLOPAFPN
    from .yolox_nano.yolo_head import YOLOXHead
    model = YOLOX(YOLOPAFPN(.33, .25, depthwise=True), YOLOXHead(1, .25, depthwise=True)).cpu()
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.eps = 1e-3
            module.momentum = .03
    model.head.initialize_biases(.01)
    return model


def _deadline(start, limit, code):
    if time.monotonic() - start > limit:
        raise TrainingFailure(code)


def _save(path, content):
    if not 0 < len(content) <= detection.MAX_ARTIFACT:
        raise TrainingFailure('detection_artifact_too_large')
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(content); stream.flush(); os.fsync(stream.fileno()); os.fchmod(stream.fileno(), 0o400)
    return hashlib.sha256(content).hexdigest()


def _box_parity(reference, actual):
    """Compare box sets, tolerating harmless ranking swaps of near-equal scores.

    COCO AP below is still computed from each backend's real scores/order, so
    matching box sets cannot hide a metric difference caused by a score swap.
    """
    if len(reference) != len(actual):
        raise TrainingFailure('detection_onnx_count_parity_failed')
    # Deterministic augmenting-path bipartite matching, not greedy nearest-box.
    # Every edge must satisfy BOTH fixed score and coordinate tolerances.
    edges = []
    for expected in reference:
        candidates = []
        for index, candidate in enumerate(actual):
            error = max(abs(x-y) for x,y in zip(expected['box'], candidate['box']))
            if (candidate['class_id'] == expected['class_id'] and error <= .1
                    and abs(candidate['score'] - expected['score']) <= 1e-4):
                candidates.append((error, index))
        edges.append([index for _, index in sorted(candidates)])
    matched = {}

    def augment(reference_index, visited):
        for actual_index in edges[reference_index]:
            if actual_index in visited:
                continue
            visited.add(actual_index)
            if actual_index not in matched or augment(matched[actual_index], visited):
                matched[actual_index] = reference_index
                return True
        return False

    if any(not augment(index, set()) for index in range(len(reference))):
        raise TrainingFailure('detection_onnx_box_parity_failed')
    max_box_error = max_score_error = 0.
    for actual_index, reference_index in matched.items():
        expected, candidate = reference[reference_index], actual[actual_index]
        max_box_error = max(max_box_error, max(abs(x-y) for x,y in zip(expected['box'], candidate['box'])))
        max_score_error = max(max_score_error, abs(expected['score']-candidate['score']))
    return max_box_error, max_score_error


def _evaluate(model, session, items, started, limit):
    import numpy as np
    import torch
    from .detection_geometry import postprocess_predictions
    from .detection_evaluation import evaluate_detections
    images, annotations, predictions, reference_predictions = [], [], [], []
    max_raw_error, max_box_error, max_score_error = 0., 0., 0.
    exact_detection_counts = 0
    serving_parity_count = 0
    for item in items:
        _deadline(started, limit, 'detection_evaluation_timeout')
        record, image = item['record'], item['image']
        image_id = record['image_id']
        images.append({'id': image_id, 'width': image.width, 'height': image.height})
        annotations.extend({'image_id': image_id, 'category_id': 0, 'bbox': box['xyxy']} for box in record['boxes'])
        array, transform = detection.preprocess(image)
        with torch.inference_mode():
            expected = model(torch.from_numpy(array[None])).numpy()
        actual = session.run(['detections'], {'images': array[None]})[0]
        if expected.shape != (1, 525, 6) or actual.shape != expected.shape or not np.isfinite(actual).all():
            raise TrainingFailure('detection_outputs_invalid')
        raw_error = float(np.max(np.abs(expected.astype(np.float64) - actual.astype(np.float64))))
        max_raw_error = max(max_raw_error, raw_error)
        if not np.allclose(expected, actual, rtol=1e-4, atol=1e-3):
            raise TrainingFailure('detection_onnx_raw_parity_failed')
        opts = dict(num_classes=1, score_threshold=detection.EVALUATION_SCORE_THRESHOLD,
                    iou_threshold=detection.NMS_IOU_THRESHOLD, max_detections=100)
        torch_boxes = postprocess_predictions(expected[0].tolist(), transform, **opts)
        onnx_boxes = postprocess_predictions(actual[0].tolist(), transform, **opts)
        box_error, score_error = _box_parity(torch_boxes, onnx_boxes)
        max_box_error, max_score_error = max(max_box_error, box_error), max(max_score_error, score_error)
        exact_detection_counts += 1
        serving_opts = dict(opts, score_threshold=detection.SCORE_THRESHOLD)
        serving_reference = postprocess_predictions(expected[0].tolist(), transform, **serving_opts)
        serving_actual = postprocess_predictions(actual[0].tolist(), transform, **serving_opts)
        _box_parity(serving_reference, serving_actual)
        serving_parity_count += 1
        for boxes, target in ((onnx_boxes, predictions), (torch_boxes, reference_predictions)):
            target.extend({'image_id': image_id, 'category_id': box['class_id'], 'bbox': box['box'], 'score': box['score']} for box in boxes)
    categories = [{'id': 0, 'name': 'Cat'}]
    actual_metrics = evaluate_detections(images, annotations, predictions, categories)
    expected_metrics = evaluate_detections(images, annotations, reference_predictions, categories)
    for key in ('map_50', 'map_50_95'):
        if actual_metrics[key] is None or abs(actual_metrics[key] - expected_metrics[key]) > 1e-6:
            raise TrainingFailure('detection_onnx_map_parity_failed')
    _deadline(started, limit, 'detection_evaluation_timeout')
    return {'onnx': actual_metrics, 'pytorch': expected_metrics, 'parity': {
        'sample_count': len(items), 'matching_detection_counts': exact_detection_counts,
        'box_matching': 'deterministic-one-to-one-perfect-bipartite/v1',
        'serving_threshold_parity_samples': serving_parity_count, 'serving_score_threshold': detection.SCORE_THRESHOLD,
        'max_absolute_raw_error': max_raw_error, 'raw_atol': .001, 'raw_rtol': .0001,
        'max_absolute_box_error_pixels': max_box_error, 'box_atol_pixels': .1,
        'max_absolute_score_error': max_score_error, 'score_atol': .0001, 'map_atol': .000001}}


def train_export(spec, runtime, body, path, log, execution_environment, submission_environment):
    import numpy as np
    import torch
    import onnx
    import onnxruntime
    import PIL
    import pycocotools.cocoeval
    from . import detection_data
    detection.validate_request(body['dataset'], body['config'])
    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(2)
    except RuntimeError:
        if torch.get_num_interop_threads() != 2:
            raise TrainingFailure('detection_thread_budget_unavailable') from None
    torch.manual_seed(detection.CONFIG['seed'])
    torch.use_deterministic_algorithms(True)
    manifest_data, splits = detection_data.load_dataset(runtime)
    log('detection_dataset_verified', {'train':128, 'validation':32, 'test':32,
                                       'manifest_sha256':detection_data.MANIFEST_SHA256})
    model = build_model()
    initial_hash = hashlib.sha256(b''.join(p.detach().numpy().tobytes() for p in model.parameters())).hexdigest()
    arrays, targets = [], []
    for item in splits['train']:
        array, transform = detection.preprocess(item['image'])
        arrays.append(array)
        target = np.zeros((100,5), dtype=np.float32)
        for index, box in enumerate(item['record']['boxes']):
            x1,y1,x2,y2 = box['xyxy']
            target[index] = [0, (x1+x2)/2*transform.scale_x, (y1+y2)/2*transform.scale_y,
                             (x2-x1)*transform.scale_x, (y2-y1)*transform.scale_y]
        targets.append(target)
    train_x, train_y = torch.from_numpy(np.stack(arrays)), torch.from_numpy(np.stack(targets))
    optimizer = torch.optim.Adam(model.parameters(), lr=detection.CONFIG['learning_rate'])
    generator = torch.Generator(device='cpu').manual_seed(detection.CONFIG['seed'])
    training_started = time.monotonic()
    model.train()
    loss_history=[]
    for epoch in range(detection.CONFIG['epochs']):
        order = torch.randperm(len(train_x), generator=generator)
        total_loss = 0.
        for offset in range(0,len(order),4):
            _deadline(training_started,spec.budgets.training_seconds,'training_timeout')
            idx=order[offset:offset+4]
            optimizer.zero_grad(set_to_none=True)
            loss=model(train_x[idx],train_y[idx])['total_loss']
            if not bool(torch.isfinite(loss)):
                raise TrainingFailure('detection_training_nonfinite')
            loss.backward(); optimizer.step()
            total_loss+=float(loss.detach())*len(idx)
        average=total_loss/len(train_x);loss_history.append(average)
        log('train_progress',{'epoch':epoch+1,'total_epochs':detection.CONFIG['epochs'],'loss':average})
    _deadline(training_started,spec.budgets.training_seconds,'training_timeout')
    training_seconds=time.monotonic()-training_started
    final_hash=hashlib.sha256(b''.join(p.detach().numpy().tobytes() for p in model.parameters())).hexdigest()
    if initial_hash==final_hash:
        raise TrainingFailure('detection_parameters_unchanged')
    model.eval()
    export_started=time.monotonic()
    buffer=io.BytesIO()
    # Fixed batch 1 keeps inference resource bounds and avoids dynamic code paths.
    torch.onnx.export(model,train_x[:1],buffer,dynamo=False,opset_version=17,
                      input_names=['images'],output_names=['detections'],external_data=False)
    proto=onnx.load_model_from_string(buffer.getvalue())
    onnx.helper.set_model_props(proto,detection.metadata())
    content=proto.SerializeToString();artifact_hash=_save(path,content)
    loaded=detection.load_model(path,artifact_hash)
    _deadline(export_started,spec.budgets.export_seconds,'detection_export_timeout')
    export_seconds=time.monotonic()-export_started
    log('artifact_written',{'sha256':artifact_hash,'bytes':len(content),'format':detection.FORMAT})
    evaluation_started=time.monotonic()
    # Validation is reported separately; it does not select configuration/weights.
    validation=_evaluate(model,loaded['session'],splits['validation'],evaluation_started,spec.budgets.evaluation_seconds)
    test=_evaluate(model,loaded['session'],splits['test'],evaluation_started,spec.budgets.evaluation_seconds)
    evaluation=dict(test['onnx'])
    evaluation.update({'contract':'held-out-coco-style-bbox/v1','artifact_sha256':artifact_hash,
        'dataset_manifest_sha256':detection_data.MANIFEST_SHA256,
        'dataset_sha256':digest(manifest_data['splits']['test']),
        'evaluation_sha256':digest(manifest_data['splits']['test']),
        'validation':validation,'test_pytorch':test['pytorch'],'test_parity':test['parity'],
        'scope':'Fixed reviewed Open Images Cat subset; not official Open Images benchmark',
        'score_threshold':detection.EVALUATION_SCORE_THRESHOLD,'nms_iou_threshold':detection.NMS_IOU_THRESHOLD,
        'execution_environment':execution_environment})
    log('evaluated',{'map_50':evaluation['map_50'],'map_50_95':evaluation['map_50_95']})
    manifest={'recipe':detection.RECIPE,'contract':detection.CONTRACT,'format':detection.FORMAT,
        'artifact_bytes':len(content),'dimensions':detection.INPUT_SIZE*detection.INPUT_SIZE*3,'training_seconds':training_seconds,
        'export_seconds':export_seconds,'evaluation_seconds':time.monotonic()-evaluation_started,
        'dataset':{'source':body['dataset']['source'],'version':body['dataset']['version'],
            'sha256':detection_data.MANIFEST_SHA256,'train_sha256':digest(manifest_data['splits']['train']),
            'validation_sha256':digest(manifest_data['splits']['validation']),
            'evaluation_sha256':digest(manifest_data['splits']['test']),
            'train_count':128,'validation_count':32,'evaluation_count':32},
        'config':dict(detection.CONFIG),'environment':submission_environment,'execution_environment':execution_environment,
        'target_profile':'local-cpu-onnx/v1','class_names':detection.CLASS_NAMES,'preprocessing':detection.metadata(),
        'parameter_count':sum(p.numel() for p in model.parameters()),'initial_parameters_sha256':initial_hash,
        'final_parameters_sha256':final_hash,'training_loss_history':loss_history,
        'dependencies':{n:importlib.metadata.version(n) for n in ('torch','numpy','onnx','onnxruntime','Pillow','pycocotools')},
        'upstream':{'repository':'https://github.com/Megvii-BaseDetection/YOLOX','commit':'6ddff4824372906469a7fae2dc3206c7aa4bbaee','code_license':'Apache-2.0','pretrained_weights':False},
        'capabilities':{'train':True,'evaluate':True,'serve_cpu':True,'device':False}}
    return artifact_hash,evaluation,manifest
