#!/usr/bin/env python3
"""Single predeclared real training run and CPU ONNX/service consistency check.

Uses own synthetic calibration only. Leaves deployment ready for browser QA;
operator must stop it afterwards. Never chooses weights or thresholds by results.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--api', default='http://127.0.0.1:8787', help='Existing loopback BFF origin; no credential extraction')
    parser.add_argument('--origin', default='http://127.0.0.1:5666')
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    url = urllib.parse.urlsplit(args.api)
    if url.scheme != 'http' or url.hostname != '127.0.0.1' or url.path or url.query or url.fragment or url.username:
        parser.error('API must be plain http://127.0.0.1:port')
    args.output.mkdir(parents=True, exist_ok=False)
    home = Path(os.environ.get('MP_HOME', '.runtime'))
    origin = urllib.parse.urlsplit(args.origin)
    if origin.scheme != 'http' or origin.hostname != '127.0.0.1' or origin.path or origin.query or origin.fragment or origin.username:
        parser.error('Origin must be plain loopback frontend origin')
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def call(path, body=None, raw=None, key=None):
        data = raw if raw is not None else json.dumps(body).encode() if body is not None else None
        headers = {'Origin': args.origin}
        if path.split('/')[1] in ('runs', 'models', 'deployments'):
            path = '/experiments'+path
        path = '/api/real-detection'+path
        if data is not None:
            headers['Content-Type'] = 'application/zip' if raw is not None else 'application/json'
        if key:
            headers['Idempotency-Key'] = key
        with opener.open(urllib.request.Request(args.api+path, data=data, headers=headers), timeout=60) as r:
            return json.load(r)
    def save(name, value):
        (args.output/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    manifest = json.loads((args.fixture/'manifest.json').read_text())
    raw = (args.fixture/'learnability.zip').read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest['archive_sha256']:
        raise ValueError('Fixture checksum mismatch')
    from scripts.make_learnability_fixture import CONFIG
    if manifest['predeclared_training_config'] != CONFIG or manifest['predeclared_thresholds'] != {'evaluation': .001, 'serving': .01}:
        raise ValueError('Predeclared settings changed')
    save('predeclared.json', {'fixture_sha256':manifest['archive_sha256'], 'config':CONFIG, 'thresholds':manifest['predeclared_thresholds']})
    key = 'learnability-'+manifest['archive_sha256']
    item = call('/dataset-imports', {'name':'Own synthetic learnability calibration; not photo accuracy', 'rights_confirmed':True, 'negative_policy':'reject'}, key=key)
    save('import.json', item)
    report = call('/dataset-imports/'+item['id']+'/archive', raw=raw)
    if not report['report']['valid']:
        raise ValueError('Fixture rejected')
    version = call('/dataset-imports/'+item['id']+'/publish', {})
    save('version.json', version)
    body = {'recipe':'yolox-nano-dataset/v1', 'dataset':{'dataset_version_id':version['id']}, 'config':CONFIG}
    run = call('/runs', body, key=key+'-run')
    save('submitted-run.json', run)
    deadline = time.monotonic()+1000
    while run['status'] in ('queued','running'):
        if time.monotonic()>deadline:
            raise RuntimeError('Training observation timeout; inspect existing run, never resubmit')
        time.sleep(2); run = call('/runs/'+run['id'])
        print(json.dumps({'run_id':run['id'], 'status':run['status']}), flush=True)
    save('run.json', run)
    if run['status'] != 'succeeded':
        raise RuntimeError('Predeclared run failed; see run.json, no retraining')
    model = call('/models/'+run['model_version_id']); save('model.json', model)
    deployment = call('/deployments', {'model_version_id':model['model_version_id']}, key=key+'-deployment')
    save('deployment.json', deployment)
    deadline = time.monotonic()+45
    while not deployment['ready']:
        if time.monotonic()>deadline:
            raise RuntimeError('Deployment readiness timeout')
        time.sleep(.5); deployment=call('/deployments/'+deployment['id'])
    save('deployment.json', deployment)
    from PIL import Image
    from model_platform import dataset_detection, detection
    from model_platform.detection_geometry import postprocess_predictions
    from model_platform.detection_evaluation import evaluate_detections
    loaded = dataset_detection.load_model(home/'artifacts'/(model['attempt_id']+'.json'), model['artifact_sha256'])
    images, annotations, predictions, samples = [], [], [], []
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        for record in (r for r in manifest['samples'] if r['split']=='test'):
            image = Image.open(io.BytesIO(archive.read(record['image']))).convert('RGB')
            w,h=image.size
            features=[{'width':w, 'height':h, 'pixels':list(image.tobytes())}]
            expected = detection.predict(loaded, features)
            actual = call('/deployments/'+deployment['id']+'/predict', {'features':features})
            if actual['predictions'] != expected or actual['artifact_sha256'] != model['artifact_sha256']:
                raise ValueError('Online/offline predictions or artifact identity differ')
            batch, transform = detection.preprocess(image)
            tensor = loaded['session'].run(['detections'], {'images':batch[None]})[0]
            boxes = postprocess_predictions(tensor[0].tolist(), transform, num_classes=2, score_threshold=.001, iou_threshold=detection.NMS_IOU_THRESHOLD, max_detections=100)
            image_id = record['image']
            images.append({'id':image_id,'width':w,'height':h})
            annotations.extend({'image_id':image_id,'category_id':b['class_id'],'bbox':b['xyxy']} for b in record['boxes'])
            predictions.extend({'image_id':image_id,'category_id':b['class_id'],'bbox':b['box'],'score':b['score']} for b in boxes)
            samples.append({'image':image_id,'width':w,'height':h,'ground_truth':record['boxes'],'inference':actual})
    metrics=evaluate_detections(images,annotations,predictions,[{'id':0,'name':'red_rectangle'},{'id':1,'name':'blue_rectangle'}])
    for metric in ('map_50','map_50_95'):
        if abs(metrics[metric]-model['evaluation'][metric])>1e-6:
            raise ValueError('Recomputed AP differs from registered ONNX evaluation')
    for a,b in zip(metrics['per_class'],model['evaluation']['per_class']):
        for metric in ('ap_50','ap_50_95'):
            if abs(a[metric]-b[metric])>1e-6:
                raise ValueError('Recomputed class AP differs')
    evidence={'scope':'Synthetic learnability/UI mechanics only; no real-photo quality claim', 'model_version_id':model['model_version_id'],
              'artifact_sha256':model['artifact_sha256'], 'deployment_id':deployment['id'], 'samples':samples,
              'recomputed_metrics':metrics,'online_offline_exact_equal':True,
              'nonempty_images':sum(bool(s['inference']['predictions'][0]['detections']) for s in samples),
              'note':'Deployment left ready only for browser QA; stop afterwards'}
    evidence['nonempty_validation_status'] = 'passed' if evidence['nonempty_images'] else 'not-achieved'
    save('acceptance.json',evidence)
    print(json.dumps({'evidence':str(args.output/'acceptance.json'),'nonempty_images':evidence['nonempty_images'],
                      'map_50':metrics['map_50'],'map_50_95':metrics['map_50_95']}),flush=True)
    if not evidence['nonempty_images']:
        raise SystemExit('Nonempty-result target not achieved; evidence retained, no tuning or retry')


if __name__=='__main__':
    main()
