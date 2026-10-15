"""Real Cat detection run/service demo; start independent API and Worker first."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model_platform import detection, detection_data
from model_platform.common import Runtime, encode
from scripts.demo_fashion import NoRedirects


def main():
    runtime=Runtime()
    base=os.environ.get('MP_API','http://127.0.0.1:8080').rstrip('/')
    url=urllib.parse.urlsplit(base)
    if (url.scheme!='http' or url.hostname not in ('127.0.0.1','localhost','::1')
            or url.username or url.password or url.path or url.query or url.fragment):
        raise ValueError('MP_API must be a plain loopback HTTP origin')
    manifest,splits=detection_data.load_dataset(runtime)
    # Fixed first three validation images; never choose good-looking predictions.
    samples=[]
    for item in splits['validation'][:3]:
        from PIL import Image
        image=item['image'].copy();image.thumbnail((384,384),Image.Resampling.BILINEAR)
        samples.append({'image_id':item['record']['image_id'],
                        'features':[{'width':image.width,'height':image.height,
                                     'pixels':list(image.tobytes())}]})
    token=json.loads((runtime.home/'credentials.json').read_text())['demo']
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}),NoRedirects())
    def call(path,payload=None,key=None):
        headers={'Authorization':'Bearer '+token}
        if payload is not None:headers['Content-Type']='application/json'
        if key:headers['Idempotency-Key']=key
        request=urllib.request.Request(base+path,data=None if payload is None else encode(payload),headers=headers)
        with opener.open(request,timeout=20) as response:return json.load(response)
    body=detection.request_body();key='detection-'+uuid.uuid4().hex
    run=call('/runs',body,key)
    assert call('/runs',body,key)['id']==run['id']
    deadline=time.monotonic()+1000
    while run['status'] in ('queued','running'):
        if time.monotonic()>deadline:raise RuntimeError('Detection training timed out')
        time.sleep(2);run=call('/runs/'+run['id'])
    if run['status']!='succeeded':
        raise RuntimeError('Detection training failed: '+repr([a['error_code'] for a in run['attempts']]))
    model=call('/models/'+run['model_version_id'])
    assert model['contract']==detection.CONTRACT and model['config']==detection.CONFIG
    assert model['dataset']['train_count']==128 and model['dataset']['evaluation_count']==32
    assert model['initial_parameters_sha256']!=model['final_parameters_sha256']
    assert hashlib.sha256(runtime.artifact(model['artifact_id']).read_bytes()).hexdigest()==model['artifact_sha256']
    assert model['evaluation']['test_parity']['matching_detection_counts']==32
    deployment=call('/deployments',{'model_version_id':model['model_version_id']})
    route='/deployments/'+deployment['id'];outputs=[]
    try:
        deadline=time.monotonic()+30
        while not deployment['ready']:
            if deployment['observed']=='failed' or time.monotonic()>deadline:raise RuntimeError('Detection service not ready')
            time.sleep(1);deployment=call(route)
        for sample in samples:
            before=time.monotonic();first=call(route+'/predict',{'features':sample['features']})
            assert first==call(route+'/predict',{'features':sample['features']})
            assert first['model_version_id']==model['model_version_id'] and first['artifact_sha256']==model['artifact_sha256']
            outputs.append({'image_id':sample['image_id'],'result':first,'two_requests_seconds':time.monotonic()-before})
        active=call(route)
        assert active['ready'] and active['pid']==deployment['pid']
    finally:
        stopped=call(route+'/stop',{})
        assert stopped['observed']=='stopped' and stopped['desired']=='stopped' and not stopped['ready']
    try:call(route+'/predict',{'features':samples[0]['features']})
    except urllib.error.HTTPError as exc:
        assert exc.code==503;exc.close()
    else:raise AssertionError('Stopped service accepted prediction')
    report={'case':'real-yolox-nano-cat-cpu/v1','run_id':run['id'],'model_version_id':model['model_version_id'],
            'deployment_id':deployment['id'],'registered_manifest':model,'attempts':run['attempts'],
            'sample_policy':'First three validation images before predictions; resized to HTTP budget',
            'sample_predictions':outputs,'verification':{'idempotency':True,'parameters_changed':True,
            'persistent_requests':6,'stable_predictions':True,'stop_observed':'stopped','post_stop_http_status':503}}
    path=runtime.home/('detection-'+run['id']+'-report.json')
    with path.open('xb') as stream:stream.write(encode(report)+b'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
