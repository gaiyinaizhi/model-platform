#!/usr/bin/env python3
"""Real CPU smoke over own synthetic data; run alongside the API and Worker.

Produces mechanics/provenance evidence only. It does not establish quality.
Reads local demo credentials without printing or sending them outside loopback.
"""
import argparse
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
from scripts.make_training_fixture import fixture_bytes
from model_platform.dataset_detection import request_body


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--api',default=os.environ.get('MP_API','http://127.0.0.1:8080'))
    parser.add_argument('--output',type=Path,default=Path('.runtime/dataset-training-evidence.json'))
    args=parser.parse_args()
    url=urllib.parse.urlsplit(args.api)
    if url.scheme!='http' or url.hostname!='127.0.0.1' or url.path or url.query or url.fragment or url.username:
        parser.error('API must be a plain http://127.0.0.1:port origin')
    home=Path(os.environ.get('MP_HOME','.runtime'))
    tokens=json.loads((home/'credentials.json').read_text())
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def call(path,body=None,key=None,raw=None,tenant='demo'):
        data=raw if raw is not None else json.dumps(body).encode() if body is not None else None
        headers={'Authorization':'Bearer '+tokens[tenant]}
        if data is not None:headers['Content-Type']='application/zip' if raw is not None else 'application/json'
        if key:headers['Idempotency-Key']=key
        req=urllib.request.Request(args.api+path,data=data,headers=headers)
        try:
            with opener.open(req,timeout=60) as response:return response.status,json.load(response)
        except urllib.error.HTTPError as error:return error.code,json.load(error)
    def success(path,body=None,key=None,raw=None):
        status,value=call(path,body,key,raw)
        if status not in (200,201,202):raise RuntimeError(f'{path}: {status} {value}')
        return value
    def publish(test):
        item=success('/dataset-imports',{'name':'Own synthetic two-class mechanics fixture','rights_confirmed':True,'negative_policy':'reject'},uuid.uuid4().hex)
        report=success('/dataset-imports/'+item['id']+'/archive',raw=fixture_bytes(test=test))
        assert report['report']['valid'],report
        return success('/dataset-imports/'+item['id']+'/publish',{})
    def train(version,epochs):
        body=request_body(version['id']);body['config']['epochs']=epochs
        key=uuid.uuid4().hex
        run=success('/runs',body,key)
        assert success('/runs',body,key)['id']==run['id']
        deadline=time.monotonic()+900
        last=None
        while run['status'] in ('queued','running'):
            if time.monotonic()>deadline:raise RuntimeError('Training exceeded smoke observation budget')
            time.sleep(2)
            run=success('/runs/'+run['id'])
            marker=(run['status'],len(run['attempts'][-1]['logs']) if run['attempts'] else 0)
            if marker!=last:
                print(json.dumps({'run_id':run['id'],'status':run['status'],'last_log':run['attempts'][-1]['logs'][-1] if run['attempts'] and run['attempts'][-1]['logs'] else None}),flush=True)
                last=marker
        assert run['status']=='succeeded',run
        model=success('/models/'+run['model_version_id'])
        assert model['dataset']['dataset_version_id']==version['id']
        assert model['initial_parameters_sha256']!=model['final_parameters_sha256']
        return run,model
    version=publish(True)
    first,m1=train(version,1);second,m2=train(version,2)
    comparison=success('/model-comparisons',{'model_version_ids':[m1['model_version_id'],m2['model_version_id']]})
    valversion=publish(False);third,m3=train(valversion,1)
    assert m1['evaluation']['role']=='held-out-test' and m3['evaluation']['role']=='validation-only'
    incompatible=call('/model-comparisons',{'model_version_ids':[m1['model_version_id'],m3['model_version_id']]})
    assert incompatible[0]==409 and incompatible[1]['error']['code']=='evaluation_incompatible',incompatible
    assert call('/runs',request_body(version['id']),uuid.uuid4().hex,tenant='other')[0]==404
    deployment=success('/deployments',{'model_version_id':m1['model_version_id']},uuid.uuid4().hex)
    deadline=time.monotonic()+45
    while not deployment['ready']:
        if time.monotonic()>deadline:raise RuntimeError('Deployment did not become ready')
        time.sleep(.5);deployment=success('/deployments/'+deployment['id'])
    features=[{'width':64,'height':64,'pixels':[35]*(64*64*3)}]
    inference=success('/deployments/'+deployment['id']+'/predict',{'features':features})
    assert inference['model_version_id']==m1['model_version_id']
    assert all(0<=box['class_id']<2 for box in inference['predictions'][0]['detections'])
    stopped=success('/deployments/'+deployment['id']+'/stop',{})
    assert not stopped['ready'] and call('/deployments/'+deployment['id']+'/predict',{'features':features})[0]==503
    evidence={'scope':'Own synthetic fixture; actual CPU training/export/ONNX evaluation/serving; not quality or production proof',
              'runs':[first,second,third],'models':[m1,m2,m3],'comparison':comparison,'incompatible':incompatible,
              'inference':inference,'stopped':stopped,'ownership_check':'other tenant submission returned 404'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:json.dump(evidence,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'evidence':str(args.output),'models':[m['model_version_id'] for m in (m1,m2,m3)],
                      'metrics':[{'split':m['evaluation']['split'],'map_50':m['evaluation']['map_50'],'map_50_95':m['evaluation']['map_50_95']} for m in (m1,m2,m3)],'quality_proof':False}),flush=True)


if __name__=='__main__':main()
