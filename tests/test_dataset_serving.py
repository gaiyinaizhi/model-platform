"""Synthetic ONNX lifecycle tests; no trained-model quality claim.

A constant graph exercises real ORT, HTTP and child lifecycle. Real trained
artifact/browser acceptance is performed separately and recorded as such.
"""
import copy
import hashlib
import importlib.util
import os
import signal
import time
import unittest

from model_platform import dataset_detection as recipe, detection, dataset_serving
from model_platform.common import digest, training_request
from model_platform.datasets import DatasetStore
import test_platform
import test_dataset_training

OPTIONAL = all(importlib.util.find_spec(n) for n in ('onnx','onnxruntime','numpy','PIL','yaml','pycocotools'))


@unittest.skipUnless(OPTIONAL, 'Optional ONNX/dataset dependencies absent')
class DatasetServingTests(unittest.TestCase):
    spawn = test_platform.PlatformTests.spawn
    start_api = test_platform.PlatformTests.start_api
    cleanup = test_platform.PlatformTests.cleanup
    call = test_platform.PlatformTests.call
    ready = test_platform.PlatformTests.ready
    publish = test_dataset_training.BindingTests.publish
    synthetic_result = test_dataset_training.BindingTests.synthetic_result

    def setUp(self):
        test_platform.PlatformTests.setUp(self)
        self.datasets = DatasetStore(self.runtime)
        self.sequence = 0
        version = self.publish()
        self.store.submit('demo','training-fixture',training_request(recipe.request_body(version['id'])))
        run, attempt = self.store.claim()
        binding = self.store.dataset_binding(run)
        import numpy as np
        import onnx
        values = np.zeros((1,525,7),dtype=np.float32)
        # One real postprocessed class-1 box; this is deliberately a constant
        # test graph, not a learned detector or evidence of recognition quality.
        values[0,0] = [80,80,40,40,1,.1,.9]
        graph = onnx.helper.make_graph([
            onnx.helper.make_node('Constant',[],['detections'],value=onnx.numpy_helper.from_array(values))],
            'synthetic-serving-test',[onnx.helper.make_tensor_value_info('images',onnx.TensorProto.FLOAT,[1,3,160,160])],
            [onnx.helper.make_tensor_value_info('detections',onnx.TensorProto.FLOAT,[1,525,7])])
        proto = onnx.helper.make_model(graph,opset_imports=[onnx.helper.make_opsetid('',17)],ir_version=9)
        onnx.helper.set_model_props(proto,recipe.metadata(binding['class_names'],binding))
        content=proto.SerializeToString();self.path=self.runtime.artifact(attempt)
        self.path.parent.mkdir(parents=True,exist_ok=True);self.path.write_bytes(content)
        self.artifact_hash=hashlib.sha256(content).hexdigest()
        evaluation,manifest=self.synthetic_result(run,binding)
        evaluation.update(artifact_sha256=self.artifact_hash,
            max_detections={'upstream':100,'scope':'global-per-image-class-aware-nms'},
            preprocessing={'version':detection.metadata()['preprocessing'],'input_size':160,
                'orientation':'raw-decoded-no-exif-transpose','epsilon_box_clamp':1e-9})
        from model_platform.detection_evaluation import evaluate_detections
        actual=evaluate_detections([{'id':1,'width':160,'height':160}],[],[],
                        [{'id':0,'name':'red_square'},{'id':1,'name':'blue_square'}])
        for field in ('protocol','evaluator','parameters'):evaluation[field]=actual[field]
        manifest['dependencies']={'pycocotools':actual['evaluator']['version']}
        evaluation['compatibility_key']=digest(recipe.compatibility(evaluation))
        manifest.update(contract=recipe.CONTRACT,format=recipe.FORMAT,dimensions=160*160*3,
                        preprocessing=recipe.metadata(binding['class_names'],binding))
        self.model_id=self.store.register(run,attempt,self.artifact_hash,evaluation,manifest)
        self.manifest=self.store.model('demo',self.model_id)

    def deploy(self,key='deploy'):
        status,result=self.call('/deployments',{'model_version_id':self.model_id},key)
        self.assertEqual(status,202,result)
        return result['id']

    def test_identity_rejects_each_inconsistent_link(self):
        loaded=recipe.load_model(self.path,self.artifact_hash)
        dataset_serving.validate_identity(self.manifest,loaded)
        paths=[('class_names',),('dataset','classes'),('dataset','class_names'),
               ('dataset','evaluation_sha256'),('preprocessing',),('evaluation','artifact_sha256'),
               ('evaluation','category_mapping'),('evaluation','score_threshold'),('evaluation','nms_iou_threshold'),
               ('evaluation','max_detections'),('evaluation','preprocessing'),('evaluation','role'),
               ('evaluation','compatibility_key'),('evaluation','contract'),('evaluation','protocol'),
               ('evaluation','evaluator'),('evaluation','parameters')]
        for path in paths:
            bad=copy.deepcopy(self.manifest);node=bad
            for key in path[:-1]:node=node[key]
            node[path[-1]]='tampered'
            with self.subTest(path=path),self.assertRaises((ValueError,TypeError,KeyError)):
                dataset_serving.validate_identity(bad,loaded)

    def test_actual_onnx_lifecycle_idempotency_ownership_restart_and_stop(self):
        did=self.deploy();ready=self.ready(did)
        for key,value in dataset_serving.public_identity(self.manifest).items():self.assertEqual(ready[key],value)
        self.assertFalse(ready['quality']['production_allowed'])
        status,replay=self.call('/deployments',{'model_version_id':self.model_id},'deploy')
        self.assertEqual(status,200);self.assertEqual(replay['id'],did)
        conflict=self.call('/deployments',{'model_version_id':'model_'+'f'*32},'deploy')
        self.assertEqual(conflict[0],409);self.assertEqual(conflict[1]['error']['code'],'idempotency_conflict')
        self.assertEqual(self.call('/deployments',{'model_version_id':self.model_id},'other',tenant='other')[0],404)
        for suffix,body in [('',None),('/stop',{}),('/predict',{'features':detection.probe_input()})]:
            self.assertEqual(self.call('/deployments/'+did+suffix,body,tenant='other')[0],404)
        status,pred=self.call('/deployments/'+did+'/predict',{'features':detection.probe_input()})
        self.assertEqual(status,200,pred);self.assertEqual(pred['predictions'][0]['detections'][0]['class_id'],1)
        self.assertEqual(pred['class_names'],['red_square','blue_square'])
        self.assertEqual(pred['artifact_sha256'],self.artifact_hash)
        self.assertEqual(self.call('/deployments/'+did+'/predict',{'features':[]})[0],400)
        self.api.terminate();self.api.wait(timeout=10);self.start_api()
        restored=self.ready(did);self.assertEqual(restored['model_version_id'],self.model_id)
        self.assertEqual(self.call('/deployments/'+did+'/predict',{'features':detection.probe_input()})[0],200)
        stopped=self.call('/deployments/'+did+'/stop',{})
        self.assertEqual(stopped[0],200);self.assertFalse(stopped[1]['ready'])
        self.assertEqual(stopped[1]['observed'],'stopped')
        self.assertEqual(self.call('/deployments/'+did+'/predict',{'features':detection.probe_input()})[0],503)
        self.assertEqual(self.call('/deployments',{'model_version_id':self.model_id},'deploy')[1]['desired'],'stopped')
        new=self.deploy('new-explicit-service');self.assertNotEqual(new,did);self.ready(new)
        self.assertEqual(self.call('/deployments/'+new+'/stop',{})[0],200)

    def test_child_death_and_corruption_never_ready(self):
        did=self.deploy();ready=self.ready(did)
        os.kill(ready['pid'],signal.SIGKILL)
        deadline=time.monotonic()+10
        while time.monotonic()<deadline:
            state=self.call('/deployments/'+did)[1]
            if state['observed']=='failed':break
            time.sleep(.05)
        self.assertEqual(state['observed'],'failed');self.assertFalse(state['ready'])
        self.assertEqual(self.call('/deployments/'+did+'/predict',{'features':detection.probe_input()})[0],503)
        self.call('/deployments/'+did+'/stop',{})
        self.path.write_bytes(b'corrupt')
        bad=self.deploy('corrupt-artifact')
        deadline=time.monotonic()+10
        while time.monotonic()<deadline:
            state=self.call('/deployments/'+bad)[1]
            if state['observed']=='failed':break
            time.sleep(.05)
        self.assertEqual(state['observed'],'failed');self.assertFalse(state['ready'])
        self.assertIn(state['error_code'],('serving_exited','startup_failed'))
        self.assertEqual(self.call('/deployments/'+bad+'/predict',{'features':detection.probe_input()})[0],503)
        self.api.terminate();self.api.wait(timeout=10);self.start_api()
        deadline=time.monotonic()+10
        while time.monotonic()<deadline:
            recovered=self.call('/deployments/'+bad)[1]
            if recovered['observed']=='failed':break
            time.sleep(.05)
        self.assertFalse(recovered['ready']);self.assertEqual(recovered['observed'],'failed')
        self.assertEqual(recovered['model_version_id'],self.model_id)
        self.assertEqual(self.call('/deployments/'+bad+'/predict',{'features':detection.probe_input()})[0],503)
        self.call('/deployments/'+bad+'/stop',{})
