"""Transport/security fixtures use opaque bytes, not claimed ONNX accuracy tests."""
from contextlib import ExitStack
import hashlib
import http.client
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from urllib.parse import urlsplit

from model_platform.common import Runtime, Problem
from model_platform.delivery import Delivery, delivery_manifest
from model_platform.store import Store
from model_platform import detection
from scripts.demo import request_body
import test_platform


def fixture(store, content=b'opaque transport fixture, not a real ONNX model'):
    store.submit('demo', 'transport', request_body())
    run, attempt = store.claim()
    digest = hashlib.sha256(content).hexdigest()
    store.runtime.artifact(attempt).write_bytes(content)
    manifest = {'recipe': detection.RECIPE, 'format': detection.FORMAT,
                'artifact_bytes': len(content), 'target_profile': 'local-cpu-onnx/v1',
                'class_names': ['Cat'], 'dependencies': {}}
    model = store.register(run, attempt, digest, {'dataset_sha256': '0'*64}, manifest)
    return model, attempt, content


class DeliveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = Store(Runtime(Path(self.temp.name)))
        self.delivery = Delivery(self.store)
        self.model, self.attempt, self.content = fixture(self.store)

    def test_verified_snapshot_manifest_and_original_mutation(self):
        with self.delivery.snapshot('demo', self.model) as (model, stream):
            self.store.runtime.artifact(self.attempt).write_bytes(b'changed')
            self.assertEqual(stream.read(), self.content)
            self.assertEqual(hashlib.sha256(self.content).hexdigest(), model['artifact_sha256'])
        self.assertTrue(stream.closed)
        with self.assertRaises(Problem) as err:
            self.delivery.manifest('demo', self.model)
        self.assertEqual(err.exception.code, 'artifact_integrity_failed')

    def test_manifest_describes_graph_and_quality(self):
        result = self.delivery.manifest('demo', self.model)
        self.assertEqual(result['schema_version'], 'model-delivery/v1')
        self.assertEqual(result['artifact']['bytes'], len(self.content))
        self.assertEqual(result['artifact']['download_url'], '/models/'+self.model+'/artifact')
        self.assertEqual(result['input']['shape'], [1,3,160,160])
        self.assertEqual(result['output']['shape'], [1,525,6])
        self.assertFalse(result['quality']['production_allowed'])
        self.assertEqual(result['quality']['status'], 'quality-failed')
        self.assertFalse(result['runtime']['device_certified'])
        self.assertNotIn(str(self.store.runtime.home), json.dumps(result))
        model = self.store.model('demo', self.model)
        model.update(recipe='yolox-nano-dataset/v1', class_names=['红,色','Blue'])
        result = delivery_manifest(model)
        self.assertEqual(result['classes'][0], {'id':0,'name':'红,色'})
        self.assertEqual(result['output']['shape'], [1,525,7])
        model.update(recipe='fashion-cnn/v1',format='fashion-onnx/v1')
        result = delivery_manifest(model)
        self.assertEqual(result['artifact']['onnx_opset'],18)
        self.assertEqual(len(result['classes']),10)

    def test_owned_lookup_precedes_format_and_capacity(self):
        with ExitStack() as stack:
            for _ in range(2): stack.enter_context(self.delivery.snapshot('demo', self.model))
            with self.assertRaises(Problem) as error:
                with self.delivery.snapshot('other', self.model): pass
            self.assertEqual(error.exception.status,404)
            with self.assertRaises(Problem) as error:
                with self.delivery.snapshot('demo', self.model): pass
            self.assertEqual(error.exception.code,'download_capacity')
        with self.delivery.snapshot('demo', self.model): pass

    def test_failure_releases_capacity_and_closes_snapshot(self):
        original = tempfile.TemporaryFile
        files=[]
        def opened(*a,**k):
            stream=original(*a,**k); files.append(stream); return stream
        path=self.store.runtime.artifact(self.attempt)
        path.write_bytes(b'X'*len(self.content))
        with mock.patch('model_platform.delivery.tempfile.TemporaryFile',side_effect=opened):
            for _ in range(3):
                with self.assertRaises(Problem):
                    with self.delivery.snapshot('demo',self.model): pass
        self.assertTrue(all(s.closed for s in files))
        path.write_bytes(self.content)
        with self.delivery.snapshot('demo',self.model): pass

    def test_missing_symlink_directory_symlink_and_oversized(self):
        path=self.store.runtime.artifact(self.attempt)
        path.unlink()
        with self.assertRaises(Problem) as e: self.delivery.manifest('demo',self.model)
        self.assertEqual(e.exception.code,'artifact_unavailable')
        outside=Path(self.temp.name)/'outside';outside.write_bytes(self.content)
        path.symlink_to(outside)
        with self.assertRaises(Problem) as e:self.delivery.manifest('demo',self.model)
        self.assertEqual(e.exception.code,'artifact_integrity_failed')
        path.unlink();path.write_bytes(self.content+b'oversized')
        with self.assertRaises(Problem):self.delivery.manifest('demo',self.model)
        path.write_bytes(self.content)
        directory=self.store.runtime.artifacts
        real=directory.with_name('actual-artifacts');directory.rename(real);directory.symlink_to(real,target_is_directory=True)
        with self.assertRaises(Problem):self.delivery.manifest('demo',self.model)

    def test_bad_registered_size_and_unsupported_format(self):
        original=self.store.model('demo',self.model)
        for key,value in [('artifact_bytes',True),('artifact_bytes',9*1024*1024),('artifact_id','../oops'),('artifact_sha256','invalid')]:
            with mock.patch.object(self.store,'model',return_value=dict(original,**{key:value})):
                with self.assertRaises(Problem):self.delivery.manifest('demo',self.model)
        with mock.patch.object(self.store,'model',return_value=dict(original,recipe='logistic-regression/v1')):
            with self.assertRaises(Problem) as e:self.delivery.manifest('demo',self.model)
            self.assertEqual(e.exception.code,'delivery_format_unsupported')


class DeliveryHttpTests(unittest.TestCase):
    for _name in ('setUp','spawn','start_api','cleanup','call'):
        locals()[_name]=getattr(test_platform.PlatformTests,_name)

    def binary(self,path,tenant='demo',extra=None):
        u=urlsplit(self.base);conn=http.client.HTTPConnection(u.hostname,u.port,timeout=5)
        try:
            headers={'Authorization':'Bearer '+self.tokens[tenant]} if tenant else {}
            headers.update(extra or {})
            conn.request('GET',path,headers=headers)
            response=conn.getresponse();return response.status,dict(response.getheaders()),response.read()
        finally:conn.close()

    def test_binary_roundtrip_auth_headers_and_corruption(self):
        model,attempt,content=fixture(self.store)
        path='/models/'+model+'/artifact'
        for tenant,status in [(None,401),('other',404)]:self.assertEqual(self.binary(path,tenant)[0],status)
        status,headers,body=self.binary(path)
        self.assertEqual((status,body),(200,content))
        self.assertEqual(headers['Content-Type'],'application/octet-stream')
        self.assertEqual(headers['X-Artifact-SHA256'],hashlib.sha256(body).hexdigest())
        self.assertEqual(int(headers['Content-Length']),len(body))
        self.assertEqual(headers['Cache-Control'],'no-store')
        self.assertEqual(headers['Content-Disposition'],'attachment; filename="'+model+'.onnx"')
        status,result=self.call('/models/'+model+'/delivery')
        self.assertEqual(status,200)
        self.assertEqual(result['artifact']['sha256'],headers['X-Artifact-SHA256'])
        self.assertEqual(self.binary(path,extra={'Range':'bytes=0-1'})[0],400)
        self.assertEqual(self.binary(path+'?path=anything')[0],400)
        self.store.runtime.artifact(attempt).write_bytes(b'corrupt')
        status,headers,body=self.binary(path)
        self.assertEqual(status,503)
        self.assertEqual(headers['Content-Type'],'application/json')
        self.assertNotIn(str(self.home).encode(),body)
        self.assertEqual(self.call('/worker-readiness',tenant=None)[0],401)
        self.assertEqual(self.call('/worker-readiness')[0],200)
