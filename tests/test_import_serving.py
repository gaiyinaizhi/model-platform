"""Opt-in real uploaded Nano lifecycle using disposable loopback test identities."""
import importlib.util,json,os,time,unittest,urllib.request,urllib.error
from pathlib import Path
import test_platform
from model_platform import pretrained_yolox as nano
ASSET=os.environ.get('MP_PRETRAINED_YOLOX_ASSET')
OPTIONAL=all(importlib.util.find_spec(x) for x in ('onnx','onnxruntime','numpy','cv2'))

@unittest.skipUnless(ASSET and OPTIONAL,'Authorized real Nano asset and optional dependencies required')
class ImportedServingTests(unittest.TestCase):
    spawn=test_platform.PlatformTests.spawn
    start_api=test_platform.PlatformTests.start_api
    cleanup=test_platform.PlatformTests.cleanup
    call=test_platform.PlatformTests.call
    ready=test_platform.PlatformTests.ready
    def setUp(self):
        test_platform.PlatformTests.setUp(self)
        self.raw=Path(ASSET).read_bytes()
        status,self.item=self.call('/model-imports',{'catalog_id':nano.PROTOCOL,'rights_confirmed':True},'import')
        self.assertEqual(status,201,self.item)
        status,r=self.upload(self.item['id']);self.assertEqual(status,200,r);self.assertEqual(r['status'],'validated',r)
        status,self.model=self.call('/model-imports/'+self.item['id']+'/register',{})
        self.assertEqual(status,201,self.model);self.mid=self.model['model_version_id']
    def upload(self,item_id,tenant='demo'):
        req=urllib.request.Request(self.base+'/model-imports/'+item_id+'/artifact',data=self.raw,
            headers={'Content-Type':'application/octet-stream','Authorization':'Bearer '+self.tokens[tenant]})
        try:
            with self.opener.open(req,timeout=40) as r:return r.status,json.load(r)
        except urllib.error.HTTPError as e:
            with e:return e.code,json.load(e)
    def deploy(self):
        status,r=self.call('/deployments',{'model_version_id':self.mid},'deploy')
        self.assertEqual(status,202,r);return r['id']
    def test_lifecycle_restart_stop_replay(self):
        self.assertEqual(self.model['origin'],'imported');self.assertIsNone(self.model['evaluation'])
        self.assertNotIn('run_id',self.model);self.assertFalse(self.model['production_allowed'])
        self.assertEqual(self.call('/runs')[1]['ids'],[])
        did=self.deploy();state=self.ready(did)
        self.assertEqual(state['artifact_sha256'],nano.ASSET_SHA256)
        status,r=self.call('/deployments/'+did+'/predict',{'features':nano.probe_input()})
        self.assertEqual(status,200,r);nano.probe_output(r['predictions'])
        self.assertEqual(r['validation_id'],self.model['validation_id'])
        self.api.terminate();self.api.wait(timeout=10);self.start_api()
        self.assertEqual(self.ready(did)['model_version_id'],self.mid)
        self.assertEqual(self.call('/deployments/'+did+'/stop',{})[0],200)
        self.assertEqual(self.call('/deployments/'+did+'/predict',{'features':nano.probe_input()})[0],503)
        self.assertEqual(self.call('/deployments',{'model_version_id':self.mid},'deploy')[1]['desired'],'stopped')
        status,r=self.call('/model-imports/'+self.item['id']+'/register',{})
        self.assertEqual(status,200);self.assertEqual(r['model_version_id'],self.mid)
        self.assertEqual(self.upload(self.item['id'])[1]['status'],'validated')
    def test_ownership_no_download_or_fake_evaluation(self):
        for path in ('/models/'+self.mid,'/model-imports/'+self.item['id']):self.assertEqual(self.call(path,tenant='other')[0],404)
        self.assertEqual(self.upload(self.item['id'],'other')[0],404)
        self.assertEqual(self.call('/model-imports/'+self.item['id']+'/register',{},tenant='other')[0],404)
        self.assertEqual(self.call('/deployments',{'model_version_id':self.mid},tenant='other')[0],404)
        for suffix in ('/artifact','/delivery'):self.assertEqual(self.call('/models/'+self.mid+suffix)[0],409)
        with self.runtime.connection() as db:
            for table in ('runs','attempts','evaluations'):self.assertEqual(db.execute('SELECT count(*) FROM '+table).fetchone()[0],0)
    def test_corruption_never_ready(self):
        from model_platform.model_imports import artifact_path
        artifact_path(self.runtime,self.model['artifact_id']).write_bytes(b'corrupt')
        did=self.deploy();deadline=time.monotonic()+10
        while time.monotonic()<deadline:
            state=self.call('/deployments/'+did)[1]
            if state['observed']=='failed':break
            time.sleep(.05)
        self.assertFalse(state['ready']);self.assertEqual(state['observed'],'failed')
        self.assertEqual(self.call('/deployments/'+did+'/predict',{'features':nano.probe_input()})[0],503)
        self.assertEqual(self.call('/deployments/'+did+'/stop',{})[0],200)
