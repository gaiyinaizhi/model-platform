"""Contract/provenance tests use synthetic records, not real quality evidence."""
import copy
import importlib.util
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest import mock

from model_platform import dataset_detection as recipe, detection, worker
from model_platform.common import Problem, Runtime, digest, environment, training_request
from model_platform.datasets import DatasetStore
from model_platform.store import Store
from scripts.make_training_fixture import fixture_bytes
import test_datasets
import test_platform

OPTIONAL = test_datasets.OPTIONAL


class RequestTests(unittest.TestCase):
    def test_minimal_bounded_config_and_no_caller_lineage(self):
        body = recipe.request_body('dsv_'+'1'*32)
        self.assertIs(training_request(body), body)
        for key, value in [('epochs',0),('epochs',21),('epochs',True),('seed',-1),('seed',True),('learning_rate',.02),('learning_rate',float('nan'))]:
            bad=copy.deepcopy(body);bad['config'][key]=value
            with self.subTest(key=key,value=value), self.assertRaises(Problem):training_request(bad)
        for part,key,value in [('dataset','tenant','other'),('dataset','source','url'),('config','code','exec'),('config','input_size',320)]:
            bad=copy.deepcopy(body);bad[part][key]=value
            with self.assertRaises(Problem):training_request(bad)

    def test_model_metadata_is_class_and_dataset_bound(self):
        binding={'dataset_version_id':'dsv_'+'1'*32,'sha256':'a'*64,'train_sha256':'b'*64,'evaluation_sha256':'c'*64,'evaluation_split':'val'}
        names=['红色, square','blue square']
        props=recipe.metadata(names,binding);model=recipe._metadata_identity(props)
        self.assertEqual(model['class_names'],names)
        for key,value in [('class_ids','[1,0]'),('recipe',detection.RECIPE),('evaluation_split','train'),('dataset_sha256','0'),('class_names','["same","same"]')]:
            bad=dict(props,**{key:value})
            with self.assertRaises(ValueError):recipe._metadata_identity(bad)


@unittest.skipUnless(OPTIONAL, 'Optional dataset dependencies absent')
class BindingTests(unittest.TestCase):
    def setUp(self):
        parent=Path(test_platform.ROOT)/'.runtime/tests';parent.mkdir(parents=True,exist_ok=True)
        self.tmp=tempfile.TemporaryDirectory(dir=parent);self.addCleanup(self.tmp.cleanup)
        self.runtime=Runtime(self.tmp.name);self.store=Store(self.runtime);self.datasets=DatasetStore(self.runtime)
        self.sequence=0

    def publish(self,raw=None,tenant='demo'):
        self.sequence+=1
        item,_=self.datasets.create(tenant,str(self.sequence),dict(test_datasets.META,name='Synthetic mechanics'))
        result=self.datasets.upload(tenant,item['id'],lambda:fixture_bytes() if raw is None else raw)
        self.assertTrue(result['report']['valid'],result)
        return self.datasets.publish(tenant,item['id'])[0]

    def submit(self,version,key='run'):
        body=recipe.request_body(version['id'])
        rid,_=self.store.submit('demo',key,training_request(body))
        return rid,body

    def test_immutable_binding_idempotency_ownership_and_roles(self):
        version=self.publish();rid,body=self.submit(version)
        self.assertEqual(self.store.submit('demo','run',body),(rid,False))
        run=self.store.run('demo',rid)
        self.assertEqual(run['request_sha256'],digest(body))
        self.assertEqual(run['dataset']['evaluation_split'],'test')
        self.assertEqual(run['dataset']['classes'],[{'id':0,'name':'red_square'},{'id':1,'name':'blue_square'}])
        self.assertEqual(run['dataset']['train_count'],8)
        with self.assertRaises(Problem) as e:self.store.submit('other','steal',body)
        self.assertEqual(e.exception.status,404)
        changed=copy.deepcopy(body);changed['config']['epochs']+=1
        with self.assertRaises(Problem) as e:self.store.submit('demo','run',changed)
        self.assertEqual(e.exception.code,'idempotency_conflict')
        with self.runtime.connection() as db:
            with self.assertRaises(sqlite3.IntegrityError):db.execute('UPDATE run_dataset_bindings SET content_sha256=? WHERE run_id=?',('x',rid))
            with self.assertRaises(sqlite3.IntegrityError):db.execute('DELETE FROM run_dataset_bindings WHERE run_id=?',(rid,))
        val=self.publish(fixture_bytes(test=False));vid,_=self.submit(val,'val')
        self.assertEqual(self.store.run('demo',vid)['dataset']['evaluation_split'],'val')
        self.assertEqual(self.store.run('demo',vid)['dataset']['test_count'],0)

    def test_tampered_snapshot_blocks_admission_and_execution(self):
        version=self.publish();rid,body=self.submit(version)
        image=self.datasets.root/'versions'/version['id']/version['manifest']['samples'][0]['image']
        original=image.read_bytes();image.write_bytes(b'corrupt')
        with self.assertRaises(Problem) as e:self.store.submit('demo','later',body)
        self.assertEqual(e.exception.code,'dataset_integrity')
        job=self.store.claim()
        with mock.patch.object(worker,'EXECUTION_ENVIRONMENT',environment()), mock.patch('model_platform.dataset_detection.execute',side_effect=AssertionError('must not train')):
            worker.execute(self.store,*job)
        failed=self.store.run('demo',rid)
        self.assertEqual(failed['attempts'][0]['error_code'],'dataset_integrity')
        self.assertIsNone(failed['model_version_id'])
        image.write_bytes(original);self.store.retry('demo',rid)
        self.assertEqual(len(self.store.run('demo',rid)['attempts']),1)

    def test_conditional_retry_keeps_published_synthetic_dataset_binding(self):
        version = self.publish()
        rid, body = self.submit(version)
        run, attempt = self.store.claim()
        self.store.fail(rid, attempt, 'synthetic_retry_fixture')
        before = self.store.run('demo', rid)
        binding = self.store.dataset_binding(run)
        self.store.retry('demo', rid, attempt)
        after = self.store.run('demo', rid)
        self.assertEqual(after['status'], 'queued')
        for key in ('id', 'recipe', 'dataset', 'config', 'environment', 'request_sha256', 'attempts'):
            self.assertEqual(after[key], before[key])
        self.assertEqual(self.store.dataset_binding(run), binding)
        self.assertEqual(self.store.submit('demo', 'run', body), (rid, False))
        _, newer = self.store.claim()
        self.store.fail(rid, newer, 'synthetic_retry_fixture')
        with self.assertRaises(Problem) as error:
            self.store.retry('demo', rid, attempt)
        self.assertEqual(error.exception.code, 'retry_conflict')
        self.assertEqual(self.store.run('demo', rid)['status'], 'failed')

    def test_import_valid_but_training_rejects_duplicate_missing_class_budget(self):
        files=test_datasets.fixture()
        # Accepted import has class1 only in validation, unsuitable train.
        version=self.publish(test_datasets.archive(files))
        with self.assertRaises(Problem) as e:self.submit(version)
        self.assertEqual(e.exception.code,'dataset_train_classes')
        files['labels/train/a.txt']=b'0 0.5 0.5 0.5 0.5\n1 0.5 0.5 0.5 0.5\n0 0.5 0.5 0.5 0.5\n'
        version=self.publish(test_datasets.archive(files))
        with self.assertRaises(Problem) as e:self.submit(version)
        self.assertEqual(e.exception.code,'dataset_training_annotations')

    def synthetic_result(self,run,binding):
        # Deliberately synthetic unit-only metrics for registration guards, no weight claims.
        config=dict(json.loads(run['input_json'])['config'],**recipe.FIXED_CONFIG)
        mapping=[{'category_id':c['id'],'coco_category_id':c['id']+1,'name':c['name']} for c in binding['classes']]
        evaluation={'contract':'dataset-coco-style-bbox/v1','protocol':'test-fixture-only','evaluator':{'version':'unit'},'parameters':{},
            'artifact_sha256':'e'*64,'dataset_version_id':binding['dataset_version_id'],'dataset_content_sha256':binding['sha256'],
            'dataset_sha256':binding['evaluation_sha256'],'evaluation_sha256':binding['evaluation_sha256'],
            'split':binding['evaluation_split'],'role':'held-out-test' if binding['evaluation_split']=='test' else 'validation-only',
            'category_mapping':mapping,'score_threshold':.001,'nms_iou_threshold':.65,'max_detections':100,'preprocessing':{},
            'map_50':None,'map_50_95':None,'per_class':[],'status':'no_ground_truth'}
        evaluation['compatibility_key']=digest(recipe.compatibility(evaluation))
        manifest={'recipe':recipe.RECIPE,'dataset':binding,'class_names':binding['class_names'],'config':json.loads(run['input_json'])['config'],
                  'training_config':config,'config_sha256':digest(config)}
        return evaluation,manifest

    def test_registration_rejects_wrong_linkages_and_comparison_nulls(self):
        v=self.publish();rid,_=self.submit(v);run,attempt=self.store.claim();binding=self.store.dataset_binding(run)
        evaluation,manifest=self.synthetic_result(run,binding)
        for field in ('artifact_sha256','dataset_version_id','dataset_content_sha256','dataset_sha256','evaluation_sha256','split','role','category_mapping'):
            wrong=dict(evaluation,**{field:'wrong'})
            with self.subTest(field=field),self.assertRaises((Problem,KeyError)):
                self.store.register(run,attempt,'e'*64,wrong,copy.deepcopy(manifest))
        mid=self.store.register(run,attempt,'e'*64,evaluation,manifest)
        self.submit(v,'second');r2,a2=self.store.claim();e2,m2=self.synthetic_result(r2,binding);mid2=self.store.register(r2,a2,'e'*64,e2,m2)
        result=self.store.compare_models('demo',[mid,mid2]);self.assertTrue(result['compatible']);self.assertEqual(result['deltas'],{'map_50':None,'map_50_95':None})
        with self.assertRaises(Problem) as e:self.store.compare_models('other',[mid,mid2])
        self.assertEqual(e.exception.status,404)
        different=self.publish(fixture_bytes(variant=1));self.submit(different,'third');r3,a3=self.store.claim();e3,m3=self.synthetic_result(r3,self.store.dataset_binding(r3));mid3=self.store.register(r3,a3,'e'*64,e3,m3)
        with self.assertRaises(Problem) as e:self.store.compare_models('demo',[mid,mid3])
        self.assertEqual(e.exception.code,'evaluation_incompatible');self.assertIn('dataset',e.exception.message)


@unittest.skipUnless(OPTIONAL, 'Optional dataset dependencies absent')
class DatasetTrainingHttpTests(unittest.TestCase):
    setUp=test_platform.PlatformTests.setUp
    spawn=test_platform.PlatformTests.spawn
    start_api=test_platform.PlatformTests.start_api
    cleanup=test_platform.PlatformTests.cleanup
    call=test_platform.PlatformTests.call

    def test_real_http_owned_dataset_run_and_catalog(self):
        datasets=DatasetStore(self.runtime)
        item,_=datasets.create('demo','fixture',test_datasets.META)
        datasets.upload('demo',item['id'],fixture_bytes)
        version=datasets.publish('demo',item['id'])[0]
        body=recipe.request_body(version['id'])
        status,run=self.call('/runs',body,'custom')
        self.assertEqual(status,202,run);self.assertEqual(run['dataset']['dataset_version_id'],version['id'])
        self.assertEqual(self.call('/runs',body,'custom')[0],200)
        self.assertEqual(self.call('/runs',body,'foreign',tenant='other')[0],404)
        catalog=self.call('/recipes')[1]['recipes'];custom=next(x for x in catalog if x['id']==recipe.RECIPE)
        self.assertEqual(custom['data_mode'],'dataset-version');self.assertEqual(custom['training']['config_schema'],recipe.CONFIG_SCHEMA)
        self.assertEqual(self.call('/model-comparisons',{'model_version_ids':[]})[0],400)


if __name__=='__main__':unittest.main()
