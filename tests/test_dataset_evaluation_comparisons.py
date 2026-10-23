"""Strict comparison fixtures and actual ORT mechanics, not quality evidence."""
import contextlib
import copy
import http.client
import json
import sqlite3
import threading
import unittest
from unittest import mock

from model_platform.common import Problem, encode, environment
from model_platform import dataset_evaluation_comparisons as comparison
from model_platform import evaluation_report_schema as schema
from model_platform import dataset_evaluation_report_schema as trained
from model_platform.evaluation_reports import report
import test_dataset_evaluation_reports as reports
import test_dataset_model_evaluations as actual


def pair(count=2, mode='measured'):
    a = reports.fixture(count, mode); b = copy.deepcopy(a)
    b['evaluation']['id'] = 'mev_' + 'f'*32
    return a, reports.rehash(b)


def project(a, b):
    return comparison.project(schema.encode(reports.rehash(a)), schema.encode(reports.rehash(b)))


class ComparisonContractTests(unittest.TestCase):
    def test_different_models_training_lineages_allowed_original_texts(self):
        a, b = pair()
        b['evaluation']['model_version_id'] = b['provenance']['model']['model_version_id'] = 'model_'+'f'*32
        for key,prefix in (('run_id','run'),('attempt_id','attempt'),('evaluation_id','eval')):
            b['provenance']['model'][key] = prefix+'_'+'f'*32
        b['provenance']['model'].update(artifact_sha256='f'*64, manifest_sha256='e'*64)
        b['provenance']['training_dataset'].update(dataset_version_id='dsv_'+'f'*32, content_sha256='f'*64,
            train_sha256='e'*64, validation_sha256='d'*64, test_sha256='c'*64, evaluation_sha256='c'*64)
        b['evaluation'].update(created_at=10, started_at=11, finished_at=12)
        value = project(a,b)
        self.assertTrue(value['compatible']);self.assertEqual(value['incompatibilities'],[])
        self.assertEqual(value['a_report'].encode(),schema.encode(reports.rehash(a)))
        self.assertEqual(value['b_report'].encode(),schema.encode(reports.rehash(b)))
        self.assertEqual(set(value),{'schema_version','a_report','b_report','compatible','incompatibilities','delta'})
        self.assertTrue(all(x == 0 for x in value['delta']['overall'].values()))

    def test_zero_null_and_class_bounds(self):
        for count in (1,2,20):
            for mode in ('zero','no_gt'):
                a,b=pair(count,mode);v=project(a,b)
                self.assertEqual(v['delta']['overall']['map_50'],0 if mode=='zero' else None)
                self.assertIsNone(v['delta']['overall']['precision'])
                self.assertEqual(len(v['delta']['per_class']),count)
                self.assertEqual(v['delta']['per_class'][0]['category_id'],0)
                self.assertIsNone(v['delta']['per_class'][-1]['ap_50'] if count>1 else None)

    def test_nonzero_deltas_and_reverse(self):
        a,b=pair()
        # Same ground truth, fewer predictions/TP. AP can legitimately be zero.
        result=b['exported_result'];m=result['metrics'];op=result['operating_point']
        m.update(map_50=0,map_50_95=0,ar_100=0)
        m['counts'].update(predictions=0,images_with_predictions=0)
        for r in m['per_class']:
            r['predictions']=0
            for k in ('ap_50','ap_50_95','ar_100'):r[k]=0 if r['ground_truth_boxes'] else None
        for r,metric in zip(op['per_class'],m['per_class']):
            r.update(tp=0,fp=0,fn=metric['ground_truth_boxes'],precision=None,
                     recall=0 if metric['ground_truth_boxes'] else None)
        op['totals'].update(tp=0,fp=0,fn=m['counts']['ground_truth_boxes'],precision=None,recall=0)
        forward=project(a,b)['delta'];reverse=project(b,a)['delta']
        self.assertEqual(forward['overall']['tp'],-1);self.assertEqual(forward['overall']['fp'],-2)
        self.assertEqual(forward['overall']['fn'],1);self.assertAlmostEqual(forward['overall']['map_50'],-.5)
        for k,v in forward['overall'].items():self.assertEqual(reverse['overall'][k],None if v is None else -v)
        self.assertEqual(forward['per_class'][1]['tp'],-1)

    def test_every_compatibility_field_order_and_null_delta(self):
        a,b=pair();original=comparison.facts(a)
        # Facts layer exercises every key, including protocol fields fixed by the schema.
        for key in original:
            changed=copy.deepcopy(original);changed[key]={'different':True}
            with mock.patch.object(comparison,'facts',side_effect=[original,changed]):
                v=project(a,b)
            self.assertFalse(v['compatible']);self.assertIsNone(v['delta'])
            self.assertEqual(v['incompatibilities'],[{'field':key,'reason':key+' differs'}])
        self.assertEqual(list(original),['dataset.version','dataset.content','dataset.scope','categories',
            'category_mapping','protocol','protocol_digest','metric_protocol','metric_evaluator',
            'metric_parameters','operating_point','submission.code','submission.python','submission.platform',
            'submission.dependencies','execution.code','execution.python','execution.platform','execution.dependencies'])

    def test_valid_rehashed_conditions_incompatible(self):
        for field in ('version','content','python','platform','code','dependencies','category','evaluator','scope','onnx'):
            a,b=pair();p=b['provenance'];r=b['exported_result']
            if field=='version':b['evaluation']['dataset_version_id']=p['dataset']['dataset_version_id']='dsv_'+'e'*32
            elif field=='content':p['dataset']['content_sha256']='e'*64
            elif field=='category':
                p['categories'][0]['name']='Changed'
                r['metrics']['per_class'][0]['name']=r['operating_point']['per_class'][0]['name']='Changed'
            elif field=='scope':
                p['dataset'].update(images=p['dataset']['images']+1,content_sha256='e'*64)
                r['metrics']['counts']['images']+=1;r['metrics']['counts']['images_without_ground_truth']+=1
                r['scope']['images']+=1
            elif field in ('evaluator','onnx'):
                deps=json.loads(p['submission_environment']['dependencies'])
                key='onnx' if field=='onnx' else 'pycocotools';deps[key]='1.0.0'
                p['submission_environment']['dependencies']=schema.encode(deps).decode()
                if field=='evaluator':
                    r['environment']['dependencies'][key]='1.0.0';r['metrics']['evaluator']['version']='1.0.0'
            elif field=='dependencies':
                deps=json.loads(p['submission_environment']['dependencies']);deps['numpy']='2.0.0'
                p['submission_environment']['dependencies']=schema.encode(deps).decode();r['environment']['dependencies']['numpy']='2.0.0'
            else:
                key='code_sha256' if field=='code' else field
                val='e'*64 if field=='code' else ('3.12.1' if field=='python' else 'Linux-other')
                p['submission_environment'][key]=r['environment'][key]=val
            expected={'version':['dataset.version'],'content':['dataset.content'],
                'python':['submission.python','execution.python'],'platform':['submission.platform','execution.platform'],
                'code':['submission.code','execution.code'],'dependencies':['submission.dependencies','execution.dependencies'],
                'category':['categories','category_mapping'],
                'evaluator':['metric_evaluator','submission.dependencies','execution.dependencies'],
                'scope':['dataset.content','dataset.scope'],'onnx':['submission.dependencies']}[field]
            with self.subTest(field=field):
                value=project(a,b);self.assertFalse(value['compatible']);self.assertIsNone(value['delta'])
                self.assertEqual([r['field'] for r in value['incompatibilities']],expected)

    def test_malformed_mixed_same_id_and_budget_rejected(self):
        a,b=pair();raw=schema.encode(a)
        for bad in (raw+b'\n',b'{}',raw[:-1]):
            with self.assertRaises(ValueError):comparison.project(bad,schema.encode(b))
        with self.assertRaises(ValueError):comparison.project(raw,raw)
        b['schema_version']='independent-evaluation-report/v1'
        with self.assertRaises(ValueError):project(a,b)
        a,b=pair()
        with mock.patch.object(comparison,'MAX_COMPARISON_BYTES',1),self.assertRaises(ValueError):project(a,b)

    def test_rehashed_ground_truth_contradiction_rejected(self):
        a,b=pair(2,'zero');b['provenance']['dataset']['boxes']=0
        c=reports.fixture(2,'no_gt');c['evaluation']['id']=b['evaluation']['id']
        # Both source content IDs remain the same, but ground truth differs.
        c['provenance']['dataset']['content_sha256']=a['provenance']['dataset']['content_sha256']
        with self.assertRaises(ValueError):project(a,c)


@unittest.skipUnless(actual.OPTIONAL,'Optional ONNX/dataset dependencies absent')
class ActualComparisonTests(unittest.TestCase):
    def setUp(self):
        self.case=actual.TrainedEvaluationTests('test_actual_read_only_child_metrics_and_immutable_registration')
        self.case.setUp();self.addCleanup(self.case.doCleanups)
        self.case.test_actual_read_only_child_metrics_and_immutable_registration()
        self.e=self.case.evaluations;self.a=self.case.detail['id']
        self.b=self.case.create('second')['id'];row=self.e.claim()
        actual.independent.execute(self.e,row,environment(),timeout=30)
        self.assertEqual(self.e.get('demo',self.b)['status'],'succeeded')

    def snapshot(self):
        with self.case.runtime.connection() as db:
            names=[r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
            return {t:[tuple(r) for r in db.execute('SELECT * FROM '+t)] for t in names}

    def test_actual_readonly_exact_reports_and_determinism(self):
        before=self.snapshot();files={p.name:p.read_bytes() for p in self.e.root.iterdir()};artifact=self.case.path.read_bytes()
        expected=[report(self.e,'demo',x).decode() for x in (self.a,self.b)]
        original=self.case.runtime.connection;statements=[]
        @contextlib.contextmanager
        def readonly():
            with original() as db:
                db.execute('PRAGMA query_only=ON');db.set_trace_callback(statements.append);yield db
        with mock.patch.object(self.case.runtime,'connection',readonly):v=comparison.compare(self.e,'demo',self.a,self.b)
        self.assertTrue(v['compatible']);self.assertEqual([v['a_report'],v['b_report']],expected)
        self.assertEqual(v,comparison.compare(self.e,'demo',self.a,self.b))
        self.assertEqual(before,self.snapshot());self.assertEqual(files,{p.name:p.read_bytes() for p in self.e.root.iterdir()})
        self.assertEqual(artifact,self.case.path.read_bytes())
        self.assertTrue(all(s.lstrip().upper().startswith(('SELECT ','BEGIN','COMMIT')) for s in statements),statements)

    def test_ownership_precedes_status_protocol_integrity(self):
        def fail(a,b,status,code,tenant='demo'):
            with self.assertRaises(Problem) as e:comparison.compare(self.e,tenant,a,b)
            self.assertEqual((e.exception.status,e.exception.code),(status,code))
        fail(self.a,self.a,400,'invalid_input');fail('bad',self.b,400,'invalid_input')
        fail(self.a,self.b,404,'not_found','other')
        queued=self.case.create('pending')['id'];missing='mev_'+'0'*32
        fail(queued,missing,404,'not_found');fail(queued,self.a,409,'evaluation_not_succeeded')
        with self.case.runtime.connection() as db:
            body=json.loads(db.execute('SELECT request_json FROM model_evaluations WHERE id=?',(queued,)).fetchone()[0])
            body['protocol']=actual.independent.PROTOCOL
            # Explicit tamper fixture: remove immutable protection only in this temporary DB.
            for trigger in list(db.execute("SELECT name FROM sqlite_master WHERE type='trigger' AND tbl_name='model_evaluations'")):
                db.execute('DROP TRIGGER '+trigger[0])
            db.execute('UPDATE model_evaluations SET request_json=? WHERE id=?',(encode(body).decode(),queued))
        fail(queued,missing,404,'not_found');fail(queued,self.a,409,'evaluation_comparison_unsupported')
        self.e.evidence_path(self.a).write_bytes(b'changed')
        fail(self.a,missing,404,'not_found');fail(self.a,self.b,409,'evaluation_integrity')

    def test_http_exact_queries_and_auth(self):
        from model_platform.api import make_handler
        from model_platform.http_support import BoundedHTTPServer
        class Auth:
            def authenticate(self,header):
                if header not in ('Bearer demo-test-only','Bearer other-test-only'):raise Problem(401,'unauthorized','Test identity')
                return 'demo' if header=='Bearer demo-test-only' else 'other'
        server=BoundedHTTPServer(('127.0.0.1',0),make_handler(self.case.store,None,Auth()))
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        def call(path,method='GET',tenant='demo'):
            c=http.client.HTTPConnection(*server.server_address,timeout=3)
            c.request(method,path,headers={'Authorization':'Bearer '+tenant+'-test-only'})
            r=c.getresponse();v=r.status,json.loads(r.read());c.close();return v
        base='/dataset-model-evaluation-comparisons';query='?evaluation_a_id='+self.a+'&evaluation_b_id='+self.b
        try:
            status,v=call(base+query);self.assertEqual(status,200);self.assertTrue(v['compatible'])
            self.assertEqual(call(base+query,tenant='other')[0],404)
            self.assertEqual(call(base+query,tenant='bad')[0],401)
            for path in (base,base+'?',base+'?evaluation_a_id='+self.a,base+query+'&x=1',base+query+'&evaluation_a_id='+self.a,
                         base+'?evaluation_a_id='+self.a+'&evaluation_a_id='+self.b):
                self.assertEqual(call(path)[0],400,path)
            self.assertEqual(call(base+query,method='POST')[0],400)
            self.assertEqual(call(base+query.replace(self.b,self.a))[0],400)
            self.assertEqual(call('/model-evaluation-comparisons'+query)[1]['error']['code'],'evaluation_comparison_unsupported')
        finally:server.shutdown();server.server_close();thread.join(timeout=3)
