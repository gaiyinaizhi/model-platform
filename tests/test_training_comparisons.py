"""Synthetic persisted records; these tests do not establish model quality."""
import contextlib
import copy
import http.client
import json
import threading
import unittest
from unittest import mock
import test_training_reports as helpers
from test_training_reports import fixture, rehash
from model_platform import training_report_schema as schema
from model_platform.common import Problem
from model_platform.training_comparisons import compare, project, facts, MAX_COMPARISON_BYTES


class ComparisonTests(unittest.TestCase):
    setUp = helpers.ReportTests.setUp
    install = helpers.ReportTests.install
    mutate = helpers.ReportTests.mutate
    mutate_json = helpers.ReportTests.mutate_json

    def second(self, change=lambda r: None):
        b = copy.deepcopy(self.v)
        for key, prefix in [('model_version_id','model'),('run_id','run'),('attempt_id','attempt'),('evaluation_id','eval')]:
            b['training'][key] = prefix + '_' + 'e' * 32
        change(b); rehash(b); schema.validate_report(b)
        self.install(b)
        return b

    def pair(self): return compare(self.store, 'demo', self.v['training']['model_version_id'], self.id)

    def test_identical_conditions_zero_and_deterministic(self):
        self.second(); value=self.pair()
        self.assertTrue(value['compatible']); self.assertEqual(value['differences'], [])
        self.assertEqual(value['deltas']['map_50'],0)
        self.assertEqual(value['deltas']['ar_100'],0)
        self.assertEqual(schema.encode(value),schema.encode(self.pair()))
        for report in value['reports']:schema.validate_report(report)

    def test_changed_training_config_train_data_and_version_allowed(self):
        def change(b):
            p=b['provenance'];d=p['dataset'];e=b['exported_evaluation']
            d.update(dataset_version_id='dsv_'+'3'*32,content_sha256='a'*64,train_sha256='b'*64)
            e.update(dataset_version_id=d['dataset_version_id'],dataset_content_sha256=d['content_sha256'])
            p['config']['seed']+=1;p['training_config']['seed']+=1
            p['config_sha256']=schema.digest(p['training_config'])
        self.second(change); self.assertTrue(self.pair()['compatible'])

    def test_execution_differences_order_and_empty_deltas(self):
        def change(b):
            p=b['provenance'];p['submission_environment']['code_sha256']='f'*64
            p['execution_environment'].update(code_sha256='f'*64,python='3.12.13',platform='Other')
            p['dependencies']['torch']='2.5.0'
        self.second(change);v=self.pair()
        self.assertFalse(v['compatible']);self.assertIsNone(v['deltas'])
        self.assertEqual(v['differences'],['execution_code','execution_python','execution_platform','dependencies'])

    def test_eval_content_difference(self):
        def change(b):
            d=b['provenance']['dataset'];e=b['exported_evaluation']
            d['test_sha256']=d['evaluation_sha256']='a'*64;e['evaluation_sha256']='a'*64
        self.second(change);self.assertEqual(self.pair()['differences'],['evaluation_content'])

    def test_split_role_difference(self):
        def change(b):
            d=b['provenance']['dataset'];e=b['exported_evaluation']
            d.update(test_count=0,test_sha256=None,evaluation_split='val',evaluation_sha256=d['validation_sha256'])
            e.update(split='val',role='validation-only',evaluation_sha256=d['validation_sha256'])
        self.second(change);self.assertEqual(self.pair()['differences'],['evaluation_content','evaluation_split','evaluation_role'])

    def test_null_preserved(self):
        def no_gt(v):
            m=v['exported_evaluation']['metrics'];m.update(status='no_ground_truth',map_50=None,map_50_95=None,ar_100=None)
            m['counts'].update(images_with_ground_truth=0,images_without_ground_truth=2,ground_truth_boxes=0,categories_with_ground_truth=0)
            for c in m['per_class']:c.update(status='no_ground_truth',ground_truth_boxes=0,ap_50=None,ap_50_95=None,ar_100=None)
            return rehash(v)
        a=no_gt(copy.deepcopy(self.v));b=copy.deepcopy(a)
        b['training']['model_version_id']='model_'+'2'*32;rehash(b)
        schema.validate_report(a);schema.validate_report(b)
        v=project([a,b]);self.assertIsNone(v['deltas']['map_50']);self.assertIsNone(v['deltas']['per_class'][0]['ap_50'])

    def test_same_eval_ground_truth_contradiction_refused(self):
        def change(b):
            m=b['exported_evaluation']['metrics']
            m['counts']['ground_truth_boxes']+=1
            m['per_class'][0]['ground_truth_boxes']+=1
        self.second(change)
        with self.assertRaises(Problem) as e:self.pair()
        self.assertEqual(e.exception.code,'training_comparison_integrity')

    def test_prediction_counts_may_differ(self):
        def change(b):
            m=b['exported_evaluation']['metrics']
            m['counts'].update(predictions=1,images_with_predictions=1)
            m['per_class'][0]['predictions']=1
        self.second(change)
        self.assertTrue(self.pair()['compatible'])

    def test_numeric_deltas_are_b_minus_a(self):
        a=copy.deepcopy(self.v);b=copy.deepcopy(a)
        am=a['exported_evaluation']['metrics'];bm=b['exported_evaluation']['metrics']
        am['map_50']=.8;bm['map_50']=.3
        am['per_class'][0]['ap_50']=.8;bm['per_class'][0]['ap_50']=.3
        v=project([a,b]);self.assertAlmostEqual(v['deltas']['map_50'],-.5)
        self.assertAlmostEqual(v['deltas']['per_class'][0]['ap_50'],-.5)

    def test_both_owners_resolved_before_integrity(self):
        self.second();self.mutate('models','manifest_json','bad')
        for a,b in [(self.id,'model_'+'f'*32),('model_'+'f'*32,self.id)]:
            with self.assertRaises(Problem) as e:compare(self.store,'demo',a,b)
            self.assertEqual(e.exception.status,404)
        with self.assertRaises(Problem) as e:self.pair()
        self.assertEqual(e.exception.code,'training_comparison_integrity')

    def test_foreign_second_model_is_404_even_with_corrupt_first(self):
        self.second()
        with self.runtime.connection() as db:
            for row in list(db.execute("SELECT name FROM sqlite_master WHERE type='trigger' AND tbl_name='models'")):
                db.execute('DROP TRIGGER '+row[0])
            db.execute("UPDATE models SET tenant='other' WHERE id=?", (self.id,))
            db.execute("UPDATE models SET manifest_json='bad' WHERE id=?", (self.v['training']['model_version_id'],))
        with self.assertRaises(Problem) as e:self.pair()
        self.assertEqual(e.exception.status,404)

    def test_class_and_protocol_diagnostics_are_structural(self):
        b=copy.deepcopy(self.v)
        b['provenance']['dataset']['classes'][0]['name']='renamed'
        b['exported_evaluation']['preprocessing']['input_size']=192
        v=project([self.v,b])
        self.assertEqual(v['differences'],['classes','evaluation_protocol'])
        self.assertIsNone(v['deltas'])

    def test_unsupported_owner_precedence_and_error_mapping(self):
        self.second();self.mutate_json('models','manifest_json',lambda m:m.update(recipe='fashion-cnn/v1'))
        with self.assertRaises(Problem) as e:compare(self.store,'demo',self.id,'model_'+'f'*32)
        self.assertEqual(e.exception.status,404)
        with self.assertRaises(Problem) as e:self.pair()
        self.assertEqual(e.exception.code,'training_comparison_unsupported')

    def test_unsuccessful_mapping(self):
        self.second();self.mutate('runs','status','failed')
        with self.assertRaises(Problem) as e:self.pair()
        self.assertEqual(e.exception.code,'training_comparison_not_succeeded')

    def test_cross_table_mismatch_refused(self):
        self.second();self.mutate('evaluations','artifact_sha256','9'*64)
        with self.assertRaises(Problem) as e:self.pair()
        self.assertEqual(e.exception.code,'training_comparison_integrity')

    def test_one_readonly_snapshot_no_schema_or_audit_write(self):
        self.second();original=self.runtime.connection;statements=[];connections=[]
        @contextlib.contextmanager
        def readonly():
            connections.append(1)
            with original() as db:
                db.execute('PRAGMA query_only=ON');db.set_trace_callback(statements.append);yield db
        with mock.patch.object(self.runtime,'connection',readonly):self.pair()
        self.assertEqual(len(connections),1)
        self.assertEqual(sum(x=='BEGIN' for x in statements),1)
        self.assertTrue(all(x.split()[0].upper() in ('BEGIN','SELECT','COMMIT') for x in statements),statements)

    def test_invalid_identifiers_duplicates_and_budget(self):
        self.second()
        for a,b in [(self.id,self.id),('x',self.id),(True,self.id)]:
            with self.assertRaises(Problem) as e:compare(self.store,'demo',a,b)
            self.assertEqual(e.exception.status,400)
        with mock.patch('model_platform.training_comparisons.MAX_COMPARISON_BYTES',10):
            with self.assertRaises(Problem) as e:self.pair()
            self.assertEqual(e.exception.code,'training_comparison_integrity')

    def test_http_exact_route_headers_and_ownership(self):
        self.second()
        from model_platform.api import make_handler
        from model_platform.http_support import BoundedHTTPServer
        class Auth:
            def authenticate(self,value):
                if value!='Bearer fixed-public-test-vector':raise Problem(401,'unauthorized','Required')
                return 'demo'
        server=BoundedHTTPServer(('127.0.0.1',0),make_handler(self.store,None,Auth()))
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        def call(path,method='GET',headers=None):
            c=http.client.HTTPConnection(*server.server_address,timeout=3)
            h={'Authorization':'Bearer fixed-public-test-vector'};h.update(headers or {})
            c.request(method,path,headers=h);r=c.getresponse();out=r.status,dict(r.getheaders()),r.read();c.close();return out
        path='/training-model-comparisons?model_a_id='+self.v['training']['model_version_id']+'&model_b_id='+self.id
        try:
            status,h,raw=call(path);self.assertEqual(status,200,raw)
            self.assertEqual(raw,schema.encode(self.pair()));self.assertEqual(h['Cache-Control'],'no-store')
            self.assertEqual(h['X-Content-Type-Options'],'nosniff');self.assertEqual(int(h['Content-Length']),len(raw))
            for p in (path+'&x=1',path.replace('model_b_id','model_a_id'),path.replace('model_a_id','%6dodel_a_id'),path.replace('?','/?'),path.replace('/training','/%74raining'),'/training-model-comparisons','http://localhost'+path):
                self.assertEqual(call(p)[0],400,p)
            self.assertEqual(call(path,'POST')[0],400)
            for k in ('Range','If-Range','Transfer-Encoding','Content-Encoding','Content-Length'):
                self.assertEqual(call(path,headers={k:'1'})[0],400,k)
        finally:server.shutdown();server.server_close();thread.join(3)

if __name__=='__main__':unittest.main()
