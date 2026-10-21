"""Synthetic persistent fixtures exercise reporting, not training/quality."""
import contextlib
import copy
import http.client
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest import mock

from model_platform.common import Runtime, Problem
from model_platform.store import Store
from model_platform import training_report_schema as schema
from model_platform.training_reports import report
from model_platform.dataset_detection import compatibility

ROOT=Path(__file__).resolve().parents[1]


def fixture():
    return json.loads((ROOT/'tests/fixtures/training-report.json').read_bytes())


def rehash(v):
    v['exported_evaluation_sha256']=schema.digest(v['exported_evaluation'])
    v['report_sha256']=schema.digest({k:x for k,x in v.items() if k!='report_sha256'})
    return v


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.runtime=Runtime(self.tmp.name)
        # Public fixed local test vector, never a real user credential.
        with mock.patch('model_platform.store.cursor_key',return_value=b't'*32):
            self.store=Store(self.runtime)
        self.v=fixture(); self.install(self.v)

    def install(self,v):
        t=v['training']; p=v['provenance']; d=p['dataset']; e=v['exported_evaluation']; s=v['training_summary']
        self.id=t['model_version_id']
        binding={k:x for k,x in d.items() if k!='content_sha256'}
        binding.update(sha256=d['content_sha256'],source='private fixture source',name='private fixture source',
                       version=d['dataset_version_id'],class_names=[c['name'] for c in d['classes']])
        body={'recipe':schema.RECIPE,'dataset':{'dataset_version_id':d['dataset_version_id']},'config':p['config']}
        ev={k:x for k,x in e.items() if k!='metrics'}; ev.update(e['metrics'])
        ev.update(dataset_sha256=d['evaluation_sha256'],category_mapping=[{'category_id':c['id'],'coco_category_id':c['id']+1,'name':c['name']} for c in d['classes']],execution_environment=p['execution_environment'])
        ev['compatibility_key']=schema.digest(compatibility(ev))
        manifest={k:x for k,x in p.items() if k not in ('dataset','submission_environment')}
        manifest.update(dataset=binding,environment=p['submission_environment'],**{k:t[k] for k in ('recipe','contract','format','run_id','attempt_id','evaluation_id','model_version_id')},artifact_id=t['attempt_id'],class_names=binding['class_names'],training_loss_history=s['loss_history'],**{k:s[k] for k in ('training_seconds','export_seconds','evaluation_seconds')})
        with self.runtime.connection() as db:
            db.execute('INSERT INTO runs VALUES(?,?,?,?,?,?,?,?,?,?)',(t['run_id'],'demo',t['run_id'],schema.digest(body),schema.encode(body).decode(),schema.encode(p['submission_environment']).decode(),'succeeded',t['created_at'],t['finished_at']+.001,self.id))
            db.execute('INSERT INTO run_dataset_bindings VALUES(?,?,?,?)',(t['run_id'],d['dataset_version_id'],d['content_sha256'],schema.encode(binding).decode()))
            db.execute('INSERT INTO attempts VALUES(?,?,?,?,?,?,?)',(t['attempt_id'],t['run_id'],1,'succeeded',t['started_at'],t['finished_at'],None))
            db.execute('INSERT INTO evaluations VALUES(?,?,?,?,?,?)',(t['evaluation_id'],t['attempt_id'],p['artifact_sha256'],d['evaluation_sha256'],schema.encode(ev).decode(),t['finished_at']-.002))
            db.execute('INSERT INTO models(id,tenant,run_id,attempt_id,evaluation_id,artifact_sha256,manifest_json,created_at) VALUES(?,?,?,?,?,?,?,?)',(self.id,'demo',t['run_id'],t['attempt_id'],t['evaluation_id'],p['artifact_sha256'],schema.encode(manifest).decode(),t['finished_at']-.001))

    def mutate(self,table,field,value):
        with self.runtime.connection() as db:
            for row in list(db.execute("SELECT name FROM sqlite_master WHERE type='trigger' AND tbl_name=?",(table,))): db.execute('DROP TRIGGER '+row[0])
            db.execute(f'UPDATE {table} SET {field}=?',(value,))

    def mutate_json(self,table,field,change):
        with self.runtime.connection() as db: value=json.loads(db.execute(f'SELECT {field} FROM {table}').fetchone()[0])
        change(value); self.mutate(table,field,schema.encode(value).decode())

    def value(self): return schema.parse_report(report(self.store,'demo',self.id))

    def test_fixture_and_deterministic_read_only_projection(self):
        raw=report(self.store,'demo',self.id); self.assertEqual(raw,schema.encode(self.v)); self.assertEqual(raw,report(self.store,'demo',self.id))
        for forbidden in (b'private fixture',b'category_mapping',b'raw_predictions',b'business_ref',b'"tenant"'): self.assertNotIn(forbidden,raw)
        original=self.runtime.connection; statements=[]
        @contextlib.contextmanager
        def readonly():
            with original() as db:
                db.execute('PRAGMA query_only=ON');db.set_trace_callback(statements.append);yield db
        with mock.patch.object(self.runtime,'connection',readonly): self.assertEqual(raw,report(self.store,'demo',self.id))
        self.assertTrue(all(x.split()[0].upper() in ('BEGIN','SELECT','COMMIT') for x in statements),statements)

    def test_ownership_before_invalid_evidence(self):
        self.mutate('models','manifest_json','bad')
        for tenant,mid in [('other',self.id),('demo','model_'+'f'*32)]:
            with self.assertRaises(Problem) as e: report(self.store,tenant,mid)
            self.assertEqual(e.exception.status,404)
        with self.assertRaises(Problem) as e: report(self.store,'demo',self.id)
        self.assertEqual(e.exception.code,'training_report_integrity')

    def test_non_success_status(self):
        self.mutate('runs','status','failed')
        with self.assertRaises(Problem) as e: self.value()
        self.assertEqual(e.exception.code,'training_report_not_succeeded')

    def test_unsupported_recipe(self):
        self.mutate_json('models','manifest_json',lambda v:v.update(recipe='fashion-cnn/v1'))
        with self.assertRaises(Problem) as e: self.value()
        self.assertEqual(e.exception.code,'training_report_unsupported')

    def test_source_binding_tamper(self):
        self.mutate('evaluations','evaluation_sha256','9'*64)
        with self.assertRaises(Problem) as e: self.value()
        self.assertEqual(e.exception.code,'training_report_integrity')

    def test_source_manifest_tamper(self):
        self.mutate_json('models','manifest_json',lambda v:v.update(run_id='run_'+'a'*32))
        with self.assertRaises(Problem): self.value()

    def test_source_compatibility_tamper(self):
        self.mutate_json('evaluations','result_json',lambda v:v.update(compatibility_key='a'*64))
        with self.assertRaises(Problem): self.value()

    def test_source_config_tamper(self):
        self.mutate_json('models','manifest_json',lambda v:v['config'].update(epochs=2))
        with self.assertRaises(Problem): self.value()

    def test_cross_table_tamper_matrix(self):
        cases=[('models','manifest_json',lambda x:x.update(artifact_id='attempt_'+'f'*32)),
               ('models','manifest_json',lambda x:x.update(artifact_sha256='e'*64)),
               ('models','manifest_json',lambda x:x.update(class_names=['other','blue_square'])),
               ('models','manifest_json',lambda x:x['dataset'].update(evaluation_count=1)),
               ('models','manifest_json',lambda x:x['environment'].update(code_sha256='1'*64)),
               ('models','manifest_json',lambda x:x['execution_environment'].update(platform='Linux-other')),
               ('evaluations','result_json',lambda x:x.update(artifact_sha256='e'*64)),
               ('evaluations','result_json',lambda x:x.update(dataset_content_sha256='e'*64)),
               ('evaluations','result_json',lambda x:x.update(role='validation-only')),
               ('evaluations','result_json',lambda x:x['category_mapping'][0].update(coco_category_id=2)),
               ('run_dataset_bindings','binding_json',lambda x:x.update(sha256='e'*64)),
               ('runs','input_json',lambda x:x['config'].update(seed=3))]
        for table,field,change in cases:
            with self.runtime.connection() as db: original=db.execute(f'SELECT {field} FROM {table}').fetchone()[0]
            self.mutate_json(table,field,change)
            with self.subTest(table=table,field=field),self.assertRaises(Problem) as error: self.value()
            self.assertEqual(error.exception.code,'training_report_integrity')
            self.mutate(table,field,original)
        self.assertEqual(self.value(),self.v)

    def test_duplicate_and_nonfinite_stored_evidence_rejected(self):
        with self.runtime.connection() as db: original=db.execute('SELECT manifest_json FROM models').fetchone()[0]
        for bad in ('{"recipe":"yolox-nano-dataset/v1","recipe":"other"}', '{"x":NaN}', '['*25+'0'+']'*25):
            self.mutate('models','manifest_json',bad)
            with self.assertRaises(Problem) as e: self.value()
            self.assertEqual(e.exception.code,'training_report_integrity')
        self.mutate('models','manifest_json',original)

    def test_source_parsing_does_not_probe_current_code_or_files(self):
        with mock.patch('model_platform.common.environment',side_effect=AssertionError('no current probe')), \
             mock.patch('model_platform.datasets.DatasetStore',side_effect=AssertionError('no dataset reads')), \
             mock.patch('builtins.open',side_effect=AssertionError('no artifact reads')):
            self.assertEqual(self.value(),self.v)

    def test_offline_file_type_limits(self):
        from scripts.validate_training_report import read_report
        path=Path(self.tmp.name)/'valid.json';path.write_bytes(schema.encode(self.v))
        link=Path(self.tmp.name)/'link.json';link.symlink_to(path)
        with self.assertRaises(OSError):read_report(link)
        with self.assertRaises((OSError,ValueError)):read_report(Path(self.tmp.name))
        path.write_bytes(b'x'*(schema.MAX_REPORT_BYTES+1))
        with self.assertRaises(ValueError):read_report(path)

    def test_unknown_fields_each_export_object(self):
        paths=[]
        def visit(v,p):
            if isinstance(v,dict):
                paths.append(p)
                for k,x in v.items():visit(x,p+[k])
            elif isinstance(v,list):
                for i,x in enumerate(v):visit(x,p+[i])
        visit(self.v,[])
        for path in paths:
            v=copy.deepcopy(self.v);target=v
            for k in path:target=target[k]
            target['unexpected']='/private'
            with self.subTest(path=path),self.assertRaises(ValueError):schema.parse_report(schema.encode(rehash(v)))

    def test_rehashed_semantic_corruption(self):
        changes=[lambda v:v['training'].update(status='failed'),lambda v:v['training'].update(attempt_number=True),
            lambda v:v['training'].update(finished_at=1),lambda v:v['provenance']['dataset'].update(test_count=0),
            lambda v:v['provenance'].update(config_sha256='a'*64),lambda v:v['provenance'].update(final_parameters_sha256='0'*64),
            lambda v:v['exported_evaluation'].update(role='validation-only'),lambda v:v['exported_evaluation']['metrics'].update(map_50=.5),
            lambda v:v['exported_evaluation']['metrics']['counts'].update(images=1),lambda v:v['exported_evaluation']['metrics']['per_class'][0].update(name='other'),
            lambda v:v['provenance']['execution_environment'].update(code_sha256='a'*64),lambda v:v['training_summary'].update(loss_history=[])]
        for change in changes:
            v=copy.deepcopy(self.v);change(v)
            with self.assertRaises(ValueError):schema.parse_report(schema.encode(rehash(v)))

    def test_distinct_python_platform_preserved_when_same_code(self):
        v=copy.deepcopy(self.v);v['provenance']['execution_environment'].update(python='3.12.13',platform='Linux-other')
        self.assertEqual(schema.parse_report(schema.encode(rehash(v)))['provenance']['execution_environment']['python'],'3.12.13')

    def test_validation_only_and_null_metrics(self):
        v=copy.deepcopy(self.v);d=v['provenance']['dataset'];e=v['exported_evaluation'];m=e['metrics']
        d.update(test_count=0,test_sha256=None,evaluation_split='val',evaluation_sha256=d['validation_sha256'])
        e.update(split='val',role='validation-only',evaluation_sha256=d['validation_sha256'])
        m.update(status='no_ground_truth',map_50=None,map_50_95=None,ar_100=None)
        m['counts'].update(images_with_ground_truth=0,images_without_ground_truth=2,ground_truth_boxes=0,categories_with_ground_truth=0)
        for r in m['per_class']:r.update(status='no_ground_truth',ground_truth_boxes=0,ap_50=None,ap_50_95=None,ar_100=None)
        schema.parse_report(schema.encode(rehash(v)))
        m['map_50']=0
        with self.assertRaises(ValueError):schema.parse_report(schema.encode(rehash(v)))

    def test_parser_limits_duplicates_noncanonical(self):
        raw=schema.encode(self.v)
        for bad in (b'',raw+b'\n',b'\xef\xbb\xbf'+raw,raw[:-1],b'['*25+b'0'+b']'*25,
                    b'{"a":1,"a":2}',b'{"a":NaN}',b'{"a":1e9999}',b'x'*(schema.MAX_REPORT_BYTES+1),
                    b'{"a":'+b'1'*65+b'}',b'{"a":"'+b'x'*(8192*6+3)+b'"}'):
            with self.subTest(length=len(bad)),self.assertRaises(ValueError):schema.parse_report(bad)

    def test_offline_cli_python_s_and_no_writes(self):
        path=Path(self.tmp.name)/'report.json';path.write_bytes(schema.encode(self.v))
        cli=ROOT/'scripts/validate_training_report.py'
        result=subprocess.run([sys.executable,'-S',str(cli),str(path)],cwd=self.tmp.name,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr);self.assertIn('unverified',result.stdout)
        self.assertEqual(path.read_bytes(),schema.encode(self.v))
        path.write_bytes(b'bad');result=subprocess.run([sys.executable,'-S',str(cli),str(path)],capture_output=True)
        self.assertEqual(result.returncode,1);self.assertNotIn(str(path).encode(),result.stderr)

    def test_http_route_headers_auth_and_no_negotiation(self):
        from model_platform.api import make_handler
        from model_platform.http_support import BoundedHTTPServer
        class Auth:
            def authenticate(self,value):
                if value!='Bearer fixed-public-test-vector':raise Problem(401,'unauthorized','Required')
                return 'demo'
        server=BoundedHTTPServer(('127.0.0.1',0),make_handler(self.store,None,Auth()))
        worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
        def call(path,method='GET',headers=None,auth=True):
            c=http.client.HTTPConnection(*server.server_address,timeout=3)
            h={'Authorization':'Bearer fixed-public-test-vector'} if auth else {};h.update(headers or {})
            c.request(method,path,headers=h);r=c.getresponse();out=r.status,dict(r.getheaders()),r.read();c.close();return out
        path=f'/models/{self.id}/training-report'
        try:
            status,h,raw=call(path);self.assertEqual(status,200,raw);schema.parse_report(raw)
            self.assertEqual(h['Content-Disposition'],'attachment; filename="training-evaluation-report.json"')
            self.assertEqual(h['Content-Length'],str(len(raw)));self.assertEqual(h['Cache-Control'],'no-store')
            self.assertEqual(h['X-Content-Type-Options'],'nosniff');self.assertEqual(h['Content-Type'],'application/json')
            self.assertEqual(call(path,auth=False)[0],401)
            self.assertEqual(call(path,headers={'Accept':'text/html','Accept-Encoding':'gzip'})[2],raw)
            for p in (path+'?',path+'/',path+'?x=1',path.replace('training-report','%74raining-report'),path.replace('/training-report','//training-report'),'http://localhost'+path):self.assertEqual(call(p)[0],400,p)
            self.assertEqual(call(path,'POST')[0],400)
            for k in ('Range','If-Range','Transfer-Encoding','Content-Encoding'):self.assertEqual(call(path,headers={k:'bad'})[0],400)
            self.assertEqual(call(path,headers={'Content-Length':'1'})[0],400)
        finally:server.shutdown();server.server_close();worker.join(3)


if __name__=='__main__':unittest.main()
