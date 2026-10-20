"""Independent queue/identity failures; synthetic fixtures are not quality evidence."""
import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

from model_platform.common import Runtime, Problem, digest, encode, environment
from model_platform.store import Store
from model_platform import model_evaluations as module
from model_platform.model_evaluations import ModelEvaluationStore, PROTOCOL
from model_platform.model_imports import ModelImportStore, CATALOG_ID
from model_platform import model_imports
from model_platform.datasets import DatasetStore
from scripts.demo import request_body


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.runtime = Runtime(self.temp.name)
        self.store = Store(self.runtime)
        DatasetStore(self.runtime)
        self.addCleanup(mock.patch.stopall)
        raw = b'synthetic opaque fixture, not ONNX'
        mock.patch.object(model_imports, 'ASSET_BYTES', len(raw)).start()
        mock.patch.object(model_imports, 'ASSET_SHA256', __import__('hashlib').sha256(raw).hexdigest()).start()
        mock.patch.object(model_imports, 'validate_isolated', return_value={'status': 'validated'}).start()
        imports = ModelImportStore(self.store)
        item, _ = imports.create('demo', 'fixture', {'catalog_id': CATALOG_ID, 'rights_confirmed': True})
        imports.upload('demo', item['id'], lambda: raw)
        model, _ = imports.register('demo', item['id'])
        self.model_id = model['model_version_id']
        self.dataset_id = 'dsv_' + 'a'*32
        with self.runtime.connection() as db:
            db.execute('INSERT INTO dataset_versions VALUES(?,?,?,?,?,?,?,?)',
                       (self.dataset_id, 'demo', 'fixture', 'a'*64, time.time(), 'fixture', '{}', 0))
        self.evaluations = ModelEvaluationStore(self.store)
        self.versions = {'numpy':'fixture', 'onnxruntime':'fixture', 'onnx':'fixture',
                         'opencv':'fixture', 'Pillow':'fixture', 'pycocotools':'fixture'}
        mock.patch.object(module, 'dependencies').start()
        mock.patch.object(module, 'installed_versions', return_value=self.versions).start()
        self.body = {'model_version_id': self.model_id, 'dataset_version_id': self.dataset_id,
                     'split': 'val', 'protocol': PROTOCOL}
        self.binding = {'model': {'model_version_id': self.model_id, 'artifact_sha256': 'b'*64},
                        'dataset': {'dataset_version_id': self.dataset_id, 'content_sha256': 'a'*64, 'source': None},
                        'category_mapping_sha256': 'c'*64, 'protocol_sha256': 'd'*64,
                        'submission_environment': module._env()}
        self.binder = mock.patch.object(self.evaluations, 'binding_for', return_value=self.binding).start()

    def create(self, key='one', tenant='demo'):
        return self.evaluations.create(tenant, key, self.body)[0]

    def test_cancelled_training_does_not_block_global_evaluation_order(self):
        training, _ = self.store.submit('demo', 'cancel-before-evaluation', request_body())
        evaluation = self.create('after-training')
        self.assertIsNone(self.evaluations.claim())
        self.store.cancel('demo', training)
        self.assertEqual(self.evaluations.claim()['id'], evaluation['id'])
        self.assertIsNone(self.store.claim())
        self.assertEqual(self.store.run('demo', training)['attempts'], [])

    def succeed(self):
        item = self.create()
        row = self.evaluations.claim()
        evidence = []
        result = {'digests': dict(self.evaluations.expected_digests(self.binding, digest(self.binding)),
                                 prediction_sha256=digest(evidence))}
        self.evaluations.succeed(row, result, evidence)
        return self.evaluations.get('demo', item['id'])

    def test_http_routes_authentication_filter_and_replay(self):
        import http.client
        import threading
        from model_platform.api import make_handler
        from model_platform.http_support import BoundedHTTPServer
        class TestAuth:
            def authenticate(self, header):
                #Test-only identity stub; no credential is created or persisted.
                if header not in ('Bearer demo-fixture','Bearer other-fixture'):
                    raise Problem(401,'unauthorized','Test identity required')
                return 'demo' if header=='Bearer demo-fixture' else 'other'
        server=BoundedHTTPServer(('127.0.0.1',0),make_handler(self.store,None,TestAuth()))
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        def call(method,path,body=None,identity='demo',key=None):
            connection=http.client.HTTPConnection(*server.server_address,timeout=3)
            headers={'Authorization':'Bearer '+identity+'-fixture','Content-Type':'application/json'}
            if key is not None:headers['Idempotency-Key']=key
            connection.request(method,path,None if body is None else encode(body),headers)
            response=connection.getresponse(); result=json.loads(response.read());status=response.status
            connection.close();return status,result
        try:
            with mock.patch.object(ModelEvaluationStore,'binding_for',return_value=self.binding):
                status,item=call('POST','/model-evaluations',self.body,key='http')
                self.assertEqual(status,202)
                self.assertEqual(call('POST','/model-evaluations',self.body,key='http')[0],200)
                self.assertEqual(call('POST','/model-evaluations',self.body)[0],400)
                self.assertEqual(call('GET','/model-evaluations/'+item['id'])[0],200)
                self.assertEqual(call('GET','/model-evaluations/'+item['id'],identity='other')[0],404)
                status,result=call('GET','/model-evaluations?model_version_id='+self.model_id)
                self.assertEqual(status,200);self.assertEqual(len(result['evaluations']),1)
                self.assertEqual(call('GET','/model-evaluations?model_version_id='+self.model_id,identity='other')[0],404)
                self.assertEqual(call('GET','/model-evaluations?limit=1')[0],400)
                self.assertEqual(call('GET','/model-evaluations?model_version_id=x&model_version_id=y')[0],400)
                self.assertEqual(call('GET','/model-evaluations',identity='unauthorized')[0],401)
        finally:
            server.shutdown();server.server_close();thread.join(timeout=3)

    def test_idempotency_conflict_strict_body_and_ownership(self):
        item = self.create()
        self.assertEqual(item, self.create())
        self.assertEqual(self.evaluations.list('other')['evaluations'], [])
        for op in (lambda: self.evaluations.get('other', item['id']),
                   lambda: self.evaluations.list('other', self.model_id)):
            with self.assertRaises(Problem) as error:
                op()
            self.assertEqual(error.exception.status, 404)
        changed = dict(self.body, dataset_version_id='dsv_'+'b'*32)
        with self.assertRaises(Problem) as error:
            self.evaluations.create('demo','one',changed)
        self.assertEqual(error.exception.code, 'idempotency_conflict')
        for changed in (dict(self.body, score_threshold=.1), dict(self.body, split='train'), dict(self.body, protocol='unknown')):
            with self.assertRaises(Problem):
                self.evaluations.create('demo','two',changed)
        with self.assertRaises(Problem):
            self.evaluations.create('demo','',self.body)

    def test_claim_once_terminal_recovery_and_immutable_identity(self):
        item = self.create()
        row = self.evaluations.claim()
        self.assertEqual(row['id'],item['id'])
        self.assertIsNone(self.evaluations.claim())
        self.assertEqual(self.evaluations.recover(),1)
        self.assertEqual(self.evaluations.recover(),0)
        failed = self.evaluations.get('demo', item['id'])
        self.assertEqual((failed['status'],failed['error_code']),('failed','evaluation_interrupted'))
        self.assertEqual(self.create()['id'],item['id'])
        with self.runtime.connection() as db:
            for sql in ("UPDATE model_evaluations SET status='queued'", "DELETE FROM model_evaluations",
                        "UPDATE model_evaluations SET tenant='other'", "UPDATE model_evaluations SET binding_sha256='bad'"):
                with self.assertRaises(sqlite3.IntegrityError):
                    db.execute(sql)

    def test_combined_queue_and_global_order(self):
        train = self.store.submit('demo','train-first',request_body())[0]
        item = self.create()
        self.assertIsNone(self.evaluations.claim())
        job = self.store.claim()
        self.assertEqual(job[0]['id'],train)
        self.store.fail(train,job[1],'fixture')
        self.assertEqual(self.evaluations.claim()['id'],item['id'])
        self.evaluations.fail(item['id'],'fixture')
        #32 queued training jobs prevent an additional independent job.
        for i in range(32):
            self.store.submit('demo','queued-'+str(i),request_body())
        with self.assertRaises(Problem) as error:
            self.create('over-capacity')
        self.assertEqual(error.exception.code,'evaluation_capacity')
        with self.assertRaises(Problem):
            self.store.retry('demo',train)

    def test_evaluation_slots_also_block_training(self):
        for i in range(32):
            self.create('evaluation-'+str(i))
        with self.assertRaises(Problem) as error:
            self.store.submit('demo','over-capacity',request_body())
        self.assertEqual(error.exception.code,'run_capacity')

    def test_success_keeps_model_provenance_and_training_tables(self):
        before=self.store.model('demo',self.model_id)
        item=self.succeed()
        self.assertEqual(item['status'],'succeeded')
        self.assertEqual(self.store.model('demo',self.model_id),before)
        self.assertIsNone(before['evaluation'])
        with self.runtime.connection() as db:
            for table in ('runs','attempts','evaluations'):
                self.assertEqual(db.execute('SELECT count(*) FROM '+table).fetchone()[0],0)
            for table in ('model_evaluations','model_evaluation_results'):
                with self.assertRaises(sqlite3.IntegrityError):db.execute('DELETE FROM '+table)
        summary=self.evaluations.list('demo')['evaluations'][0]
        self.assertIsNone(summary['result']); self.assertNotIn('binding',summary)

    def test_polling_snapshot_is_coherent_when_success_commits_during_read(self):
        for kind in ('detail','list'):
            item=self.create('race-'+kind);row=self.evaluations.claim()
            result={'digests':dict(self.evaluations.expected_digests(self.binding,digest(self.binding)),
                                   prediction_sha256=digest([]))}
            verify=self.evaluations.verified_result
            completed=False
            def commit_between(snapshot,binding):
                nonlocal completed
                if snapshot['id']==item['id'] and not completed:
                    completed=True;self.evaluations.succeed(row,result,[])
                return verify(snapshot,binding)
            with mock.patch.object(self.evaluations,'verified_result',side_effect=commit_between):
                if kind=='detail':value=self.evaluations.get('demo',item['id'])
                else:value=next(v for v in self.evaluations.list('demo')['evaluations'] if v['id']==item['id'])
            self.assertTrue(completed)
            self.assertEqual(value['status'],'running');self.assertIsNone(value['result'])
            self.assertEqual(self.evaluations.get('demo',item['id'])['status'],'succeeded')

    def test_revalidation_errors_are_integrity_errors_after_acceptance(self):
        item=self.create()
        for code in ('not_found','evaluation_model_unsupported','evaluation_dataset_incompatible'):
            with mock.patch.object(self.evaluations,'binding_for',side_effect=Problem(404,code,'fixture')):
                with self.assertRaises(Problem) as error:self.evaluations.get('demo',item['id'])
                self.assertEqual((error.exception.status,error.exception.code),(409,'evaluation_integrity'))
        with mock.patch.object(self.evaluations,'binding_for',side_effect=Problem(404,'not_found','fixture')):
            with self.assertRaises(Problem) as error:self.create('initial-rejection')
            self.assertEqual((error.exception.status,error.exception.code),(404,'not_found'))

    def test_publication_lock_wait_cannot_commit_after_deadline(self):
        import threading
        item=self.create();row=self.evaluations.claim()
        result={'digests':dict(self.evaluations.expected_digests(self.binding,digest(self.binding)),
                               prediction_sha256=digest([]))}
        locked=threading.Event()
        def hold_write_lock():
            with self.runtime.connection() as db:
                db.execute('BEGIN IMMEDIATE');locked.set();time.sleep(.25)
        holder=threading.Thread(target=hold_write_lock);holder.start();self.assertTrue(locked.wait(2))
        try:
            with self.assertRaises(Problem) as error:
                self.evaluations.succeed(row,result,[],deadline=time.monotonic()+.12)
            self.assertEqual(error.exception.code,'evaluation_timeout')
        finally:holder.join(timeout=3)
        with self.runtime.connection() as db:
            self.assertEqual(db.execute('SELECT count(*) FROM model_evaluation_results').fetchone()[0],0)
            self.assertEqual(db.execute('SELECT status FROM model_evaluations WHERE id=?',(item['id'],)).fetchone()[0],'running')

    def test_evidence_tamper_fail_closed_in_detail_and_list(self):
        item=self.succeed()
        self.evaluations.evidence_path(item['id']).write_text('[{}]')
        for op in (lambda:self.evaluations.get('demo',item['id']),lambda:self.evaluations.list('demo')):
            with self.assertRaises(Problem) as error:op()
            self.assertEqual(error.exception.code,'evaluation_integrity')

    def test_result_tamper_fail_closed(self):
        item=self.succeed()
        with self.runtime.connection() as db:
            db.execute('DROP TRIGGER model_evaluation_result_update_immutable')
            db.execute("UPDATE model_evaluation_results SET result_json='{}'")
        with self.assertRaises(Problem) as error:self.evaluations.get('demo',item['id'])
        self.assertEqual(error.exception.code,'evaluation_integrity')

    def test_binding_tamper_fail_closed(self):
        item=self.create()
        with self.runtime.connection() as db:
            db.execute('DROP TRIGGER model_evaluation_identity_immutable')
            db.execute("UPDATE model_evaluations SET binding_json='{}'")
        with self.assertRaises(Problem) as error:self.evaluations.get('demo',item['id'])
        self.assertEqual(error.exception.code,'evaluation_integrity')

    def test_source_and_environment_changes_fail_before_child_start(self):
        item=self.create(); row=self.evaluations.claim()
        with mock.patch.object(module.subprocess,'Popen') as process:
            module.execute(self.evaluations,row,dict(environment(),code_sha256='f'*64))
            process.assert_not_called()
        self.assertEqual(self.evaluations.get('demo',item['id'])['error_code'],'execution_code_changed')
        item=self.create('environment'); row=self.evaluations.claim()
        with mock.patch.object(module,'installed_versions',return_value=dict(self.versions,numpy='changed')):
            with mock.patch.object(module.subprocess,'Popen') as process:
                module.execute(self.evaluations,row,environment()); process.assert_not_called()
        self.assertEqual(self.evaluations.get('demo',item['id'])['error_code'],'execution_environment_changed')

    def test_actual_child_timeout_is_reaped(self):
        item=self.create(); row=self.evaluations.claim()
        actual=subprocess.Popen
        children=[]
        def delayed(*args,**kwargs):
            child=actual([sys.executable,'-c','import time; time.sleep(60)'],stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
            children.append(child); return child
        started=time.monotonic()
        with mock.patch.object(module.subprocess,'Popen',side_effect=delayed):
            module.execute(self.evaluations,row,environment(),timeout=.15)
        self.assertLess(time.monotonic()-started,3)
        self.assertIsNotNone(children[0].poll())
        self.assertEqual(self.evaluations.get('demo',item['id'])['error_code'],'evaluation_timeout')

    def test_orphan_evaluator_exits_when_its_parent_dies(self):
        item=self.create(); self.evaluations.claim()
        child_code = ("import os,time; from model_platform import model_evaluations as m; "
                      "m.measure=lambda *args: time.sleep(60); "
                      "m._child(" + repr(str(self.runtime.home)) + "," + repr(item['id']) + ",os.getppid())")
        parent_code = ("import subprocess,sys,time; p=subprocess.Popen([sys.executable,'-c',"
                       + repr(child_code) + "],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); "
                       "print(p.pid,flush=True); time.sleep(60)")
        parent=subprocess.Popen([sys.executable,'-c',parent_code],stdout=subprocess.PIPE,text=True)
        try:
            pid=int(parent.stdout.readline().strip()); time.sleep(.2)
            parent.kill(); parent.wait(timeout=3)
            deadline=time.monotonic()+3
            while time.monotonic()<deadline:
                try:
                    state=Path('/proc/'+str(pid)+'/stat').read_text().split()[2]
                    if state=='Z':break
                except FileNotFoundError:break
                time.sleep(.05)
            else:
                os.kill(pid,signal.SIGKILL)
                self.fail('Evaluator remained alive after its parent died')
        finally:
            if parent.poll() is None:parent.kill()
            parent.wait();parent.stdout.close()
        self.assertEqual(self.evaluations.recover(),1)

    def test_list_skips_full_dataset_rechecks_but_verifies_result_bytes(self):
        self.succeed()
        self.binder.reset_mock()
        self.evaluations.list('demo')
        self.binder.assert_not_called()
        #Storage budgets count even orphan regular files and reject symlinks.
        orphan=self.evaluations.root/'orphan.json';orphan.write_bytes(b'x'*10)
        with mock.patch.object(module,'MAX_EVIDENCE_BYTES',1),self.assertRaises(Problem):
            self.create('no-storage')
        orphan.unlink();orphan.symlink_to('/dev/null')
        with self.assertRaises(Problem):self.evaluations.disk_bytes()

    def test_failure_is_terminal_and_replay_does_not_execute(self):
        item=self.create(); self.evaluations.claim()
        self.evaluations.fail(item['id'],'evaluation_failed')
        self.assertEqual(self.create()['status'],'failed')
        self.assertIsNone(self.evaluations.claim())
        self.assertNotEqual(self.create('explicit-rerun')['id'],item['id'])

    def test_record_and_response_budgets(self):
        item=self.create()
        with mock.patch.object(module,'MAX_RECORDS',1),self.assertRaises(Problem):self.create('over')
        with mock.patch.object(module,'MAX_DETAIL_BYTES',1),self.assertRaises(Problem):self.evaluations.get('demo',item['id'])
        with mock.patch.object(module,'MAX_LIST_BYTES',1),self.assertRaises(Problem):self.evaluations.list('demo')

    def test_post_init_evidence_root_symlink_cannot_redirect_publication(self):
        item=self.create();row=self.evaluations.claim()
        original=self.evaluations.root
        parked=original.with_name('evidence-parked');original.rename(parked)
        outside=self.runtime.home/'outside';outside.mkdir()
        original.symlink_to(outside,target_is_directory=True)
        with self.assertRaises(Problem) as error:self.evaluations.disk_bytes()
        self.assertEqual(error.exception.code,'evaluation_integrity')
        result={'digests':dict(self.evaluations.expected_digests(self.binding,digest(self.binding)),
                               prediction_sha256=digest([]))}
        with self.assertRaises(Problem):self.evaluations.succeed(row,result,[])
        self.assertEqual(list(outside.iterdir()),[])
        with self.runtime.connection() as db:
            self.assertEqual(db.execute('SELECT status FROM model_evaluations WHERE id=?',(item['id'],)).fetchone()[0],'running')

    def test_evidence_symlink_and_fifo_are_rejected(self):
        item=self.succeed(); path=self.evaluations.evidence_path(item['id'])
        path.unlink(); path.symlink_to('/dev/null')
        with self.assertRaises(Problem):self.evaluations.get('demo',item['id'])
        path.unlink(); os.mkfifo(path)
        started=time.monotonic()
        with self.assertRaises(Problem):self.evaluations.get('demo',item['id'])
        self.assertLess(time.monotonic()-started,1)


class GeometryTests(unittest.TestCase):
    def test_resize_floor_and_pixel_identity(self):
        try:import numpy as np; import cv2
        except ImportError:self.skipTest('optional numpy/OpenCV missing')
        rgb=np.arange(640*246*3,dtype=np.uint8).reshape(640,246,3)
        resized=module.resize_request(rgb)
        self.assertEqual(resized.shape,(384,147,3))
        self.assertTrue(np.array_equal(resized,cv2.resize(rgb,(147,384),interpolation=cv2.INTER_LINEAR)))
        small=rgb[:10,:20]; self.assertTrue(np.array_equal(module.resize_request(small),small))
        self.assertNotEqual(module._pixel_digest(rgb),module._pixel_digest(resized))

    def test_diagnostics_bounded_without_dropping_metric_counts(self):
        sample={'image_id':'images/val/x.jpg','width':100,'height':100,'tp':0,'fp':30,'fn':30,
                'false_positive_prediction_indices':list(range(30)), 'false_negative_annotation_indices':list(range(30)),
                'predictions':[{'index':i,'category_id':0,'score':.5,'bbox':[0,0,1,1]} for i in range(30)],
                'annotations':[{'index':i,'category_id':0,'bbox':[0,0,1,1]} for i in range(30)]}
        result=module._diagnostics({'wrong_examples':[sample]*20,'wrong_examples_total':64})
        self.assertTrue(result['truncated']); self.assertEqual(len(result['items']),20)
        first=result['items'][0]
        self.assertEqual(first['fp'],30); self.assertEqual(len(first['false_positives']),20)
        self.assertTrue(first['false_positives_truncated']); self.assertTrue(first['false_negatives_truncated'])


if __name__=='__main__':unittest.main()
