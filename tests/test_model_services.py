"""Real SQLite/concurrency with explicitly fake serving, plus optional real ORT HTTP."""
import copy
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from model_platform.common import Problem, Runtime, encode, now
from model_platform.store import Store, identifier
from model_platform.model_services import ServiceManager
from model_platform import dataset_serving, dataset_detection
from scripts.demo import request_body


class FakeSupervisor:
    """No actual model inference: deterministic lifecycle/locking test double."""
    def __init__(self, store):
        self.store, self.lock, self.instances = store, threading.RLock(), {}
        self.queries = 0
        self.entered = self.release = None

    def add(self, deployment):
        row = self.store.deployment('demo', deployment)
        self.instances[deployment] = SimpleNamespace(deployment_id=deployment,
            manifest=self.store.model('demo', row['model_version_id']))
        self.store.observe(deployment, 'ready')

    def query(self, tenant, deployment):
        with self.lock:
            self.queries += 1
            return self.store.deployment(tenant, deployment)

    def predict(self, tenant, deployment, features):
        with self.lock:
            row = self.query(tenant, deployment)
            if not row['ready'] or deployment not in self.instances:
                raise Problem(503, 'not_ready', 'Fake unavailable')
            if self.entered:
                self.entered.set()
                if not self.release.wait(5):
                    raise RuntimeError('test synchronization timed out')
            return dict(deployment_id=deployment, model_version_id=row['model_version_id'],
                        contract=row['contract'], predictions=[], **dataset_serving.public_identity(self.instances[deployment].manifest))

    def stop(self, deployment):
        with self.lock:
            self.store.stop_desired('demo', deployment)
            self.store.observe(deployment, 'stopped')
            self.instances.pop(deployment, None)


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.runtime = Runtime(self.temp.name)
        self.store = Store(self.runtime)
        self.supervisor = FakeSupervisor(self.store)
        self.services = ServiceManager(self.store, self.supervisor)
        self.a = self.deployment()
        self.b = self.deployment()

    def deployment(self, names=None, tenant='demo'):
        run_id, _ = self.store.submit(tenant, identifier('key'), request_body())
        run, attempt = self.store.claim()
        model, evaluation = identifier('model'), identifier('evaluation')
        manifest = dict(model_version_id=model, run_id=run_id, attempt_id=attempt,
            evaluation_id=evaluation, artifact_sha256='a'*64, recipe=dataset_detection.RECIPE,
            contract=dataset_detection.CONTRACT, format=dataset_detection.FORMAT,
            class_names=names or ['a','b'], dataset=dict(dataset_version_id='dsv_'+'a'*32,
                sha256='b'*64, train_sha256='c'*64, evaluation_sha256='d'*64, evaluation_split='val'))
        with self.runtime.connection() as db:
            db.execute('INSERT INTO evaluations VALUES(?,?,?,?,?,?)', (evaluation, attempt, 'a'*64, 'd'*64, '{}', now()))
            db.execute('INSERT INTO models(id,tenant,run_id,attempt_id,evaluation_id,artifact_sha256,manifest_json,created_at) VALUES(?,?,?,?,?,?,?,?)',
                       (model, tenant, run_id, attempt, evaluation, 'a'*64, encode(manifest).decode(), now()))
            db.execute("UPDATE runs SET status='succeeded',model_version_id=? WHERE id=?", (model, run_id))
            db.execute("UPDATE attempts SET status='succeeded' WHERE id=?", (attempt,))
        deployment = self.store.create_deployment(tenant, model)
        if tenant == 'demo':
            self.supervisor.add(deployment)
        return deployment

    def create(self, key='create', target=None):
        return self.services.create('demo', key, {'deployment_id': target or self.a})[0]

    def switch(self, snapshot, target, key='switch'):
        return self.services.switch('demo', snapshot['id'], key,
            {'expected_revision': snapshot['revision'], 'target_deployment_id': target})[0]

    def assert_problem(self, code, callable, *args):
        with self.assertRaises(Problem) as error:
            callable(*args)
        self.assertEqual(error.exception.code, code)

    def test_history_replay_survives_switch_stop_and_missing_live_instance(self):
        first = self.create()
        second = self.switch(first, self.b)
        third = self.switch(second, self.a, 'rollback')
        self.supervisor.stop(self.a)
        self.supervisor.stop(self.b)
        queries = self.supervisor.queries
        self.assertEqual(self.services.create('demo','create',{'deployment_id':self.a}), (first,False))
        self.assertEqual(self.services.switch('demo',first['id'],'switch',
            {'expected_revision':1,'target_deployment_id':self.b}), (second,False))
        self.assertEqual(self.supervisor.queries, queries)
        self.assertEqual(third['revision'],3)
        self.assertEqual(third['previous_deployment_id'],self.b)
        self.assertFalse(self.services.query('demo',first['id'])['ready'])
        self.assert_problem('not_ready', self.services.predict, 'demo',first['id'],{})
        with self.runtime.connection() as db:
            self.assertEqual(db.execute("SELECT count(*) FROM audit WHERE action IN ('create_service','switch_service')").fetchone()[0],3)

    def test_namespace_validation_ownership_and_compatibility(self):
        first = self.create()
        self.assert_problem('idempotency_conflict', self.services.create, 'demo','create',{'deployment_id':self.b})
        self.assert_problem('idempotency_conflict', self.services.switch, 'demo',first['id'],'create',{'expected_revision':1,'target_deployment_id':self.b})
        self.assert_problem('not_found',self.services.switch,'other',first['id'],None,None)
        for body in ({}, {'deployment_id':True}, {'deployment_id':self.a,'extra':0}):
            self.assert_problem('invalid_input',self.services.create,'demo','invalid',body)
        for revision in (True,0,9007199254740992,1.0):
            self.assert_problem('invalid_input',self.services.switch,'demo',first['id'],'invalid',{'expected_revision':revision,'target_deployment_id':self.b})
        different = self.deployment(['b','a'])
        self.assert_problem('service_incompatible',self.switch,first,different)
        self.assert_problem('service_conflict',self.switch,first,self.a)
        self.assertEqual(self.services.query('demo',first['id'])['revision'],1)

    def test_concurrent_cas_has_one_winner_across_managers(self):
        first = self.create()
        other = ServiceManager(Store(self.runtime),self.supervisor)
        gate = threading.Barrier(2)
        def submit(manager,key):
            gate.wait()
            try:
                return manager.switch('demo',first['id'],key,{'expected_revision':1,'target_deployment_id':self.b})[0]['revision']
            except Problem as error:
                return error.code
        with ThreadPoolExecutor(2) as pool:
            futures=[pool.submit(submit,m,k) for m,k in [(self.services,'x'),(other,'y')]]
            self.assertCountEqual([f.result(5) for f in futures],[2,'service_conflict'])

    def test_predict_holds_binding_until_response_and_switch_waits(self):
        first=self.create()
        self.supervisor.entered,self.supervisor.release=threading.Event(),threading.Event()
        with ThreadPoolExecutor(2) as pool:
            predicting=pool.submit(self.services.predict,'demo',first['id'],{})
            self.assertTrue(self.supervisor.entered.wait(3))
            switching=pool.submit(self.switch,first,self.b)
            self.assertFalse(switching.done())
            self.supervisor.release.set()
            result=predicting.result(5)
            self.assertEqual((result['service_revision'],result['deployment_id']),(1,self.a))
            self.assertEqual(switching.result(5)['revision'],2)

    def test_stop_serializes_after_admission_and_replay_never_probes(self):
        entered,release=threading.Event(),threading.Event()
        original=self.supervisor.query
        def blocked(*args):
            result=original(*args);entered.set()
            self.assertTrue(release.wait(5))
            return result
        with patch.object(self.supervisor,'query',blocked), ThreadPoolExecutor(2) as pool:
            creating=pool.submit(self.create)
            self.assertTrue(entered.wait(3))
            stopping=pool.submit(self.supervisor.stop,self.a)
            self.assertFalse(stopping.done())
            release.set();first=creating.result(5);stopping.result(5)
        self.assertFalse(self.services.query('demo',first['id'])['ready'])
        with patch.object(self.supervisor,'query',side_effect=AssertionError('replay must not probe')):
            self.assertEqual(self.create(),first)

    def test_no_write_transaction_during_probe_and_atomic_rollback(self):
        original=self.supervisor.query
        def probe(*args):
            with self.runtime.connection() as db:
                db.execute('BEGIN IMMEDIATE')
            return original(*args)
        with patch.object(self.supervisor,'query',probe), patch.object(self.store,'audit',side_effect=RuntimeError('crash before commit')):
            with self.assertRaises(RuntimeError):self.create()
        with self.runtime.connection() as db:
            for table in ('model_services','model_service_keys','model_service_revisions'):
                self.assertEqual(db.execute('SELECT count(*) FROM '+table).fetchone()[0],0)
        self.assertEqual(self.create()['revision'],1)

    def test_missing_instance_stale_ready_and_inference_identity_rejected(self):
        first=self.create()
        self.supervisor.instances.pop(self.b)
        self.assert_problem('not_ready',self.switch,first,self.b)
        with patch.object(self.supervisor,'predict',return_value={'deployment_id':self.b}):
            self.assert_problem('not_ready',self.services.predict,'demo',first['id'],{})

    def test_protocol_change_cannot_claim_ready_or_switch_but_old_key_replays(self):
        first=self.create()
        protocol=dataset_serving.protocol()
        protocol['score_threshold'] += .01
        with patch.object(dataset_serving,'protocol',return_value=protocol):
            self.assert_problem('service_integrity',self.services.query,'demo',first['id'])
            self.assert_problem('service_incompatible',self.switch,first,self.b)
            self.assert_problem('not_ready',self.services.predict,'demo',first['id'],{})
            self.assertEqual(self.create(),first)

    def test_live_manifest_mismatch_rejects_admission_and_readiness(self):
        self.supervisor.instances[self.b].manifest['artifact_sha256']='f'*64
        self.assert_problem('service_integrity',self.create,'bad',self.b)
        first=self.create()
        self.supervisor.instances[self.a].manifest['class_names']=['b','a']
        self.assert_problem('service_integrity',self.services.query,'demo',first['id'])

    def test_candidate_state_rechecked_in_commit_transaction(self):
        original=self.services._candidate
        def candidate(*args):
            result=original(*args)
            # Simulates an unsupported out-of-band DB writer between probe/CAS.
            self.store.stop_desired('demo',self.a)
            return result
        with patch.object(self.services,'_candidate',candidate):
            self.assert_problem('not_ready',self.create)
        with self.runtime.connection() as db:
            self.assertEqual(db.execute('SELECT count(*) FROM model_service_keys').fetchone()[0],0)

    def test_replay_validates_saved_snapshot_without_current_model_reads(self):
        first=self.create()
        self.switch(first,self.b)
        with patch.object(self.store,'model',side_effect=AssertionError('no current model dependency')):
            self.assertEqual(self.create(),first)
        with self.runtime.connection() as db:
            db.execute('DROP TRIGGER model_service_revisions_update_immutable')
            bad=dict(first,ready=True)
            db.execute('UPDATE model_service_revisions SET snapshot_json=? WHERE service_id=? AND revision=1',
                       (encode(bad).decode(),first['id']))
        self.assert_problem('service_integrity',self.create)

    def test_concurrent_same_key_creates_once(self):
        gate=threading.Barrier(2)
        def create():
            gate.wait()
            return self.services.create('demo','same',{'deployment_id':self.a})
        with ThreadPoolExecutor(2) as pool:
            futures=[pool.submit(create) for _ in range(2)]
            results=[f.result(5) for f in futures]
        self.assertEqual(results[0][0],results[1][0])
        self.assertCountEqual([r[1] for r in results],[True,False])

    def test_signed_pagination_frozen_membership_and_tenant(self):
        first=self.create()
        second=self.create('second')
        page=self.services.list_page('demo',1)
        third=self.create('third')
        following=self.services.list_page('demo',cursor=page['next_cursor'])
        self.assertEqual(page['ids']+following['ids'],[second['id'],first['id']])
        self.assertNotIn(third['id'],following['ids'])
        self.assertEqual(self.services.list_ids('other'),[])
        self.assert_problem('invalid_cursor',self.services.list_page,'other',None,page['next_cursor'])


# Reuse only fixture/lifecycle helpers, not its inherited test methods.
import test_dataset_serving
@unittest.skipUnless(test_dataset_serving.OPTIONAL, 'Optional ONNX dependencies absent')
class ServiceORTTests(unittest.TestCase):
    setUp = test_dataset_serving.DatasetServingTests.setUp
    spawn = test_dataset_serving.DatasetServingTests.spawn
    start_api = test_dataset_serving.DatasetServingTests.start_api
    cleanup = test_dataset_serving.DatasetServingTests.cleanup
    call = test_dataset_serving.DatasetServingTests.call
    ready = test_dataset_serving.DatasetServingTests.ready
    publish = test_dataset_serving.DatasetServingTests.publish
    synthetic_result = test_dataset_serving.DatasetServingTests.synthetic_result
    deploy = test_dataset_serving.DatasetServingTests.deploy

    def test_real_onnx_a_b_a_restart_stop_and_original_endpoint(self):
        from model_platform.detection import probe_input
        a,b=self.deploy('a'),self.deploy('b')
        self.ready(a);self.ready(b)
        status,first=self.call('/services',{'deployment_id':a},'create')
        self.assertEqual(status,201,first)
        endpoint=first['endpoint']
        for revision,target in [(2,b),(3,a)]:
            status,snapshot=self.call('/services/'+first['id']+'/switch',
                {'expected_revision':revision-1,'target_deployment_id':target},'switch'+str(revision))
            self.assertEqual(status,200,snapshot)
            status,result=self.call(endpoint,{'features':probe_input()})
            self.assertEqual(status,200,result)
            self.assertEqual((result['deployment_id'],result['service_revision']),(target,revision))
        self.api.terminate();self.api.wait(timeout=10);self.start_api();self.ready(a)
        self.assertEqual(self.call(endpoint,{'features':probe_input()})[1]['service_revision'],3)
        self.assertEqual(self.call('/services',{'deployment_id':a},'create'),(200,first))
        self.assertEqual(self.call('/deployments/'+a+'/predict',{'features':probe_input()})[0],200)
        self.call('/deployments/'+a+'/stop',{})
        self.assertEqual(self.call(endpoint,{'features':probe_input()})[0],503)
