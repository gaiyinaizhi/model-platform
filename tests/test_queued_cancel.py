"""Real isolated SQLite/loopback HTTP cancellation; no credentials or ML assets."""
import collections
import concurrent.futures
import hashlib
import http.client
import json
import multiprocessing
from pathlib import Path
import sqlite3
import tempfile
import threading
import unittest
from unittest import mock

from model_platform.common import Runtime, Problem
from model_platform.store import Store
from model_platform.model_evaluations import active_count, ModelEvaluationStore
from model_platform.worker_readiness import public_report
from scripts.demo import request_body


def competing_operation(home, run_id, barrier, results, operation):
    # Separate processes/connections; fixed process-local cursor material only.
    import model_platform.store
    model_platform.store.cursor_key = lambda *_: b'queued-cancel-test-fixture'
    store = Store(Runtime(home))
    barrier.wait(timeout=10)
    try:
        if operation == 'claim':
            value = store.claim()
            results.put((operation, 'claimed' if value else 'empty'))
        elif operation == 'retry':
            store.retry('demo', run_id)
            results.put((operation, 'ok'))
        else:
            store.cancel('demo', run_id)
            results.put((operation, 'ok'))
    except Problem as exc:
        results.put((operation, exc.code))


class CancelTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        patch = mock.patch('model_platform.store.cursor_key', return_value=b'queued-cancel-test-fixture')
        patch.start()
        self.addCleanup(patch.stop)
        self.runtime = Runtime(self.temp.name)
        self.store = Store(self.runtime)

    def submit(self, key='one', tenant='demo'):
        return self.store.submit(tenant, key, request_body())[0]

    def test_cancel_is_terminal_retains_identity_and_replay(self):
        rid = self.submit()
        before = self.store.run('demo', rid)
        self.store.cancel('demo', rid)
        after = self.store.run('demo', rid)
        self.assertEqual(after['status'], 'cancelled')
        self.assertIsNone(after['queue']['position'])
        self.assertIsNone(after['model_version_id'])
        self.assertEqual(after['attempts'], [])
        for key in ('request_sha256', 'environment', 'dataset', 'config', 'created_at'):
            self.assertEqual(before[key], after[key])
        self.store.cancel('demo', rid)
        self.assertEqual(self.store.run('demo', rid), after)
        self.assertEqual(self.store.submit('demo', 'one', request_body()), (rid, False))
        changed = request_body()
        changed['config']['epochs'] -= 1
        with self.assertRaises(Problem) as ctx:
            self.store.submit('demo', 'one', changed)
        self.assertEqual(ctx.exception.code, 'idempotency_conflict')
        with self.assertRaises(Problem) as ctx:
            self.store.retry('demo', rid)
        self.assertEqual(ctx.exception.code, 'retry_conflict')
        with self.runtime.connection() as db:
            self.assertEqual(db.execute("SELECT count(*) FROM audit WHERE action='cancel_queued'").fetchone()[0], 1)
            self.assertEqual(db.execute('SELECT count(*) FROM attempts').fetchone()[0], 0)
            self.assertEqual(db.execute('SELECT count(*) FROM models').fetchone()[0], 0)
            self.assertEqual(active_count(db), 0)
        self.assertEqual(public_report(self.store, 'demo')['queue'], {'queued': 0, 'running': 0})
        self.assertEqual(self.store.list_ids('runs', 'demo'), [rid])
        self.assertEqual(list(self.runtime.artifacts.iterdir()), [])

    def test_foreign_missing_and_terminal_conflicts(self):
        rid = self.submit()
        for target in (rid, 'run_missing'):
            with self.assertRaises(Problem) as ctx:
                self.store.cancel('other', target)
            self.assertEqual((ctx.exception.status, ctx.exception.code), (404, 'not_found'))
        run, attempt = self.store.claim()
        with self.assertRaises(Problem) as ctx:
            self.store.cancel('demo', rid)
        self.assertEqual((ctx.exception.status, ctx.exception.code), (409, 'cancel_conflict'))
        self.assertIn('Running', ctx.exception.message)
        self.assertEqual(self.store.run('demo', rid)['attempts'][0]['status'], 'running')
        self.store.fail(rid, attempt, 'test_failure')
        with self.assertRaises(Problem):
            self.store.cancel('demo', rid)
        other = self.submit('success')
        run, attempt = self.store.claim()
        self.store.register(run, attempt, 'a'*64, {'dataset_sha256': 'b'*64}, {'contract': 'fixture', 'format': 'fixture'})
        with self.assertRaises(Problem):
            self.store.cancel('demo', other)
        self.assertEqual(self.store.run('demo', other)['status'], 'succeeded')

    def test_retry_queue_cancellation_preserves_failed_attempts(self):
        rid = self.submit()
        run, attempt = self.store.claim()
        self.store.fail(rid, attempt, 'test_failure')
        history = self.store.run('demo', rid)['attempts']
        self.store.retry('demo', rid)
        self.store.cancel('demo', rid)
        self.assertEqual(self.store.run('demo', rid)['attempts'], history)
        self.assertIsNone(self.store.claim())
        self.assertEqual(self.store.recover_worker(), 0)
        restarted = Store(self.runtime)
        self.assertEqual(restarted.run('demo', rid)['status'], 'cancelled')
        self.assertIsNone(restarted.claim())

    def test_released_capacity_accepts_new_work_and_order(self):
        from model_platform import store as module
        first = self.submit()
        second = self.submit('two')
        with mock.patch.object(module, 'MAX_ACTIVE_RUNS', 2):
            with self.assertRaises(Problem) as ctx:
                self.submit('three')
            self.assertEqual(ctx.exception.code, 'run_capacity')
            self.store.cancel('demo', first)
            third = self.submit('three')
        self.assertEqual(self.store.run('demo', second)['queue']['position'], 1)
        self.assertEqual(self.store.run('demo', third)['queue']['position'], 2)
        self.assertEqual(self.store.claim()[0]['id'], second)

    def race(self, rid, operations):
        ctx = multiprocessing.get_context('spawn')
        barrier = ctx.Barrier(len(operations))
        results = ctx.Queue()
        processes = [ctx.Process(target=competing_operation,
                                 args=(str(self.runtime.home), rid, barrier, results, op))
                     for op in operations]
        try:
            for process in processes:
                process.start()
            output = [results.get(timeout=20) for _ in processes]
            for process in processes:
                process.join(timeout=20)
                self.assertEqual(process.exitcode, 0)
            return output
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5)
            results.close()

    def test_multiprocess_cancel_claim_race_has_one_winner(self):
        for index in range(6):
            rid = self.submit(str(index))
            result = dict(self.race(rid, ['cancel', 'claim']))
            row = self.store.run('demo', rid)
            if result['cancel'] == 'ok':
                self.assertEqual(result['claim'], 'empty')
                self.assertEqual((row['status'], row['attempts']), ('cancelled', []))
            else:
                self.assertEqual(result, {'cancel': 'cancel_conflict', 'claim': 'claimed'})
                self.assertEqual(row['status'], 'running')
                self.assertEqual(len(row['attempts']), 1)
                self.store.fail(rid, row['attempts'][0]['id'], 'test_cleanup')

    def test_concurrent_duplicate_cancel_updates_audit_once(self):
        rid = self.submit()
        self.assertEqual([r[1] for r in self.race(rid, ['cancel']*4)], ['ok']*4)
        with self.runtime.connection() as db:
            self.assertEqual(db.execute("SELECT count(*) FROM audit WHERE action='cancel_queued'").fetchone()[0], 1)

    def test_retry_cancel_race_is_serialized(self):
        rid = self.submit()
        run, attempt = self.store.claim()
        self.store.fail(rid, attempt, 'test_failure')
        result = dict(self.race(rid, ['retry', 'cancel']))
        self.assertEqual(result['retry'], 'ok')
        row = self.store.run('demo', rid)
        self.assertEqual(row['status'], 'cancelled' if result['cancel'] == 'ok' else 'queued')
        self.assertEqual(len(row['attempts']), 1)
        self.assertEqual(row['attempts'][0]['status'], 'failed')

    def test_api_auth_body_state_and_capability(self):
        from model_platform.api import Auth, make_handler
        from model_platform.http_support import BoundedHTTPServer
        # Real Auth path with fixed memory-only fixture values, never a key file.
        auth = Auth.__new__(Auth)
        tokens = {'demo': '1'*64, 'other': '2'*64}
        auth.tokens = {hashlib.sha256(v.encode()).digest(): k for k, v in tokens.items()}
        auth.windows = collections.defaultdict(collections.deque)
        auth.lock = threading.Lock()
        server = BoundedHTTPServer(('127.0.0.1', 0), make_handler(self.store, None, auth))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        self.addCleanup(lambda: thread.join(timeout=2) if not thread.is_alive() else None)
        def call(path, raw=None, tenant='demo', method=None):
            headers = {'Content-Type': 'application/json'}
            if tenant:
                headers['Authorization'] = 'Bearer ' + tokens[tenant]
            conn = http.client.HTTPConnection(*server.server_address, timeout=5)
            try:
                conn.request(method or ('POST' if raw is not None else 'GET'), path, raw, headers)
                response = conn.getresponse()
                return response.status, json.loads(response.read())
            finally:
                conn.close()
        health = call('/health')[1]
        self.assertFalse(health['cancel_supported'])
        self.assertTrue(health['queued_cancel_supported'])
        rid = self.submit()
        path = '/runs/' + rid + '/cancel'
        self.assertEqual(call(path, b'{}', None)[0], 401)
        self.assertEqual(call(path, b'{"bad":1}', 'other')[0], 404)
        for raw in (b'{"reason":"x"}', b'{"x":1,"x":2}', b'null', b'[]'):
            self.assertEqual(call(path, raw)[0], 400)
        self.assertEqual(call(path, b'{}')[0], 200)
        self.assertEqual(call(path, b'{}')[1]['status'], 'cancelled')
        self.assertEqual(call('/runs/'+rid)[1]['status'], 'cancelled')
        self.assertEqual(call('/runs/'+rid+'/retry', b'{}')[0], 409)
        running = self.submit('running-http')
        run, attempt = self.store.claim()
        status, error = call('/runs/'+running+'/cancel', b'{}')
        self.assertEqual((status, error['error']['code']), (409, 'cancel_conflict'))
        self.assertIn('Running', error['error']['message'])
        self.store.fail(running, attempt, 'test_failure')
        self.assertEqual(call('/runs/'+running+'/cancel', b'{}')[0], 409)
        succeeded = self.submit('succeeded-http')
        run, attempt = self.store.claim()
        self.store.register(run, attempt, 'a'*64, {'dataset_sha256': 'b'*64},
                            {'contract': 'fixture', 'format': 'fixture'})
        self.assertEqual(call('/runs/'+succeeded+'/cancel', b'{}')[0], 409)
        self.assertFalse((self.runtime.home/'credentials.json').exists())
        self.assertFalse((self.runtime.home/'cursor-integrity.key').exists())


if __name__ == '__main__':
    unittest.main()
