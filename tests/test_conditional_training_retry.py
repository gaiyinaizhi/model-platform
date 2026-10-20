"""Isolated SQLite/HTTP retry mechanics; synthetic records are not model quality."""
import collections
import hashlib
import http.client
import json
import multiprocessing
import socket
import tempfile
import threading
import unittest
from unittest import mock

from model_platform.common import Problem, Runtime, environment
from model_platform.store import Store
from scripts.demo import request_body


def competing_retry(home, run_id, expected, operation, barrier, results):
    import model_platform.store
    model_platform.store.cursor_key = lambda *_: b'conditional-retry-test-fixture'
    store = Store(Runtime(home))
    barrier.wait(timeout=10)
    try:
        if operation == 'retry':
            store.retry('demo', run_id, expected)
            outcome = 'ok'
        elif operation == 'claim':
            outcome = 'claimed' if store.claim() else 'empty'
        elif operation == 'recover':
            outcome = store.recover_worker()
        else:
            store.cancel('demo', run_id)
            outcome = 'ok'
        results.put((operation, outcome))
    except Problem as exc:
        results.put((operation, exc.code))


class RetryFixture:
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        patch = mock.patch('model_platform.store.cursor_key', return_value=b'conditional-retry-test-fixture')
        patch.start()
        self.addCleanup(patch.stop)
        self.runtime = Runtime(self.temp.name)
        self.store = Store(self.runtime)
        self.sequence = 0

    def submit(self, tenant='demo', body=None):
        self.sequence += 1
        return self.store.submit(tenant, str(self.sequence), body or request_body())[0]

    def failed(self, tenant='demo', body=None):
        rid = self.submit(tenant, body)
        run, attempt = self.store.claim()
        self.assertEqual(run['id'], rid)
        self.store.log(attempt, 'synthetic-test-only', {'scope': 'retry mechanics'})
        self.store.fail(rid, attempt, 'fixture_failure')
        return rid, attempt

    def snapshot(self):
        with self.runtime.connection() as db:
            return tuple(db.iterdump())

    def rejected(self, rid, expected, code='retry_conflict', status=409, tenant='demo'):
        before = self.snapshot()
        with self.assertRaises(Problem) as error:
            self.store.retry(tenant, rid, expected)
        self.assertEqual((error.exception.status, error.exception.code), (status, code))
        self.assertEqual(self.snapshot(), before)

    def retry_audits(self, rid):
        with self.runtime.connection() as db:
            return db.execute("SELECT count(*) FROM audit WHERE action='retry' AND object_id=?", (rid,)).fetchone()[0]

    def register_fixture(self, run, attempt):
        return self.store.register(run, attempt, 'a'*64, {'dataset_sha256': 'b'*64},
                                   {'contract': 'synthetic-fixture', 'format': 'synthetic-fixture'})

    def race(self, rid, expected, operations):
        context = multiprocessing.get_context('spawn')
        barrier = context.Barrier(len(operations))
        results = context.Queue()
        processes = [context.Process(target=competing_retry,
                     args=(str(self.runtime.home), rid, expected, operation, barrier, results))
                     for operation in operations]
        try:
            for process in processes:
                process.start()
            outcomes = [results.get(timeout=20) for _ in processes]
            for process in processes:
                process.join(timeout=20)
                self.assertEqual(process.exitcode, 0)
            return outcomes
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5)
            results.close()


class ConditionalRetryTests(RetryFixture, unittest.TestCase):
    def test_queues_same_run_preserves_inputs_binding_history_and_schema(self):
        rid, expected = self.failed()
        # An inert binding row lets this standard-library test check preservation;
        # actual dataset-version provenance is covered by test_dataset_training.
        with self.runtime.connection() as db:
            db.execute("INSERT INTO run_dataset_bindings VALUES(?,?,?,?)",
                       (rid, 'dsv_'+'a'*32, 'b'*64, '{"scope":"synthetic-test-only"}'))
            before_row = dict(db.execute('SELECT * FROM runs WHERE id=?', (rid,)).fetchone())
            schema = list(db.execute('SELECT type,name,sql FROM sqlite_master ORDER BY name'))
            binding = list(db.execute('SELECT * FROM run_dataset_bindings'))
            migrations = list(db.execute('SELECT * FROM schema_migrations'))
        before = self.store.run('demo', rid)
        self.store.retry('demo', rid, expected)
        after = self.store.run('demo', rid)
        self.assertEqual(after['status'], 'queued')
        self.assertEqual(after['attempts'], before['attempts'])
        self.assertEqual(self.retry_audits(rid), 1)
        with self.runtime.connection() as db:
            row = dict(db.execute('SELECT * FROM runs WHERE id=?', (rid,)).fetchone())
            for key in before_row.keys() - {'status', 'updated_at'}:
                self.assertEqual(row[key], before_row[key], key)
            self.assertEqual(list(db.execute('SELECT * FROM run_dataset_bindings')), binding)
            self.assertEqual(list(db.execute('SELECT * FROM schema_migrations')), migrations)
            self.assertEqual(list(db.execute('SELECT type,name,sql FROM sqlite_master ORDER BY name')), schema)
            self.assertEqual(db.execute('SELECT count(*) FROM runs').fetchone()[0], 1)
            self.assertEqual(db.execute('SELECT count(*) FROM models').fetchone()[0], 0)
        _, attempt = self.store.claim()
        self.assertNotEqual(attempt, expected)
        self.assertEqual([a['number'] for a in self.store.run('demo', rid)['attempts']], [1, 2])

    def test_replay_after_fast_failure_cannot_enqueue_third_attempt(self):
        rid, expected = self.failed()
        self.store.retry('demo', rid, expected)
        self.rejected(rid, expected)
        run, second = self.store.claim()
        self.rejected(rid, expected)
        self.store.fail(rid, second, 'second_fixture_failure')
        self.rejected(rid, expected)
        self.assertEqual(self.retry_audits(rid), 1)
        self.assertEqual(len(self.store.run('demo', rid)['attempts']), 2)
        self.assertIsNone(self.store.claim())
        self.store.retry('demo', rid, second)
        _, third = self.store.claim()
        self.store.fail(rid, third, 'third_fixture_failure')
        self.rejected(rid, third)
        self.assertEqual(self.retry_audits(rid), 2)
        self.assertEqual(len(self.store.run('demo', rid)['attempts']), 3)

    def test_cancelled_and_succeeded_replay_remain_terminal(self):
        for terminal in ('cancelled', 'succeeded'):
            with self.subTest(terminal=terminal):
                rid, expected = self.failed()
                self.store.retry('demo', rid, expected)
                if terminal == 'cancelled':
                    self.store.cancel('demo', rid)
                else:
                    self.register_fixture(*self.store.claim())
                self.rejected(rid, expected)
                self.assertEqual(self.store.run('demo', rid)['status'], terminal)
                self.assertIsNone(self.store.claim())

    def test_foreign_missing_and_other_run_attempts_do_not_disclose_or_write(self):
        rid, expected = self.failed()
        other_run, other_attempt = self.failed()
        foreign_run, foreign_attempt = self.failed('other')
        for attempt in (other_attempt, foreign_attempt, 'attempt_'+'0'*32):
            self.rejected(rid, attempt)
        self.rejected(rid, expected, 'not_found', 404, 'other')
        self.rejected('run_'+'0'*32, expected, 'not_found', 404)
        self.rejected(foreign_run, foreign_attempt, 'not_found', 404)
        self.assertEqual(self.store.run('demo', other_run)['status'], 'failed')

    def test_state_and_history_checks_precede_source_and_capacity(self):
        rid, expected = self.failed()
        self.store.retry('demo', rid, expected)
        with mock.patch('model_platform.store.environment', side_effect=AssertionError('must not hash source')):
            self.rejected(rid, expected)
            self.rejected(rid, expected, 'not_found', 404, 'other')

    def test_failed_without_attempt_and_incoherent_history_fail_closed(self):
        for corruption in ('empty', 'gap', 'zero', 'fraction', 'running', 'succeeded', 'earlier_running', 'model'):
            with self.subTest(corruption=corruption):
                rid, expected = self.failed()
                if corruption == 'earlier_running':
                    self.store.retry('demo', rid, expected)
                    _, expected = self.store.claim()
                    self.store.fail(rid, expected, 'fixture_failure')
                with self.runtime.connection() as db:
                    if corruption == 'empty':
                        db.execute('DELETE FROM logs WHERE attempt_id=?', (expected,))
                        db.execute('DELETE FROM attempts WHERE id=?', (expected,))
                    elif corruption in ('gap', 'zero', 'fraction'):
                        db.execute('UPDATE attempts SET number=? WHERE id=?',
                                   ({'gap': 2, 'zero': 0, 'fraction': 1.5}[corruption], expected))
                    elif corruption == 'model':
                        db.execute('UPDATE runs SET model_version_id=? WHERE id=?', ('model_'+'a'*32, rid))
                    elif corruption == 'earlier_running':
                        db.execute("UPDATE attempts SET status='running' WHERE run_id=? AND number=1", (rid,))
                    else:
                        db.execute('UPDATE attempts SET status=? WHERE id=?', (corruption, expected))
                self.rejected(rid, expected)

    def test_fingerprint_rejection_is_read_only_and_legacy_behavior_unchanged(self):
        rid, expected = self.failed()
        changed = dict(environment(), code_sha256='0'*64)
        with mock.patch('model_platform.store.environment', return_value=changed):
            self.rejected(rid, expected, 'retry_code_changed')
            # Old direct/API {} clients retain their original (unguarded) behavior.
            self.store.retry('demo', rid)
        self.assertEqual(self.store.run('demo', rid)['status'], 'queued')
        self.assertEqual(self.retry_audits(rid), 1)

    def test_restored_source_can_retry_prior_execution_code_changed_failure(self):
        rid, expected = self.failed()
        with self.runtime.connection() as db:
            db.execute("UPDATE attempts SET error_code='execution_code_changed' WHERE id=?", (expected,))
        self.store.retry('demo', rid, expected)
        self.assertEqual(self.store.run('demo', rid)['status'], 'queued')

    def test_worker_loaded_source_guard_is_not_bypassed(self):
        from model_platform import worker
        rid, expected = self.failed()
        self.store.retry('demo', rid, expected)
        run, attempt = self.store.claim()
        with mock.patch.object(worker, 'EXECUTION_ENVIRONMENT', dict(environment(), code_sha256='0'*64)):
            worker.execute(self.store, run, attempt)
        row = self.store.run('demo', rid)
        self.assertEqual(row['status'], 'failed')
        self.assertEqual(row['attempts'][-1]['error_code'], 'execution_code_changed')
        self.assertIsNone(row['model_version_id'])

    def test_training_capacity_rejection_and_code_drift_precedence(self):
        rid, expected = self.failed()
        for _ in range(32):
            self.submit()
        self.rejected(rid, expected, 'run_capacity', 429)
        with mock.patch('model_platform.store.environment', return_value=dict(environment(), code_sha256='0'*64)):
            self.rejected(rid, expected, 'retry_code_changed')

    def test_independent_evaluations_share_capacity_and_are_unchanged(self):
        from model_platform.datasets import DatasetStore
        DatasetStore(self.runtime)
        rid, expected = self.failed()
        self.submit()
        model_id = self.register_fixture(*self.store.claim())
        with self.runtime.connection() as db:
            db.execute("INSERT INTO dataset_versions VALUES('dsv_fixture','other','fixture',?,1,'fixture','{}',0)", ('a'*64,))
            for index in range(32):
                db.execute("INSERT INTO model_evaluations VALUES(?,?,?,?,?,?,?,?,?,'queued',1,NULL,NULL,NULL)",
                           ('mev_'+f'{index:032x}', 'other', str(index), 'b'*64, '{}', model_id,
                            'dsv_fixture', '{}', 'c'*64))
        self.rejected(rid, expected, 'run_capacity', 429)

    def test_multiprocess_duplicate_conditional_requests_enqueue_only_once(self):
        rid, expected = self.failed()
        outcomes = [value for _, value in self.race(rid, expected, ['retry']*4)]
        self.assertEqual(collections.Counter(outcomes), {'ok': 1, 'retry_conflict': 3})
        self.assertEqual(self.retry_audits(rid), 1)
        self.assertEqual(len(self.store.run('demo', rid)['attempts']), 1)

    def test_retry_claim_and_cancel_races_preserve_serialized_history(self):
        for operation in ('claim', 'cancel'):
            for _ in range(3):
                with self.subTest(operation=operation):
                    rid, expected = self.failed()
                    outcomes = dict(self.race(rid, expected, ['retry', operation]))
                    self.assertEqual(outcomes['retry'], 'ok')
                    row = self.store.run('demo', rid)
                    if operation == 'claim' and outcomes[operation] == 'claimed':
                        self.assertEqual((row['status'], len(row['attempts'])), ('running', 2))
                        self.store.fail(rid, row['attempts'][-1]['id'], 'fixture_cleanup')
                    elif operation == 'cancel' and outcomes[operation] == 'ok':
                        self.assertEqual((row['status'], len(row['attempts'])), ('cancelled', 1))
                    else:
                        self.assertEqual(row['status'], 'queued')
                        self.store.cancel('demo', rid)
                    self.assertEqual(row['attempts'][0]['id'], expected)
                    self.rejected(rid, expected)
                    self.assertEqual(self.retry_audits(rid), 1)

    def test_stale_retry_cannot_requeue_new_failure_during_worker_recovery(self):
        rid, expected = self.failed()
        self.store.retry('demo', rid, expected)
        _, newer = self.store.claim()
        outcomes = dict(self.race(rid, expected, ['retry', 'recover']))
        self.assertEqual(outcomes, {'retry': 'retry_conflict', 'recover': 1})
        row = self.store.run('demo', rid)
        self.assertEqual(row['status'], 'failed')
        self.assertEqual(row['attempts'][-1]['id'], newer)
        self.assertEqual(row['attempts'][-1]['error_code'], 'interrupted')
        self.assertIsNone(self.store.claim())
        self.assertEqual(self.retry_audits(rid), 1)

    def test_retry_retains_original_queue_priority(self):
        rid, expected = self.failed()
        later = self.submit()
        self.store.retry('demo', rid, expected)
        self.assertEqual(self.store.run('demo', rid)['queue']['position'], 1)
        self.assertEqual(self.store.run('demo', later)['queue']['position'], 2)
        self.assertEqual(self.store.claim()[0]['id'], rid)

    def test_fixed_cat_retry_preserves_recipe_without_loading_data_or_training(self):
        from model_platform.detection import request_body as cat_body
        rid, expected = self.failed(body=cat_body())
        before = self.store.run('demo', rid)
        self.store.retry('demo', rid, expected)
        after = self.store.run('demo', rid)
        self.assertEqual(after['status'], 'queued')
        for key in ('id', 'recipe', 'dataset', 'config', 'environment', 'request_sha256', 'attempts'):
            self.assertEqual(after[key], before[key])


class ConditionalRetryHttpTests(RetryFixture, unittest.TestCase):
    def setUp(self):
        super().setUp()
        from model_platform.api import Auth, make_handler
        from model_platform.http_support import BoundedHTTPServer
        auth = Auth.__new__(Auth)
        self.tokens = {'demo': '1'*64, 'other': '2'*64}
        auth.tokens = {hashlib.sha256(value.encode()).digest(): key for key, value in self.tokens.items()}
        auth.windows = collections.defaultdict(collections.deque)
        auth.lock = threading.Lock()
        self.handler = make_handler(self.store, None, auth)
        self.server = BoundedHTTPServer(('127.0.0.1', 0), self.handler)
        thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        thread.start()
        def cleanup():
            self.server.shutdown()
            thread.join(timeout=5)
            self.server.server_close()
        self.addCleanup(cleanup)

    def call(self, path, raw=None, tenant='demo'):
        headers = {'Content-Type': 'application/json'}
        if tenant:
            headers['Authorization'] = 'Bearer ' + self.tokens[tenant]
        connection = http.client.HTTPConnection(*self.server.server_address, timeout=5)
        try:
            connection.request('POST' if raw is not None else 'GET', path, raw, headers)
            response = connection.getresponse()
            return response.status, json.loads(response.read())
        finally:
            connection.close()

    def body(self, expected):
        return json.dumps({'expected_attempt_id': expected}).encode()

    def test_auth_ownership_body_validation_and_health_capability(self):
        rid, expected = self.failed()
        route = '/runs/'+rid+'/retry'
        health = self.call('/health')[1]
        self.assertIs(health['conditional_training_retry_supported'], True)
        self.assertIs(health['queued_cancel_supported'], True)
        self.assertIs(health['cancel_supported'], False)
        before = self.snapshot()
        self.assertEqual(self.call(route, self.body(expected), None)[0], 401)
        self.assertEqual(self.call(route, b'not json', 'other')[0], 404)
        self.assertEqual(self.call('/runs/run_missing/retry', b'null')[0], 404)
        for value in (None, True, 1, [], {}, '', 'attempt_'+'A'*32, 'run_'+'a'*32,
                      'attempt_'+'a'*31, 'attempt_'+'a'*33, 'attempt_'+'a'*32+'\n'):
            with self.subTest(value=value):
                self.assertEqual(self.call(route, self.body(value))[0], 400)
        for raw in (b'null', b'[]', b'1', b'{"unexpected":true}',
                    b'{"expected_attempt_id":"x","expected_attempt_id":"y"}',
                    json.dumps({'expected_attempt_id': expected, 'config': {}}).encode()):
            self.assertEqual(self.call(route, raw)[0], 400)
        self.assertEqual(self.snapshot(), before)
        status, result = self.call(route, self.body(expected))
        self.assertEqual((status, result['id'], result['status']), (202, rid, 'queued'))
        self.assertFalse((self.runtime.home/'credentials.json').exists())

    def test_legacy_empty_body_and_conditional_code_error_remain_distinct(self):
        rid, expected = self.failed()
        route = '/runs/'+rid+'/retry'
        before = self.snapshot()
        with mock.patch('model_platform.store.environment', return_value=dict(environment(), code_sha256='0'*64)):
            status, result = self.call(route, self.body(expected))
            self.assertEqual((status, result['error']['code']), (409, 'retry_code_changed'))
            self.assertEqual(self.snapshot(), before)
            self.assertEqual(self.call(route, b'{}')[0], 202)

    def test_accepted_response_can_already_be_running_or_terminal(self):
        original = self.store.retry
        for target in ('queued', 'running', 'failed', 'succeeded', 'cancelled'):
            with self.subTest(target=target):
                rid, expected = self.failed()
                def retry_and_advance(tenant, run_id, expected_attempt_id=None):
                    original(tenant, run_id, expected_attempt_id)
                    if target == 'cancelled':
                        self.store.cancel(tenant, run_id)
                    elif target != 'queued':
                        run, attempt = self.store.claim()
                        if target == 'failed':
                            self.store.fail(run_id, attempt, 'fast_fixture_failure')
                        elif target == 'succeeded':
                            self.register_fixture(run, attempt)
                with mock.patch.object(self.store, 'retry', side_effect=retry_and_advance):
                    status, result = self.call('/runs/'+rid+'/retry', self.body(expected))
                self.assertEqual((status, result['id'], result['status']), (202, rid, target))
                self.assertEqual(result['attempts'][0]['id'], expected)
                self.assertEqual(self.retry_audits(rid), 1)
                if target == 'queued':
                    self.store.cancel('demo', rid)
                elif target == 'running':
                    self.store.fail(rid, result['attempts'][-1]['id'], 'fixture_cleanup')

    def test_lost_http_response_fast_failure_and_replay_do_not_enqueue_again(self):
        rid, expected = self.failed()
        route = '/runs/'+rid+'/retry'
        original = self.handler.respond
        committed = threading.Event()
        def lose_response(handler, status, body):
            if status == 202:
                handler.close_connection = True
                handler.connection.shutdown(socket.SHUT_RDWR)
                committed.set()
                return
            original(handler, status, body)
        with mock.patch.object(self.handler, 'respond', lose_response):
            with self.assertRaises(http.client.RemoteDisconnected):
                self.call(route, self.body(expected))
            self.assertTrue(committed.wait(timeout=5))
        self.assertEqual(self.store.run('demo', rid)['status'], 'queued')
        _, newer = self.store.claim()
        self.store.fail(rid, newer, 'fast_fixture_failure')
        before = self.snapshot()
        status, result = self.call(route, self.body(expected))
        self.assertEqual((status, result['error']['code']), (409, 'retry_conflict'))
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(self.retry_audits(rid), 1)
        self.assertIsNone(self.store.claim())
        self.assertEqual(self.call('/runs/'+rid)[1]['attempts'][-1]['id'], newer)
