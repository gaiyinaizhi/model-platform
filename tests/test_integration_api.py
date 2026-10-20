"""Discovery, deployment replay and stable history through real HTTP/SQLite."""
import concurrent.futures
import copy
import http.client
import json
import os
from pathlib import Path
import stat
import tempfile
import socket
import time
import unittest
from urllib.parse import urlsplit
from unittest import mock

import test_platform
from model_platform.common import Problem, Runtime, encode
from model_platform.pagination import cursor_key, pack_cursor
from model_platform.recipes import public_catalog
from model_platform.store import Store
from scripts.demo import request_body


class IntegrationHttpTests(unittest.TestCase):
    for _name in ('setUp', 'spawn', 'start_api', 'cleanup', 'call', 'submit', 'worker_once', 'train', 'ready'):
        locals()[_name] = getattr(test_platform.PlatformTests, _name)

    def test_discovery_authenticated_detached_and_quality_truthful(self):
        self.assertEqual(self.call('/recipes', tenant=None)[0], 401)
        status, body = self.call('/recipes')
        self.assertEqual(status, 200)
        self.assertEqual(body, {'recipes': public_catalog()})
        self.assertEqual(len(body['recipes']), 5)
        det = next(item for item in body['recipes'] if item['id']=='yolox-nano-cat/v1')
        self.assertEqual(det['task_type'], 'object-detection')
        self.assertFalse(det['quality']['production_allowed'])
        self.assertEqual(det['quality']['status'], 'quality-failed')
        self.assertEqual(det['availability']['dependencies'], 'unknown')
        self.assertEqual(det['training']['fixed_config']['epochs'], 20)
        self.assertEqual(det['prediction']['semantics']['classes'], [{'id': 0, 'name': 'Cat'}])
        encoded = json.dumps(body)
        for forbidden in (str(self.home), 'cursor-integrity', 'validate_request', 'credentials.json'):
            self.assertNotIn(forbidden, encoded)
        for token in self.tokens.values():
            self.assertNotIn(token, encoded)
        det['training']['fixed_config']['epochs'] = 999
        self.assertEqual(next(item for item in public_catalog() if item['id']=='yolox-nano-cat/v1')['training']['fixed_config']['epochs'], 20)
        self.assertEqual(self.call('/recipes', {})[0], 404)

    def test_deploy_concurrent_replay_canonical_input_and_restart(self):
        model = self.train()['model_version_id']
        body = {'model_version_id': model}
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            responses = list(pool.map(lambda _: self.call('/deployments', body, 'one'), range(8)))
        self.assertEqual([s for s, _ in responses].count(202), 1)
        self.assertEqual([s for s, _ in responses].count(200), 7)
        ids = {r['id'] for _, r in responses}
        self.assertEqual(len(ids), 1)
        deployment = ids.pop()
        # Whitespace and JSON serialization are not part of canonical input.
        status, replay = self.call('/deployments', key='one', raw=(' { "model_version_id" : "'+model+'" } ').encode())
        self.assertEqual((status, replay['id']), (200, deployment))
        self.assertEqual(self.call('/deployments', {'model_version_id': 'unknown'}, 'one')[0], 409)
        self.assertEqual(self.call('/deployments', body, 'one', tenant='other')[0], 404)
        self.ready(deployment)
        self.assertEqual(self.call('/deployments/'+deployment+'/predict', {'features': [[-2.5], [2.5]]})[0], 200)
        self.api.terminate(); self.api.wait(timeout=10)
        self.start_api()
        status, replay = self.call('/deployments', body, 'one')
        self.assertEqual((status, replay['id']), (200, deployment))
        self.ready(deployment)
        self.assertEqual(self.call('/deployments/'+deployment+'/stop', {})[0], 200)
        status, replay = self.call('/deployments', body, 'one')
        self.assertEqual((status, replay['id'], replay['desired']), (200, deployment, 'stopped'))
        self.assertEqual(self.call('/deployments/'+deployment+'/predict', {'features': [[0]]})[0], 503)
        with self.runtime.connection() as db:
            self.assertEqual(db.execute("SELECT count(*) FROM audit WHERE action='start_deployment'").fetchone()[0], 1)

    def test_replay_at_capacity_legacy_and_failed_creation_does_not_bind_key(self):
        model = self.train()['model_version_id']
        body = {'model_version_id': model}
        first = self.call('/deployments', body, 'first')[1]
        for _ in range(3):
            self.assertEqual(self.call('/deployments', body)[0], 202)
        self.assertEqual(self.call('/deployments', body, 'blocked')[0], 429)
        self.assertEqual(self.call('/deployments', body, 'first')[0], 200)
        self.assertEqual(self.call('/deployments', {'model_version_id': 'other'}, 'first')[0], 409)
        self.assertEqual(len(self.call('/deployments')[1]['ids']), 4)
        self.call('/deployments/'+first['id']+'/stop', {})
        self.assertEqual(self.call('/deployments', body, 'blocked')[0], 202)
        self.assertEqual(self.call('/deployments', body, 'blocked')[0], 200)
        self.assertEqual(self.call('/deployments', body, '')[0], 400)
        self.assertEqual(self.call('/deployments', body, 'x'*129)[0], 400)

    def test_response_loss_retry_and_tenant_scoped_key(self):
        model = self.train()['model_version_id']
        url = urlsplit(self.base)
        raw = encode({'model_version_id': model})
        connection = socket.create_connection((url.hostname, url.port), timeout=5)
        wire = (f"POST /deployments HTTP/1.1\r\nHost: localhost\r\n"
                f"Authorization: Bearer {self.tokens['demo']}\r\n"
                f"Idempotency-Key: response-lost\r\nContent-Type: application/json\r\n"
                f"Content-Length: {len(raw)}\r\nConnection: close\r\n\r\n").encode()+raw
        connection.sendall(wire)
        # Simulate the caller losing the response after the server committed.
        deadline = time.monotonic()+5
        while time.monotonic() < deadline:
            with self.runtime.connection() as db:
                row = db.execute("SELECT deployment_id FROM deployment_keys WHERE idempotency_key='response-lost'").fetchone()
            if row: break
            time.sleep(0.01)
        connection.close()
        self.assertIsNotNone(row)
        status, replay = self.call('/deployments', {'model_version_id': model}, 'response-lost')
        self.assertEqual((status, replay['id']), (200, row[0]))
        status, run = self.call('/runs', request_body(), 'other-model', tenant='other')
        self.assertEqual(status, 202)
        self.worker_once()
        other_model = self.call('/runs/'+run['id'], tenant='other')[1]['model_version_id']
        status, other = self.call('/deployments', {'model_version_id': other_model}, 'response-lost', tenant='other')
        self.assertEqual(status, 202)
        self.assertNotEqual(other['id'], replay['id'])

    def test_concurrent_distinct_keys_never_exceed_capacity(self):
        model = self.train()['model_version_id']
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            responses = list(pool.map(lambda n: self.call('/deployments', {'model_version_id': model}, f'capacity-{n}'), range(8)))
        self.assertEqual([s for s, _ in responses].count(202), 4)
        self.assertEqual([s for s, _ in responses].count(429), 4)
        with self.runtime.connection() as db:
            self.assertEqual(db.execute('SELECT count(*) FROM deployment_keys').fetchone()[0], 4)
            self.assertEqual(db.execute('SELECT count(*) FROM deployments').fetchone()[0], 4)

    def test_duplicate_deploy_header_is_rejected(self):
        url = urlsplit(self.base)
        connection = http.client.HTTPConnection(url.hostname, url.port, timeout=5)
        self.addCleanup(connection.close)
        raw = encode({'model_version_id': 'any'})
        connection.putrequest('POST', '/deployments')
        connection.putheader('Authorization', 'Bearer '+self.tokens['demo'])
        connection.putheader('Idempotency-Key', 'one')
        connection.putheader('Idempotency-Key', 'two')
        connection.putheader('Content-Type', 'application/json')
        connection.putheader('Content-Length', str(len(raw)))
        connection.endheaders(raw)
        response = connection.getresponse()
        self.assertEqual(response.status, 400)
        response.read()

    def seed_runs(self, count, tenant='demo', stamp=100):
        ids = []
        for i in range(count):
            with mock.patch('model_platform.store.now', return_value=stamp):
                run, _ = self.store.submit(tenant, f'{tenant}-{i}', request_body())
            with self.runtime.connection() as db:
                db.execute("UPDATE runs SET status='failed' WHERE id=?", (run,))
            ids.append(run)
        return ids

    def test_history_over_100_ties_snapshot_restart_scope_and_tampering(self):
        ids = self.seed_runs(107)
        self.seed_runs(2, tenant='other')
        self.assertEqual(self.call('/runs')[1], {'ids': sorted(ids, reverse=True)[:100], 'limit': 100})
        status, first = self.call('/runs?limit=17')
        self.assertEqual(status, 200)
        token = first['next_cursor']
        self.assertEqual(self.call('/runs?cursor='+token, tenant='other')[0], 400)
        self.assertEqual(self.call('/models?cursor='+token)[0], 400)
        self.assertEqual(self.call('/runs?limit=18&cursor='+token)[0], 400)
        altered = ('A' if token[0] != 'A' else 'B') + token[1:]
        self.assertEqual(self.call('/runs?cursor='+altered)[0], 400)
        # New/backdated inserts must both remain outside this traversal.
        for key, stamp in [('newer', 200), ('backdated', 1)]:
            with mock.patch('model_platform.store.now', return_value=stamp):
                self.store.submit('demo', key, request_body())
        self.api.terminate(); self.api.wait(timeout=10); self.start_api()
        collected = first['ids']
        while token:
            status, page = self.call('/runs?cursor='+token)
            self.assertEqual(status, 200, page)
            collected += page['ids']
            token = page['next_cursor']
        self.assertEqual(collected, sorted(ids, reverse=True))
        self.assertEqual(len(collected), len(set(collected)))
        self.assertEqual(self.call('/models?limit=1')[1], {'ids': [], 'limit': 1, 'next_cursor': None})

    def test_query_bounds_and_unsupported_routes(self):
        for query in ('limit=0', 'limit=101', 'limit=-1', 'limit=1.0', 'limit=', 'limit=true',
                      'limit=1&limit=2', 'other=1', 'cursor=', 'cursor=x', 'cursor='+'A'*2049,
                      'limit=1&cursor=x&extra=y', 'limit', 'limit=%FF'):
            with self.subTest(query=query[:60]):
                self.assertEqual(self.call('/runs?'+query)[0], 400)
        self.assertEqual(self.call('/health?limit=1')[0], 400)
        self.assertEqual(self.call('/runs?limit=1', request_body(), 'x')[0], 400)
        self.assertEqual(self.call('/runs?limit=1', tenant=None)[0], 401)
        self.assertEqual(self.call('/runs?limit=100')[0], 200)

    def test_models_and_deployments_paginate_without_rewriting_records(self):
        model = self.train()['model_version_id']
        before = copy.deepcopy(self.call('/models/'+model)[1])
        for _ in range(3):
            deployment = self.call('/deployments', {'model_version_id': model})[1]
            self.call('/deployments/'+deployment['id']+'/stop', {})
        for collection in ('models', 'deployments'):
            expected = self.call('/'+collection)[1]['ids']
            result, token = [], None
            while True:
                path = '/'+collection+('?cursor='+token if token else '?limit=1')
                status, page = self.call(path)
                self.assertEqual(status, 200)
                result += page['ids']; token = page['next_cursor']
                if not token: break
            self.assertEqual(result, expected)
        self.assertEqual(self.call('/models/'+model)[1], before)


class CursorKeyTests(unittest.TestCase):
    def test_atomic_initialization_mode_and_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
                keys = list(pool.map(lambda _: cursor_key(home), range(32)))
            self.assertTrue(all(key == keys[0] for key in keys))
            self.assertEqual(len(keys[0]), 32)
            self.assertEqual(stat.S_IMODE((home/'cursor-integrity.key').stat().st_mode), 0o600)
            self.assertEqual(cursor_key(home), keys[0])
            self.assertEqual(len(list(home.iterdir())), 1)
            os.chmod(home/'cursor-integrity.key', 0o644)
            with self.assertRaises(RuntimeError): cursor_key(home)

    def test_long_unicode_tenant_cursor_remains_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(Runtime(directory))
            tenant = '\U0001f600' * 160
            for key in ('one', 'two'):
                store.submit(tenant, key, request_body())
            first = store.list_page('runs', tenant, limit=1)
            self.assertLess(len(first['next_cursor']), 2048)
            second = store.list_page('runs', tenant, cursor=first['next_cursor'])
            self.assertEqual(len(second['ids']), 1)
            self.assertIsNone(second['next_cursor'])

    def test_legacy_deployment_schema_migration_preserves_technical_data(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Runtime(directory)
            # Keep the versioned queue schema complete. This fixture targets
            # the older additive deployment-key/index upgrade, not queue migration.
            Store(runtime)
            with runtime.connection() as db:
                db.execute('DROP TABLE deployment_keys')
                db.execute('DROP INDEX models_history')
                db.execute('DROP INDEX deployments_history')
                before = db.execute("SELECT sql FROM sqlite_master WHERE name='models'").fetchone()[0]
            Store(runtime)
            with runtime.connection() as db:
                self.assertEqual(db.execute("SELECT sql FROM sqlite_master WHERE name='models'").fetchone()[0], before)
                self.assertIsNotNone(db.execute("SELECT name FROM sqlite_master WHERE name='deployment_keys'").fetchone())
                for name in ('models_history', 'deployments_history'):
                    self.assertIsNotNone(db.execute(
                        "SELECT name FROM sqlite_master WHERE type='index' AND name=?", (name,)).fetchone())
