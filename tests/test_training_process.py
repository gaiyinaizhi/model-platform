"""Real subprocess mechanics; synthetic numerical data is not quality evidence."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from model_platform import training_process as supervisor, worker
from model_platform.algorithm import TrainingFailure
from model_platform.common import Runtime, environment
from model_platform.recipes import get_recipe
from model_platform.store import Store
from model_platform.training_child import TrainingRuntime
from scripts.demo import request_body


@unittest.skipUnless(sys.platform.startswith('linux'), 'Linux training supervisor')
class TrainingProcessTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.runtime = Runtime(self.temporary.name)
        self.store = Store(self.runtime)
        self.sequence = 0

    def job(self, body=None):
        self.sequence += 1
        self.store.submit('demo', str(self.sequence), body or request_body())
        return self.store.claim()

    def test_real_numeric_child_and_actual_provenance(self):
        run, attempt = self.job()
        # An in-process execution would hit this mock; the child uses source.
        with patch('model_platform.recipe_execution.execute_numeric', side_effect=AssertionError('in-process')):
            worker.execute(self.store, run, attempt)
        record = self.store.run('demo', run['id'])
        self.assertEqual(record['status'], 'succeeded')
        model = self.store.model('demo', record['model_version_id'])
        self.assertEqual(model['execution_environment'], environment())
        self.assertTrue(any(row['event'] == 'train_progress' for row in record['attempts'][0]['logs']))

    def test_known_recipe_failure_is_preserved(self):
        body = request_body()
        for row in body['dataset']['train']:
            row['x'] = [0.0]
        run, attempt = self.job(body)
        worker.execute(self.store, run, attempt)
        record = self.store.run('demo', run['id'])
        self.assertEqual(record['status'], 'failed')
        self.assertEqual(record['attempts'][0]['error_code'], 'degenerate_features')
        self.assertIsNone(record['model_version_id'])

    def test_direct_execute_collision_does_not_mutate_attempt(self):
        run, attempt = self.job()
        with self.runtime.lock('worker', inherited_lease=True):
            with self.assertRaises(RuntimeError):
                worker.execute(self.store, run, attempt)
        self.assertEqual(self.store.run('demo', run['id'])['status'], 'running')

    def test_close_only_lease_keeps_inherited_description_locked(self):
        duplicate = None
        try:
            with self.runtime.lock('worker', inherited_lease=True) as fd:
                duplicate = os.dup(fd)
            with self.assertRaises(RuntimeError):
                with self.runtime.lock('worker', inherited_lease=True):
                    pass
        finally:
            if duplicate is not None:
                os.close(duplicate)
        with self.runtime.lock('worker', inherited_lease=True):
            pass

    def test_child_database_is_read_only(self):
        child = TrainingRuntime(self.runtime.home)
        with child.connection() as database:
            import sqlite3
            with self.assertRaises(sqlite3.OperationalError):
                database.execute("UPDATE runs SET status='failed'")

    def fake_child(self, script, timeout=2):
        run, attempt = self.job()
        loaded = environment()
        processes = []
        popen = subprocess.Popen
        def spawn(*args, **kwargs):
            process = popen(*args, **kwargs)
            processes.append(process)
            return process
        with self.runtime.lock('worker', inherited_lease=True) as fd, \
                patch.object(supervisor, 'command', return_value=[sys.executable, '-c', script]), \
                patch.object(supervisor.subprocess, 'Popen', side_effect=spawn):
            try:
                return supervisor.supervise(self.runtime, get_recipe('logistic-regression/v1'), request_body(), attempt,
                    json.loads(run['environment_json']), loaded, fd, lambda *_: None,
                    time.monotonic() + timeout)
            finally:
                self.assertEqual(len(processes), 1)
                self.assertIsNotNone(processes[0].returncode)
                with self.assertRaises(ChildProcessError):
                    os.waitpid(processes[0].pid, os.WNOHANG)

    def test_timeout_kills_and_reaps_child_ignoring_term(self):
        with self.assertRaises(TrainingFailure) as caught:
            self.fake_child('import time,signal; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(30)', .15)
        self.assertEqual(caught.exception.code, 'training_timeout')

    def test_huge_output_killed_and_reaped(self):
        with self.assertRaises(TrainingFailure) as caught:
            self.fake_child("import os,sys; os.write(int(sys.argv[4]), b'x' * (2*1024*1024)); import time; time.sleep(30)")
        self.assertEqual(caught.exception.code, 'training_protocol_limit')

    def test_malformed_output_reaped(self):
        with self.assertRaises(ValueError):
            self.fake_child("import os,sys; os.write(int(sys.argv[4]), b'{bad}\\n'); import time; time.sleep(30)")

    def test_missing_result_fails(self):
        with self.assertRaises(TrainingFailure) as caught:
            self.fake_child('pass')
        self.assertEqual(caught.exception.code, 'execution_failed')

    def test_unknown_error_cannot_leak_child_text(self):
        with self.assertRaises(ValueError):
            self.fake_child("import os,sys; os.write(int(sys.argv[4]), b'{\"type\":\"error\",\"code\":\"secret-token\"}\\n')")

    def test_extra_frames_after_terminal_rejected(self):
        with self.assertRaises(ValueError):
            self.fake_child("import os,sys; os.write(int(sys.argv[4]), b'{\"type\":\"error\",\"code\":\"execution_failed\"}\\n{}\\n')")

    def test_log_and_json_sanitization(self):
        for value in (True, float('inf'), float('nan'), 'secret', 1.5, 0, 201):
            with self.subTest(value=value), self.assertRaises(ValueError):
                supervisor.validate_log('train_progress', {'epoch': value})
        with self.assertRaises(ValueError):
            supervisor.validate_log('train_progress', {'epoch': 2, 'total_epochs': 1})
        for raw in (b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":1e999}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                supervisor.decode(raw)
        with self.assertRaises(ValueError):
            supervisor.validate_log('raw_exception', {'message': '/secret'})

    def test_all_fixed_recipe_failure_codes_are_preserved(self):
        import ast
        from model_platform.training_child import ERROR_CODES
        for name in ('algorithm', 'digits', 'recipe_execution', 'fashion_training',
                     'detection_training', 'detection_data', 'dataset_training', 'dataset_detection'):
            tree = ast.parse((supervisor.ROOT / 'model_platform' / (name + '.py')).read_text())
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                    continue
                index = {'TrainingFailure': 0, '_deadline': 2, 'Problem': 1}.get(node.func.id)
                if index is not None and len(node.args) > index and isinstance(node.args[index], ast.Constant):
                    self.assertIn(node.args[index].value, ERROR_CODES, (name, node.lineno))
        from model_platform.fashion_data import DatasetError, _cache_directory
        self.assertEqual(DatasetError('redacted').code, 'dataset_corrupt')
        with self.assertRaises(DatasetError) as caught:
            with _cache_directory(self.runtime):
                pass
        self.assertEqual(caught.exception.code, 'dataset_missing')
        for code in ('dataset_missing', 'dataset_corrupt'):
            with self.subTest(code=code), self.assertRaises(TrainingFailure) as failure:
                frame = json.dumps({'type': 'error', 'code': code}).encode() + b'\n'
                self.fake_child('import os,sys; os.write(int(sys.argv[4]), ' + repr(frame) + ')')
            self.assertEqual(failure.exception.code, code)

    def test_deadline_before_commit_rolls_back_publication(self):
        run, attempt = self.job()
        from model_platform.common import Problem
        with patch('model_platform.store.time.monotonic', side_effect=[0, 0, 2]):
            with self.assertRaises(Problem) as caught:
                self.store.register(run, attempt, 'a'*64, {'dataset_sha256': 'b'*64}, {}, deadline=1)
        self.assertEqual(caught.exception.code, 'training_timeout')
        record = self.store.run('demo', run['id'])
        self.assertEqual(record['status'], 'running')
        self.assertIsNone(record['model_version_id'])
        with self.runtime.connection() as database:
            for table in ('models', 'evaluations'):
                self.assertEqual(database.execute('SELECT count(*) FROM ' + table).fetchone()[0], 0)

    def test_static_recipe_log_fields_remain_supported(self):
        import ast
        from model_platform.recipes import REGISTRY
        for name in ('algorithm', 'digits', 'recipe_execution', 'fashion_training',
                     'detection_training', 'dataset_training'):
            tree = ast.parse((supervisor.ROOT / 'model_platform' / (name + '.py')).read_text())
            for node in ast.walk(tree):
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                        and node.func.id == 'log' and len(node.args) == 2
                        and isinstance(node.args[0], ast.Constant)
                        and isinstance(node.args[1], ast.Dict)):
                    event = node.args[0].value
                    keys = {key.value for key in node.args[1].keys if isinstance(key, ast.Constant)}
                    self.assertLessEqual(keys, supervisor.LOG_KEYS[event], (name, event, keys))
        for spec in REGISTRY.values():
            supervisor.validate_log('artifact_written', {'format': spec.format})

    def test_preflight_failure_before_claim_or_recovery(self):
        self.job()
        with patch.object(supervisor, 'preflight', side_effect=RuntimeError('unavailable')), \
                patch.object(Store, 'claim') as claim, patch.object(Store, 'recover_worker') as recover:
            with self.assertRaises(RuntimeError):
                worker.run_worker(self.runtime, once=True)
        claim.assert_not_called()
        recover.assert_not_called()

    def test_parent_loaded_code_mismatch_fails_before_child(self):
        run, attempt = self.job()
        with patch.object(worker, 'EXECUTION_ENVIRONMENT', dict(environment(), code_sha256='0'*64)), \
                patch.object(supervisor, 'supervise') as spawn:
            worker.execute(self.store, run, attempt)
        spawn.assert_not_called()
        self.assertEqual(self.store.run('demo', run['id'])['attempts'][0]['error_code'], 'execution_code_changed')

    def test_postcommit_log_error_cannot_add_failed_event(self):
        run, attempt = self.job()
        original = self.store.log
        def log(attempt_id, event, details=None):
            if event == 'registered':
                raise RuntimeError('postcommit log failure')
            return original(attempt_id, event, details)
        with patch.object(self.store, 'log', side_effect=log):
            worker.execute(self.store, run, attempt)
        record = self.store.run('demo', run['id'])
        self.assertEqual(record['status'], 'succeeded')
        self.assertNotIn('failed', [row['event'] for row in record['attempts'][0]['logs']])

    def test_expired_registration_has_no_model(self):
        run, attempt = self.job()
        from model_platform.common import Problem
        with self.assertRaises(Problem) as caught:
            self.store.register(run, attempt, 'a'*64, {}, {}, deadline=time.monotonic()-1)
        self.assertEqual(caught.exception.code, 'training_timeout')
        self.assertIsNone(self.store.run('demo', run['id'])['model_version_id'])
