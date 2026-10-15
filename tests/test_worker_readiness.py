"""Stdlib worker liveness, dependency observation and tenant-privacy tests."""
import json
import os
import subprocess
import sys
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from model_platform.common import Runtime, RECIPE
from model_platform.store import Store
from model_platform.worker import run_worker
from model_platform.worker_readiness import (WorkerHeartbeat, public_report,
                                            probe_dependencies, STALE_AFTER_SECONDS)
from scripts.demo import request_body


class WorkerReadinessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.runtime = Runtime(Path(self.temp.name))
        self.store = Store(self.runtime)

    def report(self):
        return public_report(self.store, 'demo')

    def heartbeat(self, dependencies=None):
        hb = WorkerHeartbeat(self.runtime, dependencies or {}, interval=0.02).start()
        self.addCleanup(hb.stop)
        return hb

    def test_absent_is_not_ready_and_tenant_queue_is_private(self):
        own, _ = self.store.submit('demo', 'one', request_body())
        foreign, _ = self.store.submit('other', 'two', request_body())
        report = self.report()
        self.assertEqual(report['queue'], {'queued': 1, 'running': 0})
        self.assertEqual(report['worker']['state'], 'absent')
        self.assertFalse(report['worker']['ready'])
        self.assertTrue(all(not r['ready'] for r in report['recipes']))
        for secret in (own, foreign, 'other', 'demo'):
            self.assertNotIn(secret, json.dumps(report))

    def test_live_busy_stopped_and_unknown_data(self):
        hb = self.heartbeat()
        report = self.report()
        self.assertEqual(report['worker']['state'], 'idle')
        numeric = next(r for r in report['recipes'] if r['id'] == RECIPE)
        self.assertTrue(numeric['ready'])
        self.assertEqual(numeric['data_status'], 'not_checked')
        hb.set_state('busy')
        report = self.report()
        self.assertTrue(report['worker']['ready'])
        self.assertFalse(report['worker']['accepting_immediately'])
        self.assertEqual(report['worker']['state'], 'busy')
        self.assertEqual(report['worker']['reasons'], ['worker_busy'])
        numeric = next(r for r in report['recipes'] if r['id'] == RECIPE)
        self.assertTrue(numeric['ready'])
        self.assertEqual(numeric['reasons'], [])
        first = report['worker']['checked_at']
        deadline = time.monotonic() + 2
        while self.report()['worker']['checked_at'] <= first and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertGreater(self.report()['worker']['checked_at'], first)
        hb.stop()
        self.assertEqual(self.report()['worker']['state'], 'stopped')
        self.assertFalse(self.report()['worker']['ready'])

    def test_stale_and_future_clock_fail_closed(self):
        hb = self.heartbeat()
        hb._stop.set()
        hb._thread.join()
        for stamp in (time.time() - STALE_AFTER_SECONDS - 1, time.time() + 60):
            with self.runtime.connection() as db:
                db.execute('UPDATE worker_readiness SET checked_at=?', (stamp,))
            report = self.report()
            self.assertEqual(report['worker']['state'], 'stale')
            self.assertFalse(report['worker']['fresh'])
            self.assertEqual(report['worker']['reasons'], ['worker_stale'])
            self.assertTrue(all(not r['ready'] for r in report['recipes']))

    def test_tenant_position_is_not_global_and_claim_ties_are_stable(self):
        own = [self.store.submit('demo', str(i), request_body())[0] for i in range(3)]
        foreign, _ = self.store.submit('other', 'foreign', request_body())
        with self.runtime.connection() as db:
            db.execute("UPDATE runs SET created_at=1.0")
        for position, run_id in enumerate(sorted(own), 1):
            self.assertEqual(self.store.run('demo', run_id)['queue'],
                             {'scope': 'tenant', 'position': position})
        from model_platform.common import Problem
        with self.assertRaises(Problem) as caught:
            self.store.run('demo', foreign)
        self.assertEqual(caught.exception.status, 404)
        for run_id in sorted(own + [foreign]):
            run, attempt = self.store.claim()
            self.assertEqual(run['id'], run_id)
            self.assertEqual(self.store.run(run['tenant'], run_id)['queue'],
                             {'scope': 'tenant', 'position': None})
            self.store.fail(run_id, attempt, 'test_failure')
            self.assertIsNone(self.store.run(run['tenant'], run_id)['queue']['position'])

    def test_retry_keeps_original_creation_order(self):
        first, _ = self.store.submit('demo', 'first', request_body())
        run, attempt = self.store.claim()
        self.store.fail(first, attempt, 'test_failure')
        second, _ = self.store.submit('demo', 'second', request_body())
        with self.runtime.connection() as db:
            db.execute("UPDATE runs SET created_at=1.0 WHERE id=?", (first,))
            db.execute("UPDATE runs SET created_at=2.0 WHERE id=?", (second,))
        self.store.retry('demo', first)
        self.assertEqual(self.store.run('demo', first)['queue']['position'], 1)
        self.assertEqual(self.store.run('demo', second)['queue']['position'], 2)
        self.assertEqual(self.store.claim()[0]['id'], first)

    def test_probe_actual_import_and_metadata_failures_redacted(self):
        def imports(name):
            if name == 'torch':
                raise ImportError('/secret/path')
            if name == 'onnxruntime':
                raise RuntimeError('private-secret')
            return object()
        with patch('model_platform.worker_readiness.importlib.metadata.version', return_value='1.2.3'), \
                patch('model_platform.worker_readiness.importlib.import_module', side_effect=imports):
            deps = probe_dependencies()
        self.assertFalse(deps['torch']['available'])
        self.assertEqual(deps['torch']['reason'], 'dependency_missing')
        self.assertEqual(deps['onnxruntime']['reason'], 'dependency_unavailable')
        self.assertEqual(deps['Pillow']['version'], '1.2.3')
        self.heartbeat(deps)
        report = self.report()
        fashion = next(r for r in report['recipes'] if r['id'] == 'fashion-cnn/v1')
        self.assertFalse(fashion['ready'])
        self.assertIn('dependencies_unavailable', fashion['reasons'])
        self.assertNotIn('private-secret', json.dumps(report))
        self.assertNotIn('/secret', json.dumps(report))

    def test_replacement_is_bounded_and_fences_old_updates(self):
        old = self.heartbeat()
        new = self.heartbeat()
        new.set_state('busy')
        old.set_state('idle')
        old.stop()
        self.assertEqual(self.report()['worker']['state'], 'busy')
        with self.runtime.connection() as db:
            self.assertEqual(db.execute('SELECT count(*) FROM worker_readiness').fetchone()[0], 1)

    def test_once_cleanup_empty_and_claimed_failure(self):
        with patch('model_platform.worker.probe_dependencies', return_value={}):
            run_worker(self.runtime, once=True)
        self.assertEqual(self.report()['worker']['state'], 'stopped')
        from model_platform.fashion import request_body as fashion_body
        run_id, _ = self.store.submit('demo', 'fashion', fashion_body())
        from model_platform.algorithm import TrainingFailure
        with patch('model_platform.worker.probe_dependencies', return_value={}), \
                patch('model_platform.recipe_execution.execute_fashion', side_effect=TrainingFailure('optional_dependencies_missing')):
            run_worker(self.runtime, once=True)
        run = self.store.run('demo', run_id)
        self.assertEqual(run['status'], 'failed')
        self.assertEqual(run['attempts'][0]['error_code'], 'optional_dependencies_missing')
        self.assertEqual(self.report()['worker']['state'], 'stopped')

    def test_once_exception_cleanup(self):
        with patch('model_platform.worker.probe_dependencies', return_value={}), \
                patch.object(Store, 'claim', side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                run_worker(self.runtime, once=True)
        self.assertEqual(self.report()['worker']['state'], 'stopped')

    def test_real_stdlib_only_worker_once(self):
        # -S excludes site-packages: actual worker imports fail safely without
        # Torch, while the numeric recipe still performs real CPU training.
        run_id, _ = self.store.submit('demo', 'real-numeric', request_body())
        result = subprocess.run([sys.executable, '-S', '-m', 'model_platform', 'worker', '--once'],
                                cwd=Path(__file__).resolve().parents[1],
                                env=dict(os.environ, MP_HOME=str(self.runtime.home)),
                                capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.store.run('demo', run_id)['status'], 'succeeded')
        report = self.report()
        self.assertEqual(report['worker']['state'], 'stopped')
        fashion = next(r for r in report['recipes'] if r['id'] == 'fashion-cnn/v1')
        self.assertFalse(fashion['ready'])
        self.assertTrue(all(not dep['available'] for dep in fashion['dependencies']))

    def test_real_process_crash_expires_and_restart_recovers(self):
        process = subprocess.Popen([sys.executable, '-S', '-m', 'model_platform', 'worker'],
                                   cwd=Path(__file__).resolve().parents[1],
                                   env=dict(os.environ, MP_HOME=str(self.runtime.home)),
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        def cleanup():
            if process.poll() is None:
                process.kill()
            process.wait(timeout=10)
        self.addCleanup(cleanup)
        deadline = time.monotonic() + 10
        while not self.report()['worker']['ready'] and time.monotonic() < deadline:
            self.assertIsNone(process.poll())
            time.sleep(0.02)
        self.assertTrue(self.report()['worker']['ready'])
        process.kill()
        process.wait(timeout=10)
        # Advance only the observer clock rather than slowing this suite 15s.
        with patch('model_platform.worker_readiness.time.time', return_value=time.time() + STALE_AFTER_SECONDS + 1):
            self.assertEqual(self.report()['worker']['state'], 'stale')
            self.assertFalse(self.report()['worker']['ready'])
        with patch('model_platform.worker.probe_dependencies', return_value={}):
            run_worker(self.runtime, once=True)
        self.assertEqual(self.report()['worker']['state'], 'stopped')
