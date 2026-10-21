"""Isolated transactional and real owned-process cancellation; finite timeouts."""
import collections
import hashlib
import http.client
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

from model_platform.common import Runtime, Problem, environment
from model_platform.store import Store
from model_platform import worker, training_process as supervisor
from scripts.demo import request_body


class RunningCancelTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.runtime = Runtime(self.temp.name)
        self.store = Store(self.runtime)
        self.counter = 0

    def job(self):
        self.counter += 1
        self.store.submit('demo', str(self.counter), request_body())
        return self.store.claim()

    def record(self, run):
        return self.store.run('demo', run['id'])

    def register(self, run, attempt):
        return self.store.register(run, attempt, 'a'*64, {'dataset_sha256': 'b'*64}, {})

    def test_intent_replay_confirmation_and_no_publication(self):
        run, attempt = self.job()
        status, pending = self.store.cancel_running('demo', run['id'], attempt)
        self.assertEqual(status, 202)
        self.assertEqual(pending['status'], 'running')
        self.assertEqual(pending['attempts'][-1]['status'], 'running')
        self.assertEqual(pending['cancellation']['status'], 'requested')
        self.assertEqual(self.store.cancel_running('demo', run['id'], attempt), (202, pending))
        with self.assertRaises(Problem):
            self.register(run, attempt)
        # A failure without lifecycle proof cannot turn intent into confirmation.
        self.store.fail(run['id'], attempt, 'fixture')
        self.assertEqual(self.record(run), pending)
        self.store.fail(run['id'], attempt, 'fixture', termination_proof='child_reaped')
        confirmed = self.record(run)
        self.assertEqual(confirmed['status'], 'cancelled')
        self.assertEqual(confirmed['attempts'][-1]['status'], 'cancelled')
        self.assertIsNone(confirmed['attempts'][-1]['error_code'])
        self.assertIsNotNone(confirmed['cancellation']['confirmed_at'])
        self.assertEqual(self.store.cancel_running('demo', run['id'], attempt), (200, confirmed))
        with self.assertRaises(Problem):
            self.store.retry('demo', run['id'])
        with self.runtime.connection() as db:
            for table in ('models', 'evaluations'):
                self.assertEqual(db.execute('SELECT count(*) FROM '+table).fetchone()[0], 0)
            self.assertEqual(db.execute("SELECT count(*) FROM audit WHERE action='confirm_running_cancel'").fetchone()[0], 1)

    def test_confirmation_preserves_order_after_clock_rollback(self):
        run, attempt = self.job()
        with mock.patch('model_platform.store.now', return_value=200):
            self.store.cancel_running('demo', run['id'], attempt)
        with mock.patch('model_platform.store.now', return_value=100):
            self.store.confirm_cancel(run['id'], attempt, 'never_started')
        record = self.record(run)
        self.assertEqual(record['cancellation']['requested_at'], 200)
        self.assertEqual(record['cancellation']['confirmed_at'], 200)
        self.assertEqual(record['attempts'][-1]['finished_at'], 200)

    def test_prior_failures_and_stale_fail_do_not_mutate_current_attempt(self):
        run, old = self.job()
        self.store.fail(run['id'], old, 'fixture')
        previous = self.record(run)['attempts'][0]
        self.store.retry('demo', run['id'])
        run, current = self.store.claim()
        self.store.fail(run['id'], old, 'late_failure')
        self.assertEqual(self.record(run)['status'], 'running')
        other, other_attempt = self.job()
        self.store.fail(run['id'], other_attempt, 'wrong_run')
        self.assertEqual(self.record(other)['status'], 'running')
        for tenant, rid, aid, code in [('other', run['id'], current, 404), ('demo', run['id'], old, 409),
                                      ('demo', run['id'], other_attempt, 409), ('demo', 'missing', current, 404)]:
            with self.assertRaises(Problem) as caught:
                self.store.cancel_running(tenant, rid, aid)
            self.assertEqual(caught.exception.status, code)
        self.store.cancel_running('demo', run['id'], current)
        self.store.confirm_cancel(run['id'], current, 'never_started')
        self.assertEqual(self.record(run)['attempts'][0], previous)

    def test_registration_and_failure_first_reject_intent(self):
        for action in ('register', 'fail'):
            run, attempt = self.job()
            if action == 'register':
                self.register(run, attempt)
            else:
                self.store.fail(run['id'], attempt, 'fixture')
            with self.assertRaises(Problem) as caught:
                self.store.cancel_running('demo', run['id'], attempt)
            self.assertEqual(caught.exception.code, 'cancel_conflict')
            self.assertIsNone(self.record(run)['cancellation'])

    def test_concurrent_registration_or_failure_vs_intent(self):
        for action in ('register', 'fail'):
            for _ in range(5):
                run, attempt = self.job()
                barrier = threading.Barrier(2, timeout=3)
                outcomes = {}
                def cancel():
                    barrier.wait()
                    try:
                        outcomes['cancel'] = self.store.cancel_running('demo', run['id'], attempt)[0]
                    except Problem as exc:
                        outcomes['cancel'] = exc.status
                def finish():
                    barrier.wait()
                    try:
                        if action == 'register':
                            self.register(run, attempt)
                        else:
                            self.store.fail(run['id'], attempt, 'fixture', termination_proof='child_reaped')
                    except Problem as exc:
                        outcomes['finish'] = exc.code
                threads = [threading.Thread(target=cancel), threading.Thread(target=finish)]
                for thread in threads: thread.start()
                for thread in threads:
                    thread.join(5)
                    self.assertFalse(thread.is_alive())
                record = self.record(run)
                if outcomes['cancel'] == 202:
                    self.assertIsNone(record['model_version_id'])
                    if action == 'register':
                        self.assertEqual(outcomes.get('finish'), 'cancel_requested')
                        self.store.confirm_cancel(run['id'], attempt, 'child_reaped')
                    self.assertEqual(self.record(run)['status'], 'cancelled')
                else:
                    self.assertEqual(outcomes['cancel'], 409)
                    self.assertEqual(record['status'], 'succeeded' if action == 'register' else 'failed')

    def test_nochild_cancel_and_worker_recovery(self):
        run, attempt = self.job()
        self.store.cancel_running('demo', run['id'], attempt)
        with mock.patch.object(supervisor.subprocess, 'Popen') as spawn:
            # Supply an actual held lease so the separate preflight probe isn't mocked.
            with self.runtime.lock('worker', inherited_lease=True) as lease:
                worker.execute(self.store, run, attempt, lease)
        spawn.assert_not_called()
        with self.runtime.connection() as db:
            self.assertEqual(db.execute('SELECT confirmation_kind FROM training_cancellations').fetchone()[0], 'never_started')
        run, attempt = self.job()
        self.store.cancel_running('demo', run['id'], attempt)
        with self.runtime.lock('worker', inherited_lease=True):
            self.assertEqual(self.store.recover_worker(), 1)
        self.assertEqual(self.record(run)['status'], 'cancelled')
        with self.runtime.connection() as db:
            self.assertEqual(db.execute('SELECT confirmation_kind FROM training_cancellations WHERE attempt_id=?', (attempt,)).fetchone()[0], 'lease_recovered')

    def run_cancel_child(self, script, *, spawn_gap=False):
        run, attempt = self.job()
        processes = []
        original = subprocess.Popen
        def spawn(*args, **kwargs):
            child = original(*args, **kwargs)
            processes.append(child)
            if spawn_gap:
                self.store.cancel_running('demo', run['id'], attempt)
            return child
        def request():
            end = time.monotonic()+3
            while not processes and time.monotonic() < end: time.sleep(.01)
            if processes:
                time.sleep(.15)
                self.store.cancel_running('demo', run['id'], attempt)
        thread = None
        if not spawn_gap:
            thread = threading.Thread(target=request)
            thread.start()
        start = time.monotonic()
        with self.runtime.lock('worker', inherited_lease=True) as lease, \
                mock.patch.object(supervisor, 'command', return_value=[sys.executable, '-c', script]), \
                mock.patch.object(supervisor.subprocess, 'Popen', side_effect=spawn):
            worker.execute(self.store, run, attempt, lease)
        if thread:
            thread.join(5)
            self.assertFalse(thread.is_alive())
        self.assertLess(time.monotonic()-start, 6)
        self.assertEqual(self.record(run)['status'], 'cancelled')
        self.assertIsNotNone(processes[0].returncode)
        with self.assertRaises(ChildProcessError): os.waitpid(processes[0].pid, os.WNOHANG)
        # Cancellation never strands the subsequent actual numerical training.
        next_run, next_attempt = self.job()
        worker.execute(self.store, next_run, next_attempt)
        self.assertEqual(self.record(next_run)['status'], 'succeeded')

    def test_cancel_spawn_gap(self):
        self.run_cancel_child('import time; time.sleep(30)', spawn_gap=True)

    def test_cancel_result_before_exit_and_eof_wait(self):
        self.run_cancel_child('import os,sys,time; fd=int(sys.argv[4]); os.write(fd,b\'{"type":"result"}\\n\'); os.close(fd); time.sleep(30)')

    def test_cancel_read_loop_child_ignores_sigterm(self):
        self.run_cancel_child('import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(30)')

    def test_uncertain_kill_or_reap_bypasses_failure_and_queue_continuation(self):
        for failure in (OSError('cannot signal'), subprocess.TimeoutExpired('child', 5)):
            lifecycle = supervisor.ExecutionLifecycle()
            child = mock.Mock()
            child.poll.return_value = None
            child.kill.side_effect = failure if isinstance(failure, OSError) else None
            child.wait.side_effect = subprocess.TimeoutExpired('child', 5)
            with self.assertRaises(supervisor.TerminationUncertain):
                supervisor.reap_owned(child, lifecycle)
            child.wait.assert_called_once_with(timeout=5)
            self.assertIsNone(lifecycle.proof)
        run, attempt = self.job()
        self.store.cancel_running('demo', run['id'], attempt)
        with mock.patch.object(supervisor, 'supervise', side_effect=supervisor.TerminationUncertain()), \
                mock.patch.object(self.store, 'cancellation_requested', return_value=False), \
                mock.patch.object(self.store, 'fail') as fail:
            with self.runtime.lock('worker', inherited_lease=True) as lease:
                with self.assertRaises(supervisor.TerminationUncertain):
                    worker.execute(self.store, run, attempt, lease)
        fail.assert_not_called()
        self.assertEqual(self.record(run)['status'], 'running')
        self.assertEqual(self.record(run)['cancellation']['status'], 'requested')

    def test_parent_sigkill_and_inherited_lease_recovery(self):
        import signal
        from pathlib import Path
        for install_parent_death in (True, False):
            run, attempt = self.job()
            marker = self.runtime.home / ('child-' + attempt)
            child_code = ('import os,sys,time,ctypes,signal; '
                          + ('ctypes.CDLL(None).prctl(1,signal.SIGKILL,0,0,0); ' if install_parent_death else '')
                          + 'open(' + repr(str(marker)) + ',"w").write(str(os.getpid())); time.sleep(30)')
            # Explicit context keeps DB and worker lease lifetime visible.
            parent_code = "\n".join([
                "import sys", "from unittest.mock import patch",
                "from model_platform.common import Runtime", "from model_platform.store import Store",
                "from model_platform import worker, training_process", "store=Store(Runtime(sys.argv[1]))",
                "with store.runtime.connection() as db:",
                "    run=dict(db.execute('SELECT * FROM runs WHERE id=?',(sys.argv[2],)).fetchone())",
                "with store.runtime.lock('worker', inherited_lease=True) as lease:",
                "    with patch.object(training_process,'command',return_value=[sys.executable,'-c',sys.argv[4]]):",
                "        worker.execute(store,run,sys.argv[3],lease)"])
            parent = subprocess.Popen([sys.executable, '-c', parent_code, str(self.runtime.home), run['id'], attempt, child_code],
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            child_pid = None
            try:
                end = time.monotonic()+5
                while not marker.exists() and time.monotonic()<end:
                    self.assertIsNone(parent.poll())
                    time.sleep(.02)
                self.assertTrue(marker.exists())
                child_pid = int(marker.read_text())
                os.kill(parent.pid, signal.SIGSTOP)
                self.store.cancel_running('demo', run['id'], attempt)
                parent.kill(); parent.wait(timeout=3)
                if not install_parent_death:
                    with self.assertRaises(RuntimeError):
                        with self.runtime.lock('worker', inherited_lease=True): pass
                    self.assertEqual(self.record(run)['status'], 'running')
                    os.kill(child_pid, signal.SIGKILL)
                end = time.monotonic()+5
                while True:
                    try:
                        with self.runtime.lock('worker', inherited_lease=True):
                            self.store.recover_worker()
                        break
                    except RuntimeError:
                        if time.monotonic() >= end: raise
                        time.sleep(.02)
                self.assertEqual(self.record(run)['status'], 'cancelled')
                with self.runtime.connection() as db:
                    self.assertEqual(db.execute('SELECT confirmation_kind FROM training_cancellations WHERE attempt_id=?', (attempt,)).fetchone()[0], 'lease_recovered')
            finally:
                if parent.poll() is None:
                    parent.kill(); parent.wait(timeout=3)
                if child_pid is not None:
                    try: os.kill(child_pid, signal.SIGKILL)
                    except ProcessLookupError: pass

    def test_api_exact_body_ownership_and_capability(self):
        from model_platform.api import Auth, make_handler
        from model_platform.http_support import BoundedHTTPServer
        auth = Auth.__new__(Auth)
        tokens = {'demo': '1'*64, 'other': '2'*64}
        auth.tokens = {hashlib.sha256(v.encode()).digest(): k for k,v in tokens.items()}
        auth.windows = collections.defaultdict(collections.deque)
        auth.lock = threading.Lock()
        server = BoundedHTTPServer(('127.0.0.1', 0), make_handler(self.store, None, auth))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        def cleanup():
            server.shutdown(); server.server_close(); thread.join(3)
        self.addCleanup(cleanup)
        def call(path, body=None, tenant='demo'):
            conn = http.client.HTTPConnection(*server.server_address, timeout=3)
            try:
                conn.request('POST' if body is not None else 'GET', path, body,
                             {'Authorization': 'Bearer '+tokens[tenant], 'Content-Type': 'application/json'})
                response = conn.getresponse()
                return response.status, json.loads(response.read())
            finally: conn.close()
        health = call('/health')[1]
        self.assertTrue(health['running_cancel_supported'])
        self.assertTrue(health['queued_cancel_supported'])
        self.assertFalse(health['cancel_supported'])
        run, attempt = self.job()
        path = '/runs/'+run['id']+'/cancel-running'
        self.assertEqual(call(path, b'not json', 'other')[0], 404)
        for body in ({}, [], None, {'expected_attempt_id': 1}, {'expected_attempt_id': attempt.upper()},
                     {'expected_attempt_id': attempt, 'extra': True}):
            self.assertEqual(call(path, json.dumps(body))[0], 400)
        status, pending = call(path, json.dumps({'expected_attempt_id': attempt}))
        self.assertEqual(status, 202)
        self.assertEqual(pending['cancellation']['status'], 'requested')
        self.store.confirm_cancel(run['id'], attempt, 'never_started')
        status, confirmed = call(path, json.dumps({'expected_attempt_id': attempt}))
        self.assertEqual(status, 200)
        self.assertEqual(confirmed['status'], 'cancelled')
        self.assertEqual(call('/runs/'+run['id'])[1], confirmed)


if __name__ == '__main__':
    unittest.main()
