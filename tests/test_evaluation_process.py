"""Owned-process evaluation mechanics; opaque fixtures are not detector evidence."""
import contextlib
import ctypes
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

from model_platform import evaluation_process as supervisor
from model_platform import evaluation_child, model_evaluations, training_process, worker
from model_platform.common import digest, encode, environment
from scripts.demo import request_body
from tests import test_model_evaluations as queue_fixtures


@unittest.skipUnless(sys.platform.startswith('linux'), 'Linux evaluation supervisor')
class EvaluationProcessTests(unittest.TestCase):
    def setUp(self):
        # Compose the synthetic queue setup, without inheriting its test methods.
        queue_fixtures.QueueTests.setUp(self)
        self.sequence = 0

    def job(self, *, claim=True):
        self.sequence += 1
        item = queue_fixtures.QueueTests.create(self, 'lifecycle-' + str(self.sequence))
        return item, self.evaluations.claim() if claim else None

    def record(self, item):
        with self.runtime.connection() as database:
            return dict(database.execute('SELECT * FROM model_evaluations WHERE id=?',
                                         (item['id'],)).fetchone())

    def payload(self):
        evidence = []
        return {'result': {'digests': dict(
            self.evaluations.expected_digests(self.binding, digest(self.binding)),
            prediction_sha256=digest(evidence))}, 'predictions': evidence}

    def assert_reaped(self, child):
        self.assertIsNotNone(child.returncode)
        with self.assertRaises(ChildProcessError):
            os.waitpid(child.pid, os.WNOHANG)

    def assert_unpublished(self, item):
        self.assertEqual(self.record(item)['status'], 'running')
        self.assertFalse(self.evaluations.evidence_path(item['id']).exists())
        with self.runtime.connection() as database:
            self.assertEqual(database.execute(
                'SELECT count(*) FROM model_evaluation_results WHERE evaluation_id=?',
                (item['id'],)).fetchone()[0], 0)

    def fake_execute(self, script, *, timeout=2):
        item, row = self.job()
        children = []
        real_popen = subprocess.Popen
        original_fail, original_succeed = self.evaluations.fail, self.evaluations.succeed

        def spawn(*args, **kwargs):
            child = real_popen(*args, **kwargs)
            children.append(child)
            return child

        def terminal(operation, *args, **kwargs):
            self.assertEqual(len(children), 1)
            self.assert_reaped(children[0])
            return operation(*args, **kwargs)

        with self.runtime.lock('worker', inherited_lease=True) as lease, \
                mock.patch.object(supervisor, 'command', return_value=[sys.executable, '-c', script]), \
                mock.patch.object(supervisor.subprocess, 'Popen', side_effect=spawn), \
                mock.patch.object(self.evaluations, 'fail', side_effect=lambda *a, **k: terminal(original_fail, *a, **k)), \
                mock.patch.object(self.evaluations, 'succeed', side_effect=lambda *a, **k: terminal(original_succeed, *a, **k)):
            try:
                model_evaluations.execute(self.evaluations, row, environment(), timeout,
                                          lease_fd=lease)
            finally:
                for child in children:
                    if child.poll() is None:
                        child.kill()
                    child.wait(timeout=3)
        self.assertEqual(len(children), 1)
        self.assert_reaped(children[0])
        return item, children[0]

    def test_direct_execute_collision_does_not_mutate(self):
        item, row = self.job()
        before = self.record(item)
        with self.runtime.lock('worker', inherited_lease=True), \
                mock.patch.object(supervisor, 'supervise') as spawn:
            with self.assertRaises(RuntimeError):
                model_evaluations.execute(self.evaluations, row, environment())
        spawn.assert_not_called()
        self.assertEqual(self.record(item), before)
        self.assert_unpublished(item)

    def test_child_constructor_and_database_are_read_only(self):
        missing = self.runtime.home / 'missing-child-home'
        evaluation_child.EvaluationReader(evaluation_child.EvaluationRuntime(missing))
        self.assertFalse(missing.exists())
        before = {path.relative_to(self.runtime.home) for path in self.runtime.home.rglob('*')}
        cursor_before = (self.runtime.home / 'cursor-integrity.key').read_bytes()
        with self.runtime.connection() as database:
            schema = database.execute('SELECT sql FROM sqlite_master ORDER BY name').fetchall()
            rows_before = database.execute('SELECT count(*) FROM model_evaluation_results').fetchone()[0]
        child_runtime = evaluation_child.EvaluationRuntime(self.runtime.home)
        reader = evaluation_child.EvaluationReader(child_runtime)
        self.assertFalse(any(hasattr(reader, name) for name in ('claim', 'recover', 'fail', 'succeed')))
        self.assertEqual({path.relative_to(self.runtime.home) for path in self.runtime.home.rglob('*')}, before)
        with child_runtime.connection() as database:
            for sql in ("UPDATE model_evaluations SET status='failed'", 'CREATE TABLE forbidden (id TEXT)',
                        "INSERT INTO model_evaluation_results VALUES('mev_fake','demo','{}','x','y')"):
                with self.subTest(sql=sql), self.assertRaises(sqlite3.OperationalError):
                    database.execute(sql)
            self.assertEqual([tuple(row) for row in database.execute(
                'SELECT sql FROM sqlite_master ORDER BY name')], [tuple(row) for row in schema])
            self.assertEqual(database.execute('SELECT count(*) FROM model_evaluation_results').fetchone()[0], rows_before)
        self.assertEqual((self.runtime.home / 'cursor-integrity.key').read_bytes(), cursor_before)
        # SQLite may create WAL/shared-memory coordination files even for a
        # mode=ro connection. No durable schema, cursor or evidence is created.
        transient = {Path('platform.sqlite3-wal'), Path('platform.sqlite3-shm')}
        after = {path.relative_to(self.runtime.home) for path in self.runtime.home.rglob('*')}
        self.assertEqual(after - transient, before - transient)

    def test_result_and_eof_do_not_publish_before_actual_exit(self):
        for include_result in (True, False):
            with self.subTest(include_result=include_result):
                item, row = self.job()
                marker = self.runtime.home / ('ready-' + item['id'])
                release = self.runtime.home / ('release-' + item['id'])
                raw = encode(self.payload()) if include_result else b''
                script = '\n'.join([
                    'import os,sys,time,pathlib', 'fd=int(sys.argv[4])',
                    'os.write(fd,' + repr(raw) + ')', 'os.close(fd)',
                    'pathlib.Path(' + repr(str(marker)) + ').touch()',
                    'while not pathlib.Path(' + repr(str(release)) + ').exists(): time.sleep(.01)'])
                errors, children = [], []
                real_popen = subprocess.Popen

                def spawn(*args, **kwargs):
                    child = real_popen(*args, **kwargs)
                    children.append(child)
                    return child

                def run(lease):
                    try:
                        model_evaluations.execute(self.evaluations, row, environment(), 5, lease_fd=lease)
                    except BaseException as exc:
                        errors.append(exc)

                original_fail, original_succeed = self.evaluations.fail, self.evaluations.succeed

                def terminal(operation, *args, **kwargs):
                    self.assert_reaped(children[0])
                    return operation(*args, **kwargs)

                with self.runtime.lock('worker', inherited_lease=True) as lease, \
                        mock.patch.object(supervisor, 'command', return_value=[sys.executable, '-c', script]), \
                        mock.patch.object(supervisor.subprocess, 'Popen', side_effect=spawn), \
                        mock.patch.object(self.evaluations, 'fail', side_effect=lambda *a, **k: terminal(original_fail, *a, **k)), \
                        mock.patch.object(self.evaluations, 'succeed', side_effect=lambda *a, **k: terminal(original_succeed, *a, **k)):
                    thread = threading.Thread(target=run, args=(lease,))
                    thread.start()
                    try:
                        self.wait_for(lambda: marker.exists(), 'child did not report its EOF')
                        time.sleep(.15)
                        self.assertTrue(thread.is_alive())
                        self.assertIsNone(children[0].poll())
                        self.assert_unpublished(item)
                    finally:
                        release.touch()
                        thread.join(timeout=6)
                        for child in children:
                            if child.poll() is None:
                                child.kill()
                            child.wait(timeout=3)
                self.assertFalse(thread.is_alive())
                self.assertEqual(errors, [])
                self.assert_reaped(children[0])
                self.assertEqual(self.record(item)['status'], 'succeeded' if include_result else 'failed')

    def test_timeout_ignoring_sigterm_is_killed_and_reaped_before_failure(self):
        started = time.monotonic()
        item, child = self.fake_execute(
            'import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)', timeout=.15)
        self.assertLess(time.monotonic() - started, 3)
        self.assertEqual(child.returncode, -signal.SIGKILL)
        self.assertEqual(self.record(item)['error_code'], 'evaluation_timeout')

    def test_overflow_is_bounded_and_reaped_before_failure(self):
        script = '\n'.join(['import os,sys,time', 'fd=int(sys.argv[4])',
                            'for _ in range(130): os.write(fd,b"x"*65536)', 'time.sleep(60)'])
        item, child = self.fake_execute(script)
        self.assertEqual(child.returncode, -signal.SIGKILL)
        self.assertEqual(self.record(item)['error_code'], 'evaluation_result_limit')

    def test_strict_output_rejects_ambiguous_or_untrusted_payloads(self):
        cases = [b'', b'{bad}', b'[]', b'{}', b'null',
                 b'{"result":{},"predictions":{}}', b'{"result":[],"predictions":[]}',
                 b'{"result":{},"predictions":[],"extra":1}',
                 b'{"error_code":"evaluation_failed","extra":1}',
                 b'{"result":{},"predictions":[],"error_code":"evaluation_failed"}',
                 b'{"error_code":"/private/secret-token"}', b'{"error_code":1}',
                 b'{"result":{},"result":{},"predictions":[]}',
                 b'{"result":{"x":1,"x":2},"predictions":[]}',
                 b'{"result":{"x":NaN},"predictions":[]}',
                 b'{"result":{"x":Infinity},"predictions":[]}',
                 b'{"result":{"x":-Infinity},"predictions":[]}',
                 b'{"result":{"x":1e309},"predictions":[]}',
                 b'{"result":{},"predictions":[]}{}',
                 b'{"result":{},"predictions":[]}\n{"error_code":"evaluation_failed"}',
                 b'{"result":{},"predictions":[]}trailing']
        for raw in cases:
            with self.subTest(raw=raw):
                item, _ = self.fake_execute('import os,sys; os.write(int(sys.argv[4]),' + repr(raw) + ')')
                record = self.record(item)
                self.assertEqual((record['status'], record['error_code']), ('failed', 'evaluation_failed'))
                self.assertFalse(self.evaluations.evidence_path(item['id']).exists())

    def test_valid_child_errors_are_preserved(self):
        for code in sorted(supervisor.ERROR_CODES):
            with self.subTest(code=code):
                item, _ = self.fake_execute('import os,sys; os.write(int(sys.argv[4]),' +
                                           repr(encode({'error_code': code})) + ')')
                self.assertEqual(self.record(item)['error_code'], code)

    def test_nonzero_exit_cannot_publish_valid_result(self):
        item, _ = self.fake_execute('import os,sys; os.write(int(sys.argv[4]),' +
                                   repr(encode(self.payload())) + '); sys.exit(4)')
        self.assertEqual(self.record(item)['error_code'], 'evaluation_failed')
        self.assertFalse(self.evaluations.evidence_path(item['id']).exists())

    def test_source_mismatch_is_rejected_by_real_bootstrap(self):
        item, _ = self.job()
        lifecycle = supervisor.ExecutionLifecycle()
        with self.runtime.lock('worker', inherited_lease=True) as lease:
            with self.assertRaises(model_evaluations.Problem) as failure:
                supervisor.supervise(self.runtime, item['id'], dict(environment(), code_sha256='0'*64),
                                     lease, time.monotonic() + 3, lifecycle=lifecycle)
        self.assertEqual(failure.exception.code, 'execution_code_changed')
        self.assertEqual(lifecycle.proof, 'child_reaped')
        self.assert_unpublished(item)

    def test_private_request_has_exact_shape_and_bounded_values(self):
        request = {'home': str(self.runtime.home), 'evaluation_id': 'mev_' + 'a'*32,
                   'loaded_environment': environment()}
        self.assertEqual(supervisor.validate_request(request), request)
        invalid = [[], {}, dict(request, extra=True), dict(request, home='relative'),
                   dict(request, home=''), dict(request, home='/' + 'a'*4096),
                   dict(request, home='/x\x00y'), dict(request, evaluation_id='run_' + 'a'*32),
                   dict(request, evaluation_id='mev_' + 'A'*32), dict(request, loaded_environment=[])]
        loaded = request['loaded_environment']
        for replacement in ({}, dict(loaded, extra=1), dict(loaded, code_sha256='x'*64),
                            dict(loaded, python=''), dict(loaded, platform='x'*2049),
                            dict(loaded, dependencies='x'*4097), dict(loaded, supported_recipes=[]),
                            dict(loaded, supported_recipes=['x'*129]), dict(loaded, supported_recipes=[1]),
                            dict(loaded, supported_recipes=['x']*65)):
            invalid.append(dict(request, loaded_environment=replacement))
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                supervisor.validate_request(value)

    def test_oversized_request_never_starts_child(self):
        item, _ = self.job()
        loaded = dict(environment(), platform='x'*2048, dependencies='x'*4096,
                      supported_recipes=['x'*128]*64)
        runtime = mock.Mock(home=Path('/' + 'x'*4095))
        lifecycle = supervisor.ExecutionLifecycle()
        with mock.patch.object(supervisor.subprocess, 'Popen') as spawn:
            with self.assertRaises(model_evaluations.Problem) as failure:
                supervisor.supervise(runtime, item['id'], loaded, 1, time.monotonic()+3, lifecycle=lifecycle)
        spawn.assert_not_called()
        self.assertEqual(failure.exception.code, 'evaluation_result_limit')
        self.assertEqual(lifecycle.proof, 'never_started')

    def test_child_main_rejects_invalid_private_requests_without_measurement(self):
        item, _ = self.job()
        loaded = environment()
        request = {'home': str(self.runtime.home), 'evaluation_id': item['id'],
                   'loaded_environment': loaded}
        raw = encode(request)
        cases = [b'x' * (supervisor.MAX_INPUT + 1), b'{bad}', b'[]',
                 encode(dict(request, extra=1)), encode(dict(request, home='relative')),
                 encode(dict(request, loaded_environment=dict(loaded, platform='different-fixture-platform'))),
                 raw[:-1] + b',"home":"/duplicate"}', raw + b'{}',
                 raw.replace(b'"loaded_environment":{', b'"loaded_environment":{"bad":1e309,'),
                 raw.replace(b'"loaded_environment":{', b'"loaded_environment":{"bad":NaN,')]
        # Exercise the actual child main and its read-only row lookup, replacing
        # only metric computation with an unmistakable synthetic payload.
        prefix = supervisor.BOOTSTRAP.split('from model_platform.evaluation_child import main')[0]
        script = prefix + '\n'.join([
            'from model_platform import evaluation_child as child',
            'child.measure=lambda *args: {"result":{"synthetic":True},"predictions":[]}',
            'child.main(int(sys.argv[3]),int(sys.argv[4]),int(sys.argv[5]),sys.argv[6])'])

        def invoke(payload):
            with self.runtime.lock('worker', inherited_lease=True) as lease, \
                    tempfile.TemporaryFile() as request_file, tempfile.TemporaryFile() as result:
                request_file.write(payload)
                request_file.seek(0)
                child = subprocess.Popen([sys.executable, '-I', '-B', '-c', script,
                    str(os.getpid()), str(supervisor.ROOT), str(request_file.fileno()),
                    str(result.fileno()), str(lease), loaded['code_sha256']],
                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    close_fds=True, pass_fds=(request_file.fileno(), result.fileno(), lease))
                try:
                    self.assertEqual(child.wait(timeout=3), 0)
                finally:
                    if child.poll() is None:
                        child.kill()
                    child.wait(timeout=3)
                self.assert_reaped(child)
                result.seek(0)
                return supervisor.decode(result.read(supervisor.MAX_CHILD_BYTES + 1))

        self.assertEqual(invoke(raw), {'result': {'synthetic': True}, 'predictions': []})
        for payload in cases:
            with self.subTest(payload=payload[:100]):
                result = invoke(payload)
                if len(payload) > supervisor.MAX_INPUT:
                    expected = 'evaluation_result_limit'
                elif b'different-fixture-platform' in payload:
                    expected = 'execution_code_changed'
                else:
                    expected = 'evaluation_failed'
                self.assertEqual(result, {'error_code': expected})
                self.assert_unpublished(item)

    def test_uncertain_spawn_or_termination_stops_queue_and_keeps_live_lease(self):
        for failure_kind in ('spawn', 'kill', 'wait'):
            with self.subTest(failure_kind=failure_kind):
                item, _ = self.job(claim=False)
                next_run = self.store.submit('demo', 'after-' + failure_kind, request_body())[0]
                children, originals = [], []
                real_popen = subprocess.Popen
                actual_execute = model_evaluations.execute

                def spawn(*args, **kwargs):
                    child = real_popen(*args, **kwargs)
                    children.append(child)
                    originals.append((child.kill, child.wait))
                    if failure_kind == 'spawn':
                        raise OSError('injected loss of process handle after spawn')
                    if failure_kind == 'kill':
                        child.kill = mock.Mock(side_effect=OSError('injected signal failure'))
                    else:
                        child.kill = mock.Mock()  # A kill reported success, without confirmed exit.
                    child.wait = mock.Mock(side_effect=subprocess.TimeoutExpired('owned-child', .01))
                    return child

                def bounded_execute(evaluations, row, loaded, *, lease_fd):
                    return actual_execute(evaluations, row, loaded, .15, lease_fd=lease_fd)

                heartbeat = mock.Mock()
                heartbeat.start.return_value = heartbeat
                try:
                    with mock.patch.object(training_process, 'preflight'), \
                            mock.patch.object(worker, 'Store', return_value=self.store), \
                            mock.patch.object(model_evaluations, 'ModelEvaluationStore', return_value=self.evaluations), \
                            mock.patch.object(worker, 'WorkerHeartbeat', return_value=heartbeat), \
                            mock.patch.object(worker, 'probe_dependencies', return_value={}), \
                            mock.patch.object(worker, 'EXECUTION_ENVIRONMENT', environment()), \
                            mock.patch.object(model_evaluations, 'execute', side_effect=bounded_execute), \
                            mock.patch.object(supervisor, 'command', return_value=[sys.executable, '-c', 'import time; time.sleep(60)']), \
                            mock.patch.object(supervisor.subprocess, 'Popen', side_effect=spawn), \
                            mock.patch.object(self.evaluations, 'claim', wraps=self.evaluations.claim) as claim, \
                            mock.patch.object(self.store, 'claim', wraps=self.store.claim) as training_claim, \
                            mock.patch.object(self.evaluations, 'fail', wraps=self.evaluations.fail) as fail, \
                            mock.patch.object(self.evaluations, 'succeed', wraps=self.evaluations.succeed) as succeed:
                        with self.assertRaises(supervisor.TerminationUncertain):
                            worker.run_worker(self.runtime)
                    self.assertEqual(claim.call_count, 1)
                    training_claim.assert_not_called()
                    fail.assert_not_called()
                    succeed.assert_not_called()
                    heartbeat.stop.assert_called_once()
                    self.assertEqual(len(children), 1)
                    self.assertIsNone(children[0].poll())
                    self.assert_unpublished(item)
                    self.assertEqual(self.store.run('demo', next_run)['status'], 'queued')
                    with self.assertRaises(RuntimeError):
                        with self.runtime.lock('worker', inherited_lease=True):
                            pass
                finally:
                    for child, (kill, wait) in zip(children, originals):
                        if child.poll() is None:
                            kill()
                        wait(timeout=3)
                        child.kill, child.wait = kill, wait
                with mock.patch.object(worker, 'EXECUTION_ENVIRONMENT', environment()):
                    worker.run_worker(self.runtime, once=True)
                self.assertEqual(self.record(item)['error_code'], 'evaluation_interrupted')
                self.assertEqual(self.store.run('demo', next_run)['status'], 'succeeded')

    def wait_for(self, predicate, message, timeout=5):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(.01)
        self.fail(message)

    @contextlib.contextmanager
    def adopt_owned_descendants(self):
        # This test process reaps its own orphaned grandchild. Restore the prior
        # Linux setting; never reap or signal unrelated process identities.
        libc = ctypes.CDLL(None, use_errno=True)
        previous = ctypes.c_int()
        self.assertEqual(libc.prctl(37, ctypes.byref(previous), 0, 0, 0), 0)
        self.assertEqual(libc.prctl(36, 1, 0, 0, 0), 0)
        try:
            yield
        finally:
            self.assertEqual(libc.prctl(36, previous.value, 0, 0, 0), 0)

    def parent_death_case(self, *, stopped_before_guard):
        if not hasattr(os, 'pidfd_open') or not hasattr(signal, 'pidfd_send_signal'):
            self.skipTest('Linux pidfd identity-safe test cleanup unavailable')
        item, _ = self.job()
        next_run = self.store.submit('demo', 'after-parent-' + str(stopped_before_guard), request_body())[0]
        marker = self.runtime.home / ('owned-child-' + item['id'])
        mark = 'open(' + repr(str(marker)) + ',"w").write(str(os.getpid()))\n'
        if stopped_before_guard:
            script = ('import os,signal\n' + mark + 'os.kill(os.getpid(),signal.SIGSTOP)\n' +
                      supervisor.BOOTSTRAP)
        else:
            # Keep the production guard, isolated imports, fd and source checks.
            # Replace only measurement with a native sleep that holds the GIL:
            # a Python watchdog could not run until its full 60 seconds elapsed.
            prefix = supervisor.BOOTSTRAP.split('from model_platform.evaluation_child import main')[0]
            script = prefix + mark + 'ctypes.PyDLL(None).sleep(60)\n'
        parent_code = '\n'.join([
            'import sys,time', 'from unittest.mock import patch',
            'from model_platform.common import Runtime,environment',
            'from model_platform import evaluation_process as supervisor',
            'runtime=Runtime(sys.argv[1])',
            'with runtime.lock("worker", inherited_lease=True) as lease:',
            '    with patch.object(supervisor,"command",return_value=[sys.executable,"-I","-B","-c",sys.argv[3]]):',
            '        supervisor.supervise(runtime,sys.argv[2],environment(),lease,time.monotonic()+60)'])
        pidfd = child_pid = child_status = None
        with self.adopt_owned_descendants():
            parent = subprocess.Popen([sys.executable, '-c', parent_code, str(self.runtime.home),
                                       item['id'], script], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:
                self.wait_for(lambda: marker.exists() and marker.stat().st_size > 0,
                              'owned evaluator did not reach its test marker')
                child_pid = int(marker.read_text())
                # The marker belongs to this isolated home. Verify the actual
                # parent before pinning identity with a pidfd and sending signals.
                stat = Path('/proc/' + str(child_pid) + '/stat').read_text().split()
                self.assertEqual(int(stat[3]), parent.pid)
                pidfd = os.pidfd_open(child_pid)
                if stopped_before_guard:
                    self.wait_for(lambda: Path('/proc/' + str(child_pid) + '/stat').read_text().split()[2] == 'T',
                                  'evaluator did not stop before bootstrap')
                else:
                    time.sleep(.15)
                with self.assertRaises(RuntimeError):
                    with self.runtime.lock('worker', inherited_lease=True):
                        pass
                parent.kill()
                parent.wait(timeout=3)
                if stopped_before_guard:
                    with mock.patch.object(training_process, 'preflight'), \
                            mock.patch.object(worker.Store, 'recover_worker') as recover_training, \
                            mock.patch.object(model_evaluations.ModelEvaluationStore, 'recover') as recover_evaluation, \
                            mock.patch.object(worker.Store, 'claim') as claim_training, \
                            mock.patch.object(model_evaluations.ModelEvaluationStore, 'claim') as claim_evaluation:
                        with self.assertRaises(RuntimeError):
                            worker.run_worker(self.runtime, once=True)
                    for operation in (recover_training, recover_evaluation, claim_training, claim_evaluation):
                        operation.assert_not_called()
                    self.assert_unpublished(item)
                    self.assertEqual(self.store.run('demo', next_run)['status'], 'queued')
                    signal.pidfd_send_signal(pidfd, signal.SIGCONT)

                def reap_child():
                    nonlocal child_status
                    pid, status = os.waitpid(child_pid, os.WNOHANG)
                    if pid:
                        child_status = status
                        return True
                    return False

                self.wait_for(reap_child, 'owned evaluator survived its parent or resumed bootstrap')
                if stopped_before_guard:
                    self.assertTrue(os.WIFEXITED(child_status))
                    self.assertEqual(os.WEXITSTATUS(child_status), 70)
                else:
                    self.assertTrue(os.WIFSIGNALED(child_status))
                    self.assertEqual(os.WTERMSIG(child_status), signal.SIGKILL)
                with mock.patch.object(worker, 'EXECUTION_ENVIRONMENT', environment()):
                    worker.run_worker(self.runtime, once=True)
                self.assertEqual(self.record(item)['error_code'], 'evaluation_interrupted')
                self.assertEqual(self.store.run('demo', next_run)['status'], 'succeeded')
            finally:
                if parent.poll() is None:
                    parent.kill()
                parent.wait(timeout=3)
                if pidfd is not None:
                    if child_status is None:
                        try:
                            signal.pidfd_send_signal(pidfd, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        os.waitpid(child_pid, 0)
                    os.close(pidfd)

    def test_kernel_parent_death_interrupts_native_call_holding_gil(self):
        self.parent_death_case(stopped_before_guard=False)

    def test_stopped_before_bootstrap_lease_fences_recovery_until_exit(self):
        self.parent_death_case(stopped_before_guard=True)


if __name__ == '__main__':
    unittest.main()
