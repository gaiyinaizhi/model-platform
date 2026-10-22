"""Private independent-evaluation supervision; no queue or publication authority."""
import os
from pathlib import Path
import re
import selectors
import signal
import subprocess
import sys
import tempfile
import time

from .common import Problem, encode
from .model_evaluations import ID, MAX_CHILD_BYTES
from .training_process import ExecutionLifecycle, TerminationUncertain, decode, reap_owned

MAX_INPUT = 16 * 1024
ROOT = Path(__file__).resolve().parents[1]
ERROR_CODES = frozenset('''evaluation_failed evaluation_dependencies_missing
evaluation_integrity evaluation_result_limit evaluation_timeout execution_code_changed
execution_environment_changed import_integrity not_found invalid_input'''.split())

# The inherited close-only Worker lease covers interpreter startup. Install the
# kernel guard before project/numerical imports; it also works during native code.
BOOTSTRAP = '''import ctypes, os, signal, sys
if not sys.platform.startswith("linux"):
    sys.exit(70)
libc = ctypes.CDLL(None, use_errno=True)
if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
    sys.exit(70)
if os.getppid() != int(sys.argv[1]):
    sys.exit(70)
for fd in map(int, sys.argv[3:6]):
    os.set_inheritable(fd, False)
import hashlib, pathlib
root = pathlib.Path(sys.argv[2])
h = hashlib.sha256()
for p in sorted((root / "model_platform").rglob("*.py")):
    h.update(p.relative_to(root / "model_platform").as_posix().encode())
    h.update(p.read_bytes())
if h.hexdigest() != sys.argv[6]:
    sys.exit(71)
sys.path.insert(0, str(root))
from model_platform.evaluation_child import main
main(int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6])
'''


def command(cache):
    return [sys.executable, '-I', '-B', *(['-S'] if sys.flags.no_site else []),
            '-X', 'pycache_prefix=' + str(cache), '-c', BOOTSTRAP]


def validate_request(value):
    if type(value) is not dict or set(value) != {'home', 'evaluation_id', 'loaded_environment'}:
        raise ValueError('Invalid evaluation request')
    home, item_id, loaded = value['home'], value['evaluation_id'], value['loaded_environment']
    if (type(home) is not str or not 1 <= len(home) <= 4096 or '\x00' in home
            or not Path(home).is_absolute() or type(item_id) is not str or not ID.fullmatch(item_id)):
        raise ValueError('Invalid evaluation identity')
    if type(loaded) is not dict or set(loaded) != {
            'python', 'platform', 'code_sha256', 'dependencies', 'supported_recipes'}:
        raise ValueError('Invalid execution environment')
    for key, maximum in (('python', 64), ('platform', 2048), ('dependencies', 4096)):
        if type(loaded[key]) is not str or not 1 <= len(loaded[key]) <= maximum:
            raise ValueError('Invalid execution environment value')
    if type(loaded['code_sha256']) is not str or not re.fullmatch('[a-f0-9]{64}', loaded['code_sha256']):
        raise ValueError('Invalid execution source identity')
    recipes = loaded['supported_recipes']
    if (type(recipes) is not list or not 1 <= len(recipes) <= 64
            or any(type(item) is not str or not 1 <= len(item) <= 128 for item in recipes)):
        raise ValueError('Invalid execution recipes')
    return value


def validate_result(value):
    if type(value) is not dict:
        raise ValueError('Invalid evaluation result')
    if set(value) == {'error_code'}:
        if type(value['error_code']) is not str or value['error_code'] not in ERROR_CODES:
            raise ValueError('Invalid evaluation error')
    elif (set(value) != {'result', 'predictions'} or type(value['result']) is not dict
            or type(value['predictions']) is not list):
        raise ValueError('Invalid evaluation result envelope')
    return value


def supervise(runtime, evaluation_id, loaded_environment, lease_fd, deadline, lifecycle=None):
    """Return only after reaping this invocation's child; uncertainty aborts the Worker."""
    lifecycle = lifecycle or ExecutionLifecycle()
    payload = encode(validate_request({'home': str(runtime.home), 'evaluation_id': evaluation_id,
                                       'loaded_environment': loaded_environment}))
    if len(payload) > MAX_INPUT:
        raise Problem(409, 'evaluation_result_limit', 'Evaluation request exceeds bounded channel')
    process = None
    read_fd, write_fd = os.pipe()
    try:
        with tempfile.TemporaryFile() as request, tempfile.TemporaryDirectory(prefix='mp-evaluation-cache-') as cache:
            request.write(payload)
            request.seek(0)
            if time.monotonic() >= deadline:
                raise Problem(409, 'evaluation_timeout', 'Evaluation deadline exceeded')
            args = command(cache) + [str(os.getpid()), str(ROOT), str(request.fileno()),
                str(write_fd), str(lease_fd), loaded_environment['code_sha256']]
            # A failed Popen can have an uncertain spawn outcome. Never publish
            # a terminal record or continue the queue without an owned handle.
            lifecycle.proof = None
            process = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, close_fds=True,
                pass_fds=(request.fileno(), write_fd, lease_fd), cwd=ROOT,
                env={'PATH': os.defpath, 'LANG': 'C.UTF-8', 'OPENBLAS_NUM_THREADS': '1',
                     'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'})
            os.close(write_fd)
            write_fd = None
            raw = bytearray()
            with selectors.DefaultSelector() as selector:
                selector.register(read_fd, selectors.EVENT_READ)
                while selector.get_map():
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise Problem(409, 'evaluation_timeout', 'Evaluation deadline exceeded')
                    for key, _ in selector.select(min(remaining, .1)):
                        chunk = os.read(key.fd, 65536)
                        if not chunk:
                            selector.unregister(read_fd)
                            break
                        raw.extend(chunk)
                        if len(raw) > MAX_CHILD_BYTES:
                            raise Problem(409, 'evaluation_result_limit', 'Evaluator result exceeds bounded channel')
            # Neither a complete payload nor pipe EOF proves termination.
            while process.poll() is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise Problem(409, 'evaluation_timeout', 'Evaluation deadline exceeded')
                try:
                    process.wait(timeout=min(.1, remaining))
                except subprocess.TimeoutExpired:
                    pass
            if time.monotonic() >= deadline or process.returncode in (-signal.SIGALRM, -signal.SIGXCPU):
                raise Problem(409, 'evaluation_timeout', 'Evaluation deadline exceeded')
            if process.returncode == 71:
                raise Problem(409, 'execution_code_changed', 'Execution source changed after submission')
            if process.returncode != 0 or not raw:
                raise Problem(409, 'evaluation_failed', 'Isolated evaluation did not finish successfully')
            result = validate_result(decode(raw))
            if 'error_code' in result:
                raise Problem(409, result['error_code'], 'Isolated evaluation failed')
            return result
    finally:
        try:
            if process is not None:
                reap_owned(process, lifecycle)
            elif lifecycle.proof is None:
                raise TerminationUncertain('Evaluation process construction outcome is unconfirmed')
        finally:
            for fd in (read_fd, write_fd):
                if fd is not None:
                    try:
                        os.close(fd)
                    except OSError:
                        pass
