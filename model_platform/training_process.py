"""Private Linux training supervisor; no queue or publication authority."""
import hashlib
import json
import math
import os
from pathlib import Path
import selectors
import subprocess
import sys
import tempfile
import time

from .algorithm import TrainingFailure
from .common import encode

MAX_INPUT = 4 * 1024 * 1024
MAX_FRAME = 1024 * 1024
MAX_TOTAL = 2 * 1024 * 1024
MAX_LOGS = 256
ROOT = Path(__file__).resolve().parents[1]
# Runs before project/recipe imports. The inherited lease covers even the
# interpreter startup gap. SIGKILL remains effective inside native extensions.
BOOTSTRAP = '''import ctypes, os, signal, sys
if not sys.platform.startswith("linux"):
    sys.exit(70)
libc = ctypes.CDLL(None, use_errno=True)
if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
    sys.exit(70)
if os.getppid() != int(sys.argv[1]):
    sys.exit(70)
if len(sys.argv) == 2:
    sys.exit(0)
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
from model_platform.training_child import main
main(int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6])
'''


def command(cache):
    return [sys.executable, '-I', '-B', *(['-S'] if sys.flags.no_site else []),
            '-X', 'pycache_prefix=' + str(cache), '-c', BOOTSTRAP]


def preflight():
    """Fail before claiming work if kernel parent-death support is unavailable."""
    if not sys.platform.startswith('linux'):
        raise RuntimeError('Supervised training requires Linux')
    with tempfile.TemporaryDirectory(prefix='mp-training-cache-') as cache:
        result = subprocess.run(command(cache) + [str(os.getpid())],
                                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL, timeout=10)
    if result.returncode:
        raise RuntimeError('Training parent-death supervision unavailable')


def decode(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate field')
            result[key] = value
        return result
    def bad(_):
        raise ValueError('Nonfinite number')
    def finite(value):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError('Nonfinite number')
        return number
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=bad, parse_float=finite)


LOG_KEYS = {
    'train_progress': {'epoch', 'total_epochs', 'loss', 'training_log_loss'},
    'artifact_written': {'sha256', 'bytes', 'format'},
    'evaluated': {'accuracy', 'log_loss', 'sample_count', 'parity_max_absolute_logit_error',
                  'map_50', 'map_50_95', 'split', 'role'},
    'fashion_dataset_verified': {'official_training_count', 'selected_training_count',
                                'evaluation_count', 'selection_sha256'},
    'detection_dataset_verified': {'train', 'validation', 'test', 'manifest_sha256'},
    'dataset_version_verified': {'dataset_version_id', 'sha256', 'train_count',
                                'validation_count', 'test_count', 'evaluation_split'},
}


def validate_log(event, details):
    if event not in LOG_KEYS or type(details) is not dict or not details.keys() <= LOG_KEYS[event]:
        raise ValueError('Invalid log')
    strings = {'format': {'logistic-json/v1', 'softmax-json/v1', 'fashion-onnx/v1',
                          'yolox-nano-onnx/v1', 'yolox-nano-dataset-onnx/v1'},
               'split': {'val', 'test'}, 'evaluation_split': {'val', 'test'},
               'role': {'held-out-test', 'validation-only'}}
    counters = {'epoch': (1, 200), 'total_epochs': (1, 200), 'bytes': (1, 16 * 1024 * 1024),
                **{key: (0, 60000) for key in ('sample_count', 'official_training_count',
                    'selected_training_count', 'evaluation_count', 'train', 'validation', 'test',
                    'train_count', 'validation_count', 'test_count')}}
    import re
    for key, value in details.items():
        if key.endswith('sha256'):
            valid = type(value) is str and re.fullmatch('[a-f0-9]{64}', value)
        elif key == 'dataset_version_id':
            valid = type(value) is str and re.fullmatch('dsv_[a-f0-9]{32}', value)
        elif key in counters:
            low, high = counters[key]
            valid = type(value) is int and low <= value <= high
        elif key in strings:
            valid = type(value) is str and value in strings[key]
        else:
            valid = type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1e15
        if not valid:
            raise ValueError('Invalid log value')
    if ('epoch' in details and 'total_epochs' in details
            and details['epoch'] > details['total_epochs']):
        raise ValueError('Invalid epoch count')
    if len(encode(details)) > 4096:
        raise ValueError('Log limit')


def supervise(runtime, spec, body, attempt, submission, loaded, lease_fd, log, deadline):
    payload = encode({'home': str(runtime.home), 'body': body, 'attempt': attempt,
                      'submission': submission, 'loaded': loaded})
    if len(payload) > MAX_INPUT:
        raise TrainingFailure('training_protocol_limit')
    process = None
    read_fd, write_fd = os.pipe()
    terminal = None
    try:
        with tempfile.TemporaryFile() as request, tempfile.TemporaryDirectory(prefix='mp-training-cache-') as cache:
            request.write(payload)
            request.seek(0)
            process = subprocess.Popen(command(cache) + [str(os.getpid()), str(ROOT),
                str(request.fileno()), str(write_fd), str(lease_fd), loaded['code_sha256']],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                close_fds=True, pass_fds=(request.fileno(), write_fd, lease_fd), cwd=ROOT,
                env={'PATH': os.defpath, 'LANG': 'C.UTF-8', 'OPENBLAS_NUM_THREADS': '1',
                     'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'})
            os.close(write_fd)
            write_fd = None
            total = logs = 0
            buffer = b''
            with selectors.DefaultSelector() as selector:
                selector.register(read_fd, selectors.EVENT_READ)
                while selector.get_map():
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TrainingFailure('training_timeout')
                    for key, _ in selector.select(min(remaining, .1)):
                        chunk = os.read(key.fd, 65536)
                        if not chunk:
                            selector.unregister(read_fd)
                            if buffer:
                                raise ValueError('Truncated frame')
                            break
                        total += len(chunk)
                        buffer += chunk
                        if total > MAX_TOTAL:
                            raise TrainingFailure('training_protocol_limit')
                        while b'\n' in buffer:
                            raw, buffer = buffer.split(b'\n', 1)
                            if len(raw) > MAX_FRAME:
                                raise TrainingFailure('training_protocol_limit')
                            frame = decode(raw)
                            if type(frame) is not dict or terminal is not None:
                                raise ValueError('Invalid frame ordering')
                            if frame.get('type') == 'log' and set(frame) == {'type', 'event', 'details'}:
                                logs += 1
                                if logs > MAX_LOGS or len(raw) > 4096:
                                    raise TrainingFailure('training_protocol_limit')
                                validate_log(frame['event'], frame['details'])
                                log(frame['event'], frame['details'])
                            elif frame.get('type') in ('result', 'error'):
                                terminal = frame
                            else:
                                raise ValueError('Invalid frame')
                        if len(buffer) > MAX_FRAME:
                            raise TrainingFailure('training_protocol_limit')
            process.wait(timeout=max(.001, deadline - time.monotonic()))
            if time.monotonic() >= deadline:
                raise TrainingFailure('training_timeout')
            if process.returncode == 71:
                raise TrainingFailure('execution_code_changed')
            if process.returncode != 0 or terminal is None:
                raise TrainingFailure('execution_failed')
            if terminal.get('type') == 'error':
                from .training_child import ERROR_CODES
                if set(terminal) != {'type', 'code'} or terminal['code'] not in ERROR_CODES:
                    raise ValueError('Invalid error')
                raise TrainingFailure(terminal['code'])
            if set(terminal) != {'type', 'artifact_hash', 'evaluation', 'manifest', 'environment'}:
                raise ValueError('Invalid result')
            evaluation, manifest = terminal['evaluation'], terminal['manifest']
            if (type(evaluation) is not dict or type(manifest) is not dict
                    or terminal['environment'] != loaded
                    or manifest.get('execution_environment') != loaded
                    or manifest.get('environment') != submission
                    or manifest.get('recipe') != spec.id):
                raise ValueError('Invalid provenance')
            path = runtime.artifact(attempt)
            with path.open('rb') as artifact:
                data = artifact.read(spec.budgets.artifact_bytes + 1)
            if not data or len(data) > spec.budgets.artifact_bytes or hashlib.sha256(data).hexdigest() != terminal['artifact_hash']:
                raise ValueError('Invalid artifact')
            return terminal['artifact_hash'], evaluation, manifest
    except subprocess.TimeoutExpired:
        raise TrainingFailure('training_timeout') from None
    finally:
        if process is not None:
            if process.poll() is None:
                process.kill()
            # Never release lease or publish failure before owned child is dead.
            process.wait()
        os.close(read_fd)
        if write_fd is not None:
            os.close(write_fd)
