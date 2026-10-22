"""Private read-only evaluator entry point; only the parent may publish evidence."""
import contextlib
import os
from pathlib import Path
import resource
import sqlite3

from .common import Problem, encode, environment
from .evaluation_process import MAX_INPUT, ERROR_CODES, decode, validate_request, validate_result
from .model_evaluations import ModelEvaluationStore, MAX_CHILD_BYTES, ADDRESS_BYTES, CPU_SECONDS, measure
from .store import Store


class EvaluationRuntime:
    """Read existing runtime paths only; construction creates no directories."""
    def __init__(self, home):
        self.home = Path(home).resolve()
        self.db = self.home / 'platform.sqlite3'

    @contextlib.contextmanager
    def connection(self):
        connection = sqlite3.connect(self.db.as_uri() + '?mode=ro', uri=True, timeout=10)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute('PRAGMA query_only=ON')
            yield connection
        finally:
            connection.close()


class EvaluationStore:
    """Reuse only model reads, without Store initialization or write methods."""
    _owned = staticmethod(Store._owned)
    model = Store.model

    def __init__(self, runtime):
        self.runtime = runtime


class EvaluationReader:
    """The measurement's existing read contract, without queue/evidence methods."""
    binding_for = ModelEvaluationStore.binding_for
    verify_binding = ModelEvaluationStore.verify_binding
    verify_code = staticmethod(ModelEvaluationStore.verify_code)
    expected_digests = staticmethod(ModelEvaluationStore.expected_digests)

    def __init__(self, runtime):
        self.runtime = runtime
        self.store = EvaluationStore(runtime)


def main(input_fd, output_fd, lease_fd, expected_hash):
    # Keep lease_fd open until process exit. Never unlock this shared lease.
    resource.setrlimit(resource.RLIMIT_AS, (ADDRESS_BYTES, ADDRESS_BYTES))
    resource.setrlimit(resource.RLIMIT_CPU, (CPU_SECONDS, CPU_SECONDS))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    try:
        with os.fdopen(input_fd, 'rb') as request:
            raw = request.read(MAX_INPUT + 1)
        if len(raw) > MAX_INPUT:
            raise Problem(409, 'evaluation_result_limit', 'Evaluation request exceeds bounded channel')
        request = validate_request(decode(raw))
        actual = environment()
        if actual != request['loaded_environment'] or actual['code_sha256'] != expected_hash:
            raise Problem(409, 'execution_code_changed', 'Execution source changed after submission')
        runtime = EvaluationRuntime(request['home'])
        evaluations = EvaluationReader(runtime)
        with runtime.connection() as db:
            row = db.execute("SELECT * FROM model_evaluations WHERE id=? AND status='running'",
                             (request['evaluation_id'],)).fetchone()
        if row is None:
            raise Problem(409, 'evaluation_integrity', 'Evaluation is not running')
        payload = validate_result(measure(evaluations, row))
        if environment() != actual:
            raise Problem(409, 'execution_code_changed', 'Execution source changed during evaluation')
        raw = encode(payload)
        if len(raw) > MAX_CHILD_BYTES:
            raise Problem(409, 'evaluation_result_limit', 'Evaluator result exceeds bounded channel')
    except Problem as exc:
        raw = encode({'error_code': exc.code if exc.code in ERROR_CODES else 'evaluation_failed'})
    except ImportError:
        raw = encode({'error_code': 'evaluation_dependencies_missing'})
    except Exception:
        raw = encode({'error_code': 'evaluation_failed'})
    view = memoryview(raw)
    while view:
        view = view[os.write(output_fd, view):]
