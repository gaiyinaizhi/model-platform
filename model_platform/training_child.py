"""Private child entry point: executes a static recipe, never publishes a model."""
import os
import contextlib
import sqlite3

from .algorithm import TrainingFailure
from .common import Runtime, Problem, encode, environment
from .training_process import MAX_INPUT, MAX_FRAME, decode, validate_log

ERROR_CODES = frozenset('''execution_failed execution_code_changed training_timeout
optional_dependencies_missing degenerate_features invalid_input dataset_integrity
dataset_missing dataset_corrupt
not_found dataset_training_purpose dataset_training_budget dataset_training_annotations
 dataset_train_classes fashion_artifact_too_large fashion_dataset_invalid
fashion_dataset_unavailable fashion_dataset_unavailable_or_invalid fashion_dataset_timeout fashion_evaluation_empty
fashion_logits_invalid fashion_onnx_parity_failed fashion_thread_budget_unavailable
fashion_training_nonfinite fashion_export_timeout fashion_evaluation_timeout
fashion_total_timeout detection_artifact_too_large detection_dataset_unavailable_or_invalid
 detection_onnx_box_parity_failed detection_onnx_count_parity_failed
 detection_onnx_map_parity_failed detection_onnx_per_class_parity_failed
 detection_onnx_raw_parity_failed detection_outputs_invalid detection_parameters_unchanged
 detection_thread_budget_unavailable detection_training_nonfinite detection_export_timeout
 detection_evaluation_timeout training_protocol_limit'''.split())


class TrainingRuntime(Runtime):
    @contextlib.contextmanager
    def connection(self):
        # Read immutable dataset metadata only. No child lifecycle DB writes.
        connection = sqlite3.connect(self.db.as_uri() + '?mode=ro', uri=True, timeout=10)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
        finally:
            connection.close()


def main(input_fd, output_fd, lease_fd, expected_hash):
    # Keep lease_fd open until process exit; never LOCK_UN this shared OFD.
    def emit(value):
        raw = encode(value)
        if len(raw) > MAX_FRAME:
            raise TrainingFailure('training_protocol_limit')
        view = memoryview(raw + b'\n')
        while view:
            view = view[os.write(output_fd, view):]

    def log(event, details):
        validate_log(event, details)
        emit({'type': 'log', 'event': event, 'details': details})

    try:
        with os.fdopen(input_fd, 'rb') as request:
            raw = request.read(MAX_INPUT + 1)
        if len(raw) > MAX_INPUT:
            raise TrainingFailure('training_protocol_limit')
        request = decode(raw)
        actual = environment()
        if (actual != request['loaded'] or actual['code_sha256'] != expected_hash
                or expected_hash != request['submission']['code_sha256']):
            raise TrainingFailure('execution_code_changed')
        from .recipes import get_recipe
        from .common import RECIPE, training_request
        body = request['body']
        training_request({k: v for k, v in body.items() if not k.startswith('_')})
        spec = get_recipe(body.get('recipe', RECIPE))
        runtime = TrainingRuntime(request['home'])
        artifact_hash, evaluation, manifest = spec.execute(spec, runtime, body,
            runtime.artifact(request['attempt']), log, actual, request['submission'])
        if environment() != actual:
            raise TrainingFailure('execution_code_changed')
        emit({'type': 'result', 'artifact_hash': artifact_hash, 'evaluation': evaluation,
              'manifest': manifest, 'environment': actual})
    except (Problem, TrainingFailure) as exc:
        emit({'type': 'error', 'code': exc.code if exc.code in ERROR_CODES else 'execution_failed'})
    except Exception:
        emit({'type': 'error', 'code': 'execution_failed'})
