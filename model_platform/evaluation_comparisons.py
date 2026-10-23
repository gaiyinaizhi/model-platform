"""Read-only, strict repeatability comparisons of independent measurements.

This is neither the trained-model comparison nor a production quality gate.
"""
import json
import math

from .common import Problem, encode
from .model_evaluations import ID
from .evaluation_comparison_validation import validate

SCHEMA_VERSION = 'independent-evaluation-comparison/v1'
MAX_COMPARISON_BYTES = 1100 * 1024


def facts(detail):
    """Ordered compatibility facts. Environment comparisons are A-to-B only."""
    binding, result = detail['binding'], detail['result']
    dataset, metrics = binding['dataset'], result['metrics']
    submission, execution = binding['submission_environment'], result['environment']
    operating = result['operating_point']
    dependencies = json.loads(submission['dependencies'])
    if not isinstance(dependencies, dict) or not isinstance(execution['dependencies'], dict):
        raise ValueError('Invalid dependency evidence')
    return {
        'dataset.version': dataset['dataset_version_id'],
        'dataset.content': dataset['content_sha256'],
        'dataset.samples': dataset['samples'],
        'dataset.source_selection': dataset['source']['selection_sha256'] if dataset['source'] else None,
        'split': dataset['split'],
        'categories': binding['categories'],
        'category_mapping': binding['category_mapping_sha256'],
        'protocol': binding['protocol'],
        'protocol_digest': binding['protocol_sha256'],
        'metric_protocol': metrics['protocol'],
        'metric_evaluator': metrics['evaluator'],
        'metric_parameters': metrics['parameters'],
        'operating_point': {key: operating[key] for key in
                            ('protocol', 'iou_threshold', 'score_threshold', 'score_operator')},
        'submission.code': submission['code_sha256'],
        'submission.python': submission['python'],
        'submission.platform': submission['platform'],
        'submission.dependencies': dependencies,
        'execution.code': execution['code_sha256'],
        'execution.python': execution['python'],
        'execution.platform': execution['platform'],
        'execution.dependencies': execution['dependencies'],
    }


def same_json(a, b):
    """JSON structural equality across Python/JS number representations."""
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if type(a) in (int, float) and type(b) in (int, float):
        return a == b
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same_json(a[key], b[key]) for key in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(same_json(x, y) for x, y in zip(a, b))
    return a == b


def measurements(detail):
    result = detail['result']
    values = {key: result['metrics'][key] for key in ('map_50', 'map_50_95', 'ar_100')}
    values.update({key: result['operating_point']['totals'][key]
                   for key in ('precision', 'recall', 'tp', 'fp', 'fn')})
    for key, value in values.items():
        if key in ('tp', 'fp', 'fn'):
            if type(value) is not int or not 0 <= value <= 100000:
                raise ValueError('Invalid count')
        elif value is not None and (type(value) not in (float, int) or not math.isfinite(value) or not 0 <= value <= 1):
            raise ValueError('Invalid metric')
    return values


def compare(evaluations, tenant, evaluation_a_id, evaluation_b_id):
    for item_id in (evaluation_a_id, evaluation_b_id):
        if not isinstance(item_id, str) or not ID.fullmatch(item_id):
            raise Problem(400, 'invalid_input', 'Expected an independent evaluation ID')
    if evaluation_a_id == evaluation_b_id:
        raise Problem(400, 'invalid_input', 'Choose two distinct evaluation records')
    # Resolve BOTH owners before state or evidence checks to avoid an oracle
    # that reveals state/tamper on one side while the other is inaccessible.
    rows = [evaluations._row(tenant, item_id) for item_id in (evaluation_a_id, evaluation_b_id)]
    from .model_evaluations import require_imported_capability
    for row in rows:
        require_imported_capability(row, 'comparison')
    if any(row['status'] != 'succeeded' for row in rows):
        raise Problem(409, 'evaluation_not_succeeded', 'Both evaluation records must have succeeded')
    a, b = [evaluations.detail(row) for row in rows]
    try:
        validate(a)
        validate(b)
        af, bf = facts(a), facts(b)
        am, bm = measurements(a), measurements(b)
        reasons = [{'field': key, 'reason': key + ' differs'} for key in af
                   if not same_json(af[key], bf[key])]
        delta = None if reasons else {key: None if am[key] is None or bm[key] is None
                                     else bm[key] - am[key] for key in am}
        response = {'schema_version': SCHEMA_VERSION, 'a': a, 'b': b,
                    'compatible': not reasons, 'incompatibilities': reasons, 'delta': delta}
        if len(encode(response)) > MAX_COMPARISON_BYTES:
            raise ValueError('Comparison response exceeds the bounded contract')
        return response
    except (KeyError, TypeError, ValueError, OverflowError):
        raise Problem(409, 'evaluation_integrity', 'Evaluation comparison evidence is invalid') from None
