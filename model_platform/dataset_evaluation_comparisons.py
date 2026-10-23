"""Read-only comparison of verified trained-model independent report bytes.

Different models and training lineages are permitted; this is not a quality gate.
"""
import json

from .common import Problem, encode
from . import dataset_evaluation_report_schema as trained
from . import evaluation_report_schema as schema
from .evaluation_comparisons import same_json
from .model_evaluations import ID, request

SCHEMA_VERSION = 'dataset-trained-independent-evaluation-comparison/v1'
MAX_COMPARISON_BYTES = 2 * 1024 * 1024 + 64 * 1024


def facts(report):
    p, result = report['provenance'], report['exported_result']
    dataset, metrics = p['dataset'], result['metrics']
    submission, execution = p['submission_environment'], result['environment']
    return {
        'dataset.version': dataset['dataset_version_id'],
        'dataset.content': dataset['content_sha256'],
        'dataset.scope': {key: dataset[key] for key in
                          ('purpose', 'split', 'images', 'boxes', 'source_selection_sha256')},
        'categories': p['categories'],
        'category_mapping': p['category_mapping_sha256'],
        'protocol': p['protocol'],
        'protocol_digest': p['protocol_sha256'],
        'metric_protocol': metrics['protocol'],
        'metric_evaluator': metrics['evaluator'],
        'metric_parameters': metrics['parameters'],
        'operating_point': {key: result['operating_point'][key] for key in
                            ('protocol', 'iou_threshold', 'score_threshold', 'score_operator')},
        'submission.code': submission['code_sha256'],
        'submission.python': submission['python'],
        'submission.platform': submission['platform'],
        'submission.dependencies': json.loads(submission['dependencies']),
        'execution.code': execution['code_sha256'],
        'execution.python': execution['python'],
        'execution.platform': execution['platform'],
        'execution.dependencies': execution['dependencies'],
    }


def measurements(report):
    result = report['exported_result']
    return {**{key: result['metrics'][key] for key in ('map_50', 'map_50_95', 'ar_100')},
            **{key: result['operating_point']['totals'][key]
               for key in ('precision', 'recall', 'tp', 'fp', 'fn')}}


def _ground_truth(report):
    metrics = report['exported_result']['metrics']
    return {'counts': {key: metrics['counts'][key] for key in
                      ('images', 'images_with_ground_truth', 'images_without_ground_truth',
                       'ground_truth_boxes', 'categories', 'categories_with_ground_truth')},
            'per_class': [row['ground_truth_boxes'] for row in metrics['per_class']]}


def project(a_raw, b_raw):
    """Pure portable projection, preserving canonical report text exactly."""
    a, b = schema.parse_report(a_raw), schema.parse_report(b_raw)
    if (a['schema_version'] != trained.SCHEMA_VERSION or b['schema_version'] != trained.SCHEMA_VERSION
            or a['evaluation']['id'] == b['evaluation']['id']):
        raise ValueError('Expected two distinct trained evaluation reports')
    af, bf = facts(a), facts(b)
    # The same frozen content and category map cannot represent contradictory GT.
    if (same_json(af['dataset.content'], bf['dataset.content'])
            and same_json(af['categories'], bf['categories'])
            and not same_json(_ground_truth(a), _ground_truth(b))):
        raise ValueError('Contradictory ground-truth evidence')
    reasons = [{'field': key, 'reason': key + ' differs'}
               for key in af if not same_json(af[key], bf[key])]
    am, bm = measurements(a), measurements(b)
    def diff(x, y):
        return None if x is None or y is None else y - x
    delta = None
    if not reasons:
        ar, br = a['exported_result'], b['exported_result']
        per_class = []
        for x, y, xo, yo in zip(ar['metrics']['per_class'], br['metrics']['per_class'],
                                ar['operating_point']['per_class'], br['operating_point']['per_class']):
            per_class.append({'category_id': x['category_id'], 'name': x['name'],
                              **{key: diff(x[key], y[key]) for key in ('ap_50', 'ap_50_95', 'ar_100')},
                              **{key: diff(xo[key], yo[key]) for key in ('precision', 'recall', 'tp', 'fp', 'fn')}})
        delta = {'overall': {key: diff(am[key], bm[key]) for key in am}, 'per_class': per_class}
    value = {'schema_version': SCHEMA_VERSION, 'a_report': a_raw.decode('utf-8'),
             'b_report': b_raw.decode('utf-8'), 'compatible': not reasons,
             'incompatibilities': reasons, 'delta': delta}
    if len(encode(value)) > MAX_COMPARISON_BYTES:
        raise ValueError('Comparison exceeds the bounded evidence budget')
    return value


def compare(evaluations, tenant, evaluation_a_id, evaluation_b_id):
    if (any(type(item) is not str or not ID.fullmatch(item)
            for item in (evaluation_a_id, evaluation_b_id)) or evaluation_a_id == evaluation_b_id):
        raise Problem(400, 'invalid_input', 'Choose two distinct evaluation IDs')
    # One read snapshot and BOTH ownership checks precede capability/state/integrity.
    with evaluations.runtime.connection() as db:
        db.execute('BEGIN')
        rows = [db.execute(evaluations.read_query() + 'WHERE e.tenant=? AND e.id=?',
                           (tenant, item)).fetchone() for item in (evaluation_a_id, evaluation_b_id)]
    if any(row is None for row in rows):
        raise Problem(404, 'not_found', 'Evaluation resource not found')
    try:
        protocols = [request(json.loads(row['request_json']))['protocol'] for row in rows]
    except (Problem, ValueError, TypeError, KeyError):
        raise Problem(409, 'evaluation_integrity', 'Evaluation request identity is invalid') from None
    if any(protocol != trained.PROTOCOL for protocol in protocols):
        raise Problem(409, 'evaluation_comparison_unsupported', 'Expected trained independent evaluations')
    if any(row['status'] != 'succeeded' for row in rows):
        raise Problem(409, 'evaluation_not_succeeded', 'Both evaluation records must have succeeded')
    try:
        details = [evaluations.detail(row) for row in rows]
        raws = [trained.project(detail) for detail in details]
        result = project(*raws)
        if any(json.loads(raw)['evaluation']['id'] != item for raw, item in
               zip(raws, (evaluation_a_id, evaluation_b_id))):
            raise ValueError('Mismatched evaluation identity')
        return result
    except (ValueError, KeyError, TypeError, OverflowError, RecursionError):
        raise Problem(409, 'evaluation_integrity', 'Evaluation comparison evidence is invalid') from None
