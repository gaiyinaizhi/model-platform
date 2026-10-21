"""Read-only comparison of validated dataset-training evidence, not quality approval."""
from .common import Problem
from . import training_report_schema as schema
from .training_reports import snapshot_report
from .evaluation_comparisons import same_json

SCHEMA_VERSION = 'training-model-comparison/v1'
MAX_COMPARISON_BYTES = 2 * schema.MAX_REPORT_BYTES + 16 * 1024
SCOPE = {'internal_consistency_only': True, 'statistical_significance': False,
         'production_allowed': False, 'full_source_hash_comparison': True}


def facts(report):
    e = report['exported_evaluation']
    p = report['provenance']
    return {
        'evaluation_content': e['evaluation_sha256'],
        'evaluation_split': e['split'],
        'evaluation_role': e['role'],
        'classes': p['dataset']['classes'],
        'evaluation_protocol': {
            **{k: e[k] for k in ('contract', 'score_threshold', 'nms_iou_threshold',
                                 'max_detections', 'preprocessing')},
            'metrics': {k: e['metrics'][k] for k in ('protocol', 'evaluator', 'parameters')}},
        'execution_code': p['execution_environment']['code_sha256'],
        'execution_python': p['execution_environment']['python'],
        'execution_platform': p['execution_environment']['platform'],
        'dependencies': p['dependencies'],
    }


def difference(a, b):
    return None if a is None or b is None else b - a


def project(reports):
    a, b = reports
    af, bf = facts(a), facts(b)
    differences = [key for key in af if not same_json(af[key], bf[key])]
    # Identical ordered evaluation content and categories cannot coherently
    # describe different ground truth, even when execution conditions differ.
    if same_json(af['evaluation_content'], bf['evaluation_content']) and same_json(af['classes'], bf['classes']):
        def ground_truth(report):
            metrics = report['exported_evaluation']['metrics']
            return {'counts': {key: metrics['counts'][key] for key in (
                'images', 'images_with_ground_truth', 'images_without_ground_truth',
                'ground_truth_boxes', 'categories', 'categories_with_ground_truth')},
                'per_class': [row['ground_truth_boxes'] for row in metrics['per_class']]}
        if not same_json(ground_truth(a), ground_truth(b)):
            raise Problem(409, 'training_comparison_integrity', 'Evaluation ground-truth evidence is contradictory')
    deltas = None
    if not differences:
        am, bm = a['exported_evaluation']['metrics'], b['exported_evaluation']['metrics']
        deltas = {key: difference(am[key], bm[key]) for key in ('map_50', 'map_50_95', 'ar_100')}
        deltas['per_class'] = [
            {'category_id': x['category_id'], 'name': x['name'],
             **{key: difference(x[key], y[key]) for key in ('ap_50', 'ap_50_95', 'ar_100')}}
            for x, y in zip(am['per_class'], bm['per_class'])]
    return {'schema_version': SCHEMA_VERSION,
            'model_version_ids': [r['training']['model_version_id'] for r in reports],
            'compatible': not differences, 'differences': differences, 'reports': reports,
            'deltas': deltas, 'scope': dict(SCOPE)}


def compare(store, tenant, model_a_id, model_b_id):
    try:
        schema.ident(model_a_id, 'model'); schema.ident(model_b_id, 'model')
        schema.check(model_a_id != model_b_id)
    except ValueError:
        raise Problem(400, 'invalid_input', 'Choose two distinct registered model IDs') from None
    with store.runtime.connection() as db:
        db.execute('BEGIN')
        # Resolve BOTH ownerships first: an owned corrupt record cannot disclose
        # anything about a missing/foreign second model through error precedence.
        models = [store._owned(db, 'models', tenant, mid) for mid in (model_a_id, model_b_id)]
        try:
            reports = [snapshot_report(store, db, tenant, model) for model in models]
        except Problem as exc:
            code = {'training_report_unsupported': 'training_comparison_unsupported',
                    'training_report_not_succeeded': 'training_comparison_not_succeeded',
                    'training_report_integrity': 'training_comparison_integrity'}.get(exc.code)
            if code is None:
                raise
            raise Problem(409, code, 'Training comparison evidence is unavailable or invalid') from None
        result = project(reports)
        if len(schema.encode(result)) > MAX_COMPARISON_BYTES:
            raise Problem(409, 'training_comparison_integrity', 'Training comparison exceeds the evidence budget')
        return result
