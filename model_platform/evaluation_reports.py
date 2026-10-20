"""Read-only bounded projection of one captured, verified evaluation."""
from .common import Problem
from .evaluation_comparison_validation import validate as validate_source
from .model_evaluations import ID
from .evaluation_report_schema import (SCHEMA_VERSION, CANONICALIZATION, EVALUATION_KEYS,
    DATASET_KEYS, RESULT_KEYS, VERIFICATION, LIMITATIONS, encode, digest, validate_report)


def report(evaluations, tenant, evaluation_id):
    if type(evaluation_id) is not str or not ID.fullmatch(evaluation_id):
        raise Problem(400, 'invalid_input', 'Expected an independent evaluation ID')
    row = evaluations._row(tenant, evaluation_id)
    if row['status'] != 'succeeded':
        raise Problem(409, 'evaluation_not_succeeded', 'Evaluation record must have succeeded')
    detail = evaluations.detail(row)
    try:
        validate_source(detail)
        binding, result = detail['binding'], detail['result']
        dataset = binding['dataset']
        projected_dataset = {key: dataset[key] for key in DATASET_KEYS if key != 'source_selection_sha256'}
        projected_dataset['source_selection_sha256'] = (dataset['source']['selection_sha256']
                                                        if dataset['source'] is not None else None)
        provenance = {key: binding[key] for key in ('model', 'categories', 'category_mapping_sha256',
                      'protocol', 'protocol_sha256', 'submission_environment')}
        provenance['dataset'] = projected_dataset
        exported = {key: result[key] for key in RESULT_KEYS}
        value = {'schema_version': SCHEMA_VERSION, 'canonicalization': CANONICALIZATION,
                 'evaluation': {key: detail[key] for key in EVALUATION_KEYS},
                 'provenance': provenance, 'exported_result': exported,
                 'source_result_sha256': detail['result_sha256'],
                 'source_binding_sha256': result['digests']['binding_sha256'],
                 'source_prediction_sha256': result['digests']['prediction_sha256'],
                 'exported_result_sha256': digest(exported),
                 'verification': VERIFICATION, 'limitations': LIMITATIONS}
        value['report_sha256'] = digest(value)
        validate_report(value)
        return encode(value)
    except (ValueError, KeyError, TypeError, OverflowError, RecursionError):
        raise Problem(409, 'evaluation_integrity', 'Evaluation report evidence is invalid') from None
