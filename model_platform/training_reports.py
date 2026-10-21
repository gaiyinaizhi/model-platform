"""Read-only projection from one captured trained-model database snapshot."""
from .common import Problem
from . import training_report_schema as schema


def _json(value):
    schema.check(type(value) is str)
    return schema.parse_value(value.encode('utf-8'))


def _project(obj, keys):
    schema.check(type(obj) is dict)
    return {k: obj[k] for k in keys}


def report(store, tenant, model_id):
    try:
        schema.ident(model_id,'model')
    except ValueError:
        raise Problem(400,'invalid_input','Expected a registered model ID') from None
    # No Store/Runtime/DatasetStore constructor, environment probe, file access or
    # implicit migration. One transaction captures all associations together.
    with store.runtime.connection() as db:
        db.execute('BEGIN')
        model=store._owned(db,'models',tenant,model_id)
        return schema.encode(snapshot_report(store, db, tenant, model))


def snapshot_report(store, db, tenant, model):
    """Validate one already-owned row within the caller's read transaction."""
    model_id = model['id']
    try:
        manifest=_json(model['manifest_json'])
        if model['origin']!='trained' or manifest.get('recipe')!=schema.RECIPE:
            raise Problem(409,'training_report_unsupported','Only dataset-trained model reports are supported')
        run=db.execute('SELECT * FROM runs WHERE id=? AND tenant=?',(model['run_id'],tenant)).fetchone()
        attempt=db.execute('SELECT * FROM attempts WHERE id=? AND run_id=?',(model['attempt_id'],model['run_id'])).fetchone()
        ev=db.execute('SELECT * FROM evaluations WHERE id=?',(model['evaluation_id'],)).fetchone()
        bind=db.execute('SELECT * FROM run_dataset_bindings WHERE run_id=?',(model['run_id'],)).fetchone()
        schema.check(all(x is not None for x in (run,attempt,ev,bind)))
        if run['status']!='succeeded' or attempt['status']!='succeeded':
            raise Problem(409,'training_report_not_succeeded','A succeeded training record is required')
        schema.check(run['model_version_id']==model_id and attempt['error_code'] is None)
        schema.check(ev['attempt_id']==attempt['id'])
        schema.check(ev['artifact_sha256']==model['artifact_sha256'])
        body=_json(run['input_json']); binding=_json(bind['binding_json']); evaluation=_json(ev['result_json'])
        submission=_json(run['environment_json'])
        schema.check(schema.digest(body)==run['request_sha256'])
        schema.check(body['recipe']==schema.RECIPE and body['dataset']=={'dataset_version_id':bind['dataset_version_id']})
        schema.check(binding['dataset_version_id']==bind['dataset_version_id'] and binding['sha256']==bind['content_sha256'])
        schema.check(manifest['dataset']==binding and manifest['config']==body['config'])
        schema.check(manifest['environment']==submission and manifest['execution_environment']==evaluation['execution_environment'])
        schema.check(manifest.get('origin','trained')=='trained')
        for k,expected in [('model_version_id',model_id),('run_id',run['id']),('attempt_id',attempt['id']),
                           ('evaluation_id',ev['id']),('artifact_sha256',model['artifact_sha256']),('artifact_id',attempt['id'])]:
            schema.check(manifest[k]==expected)
        schema.check(all(model[k] is None for k in ('import_id','validation_id','catalog_revision')))
        schema.check(evaluation['dataset_sha256']==ev['evaluation_sha256']==binding['evaluation_sha256'])
        classes=binding['classes']
        schema.check(manifest['class_names']==binding['class_names']==[c['name'] for c in classes])
        schema.check(evaluation['category_mapping']==[
            {'category_id':c['id'],'coco_category_id':c['id']+1,'name':c['name']} for c in classes])
        # Existing training compatibility records the real evaluator parameters.
        from .dataset_detection import compatibility
        schema.check(evaluation['compatibility_key']==schema.digest(compatibility(evaluation)))
        history=list(db.execute('SELECT id,number,status FROM attempts WHERE run_id=? ORDER BY number',(run['id'],)))
        schema.check(1<=len(history)<=3 and history[-1]['id']==attempt['id'])
        schema.check([r['number'] for r in history]==list(range(1,len(history)+1)))
        schema.check(all(r['status']=='failed' for r in history[:-1]))
        training={'model_version_id':model_id,'run_id':run['id'],'attempt_id':attempt['id'],'evaluation_id':ev['id'],
                  'status':'succeeded','attempt_number':attempt['number'],'created_at':run['created_at'],
                  'started_at':attempt['started_at'],'finished_at':attempt['finished_at'],
                  **_project(manifest,('recipe','contract','format'))}
        for stamp in (model['created_at'],ev['created_at']):
            schema.finite(stamp); schema.check(attempt['started_at']<=stamp<=attempt['finished_at'])
        schema.finite(run['updated_at']); schema.check(run['updated_at']>=attempt['finished_at'])
        dataset={k:binding['sha256'] if k=='content_sha256' else binding[k] for k in schema.DATASET_KEYS}
        provenance={**_project(manifest,('config','training_config','config_sha256','dependencies','artifact_sha256','artifact_bytes',
                     'initial_parameters_sha256','final_parameters_sha256')),
                    'dataset':dataset,'submission_environment':_project(submission,('python','platform','code_sha256')),
                    'execution_environment':_project(manifest['execution_environment'],('python','platform','code_sha256'))}
        summary={**_project(manifest,('training_seconds','export_seconds','evaluation_seconds')),
                 'loss_history':manifest['training_loss_history']}
        exported={k:evaluation[k] for k in schema.EVALUATION_KEYS if k!='metrics'}
        exported['metrics']=_project(evaluation,schema.METRIC_KEYS)
        value={'schema_version':schema.SCHEMA_VERSION,'canonicalization':schema.CANONICALIZATION,
               'training':training,'provenance':provenance,'training_summary':summary,
               'exported_evaluation':exported,'exported_evaluation_sha256':schema.digest(exported),
               'verification':schema.VERIFICATION,'limitations':schema.LIMITATIONS}
        value['report_sha256']=schema.digest(value)
        schema.validate_report(value)
        return value
    except (ValueError,KeyError,TypeError,OverflowError,RecursionError,AttributeError,IndexError):
        raise Problem(409,'training_report_integrity','Training report evidence is invalid') from None
