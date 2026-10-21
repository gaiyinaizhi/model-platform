"""Portable stdlib-only training report projection and internal-consistency checks.

No source authenticity, artifact verification, inference, runtime or I/O implied.
"""
import json
import math
import re
from .evaluation_report_schema import (encode, digest, _bounds, _unique, _number,
    MAX_REPORT_BYTES, MAX_DEPTH, MAX_STRING)

SCHEMA_VERSION = 'training-evaluation-report/v1'
CANONICALIZATION = 'python-json-sort-keys-ascii-compact/v1'
RECIPE = 'yolox-nano-dataset/v1'
CONTRACT = 'rgb-dataset-detection/v1'
FORMAT = 'yolox-nano-dataset-onnx/v1'
FIXED_CONFIG = {'batch_size': 4, 'input_size': 160, 'from_scratch': True,
                'optimizer': 'Adam', 'checkpoint_selection': 'final-epoch'}
PACKAGES = ('torch', 'numpy', 'onnx', 'onnxruntime', 'Pillow', 'pycocotools', 'PyYAML')
VERIFICATION = {'internal_consistency_only': True, 'trusted_authenticity': False,
                'signature_present': False, 'reevaluated': False,
                'artifact_bytes_verified': False, 'dataset_bytes_verified': False,
                'prediction_bytes_included': False}
LIMITATIONS = {'production_allowed': False, 'official_coco_benchmark': False,
               'device_certified': False, 'external_audit': False,
               'quality_pass_asserted': False}
TRAINING_KEYS = ('model_version_id','run_id','attempt_id','evaluation_id','status','attempt_number',
                 'created_at','started_at','finished_at','recipe','contract','format')
DATASET_KEYS = ('dataset_version_id','content_sha256','train_count','validation_count','test_count',
                'evaluation_count','evaluation_split','train_sha256','validation_sha256','test_sha256',
                'evaluation_sha256','negative_policy','classes')
PROVENANCE_KEYS = ('dataset','config','training_config','config_sha256','submission_environment',
                   'execution_environment','dependencies','artifact_sha256','artifact_bytes',
                   'initial_parameters_sha256','final_parameters_sha256')
SUMMARY_KEYS = ('loss_history','training_seconds','export_seconds','evaluation_seconds')
METRIC_KEYS = ('protocol','status','map_50','map_50_95','ar_100','evaluator','parameters','counts','per_class')
EVALUATION_KEYS = ('contract','split','role','artifact_sha256','dataset_version_id','dataset_content_sha256',
                   'evaluation_sha256','score_threshold','nms_iou_threshold','max_detections',
                   'preprocessing','metrics')
TOP_KEYS = ('schema_version','canonicalization','training','provenance','training_summary',
            'exported_evaluation','exported_evaluation_sha256','verification','limitations','report_sha256')
PARAMETERS = {'iou_type': 'bbox', 'area_ranges': [[0.0, 10000000000.0]], 'area_range_labels': ['all'],
              'max_detections': [1,10,100], 'reported_max_detections': 100,
              'max_detections_scope': 'per_image_per_category', 'use_categories': True,
              'box_format': 'xyxy_original_image_pixels_continuous',
              'category_averaging': 'macro_over_categories_with_ground_truth', 'no_ground_truth_ap': None,
              'crowd_ignore_group_policy': 'rejected', 'score_filtering': 'none', 'nms': 'none',
              'score_tie_order': 'stable_image_input_order_then_prediction_input_order'}
PREPROCESSING = {'version':'rgb-to-bgr-pillow-bilinear-top-left-114-f32-0-255/v1',
                 'input_size':160,'orientation':'raw-decoded-no-exif-transpose','epsilon_box_clamp':1e-9}
COUNT_KEYS = ('images','images_with_ground_truth','images_without_ground_truth','images_with_predictions',
              'ground_truth_boxes','predictions','predictions_above_max_detections','categories','categories_with_ground_truth')
CLASS_KEYS = ('category_id','name','ground_truth_boxes','predictions','status','ap_50','ap_50_95','ar_100')


def check(condition):
    if not condition:
        raise ValueError('Invalid training report')


def obj(v, keys):
    check(type(v) is dict and set(v) == set(keys))
    return v


def exact(v, expected):
    # Enforce bool/number distinctions recursively; int vs float is a JSON number.
    if type(expected) is dict:
        obj(v, expected)
        for k in expected: exact(v[k], expected[k])
    elif type(expected) is list:
        check(type(v) is list and len(v) == len(expected))
        for a,b in zip(v,expected): exact(a,b)
    elif type(expected) in (int,float):
        check(type(v) in (int,float) and v == expected)
    else:
        check(type(v) is type(expected) and v == expected)


def integer(v, low, high):
    check(type(v) is int and low <= v <= high)


def finite(v, low=0, high=1e12):
    check(type(v) in (int,float) and math.isfinite(v) and low <= v <= high)


def sha(v):
    check(type(v) is str and re.fullmatch('[0-9a-f]{64}',v) is not None)


def ident(v, prefix):
    check(type(v) is str and re.fullmatch(prefix+'_[0-9a-f]{32}',v) is not None)


def name(v):
    check(type(v) is str and 1 <= len(v) <= 80 and v.strip()
          and not any(ord(c)<32 or ord(c)==127 for c in v))


def environment(v):
    obj(v, ('python','platform','code_sha256'))
    check(type(v['python']) is str and re.fullmatch('[A-Za-z0-9.+_-]{1,160}',v['python']))
    check(type(v['platform']) is str and re.fullmatch('[A-Za-z0-9 ._()+-]{1,240}',v['platform']))
    sha(v['code_sha256'])


def metric(v):
    if v is not None: finite(v,0,1)


def _validate(v):
    _bounds(v)
    obj(v, TOP_KEYS)
    exact(v['schema_version'],SCHEMA_VERSION); exact(v['canonicalization'],CANONICALIZATION)
    t=obj(v['training'],TRAINING_KEYS)
    for k,prefix in [('model_version_id','model'),('run_id','run'),('attempt_id','attempt'),('evaluation_id','eval')]: ident(t[k],prefix)
    for k,c in [('status','succeeded'),('recipe',RECIPE),('contract',CONTRACT),('format',FORMAT)]: exact(t[k],c)
    integer(t['attempt_number'],1,3)
    for k in ('created_at','started_at','finished_at'): finite(t[k])
    check(t['created_at'] <= t['started_at'] <= t['finished_at'])
    p=obj(v['provenance'],PROVENANCE_KEYS); d=obj(p['dataset'],DATASET_KEYS)
    ident(d['dataset_version_id'],'dsv')
    for k in ('content_sha256','train_sha256','validation_sha256','evaluation_sha256'): sha(d[k])
    integer(d['train_count'],1,128); integer(d['validation_count'],1,64); integer(d['test_count'],0,64)
    integer(d['evaluation_count'],1,64)
    check(d['evaluation_split'] in ('val','test'))
    if d['test_count']:
        sha(d['test_sha256']); exact(d['evaluation_split'],'test')
        check(d['evaluation_count']==d['test_count'] and d['evaluation_sha256']==d['test_sha256'])
    else:
        exact(d['test_sha256'],None); exact(d['evaluation_split'],'val')
        check(d['evaluation_count']==d['validation_count'] and d['evaluation_sha256']==d['validation_sha256'])
    check(d['negative_policy'] in ('allow_empty','reject'))
    classes=d['classes']; check(type(classes) is list and 1<=len(classes)<=20)
    for i,c in enumerate(classes):
        obj(c,('id','name')); integer(c['id'],i,i); name(c['name'])
    check(len({c['name'] for c in classes})==len(classes))
    cfg=obj(p['config'],('epochs','learning_rate','seed'))
    integer(cfg['epochs'],1,20); finite(cfg['learning_rate'],.0001,.01); integer(cfg['seed'],0,2**32-1)
    exact(p['training_config'],dict(cfg,**FIXED_CONFIG))
    sha(p['config_sha256']); check(p['config_sha256']==digest(p['training_config']))
    environment(p['submission_environment']); environment(p['execution_environment'])
    exact(p['submission_environment']['code_sha256'],p['execution_environment']['code_sha256'])
    deps=obj(p['dependencies'],PACKAGES)
    for value in deps.values(): check(type(value) is str and re.fullmatch('[A-Za-z0-9.+_-]{1,160}',value))
    for k in ('artifact_sha256','initial_parameters_sha256','final_parameters_sha256'): sha(p[k])
    check(p['initial_parameters_sha256']!=p['final_parameters_sha256'])
    integer(p['artifact_bytes'],1,8*1024*1024)
    s=obj(v['training_summary'],SUMMARY_KEYS)
    check(type(s['loss_history']) is list and len(s['loss_history'])==cfg['epochs'])
    for loss in s['loss_history']: finite(loss, -1e12, 1e12)
    for k in ('training_seconds','export_seconds','evaluation_seconds'): finite(s[k])
    e=obj(v['exported_evaluation'],EVALUATION_KEYS)
    exact(e['contract'],'dataset-coco-style-bbox/v1')
    exact(e['split'],d['evaluation_split']); exact(e['role'],'held-out-test' if e['split']=='test' else 'validation-only')
    for a,b in [('dataset_version_id','dataset_version_id'),('dataset_content_sha256','content_sha256'),('evaluation_sha256','evaluation_sha256')]: exact(e[a],d[b])
    exact(e['artifact_sha256'],p['artifact_sha256'])
    exact(e['score_threshold'],.001); exact(e['nms_iou_threshold'],.65)
    exact(e['max_detections'],{'upstream':100,'scope':'global-per-image-class-aware-nms'})
    exact(e['preprocessing'],PREPROCESSING)
    m=obj(e['metrics'],METRIC_KEYS); exact(m['protocol'],'coco-style-bbox-subset/v1')
    exact(m['evaluator'],{'implementation':'pycocotools.cocoeval.COCOeval','version':deps['pycocotools']})
    params=obj(m['parameters'],(*PARAMETERS,'iou_thresholds','recall_thresholds'))
    for k,c in PARAMETERS.items(): exact(params[k],c)
    for k,n,start,step in [('iou_thresholds',10,.5,.05),('recall_thresholds',101,0,.01)]:
        values=params[k]; check(type(values) is list and len(values)==n)
        for i,x in enumerate(values): finite(x,0,1); check(abs(x-(start+i*step))<=1e-12)
    counts=obj(m['counts'],COUNT_KEYS)
    for k in COUNT_KEYS: integer(counts[k],0,6400)
    exact(counts['images'],d['evaluation_count']); exact(counts['categories'],len(classes))
    check(counts['images_with_ground_truth']+counts['images_without_ground_truth']==counts['images'])
    check(counts['images_with_predictions']<=counts['images'] and counts['categories_with_ground_truth']<=len(classes))
    exact(counts['predictions_above_max_detections'],0)
    check(counts['ground_truth_boxes']<=counts['images_with_ground_truth']*100 and counts['ground_truth_boxes']>=counts['images_with_ground_truth'])
    check(counts['predictions']<=counts['images_with_predictions']*100 and counts['predictions']>=counts['images_with_predictions'])
    rows=m['per_class']; check(type(rows) is list and len(rows)==len(classes))
    present=[]
    for i,row in enumerate(rows):
        obj(row,CLASS_KEYS); integer(row['category_id'],i,i); exact(row['name'],classes[i]['name'])
        integer(row['ground_truth_boxes'],0,6400); integer(row['predictions'],0,6400)
        exact(row['status'],'evaluated' if row['ground_truth_boxes'] else 'no_ground_truth')
        if row['ground_truth_boxes']: present.append(row)
        for key in ('ap_50','ap_50_95','ar_100'):
            metric(row[key]); check((row[key] is not None) == bool(row['ground_truth_boxes']))
            if row['ground_truth_boxes'] and not row['predictions']: exact(row[key],0)
    check(sum(r['ground_truth_boxes'] for r in rows)==counts['ground_truth_boxes'])
    check(sum(r['predictions'] for r in rows)==counts['predictions'])
    exact(counts['categories_with_ground_truth'],len(present))
    exact(m['status'],'evaluated' if present else 'no_ground_truth')
    for key,ck in [('map_50','ap_50'),('map_50_95','ap_50_95'),('ar_100','ar_100')]:
        metric(m[key]); check((m[key] is not None)==bool(present))
        if present: check(abs(m[key]-sum(r[ck] for r in present)/len(present))<=1e-10)
    exact(v['verification'],VERIFICATION); exact(v['limitations'],LIMITATIONS)
    sha(v['exported_evaluation_sha256']); check(v['exported_evaluation_sha256']==digest(e))
    sha(v['report_sha256']); check(v['report_sha256']==digest({k:x for k,x in v.items() if k!='report_sha256'}))
    check(len(encode(v))<=MAX_REPORT_BYTES)
    return v


def validate_report(value):
    try: return _validate(value)
    except (ValueError,TypeError,KeyError,OverflowError,RecursionError,AttributeError):
        raise ValueError('Invalid training evaluation report') from None


def parse_value(raw):
    try:
        check(type(raw) is bytes and 0<len(raw)<=MAX_REPORT_BYTES)
        text=raw.decode('utf-8',errors='strict'); depth=0; quoted=escaped=False; length=0
        for c in text:
            if quoted:
                length+=1; check(length<=MAX_STRING*6+2)
                if escaped: escaped=False
                elif c=='\\': escaped=True
                elif c=='"': quoted=False
            elif c=='"': quoted=True; length=0
            elif c in '[{': depth+=1; check(depth<=MAX_DEPTH)
            elif c in ']}': depth-=1; check(depth>=0)
        check(depth==0 and not quoted)
        value=json.loads(text,object_pairs_hook=_unique,parse_int=lambda s:_number(s,int),
                         parse_float=lambda s:_number(s,float),parse_constant=lambda _:check(False))
        _bounds(value); check(encode(value)==raw)
        return value
    except (ValueError,TypeError,OverflowError,RecursionError,UnicodeError):
        raise ValueError('Invalid training evaluation report') from None


def parse_report(raw):
    return validate_report(parse_value(raw))
