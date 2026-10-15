"""Dataset detection serving identity, separate from training and business approval."""
from . import dataset_detection as recipe, detection
from .common import digest
from .detection_evaluation import PROTOCOL
import math


def protocol():
    return {'preprocessing': detection.metadata()['preprocessing'],
            'box_format': 'original-xyxy-half-open', 'input_size': detection.INPUT_SIZE,
            'score_threshold': detection.SCORE_THRESHOLD,
            'evaluation_score_threshold': detection.EVALUATION_SCORE_THRESHOLD,
            'nms_iou_threshold': detection.NMS_IOU_THRESHOLD, 'max_detections': 100}


def public_identity(manifest):
    return {'recipe': manifest['recipe'], 'artifact_sha256': manifest['artifact_sha256'],
            'run_id': manifest['run_id'], 'attempt_id': manifest['attempt_id'],
            'evaluation_id': manifest['evaluation_id'],
            'dataset_version_id': manifest['dataset']['dataset_version_id'],
            'dataset_sha256': manifest['dataset']['sha256'],
            'train_sha256': manifest['dataset']['train_sha256'],
            'evaluation_sha256': manifest['dataset']['evaluation_sha256'],
            'evaluation_split': manifest['dataset']['evaluation_split'],
            'class_names': list(manifest['class_names']), 'serving_protocol': protocol(),
            'quality': {'production_allowed': False, 'status': 'development-only',
                        'warnings': ['research-only', 'deployment-readiness-is-not-model-quality']}}


def validate_evaluation_protocol(evaluation, dependencies):
    expected = {
        'iou_type':'bbox', 'area_ranges':[[0.0,10000000000.0]], 'area_range_labels':['all'],
        'max_detections':[1,10,100], 'reported_max_detections':100,
        'max_detections_scope':'per_image_per_category', 'use_categories':True,
        'box_format':'xyxy_original_image_pixels_continuous',
        'category_averaging':'macro_over_categories_with_ground_truth', 'no_ground_truth_ap':None,
        'crowd_ignore_group_policy':'rejected', 'score_filtering':'none', 'nms':'none',
        'score_tie_order':'stable_image_input_order_then_prediction_input_order'}
    params = evaluation['parameters']
    if (evaluation['contract'] != 'dataset-coco-style-bbox/v1' or evaluation['protocol'] != PROTOCOL
            or evaluation['evaluator'] != {'implementation':'pycocotools.cocoeval.COCOeval',
                                          'version':dependencies['pycocotools']}
            or set(params) != set(expected) | {'iou_thresholds','recall_thresholds'}
            or any(params[key] != value for key,value in expected.items())):
        raise ValueError('Dataset evaluation protocol mismatch')
    for key, count, start, step in [('iou_thresholds',10,.5,.05), ('recall_thresholds',101,0,.01)]:
        values = params[key]
        if (not isinstance(values,list) or len(values)!=count
                or any(type(v) not in (float,int) or not math.isclose(v,start+i*step,rel_tol=0,abs_tol=1e-14)
                       for i,v in enumerate(values))):
            raise ValueError('Dataset evaluation thresholds mismatch')


def validate_identity(manifest, model):
    """Reject inconsistent model, evaluation and embedded artifact identities.

    No raw dataset is needed for serving. The immutable registered binding and
    artifact metadata remain authoritative after deployment/recovery.
    """
    data = manifest['dataset']
    expected = {key: data[key] for key in ('dataset_version_id', 'train_sha256', 'evaluation_sha256', 'evaluation_split')}
    expected['dataset_sha256'] = data['sha256']
    names = manifest['class_names']
    evaluation = manifest['evaluation']
    validate_evaluation_protocol(evaluation, manifest['dependencies'])
    mapping = [{'category_id': i, 'coco_category_id': i+1, 'name': name} for i, name in enumerate(names)]
    if (model['class_names'] != names or data['class_names'] != names
            or data['classes'] != [{'id': i, 'name': name} for i, name in enumerate(names)]
            or model['dataset_lineage'] != expected
            or manifest['preprocessing'] != recipe.metadata(names, data)
            or evaluation['artifact_sha256'] != manifest['artifact_sha256']
            or evaluation['dataset_version_id'] != data['dataset_version_id']
            or evaluation['dataset_content_sha256'] != data['sha256']
            or evaluation['evaluation_sha256'] != data['evaluation_sha256']
            or evaluation['dataset_sha256'] != data['evaluation_sha256']
            or evaluation['split'] != data['evaluation_split']
            or evaluation['role'] != ('held-out-test' if data['evaluation_split'] == 'test' else 'validation-only')
            or evaluation['category_mapping'] != mapping
            or evaluation['score_threshold'] != detection.EVALUATION_SCORE_THRESHOLD
            or evaluation['nms_iou_threshold'] != detection.NMS_IOU_THRESHOLD
            or evaluation['max_detections'] != {'upstream':100, 'scope':'global-per-image-class-aware-nms'}
            or evaluation['preprocessing'] != {'version':protocol()['preprocessing'], 'input_size':160,
                'orientation':'raw-decoded-no-exif-transpose', 'epsilon_box_clamp':1e-9}
            or evaluation['compatibility_key'] != digest(recipe.compatibility(evaluation))):
        raise ValueError('Dataset serving identity mismatch')
