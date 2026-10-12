"""COCO-style bbox AP on an explicitly supplied, non-crowd image subset.

This uses the official PyPI pycocotools COCOeval implementation, not a local
approximation of AP. It does not implement the Open Images evaluation protocol
or claim a full COCO benchmark. Inputs are original-image continuous xyxy pixel
coordinates (no inclusive-pixel +1), with contiguous zero-based category IDs.
The optional numerical dependencies are imported only when evaluation runs.
"""
from collections import Counter
from contextlib import redirect_stdout
from importlib.metadata import version
from io import StringIO
import math


PROTOCOL = "coco-style-bbox-subset/v1"
_AREA_MAX = 10000000000.0


def _record(value, required, optional, label):
    if not isinstance(value, dict) or not required <= value.keys() or value.keys() - required - optional:
        raise ValueError(f"{label} has missing or unsupported fields")


def _identifier(value, label):
    if not ((type(value) is int and value >= 0) or
            (type(value) is str and value.strip() and len(value) <= 256)):
        raise ValueError(f"{label} must be a nonnegative integer or nonempty string")
    return value


def _number(value, label):
    if type(value) not in (int, float):
        raise ValueError(f"{label} must be a finite number")
    try:
        number = float(value)
    except (OverflowError, ValueError):
        raise ValueError(f"{label} must be a finite number") from None
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number")
    return number


def _bbox(value, image, label):
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"{label} must contain four xyxy coordinates")
    x1, y1, x2, y2 = [_number(item, label) for item in value]
    if not (0 <= x1 < x2 <= image["width"] and 0 <= y1 < y2 <= image["height"]):
        raise ValueError(f"{label} must be nondegenerate and inside the original image")
    width, height = x2 - x1, y2 - y1
    area = width * height
    # COCO's standard all-area range has this finite upper bound. Reject rather
    # than silently ignore boxes outside that range or a floating-point underflow.
    if not 0 < area <= _AREA_MAX:
        raise ValueError(f"{label} area is outside the supported COCO all-area range")
    return [x1, y1, width, height], area


def _prepare(images, annotations, predictions, categories):
    """Validate before importing optional libraries, and copy every record."""
    for label, values in (("images", images), ("annotations", annotations),
                          ("predictions", predictions), ("categories", categories)):
        if not isinstance(values, (list, tuple)):
            raise ValueError(f"{label} must be a list or tuple")
    if not categories:
        raise ValueError("categories must not be empty")

    image_lookup, coco_images = {}, []
    for index, image in enumerate(images):
        _record(image, {"id", "width", "height"}, set(), "image")
        image_id = _identifier(image["id"], "image id")
        if image_id in image_lookup:
            raise ValueError("duplicate image id")
        for dimension in ("width", "height"):
            if type(image[dimension]) is not int or not 0 < image[dimension] <= 2147483647:
                raise ValueError(f"image {dimension} must be a positive 32-bit integer")
        converted = {"id": index + 1, "width": image["width"], "height": image["height"]}
        image_lookup[image_id] = converted
        coco_images.append(converted)

    category_lookup = {}
    for category in categories:
        _record(category, {"id", "name"}, set(), "category")
        category_id = category["id"]
        if type(category_id) is not int or category_id < 0:
            raise ValueError("category id must be a nonnegative integer")
        if category_id in category_lookup:
            raise ValueError("duplicate category id")
        if not isinstance(category["name"], str) or not category["name"].strip():
            raise ValueError("category name must be a nonempty string")
        category_lookup[category_id] = {"id": category_id + 1, "name": category["name"]}
    if sorted(category_lookup) != list(range(len(categories))):
        raise ValueError("category ids must be contiguous and zero-based")
    coco_categories = [category_lookup[key] for key in sorted(category_lookup)]

    converted_sets = []
    for label, rows in (("annotation", annotations), ("prediction", predictions)):
        converted, external_ids, ground_truth_keys = [], set(), set()
        for index, row in enumerate(rows):
            required = {"image_id", "category_id", "bbox"}
            optional = {"id", "iscrowd", "ignore", "group_of"}
            if label == "prediction":
                required = required | {"score"}
                optional = {"id"}
            _record(row, required, optional, label)
            if "id" in row:
                external_id = _identifier(row["id"], f"{label} id")
                if external_id in external_ids:
                    raise ValueError(f"duplicate {label} id")
                external_ids.add(external_id)
            image_id = _identifier(row["image_id"], f"{label} image_id")
            if image_id not in image_lookup:
                raise ValueError(f"unknown {label} image_id")
            category_id = row["category_id"]
            if type(category_id) is not int or category_id not in category_lookup:
                raise ValueError(f"unknown {label} category_id")
            image = image_lookup[image_id]
            box, area = _bbox(row["bbox"], image, f"{label} bbox")
            record = {"id": index + 1, "image_id": image["id"],
                      "category_id": category_id + 1, "bbox": box,
                      "area": area, "iscrowd": 0}
            if label == "annotation":
                for flag in ("iscrowd", "ignore", "group_of"):
                    # Neither unsupported ignored instances nor malformed flags
                    # may be silently converted into ordinary positive boxes.
                    if flag in row and not (type(row[flag]) in (bool, int) and row[flag] == 0):
                        raise ValueError(f"{flag} annotations are unsupported; select a non-crowd, non-ignored subset")
                key = (image["id"], category_id, *box)
                if key in ground_truth_keys:
                    raise ValueError("duplicate ground-truth annotation")
                ground_truth_keys.add(key)
            else:
                score = _number(row["score"], "prediction score")
                if not 0 <= score <= 1:
                    raise ValueError("prediction score must lie in [0, 1]")
                record["score"] = score
                # Duplicate detections must reach COCOeval: they are potential
                # false positives, never deduplicated to inflate measured AP.
            converted.append(record)
        converted_sets.append(converted)
    return coco_images, converted_sets[0], converted_sets[1], coco_categories


def evaluate_detections(images, annotations, predictions, categories, *, include_per_class=True):
    """Return measured AP plus the complete evaluation settings and counts.

    images: [{id: str|int, width: int, height: int}], including negative images.
    categories: [{id: int, name: str}], IDs exactly 0..K-1, in any order.
    annotations: [{image_id, category_id, bbox: [x1,y1,x2,y2]}]. Optional unique
        id is accepted; optional iscrowd, ignore and group_of must be false/0.
    predictions: [{image_id, category_id, bbox: [x1,y1,x2,y2], score: float}].
        Optional unique id is accepted. Repeated detection boxes are preserved.

    Image IDs are mapped to positive COCO IDs in supplied image order; platform
    categories are mapped to COCO category_id + 1. Box coordinates must already
    be restored to original image pixels, with 0 <= x <= width, 0 <= y <= height.
    No clipping, rescaling, score filtering or NMS is performed here. Equal-score
    ties follow stable COCO ordering: image input order, then prediction order.

    AP uses 101 recall points, ten IoU thresholds 0.50..0.95, area=all and
    maxDets=100 per image/category. Categories without GT are excluded from the
    macro average and have None AP, as does an entirely GT-free evaluation.
    Empty detections with present GT yield measured zero AP. Invalid data raises
    ValueError; missing optional dependencies raises RuntimeError.
    """
    if type(include_per_class) is not bool:
        raise ValueError("include_per_class must be boolean")
    coco_images, ground_truth, detections, coco_categories = _prepare(
        images, annotations, predictions, categories)
    try:
        import numpy as np
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as exc:
        raise RuntimeError("Detection evaluation requires optional requirements-detection.lock dependencies") from exc

    with redirect_stdout(StringIO()):
        gt = COCO()
        gt.dataset = {"images": coco_images, "annotations": ground_truth,
                      "categories": coco_categories, "info": {}}
        gt.createIndex()
        dt = COCO()
        # Constructing both indexes also handles empty predictions: loadRes([])
        # in pycocotools indexes its first result and cannot handle that case.
        dt.dataset = {"images": coco_images, "annotations": detections,
                      "categories": coco_categories, "info": {}}
        dt.createIndex()
        evaluator = COCOeval(gt, dt, iouType="bbox")
        parameters = evaluator.params
        parameters.imgIds = [image["id"] for image in coco_images]
        parameters.catIds = [category["id"] for category in coco_categories]
        parameters.iouThrs = np.linspace(0.5, 0.95, 10, endpoint=True)
        parameters.recThrs = np.linspace(0, 1, 101, endpoint=True)
        parameters.areaRng = [[0.0, _AREA_MAX]]
        parameters.areaRngLbl = ["all"]
        parameters.maxDets = [1, 10, 100]
        parameters.useCats = 1
        evaluator.evaluate()
        evaluator.accumulate()

    # COCOeval's accumulated tensor is [IoU, recall, category, area, maxDets].
    # The standard summary uses the mean of entries > -1; use the same reduction
    # directly, as summarize() also prints irrelevant small/medium/large metrics.
    precision = evaluator.eval["precision"][:, :, :, 0, 2]

    def ap(values):
        valid = values[values > -1]
        return float(np.mean(valid)) if valid.size else None

    gt_counts = Counter(row["category_id"] for row in ground_truth)
    dt_counts = Counter(row["category_id"] for row in detections)
    per_image_category = Counter((row["image_id"], row["category_id"]) for row in detections)
    gt_images = {row["image_id"] for row in ground_truth}
    dt_images = {row["image_id"] for row in detections}
    result = {
        "protocol": PROTOCOL,
        "status": "evaluated" if ground_truth else "no_ground_truth",
        "map_50": ap(precision[0]),
        "map_50_95": ap(precision),
        "evaluator": {"implementation": "pycocotools.cocoeval.COCOeval",
                      "version": version("pycocotools")},
        "parameters": {
            "iou_type": "bbox", "iou_thresholds": parameters.iouThrs.tolist(),
            "recall_thresholds": parameters.recThrs.tolist(),
            "area_ranges": [[0.0, _AREA_MAX]], "area_range_labels": ["all"],
            "max_detections": [1, 10, 100], "reported_max_detections": 100,
            "max_detections_scope": "per_image_per_category", "use_categories": True,
            "box_format": "xyxy_original_image_pixels_continuous",
            "category_averaging": "macro_over_categories_with_ground_truth",
            "no_ground_truth_ap": None,
            "crowd_ignore_group_policy": "rejected",
            "score_filtering": "none", "nms": "none",
            "score_tie_order": "stable_image_input_order_then_prediction_input_order",
        },
        "counts": {
            "images": len(coco_images), "images_with_ground_truth": len(gt_images),
            "images_without_ground_truth": len(coco_images) - len(gt_images),
            "images_with_predictions": len(dt_images),
            "ground_truth_boxes": len(ground_truth), "predictions": len(detections),
            "predictions_above_max_detections": sum(max(0, count - 100) for count in per_image_category.values()),
            "categories": len(coco_categories), "categories_with_ground_truth": len(gt_counts),
        },
        "category_mapping": [
            {"category_id": category["id"] - 1, "coco_category_id": category["id"], "name": category["name"]}
            for category in coco_categories
        ],
        "limitations": (
            "COCO-style bbox AP on the supplied non-crowd subset only; not the full official "
            "COCO or Open Images benchmark, external audit, or device validation. Categories "
            "without ground truth are excluded from AP, including their predictions. "
            "Dataset split/label completeness and upstream score/NMS settings must be recorded separately."
        ),
    }
    if include_per_class:
        result["per_class"] = [
            {"category_id": category["id"] - 1, "name": category["name"],
             "ground_truth_boxes": gt_counts[category["id"]], "predictions": dt_counts[category["id"]],
             "status": "evaluated" if gt_counts[category["id"]] else "no_ground_truth",
             "ap_50": ap(precision[0, :, index]), "ap_50_95": ap(precision[:, :, index])}
            for index, category in enumerate(coco_categories)
        ]
    return result
