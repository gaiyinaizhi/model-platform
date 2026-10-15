"""Standard-library-only image validation and decoded YOLOX box geometry.

Images use inline, flattened RGB HWC bytes. The geometry contract is a top-left
letterbox: resize by the minimum target/source scale, floor both resized sides,
then pad only the right and bottom with 114. Pixel interpolation, RGB-to-BGR
conversion and float32 CHW packing belong to the optional inference backend.

Boxes use continuous, half-open ``[x1, y1, x2, y2]`` image coordinates. In
particular, areas never add an inclusive-pixel ``+1``. Model outputs must already
be decoded to padded-input ``cx, cy, width, height, objectness, class scores``;
this module does not decode YOLOX grid/stride offsets or apply a sigmoid.
"""

from dataclasses import dataclass
import math

from .common import Problem, object_fields


PAD_VALUE = 114
MAX_DETECTIONS = 100


def _positive_integer(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite(value, name):
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be a finite number")
    try:
        value = float(value)
    except (OverflowError, ValueError):
        raise ValueError(f"{name} must be a finite number") from None
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return value


def _probability(value, name):
    value = _finite(value, name)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in 0..1")
    return value


def validate_images(images, *, max_images=1, max_side=512, min_side=1):
    """Validate bounded inline RGB images without importing an imaging library.

    Each image is exactly ``{width, height, pixels}``; dimensions and every
    flattened row-major interleaved R/G/B pixel channel must be integers (not
    booleans), with channels in 0..255. URLs, paths, extra fields, empty batches
    and inconsistent byte counts are rejected with HTTP-400 ``Problem``.
    The caller owns target-specific checks for floor-resize-to-zero cases.
    Invalid configured bounds raise ``ValueError`` rather than a client error.
    """
    _positive_integer(max_images, "max_images")
    _positive_integer(max_side, "max_side")
    _positive_integer(min_side, "min_side")
    if min_side > max_side:
        raise ValueError("min_side must not exceed max_side")
    if not isinstance(images, list) or not 1 <= len(images) <= max_images:
        raise Problem(400, "invalid_input", f"Detection requires 1..{max_images} inline RGB images")
    for image in images:
        object_fields(image, ("width", "height", "pixels"))
        width, height = image["width"], image["height"]
        if any(type(side) is not int or not min_side <= side <= max_side
               for side in (width, height)):
            raise Problem(400, "invalid_input", f"Image dimensions must be integers in {min_side}..{max_side}")
        pixels = image["pixels"]
        if (not isinstance(pixels, list) or len(pixels) != width * height * 3
                or any(type(value) is not int or not 0 <= value <= 255 for value in pixels)):
            raise Problem(400, "invalid_input", "Expected width*height*3 RGB integer channels in 0..255")


def _resized_dimensions(width, height, target_width, target_height):
    # Integer ratios implement exact mathematical floor, avoiding an accidental
    # one-pixel short limiting side from floating-point multiplication.
    if target_width * height <= target_height * width:
        return target_width, height * target_width // width
    return width * target_height // height, target_height


@dataclass(frozen=True)
class LetterboxTransform:
    """Realized top-left resize and padding, including separate x/y scales.

    Use ``letterbox_transform`` to construct this record. The separate scales
    account for the integer flooring of each side; a single ideal scale loses
    subpixel accuracy when undoing a non-integral resize.
    """

    original_width: int
    original_height: int
    target_width: int
    target_height: int
    resized_width: int
    resized_height: int

    def __post_init__(self):
        for name in ("original_width", "original_height", "target_width", "target_height",
                     "resized_width", "resized_height"):
            _positive_integer(getattr(self, name), name)
        expected = _resized_dimensions(self.original_width, self.original_height,
                                       self.target_width, self.target_height)
        if (self.resized_width, self.resized_height) != expected:
            raise ValueError("Resized dimensions do not match floor letterbox geometry")

    @property
    def scale_x(self):
        """Realized horizontal scale: integer resized width / original width."""
        return self.resized_width / self.original_width

    @property
    def scale_y(self):
        """Realized vertical scale: integer resized height / original height."""
        return self.resized_height / self.original_height

    @property
    def pad_left(self):
        return 0

    @property
    def pad_top(self):
        return 0

    @property
    def pad_right(self):
        return self.target_width - self.resized_width

    @property
    def pad_bottom(self):
        return self.target_height - self.resized_height

    @property
    def pad_value(self):
        return PAD_VALUE


def letterbox_transform(width, height, *, target_width=416, target_height=416):
    """Return exact integer top-left letterbox geometry, without changing pixels.

    Upscaling is allowed. A source aspect ratio that floors either resized side
    to zero is rejected with ``ValueError``; it is never silently stretched.
    """
    for name, value in (("width", width), ("height", height), ("target_width", target_width),
                        ("target_height", target_height)):
        _positive_integer(value, name)
    resized_width, resized_height = _resized_dimensions(width, height, target_width, target_height)
    if not resized_width or not resized_height:
        raise ValueError("Image aspect ratio produces an empty resized side")
    return LetterboxTransform(width, height, target_width, target_height, resized_width, resized_height)


def _box(value):
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("A box must contain exactly four finite coordinates")
    return [_finite(coordinate, "Box coordinate") for coordinate in value]


def restore_box(cxcywh, transform):
    """Map a decoded padded-input box to clipped original-image half-open xyxy.

    Nonpositive model widths/heights and boxes wholly outside the image return
    ``None``. Malformed/non-finite model values raise ``ValueError``. Clipping
    uses image width/height, not their last inclusive pixel indices.
    """
    if not isinstance(transform, LetterboxTransform):
        raise ValueError("Expected a LetterboxTransform")
    cx, cy, width, height = _box(cxcywh)
    if width <= 0.0 or height <= 0.0:
        return None
    box = [(cx - width / 2.0) / transform.scale_x,
           (cy - height / 2.0) / transform.scale_y,
           (cx + width / 2.0) / transform.scale_x,
           (cy + height / 2.0) / transform.scale_y]
    box = [_finite(value, "Restored box coordinate") for value in box]
    box[0] = min(float(transform.original_width), max(0.0, box[0]))
    box[2] = min(float(transform.original_width), max(0.0, box[2]))
    box[1] = min(float(transform.original_height), max(0.0, box[1]))
    box[3] = min(float(transform.original_height), max(0.0, box[3]))
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    return box


def box_iou(first, second):
    """Return continuous xyxy intersection-over-union with no inclusive +1.

    Zero/negative-area boxes have IoU zero. Malformed, non-finite or numerically
    overflowing coordinates/areas raise ``ValueError``.
    """
    first, second = _box(first), _box(second)
    first_area = _finite(max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1]), "Box area")
    second_area = _finite(max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1]), "Box area")
    if not first_area or not second_area:
        return 0.0
    intersection = _finite(max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
                           * max(0.0, min(first[3], second[3]) - max(first[1], second[1])), "Intersection area")
    union = _finite(first_area + second_area - intersection, "Union area")
    return intersection / union


def class_aware_nms(detections, *, iou_threshold, max_detections=MAX_DETECTIONS):
    """Keep at most 100 detections, suppressing overlap only within each class.

    Each detection is exactly ``{class_id, score, box}``. Higher scores win;
    equal scores preserve input order. Suppression uses strict IoU > threshold.
    Degenerate boxes are discarded, but all entries are validated before a
    result is returned. The output contains fresh dictionaries and box lists.
    """
    iou_threshold = _probability(iou_threshold, "iou_threshold")
    if type(max_detections) is not int or not 1 <= max_detections <= MAX_DETECTIONS:
        raise ValueError(f"max_detections must be an integer in 1..{MAX_DETECTIONS}")
    if not isinstance(detections, (list, tuple)):
        raise ValueError("Detections must be a sequence")
    candidates = []
    for detection in detections:
        if not isinstance(detection, dict) or detection.keys() != {"class_id", "score", "box"}:
            raise ValueError("Invalid detection fields")
        class_id = detection["class_id"]
        if type(class_id) is not int or class_id < 0:
            raise ValueError("class_id must be a nonnegative integer")
        score = _probability(detection["score"], "Detection score")
        box = _box(detection["box"])
        if box[2] > box[0] and box[3] > box[1]:
            # Validate area even when this candidate would never be compared.
            box_iou(box, box)
            candidates.append({"class_id": class_id, "score": score, "box": box})
    candidates.sort(key=lambda detection: -detection["score"])
    kept = []
    for candidate in candidates:
        if any(candidate["class_id"] == previous["class_id"]
               and box_iou(candidate["box"], previous["box"]) > iou_threshold for previous in kept):
            continue
        kept.append(candidate)
        if len(kept) == max_detections:
            break
    return kept


def postprocess_predictions(rows, transform, *, num_classes=80, score_threshold,
                            iou_threshold, max_detections=MAX_DETECTIONS):
    """Validate decoded YOLOX rows, restore boxes, and perform class-aware NMS.

    A row is ``[cx, cy, width, height, objectness, class_0, ...]``. Objectness
    and class confidences must be finite in 0..1. Only the largest class score
    is used; a class-score tie chooses the smallest zero-based class ID. The
    returned score is objectness * class confidence, included when it is >=
    the supplied score threshold. All rows, including low-scoring rows, are
    validated; no invalid output is hidden by filtering or the output cap.
    """
    _positive_integer(num_classes, "num_classes")
    score_threshold = _probability(score_threshold, "score_threshold")
    if not isinstance(transform, LetterboxTransform):
        raise ValueError("Expected a LetterboxTransform")
    if not isinstance(rows, (list, tuple)):
        raise ValueError("Prediction rows must be a sequence")
    detections = []
    for row in rows:
        if not isinstance(row, (list, tuple)) or len(row) != 5 + num_classes:
            raise ValueError("Prediction rows do not match the decoded class contract")
        coordinates = _box(row[:4])
        objectness = _probability(row[4], "Objectness")
        probabilities = [_probability(value, "Class confidence") for value in row[5:]]
        class_id = max(range(num_classes), key=probabilities.__getitem__)
        score = objectness * probabilities[class_id]
        box = restore_box(coordinates, transform)
        if score >= score_threshold and box is not None:
            detections.append({"class_id": class_id, "score": score, "box": box})
    return class_aware_nms(detections, iou_threshold=iou_threshold, max_detections=max_detections)
