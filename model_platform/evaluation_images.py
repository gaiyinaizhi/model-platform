"""Authenticated read-only previews of already-recorded diagnostic images."""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import struct
import subprocess
import sys
import threading
import time
import zlib

from .common import Problem
from .dataset_validation import safe_path, MAX_FILE_BYTES, MAX_PIXELS as DATASET_PIXELS
from .evaluation_comparison_validation import validate
from .model_evaluations import ID, MAX_CHILD_BYTES, EXAMPLE_LIMIT, EXAMPLE_BOX_LIMIT
from .safe_files import read_file

MAX_SOURCE_BYTES = min(8 * 1024 * 1024, MAX_FILE_BYTES)
MAX_PIXELS = min(16_000_000, DATASET_PIXELS)
MAX_OUTPUT_BYTES = 2 * 1024 * 1024
MAX_EDGE = 1024
WALL_SECONDS = 5
_SLOTS = threading.BoundedSemaphore(2)
_INDEX = re.compile(r'(?:[0-9]|1[0-9])\Z')
_SHA = re.compile(r'[a-f0-9]{64}\Z')
_DATASET_ID = re.compile(r'dsv_[a-f0-9]{32}\Z')
_MESSAGE = 'Evaluation image or evidence is unavailable or has changed'


def _check(value):
    if not value:
        raise ValueError('Invalid preview evidence')


def _integer(value, maximum, minimum=0):
    _check(type(value) is int and minimum <= value <= maximum)
    return value


def _sha(value):
    _check(type(value) is str and _SHA.fullmatch(value))


def _dimensions(value):
    w, h = (_integer(value[key], MAX_PIXELS, 1) for key in ('width', 'height'))
    _check(w * h <= MAX_PIXELS)
    return w, h


def _box(value, width, height):
    _check(type(value) is list and len(value) == 4 and all(
        type(n) in (int, float) and math.isfinite(n) for n in value))
    _check(0 <= value[0] < value[2] <= width and 0 <= value[1] < value[3] <= height)


def _selected(detail, index, evidence):
    """Bind the diagnostic ordinal to exactly one immutable sample/evidence row."""
    binding, result = detail['binding'], detail['result']
    dataset, examples = binding['dataset'], result['wrong_examples']
    _check(type(dataset['dataset_version_id']) is str and _DATASET_ID.fullmatch(dataset['dataset_version_id']))
    _check(type(examples) is dict and examples['limit'] == EXAMPLE_LIMIT)
    items = examples['items']
    total = _integer(examples['total'], dataset['images'])
    _check(type(items) is list and len(items) == min(EXAMPLE_LIMIT, total)
           and type(examples['truncated']) is bool and examples['truncated'] == (total > len(items)))
    samples = dataset['samples']
    _check(type(evidence) is list and len(evidence) == len(samples))
    ids, sample_by_id, prediction_offsets = [], {}, {}
    offset = 0
    for sample, recorded in zip(samples, evidence):
        name = sample['image']
        _check(type(name) is str and safe_path(name) and name.startswith('images/val/')
               and not name.endswith('/') and name not in sample_by_id)
        _check(type(sample['label']) is str and safe_path(sample['label'])
               and sample['label'].startswith('labels/val/') and not sample['label'].endswith('/'))
        # Only the selected image has the tighter preview pixel budget. Other
        # valid historical images must not prevent its preview.
        _check(type(sample['width']) is int and type(sample['height']) is int
               and 0 < sample['width'] * sample['height'] <= DATASET_PIXELS)
        for key in ('image_sha256', 'label_sha256', 'pixel_sha256'):
            _sha(sample[key])
        _check(recorded['image_id'] == name and recorded['width'] == sample['width']
               and recorded['height'] == sample['height']
               and recorded['original_rgb_sha256'] == sample['pixel_sha256'])
        _sha(recorded['request_rgb_sha256'])
        _integer(recorded['request_width'], 384, 1)
        _integer(recorded['request_height'], 384, 1)
        _check(type(recorded['detections']) is list and len(recorded['detections']) <= 100)
        prediction_offsets[name] = (offset, recorded['detections'])
        offset += len(recorded['detections'])
        ids.append(name)
        sample_by_id[name] = sample
    seen, last = set(), -1
    for item in items:
        name = item['image_id']
        _check(type(name) is str and name in sample_by_id and name not in seen)
        position = ids.index(name)
        _check(position > last)
        seen.add(name)
        last = position
        sample = sample_by_id[name]
        _check(item['width'] == sample['width'] and item['height'] == sample['height'])
        tp, fp, fn = (_integer(item[key], maximum) for key, maximum in
                      (('tp', 100), ('fp', 100), ('fn', 10000)))
        _check(fp + fn > 0 and tp + fp <= 100)
        for key, count, index_name in (('false_positives', fp, 'prediction_index'),
                                       ('false_negatives', fn, 'annotation_index')):
            entries = item[key]
            _check(type(entries) is list and len(entries) == min(EXAMPLE_BOX_LIMIT, count)
                   and type(item[key + '_truncated']) is bool
                   and item[key + '_truncated'] == (count > len(entries)))
            indices = set()
            for entry in entries:
                row_index = _integer(entry[index_name], 6399 if key == 'false_positives' else 9999)
                _check(row_index not in indices)
                indices.add(row_index)
                _integer(entry['category_id'], 79)
                _box(entry['bbox'], sample['width'], sample['height'])
                if key == 'false_positives':
                    _check(type(entry['score']) in (int, float) and .3 < entry['score'] <= 1)
                    start, detections = prediction_offsets[name]
                    _check(0 <= row_index - start < len(detections))
                    recorded = detections[row_index - start]
                    _check(all(entry[k] == recorded[k] for k in ('category_id', 'score', 'bbox')))
    if index >= len(items):
        raise Problem(404, 'not_found', 'Evaluation error image not found')
    sample = sample_by_id[items[index]['image_id']]
    _dimensions(sample)
    return sample


def validate_png(raw, expected):
    """Require a complete metadata-free RGB PNG before sending success headers."""
    _check(type(raw) is bytes and 0 < len(raw) <= MAX_OUTPUT_BYTES and raw[:8] == b'\x89PNG\r\n\x1a\n')
    offset, kinds = 8, []
    while offset < len(raw):
        _check(offset + 12 <= len(raw))
        length = struct.unpack('>I', raw[offset:offset + 4])[0]
        kind = raw[offset + 4:offset + 8]
        end = offset + 12 + length
        _check(end <= len(raw) and kind in (b'IHDR', b'IDAT', b'IEND'))
        payload = raw[offset + 8:end - 4]
        _check(zlib.crc32(kind + payload) & 0xffffffff == struct.unpack('>I', raw[end - 4:end])[0])
        if kind == b'IHDR':
            _check(not kinds and length == 13)
            width, height, depth, color, compression, filtering, interlace = struct.unpack('>IIBBBBB', payload)
            _check((width, height) == expected and depth == 8 and color == 2
                   and compression == filtering == interlace == 0)
        elif kind == b'IDAT':
            _check(kinds and kinds[-1] in (b'IHDR', b'IDAT') and length > 0)
        else:
            _check(length == 0 and kinds and kinds[-1] == b'IDAT' and end == len(raw))
        kinds.append(kind)
        offset = end
    _check(kinds and kinds[-1] == b'IEND')


def _decode(raw, sample):
    width, height = _dimensions(sample)
    _check(type(raw) is bytes and 0 < len(raw) <= MAX_SOURCE_BYTES)
    process, timer = None, None
    timed_out = threading.Event()
    deadline = time.monotonic() + WALL_SECONDS
    try:
        # Isolated Python ignores user environment/import paths and cannot write
        # bytecode. The child receives only bounded bytes and identity facts.
        process = subprocess.Popen([sys.executable, '-I', '-B',
            str(Path(__file__).with_name('evaluation_image_decode.py')),
            str(width), str(height), sample['pixel_sha256'], str(os.getpid())], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, close_fds=True)
        def expire():
            timed_out.set()
            try:
                process.kill()
            except ProcessLookupError:
                pass
        timer = threading.Timer(max(0, deadline - time.monotonic()), expire)
        timer.daemon = True
        timer.start()
        try:
            process.stdin.write(raw)
            process.stdin.close()
            # Independently bound the parent read, even if the child violates
            # its own output contract. The timer bounds stdin and stdout stalls.
            output = process.stdout.read(MAX_OUTPUT_BYTES + 1)
            if len(output) > MAX_OUTPUT_BYTES:
                process.kill()
            process.wait()
        except BrokenPipeError:
            process.wait()
            output = b''
        if timed_out.is_set() or time.monotonic() > deadline or process.returncode in (-signal.SIGALRM, -signal.SIGXCPU):
            raise Problem(503, 'preview_timeout', 'Image preview deadline exceeded')
        if process.returncode == 3:
            raise Problem(503, 'preview_unavailable', 'Image preview renderer is unavailable')
        if process.returncode != 0:
            raise Problem(409, 'evaluation_integrity', _MESSAGE)
        scale = min(1.0, MAX_EDGE / max(width, height))
        validate_png(output, (max(1, int(width * scale)), max(1, int(height * scale))))
        return output
    except OSError:
        raise Problem(503, 'preview_unavailable', 'Image preview renderer is unavailable') from None
    finally:
        if timer is not None:
            timer.cancel()
        if process is not None:
            if process.poll() is None:
                process.kill()
            process.wait()
            for stream in (process.stdin, process.stdout):
                if stream is not None:
                    try:
                        stream.close()
                    except BrokenPipeError:
                        pass


def preview(evaluations, tenant, item_id, index):
    if not isinstance(item_id, str) or not ID.fullmatch(item_id) or not isinstance(index, str) or not _INDEX.fullmatch(index):
        raise Problem(400, 'invalid_input', 'Expected an evaluation ID and canonical error index 0..19')
    row = evaluations._row(tenant, item_id)
    from .model_evaluations import require_imported_capability
    require_imported_capability(row, 'preview')
    # Ownership lookup must precede state, integrity and filesystem inspection.
    if row['status'] != 'succeeded':
        raise Problem(409, 'evaluation_not_succeeded', 'Evaluation must have succeeded')
    if not _SLOTS.acquire(blocking=False):
        raise Problem(429, 'preview_busy', 'Image preview budget is occupied; retry later')
    try:
        # Validate the persisted storage identity before full binding checks
        # construct its snapshot path, even if an operator corrupted the DB.
        _check(type(row['dataset_version_id']) is str and _DATASET_ID.fullmatch(row['dataset_version_id']))
        detail = evaluations.detail(row)
        validate(detail)
        evidence = read_file(evaluations.evidence_path(row['id']), MAX_CHILD_BYTES)
        _check(hashlib.sha256(evidence).hexdigest() == row['terminal_prediction_sha256'])
        sample = _selected(detail, int(index), json.loads(evidence))
        root = evaluations.runtime.home / 'datasets' / 'versions' / detail['dataset_version_id']
        raw = read_file(root / sample['image'], MAX_SOURCE_BYTES)
        _check(hashlib.sha256(raw).hexdigest() == sample['image_sha256'])
        return _decode(raw, sample)
    except Problem as exc:
        if exc.code in ('not_found', 'preview_timeout', 'preview_unavailable'):
            raise
        raise Problem(409, 'evaluation_integrity', _MESSAGE) from None
    except (ValueError, TypeError, KeyError, AttributeError, OSError, OverflowError, RecursionError):
        raise Problem(409, 'evaluation_integrity', _MESSAGE) from None
    finally:
        _SLOTS.release()
