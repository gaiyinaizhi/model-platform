"""Independent, immutable imported-model measurements; never a training run."""
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys
import threading
import time
import uuid

from .common import Problem, digest, encode, environment, now, object_fields, text_value

PROTOCOL = 'imported-coco-serving-evaluation/v1'
MAX_IMAGES = 64
MAX_BOXES = 10000
MAX_RECORDS = 256
MAX_EVIDENCE_BYTES = 64 * 1024 * 1024
MAX_CHILD_BYTES = 8 * 1024 * 1024
MAX_DETAIL_BYTES = 512 * 1024
MAX_LIST_BYTES = 256 * 1024
WALL_SECONDS = 300
CPU_SECONDS = 240
ADDRESS_BYTES = 2 * 1024 ** 3
EXAMPLE_LIMIT = 20
EXAMPLE_BOX_LIMIT = 20
ID = re.compile(r'^mev_[0-9a-f]{32}$')

SCHEMA = """
CREATE TABLE IF NOT EXISTS model_evaluations (
 id TEXT PRIMARY KEY, tenant TEXT NOT NULL, idempotency_key TEXT NOT NULL,
 request_sha256 TEXT NOT NULL, request_json TEXT NOT NULL,
 model_version_id TEXT NOT NULL REFERENCES models(id),
 dataset_version_id TEXT NOT NULL REFERENCES dataset_versions(id),
 binding_json TEXT NOT NULL, binding_sha256 TEXT NOT NULL,
 status TEXT NOT NULL CHECK(status IN ('queued','running','succeeded','failed')),
 created_at REAL NOT NULL, started_at REAL, finished_at REAL, error_code TEXT,
 UNIQUE(tenant,idempotency_key), UNIQUE(id,tenant),
 CHECK((status='queued' AND started_at IS NULL AND finished_at IS NULL AND error_code IS NULL)
    OR (status='running' AND started_at IS NOT NULL AND finished_at IS NULL AND error_code IS NULL)
    OR (status='succeeded' AND started_at IS NOT NULL AND finished_at IS NOT NULL AND error_code IS NULL)
    OR (status='failed' AND started_at IS NOT NULL AND finished_at IS NOT NULL AND error_code IS NOT NULL))
);
CREATE TABLE IF NOT EXISTS model_evaluation_results (
 evaluation_id TEXT PRIMARY KEY, tenant TEXT NOT NULL,
 result_json TEXT NOT NULL, result_sha256 TEXT NOT NULL, prediction_sha256 TEXT NOT NULL,
 FOREIGN KEY(evaluation_id,tenant) REFERENCES model_evaluations(id,tenant)
);
CREATE INDEX IF NOT EXISTS model_evaluations_history ON model_evaluations(tenant,created_at DESC,id DESC);
CREATE TRIGGER IF NOT EXISTS model_evaluation_identity_immutable BEFORE UPDATE OF
 id,tenant,idempotency_key,request_sha256,request_json,model_version_id,dataset_version_id,binding_json,binding_sha256,created_at
 ON model_evaluations BEGIN SELECT RAISE(ABORT,'evaluation identity is immutable'); END;
CREATE TRIGGER IF NOT EXISTS model_evaluation_transition BEFORE UPDATE OF status ON model_evaluations
 WHEN NOT ((OLD.status='queued' AND NEW.status='running') OR
           (OLD.status='running' AND NEW.status IN ('succeeded','failed')))
 BEGIN SELECT RAISE(ABORT,'invalid evaluation transition'); END;
CREATE TRIGGER IF NOT EXISTS model_evaluation_terminal_immutable BEFORE UPDATE ON model_evaluations
 WHEN OLD.status IN ('succeeded','failed') BEGIN SELECT RAISE(ABORT,'terminal evaluation is immutable'); END;
CREATE TRIGGER IF NOT EXISTS model_evaluation_delete_immutable BEFORE DELETE ON model_evaluations
 BEGIN SELECT RAISE(ABORT,'evaluations are immutable'); END;
CREATE TRIGGER IF NOT EXISTS model_evaluation_result_update_immutable BEFORE UPDATE ON model_evaluation_results
 BEGIN SELECT RAISE(ABORT,'evaluation results are immutable'); END;
CREATE TRIGGER IF NOT EXISTS model_evaluation_result_delete_immutable BEFORE DELETE ON model_evaluation_results
 BEGIN SELECT RAISE(ABORT,'evaluation results are immutable'); END;
CREATE TRIGGER IF NOT EXISTS model_evaluation_result_owner BEFORE INSERT ON model_evaluation_results
 WHEN NOT EXISTS (SELECT 1 FROM model_evaluations WHERE id=NEW.evaluation_id AND tenant=NEW.tenant AND status='running')
 BEGIN SELECT RAISE(ABORT,'evaluation result requires running owner'); END;
CREATE TRIGGER IF NOT EXISTS model_evaluation_success_result BEFORE UPDATE OF status ON model_evaluations
 WHEN NEW.status='succeeded' AND NOT EXISTS (SELECT 1 FROM model_evaluation_results WHERE evaluation_id=NEW.id AND tenant=NEW.tenant)
 BEGIN SELECT RAISE(ABORT,'success requires immutable result'); END;
"""


def protocol():
    return {'id': PROTOCOL, 'inference_protocol': 'yolox-nano-coco-onnx/v1',
            'max_input_side': 384, 'input_size': 416, 'score_threshold': .3, 'score_operator': '>',
            'nms_iou_threshold': .45, 'nms_class_agnostic': True, 'max_detections': 100,
            'resize': 'opencv-inter-linear-floor-max384-no-upscale/v1',
            'metric_protocol': 'coco-style-bbox-subset/v1', 'matching_protocol': 'class-aware-greedy-iou50/v1'}


def dependencies():
    # API checks installation without importing numerical libraries into HTTP.
    for name in ('numpy', 'onnxruntime', 'onnx', 'cv2', 'PIL', 'pycocotools'):
        try:
            found = importlib.util.find_spec(name)
        except (ValueError, ImportError):
            found = None
        if found is None:
            raise Problem(503, 'evaluation_dependencies_missing', 'Independent evaluation requires the optional detection and pretrained dependencies')


def installed_versions():
    import importlib.metadata
    try:
        return {name: importlib.metadata.version(distribution) for name, distribution in
                [('numpy', 'numpy'), ('onnxruntime', 'onnxruntime'), ('onnx', 'onnx'),
                 ('opencv', 'opencv-python-headless'), ('Pillow', 'Pillow'), ('pycocotools', 'pycocotools')]}
    except importlib.metadata.PackageNotFoundError:
        raise Problem(503, 'evaluation_dependencies_missing', 'Evaluation package metadata is unavailable') from None


def _env(value=None):
    if value is None:
        value = dict(environment(), dependencies=encode(installed_versions()).decode())
    return {key: value[key] for key in ('python', 'platform', 'code_sha256', 'dependencies')}



def request(body):
    object_fields(body, ('model_version_id', 'dataset_version_id', 'split', 'protocol'))
    for key in ('model_version_id', 'dataset_version_id'):
        text_value(body[key], key, 80)
    if body['split'] != 'val' or body['protocol'] != PROTOCOL:
        raise Problem(400, 'invalid_input', 'Choose the fixed imported evaluation protocol and val split')
    return body


def active_count(db):
    return (db.execute("SELECT count(*) FROM runs WHERE status IN ('queued','running')").fetchone()[0]
            + db.execute("SELECT count(*) FROM model_evaluations WHERE status IN ('queued','running')").fetchone()[0])


@contextlib.contextmanager
def _directory(path):
    parts = Path(os.path.abspath(path)).parts
    fd = os.open(parts[0], os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in parts[1:]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = child
        yield fd
    finally:
        os.close(fd)


def _read_file(path, maximum):
    """No symlinks at any path component; read bounded regular bytes once."""
    parts = Path(os.path.abspath(path)).parts
    directory = os.open(parts[0], os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in parts[1:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = child
        fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
    finally:
        os.close(directory)
    with os.fdopen(fd, 'rb') as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or not 0 <= before.st_size <= maximum:
            raise ValueError('Unsafe evidence file')
        raw = stream.read(maximum + 1)
        after = os.fstat(stream.fileno())
        if len(raw) != before.st_size or any(getattr(before, k) != getattr(after, k)
                for k in ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')):
            raise ValueError('Evidence changed during read')
    return raw


class ModelEvaluationStore:
    def __init__(self, store):
        self.store, self.runtime = store, store.runtime
        self.root = self.runtime.home / 'model-evaluation-evidence'
        if self.root.is_symlink():
            raise Problem(409, 'evaluation_integrity', 'Evaluation evidence storage is invalid')
        self.root.mkdir(mode=0o700, exist_ok=True)

    def binding_for(self, tenant, body, submitted_environment=None):
        from .datasets import DatasetStore
        from .model_imports import validate_manifest, verify_artifact
        from . import pretrained_yolox as nano
        with self.runtime.connection() as db:
            row = self.store._owned(db, 'models', tenant, body['model_version_id'])
            manifest = json.loads(row['manifest_json'])
            if manifest.get('origin') != 'imported' or manifest.get('protocol') != nano.PROTOCOL:
                raise Problem(409, 'evaluation_model_unsupported', 'Only the reviewed imported COCO model is supported')
            validate_manifest(db, row, manifest)
        if (manifest.get('artifact_sha256') != nano.ASSET_SHA256 or manifest.get('artifact_bytes') != nano.ASSET_BYTES
                or manifest.get('class_names') != list(nano.CLASS_NAMES)
                or manifest.get('category_ids') != list(nano.COCO_CATEGORY_IDS)
                or manifest.get('label_table_sha256') != nano.LABEL_TABLE_SHA256
                or manifest.get('evaluation') is not None
                or manifest.get('local_evaluation_status') != 'not_evaluated'):
            raise Problem(409, 'import_integrity', 'Imported model identity does not match the reviewed protocol')
        verify_artifact(self.runtime, manifest['artifact_id'], manifest['artifact_sha256'])
        dataset = DatasetStore(self.runtime).get_version(tenant, body['dataset_version_id'])
        dm = dataset['manifest']
        samples = dm['samples']
        if (dm.get('purpose', 'training') != 'evaluation'
                or [c['name'] for c in dm['classes']] != list(nano.CLASS_NAMES)
                or [c['id'] for c in dm['classes']] != list(range(80))
                or not 1 <= len(samples) <= MAX_IMAGES or any(s['split'] != 'val' for s in samples)
                or not 0 <= dm['counts']['boxes'] <= MAX_BOXES
                or any(not s.get('pixel_sha256') for s in samples)):
            raise Problem(409, 'evaluation_dataset_incompatible', 'Evaluation requires val-only, 1..64 images, at most10000 boxes and exact80 class ordering')
        categories = [{'class_id': i, 'name': name, 'source_category_id': cid,
                       'evaluator_internal_category_id': i + 1}
                      for i, (name, cid) in enumerate(zip(nano.CLASS_NAMES, nano.COCO_CATEGORY_IDS))]
        return {'model': {**{key: manifest[key] for key in ('model_version_id', 'origin', 'import_id',
                'validation_id', 'catalog_id', 'catalog_revision', 'artifact_sha256', 'artifact_bytes')},
                'manifest_sha256': digest(manifest)},
                'dataset': {'dataset_version_id': dataset['id'], 'content_sha256': dataset['content_sha256'],
                    'purpose': 'evaluation', 'split': 'val', 'images': len(samples), 'boxes': dm['counts']['boxes'],
                    'samples': [{key: s[key] for key in ('image', 'label', 'width', 'height', 'image_sha256',
                                                       'label_sha256', 'pixel_sha256')} for s in samples],
                    'source': dm.get('source')},
                'categories': categories, 'category_mapping_sha256': digest(categories),
                'protocol': protocol(), 'protocol_sha256': digest(protocol()),
                'submission_environment': _env(submitted_environment)}

    def create(self, tenant, key, body):
        request(body)
        key = text_value(key, 'Idempotency-Key', 128)
        # Existing keys replay even if dependencies later become unavailable.
        with self.runtime.connection() as db:
            old = db.execute('SELECT * FROM model_evaluations WHERE tenant=? AND idempotency_key=?', (tenant, key)).fetchone()
        if old:
            if old['request_sha256'] != digest(body):
                raise Problem(409, 'idempotency_conflict', 'Key is bound to another evaluation request')
            return self.get(tenant, old['id']), False
        dependencies()
        binding = self.binding_for(tenant, body)
        with self.runtime.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            old = db.execute('SELECT * FROM model_evaluations WHERE tenant=? AND idempotency_key=?', (tenant, key)).fetchone()
            if old:
                if old['request_sha256'] != digest(body):
                    raise Problem(409, 'idempotency_conflict', 'Key is bound to another evaluation request')
                item_id, created = old['id'], False
            else:
                if active_count(db) >= 32 or db.execute('SELECT count(*) FROM model_evaluations').fetchone()[0] >= MAX_RECORDS:
                    raise Problem(429, 'evaluation_capacity', 'Combined queue or retained evaluation record budget reached')
                if self.disk_bytes() >= MAX_EVIDENCE_BYTES:
                    raise Problem(429, 'evaluation_capacity', 'Evaluation evidence storage budget reached')
                item_id, created = 'mev_' + uuid.uuid4().hex, True
                db.execute('INSERT INTO model_evaluations VALUES(?,?,?,?,?,?,?,?,?,\'queued\',?,NULL,NULL,NULL)',
                    (item_id, tenant, key, digest(body), encode(body).decode(), body['model_version_id'],
                     body['dataset_version_id'], encode(binding).decode(), digest(binding), now()))
                self.store.audit(db, tenant, 'submit_model_evaluation', item_id)
        return self.get(tenant, item_id), created

    @staticmethod
    def read_query():
        return ("SELECT e.*,r.result_json AS terminal_result_json,"
                "r.result_sha256 AS terminal_result_sha256,r.prediction_sha256 AS terminal_prediction_sha256 "
                "FROM model_evaluations e LEFT JOIN model_evaluation_results r "
                "ON r.evaluation_id=e.id AND r.tenant=e.tenant ")

    def _row(self, tenant, item_id):
        with self.runtime.connection() as db:
            row = db.execute(self.read_query() + 'WHERE e.tenant=? AND e.id=?', (tenant, item_id)).fetchone()
        if row is None:
            raise Problem(404, 'not_found', 'Evaluation resource not found')
        return row


    def verify_binding(self, row, *, files=True):
        try:
            body, binding = json.loads(row['request_json']), json.loads(row['binding_json'])
            request(body)
            if (digest(body) != row['request_sha256'] or digest(binding) != row['binding_sha256']
                    or body['model_version_id'] != row['model_version_id']
                    or body['dataset_version_id'] != row['dataset_version_id']
                    or binding['model']['model_version_id'] != row['model_version_id']
                    or binding['dataset']['dataset_version_id'] != row['dataset_version_id']):
                raise ValueError('binding')
            if files and self.binding_for(row['tenant'], body, binding['submission_environment']) != binding:
                raise ValueError('resource binding changed')
            if not files:
                with self.runtime.connection() as db:
                    self.store._owned(db, 'models', row['tenant'], row['model_version_id'])
                    self.store._owned(db, 'dataset_versions', row['tenant'], row['dataset_version_id'])
            return binding
        except (Problem, ValueError, TypeError, KeyError, OSError):
            raise Problem(409, 'evaluation_integrity', 'Evaluation identity or evidence has changed') from None

    @staticmethod
    def summary(row):
        return {**{key: row[key] for key in ('id', 'status', 'created_at', 'started_at', 'finished_at',
                'model_version_id', 'dataset_version_id', 'request_sha256', 'error_code')},
                'split': 'val', 'protocol': PROTOCOL, 'result': None}

    def get(self, tenant, item_id):
        row = self._row(tenant, item_id)
        binding = self.verify_binding(row)
        value = dict(self.summary(row), binding=binding, result_sha256=None)
        parsed, result_hash = self.verified_result(row, binding)
        value.update(result=parsed, result_sha256=result_hash)
        if len(encode(value)) > MAX_DETAIL_BYTES:
            raise Problem(409, 'evaluation_integrity', 'Evaluation response exceeds the bounded contract')
        return value

    def verified_result(self, row, binding):
        #Status and terminal result came from the same SQLite SELECT snapshot.
        result = ({'result_json': row['terminal_result_json'], 'result_sha256': row['terminal_result_sha256'],
                   'prediction_sha256': row['terminal_prediction_sha256']}
                  if row['terminal_result_json'] is not None else None)
        if row['status'] != 'succeeded':
            if result:
                raise Problem(409, 'evaluation_integrity', 'Non-success evaluation has unexpected result evidence')
            return None, None
        try:
            if not result:
                raise ValueError('missing result')
            parsed = json.loads(result['result_json'])
            expected = self.expected_digests(binding, row['binding_sha256'])
            if (digest(parsed) != result['result_sha256']
                    or any(parsed['digests'].get(k) != v for k, v in expected.items())
                    or parsed['digests']['prediction_sha256'] != result['prediction_sha256']):
                raise ValueError('result binding')
            raw = _read_file(self.evidence_path(row['id']), MAX_CHILD_BYTES)
            if hashlib.sha256(raw).hexdigest() != result['prediction_sha256']:
                raise ValueError('prediction evidence')
            return parsed, result['result_sha256']
        except (ValueError, TypeError, KeyError, OSError):
            raise Problem(409, 'evaluation_integrity', 'Evaluation result or prediction evidence has changed') from None

    def list(self, tenant, model_version_id=None):
        with self.runtime.connection() as db:
            args = [tenant]
            where = 'e.tenant=?'
            if model_version_id is not None:
                self.store._owned(db, 'models', tenant, model_version_id)
                where += ' AND e.model_version_id=?'
                args.append(model_version_id)
            rows = db.execute(self.read_query() + 'WHERE ' + where + ' ORDER BY e.created_at DESC,e.id DESC LIMIT 100', args).fetchall()
        items = []
        for row in rows:
            binding = self.verify_binding(row, files=False)
            self.verified_result(row, binding)
            items.append(self.summary(row))
        result = {'evaluations': items, 'limit': 100}
        if len(encode(result)) > MAX_LIST_BYTES:
            raise Problem(409, 'evaluation_integrity', 'Evaluation list exceeds the bounded contract')
        return result

    def recover(self):
        # Only the Worker lock owner calls this. API restart does not fail live work.
        with self.runtime.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            result = db.execute("UPDATE model_evaluations SET status='failed',finished_at=?,error_code='evaluation_interrupted' WHERE status='running'", (now(),))
            return result.rowcount

    def claim(self):
        with self.runtime.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute("SELECT * FROM model_evaluations WHERE status='queued' ORDER BY created_at,id LIMIT 1").fetchone()
            if row is None:
                return None
            # A single Worker calls this before Store.claim(); compare globally.
            train = db.execute("SELECT created_at,id FROM runs WHERE status='queued' ORDER BY created_at,id LIMIT 1").fetchone()
            if train and (train['created_at'], train['id']) < (row['created_at'], row['id']):
                return None
            db.execute("UPDATE model_evaluations SET status='running',started_at=? WHERE id=? AND status='queued'", (now(), row['id']))
            return dict(db.execute('SELECT * FROM model_evaluations WHERE id=?', (row['id'],)).fetchone())

    def fail(self, item_id, code):
        if not re.fullmatch(r'[a-z_]{1,80}', code):
            code = 'evaluation_failed'
        with self.runtime.connection() as db:
            db.execute("UPDATE model_evaluations SET status='failed',finished_at=?,error_code=? WHERE id=? AND status='running'", (now(), code, item_id))

    def evidence_path(self, item_id):
        if not ID.fullmatch(item_id):
            raise Problem(409, 'evaluation_integrity', 'Invalid evaluation evidence identity')
        return self.root / (item_id + '.json')

    def disk_bytes(self):
        total = 0
        try:
            with _directory(self.root) as directory:
                for name in os.listdir(directory):
                    info = os.stat(name, dir_fd=directory, follow_symlinks=False)
                    if not stat.S_ISREG(info.st_mode):
                        raise ValueError('Unexpected storage entry')
                    total += info.st_size
        except (OSError, ValueError):
            raise Problem(409, 'evaluation_integrity', 'Evaluation evidence storage integrity failed') from None
        return total

    @staticmethod
    def expected_digests(binding, binding_hash):
        source = binding['dataset']['source']
        return {'binding_sha256': binding_hash,
                'model_artifact_sha256': binding['model']['artifact_sha256'],
                'dataset_content_sha256': binding['dataset']['content_sha256'],
                'category_mapping_sha256': binding['category_mapping_sha256'],
                'protocol_sha256': binding['protocol_sha256'],
                'source_selection_sha256': source['selection_sha256'] if source else None,
                'execution_code_sha256': binding['submission_environment']['code_sha256']}

    def succeed(self, row, result, predictions, *, deadline=None):
        binding = self.verify_binding(row)
        self.verify_code(binding)
        prediction_raw = encode(predictions)
        prediction_hash = hashlib.sha256(prediction_raw).hexdigest()
        expected = dict(self.expected_digests(binding, row['binding_sha256']), prediction_sha256=prediction_hash)
        if result.get('digests') != expected or len(prediction_raw) > MAX_CHILD_BYTES:
            raise Problem(409, 'evaluation_integrity', 'Execution evidence does not match the immutable binding')
        detail = dict(self.summary(row), status='succeeded', result=result, binding=binding, result_sha256=digest(result))
        if len(encode(detail)) > MAX_DETAIL_BYTES:
            raise Problem(409, 'evaluation_result_limit', 'Evaluation result exceeds the public response budget')
        if self.disk_bytes() + len(prediction_raw) > MAX_EVIDENCE_BYTES:
            raise Problem(429, 'evaluation_capacity', 'Evaluation evidence storage budget reached')
        target = self.evidence_path(row['id'])
        # Exclusive creation never overwrites previous or orphan evidence.
        try:
            with _directory(self.root) as directory:
                fd = os.open(target.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=directory)
                with os.fdopen(fd, 'wb') as stream:
                    stream.write(prediction_raw)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.fsync(directory)
        except OSError:
            raise Problem(409, 'evaluation_integrity', 'Evaluation evidence could not be exclusively published') from None
        if hashlib.sha256(_read_file(target, MAX_CHILD_BYTES)).hexdigest() != prediction_hash:
            raise Problem(409, 'evaluation_integrity', 'Published prediction evidence changed')
        self.verify_binding(row)
        self.verify_code(binding)
        if deadline is not None and time.monotonic() >= deadline:
            raise Problem(409, 'evaluation_timeout', 'Evaluation deadline exceeded before publication')
        with self.runtime.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            if deadline is not None and time.monotonic() >= deadline:
                raise Problem(409, 'evaluation_timeout', 'Evaluation deadline exceeded while acquiring publication transaction')
            current = self.store._owned(db, 'model_evaluations', row['tenant'], row['id'])
            if current['status'] != 'running' or current['binding_sha256'] != row['binding_sha256']:
                raise Problem(409, 'evaluation_integrity', 'Execution no longer owns evaluation publication')
            if deadline is not None and time.monotonic() >= deadline:
                raise Problem(409, 'evaluation_timeout', 'Evaluation deadline exceeded before terminal result insertion')
            db.execute('INSERT INTO model_evaluation_results VALUES(?,?,?,?,?)',
                       (row['id'], row['tenant'], encode(result).decode(), digest(result), prediction_hash))
            db.execute("UPDATE model_evaluations SET status='succeeded',finished_at=? WHERE id=?", (now(), row['id']))
            self.store.audit(db, row['tenant'], 'complete_model_evaluation', row['id'])

    @staticmethod
    def verify_code(binding, loaded_environment=None):
        expected = binding['submission_environment']['code_sha256']
        if environment()['code_sha256'] != expected or (loaded_environment and loaded_environment['code_sha256'] != expected):
            raise Problem(409, 'execution_code_changed', 'Execution source changed after submission')
        current = _env()
        if any(current[k] != binding['submission_environment'][k] for k in ('python', 'platform', 'dependencies')):
            raise Problem(409, 'execution_environment_changed', 'Execution environment changed after submission')


def _pixel_digest(rgb):
    h, w = rgb.shape[:2]
    return hashlib.sha256(f'{w}x{h}:'.encode('ascii') + rgb.tobytes()).hexdigest()


def resize_request(rgb):
    import cv2
    h, w = rgb.shape[:2]
    if max(w, h) <= 384:
        return rgb.copy()
    ratio = 384 / max(w, h)
    return cv2.resize(rgb, (max(1, int(w * ratio)), max(1, int(h * ratio))), interpolation=cv2.INTER_LINEAR)


def _diagnostics(matching):
    examples = []
    for item in matching['wrong_examples'][:EXAMPLE_LIMIT]:
        fp = set(item['false_positive_prediction_indices'])
        fn = set(item['false_negative_annotation_indices'])
        predictions = [p for p in item['predictions'] if p['index'] in fp]
        annotations = [a for a in item['annotations'] if a['index'] in fn]
        examples.append({**{k: item[k] for k in ('image_id', 'width', 'height', 'tp', 'fp', 'fn')},
            'false_positives': [{'prediction_index': p['index'], 'category_id': p['category_id'],
                                'score': p['score'], 'bbox': p['bbox']} for p in predictions[:EXAMPLE_BOX_LIMIT]],
            'false_negatives': [{'annotation_index': a['index'], 'category_id': a['category_id'],
                                'bbox': a['bbox']} for a in annotations[:EXAMPLE_BOX_LIMIT]],
            'false_positives_truncated': len(predictions) > EXAMPLE_BOX_LIMIT,
            'false_negatives_truncated': len(annotations) > EXAMPLE_BOX_LIMIT})
    return {'total': matching['wrong_examples_total'], 'limit': EXAMPLE_LIMIT,
            'truncated': matching['wrong_examples_total'] > len(examples), 'items': examples}


def measure(evaluations, row):
    """Runs only inside the bounded evaluator process."""
    import cv2
    import numpy as np
    import importlib.metadata
    from PIL import Image
    from . import pretrained_yolox as nano
    from .detection_evaluation import evaluate_detections, match_detections
    from .model_imports import verify_artifact
    binding = evaluations.verify_binding(row)
    evaluations.verify_code(binding)
    cv2.setNumThreads(1)
    manifest = evaluations.store.model(row['tenant'], row['model_version_id'])
    content = nano._read(verify_artifact(evaluations.runtime, manifest['artifact_id']), binding['model']['artifact_sha256'])
    signal.alarm(30)
    nano._static_check(content)
    model = {'session': nano._session(content)}
    signal.alarm(0)
    images, annotations, predictions, evidence = [], [], [], []
    root = evaluations.runtime.home / 'datasets' / 'versions' / row['dataset_version_id']
    for sample in binding['dataset']['samples']:
        raw = _read_file(root / sample['image'], 8 * 1024 * 1024)
        labels = _read_file(root / sample['label'], 1024 * 1024)
        if hashlib.sha256(raw).hexdigest() != sample['image_sha256'] or hashlib.sha256(labels).hexdigest() != sample['label_sha256']:
            raise Problem(409, 'evaluation_integrity', 'Dataset file changed before inference')
        with Image.open(io.BytesIO(raw)) as image:
            image.load()
            rgb = np.asarray(image.convert('RGB'))
        height, width = rgb.shape[:2]
        if (width, height) != (sample['width'], sample['height']) or _pixel_digest(rgb) != sample['pixel_sha256']:
            raise Problem(409, 'evaluation_integrity', 'Decoded dataset RGB changed')
        resized = resize_request(rgb)
        rh, rw = resized.shape[:2]
        image_id = sample['image']
        images.append({'id': image_id, 'width': width, 'height': height})
        for line in labels.decode('utf-8').splitlines():
            if not line.strip():
                continue
            cid, x, y, w, h = line.split()
            x, y, w, h = map(float, (x, y, w, h))
            # Validated YOLO roundoff may extend only by the import's 1e-9 tolerance.
            box = [max(0., (x-w/2)*width), max(0., (y-h/2)*height),
                   min(float(width), (x+w/2)*width), min(float(height), (y+h/2)*height)]
            annotations.append({'image_id': image_id, 'category_id': int(cid), 'bbox': box})
        signal.alarm(3)
        returned = nano.predict(model, [{'width': rw, 'height': rh, 'pixels': resized.reshape(-1).tolist()}])[0]
        signal.alarm(0)
        restored = []
        for detection in returned['detections']:
            box = detection['box']
            restored.append({'category_id': detection['class_id'], 'score': detection['score'],
                             'bbox': [box[0]*width/rw, box[1]*height/rh, box[2]*width/rw, box[3]*height/rh]})
        predictions.extend(dict(p, image_id=image_id) for p in restored)
        evidence.append({'image_id': image_id, 'width': width, 'height': height,
                         'request_width': rw, 'request_height': rh, 'original_rgb_sha256': sample['pixel_sha256'],
                         'request_rgb_sha256': _pixel_digest(resized), 'detections': restored})
    if len(annotations) != binding['dataset']['boxes'] or len(predictions) > MAX_IMAGES * 100:
        raise Problem(409, 'evaluation_integrity', 'Evaluation counts do not match the bound dataset')
    categories = [{'id': c['class_id'], 'name': c['name']} for c in binding['categories']]
    metrics = evaluate_detections(images, annotations, predictions, categories)
    metrics = {k: metrics[k] for k in ('protocol', 'status', 'map_50', 'map_50_95', 'ar_100',
                                     'evaluator', 'parameters', 'counts', 'per_class')}
    #AR reporting metadata belongs only to this independent protocol. Preserve
    #the shared evaluator's legacy parameter identity for trained-model serving/comparison.
    metrics['parameters'] = dict(metrics['parameters'],
        ar_100_iou_thresholds=list(metrics['parameters']['iou_thresholds']), ar_100_area='all')
    matching = match_detections(images, annotations, predictions, categories, max_examples=EXAMPLE_LIMIT)
    versions = {name: importlib.metadata.version(distribution) for name, distribution in
                [('numpy', 'numpy'), ('onnxruntime', 'onnxruntime'), ('opencv', 'opencv-python-headless'),
                 ('Pillow', 'Pillow'), ('pycocotools', 'pycocotools')]}
    result = {'metrics': metrics, 'operating_point': {'protocol': matching['protocol'], 'iou_threshold': .5,
              'score_threshold': .3, 'score_operator': '>', 'totals': matching['totals'], 'per_class': matching['per_class']},
              'wrong_examples': _diagnostics(matching),
              'digests': dict(evaluations.expected_digests(binding, row['binding_sha256']), prediction_sha256=digest(evidence)),
              'environment': dict(_env(), dependencies=versions),
              'scope': {'images': len(images), 'split': 'val', 'official_benchmark': False, 'external_audit': False,
                        'production_allowed': False, 'device_certified': False},
              'warnings': ['Fixed non-crowd subset at the serving operating point; not the official COCO benchmark.',
                           'All80 classes retained; categories without ground truth excluded from AP/AR macro averages.',
                           'Dataset source and label completeness are declared provenance, not an external audit.',
                           'COCO pretraining overlap is not independently excluded; weight license remains unverified.',
                           'Logical budgets and process separation are not production multi-tenant isolation.',
                           'Uses unchanged server inference with declared OpenCV caller resize; browser JPEG decoding or resizing may differ.']}
    evaluations.verify_binding(row)
    evaluations.verify_code(binding)
    return {'result': result, 'predictions': evidence}


def execute(evaluations, row, loaded_environment, timeout=WALL_SECONDS):
    process = None
    deadline = time.monotonic() + timeout
    try:
        binding = evaluations.verify_binding(row)
        evaluations.verify_code(binding, loaded_environment)
        env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        process = subprocess.Popen([sys.executable, '-m', 'model_platform.model_evaluations',
            '--evaluate', str(evaluations.runtime.home), row['id'], str(os.getpid())],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=env,
            cwd=str(Path(__file__).resolve().parents[1]))
        # Read incrementally, with a watchdog killing the exact child process.
        timed_out = threading.Event()
        def stop_child():
            timed_out.set()
            if process.poll() is None:
                process.kill()
        timer = threading.Timer(max(0, deadline - time.monotonic()), stop_child)
        timer.start()
        try:
            raw = process.stdout.read(MAX_CHILD_BYTES + 1)
            if len(raw) > MAX_CHILD_BYTES:
                process.kill()
                raise Problem(409, 'evaluation_result_limit', 'Evaluator result exceeds bounded channel')
            process.wait()
        finally:
            timer.cancel()
        if timed_out.is_set() or process.returncode in (-signal.SIGALRM, -signal.SIGXCPU):
            raise Problem(409, 'evaluation_timeout', 'Evaluation deadline exceeded')
        if process.returncode != 0:
            raise Problem(409, 'evaluation_failed', 'Isolated evaluation did not finish successfully')
        payload = json.loads(raw)
        if payload.get('error_code'):
            raise Problem(409, payload['error_code'], 'Isolated evaluation failed')
        evaluations.verify_code(binding, loaded_environment)
        if time.monotonic() >= deadline:
            raise Problem(409, 'evaluation_timeout', 'Evaluation deadline exceeded')
        evaluations.succeed(row, payload['result'], payload['predictions'], deadline=deadline)
    except Problem as exc:
        evaluations.fail(row['id'], exc.code)
    except Exception:
        evaluations.fail(row['id'], 'evaluation_failed')
    finally:
        if process is not None:
            if process.poll() is None:
                process.kill()
            process.wait()
            if process.stdout:
                process.stdout.close()


def _child(home, item_id, parent):
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (ADDRESS_BYTES, ADDRESS_BYTES))
    resource.setrlimit(resource.RLIMIT_CPU, (CPU_SECONDS, CPU_SECONDS))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    def watch_parent():
        while True:
            if os.getppid() != parent:
                os._exit(3)
            time.sleep(.2)
    threading.Thread(target=watch_parent, daemon=True).start()
    from .common import Runtime
    from .store import Store
    runtime = Runtime(home)
    evaluations = ModelEvaluationStore(Store(runtime))
    try:
        with runtime.connection() as db:
            row = db.execute('SELECT * FROM model_evaluations WHERE id=? AND status=\'running\'', (item_id,)).fetchone()
        if not row:
            raise Problem(409, 'evaluation_integrity', 'Evaluation is not running')
        payload = measure(evaluations, row)
    except Problem as exc:
        payload = {'error_code': exc.code}
    except ImportError:
        payload = {'error_code': 'evaluation_dependencies_missing'}
    except Exception:
        payload = {'error_code': 'evaluation_failed'}
    raw = encode(payload)
    if len(raw) > MAX_CHILD_BYTES:
        raw = encode({'error_code': 'evaluation_result_limit'})
    sys.stdout.buffer.write(raw)


if __name__ == '__main__':
    if len(sys.argv) != 5 or sys.argv[1] != '--evaluate':
        raise SystemExit(2)
    _child(sys.argv[2], sys.argv[3], int(sys.argv[4]))
