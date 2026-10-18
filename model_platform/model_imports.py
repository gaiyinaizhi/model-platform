"""Allowlisted byte uploads, separate validation, and explicit registration."""
import contextlib
import hashlib
import json
import os
import re
import stat
import threading
import time

from .common import Problem, digest, encode, now, object_fields, text_value
from .store import identifier

CATALOG_ID = 'yolox-nano-coco-onnx/v1'
CATALOG_REVISION = 'yolox-nano-coco-onnx/2026-10-17'
ASSET_SHA256 = 'c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d'
ASSET_BYTES = 3659407
MAX_IMPORTS = 64
MAX_STORED_BYTES = 256 * 1024 * 1024
SOURCE_URL = 'https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.onnx'


def read_artifact_body(stream, connection, size=ASSET_BYTES, timeout=15):
    """An absolute upload deadline also bounds slow clients that keep sending."""
    deadline = time.monotonic() + timeout
    original_timeout = connection.gettimeout()
    chunks, received = [], 0
    try:
        while received < size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('Upload deadline exceeded')
            connection.settimeout(remaining)
            chunk = stream.read1(min(64 * 1024, size - received))
            if not chunk:
                raise ValueError('Incomplete upload')
            received += len(chunk)
            chunks.append(chunk)
        if time.monotonic() > deadline:
            raise TimeoutError('Upload deadline exceeded')
        return b''.join(chunks)
    except (ValueError, OSError):
        raise Problem(400, 'invalid_artifact_body', 'Incomplete or timed-out upload; retry with the same import ID') from None
    finally:
        connection.settimeout(original_timeout)


def catalog():
    return {'entries': [{'id': CATALOG_ID, 'catalog_id': CATALOG_ID,
        'catalog_revision': CATALOG_REVISION, 'name': 'YOLOX-Nano COCO',
        'enabled': True, 'reason': None, 'artifact_sha256': ASSET_SHA256,
        'artifact_bytes': ASSET_BYTES, 'protocol': CATALOG_ID, 'source_url': SOURCE_URL,
        'rights_confirmation_required': True, 'redistribution_allowed': False,
        'publisher_authenticated': False, 'provenance': 'user_supplied_reviewed_hash',
        'weight_license': 'unverified'}]}


def artifact_path(runtime, artifact_id):
    if not isinstance(artifact_id, str) or not re.fullmatch(r'imp_artifact_[0-9a-f]{32}', artifact_id):
        raise Problem(409, 'import_integrity', 'Invalid imported artifact identity')
    root = runtime.home / 'import-artifacts'
    if root.is_symlink():
        raise Problem(409, 'import_integrity', 'Imported artifact storage is invalid')
    return root / (artifact_id + '.onnx')


@contextlib.contextmanager
def storage_directory(runtime):
    parts = (runtime.home / 'import-artifacts').parts
    directory = os.open(parts[0], os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in parts[1:]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = child
        yield directory
    finally:
        os.close(directory)


def verify_artifact(runtime, artifact_id, expected_hash=None):
    expected_hash = expected_hash or ASSET_SHA256
    path = artifact_path(runtime, artifact_id)
    try:
        with storage_directory(runtime) as directory:
            fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
            with os.fdopen(fd, 'rb') as stream:
                info = os.fstat(stream.fileno())
                if not stat.S_ISREG(info.st_mode) or info.st_size != ASSET_BYTES:
                    raise ValueError('size')
                actual = hashlib.sha256(stream.read(ASSET_BYTES + 1)).hexdigest()
                after = os.fstat(stream.fileno())
                if (info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns,info.st_ctime_ns) != (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns):
                    raise ValueError('changed while reading')
        if actual != expected_hash or expected_hash != ASSET_SHA256:
            raise ValueError('hash')
    except (OSError, ValueError):
        raise Problem(409, 'import_integrity', 'Imported artifact failed identity verification') from None
    return path


def validate_manifest(db, row, manifest):
    """Bind persisted origin, tenant, import, validation, catalog and bytes."""
    if 'origin' not in row.keys() or row['origin'] != 'imported':
        if manifest.get('origin', 'trained') != 'trained':
            raise Problem(409, 'import_integrity', 'Model origin binding failed')
        return
    imported = db.execute('SELECT * FROM model_imports WHERE id=? AND tenant=?', (row['import_id'], row['tenant'])).fetchone()
    validation = db.execute('SELECT * FROM import_validations WHERE id=? AND import_id=? AND tenant=? AND catalog_revision=? AND artifact_sha256=?',
        (row['validation_id'], row['import_id'], row['tenant'], row['catalog_revision'], row['artifact_sha256'])).fetchone()
    expected = {'origin': 'imported', 'model_version_id': row['id'], 'import_id': row['import_id'],
        'validation_id': row['validation_id'], 'catalog_revision': row['catalog_revision'],
        'artifact_sha256': row['artifact_sha256'], 'catalog_id': CATALOG_ID}
    if (not imported or not validation or imported['status'] != 'validated'
            or imported['validation_id'] != row['validation_id']
            or imported['artifact_sha256'] != row['artifact_sha256']
            or imported['catalog_revision'] != CATALOG_REVISION
            or imported['catalog_id'] != CATALOG_ID or row['artifact_sha256'] != ASSET_SHA256
            or manifest.get('artifact_id') != imported['artifact_id']
            or any(manifest.get(key) != value for key, value in expected.items())
            or any(key in manifest for key in ('recipe','run_id','attempt_id','evaluation_id'))):
        raise Problem(409, 'import_integrity', 'Model import validation binding failed')


def public_identity(manifest):
    return {key: manifest[key] for key in ('origin', 'import_id', 'validation_id', 'protocol',
        'class_names', 'artifact_sha256', 'local_evaluation_status', 'production_allowed', 'device_certified') if key in manifest}


def validate_isolated(path):
    from .pretrained_yolox import validate_asset
    report = validate_asset(path)
    if not isinstance(report, dict) or report.get('status') != 'validated':
        raise Problem(422, 'validation_failed', 'Imported model validation failed')
    return report


class ModelImportStore:
    def __init__(self, store):
        self.store, self.runtime = store, store.runtime
        self.operation_lock = threading.Lock()
        self.root = self.runtime.home / 'import-artifacts'
        if self.root.is_symlink():
            raise Problem(409, 'import_integrity', 'Imported artifact storage is invalid')
        self.root.mkdir(mode=0o700, exist_ok=True)

    def require_schema(self):
        with self.runtime.connection() as db:
            if 'origin' not in {r['name'] for r in db.execute('PRAGMA table_info(models)')}:
                raise Problem(409, 'migration_required', 'Stop services and run the explicit imported-origin migration')

    @contextlib.contextmanager
    def operation(self):
        if not self.operation_lock.acquire(blocking=False):
            raise Problem(429, 'model_import_busy', 'Another import operation is active')
        try:
            try:
                with self.runtime.lock('model-import'):
                    yield
            except RuntimeError as exc:
                if str(exc).startswith('Another model-import'):
                    raise Problem(429, 'model_import_busy', 'Another import operation owns this runtime') from None
                raise
        finally:
            self.operation_lock.release()

    def recover(self):
        # API caller holds api.lock; this lock excludes another import operation.
        with self.operation(), self.runtime.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            db.execute("UPDATE model_imports SET status='rejected',reason='validation_interrupted' WHERE status='validating'")

    def create(self, tenant, key, body):
        self.require_schema()
        object_fields(body, ('catalog_id', 'rights_confirmed'))
        key = text_value(key, 'Idempotency-Key', 128)
        if body['catalog_id'] != CATALOG_ID or body['rights_confirmed'] is not True:
            raise Problem(400, 'invalid_input', 'Choose the reviewed catalog entry and confirm usage rights')
        with self.runtime.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            old = db.execute('SELECT * FROM model_imports WHERE tenant=? AND idempotency_key=?', (tenant, key)).fetchone()
            if old:
                if old['request_sha256'] != digest(body):
                    raise Problem(409, 'idempotency_conflict', 'Key is bound to another import request')
                item_id, created = old['id'], False
            else:
                if db.execute('SELECT count(*) FROM model_imports').fetchone()[0] >= MAX_IMPORTS:
                    raise Problem(429, 'model_import_capacity', 'Runtime import record limit reached')
                item_id, created = identifier('import'), True
                db.execute('INSERT INTO model_imports(id,tenant,idempotency_key,request_sha256,catalog_id,catalog_revision,rights_confirmed,created_at,status) VALUES(?,?,?,?,?,?,1,?,?)',
                    (item_id, tenant, key, digest(body), CATALOG_ID, CATALOG_REVISION, now(), 'created'))
                self.store.audit(db, tenant, 'create_model_import', item_id)
        return self.get_import(tenant, item_id), created

    def get_import(self, tenant, item_id):
        with self.runtime.connection() as db:
            row = self.store._owned(db, 'model_imports', tenant, item_id)
            validation = db.execute('SELECT report_json FROM import_validations WHERE id=?', (row['validation_id'],)).fetchone()
            model = db.execute('SELECT id FROM models WHERE tenant=? AND import_id=?', (tenant, item_id)).fetchone() if 'origin' in {r['name'] for r in db.execute('PRAGMA table_info(models)')} else None
            result = {k: row[k] for k in ('id', 'catalog_id', 'catalog_revision', 'status', 'created_at', 'artifact_sha256', 'validation_id', 'reason')}
            result.update(validation=json.loads(validation[0]) if validation else None,
                model_version_id=model[0] if model else None)
            return result

    def disk_bytes(self):
        total = 0
        try:
            with storage_directory(self.runtime) as directory:
                for name in os.listdir(directory):
                    info = os.stat(name, dir_fd=directory, follow_symlinks=False)
                    if not stat.S_ISREG(info.st_mode):
                        raise Problem(409, 'import_integrity', 'Unexpected imported artifact storage entry')
                    total += info.st_size
        except OSError:
            raise Problem(409, 'import_integrity', 'Imported artifact storage is unavailable') from None
        return total

    def upload(self, tenant, item_id, read_body):
        self.require_schema()
        with self.operation():
            with self.runtime.connection() as db:
                row = self.store._owned(db, 'model_imports', tenant, item_id)
            raw = read_body()
            if not isinstance(raw, bytes) or len(raw) != ASSET_BYTES:
                raise Problem(400, 'invalid_artifact_body', 'Exact catalog artifact length required; incomplete uploads are not bound')
            actual = hashlib.sha256(raw).hexdigest()
            if row['artifact_sha256']:
                if row['artifact_sha256'] != actual:
                    raise Problem(409, 'artifact_conflict', 'Import is bound to different bytes')
                return self.get_import(tenant, item_id)
            if row['status'] != 'created':
                return self.get_import(tenant, item_id)
            if self.disk_bytes() + len(raw) > MAX_STORED_BYTES:
                raise Problem(429, 'model_import_capacity', 'Imported artifact byte budget reached')
            artifact_id = identifier('imp_artifact')
            with self.runtime.connection() as db:
                db.execute('BEGIN IMMEDIATE')
                db.execute("UPDATE model_imports SET status='validating',artifact_sha256=?,artifact_id=? WHERE id=? AND status='created'",
                    (actual, artifact_id, item_id))
            try:
                if actual != ASSET_SHA256:
                    raise Problem(422, 'artifact_hash_mismatch', 'Bytes do not match the reviewed catalog artifact')
                target = artifact_path(self.runtime, artifact_id)
                temporary = self.root / (artifact_id + '.tmp')
                with storage_directory(self.runtime) as directory:
                    fd = os.open(temporary.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=directory)
                    with os.fdopen(fd, 'wb') as stream:
                        stream.write(raw)
                        stream.flush()
                        os.fsync(stream.fileno())
                    os.replace(temporary.name, target.name, src_dir_fd=directory, dst_dir_fd=directory)
                    os.fsync(directory)
                report = validate_isolated(verify_artifact(self.runtime, artifact_id))
                report = dict(report, validator='import-validation/v1', import_id=item_id,
                    catalog_id=CATALOG_ID, catalog_revision=CATALOG_REVISION, artifact_sha256=actual)
                validation_id = identifier('validation')
                with self.runtime.connection() as db:
                    db.execute('BEGIN IMMEDIATE')
                    owned = self.store._owned(db, 'model_imports', tenant, item_id)
                    if owned['status'] != 'validating':
                        raise Problem(409, 'import_conflict', 'Validation no longer owns this import')
                    db.execute('INSERT INTO import_validations VALUES(?,?,?,?,?,?,?)',
                        (validation_id, item_id, tenant, CATALOG_REVISION, actual, encode(report).decode(), now()))
                    db.execute("UPDATE model_imports SET status='validated',validation_id=? WHERE id=?", (validation_id, item_id))
            except Exception as exc:
                code = exc.code if isinstance(exc, Problem) else 'validation_failed'
                with self.runtime.connection() as db:
                    db.execute("UPDATE model_imports SET status='rejected',reason=? WHERE id=? AND status='validating'", (code, item_id))
            return self.get_import(tenant, item_id)

    def register(self, tenant, item_id):
        self.require_schema()
        with self.operation(), self.runtime.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            row = self.store._owned(db, 'model_imports', tenant, item_id)
            old = db.execute('SELECT id FROM models WHERE import_id=? AND tenant=?', (item_id, tenant)).fetchone()
            if old:
                model_id, created = old[0], False
            else:
                if row['status'] != 'validated':
                    raise Problem(409, 'import_not_validated', 'Only a validated import can be registered')
                validation = db.execute('SELECT * FROM import_validations WHERE id=? AND import_id=? AND tenant=? AND catalog_revision=? AND artifact_sha256=?',
                    (row['validation_id'], item_id, tenant, row['catalog_revision'], row['artifact_sha256'])).fetchone()
                if (not validation or row['catalog_id'] != CATALOG_ID or row['catalog_revision'] != CATALOG_REVISION
                        or row['artifact_sha256'] != ASSET_SHA256):
                    raise Problem(409, 'import_integrity', 'Import validation binding failed')
                verify_artifact(self.runtime, row['artifact_id'])
                from . import pretrained_yolox as nano
                model_id, created = identifier('model'), True
                manifest = {'model_version_id': model_id, 'origin': 'imported', 'import_id': item_id,
                    'validation_id': row['validation_id'], 'catalog_id': CATALOG_ID,
                    'catalog_revision': CATALOG_REVISION, 'protocol': nano.PROTOCOL,
                    'format': nano.FORMAT, 'contract': nano.CONTRACT, 'dimensions': 519168,
                    'artifact_id': row['artifact_id'], 'artifact_sha256': ASSET_SHA256,
                    'artifact_bytes': ASSET_BYTES, 'local_evaluation_status': 'not_evaluated',
                    'production_allowed': False, 'device_certified': False, 'evaluation': None,
                    'class_names': list(nano.CLASS_NAMES), 'category_ids': list(nano.COCO_CATEGORY_IDS),
                    'label_table_sha256': nano.LABEL_TABLE_SHA256,
                    'source': {'url': SOURCE_URL, 'redistribution_allowed': False,
                        'publisher_authenticated': False, 'release_asset_id': 42724905,
                        'release_asset_bytes': ASSET_BYTES, 'publisher_digest': None,
                        'provenance': 'user_supplied_reviewed_hash', 'source_commit': nano.SOURCE_COMMIT,
                        'repository_code_license': 'Apache-2.0', 'weight_license': 'unverified'}}
                db.execute("INSERT INTO models(id,tenant,artifact_sha256,manifest_json,created_at,origin,import_id,validation_id,catalog_revision) VALUES(?,?,?,?,?,'imported',?,?,?)",
                    (model_id, tenant, ASSET_SHA256, encode(manifest).decode(), now(), item_id, row['validation_id'], CATALOG_REVISION))
                self.store.audit(db, tenant, 'register_imported_model', model_id)
        return self.store.model(tenant, model_id), created
