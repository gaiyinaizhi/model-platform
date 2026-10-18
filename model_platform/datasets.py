"""Tenant-owned staged dataset imports and logically immutable copied versions."""
import contextlib
import json
import os
from pathlib import Path
import shutil
import stat
import threading
import uuid

from .common import Problem, digest, encode, now, object_fields, text_value
from .dataset_validation import (Validation, MAX_ARCHIVE_BYTES, dependencies, sha256,
                                 safe_path, validate_source, verify_source_samples)

MAX_DATASET_BYTES = 512 * 1024 * 1024
MAX_IMPORTS = 256
MAX_VERSIONS = 100
SCHEMA = """
CREATE TABLE IF NOT EXISTS dataset_imports (
 id TEXT PRIMARY KEY, tenant TEXT NOT NULL, idempotency_key TEXT NOT NULL,
 request_sha256 TEXT NOT NULL, name TEXT NOT NULL, negative_policy TEXT NOT NULL,
 purpose TEXT NOT NULL DEFAULT 'training' CHECK(purpose IN ('training','evaluation')), source_json TEXT,
 created_at REAL NOT NULL, status TEXT NOT NULL CHECK(status IN ('awaiting_upload','valid','invalid','published')),
 archive_sha256 TEXT, report_json TEXT, manifest_json TEXT, version_id TEXT, stored_bytes INTEGER NOT NULL DEFAULT 0,
 UNIQUE(tenant,idempotency_key)
);
CREATE TABLE IF NOT EXISTS dataset_versions (
 id TEXT PRIMARY KEY, tenant TEXT NOT NULL, name TEXT NOT NULL,
 content_sha256 TEXT NOT NULL, created_at REAL NOT NULL, source_import_id TEXT NOT NULL,
 manifest_json TEXT NOT NULL, stored_bytes INTEGER NOT NULL,
 UNIQUE(tenant,name,content_sha256)
);
CREATE TRIGGER IF NOT EXISTS dataset_import_identity_immutable
 BEFORE UPDATE OF id,tenant,idempotency_key,request_sha256,name,negative_policy,created_at ON dataset_imports
 BEGIN SELECT RAISE(ABORT,'dataset import identity is immutable'); END;
CREATE TRIGGER IF NOT EXISTS dataset_import_report_immutable
 BEFORE UPDATE OF archive_sha256,report_json,manifest_json ON dataset_imports
 WHEN OLD.status <> 'awaiting_upload'
 BEGIN SELECT RAISE(ABORT,'dataset validation report is immutable'); END;
CREATE TRIGGER IF NOT EXISTS dataset_versions_update_immutable BEFORE UPDATE ON dataset_versions
 BEGIN SELECT RAISE(ABORT,'dataset versions are immutable'); END;
CREATE TRIGGER IF NOT EXISTS dataset_versions_delete_immutable BEFORE DELETE ON dataset_versions
 BEGIN SELECT RAISE(ABORT,'dataset versions are immutable'); END;
"""


def identifier(prefix):
    return prefix + "_" + uuid.uuid4().hex


class DatasetStore:
    def __init__(self, runtime, *, read_only=False):
        self.runtime = runtime
        self.root = runtime.home / "datasets"
        self.operation_lock = threading.Lock()
        # Existing immutable-version reads must not initialize schema or acquire
        # a write transaction (notably read-only evaluation comparisons).
        if read_only:
            return
        self.root.mkdir(mode=0o700, exist_ok=True)
        for directory in ("staging", "versions", "temporary"):
            (self.root / directory).mkdir(mode=0o700, exist_ok=True)
        with runtime.connection() as db:
            db.executescript(SCHEMA)
            # Additive migration preserves old request digests and manifest bytes.
            # BEGIN IMMEDIATE also serializes concurrent Store initializations.
            db.execute("BEGIN IMMEDIATE")
            columns = {row["name"] for row in db.execute("PRAGMA table_info(dataset_imports)")}
            if "purpose" not in columns:
                db.execute("ALTER TABLE dataset_imports ADD COLUMN purpose TEXT NOT NULL DEFAULT 'training' CHECK(purpose IN ('training','evaluation'))")
            if "source_json" not in columns:
                db.execute("ALTER TABLE dataset_imports ADD COLUMN source_json TEXT")
            db.execute("""CREATE TRIGGER IF NOT EXISTS dataset_import_purpose_immutable
                BEFORE UPDATE OF purpose,source_json ON dataset_imports
                BEGIN SELECT RAISE(ABORT,'dataset import purpose and source are immutable'); END""")

    @contextlib.contextmanager
    def operation(self):
        if not self.operation_lock.acquire(blocking=False):
            raise Problem(429, "dataset_busy", "Another dataset validation or publication is in progress; retry later")
        try:
            try:
                with self.runtime.lock("dataset-import"):
                    yield
            except RuntimeError as exc:
                if str(exc).startswith("Another dataset-import"):
                    raise Problem(429, "dataset_busy", "Another dataset operation owns this runtime") from None
                raise
        finally:
            self.operation_lock.release()

    @staticmethod
    def owned(db, table, tenant, item_id):
        row = db.execute(f"SELECT * FROM {table} WHERE tenant=? AND id=?", (tenant, item_id)).fetchone()
        if row is None:
            raise Problem(404, "not_found", "Dataset resource not found")
        return row

    def create(self, tenant, key, body):
        object_fields(body, ("name", "rights_confirmed", "negative_policy"), ("purpose", "source"))
        name = text_value(body["name"], "name", 100)
        if not name.strip() or name != name.strip() or any(ord(c) == 127 for c in name):
            raise Problem(400, "invalid_input", "Dataset name must be nonempty without surrounding spaces")
        if body["rights_confirmed"] is not True or body["negative_policy"] not in ("reject", "allow_empty"):
            raise Problem(400, "invalid_input", "Confirm dataset usage rights and choose reject or allow_empty negative policy")
        purpose = body.get("purpose", "training")
        if purpose not in ("training", "evaluation"):
            raise Problem(400, "invalid_input", "Dataset purpose must be training or evaluation")
        if "source" in body:
            if purpose != "evaluation":
                raise Problem(400, "invalid_input", "Source declarations are only supported for evaluation imports")
            validate_source(body["source"])
        source = body.get("source")
        key = text_value(key, "Idempotency-Key", 128)
        # Do not insert default fields before hashing: legacy omitted-purpose
        # requests must replay against the very same persisted request digest.
        request_hash = digest(body)
        with self.runtime.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            old = db.execute("SELECT * FROM dataset_imports WHERE tenant=? AND idempotency_key=?", (tenant, key)).fetchone()
            if old:
                if old["request_sha256"] != request_hash:
                    raise Problem(409, "idempotency_conflict", "Key already bound to different dataset metadata")
                return self.import_value(old), False
            if db.execute("SELECT count(*) FROM dataset_imports").fetchone()[0] >= MAX_IMPORTS:
                raise Problem(429, "dataset_capacity", "Runtime limit of 256 dataset import records reached")
            item_id = identifier("dsi")
            db.execute("INSERT INTO dataset_imports(id,tenant,idempotency_key,request_sha256,name,negative_policy,purpose,source_json,created_at,status) VALUES(?,?,?,?,?,?,?,?,?,?)",
                       (item_id, tenant, key, request_hash, name, body["negative_policy"], purpose,
                        encode(source).decode() if source is not None else None, now(), "awaiting_upload"))
            return self.import_value(self.owned(db, "dataset_imports", tenant, item_id)), True

    @staticmethod
    def import_value(row):
        purpose = row["purpose"]
        source = json.loads(row["source_json"]) if row["source_json"] else None
        report = json.loads(row["report_json"]) if row["report_json"] else None
        if report is not None:
            report.setdefault("purpose", purpose)
            report.setdefault("source", source)
        return {"id": row["id"], "name": row["name"], "status": row["status"],
                "purpose": purpose, "source": source,
                "negative_policy": row["negative_policy"], "created_at": row["created_at"],
                "version_id": row["version_id"], "report": report}

    def get_import(self, tenant, item_id):
        with self.runtime.connection() as db:
            return self.import_value(self.owned(db, "dataset_imports", tenant, item_id))

    @staticmethod
    def disk_bytes(root):
        total = 0
        for directory, dirs, files in os.walk(root, followlinks=False):
            if any((Path(directory) / name).is_symlink() for name in dirs + files):
                raise Problem(409, "dataset_integrity", "Dataset storage contains an unexpected symbolic link")
            total += sum((Path(directory) / name).stat().st_size for name in files)
        return total

    def upload(self, tenant, item_id, read_body):
        with self.operation():
            with self.runtime.connection() as db:
                row = self.owned(db, "dataset_imports", tenant, item_id)
            dependencies()  # Fail explicitly before accepting a body; no fallback validation.
            raw = read_body()
            if not isinstance(raw, bytes) or not 0 < len(raw) <= MAX_ARCHIVE_BYTES:
                raise Problem(413, "body_limit", "ZIP body must be 1 byte to 32 MiB")
            archive_hash = sha256(raw)
            if row["archive_sha256"] is not None:
                if row["archive_sha256"] != archive_hash:
                    raise Problem(409, "dataset_upload_conflict", "Import already validated different ZIP bytes; create a new import")
                return self.get_import(tenant, item_id)
            source = json.loads(row["source_json"]) if row["source_json"] else None
            report, manifest, files = Validation(row["negative_policy"], row["purpose"], source).validate(raw)
            target = self.root / "staging" / item_id
            temporary = self.root / "temporary" / identifier("upload")
            stored_bytes = sum(len(b) for b in files.values())
            if self.disk_bytes(self.root) + stored_bytes > MAX_DATASET_BYTES:
                raise Problem(429, "dataset_capacity", "Dataset runtime storage budget (512 MiB) reached")
            moved = False
            try:
                if manifest:
                    temporary.mkdir(mode=0o700)
                    for path, data in files.items():
                        destination = temporary / path
                        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                        with destination.open("xb") as handle:
                            handle.write(data)
                            handle.flush()
                            os.fsync(handle.fileno())
                    if target.exists():
                        raise Problem(409, "dataset_integrity", "Unregistered staging snapshot exists; operator review required")
                    temporary.rename(target)
                    moved = True
                with self.runtime.connection() as db:
                    db.execute("BEGIN IMMEDIATE")
                    current = self.owned(db, "dataset_imports", tenant, item_id)
                    if current["archive_sha256"] is not None:
                        raise Problem(409, "dataset_upload_conflict", "Import has already been validated")
                    db.execute("UPDATE dataset_imports SET status=?,archive_sha256=?,report_json=?,manifest_json=?,stored_bytes=? WHERE id=?",
                               ("valid" if report["valid"] else "invalid", archive_hash, encode(report).decode(),
                                encode(manifest).decode() if manifest else None, stored_bytes, item_id))
            except Exception:
                if moved:
                    shutil.rmtree(target)
                raise
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
            return self.get_import(tenant, item_id)

    @staticmethod
    def verify_snapshot(path, manifest, content_hash):
        try:
            if digest(manifest) != content_hash or not path.is_dir() or path.is_symlink():
                raise ValueError("Manifest mismatch")
            if any(not isinstance(entry["path"], str) or not safe_path(entry["path"])
                   or entry["path"].endswith("/") for entry in manifest["files"]):
                raise ValueError("Unsafe manifest path")
            expected = {f["path"] for f in manifest["files"]}
            if len(expected) != len(manifest["files"]):
                raise ValueError("Duplicate manifest path")
            actual = set()
            for directory, dirs, files in os.walk(path, followlinks=False):
                if any((Path(directory) / name).is_symlink() for name in dirs):
                    raise ValueError("Symbolic link")
                for name in files:
                    item = Path(directory) / name
                    if not stat.S_ISREG(item.lstat().st_mode):
                        raise ValueError("Not regular")
                    actual.add(item.relative_to(path).as_posix())
            if actual != expected:
                raise ValueError("File set changed")
            for entry in manifest["files"]:
                item = path / entry["path"]
                if item.stat().st_size != entry["bytes"] or sha256(item.read_bytes()) != entry["sha256"]:
                    raise ValueError("File hash changed")
            if manifest.get("source") is not None:
                if manifest.get("purpose") != "evaluation":
                    raise ValueError("Source purpose mismatch")
                verify_source_samples(manifest["source"], manifest["samples"], manifest["classes"])
        except (OSError, ValueError, KeyError, TypeError, Problem):
            raise Problem(409, "dataset_integrity", "Dataset snapshot is missing or has changed; publication/use is blocked") from None

    @staticmethod
    def verify_import_identity(row, manifest):
        source = json.loads(row["source_json"]) if row["source_json"] else None
        if (manifest.get("purpose", "training") != row["purpose"]
                or manifest.get("source") != source or manifest["negative_policy"] != row["negative_policy"]):
            raise Problem(409, "dataset_integrity", "Dataset purpose or source binding has changed")

    def publish(self, tenant, item_id):
        with self.operation():
            with self.runtime.connection() as db:
                row = self.owned(db, "dataset_imports", tenant, item_id)
                if row["status"] == "published":
                    return self.get_version(tenant, row["version_id"]), False
                if row["status"] != "valid":
                    raise Problem(409, "dataset_not_valid", "Only an import with a valid report can be published")
                manifest = json.loads(row["manifest_json"])
                self.verify_import_identity(row, manifest)
                content_hash = json.loads(row["report_json"])["content_sha256"]
                old = db.execute("SELECT id FROM dataset_versions WHERE tenant=? AND name=? AND content_sha256=?",
                                 (tenant, row["name"], content_hash)).fetchone()
            source = self.root / "staging" / item_id
            self.verify_snapshot(source, manifest, content_hash)
            if old:
                result = self.get_version(tenant, old["id"])
                with self.runtime.connection() as db:
                    db.execute("UPDATE dataset_imports SET status='published',version_id=?,stored_bytes=0 WHERE id=?", (old["id"], item_id))
                shutil.rmtree(source)
                return result, False
            with self.runtime.connection() as db:
                if db.execute("SELECT count(*) FROM dataset_versions").fetchone()[0] >= MAX_VERSIONS:
                    raise Problem(429, "dataset_capacity", "Runtime limit of 100 dataset versions reached")
            # Copy, never link: published bytes must not share mutable staging inodes.
            if self.disk_bytes(self.root) + row["stored_bytes"] > MAX_DATASET_BYTES:
                raise Problem(429, "dataset_capacity", "Publication needs snapshot-copy space within the 512 MiB runtime budget")
            version_id = identifier("dsv")
            temporary = self.root / "temporary" / version_id
            target = self.root / "versions" / version_id
            moved = False
            try:
                shutil.copytree(source, temporary)
                self.verify_snapshot(temporary, manifest, content_hash)
                temporary.rename(target)
                moved = True
                with self.runtime.connection() as db:
                    db.execute("BEGIN IMMEDIATE")
                    db.execute("INSERT INTO dataset_versions VALUES(?,?,?,?,?,?,?,?)",
                               (version_id, tenant, row["name"], content_hash, now(), item_id,
                                encode(manifest).decode(), row["stored_bytes"]))
                    db.execute("UPDATE dataset_imports SET status='published',version_id=?,stored_bytes=0 WHERE id=?", (version_id, item_id))
            except Exception:
                if moved:
                    shutil.rmtree(target)
                raise
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
            # After DB commit, an interrupted cleanup leaves an orphan, never a false version.
            shutil.rmtree(source)
            return self.get_version(tenant, version_id), True

    @staticmethod
    def version_value(row, full=False):
        manifest = json.loads(row["manifest_json"])
        result = {k: row[k] for k in ("id", "name", "created_at", "content_sha256", "source_import_id")}
        source = manifest.get("source")
        # Full selection evidence belongs to detail; multiplying 128 KiB by
        # every retained version would exceed the existing 8 MiB BFF budget.
        if source is not None and not full:
            source = {key: source[key] for key in ("name", "version", "annotation_sha256",
                                                   "selection_sha256", "selection_policy")}
        result.update(purpose=manifest.get("purpose", "training"), source=source)
        result.update({k: manifest[k] for k in ("negative_policy", "counts", "classes", "splits")})
        if full:
            result["manifest"] = manifest
        return result

    def get_version(self, tenant, version_id):
        with self.runtime.connection() as db:
            row = self.owned(db, "dataset_versions", tenant, version_id)
            imported = db.execute("SELECT * FROM dataset_imports WHERE tenant=? AND id=?",
                                  (tenant, row["source_import_id"])).fetchone()
            if imported is None or imported["name"] != row["name"]:
                raise Problem(409, "dataset_integrity", "Dataset import ownership binding has changed")
            manifest = json.loads(row["manifest_json"])
            self.verify_import_identity(imported, manifest)
        self.verify_snapshot(self.root / "versions" / version_id, manifest, row["content_sha256"])
        return self.version_value(row, True)

    def list_versions(self, tenant):
        with self.runtime.connection() as db:
            rows = db.execute("SELECT * FROM dataset_versions WHERE tenant=? ORDER BY created_at DESC,id DESC LIMIT 100", (tenant,)).fetchall()
        return {"versions": [self.version_value(row) for row in rows], "limit": 100}
