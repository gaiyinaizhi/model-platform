"""One pinned, official Fashion-MNIST reference; no arbitrary dataset paths or URLs.

Only fetch_dataset performs network I/O. Workers call load_dataset, which reads
and authenticates all four compressed source files again before using any pixels.
"""
from contextlib import contextmanager
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import secrets
import stat
import struct
import time
import urllib.parse
import urllib.request

DATASET_REF = "fashion-mnist-official/v1"
CACHE_NAME = "fashion-mnist-v1"
DATASET_VERSION = CACHE_NAME
SOURCE_COMMIT = "b2617bb6d3ffa2e429640350f613e3291e10b141"
SOURCE_URL = "https://github.com/zalandoresearch/fashion-mnist"
RAW_BASE = f"https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/{SOURCE_COMMIT}/"
LICENSE_URL = RAW_BASE + "LICENSE"
LICENSE_SHA256 = "13ef4788476d292858fa60eb9a5f74aeca5c65770bc885ccaa05823a17ef7be1"
LICENSE_BYTES = 1109
LABELS = ("T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal",
          "Shirt", "Sneaker", "Bag", "Ankle boot")
# SHA-256 and byte counts verified from the fixed official commit; MD5 values
# are independently published in the official README. MD5 is not the trust root.
FILES = {
    "train-images-idx3-ubyte.gz": {
        "compressed_bytes": 26421880, "raw_bytes": 47040016,
        "sha256": "3aede38d61863908ad78613f6a32ed271626dd12800ba2636569512369268a84",
        "md5": "8d4fb7e6c68d591d4c3dfef9ec88bf0d", "count": 60000, "kind": "images"},
    "train-labels-idx1-ubyte.gz": {
        "compressed_bytes": 29515, "raw_bytes": 60008,
        "sha256": "a04f17134ac03560a47e3764e11b92fc97de4d1bfaf8ba1a3aa29af54cc90845",
        "md5": "25c81989df183df01b3e8a0aad5dffbe", "count": 60000, "kind": "labels"},
    "t10k-images-idx3-ubyte.gz": {
        "compressed_bytes": 4422102, "raw_bytes": 7840016,
        "sha256": "346e55b948d973a97e58d2351dde16a484bd415d4595297633bb08f03db6a073",
        "md5": "bef4ecab320f06d8554ea6380940ec79", "count": 10000, "kind": "images"},
    "t10k-labels-idx1-ubyte.gz": {
        "compressed_bytes": 5148, "raw_bytes": 10008,
        "sha256": "67da17c76eaffca5446c3361aaab5c3cd6d1c2608764d35dfb1850b086bf8dd5",
        "md5": "bb300cfdad3c16e7a12a480ee83cd310", "count": 10000, "kind": "labels"},
}
DATASET_SHA256 = hashlib.sha256(json.dumps(FILES, sort_keys=True,
                                         separators=(",", ":")).encode()).hexdigest()
DOWNLOAD_TIMEOUT_SECONDS = 30
DOWNLOAD_DEADLINE_SECONDS = 180


class DatasetError(ValueError):
    """The fixed reference is missing, unsafe, corrupt, or unavailable."""

    def __init__(self, message, code="dataset_corrupt"):
        self.code = code
        super().__init__(message)


def provenance():
    """Fresh JSON-safe source facts; informational manifests are never trusted."""
    return {"reference": DATASET_REF, "source_url": SOURCE_URL,
            "source_commit": SOURCE_COMMIT, "license": "MIT", "license_url": LICENSE_URL,
            "license_sha256": LICENSE_SHA256, "dataset_sha256": DATASET_SHA256,
            "official_split": True, "train_count": 60000, "evaluation_count": 10000,
            "shape": [28, 28], "pixel_range": [0, 255], "class_order": list(range(10)),
            "labels": list(LABELS), "files": {name: dict(spec) for name, spec in FILES.items()},
            "filtering": "none; official split and record order retained"}


@contextmanager
def _cache_directory(runtime, create=False):
    """Use no-follow directory descriptors for the fixed cache components."""
    home = Path(runtime.home)
    if not home.is_absolute():
        home = home.absolute()
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    fd = None
    try:
        # Walk even the runtime's ancestors without following a symlink. Runtime
        # normally supplies a resolved absolute path; this also supports tests.
        fd = os.open(home.anchor, flags)
        for component in home.parts[1:]:
            child = os.open(component, flags, dir_fd=fd)
            os.close(fd)
            fd = child
        for component in ("datasets", CACHE_NAME):
            if create:
                try:
                    os.mkdir(component, mode=0o700, dir_fd=fd)
                except FileExistsError:
                    pass
            child = os.open(component, flags, dir_fd=fd)
            os.close(fd)
            fd = child
        yield fd
    except OSError as exc:
        code = "dataset_missing" if isinstance(exc, FileNotFoundError) else "dataset_corrupt"
        raise DatasetError("Fashion-MNIST cache is missing or unsafe; run scripts/fetch_fashion.py", code) from exc
    finally:
        if fd is not None:
            os.close(fd)


def _read(fd, name, expected_size):
    """Bounded regular-file read; O_NOFOLLOW also rejects a symlink race."""
    handle = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
    with os.fdopen(handle, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size != expected_size:
            raise DatasetError(f"Unexpected cached size or file type: {name}")
        content = stream.read(expected_size + 1)
    if len(content) != expected_size:
        raise DatasetError(f"Cached file changed while reading: {name}")
    return content


def _verify(content, spec, name):
    if len(content) != spec["compressed_bytes"] or hashlib.sha256(content).hexdigest() != spec["sha256"]:
        raise DatasetError(f"Pinned SHA-256 or byte count mismatch: {name}")
    if "md5" in spec and hashlib.md5(content, usedforsecurity=False).hexdigest() != spec["md5"]:
        raise DatasetError(f"Official MD5 mismatch: {name}")


def _decode(content, spec, name):
    _verify(content, spec, name)
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(content), mode="rb") as stream:
            raw = stream.read(spec["raw_bytes"] + 1)
    except (OSError, EOFError) as exc:
        raise DatasetError(f"Invalid gzip data: {name}") from exc
    if len(raw) != spec["raw_bytes"]:
        raise DatasetError(f"Decompressed IDX size exceeds or misses the fixed bound: {name}")
    if spec["kind"] == "images":
        if struct.unpack(">IIII", raw[:16]) != (2051, spec["count"], 28, 28):
            raise DatasetError(f"Invalid official image IDX header: {name}")
        return raw[16:]
    if struct.unpack(">II", raw[:8]) != (2049, spec["count"]):
        raise DatasetError(f"Invalid official label IDX header: {name}")
    labels = raw[8:]
    if set(labels) != set(range(10)) or any(labels.count(label) != spec["count"] // 10 for label in range(10)):
        raise DatasetError(f"Invalid official label values or class counts: {name}")
    return labels


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise DatasetError("Redirects are not accepted for the pinned official source")


def _download(name, expected_size):
    # Private fixed-name dispatch, not a general URL fetch facility.
    if name not in FILES and name != "LICENSE":
        raise DatasetError("Unknown official dataset file")
    url = LICENSE_URL if name == "LICENSE" else RAW_BASE + "data/fashion/" + name
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https" or parsed.netloc != "raw.githubusercontent.com":
        raise DatasetError("Only the official pinned HTTPS source is allowed")
    opener = urllib.request.build_opener(_NoRedirect())
    request = urllib.request.Request(url, headers={"Accept-Encoding": "identity"})
    deadline, chunks, total = time.monotonic() + DOWNLOAD_DEADLINE_SECONDS, [], 0
    with opener.open(request, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
        if response.status != 200 or response.geturl() != url:
            raise DatasetError("Unexpected response from official dataset source")
        declared = response.headers.get("Content-Length")
        if declared is not None and int(declared) != expected_size:
            raise DatasetError("Official source Content-Length differs from pinned byte count")
        while True:
            if time.monotonic() > deadline:
                raise DatasetError("Official dataset download exceeded its time budget")
            chunk = response.read1(min(1024 * 1024, expected_size + 1 - total))
            if time.monotonic() > deadline:
                raise DatasetError("Official dataset download exceeded its time budget")
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > expected_size:
                raise DatasetError("Official dataset download exceeded its byte budget")
    if total != expected_size:
        raise DatasetError("Official dataset download did not match its pinned byte count")
    return b"".join(chunks)


def _write(fd, name, content):
    """Atomic 0600 output, refusing existing links and non-regular files."""
    try:
        existing = os.stat(name, dir_fd=fd, follow_symlinks=False)
    except FileNotFoundError:
        existing = None
    if existing is not None and not stat.S_ISREG(existing.st_mode):
        raise DatasetError("Cache output must be a regular non-symlink file")
    temporary = ".fashion-" + secrets.token_hex(16)
    handle = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                     0o600, dir_fd=fd)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, name, src_dir_fd=fd, dst_dir_fd=fd)
        os.fsync(fd)
    finally:
        try:
            os.unlink(temporary, dir_fd=fd)
        except FileNotFoundError:
            pass


def fetch_dataset(runtime):
    """Fetch/verify only this reference; corrupt existing data is never replaced."""
    with _cache_directory(runtime, create=True) as fd:
        for name, spec in FILES.items():
            try:
                content = _read(fd, name, spec["compressed_bytes"])
                downloaded = False
            except FileNotFoundError:
                content, downloaded = _download(name, spec["compressed_bytes"]), True
            _decode(content, spec, name)
            if downloaded:
                _write(fd, name, content)
        license_spec = {"compressed_bytes": LICENSE_BYTES, "sha256": LICENSE_SHA256}
        try:
            license_content = _read(fd, "LICENSE", LICENSE_BYTES)
            downloaded = False
        except FileNotFoundError:
            license_content, downloaded = _download("LICENSE", LICENSE_BYTES), True
        _verify(license_content, license_spec, "LICENSE")
        if downloaded:
            _write(fd, "LICENSE", license_content)
        metadata = provenance()
        _write(fd, "metadata.json", json.dumps(metadata, sort_keys=True, indent=2).encode() + b"\n")
    return metadata


def load_dataset(runtime):
    """Offline full split: uint8 (60000,784), (60000,), (10000,784), (10000,)."""
    import numpy as np
    arrays = []
    with _cache_directory(runtime) as fd:
        for name, spec in FILES.items():
            # Recompute official source digests on every Worker load. No trust in
            # the download step, request declaration, metadata file or raw cache.
            raw = _decode(_read(fd, name, spec["compressed_bytes"]), spec, name)
            array = np.frombuffer(raw, dtype=np.uint8)
            arrays.append(array.reshape(spec["count"], 784).copy()
                          if spec["kind"] == "images" else array.copy())
    return tuple(arrays)
