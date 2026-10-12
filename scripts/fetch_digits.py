"""Fetch the complete pinned UCI optical-digits split into private runtime files."""
import csv
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import urllib.parse
import urllib.request
import zipfile


SOURCE_URL = "https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits"
ARCHIVE_URL = "https://archive.ics.uci.edu/static/public/80/optical%2Brecognition%2Bof%2Bhandwritten%2Bdigits.zip"
ARCHIVE_NAME = "optical-recognition-of-handwritten-digits.zip"
ARCHIVE_SHA256 = "0d7b054fea010270e9b3f06411c654c5e59547732ad626381980baffe0a23fb0"
FILE_SHA256 = {
    "optdigits.tra": "e1b683cc211604fe8fd8c4417e6a69f31380e0c61d4af22e93cc21e9257ffedd",
    "optdigits.tes": "6ebb3d2fee246a4e99363262ddf8a00a3c41bee6014c373ed9d9216ba7f651b8",
    "optdigits.names": "3e82f7202d72a2b7dbdbc324c8c90fe8853164f5d6ab978a071357ad3de89f02",
}
DATASET_VERSION = "uci-optdigits-official-split-v1"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
ATTRIBUTION = ("Alpaydin, E. & Kaynak, C. (1998). Optical Recognition of Handwritten Digits "
               "[Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C50P49.")
CONFIG = {"epochs": 40, "learning_rate": 0.03, "seed": 1729}
MAX_ARCHIVE_BYTES = 2 * 1024 * 1024


class OfficialRedirectsOnly(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        destination = urllib.parse.urlsplit(newurl)
        if destination.scheme != "https" or destination.netloc != "archive.ics.uci.edu":
            raise ValueError("Dataset redirect left the official HTTPS UCI source")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def write_private(path, content):
    """Atomic runtime output; do not follow an existing output symlink."""
    if path.is_symlink():
        raise ValueError("Runtime output must not be a symlink")
    fd, temporary = tempfile.mkstemp(prefix=".digits-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def json_bytes(value):
    # Compact encoding keeps the complete official split below 1 MiB.
    return json.dumps(value, separators=(",", ":"), allow_nan=False).encode()


def verify(content, expected, name):
    if hashlib.sha256(content).hexdigest() != expected:
        raise ValueError(f"Pinned SHA-256 mismatch for {name}; no data was accepted")


def parse_rows(content, expected_count):
    rows = []
    for values in csv.reader(io.StringIO(content.decode("ascii"))):
        if len(values) != 65:
            raise ValueError("Expected exactly 64 integer pixels and one digit label")
        # int() rejects decimal-valued pixels; the source files are integer CSV.
        values = [int(value) for value in values]
        if any(not 0 <= value <= 16 for value in values[:-1]) or not 0 <= values[-1] <= 9:
            raise ValueError("Invalid official digit pixel or label")
        rows.append({"x": values[:-1], "y": values[-1]})
    if len(rows) != expected_count or {row["y"] for row in rows} != set(range(10)):
        raise ValueError("Official split row count or class coverage mismatch")
    return rows


def audit_split(train, evaluation):
    """Count duplicates without deleting, moving or resampling any official row."""
    result, feature_sets, row_sets = {}, {}, {}
    for name, rows in (("train", train), ("evaluation", evaluation)):
        features = Counter(tuple(row["x"]) for row in rows)
        labeled = Counter((tuple(row["x"]), row["y"]) for row in rows)
        feature_sets[name], row_sets[name] = set(features), set(labeled)
        result[name] = {
            "sample_count": len(rows),
            "class_counts": {str(label): sum(row["y"] == label for row in rows) for label in range(10)},
            "duplicate_feature_rows": len(rows) - len(features),
            "duplicate_labeled_rows": len(rows) - len(labeled),
        }
    result["cross_split_shared_feature_vectors"] = len(feature_sets["train"] & feature_sets["evaluation"])
    result["cross_split_shared_labeled_rows"] = len(row_sets["train"] & row_sets["evaluation"])
    result["filtering"] = "none; every official row is retained in its original split and order"
    return result


def prepare(home=None):
    home = Path(home or os.environ.get("MP_HOME", ".runtime")).resolve()
    folder, source = home / "digits", home / "digits" / "source"
    for path in (home, folder, source):
        if path.is_symlink():
            raise ValueError("Runtime directories must not be symlinks")
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
    archive = source / ARCHIVE_NAME
    if archive.is_symlink():
        raise ValueError("Dataset archive must not be a symlink")
    downloaded = not archive.exists()
    if downloaded:
        opener = urllib.request.build_opener(OfficialRedirectsOnly())
        with opener.open(ARCHIVE_URL, timeout=45) as response:
            content = response.read(MAX_ARCHIVE_BYTES + 1)
        if len(content) > MAX_ARCHIVE_BYTES:
            raise ValueError("Official dataset archive exceeds the download budget")
    else:
        if archive.stat().st_size > MAX_ARCHIVE_BYTES:
            raise ValueError("Cached dataset archive exceeds the size budget")
        content = archive.read_bytes()
    verify(content, ARCHIVE_SHA256, ARCHIVE_NAME)
    raw = {}
    with zipfile.ZipFile(io.BytesIO(content)) as bundle:
        # Never extract paths supplied by an archive. Read only fixed members.
        for name, expected in FILE_SHA256.items():
            if bundle.namelist().count(name) != 1 or bundle.getinfo(name).file_size > MAX_ARCHIVE_BYTES:
                raise ValueError("Official archive member count or size mismatch")
            raw[name] = bundle.read(name)
            verify(raw[name], expected, name)
    train, evaluation = parse_rows(raw["optdigits.tra"], 3823), parse_rows(raw["optdigits.tes"], 1797)
    audit = audit_split(train, evaluation)
    provenance = {
        "archive_sha256": ARCHIVE_SHA256,
        "train_sha256": FILE_SHA256["optdigits.tra"],
        "evaluation_sha256": FILE_SHA256["optdigits.tes"],
        "official_split": True, "license": "CC-BY-4.0", "source_url": SOURCE_URL,
    }
    body = {
        "recipe": "softmax-digits/v1",
        "dataset": {"source": SOURCE_URL, "version": DATASET_VERSION, "rights_confirmed": True,
                    "provenance": provenance, "train": train, "evaluation": evaluation},
        "config": dict(CONFIG),
    }
    metadata = {
        "dataset": "Optical Recognition of Handwritten Digits", "version": DATASET_VERSION,
        "source_url": SOURCE_URL, "archive_url": ARCHIVE_URL,
        "archive_sha256": ARCHIVE_SHA256, "file_sha256": FILE_SHA256,
        "attribution": ATTRIBUTION, "doi": "https://doi.org/10.24432/C50P49",
        "license": "CC-BY-4.0", "license_url": LICENSE_URL,
        "license_verified_at_source_on": "2026-10-12",
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "official_split": True, "training_writers": 30, "evaluation_writers": 13,
        "dimensions": 64, "pixel_range": [0, 16], "labels": list(range(10)),
        "preprocessing": "preserve official 8x8 integer features; recipe scales each pixel by 16",
        "config": dict(CONFIG), "audit": audit,
    }
    if downloaded:
        write_private(archive, content)
    for name, data in raw.items():
        write_private(source / name, data)
    attribution = (ATTRIBUTION + "\nSource: " + SOURCE_URL + "\nLicense: CC BY 4.0 " + LICENSE_URL
                   + "\nOriginal files and split retained without filtering; model input is pixels / 16.\n")
    write_private(source / "ATTRIBUTION.txt", attribution.encode())
    write_private(folder / "metadata.json", json_bytes(metadata))
    write_private(folder / "request.json", json_bytes(body))
    return body, metadata


def main():
    body, metadata = prepare()
    print(json.dumps({"source_url": SOURCE_URL, "license": metadata["license"],
                      "archive_sha256": ARCHIVE_SHA256, "audit": metadata["audit"],
                      "request_bytes": len(json_bytes(body)),
                      "runtime_subdirectory": "digits", "status": "verified"}, indent=2))


if __name__ == "__main__":
    main()
