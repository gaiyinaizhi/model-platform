"""Fixed-reference data safety tests; synthetic fixtures are not model evidence.

All mandatory tests use only stdlib and never contact the network. The separately
labelled full-source check runs only when NumPy and the official local cache exist.
"""
import gzip
import hashlib
import importlib.util
import io
import os
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from model_platform import fashion_data as data

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_CACHE = ROOT / ".runtime" / "datasets" / data.CACHE_NAME
HAS_OFFICIAL_CACHE = all((OFFICIAL_CACHE / name).is_file() for name in data.FILES)
HAS_NUMPY = importlib.util.find_spec("numpy") is not None


def synthetic_image(raw=None, **changes):
    """One made-up zero image, not an official training or evaluation record."""
    raw = raw if raw is not None else struct.pack(">IIII", 2051, 1, 28, 28) + bytes(784)
    content = gzip.compress(raw, mtime=0)
    spec = {"compressed_bytes": len(content), "raw_bytes": len(raw),
            "sha256": hashlib.sha256(content).hexdigest(), "count": 1, "kind": "images"}
    spec.update(changes)
    return content, spec


class FixedSourceSafetyTests(unittest.TestCase):
    def setUp(self):
        test_root = ROOT / ".runtime" / "tests"
        test_root.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="fashion-data-", dir=test_root)
        self.addCleanup(self.temporary.cleanup)
        self.home = Path(self.temporary.name)
        self.runtime = SimpleNamespace(home=self.home)
        self.cache = self.home / "datasets" / data.CACHE_NAME

    def assert_dataset_error(self, code, call):
        with self.assertRaises(data.DatasetError) as caught:
            call()
        self.assertEqual(caught.exception.code, code)

    def read_cached(self, name="test.gz", size=1):
        with data._cache_directory(self.runtime) as directory:
            return data._read(directory, name, size)

    def test_missing_cache_is_actionable(self):
        self.assert_dataset_error("dataset_missing", self.read_cached)

    def test_missing_file_is_actionable(self):
        self.cache.mkdir(parents=True)
        self.assert_dataset_error("dataset_missing", self.read_cached)

    def test_symlink_cache_directory_is_rejected(self):
        target = self.home / "actual-datasets"
        (target / data.CACHE_NAME).mkdir(parents=True)
        (self.home / "datasets").symlink_to(target, target_is_directory=True)
        self.assert_dataset_error("dataset_corrupt", self.read_cached)

    def test_symlink_cache_file_is_rejected(self):
        self.cache.mkdir(parents=True)
        target = self.home / "ordinary-file"
        target.write_bytes(b"x")
        (self.cache / "test.gz").symlink_to(target)
        self.assert_dataset_error("dataset_corrupt", self.read_cached)

    def test_wrong_compressed_size_is_rejected(self):
        self.cache.mkdir(parents=True)
        (self.cache / "test.gz").write_bytes(b"too large")
        self.assert_dataset_error("dataset_corrupt", self.read_cached)

    @unittest.skipUnless(hasattr(os, "mkfifo"), "POSIX FIFO safety check")
    def test_fifo_is_rejected_without_open_blocking(self):
        self.cache.mkdir(parents=True)
        os.mkfifo(self.cache / "test.gz")
        self.assert_dataset_error("dataset_corrupt", self.read_cached)

    def test_valid_synthetic_idx_is_parsed(self):
        content, spec = synthetic_image()
        self.assertEqual(data._decode(content, spec, "synthetic"), bytes(784))

    def test_same_size_hash_corruption_is_rejected(self):
        content, spec = synthetic_image(sha256="0" * 64)
        self.assert_dataset_error("dataset_corrupt", lambda: data._decode(content, spec, "synthetic"))

    def test_wrong_idx_count_is_rejected(self):
        content, spec = synthetic_image(count=2)
        self.assert_dataset_error("dataset_corrupt", lambda: data._decode(content, spec, "synthetic"))

    def test_decompression_is_bounded(self):
        # A small gzip expands far beyond its declared allowance. This is a
        # synthetic malformed fixture, not a modified official source file.
        content, spec = synthetic_image(bytes(1024 * 1024), raw_bytes=16)
        self.assert_dataset_error("dataset_corrupt", lambda: data._decode(content, spec, "synthetic"))

    def test_label_values_and_counts_are_checked(self):
        raw = struct.pack(">II", 2049, 10) + bytes(10)
        content = gzip.compress(raw, mtime=0)
        spec = {"compressed_bytes": len(content), "raw_bytes": len(raw),
                "sha256": hashlib.sha256(content).hexdigest(), "count": 10, "kind": "labels"}
        self.assert_dataset_error("dataset_corrupt", lambda: data._decode(content, spec, "synthetic"))

    def test_arbitrary_download_name_is_rejected_before_network(self):
        with patch.object(data.urllib.request, "build_opener", side_effect=AssertionError("network forbidden")):
            self.assert_dataset_error("dataset_corrupt", lambda: data._download("https://example.com/data", 1))

    def test_redirect_to_another_source_is_rejected(self):
        self.assert_dataset_error("dataset_corrupt", lambda: data._NoRedirect().redirect_request(
            None, None, 302, "Found", {}, "https://example.com/data"))

    def test_download_byte_budget_is_enforced_without_network(self):
        name = next(iter(data.FILES))
        url = data.RAW_BASE + "data/fashion/" + name

        class SyntheticResponse(io.BytesIO):
            status = 200
            headers = {}

            def geturl(self):
                return url

        opener = SimpleNamespace(open=lambda *args, **kwargs: SyntheticResponse(b"12"))
        with patch.object(data.urllib.request, "build_opener", return_value=opener):
            self.assert_dataset_error("dataset_corrupt", lambda: data._download(name, 1))

    def test_atomic_output_refuses_existing_symlink(self):
        self.cache.mkdir(parents=True)
        target = self.home / "original"
        target.write_bytes(b"original")
        (self.cache / "metadata.json").symlink_to(target)
        with data._cache_directory(self.runtime) as directory:
            self.assert_dataset_error("dataset_corrupt", lambda: data._write(directory, "metadata.json", b"new"))
        self.assertEqual(target.read_bytes(), b"original")


class OptionalOfficialCacheTests(unittest.TestCase):
    @unittest.skipUnless(HAS_NUMPY and HAS_OFFICIAL_CACHE, "requires optional NumPy and the official local cache")
    def test_full_source_is_reverified_offline(self):
        """Actual pinned source, separate from all synthetic parser fixtures."""
        runtime = SimpleNamespace(home=ROOT / ".runtime")
        with patch.object(data.urllib.request, "build_opener", side_effect=AssertionError("network forbidden")):
            train_x, train_y, test_x, test_y = data.load_dataset(runtime)
        self.assertEqual([array.shape for array in (train_x, train_y, test_x, test_y)],
                         [(60000, 784), (60000,), (10000, 784), (10000,)])
        self.assertTrue(all(str(array.dtype) == "uint8" for array in (train_x, train_y, test_x, test_y)))
        self.assertEqual([int((train_y == label).sum()) for label in range(10)], [6000] * 10)
        self.assertEqual([int((test_y == label).sum()) for label in range(10)], [1000] * 10)


if __name__ == "__main__":
    unittest.main()
