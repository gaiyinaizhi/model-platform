"""Offline downloader boundary regressions; never contact image hosts in CI."""
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from scripts import fetch_detection as fetch
from model_platform import detection_data


class FetchDetectionTests(unittest.TestCase):
    def test_manifest_pin_matches_runtime_loader(self):
        self.assertEqual(fetch.EXPECTED_MANIFEST_SHA256,detection_data.MANIFEST_SHA256)
        self.assertEqual(fetch.canonical({'b':1,'a':'猫'}),fetch.canonical({'a':'猫','b':1}))

    def test_official_https_only_including_redirects(self):
        for url in ['http://storage.googleapis.com/openimages/x','https://evil.invalid/a',
                    'https://user@storage.googleapis.com/x','https://storage.googleapis.com:444/x',
                    'file:///tmp/image.jpg','https://open-images-dataset.s3.amazonaws.com.evil.invalid/x']:
            with self.subTest(url=url),self.assertRaises(ValueError):fetch.check_url(url)
        for url in ['https://storage.googleapis.com/openimages/x','https://open-images-dataset.s3.amazonaws.com/train/a.jpg']:
            self.assertEqual(fetch.check_url(url),url)
        with self.assertRaises(ValueError):
            fetch.SafeRedirect().redirect_request(None,None,302,'',{},'https://evil.invalid/x')

    def test_cache_path_escape_rejected(self):
        root=Path(__file__).resolve().parents[1]/'.runtime'/'tests';root.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as temp:
            base=Path(temp)
            for relative in ['../escape.jpg','/tmp/escape.jpg']:
                with self.subTest(path=relative),self.assertRaises(ValueError):fetch.local_image(base,relative)
            (base/'escape.jpg').symlink_to(base.parent/'outside.jpg')
            with self.assertRaises(ValueError):fetch.local_image(base,'escape.jpg')

    def test_manifest_tampering_and_symlink_fail_before_decode(self):
        root=Path(__file__).resolve().parents[1]/'.runtime'/'tests';root.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as temp:
            base=Path(temp);path=base/'manifest.json';path.write_text('{}')
            with self.assertRaises(ValueError):fetch.verify_cache(base)
            path.unlink();path.symlink_to(base/'elsewhere')
            with self.assertRaises(ValueError):fetch.verify_cache(base)

    def test_download_budget_rejects_header_and_stream_overflow(self):
        for header,payload in [('100',b'a'),('0',b'12345')]:
            response=io.BytesIO(payload);response.headers={'Content-Length':header}
            with mock.patch.object(fetch,'open_official',return_value=response),self.assertRaises(ValueError):
                fetch.read_bounded('https://storage.googleapis.com/x',4)
        response=io.BytesIO(b'1234');response.headers={'Content-Length':'4'}
        with mock.patch.object(fetch,'open_official',return_value=response):
            self.assertEqual(fetch.read_bounded('https://storage.googleapis.com/x',4),b'1234')


if __name__=='__main__':unittest.main()
