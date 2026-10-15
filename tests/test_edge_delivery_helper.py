"""Local helper transport/security tests; fixtures are not model-quality evidence."""
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
import unittest
from scripts.verify_edge_delivery import download, validate_origin

MODEL = 'model_' + 'a' * 32


class HelperTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.credentials = self.root / 'credentials.json'
        self.credentials.write_text(json.dumps({'demo': 'disposable-local-fixture'}))
        self.content = b'fixture bytes, not ONNX'
        self.manifest = {'schema_version': 'model-delivery/v1', 'model_version_id': MODEL,
                         'artifact': {'format': 'yolox-nano-dataset-onnx/v1', 'media_type': 'application/octet-stream', 'download_url': '/models/' + MODEL + '/artifact',
                                      'bytes': len(self.content), 'sha256': hashlib.sha256(self.content).hexdigest()}}
        owner = self
        self.redirect = False
        self.response_hash = None
        self.response_type = None
        self.requests = []
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args): pass
            def do_GET(self):
                owner.requests.append(self.path)
                if owner.redirect:
                    self.send_response(302)
                    self.send_header('Location', '/unexpected')
                    self.end_headers()
                    return
                data = json.dumps(owner.manifest).encode() if self.path.endswith('/delivery') else owner.content
                self.send_response(200)
                self.send_header('Content-Length', str(len(data)))
                self.send_header('Content-Type', owner.response_type or 'application/octet-stream')
                self.send_header('X-Artifact-SHA256', owner.response_hash or owner.manifest['artifact']['sha256'])
                self.end_headers()
                self.wfile.write(data)
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.api = 'http://127.0.0.1:' + str(self.server.server_port)
        self.output = self.root / 'download'

    def tearDown(self):
        self.server.shutdown(); self.server.server_close(); self.thread.join()
        self.tmp.cleanup()

    def run_download(self):
        return download(self.api, self.credentials, 'demo', MODEL, self.output)

    def test_success_and_no_overwrite(self):
        self.assertTrue(self.run_download()['verified'])
        self.assertEqual((self.output / 'model.onnx').read_bytes(), self.content)
        with self.assertRaises(FileExistsError): self.run_download()
        self.assertEqual((self.output / 'model.onnx').read_bytes(), self.content)

    def test_hash_mismatch_removes_partial(self):
        self.manifest['artifact']['sha256'] = '0' * 64
        with self.assertRaises(ValueError): self.run_download()
        self.assertFalse(self.output.exists())

    def test_size_mismatch_removes_partial(self):
        self.manifest['artifact']['bytes'] += 1
        with self.assertRaises(ValueError): self.run_download()
        self.assertFalse(self.output.exists())

    def test_wrong_model_identity_rejected(self):
        self.manifest['model_version_id'] = 'model_' + 'b' * 32
        with self.assertRaises(ValueError): self.run_download()
        self.assertFalse(self.output.exists())

    def test_untrusted_artifact_url_never_requested(self):
        self.manifest['artifact']['download_url'] = 'http://example.com/secret'
        with self.assertRaises(ValueError): self.run_download()
        self.assertEqual(self.requests, ['/models/' + MODEL + '/delivery'])

    def test_redirect_not_followed(self):
        self.redirect = True
        with self.assertRaises(ValueError): self.run_download()
        self.assertEqual(len(self.requests), 1)

    def test_response_hash_header_rejected(self):
        self.response_hash = '0' * 64
        with self.assertRaises(ValueError): self.run_download()
        self.assertFalse(self.output.exists())

    def test_response_type_rejected(self):
        self.response_type = 'text/html'
        with self.assertRaises(ValueError): self.run_download()
        self.assertFalse(self.output.exists())

    def test_unsupported_format_rejected(self):
        self.manifest['artifact']['format'] = 'arbitrary-pickle/v1'
        with self.assertRaises(ValueError): self.run_download()
        self.assertFalse(self.output.exists())

    def test_origins_rejected(self):
        for origin in ['https://127.0.0.1:8080', 'http://localhost:8080', 'http://example.com:80',
                       'http://127.0.0.1:80/path', 'http://user@127.0.0.1:80', 'http://127.0.0.1:80?secret']:
            with self.subTest(origin=origin), self.assertRaises(ValueError): validate_origin(origin)


if __name__ == '__main__': unittest.main()
