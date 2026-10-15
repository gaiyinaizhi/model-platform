#!/usr/bin/env python3
"""Download and verify an authenticated loopback ONNX delivery, without printing credentials.

Transfer integrity only: this does not certify inference quality or target hardware.
The output directory must not exist. Credentials are read from an existing local file.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import urllib.error
import urllib.parse
import urllib.request

MAX_ARTIFACT_BYTES = 64 * 1024 * 1024
MAX_MANIFEST_BYTES = 2 * 1024 * 1024


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError('Redirects are not permitted')


def validate_origin(value):
    url = urllib.parse.urlsplit(value)
    if (url.scheme != 'http' or url.hostname != '127.0.0.1'
            or url.path or url.query or url.fragment or url.username or url.password
            or not url.port or not 1 <= url.port <= 65535):
        raise ValueError('API must be a plain http://127.0.0.1:port origin')
    return value


def download(api, credentials, tenant, model_id, output_dir):
    """Return sanitized transfer evidence; remove partial output on any failure."""
    validate_origin(api)
    if not re.fullmatch(r'model_[0-9a-f]{32}', model_id):
        raise ValueError('Invalid model ID')
    tokens = json.loads(Path(credentials).read_text())
    token = tokens.get(tenant)
    if not isinstance(token, str) or not token or any(ord(c) < 32 or ord(c) > 126 for c in token):
        raise ValueError('Credentials do not contain a valid token for the selected tenant')
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    def get(path):
        request = urllib.request.Request(api + path, headers={'Authorization': 'Bearer ' + token})
        return opener.open(request, timeout=60)
    with get('/models/' + model_id + '/delivery') as response:
        if response.status != 200:
            raise ValueError('Manifest response was not successful')
        raw = response.read(MAX_MANIFEST_BYTES + 1)
    if len(raw) > MAX_MANIFEST_BYTES:
        raise ValueError('Manifest exceeds local safety limit')
    manifest = json.loads(raw)
    if manifest.get('schema_version') != 'model-delivery/v1' or manifest.get('model_version_id') != model_id:
        raise ValueError('Manifest identity does not match the request')
    artifact = manifest.get('artifact', {})
    expected_path = '/models/' + model_id + '/artifact'
    size, expected_hash = artifact.get('bytes'), artifact.get('sha256')
    if (artifact.get('format') not in ('fashion-onnx/v1', 'yolox-nano-onnx/v1', 'yolox-nano-dataset-onnx/v1')
            or artifact.get('media_type') != 'application/octet-stream'
            or artifact.get('download_url') != expected_path or type(size) is not int
            or not 0 < size <= MAX_ARTIFACT_BYTES or not isinstance(expected_hash, str)
            or not re.fullmatch('[0-9a-f]{64}', expected_hash)):
        raise ValueError('Manifest artifact contract is invalid')
    output_dir = Path(output_dir)
    output_dir.mkdir(mode=0o700, parents=False, exist_ok=False)
    try:
        digest, total = hashlib.sha256(), 0
        with get(expected_path) as response, (output_dir / 'model.onnx').open('xb') as target:
            if (response.status != 200 or response.headers.get('Content-Length') != str(size)
                    or response.headers.get('X-Artifact-SHA256') != expected_hash
                    or response.headers.get_content_type() != 'application/octet-stream'):
                raise ValueError('Artifact response size is inconsistent')
            while True:
                chunk = response.read(min(64 * 1024, size + 1 - total))
                if not chunk:
                    break
                total += len(chunk)
                if total > size:
                    raise ValueError('Artifact exceeds declared size')
                digest.update(chunk)
                target.write(chunk)
        if total != size or digest.hexdigest() != expected_hash:
            raise ValueError('Artifact SHA-256 or size verification failed')
        with (output_dir / 'manifest.json').open('x') as target:
            json.dump(manifest, target, indent=2, allow_nan=False)
            target.write('\n')
    except BaseException:
        shutil.rmtree(output_dir)
        raise
    return {'model_version_id': model_id, 'sha256': expected_hash, 'bytes': size,
            'verified': True, 'scope': 'Transfer integrity only; no device or model-quality approval'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--api', default=os.environ.get('MP_API', 'http://127.0.0.1:8080'))
    parser.add_argument('--credentials', type=Path, default=Path(os.environ.get('MP_HOME', '.runtime')) / 'credentials.json')
    parser.add_argument('--tenant', default='demo')
    parser.add_argument('--model-id', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    try:
        result = download(args.api, args.credentials, args.tenant, args.model_id, args.output_dir)
    except Exception as error:
        # Never expose response bodies, credentials, or private filesystem paths.
        parser.exit(1, 'Delivery verification failed (' + type(error).__name__ + '); no verified download was produced\n')
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()
