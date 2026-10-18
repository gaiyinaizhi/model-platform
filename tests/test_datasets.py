"""Synthetic, offline validation, HTTP and transactional dataset acceptance."""
import concurrent.futures
import contextlib
import copy
import hashlib
import http.client
import importlib.util
import io
import json
from pathlib import Path
import sqlite3
import shutil
import stat
import struct
import tempfile
import threading
import unittest
from unittest import mock
import zipfile
import zlib

from model_platform.api import Auth, make_handler
from model_platform.common import Runtime, Problem
from model_platform.dataset_validation import Validation
from model_platform.datasets import DatasetStore
from model_platform.http_support import BoundedHTTPServer
from model_platform.store import Store

OPTIONAL = importlib.util.find_spec('PIL') is not None and importlib.util.find_spec('yaml') is not None
META = {'name': 'Tiny detection', 'rights_confirmed': True, 'negative_policy': 'reject'}


def png(color=(255, 0, 0), width=8, height=8):
    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))
    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0))
            + chunk(b'IDAT', zlib.compress((b'\0' + bytes(color) * width) * height)) + chunk(b'IEND', b''))


def fixture():
    return {'data.yaml': b'path: .\ntrain: images/train\nval: images/val\nnc: 2\nnames: [cat, dog]\n',
            'images/train/a.png': png(), 'labels/train/a.txt': b'0 0.5 0.5 0.5 0.5\n',
            'images/val/b.png': png((0, 0, 255)), 'labels/val/b.txt': b'1 0.5 0.5 1 1\n'}


def archive(files=None, compression=zipfile.ZIP_STORED, prefix=''):
    output = io.BytesIO()
    with zipfile.ZipFile(output, 'w', compression=compression) as zipout:
        for name, content in (fixture() if files is None else files).items():
            # Replays compare ZIP bytes, so fixture timestamps must not depend on
            # crossing a two-second DOS timestamp boundary during a test.
            info = zipfile.ZipInfo(prefix + name, date_time=(1980, 1, 1, 0, 0, 0))
            zipout.writestr(info, content, compress_type=compression)
    return output.getvalue()


class ArchiveFixtureTests(unittest.TestCase):
    def test_archive_bytes_and_compression_are_deterministic(self):
        for compression in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
            raw = archive(compression=compression)
            self.assertEqual(raw, archive(compression=compression))
            with zipfile.ZipFile(io.BytesIO(raw)) as zipped:
                for info in zipped.infolist():
                    self.assertEqual(info.date_time, (1980, 1, 1, 0, 0, 0))
                    self.assertEqual(info.compress_type, compression)


@unittest.skipUnless(OPTIONAL, 'Optional Pillow/PyYAML dataset dependencies are not installed')
class ValidationTests(unittest.TestCase):
    def report(self, files=None, policy='reject', raw=None):
        return Validation(policy).validate(raw if raw is not None else archive(files))[0]

    def codes(self, report):
        return {x['code'] for x in report['errors']}

    def test_valid_manifest_counts_hashes_and_single_wrapper(self):
        report, manifest, files = Validation('reject').validate(archive())
        self.assertTrue(report['valid'], report)
        self.assertEqual(report['counts'], {'images': 2, 'labels': 2, 'boxes': 2, 'negative_images': 0})
        self.assertEqual(report['classes'][0], {'id': 0, 'name': 'cat', 'boxes': 1, 'images': 1})
        self.assertEqual(manifest['samples'][0]['width'], 8)
        self.assertEqual(len(manifest['files']), 5)
        self.assertEqual(self.report(raw=archive(prefix='dataset/'))['content_sha256'], report['content_sha256'])
        self.assertEqual(self.report(raw=archive(compression=zipfile.ZIP_DEFLATED))['content_sha256'], report['content_sha256'])
        for item in manifest['files']:
            self.assertEqual(item['sha256'], hashlib.sha256(files[item['path']]).hexdigest())

    def test_optional_test_split_and_contiguous_name_mapping(self):
        files = fixture()
        files['data.yaml'] = b'train: images/train\nval: images/val\ntest: images/test\nnames:\n  0: cat\n  1: dog\n'
        files['images/test/c.png'] = png((0, 255, 0))
        files['labels/test/c.txt'] = b'0 0.5 0.5 0.2 0.2\n'
        self.assertTrue(self.report(files)['valid'])

    def test_missing_yaml_and_invalid_zip(self):
        files = fixture(); del files['data.yaml']
        self.assertIn('data_yaml', self.codes(self.report(files)))
        self.assertIn('invalid_zip', self.codes(self.report(raw=b'no zip')))
        self.assertIn('invalid_zip', self.codes(self.report(raw=archive()[:-16])))

    def test_corrupt_deflate_has_readable_report(self):
        raw = bytearray(archive(compression=zipfile.ZIP_DEFLATED))
        offset = 30 + int.from_bytes(raw[26:28], 'little') + int.from_bytes(raw[28:30], 'little')
        raw[offset:offset + 4] = b'\xff' * 4
        self.assertIn('invalid_zip', self.codes(self.report(raw=bytes(raw))))

    def test_unsafe_paths_and_special_entries(self):
        for name in ('../x', '/x', 'C:/x', 'foo\\bar', './x', 'a//b', 'a/../b', 'a\nx'):
            with self.subTest(name=name):
                self.assertIn('unsafe_path', self.codes(self.report({**fixture(), name: b'x'})))
        for mode in (stat.S_IFLNK, stat.S_IFIFO, stat.S_IFCHR):
            raw = io.BytesIO()
            with zipfile.ZipFile(raw, 'w') as z:
                info = zipfile.ZipInfo('bad'); info.create_system = 3; info.external_attr = (mode | 0o644) << 16
                z.writestr(info, b'../../outside')
            self.assertIn('unsafe_entry', self.codes(self.report(raw=raw.getvalue())))

    def test_unicode_filename_byte_limit_and_short_names(self):
        files = fixture()
        for kind, extension in (('images', '.png'), ('labels', '.txt')):
            old = f'{kind}/train/a{extension}'
            files[f'{kind}/train/' + '猫' * 100 + extension] = files.pop(old)
        self.assertIn('unsafe_path', self.codes(self.report(files)))
        files = fixture()
        for kind, extension in (('images', '.png'), ('labels', '.txt')):
            old = f'{kind}/train/a{extension}'
            files[f'{kind}/train/猫{extension}'] = files.pop(old)
        self.assertTrue(self.report(files)['valid'])

    def test_duplicate_case_collision_and_parent_conflict(self):
        for paths in (('a', 'A'), ('a', 'a'), ('a', 'a/b')):
            raw = io.BytesIO()
            with zipfile.ZipFile(raw, 'w') as z:
                for p in paths:
                    with __import__('warnings').catch_warnings():
                        __import__('warnings').simplefilter('ignore', UserWarning)
                        z.writestr(p, b'x')
            self.assertTrue(self.codes(self.report(raw=raw.getvalue())) & {'duplicate_path', 'path_conflict'})

    def test_bounded_files_entries_expansion_and_ratio(self):
        for constant, limit, expected in [('MAX_FILE_BYTES', 50, 'file_size'), ('MAX_UNPACKED_BYTES', 50, 'archive_size'), ('MAX_ENTRIES', 1, 'archive_entries')]:
            with mock.patch('model_platform.dataset_validation.' + constant, limit):
                self.assertIn(expected, self.codes(self.report()))
        self.assertIn('compression_ratio', self.codes(self.report(raw=archive({'z': b'0' * 100000}, compression=zipfile.ZIP_DEFLATED))))

    def test_yaml_tags_aliases_duplicates_urls_and_bad_names(self):
        examples = [b'!!python/object/apply:os.system [echo bad]', b'a: &a [*a]',
                    b'names: [cat]\nnames: [dog]', b'names: [cat]\ntrain: https://example.com/x\nval: images/val',
                    b'names: [cat]\ntrain: ../images/train\nval: images/val',
                    b'names: [cat]\ntrain: images/train\nval: images/val\ndownload: echo bad',
                    b'names: {1: cat}\ntrain: images/train\nval: images/val',
                    b'names: [cat, cat]\ntrain: images/train\nval: images/val',
                    b'names: [cat]\nnc: 2\ntrain: images/train\nval: images/val',
                    b'names: [true]\ntrain: images/train\nval: images/val',
                    b'names: [cat]\npath: /tmp\ntrain: images/train\nval: images/val']
        for config in examples:
            with self.subTest(config=config):
                self.assertIn('data_yaml', self.codes(self.report({**fixture(), 'data.yaml': config})))

    def test_invalid_box_class_and_per_file_lines(self):
        labels = ['0 nan .5 .2 .2', '0 .5 inf .2 .2', '0 1 .5 .5 .5', '0 .5 .5 0 .5',
                  '0 .5 .5 -1 .5', '0 .5 .5 2 .5', '0 .5 .5 .5 .5 9', '-1 .5 .5 .5 .5',
                  '9 .5 .5 .5 .5', '0.0 .5 .5 .5 .5']
        files = fixture(); files['labels/train/a.txt'] = '\n'.join(labels).encode()
        errors = self.report(files)['errors']
        self.assertEqual(len(errors), 10)
        self.assertEqual([e['line'] for e in errors], list(range(1, 11)))
        self.assertTrue(all(e['path'] == 'labels/train/a.txt' for e in errors))

    def test_explicit_negatives_missing_orphan_labels(self):
        files = fixture(); files['labels/train/a.txt'] = b''
        self.assertIn('negative_not_allowed', self.codes(self.report(files)))
        allowed = self.report(files, policy='allow_empty')
        self.assertTrue(allowed['valid']); self.assertEqual(allowed['counts']['negative_images'], 1)
        del files['labels/train/a.txt']
        self.assertIn('missing_label', self.codes(self.report(files, policy='allow_empty')))
        files['labels/train/absent.txt'] = b''
        self.assertIn('orphan_label', self.codes(self.report(files)))

    def test_cross_split_duplicate_bytes_and_pixels(self):
        files = fixture(); files['images/val/b.png'] = files['images/train/a.png']
        self.assertIn('split_overlap', self.codes(self.report(files)))
        # PNG encoding/comment bytes differ while decoded RGB is identical.
        files['images/val/b.png'] += b'ignored trailer'
        self.assertIn('split_overlap', self.codes(self.report(files)))

    def test_images_invalid_extension_dimensions_and_pixel_budget(self):
        files = fixture(); files['images/train/a.png'] = b'not an image'
        self.assertIn('invalid_image', self.codes(self.report(files)))
        files = fixture(); files['images/train/a.jpg'] = files.pop('images/train/a.png')
        self.assertIn('invalid_image', self.codes(self.report(files)))
        with mock.patch('model_platform.dataset_validation.MAX_PIXELS', 10):
            self.assertIn('invalid_image', self.codes(self.report()))
        with mock.patch('model_platform.dataset_validation.MAX_TOTAL_PIXELS', 100):
            self.assertIn('pixel_limit', self.codes(self.report()))

    def test_unexpected_empty_undeclared_and_ambiguous_paths(self):
        files = fixture(); files['other.py'] = b'print(1)'
        self.assertIn('unexpected_file', self.codes(self.report(files)))
        files = fixture(); del files['images/val/b.png']
        self.assertIn('empty_split', self.codes(self.report(files)))
        files = fixture(); files['images/train/a.jpg'] = files['images/train/a.png']
        self.assertIn('ambiguous_image', self.codes(self.report(files)))
        files = fixture(); files['images/test/c.png'] = png((0, 255, 0)); files['labels/test/c.txt'] = b''
        self.assertIn('undeclared_split', self.codes(self.report(files)))

    def test_error_report_is_bounded(self):
        files = fixture(); files['labels/train/a.txt'] = b'bad\n' * 1000
        report = self.report(files)
        self.assertEqual(len(report['errors']), 200); self.assertTrue(report['errors_truncated'])
        self.assertFalse(report['valid'])


class StoreFixture(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1] / '.runtime' / 'tests'
        root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=root)
        self.addCleanup(self.temp.cleanup)
        self.runtime = Runtime(self.temp.name)
        self.store = DatasetStore(self.runtime)

    def create(self, tenant='a', key='one', body=None):
        return self.store.create(tenant, key, body or META)[0]

    def validate(self, tenant='a', key='one', raw=None):
        draft = self.create(tenant, key)
        return self.store.upload(tenant, draft['id'], lambda: raw or archive())

    def assertProblem(self, code, action):
        with self.assertRaises(Problem) as caught:
            action()
        self.assertEqual(caught.exception.code, code)


class DatasetMetadataTests(StoreFixture):
    def test_create_idempotency_and_tenant(self):
        first = self.create()
        self.assertEqual(first, self.create())
        self.assertNotEqual(first['id'], self.create(tenant='b')['id'])
        self.assertProblem('idempotency_conflict', lambda: self.create(body={**META, 'name': 'Different'}))
        self.assertProblem('not_found', lambda: self.store.get_import('b', first['id']))
        self.assertProblem('dataset_not_valid', lambda: self.store.publish('a', first['id']))

    def test_metadata_validation_and_capacity(self):
        for body in ({**META, 'rights_confirmed': False}, {**META, 'name': ''}, {**META, 'name': ' x'},
                     {**META, 'negative_policy': 'implicit'}, {**META, 'tenant': 'b'}):
            self.assertProblem('invalid_input', lambda: self.create(body=body))
        with mock.patch('model_platform.datasets.MAX_IMPORTS', 0):
            self.assertProblem('dataset_capacity', self.create)

    def test_missing_dependencies_explicit_and_draft_retryable(self):
        draft = self.create()
        with mock.patch('model_platform.datasets.dependencies', side_effect=Problem(503, 'dataset_dependencies_missing', 'missing')):
            self.assertProblem('dataset_dependencies_missing', lambda: self.store.upload('a', draft['id'], lambda: archive()))
        self.assertEqual(self.store.get_import('a', draft['id'])['status'], 'awaiting_upload')


@unittest.skipUnless(OPTIONAL, 'Optional Pillow/PyYAML dataset dependencies are not installed')
class DatasetStoreTests(StoreFixture):
    def test_publish_replay_dedup_repack_and_tenant_isolation(self):
        draft = self.validate()
        version, created = self.store.publish('a', draft['id'])
        self.assertTrue(created)
        replay, created = self.store.publish('a', draft['id'])
        self.assertFalse(created); self.assertEqual(version, replay)
        second = self.validate(key='two', raw=archive(compression=zipfile.ZIP_DEFLATED))
        result, created = self.store.publish('a', second['id'])
        self.assertFalse(created); self.assertEqual(result['id'], version['id'])
        other = self.validate(tenant='b')
        other_version, created = self.store.publish('b', other['id'])
        self.assertTrue(created); self.assertNotEqual(other_version['id'], version['id'])
        self.assertEqual(len(self.store.list_versions('a')['versions']), 1)
        self.assertProblem('not_found', lambda: self.store.get_version('b', version['id']))
        self.assertProblem('not_found', lambda: self.store.publish('b', draft['id']))
        self.assertProblem('not_found', lambda: self.store.upload('b', draft['id'], lambda: archive()))

    def test_upload_replay_conflict_invalid_and_no_extraction_side_effects(self):
        draft = self.validate()
        self.assertEqual(self.store.upload('a', draft['id'], lambda: archive()), draft)
        self.assertProblem('dataset_upload_conflict', lambda: self.store.upload('a', draft['id'], lambda: b'different'))
        bad = self.validate(key='bad', raw=archive({'../escape': b'evil'}))
        self.assertEqual(bad['status'], 'invalid')
        self.assertFalse((self.store.root / 'staging' / bad['id']).exists())
        self.assertProblem('dataset_not_valid', lambda: self.store.publish('a', bad['id']))
        self.assertEqual(self.store.upload('a', bad['id'], lambda: archive({'../escape': b'evil'})), bad)

    def test_source_bytes_isolated_and_published_database_immutable(self):
        raw = archive(); draft = self.validate(raw=raw)
        original_source = self.store.root / 'staging' / draft['id'] / 'images/train/a.png'
        held = original_source.open('r+b')
        self.addCleanup(held.close)
        version, _ = self.store.publish('a', draft['id'])
        held.seek(0); held.write(b'bad'); held.flush()
        self.assertEqual(self.store.get_version('a', version['id']), version)
        self.assertFalse(original_source.exists())
        with self.runtime.connection() as db:
            for sql in ('UPDATE dataset_versions SET name="changed"', 'DELETE FROM dataset_versions',
                        'UPDATE dataset_imports SET report_json="{}"'):
                with self.assertRaises(sqlite3.IntegrityError):
                    db.execute(sql)

    def test_staging_and_published_tampering_detected(self):
        for kind in ('bytes', 'extra', 'missing', 'symlink'):
            with self.subTest(kind=kind):
                draft = self.validate(key=kind)
                path = self.store.root / 'staging' / draft['id'] / 'labels/train/a.txt'
                if kind == 'bytes': path.write_bytes(b'changed')
                elif kind == 'extra': path.with_name('extra.txt').write_bytes(b'x')
                elif kind == 'missing': path.unlink()
                else:
                    path.unlink(); path.symlink_to('/tmp')
                self.assertProblem('dataset_integrity', lambda: self.store.publish('a', draft['id']))
                shutil.rmtree(self.store.root / 'staging' / draft['id'])
        draft = self.validate(key='good'); version, _ = self.store.publish('a', draft['id'])
        (self.store.root / 'versions' / version['id'] / 'labels/train/a.txt').write_bytes(b'changed')
        self.assertProblem('dataset_integrity', lambda: self.store.get_version('a', version['id']))

    def test_copy_and_rename_failure_leaves_retryable_import(self):
        draft = self.validate()
        with mock.patch('model_platform.datasets.shutil.copytree', side_effect=OSError('disk full')):
            with self.assertRaises(OSError): self.store.publish('a', draft['id'])
        self.assertEqual(self.store.get_import('a', draft['id'])['status'], 'valid')
        with mock.patch('pathlib.Path.rename', side_effect=OSError('rename failed')):
            with self.assertRaises(OSError): self.store.publish('a', draft['id'])
        self.assertEqual(list((self.store.root / 'temporary').iterdir()), [])
        self.assertEqual(list((self.store.root / 'versions').iterdir()), [])
        self.assertTrue(self.store.publish('a', draft['id'])[1])

    def test_database_publish_failure_cleans_snapshot(self):
        draft = self.validate()
        with self.runtime.connection() as db:
            db.execute("CREATE TRIGGER fail_publish BEFORE INSERT ON dataset_versions BEGIN SELECT RAISE(ABORT,'test failure'); END")
        with self.assertRaises(sqlite3.IntegrityError): self.store.publish('a', draft['id'])
        self.assertEqual(list((self.store.root / 'versions').iterdir()), [])
        self.assertEqual(self.store.get_import('a', draft['id'])['status'], 'valid')
        with self.runtime.connection() as db: db.execute('DROP TRIGGER fail_publish')
        self.assertTrue(self.store.publish('a', draft['id'])[1])

    def test_upload_write_failure_keeps_draft_retryable(self):
        draft = self.create()
        with mock.patch('pathlib.Path.rename', side_effect=OSError('rename failed')):
            with self.assertRaises(OSError): self.store.upload('a', draft['id'], archive)
        self.assertEqual(self.store.get_import('a', draft['id'])['status'], 'awaiting_upload')
        self.assertEqual(list((self.store.root / 'temporary').iterdir()), [])
        self.assertEqual(self.store.upload('a', draft['id'], archive)['status'], 'valid')

    def test_capacity_and_global_validation_slot(self):
        draft = self.create()
        with mock.patch('model_platform.datasets.MAX_DATASET_BYTES', 1):
            self.assertProblem('dataset_capacity', lambda: self.store.upload('a', draft['id'], archive))
        self.assertEqual(self.store.get_import('a', draft['id'])['status'], 'awaiting_upload')
        with self.store.operation():
            self.assertProblem('dataset_busy', lambda: self.store.upload('a', draft['id'], archive))
        self.store.upload('a', draft['id'], archive)
        with mock.patch('model_platform.datasets.MAX_VERSIONS', 0):
            self.assertProblem('dataset_capacity', lambda: self.store.publish('a', draft['id']))

    def test_concurrent_publish_never_duplicates_version(self):
        draft = self.validate()
        def publish(_):
            try: return self.store.publish('a', draft['id'])
            except Problem as p:
                self.assertEqual(p.code, 'dataset_busy')
                return None
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(publish, range(4)))
        self.assertEqual(sum(bool(r and r[1]) for r in results), 1)
        self.assertEqual(len(self.store.list_versions('a')['versions']), 1)
        self.assertFalse(self.store.publish('a', draft['id'])[1])


@unittest.skipUnless(OPTIONAL, 'Optional Pillow/PyYAML dataset dependencies are not installed')
class DatasetHttpTests(StoreFixture):
    def setUp(self):
        super().setUp()
        tokens = {'a': 'a' * 64, 'b': 'b' * 64}
        credentials = self.runtime.home / 'credentials.json'
        credentials.write_text(json.dumps(tokens))
        self.server = BoundedHTTPServer(('127.0.0.1', 0), make_handler(Store(self.runtime), None, Auth(credentials)))
        thread = threading.Thread(target=self.server.serve_forever, daemon=True); thread.start()
        self.addCleanup(self.server.server_close); self.addCleanup(self.server.shutdown)

    def request(self, path, body=None, token='a', headers=None, raw=None):
        connection = http.client.HTTPConnection(*self.server.server_address, timeout=5)
        auth = {'Authorization': 'Bearer ' + token * 64} if token else {}
        if raw is not None:
            auth['Content-Type'] = 'application/zip'
        elif body is not None:
            raw = json.dumps(body).encode(); auth['Content-Type'] = 'application/json'
        auth.update(headers or {})
        connection.request('POST' if raw is not None else 'GET', path, raw, auth)
        response = connection.getresponse()
        result = (response.status, json.loads(response.read()))
        connection.close()
        return result

    def test_full_http_flow_auth_replay_and_existing_cat_catalog_unchanged(self):
        self.assertEqual(self.request('/dataset-versions', token=None)[0], 401)
        status, draft = self.request('/dataset-imports', META, headers={'Idempotency-Key': 'demo'})
        self.assertEqual(status, 201)
        self.assertEqual(self.request('/dataset-imports', META, headers={'Idempotency-Key': 'demo'}), (200, draft))
        path = '/dataset-imports/' + draft['id']
        self.assertEqual(self.request(path, token='b')[0], 404)
        status, report = self.request(path + '/archive', raw=archive())
        self.assertEqual(status, 200); self.assertTrue(report['report']['valid'])
        status, version = self.request(path + '/publish', {})
        self.assertEqual(status, 201)
        self.assertEqual(self.request(path + '/publish', {}), (200, version))
        self.assertEqual(self.request('/dataset-versions/' + version['id']), (200, version))
        self.assertEqual(self.request('/dataset-versions/' + version['id'], token='b')[0], 404)
        self.assertEqual(len(self.request('/dataset-versions')[1]['versions']), 1)
        recipes = self.request('/recipes')[1]['recipes']
        self.assertEqual(len(recipes), 5)
        cat = next(r for r in recipes if r['id'] == 'yolox-nano-cat/v1')
        self.assertFalse(cat['quality']['production_allowed'])
        self.assertNotIn(str(self.runtime.home), json.dumps(version))

    def test_http_metadata_headers_errors_and_validation_report(self):
        self.assertEqual(self.request('/dataset-imports', META)[0], 400)
        draft = self.request('/dataset-imports', META, headers={'Idempotency-Key': 'draft'})[1]
        path = '/dataset-imports/' + draft['id']
        self.assertEqual(self.request(path + '/archive', raw=b'PK', headers={'Content-Type': 'application/json'})[0], 400)
        self.assertEqual(self.request(path + '/archive', raw=b'PK', headers={'Content-Length': str(33*1024*1024)})[0], 413)
        self.assertEqual(self.request(path + '/archive', raw=b'PK', headers={'Transfer-Encoding': 'chunked'})[0], 400)
        self.assertEqual(self.request(path + '/archive', raw=b'PK', headers={'Content-Encoding': 'gzip'})[0], 400)
        status, invalid = self.request(path + '/archive', raw=b'not zip')
        self.assertEqual(status, 200); self.assertEqual(invalid['status'], 'invalid')
        self.assertEqual(invalid['report']['errors'][0]['code'], 'invalid_zip')
        self.assertEqual(self.request(path + '/publish', {})[0], 409)


if __name__ == '__main__':
    unittest.main()
