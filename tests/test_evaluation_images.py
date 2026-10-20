"""Synthetic preview/security fixtures. No inference, quality or asset evidence."""
import contextlib
import copy
import hashlib
import http.client
import importlib.util
import io
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

from model_platform.common import Runtime, Problem, digest, encode
from model_platform.store import Store
from model_platform.datasets import DatasetStore
from model_platform import model_imports, model_evaluations, pretrained_yolox
from model_platform.model_imports import ModelImportStore, CATALOG_ID
from model_platform.model_evaluations import ModelEvaluationStore, PROTOCOL
from model_platform import evaluation_images as previews
from model_platform.safe_files import read_file
from test_evaluation_comparisons import complete_result
from test_dataset_evaluation import evaluation_files
from test_datasets import archive

OPTIONAL = bool(importlib.util.find_spec('PIL') and importlib.util.find_spec('yaml'))


@unittest.skipUnless(OPTIONAL, 'Optional Pillow/PyYAML dependencies absent')
class PreviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        # Fixed in-memory fixture material, not created/persisted credentials.
        self.patches = []
        def patch(*args, **kwargs):
            p = mock.patch(*args, **kwargs)
            self.patches.append(p)
            return p.start()
        self.addCleanup(lambda: [p.stop() for p in reversed(self.patches)])
        patch('model_platform.store.cursor_key', return_value=b'fixed-preview-test-only')
        self.runtime = Runtime(self.temp.name)
        self.store = Store(self.runtime)
        self.datasets = DatasetStore(self.runtime)
        raw = b'Synthetic opaque bytes, never loaded as a model'
        for name in ('model_imports', 'pretrained_yolox'):
            patch('model_platform.' + name + '.ASSET_BYTES', len(raw))
            patch('model_platform.' + name + '.ASSET_SHA256', hashlib.sha256(raw).hexdigest())
        patch('model_platform.model_imports.validate_isolated', return_value={'status': 'validated'})
        imports = ModelImportStore(self.store)
        item, _ = imports.create('demo', 'fixture', {'catalog_id': CATALOG_ID, 'rights_confirmed': True})
        imports.upload('demo', item['id'], lambda: raw)
        self.model, _ = imports.register('demo', item['id'])
        self.model_id = self.model['model_version_id']
        patch('model_platform.model_evaluations.dependencies')
        patch('model_platform.model_evaluations.installed_versions', return_value={
            name: 'fixture' for name in ('numpy', 'onnx', 'onnxruntime', 'opencv', 'Pillow', 'pycocotools')})
        self.evaluations = ModelEvaluationStore(self.store)
        self.counter = 0

    def successful(self, *, negative=False, zero=False, raw=None):
        self.counter += 1
        files = evaluation_files(True, negative=negative)
        files['labels/val/000_a.txt'] = b'' if negative else b'0 0.625 0.625 0.25 0.25\n'
        if raw is not None:
            files['images/val/000_a.png'] = raw
        imported, _ = self.datasets.create('demo', 'dataset' + str(self.counter), {
            'name': 'fixture' + str(self.counter), 'rights_confirmed': True,
            'negative_policy': 'allow_empty', 'purpose': 'evaluation'})
        self.datasets.upload('demo', imported['id'], lambda: archive(files))
        dataset, _ = self.datasets.publish('demo', imported['id'])
        item, _ = self.evaluations.create('demo', 'evaluation' + str(self.counter), {
            'model_version_id': self.model_id, 'dataset_version_id': dataset['id'],
            'split': 'val', 'protocol': PROTOCOL})
        row = self.evaluations.claim()
        binding = json.loads(row['binding_json'])
        sample = binding['dataset']['samples'][0]
        width, height = sample['width'], sample['height']
        detections = [] if zero else [{'category_id': 0, 'score': .9, 'bbox': [0., 0., 2., 2.]}]
        evidence = [{'image_id': sample['image'], 'width': width, 'height': height,
                     'request_width': min(384, width), 'request_height': min(384, height),
                     'original_rgb_sha256': sample['pixel_sha256'], 'request_rgb_sha256': sample['pixel_sha256'],
                     'detections': detections}]
        result = complete_result(binding)
        m = result['metrics']
        m.update(status='no_ground_truth' if negative else 'evaluated',
                 map_50=None if negative else 0., map_50_95=None if negative else 0., ar_100=None if negative else 0.)
        m['counts'].update(images_with_ground_truth=0 if negative else 1,
                           images_without_ground_truth=1 if negative else 0,
                           images_with_predictions=0 if zero else 1, ground_truth_boxes=0 if negative else 1,
                           predictions=0 if zero else 1, categories_with_ground_truth=0 if negative else 1)
        m['per_class'][0].update(ground_truth_boxes=0 if negative else 1, predictions=0 if zero else 1,
                                 status='no_ground_truth' if negative else 'evaluated',
                                 ap_50=None if negative else 0., ap_50_95=None if negative else 0.,
                                 ar_100=None if negative else 0.)
        for counts in (result['operating_point']['totals'], result['operating_point']['per_class'][0]):
            counts.update(tp=0, fp=0 if zero else 1, fn=0 if negative else 1,
                          precision=None if zero else 0., recall=None if negative else 0.)
        wrong = [] if zero and negative else [{
            'image_id': sample['image'], 'width': width, 'height': height, 'tp': 0,
            'fp': 0 if zero else 1, 'fn': 0 if negative else 1,
            'false_positives': [] if zero else [dict(detections[0], prediction_index=0)],
            'false_negatives': [] if negative else [{'annotation_index': 0, 'category_id': 0,
                                                    'bbox': [width*.5, height*.5, width*.75, height*.75]}],
            'false_positives_truncated': False, 'false_negatives_truncated': False}]
        result['wrong_examples'] = {'total': len(wrong), 'limit': 20, 'truncated': False, 'items': wrong}
        result['digests']['prediction_sha256'] = digest(evidence)
        self.evaluations.succeed(row, result, evidence)
        self.dataset, self.sample = dataset, sample
        self.image_path = self.runtime.home / 'datasets' / 'versions' / dataset['id'] / sample['image']
        self.item_id = item['id']
        return item['id']

    def preview(self, item_id=None, index='0', tenant='demo'):
        return previews.preview(self.evaluations, tenant, item_id or self.item_id, index)

    def assert_problem(self, status, code, call):
        with self.assertRaises(Problem) as caught:
            call()
        self.assertEqual((caught.exception.status, caught.exception.code), (status, code))
        self.assertNotIn(str(self.runtime.home), caught.exception.message)

    def rewrite_result(self, change):
        value = self.evaluations.get('demo', self.item_id)['result']
        change(value)
        with self.runtime.connection() as db:
            db.execute('DROP TRIGGER IF EXISTS model_evaluation_result_update_immutable')
            db.execute('UPDATE model_evaluation_results SET result_json=?,result_sha256=? WHERE evaluation_id=?',
                       (encode(value).decode(), digest(value), self.item_id))

    def test_real_child_png_headers_read_only_and_no_ml_import(self):
        from PIL import Image
        self.successful()
        before = self.store.model('demo', self.model_id)
        original, statements = self.runtime.connection, []
        @contextlib.contextmanager
        def read_only():
            with original() as db:
                db.execute('PRAGMA query_only=ON')
                db.set_trace_callback(statements.append)
                yield db
        paths_before = {p.relative_to(self.runtime.home): p.read_bytes()
                        for p in self.runtime.home.rglob('*') if p.is_file() and p.suffix != '-shm'}
        with mock.patch.object(self.runtime, 'connection', read_only), mock.patch.dict(
                sys.modules, {name: None for name in ('torch', 'cv2', 'numpy', 'onnx', 'onnxruntime', 'pycocotools')}):
            raw = self.preview()
        with Image.open(io.BytesIO(raw)) as image:
            self.assertEqual((image.format, image.mode, image.size, image.info), ('PNG', 'RGB', (8, 8), {}))
            self.assertEqual(image.getpixel((0, 0)), (255, 0, 0))
        self.assertEqual(self.store.model('demo', self.model_id), before)
        self.assertTrue(statements)
        self.assertTrue(all(s.lstrip().upper().startswith('SELECT ') for s in statements), statements)
        self.assertEqual({p.relative_to(self.runtime.home): p.read_bytes()
                          for p in self.runtime.home.rglob('*') if p.is_file() and p.suffix != '-shm'}, paths_before)
        with original() as db:
            for table in ('runs', 'attempts', 'evaluations'):
                self.assertEqual(db.execute('SELECT count(*) FROM ' + table).fetchone()[0], 0)
        self.assertFalse((self.runtime.home / 'credentials.json').exists())

    def test_owner_before_state_and_integrity_or_filesystem(self):
        item_id = self.successful()
        self.image_path.unlink()
        with mock.patch.object(self.evaluations, 'detail', side_effect=AssertionError('must not inspect')):
            self.assert_problem(404, 'not_found', lambda: self.preview(tenant='other'))
            self.assert_problem(404, 'not_found', lambda: self.preview('mev_' + 'f'*32))
        self.assert_problem(409, 'evaluation_integrity', self.preview)
        for state in ('queued', 'running', 'failed'):
            row = dict(self.evaluations._row('demo', item_id), status=state)
            with mock.patch.object(self.evaluations, '_row', return_value=row), mock.patch.object(
                    self.evaluations, 'detail', side_effect=AssertionError('must not inspect')):
                self.assert_problem(409, 'evaluation_not_succeeded', self.preview)
        row = dict(self.evaluations._row('demo', item_id), dataset_version_id='../../outside')
        with mock.patch.object(self.evaluations, '_row', return_value=row), mock.patch.object(
                self.evaluations, 'detail', side_effect=AssertionError('must reject before any snapshot read')):
            self.assert_problem(409, 'evaluation_integrity', self.preview)

    def test_strict_indices_id_and_missing_example(self):
        self.successful()
        for value in ('', '-1', '20', '00', '01', '+0', '0.0', '０', '0/../1', 0, None):
            with self.subTest(index=value):
                self.assert_problem(400, 'invalid_input', lambda: self.preview(index=value))
        for value in ('mev_bad', self.item_id.upper(), '../../other', 'http://example.test'):
            self.assert_problem(400, 'invalid_input', lambda: self.preview(value))
        self.assert_problem(404, 'not_found', lambda: self.preview(index='1'))

    def test_no_ground_truth_false_positive_and_zero_examples(self):
        self.successful(negative=True)
        self.assertTrue(self.preview().startswith(b'\x89PNG'))
        self.successful(negative=True, zero=True)
        self.assert_problem(404, 'not_found', self.preview)

    def test_binding_result_private_evidence_and_protocol_tamper(self):
        for target in ('binding', 'result', 'prediction', 'protocol'):
            self.successful()
            with self.subTest(target=target):
                if target == 'prediction':
                    self.evaluations.evidence_path(self.item_id).write_bytes(b'[]')
                elif target == 'protocol':
                    self.rewrite_result(lambda r: r['metrics'].update(protocol='different'))
                else:
                    with self.runtime.connection() as db:
                        if target == 'binding':
                            db.execute('DROP TRIGGER IF EXISTS model_evaluation_identity_immutable')
                            db.execute('DROP TRIGGER IF EXISTS model_evaluation_terminal_immutable')
                            db.execute('UPDATE model_evaluations SET binding_sha256=? WHERE id=?', ('bad', self.item_id))
                        else:
                            db.execute('DROP TRIGGER IF EXISTS model_evaluation_result_update_immutable')
                            db.execute('UPDATE model_evaluation_results SET result_json=? WHERE evaluation_id=?', ('{}', self.item_id))
                self.assert_problem(409, 'evaluation_integrity', self.preview)

    def test_hash_consistent_wrong_membership_dimension_prediction_or_shape(self):
        changes = [lambda e: e.update(image_id='images/val/missing.png'),
                   lambda e: e.update(image_id='../../external'),
                   lambda e: e.update(width=9),
                   lambda e: e.update(fp=0),
                   lambda e: e['false_positives'][0].update(prediction_index=1),
                   lambda e: e['false_positives'][0].update(score=.8),
                   lambda e: e['false_negatives'][0].update(bbox=[0, 0, 999, 999])]
        for change in changes:
            self.successful()
            self.rewrite_result(lambda r: change(r['wrong_examples']['items'][0]))
            self.assert_problem(409, 'evaluation_integrity', self.preview)

    def test_raw_tamper_missing_final_symlink_fifo_and_oversized(self):
        for kind in ('changed', 'missing', 'symlink', 'fifo', 'oversized'):
            self.successful()
            original = self.image_path.read_bytes()
            with self.subTest(kind=kind):
                if kind == 'changed':
                    self.image_path.write_bytes(b'changed')
                elif kind == 'oversized':
                    with self.image_path.open('wb') as stream:
                        stream.truncate(previews.MAX_SOURCE_BYTES + 1)
                else:
                    self.image_path.unlink()
                    if kind == 'symlink':
                        self.image_path.symlink_to(self.runtime.home / 'platform.sqlite3')
                    elif kind == 'fifo':
                        os.mkfifo(self.image_path)
                self.assert_problem(409, 'evaluation_integrity', self.preview)
                if self.image_path.exists() or self.image_path.is_symlink():
                    self.image_path.unlink()
                self.image_path.write_bytes(original)

    def test_dataset_ancestor_and_final_evidence_symlinks_and_fifo(self):
        for ancestor in ('datasets', 'versions', 'version', 'images', 'val'):
            self.successful()
            path = {'datasets': self.runtime.home / 'datasets',
                    'versions': self.runtime.home / 'datasets/versions',
                    'version': self.image_path.parents[2],
                    'images': self.image_path.parents[1], 'val': self.image_path.parent}[ancestor]
            moved = path.with_name(path.name + '-held')
            path.rename(moved); path.symlink_to(moved, target_is_directory=True)
            try:
                self.assert_problem(409, 'evaluation_integrity', self.preview)
            finally:
                path.unlink(); moved.rename(path)
        for kind in ('symlink', 'fifo', 'parent'):
            self.successful()
            path = self.evaluations.evidence_path(self.item_id)
            if kind == 'parent':
                path = path.parent
                moved = path.with_name(path.name + '-held')
                path.rename(moved); path.symlink_to(moved, target_is_directory=True)
                try:
                    self.assert_problem(409, 'evaluation_integrity', self.preview)
                finally:
                    path.unlink(); moved.rename(path)
            else:
                original = path.read_bytes()
                path.unlink()
                if kind == 'symlink': path.symlink_to(self.image_path)
                else: os.mkfifo(path)
                self.assert_problem(409, 'evaluation_integrity', self.preview)
                path.unlink(); path.write_bytes(original)

    def test_model_ancestor_symlink_is_rejected(self):
        self.successful()
        path = self.runtime.home / 'import-artifacts'; moved = path.with_name('artifact-held')
        path.rename(moved); path.symlink_to(moved, target_is_directory=True)
        try:
            self.assert_problem(409, 'evaluation_integrity', self.preview)
        finally:
            path.unlink(); moved.rename(path)

    def test_concurrency_and_release_on_success_and_failure(self):
        self.successful()
        self.assertTrue(previews._SLOTS.acquire(False)); self.assertTrue(previews._SLOTS.acquire(False))
        try:
            self.assert_problem(429, 'preview_busy', self.preview)
            self.assert_problem(404, 'not_found', lambda: self.preview(tenant='other'))
        finally:
            previews._SLOTS.release(); previews._SLOTS.release()
        self.assertTrue(self.preview())
        with mock.patch.object(previews, '_decode', side_effect=Problem(503, 'preview_unavailable', 'Unavailable')):
            self.assert_problem(503, 'preview_unavailable', self.preview)
        self.assertTrue(previews._SLOTS.acquire(False)); self.assertTrue(previews._SLOTS.acquire(False))
        previews._SLOTS.release(); previews._SLOTS.release()

    def test_actual_timeout_kills_reaps_and_releases(self):
        self.successful()
        popen, processes = subprocess.Popen, []
        def sleepy(*args, **kwargs):
            process = popen([sys.executable, '-I', '-B', '-c', 'import time; time.sleep(30)'], **kwargs)
            processes.append(process)
            return process
        start = time.monotonic()
        with mock.patch.object(previews.subprocess, 'Popen', side_effect=sleepy), mock.patch.object(previews, 'WALL_SECONDS', .1):
            self.assert_problem(503, 'preview_timeout', self.preview)
        self.assertLess(time.monotonic() - start, 3)
        self.assertEqual(len(processes), 1); self.assertIsNotNone(processes[0].poll())
        self.assertTrue(self.preview())

    def test_actual_child_failure_and_dependency_unavailable_release(self):
        self.successful()
        popen = subprocess.Popen
        for exit_code, status, code in ((3, 503, 'preview_unavailable'), (4, 409, 'evaluation_integrity')):
            def fail(*args, **kwargs):
                return popen([sys.executable, '-I', '-B', '-c',
                              'import sys; sys.stdin.buffer.read(); raise SystemExit(' + str(exit_code) + ')'], **kwargs)
            with mock.patch.object(previews.subprocess, 'Popen', side_effect=fail):
                self.assert_problem(status, code, self.preview)
        self.assertTrue(self.preview())

    def test_parent_bounded_stdout_kills_unexpected_oversized_child(self):
        self.successful()
        popen, processes = subprocess.Popen, []
        def overrun(*args, **kwargs):
            process = popen([sys.executable, '-I', '-B', '-c',
                'import sys,time; sys.stdin.buffer.read(); sys.stdout.buffer.write(b"x"*(2*1024*1024+1)); '
                'sys.stdout.flush(); time.sleep(30)'], **kwargs)
            processes.append(process)
            return process
        with mock.patch.object(previews.subprocess, 'Popen', side_effect=overrun):
            self.assert_problem(409, 'evaluation_integrity', self.preview)
        self.assertIsNotNone(processes[0].poll())

    def test_orphan_renderer_exits_when_parent_dies(self):
        decoder_path = str(Path(previews.__file__).with_name('evaluation_image_decode.py'))
        # Exercise the actual child watchdog with a deliberately stalled render,
        # independent of Pillow timing. Only these synthetic child PIDs are used.
        child_code = (
            'import importlib.util,os,sys,time; '
            'spec=importlib.util.spec_from_file_location("decoder",' + repr(decoder_path) + '); '
            'module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); '
            'sys.argv=["decoder","8","8","' + 'a'*64 + '",sys.argv[1]]; '
            'module.render=lambda *args: time.sleep(30); raise SystemExit(module.main())')
        parent_code = (
            'import os,subprocess,sys,time; p=subprocess.Popen([sys.executable,"-I","-B","-c",' + repr(child_code) + ',str(os.getpid())],'
            'stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); '
            'p.stdin.write(b"fixture");p.stdin.close();print(p.pid,flush=True);time.sleep(30)')
        parent = subprocess.Popen([sys.executable, '-I', '-B', '-c', parent_code], stdout=subprocess.PIPE, text=True)
        child_pid = None
        try:
            child_pid = int(parent.stdout.readline().strip()); time.sleep(.2)
            parent.kill(); parent.wait(timeout=3)
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                status = Path('/proc') / str(child_pid) / 'stat'
                if not status.exists() or status.read_text().split()[2] == 'Z':
                    break
                time.sleep(.05)
            else:
                self.fail('Renderer remained alive after its parent died')
        finally:
            if parent.poll() is None: parent.kill()
            parent.wait(); parent.stdout.close()
            if child_pid is not None:
                with contextlib.suppress(ProcessLookupError): os.kill(child_pid, 9)

    def test_http_auth_strict_routes_and_fixed_binary_headers(self):
        from model_platform.api import make_handler
        from model_platform.http_support import BoundedHTTPServer
        self.successful()
        class TestAuth:
            def authenticate(self, header):
                if header not in ('Bearer demo-fixture', 'Bearer other-fixture'):
                    raise Problem(401, 'unauthorized', 'Fixture identity required')
                return 'other' if header == 'Bearer other-fixture' else 'demo'
        server = BoundedHTTPServer(('127.0.0.1', 0), make_handler(self.store, None, TestAuth()))
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        base = '/model-evaluations/' + self.item_id + '/error-images/0'
        def call(path=base, method='GET', auth='demo', extra=None, body=None):
            connection = http.client.HTTPConnection(*server.server_address, timeout=8)
            headers = {} if auth is None else {'Authorization': 'Bearer ' + auth + '-fixture'}
            headers.update(extra or {})
            connection.request(method, path, body=body, headers=headers)
            response = connection.getresponse(); raw = response.read()
            result = response.status, dict(response.getheaders()), raw
            connection.close(); return result
        try:
            status, headers, raw = call()
            self.assertEqual(status, 200); self.assertEqual(headers['Content-Type'], 'image/png')
            self.assertEqual(int(headers['Content-Length']), len(raw))
            self.assertEqual(headers['Cache-Control'], 'no-store')
            self.assertEqual(headers['X-Content-Type-Options'], 'nosniff')
            self.assertEqual(headers['Cross-Origin-Resource-Policy'], 'same-origin')
            self.assertEqual(call(auth=None)[0], 401); self.assertEqual(call(auth='other')[0], 404)
            for path in (base+'?', base+'?x=1', base+'/', base.replace('/error-images/', '//error-images/'),
                         base.replace('/error-images/', '/./error-images/'), base.replace('/0', '/00'),
                         base.replace('/0', '/20'), base.replace('/0', '/%30'), 'http://localhost'+base, '/'+base):
                with self.subTest(path=path): self.assertEqual(call(path)[0], 400)
            self.assertEqual(call(method='POST')[0], 400)
            self.assertEqual(call(extra={'Range': 'bytes=0-10'})[0], 400)
            self.assertEqual(call(extra={'If-Range': 'anything'})[0], 400)
            self.assertEqual(call(body=b'body')[0], 400)
        finally:
            server.shutdown(); server.server_close(); thread.join(timeout=3)


@unittest.skipUnless(importlib.util.find_spec('PIL'), 'Optional Pillow dependency absent')
class DecoderTests(unittest.TestCase):
    def image(self, width=8, height=8, *, mode='RGB', format='PNG', **options):
        from PIL import Image
        image = Image.new(mode, (width, height), (255, 0, 0) if mode == 'RGB' else 127)
        stream = io.BytesIO(); image.save(stream, format=format, **options)
        raw = stream.getvalue()
        # Re-decode JPEG to bind the actual input, without orientation changes.
        with Image.open(io.BytesIO(raw)) as source:
            pixels = source.convert('RGB').tobytes()
        return raw, {'width': width, 'height': height,
                     'pixel_sha256': hashlib.sha256(f'{width}x{height}:'.encode() + pixels).hexdigest()}

    def test_metadata_exif_orientation_rgb_and_floor_ratio(self):
        from PIL import Image, PngImagePlugin
        metadata = PngImagePlugin.PngInfo(); metadata.add_text('private', 'must be stripped')
        for width, height, format, options in ((2051, 1025, 'PNG', {'pnginfo': metadata}),
                                               (8, 16, 'JPEG', {'exif': Image.Exif()}),
                                               (17, 2051, 'PNG', {})):
            if 'exif' in options: options['exif'][274] = 6
            raw, sample = self.image(width, height, format=format, **options)
            result = previews._decode(raw, sample)
            scale = min(1, 1024/max(width, height))
            with Image.open(io.BytesIO(result)) as image:
                self.assertEqual(image.size, (max(1, int(width*scale)), max(1, int(height*scale))))
                self.assertEqual(image.mode, 'RGB'); self.assertEqual(image.info, {})
        raw, sample = self.image(mode='L')
        with Image.open(io.BytesIO(previews._decode(raw, sample))) as image:
            self.assertEqual(image.mode, 'RGB')

    def test_pixel_hash_dimensions_corrupt_multiframe_and_source_budget(self):
        from PIL import Image
        raw, sample = self.image()
        for content, identity in ((raw, dict(sample, pixel_sha256='f'*64)),
                                  (raw, dict(sample, width=9)), (b'broken', sample),
                                  (raw, dict(sample, width=4001, height=4000)),
                                  (b'x'*(previews.MAX_SOURCE_BYTES+1), sample)):
            with self.assertRaises((Problem, ValueError)):
                previews._decode(content, identity)
        image = Image.new('RGB', (8, 8), 'red'); second = Image.new('RGB', (8, 8), 'blue')
        stream = io.BytesIO(); image.save(stream, format='PNG', save_all=True, append_images=[second])
        with self.assertRaises(Problem): previews._decode(stream.getvalue(), sample)

    def test_actual_png_output_budget(self):
        from PIL import Image
        pixels = random.Random(11).randbytes(1024*1024*3)
        image = Image.frombytes('RGB', (1024, 1024), pixels)
        stream = io.BytesIO(); image.save(stream, format='PNG')
        sample = {'width': 1024, 'height': 1024,
                  'pixel_sha256': hashlib.sha256(b'1024x1024:' + pixels).hexdigest()}
        with self.assertRaises(Problem) as caught: previews._decode(stream.getvalue(), sample)
        self.assertEqual(caught.exception.code, 'evaluation_integrity')

    def test_png_parser_rejects_metadata_truncation_crc_and_dimension_mismatch(self):
        from PIL import PngImagePlugin
        raw, sample = self.image()
        previews.validate_png(raw, (8, 8))
        metadata = PngImagePlugin.PngInfo(); metadata.add_text('leak', 'secret-fixture')
        bad_metadata, _ = self.image(pnginfo=metadata)
        for bad, dimensions in ((bad_metadata, (8, 8)), (raw[:-1], (8, 8)),
                                (raw + b'extra', (8, 8)), (raw, (9, 8)),
                                (raw[:-6] + b'broken', (8, 8))):
            with self.assertRaises(ValueError): previews.validate_png(bad, dimensions)


class SafeSnapshotTests(unittest.TestCase):
    def test_bounded_nofollow_snapshot_does_not_use_path_reads(self):
        with tempfile.TemporaryDirectory() as home:
            path = Path(home) / 'dataset'; path.mkdir(); (path / 'file').write_bytes(b'fixture')
            manifest = {'files': [{'path': 'file', 'bytes': 7, 'sha256': hashlib.sha256(b'fixture').hexdigest()}]}
            with mock.patch.object(Path, 'read_bytes', side_effect=AssertionError('Unsafe path read')):
                DatasetStore.verify_snapshot(path, manifest, digest(manifest))
            for value in ('../file', '/file', 'dir/../file'):
                altered = copy.deepcopy(manifest); altered['files'][0]['path'] = value
                with self.assertRaises(Problem): DatasetStore.verify_snapshot(path, altered, digest(altered))
            for value in (-1, True, 8*1024*1024+1):
                altered = copy.deepcopy(manifest); altered['files'][0]['bytes'] = value
                with self.assertRaises(Problem): DatasetStore.verify_snapshot(path, altered, digest(altered))

    def test_runtime_parent_symlink_and_fifo_do_not_block(self):
        with tempfile.TemporaryDirectory() as home:
            root = Path(home); actual = root / 'actual'; actual.mkdir(); (actual / 'file').write_bytes(b'a')
            (root / 'link').symlink_to(actual, target_is_directory=True)
            with self.assertRaises(OSError): read_file(root / 'link/file', 1)
            os.mkfifo(actual / 'fifo')
            start = time.monotonic()
            with self.assertRaises(ValueError): read_file(actual / 'fifo', 1)
            self.assertLess(time.monotonic() - start, 1)
