"""Synthetic contract fixtures only; no inference or credentials are generated."""
import copy
import http.client
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest import mock

from model_platform.common import Problem
from model_platform import evaluation_report_schema as schema
from model_platform.evaluation_reports import report
from model_platform.model_evaluations import ModelEvaluationStore, protocol
from model_platform.pretrained_yolox import CLASS_NAMES, COCO_CATEGORY_IDS
import test_evaluation_comparisons as comparison


def rehash(value):
    value['exported_result_sha256'] = schema.digest(value['exported_result'])
    value['report_sha256'] = schema.digest({k: v for k, v in value.items() if k != 'report_sha256'})
    return value


class ReportTests(unittest.TestCase):
    def setUp(self):
        # Comparison fixture mocks cursor_key BEFORE Store setup.
        self.case = comparison.ComparisonTests('test_same_model_distinct_records_zero_and_read_only_sql')
        self.case.setUp()
        self.addCleanup(self.case.doCleanups)
        self.evaluations = self.case.evaluations
        self.binding = self.case.binding
        self.binding['model'].update(origin='imported', import_id='import_' + 'c'*32,
            validation_id='validation_' + 'd'*32, catalog_id='yolox-nano-coco-onnx/v1',
            catalog_revision='yolox-nano-coco-onnx/2026-10-17', artifact_bytes=3659407,
            artifact_sha256='c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d',
            manifest_sha256='e'*64)
        self.item = self.case.success('report', lambda result: result.update(warnings=list(schema.WARNINGS)))

    def value(self):
        return json.loads(report(self.evaluations, 'demo', self.item))

    def assert_invalid(self, value):
        with self.assertRaises(ValueError):
            schema.parse_report(schema.encode(rehash(value)))

    def test_determinism_projection_reference_hashes_read_only(self):
        raw = report(self.evaluations, 'demo', self.item)
        self.assertEqual(raw, report(self.evaluations, 'demo', self.item))
        value = schema.parse_report(raw)
        detail = self.evaluations.get('demo', self.item)
        self.assertEqual(value['source_result_sha256'], detail['result_sha256'])
        self.assertEqual(value['source_binding_sha256'], schema.digest(detail['binding']))
        self.assertNotEqual(value['source_result_sha256'], value['exported_result_sha256'])
        for forbidden in (b'fixture.png', b'wrong_examples', b'samples', b'"label"', b'"digests"'):
            self.assertNotIn(forbidden, raw)
        self.assertEqual(value['exported_result']['metrics']['per_class'][0]['ap_50'], .4)
        self.assertIsNone(value['exported_result']['metrics']['per_class'][1]['ap_50'])
        original = self.case.fixture.runtime.connection
        import contextlib
        statements = []
        @contextlib.contextmanager
        def read_only():
            with original() as db:
                db.execute('PRAGMA query_only=ON')
                db.set_trace_callback(statements.append)
                yield db
        with mock.patch.object(self.case.fixture.runtime, 'connection', read_only):
            self.assertEqual(raw, report(self.evaluations, 'demo', self.item))
        self.assertTrue(statements)
        self.assertTrue(all(s.lstrip().upper().startswith('SELECT ') for s in statements))

    def test_unknown_fields_at_every_object_rejected(self):
        original = self.value()
        paths = []
        def collect(value, path):
            if isinstance(value, dict):
                paths.append(path)
                for key, item in value.items(): collect(item, path + [key])
            elif isinstance(value, list):
                for index, item in enumerate(value): collect(item, path + [index])
        collect(original, [])
        for path in paths:
            value = copy.deepcopy(original)
            target = value
            for key in path: target = target[key]
            target['unexpected_private_path'] = '/secret'
            with self.subTest(path=path): self.assert_invalid(value)

    def test_rehashed_semantic_corruption(self):
        changes = [
            lambda v: v['evaluation'].update(model_version_id='model_'+'f'*32),
            lambda v: v['evaluation'].update(dataset_version_id='dsv_'+'f'*32),
            lambda v: v['evaluation'].update(finished_at=0),
            lambda v: v['evaluation'].update(started_at=True),
            lambda v: v['provenance']['model'].update(origin='trained'),
            lambda v: v['provenance']['model'].update(artifact_bytes=True),
            lambda v: v['provenance']['dataset'].update(images=2),
            lambda v: v['provenance']['categories'][0].update(class_id=True),
            lambda v: v['exported_result']['metrics']['counts'].update(predictions=5),
            lambda v: v['exported_result']['metrics']['per_class'][1].update(ap_50=0),
            lambda v: v['exported_result']['metrics'].update(map_50=.9),
            lambda v: v['exported_result']['operating_point']['totals'].update(precision=.9),
            lambda v: v['exported_result']['environment'].update(code_sha256='f'*64),
            lambda v: v['exported_result']['scope'].update(production_allowed=True),
            lambda v: v['exported_result']['warnings'].append('/private/path'),
            lambda v: v['verification'].update(trusted_authenticity=True),
            lambda v: v['limitations'].update(weight_license_verified=True),
            lambda v: v['provenance']['submission_environment'].update(dependencies='{"numpy":"x","numpy":"x"}'),
        ]
        for index, change in enumerate(changes):
            value = self.value(); change(value)
            with self.subTest(index=index): self.assert_invalid(value)

    def test_ownership_state_and_tamper(self):
        def failure(status, code, tenant, item):
            with self.assertRaises(Problem) as caught: report(self.evaluations, tenant, item)
            self.assertEqual((caught.exception.status, caught.exception.code), (status, code))
        failure(404, 'not_found', 'other', self.item)
        failure(404, 'not_found', 'demo', 'mev_'+'f'*32)
        queued = self.case.fixture.create('pending')['id']
        for state in ('queued', 'running', 'failed'):
            if state == 'running': self.evaluations.claim()
            if state == 'failed': self.evaluations.fail(queued, 'fixture')
            failure(409, 'evaluation_not_succeeded', 'demo', queued)
            failure(404, 'not_found', 'other', queued)
        self.evaluations.evidence_path(self.item).write_text('[{}]')
        failure(404, 'not_found', 'other', self.item)
        failure(409, 'evaluation_integrity', 'demo', self.item)

    def test_report_preserves_measured_zero_negative_zero_and_null(self):
        value = self.value()
        metrics = value['exported_result']['metrics']
        metrics['map_50'] = -0.0
        metrics['per_class'][0]['ap_50'] = -0.0
        raw = schema.encode(rehash(value))
        self.assertIn(b'"map_50":-0.0', raw)
        checked = schema.parse_report(raw)
        self.assertIsNone(checked['exported_result']['metrics']['per_class'][1]['ap_50'])
        self.assertEqual(checked['exported_result']['metrics']['per_class'][1]['predictions'], 0)

    def test_semantic_source_validator_remains_mandatory(self):
        with mock.patch('model_platform.evaluation_reports.validate_source', side_effect=ValueError('invalid')):
            with self.assertRaises(Problem) as caught: report(self.evaluations, 'demo', self.item)
        self.assertEqual(caught.exception.code, 'evaluation_integrity')

    def test_noncanonical_duplicate_truncated_limits_numbers(self):
        raw = report(self.evaluations, 'demo', self.item)
        bad = [raw+b'\n', b'\xef\xbb\xbf'+raw, raw[:-1], b'\xff',
               raw.replace(b'"schema_version":', b'"schema_version":"x","schema_version":'),
               b'['*25+b'0'+b']'*25, b'['+b'0,'*21000+b'0]',
               b'{"x":'+b'9'*65+b'}', b'{"x":1e999}', b'{"x":NaN}', b'{"x":Infinity}',
               b'"'+b'x'*(schema.MAX_STRING*6+3)+b'"', b'x'*(schema.MAX_REPORT_BYTES+1)]
        for index, content in enumerate(bad):
            with self.subTest(index=index), self.assertRaises(ValueError): schema.parse_report(content)
        value = self.value(); value['report_sha256'] = '0'*64
        with self.assertRaises(ValueError): schema.parse_report(schema.encode(value))
        value = self.value(); value['exported_result_sha256'] = '0'*64
        value['report_sha256'] = schema.digest({k: v for k,v in value.items() if k != 'report_sha256'})
        with self.assertRaises(ValueError): schema.parse_report(schema.encode(value))

    def test_offline_cli_stdlib_only_and_regular_file(self):
        from scripts.validate_evaluation_report import read_report
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'report.json'
            path.write_bytes(report(self.evaluations, 'demo', self.item))
            result = subprocess.run([sys.executable, '-S', '-m', 'scripts.validate_evaluation_report', str(path)],
                                    capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('Authenticity is unverified', result.stdout)
            script = Path(__file__).resolve().parents[1] / 'scripts' / 'validate_evaluation_report.py'
            clean_environment = {key: value for key, value in os.environ.items() if key != 'PYTHONPATH'}
            direct = subprocess.run([sys.executable, '-S', str(script), str(path)],
                                    cwd=directory, env=clean_environment,
                                    capture_output=True, text=True, timeout=10)
            self.assertEqual(direct.returncode, 0, direct.stderr)
            self.assertIn('Authenticity is unverified', direct.stdout)
            path.write_bytes(b'{}')
            result = subprocess.run([sys.executable, '-S', '-m', 'scripts.validate_evaluation_report', str(path)],
                                    capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 1)
            self.assertNotIn(str(path), result.stderr)
            with self.assertRaises((OSError, ValueError)): read_report(directory)
            if hasattr(os, 'mkfifo'):
                fifo = Path(directory) / 'fifo'; os.mkfifo(fifo)
                with self.assertRaises(ValueError): read_report(fifo)

    def test_http_transport_and_authentication(self):
        from model_platform.api import make_handler
        from model_platform.http_support import BoundedHTTPServer
        class Auth:
            def authenticate(self, header):
                if header != 'Bearer report-fixture': raise Problem(401, 'unauthorized', 'Fixture identity required')
                return 'demo'
        server = BoundedHTTPServer(('127.0.0.1', 0), make_handler(self.case.fixture.store, None, Auth()))
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        def call(path, method='GET', headers=None, auth=True):
            connection = http.client.HTTPConnection(*server.server_address, timeout=3)
            fields = {'Authorization': 'Bearer report-fixture'} if auth else {}
            fields.update(headers or {})
            connection.request(method, path, headers=fields)
            response = connection.getresponse(); body = response.read()
            result = response.status, dict(response.getheaders()), body
            connection.close(); return result
        path = '/model-evaluations/'+self.item+'/report'
        try:
            with mock.patch.object(ModelEvaluationStore, 'binding_for', return_value=self.binding):
                status, headers, body = call(path)
                self.assertEqual(status, 200)
                schema.parse_report(body)
                self.assertEqual(headers['Content-Length'], str(len(body)))
                self.assertEqual(headers['Content-Type'], 'application/json')
                self.assertEqual(headers['Content-Disposition'], 'attachment; filename="independent-evaluation-report.json"')
                self.assertEqual(headers['Cache-Control'], 'no-store')
                self.assertEqual(headers['X-Content-Type-Options'], 'nosniff')
                self.assertEqual(call(path, auth=False)[0], 401)
                self.assertEqual(call(path, headers={'Accept':'text/html','Accept-Encoding':'gzip'})[2], body)
                for bad in (path+'?', path+'?x=1', path+'/', path.replace('/report','//report'),
                            path.replace('report','%72eport'), 'http://localhost'+path):
                    self.assertEqual(call(bad)[0], 400, bad)
                self.assertEqual(call(path, 'POST')[0], 400)
                for header in ('Range', 'If-Range', 'Content-Encoding', 'Transfer-Encoding'):
                    self.assertEqual(call(path, headers={header:'x'})[0], 400, header)
                self.assertEqual(call(path, headers={'Content-Length':'1'})[0], 400)
        finally:
            server.shutdown(); server.server_close(); thread.join(timeout=3)


class PortableSchemaTests(unittest.TestCase):
    def test_pinned_constant_parity(self):
        self.assertEqual(schema.CLASS_NAMES, CLASS_NAMES)
        self.assertEqual(schema.COCO_CATEGORY_IDS, COCO_CATEGORY_IDS)
        self.assertEqual(schema.protocol(), protocol())
        import model_platform.evaluation_comparison_validation as original
        for name in ('_PARAMETERS', '_OPERATING', '_PACKAGES'):
            self.assertEqual(getattr(schema, name), getattr(original, name))
        import inspect
        for name in ('_metrics', '_environment', '_parameters', '_ratios', '_rate', '_fixed'):
            self.assertEqual(inspect.getsource(getattr(schema, name)), inspect.getsource(getattr(original, name)))

    def test_canonical_numbers_unicode(self):
        self.assertEqual(schema.encode({'z': None, 'a': '雪'}), b'{"a":"\\u96ea","z":null}')
        self.assertNotEqual(schema.digest(1), schema.digest(1.0))
        self.assertNotEqual(schema.digest(0.0), schema.digest(-0.0))
        self.assertNotEqual(schema.digest(None), schema.digest(0))
        for value in (float('nan'), float('inf'), -float('inf')):
            with self.assertRaises(ValueError): schema.encode(value)


if __name__ == '__main__':
    unittest.main()
