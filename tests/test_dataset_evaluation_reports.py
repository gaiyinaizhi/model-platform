"""Portable contract fixtures plus actual self-generated ORT evaluation integration.

The fixture measures a constant ONNX graph, not a learned model's quality.
"""
import contextlib
import copy
import hashlib
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

from model_platform import evaluation_report_schema as schema
from model_platform import dataset_evaluation_report_schema as trained
from model_platform.common import Problem, encode, environment
from model_platform.evaluation_reports import report
import test_dataset_model_evaluations as actual


def rehash(value):
    p = value['provenance']
    p['category_mapping_sha256'] = schema.digest(p['categories'])
    p['protocol_sha256'] = schema.digest(p['protocol'])
    value['exported_result_sha256'] = schema.digest(value['exported_result'])
    value['report_sha256'] = schema.digest({k: v for k, v in value.items() if k != 'report_sha256'})
    return value


def fixture(count=2, mode='measured'):
    value = json.loads((Path(__file__).parent / 'fixtures/dataset-evaluation-report.json').read_bytes())
    if count == 2 and mode == 'measured': return value
    p, result = value['provenance'], value['exported_result']
    categories = [{'class_id': i, 'name': '类别'+str(i), 'evaluator_internal_category_id': i+1} for i in range(count)]
    p['categories'] = categories
    no_gt = mode == 'no_gt'; gt = 0 if no_gt else 1
    p['dataset'].update(images=1, boxes=gt)
    result['scope']['images'] = 1
    metrics = result['metrics']; operating = result['operating_point']
    metrics.update(status='no_ground_truth' if no_gt else 'evaluated', map_50=None if no_gt else 0,
                   map_50_95=None if no_gt else 0, ar_100=None if no_gt else 0)
    metrics['counts'].update(images=1, images_with_ground_truth=gt, images_without_ground_truth=1-gt,
                            images_with_predictions=0, ground_truth_boxes=gt, predictions=0,
                            categories=count, categories_with_ground_truth=gt)
    metrics['per_class'] = [{'category_id': i, 'name': c['name'], 'ground_truth_boxes': gt if i == 0 else 0,
        'predictions': 0, 'status': 'evaluated' if i == 0 and gt else 'no_ground_truth',
        **{k: 0 if i == 0 and gt else None for k in ('ap_50', 'ap_50_95', 'ar_100')}} for i,c in enumerate(categories)]
    operating['per_class'] = [{'category_id': i, 'name': c['name'], 'tp': 0, 'fp': 0,
        'fn': gt if i == 0 else 0, 'precision': None, 'recall': 0 if i == 0 and gt else None} for i,c in enumerate(categories)]
    operating['totals'] = {'tp': 0, 'fp': 0, 'fn': gt, 'precision': None, 'recall': 0 if gt else None}
    return rehash(value)


class PortableTrainedReportTests(unittest.TestCase):
    def invalid(self, value):
        with self.assertRaises(ValueError): schema.parse_report(schema.encode(rehash(value)))

    def test_discriminated_fixtures_zero_null_unicode_limits(self):
        for count in (1, 2, 20):
            for mode in ('zero', 'no_gt'):
                value = fixture(count, mode)
                self.assertEqual(schema.parse_report(schema.encode(value)), value)
                self.assertEqual(value['exported_result']['metrics']['map_50'], 0 if mode == 'zero' else None)
        value = fixture(); self.assertEqual(schema.parse_report(schema.encode(value)), value)
        self.assertEqual(value['exported_result']['operating_point']['totals']['tp'], 1)
        self.assertEqual(value['exported_result']['operating_point']['totals']['fp'], 2)
        value = fixture(1, 'zero'); value['exported_result']['metrics']['map_50'] = -0.0
        value['exported_result']['metrics']['per_class'][0]['ap_50'] = -0.0
        self.assertIn(b'"map_50":-0.0', schema.encode(schema.parse_report(schema.encode(rehash(value)))))
        value = fixture(); t = value['provenance']['training_dataset']
        t.update(test_sha256=None, evaluation_split='val', evaluation_sha256=t['validation_sha256'])
        schema.parse_report(schema.encode(rehash(value)))

    def test_unicode_name_parity(self):
        for name, valid in (('\ufeff', True), ('\u0085', False), ('\u0085x', True),
                            ('\u3000', False), ('😀'*80, True), ('😀'*81, False)):
            value = fixture(1, 'zero')
            value['provenance']['categories'][0]['name'] = name
            value['exported_result']['metrics']['per_class'][0]['name'] = name
            value['exported_result']['operating_point']['per_class'][0]['name'] = name
            with self.subTest(name=repr(name)):
                if valid: schema.parse_report(schema.encode(rehash(value)))
                else: self.invalid(value)

    def test_runtime_constant_parity(self):
        self.assertEqual(trained.protocol(), actual.trained.protocol())
        self.assertEqual(trained.MODEL_KEYS, actual.trained.MODEL_KEYS)
        self.assertEqual(trained.TRAINING_KEYS, actual.trained.TRAINING_KEYS)
        self.assertEqual(trained.WARNINGS, fixture()['exported_result']['warnings'])

    def test_unknown_fields_at_every_object_rejected(self):
        original = fixture(); paths = []
        def collect(value, path):
            if isinstance(value, dict):
                paths.append(path)
                for key, item in value.items(): collect(item, path+[key])
            elif isinstance(value, list):
                for index, item in enumerate(value): collect(item, path+[index])
        collect(original, [])
        for path in paths:
            value = copy.deepcopy(original); target = value
            for key in path: target = target[key]
            target['private_path'] = '/secret'
            with self.subTest(path=path): self.invalid(value)

    def test_rehashed_identity_protocol_and_semantic_corruption(self):
        changes = [
            lambda v: v.update(schema_version='independent-evaluation-report/v1'),
            lambda v: v['evaluation'].update(protocol='imported-coco-serving-evaluation/v1'),
            lambda v: v['evaluation'].update(model_version_id='model_'+'f'*32),
            lambda v: v['evaluation'].update(dataset_version_id='dsv_'+'f'*32),
            lambda v: v['evaluation'].update(started_at=True),
            lambda v: v['evaluation'].update(finished_at=0),
            lambda v: v['provenance']['model'].update(origin='imported'),
            lambda v: v['provenance']['model'].update(recipe='other'),
            lambda v: v['provenance']['model'].update(artifact_bytes=True),
            lambda v: v['provenance']['model'].update(artifact_bytes=8388609),
            lambda v: v['provenance']['model'].update(run_id='mev_'+'f'*32),
            lambda v: v['provenance']['training_dataset'].update(test_sha256=None),
            lambda v: v['provenance']['training_dataset'].update(evaluation_sha256='f'*64),
            lambda v: v['provenance']['training_dataset'].update(evaluation_split='train'),
            lambda v: v['provenance']['training_dataset'].update(dataset_version_id=v['evaluation']['dataset_version_id']),
            lambda v: v['provenance']['dataset'].update(source_selection_sha256='f'*64),
            lambda v: v['provenance']['categories'][0].update(class_id=True),
            lambda v: v['provenance']['categories'][0].update(class_id=0.0),
            lambda v: v['provenance']['categories'][0].update(evaluator_internal_category_id=2),
            lambda v: v['provenance']['categories'][0].update(name=v['provenance']['categories'][1]['name']),
            lambda v: v['provenance']['categories'][0].update(name='\x00private'),
            lambda v: v['provenance']['categories'][0].update(name=' '*4),
            lambda v: v['provenance']['categories'][0].update(name='x'*81),
            lambda v: v['provenance'].update(categories=[]),
            lambda v: v['provenance'].update(categories=v['provenance']['categories']*11),
            lambda v: v['provenance']['protocol'].update(input_size=416),
            lambda v: v['provenance']['protocol'].update(score_operator='>'),
            lambda v: v['provenance']['protocol'].update(nms_class_agnostic=True),
            lambda v: v['exported_result'].update(protocol='imported-coco-serving-evaluation/v1'),
            lambda v: v['exported_result']['metrics']['counts'].update(categories=80),
            lambda v: v['exported_result']['metrics']['counts'].update(predictions=4),
            lambda v: v['exported_result']['metrics']['per_class'][0].update(category_id=0.0),
            lambda v: v['exported_result']['metrics'].update(map_50=.9),
            lambda v: v['exported_result']['operating_point'].update(score_threshold=.001),
            lambda v: v['exported_result']['operating_point']['totals'].update(precision=.9),
            lambda v: v['exported_result']['environment'].update(code_sha256='f'*64),
            lambda v: v['exported_result']['scope'].update(production_allowed=True),
            lambda v: v['exported_result']['warnings'].append('/private/path'),
            lambda v: v['verification'].update(trusted_authenticity=True),
            lambda v: v['limitations'].update(cross_dataset_overlap='excluded'),
            lambda v: v['provenance']['submission_environment'].update(dependencies='{"numpy":"x","numpy":"x"}'),
        ]
        for index, change in enumerate(changes):
            value = fixture(); change(value)
            with self.subTest(index=index): self.invalid(value)
        for count in (1, 20):
            value = fixture(count, 'no_gt'); value['exported_result']['metrics']['per_class'][0]['ap_50'] = 0
            self.invalid(value)

    def test_reference_hashes_not_source_authenticity(self):
        value = fixture()
        for key in ('source_result_sha256', 'source_binding_sha256', 'source_prediction_sha256'): value[key] = 'f'*64
        self.assertEqual(schema.parse_report(schema.encode(rehash(value))), value)
        for key in ('exported_result_sha256', 'report_sha256'):
            value = fixture(); value[key] = '0'*64
            if key == 'exported_result_sha256':
                value['report_sha256'] = schema.digest({k: v for k,v in value.items() if k != 'report_sha256'})
            with self.assertRaises(ValueError): schema.parse_report(schema.encode(value))

    def test_parser_bounds_and_offline_cli_stdlib_only(self):
        raw = schema.encode(fixture())
        bad = [raw+b'\n', b'\xef\xbb\xbf'+raw, raw[:-1], b'\xff',
               raw.replace(b'"schema_version":', b'"schema_version":"x","schema_version":'),
               b'['*25+b'0'+b']'*25, b'{"x":'+b'9'*65+b'}', b'{"x":1e999}',
               b'{"x":NaN}', b'{"x":Infinity}', b'x'*(schema.MAX_REPORT_BYTES+1)]
        for content in bad:
            with self.assertRaises(ValueError): schema.parse_report(content)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'report.json'; path.write_bytes(raw)
            script=Path(__file__).resolve().parents[1]/'scripts/validate_evaluation_report.py'
            result=subprocess.run([sys.executable,'-S',str(script),str(path)], cwd=d,
                env={k:v for k,v in os.environ.items() if k!='PYTHONPATH'}, capture_output=True,text=True,timeout=10)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('Authenticity is unverified',result.stdout)
            from scripts.validate_evaluation_report import read_report
            link=Path(d)/'link';link.symlink_to(path)
            with self.assertRaises((OSError,ValueError)):read_report(link)
            with self.assertRaises((OSError,ValueError)):read_report(d)
            if hasattr(os,'mkfifo'):
                fifo=Path(d)/'fifo';os.mkfifo(fifo)
                with self.assertRaises(ValueError):read_report(fifo)
            path.write_bytes(b'{}')
            result=subprocess.run([sys.executable,'-S',str(script),str(path)],cwd=d,capture_output=True,text=True,timeout=10)
            self.assertEqual(result.returncode,1);self.assertNotIn(d,result.stderr)


@unittest.skipUnless(actual.OPTIONAL, 'Optional ONNX/dataset dependencies absent')
class TrainedReportIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.case=actual.TrainedEvaluationTests('test_actual_read_only_child_metrics_and_immutable_registration')
        self.case.setUp();self.addCleanup(self.case.doCleanups)
        self.case.test_actual_read_only_child_metrics_and_immutable_registration()
        self.evaluations=self.case.evaluations;self.item=self.case.detail['id']

    def test_actual_report_read_only_projection_and_determinism(self):
        def snapshot_all():
            with self.case.runtime.connection() as db:
                tables=[r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
                return {t:[tuple(r) for r in db.execute('SELECT * FROM '+t)] for t in tables}
        before=self.case.snapshot(); all_before=snapshot_all(); raw=report(self.evaluations,'demo',self.item)
        self.assertEqual(raw,report(self.evaluations,'demo',self.item));value=schema.parse_report(raw)
        self.assertEqual(value['source_binding_sha256'],schema.digest(self.case.detail['binding']))
        self.assertEqual(value['source_result_sha256'],self.case.detail['result_sha256'])
        self.assertEqual(value['exported_result']['operating_point']['totals'], self.case.detail['result']['operating_point']['totals'])
        for forbidden in (b'wrong_examples',b'images/val/',b'"samples"',b'"digests"',b'"label"'):
            self.assertNotIn(forbidden,raw)
        original=self.case.runtime.connection;statements=[]
        @contextlib.contextmanager
        def readonly():
            with original() as db:
                db.execute('PRAGMA query_only=ON');db.set_trace_callback(statements.append);yield db
        with mock.patch.object(self.case.runtime,'connection',readonly):
            self.assertEqual(raw,report(self.evaluations,'demo',self.item))
        self.assertTrue(statements)
        self.assertTrue(all(s.lstrip().upper().startswith(('SELECT ','BEGIN','COMMIT')) for s in statements),statements)
        self.assertEqual(before,self.case.snapshot());self.assertEqual(all_before,snapshot_all())

    def test_owner_status_and_source_integrity(self):
        def fail(tenant,item,status,code):
            with self.assertRaises(Problem) as caught:report(self.evaluations,tenant,item)
            self.assertEqual((caught.exception.status,caught.exception.code),(status,code))
        fail('other',self.item,404,'not_found');fail('demo','mev_'+'f'*32,404,'not_found')
        queued=self.case.create('pending')['id']
        for state in ('queued','running','failed'):
            if state=='running':self.evaluations.claim()
            if state=='failed':self.evaluations.fail(queued,'evaluation_interrupted')
            fail('demo',queued,409,'evaluation_not_succeeded');fail('other',queued,404,'not_found')
        raw=self.case.path.read_bytes();self.case.path.write_bytes(b'changed')
        fail('other',self.item,404,'not_found');fail('demo',self.item,409,'evaluation_integrity')
        self.case.path.write_bytes(raw)
        evidence=self.evaluations.evidence_path(self.item);evidence.write_bytes(b'[]')
        fail('demo',self.item,409,'evaluation_integrity')

    def test_no_original_images_needed_but_evaluation_and_lineage_verified(self):
        import shutil
        shutil.rmtree(self.case.datasets.root / 'versions' / self.case.original['id'])
        raw=report(self.evaluations,'demo',self.item);schema.parse_report(raw)
        path=self.case.datasets.root / 'versions' / self.case.version['id'] / self.case.version['manifest']['samples'][0]['image']
        old=path.read_bytes();path.write_bytes(b'changed')
        with self.assertRaises(Problem) as caught:report(self.evaluations,'demo',self.item)
        self.assertEqual(caught.exception.code,'evaluation_integrity');path.write_bytes(old)
        with self.case.runtime.connection() as db:
            db.execute("UPDATE models SET manifest_json=json_set(manifest_json,'$.class_names[0]','tampered')")
        with self.assertRaises(Problem) as caught:report(self.evaluations,'demo',self.item)
        self.assertEqual(caught.exception.code,'evaluation_integrity')

    def test_warning_rejection_no_silent_projection(self):
        detail=copy.deepcopy(self.case.detail);detail['result']['warnings'][0]='/private/path'
        with mock.patch.object(self.evaluations,'detail',return_value=detail):
            with self.assertRaises(Problem) as caught:report(self.evaluations,'demo',self.item)
        self.assertEqual(caught.exception.code,'evaluation_integrity')

    def test_http_download_exact_bytes_headers_owner_and_status(self):
        from model_platform.api import make_handler
        from model_platform.http_support import BoundedHTTPServer
        class Auth:
            def authenticate(self,header):
                if header not in ('Bearer demo-test-only','Bearer other-test-only'):
                    raise Problem(401,'unauthorized','Test identity required')
                return 'demo' if header=='Bearer demo-test-only' else 'other'
        server=BoundedHTTPServer(('127.0.0.1',0),make_handler(self.case.store,None,Auth()))
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        def call(path,tenant='demo',headers=None):
            connection=http.client.HTTPConnection(*server.server_address,timeout=3)
            fields={'Authorization':'Bearer '+tenant+'-test-only'};fields.update(headers or {})
            connection.request('GET',path,headers=fields);response=connection.getresponse()
            result=response.status,dict(response.getheaders()),response.read();connection.close();return result
        path='/model-evaluations/'+self.item+'/report'
        try:
            status,headers,raw=call(path);self.assertEqual(status,200)
            self.assertEqual(raw,report(self.evaluations,'demo',self.item));schema.parse_report(raw)
            self.assertEqual(headers['Content-Length'],str(len(raw)));self.assertEqual(headers['Content-Type'],'application/json')
            self.assertEqual(headers['Content-Disposition'],'attachment; filename="independent-evaluation-report.json"')
            self.assertEqual(headers['Cache-Control'],'no-store');self.assertEqual(headers['X-Content-Type-Options'],'nosniff')
            self.assertEqual(call(path,'other')[0],404);self.assertEqual(call(path,'unauthorized')[0],401)
            for bad in (path+'?',path+'?x=1',path+'/',path.replace('/report','//report'),path.replace('report','%72eport')):
                self.assertEqual(call(bad)[0],400,bad)
            for h in ('Range','If-Range','Content-Encoding','Transfer-Encoding'):
                self.assertEqual(call(path,headers={h:'x'})[0],400,h)
        finally:server.shutdown();server.server_close();thread.join(timeout=3)


if __name__=='__main__':unittest.main()
