"""Self-generated ONNX/registered-record fixtures, not learned-model quality.

Actual ORT, pycocotools, read-only supervised child and immutable publications
are exercised without private model assets or user datasets.
"""
import contextlib
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock

from model_platform.common import Runtime, Problem, digest, encode, environment
from model_platform.store import Store
from model_platform.datasets import DatasetStore
from model_platform import dataset_detection as recipe, detection, dataset_model_evaluations as trained
from model_platform import model_evaluations as independent
from model_platform.evaluation_child import EvaluationReader, EvaluationRuntime
from model_platform.evaluation_comparisons import compare
from model_platform.evaluation_reports import report
from model_platform.evaluation_images import preview
from model_platform import evaluation_process
from model_platform.training_process import ExecutionLifecycle
from scripts.make_training_fixture import fixture_bytes
import test_training_reports
import test_datasets

OPTIONAL = all(importlib.util.find_spec(name) for name in ('numpy', 'onnx', 'onnxruntime', 'cv2', 'PIL', 'yaml', 'pycocotools'))


@unittest.skipUnless(OPTIONAL, 'Optional ONNX/dataset dependencies absent')
class TrainedEvaluationTests(unittest.TestCase):
    install = test_training_reports.ReportTests.install

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.runtime = Runtime(self.tmp.name); self.store = Store(self.runtime)
        self.datasets = DatasetStore(self.runtime); self.sequence = 0
        original = self.publish(fixture_bytes(), purpose='training')
        self.original = original
        binding = recipe.bind_version(self.datasets, 'demo', original['id'])
        v = test_training_reports.fixture()
        p, e = v['provenance'], v['exported_evaluation']
        p['dataset'] = {key: binding['sha256'] if key == 'content_sha256' else binding[key]
                        for key in p['dataset']}
        e.update(dataset_version_id=original['id'], dataset_content_sha256=original['content_sha256'],
                 evaluation_sha256=binding['evaluation_sha256'])
        import numpy as np
        import onnx
        from importlib.metadata import version
        values = np.zeros((1, 525, 7), dtype=np.float32)
        values[0, 0] = [80, 80, 40, 40, 1, .1, .9]
        graph = onnx.helper.make_graph([
            onnx.helper.make_node('Constant', [], ['detections'], value=onnx.numpy_helper.from_array(values))],
            'self-generated-independent-test',
            [onnx.helper.make_tensor_value_info('images', onnx.TensorProto.FLOAT, [1, 3, 160, 160])],
            [onnx.helper.make_tensor_value_info('detections', onnx.TensorProto.FLOAT, [1, 525, 7])])
        proto = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid('', 17)], ir_version=9)
        onnx.helper.set_model_props(proto, recipe.metadata(binding['class_names'], binding))
        raw = proto.SerializeToString(); self.path = self.runtime.artifact(v['training']['attempt_id'])
        self.path.write_bytes(raw)
        p['artifact_sha256'] = e['artifact_sha256'] = hashlib.sha256(raw).hexdigest()
        p['artifact_bytes'] = len(raw)
        p['dependencies']['pycocotools'] = e['metrics']['evaluator']['version'] = version('pycocotools')
        self.install(test_training_reports.rehash(v))
        # Complete the serving metadata intentionally omitted by the standalone
        # training-report fixture. No product registration code is mocked.
        with self.runtime.connection() as db:
            for trigger in list(db.execute("SELECT name FROM sqlite_master WHERE type='trigger' AND tbl_name='models'")):
                db.execute('DROP TRIGGER ' + trigger[0])
            manifest = json.loads(db.execute('SELECT manifest_json FROM models').fetchone()[0])
            manifest.update(dimensions=160*160*3, preprocessing=recipe.metadata(binding['class_names'], binding))
            db.execute('UPDATE models SET manifest_json=?', (encode(manifest).decode(),))
        self.evaluations = independent.ModelEvaluationStore(self.store)
        self.version = self.publish(self.evaluation_zip())
        self.body = {'model_version_id': self.id, 'dataset_version_id': self.version['id'],
                     'split': 'val', 'protocol': trained.PROTOCOL}

    def publish(self, raw, purpose='evaluation', tenant='demo'):
        self.sequence += 1
        item, _ = self.datasets.create(tenant, str(self.sequence), {'name': 'Synthetic evaluation mechanics',
            'rights_confirmed': True, 'negative_policy': 'allow_empty', 'purpose': purpose})
        uploaded = self.datasets.upload(tenant, item['id'], lambda: raw)
        self.assertTrue(uploaded['report']['valid'], uploaded)
        return self.datasets.publish(tenant, item['id'])[0]

    def evaluation_zip(self, names=('red_square', 'blue_square'), no_gt=False, size=(160, 160)):
        files = {'data.yaml': ('val: images/val\nnames: ' + json.dumps(list(names)) + '\n').encode()}
        for i, label in enumerate((b'1 0.5 0.5 0.25 0.25\n', b'0 0.5 0.5 0.25 0.25\n', b'')):
            files[f'images/val/{i}.png'] = test_datasets.png((71+i, 23, 49), *size)
            files[f'labels/val/{i}.txt'] = b'' if no_gt else label
        return test_datasets.archive(files)

    def create(self, key='new', body=None):
        return self.evaluations.create('demo', key, body or self.body)[0]

    def snapshot(self):
        with self.runtime.connection() as db:
            return {table: [tuple(r) for r in db.execute('SELECT * FROM ' + table)]
                    for table in ('runs', 'attempts', 'evaluations', 'models', 'run_dataset_bindings', 'deployments', 'model_services')}

    def test_actual_read_only_child_metrics_and_immutable_registration(self):
        before = self.snapshot(); item = self.create(); row = self.evaluations.claim()
        lifecycle = ExecutionLifecycle()
        with self.runtime.lock('worker', inherited_lease=True) as lease:
            payload = evaluation_process.supervise(self.runtime, item['id'], environment(), lease,
                                                   time.monotonic()+30, lifecycle=lifecycle)
        self.assertEqual(lifecycle.proof, 'child_reaped'); self.assertNotIn('error_code', payload)
        with self.runtime.connection() as db:
            self.assertEqual(db.execute('SELECT status FROM model_evaluations').fetchone()[0], 'running')
            self.assertEqual(db.execute('SELECT count(*) FROM model_evaluation_results').fetchone()[0], 0)
        self.assertEqual(list(self.evaluations.root.iterdir()), [])
        self.evaluations.succeed(row, **payload)
        detail = self.evaluations.get('demo', item['id'])
        self.assertEqual(detail['protocol'], trained.PROTOCOL)
        self.assertEqual(detail['result']['protocol'], trained.PROTOCOL)
        self.assertEqual(detail['binding']['model']['origin'], 'trained')
        self.assertEqual(detail['binding']['model']['evaluation_id'], self.store.model('demo', self.id)['evaluation_id'])
        self.assertNotEqual(detail['binding']['training_dataset']['dataset_version_id'], detail['dataset_version_id'])
        for key in ('map_50', 'map_50_95', 'ar_100'):
            self.assertAlmostEqual(detail['result']['metrics'][key], .5)
        self.assertEqual(detail['result']['operating_point']['totals'],
                         {'tp': 1, 'fp': 2, 'fn': 1, 'precision': 1/3, 'recall': .5})
        self.assertEqual(before, self.snapshot())
        self.assertEqual(self.create()['id'], item['id'])
        self.assertEqual(self.evaluations.list('demo')['evaluations'][0]['protocol'], trained.PROTOCOL)
        # The parent test runner may copy this safe generated wire fixture for
        # frontend conformance checks; it contains no raw images or credentials.
        self.detail = detail

    def test_http_submit_poll_history_and_unsupported_boundaries(self):
        import http.client
        import threading
        from model_platform.api import make_handler
        from model_platform.http_support import BoundedHTTPServer
        class Auth:
            def authenticate(self, header):
                if header not in ('Bearer demo-test-only', 'Bearer other-test-only'):
                    raise Problem(401, 'unauthorized', 'Test-only identity required')
                return 'demo' if header == 'Bearer demo-test-only' else 'other'
        server = BoundedHTTPServer(('127.0.0.1', 0), make_handler(self.store, None, Auth()))
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        def call(method, path, body=None, tenant='demo', key=None):
            connection = http.client.HTTPConnection(*server.server_address, timeout=3)
            headers = {'Authorization': 'Bearer '+tenant+'-test-only', 'Content-Type': 'application/json'}
            if key is not None: headers['Idempotency-Key'] = key
            connection.request(method, path, None if body is None else encode(body), headers)
            response = connection.getresponse(); value = json.loads(response.read()); status = response.status
            connection.close(); return status, value
        try:
            status, item = call('POST', '/model-evaluations', self.body, key='http')
            self.assertEqual(status, 202, item)
            self.assertEqual(call('POST', '/model-evaluations', self.body, key='http')[0], 200)
            self.assertEqual(call('POST', '/model-evaluations', self.body)[0], 400)
            self.assertEqual(call('POST', '/model-evaluations', self.body, tenant='other', key='foreign')[0], 404)
            path = '/model-evaluations/'+item['id']
            self.assertEqual(call('GET', path)[1]['protocol'], trained.PROTOCOL)
            self.assertEqual(call('GET', path, tenant='other')[0], 404)
            self.assertEqual(call('GET', path, tenant='unauthorized')[0], 401)
            status, history = call('GET', '/model-evaluations?model_version_id='+self.id)
            self.assertEqual(status, 200); self.assertEqual(history['evaluations'][0]['id'], item['id'])
            self.assertEqual(call('GET', path+'/report')[1]['error']['code'], 'evaluation_not_succeeded')
            row = self.evaluations.claim(); independent.execute(self.evaluations, row, environment(), timeout=30)
            status, result = call('GET', path)
            self.assertEqual(status, 200); self.assertEqual(result['status'], 'succeeded')
        finally:
            server.shutdown(); server.server_close(); thread.join(timeout=3)

    def test_result_discriminator_and_semantic_tamper_rejected(self):
        item = self.create(); row = self.evaluations.claim()
        reader = EvaluationReader(EvaluationRuntime(self.runtime.home))
        payload = trained.measure(reader, row)
        for mutate in (lambda r: r.update(protocol=independent.PROTOCOL),
                       lambda r: r['operating_point'].update(score_threshold=.001),
                       lambda r: r['metrics']['per_class'][0].update(name='other'),
                       lambda r: r['metrics'].update(map_50=.7),
                       lambda r: r['wrong_examples']['items'][0]['false_positives'][0].update(category_id=20)):
            result = copy.deepcopy(payload['result']); mutate(result)
            with self.assertRaises(Problem) as error: self.evaluations.succeed(row, result, payload['predictions'])
            self.assertEqual(error.exception.code, 'evaluation_integrity')
        self.assertEqual(list(self.evaluations.root.iterdir()), [])
        self.evaluations.succeed(row, **payload)
        self.assertEqual(self.evaluations.get('demo', item['id'])['status'], 'succeeded')

    def test_no_ground_truth_is_null_and_full_execution_succeeds(self):
        version = self.publish(self.evaluation_zip(no_gt=True))
        item = self.create(body=dict(self.body, dataset_version_id=version['id']))
        row = self.evaluations.claim(); independent.execute(self.evaluations, row, environment(), timeout=30)
        detail = self.evaluations.get('demo', item['id'])
        self.assertEqual(detail['status'], 'succeeded', detail)
        for key in ('map_50', 'map_50_95', 'ar_100'):
            self.assertIsNone(detail['result']['metrics'][key])
        self.assertEqual(detail['result']['operating_point']['totals'],
                         {'tp': 0, 'fp': 3, 'fn': 0, 'precision': 0, 'recall': None})

    def test_owner_before_support_and_dataset_compatibility(self):
        for body in (dict(self.body, protocol=independent.PROTOCOL, dataset_version_id='dsv_'+'f'*32),
                     dict(self.body, model_version_id='model_'+'f'*32)):
            with self.assertRaises(Problem) as error: self.create(body=body)
            self.assertEqual(error.exception.status, 404)
        with self.assertRaises(Problem) as error: self.create(body=dict(self.body, protocol=independent.PROTOCOL))
        self.assertEqual(error.exception.code, 'evaluation_model_unsupported')
        for version in (self.original, self.publish(self.evaluation_zip(names=('blue_square', 'red_square'))),
                        self.publish(self.evaluation_zip(names=('renamed', 'blue_square')))):
            with self.assertRaises(Problem) as error: self.create(body=dict(self.body, dataset_version_id=version['id']))
            self.assertEqual(error.exception.code, 'evaluation_dataset_incompatible')
        with self.assertRaises(Problem) as error: self.evaluations.create('other', 'new', self.body)
        self.assertEqual(error.exception.status, 404)

    def test_replay_conflict_unsupported_and_owner_order(self):
        item = self.create()
        self.assertEqual(self.create()['id'], item['id'])
        second = self.create('second')
        with self.assertRaises(Problem) as error: self.create(body=dict(self.body, dataset_version_id=self.original['id']))
        self.assertEqual(error.exception.code, 'idempotency_conflict')
        for action, code in ((lambda: report(self.evaluations, 'demo', item['id']), 'evaluation_not_succeeded'),
                             (lambda: compare(self.evaluations, 'demo', item['id'], second['id']), 'evaluation_comparison_unsupported'),
                             (lambda: preview(self.evaluations, 'demo', item['id'], '0'), 'evaluation_preview_unsupported')):
            with self.assertRaises(Problem) as error: action()
            self.assertEqual(error.exception.code, code)
        for a, b in ((item['id'], 'mev_'+'f'*32), ('mev_'+'f'*32, item['id'])):
            with self.assertRaises(Problem) as error: compare(self.evaluations, 'demo', a, b)
            self.assertEqual(error.exception.status, 404)
        self.assertEqual(self.evaluations.list('other')['evaluations'], [])
        with self.assertRaises(Problem) as error: self.evaluations.get('other', item['id'])
        self.assertEqual(error.exception.status, 404)

    def test_no_original_training_files_required_and_overlap_not_certified(self):
        import shutil
        shutil.rmtree(self.datasets.root / 'versions' / self.original['id'])
        item = self.create()
        self.assertEqual(item['status'], 'queued')
        self.assertEqual(item['binding']['dataset']['source'], None)

    def test_artifact_and_dataset_tamper_fail_closed(self):
        item = self.create(); raw = self.path.read_bytes(); self.path.write_bytes(b'changed')
        with self.assertRaises(Problem) as error: self.evaluations.get('demo', item['id'])
        self.assertEqual(error.exception.code, 'evaluation_integrity')
        self.path.write_bytes(raw)
        path = self.datasets.root / 'versions' / self.version['id'] / self.version['manifest']['samples'][0]['image']
        path.write_bytes(b'changed')
        row = self.evaluations.claim()
        with mock.patch.object(evaluation_process, 'supervise') as child:
            independent.execute(self.evaluations, row, environment()); child.assert_not_called()
        with self.runtime.connection() as db:
            self.assertEqual(db.execute('SELECT status,error_code FROM model_evaluations').fetchone()[:],
                             ('failed', 'evaluation_integrity'))

    def test_manifest_and_cross_table_tamper_fail_closed(self):
        item = self.create()
        with self.runtime.connection() as db:
            db.execute("UPDATE models SET manifest_json=json_set(manifest_json,'$.class_names[0]','tampered')")
        with self.assertRaises(Problem) as error: self.evaluations.get('demo', item['id'])
        self.assertEqual(error.exception.code, 'evaluation_integrity')

    def test_source_hash_failure_and_interruption_close_without_result(self):
        item = self.create(); row = self.evaluations.claim()
        independent.execute(self.evaluations, row, dict(environment(), code_sha256='f'*64))
        self.assertEqual(self.evaluations.get('demo', item['id'])['error_code'], 'execution_code_changed')
        other = self.create('interrupted'); self.evaluations.claim(); self.evaluations.recover()
        self.assertEqual(self.evaluations.get('demo', other['id'])['error_code'], 'evaluation_interrupted')

    def test_post_resize_extreme_aspect_ratio_is_rejected(self):
        version = self.publish(self.evaluation_zip(size=(384, 1)))
        with self.assertRaises(Problem) as error: self.create(body=dict(self.body, dataset_version_id=version['id']))
        self.assertEqual(error.exception.code, 'evaluation_dataset_incompatible')

    def test_binding_class_limits_and_test_null_relation(self):
        value = self.create()['binding']
        for count in (1, 20):
            binding = copy.deepcopy(value)
            binding['categories'] = [{'class_id': i, 'name': 'class'+str(i), 'evaluator_internal_category_id': i+1} for i in range(count)]
            binding['category_mapping_sha256'] = digest(binding['categories'])
            trained.validate_binding(binding)
        for count in (0, 21):
            binding = copy.deepcopy(value); binding['categories'] = binding['categories'][:1]*count
            binding['category_mapping_sha256'] = digest(binding['categories'])
            with self.assertRaises(ValueError): trained.validate_binding(binding)
        binding = copy.deepcopy(value); binding['training_dataset'].update(test_sha256=None)
        with self.assertRaises(ValueError): trained.validate_binding(binding)
        binding['training_dataset'].update(evaluation_split='val', evaluation_sha256=binding['training_dataset']['validation_sha256'])
        trained.validate_binding(binding)


class ProtocolBoundaryTests(unittest.TestCase):
    def test_inclusive_score_and_strict_nms_boundary(self):
        from model_platform.detection_geometry import letterbox_transform, postprocess_predictions, class_aware_nms, box_iou
        transform = letterbox_transform(160, 160, target_width=160, target_height=160)
        found = postprocess_predictions([[80, 80, 40, 40, 1, .01]], transform,
                                       num_classes=1, score_threshold=.01, iou_threshold=.65)
        self.assertEqual(len(found), 1); self.assertEqual(found[0]['score'], .01)
        first = {'class_id': 0, 'score': .9, 'box': [0, 0, 33, 1]}
        second = {'class_id': 0, 'score': .8, 'box': [7, 0, 40, 1]}
        self.assertEqual(box_iou(first['box'], second['box']), .65)
        self.assertEqual(len(class_aware_nms([first, second], iou_threshold=.65)), 2)
        second['box'] = [6.9, 0, 39.9, 1]
        self.assertEqual(len(class_aware_nms([first, second], iou_threshold=.65)), 1)
        second['class_id'] = 1
        self.assertEqual(len(class_aware_nms([first, second], iou_threshold=.65)), 2)
        self.assertEqual(trained.protocol()['score_operator'], '>=')
        self.assertEqual(independent.protocol()['score_operator'], '>')


if __name__ == '__main__': unittest.main()
