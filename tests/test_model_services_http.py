"""Actual HTTP/ORT/child-process tests using an explicitly synthetic constant graph.

Different deployments here use the same model; separate trained acceptance proves
model-version A/B changes. Neither fixture establishes recognition quality.
"""
import unittest

from model_platform import detection
import test_dataset_serving as dataset_fixture


@unittest.skipUnless(dataset_fixture.OPTIONAL, 'Optional ONNX/dataset dependencies absent')
class ModelServicesHttpTests(unittest.TestCase):
    setUp = dataset_fixture.DatasetServingTests.setUp
    spawn = dataset_fixture.DatasetServingTests.spawn
    start_api = dataset_fixture.DatasetServingTests.start_api
    cleanup = dataset_fixture.DatasetServingTests.cleanup
    call = dataset_fixture.DatasetServingTests.call
    ready = dataset_fixture.DatasetServingTests.ready
    publish = dataset_fixture.DatasetServingTests.publish
    synthetic_result = dataset_fixture.DatasetServingTests.synthetic_result
    deploy = dataset_fixture.DatasetServingTests.deploy

    def service(self):
        a, b = self.deploy('a'), self.deploy('b')
        self.ready(a); self.ready(b)
        code, value = self.call('/services', {'deployment_id': a}, 'create')
        self.assertEqual(code, 201, value)
        self.assertNotIn('ready', value)
        return a, b, value

    def test_stable_prediction_rollback_restart_stop_and_old_replay(self):
        a, b, first = self.service()
        endpoint = first['endpoint']; sid = first['id']
        self.assertEqual(endpoint, f'/services/{sid}/predict')
        self.assertTrue(self.call('/health')[1]['model_services_supported'])
        def predict(expected, revision):
            code, result = self.call(endpoint, {'features': detection.probe_input()})
            self.assertEqual(code, 200, result)
            self.assertEqual((result['service_id'], result['service_revision'], result['deployment_id']),
                             (sid, revision, expected))
            self.assertEqual(result['model_version_id'], self.model_id)
            self.assertEqual(result['artifact_sha256'], self.artifact_hash)
        predict(a, 1)
        request = {'expected_revision': 1, 'target_deployment_id': b}
        code, second = self.call(f'/services/{sid}/switch', request, 'switch-b')
        self.assertEqual(code, 200, second); self.assertEqual(second['revision'], 2)
        predict(b, 2)
        code, third = self.call(f'/services/{sid}/switch',
                              {'expected_revision': 2, 'target_deployment_id': a}, 'rollback-a')
        self.assertEqual(code, 200, third); self.assertEqual(third['revision'], 3)
        predict(a, 3)
        self.assertEqual(self.call('/services', {'deployment_id': a}, 'create'), (200, first))
        self.assertEqual(self.call(f'/services/{sid}/switch', request, 'switch-b'), (200, second))
        self.assertEqual(self.call(f'/services/{sid}')[1]['revision'], 3)
        self.api.terminate(); self.api.wait(timeout=10); self.start_api()
        self.ready(a); self.ready(b)
        predict(a, 3)
        self.assertEqual(self.call('/deployments/'+a+'/stop', {})[0], 200)
        self.assertEqual(self.call(endpoint, {'features': detection.probe_input()})[0], 503)
        current = self.call('/services/'+sid)[1]
        self.assertFalse(current['ready']); self.assertEqual(current['active_deployment_id'], a)
        self.assertEqual(self.call(f'/services/{sid}/switch', request, 'switch-b'), (200, second))
        # Existing deployment prediction remains an independent, compatible route.
        self.assertEqual(self.call('/deployments/'+b+'/predict', {'features': detection.probe_input()})[0], 200)

    def test_http_validation_isolation_and_pagination(self):
        a, b, first = self.service(); sid = first['id']
        for body in ({}, {'deployment_id': a, 'extra': 1}, {'deployment_id': 'invalid'}):
            self.assertEqual(self.call('/services', body, 'bad-create')[0], 400)
        self.assertEqual(self.call('/services', {'deployment_id': a})[0], 400)
        self.assertEqual(self.call('/services', {'deployment_id': a}, 'other-create', tenant='other')[0], 404)
        self.assertEqual(self.call('/services/'+sid, tenant='other')[0], 404)
        for action in ('switch', 'predict'):
            self.assertEqual(self.call('/services/'+sid+'/'+action, raw=b'{bad', tenant='other')[0], 404)
        for body in ({'expected_revision': True, 'target_deployment_id': b},
                     {'expected_revision': 1, 'target_deployment_id': b, 'extra': 1}):
            self.assertEqual(self.call('/services/'+sid+'/switch', body, 'bad-switch')[0], 400)
        self.assertEqual(self.call('/services/'+sid+'/switch', {'expected_revision': 1, 'target_deployment_id': a}, 'same')[0], 409)
        self.assertEqual(self.call('/services/'+sid+'/switch', {'expected_revision': 2, 'target_deployment_id': b}, 'stale')[0], 409)
        self.assertEqual(self.call('/services', {'deployment_id': b}, 'create')[0], 409)
        self.assertEqual(self.call('/services', {'deployment_id': b}, 'create-2')[0], 201)
        status, page = self.call('/services?limit=1')
        self.assertEqual(status, 200); self.assertEqual(len(page['ids']), 1); self.assertTrue(page['next_cursor'])
        status, other = self.call('/services?cursor='+page['next_cursor'])
        self.assertEqual(status, 200); self.assertEqual(len(other['ids']), 1)
        self.assertNotEqual(page['ids'], other['ids'])
        self.assertEqual(self.call('/services?cursor='+page['next_cursor'], tenant='other')[0], 400)
        self.assertEqual(self.call('/services?limit=101')[0], 400)
        self.assertEqual(self.call('/services?limit=1&limit=1')[0], 400)
        self.assertEqual(self.call('/services/'+sid+'?limit=1')[0], 400)
