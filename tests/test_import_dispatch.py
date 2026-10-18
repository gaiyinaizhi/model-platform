"""Pure serving dispatch boundaries; no credentials or external assets."""
import copy
import unittest
from model_platform import runtime_dispatch as dispatch, pretrained_yolox as nano

class ImportDispatchTests(unittest.TestCase):
    def manifest(self):
        return dict(origin='imported',protocol=nano.PROTOCOL,contract=nano.CONTRACT,format=nano.FORMAT,
                    dimensions=416*416*3,class_names=list(nano.CLASS_NAMES),artifact_sha256=nano.ASSET_SHA256,
                    artifact_bytes=nano.ASSET_BYTES,local_evaluation_status='not_evaluated',
                    production_allowed=False,device_certified=False,evaluation=None)
    def test_not_trainable(self):
        from model_platform.recipes import recipe_ids
        self.assertNotIn(nano.PROTOCOL,recipe_ids())
        self.assertIsInstance(dispatch.for_manifest(self.manifest()),dispatch.ImportedSpec)
    def test_forged_lineage_or_protocol(self):
        for key in ('recipe','run_id','attempt_id','evaluation_id','evaluation','protocol','format','contract','origin'):
            bad=self.manifest();bad[key]='forged'
            with self.subTest(key=key),self.assertRaises(ValueError):dispatch.for_manifest(bad)
    def test_loaded_binding(self):
        m=self.manifest();loaded=dict(m,validation={'sha256':nano.ASSET_SHA256})
        dispatch.validate_loaded_identity(m,loaded)
        for key in ('protocol','contract','format','dimensions','artifact_sha256','artifact_bytes','class_names','local_evaluation_status','production_allowed','device_certified'):
            bad=copy.deepcopy(m);bad[key]='tampered'
            with self.subTest(key=key),self.assertRaises(ValueError):dispatch.validate_loaded_identity(bad,loaded)
        loaded['validation']['sha256']='0'*64
        with self.assertRaises(ValueError):dispatch.validate_loaded_identity(m,loaded)
