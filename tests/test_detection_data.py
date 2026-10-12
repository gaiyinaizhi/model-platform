"""Synthetic cache-integrity tests, not evidence of detection model quality."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from model_platform.algorithm import TrainingFailure
from model_platform.common import Runtime
from model_platform import detection_data

HAS_PIL = importlib.util.find_spec('PIL') is not None


@unittest.skipUnless(HAS_PIL, 'optional Pillow is not installed')
class DetectionCacheTests(unittest.TestCase):
    def setUp(self):
        from PIL import Image
        base=Path(__file__).resolve().parents[1]/'.runtime'/'tests';base.mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=base);self.addCleanup(self.temp.cleanup)
        self.runtime=Runtime(self.temp.name)
        self.root=self.runtime.home/'datasets'/detection_data.REFERENCE_DIRECTORY
        self.root.mkdir(parents=True)
        self.manifest={'splits':{}}
        index=0
        for split,count in [('train',128),('validation',32),('test',32)]:
            records=[]
            for _ in range(count):
                image=Image.new('RGB',(8,8),(index,13,42))
                path=self.root/f'{index}.png';image.save(path)
                records.append({'image_id':str(index),'image_path':path.name,'width':8,'height':8,
                    'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                    'pixel_sha256':hashlib.sha256(b'8,8:'+image.tobytes()).hexdigest(),'boxes':[]})
                index+=1
            self.manifest['splits'][split]=records
        self.write_manifest()

    def write_manifest(self):
        raw=json.dumps(self.manifest,sort_keys=True).encode()
        (self.root/'manifest.json').write_bytes(raw)
        self.hash=hashlib.sha256(raw).hexdigest()

    def load(self):
        with mock.patch.object(detection_data,'MANIFEST_SHA256',self.hash):
            return detection_data.load_dataset(self.runtime)

    def test_valid_cache_verifies_every_split(self):
        manifest,splits=self.load()
        self.assertEqual({k:len(v) for k,v in splits.items()},{'train':128,'validation':32,'test':32})

    def test_manifest_and_image_tampering_rejected(self):
        with (self.root/'manifest.json').open('ab') as stream:stream.write(b' ')
        with self.assertRaises(TrainingFailure):self.load()
        self.write_manifest()
        with (self.root/'0.png').open('ab') as stream:stream.write(b'junk')
        with self.assertRaises(TrainingFailure):self.load()

    def test_pixel_digest_dimensions_and_split_overlap_rejected(self):
        record=self.manifest['splits']['test'][0]
        for field,value in [('pixel_sha256','0'*64),('width',9),('image_id','0')]:
            original=record[field];record[field]=value;self.write_manifest()
            with self.assertRaises(TrainingFailure):self.load()
            record[field]=original
        record.update({k:self.manifest['splits']['train'][0][k] for k in ('image_path','sha256','pixel_sha256')})
        self.write_manifest()
        with self.assertRaises(TrainingFailure):self.load()

    def test_path_traversal_and_symlinks_rejected(self):
        record=self.manifest['splits']['train'][0]
        for path in ('../outside.png','/tmp/outside.png'):
            record['image_path']=path;self.write_manifest()
            with self.assertRaises(TrainingFailure):self.load()
        record['image_path']='0.png';self.write_manifest()
        original=self.root/'0.png';moved=self.root/'real.png';original.rename(moved);original.symlink_to(moved)
        with self.assertRaises(TrainingFailure):self.load()

    def test_missing_truncated_and_crowd_cache_rejected(self):
        self.manifest['splits']['train'][0]['boxes']=[{'class_id':0,'category_id':1,'iscrowd':1,'xyxy':[0,0,4,4]}]
        self.write_manifest()
        with self.assertRaises(TrainingFailure):self.load()
        self.manifest['splits']['train'].pop();self.write_manifest()
        with self.assertRaises(TrainingFailure):self.load()


if __name__=='__main__':unittest.main()
