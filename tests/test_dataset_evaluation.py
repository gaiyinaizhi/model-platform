"""Synthetic evaluation-purpose dataset regressions; no model quality claims."""
import copy
import json
import sqlite3
import unittest
from unittest import mock

from model_platform import dataset_detection
from model_platform.common import Problem, digest, encode, now
from model_platform.dataset_validation import Validation, sha256, validate_source
from model_platform.datasets import DatasetStore
from model_platform.pretrained_yolox import CLASS_NAMES, COCO_CATEGORY_IDS
from model_platform.store import Store
import test_datasets


def evaluation_files(full_classes=False, negative=False):
    names = list(CLASS_NAMES) if full_classes else ['cat']
    return {'data.yaml': ('val: images/val\nnc: ' + str(len(names)) + '\nnames: '
                          + json.dumps(names) + '\n').encode(),
            'images/val/000_a.png': test_datasets.png(),
            'labels/val/000_a.txt': b'' if negative else
                (b'15 0.5 0.5 0.5 0.5\n' if full_classes else b'0 0.5 0.5 0.5 0.5\n')}


def source_fixture(files=None):
    files = evaluation_files(True) if files is None else files
    manifest = {
        'schema': 'coco-subset-source/v1',
        'source': {'name': 'COCO', 'version': 'val2017', 'annotation_sha256': 'a' * 64},
        'selection': {'policy': 'synthetic-validation-fixture/v1', 'seed': 'unit-test-only',
                      'ordered_image_ids': [139], 'excluded_counts': {'crowd': 2}},
        'categories': [{'class_id': i, 'source_category_id': category_id, 'name': name}
                       for i, (category_id, name) in enumerate(zip(COCO_CATEGORY_IDS, CLASS_NAMES))],
        'licenses': [{'license_id': 4, 'name': 'Attribution License'}],
        'images': [{'image_id': 139, 'file_name': '000_a.png', 'width': 8, 'height': 8,
                    'image_sha256': sha256(files['images/val/000_a.png']),
                    'pixel_sha256': sha256(b'8x8:' + bytes((255, 0, 0)) * 64),
                    'license_id': 4, 'annotation_ids': [71]}]}
    return {'name': 'COCO', 'version': 'val2017', 'annotation_sha256': 'a' * 64,
            'selection_sha256': digest(manifest), 'selection_policy': 'synthetic-validation-fixture/v1',
            'manifest': manifest}


def rehash(source):
    source['selection_sha256'] = digest(source['manifest'])
    return source


@unittest.skipUnless(test_datasets.OPTIONAL, 'Optional Pillow/PyYAML dependencies absent')
class EvaluationValidationTests(unittest.TestCase):
    def report(self, files=None, source=None, policy='reject'):
        return Validation(policy, 'evaluation', source).validate(
            test_datasets.archive(evaluation_files() if files is None else files))

    def codes(self, report):
        return {error['code'] for error in report['errors']}

    def test_val_only_purpose_rgb_hash_and_no_synthetic_training_split(self):
        report, manifest, _ = self.report()
        self.assertTrue(report['valid'], report)
        self.assertEqual(report['purpose'], 'evaluation')
        self.assertEqual(manifest['purpose'], 'evaluation')
        self.assertIsNone(manifest['source'])
        self.assertEqual(report['splits']['train']['images'], 0)
        self.assertEqual(report['splits']['val']['images'], 1)
        self.assertEqual(report['splits']['test']['images'], 0)
        self.assertEqual(manifest['samples'][0]['pixel_sha256'], sha256(b'8x8:' + bytes((255, 0, 0)) * 64))
        self.assertFalse(Validation('reject').validate(test_datasets.archive(evaluation_files()))[0]['valid'])
        trained = Validation('reject').validate(test_datasets.archive())[1]
        self.assertEqual(trained['purpose'], 'training')
        self.assertNotIn('pixel_sha256', trained['samples'][0])

    def test_evaluation_rejects_train_test_declarations_and_files(self):
        for split in ('train', 'test'):
            files = evaluation_files()
            files['data.yaml'] += f'{split}: images/{split}\n'.encode()
            self.assertIn('data_yaml', self.codes(self.report(files)[0]))
            for kind, suffix, value in (('images', '.png', test_datasets.png()), ('labels', '.txt', b'')):
                files = evaluation_files()
                files[f'{kind}/{split}/extra{suffix}'] = value
                self.assertIn('evaluation_split', self.codes(self.report(files)[0]))

    def test_no_empty_val_missing_labels_implicit_negatives_or_unsafe_metadata(self):
        files = evaluation_files(); del files['images/val/000_a.png']
        self.assertIn('empty_split', self.codes(self.report(files)[0]))
        files = evaluation_files(); del files['labels/val/000_a.txt']
        self.assertIn('missing_label', self.codes(self.report(files)[0]))
        files = evaluation_files(negative=True)
        self.assertIn('negative_not_allowed', self.codes(self.report(files)[0]))
        self.assertTrue(self.report(files, policy='allow_empty')[0]['valid'])
        for yaml in (b'val: https://example.com/data\nnames: [cat]',
                     b'val: images/val\nnames: [cat]\ndownload: anything',
                     b'val: images/val\nnames: &classes [cat]\nother: *classes'):
            files = evaluation_files(); files['data.yaml'] = yaml
            self.assertIn('data_yaml', self.codes(self.report(files)[0]))

    def test_evaluation_preserves_archive_image_pixel_and_box_budgets(self):
        for constant, limit, expected in (
                ('MAX_ENTRIES', 1, 'archive_entries'), ('MAX_FILE_BYTES', 20, 'file_size'),
                ('MAX_UNPACKED_BYTES', 10, 'archive_size'), ('MAX_IMAGES', 0, 'image_limit'),
                ('MAX_PIXELS', 10, 'invalid_image'), ('MAX_TOTAL_PIXELS', 10, 'pixel_limit'),
                ('MAX_BOXES', 0, 'box_limit')):
            with self.subTest(constant=constant), mock.patch('model_platform.dataset_validation.' + constant, limit):
                self.assertIn(expected, self.codes(self.report()[0]))

    def test_full_source_manifest_bound_to_real_raw_rgb_label_counts_and_order(self):
        files = evaluation_files(True); source = source_fixture(files)
        report, manifest, _ = self.report(files, source)
        self.assertTrue(report['valid'], report)
        self.assertEqual(manifest['source'], source)
        self.assertEqual(manifest['source']['manifest']['categories'][15],
                         {'class_id': 15, 'source_category_id': 17, 'name': 'cat'})
        for key, value in (('file_name', 'other.png'), ('image_sha256', '0' * 64),
                           ('pixel_sha256', '1' * 64), ('width', 9), ('annotation_ids', [71, 72])):
            changed = copy.deepcopy(source); changed['manifest']['images'][0][key] = value; rehash(changed)
            with self.subTest(key=key):
                self.assertIn('source_mismatch', self.codes(self.report(files, changed)[0]))
        self.assertIn('source_mismatch', self.codes(self.report(evaluation_files(), source)[0]))
        files['labels/val/000_a.txt'] += b'15 0.6 0.6 0.2 0.2\n'
        self.assertIn('source_mismatch', self.codes(self.report(files, source)[0]))

    def test_source_requires_all_selected_files_and_exact_preinference_order(self):
        files = evaluation_files(True); source = source_fixture(files)
        files['images/val/001_b.png'] = test_datasets.png((0, 255, 0))
        files['labels/val/001_b.txt'] = b'0 0.5 0.5 0.5 0.5\n'
        self.assertIn('source_mismatch', self.codes(self.report(files, source)[0]))
        second = copy.deepcopy(source['manifest']['images'][0])
        second.update(image_id=140, file_name='001_b.png', image_sha256=sha256(files['images/val/001_b.png']),
                      pixel_sha256=sha256(b'8x8:' + bytes((0, 255, 0)) * 64), annotation_ids=[72])
        source['manifest']['images'].append(second)
        source['manifest']['selection']['ordered_image_ids'].append(140)
        rehash(source)
        self.assertTrue(self.report(files, source)[0]['valid'])
        source['manifest']['images'].reverse(); source['manifest']['selection']['ordered_image_ids'].reverse(); rehash(source)
        self.assertIn('source_mismatch', self.codes(self.report(files, source)[0]))


class SourceContractTests(unittest.TestCase):
    def assert_invalid(self, source):
        with self.assertRaises(Problem) as caught:
            validate_source(source)
        self.assertEqual(caught.exception.code, 'invalid_input')

    def test_strict_source_fields_hash_schema_and_bounds(self):
        source = source_fixture()
        self.assertEqual(validate_source(source), source)
        for key, value in (('name', 'Different'), ('version', 'train2017'), ('selection_sha256', 'b' * 64),
                           ('annotation_sha256', 'A' * 64), ('selection_policy', 'not-the-same')):
            self.assert_invalid(dict(source, **{key: value}))
        self.assert_invalid(dict(source, url='https://example.com/arbitrary'))
        self.assert_invalid(dict(source, manifest={**source['manifest'], 'source_url': '/tmp/private'}))
        for key, value in (('schema', 'other/v1'), ('licenses', []), ('categories', []), ('images', [])):
            changed = copy.deepcopy(source); changed['manifest'][key] = value; self.assert_invalid(rehash(changed))
        for key, value in (('image_id', True), ('file_name', '../outside.png'), ('file_name', '/absolute.png'),
                           ('file_name', 'https://example.com/x.png'), ('width', 0), ('license_id', 999),
                           ('annotation_ids', [71, 71]), ('annotation_ids', [False]),
                           ('annotation_ids', [2**53]), ('pixel_sha256', 'bad')):
            changed = copy.deepcopy(source); changed['manifest']['images'][0][key] = value; self.assert_invalid(rehash(changed))
        for key, value in (('class_id', True), ('source_category_id', 16), ('name', 'Cat')):
            changed = copy.deepcopy(source); changed['manifest']['categories'][15][key] = value; self.assert_invalid(rehash(changed))
        changed = copy.deepcopy(source); changed['manifest']['selection']['seed'] = 'x' * 161
        self.assert_invalid(rehash(changed))
        changed = copy.deepcopy(source); changed['manifest']['selection']['excluded_counts'] = {str(i): 0 for i in range(33)}
        self.assert_invalid(rehash(changed))
        changed = copy.deepcopy(source); changed['manifest']['selection']['excluded_counts']['crowd'] = True
        self.assert_invalid(rehash(changed))
        with mock.patch('model_platform.dataset_validation.MAX_SOURCE_BYTES', 10):
            self.assert_invalid(source)

    def test_annotation_ids_globally_unique_not_just_per_image(self):
        source = source_fixture()
        second = dict(source['manifest']['images'][0], image_id=140, file_name='001_b.png')
        source['manifest']['images'].append(second)
        source['manifest']['selection']['ordered_image_ids'].append(140)
        self.assert_invalid(rehash(source))

    def test_canonical_source_hash_unicode_and_safe_integer_boundary(self):
        source = source_fixture()
        source['manifest']['selection']['seed'] = 'unicode-猫-🐈'
        source['manifest']['selection']['excluded_counts'] = {'🐈': 1, '\uffff': 2}
        source['manifest']['images'][0]['annotation_ids'] = [2**53 - 1]
        rehash(source)
        self.assertEqual(validate_source(source), source)
        self.assertEqual(source['selection_sha256'], digest(json.loads(encode(source['manifest']))))
        self.assertIn(b'\\ud83d\\udc08', encode(source['manifest']))
        source['manifest']['images'][0]['annotation_ids'] = [2.0]
        self.assert_invalid(rehash(source))


@unittest.skipUnless(test_datasets.OPTIONAL, 'Optional Pillow/PyYAML dependencies absent')
class EvaluationDatasetStoreTests(test_datasets.StoreFixture):
    def publish_evaluation(self, source=None, key='eval', files=None):
        body = dict(test_datasets.META, purpose='evaluation')
        if source is not None: body['source'] = source
        imported, _ = self.store.create('a', key, body)
        uploaded = self.store.upload('a', imported['id'], lambda: test_datasets.archive(
            evaluation_files(source is not None) if files is None else files))
        self.assertEqual(uploaded['purpose'], 'evaluation')
        self.assertTrue(uploaded['report']['valid'], uploaded)
        return self.store.publish('a', imported['id'])[0]

    def test_metadata_defaults_strict_inputs_and_raw_request_idempotency(self):
        created = self.create()
        self.assertEqual(created['purpose'], 'training'); self.assertIsNone(created['source'])
        with self.runtime.connection() as db:
            self.assertEqual(db.execute('SELECT request_sha256 FROM dataset_imports WHERE id=?', (created['id'],)).fetchone()[0],
                             digest(test_datasets.META))
        self.assertEqual(self.create(), created)
        self.assertProblem('idempotency_conflict', lambda: self.create(body=dict(test_datasets.META, purpose='training')))
        for body in (dict(test_datasets.META, purpose=None), dict(test_datasets.META, purpose=True),
                     dict(test_datasets.META, source=source_fixture()),
                     dict(test_datasets.META, purpose='evaluation', source=None)):
            self.assertProblem('invalid_input', lambda: self.create(key='bad', body=body))

    def test_purpose_source_immutable_copy_and_tenant_isolation(self):
        source = source_fixture(); version = self.publish_evaluation(source)
        self.assertEqual(version['source'], source)
        self.assertEqual(version['manifest']['source'], source)
        self.assertEqual(version['purpose'], 'evaluation')
        self.assertEqual(self.store.get_import('a', version['source_import_id'])['report']['source'], source)
        self.assertEqual(self.store.list_versions('a')['versions'][0]['source'],
                         {key: value for key, value in source.items() if key != 'manifest'})
        self.assertEqual(self.store.get_version('a', version['id']), version)
        source['manifest']['images'][0]['width'] = 99
        self.assertEqual(self.store.get_version('a', version['id']), version)
        self.assertProblem('not_found', lambda: self.store.get_version('b', version['id']))
        with self.runtime.connection() as db:
            for sql in ("UPDATE dataset_imports SET purpose='training'", "UPDATE dataset_imports SET source_json=NULL",
                        "UPDATE dataset_versions SET manifest_json='{}'"):
                with self.assertRaises(sqlite3.IntegrityError): db.execute(sql)

    def test_source_selection_change_requires_new_content_version(self):
        source = source_fixture(); first = self.publish_evaluation(source)
        source['manifest']['selection']['seed'] = 'another-frozen-seed'; rehash(source)
        second = self.publish_evaluation(source, key='eval-2')
        self.assertNotEqual(first['id'], second['id'])
        self.assertNotEqual(first['content_sha256'], second['content_sha256'])

    def test_maximum_record_list_uses_compact_source_below_bff_budget(self):
        source = source_fixture()
        source['manifest']['images'][0]['annotation_ids'] = list(range(100_000_000, 100_010_000))
        rehash(source); validate_source(source)
        self.assertGreater(len(encode(source)), 100_000)
        report, manifest, _ = Validation('reject', 'evaluation').validate(test_datasets.archive(evaluation_files(True)))
        manifest['source'] = source
        row = {'id': 'dsv_' + '1' * 32, 'name': 'Synthetic summary budget', 'created_at': now(),
               'content_sha256': digest(manifest), 'source_import_id': 'dsi_' + '1' * 32,
               'manifest_json': encode(manifest).decode()}
        summary = DatasetStore.version_value(row)
        detail = DatasetStore.version_value(row, True)
        self.assertNotIn('manifest', summary['source'])
        self.assertEqual(detail['source'], source)
        self.assertEqual(detail['manifest']['source'], source)
        self.assertLess(len(encode({'versions': [summary] * 100, 'limit': 100})), 8 * 1024 * 1024)
        self.assertGreater(len(encode({'versions': [detail] * 100, 'limit': 100})), 8 * 1024 * 1024)

    def test_training_rejects_evaluation_purpose_before_budget_or_enqueue(self):
        version = self.publish_evaluation()
        self.assertProblem('dataset_training_purpose', lambda: dataset_detection.bind_version(self.store, 'a', version['id']))
        training = Store(self.runtime)
        self.assertProblem('dataset_training_purpose', lambda: training.submit('a', 'no-train', dataset_detection.request_body(version['id'])))
        with self.runtime.connection() as db:
            self.assertEqual(db.execute('SELECT count(*) FROM runs').fetchone()[0], 0)
        self.assertProblem('not_found', lambda: dataset_detection.bind_version(self.store, 'b', version['id']))

    def test_source_tampering_in_staging_blocks_publication(self):
        body = dict(test_datasets.META, purpose='evaluation', source=source_fixture())
        imported, _ = self.store.create('a', 'eval', body)
        self.store.upload('a', imported['id'], lambda: test_datasets.archive(evaluation_files(True)))
        image = self.store.root / 'staging' / imported['id'] / 'images/val/000_a.png'
        image.write_bytes(test_datasets.png((0, 0, 255)))
        self.assertProblem('dataset_integrity', lambda: self.store.publish('a', imported['id']))

    def test_additive_migration_preserves_legacy_manifest_report_and_replay(self):
        # Recreate the exact old table shape and store a valid pre-feature import.
        report, manifest, files = Validation('reject').validate(test_datasets.archive())
        report.pop('purpose'); report.pop('source'); manifest.pop('purpose'); manifest.pop('source')
        report['content_sha256'] = digest(manifest)
        original_manifest, original_report = encode(manifest).decode(), encode(report).decode()
        import_id = 'dsi_' + '1' * 32
        with self.runtime.connection() as db:
            db.executescript('''DROP TABLE dataset_imports;
                CREATE TABLE dataset_imports (id TEXT PRIMARY KEY,tenant TEXT NOT NULL,idempotency_key TEXT NOT NULL,
                  request_sha256 TEXT NOT NULL,name TEXT NOT NULL,negative_policy TEXT NOT NULL,
                  created_at REAL NOT NULL,status TEXT NOT NULL,archive_sha256 TEXT,report_json TEXT,
                  manifest_json TEXT,version_id TEXT,stored_bytes INTEGER NOT NULL DEFAULT 0,
                  UNIQUE(tenant,idempotency_key));''')
            db.execute('INSERT INTO dataset_imports VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)',
                       (import_id, 'a', 'old', digest(test_datasets.META), test_datasets.META['name'], 'reject',
                        now(), 'valid', sha256(test_datasets.archive()), original_report, original_manifest,
                        None, sum(map(len, files.values()))))
        staging = self.store.root / 'staging' / import_id
        for name, data in files.items():
            destination = staging / name; destination.parent.mkdir(parents=True, exist_ok=True); destination.write_bytes(data)
        self.store = DatasetStore(self.runtime)
        old, created = self.store.create('a', 'old', test_datasets.META)
        self.assertFalse(created); self.assertEqual(old['purpose'], 'training'); self.assertEqual(old['report']['purpose'], 'training')
        self.assertEqual(self.store.upload('a', import_id, test_datasets.archive), old)
        version, created = self.store.publish('a', import_id)
        self.assertTrue(created); self.assertEqual(version['purpose'], 'training')
        self.assertNotIn('purpose', version['manifest']); self.assertNotIn('pixel_sha256', version['manifest']['samples'][0])
        self.assertEqual(version['content_sha256'], digest(manifest))
        with self.runtime.connection() as db:
            row = db.execute('SELECT * FROM dataset_imports WHERE id=?', (import_id,)).fetchone()
            self.assertEqual(row['manifest_json'], original_manifest); self.assertEqual(row['report_json'], original_report)
            self.assertEqual(row['request_sha256'], digest(test_datasets.META))
        self.assertEqual(self.store.create('a', 'old', test_datasets.META)[0]['id'], import_id)


if __name__ == '__main__':
    unittest.main()
