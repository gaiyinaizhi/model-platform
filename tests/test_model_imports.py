"""Pure temporary stores: opaque fixtures do not claim actual ONNX validation."""
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import time
import unittest
from unittest import mock

from model_platform.common import Runtime, Problem
from model_platform.store import Store
from model_platform.model_imports import ModelImportStore, CATALOG_ID, CATALOG_REVISION, artifact_path
from model_platform import model_imports as imports
from model_platform.migrate_imports import migrate
from model_platform.migrate_services import migrate as migrate_services
from model_platform.service_schema import TABLES as SERVICE_TABLES, VERSION as SERVICE_VERSION
from scripts.demo import request_body


class ImportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.runtime = Runtime(Path(self.temp.name))
        self.store = Store(self.runtime)
        self.imports = ModelImportStore(self.store)
        self.raw = b'opaque transport fixture; NOT an ONNX graph'
        self.sha = hashlib.sha256(self.raw).hexdigest()
        self.addCleanup(mock.patch.stopall)
        mock.patch.object(imports, 'ASSET_BYTES', len(self.raw)).start()
        mock.patch.object(imports, 'ASSET_SHA256', self.sha).start()
        self.validator = mock.patch.object(imports, 'validate_isolated', return_value={'status':'validated', 'fixture': True}).start()

    def create(self, key='one', tenant='demo'):
        return self.imports.create(tenant, key, {'catalog_id': CATALOG_ID, 'rights_confirmed': True})[0]

    def validated(self, key='one', tenant='demo'):
        item = self.create(key, tenant)
        return self.imports.upload(tenant, item['id'], lambda: self.raw)

    def test_create_replay_and_foreign_tenant(self):
        first = self.create()
        self.assertEqual(first, self.create())
        with self.assertRaises(Problem) as e:
            self.imports.get_import('other', first['id'])
        self.assertEqual(e.exception.status, 404)
        with self.assertRaises(Problem):
            self.imports.upload('other', first['id'], lambda: self.raw)
        self.validator.assert_not_called()

    def test_validation_registration_and_replay(self):
        item = self.validated()
        self.assertEqual(item['status'], 'validated')
        self.assertEqual(item, self.imports.upload('demo', item['id'], lambda: self.raw))
        self.assertEqual(self.validator.call_count, 1)
        model, created = self.imports.register('demo', item['id'])
        self.assertTrue(created)
        self.assertEqual(self.imports.register('demo', item['id']), (model, False))
        self.assertEqual(model['origin'], 'imported')
        self.assertIsNone(model['evaluation'])
        self.assertNotIn('recipe', model)
        self.assertNotIn('run_id', model)
        self.assertFalse(model['production_allowed'])
        self.assertEqual(self.store.list_ids('runs', 'demo'), [])
        with self.runtime.connection() as db:
            for table in ('models', 'import_validations'):
                with self.assertRaises(sqlite3.IntegrityError):
                    db.execute(f'DELETE FROM {table}')

    def test_hash_rejection_and_incomplete_unbound(self):
        item = self.create()
        with self.assertRaises(Problem):
            self.imports.upload('demo', item['id'], lambda: b'partial')
        self.assertEqual(self.imports.get_import('demo', item['id'])['status'], 'created')
        bad = b'x' * len(self.raw)
        rejected = self.imports.upload('demo', item['id'], lambda: bad)
        self.assertEqual(rejected['reason'], 'artifact_hash_mismatch')
        self.assertEqual(rejected, self.imports.upload('demo', item['id'], lambda: bad))
        self.validator.assert_not_called()
        with self.assertRaises(Problem):
            self.imports.upload('demo', item['id'], lambda: self.raw)
        with self.assertRaises(Problem):
            self.imports.register('demo', item['id'])

    def test_interruption_is_terminal_and_capacity(self):
        item = self.create()
        with self.runtime.connection() as db:
            db.execute("UPDATE model_imports SET status='validating',artifact_sha256=? WHERE id=?", (self.sha, item['id']))
        self.imports.recover()
        rejected = self.imports.get_import('demo', item['id'])
        self.assertEqual(rejected['reason'], 'validation_interrupted')
        self.assertEqual(rejected, self.imports.upload('demo', item['id'], lambda: self.raw))
        self.validator.assert_not_called()
        with mock.patch.object(imports, 'MAX_IMPORTS', 1), self.assertRaises(Problem):
            self.create('two')

    def test_busy_storage_tampering_and_quota(self):
        item = self.create()
        with self.imports.operation(), self.assertRaises(Problem):
            self.imports.upload('demo', item['id'], lambda: self.raw)
        with mock.patch.object(imports, 'MAX_STORED_BYTES', 0), self.assertRaises(Problem):
            self.imports.upload('demo', item['id'], lambda: self.raw)
        item = self.imports.upload('demo', item['id'], lambda: self.raw)
        with self.runtime.connection() as db:
            aid = db.execute('SELECT artifact_id FROM model_imports WHERE id=?', (item['id'],)).fetchone()[0]
        path = artifact_path(self.runtime, aid)
        path.unlink()
        path.symlink_to('/dev/null')
        with self.assertRaises(Problem):
            self.imports.register('demo', item['id'])

    def test_fifo_rejected_without_blocking(self):
        item = self.validated()
        with self.runtime.connection() as db:
            aid = db.execute('SELECT artifact_id FROM model_imports WHERE id=?', (item['id'],)).fetchone()[0]
        path = artifact_path(self.runtime, aid)
        path.unlink()
        os.mkfifo(path)
        start = time.monotonic()
        with self.assertRaises(Problem):
            self.imports.register('demo', item['id'])
        self.assertLess(time.monotonic() - start, 1)

    def test_stored_manifest_binding_is_checked(self):
        first = self.validated('first')
        second = self.validated('second')
        model, _ = self.imports.register('demo', first['id'])
        with self.runtime.connection() as db:
            db.execute('DROP TRIGGER models_update_immutable')
            manifest = dict(model, validation_id=second['validation_id'])
            db.execute('UPDATE models SET manifest_json=? WHERE id=?', (json.dumps(manifest), model['model_version_id']))
        with self.assertRaises(Problem):
            self.store.model('demo', model['model_version_id'])

    def test_absolute_upload_deadline_leaves_import_unbound(self):
        item = self.create()
        stream = mock.Mock()
        stream.read1.return_value = b'x'
        connection = mock.Mock()
        connection.gettimeout.return_value = 10
        with mock.patch.object(imports.time, 'monotonic', side_effect=[0, 0, 16]):
            with self.assertRaises(Problem):
                self.imports.upload('demo', item['id'], lambda: imports.read_artifact_body(stream, connection, len(self.raw)))
        self.assertEqual(self.imports.get_import('demo', item['id'])['status'], 'created')
        connection.settimeout.assert_called_with(10)
        self.validator.assert_not_called()

    def test_cross_import_and_tenant_validation_fk(self):
        first = self.validated('first')
        second = self.validated('second')
        third = self.validated('third', 'other')
        with self.runtime.connection() as db:
            for validation_id in (second['validation_id'], third['validation_id']):
                with self.assertRaises(sqlite3.IntegrityError):
                    db.execute("INSERT INTO models(id,tenant,artifact_sha256,manifest_json,created_at,origin,import_id,validation_id,catalog_revision) VALUES('bad','demo',?,'{}',1,'imported',?,?,?)",
                        (self.sha, first['id'], validation_id, CATALOG_REVISION))
        self.assertEqual(self.store.list_ids('models', 'demo'), [])

    def test_validation_failure_terminal(self):
        self.validator.side_effect = RuntimeError('opaque failure')
        item = self.validated()
        self.assertEqual(item['status'], 'rejected')
        self.assertEqual(item['reason'], 'validation_failed')
        self.assertEqual(item, self.imports.upload('demo', item['id'], lambda: self.raw))


class MigrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.runtime = Runtime(Path(self.temp.name))
        self.store = Store(self.runtime)
        self.store.submit('demo', 'legacy', request_body())
        run, attempt = self.store.claim()
        self.mid = self.store.register(run, attempt, 'a'*64, {'dataset_sha256':'b'*64}, {'contract':'fixture','format':'fixture'})
        self.did = self.store.submit_deployment('demo', self.mid, 'legacy-deployment')[0]
        with self.runtime.connection() as db:
            db.execute('PRAGMA foreign_keys=OFF')
            db.execute('BEGIN IMMEDIATE')
            # The historical fixture predates services; keep later markers and
            # objects out, and preserve the original model immutability guards.
            for table in reversed(SERVICE_TABLES):
                db.execute('DROP TABLE ' + table)
            db.execute('DELETE FROM schema_migrations WHERE version=?', (SERVICE_VERSION,))
            model_objects = [row[0] for row in db.execute(
                "SELECT sql FROM sqlite_master WHERE type IN ('index','trigger') AND tbl_name='models' AND sql IS NOT NULL")]
            db.execute('CREATE TABLE old_models (id TEXT PRIMARY KEY, tenant TEXT NOT NULL, run_id TEXT NOT NULL UNIQUE REFERENCES runs(id), attempt_id TEXT NOT NULL UNIQUE REFERENCES attempts(id), evaluation_id TEXT NOT NULL UNIQUE REFERENCES evaluations(id), artifact_sha256 TEXT NOT NULL, manifest_json TEXT NOT NULL, created_at REAL NOT NULL)')
            db.execute('INSERT INTO old_models SELECT id,tenant,run_id,attempt_id,evaluation_id,artifact_sha256,manifest_json,created_at FROM models')
            db.execute('UPDATE old_models SET rowid=41')
            db.execute('DROP TABLE models')
            db.execute('ALTER TABLE old_models RENAME TO models')
            for sql in model_objects:
                db.execute(sql)
            db.execute("DELETE FROM schema_migrations WHERE version='imported-origin/v1'")
            before = list(db.iterdump())
        # Do not use current startup as an implicit old-schema repair. The
        # retained test-only Store reference is solely for legacy fixture writes.
        with mock.patch('model_platform.store.cursor_key', side_effect=AssertionError('no cursor write')):
            with self.assertRaises(RuntimeError):
                Store(self.runtime)
        with self.runtime.connection() as db:
            self.assertEqual(list(db.iterdump()), before)

    def test_migration_preserves_bytes_links_rowids_and_replays(self):
        with self.runtime.connection() as db:
            before = tuple(db.execute('SELECT rowid,* FROM models').fetchone())
            deployment = tuple(db.execute('SELECT * FROM deployments').fetchone())
        self.assertTrue(migrate(self.runtime))
        self.assertFalse(migrate(self.runtime))
        with self.runtime.connection() as db:
            after = tuple(db.execute('SELECT rowid,id,tenant,run_id,attempt_id,evaluation_id,artifact_sha256,manifest_json,created_at FROM models').fetchone())
            self.assertEqual(before, after)
            self.assertEqual(deployment, tuple(db.execute('SELECT * FROM deployments').fetchone()))
            self.assertEqual(db.execute('PRAGMA foreign_key_check').fetchall(), [])
            with self.assertRaises(sqlite3.IntegrityError):
                db.execute('DELETE FROM models')
        self.assertTrue(migrate_services(self.runtime))
        self.store = Store(self.runtime)
        self.assertEqual(self.store.submit_deployment('demo', self.mid, 'legacy-deployment'), (self.did, False))

    def test_history_cursor_survives_migration_and_new_inserts(self):
        self.store.submit('demo', 'second', request_body())
        run, attempt = self.store.claim()
        self.store.register(run, attempt, 'c'*64, {'dataset_sha256':'d'*64}, {'contract':'fixture','format':'fixture'})
        first = self.store.list_page('models', 'demo', limit=1)
        self.assertIsNotNone(first['next_cursor'])
        migrate(self.runtime)
        self.assertTrue(migrate_services(self.runtime))
        self.store = Store(self.runtime)
        self.store.submit('demo', 'third', request_body())
        run, attempt = self.store.claim()
        newest = self.store.register(run, attempt, 'e'*64, {'dataset_sha256':'f'*64}, {'contract':'fixture','format':'fixture'})
        second = self.store.list_page('models', 'demo', cursor=first['next_cursor'])
        self.assertEqual(second['ids'], [self.mid])
        self.assertNotIn(newest, second['ids'])

    def test_migrated_schema_tampered_trigger_is_rejected(self):
        migrate(self.runtime)
        with self.runtime.connection() as db:
            db.execute('DROP TRIGGER models_update_immutable')
            db.execute('CREATE TRIGGER models_update_immutable BEFORE UPDATE ON models BEGIN SELECT 1; END')
        with self.assertRaises(RuntimeError):
            migrate(self.runtime)

    def test_fault_rolls_back_and_backup_exists(self):
        with self.assertRaises(RuntimeError):
            migrate(self.runtime, fault=lambda db: (_ for _ in ()).throw(RuntimeError('injected')))
        with self.runtime.connection() as db:
            self.assertNotIn('origin', [r['name'] for r in db.execute('PRAGMA table_info(models)')])
            self.assertEqual(db.execute('PRAGMA foreign_key_check').fetchall(), [])
        self.assertTrue((self.runtime.home / 'before-imported-origin.sqlite3').is_file())

    def test_active_api_lock_blocks_migration(self):
        with self.runtime.lock('api'), self.assertRaises(RuntimeError):
            migrate(self.runtime)

    def test_partial_schema_rejected(self):
        with self.runtime.connection() as db:
            db.execute("ALTER TABLE models ADD COLUMN origin TEXT DEFAULT 'trained'")
        with self.assertRaises(RuntimeError):
            migrate(self.runtime)
