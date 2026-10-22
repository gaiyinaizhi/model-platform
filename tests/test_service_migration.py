"""Isolated offline SQLite migration/constraint fixtures; no production runtimes."""
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest import mock

from model_platform.common import Runtime
from model_platform.migrate_services import BACKUP_NAME, _snapshot, migrate
from model_platform.service_schema import (SCHEMA, TABLES, VERSION, required_objects,
                                           startup_check, verify)
from model_platform.store import Store
from scripts.demo import request_body


def predecessor_fixture(runtime):
    """Remove only fresh service objects to model the supported predecessor."""
    with runtime.connection() as db:
        for name in reversed(TABLES):
            db.execute('DROP TABLE IF EXISTS ' + name)
        db.execute('DELETE FROM schema_migrations WHERE version=?', (VERSION,))


class ServiceMigrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.runtime = Runtime(self.temp.name)
        self.store = Store(self.runtime)
        self.run = self.store.submit('demo', 'first', request_body())[0]
        run, attempt = self.store.claim()
        self.store.log(attempt, 'fixture')
        self.model = self.store.register(run, attempt, 'a'*64, {'dataset_sha256': 'b'*64},
                                         {'contract': 'fixture', 'format': 'fixture'})
        with self.runtime.connection() as db:
            self.model = db.execute('SELECT id FROM models').fetchone()[0]
            db.execute('CREATE INDEX fixture_status ON attempts(status)')
            db.execute('CREATE TRIGGER fixture_insert AFTER INSERT ON attempts BEGIN SELECT 1; END')
            db.execute('CREATE VIEW fixture_view AS SELECT id FROM attempts')
            db.execute('CREATE TABLE fixture_external(key TEXT PRIMARY KEY,value TEXT) WITHOUT ROWID')
            db.execute("INSERT INTO fixture_external VALUES('keep','unchanged')")
            db.execute('UPDATE attempts SET rowid=rowid+100')
        predecessor_fixture(self.runtime)

    def snapshot(self):
        with self.runtime.connection() as db:
            return _snapshot(db)

    def schema(self):
        with self.runtime.connection() as db:
            return list(map(tuple, db.execute('SELECT type,name,tbl_name,sql FROM sqlite_master ORDER BY type,name')))

    def test_legacy_startup_fails_before_cursor_or_schema_writes(self):
        before, schema = self.snapshot(), self.schema()
        with self.runtime.connection() as db:
            with self.assertRaisesRegex(RuntimeError, 'migrate_services'):
                startup_check(db)
        with mock.patch('model_platform.store.cursor_key', side_effect=AssertionError('cursor must not be touched')):
            with self.assertRaisesRegex(RuntimeError, 'migrate_services'):
                Store(self.runtime)
        self.assertEqual(before, self.snapshot())
        self.assertEqual(schema, self.schema())

    def test_additive_migration_preserves_every_row_and_definition(self):
        before, schema = self.snapshot(), self.schema()
        self.assertTrue(migrate(self.runtime))
        self.assertFalse(migrate(self.runtime))
        after = self.snapshot()
        for name in TABLES:
            self.assertEqual(after.pop(name), [])
        marker = after['schema_migrations'].pop()
        self.assertEqual(marker[1], VERSION)
        self.assertEqual(after, before)
        actual_schema = self.schema()
        for row in schema:
            self.assertIn(row, actual_schema)
        with self.runtime.connection() as db:
            verify(db)
        Store(self.runtime)

    def test_backup_private_exact_and_restore_is_legacy(self):
        before = self.snapshot()
        migrate(self.runtime)
        backup = self.runtime.home / BACKUP_NAME
        self.assertEqual(backup.stat().st_mode & 0o777, 0o600)
        with sqlite3.connect(backup) as db:
            self.assertEqual(_snapshot(db), before)
            verify(db, legacy=True)
        with tempfile.TemporaryDirectory() as directory:
            with sqlite3.connect(backup) as source, sqlite3.connect(Path(directory)/'restore.sqlite3') as target:
                source.backup(target)
                with self.assertRaisesRegex(RuntimeError, 'migrate_services'):
                    startup_check(target)

    def test_failure_rolls_back_ddl_marker_rows_and_keeps_backup(self):
        before, schema = self.snapshot(), self.schema()
        def fault(db):
            raise RuntimeError('injected')
        with self.assertRaisesRegex(RuntimeError, 'injected'):
            migrate(self.runtime, fault=fault)
        self.assertEqual(before, self.snapshot())
        self.assertEqual(schema, self.schema())
        self.assertTrue((self.runtime.home / BACKUP_NAME).exists())
        with self.assertRaises(FileExistsError):
            migrate(self.runtime)

    def test_failure_after_marker_insert_rolls_back_marker_and_schema(self):
        before, schema = self.snapshot(), self.schema()
        original = verify
        def final_failure(db, *, legacy=False):
            if not legacy:
                self.assertTrue(db.execute('SELECT 1 FROM schema_migrations WHERE version=?', (VERSION,)).fetchone())
                raise RuntimeError('final validation injected')
            return original(db, legacy=legacy)
        with mock.patch('model_platform.migrate_services.verify', side_effect=final_failure):
            with self.assertRaisesRegex(RuntimeError, 'final validation injected'):
                migrate(self.runtime)
        self.assertEqual(before, self.snapshot())
        self.assertEqual(schema, self.schema())
        self.assertTrue((self.runtime.home / BACKUP_NAME).exists())

    def test_backup_path_swap_never_writes_replacement(self):
        before, schema = self.snapshot(), self.schema()
        backup = self.runtime.home / BACKUP_NAME
        victim = self.runtime.home / 'victim'
        victim.write_bytes(b'untouched')
        real_copy = __import__('shutil').copyfileobj
        def swap_path(source, target):
            backup.unlink()
            backup.symlink_to(victim)
            real_copy(source, target)
        with mock.patch('model_platform.migrate_services.shutil.copyfileobj', side_effect=swap_path):
            with self.assertRaisesRegex(RuntimeError, 'Backup identity changed'):
                migrate(self.runtime)
        self.assertEqual(victim.read_bytes(), b'untouched')
        self.assertEqual(before, self.snapshot())
        self.assertEqual(schema, self.schema())

    def test_existing_row_corruption_rolls_back(self):
        before = self.snapshot()
        with self.assertRaisesRegex(RuntimeError, 'changed existing rows'):
            migrate(self.runtime, fault=lambda db: db.execute("UPDATE attempts SET error_code='bad'"))
        self.assertEqual(before, self.snapshot())

    def test_existing_schema_corruption_rolls_back(self):
        schema = self.schema()
        with self.assertRaisesRegex(RuntimeError, 'changed existing schema'):
            migrate(self.runtime, fault=lambda db: db.execute('DROP INDEX fixture_status'))
        self.assertEqual(schema, self.schema())

    def test_all_locks_prevent_backup(self):
        for name in ('api', 'worker', 'model-import'):
            with self.subTest(name=name), self.runtime.lock(name), self.assertRaises(RuntimeError):
                migrate(self.runtime)
            self.assertFalse((self.runtime.home / BACKUP_NAME).exists())

    def test_serving_live_ambiguous_and_inaccessible_refuse_without_signals(self):
        deployment = self.store.create_deployment('demo', self.model)
        for pid, effect in ((os.getpid(), None), (0, None), (-1, None), ('unknown', None),
                            (12345, PermissionError())):
            with self.subTest(pid=pid, effect=effect):
                with self.runtime.connection() as db:
                    db.execute('UPDATE deployments SET pid=? WHERE id=?', (pid, deployment))
                with mock.patch('model_platform.migrate_services.os.kill', side_effect=effect) as probe:
                    with self.assertRaisesRegex(RuntimeError, 'serving'):
                        migrate(self.runtime)
                    if type(pid) is int and pid > 0:
                        probe.assert_called_once_with(pid, 0)
                    else:
                        probe.assert_not_called()
                self.assertFalse((self.runtime.home / BACKUP_NAME).exists())

    def test_dead_serving_pid_only_liveness_probe(self):
        deployment = self.store.create_deployment('demo', self.model)
        with self.runtime.connection() as db:
            db.execute('UPDATE deployments SET pid=12345 WHERE id=?', (deployment,))
        with mock.patch('model_platform.migrate_services.os.kill', side_effect=ProcessLookupError) as probe:
            self.assertTrue(migrate(self.runtime))
        probe.assert_called_once_with(12345, 0)

    def test_existing_backup_and_symlink_never_overwritten(self):
        backup = self.runtime.home / BACKUP_NAME
        backup.write_bytes(b'precious')
        with self.assertRaises(FileExistsError):
            migrate(self.runtime)
        self.assertEqual(backup.read_bytes(), b'precious')
        backup.unlink()
        target = self.runtime.home / 'unrelated'
        target.write_bytes(b'untouched')
        backup.symlink_to(target)
        with self.assertRaises(FileExistsError):
            migrate(self.runtime)
        self.assertEqual(target.read_bytes(), b'untouched')

    def test_partial_marker_or_objects_refused(self):
        with self.runtime.connection() as db:
            db.execute('INSERT INTO schema_migrations VALUES(?)', (VERSION,))
        with self.assertRaises(RuntimeError):
            migrate(self.runtime)
        with self.runtime.connection() as db:
            db.execute('DELETE FROM schema_migrations WHERE version=?', (VERSION,))
            db.executescript(SCHEMA)
        with self.assertRaisesRegex(RuntimeError, 'Partial'):
            migrate(self.runtime)
        self.assertFalse((self.runtime.home / BACKUP_NAME).exists())

    def test_running_cancel_and_full_predecessor_required(self):
        for sql in ("DELETE FROM schema_migrations WHERE version='running-training-cancel/v1'",
                    'DROP INDEX deployments_history', 'DROP TRIGGER evaluations_update_immutable'):
            with self.subTest(sql=sql), self.runtime.connection() as db:
                db.execute('BEGIN')
                db.execute(sql)
                with self.assertRaises(RuntimeError):
                    verify(db, legacy=True)
                db.rollback()
        self.assertFalse((self.runtime.home / BACKUP_NAME).exists())

    def test_foreign_key_corruption_prevents_backup(self):
        with self.runtime.connection() as db:
            db.execute('PRAGMA foreign_keys=OFF')
            db.execute("INSERT INTO logs(attempt_id,time,event,details) VALUES('missing',0,'fixture','{}')")
        with self.assertRaisesRegex(RuntimeError, 'foreign key'):
            migrate(self.runtime)
        self.assertFalse((self.runtime.home / BACKUP_NAME).exists())

    def test_fresh_and_migrated_definitions_match(self):
        migrate(self.runtime)
        with tempfile.TemporaryDirectory() as directory:
            fresh = Runtime(directory)
            Store(fresh)
            with fresh.connection() as db:
                verify(db)
                expected = {(r[0], r[1]): r[2] for r in db.execute('SELECT type,name,sql FROM sqlite_master')
                            if (r[0], r[1]) in required_objects()}
            with self.runtime.connection() as db:
                actual = {(r[0], r[1]): r[2] for r in db.execute('SELECT type,name,sql FROM sqlite_master')
                          if (r[0], r[1]) in required_objects()}
            self.assertEqual(actual, expected)


class ServiceSchemaTests(unittest.TestCase):
    def runtime(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        runtime = Runtime(temporary.name)
        Store(runtime)
        return runtime

    def test_empty_schema_only_is_fresh(self):
        with sqlite3.connect(':memory:') as db:
            self.assertTrue(startup_check(db))
            db.execute('CREATE VIEW unexpected AS SELECT 1')
            with self.assertRaises(RuntimeError):
                startup_check(db)

    def test_every_required_object_checked_before_startup_repair(self):
        for (kind, name), definition in required_objects().items():
            if definition is None:
                continue
            with self.subTest(name=name):
                runtime = self.runtime()
                with runtime.connection() as db:
                    db.execute('PRAGMA foreign_keys=OFF')
                    db.execute(f'DROP {kind} "{name}"')
                    before = list(db.iterdump())
                with mock.patch('model_platform.store.cursor_key', side_effect=AssertionError('no cursor write')):
                    with self.assertRaises(RuntimeError):
                        Store(runtime)
                with runtime.connection() as db:
                    self.assertEqual(list(db.iterdump()), before)
                with self.assertRaises(RuntimeError):
                    migrate(runtime)
                self.assertFalse((runtime.home / BACKUP_NAME).exists())

    def test_user_extensions_survive_but_tampered_definition_refused(self):
        runtime = self.runtime()
        with runtime.connection() as db:
            db.execute('CREATE INDEX custom_service_index ON model_services(updated_at)')
            db.execute('CREATE TRIGGER custom_service_trigger AFTER INSERT ON model_services BEGIN SELECT 1; END')
            verify(db)
            db.execute('DROP INDEX model_services_history')
            db.execute('CREATE INDEX model_services_history ON model_services(updated_at)')
            with self.assertRaises(RuntimeError):
                verify(db)

    def test_immutable_history_keys_and_legal_transition(self):
        runtime = self.runtime()
        with runtime.connection() as db:
            # FK-off fixture isolates new-table logical constraints from model training.
            db.execute('PRAGMA foreign_keys=OFF')
            for name, tenant in (('deployment_a', 'demo'), ('deployment_b', 'demo'), ('deployment_c', 'other')):
                db.execute("INSERT INTO deployments VALUES(?,?,'model','running','ready',NULL,NULL,NULL,1)", (name, tenant))
            db.execute("INSERT INTO model_services VALUES('service','demo',1,'deployment_a',NULL,'{}',1,1)")
            db.execute("INSERT INTO model_service_revisions VALUES('service',1,'demo','{}',1)")
            db.execute("INSERT INTO model_service_keys VALUES('demo','key','hash','service',1,1)")
            for sql in (
                'INSERT OR REPLACE INTO model_services SELECT * FROM model_services',
                'INSERT OR REPLACE INTO model_service_revisions SELECT * FROM model_service_revisions',
                'INSERT OR REPLACE INTO model_service_keys SELECT * FROM model_service_keys',
                "INSERT INTO model_service_revisions VALUES('service',2,'other','{}',1)",
                "INSERT INTO model_service_keys VALUES('other','key','hash','service',1,1)",
                "UPDATE model_services SET revision=3,active_deployment_id='deployment_b',previous_deployment_id='deployment_a'",
                "UPDATE model_services SET revision=2,active_deployment_id='deployment_c',previous_deployment_id='deployment_a'",
                'UPDATE model_services SET revision=2',
                'UPDATE model_services SET tenant=tenant',
                'UPDATE model_service_revisions SET snapshot_json=snapshot_json',
                'UPDATE model_service_keys SET request_sha256=request_sha256',
                'DELETE FROM model_service_keys', 'DELETE FROM model_service_revisions', 'DELETE FROM model_services',
            ):
                with self.subTest(sql=sql), self.assertRaises(sqlite3.IntegrityError):
                    db.execute(sql)
            db.execute("UPDATE model_services SET revision=2,active_deployment_id='deployment_b',previous_deployment_id='deployment_a',updated_at=2")
            db.execute("INSERT INTO model_service_revisions VALUES('service',2,'demo','{}',2)")
            db.execute("INSERT INTO model_service_keys VALUES('demo','key2','hash2','service',2,2)")
            self.assertEqual(db.execute('SELECT count(*) FROM model_service_revisions').fetchone()[0], 2)


if __name__ == '__main__':
    unittest.main()
