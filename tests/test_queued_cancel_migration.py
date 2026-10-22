"""Offline fixture migrations, no retained runtime, no credentials, no ML execution."""
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest import mock

from model_platform.common import Runtime
from model_platform.store import Store
from model_platform.run_schema import (LEGACY_RUN_TABLE, RUN_TABLE, VERSION,
                                       normalize, verify)
from model_platform.migrate_queued_cancel import migrate, _snapshot
from scripts.demo import request_body
from test_running_cancel_migration import legacy_attempts_fixture
from model_platform.migrate_running_cancel import migrate as migrate_running_cancel


class QueueMigrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        patch = mock.patch('model_platform.store.cursor_key', return_value=b'queued-migration-fixture')
        patch.start()
        self.addCleanup(patch.stop)
        self.runtime = Runtime(self.temp.name)
        self.store = Store(self.runtime)
        self.first = self.store.submit('demo', 'first', request_body())[0]
        run, attempt = self.store.claim()
        self.store.register(run, attempt, 'a'*64, {'dataset_sha256': 'b'*64},
                            {'contract': 'fixture', 'format': 'fixture'})
        self.queued = self.store.submit('demo', 'queued', request_body())[0]
        with self.runtime.connection() as db:
            db.execute('UPDATE runs SET rowid=rowid+40')
            db.execute('INSERT INTO run_dataset_bindings VALUES(?,?,?,?)',
                       (self.first, 'fixture-dataset', 'c'*64, '{"fixture":true}'))
            db.execute('CREATE INDEX fixture_runs_tenant ON runs(tenant)')
            db.execute("CREATE TRIGGER fixture_runs_audit AFTER INSERT ON runs BEGIN SELECT 1; END")
        self.downgrade_fixture()

    def downgrade_fixture(self):
        # The queued migration fixture predates running cancellation too.
        legacy_attempts_fixture(self.runtime)
        with self.runtime.connection() as db:
            db.execute('PRAGMA foreign_keys=OFF')
            db.execute('BEGIN IMMEDIATE')
            objects = [r[0] for r in db.execute(
                "SELECT sql FROM sqlite_master WHERE tbl_name='runs' AND type IN ('index','trigger') AND sql IS NOT NULL")]
            db.execute(LEGACY_RUN_TABLE.format(name='legacy_runs'))
            db.execute('INSERT INTO legacy_runs(rowid,id,tenant,idempotency_key,request_sha256,input_json,environment_json,status,created_at,updated_at,model_version_id) '
                       'SELECT rowid,id,tenant,idempotency_key,request_sha256,input_json,environment_json,status,created_at,updated_at,model_version_id FROM runs')
            db.execute('DROP TABLE runs')
            db.execute('ALTER TABLE legacy_runs RENAME TO runs')
            for sql in objects:
                db.execute(sql)
            db.execute('DELETE FROM schema_migrations WHERE version=?', (VERSION,))

    def snapshot(self):
        with self.runtime.connection() as db:
            return _snapshot(db)

    def test_old_startup_fails_without_writes_or_cursor_creation(self):
        before = self.snapshot()
        with mock.patch('model_platform.store.cursor_key', side_effect=AssertionError('must not create cursor')):
            with self.assertRaisesRegex(RuntimeError, 'migrate_queued_cancel'):
                Store(self.runtime)
        self.assertEqual(self.snapshot(), before)

    def test_complete_migration_preserves_history_rowids_fks_and_cursor(self):
        cursor = self.store.list_page('runs', 'demo', limit=1)['next_cursor']
        before = self.snapshot()
        self.assertTrue(migrate(self.runtime))
        self.assertFalse(migrate(self.runtime))
        after = self.snapshot()
        markers_before = before.pop('schema_migrations')
        self.assertEqual(markers_before, after['schema_migrations'][:len(markers_before)])
        self.assertEqual([row[-1] for row in after['schema_migrations'][len(markers_before):]], [VERSION])
        after.pop('schema_migrations')
        self.assertEqual(before, after)
        migrate_running_cancel(self.runtime)
        current = Store(self.runtime)
        third = current.submit('demo', 'third', request_body())[0]
        page = current.list_page('runs', 'demo', cursor=cursor)
        self.assertEqual(page['ids'], [self.first])
        self.assertNotIn(third, page['ids'])
        current.cancel('demo', self.queued)
        self.assertEqual(current.run('demo', self.queued)['status'], 'cancelled')
        with self.runtime.connection() as db:
            verify(db)
            self.assertEqual(db.execute('PRAGMA foreign_key_check').fetchall(), [])
            self.assertEqual(db.execute('PRAGMA integrity_check').fetchone()[0], 'ok')
            for sql in ("UPDATE runs SET input_json='{}'", "DELETE FROM evaluations", "DELETE FROM models",
                        "DELETE FROM run_dataset_bindings"):
                with self.assertRaises(sqlite3.IntegrityError):
                    db.execute(sql)
            self.assertTrue(db.execute("SELECT 1 FROM sqlite_master WHERE name='fixture_runs_tenant'").fetchone())
            self.assertTrue(db.execute("SELECT 1 FROM sqlite_master WHERE name='fixture_runs_audit'").fetchone())

    def test_interrupted_running_records_preserved_then_normal_recovery(self):
        run, attempt = self.store.claim()
        before = self.snapshot()
        migrate(self.runtime)
        after = self.snapshot()
        for table in ('runs', 'attempts', 'logs', 'models', 'evaluations'):
            self.assertEqual(before[table], after[table])
        migrate_running_cancel(self.runtime)
        current = Store(self.runtime)
        self.assertEqual(current.recover_worker(), 1)
        row = current.run('demo', self.queued)
        self.assertEqual(row['status'], 'failed')
        self.assertEqual(row['attempts'][0]['error_code'], 'interrupted')

    def test_independent_evaluation_rows_are_unchanged_then_recovered(self):
        from model_platform.datasets import DatasetStore
        from model_platform.model_evaluations import ModelEvaluationStore
        DatasetStore(self.runtime)
        with self.runtime.connection() as db:
            mid = db.execute('SELECT id FROM models').fetchone()[0]
            db.execute("INSERT INTO dataset_versions VALUES('dsv_fixture','demo','fixture',?,1,'fixture','{}',0)", ('a'*64,))
            db.execute("INSERT INTO model_evaluations VALUES('mev_fixture','demo','fixture',?,'{}',?,'dsv_fixture','{}',?,'running',1,2,NULL,NULL)",
                       ('b'*64, mid, 'c'*64))
        before = self.snapshot()
        migrate(self.runtime)
        self.assertEqual(before['model_evaluations'], self.snapshot()['model_evaluations'])
        migrate_running_cancel(self.runtime)
        evaluations = ModelEvaluationStore(Store(self.runtime))
        self.assertEqual(evaluations.recover(), 1)
        with self.runtime.connection() as db:
            row = db.execute("SELECT status,error_code FROM model_evaluations WHERE id='mev_fixture'").fetchone()
            self.assertEqual(tuple(row), ('failed', 'evaluation_interrupted'))
            self.assertEqual(db.execute('SELECT count(*) FROM model_evaluation_results').fetchone()[0], 0)

    def test_backup_is_private_complete_and_restore_is_old_state(self):
        before = self.snapshot()
        migrate(self.runtime)
        backup = self.runtime.home / 'before-queued-training-cancel.sqlite3'
        self.assertEqual(backup.stat().st_mode & 0o777, 0o600)
        with sqlite3.connect(backup) as db:
            db.row_factory = sqlite3.Row
            self.assertEqual(_snapshot(db), before)
        # Isolated restore copy; no mutation or automatic rollback of runtime.
        with tempfile.TemporaryDirectory() as other:
            restored = Path(other)/'restored.sqlite3'
            with sqlite3.connect(backup) as source, sqlite3.connect(restored) as target:
                source.backup(target)
            with sqlite3.connect(restored) as db:
                self.assertEqual(normalize(db.execute("SELECT sql FROM sqlite_master WHERE name='runs'").fetchone()[0]),
                                 normalize(LEGACY_RUN_TABLE.format(name='runs')))

    def test_failure_rolls_back_complete_schema_and_data(self):
        before = self.snapshot()
        with self.assertRaisesRegex(RuntimeError, 'injected'):
            migrate(self.runtime, fault=lambda db: (_ for _ in ()).throw(RuntimeError('injected')))
        self.assertEqual(self.snapshot(), before)
        with self.runtime.connection() as db:
            verify(db, legacy=True)
            self.assertEqual(db.execute('PRAGMA foreign_key_check').fetchall(), [])
        self.assertTrue((self.runtime.home/'before-queued-training-cancel.sqlite3').exists())
        with self.assertRaises(FileExistsError):
            migrate(self.runtime)

    def test_locks_refuse_without_backup(self):
        for name in ('api', 'worker', 'model-import'):
            with self.subTest(name=name):
                with self.runtime.lock(name), self.assertRaises(RuntimeError):
                    migrate(self.runtime)
                self.assertFalse((self.runtime.home/'before-queued-training-cancel.sqlite3').exists())

    def test_live_serving_pid_is_checked_never_killed(self):
        with self.runtime.connection() as db:
            model_id = db.execute('SELECT id FROM models').fetchone()[0]
        deployment = self.store.create_deployment('demo', model_id)
        with self.runtime.connection() as db:
            db.execute('UPDATE deployments SET pid=? WHERE id=?', (os.getpid(), deployment))
        with mock.patch('model_platform.migrate_queued_cancel.os.kill') as probe:
            with self.assertRaisesRegex(RuntimeError, 'serving'):
                migrate(self.runtime)
            probe.assert_called_once_with(os.getpid(), 0)
        self.assertFalse((self.runtime.home/'before-queued-training-cancel.sqlite3').exists())

    def test_backup_path_no_overwrite_and_no_symlink(self):
        backup = self.runtime.home/'before-queued-training-cancel.sqlite3'
        backup.write_bytes(b'precious')
        with self.assertRaises(FileExistsError):
            migrate(self.runtime)
        self.assertEqual(backup.read_bytes(), b'precious')
        backup.unlink()
        target = self.runtime.home/'unrelated'
        target.write_bytes(b'untouched')
        backup.symlink_to(target)
        with self.assertRaises(FileExistsError):
            migrate(self.runtime)
        self.assertEqual(target.read_bytes(), b'untouched')

    def test_partial_marker_or_trigger_fails_closed(self):
        migrate(self.runtime)
        with self.runtime.connection() as db:
            db.execute('DROP TRIGGER runs_input_immutable')
            db.execute('CREATE TRIGGER runs_input_immutable BEFORE UPDATE ON runs BEGIN SELECT 1; END')
        with self.assertRaises(RuntimeError):
            migrate(self.runtime)
        with mock.patch('model_platform.store.cursor_key', side_effect=AssertionError):
            with self.assertRaises(RuntimeError):
                Store(self.runtime)

    def test_missing_current_marker_is_not_silently_repaired(self):
        migrate(self.runtime)
        with self.runtime.connection() as db:
            db.execute('DELETE FROM schema_migrations WHERE version=?', (VERSION,))
        with self.assertRaisesRegex(RuntimeError, 'marker'):
            Store(self.runtime)
        with self.assertRaisesRegex(RuntimeError, 'marker'):
            migrate(self.runtime)

    def test_legacy_marker_and_unknown_schema_are_rejected_without_backup(self):
        with self.runtime.connection() as db:
            db.execute('INSERT INTO schema_migrations VALUES(?)', (VERSION,))
        with self.assertRaisesRegex(RuntimeError, 'marker'):
            migrate(self.runtime)
        with self.runtime.connection() as db:
            db.execute('DELETE FROM schema_migrations WHERE version=?', (VERSION,))
            db.execute('ALTER TABLE runs ADD COLUMN unexpected TEXT')
        with self.assertRaisesRegex(RuntimeError, 'schema'):
            migrate(self.runtime)
        with self.assertRaisesRegex(RuntimeError, 'schema'):
            Store(self.runtime)
        self.assertFalse((self.runtime.home/'before-queued-training-cancel.sqlite3').exists())

    def test_fresh_and_migrated_required_definitions_match(self):
        migrate(self.runtime)
        with tempfile.TemporaryDirectory() as other:
            fresh = Runtime(other)
            Store(fresh)
            def definitions(runtime):
                with runtime.connection() as db:
                    return {r['name']: normalize(r['sql']) for r in db.execute(
                        "SELECT name,sql FROM sqlite_master WHERE name IN ('runs','runs_history','runs_input_immutable')")}
            self.assertEqual(definitions(self.runtime), definitions(fresh))





class QueueStartupNamespaceTests(unittest.TestCase):
    def test_dataset_only_and_missing_queue_in_full_platform_fail_closed(self):
        from model_platform.datasets import DatasetStore
        for dataset_only in (True, False):
            with self.subTest(dataset_only=dataset_only), tempfile.TemporaryDirectory() as directory:
                runtime = Runtime(directory)
                if dataset_only:
                    DatasetStore(runtime)
                else:
                    Store(runtime)
                    with runtime.connection() as db:
                        db.execute('PRAGMA foreign_keys=OFF')
                        db.execute('DROP TABLE runs')
                with runtime.connection() as db:
                    before = list(db.iterdump())
                with mock.patch('model_platform.store.cursor_key', side_effect=AssertionError):
                    with self.assertRaisesRegex(RuntimeError, 'Incomplete platform schema'):
                        Store(runtime)
                with runtime.connection() as db:
                    self.assertEqual(list(db.iterdump()), before)

    def test_absent_queue_with_partial_or_unknown_tables_fails_without_writes(self):
        for tables in (('dataset_imports',), ('dataset_imports', 'dataset_versions'),
                       ('attempts',), ('models',), ('schema_migrations',), ('unexpected',)):
            with self.subTest(tables=tables), tempfile.TemporaryDirectory() as directory:
                runtime = Runtime(directory)
                with runtime.connection() as db:
                    for table in tables:
                        db.execute(f'CREATE TABLE {table}(fixture TEXT)')
                    before = list(db.iterdump())
                with mock.patch('model_platform.store.cursor_key', side_effect=AssertionError):
                    with self.assertRaisesRegex(RuntimeError, 'Incomplete platform schema'):
                        Store(runtime)
                with runtime.connection() as db:
                    self.assertEqual(list(db.iterdump()), before)


if __name__ == '__main__':
    unittest.main()
