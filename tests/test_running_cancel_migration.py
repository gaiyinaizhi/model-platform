"""Offline temporary SQLite fixtures; no production data or ML execution."""
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest import mock

from model_platform.common import Runtime
from model_platform.migrate_running_cancel import BACKUP_NAME, _snapshot, migrate
from model_platform.running_cancel_schema import (
    ATTEMPT_TABLE, LEGACY_ATTEMPT_TABLE, SCHEMA, VERSION, normalize, required_objects,
    startup_check, verify,
)
from model_platform.store import Store
from scripts.demo import request_body


def legacy_attempts_fixture(runtime):
    """Test-only rebuild of a fresh current DB to the exact supported predecessor."""
    with runtime.connection() as db:
        db.execute('PRAGMA foreign_keys=OFF')
        db.execute('PRAGMA legacy_alter_table=ON')
        db.execute('BEGIN IMMEDIATE')
        db.execute('DROP TABLE training_cancellations')
        objects = [row[0] for row in db.execute(
            "SELECT sql FROM sqlite_master WHERE tbl_name='attempts' "
            "AND type IN ('index','trigger') AND sql IS NOT NULL")]
        db.execute(LEGACY_ATTEMPT_TABLE.format(name='legacy_attempts'))
        db.execute('INSERT INTO legacy_attempts(rowid,id,run_id,number,status,started_at,finished_at,error_code) '
                   'SELECT rowid,id,run_id,number,status,started_at,finished_at,error_code FROM attempts')
        db.execute('DROP TABLE attempts')
        db.execute('ALTER TABLE legacy_attempts RENAME TO attempts')
        for sql in objects:
            db.execute(sql)
        db.execute('DELETE FROM schema_migrations WHERE version=?', (VERSION,))


class RunningMigrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        patch = mock.patch('model_platform.store.cursor_key', return_value=b'running-migration-fixture')
        patch.start()
        self.addCleanup(patch.stop)
        self.runtime = Runtime(self.temp.name)
        self.store = Store(self.runtime)
        self.first = self.store.submit('demo', 'succeeded', request_body())[0]
        run, self.completed_attempt = self.store.claim()
        self.store.log(self.completed_attempt, 'fixture')
        self.store.register(run, self.completed_attempt, 'a'*64, {'dataset_sha256': 'b'*64},
                            {'contract': 'fixture', 'format': 'fixture'})
        self.running = self.store.submit('demo', 'running', request_body())[0]
        run, self.failed_attempt = self.store.claim()
        self.store.fail(run['id'], self.failed_attempt, 'fixture_failure')
        self.store.retry('demo', self.running)
        run, self.running_attempt = self.store.claim()
        self.queued = self.store.submit('demo', 'queued', request_body())[0]
        with self.runtime.connection() as db:
            db.execute('UPDATE attempts SET rowid=rowid+100')
            db.execute('UPDATE runs SET rowid=rowid+40')
            db.execute('CREATE INDEX fixture_attempt_status ON attempts(status,number)')
            db.execute('CREATE TRIGGER fixture_attempt_audit AFTER INSERT ON attempts BEGIN SELECT 1; END')
            db.execute('CREATE VIEW fixture_attempt_view AS SELECT id,run_id,status FROM attempts')
            db.execute('CREATE TABLE fixture_external(key TEXT PRIMARY KEY,value TEXT) WITHOUT ROWID')
            db.execute("INSERT INTO fixture_external VALUES('keep','unchanged')")
            db.execute('INSERT INTO run_dataset_bindings VALUES(?,?,?,?)',
                       (self.first, 'fixture-dataset', 'c'*64, '{"fixture":true}'))
        legacy_attempts_fixture(self.runtime)

    def snapshot(self):
        with self.runtime.connection() as db:
            return _snapshot(db)

    def schema(self):
        with self.runtime.connection() as db:
            return list(map(tuple, db.execute('SELECT type,name,tbl_name,sql FROM sqlite_master ORDER BY type,name')))

    def test_old_startup_fails_before_cursor_or_schema_writes(self):
        before, schema = self.snapshot(), self.schema()
        with mock.patch('model_platform.store.cursor_key', side_effect=AssertionError('must not create cursor')):
            with self.assertRaisesRegex(RuntimeError, 'migrate_running_cancel'):
                Store(self.runtime)
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(self.schema(), schema)

    def test_migration_preserves_rows_rowids_fks_indexes_triggers_views_and_cursor(self):
        cursor = self.store.list_page('runs', 'demo', limit=1)['next_cursor']
        before = self.snapshot()
        self.assertTrue(migrate(self.runtime))
        self.assertFalse(migrate(self.runtime))
        after = self.snapshot()
        self.assertEqual(after.pop('training_cancellations'), [])
        added = after['schema_migrations'].pop()
        self.assertEqual(added[1], VERSION)
        self.assertEqual(before, after)
        current = Store(self.runtime)
        current.submit('demo', 'later', request_body())
        page = current.list_page('runs', 'demo', cursor=cursor)
        self.assertEqual(page['ids'], [self.running])
        with self.runtime.connection() as db:
            verify(db)
            self.assertEqual(db.execute('PRAGMA foreign_key_check').fetchall(), [])
            self.assertEqual(db.execute('PRAGMA integrity_check').fetchone()[0], 'ok')
            self.assertEqual(len(db.execute('SELECT * FROM fixture_attempt_view').fetchall()), 3)
            for name in ('fixture_attempt_status', 'fixture_attempt_audit'):
                self.assertTrue(db.execute('SELECT 1 FROM sqlite_master WHERE name=?', (name,)).fetchone())
            for sql in ("UPDATE runs SET input_json='{}'", 'DELETE FROM models',
                        'DELETE FROM evaluations', 'DELETE FROM run_dataset_bindings'):
                with self.assertRaises(sqlite3.IntegrityError):
                    db.execute(sql)
        status, projection = current.cancel_running('demo', self.running, self.running_attempt)
        self.assertEqual(status, 202)
        self.assertEqual(projection['status'], 'running')
        current.confirm_cancel(self.running, self.running_attempt, 'never_started')
        row = current.run('demo', self.running)
        self.assertEqual([attempt['status'] for attempt in row['attempts']], ['failed', 'cancelled'])

    def test_backup_is_private_exact_and_restore_remains_legacy(self):
        before = self.snapshot()
        migrate(self.runtime)
        backup = self.runtime.home / BACKUP_NAME
        self.assertEqual(backup.stat().st_mode & 0o777, 0o600)
        with sqlite3.connect(backup) as db:
            self.assertEqual(_snapshot(db), before)
            self.assertEqual(normalize(db.execute("SELECT sql FROM sqlite_master WHERE name='attempts'").fetchone()[0]),
                             normalize(LEGACY_ATTEMPT_TABLE.format(name='attempts')))
        with tempfile.TemporaryDirectory() as directory:
            with sqlite3.connect(backup) as source, sqlite3.connect(Path(directory)/'restored.sqlite3') as target:
                source.backup(target)
                verify(target, legacy=True)

    def test_failure_rolls_back_all_rows_and_schema_and_retains_backup(self):
        before, schema = self.snapshot(), self.schema()
        with self.assertRaisesRegex(RuntimeError, 'injected'):
            migrate(self.runtime, fault=lambda db: (_ for _ in ()).throw(RuntimeError('injected')))
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(self.schema(), schema)
        with self.runtime.connection() as db:
            verify(db, legacy=True)
            self.assertEqual(db.execute('PRAGMA foreign_key_check').fetchall(), [])
        self.assertTrue((self.runtime.home/BACKUP_NAME).exists())
        with self.assertRaises(FileExistsError):
            migrate(self.runtime)

    def test_integrity_failure_rolls_back_injected_changes(self):
        before, schema = self.snapshot(), self.schema()
        def change(db):
            db.execute("UPDATE attempts SET error_code='corruption' WHERE id=?", (self.failed_attempt,))
        with self.assertRaisesRegex(RuntimeError, 'changed existing rows'):
            migrate(self.runtime, fault=change)
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(self.schema(), schema)

    def test_all_runtime_locks_refuse_without_backup(self):
        for name in ('api', 'worker', 'model-import'):
            with self.subTest(name=name):
                with self.runtime.lock(name), self.assertRaises(RuntimeError):
                    migrate(self.runtime)
                self.assertFalse((self.runtime.home/BACKUP_NAME).exists())

    def test_live_or_ambiguous_serving_process_refuses_never_kills(self):
        with self.runtime.connection() as db:
            model_id = db.execute('SELECT id FROM models').fetchone()[0]
        deployment = self.store.create_deployment('demo', model_id)
        for pid in (os.getpid(), 0, -1, 'unknown'):
            with self.subTest(pid=pid):
                with self.runtime.connection() as db:
                    db.execute('UPDATE deployments SET pid=? WHERE id=?', (pid, deployment))
                with mock.patch('model_platform.migrate_running_cancel.os.kill') as probe:
                    with self.assertRaisesRegex(RuntimeError, 'serving'):
                        migrate(self.runtime)
                    if pid == os.getpid():
                        probe.assert_called_once_with(pid, 0)
                    else:
                        probe.assert_not_called()
                self.assertFalse((self.runtime.home/BACKUP_NAME).exists())

    def test_dead_serving_pid_is_only_probed(self):
        with self.runtime.connection() as db:
            model_id = db.execute('SELECT id FROM models').fetchone()[0]
        deployment = self.store.create_deployment('demo', model_id)
        with self.runtime.connection() as db:
            db.execute('UPDATE deployments SET pid=12345 WHERE id=?', (deployment,))
        with mock.patch('model_platform.migrate_running_cancel.os.kill', side_effect=ProcessLookupError) as probe:
            self.assertTrue(migrate(self.runtime))
        probe.assert_called_once_with(12345, 0)

    def test_backup_no_overwrite_or_symlink(self):
        backup = self.runtime.home/BACKUP_NAME
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

    def test_legacy_marker_partial_objects_and_unknown_attempts_refused(self):
        with self.runtime.connection() as db:
            db.execute('INSERT INTO schema_migrations VALUES(?)', (VERSION,))
        with self.assertRaisesRegex(RuntimeError, 'marker'):
            migrate(self.runtime)
        with self.runtime.connection() as db:
            db.execute('DELETE FROM schema_migrations WHERE version=?', (VERSION,))
            db.executescript(SCHEMA)
        with self.assertRaisesRegex(RuntimeError, 'Partial'):
            migrate(self.runtime)
        with self.runtime.connection() as db:
            db.execute('DROP TABLE training_cancellations')
            db.execute('ALTER TABLE attempts ADD COLUMN unexpected TEXT')
        for action in (lambda: migrate(self.runtime), lambda: Store(self.runtime)):
            with self.assertRaisesRegex(RuntimeError, 'schema'):
                action()
        self.assertFalse((self.runtime.home/BACKUP_NAME).exists())

    def test_queued_migration_remains_a_required_prerequisite(self):
        from model_platform.run_schema import VERSION as QUEUED_VERSION
        with self.runtime.connection() as db:
            db.execute('DELETE FROM schema_migrations WHERE version=?', (QUEUED_VERSION,))
        with self.assertRaisesRegex(RuntimeError, 'queue migration marker'):
            migrate(self.runtime)
        self.assertFalse((self.runtime.home/BACKUP_NAME).exists())

    def test_foreign_key_corruption_refused_before_backup(self):
        with self.runtime.connection() as db:
            db.execute('PRAGMA foreign_keys=OFF')
            db.execute("INSERT INTO logs(attempt_id,time,event,details) VALUES('missing',0,'fixture','{}')")
        with self.assertRaisesRegex(RuntimeError, 'foreign key'):
            migrate(self.runtime)
        self.assertFalse((self.runtime.home/BACKUP_NAME).exists())

    def test_fresh_and_migrated_required_definitions_match(self):
        migrate(self.runtime)
        with tempfile.TemporaryDirectory() as directory:
            fresh = Runtime(directory)
            Store(fresh)
            def definitions(runtime):
                with runtime.connection() as db:
                    return {(kind, name): normalize(sql) if sql else sql
                            for kind, name, sql in db.execute('SELECT type,name,sql FROM sqlite_master')
                            if (kind, name) in required_objects()}
            self.assertEqual(definitions(self.runtime), definitions(fresh))


class RunningStartupAndConstraintTests(unittest.TestCase):
    def new_runtime(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        return Runtime(temporary.name)

    def test_all_required_objects_checked_before_any_startup_repair(self):
        for (kind, name), sql in required_objects().items():
            if sql is None:
                continue
            with self.subTest(name=name):
                runtime = self.new_runtime()
                Store(runtime)
                with runtime.connection() as db:
                    db.execute('PRAGMA foreign_keys=OFF')
                    db.execute(f'DROP {kind} "{name}"')
                    before = list(db.iterdump())
                with mock.patch('model_platform.store.cursor_key', side_effect=AssertionError('must not create cursor')):
                    with self.assertRaises(RuntimeError):
                        Store(runtime)
                with runtime.connection() as db:
                    self.assertEqual(list(db.iterdump()), before)
                with self.assertRaises(RuntimeError):
                    migrate(runtime)
                self.assertFalse((runtime.home/BACKUP_NAME).exists())

    def test_current_missing_marker_and_tampered_trigger_index_refused(self):
        alterations = (
            "DELETE FROM schema_migrations WHERE version='running-training-cancel/v1'",
            'DROP TRIGGER training_cancellations_confirm_once; '
            'CREATE TRIGGER training_cancellations_confirm_once BEFORE UPDATE ON training_cancellations BEGIN SELECT 1; END;',
            'DROP INDEX training_cancellations_pending; '
            'CREATE INDEX training_cancellations_pending ON training_cancellations(requested_at);',
        )
        for alteration in alterations:
            with self.subTest(alteration=alteration):
                runtime = self.new_runtime()
                Store(runtime)
                with runtime.connection() as db:
                    db.executescript(alteration)
                    before = list(db.iterdump())
                for action in (lambda: Store(runtime), lambda: migrate(runtime)):
                    with self.assertRaises(RuntimeError):
                        action()
                with runtime.connection() as db:
                    self.assertEqual(list(db.iterdump()), before)

    def test_empty_is_fresh_partial_is_not(self):
        runtime = self.new_runtime()
        with runtime.connection() as db:
            self.assertTrue(startup_check(db))
            db.execute('CREATE TABLE unexpected(fixture TEXT)')
            with self.assertRaisesRegex(RuntimeError, 'Incomplete'):
                startup_check(db)

    def test_request_binding_immutable_identity_and_exactly_one_confirmation(self):
        runtime = self.new_runtime()
        store = Store(runtime)
        first = store.submit('demo', 'first', request_body())[0]
        run, attempt = store.claim()
        other = store.submit('demo', 'other', request_body())[0]
        with runtime.connection() as db:
            for values in ((attempt, other, 1, None, None), ('missing', first, 1, None, None),
                           (None, first, 1, None, None), (attempt, first, 1, 2, 'child_reaped')):
                with self.subTest(values=values), self.assertRaises(sqlite3.IntegrityError):
                    db.execute('INSERT INTO training_cancellations VALUES(?,?,?,?,?)', values)
            db.execute('INSERT INTO training_cancellations VALUES(?,?,1,NULL,NULL)', (attempt, first))
            with self.assertRaises(sqlite3.IntegrityError):
                db.execute('INSERT OR REPLACE INTO training_cancellations VALUES(?,?,2,NULL,NULL)', (attempt, first))
            for sql in ("UPDATE training_cancellations SET run_id=run_id",
                        "UPDATE training_cancellations SET attempt_id=attempt_id",
                        "UPDATE training_cancellations SET requested_at=2",
                        "DELETE FROM training_cancellations",
                        "UPDATE training_cancellations SET confirmed_at=2",
                        "UPDATE training_cancellations SET confirmation_kind='child_reaped'",
                        "UPDATE training_cancellations SET confirmed_at=2,confirmation_kind='invalid'",
                        "UPDATE training_cancellations SET confirmed_at=NULL,confirmation_kind=NULL"):
                with self.subTest(sql=sql), self.assertRaises(sqlite3.IntegrityError):
                    db.execute(sql)
            db.execute("UPDATE training_cancellations SET confirmed_at=2,confirmation_kind='child_reaped'")
            for sql in ("UPDATE training_cancellations SET confirmed_at=3",
                        "UPDATE training_cancellations SET confirmation_kind='lease_recovered'",
                        "UPDATE training_cancellations SET confirmed_at=2,confirmation_kind='child_reaped'",
                        "UPDATE training_cancellations SET confirmed_at=NULL,confirmation_kind=NULL"):
                with self.subTest(sql=sql), self.assertRaises(sqlite3.IntegrityError):
                    db.execute(sql)
            self.assertEqual(tuple(db.execute('SELECT confirmed_at,confirmation_kind FROM training_cancellations').fetchone()),
                             (2.0, 'child_reaped'))


if __name__ == '__main__':
    unittest.main()
