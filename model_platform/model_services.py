"""Stable bindings to existing, actually probed deployments; no routing fallback."""
import hashlib
import json
import math
import re

from .common import Problem, digest, encode, now, object_fields, text_value
from .pagination import pack_cursor, unpack_cursor
from .store import identifier

MAX_REVISION = 9007199254740991
IDENTITY_FIELDS = ('recipe', 'contract', 'class_names', 'serving_protocol')
SNAPSHOT_FIELDS = {'id', 'revision', 'active_deployment_id', 'previous_deployment_id',
                   'model_version_id', 'artifact_sha256', 'contract', 'class_names',
                   'serving_protocol', 'created_at', 'updated_at', 'endpoint'}


def _id(value, prefix):
    if not isinstance(value, str) or not re.fullmatch(prefix + r'_[0-9a-f]{32}', value):
        raise Problem(400, 'invalid_input', 'Invalid resource identifier')
    return value


def _integrity():
    return Problem(409, 'service_integrity', 'Service configuration integrity failed')


def _identity(deployment):
    return {key: deployment[key] for key in IDENTITY_FIELDS}


class ServiceManager:
    def __init__(self, store, supervisor):
        self.store, self.supervisor = store, supervisor
        self.runtime = store.runtime

    def owned(self, tenant, service_id):
        with self.supervisor.lock:
            with self.runtime.connection() as db:
                self._owned(db, tenant, service_id)

    def _owned(self, db, tenant, service_id):
        return self.store._owned(db, 'model_services', tenant, service_id)

    def _snapshot(self, db, tenant, service_id, revision):
        """Validate historical facts only. Never probe or depend on current files."""
        service = self._owned(db, tenant, service_id)
        history = db.execute('SELECT * FROM model_service_revisions WHERE tenant=? AND service_id=? AND revision=?',
                             (tenant, service_id, revision)).fetchone()
        try:
            if history is None:
                raise ValueError()
            snap = json.loads(history['snapshot_json'])
            identity = json.loads(service['identity_json'])
            if (set(snap) != SNAPSHOT_FIELDS or set(identity) != set(IDENTITY_FIELDS)
                    or identity['recipe'] != 'yolox-nano-dataset/v1'
                    or snap['id'] != service_id or type(snap['revision']) is not int
                    or snap['revision'] != revision or not 1 <= revision <= MAX_REVISION
                    or revision > service['revision']
                    or snap['created_at'] != service['created_at']
                    or snap['updated_at'] != history['created_at']
                    or snap['endpoint'] != f'/services/{service_id}/predict'
                    or any(snap[k] != identity[k] for k in IDENTITY_FIELDS if k != 'recipe')
                    or not re.fullmatch(r'[0-9a-f]{64}', snap['artifact_sha256'])
                    or any(type(snap[k]) not in (float, int) or not math.isfinite(snap[k])
                           for k in ('created_at', 'updated_at'))):
                raise ValueError()
            _id(snap['active_deployment_id'], 'deployment')
            _id(snap['model_version_id'], 'model')
            if revision == 1:
                if snap['previous_deployment_id'] is not None or snap['updated_at'] != snap['created_at']:
                    raise ValueError()
            else:
                previous = db.execute('SELECT snapshot_json FROM model_service_revisions WHERE tenant=? AND service_id=? AND revision=?',
                                      (tenant, service_id, revision - 1)).fetchone()
                if not previous or json.loads(previous[0])['active_deployment_id'] != snap['previous_deployment_id']:
                    raise ValueError()
            if revision == service['revision'] and any(snap[k] != service[k] for k in
                    ('active_deployment_id', 'previous_deployment_id', 'updated_at')):
                raise ValueError()
            return snap
        except (ValueError, TypeError, KeyError, Problem):
            raise _integrity() from None

    def _replay(self, db, tenant, key, request_hash):
        row = db.execute('SELECT * FROM model_service_keys WHERE tenant=? AND idempotency_key=?', (tenant, key)).fetchone()
        if row is None:
            return None
        if row['request_sha256'] != request_hash:
            raise Problem(409, 'idempotency_conflict', 'Key already bound to different input')
        snapshot = self._snapshot(db, tenant, row['service_id'], row['revision'])
        if snapshot['updated_at'] != row['created_at']:
            raise _integrity()
        return snapshot

    def _candidate(self, tenant, deployment_id):
        candidate = self.supervisor.query(tenant, deployment_id)
        instance = self.supervisor.instances.get(deployment_id)
        if (not candidate.get('ready') or candidate['desired'] != 'running'
                or candidate['observed'] != 'ready' or instance is None):
            raise Problem(503, 'not_ready', 'Deployment is not actually ready')
        if candidate.get('recipe') != 'yolox-nano-dataset/v1':
            raise Problem(409, 'service_incompatible', 'Only dataset detection deployments are supported')
        from .dataset_serving import public_identity
        manifest = self.store.model(tenant, candidate['model_version_id'])
        live = dict(public_identity(instance.manifest), contract=instance.manifest['contract'],
                    model_version_id=instance.manifest['model_version_id'])
        persisted = dict(public_identity(manifest), contract=manifest['contract'], model_version_id=manifest['model_version_id'])
        fields = (*IDENTITY_FIELDS, 'model_version_id', 'artifact_sha256')
        if (instance.deployment_id != deployment_id
                or any(candidate.get(k) != live.get(k) or live.get(k) != persisted.get(k) for k in fields)):
            raise _integrity()
        return candidate

    def _recheck_candidate(self, db, tenant, candidate):
        deployment = self.store._owned(db, 'deployments', tenant, candidate['id'])
        model = self.store._owned(db, 'models', tenant, deployment['model_version_id'])
        from .dataset_serving import public_identity
        from .model_imports import validate_manifest
        manifest = json.loads(model['manifest_json'])
        validate_manifest(db, model, manifest)
        current = dict(public_identity(manifest), contract=manifest['contract'], model_version_id=model['id'])
        if deployment['desired'] != 'running' or deployment['observed'] != 'ready':
            raise Problem(503, 'not_ready', 'Deployment is not actually ready')
        if (model['artifact_sha256'] != candidate['artifact_sha256']
                or any(current.get(k) != candidate.get(k) for k in (*IDENTITY_FIELDS, 'model_version_id', 'artifact_sha256'))):
            raise _integrity()

    def create(self, tenant, key, body):
        object_fields(body, ('deployment_id',))
        _id(body['deployment_id'], 'deployment')
        text_value(key, 'Idempotency-Key', 128)
        return self._change(tenant, None, key, body)

    def switch(self, tenant, service_id, key, body):
        with self.supervisor.lock:
            with self.runtime.connection() as db:
                self._owned(db, tenant, service_id)
            object_fields(body, ('expected_revision', 'target_deployment_id'))
            if type(body['expected_revision']) is not int or not 1 <= body['expected_revision'] <= MAX_REVISION:
                raise Problem(400, 'invalid_input', 'Invalid expected revision')
            _id(body['target_deployment_id'], 'deployment')
            text_value(key, 'Idempotency-Key', 128)
            return self._change(tenant, service_id, key, body)

    def _change(self, tenant, service_id, key, body):
        request_hash = digest({'operation': 'switch' if service_id else 'create', 'service_id': service_id, 'body': body})
        with self.supervisor.lock:
            with self.runtime.connection() as db:
                db.execute('BEGIN')
                replay = self._replay(db, tenant, key, request_hash)
                if replay is not None:
                    return replay, False
                if service_id:
                    service = self._owned(db, tenant, service_id)
                    self._check_switch(service, body)
            candidate = self._candidate(tenant, body['target_deployment_id'] if service_id else body['deployment_id'])
            identity = _identity(candidate)
            with self.runtime.connection() as db:
                db.execute('BEGIN IMMEDIATE')
                replay = self._replay(db, tenant, key, request_hash)
                if replay is not None:
                    return replay, False
                stamp = now()
                if service_id:
                    service = self._owned(db, tenant, service_id)
                    self._check_switch(service, body)
                    self._snapshot(db, tenant, service_id, service['revision'])
                    if json.loads(service['identity_json']) != identity:
                        raise Problem(409, 'service_incompatible', 'Deployment serving identity differs')
                    revision, previous, created = service['revision'] + 1, service['active_deployment_id'], service['created_at']
                else:
                    service_id, revision, previous, created = identifier('service'), 1, None, stamp
                self._recheck_candidate(db, tenant, candidate)
                snapshot = dict(id=service_id, revision=revision, active_deployment_id=candidate['id'],
                                previous_deployment_id=previous, created_at=created, updated_at=stamp,
                                endpoint=f'/services/{service_id}/predict',
                                **{k: candidate[k] for k in ('model_version_id', 'artifact_sha256', 'contract', 'class_names', 'serving_protocol')})
                if revision == 1:
                    db.execute('INSERT INTO model_services VALUES(?,?,?,?,?,?,?,?)',
                               (service_id, tenant, revision, candidate['id'], previous, encode(identity).decode(), created, stamp))
                else:
                    db.execute('UPDATE model_services SET revision=?,active_deployment_id=?,previous_deployment_id=?,updated_at=? WHERE id=? AND tenant=? AND revision=?',
                               (revision, candidate['id'], previous, stamp, service_id, tenant, revision - 1))
                db.execute('INSERT INTO model_service_revisions VALUES(?,?,?,?,?)',
                           (service_id, revision, tenant, encode(snapshot).decode(), stamp))
                db.execute('INSERT INTO model_service_keys VALUES(?,?,?,?,?,?)',
                           (tenant, key, request_hash, service_id, revision, stamp))
                self.store.audit(db, tenant, 'switch_service' if revision > 1 else 'create_service', service_id)
                return snapshot, True

    @staticmethod
    def _check_switch(service, body):
        if (service['revision'] != body['expected_revision'] or service['revision'] >= MAX_REVISION
                or service['active_deployment_id'] == body['target_deployment_id']):
            raise Problem(409, 'service_conflict', 'Service revision or target conflicts')

    def _current(self, tenant, service_id):
        with self.runtime.connection() as db:
            db.execute('BEGIN')
            service = self._owned(db, tenant, service_id)
            return self._snapshot(db, tenant, service_id, service['revision'])

    def query(self, tenant, service_id):
        with self.supervisor.lock:
            snapshot = self._current(tenant, service_id)
            deployment = self.supervisor.query(tenant, snapshot['active_deployment_id'])
            result = dict(snapshot, **{k: deployment.get(k) for k in ('ready', 'checked_at', 'error_code', 'desired', 'observed')})
            instance = self.supervisor.instances.get(snapshot['active_deployment_id'])
            if instance is None:
                result['ready'] = False
            if result['ready']:
                from .dataset_serving import public_identity
                live = dict(public_identity(instance.manifest),
                            contract=instance.manifest['contract'],
                            model_version_id=instance.manifest['model_version_id'])
                expected = {k: snapshot[k] for k in ('model_version_id', 'artifact_sha256',
                            'contract', 'class_names', 'serving_protocol')}
                expected['recipe'] = 'yolox-nano-dataset/v1'
                if (instance.deployment_id != snapshot['active_deployment_id']
                        or any(deployment.get(k) != v or live.get(k) != v for k, v in expected.items())):
                    raise _integrity()
            return result

    def predict(self, tenant, service_id, features):
        with self.supervisor.lock:
            snapshot = self._current(tenant, service_id)
            result = self.supervisor.predict(tenant, snapshot['active_deployment_id'], features)
            expected = {k: snapshot[k] for k in ('model_version_id', 'artifact_sha256', 'contract', 'class_names', 'serving_protocol')}
            expected.update(deployment_id=snapshot['active_deployment_id'], recipe='yolox-nano-dataset/v1')
            if any(result.get(k) != value for k, value in expected.items()):
                raise Problem(503, 'not_ready', 'Serving identity does not match service binding')
            return dict(result, service_id=service_id, service_revision=snapshot['revision'])

    def list_ids(self, tenant):
        return self.list_page(tenant)['ids']

    def list_page(self, tenant, limit=None, cursor=None):
        state = unpack_cursor(self.store.cursor_key, cursor, tenant, 'services') if cursor is not None else None
        limit = (state['limit'] if state else 100) if limit is None else limit
        if type(limit) is not int or not 1 <= limit <= 100:
            raise Problem(400, 'invalid_input', 'Limit must be 1..100')
        if state and limit != state['limit']:
            raise Problem(400, 'invalid_cursor', 'Cursor page size cannot change')
        with self.runtime.connection() as db:
            db.execute('BEGIN')
            high = state['high'] if state else db.execute('SELECT COALESCE(MAX(rowid),0) FROM model_services').fetchone()[0]
            sql, args = 'SELECT id,created_at FROM model_services WHERE tenant=? AND rowid<=?', [tenant, high]
            if state:
                sql += ' AND (created_at,id)<(?,?)'
                args.extend(state['last'])
            rows = db.execute(sql + ' ORDER BY created_at DESC,id DESC LIMIT ?', (*args, limit + 1)).fetchall()
        cursor = None
        if len(rows) > limit:
            last = rows[limit - 1]
            cursor = pack_cursor(self.store.cursor_key, dict(v=1, tenant=hashlib.sha256(tenant.encode()).hexdigest(),
                                 collection='services', limit=limit, high=high, last=[last['created_at'], last['id']]))
        return dict(ids=[row['id'] for row in rows[:limit]], limit=limit, next_cursor=cursor)
