"""Authenticated local API for owned technical resources; no training here."""
import collections
import hashlib
import hmac
import json
import threading
import time
from urllib.parse import urlsplit, parse_qsl

from .common import (CONTRACT, ID, MAX_ACTIVE_RUNS, MAX_DEPLOYMENTS, Problem,
                     RECIPE, object_fields, text_value, training_request)
from .http_support import BoundedHTTPServer, JsonHandler
from .recipes import recipe_ids, public_catalog
from .store import Store
from .datasets import DatasetStore
from .dataset_validation import MAX_ARCHIVE_BYTES
from .supervisor import Supervisor
from .delivery import Delivery, CHUNK


class Auth:
    def __init__(self, path):
        raw = json.loads(path.read_text())
        if not isinstance(raw, dict) or not raw:
            raise RuntimeError("Initialize local credentials first")
        self.tokens = {}
        for tenant, token in raw.items():
            text_value(tenant, "tenant")
            if not isinstance(token, str) or len(token) != 64 or any(c not in "0123456789abcdef" for c in token):
                raise RuntimeError("Invalid local token")
            hashed = hashlib.sha256(token.encode()).digest()
            if hashed in self.tokens:
                raise RuntimeError("Tokens must be unique per tenant")
            self.tokens[hashed] = tenant
        self.windows = collections.defaultdict(collections.deque)
        self.lock = threading.Lock()

    def authenticate(self, header):
        if not header.startswith("Bearer ") or len(header) != 71:
            raise Problem(401, "unauthorized", "Bearer authorization required")
        hashed = hashlib.sha256(header[7:].encode()).digest()
        tenant = next((value for key, value in self.tokens.items() if hmac.compare_digest(hashed, key)), None)
        if not tenant:
            raise Problem(401, "unauthorized", "Invalid authorization")
        with self.lock:
            stamp = time.monotonic()
            window = self.windows[tenant]
            while window and window[0] < stamp - 60:
                window.popleft()
            if len(window) >= 120:
                raise Problem(429, "rate_limit", "120 requests per minute per tenant")
            window.append(stamp)
        return tenant


def make_handler(store, supervisor, auth):
    datasets = DatasetStore(store.runtime)
    delivery = Delivery(store)
    from .model_imports import ModelImportStore, catalog, ASSET_BYTES
    imports = ModelImportStore(store)
    from .model_evaluations import ModelEvaluationStore
    evaluations = ModelEvaluationStore(store)

    class Handler(JsonHandler):
        def do_GET(self):
            self.dispatch("GET")

        def do_POST(self):
            self.dispatch("POST")

        def dispatch(self, method):
            try:
                if len(self.headers.get_all("Authorization", [])) != 1:
                    raise Problem(401, "unauthorized", "One authorization header required")
                tenant = auth.authenticate(self.headers.get("Authorization", ""))
                url = urlsplit(self.path)
                if url.fragment or "%" in url.path:
                    raise Problem(400, "invalid_input", "Fragments and encoded paths are unsupported")
                parts = url.path.strip("/").split("/")
                if 'training-report' in url.path.split('/'):
                    if (method != 'GET' or url.scheme or url.netloc or '?' in self.path
                            or self.requestline.split()[1] != self.path
                            or len(parts) != 3 or parts[0] != 'models'
                            or parts[2] != 'training-report' or self.path != '/' + '/'.join(parts)
                            or any(self.headers.get_all(key, []) for key in
                                   ('Range', 'If-Range', 'Transfer-Encoding', 'Content-Encoding'))
                            or self.headers.get_all('Content-Length', []) not in ([], ['0'])):
                        raise Problem(400, 'invalid_input', 'Use the exact body-free report route without query or range')
                    from .training_reports import report as training_report
                    content = training_report(store, tenant, parts[1])
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Content-Disposition', 'attachment; filename="training-evaluation-report.json"')
                    self.send_header('Content-Length', str(len(content)))
                    self.send_header('Cache-Control', 'no-store')
                    self.send_header('X-Content-Type-Options', 'nosniff')
                    self.send_header('Connection', 'close')
                    self.end_headers()
                    self.close_connection = True
                    try:
                        self.wfile.write(content)
                    except OSError:
                        pass
                    return
                if 'report' in url.path.split('/'):
                    if (method != 'GET' or url.scheme or url.netloc or '?' in self.path
                            or self.requestline.split()[1] != self.path
                            or len(parts) != 3 or parts[0] != 'model-evaluations'
                            or parts[2] != 'report' or self.path != '/' + '/'.join(parts)
                            or any(self.headers.get_all(key, []) for key in
                                   ('Range', 'If-Range', 'Transfer-Encoding', 'Content-Encoding'))
                            or self.headers.get_all('Content-Length', []) not in ([], ['0'])):
                        raise Problem(400, 'invalid_input', 'Use the exact body-free report route without query or range')
                    from .evaluation_reports import report
                    content = report(evaluations, tenant, parts[1])
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Content-Disposition', 'attachment; filename="independent-evaluation-report.json"')
                    self.send_header('Content-Length', str(len(content)))
                    self.send_header('Cache-Control', 'no-store')
                    self.send_header('X-Content-Type-Options', 'nosniff')
                    self.send_header('Connection', 'close')
                    self.end_headers()
                    self.close_connection = True
                    try:
                        self.wfile.write(content)
                    except OSError:
                        pass
                    return
                if 'error-images' in url.path.split('/'):
                    if (method != 'GET' or url.scheme or url.netloc or '?' in self.path
                            or self.requestline.split()[1] != self.path
                            or len(parts) != 4 or parts[0] != 'model-evaluations'
                            or parts[2] != 'error-images'
                            or self.path != '/' + '/'.join(parts)
                            or any(self.headers.get_all(key, []) for key in
                                   ('Range', 'If-Range', 'Transfer-Encoding', 'Content-Encoding'))
                            or self.headers.get_all('Content-Length', []) not in ([], ['0'])):
                        raise Problem(400, 'invalid_input', 'Use the exact body-free error-image route without query or range')
                    from .evaluation_images import preview
                    content = preview(evaluations, tenant, parts[1], parts[3])
                    self.send_response(200)
                    self.send_header('Content-Type', 'image/png')
                    self.send_header('Content-Length', str(len(content)))
                    self.send_header('Cache-Control', 'no-store')
                    self.send_header('X-Content-Type-Options', 'nosniff')
                    self.send_header('Cross-Origin-Resource-Policy', 'same-origin')
                    self.send_header('Connection', 'close')
                    self.end_headers()
                    self.close_connection = True
                    try:
                        self.wfile.write(content)
                    except OSError:
                        # A binary success can never be followed by JSON.
                        pass
                    return
                if url.query:
                    if method != "GET" or len(parts) != 1 or parts[0] not in ("runs", "models", "deployments", "model-evaluations", "model-evaluation-comparisons"):
                        raise Problem(400, "invalid_input", "Query parameters unsupported on this route")
                    if len(url.query) > 2200:
                        raise Problem(400, "invalid_input", "Query too long")
                    try:
                        pairs = parse_qsl(url.query, keep_blank_values=True, strict_parsing=True, max_num_fields=2)
                    except ValueError:
                        raise Problem(400, "invalid_input", "Invalid pagination query") from None
                    query = dict(pairs)
                    if parts == ["model-evaluation-comparisons"]:
                        if len(query) != len(pairs) or set(query) != {"evaluation_a_id", "evaluation_b_id"}:
                            raise Problem(400, "invalid_input", "Use exactly evaluation_a_id and evaluation_b_id")
                        from .evaluation_comparisons import compare
                        self.respond(200, compare(evaluations, tenant, query["evaluation_a_id"], query["evaluation_b_id"]))
                        return
                    if parts == ["model-evaluations"]:
                        if len(query) != len(pairs) or set(query) != {"model_version_id"}:
                            raise Problem(400, "invalid_input", "Use exactly one model_version_id filter")
                        result = evaluations.list(tenant, text_value(query["model_version_id"], "model_version_id", 80))
                        self.respond(200, result)
                        return
                    if len(query) != len(pairs) or set(query) - {"limit", "cursor"}:
                        raise Problem(400, "invalid_input", "Unknown or duplicate query fields")
                    limit = query.get("limit")
                    if limit is not None and (not limit.isascii() or not limit.isdecimal() or not 1 <= len(limit) <= 3):
                        raise Problem(400, "invalid_input", "Limit must be 1..100")
                    status, result = 200, store.list_page(parts[0], tenant,
                        int(limit) if limit is not None else None, query.get("cursor"))
                else:
                    status, result = self.route(method, parts, tenant)
                if result is not None:
                    self.respond(status, result)
            except Problem as exc:
                self.handle_problem(exc)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception:
                self.handle_problem(Problem(500, "internal_error", "Internal operation failed"))

        def archive_body(self):
            if self.headers.get("Transfer-Encoding"):
                raise Problem(400, "invalid_input", "Chunked requests are unsupported")
            lengths = self.headers.get_all("Content-Length", [])
            if len(lengths) != 1 or not lengths[0].isascii() or not lengths[0].isdecimal():
                raise Problem(400, "invalid_input", "One decimal Content-Length required")
            if len(lengths[0]) > 10 or not 0 < int(lengths[0]) <= MAX_ARCHIVE_BYTES:
                raise Problem(413, "body_limit", "ZIP upload must be 1 byte to 32 MiB")
            if len(self.headers.get_all("Content-Type", [])) != 1 or self.headers.get_content_type() != "application/zip":
                raise Problem(400, "invalid_input", "application/zip required")
            if self.headers.get("Content-Encoding"):
                raise Problem(400, "invalid_input", "Content-Encoding is unsupported")
            import socket
            size = int(lengths[0])
            try:
                raw = self.rfile.read(size)
                if len(raw) != size:
                    raise ValueError("Truncated ZIP")
            except (ValueError, socket.timeout):
                raise Problem(400, "invalid_archive_body", "Incomplete ZIP upload; retry with the same import ID") from None
            return raw

        def import_artifact_body(self):
            if self.headers.get('Transfer-Encoding') or self.headers.get('Content-Encoding'):
                raise Problem(400, 'invalid_input', 'Chunked or compressed uploads are unsupported')
            lengths = self.headers.get_all('Content-Length', [])
            if len(lengths) != 1 or lengths[0] != str(ASSET_BYTES):
                raise Problem(413, 'body_limit', 'Exact catalog artifact Content-Length required')
            if len(self.headers.get_all('Content-Type', [])) != 1 or self.headers.get_content_type() != 'application/octet-stream':
                raise Problem(400, 'invalid_input', 'application/octet-stream required')
            from .model_imports import read_artifact_body
            return read_artifact_body(self.rfile, self.connection, ASSET_BYTES)

        def route(self, method, parts, tenant):
            if parts == ["model-evaluation-comparisons"]:
                raise Problem(400, "invalid_input", "Use GET with evaluation_a_id and evaluation_b_id")
            if parts == ["model-evaluations"]:
                if method == "GET":
                    return 200, evaluations.list(tenant)
                if method == "POST":
                    keys = self.headers.get_all("Idempotency-Key", [])
                    if len(keys) != 1:
                        raise Problem(400, "invalid_input", "One Idempotency-Key required")
                    result, created = evaluations.create(tenant, keys[0], self.body())
                    return (202 if created else 200), result
            if len(parts) == 2 and parts[0] == "model-evaluations" and method == "GET":
                return 200, evaluations.get(tenant, parts[1])
            if parts == ['model-import-catalog'] and method == 'GET':
                return 200, catalog()
            if parts == ['model-imports'] and method == 'POST':
                keys = self.headers.get_all('Idempotency-Key', [])
                if len(keys) != 1:
                    raise Problem(400, 'invalid_input', 'One Idempotency-Key required')
                result, created = imports.create(tenant, keys[0], self.body())
                return (201 if created else 200), result
            if len(parts) == 2 and parts[0] == 'model-imports' and method == 'GET':
                return 200, imports.get_import(tenant, parts[1])
            if len(parts) == 3 and parts[0] == 'model-imports' and method == 'POST':
                if parts[2] == 'artifact':
                    return 200, imports.upload(tenant, parts[1], self.import_artifact_body)
                if parts[2] == 'register':
                    object_fields(self.body(), ())
                    result, created = imports.register(tenant, parts[1])
                    return (201 if created else 200), result
            if method == "GET" and parts == ["worker-readiness"]:
                from .worker_readiness import public_report
                return 200, public_report(store, tenant)
            if method == "GET" and len(parts) == 3 and parts[0] == "models":
                if parts[2] == "delivery":
                    return 200, delivery.manifest(tenant, parts[1])
                if parts[2] == "artifact":
                    # Range/conditional caching is deliberately unsupported in v1.
                    if self.headers.get("Range") or self.headers.get("If-Range"):
                        raise Problem(400, "range_unsupported", "Partial artifact downloads are unsupported")
                    with delivery.snapshot(tenant, parts[1]) as (model, stream):
                        self.send_response(200)
                        self.send_header("Content-Type", "application/octet-stream")
                        self.send_header("Content-Length", str(model['artifact_bytes']))
                        self.send_header("X-Artifact-SHA256", model['artifact_sha256'])
                        self.send_header("ETag", '"' + model['artifact_sha256'] + '"')
                        self.send_header("Content-Disposition", 'attachment; filename="' + model['model_version_id'] + '.onnx"')
                        self.send_header("Cache-Control", "no-store")
                        self.send_header("X-Content-Type-Options", "nosniff")
                        self.send_header("Connection", "close")
                        self.end_headers()
                        self.close_connection = True
                        try:
                            while chunk := stream.read(CHUNK):
                                self.wfile.write(chunk)
                        except OSError:
                            # Never append a JSON error after binary success headers.
                            pass
                    return 200, None
            if parts == ["model-comparisons"] and method == "POST":
                body = self.body()
                object_fields(body, ('model_version_ids',))
                return 200, store.compare_models(tenant, body['model_version_ids'])
            if parts == ["dataset-imports"] and method == "POST":
                keys = self.headers.get_all("Idempotency-Key", [])
                if len(keys) != 1:
                    raise Problem(400, "invalid_input", "One Idempotency-Key required")
                result, created = datasets.create(tenant, keys[0], self.body())
                return (201 if created else 200), result
            if parts == ["dataset-versions"] and method == "GET":
                return 200, datasets.list_versions(tenant)
            if len(parts) == 2 and method == "GET":
                if parts[0] == "dataset-imports":
                    return 200, datasets.get_import(tenant, parts[1])
                if parts[0] == "dataset-versions":
                    return 200, datasets.get_version(tenant, parts[1])
            if len(parts) == 3 and parts[0] == "dataset-imports" and method == "POST":
                if parts[2] == "archive":
                    return 200, datasets.upload(tenant, parts[1], self.archive_body)
                if parts[2] == "publish":
                    object_fields(self.body(), ())
                    result, created = datasets.publish(tenant, parts[1])
                    return (201 if created else 200), result
            if method == "GET" and parts == ["recipes"]:
                return 200, {"recipes": public_catalog()}
            if method == "GET" and parts == ["health"]:
                return 200, {"status": "ok", "recipe": RECIPE, "contract": CONTRACT, "cancel_supported": False,
                             "queued_cancel_supported": True,
                             "conditional_training_retry_supported": True,
                             "recipes": recipe_ids(),
                             "budgets": {"active_runs": MAX_ACTIVE_RUNS, "deployments": MAX_DEPLOYMENTS}}
            if len(parts) == 1 and parts[0] in ("runs", "models", "deployments"):
                if method == "GET":
                    return 200, {"ids": store.list_ids(parts[0], tenant), "limit": 100}
                if parts[0] == "runs" and method == "POST":
                    keys = self.headers.get_all("Idempotency-Key", [])
                    if len(keys) != 1:
                        raise Problem(400, "invalid_input", "One Idempotency-Key required")
                    key = text_value(keys[0], "Idempotency-Key", 128)
                    body = training_request(self.body())
                    run_id, created = store.submit(tenant, key, body)
                    return (202 if created else 200), store.run(tenant, run_id)
                if parts[0] == "deployments" and method == "POST":
                    body = self.body()
                    object_fields(body, ("model_version_id",))
                    model_id = text_value(body["model_version_id"], "model_version_id")
                    keys = self.headers.get_all("Idempotency-Key", [])
                    if len(keys) > 1:
                        raise Problem(400, "invalid_input", "At most one Idempotency-Key allowed")
                    key = text_value(keys[0], "Idempotency-Key", 128) if keys else None
                    deployment_id, created = store.submit_deployment(tenant, model_id, key)
                    return (202 if created else 200), store.deployment(tenant, deployment_id)
            if len(parts) == 2 and method == "GET":
                collection, object_id = parts
                if collection == "runs":
                    return 200, store.run(tenant, object_id)
                if collection == "models":
                    return 200, store.model(tenant, object_id)
                if collection == "deployments":
                    return 200, supervisor.query(tenant, object_id)
            if len(parts) == 3 and method == "POST":
                collection, object_id, action = parts
                if collection == "runs":
                    store.run(tenant, object_id)
                    if action == "cancel":
                        object_fields(self.body(), ())
                        store.cancel(tenant, object_id)
                        return 200, store.run(tenant, object_id)
                    if action == "retry":
                        body = self.body()
                        object_fields(body, (), ("expected_attempt_id",))
                        expected = body.get("expected_attempt_id")
                        if "expected_attempt_id" in body and (
                                not isinstance(expected, str) or not expected.startswith("attempt_")
                                or not ID.fullmatch(expected)):
                            raise Problem(400, "invalid_input", "Expected a failed attempt ID")
                        store.retry(tenant, object_id, expected)
                        return 202, store.run(tenant, object_id)
                if collection == "deployments":
                    if action == "stop":
                        object_fields(self.body(), ())
                        return 200, supervisor.stop(tenant, object_id)
                    if action == "predict":
                        body = self.body()
                        object_fields(body, ("features",))
                        return 200, supervisor.predict(tenant, object_id, body["features"])
            raise Problem(404, "not_found", "Route not found")
    return Handler


def run_api(runtime, port):
    with runtime.lock("api"):
        store = Store(runtime)
        from .model_imports import ModelImportStore
        ModelImportStore(store).recover()
        auth = Auth(runtime.home / "credentials.json")
        supervisor = Supervisor(store)
        server = None
        try:
            server = BoundedHTTPServer(("127.0.0.1", port), make_handler(store, supervisor, auth))
            print(f"api listening on http://127.0.0.1:{server.server_address[1]}", flush=True)
            server.serve_forever(poll_interval=0.25)
        finally:
            if server:
                server.server_close()
            supervisor.close()
