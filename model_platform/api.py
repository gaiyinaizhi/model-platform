"""Authenticated local API for owned technical resources; no training here."""
import collections
import hashlib
import hmac
import json
import threading
import time
from urllib.parse import urlsplit, parse_qsl

from .common import (CONTRACT, MAX_ACTIVE_RUNS, MAX_DEPLOYMENTS, Problem,
                     RECIPE, object_fields, text_value, training_request)
from .http_support import BoundedHTTPServer, JsonHandler
from .recipes import recipe_ids, public_catalog
from .store import Store
from .datasets import DatasetStore
from .dataset_validation import MAX_ARCHIVE_BYTES
from .supervisor import Supervisor


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
                if url.query:
                    if method != "GET" or len(parts) != 1 or parts[0] not in ("runs", "models", "deployments"):
                        raise Problem(400, "invalid_input", "Query parameters unsupported on this route")
                    if len(url.query) > 2200:
                        raise Problem(400, "invalid_input", "Query too long")
                    try:
                        pairs = parse_qsl(url.query, keep_blank_values=True, strict_parsing=True, max_num_fields=2)
                    except ValueError:
                        raise Problem(400, "invalid_input", "Invalid pagination query") from None
                    query = dict(pairs)
                    if len(query) != len(pairs) or set(query) - {"limit", "cursor"}:
                        raise Problem(400, "invalid_input", "Unknown or duplicate query fields")
                    limit = query.get("limit")
                    if limit is not None and (not limit.isascii() or not limit.isdecimal() or not 1 <= len(limit) <= 3):
                        raise Problem(400, "invalid_input", "Limit must be 1..100")
                    status, result = 200, store.list_page(parts[0], tenant,
                        int(limit) if limit is not None else None, query.get("cursor"))
                else:
                    status, result = self.route(method, parts, tenant)
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

        def route(self, method, parts, tenant):
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
                        raise Problem(501, "cancel_unsupported", "Training cancellation is not implemented")
                    if action == "retry":
                        object_fields(self.body(), ())
                        store.retry(tenant, object_id)
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
