"""One loopback-only inference process, loaded from a registered immutable version."""
import hmac
import json
import os
import socket
import sys
from http.server import HTTPServer

from .common import CONTRACT, Problem, object_fields
from .http_support import JsonHandler
from .store import Store


def serve(runtime, deployment_id, fd, parent_pid):
    capability = sys.stdin.readline().strip()
    if len(capability) != 64 or os.getppid() != parent_pid:
        raise RuntimeError("Invalid serving parent capability")
    store = Store(runtime)
    with runtime.connection() as db:
        row = db.execute("SELECT tenant,model_version_id FROM deployments WHERE id=? AND desired='running'", (deployment_id,)).fetchone()
    if not row:
        raise RuntimeError("Deployment no longer requested")
    manifest = store.model(row["tenant"], row["model_version_id"])
    from .recipes import for_manifest
    spec = for_manifest(manifest)
    if manifest["artifact_id"] != manifest["attempt_id"]:
        raise ValueError("Invalid artifact binding")
    model = spec.load(runtime.artifact(manifest["attempt_id"]), manifest["artifact_sha256"])
    if any(model[key] != manifest[key] for key in ("recipe", "contract", "format", "dimensions")):
        raise ValueError("Model manifest contract mismatch")

    if manifest.get('recipe') == 'yolox-nano-dataset/v1':
        from .dataset_serving import validate_identity
        validate_identity(manifest, model)

    class Handler(JsonHandler):
        def do_POST(self):
            try:
                if self.path != "/predict":
                    raise Problem(404, "not_found", "Route not found")
                token = self.headers.get("Authorization", "")
                if not hmac.compare_digest(token.encode(), ("Bearer " + capability).encode()):
                    raise Problem(401, "unauthorized", "Internal authorization required")
                # A child orphan must never accept calls after parent death.
                if os.getppid() != parent_pid:
                    raise Problem(503, "parent_gone", "Supervisor no longer alive")
                body = self.body()
                object_fields(body, ("features",))
                result = spec.predict(model, body["features"])
                identity = {}
                if manifest.get("recipe") == "yolox-nano-dataset/v1":
                    from .dataset_serving import public_identity
                    identity = public_identity(manifest)
                self.respond(200, {**identity, "deployment_id": deployment_id, "model_version_id": manifest["model_version_id"],
                                   "artifact_sha256": manifest["artifact_sha256"], "contract": model["contract"], "predictions": result})
            except Problem as exc:
                self.handle_problem(exc)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception:
                self.handle_problem(Problem(503, "inference_failed", "Inference failed"))

    inherited = socket.socket(fileno=fd)
    server = HTTPServer(inherited.getsockname(), Handler, bind_and_activate=False)
    server.socket.close()
    server.socket = inherited
    server.server_address = inherited.getsockname()
    server.timeout = 0.5
    try:
        while os.getppid() == parent_pid:
            server.handle_request()
    finally:
        server.server_close()
