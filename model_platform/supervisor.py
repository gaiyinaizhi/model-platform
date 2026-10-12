"""Local process lifecycle with actual inference probes and durable desired state."""
import json
import os
import secrets
import socket
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass

from .common import Problem, encode


@dataclass
class Instance:
    process: subprocess.Popen
    port: int
    capability: str
    manifest: dict


class Supervisor:
    def __init__(self, store):
        self.store = store
        self.instances = {}
        self.lock = threading.RLock()
        self.done = threading.Event()
        # Never trust persisted PID or ready status after ownership changes.
        for row in store.deployment_rows(pending_only=True):
            store.observe(row["id"], "starting" if row["desired"] == "running" else "stopped")
        self.thread = threading.Thread(target=self._loop, name="serving-supervisor", daemon=True)
        self.thread.start()

    @staticmethod
    def call(instance, xs):
        request = urllib.request.Request(f"http://127.0.0.1:{instance.port}/predict", data=encode({"features": xs}),
                                         headers={"Authorization": "Bearer " + instance.capability, "Content-Type": "application/json"})
        # Explicit no proxy: internal serving calls never leave loopback.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(request, timeout=2) as response:
            raw = response.read(65537)
            if len(raw) > 65536:
                raise ValueError("Internal response too large")
            result = json.loads(raw)
        if (result.get("model_version_id") != instance.manifest["model_version_id"] or
                result.get("artifact_sha256") != instance.manifest["artifact_sha256"] or
                result.get("contract") != instance.manifest["contract"]):
            raise ValueError("Serving version mismatch")
        return result

    def _spawn(self, row):
        manifest = self.store.model(row["tenant"], row["model_version_id"])
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(("127.0.0.1", 0))
            sock.listen(8)
            port = sock.getsockname()[1]
            capability = secrets.token_hex(32)
            env = os.environ.copy()
            env["MP_HOME"] = str(self.store.runtime.home)
            # Child exception output is suppressed; public errors are stable sanitized codes.
            process = subprocess.Popen([sys.executable, "-m", "model_platform", "serve", "--deployment", row["id"],
                                        "--fd", str(sock.fileno()), "--parent", str(os.getpid())],
                                       stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                       pass_fds=(sock.fileno(),), env=env)
            instance = Instance(process, port, capability, manifest)
            self.instances[row["id"]] = instance
            process.stdin.write((capability + "\n").encode())
            process.stdin.close()
            self.store.observe(row["id"], "starting", pid=process.pid)
            return instance
        finally:
            sock.close()

    def _probe(self, deployment_id, instance):
        if instance.process.poll() is not None:
            self.store.observe(deployment_id, "failed", "serving_exited")
            return False
        try:
            from .recipes import for_manifest
            spec = for_manifest(instance.manifest)
            result = self.call(instance, spec.readiness_input(instance.manifest["dimensions"]))
            spec.validate_readiness(result["predictions"])
            self.store.observe(deployment_id, "ready", pid=instance.process.pid)
            return True
        except Exception:
            self.store.observe(deployment_id, "unavailable", "probe_failed", instance.process.pid)
            return False

    def _terminate(self, deployment_id):
        instance = self.instances.get(deployment_id)
        if instance:
            if instance.process.poll() is None:
                instance.process.terminate()
                try:
                    instance.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    instance.process.kill()
                    instance.process.wait(timeout=3)
            del self.instances[deployment_id]
        self.store.observe(deployment_id, "stopped")

    def _loop(self):
        while not self.done.is_set():
            with self.lock:
                # Completed stopped history must not cause repeated write
                # transactions or compete with live deployments and training.
                for row in self.store.deployment_rows(pending_only=True):
                    try:
                        if row["desired"] == "stopped":
                            self._terminate(row["id"])
                            continue
                        instance = self.instances.get(row["id"])
                        if not instance:
                            # One launch per supervisor lifetime; avoid restart storms.
                            if row["observed"] == "failed":
                                continue
                            instance = self._spawn(row)
                        self._probe(row["id"], instance)
                    except Exception:
                        self.store.observe(row["id"], "failed", "startup_failed")
            self.done.wait(0.5)

    def query(self, tenant, deployment_id):
        with self.lock:
            row = self.store.deployment(tenant, deployment_id)
            instance = self.instances.get(deployment_id)
            if row["desired"] == "running" and instance:
                self._probe(deployment_id, instance)
            return self.store.deployment(tenant, deployment_id)

    def predict(self, tenant, deployment_id, xs):
        with self.lock:
            row = self.store.deployment(tenant, deployment_id)
            instance = self.instances.get(deployment_id)
            if row["desired"] != "running" or not instance or not self._probe(deployment_id, instance):
                raise Problem(503, "not_ready", "Deployment is not actually ready")
            from .recipes import for_manifest
            for_manifest(instance.manifest).validate_features(xs, instance.manifest["dimensions"])
            try:
                return self.call(instance, xs)
            except Exception:
                self.store.observe(deployment_id, "unavailable", "inference_failed", instance.process.pid)
                raise Problem(503, "inference_failed", "Serving call failed") from None

    def stop(self, tenant, deployment_id):
        with self.lock:
            self.store.stop_desired(tenant, deployment_id)
            try:
                self._terminate(deployment_id)
            except Exception:
                self.store.observe(deployment_id, "failed", "stop_failed")
                raise Problem(503, "stop_failed", "Stop requested; process termination unconfirmed") from None
            return self.store.deployment(tenant, deployment_id)

    def close(self):
        self.done.set()
        self.thread.join(timeout=12)
        with self.lock:
            for deployment_id in list(self.instances):
                self._terminate(deployment_id)
