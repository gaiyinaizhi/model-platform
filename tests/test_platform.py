"""Real subprocess/HTTP/SQLite CPU acceptance, with no third-party dependencies."""
import concurrent.futures
import hashlib
import http.client
import json
import os
from pathlib import Path
import signal
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request

from model_platform.common import Runtime, digest
from model_platform.evaluation import evaluate
from model_platform.store import Store
from scripts.demo import request_body

ROOT = Path(__file__).resolve().parents[1]


class PlatformTests(unittest.TestCase):
    def setUp(self):
        local = ROOT / ".runtime" / "tests"
        local.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=local)
        self.home = Path(self.temp.name)
        self.env = dict(os.environ, MP_HOME=str(self.home))
        self.processes, self.logs = [], []
        self.addCleanup(self.cleanup)
        init = subprocess.run([sys.executable, "-m", "model_platform", "init", "--tenant", "demo", "--tenant", "other"],
                              env=self.env, cwd=ROOT, capture_output=True, text=True, timeout=10)
        self.assertEqual(init.returncode, 0, init.stderr)
        self.tokens = json.loads((self.home / "credentials.json").read_text())
        self.runtime = Runtime(self.home)
        self.store = Store(self.runtime)
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self.start_api()

    def spawn(self, *args, cwd=ROOT):
        log = (self.home / f"process-{len(self.processes)}.log").open("w")
        self.logs.append(log)
        p = subprocess.Popen([sys.executable, "-m", "model_platform", *args], env=self.env, cwd=cwd, stdout=log, stderr=log)
        self.processes.append(p)
        return p, Path(log.name)

    def start_api(self, cwd=ROOT):
        self.api, log_path = self.spawn("api", "--port", "0", cwd=cwd)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            output = log_path.read_text()
            if "api listening on " in output:
                self.base = output.split("api listening on ", 1)[1].splitlines()[0]
                self.assertEqual(self.call("/health")[0], 200)
                return
            if self.api.poll() is not None:
                self.fail("API failed: " + output)
            time.sleep(0.02)
        self.fail("API startup timeout")

    def cleanup(self):
        for p in reversed(self.processes):
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait(timeout=5)
        for log in self.logs:
            log.close()
        self.temp.cleanup()

    def call(self, path, body=None, key=None, tenant="demo", raw=None):
        headers = {"Authorization": "Bearer " + self.tokens[tenant]} if tenant else {}
        if key is not None:
            headers["Idempotency-Key"] = key
        data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
        if data is not None:
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(self.base + path, data=data, headers=headers)
        try:
            with self.opener.open(request, timeout=5) as response:
                return response.status, json.load(response)
        except urllib.error.HTTPError as exc:
            try:
                return exc.code, json.load(exc)
            finally:
                exc.close()

    def worker_once(self):
        p, path = self.spawn("worker", "--once")
        self.assertEqual(p.wait(timeout=15), 0, path.read_text())

    def submit(self, body=None, key="test"):
        status, run = self.call("/runs", body or request_body(), key)
        self.assertEqual(status, 202, run)
        return run

    def train(self, body=None, key="test"):
        run = self.submit(body, key)
        self.worker_once()
        status, run = self.call("/runs/" + run["id"])
        self.assertEqual(status, 200)
        self.assertEqual(run["status"], "succeeded", run)
        return run

    def ready(self, deployment_id):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            status, deployment = self.call("/deployments/" + deployment_id)
            self.assertEqual(status, 200)
            if deployment["ready"]:
                return deployment
            self.assertNotEqual(deployment["observed"], "failed", deployment)
            time.sleep(0.1)
        self.fail("Deployment startup timeout")

    def deploy(self, model_id):
        status, deployment = self.call("/deployments", {"model_version_id": model_id})
        self.assertEqual(status, 202, deployment)
        return self.ready(deployment["id"])

    def test_idempotency_concurrent_and_scoped(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            results = list(pool.map(lambda _: self.call("/runs", request_body(), "same"), range(6)))
        self.assertEqual(sum(status == 202 for status, _ in results), 1)
        ids = {row["id"] for _, row in results}
        self.assertEqual(len(ids), 1)
        changed = request_body()
        changed["config"]["epochs"] = 99
        self.assertEqual(self.call("/runs", changed, "same")[0], 409)
        status, other = self.call("/runs", request_body(), "same", "other")
        self.assertEqual(status, 202)
        self.assertNotIn(other["id"], ids)
        self.assertEqual(self.call("/runs/" + next(iter(ids)), tenant="other")[0], 404)
        self.assertEqual(self.call("/health", tenant=None)[0], 401)
        self.assertEqual(self.call("/runs", request_body())[0], 400)
        self.assertEqual(self.call("/runs")[1]["ids"], list(ids))

    def test_input_validation_and_transport_limits(self):
        cases = []
        bad = request_body(); bad["tenant"] = "other"; cases.append(bad)
        bad = request_body(); bad["dataset"]["rights_confirmed"] = False; cases.append(bad)
        bad = request_body(); bad["config"]["epochs"] = True; cases.append(bad)
        bad = request_body(); bad["config"]["epochs"] = 201; cases.append(bad)
        bad = request_body(); bad["dataset"]["train"][0]["x"] = [float("nan")]; cases.append(bad)
        bad = request_body(); bad["dataset"]["train"][0]["x"] = [True]; cases.append(bad)
        bad = request_body(); bad["dataset"]["train"][0]["x"] = [1e7]; cases.append(bad)
        bad = request_body(); bad["dataset"]["train"][0]["x"] = [1, 2]; cases.append(bad)
        bad = request_body(); bad["dataset"]["evaluation"][0] = bad["dataset"]["train"][0]; cases.append(bad)
        bad = request_body(); bad["dataset"]["url"] = "http://example.invalid"; cases.append(bad)
        bad = request_body(); bad["dataset"]["train"] = []; cases.append(bad)
        bad = request_body(); bad["config"]["learning_rate"] = "0.1"; cases.append(bad)
        bad = request_body(); bad["dataset"]["train"][0]["y"] = True; cases.append(bad)
        bad = request_body(); bad["dataset"]["train"][0]["x"] = [10**1000]; cases.append(bad)
        for i, body in enumerate(cases):
            with self.subTest(i=i):
                self.assertEqual(self.call("/runs", body, str(i))[0], 400)
        self.assertEqual(self.call("/runs", key="duplicate", raw=b'{"dataset":{},"dataset":{},"config":{}}')[0], 400)
        # Header-only oversized request verifies early rejection without forcing
        # clients to finish sending a body after the server has already closed.
        conn = http.client.HTTPConnection(self.base.removeprefix("http://"), timeout=5)
        try:
            conn.putrequest("POST", "/runs")
            conn.putheader("Authorization", "Bearer " + self.tokens["demo"])
            conn.putheader("Idempotency-Key", "large")
            conn.putheader("Content-Type", "application/json")
            conn.putheader("Content-Length", str(2 * 1024 * 1024 + 1))
            conn.endheaders()
            response = conn.getresponse()
            self.assertEqual(response.status, 413)
            response.read()
        finally:
            conn.close()
        self.assertEqual(self.call("/runs")[1]["ids"], [])

    def test_real_training_evaluation_lineage_and_immutable_versions(self):
        run = self.train()
        status, model = self.call("/models/" + run["model_version_id"])
        self.assertEqual(status, 200)
        self.assertEqual(model["run_id"], run["id"])
        self.assertEqual(model["attempt_id"], run["attempts"][0]["id"])
        path = self.runtime.artifact(model["attempt_id"])
        weights = json.loads(path.read_text())["weights"]
        self.assertTrue(any(w != 0 for w in weights), "Actual parameters must have changed")
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), model["artifact_sha256"])
        expected = evaluate(path, model["artifact_sha256"], request_body()["dataset"]["evaluation"])
        self.assertEqual(expected, model["evaluation"])
        self.assertEqual(model["evaluation"]["accuracy"], 1.0)
        self.assertGreater(model["evaluation"]["log_loss"], 0)
        self.assertEqual(model["dataset"]["train_sha256"], digest(request_body()["dataset"]["train"]))
        self.worker_once()
        self.assertEqual(self.call("/models")[1]["ids"], [model["model_version_id"]])
        self.assertEqual(self.call("/runs/" + run["id"] + "/retry", {})[0], 409)
        self.assertEqual(self.call("/runs/" + run["id"] + "/cancel", {})[0], 501)
        self.assertEqual(self.call("/models/" + model["model_version_id"], tenant="other")[0], 404)
        with self.runtime.connection() as db:
            for table in ("models", "evaluations"):
                with self.assertRaises(sqlite3.IntegrityError):
                    db.execute(f"UPDATE {table} SET created_at=0")
                with self.assertRaises(sqlite3.IntegrityError):
                    db.execute(f"DELETE FROM {table}")
            with self.assertRaises(sqlite3.IntegrityError):
                db.execute("UPDATE runs SET input_json='{}'")
        second = self.train(key="distinct")
        self.assertNotEqual(run["model_version_id"], second["model_version_id"])
        self.assertEqual(len(self.call("/models")[1]["ids"]), 2)

    def test_training_failure_retry_and_attempt_limit(self):
        body = request_body()
        body["dataset"]["train"] = [{"x": [0], "y": i % 2} for i in range(4)]
        run = self.submit(body)
        for number in range(1, 4):
            self.worker_once()
            result = self.call("/runs/" + run["id"])[1]
            self.assertEqual(result["status"], "failed")
            self.assertEqual(len(result["attempts"]), number)
            self.assertTrue(all(a["error_code"] == "degenerate_features" for a in result["attempts"]))
            self.assertTrue(all(a["logs"] for a in result["attempts"]))
            self.assertIsNone(result["model_version_id"])
            status, _ = self.call("/runs/" + run["id"] + "/retry", {})
            self.assertEqual(status, 202 if number < 3 else 409)
        self.assertEqual(self.call("/models")[1]["ids"], [])

    def test_holdout_larger_than_online_inference_limit(self):
        body = request_body()
        body["dataset"]["train"] = [{"x": [x], "y": int(x > 0)} for x in (-4, -2, 2, 4)]
        body["dataset"]["evaluation"] = [{"x": [i / 100 + 0.001], "y": int(i >= 0)}
                                            for i in range(-498, 498)]
        run = self.train(body)
        model = self.call("/models/" + run["model_version_id"])[1]
        self.assertEqual(model["evaluation"]["sample_count"], 996)
        self.assertEqual(sum(map(sum, model["evaluation"]["confusion"])), 996)
        self.assertEqual(model["evaluation"]["dataset_sha256"], digest(body["dataset"]["evaluation"]))
        deployment = self.deploy(run["model_version_id"])
        route = "/deployments/" + deployment["id"] + "/predict"
        self.assertEqual(self.call(route, {"features": [[1]] * 128})[0], 200)
        self.assertEqual(self.call(route, {"features": [[1]] * 129})[0], 400)

    def test_small_nonconstant_features_train_without_variance_underflow(self):
        body = request_body()
        for split in ("train", "evaluation"):
            for row in body["dataset"][split]:
                row["x"] = [value * 1e-200 for value in row["x"]]
        run = self.train(body)
        model = self.call("/models/" + run["model_version_id"])[1]
        self.assertEqual(model["evaluation"]["accuracy"], 1.0)

    def test_deployment_budget_released_by_stop(self):
        run = self.train()
        deployments = []
        for _ in range(4):
            status, deployment = self.call("/deployments", {"model_version_id": run["model_version_id"]})
            self.assertEqual(status, 202)
            deployments.append(deployment)
        self.assertEqual(self.call("/deployments", {"model_version_id": run["model_version_id"]})[0], 429)
        self.assertEqual(self.call("/deployments/" + deployments[0]["id"] + "/stop", {})[0], 200)
        self.assertEqual(self.call("/deployments", {"model_version_id": run["model_version_id"]})[0], 202)

    def test_busy_readiness_and_owned_queue_http_contract(self):
        from model_platform.worker_readiness import WorkerHeartbeat
        own = self.submit(key='own-queue')['id']
        foreign, _ = self.store.submit('other', 'private-queue', request_body())
        hb = WorkerHeartbeat(self.runtime, {}).start()
        self.addCleanup(hb.stop)
        hb.set_state('busy')
        code, report = self.call('/worker-readiness')
        self.assertEqual(code, 200)
        self.assertTrue(report['worker']['ready'])
        self.assertFalse(report['worker']['accepting_immediately'])
        self.assertEqual(report['worker']['reasons'], ['worker_busy'])
        self.assertEqual(report['queue'], {'queued': 1, 'running': 0})
        self.assertNotIn(foreign, json.dumps(report))
        code, run = self.call('/runs/' + own)
        self.assertEqual(code, 200)
        self.assertEqual(run['queue'], {'scope': 'tenant', 'position': 1})
        self.assertIsNone(run['model_version_id'])
        self.assertNotIn('evaluation', run)
        self.assertEqual(self.call('/runs/' + foreign)[0], 404)
        claimed, attempt = self.store.claim()
        self.store.fail(claimed['id'], attempt, 'evaluation_failed')
        code, failed = self.call('/runs/' + claimed['id'], tenant=claimed['tenant'])
        self.assertIsNone(failed['queue']['position'])
        self.assertIsNone(failed['model_version_id'])
        self.assertEqual(failed['attempts'][0]['error_code'], 'evaluation_failed')

    def test_actual_serving_readiness_stop_and_tenant_isolation(self):
        run = self.train()
        deployment = self.deploy(run["model_version_id"])
        self.assertIsInstance(deployment["pid"], int)
        self.assertGreater(deployment["checked_at"], 0)
        route = "/deployments/" + deployment["id"]
        status, output = self.call(route + "/predict", {"features": [[-2.5], [2.5]]})
        self.assertEqual(status, 200)
        self.assertEqual([p["class"] for p in output["predictions"]], [0, 1])
        self.assertEqual(output["model_version_id"], run["model_version_id"])
        self.assertEqual(self.call(route + "/predict", {"features": [[1, 2]]})[0], 400)
        self.assertEqual(self.call(route + "/predict", {"features": [[1]]}, tenant="other")[0], 404)
        self.assertEqual(self.call(route + "/stop", {}, tenant="other")[0], 404)
        self.assertEqual(self.call("/deployments", {"model_version_id": run["model_version_id"]}, tenant="other")[0], 404)
        status, stopped = self.call(route + "/stop", {})
        self.assertEqual(status, 200)
        self.assertEqual(stopped["desired"], "stopped")
        self.assertEqual(stopped["observed"], "stopped")
        self.assertFalse(stopped["ready"])
        self.assertEqual(self.call(route + "/predict", {"features": [[1]]})[0], 503)
        with self.assertRaises(ProcessLookupError):
            os.kill(deployment["pid"], 0)
        self.assertEqual(self.call(route + "/stop", {})[0], 200)

    def test_api_crash_recovery_persists_runs_versions_and_desired(self):
        run = self.train()
        running = self.deploy(run["model_version_id"])
        stopped = self.deploy(run["model_version_id"])
        self.assertEqual(self.call("/deployments/" + stopped["id"] + "/stop", {})[0], 200)
        queued = self.submit(key="queued")
        self.api.kill()
        self.api.wait(timeout=5)
        self.start_api()
        recovered = self.ready(running["id"])
        self.assertNotEqual(recovered["pid"], running["pid"])
        self.assertGreater(recovered["checked_at"], running["checked_at"])
        self.assertEqual(self.call("/runs/" + run["id"])[1]["model_version_id"], run["model_version_id"])
        self.assertEqual(self.call("/deployments/" + stopped["id"])[1]["observed"], "stopped")
        self.assertEqual(self.call("/deployments/" + stopped["id"] + "/predict", {"features": [[1]]})[0], 503)
        self.assertEqual(self.call("/deployments/" + running["id"] + "/predict", {"features": [[1]]})[0], 200)
        self.worker_once()
        self.assertEqual(self.call("/runs/" + queued["id"])[1]["status"], "succeeded")
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            try:
                os.kill(running["pid"], 0)
            except ProcessLookupError:
                break
            time.sleep(0.1)
        else:
            self.fail("Orphan inference process did not exit")

    def test_stopped_deployment_history_is_not_rewritten(self):
        run = self.train()
        deployment = self.deploy(run["model_version_id"])
        route = "/deployments/" + deployment["id"]
        status, stopped = self.call(route + "/stop", {})
        self.assertEqual(status, 200)
        self.assertEqual(self.store.deployment_rows(pending_only=True), [])
        time.sleep(1.1)
        current = self.call(route)[1]
        self.assertEqual(current["checked_at"], stopped["checked_at"])
        self.assertEqual(current["observed"], "stopped")

    def test_worker_rejects_source_changed_after_process_start(self):
        # Copy only source into this test's private, disposable runtime.
        copied = self.home / "source"
        shutil.copytree(ROOT / "model_platform", copied / "model_platform", ignore=shutil.ignore_patterns("__pycache__"))
        worker, log_path = self.spawn("worker", cwd=copied)
        deadline = time.monotonic() + 8
        while "worker started" not in log_path.read_text():
            self.assertIsNone(worker.poll(), log_path.read_text())
            if time.monotonic() > deadline:
                self.fail("Copied worker startup timeout")
            time.sleep(0.02)
        algorithm = copied / "model_platform" / "algorithm.py"
        algorithm.write_text(algorithm.read_text() + "\n# Changed after worker startup.\n")
        self.api.terminate()
        self.api.wait(timeout=5)
        self.start_api(cwd=copied)
        run = self.submit()
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            result = self.call("/runs/" + run["id"])[1]
            if result["status"] not in ("queued", "running"):
                break
            time.sleep(0.05)
        self.assertEqual(result["status"], "failed", result)
        self.assertEqual(result["attempts"][0]["error_code"], "execution_code_changed")
        self.assertEqual(self.call("/models")[1]["ids"], [])

    def test_worker_crash_is_recorded_and_retry_adds_attempt(self):
        body = request_body()
        body["dataset"]["train"] = [{"x": [i + 10 * j for j in range(16)], "y": i % 2} for i in range(996)]
        body["dataset"]["evaluation"] = [{"x": [-i - 1 - j for j in range(16)], "y": i % 2} for i in range(4)]
        body["config"]["epochs"] = 200
        run = self.submit(body)
        worker, log_path = self.spawn("worker")
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            with self.runtime.connection() as db:
                status = db.execute("SELECT status FROM runs WHERE id=?", (run["id"],)).fetchone()[0]
            if status == "running":
                break
            time.sleep(0.005)
        else:
            self.fail("Worker did not claim job: " + log_path.read_text())
        # While owned, a second worker must refuse before recovery or claiming.
        second, second_log = self.spawn("worker", "--once")
        self.assertNotEqual(second.wait(timeout=5), 0, second_log.read_text())
        worker.kill()
        worker.wait(timeout=5)
        self.worker_once()
        failed = self.call("/runs/" + run["id"])[1]
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["attempts"][0]["error_code"], "interrupted")
        self.assertEqual(self.call("/runs/" + run["id"] + "/retry", {})[0], 202)
        self.worker_once()
        recovered = self.call("/runs/" + run["id"])[1]
        self.assertEqual(recovered["status"], "succeeded", recovered)
        self.assertEqual([a["status"] for a in recovered["attempts"]], ["failed", "succeeded"])

    def test_corrupt_artifact_cannot_become_ready(self):
        run = self.train()
        model = self.call("/models/" + run["model_version_id"])[1]
        path = self.runtime.artifact(model["attempt_id"])
        path.chmod(0o600)
        path.write_text('{"malicious": "not a valid model"}')
        status, deployment = self.call("/deployments", {"model_version_id": run["model_version_id"]})
        self.assertEqual(status, 202)
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            deployment = self.call("/deployments/" + deployment["id"])[1]
            if deployment["observed"] == "failed":
                break
            time.sleep(0.1)
        self.assertEqual(deployment["observed"], "failed")
        self.assertFalse(deployment["ready"])
        self.assertEqual(self.call("/deployments/" + deployment["id"] + "/predict", {"features": [[1]]})[0], 503)

    def test_serving_crash_never_keeps_ready(self):
        run = self.train()
        deployment = self.deploy(run["model_version_id"])
        os.kill(deployment["pid"], signal.SIGKILL)
        time.sleep(0.1)
        result = self.call("/deployments/" + deployment["id"])[1]
        self.assertFalse(result["ready"])
        self.assertEqual(result["observed"], "failed")
        self.assertEqual(self.call("/deployments/" + deployment["id"] + "/predict", {"features": [[1]]})[0], 503)

    def test_queue_capacity_and_rate_limits(self):
        for i in range(32):
            self.submit(key=str(i))
        self.assertEqual(self.call("/runs", request_body(), "overflow")[0], 429)
        self.assertEqual(self.call("/runs", request_body(), "0")[0], 200)
        statuses = [self.call("/health")[0] for _ in range(90)]
        self.assertIn(429, statuses)
        self.assertEqual(self.call("/health", tenant="other")[0], 200)

    def test_single_api_lock_and_init_never_replaces_credentials(self):
        second, log = self.spawn("api", "--port", "0")
        self.assertNotEqual(second.wait(timeout=5), 0)
        self.assertIn("Another api owns this runtime", log.read_text())
        old = (self.home / "credentials.json").read_bytes()
        p, _ = self.spawn("init", "--tenant", "demo")
        self.assertNotEqual(p.wait(timeout=5), 0)
        self.assertEqual((self.home / "credentials.json").read_bytes(), old)
        self.assertEqual((self.home / "credentials.json").stat().st_mode & 0o777, 0o600)
        self.assertEqual(self.home.stat().st_mode & 0o777, 0o700)


if __name__ == "__main__":
    unittest.main()
