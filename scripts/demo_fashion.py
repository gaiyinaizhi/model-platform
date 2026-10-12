"""One real fixed Fashion-MNIST CNN/ONNX run through an independent API/Worker.

First run scripts/fetch_fashion.py in the same MP_HOME, then start API and Worker
as separate processes using .venv-fashion. This demo verifies the official cache
offline, submits the fixed 6000-row training subset recipe, evaluates all 10000
official test rows, repeatedly calls its registered service, and verifies stop.
"""
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model_platform import fashion, fashion_data
from model_platform.common import Runtime, encode


class NoRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # A runtime bearer token must never follow an HTTP redirect.
        return None


def main():
    runtime = Runtime()
    base = os.environ.get("MP_API", "http://127.0.0.1:8080").rstrip("/")
    target = urllib.parse.urlsplit(base)
    if (target.scheme != "http" or target.hostname not in ("127.0.0.1", "localhost", "::1")
            or target.username or target.password or target.path or target.query or target.fragment):
        raise ValueError("MP_API must be a plain HTTP loopback origin for this local-only demonstration")
    # Recompute all pinned source hashes before submitting. No network, arbitrary
    # path/URL input, alternate mirror, modified subset or copied digit data.
    train_x, train_y, test_x, test_y = fashion_data.load_dataset(runtime)
    if (train_x.shape != (60000, 784) or train_y.shape != (60000,)
            or test_x.shape != (10000, 784) or test_y.shape != (10000,)):
        raise AssertionError("Official dataset split shape mismatch")
    del train_x, train_y
    chosen = {}
    for index, label in enumerate(test_y):
        chosen.setdefault(int(label), index)
    samples = [{"official_test_row_index": chosen[label], "expected_class": label,
                "expected_label": fashion_data.LABELS[label]} for label in range(10)]
    features = [test_x[sample["official_test_row_index"]].tolist() for sample in samples]
    del test_x, test_y
    token = json.loads((runtime.home / "credentials.json").read_text())["demo"]
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirects())

    def call(path, payload=None, key=None):
        headers = {"Authorization": "Bearer " + token}
        if key is not None:
            headers["Idempotency-Key"] = key
        if payload is not None:
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(base + path, data=None if payload is None else encode(payload), headers=headers)
        with opener.open(request, timeout=15) as response:
            return json.load(response)

    body = fashion.request_body()
    started = time.perf_counter()
    key = "fashion-" + uuid.uuid4().hex
    run = call("/runs", body, key)
    if call("/runs", body, key)["id"] != run["id"]:
        raise AssertionError("Idempotent submission created another training run")
    deadline = time.monotonic() + 900
    while run["status"] in ("queued", "running"):
        if time.monotonic() >= deadline:
            raise RuntimeError("Training did not finish within 900 seconds; inspect the independent Worker")
        time.sleep(2)  # At most 30 status requests/minute; respect tenant budgets.
        run = call("/runs/" + run["id"])
    if run["status"] != "succeeded":
        codes = [attempt.get("error_code") for attempt in run.get("attempts", [])]
        raise RuntimeError("Fashion training failed with sanitized attempt codes: " + repr(codes))
    submission_to_registration_seconds = time.perf_counter() - started
    model = call("/models/" + run["model_version_id"])
    evaluation = model["evaluation"]
    if ((model["recipe"], model["contract"], model["format"]) != (fashion.RECIPE, fashion.CONTRACT, fashion.FORMAT)
            or model["config"] != body["config"] or model["dataset"]["train_count"] != 6000
            or model["dataset"]["official_train_count"] != 60000
            or model["dataset"]["evaluation_count"] != 10000 or evaluation["sample_count"] != 10000
            or sum(map(sum, evaluation["confusion"])) != 10000):
        raise AssertionError("Registered model does not match the fixed official training/evaluation protocol")
    parity = evaluation["parity"]
    if (not parity["passed"] or parity["full_test_sample_count"] != 10000
            or parity["full_test_argmax_agreement_count"] != 10000
            or parity["full_test_max_absolute_logit_error"] > parity["absolute_logit_tolerance"]):
        raise AssertionError("Exported ONNX did not pass complete official-test parity")
    artifact = runtime.artifact(model["artifact_id"])
    if artifact.is_symlink() or model["artifact_id"] != model["attempt_id"]:
        raise AssertionError("Unexpected registered artifact identity")
    content = artifact.read_bytes()
    if (len(content) != model["artifact_bytes"] or len(content) > fashion.MAX_ARTIFACT
            or hashlib.sha256(content).hexdigest() != model["artifact_sha256"]):
        raise AssertionError("Registered ONNX artifact bytes failed integrity verification")
    training_seconds = model["training_seconds"]
    if not isinstance(training_seconds, (int, float)) or not math.isfinite(training_seconds) or training_seconds <= 0:
        raise AssertionError("Registered training time is invalid")

    deployment = call("/deployments", {"model_version_id": model["model_version_id"]})
    route = "/deployments/" + deployment["id"]
    prediction, latencies, stopped = None, [], None
    try:
        deadline = time.monotonic() + 30
        while not deployment["ready"]:
            if deployment["observed"] == "failed" or time.monotonic() >= deadline:
                raise RuntimeError("Fashion deployment failed to become ready")
            time.sleep(1)
            deployment = call(route)
        if deployment["runtime"] != "onnxruntime-cpu/v1" or deployment["contract"] != fashion.CONTRACT:
            raise AssertionError("Deployment is not the bound Fashion ONNX CPU service")
        inference_started = time.perf_counter()
        for index in range(10):
            before = time.perf_counter()
            result = call(route + "/predict", {"features": features})
            latencies.append((time.perf_counter() - before) * 1000)
            if (result["model_version_id"] != model["model_version_id"]
                    or result["artifact_sha256"] != model["artifact_sha256"]
                    or result["contract"] != fashion.CONTRACT or len(result["predictions"]) != len(features)):
                raise AssertionError("Serving returned the wrong bound model, contract or prediction count")
            for output in result["predictions"]:
                probabilities = output["probabilities"]
                if (len(probabilities) != 10 or not all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities)
                        or abs(sum(probabilities) - 1) > 1e-8
                        or output["class"] != max(range(10), key=probabilities.__getitem__)):
                    raise AssertionError("Serving returned invalid ten-class probabilities")
            if prediction is not None and result != prediction:
                raise AssertionError("Repeated serving calls changed deterministic predictions")
            prediction = result
            if index < 9:
                time.sleep(0.2)
        inference_window_seconds = time.perf_counter() - inference_started
        active = call(route)
        if not active["ready"] or active["desired"] != "running" or active["pid"] != deployment["pid"]:
            raise AssertionError("Continuous serving process changed or stopped after repeated predictions")
    finally:
        stopped = call(route + "/stop", {})
        if (stopped["observed"], stopped["desired"], stopped["ready"]) != ("stopped", "stopped", False):
            raise AssertionError("Deployment stop was not confirmed")
        try:
            call(route + "/predict", {"features": features[:1]})
        except urllib.error.HTTPError as exc:
            status = exc.code
            exc.close()
            if status != 503:
                raise AssertionError(f"Stopped serving returned HTTP {status}, expected 503") from None
        else:
            raise AssertionError("Stopped serving still accepted inference")

    report = {
        "case": "fashion-mnist-cnn-onnx-cpu/v1",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run["id"], "model_version_id": model["model_version_id"],
        "deployment_id": deployment["id"], "artifact_sha256": model["artifact_sha256"],
        "artifact_bytes": model["artifact_bytes"], "recipe": model["recipe"], "config": model["config"],
        "source": fashion_data.provenance(), "dataset_audit": model["dataset"],
        "evaluation": evaluation, "registered_manifest": model, "attempts": run["attempts"],
        "environment": model["execution_environment"], "dependencies": model["dependencies"],
        "demo_environment": {"python": platform.python_version(), "platform": platform.platform(),
                             "logical_cpu_count": os.cpu_count(), "gpu_used": False},
        "timing": {"training_seconds": training_seconds,
                   "submission_to_registration_observed_seconds": submission_to_registration_seconds,
                   "inference_window_seconds": inference_window_seconds,
                   "inference_requests": len(latencies), "batch_size": len(features),
                   "http_latency_ms": {"observations": latencies, "min": min(latencies),
                                       "median": statistics.median(latencies), "mean": statistics.mean(latencies),
                                       "max": max(latencies)},
                   "interpretation": "Shared local CPU observation, not a benchmark. HTTP timings include authentication, readiness probes, serialization and loopback transport."},
        "sample_selection": "First official test occurrence of each true label, chosen before model submission without looking at predictions",
        "sample_predictions": [{**sample, **output, "predicted_label": fashion_data.LABELS[output["class"]]}
                               for sample, output in zip(samples, prediction["predictions"])],
        "verification": {"official_cache_hashes_verified": True, "idempotent_submission": True,
                         "registered_artifact": True, "full_test_export_parity": True,
                         "repeated_inference_requests": len(latencies), "stable_predictions": True,
                         "continuous_serving_pid": deployment["pid"],
                         "stop_observed": stopped["observed"], "post_stop_http_status": 503},
        "limitations": "6000 stratified official training rows, not all 60000; all 10000 official test rows. Small grayscale clothing CNN only; no external audit, test tuning, production isolation, GPU, robots, drones or device validation.",
    }
    report_directory = runtime.home / "fashion"
    if report_directory.is_symlink():
        raise ValueError("Report directory must not be a symlink")
    report_directory.mkdir(mode=0o700, exist_ok=True)
    report_path = report_directory / (run["id"] + "-report.json")
    descriptor = os.open(report_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encode(report) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
