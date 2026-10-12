"""Run real UCI digits training, registered serving, repeated inference and stop."""
import json
import math
import os
from pathlib import Path
import platform
import statistics
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

if __package__:
    from .fetch_digits import prepare, json_bytes, write_private
else:
    from fetch_digits import prepare, json_bytes, write_private


class NoRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Never forward a runtime bearer token to a redirected destination.
        return None


def main():
    home = Path(os.environ.get("MP_HOME", ".runtime")).resolve()
    body, metadata = prepare(home)
    token = json.loads((home / "credentials.json").read_text())["demo"]
    base = os.environ.get("MP_API", "http://127.0.0.1:8080").rstrip("/")
    target = urllib.parse.urlsplit(base)
    if target.scheme != "http" or target.hostname not in ("127.0.0.1", "localhost", "::1") or target.username or target.password:
        raise ValueError("MP_API must be an HTTP loopback address for this local-only demonstration")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirects())

    def call(path, payload=None, key=None):
        headers = {"Authorization": "Bearer " + token}
        if key:
            headers["Idempotency-Key"] = key
        if payload is not None:
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(base + path, data=None if payload is None else json_bytes(payload), headers=headers)
        with opener.open(request, timeout=15) as response:
            return json.load(response)

    started = time.perf_counter()
    key = "digits-" + uuid.uuid4().hex
    run = call("/runs", body, key)
    if call("/runs", body, key)["id"] != run["id"]:
        raise AssertionError("Idempotent submission created another training run")
    deadline = time.monotonic() + 180
    while run["status"] in ("queued", "running"):
        if time.monotonic() >= deadline:
            raise RuntimeError("Training did not finish within 180 seconds; inspect the independent worker")
        # Stay below the shared per-tenant request budget during a slow run.
        time.sleep(1)
        run = call("/runs/" + run["id"])
    if run["status"] != "succeeded":
        raise RuntimeError("Digits training failed; inspect sanitized attempt history")
    training_completion_seconds = time.perf_counter() - started
    model = call("/models/" + run["model_version_id"])
    if model["recipe"] != body["recipe"] or model["evaluation"]["sample_count"] != 1797:
        raise AssertionError("Registered model does not match the complete official digits evaluation")
    training_seconds = model["training_seconds"]
    if not isinstance(training_seconds, (int, float)) or not math.isfinite(training_seconds) or training_seconds < 0:
        raise AssertionError("Registered training time is invalid")

    # First occurrence of each label, selected by index, without looking at predictions.
    chosen = {}
    for index, row in enumerate(body["dataset"]["evaluation"]):
        chosen.setdefault(row["y"], (index, row))
    samples = [chosen[label] for label in range(10)]
    features = [row["x"] for _, row in samples]
    deployment = call("/deployments", {"model_version_id": model["model_version_id"]})
    prediction, latencies, stopped = None, [], None
    try:
        deadline = time.monotonic() + 20
        while not deployment["ready"]:
            if deployment["observed"] == "failed" or time.monotonic() >= deadline:
                raise RuntimeError("Digits deployment failed to become ready")
            time.sleep(0.5)
            deployment = call("/deployments/" + deployment["id"])
        inference_started = time.perf_counter()
        for index in range(10):
            before = time.perf_counter()
            result = call("/deployments/" + deployment["id"] + "/predict", {"features": features})
            latencies.append((time.perf_counter() - before) * 1000)
            if result["model_version_id"] != model["model_version_id"] or result["artifact_sha256"] != model["artifact_sha256"]:
                raise AssertionError("Serving returned the wrong model version or artifact")
            if len(result["predictions"]) != len(samples):
                raise AssertionError("Serving returned an unexpected prediction count")
            for output in result["predictions"]:
                ps = output["probabilities"]
                if (len(ps) != 10 or not all(math.isfinite(p) and 0 <= p <= 1 for p in ps)
                        or abs(sum(ps) - 1) > 1e-8 or output["class"] != max(range(10), key=ps.__getitem__)):
                    raise AssertionError("Serving returned invalid ten-class probabilities")
            if prediction is not None and result != prediction:
                raise AssertionError("Repeated serving calls changed deterministic predictions")
            prediction = result
            if index < 9:
                time.sleep(0.2)
        inference_window_seconds = time.perf_counter() - inference_started
        active = call("/deployments/" + deployment["id"])
        if not active["ready"] or active["desired"] != "running":
            raise AssertionError("Continuous serving was not ready after repeated predictions")
    finally:
        stopped = call("/deployments/" + deployment["id"] + "/stop", {})
        if stopped["observed"] != "stopped" or stopped["desired"] != "stopped" or stopped["ready"]:
            raise AssertionError("Deployment stop was not confirmed")
        try:
            call("/deployments/" + deployment["id"] + "/predict", {"features": features[:1]})
        except urllib.error.HTTPError as exc:
            code = exc.code
            exc.close()
            if code != 503:
                raise AssertionError(f"Stopped serving returned HTTP {code}, expected 503") from None
        else:
            raise AssertionError("Stopped serving still accepted inference")

    report = {
        "case": "uci-optical-digits-cpu/v1", "run_id": run["id"], "model_version_id": model["model_version_id"],
        "deployment_id": deployment["id"], "artifact_sha256": model["artifact_sha256"],
        "recipe": model["recipe"], "config": model["config"],
        "source": {key: metadata[key] for key in ("source_url", "archive_url", "attribution", "doi", "license", "license_url", "archive_sha256", "file_sha256")},
        "dataset_audit": metadata["audit"], "evaluation": model["evaluation"],
        "environment": model["execution_environment"],
        "demo_environment": {"python": platform.python_version(), "platform": platform.platform(),
                             "logical_cpu_count": os.cpu_count(), "gpu_used": False},
        "timing": {"training_seconds": training_seconds,
                   "submission_to_registration_observed_seconds": training_completion_seconds,
                   "inference_window_seconds": inference_window_seconds,
                   "inference_requests": len(latencies), "batch_size": len(features),
                   "http_latency_ms": {"observations": latencies, "min": min(latencies),
                                       "median": statistics.median(latencies), "mean": statistics.mean(latencies),
                                       "max": max(latencies)},
                   "interpretation": "Shared local CPU observation, not a benchmark. HTTP timings include authentication, readiness probes, serialization and loopback transport."},
        "sample_predictions": [{"official_test_row_index": index, "expected_class": row["y"], **output}
                               for (index, row), output in zip(samples, prediction["predictions"])],
        "verification": {"idempotent_submission": True, "registered_artifact": True,
                         "repeated_inference_requests": len(latencies), "stable_predictions": True,
                         "stop_observed": stopped["observed"], "post_stop_http_status": 503},
        "limitations": "Real 8x8 handwritten digit classification only; not unconstrained images, production isolation, GPU, robots, drones or device validation.",
    }
    report_path = home / "digits" / (run["id"] + "-report.json")
    write_private(report_path, json_bytes(report))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
