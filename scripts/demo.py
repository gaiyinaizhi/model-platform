"""Real synthetic-data CPU walkthrough; API and worker must already be running."""
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request
import uuid


def request_body():
    # Generated public-domain numeric example, not real user data.
    return {"dataset": {"source": "generated-public-domain-example", "version": "numeric-v1", "rights_confirmed": True,
                        "train": [{"x": [x], "y": int(x > 0)} for x in (-4, -3, -2, -1, 1, 2, 3, 4)],
                        "evaluation": [{"x": [x], "y": int(x > 0)} for x in (-3.5, -1.5, 1.5, 3.5)]},
            "config": {"epochs": 100, "learning_rate": 0.2}, "business_ref": "optional-demo-reference"}


def main():
    home = Path(os.environ.get("MP_HOME", ".runtime"))
    token = json.loads((home / "credentials.json").read_text())["demo"]
    base = os.environ.get("MP_API", "http://127.0.0.1:8080")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def call(path, body=None, key=None):
        headers = {"Authorization": "Bearer " + token}
        if key:
            headers["Idempotency-Key"] = key
        if body is not None:
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(base + path, data=None if body is None else json.dumps(body).encode(), headers=headers)
        with opener.open(request, timeout=5) as response:
            return json.load(response)

    key = "demo-" + uuid.uuid4().hex
    run = call("/runs", request_body(), key)
    assert call("/runs", request_body(), key)["id"] == run["id"]
    deadline = time.monotonic() + 20
    while run["status"] in ("queued", "running"):
        if time.monotonic() > deadline:
            raise RuntimeError("Training not complete; check independent worker")
        time.sleep(0.5)
        run = call("/runs/" + run["id"])
    if run["status"] != "succeeded":
        raise RuntimeError("Training failed; inspect sanitized attempt logs")
    model = call("/models/" + run["model_version_id"])
    deployment = call("/deployments", {"model_version_id": model["model_version_id"]})
    try:
        deadline = time.monotonic() + 10
        while not deployment["ready"]:
            if deployment["observed"] == "failed" or time.monotonic() > deadline:
                raise RuntimeError("Deployment not ready")
            time.sleep(0.5)
            deployment = call("/deployments/" + deployment["id"])
        prediction = call("/deployments/" + deployment["id"] + "/predict", {"features": [[-2.5], [2.5]]})
        print(json.dumps({"run_id": run["id"], "model_version_id": model["model_version_id"],
                          "evaluation": {k: model["evaluation"][k] for k in ("accuracy", "log_loss", "sample_count")},
                          "prediction": prediction}, indent=2))
    finally:
        stopped = call("/deployments/" + deployment["id"] + "/stop", {})
        assert stopped["observed"] == "stopped"
    try:
        call("/deployments/" + deployment["id"] + "/predict", {"features": [[1.0]]})
    except urllib.error.HTTPError as exc:
        assert exc.code == 503
        exc.close()
        print("stop verified: subsequent inference rejected (503)")
    else:
        raise AssertionError("Stopped inference accepted")


if __name__ == "__main__":
    main()
