"""Download and verify the one official Fashion-MNIST reference (standard library)."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model_platform.common import Runtime
from model_platform.fashion_data import CACHE_NAME, DATASET_REF, fetch_dataset


def prepare(home=None):
    return fetch_dataset(Runtime(home))


def main():
    metadata = prepare()
    print(json.dumps({"reference": DATASET_REF, "status": "verified",
                      "source_url": metadata["source_url"], "source_commit": metadata["source_commit"],
                      "license": metadata["license"], "dataset_sha256": metadata["dataset_sha256"],
                      "train_count": metadata["train_count"], "evaluation_count": metadata["evaluation_count"],
                      "cache_subdirectory": "datasets/" + CACHE_NAME,
                      "files": metadata["files"]}, indent=2))


if __name__ == "__main__":
    main()
