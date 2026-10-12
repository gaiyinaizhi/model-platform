"""Fixed, optional Fashion-MNIST CNN training and verified ONNX delivery.

Third-party packages are imported only by this recipe. No Python objects,
optimizer state, pickle files or checkpoints are persisted. The test split is
used once, after training and export, and never selects training parameters.
"""
import hashlib
import importlib.metadata
import os
from pathlib import Path
import random
import time

from .algorithm import TrainingFailure
from .common import digest


SEED = 20261012
TRAIN_COUNT = 6000
PER_CLASS = 600
EPOCHS = 5
LEARNING_RATE = 0.001
BATCH_SIZE = 128
CPU_THREADS = 2
MAX_ARTIFACT_BYTES = 2 * 1024 * 1024
TRAINING_SECONDS = 600
EXPORT_SECONDS = 120
EVALUATION_SECONDS = 120
PARITY_TOLERANCE = 1e-4


def _deadline(started, budget, code):
    # Cooperative phase bounds; an individual native call is not preempted.
    if time.monotonic() - started > budget:
        raise TrainingFailure(code)


def select_training_indices(labels):
    """Select 600 official training rows per class using a fixed Python RNG.

    Sampling is independent of image values, test records and predictions.
    Sorting preserves official row order before epoch-specific shuffling.
    """
    rng = random.Random(SEED)
    groups = [[] for _ in range(10)]
    for index, value in enumerate(labels):
        label = int(value)
        if not 0 <= label < 10:
            raise TrainingFailure("fashion_dataset_invalid")
        groups[label].append(index)
    if any(len(group) != 6000 for group in groups):
        raise TrainingFailure("fashion_dataset_invalid")
    return sorted(index for group in groups for index in rng.sample(group, PER_CLASS))


def _arrays_hash(images, labels):
    """Hash typed, shaped pixel/label arrays, without ambiguous concatenation."""
    result = hashlib.sha256()
    result.update(b"fashion-uint8-flat-images-and-labels/v1\0")
    result.update(str(tuple(images.shape)).encode("ascii") + b"\0")
    result.update(images.tobytes(order="C"))
    result.update(str(tuple(labels.shape)).encode("ascii") + b"\0")
    result.update(labels.tobytes(order="C"))
    return result.hexdigest()


def _check_arrays(np, train_x, train_y, test_x, test_y):
    for array, shape in ((train_x, (60000, 784)), (train_y, (60000,)),
                         (test_x, (10000, 784)), (test_y, (10000,))):
        if not isinstance(array, np.ndarray) or array.shape != shape or array.dtype != np.uint8:
            raise TrainingFailure("fashion_dataset_invalid")
    if (np.any(train_y > 9) or np.any(test_y > 9)
            or not np.array_equal(np.bincount(test_y, minlength=10), np.full(10, 1000))):
        raise TrainingFailure("fashion_dataset_invalid")


def _normalized(np, rows):
    return np.ascontiguousarray(rows.reshape(-1, 1, 28, 28), dtype=np.float32) / np.float32(255)


def _check_logits(np, logits, count):
    if (not 1 <= count <= BATCH_SIZE or not isinstance(logits, np.ndarray) or logits.shape != (count, 10)
            or logits.dtype != np.float32 or not np.all(np.isfinite(logits))):
        raise TrainingFailure("fashion_logits_invalid")


class _Metrics:
    """Streaming ten-class metrics; log loss is computed from stable logits."""

    def __init__(self, np):
        self.np = np
        self.confusion = np.zeros((10, 10), dtype=np.int64)
        self.loss = 0.0
        self.count = 0

    def update(self, labels, logits):
        np = self.np
        _check_logits(np, logits, len(labels))
        predictions = logits.argmax(axis=1)
        np.add.at(self.confusion, (labels, predictions), 1)
        scores = logits.astype(np.float64)
        self.loss += float((np.logaddexp.reduce(scores, axis=1)
                            - scores[np.arange(len(labels)), labels]).sum())
        self.count += len(labels)

    def result(self):
        if self.count == 0:
            raise TrainingFailure("fashion_evaluation_empty")
        matrix = self.confusion.tolist()
        f1s = []
        for label in range(10):
            denominator = sum(matrix[label]) + sum(row[label] for row in matrix)
            f1s.append(2 * matrix[label][label] / denominator if denominator else 0.0)
        return {"accuracy": sum(matrix[i][i] for i in range(10)) / self.count,
                "macro_f1": sum(f1s) / 10, "log_loss": self.loss / self.count,
                "confusion": matrix, "sample_count": self.count,
                "class_order": list(range(10))}


def _parity(np, reference, actual):
    _check_logits(np, reference, len(reference))
    _check_logits(np, actual, len(reference))
    max_error = float(np.max(np.abs(reference.astype(np.float64) - actual.astype(np.float64))))
    agreements = int(np.count_nonzero(reference.argmax(axis=1) == actual.argmax(axis=1)))
    if max_error > PARITY_TOLERANCE or agreements != len(reference):
        raise TrainingFailure("fashion_onnx_parity_failed")
    return max_error, agreements


def _write_artifact(path, content):
    if not content or len(content) > MAX_ARTIFACT_BYTES:
        raise TrainingFailure("fashion_artifact_too_large")
    # O_EXCL preserves attempts and refuses pre-existing files and symlinks.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
        os.fchmod(stream.fileno(), 0o400)
    return hashlib.sha256(content).hexdigest()


def train_export(runtime, body, path, log, execution_environment):
    """Train the fixed recipe; return (artifact hash, evaluation, manifest)."""
    try:
        import numpy as np
        import onnx
        import onnxruntime  # Verify all optional dependencies before training.
        import onnxscript  # Required by the dynamo exporter, even before export.
        import torch
    except ImportError:
        raise TrainingFailure("optional_dependencies_missing") from None
    from . import fashion, fashion_data

    path = Path(path)
    all_started = time.monotonic()
    fashion.validate_request(body["dataset"], body["config"])
    torch.set_num_threads(CPU_THREADS)
    if torch.get_num_interop_threads() != CPU_THREADS:
        try:
            torch.set_num_interop_threads(CPU_THREADS)
        except RuntimeError:
            raise TrainingFailure("fashion_thread_budget_unavailable") from None
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)

    dataset_started = time.monotonic()
    try:
        train_x, train_y, test_x, test_y = fashion_data.load_dataset(runtime)
    except fashion_data.DatasetError as exc:
        code = getattr(exc, "code", "fashion_dataset_unavailable_or_invalid")
        raise TrainingFailure(code) from None
    _check_arrays(np, train_x, train_y, test_x, test_y)
    indices = select_training_indices(train_y)
    selected_x, selected_y = train_x[indices], train_y[indices]
    train_sha256 = _arrays_hash(selected_x, selected_y)
    evaluation_sha256 = _arrays_hash(test_x, test_y)
    dataset_seconds = time.monotonic() - dataset_started
    log("fashion_dataset_verified", {"official_training_count": len(train_x),
                                      "selected_training_count": len(indices),
                                      "evaluation_count": len(test_x),
                                      "selection_sha256": digest(indices)})

    # Original implementation of a small, fixed architecture. No pretrained
    # weights, augmentation, validation selection or external model code.
    model = torch.nn.Sequential(
        torch.nn.Conv2d(1, 8, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2),
        torch.nn.Conv2d(8, 16, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2),
        torch.nn.Flatten(start_dim=1),
        torch.nn.Linear(16 * 7 * 7, 10),
    ).cpu()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE,
                                 betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
    train_tensor = torch.from_numpy(_normalized(np, selected_x))
    label_tensor = torch.from_numpy(selected_y.astype(np.int64))
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    training_started = time.monotonic()
    model.train()
    for epoch in range(EPOCHS):
        order = torch.randperm(TRAIN_COUNT, generator=generator)
        loss_sum = 0.0
        for offset in range(0, TRAIN_COUNT, BATCH_SIZE):
            _deadline(training_started, TRAINING_SECONDS, "training_timeout")
            rows = order[offset:offset + BATCH_SIZE]
            optimizer.zero_grad(set_to_none=True)
            logits = model(train_tensor[rows])
            loss = torch.nn.functional.cross_entropy(logits, label_tensor[rows])
            if not bool(torch.isfinite(loss)):
                raise TrainingFailure("fashion_training_nonfinite")
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(rows)
            _deadline(training_started, TRAINING_SECONDS, "training_timeout")
        log("train_progress", {"epoch": epoch + 1, "total_epochs": EPOCHS,
                                "training_log_loss": loss_sum / TRAIN_COUNT})
    training_seconds = time.monotonic() - training_started
    model.eval()

    export_started = time.monotonic()
    # A sample batch of two avoids the exporter specializing the batch to one.
    # Nothing is written by the exporter: only the verified ONNX protobuf below
    # reaches disk, with external tensor data explicitly disabled.
    program = torch.onnx.export(
        model, (train_tensor[:2],), f=None, dynamo=True, external_data=False,
        input_names=["pixels"], output_names=["logits"], opset_version=18,
        dynamic_shapes=({0: torch.export.Dim("batch", min=1, max=BATCH_SIZE)},),
    )
    _deadline(export_started, EXPORT_SECONDS, "fashion_export_timeout")
    proto = program.model_proto
    metadata = {"recipe": fashion.RECIPE, "contract": fashion.CONTRACT,
                "format": fashion.FORMAT, "dimensions": "784", "pixel_scale": "255",
                "classes": ",".join(str(i) for i in range(10))}
    onnx.helper.set_model_props(proto, metadata)
    content = proto.SerializeToString()
    if len(content) > MAX_ARTIFACT_BYTES:
        raise TrainingFailure("fashion_artifact_too_large")
    artifact_hash = _write_artifact(path, content)
    # The shared loader re-reads and hashes the immutable bytes, validates the
    # ONNX metadata/operator/tensor whitelist, then constructs its CPU session.
    session = fashion.load_model(path, artifact_hash)["session"]
    _deadline(export_started, EXPORT_SECONDS, "fashion_export_timeout")
    export_seconds = time.monotonic() - export_started
    log("artifact_written", {"sha256": artifact_hash, "bytes": len(content), "format": fashion.FORMAT})

    evaluation_started = time.monotonic()
    parity_batches = []
    with torch.inference_mode():
        # Every pre-evaluation parity record comes from fixed TRAIN samples.
        for count in (1, 2, 17, 128):
            batch = train_tensor[:count].numpy()
            reference = model(torch.from_numpy(batch)).numpy()
            actual = session.run(["logits"], {"pixels": batch})[0]
            max_error, agreements = _parity(np, reference, actual)
            parity_batches.append({"batch_size": count, "max_absolute_logit_error": max_error,
                                   "argmax_agreement_count": agreements, "argmax_agreement": agreements / count})
            _deadline(evaluation_started, EVALUATION_SECONDS, "fashion_evaluation_timeout")
        ort_metrics, pytorch_metrics = _Metrics(np), _Metrics(np)
        ort_seconds, pytorch_seconds, test_max_error, agreement_count = 0.0, 0.0, 0.0, 0
        for offset in range(0, len(test_x), BATCH_SIZE):
            _deadline(evaluation_started, EVALUATION_SECONDS, "fashion_evaluation_timeout")
            batch = _normalized(np, test_x[offset:offset + BATCH_SIZE])
            labels = test_y[offset:offset + BATCH_SIZE]
            before = time.monotonic()
            actual = session.run(["logits"], {"pixels": batch})[0]
            ort_seconds += time.monotonic() - before
            before = time.monotonic()
            reference = model(torch.from_numpy(batch)).numpy()
            pytorch_seconds += time.monotonic() - before
            max_error, agreements = _parity(np, reference, actual)
            test_max_error = max(test_max_error, max_error)
            agreement_count += agreements
            ort_metrics.update(labels, actual)
            pytorch_metrics.update(labels, reference)
            _deadline(evaluation_started, EVALUATION_SECONDS, "fashion_evaluation_timeout")
    evaluation_seconds = time.monotonic() - evaluation_started
    dependency_names = ("torch", "numpy", "onnx", "onnxruntime", "onnxscript")
    dependencies = {name: importlib.metadata.version(name) for name in dependency_names}
    parity = {"absolute_logit_tolerance": PARITY_TOLERANCE,
              "training_sample_policy": "First 128 rows in the fixed training selection; never test-selected",
              "training_indices_sha256": digest(indices[:128]), "training_batches": parity_batches,
              "full_test_sample_count": len(test_x), "full_test_max_absolute_logit_error": test_max_error,
              "full_test_argmax_agreement_count": agreement_count,
              "full_test_argmax_agreement": agreement_count / len(test_x), "passed": True}
    evaluation = {"protocol": "fashion-official-test-onnx/v1", **ort_metrics.result(),
                  "artifact_sha256": artifact_hash, "dataset_sha256": evaluation_sha256,
                  "backend": "onnxruntime-cpu", "pytorch_reference": pytorch_metrics.result(),
                  "parity": parity, "inference_seconds": ort_seconds,
                  "pytorch_inference_seconds": pytorch_seconds,
                  "evaluation_seconds": evaluation_seconds,
                  "environment": execution_environment, "dependencies": dependencies,
                  "timing_scope": "Batched CPU forward calls; shared-host observation, not a benchmark",
                  "limitations": "Official test used only after fixed training/export; not external audit, production quality or device validation"}
    provenance = fashion_data.provenance()
    manifest = {
        "recipe": fashion.RECIPE, "contract": fashion.CONTRACT, "format": fashion.FORMAT,
        "dimensions": 784, "classes": list(range(10)), "training_seconds": training_seconds,
        "artifact_bytes": len(content), "config": body["config"],
        "training": {"optimizer": "Adam", "betas": [0.9, 0.999], "epsilon": 1e-8,
                     "weight_decay": 0, "loss": "cross_entropy", "deterministic_algorithms": True,
                     "shuffle": "torch.randperm per epoch using a CPU generator seeded with 20261012",
                     "augmentation": False, "early_stopping": False, "test_parameter_selection": False},
        "environment": execution_environment, "execution_environment": execution_environment,
        "dependencies": dependencies, "target_profile": "local-cpu-onnxruntime/v1",
        "dataset": {"source": body["dataset"]["source"], "version": body["dataset"]["version"],
                    "sha256": digest(provenance), "train_sha256": train_sha256,
                    "evaluation_sha256": evaluation_sha256, "provenance": provenance,
                    "official_train_count": len(train_x), "train_count": TRAIN_COUNT,
                    "evaluation_count": len(test_x), "training_class_counts": [PER_CLASS] * 10,
                    "training_selection_sha256": digest(indices), "selection_seed": SEED,
                    "selection_policy": "Python random.Random(seed).sample 600 indices per class, then sort; official test unchanged",
                    "split_policy": "Stratified subset from official training split only; all 10000 official test records retained"},
        "model": {"architecture": "Conv2d(1,8,3,padding=1),ReLU,MaxPool2d(2),Conv2d(8,16,3,padding=1),ReLU,MaxPool2d(2),Flatten,Linear(784,10)",
                  "parameters": sum(parameter.numel() for parameter in model.parameters()),
                  "input": {"name": "pixels", "dtype": "float32", "shape": ["batch", 1, 28, 28],
                            "preprocessing": "uint8 pixels / 255; no learned normalization"},
                  "output": {"name": "logits", "dtype": "float32", "shape": ["batch", 10]},
                  "onnx_opset": 18, "exporter": "torch.onnx dynamo", "external_data": False},
        "resources": {"device": "cpu", "gpu_used": False, "cpu_threads": CPU_THREADS,
                      "batch_size": BATCH_SIZE, "training_loop_limit_seconds": TRAINING_SECONDS,
                      "export_limit_seconds": EXPORT_SECONDS, "evaluation_limit_seconds": EVALUATION_SECONDS,
                      "artifact_limit_bytes": MAX_ARTIFACT_BYTES,
                      "limits_scope": "Cooperative checks between native calls; no hard CPU/RSS/native-call timeout isolation",
                      "dataset_seconds": dataset_seconds, "export_seconds": export_seconds,
                      "evaluation_seconds": evaluation_seconds, "total_seconds": time.monotonic() - all_started},
        "parity": parity,
        "capabilities": {"train": True, "evaluate": True, "serve_cpu": True, "device": False},
    }
    log("evaluated", {"accuracy": evaluation["accuracy"], "log_loss": evaluation["log_loss"],
                      "sample_count": evaluation["sample_count"], "parity_max_absolute_logit_error": test_max_error})
    return artifact_hash, evaluation, manifest
