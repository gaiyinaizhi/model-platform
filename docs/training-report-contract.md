# Dataset training evaluation report v1

Only succeeded `yolox-nano-dataset/v1` registered models are supported. This exports already recorded primary ONNX evaluation facts, not a new evaluation, independent imported-COCO report, comparison or release approval.

## Download and errors

`GET /models/{model_id}/training-report` requires Bearer authentication. Only the exact body-free path is accepted: no query (including `?`), encoded segments, extra slash, absolute target, Range/If-Range, transfer/content encoding or duplicate/nonzero Content-Length. Accept headers do not negotiate HTML or compression.

Ownership is resolved first (unknown/foreign 404). Unsupported model recipes/origins return 409 `training_report_unsupported`; nonsucceeded associations return 409 `training_report_not_succeeded`; missing, malformed or inconsistent evidence returns 409 `training_report_integrity`. Bad model IDs and malformed requests return 400. Errors never echo stored evidence or private paths.

The response is canonical JSON, at most 512 KiB, with fixed filename `training-evaluation-report.json`, Content-Length, `Cache-Control: no-store` and `X-Content-Type-Options: nosniff`. Unchanged evidence produces identical bytes. No download timestamp is added.

## Read-only evidence validation

One database transaction captures the owned model, its run, successful final attempt, evaluation and immutable run-dataset binding. The endpoint validates IDs/ownership/status, contiguous attempt history, artifact and dataset identities, original request digest, configurations, category mapping, the existing compatibility digest, environments, timestamps and the report's semantic constraints. It does not instantiate Store/DatasetStore, create keys, change schema or audit, read dataset/model files, inspect current execution environment, download or infer anything.

The recorded source rows are logically immutable, not externally authenticated. The report does not establish that current artifact/dataset bytes still match their recorded hashes. Same-user administrators can alter the database, and anyone can rewrite a report and recompute its unkeyed checksums.

## Fixed disclosure projection

`training-evaluation-report/v1` has exact top-level fields:

- `training`: model/run/attempt/evaluation IDs, succeeded status, attempt number, original run creation and attempt start/finish times, fixed recipe/contract/format.
- `provenance`: dataset version/content and train/val/test/evaluation split hashes and counts, classes and negative policy; submitted and actual training configuration plus digest; separate submission/execution Python/platform/code hash; seven recorded package versions; artifact digest/size; initial/final parameter digests.
- `training_summary`: actual loss history and recorded training/export/evaluation durations.
- `exported_evaluation`: primary ONNX evaluation contract, split/role, artifact/dataset identity, thresholds/preprocessing, aggregate and all class metrics, counts, evaluator and metric parameters.
- `exported_evaluation_sha256`, `report_sha256`, `verification`, `limitations`, plus schema and canonicalization IDs.

Raw images, labels, sample paths, dataset names, logs, business references, tenant, credentials, arbitrary warning text and prediction details are excluded. Nested validation/PyTorch/parity results are not exported by this version. Their omission is not a claim that they were revalidated.

If an independent test split was supplied the role is `held-out-test`; otherwise the actual validation split is explicitly `validation-only`. Missing ground truth keeps AP/AR `null`, not zero. Actual zero metrics remain zero. Every defined category is retained, while aggregate AP/AR averages only categories containing ground truth. The report never asserts quality passed, official COCO benchmark status, device certification, production eligibility or external audit.

The worker enforces identical source hashes at submission/execution. Python and platform may differ and are preserved separately. The reported dependency versions are recorded model manifest facts, not current installed versions.

## Canonicalization and offline validation

Canonicalization is `python-json-sort-keys-ascii-compact/v1`: Python `json.dumps(...,sort_keys=True,ensure_ascii=True,separators=(',',':'),allow_nan=False).encode('ascii')`, without trailing newline. This is not RFC8785. Float/integer and negative-zero representations remain byte-significant.

The exported-evaluation checksum hashes the exact exported object. The report checksum hashes the entire report excluding only its own top-level `report_sha256`. These are internal consistency checks, not signatures or origin proofs. Browser/BFF implementations validate original numeric bytes without JavaScript reserialization.

From the repository root:

```sh
python -S -m scripts.validate_training_report training-evaluation-report.json
```

Direct execution from another working directory is also supported:

```sh
python -S /path/to/backend/scripts/validate_training_report.py /path/to/report.json
```

Only Python standard library is needed. No credentials, runtime initialization, network, input writes or optional ML libraries are used. The file must be stable, regular and nonsymlinked. Parser budgets are 512 KiB, nesting 24, 20,000 nodes, 8,192 decoded string characters and 64-character numeric tokens; duplicate keys, unknown keys, noncanonical bytes, nonfinite values and invalid semantics are rejected. Successful validation explicitly reports unverified authenticity.

The old `independent-evaluation-report/v1` schema, route, 80-class imported model protocol and CLI remain unchanged.
