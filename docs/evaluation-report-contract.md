# Independent evaluation report v1

Approved contract, local implementation on backend baseline `85561ba`; publication and acceptance are separate. This is a bounded report for one independent imported-model evaluation, not a comparison bundle, production grade, certification, or re-evaluation.

## Download

`GET /model-evaluations/{mev_id}/report` is authenticated and body-free. The exact path is required: no encoded path, query (including empty `?`), trailing or duplicate slash, absolute URL, range, transfer/content encoding, nonzero or duplicate Content-Length. Accept and Accept-Encoding do not negotiate a different representation; the response is always uncompressed canonical JSON. Other existing routes are unchanged.

Ownership is resolved first: unknown/foreign records return 404. Queued/running/failed return 409 `evaluation_not_succeeded`. A captured succeeded row is verified using existing model/dataset binding, result and private prediction evidence checks, then the existing strict comparison semantic validator, then the report schema. Invalid evidence returns 409 `evaluation_integrity`. No inference, database writes, key generation or current-environment fingerprinting occurs.

Response: at most 512 KiB, `application/json`, `Content-Disposition: attachment; filename="independent-evaluation-report.json"`, fixed Content-Length, Cache-Control no-store, X-Content-Type-Options nosniff. Repeated downloads of unchanged evidence have identical bytes.

## Schema and disclosure boundary

Exact schema: `independent-evaluation-report/v1`. It contains evaluation identity/state/timestamps, model's nine safe binding fields, dataset identity/content hash/purpose/split/counts/source-selection hash, fixed 80 categories and protocol with their hashes, original submission environment, aggregate/full 80-class metrics and operating point, execution environment, scope, and the six original fixed warning strings. All nested fields are allowlisted and semantic constraints checked. There are no sample paths, wrong examples, raw predictions, images, weights or arbitrary warning text. The submission environment remains an object; only its dependencies field is the original canonical encoded JSON string.

`source_result_sha256`, `source_binding_sha256`, and `source_prediction_sha256` are references only. Original bytes are absent, so offline verification cannot recompute them. `exported_result_sha256` hashes the exported result; `report_sha256` hashes the entire report with only that top-level field excluded. Both are recomputable, unkeyed checksums, never signatures.

Canonicalization: `python-json-sort-keys-ascii-compact/v1`, specifically Python `json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(',', ':'), allow_nan=False).encode('ascii')`, without newline or BOM. This is Python-specific, not RFC8785/JCS. Integer/float and negative-zero representations remain distinct; Unicode is escaped without normalization.

The verification and limitations objects state internal consistency only, no trusted authenticity/signature/re-evaluation, absent source bytes, no production allowance/official COCO benchmark/device certification/weight license verification, and unknown pretraining overlap. Anyone editing the report can recompute its checksums. Passing does not prove origin or genuine model performance.

## Offline command

From this repository root, using Python 3.10+ and only its standard library:

```sh
python -S -m scripts.validate_evaluation_report independent-evaluation-report.json
```

Direct invocation also works from any working directory without PYTHONPATH:

```sh
python -S /path/to/backend/scripts/validate_evaluation_report.py /path/to/report.json
```

This resolves only the source tree containing the trusted validator script, never code adjacent to the input report.

No optional ML dependencies, runtime setup, network, credentials, or input-file writes are used. The closed schema dispatcher imports only the statically named portable trained-report module for its exact discriminator. `evaluation_report_schema.py` is the shared server/offline implementation. Its pinned protocol/category constants and mirrored metric helpers have parity tests against existing evaluation code; changes require coordinated contract review.

The CLI accepts one local regular file, reads at most 512 KiB + 1, and rejects unstable reads, invalid UTF-8/JSON, duplicate keys, noncanonical bytes, unknown fields, invalid types/finite ranges/counts/ratios/categories/environments/scope or digests. Limits: preparse nesting 24, decoded nodes 20,000, decoded strings 8,192 characters, numeric tokens 64 characters. It rejects huge string tokens before JSON decoding. It exits nonzero with a concise error that does not echo file contents or paths. Success explicitly says internal consistency, unverified authenticity and reference-only source hashes.

## Verification status

Focused tests cover repeated bytes, no disclosure, every nested object rejecting unknown keys, rehashed semantic corruption, ownership/state/tamper precedence, strict route/header/auth behavior, parser resource limits/duplicates/truncation/nonfinite values, regular-file handling, and CLI execution under Python `-S`. Fixtures are synthetic and do not establish model quality. Actual commands and results are recorded separately after visible Terminal execution; no passing claim is made here before execution.


## Trained protocol extension

The same endpoint and CLI additionally accept the separate `dataset-trained-independent-evaluation-report/v1` schema; see [trained report contract](dataset-evaluation-report.md). All imported schema fields and output bytes remain unchanged. Never infer that an imported-v1 report accepts trained model fields or thresholds.
