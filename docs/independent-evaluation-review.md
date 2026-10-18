# Independent evaluation contract review

2026-10-18. Independent review of the proposed imported-model evaluation increment. This document records review findings, not completed feature acceptance.

## Current decision

**Approved for dependent API/UI implementation at 14:19 UTC.** The reviewed contract SHA-256 was `1b9a2c2d886ec0dc4c3e5d2f6f1daaa14e508e57dacca2197669aa9cdf6e8e22`. This is design approval only, not feature or real-data/browser acceptance. The draft's feature scope, tenant ownership, immutable separation from model registration, worker lifecycle, fixed inference protocol, metric definitions and pre-inference subset selection are appropriate.

The following contract clarifications were requested before implementing the cross-layer API:

1. Specify exact binding, result, metric, operating-point, example and digest field names/types/nullability. List summaries must omit heavy result/binding payloads.
2. Resolve the public detail response budget versus the existing 1 MiB BFF response limit. The 8 MiB subprocess result budget is separate. Bound per-example boxes and per-image matching arrays, or introduce a narrowly scoped evaluation response limit.
3. Bind both original decoded RGB and max384 request RGB hashes to prediction evidence; define the pixel/hash encoding. Original encoded JPEG hashes alone do not identify the exact inference input.
4. Specify where the complete frozen selection/source manifest is persisted and verified. The five-field source declaration alone does not contain the required image IDs, original annotation IDs, exclusions and licensing metadata.
5. Preserve replay compatibility for existing dataset imports whose idempotency digest was calculated from an omitted-purpose request. Legacy manifest compatibility alone is insufficient.

All five clarifications were resolved in sections 2, 5, 8 and 9: exact wire schema; 512 KiB detail/256 KiB list response limits with bounded diagnostics; original and request RGB digests; complete bounded source manifest bound to its canonical hash and snapshot; original-body legacy idempotency hashing. Three final consistency fixes were verified: `model_` ID prefix, globally unique source annotation IDs and per-image annotation count equal to validated label rows, and explicit SHA256(UTF8(seed+":"+decimal(image_id))) selection ordering with numeric-ID tie break.

The author, parent, frontend owner and dataset owner were notified of approval directly. No feature implementation was changed by this reviewer. Canonical source/result hashing must consistently reuse `common.encode`/`digest`; the independently retained original annotations and acquisition/attribution evidence remain required for the real acceptance run.

At 14:19:48 UTC the author reported that the already frozen preparer used a 12-digit zero-padded image ID rather than unpadded decimal ID in the selection hash. This precision correction is accepted: retain SHA256(ASCII(seed+":"+12-digit-zero-padded-image-id)), ascending digest then numeric ID, and do not reselect. The author reported frozen source manifest SHA-256 `6638afe4a6a9d88169781730d994f709d4011336c82023a1a14cb3112a507f4a`; this report is not independent verification of that manifest's bytes. Design approval stands with the corrected algorithm documented.

## Important integration constraints

- Existing `imports.ts` correctly requires imported registration-time `evaluation:null` and `local_evaluation_status:not_evaluated`; the new independent result must not rewrite those facts.
- Existing dataset BFF validators reconstruct whitelisted fields and will drop new purpose/source/RGB metadata unless explicitly extended.
- Existing BFF query routing, response budgets and error-code/status allowlist need exact, narrow evaluation-route additions. Do not relax unrelated request or serving limits.
- The evaluator's internal `class_id+1` numbering may be retained. A consistent bijection is metrically equivalent to using sparse COCO IDs. The immutable public/source mapping must distinguish the model class index, original sparse source category ID, and temporary evaluator category ID. All 80 classes and every prediction/selected-image annotation must remain present.
- Fixed serving threshold/NMS/top100 metrics are not a full or low-threshold COCO benchmark. Matching outcomes at IoU .5 are not causal diagnoses; operating-point recall and AR@100 must remain separate.
- Frontend AGENTS references `ai/trellis/authority.md` and `ai/workflows/feature-lifecycle.md`, but those files are absent in this workspace. No delivery-state workflow verification is claimed.

## Metrics code review and executed checks

Reviewed the parent's `ar_100` addition and `match_detections` implementation in `model_platform/detection_evaluation.py`.

- AR uses the accumulated recall tensor at area all/maxDets100 and averages all ten IoU thresholds over categories with GT. Per-class AR and undefined/null cases are handled correctly.
- Matching uses descending score, original prediction-order ties, highest-IoU unmatched same-class GT and original annotation-order ties. Duplicate/wrong-class predictions remain FP, unmatched GT remain FN, empty denominators remain null, and inputs are not modified.
- The helper's potentially large wrong-example payload is acceptable internally only if the endpoint applies its declared serialization budgets without changing metric inputs or prediction digests.

Executed from the backend directory with the available shared Python 3.12 environment:

```text
/workspace/shared/pretrained-rebuild-20261018/venv/bin/python -m unittest discover -s tests -p test_detection_evaluation.py -v
26 tests passed; no skips

/workspace/shared/pretrained-rebuild-20261018/venv/bin/python -m unittest discover -s tests -p test_independent_metrics.py -v
8 tests passed; no skips
```

These are synthetic arithmetic/validation regressions. They are not real imported-artifact, fixed64, API, lifecycle or browser acceptance evidence.

## Acceptance matrix identified during design review

- Old omitted-purpose replay; explicit training/evaluation separation; source/purpose immutability and val-only training rejection
- Unknown/foreign model/dataset/evaluation IDs, same-tenant ID mix-and-match, class order/sparse-map mismatch, strict unknown fields and conflicting idempotency requests
- Model/dataset/source/request-pixel/result/prediction tamper, execution-code changes, malformed JSON, nonfinite metrics and resource-budget refusal
- Transactional duplicate claim/commit, shared training/evaluation capacity and ordering, process interruption, orphan-child cleanup, API-only restart, worker restart and timeout classified as failure
- Empty predictions, GT-free categories, negative images, cross-class overlap, duplicate predictions, exact-IoU boundary/tie ordering, bounded example truncation without metric-input filtering
- All64 frozen images and all selected-image annotations/predictions, per-image acquisition/license/exclusion evidence, original/request RGB hashes and actual imported-model inference
- Actual BFF/UI submit/poll/reload/results/failure evidence, with truthful registration-time provenance retained

## Implementation re-review and final evidence audit

Reviewed the implemented evaluation queue/store, API, worker wiring, purpose/source validation, BFF validators and browser operation journal. The 14:32 UTC source freeze was superseded after the broader optional suite found a legacy dataset-serving compatibility regression. The narrow correction and two further lifecycle fixes described below passed independent re-review; final backend source freeze was approved at 14:44 UTC. No blocking implementation finding remains in the reviewed scope.

- Evaluation reads originally fetched mutable state and terminal result in separate SQLite snapshots, allowing a normal running-to-success transition to produce a false integrity error. Detail and list now use a coherent joined snapshot; a deterministic race regression passes.
- Evidence publication originally protected only its final filename against symlinks. It now walks all directory components using no-follow directory descriptors, publishes exclusively through the descriptor, fsyncs and rereads the bytes before commit. A post-initialization root-symlink regression rejects redirection.
- The browser originally treated first-request `409 evaluation_integrity` as proof of rejection, although a post-insert integrity read can fail. It now preserves the original request key/body and blocks rotation until reconciliation.
- The BFF now preserves the new classified terminal errors and verifies reported Python/platform/dependencies against the pinned submission environment. Its documentation correctly delegates complete Python-float result hashing and private prediction integrity verification to the backend.
- Dataset version lists use compact five-field source summaries; full source manifests remain in details and immutable bindings. Source IDs/counts are JSON-safe integers; canonical Unicode hashing and safe nested source paths are covered.
- Bound-resource revalidation now normalizes resource failures to `evaluation_integrity`, preventing post-insert errors from masquerading as definite request rejection. Initial pre-insert validation still returns the original classified errors.
- Publication now rechecks the deadline after acquiring the SQLite write transaction and immediately before inserting the terminal result. A write-lock delay crossing the deadline cannot commit success.

Independent executed focused checks on the final implementation: 115 backend tests across model evaluations (22), evaluation-purpose datasets (16), legacy datasets (30), dataset training (7), detector evaluation (26), independent matching/AR (10), and legacy real-ONNX serving/compatibility (4); 29 BFF evaluation tests including the refreshed actual final fixed64 fixture; 31 browser dataset/evaluation domain tests. All passed, with no skips. These focused counts do not claim the entire repository suite or visible-browser acceptance.

The independent final evidence audit is recorded in [independent-evaluation-review-evidence.json](independent-evaluation-review-evidence.json). Both final records have identical binding, result and full prediction evidence. The review verified all 64 original encoded-image, label, decoded-RGB and resized-request-RGB hashes, the full 80-class sparse source mapping, and the unchanged imported manifest. Training runs, attempts and training-bound evaluation tables remain empty in the dedicated acceptance runtime.

Recomputed from the complete 457 GT boxes and 273 saved predictions: AP50 `0.41460834340789027`, AP50:95 `0.2747798977167655`, AR@100 `0.288550124601401`; TP 190, FP 83, FN 267. Every per-class result and bounded wrong-example diagnostic also matched exactly. The 64-image subset contains 60 classes with GT and two negative images. This verifies stored evidence and arithmetic, not absence of pretraining overlap, independent licensing certification, full official COCO performance or production quality.

Final reviewed execution source SHA-256: `ba40e485af5ee0237dd055ce94c5084f97f85a0c54aece699ef13695d2e0786c`. Prediction SHA-256: `ef25f77939bfd3de12b1bd66443bc9547a5b199125d094ed8f8a2d2f414f0d25`. Result SHA-256: `a7256a3458a97466cdbaa12ff754b5aa8cf568429a444a12dbbc7d835d750716`.

Visible browser/API acceptance was performed and reported by the parent reviewer using its cloud Chromium and local loopback services. This reviewer independently cross-checked the two resulting records directly against read-only SQLite and all actual image, label and prediction files; [browser persistence review evidence](independent-evaluation-browser-review-evidence.json) records the result. Both browser records have the final source hash, identical complete predictions/metrics, unchanged original imported manifest bytes and zero training/deployment records. Their result digest differs from the worker pair because their bound model/dataset identities differ, as expected. This cross-check does not claim that this reviewer personally drove or visually inspected the browser.

The parent's final broader optional-suite log and result were inspected at 14:47 UTC: 344 tests passed, zero failures/errors/skips. The only two explicit exclusions were the official Fashion cache verification and pinned upstream-reference helper test; authorized real imported-artifact tests were included. The earlier failing broad-suite result is retained separately, rather than counted as passing. No images, model weights, credentials or private runtime files were added to this review report. Controlled local-service and temporary-credential cleanup remains the parent's final handoff check.

## Legacy compatibility follow-through

The shared evaluator's additional AR parameter metadata broke the existing exact dataset-serving protocol validator and would also change legacy trained-model comparison signatures. The approved narrow correction keeps shared evaluator parameters byte-for-byte legacy-shaped, retains additive AR metric values, and attaches the two AR parameter metadata fields only in the independent-evaluation wrapper. This avoids changing existing serving validation, historical manifests or comparison identity. The real-ONNX lifecycle and explicit comparison-identity regression passed independently. The refreshed source-bound pair was independently audited at 14:45 UTC, including exact public AR metadata and all stored metrics. The preceding source's historical audit is separately retained and marked superseded.
