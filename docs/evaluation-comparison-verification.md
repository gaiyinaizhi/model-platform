# Independent evaluation comparison verification

2026-10-18, isolated assistant-cloud source. Work in progress; no remote publication or deployment.

## Completed

- Base main trees independently fetched from GitHub and all available source blobs matched: backend `3c8ce989fdbbaf388aa5027c632cbf2ad3b9819e`, frontend `5f00a67dbdff31d80452b7599bf8b93993cfbf86`. The unchanged frontend demo video was not materialized and must be preserved from the baseline.
- API contract independently reviewed before dependent implementation.
- `PYTHONPATH=tests python -m unittest test_evaluation_comparisons -v`: 22 tests passed, including HTTP/ownership/state/tamper/null/direction/read-only cases. Synthetic fixtures; no model-quality claim. This suite stubs pagination signing material and HTTP identity in memory and generates no credentials.
- `python -m compileall -q model_platform`: passed.
- Read-only retained-evidence verification: both final real repeat pairs yield zero across all eight metric/count deltas with identical prediction digests. SELECT-only SQL traces and unchanged database bytes verified; no new inference, data selection or model execution.
- An actual retained cross-code pair is incompatible on submission/execution code even though prediction digests match; delta is null.

Detailed identities and measurements: [retained evidence](evaluation-comparison-retained-evidence.json). These are previously computed 64-image COCO subset measurements, not a new benchmark, model ranking, production gate or official COCO result.

## Aggregate and review

- Final frozen optional regression: the audited `offline_optional.py` harness with `PYTHONPATH=.` from the backend source: **362 passed, no skips/failures/errors**. It explicitly excludes the six existing official-cache/private-asset integrations: official Fashion cache, private pretrained validation, pinned upstream parity, and three private imported-serving tests. No new real-model inference or evaluation was requested by this comparison slice.
- Final standard-library run: `python -S -m unittest discover -s tests -v`: **368 discovered, 259 passed, 109 expected optional-dependency skips**, no failures/errors.
- Independent implementation re-review approved the fixed backend: numeric JSON representation parity, malformed persisted JSON classification, comparison-critical schema/count/environment checks, no numerical imports, and real retained read-only checks.
- Additional validator review accepted five retained real records, rejected 25 malformed mutations and three invalid null substitutions, and accepted the synthetic empty-ground-truth boundary. Diagnostic geometry/warnings retain existing BFF validation.
- An initial aggregate was invalidated by concurrent runtime-source edits (`execution_code_changed` in three tests). Those edits were then frozen; the final complete aggregates above passed.

Frontend/browser final acceptance and temporary credential cleanup are recorded separately in the final source delivery. No remote writes, PR edits, merges or deployments were performed.

The published launcher pins and source hashes remain unchanged. This increment is tested with an isolated unpublished-source harness; it requires a later separately authorized backend/frontend publication and launcher synchronization before the published launcher can serve the new endpoint.
