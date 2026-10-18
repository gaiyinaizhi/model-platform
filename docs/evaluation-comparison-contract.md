# Independent evaluation comparison, v1

Base: model-platform `3c8ce989fdbbaf388aa5027c632cbf2ad3b9819e`; vision-lab `5f00a67dbdff31d80452b7599bf8b93993cfbf86`.

**Independently reviewed and approved before dependent implementation on 2026-10-18.** This contract incorporates the review amendments below. Approval covers this contract, not implementation or acceptance results.

## Route, ownership, and read-only boundary

`GET /model-evaluation-comparisons?evaluation_a_id=mev_<32 lowercase hex>&evaluation_b_id=mev_<32 lowercase hex>`. Require exactly these two unique, nonempty query parameters and two distinct IDs. The BFF mirrors this route under its existing `/api/real-detection` prefix, using its existing server-only credential/session boundary. No caller-provided tenant.

Malformed IDs, duplicate/unknown/missing query parameters, and the same ID return `400 invalid_input`. Same model in distinct evaluations is allowed. Resolve ownership of **both** IDs before checking either record's state or integrity; either missing/foreign returns the existing `404 not_found`, even if the other record is nonterminal or corrupt. Use the existing joined evaluation/result read for each captured row. Do not call `get(A)` before resolving ownership of B. After both ownership checks, require both captured states to be `succeeded`; queued/running/failed return `409 evaluation_not_succeeded`. Then perform existing full binding/model/dataset files/result/private prediction-evidence verification and bounded detail construction on those captured immutable terminal rows. Tamper returns the existing `409 evaluation_integrity`. Reusing captured rows avoids accidental state/result rereads or changing error precedence.

No database writes, audit insertion, persisted comparison, inference, training, model mutation, new credential, or new numerical/ML dependency import. Existing authentication/rate-limit bookkeeping is unchanged. The comparison request must not trigger store initialization or migrations: add a keyword-only read-only DatasetStore construction path that sets read-access fields/locks but skips all mkdir/schema/migration/BEGIN IMMEDIATE operations; binding verification uses that path to retrieve already-published versions. Existing dataset lifecycle construction retains its default initialization behavior. Historical environment fingerprints are compared to each other, not to the current API process environment. There is no quality ranking, winner, production gate, or serving permission.

## Response and bounds

The 200 response has these fields:

```text
{
  schema_version: "independent-evaluation-comparison/v1",
  a: EvaluationDetail,
  b: EvaluationDetail,
  compatible: boolean,
  incompatibilities: [{field: ComparisonField, reason: "<field> differs"}],
  delta: null | {
    map_50: number|null, map_50_95: number|null, ar_100: number|null,
    precision: number|null, recall: number|null,
    tp: integer, fp: integer, fn: integer
  }
}
```

A and B must exactly match the requested IDs in that order and must both be succeeded details with non-null results and result hashes. Existing verified bounded `EvaluationDetail` is reused for identities, protocol, ordered categories, evidence digests, results, warnings, and scope. Model identity/artifact and prediction/result/binding digests may differ and are displayed rather than required to match.

The entire encoded response is at most **1100 KiB (1,126,400 bytes)**, with each detail still bounded to 512 KiB. Apply this limit to backend serialization, BFF raw upstream body reading, BFF raw/projected comparison validation, BFF outbound serialization, and browser response validation. Narrowly add the new comparison route's limit; do not broaden other HTTP route limits. Existing evaluation detail/list limits remain unchanged.

## Exact compatibility rules

Comparison is strict structural equality: object key order is irrelevant; array order and every value are significant. No numeric tolerance, version-range compatibility, platform equivalence, inferred dataset equivalence, or ignored code/environment difference. `null` source selection compares equal only to `null`.

Return exactly one incompatibility for each unequal row below, in this fixed order, with literal reason `<field> differs`. Do not include value dumps, arbitrary upstream messages, duplicate reasons, or unrecognized field IDs. This gives a maximum of 21 incompatibilities; UI labels are localized from the fixed IDs.

| ComparisonField | Value extracted from each verified detail |
| --- | --- |
| `dataset.version` | `binding.dataset.dataset_version_id` |
| `dataset.content` | `binding.dataset.content_sha256` |
| `dataset.samples` | Entire ordered `binding.dataset.samples` array |
| `dataset.source_selection` | `binding.dataset.source.selection_sha256`, or null if source is null |
| `split` | `binding.dataset.split` (must already be val and agree with summary/result scope) |
| `categories` | Entire ordered `binding.categories` array |
| `category_mapping` | `binding.category_mapping_sha256` |
| `protocol` | Entire `binding.protocol` object |
| `protocol_digest` | `binding.protocol_sha256` |
| `metric_protocol` | `result.metrics.protocol` |
| `metric_evaluator` | Entire `result.metrics.evaluator` object |
| `metric_parameters` | Entire `result.metrics.parameters` object |
| `operating_point` | Object containing only `result.operating_point`'s `protocol`, `iou_threshold`, `score_threshold`, and `score_operator` |
| `submission.code` | `binding.submission_environment.code_sha256` |
| `submission.python` | `binding.submission_environment.python` |
| `submission.platform` | `binding.submission_environment.platform` |
| `submission.dependencies` | Parsed dependency mapping from `binding.submission_environment.dependencies` |
| `execution.code` | `result.environment.code_sha256` |
| `execution.python` | `result.environment.python` |
| `execution.platform` | `result.environment.platform` |
| `execution.dependencies` | Entire `result.environment.dependencies` mapping |

Submission dependencies are an existing canonical JSON string containing six packages, including onnx. Execution dependencies are an existing five-package object without onnx. Parse and structurally compare submission A to submission B, and execution A to execution B. Do not compare these two differently shaped representations to each other, and do not compare dependency strings by raw JSON key order. Keep the existing detail validator's canonical submission-string requirement; semantic comparison does not relax detail validation.

Existing per-detail validity and integrity checks run before compatibility. Unsupported or corrupt fixed-v1 protocols/categories, missing fields, nonfinite metrics, invalid counts, and inconsistent per-detail digests do not become ordinary incompatibility; backend integrity failures and BFF malformed-upstream rejection remain failures. Synthetic comparator-unit fixtures may exercise otherwise unreachable mismatches, clearly labeled as fixtures; they do not authorize weakening existing fixed-v1 validation.

`compatible` is true exactly when the ordered incompatibilities array is empty. Otherwise `delta` is null. A compatible pair always has the complete delta object. Every delta is **B minus A**, calculated from the three `result.metrics` aggregate rates and the five `result.operating_point.totals` fields. Null on either rate side stays null. All non-null rate deltas are finite in [-1,1]; signed count deltas are exact integers. No metric/count result, diagnostic example, timestamp, model ID, result hash, or prediction hash is itself a compatibility prerequisite.

## BFF trust boundary

The BFF must validate both raw details with the existing `publicEvaluation` boundary, reject non-succeeded details, and verify the requested A/B identities and schema. Independently recompute the exact ordered incompatibilities, compatible flag, and B-minus-A deltas, and reject disagreement as `502 upstream_invalid_response`; do not trust upstream arithmetic or compatibility booleans.

Before compatibility recomputation, compare each detail's complete original recorded comparison fields to the corresponding `publicEvaluation`-projected fields using the same structural extraction rules. Existing projection normalizes protocol/evaluator/parameters and may erase unknown metadata. Any comparison-relevant projection loss is `502 upstream_invalid_response`, even if both raw sides carry the same unknown extension; do not forward arbitrary raw metadata. This fail-closed v1 rule keeps the public details sufficient for browser recomputation. Then independently recompute compatibility from the validated projected fields, which are now proven equal to the raw comparison values. Only existing allowlisted detail data and the fixed comparison response fields are forwarded; no arbitrary raw object or error text is exposed. Preserve backend-issued result hashes; the BFF/browser do not claim to recompute complete Python floating-point result hashes or private prediction evidence.

No whole-object canonical hashes are needed to compare metrics: compare parsed values structurally and compute deltas directly. Floating-point deltas use direct subtraction in both runtimes with normal finite JSON numbers; count deltas are exact. Reject altered delta values, missing/extra delta fields, duplicate/unknown/out-of-order reasons, A/B reversal, impossible compatibility, incompatible responses with deltas, and compatible responses without deltas.

## UI behavior

Provide a separate comparison panel on pretrained-import, with tenant-wide latest-100 successful history records and explicit model identity for each selection. This is a bounded recent history window, not complete history or pagination. Fetch the corresponding model identity for each A/B as needed, allowing both same-model repeats and different-model evaluations. Preserve A/B ordering and do not create a new evaluation as part of comparison.

Show compatibility and all reasons, side-by-side AP50/AP50:95/AR@100, distinct operating-point precision/recall, TP/FP/FN, and signed deltas plus model/data/protocol/result/prediction identities. Metrics remain fractions on the wire; format rate deltas as signed percentage points and count deltas as signed integers. Zero is neutral, never an improvement badge. Incompatible pairs show their individual evidence and reasons with no calculated delta or inferred winner.

Selection changes immediately invalidate prior results. Only the current pair/request may update result, error, loading, or retry state. Reset, navigation/disposal, newer requests, and failed refreshes fence stale success **and** stale error/finally callbacks. Reload either reconstructs the pair from validated URL IDs or clearly requires reselection without retaining old results. Retry targets the current exact pair; it does not reuse a superseded pair or start inference. Existing imported registration provenance, old training comparison, serving, deployment, and quality restrictions remain unchanged.

## Acceptance

- HTTP query validation; missing/foreign ownership precedence with either side nonterminal/tampered; distinct-ID requirement; succeeded-only behavior
- Full binding/model/dataset/result/prediction tamper rejection; existing per-detail bounds; no database, result, audit, or provenance mutation and no inference/dependency import; instrument SQLite statements/authorizer so schema initialization and write-transaction attempts are detected even when row counts do not change
- Every compatibility-field fixture; semantic object/dependency ordering; ordered sample/category differences; strict code/Python/platform/dependency differences; different dataset versions with identical contents remain incompatible
- Same-model repeated evaluations allowed; different models allowed when evidence is compatible; null preservation; positive/negative/zero B-minus-A arithmetic; swapped pair negates defined deltas
- BFF malformed, swapped, inconsistent compatibility/reason/arithmetic responses and comparison-relevant projection loss/unknown extensions rejected; raw and projected comparison bounds; near-limit valid pair above 1 MiB accepted without raising unrelated limits
- UI stale success/error/finally, reset, disposal/navigation, reload, changed selection while request pending, duplicate compare click, and exact-pair retry
- Existing retained real repeated records can prove zero deltas without inference; fixtures remain labeled as fixtures. The current one artifact hash is repeatability evidence, not learned-model ranking
- Full affected tests, typechecks/build, and cloud browser QA; source-only Library checkpoint excludes credentials/assets/screenshots

## Review record

Read both AGENTS files, backend README/architecture/roadmap, existing independent-evaluation contract/docs, `model_evaluations.py` binding/get/result verification, backend API query routing, BFF `evaluations.ts`/source validators/server limits, frontend evaluation API and panel, and synthetic fixture boundaries. No implementation was edited and no implementation test pass is claimed by this contract review. The frontend AGENTS-linked `ai/trellis/authority.md` and `ai/workflows/feature-lifecycle.md` remain absent in this verified base; no delivery authority or Trellis state is invented.

Approved scope: read-only comparison with strict code/dependency/environment equality, both-ownership-first privacy precedence, existing integrity checks, route-specific bounded transport, validated/recomputed BFF response, and race-safe UI. Implementation must be reviewed and verified separately.
