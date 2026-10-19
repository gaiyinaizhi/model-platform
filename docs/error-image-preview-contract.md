# Independent evaluation error-image preview v1

Status: user authorized optional preview expansion on 2026-10-19; independently reviewed before dependent implementation. Baselines: backend main b866af42b04ac7958e665b8fac02fbe2ebf42ad7; frontend main 429ff09fb0c181e308a74a78db9456837ab135d3. No publication or deployment authorized.

## Explicit boundary amendment

The earlier independent-evaluation contract's no-image-byte boundary is amended only for this optional sanitized, authenticated, bounded preview. Detail/list/comparison JSON still returns no image bytes, arbitrary image URLs or local paths. Raw images, arbitrary dataset browsing, TP galleries, tuning, diagnosis, new inference/training, data/metric/model mutation and external fetch remain excluded. Preview availability does not change evidence or metrics.

## Route and privacy ordering

GET /model-evaluations/{mev_32lowercasehex}/error-images/{index}; BFF mirrors beneath /api/real-detection. Index is canonical decimal 0..19 into verified result.wrong_examples.items, not an image ID or dataset index. No query (including bare ?), body, Range/If-Range, encoded/absolute/dot/double-slash paths or transform parameters. Invalid request: 400 invalid_input. Authentication precedes tenant-owned captured joined evaluation/result lookup; missing or foreign: 404 not_found before status/integrity/filesystem inspection. Own non-success: 409 evaluation_not_succeeded. Missing bounded example: 404 not_found. Failed binding/result/prediction/sample evidence: 409 evaluation_integrity. Errors are fixed sanitized messages without paths or upstream detail.

## Integrity and read-only requirements

Verify existing full model/import/dataset/protocol/result/private prediction evidence bindings on the captured successful record. Historical environments are not compared to the current process. Require exactly one bound sample matching the selected diagnostic image identity and original dimensions. Verify raw image SHA256 and decoded RGB SHA256. Every file and ancestor read is descriptor-relative no-follow, regular-file bounded I/O: no symlinks, traversal, FIFO/device, external references or fallback fetch. A safe final image open alone is insufficient if earlier full binding verification follows ancestors. Read-only DatasetStore construction; no schema initialization, migration, audit insertion, persisted preview, DB writes, training, inference, worker or credentials. Existing auth/rate bookkeeping is unchanged.

## Bounded rendering

Source PNG/JPEG single-frame only; at most 8 MiB source and 16,000,000 decoded pixels, with existing stricter supported-image limits preserved. Rendering occurs in a bounded child with global concurrency 2, five-second wall deadline, bounded stdin/stdout, child kill/reap on errors/timeouts and resource limits. No ML dependency import. Decode raw RGB without EXIF transposition or ICC conversion so original-coordinate evidence remains valid. Create a fresh RGB raster with no inherited metadata, shrink-only max edge 1024, floor realized dimensions (minimum 1), preserve aspect up to integer rounding. PNG output at most 2 MiB; over-budget output fails closed. No original/raw-byte response and no filesystem cache. Dependency unavailable: 503 preview_unavailable; timeout: 503 preview_timeout; occupied budget: 429 preview_busy. These logical/process limits are not a production multi-tenant sandbox.

## Binary trust boundary

On success: image/png, exact Content-Length, Cache-Control: no-store, X-Content-Type-Options: nosniff, Cross-Origin-Resource-Policy: same-origin. BFF keeps existing Host/Origin/Fetch gates and server-only credential; fixed authorized backend route only, no redirect following. Dedicated 2 MiB binary response budget, deadline and concurrency do not widen any JSON limit. Buffer and validate complete PNG signature/IHDR/dimensions/structure and unique MIME/length/encoding headers before releasing a success response; reject metadata and malformed/truncated bodies. Preserve sanitized permitted failures, never leak arbitrary upstream bodies. Browser uses same-origin fetch, redirect:error, bounded body/MIME/signature/dimension validation and ephemeral object URL only.

## UI

Explicit single-error click loads only that image. Raster and FP/FN overlays share original-coordinate viewBox, preserve aspect and include edge boxes. Label FP/FN and class; FP includes recorded confidence. Keep original coordinate/source-index details, count/truncation and empty states. No causal diagnosis, quality rank or approval claim. Close, changed example/evaluation, refresh, navigation/unmount and retry abort/fence pending work; stale success/error/finally cannot restore a closed or replaced view, and every replaced/stale/disposed object URL is revoked. Retry is the current exact pair. No persistent image cache.

## Acceptance

Backend auth/tenant-before-integrity 401/404; strict route/status/index/query/range; binding/protocol/result/prediction/raw/RGB/dimension tampering; missing file; ancestor and final symlink/FIFO/traversal; byte/pixel/output budgets; actual timeout kill/reap and concurrency release; fresh PNG metadata stripping/orientation/ratio; SQL/read-only checks; no-GT FP and zero-example cases. BFF own-origin/Host, redirect/MIME/length/chunked/truncated/oversized/errors and unrelated JSON-budget regressions. Real Vue DOM and credential-free synthetic cloud-browser QA for portrait/landscape/edge overlays, close/switch/retry/delayed-error/Back and object-URL lifecycle. Full applicable tests, types/lint/build and independent final-diff review. Synthetic fixture checks do not establish model quality or real retained-record E2E; no real64 data or weights downloaded. Source-only checkpoint excludes credentials, datasets, weights, images/screenshots and generated outputs. Existing publication source pin remains unchanged until separately authorized publication; local source labels must be honest.

Routing clarification: an absolute URL, foreign prefix, fragment or backslash rejected by the existing BFF global route parser may return the existing 404 routing miss rather than endpoint-level400. It must never reach upstream; endpoint-recognized malformed indices, query/range and encoded path variants remain400. No global routing behavior is broadened.
