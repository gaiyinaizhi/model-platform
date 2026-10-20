# Error-image preview verification

2026-10-19, source prepared for an authorized draft pull request. Backend remote main `b866af42b04ac7958e665b8fac02fbe2ebf42ad7`; frontend remote main `429ff09fb0c181e308a74a78db9456837ab135d3` were read and verified before implementation. The approved contract is [error-image-preview-contract.md](error-image-preview-contract.md).

## Scope and safeguards

One authenticated successful evaluation and canonical wrong-example ordinal0..19 selects an optional preview. Ownership precedes state and integrity; full immutable bindings/private prediction evidence and selected raw/RGB image hashes and dimensions are checked. The shared dataset snapshot verifier was hardened to bounded descriptor-relative no-follow reads so preview integrity checks cannot traverse a symlink before the final image open. Existing dataset workflows are included in regression coverage. No database/schema/audit/metric/model mutation, inference, training or external image fetch.

Pillow-only child rendering limits: two active previews, source≤8MiB, decoded≤16,000,000 pixels, longest output edge1024 (floor, shrink-only), PNG≤2MiB, five-second wall timeout, CPU5s/AS512MiB, parent-death termination and kill/reap. Fresh RGB PNG drops all metadata without EXIF/ICC transforms. Pillow is an existing optional image dependency; no dependency version was changed. Missing dependency fails503. This is process/logic budgeting, not a production OS-user or tenant sandbox.

The BFF retains server-only credentials and existing Host/Origin/Fetch gates, enforces a separate binary size/concurrency budget and complete PNG CRC/chunk/decoded scanline validation. Only IHDR/contiguous IDAT/IEND RGB8 PNG is accepted. It does not broaden JSON limits or follow redirects. The browser fetches only one clicked example, verifies size/type/original-derived output dimensions, uses an ephemeral object URL and fences all stale outcomes. SVG raster and FP/FN coordinates share one original-image viewBox. Larger clamped labels and the retained coordinate table show class/FP confidence, source indexes and truncation.

## Automated verification

- Affected Python suites: `test_evaluation_images`, `test_evaluation_comparisons`, `test_model_evaluations`, `test_datasets`, `test_dataset_evaluation`: 111 total,110 passed,1 skipped. All21 new preview tests passed, including real renderer timeout/reap, stdout overrun and parent-death behavior. Existing optional NumPy/OpenCV resize test skipped because OpenCV is unavailable
- Supported Python3.12.14, Pillow12.3.0, PyYAML6.0.3; an additional Python3.13.5/Pillow11.1.0 run also passed the same affected coverage. No unsupported launcher compatibility is claimed
- Full affected frontend/BFF Node aggregate: 467 tests,465 passed,0 failed,2 existing skips requiring unavailable retained real64/ONNX evidence. Includes33 new BFF tests and12 new browser API/actual Vue DOM tests
- BFF and Vue TypeScript checks, changed/new Vue/TS ESLint and component stylelint passed. Python compile check passed
- Fresh forced production build:11 successful tasks,0 cached. Source archive has no .git; Turbo's corresponding status warning is non-fatal
- Launcher regression:21 passed on supported Python3.12.14. An initial3.13 launcher run failed its intentional platform gate and was rerun with the supported interpreter

The affected backend harness substitutes fixed in-memory identity material for older test-only credential-file setup, audits against identity-file creation, and blocks model prediction. It preserves actual Auth/loopback HTTP/SQLite checks. Full historical backend tests with real synthetic training/inference or credential initialization were not run under this task's read-only/no-credential scope. Initial fixture-isolation/type errors were fixed and complete affected suites rerun; they are not outstanding product failures.

## Actual cloud-browser inspection

A temporary credential-free synthetic Vite fixture mounted the real independent-evaluation parent and preview component in cloud Chromium, on loopback only. Synthetic RGB grids and known FP/FN coordinates were used, with an explicit fixture/no-quality notice outside product UI. Verified landscape1501×701→1024×478 and portrait701×1501→478×1024, full-edge FP and bottom/right FN alignment, readable class/confidence labels, original coordinate rows and truncation, one-click lazy loading, safe failure/exact retry, close while delayed error was pending, example switch during slow request, query navigation/Back and full page exit/Back. Fixture DOM counters confirmed URL creation/revocation balances and no stale preview reappeared. Portrait label readability was improved after visual inspection and checks rerun.

This is component/transport and integrity acceptance with generated nonsensitive fixtures, not live retained-evaluation end-to-end acceptance, detector quality evidence, new AP/AR measurement or production approval. Real64 data/model weights were unavailable and were not downloaded. No runtime credentials were created during local verification. Publication is limited to a new draft pull request; no merge or deployment is authorized.

## Source identity and publication

The implementation checkpoint retained the previous frontend lock until publication. The authorized publication sequence creates a backend commit from the verified current main, then pins the frontend launcher to that actual remote commit and refreshes every runtime-source hash, including the new renderer modules. No placeholder SHA or weakened source check is permitted. The paired frontend pull request records the exact backend commit and synchronization verification; source archives are not full Git checkouts and omit unchanged binary assets.

Source archive excludes dependencies, generated builds, credentials and private environment files (unchanged public frontend build settings are retained), runtime state, dataset/model binaries, images and screenshots. Existing historical verification files remain historical; this report describes this increment.
