# Independent-evaluation child supervision

This increment strengthens only the execution lifecycle of the existing imported-COCO independent evaluator. It does not add evaluation for trained dataset models, cancellation, automatic retry, a new schema, or a public API. Measurement and inference protocols remain unchanged.

## Ownership and parent death

The single Worker passes its existing close-only lease descriptor to the evaluator. The parent and child retain the same locked open-file description; neither explicitly unlocks it. A successor cannot recover or claim work until the old evaluator has released its inherited lease, even if its parent has died during interpreter startup.

A minimal Linux bootstrap installs `PR_SET_PDEATHSIG(SIGKILL)` and verifies its expected parent before importing project or numerical code. This protects the controlled direct child even while native code holds Python's GIL. The existing Worker preflight checks kernel support before recovery or queue claims. Direct internal evaluation execution also obtains the Worker lease; a collision leaves the existing record untouched.

This is not an OS sandbox, a guarantee of instant kernel termination, or management of arbitrary descendants. No cgroup, container, separate OS identity, or new production multi-tenant security boundary is introduced. A child whose termination is not yet confirmed may continue to retain the lease and block progress.

## Termination before terminal state

The parent reads the private result pipe incrementally against one monotonic deadline. The existing 300-second wall, 240-second CPU, 2-GiB address-space and 8-MiB output budgets remain. Existing load and per-image inference time limits are unchanged. Receiving a complete result or pipe EOF does not prove that the child exited.

The parent writes success, failure and result evidence only after it has reaped its own child, or has proof no child was started. Successful publication specifically requires `child_reaped`; a pre-spawn failure may use `never_started`. Cleanup addresses only the actual `Popen` handle, never a stored PID. If process construction or kill/reap outcome is uncertain, execution raises a fail-closed exception that exits the Worker loop. The record remains `running`, still consumes capacity, and no next task is claimed. A later Worker may mark it `failed/evaluation_interrupted` only after acquiring the inherited lease. There is no automatic rerun.

## Read-only child and private transport

The child opens existing SQLite data with `mode=ro` and `query_only=ON`. It reuses the existing model/data validation reads, but does not invoke runtime/schema/cursor initialization or expose queue/result publication methods. It creates no runtime or evidence directory. SQLite may still create or update transient WAL/shared-memory coordination sidecars when opening a read-only connection. Only the parent writes immutable prediction evidence and terminal records. This is a programming boundary for reviewed code, not a hostile-code sandbox.

The isolated interpreter verifies loaded source before project import and uses a minimal environment plus dedicated request/result descriptors. Requests are at most 16 KiB and have exactly `home`, `evaluation_id`, and the bounded existing `loaded_environment`. The single success document has exactly `result` (object) and `predictions` (array); the single error document has exactly one allowlisted existing `error_code`. Unknown/mixed fields, duplicate keys, nonfinite or overflowed numbers, extra JSON documents, oversized output and nonzero child exit are rejected. Raw exception text is not returned.

## Compatibility and verification

Public API, statuses, idempotency, queue fairness/capacity, imported-asset restriction, metrics, report/comparison formats and database schema are unchanged. No migration is required. Existing stored result/evidence bytes are untouched. New execution source hashes necessarily differ; byte-equivalence checks compare the unchanged measurement function and new transport under the same final source/environment, not hashes across source revisions.

Focused tests are `tests/test_evaluation_process.py` (15 lifecycle/request/read-only tests), `tests/test_evaluation_process_metrics.py` (2 metric transport tests), and the existing `tests/test_model_evaluations.py`. Fault injection is confined to test-owned processes and self-generated temporary data. Metric-equivalence fixtures use deterministic detector outputs with real image resizing and official pycocotools calculations; they are not actual imported-asset inference or recognition-quality evidence. Full commands and completed results are recorded in the accompanying verification evidence, not inferred from test existence.

The frontend's published source pin remains unchanged. Using this local increment requires an explicitly selected matching backend; this work does not publish, upgrade a production runtime, deploy a service, or merge a repository change.


### Completed local verification (2026-10-22)

The frozen source passed both complete backend runs: optional Python environment 583 tests, 577 passed and 6 explicitly skipped for unavailable official cache/private reviewed assets; standard-library-only Python 583 tests, 449 passed and 134 explicitly skipped for optional prerequisites. The 15 new lifecycle tests passed in both; the 2 new measurement-transport tests passed with optional dependencies and skip without them. `compileall` and `pip check` passed. No historical 15-second Worker timeout recurred.

Unchanged frontend compatibility regressions also passed: 489 frontend tests; 355 BFF tests with 2 explicit actual-asset/real64-fixture skips; 50 launcher-script tests; both typechecks and the production build. These do not substitute for a fresh actual-model or browser acceptance run. No real private weights or user dataset was accessed for this increment. No frontend source or backend pin changed.
