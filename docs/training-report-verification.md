# Training report verification

This slice adds read-only `training-evaluation-report/v1` for succeeded dataset-trained models. Synthetic fixtures are schema/security tests, never model-quality evidence.

## Automated checks

- `python -m unittest discover -s tests -p 'test_training_reports.py' -v`: 19 passed. Covers deterministic fixture bytes, owner-first failures, cross-table evidence tampering, exact schema and semantic rehash attacks, actual AP zero versus unavailable metrics, distinct submission/execution platforms, bounded canonical parsing, strict HTTP request framing, standalone stdlib offline validation and source read-only behavior.
- `python -m unittest discover -s tests -p 'test_evaluation_report*.py' -v`: 11 passed; existing independent imported-COCO report regressions.
- `python -m unittest discover -s tests -v`: 464 tests, OK with 55 environment-dependent skips, 61.226 seconds.
- Optional CPU suite, final complete environment: 458 selected tests in 130.745 seconds, zero failures and zero skips, runner exit 0. The same six asset-dependent integrations remain explicitly excluded by the existing runner; exclusions and test gates were not changed. Previous 439 selected tests plus 19 new report tests account for the new total.
- Environment diagnosis: the shared training venv initially lacked OpenCV, giving six skips (one geometry and five adapter/protocol tests). An independent ordinary-copy venv retained the training dependencies and added the existing locked opencv-python-headless 4.13.0.92 distribution from the preview environment; all 104 installed RECORD hashes were verified. No shared/QA venv or product lock changed. The initial 458-test run with six skips is retained as diagnostic evidence, not claimed as a gate pass.
- `python -m compileall -q model_platform scripts tests`: passed.

Optional CPU suite command (requires the existing optional ML environment): `PYTHONPATH=. python scripts/run_optional_regressions.py`. Running the script directly without `PYTHONPATH=.` does not import repository modules correctly; it stops during exclusion validation rather than testing product behavior.

## Read-only coverage

The new test suite enables SQLite `PRAGMA query_only=ON` and traces report statements, accepting only BEGIN, SELECT and COMMIT. It also rejects attempted dataset-store construction, source file access and current environment probing. The offline CLI runs from outside the repository using `python -S`, with before/after file snapshots. These tests cover the report path; they do not claim training itself is read-only.

## Acceptance boundaries

Real CPU training, download, offline verification and browser/BFF acceptance are recorded separately in the integration evidence for this slice. The synthetic canonical fixture is not a substitute for that run. Existing published source locks remain unchanged; local acceptance must use an accurate temporary lock for the extended source and existing development preflight.

Checksums establish report internal consistency only. They do not authenticate the producing server or confirm the current bytes of the artifact or dataset. No production quality or official benchmark claim is made.
