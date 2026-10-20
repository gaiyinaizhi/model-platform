"""Offline optional CI suite, with a spawn-safe entry point."""

def main():
    from collections import Counter
    import unittest
    discovered = unittest.defaultTestLoader.discover("tests")
    excluded_ids = {
        "test_fashion_data.OptionalOfficialCacheTests.test_full_source_is_reverified_offline": "official Fashion cache",
        "test_pretrained_yolox.ActualAssetTests.test_validation_load_predict_tamper": "private pretrained asset",
        "test_pretrained_yolox.UpstreamParityTests.test_exact_upstream_helpers": "private asset and pinned upstream sources",
        "test_import_serving.ImportedServingTests.test_lifecycle_restart_stop_replay": "private pretrained asset",
        "test_import_serving.ImportedServingTests.test_ownership_no_download_or_fake_evaluation": "private pretrained asset",
        "test_import_serving.ImportedServingTests.test_corruption_never_ready": "private pretrained asset",
    }
    def flatten(suite):
        for test in suite:
            if isinstance(test, unittest.TestSuite):
                yield from flatten(test)
            else:
                yield test
    tests = list(flatten(discovered))
    counts = Counter(test.id() for test in tests)
    for excluded_id, reason in excluded_ids.items():
        if counts[excluded_id] != 1:
            raise SystemExit(f"Expected exactly one {excluded_id}; discovered {counts[excluded_id]}")
        print(f"Explicitly excluded integration: {excluded_id} ({reason})")
    suite = unittest.TestSuite(test for test in tests if test.id() not in excluded_ids)
    print(f"Discovered {len(tests)} tests; selected {suite.countTestCases()}; explicitly excluded {len(excluded_ids)}")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.testsRun or result.skipped or not result.wasSuccessful():
        raise SystemExit("Optional ML tests must run successfully with no skips")


if __name__ == "__main__":
    main()
