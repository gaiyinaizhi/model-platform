"""Offline tests of the pinned official-data reader; fixtures are synthetic."""
import hashlib
import unittest
from urllib.request import Request

from scripts.fetch_digits import OfficialRedirectsOnly, audit_split, parse_rows, verify


class DigitsSourceTests(unittest.TestCase):
    def source(self):
        return '\n'.join(','.join(map(str, [label] * 64 + [label])) for label in range(10)).encode()

    def test_integer_csv_and_complete_count(self):
        rows = parse_rows(self.source(), 10)
        self.assertEqual(len(rows), 10)
        self.assertEqual(rows[9], {'x': [9] * 64, 'y': 9})
        for raw, count in ((self.source(), 11), (b'header\n' + self.source(), 10),
                           (self.source().replace(b'0,', b'0.0,', 1), 10),
                           (self.source().replace(b'0,', b'17,', 1), 10)):
            with self.subTest(raw=raw[:12], count=count), self.assertRaises(ValueError):
                parse_rows(raw, count)

    def test_hash_mismatch_is_fail_closed(self):
        raw = self.source()
        verify(raw, hashlib.sha256(raw).hexdigest(), 'synthetic')
        with self.assertRaises(ValueError):
            verify(raw + b'\n', hashlib.sha256(raw).hexdigest(), 'synthetic')

    def test_audit_reports_without_filtering(self):
        train = [{'x': [0] * 64, 'y': 0}] * 2
        evaluation = [{'x': [0] * 64, 'y': 1}]
        audit = audit_split(train, evaluation)
        self.assertEqual(audit['train']['duplicate_feature_rows'], 1)
        self.assertEqual(audit['cross_split_shared_feature_vectors'], 1)
        self.assertEqual(audit['cross_split_shared_labeled_rows'], 0)
        self.assertEqual((len(train), len(evaluation)), (2, 1))

    def test_redirect_cannot_leave_official_https_host(self):
        handler = OfficialRedirectsOnly()
        request = Request('https://archive.ics.uci.edu/example.zip')
        for destination in ('https://example.com/data.zip', 'http://archive.ics.uci.edu/data.zip',
                            'https://archive.ics.uci.edu.evil.example/data.zip'):
            with self.subTest(destination=destination), self.assertRaises(ValueError):
                handler.redirect_request(request, None, 302, '', {}, destination)


if __name__ == '__main__':
    unittest.main()
