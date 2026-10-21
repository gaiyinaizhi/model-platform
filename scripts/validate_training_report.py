"""Offline report check. Run from the repository: python -m scripts.validate_training_report FILE"""
import argparse
import os
from pathlib import Path
import stat
import sys

# Resolve only the trusted source tree containing this script, never the report
# directory or caller's working directory. This also supports direct invocation
# from another directory with Python -S and no PYTHONPATH/site packages.
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model_platform.training_report_schema import MAX_REPORT_BYTES, parse_report


def read_report(path):
    flags = os.O_RDONLY | getattr(os, 'O_NONBLOCK', 0) | getattr(os, 'O_NOFOLLOW', 0)
    fd = os.open(path, flags)
    with os.fdopen(fd, 'rb') as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= MAX_REPORT_BYTES:
            raise ValueError('Invalid report file')
        raw = stream.read(MAX_REPORT_BYTES + 1)
        after = os.fstat(stream.fileno())
        if len(raw) != before.st_size or any(getattr(before, k) != getattr(after, k) for k in
                ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')):
            raise ValueError('Report file changed')
    return parse_report(raw)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Check internal consistency of an training evaluation report; not authenticity.')
    parser.add_argument('file', help='One local regular JSON report file (at most 512 KiB)')
    args = parser.parse_args(argv)
    try:
        read_report(args.file)
    except (OSError, ValueError):
        print('Invalid or unreadable training evaluation report', file=sys.stderr)
        return 1
    print('Valid internal consistency only. Authenticity is unverified; no signature or re-evaluation. '
          'Artifact, dataset and prediction bytes are absent; no source authenticity or business quality is established.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
