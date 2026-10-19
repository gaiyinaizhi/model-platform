"""Bounded read-only filesystem access without following any symlink."""
import contextlib
import os
from pathlib import Path
import stat


@contextlib.contextmanager
def directory(path):
    parts = Path(os.path.abspath(path)).parts
    fd = os.open(parts[0], os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in parts[1:]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = child
        yield fd
    finally:
        os.close(fd)


def read_at(parent, name, maximum):
    if not name or name in ('.', '..') or '/' in name or '\\' in name:
        raise ValueError('Unsafe file name')
    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
    with os.fdopen(fd, 'rb') as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or not 0 <= before.st_size <= maximum:
            raise ValueError('Unsafe or oversized file')
        raw = stream.read(maximum + 1)
        after = os.fstat(stream.fileno())
        if len(raw) != before.st_size or any(getattr(before, key) != getattr(after, key)
                for key in ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')):
            raise ValueError('File changed during read')
        return raw


def read_file(path, maximum):
    path = Path(path)
    with directory(path.parent) as parent:
        return read_at(parent, path.name, maximum)
