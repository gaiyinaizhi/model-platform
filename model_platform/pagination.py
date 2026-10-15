"""Bounded, authenticated cursors. The key signs cursors, never authenticates users."""
import base64
import hashlib
import hmac
import json
import math
import os
import re
import secrets
import stat
import tempfile

from .common import Problem, encode

MAX_CURSOR = 2048


def cursor_key(home):
    """Publish a complete 0600 key atomically, including concurrent first startup."""
    path = home / 'cursor-integrity.key'
    if not path.exists():
        fd, temporary = tempfile.mkstemp(prefix='.cursor-key-', dir=home)
        try:
            with os.fdopen(fd, 'wb') as stream:
                stream.write(secrets.token_bytes(32))
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError:
                pass
        finally:
            os.unlink(temporary)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600:
            raise RuntimeError('Invalid cursor integrity key permissions')
        key = stream.read(33)
    if len(key) != 32:
        raise RuntimeError('Invalid cursor integrity key')
    return key


def pack_cursor(key, payload):
    raw = encode(payload)
    tag = hmac.new(key, raw, hashlib.sha256).digest()
    return base64.urlsafe_b64encode(raw + tag).decode().rstrip('=')


def unpack_cursor(key, token, tenant, collection):
    try:
        if not isinstance(token, str) or not 1 <= len(token) <= MAX_CURSOR or not re.fullmatch(r'[A-Za-z0-9_-]+', token):
            raise ValueError()
        decoded = base64.urlsafe_b64decode(token + '=' * (-len(token) % 4))
        raw, tag = decoded[:-32], decoded[-32:]
        if not hmac.compare_digest(tag, hmac.new(key, raw, hashlib.sha256).digest()):
            raise ValueError()
        value = json.loads(raw)
        if (not isinstance(value, dict) or set(value) != {'v', 'tenant', 'collection', 'limit', 'high', 'last'}
                or type(value['v']) is not int or value['v'] != 1
                or value['tenant'] != hashlib.sha256(tenant.encode()).hexdigest() or value['collection'] != collection
                or type(value['limit']) is not int or not 1 <= value['limit'] <= 100
                or type(value['high']) is not int or value['high'] < 0
                or not isinstance(value['last'], list) or len(value['last']) != 2
                or type(value['last'][0]) not in (int, float) or not math.isfinite(value['last'][0])
                or not isinstance(value['last'][1], str) or len(value['last'][1]) > 160):
            raise ValueError()
        # Reject alternate encodings rather than accepting ambiguous token forms.
        if pack_cursor(key, value) != token:
            raise ValueError()
        return value
    except (ValueError, TypeError, KeyError, OverflowError):
        raise Problem(400, 'invalid_cursor', 'Invalid cursor for this collection and tenant') from None
