"""Short-lived server-owned plans; client JSON is a handle, never authority."""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import time


def _identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def save_plan(root, envelope, *, resource_hashes, lease_identity, ttl_s=120., now=None):
    if not lease_identity or not resource_hashes:
        raise ValueError('Live lease identity and resource hashes required')
    if not math.isfinite(ttl_s) or not 0 < ttl_s <= 120:
        raise ValueError('Plan lifetime must be positive and at most120 seconds')
    root = Path(root)
    if root.is_symlink():
        raise ValueError('Plan store cannot be a symlink')
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    now = time.monotonic() if now is None else now
    record = {'envelope': envelope, 'resource_hashes': resource_hashes,
              'lease_fingerprint': _identity(lease_identity), 'created_mono': now,
              'expires_mono': now+ttl_s}
    encoded = json.dumps(record, sort_keys=True, allow_nan=False).encode()
    identifier = secrets.token_hex(16)
    path = root/(identifier+'.json')
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, 'wb') as out:
        out.write(encoded)
    return {'plan_id': identifier, 'record_sha256': hashlib.sha256(encoded).hexdigest(),
            'expires_in_s': ttl_s}


def consume_plan(root, handle, *, resource_hashes, lease_identity, now=None):
    identifier = handle.get('plan_id', '')
    if not isinstance(identifier, str) or not re.fullmatch('[0-9a-f]{32}', identifier):
        raise ValueError('Invalid plan identifier')
    path = Path(root)/(identifier+'.json')
    if path.is_symlink():
        raise ValueError('Plan record cannot be a symlink')
    encoded = path.read_bytes()
    if hashlib.sha256(encoded).hexdigest() != handle.get('record_sha256'):
        raise ValueError('Plan record identity changed')
    record = json.loads(encoded)
    now = time.monotonic() if now is None else now
    if not record['created_mono'] <= now < record['expires_mono']:
        raise ValueError('Plan expired or monotonic clock reset')
    if not lease_identity or record['lease_fingerprint'] != _identity(lease_identity):
        raise ValueError('Plan belongs to a different writer lease')
    if record['resource_hashes'] != resource_hashes:
        raise ValueError('Installed resources changed since planning')
    # Existing runtime motor lock serializes consumers. Unlink before execution:
    # failures require replanning from measured state, never replay stale motion.
    path.unlink()
    return record['envelope']
