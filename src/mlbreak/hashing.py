"""Canonical hashing helpers used for artifact, dataset and result fingerprints."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np


def sha256_arrays(*arrays: np.ndarray) -> str:
    """Hash arrays including dtype and shape, so reshapes/casts change the hash."""
    h = hashlib.sha256()
    for arr in arrays:
        a = np.ascontiguousarray(arr)
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def sha256_json(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode()).hexdigest()
