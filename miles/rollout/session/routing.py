"""Process-stable session routing for the multi-process session server.

Stdlib only — imported by the headless worker / router without FastAPI.
"""

from __future__ import annotations

import hashlib
import uuid


def new_session_id() -> str:
    """Generate a fresh session_id (32-char lowercase hex)."""
    return uuid.uuid4().hex


def worker_index_for_session(session_id: str, n_worker: int) -> int:
    """Map *session_id* to a worker index in ``range(n_worker)``.

    Uses blake2b (not the builtin ``hash()``, which PYTHONHASHSEED salts per
    process) so the router and every worker derive the same owner.
    """
    if n_worker < 1:
        raise ValueError(f"n_worker must be >= 1, got {n_worker}")
    digest = hashlib.blake2b(session_id.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % n_worker
