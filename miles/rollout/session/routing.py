"""Process-stable session routing for the multi-process session server.

A future router process generates a session_id with :func:`new_session_id`, maps
it to an owning worker with :func:`worker_index_for_session`, and dispatches over
IPC; each worker re-derives the same index to claim ownership. The mapping must
therefore be identical across processes and across runs, so it uses
``hashlib.blake2b`` rather than the builtin ``hash()`` (which is salted by
PYTHONHASHSEED and so differs per process).

Stdlib only — a headless worker / router can import this without FastAPI.
"""

from __future__ import annotations

import hashlib
import uuid


def new_session_id() -> str:
    """Generate a fresh session_id (32-char lowercase hex)."""
    return uuid.uuid4().hex


def worker_index_for_session(session_id: str, n_worker: int) -> int:
    """Map *session_id* to a worker index in ``range(n_worker)``.

    Deterministic and process-stable: the same (session_id, n_worker) always
    yields the same index regardless of PYTHONHASHSEED or process, so the router
    and every worker agree on ownership.
    """
    if n_worker < 1:
        raise ValueError(f"n_worker must be >= 1, got {n_worker}")
    digest = hashlib.blake2b(session_id.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % n_worker
