"""Stability regression: a client disconnect mid-chat must NOT destabilize the
multi-process session server.

This pins the invariant established while root-causing the benchmark's apparent
"worker death under load" (which turned out to be harness-induced, not a worker
defect): a client that drops its connection while its chat is parked in the
upstream MUST NOT

  (a) kill any worker process,
  (b) trip the supervisor's fail-fast, or
  (c) cause 503s / failures for OTHER concurrent sessions.

The router ``await``\\s the worker reply under :func:`asyncio.shield`, so a
client-disconnect cancel of the router handler does not cancel the worker's
in-flight task — the worker drains its reply and the channel stays open. This
test reuses the real spawned supervisor + mock-backend harness from
``test_session_multiprocess`` and complements ``test_worker_death_triggers_fail_fast``
(that one asserts a REAL worker death DOES fail-fast; this one asserts a mere
client disconnect does NOT).
"""

from __future__ import annotations

import socket
import threading
import time

import pytest
import requests

# Reuse the real multi-process harness (spawned router + N workers + mock backend).
from tests.fast.router.test_session_multiprocess import (
    HF_CHECKPOINT,
    N_WORKERS,
    _args,
    _chat,
    _create,
    _default_process_fn,
    _pid_alive,
    _start_supervisor,
)

from miles.rollout.session.routing import worker_index_for_session
from miles.utils.test_utils.mock_sglang_server import with_mock_server


def _disconnect_mid_chat(url: str, sid: str) -> None:
    """Open a chat request to ``sid`` then drop the connection before reading the
    reply — a real client-side disconnect while the worker is parked upstream.

    Speaks raw HTTP/1.1 over a socket and closes it right after sending the
    request body, so the router sees the peer go away mid-flight (rather than a
    clean response read). No reply is awaited.
    """
    host, port = url.removeprefix("http://").split(":")
    body = b'{"messages": [{"role": "user", "content": "drop me"}]}'
    req = (
        f"POST /sessions/{sid}/v1/chat/completions HTTP/1.1\r\n"
        f"Host: {host}:{port}\r\n"
        f"Content-Type: application/json\r\n"
        f"Content-Length: {len(body)}\r\n"
        f"Connection: close\r\n\r\n"
    ).encode() + body
    s = socket.create_connection((host, int(port)), timeout=5.0)
    try:
        s.sendall(req)
        # Give the router time to dispatch to the worker (which parks in the
        # latency-gated backend), then abruptly close WITHOUT reading the reply.
        time.sleep(0.2)
    finally:
        s.close()


def test_client_disconnect_midchat_does_not_destabilize_group():
    """A mid-chat client disconnect on one session leaves the whole group healthy
    and other sessions unaffected (no worker death, no fail-fast, no cross-session
    503s)."""
    # Latency-gated backend so the disconnected chat is provably in-flight upstream
    # at the moment the client drops.
    with with_mock_server(model_name=HF_CHECKPOINT, process_fn=_default_process_fn, latency=0.6) as backend:
        supervisor, url = _start_supervisor(_args(), backend.url)
        try:
            # Two sessions on DIFFERENT workers, so we exercise both the owning
            # worker of the dropped chat and an unrelated worker.
            sid_drop = _create(url)
            sid_other = None
            for _ in range(64):
                cand = _create(url)
                if worker_index_for_session(cand, N_WORKERS) != worker_index_for_session(sid_drop, N_WORKERS):
                    sid_other = cand
                    break
            assert sid_other is not None, "could not place two sessions on distinct workers"

            worker_pids = [w.pid for w in supervisor._workers]
            router_pid = supervisor._router.pid
            backend.reset_stats()

            # Fire-and-drop the chat for sid_drop; it parks in the latency-gated backend.
            t = threading.Thread(target=_disconnect_mid_chat, args=(url, sid_drop), daemon=True)
            t.start()
            # Wait until the dropped chat has actually reached the backend (in flight).
            deadline = time.time() + 10.0
            while len(backend.request_log) < 1 and time.time() < deadline:
                time.sleep(0.01)
            assert len(backend.request_log) >= 1, "dropped chat never reached the backend"
            t.join(timeout=10.0)

            # (b) supervisor did NOT fail-fast.
            assert not supervisor.failed, f"client disconnect tripped fail-fast: {supervisor._failure!r}"
            supervisor.check()  # raises if a failure was recorded

            # (a) every worker + the router are still alive.
            assert all(_pid_alive(p) for p in worker_pids), "a worker died after a client disconnect"
            assert _pid_alive(router_pid), "router died after a client disconnect"

            # (c) a concurrent OTHER session still succeeds (no cross-session 503)
            # and /health stays green — the disconnect is isolated to its session.
            assert requests.get(f"{url}/health", timeout=10.0).status_code == 200
            other = _chat(url, sid_other, content="still works", timeout=20.0)
            assert (
                other.status_code == 200
            ), f"other session failed after a disconnect: {other.status_code} {other.text}"

            # The disconnected session itself is intact and owned by a live worker:
            # under asyncio.shield the worker drained + committed its turn rather
            # than being abandoned mid-commit, so GET records is consistent (the
            # exact record count is timing-dependent — a clean disconnect before
            # the commit lock leaves 0, after leaves 1 — but the worker is alive
            # and the session reachable either way, which is the invariant here).
            got = requests.get(f"{url}/sessions/{sid_drop}", timeout=10.0)
            assert got.status_code == 200, f"disconnected session unreachable: {got.status_code} {got.text}"
            assert len(got.json()["records"]) <= 1
        finally:
            supervisor.shutdown()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
