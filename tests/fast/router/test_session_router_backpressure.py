"""Unit tests for SessionRouter backpressure + disconnect accounting.

Pins the invariant that a client disconnect (HTTP handler cancel) does NOT free
the per-worker in-flight slot early: the slot is released only when the
worker's IPC reply actually arrives (the router-owned task's done-callback), so
a disconnected-but-still-running request keeps counting against the per-worker
cap rather than leaking the count while it still consumes worker/httpx
resources. Uses a fake channel so no real worker/process is needed.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from miles.rollout.session.session_ipc import IpcChannelClosed, encode_envelope
from miles.rollout.session.session_router import SessionRouter


class _GatedChannel:
    """A fake IpcChannel whose request() resolves only when released."""

    def __init__(self):
        self._release = asyncio.Event()
        self._reply = b""

    def set_reply(self, reply: bytes) -> None:
        self._reply = reply

    def release(self) -> None:
        self._release.set()

    async def request(self, payload: bytes) -> bytes:
        await self._release.wait()
        return self._reply


def _router(channels, max_inflight=512):
    args = SimpleNamespace(session_server_instance_id="t", session_router_health_timeout=5.0)
    return SessionRouter(args, channels, max_inflight=max_inflight)


@pytest.mark.asyncio
async def test_disconnect_does_not_free_inflight_until_worker_reply():
    ch = _GatedChannel()
    ch.set_reply(encode_envelope({"status_code": 200, "headers": {}, "media_type": None}, b"{}"))
    router = _router([ch])

    # Dispatch as the HTTP handler would; cancelling it simulates a client
    # disconnect. The shielded IPC task keeps running.
    handler = asyncio.create_task(router._dispatch(0, b"payload"))
    await asyncio.sleep(0)  # let _dispatch start the IPC task + take the slot
    assert router._inflight[0] == 1

    handler.cancel()
    with pytest.raises(asyncio.CancelledError):
        await handler
    # The slot is STILL held: the worker request is in flight (disconnect must
    # not leak the count nor abandon the worker mid-send).
    assert router._inflight[0] == 1
    assert len(router._ipc_tasks) == 1

    # When the worker finally replies, the done-callback frees the slot.
    ch.release()
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert router._inflight[0] == 0
    assert len(router._ipc_tasks) == 0


@pytest.mark.asyncio
async def test_saturated_worker_returns_503_without_starting_a_task():
    ch = _GatedChannel()
    router = _router([ch], max_inflight=1)

    inflight = asyncio.create_task(router._dispatch(0, b"a"))
    await asyncio.sleep(0)
    assert router._inflight[0] == 1

    # Second dispatch on the saturated worker fast-fails 503 and starts no task.
    resp = await router._dispatch(0, b"b")
    assert resp.status_code == 503
    assert router._inflight[0] == 1
    assert len(router._ipc_tasks) == 1

    ch.set_reply(encode_envelope({"status_code": 200, "headers": {}, "media_type": None}, b"{}"))
    ch.release()
    done = await inflight
    assert done.status_code == 200
    await asyncio.sleep(0)
    assert router._inflight[0] == 0


@pytest.mark.asyncio
async def test_channel_closed_reply_frees_slot_and_returns_503():
    class _ClosingChannel:
        async def request(self, payload: bytes) -> bytes:
            raise IpcChannelClosed("worker died")

    router = _router([_ClosingChannel()])
    resp = await router._dispatch(0, b"x")
    assert resp.status_code == 503
    await asyncio.sleep(0)
    assert router._inflight[0] == 0
    assert len(router._ipc_tasks) == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
