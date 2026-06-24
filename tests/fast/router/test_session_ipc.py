"""Unit tests for the framed, multiplexed session IPC channel.

Covers the m3-design-contract §"IPC transport" requirements directly:
* framing round-trip over a real in-process socketpair;
* out-of-order multiplexing — a large chunked reply must not head-of-line
  block small replies;
* frame/body size caps fail deterministically;
* reader robustness — peer EOF fails all pending request futures.
"""

from __future__ import annotations

import asyncio
import socket

import pytest

from miles.rollout.session.session_ipc import DEFAULT_MAX_BODY_SIZE, IpcChannel, IpcChannelClosed, IpcError


async def _channel_pair(*, server_handler=None, client_max_body_size=DEFAULT_MAX_BODY_SIZE, **kwargs):
    """Build a (client, server) IpcChannel pair over a connected socketpair."""
    s1, s2 = socket.socketpair()
    r1, w1 = await asyncio.open_unix_connection(sock=s1)
    r2, w2 = await asyncio.open_unix_connection(sock=s2)
    server = IpcChannel(r2, w2, request_handler=server_handler, **kwargs)
    client = IpcChannel(r1, w1, max_body_size=client_max_body_size, **kwargs)
    return client, server


@pytest.mark.asyncio
async def test_framing_round_trip_small_and_multichunk():
    """A request/reply round-trips faithfully for both a tiny payload and a
    payload several chunks long (echo handler)."""

    async def echo(_request_id: int, payload: bytes) -> bytes:
        return payload

    client, server = await _channel_pair(server_handler=echo, max_chunk_size=64)
    try:
        assert await client.request(b"hi") == b"hi"
        assert await client.request(b"") == b""  # empty body terminates cleanly
        big = bytes(range(256)) * 50  # > many 64B chunks
        assert await client.request(big) == big
    finally:
        await client.close()
        await server.close()


@pytest.mark.asyncio
async def test_out_of_order_multiplexing_no_head_of_line_block():
    """A large multi-chunk reply must NOT block small replies.

    The server echoes after a per-request gate: the BIG request's handler waits
    on an event before returning its (heavily chunked) body, while SMALL
    requests return immediately. We fire the big request first, then many small
    ones; every small reply must resolve while the big one is still gated,
    proving replies are multiplexed by request_id and chunked bodies interleave.
    """

    release_big = asyncio.Event()

    async def handler(_request_id: int, payload: bytes) -> bytes:
        if payload == b"BIG":
            await release_big.wait()
            return b"X" * (4 << 20)  # 4 MiB, many chunks at 64 KiB
        return b"small-" + payload

    client, server = await _channel_pair(server_handler=handler, max_chunk_size=64 << 10)
    try:
        big_task = asyncio.create_task(client.request(b"BIG"))
        # Small requests fired while BIG is gated must all complete first.
        small = await asyncio.gather(*(client.request(str(i).encode()) for i in range(8)))
        assert small == [b"small-" + str(i).encode() for i in range(8)]
        assert not big_task.done(), "big reply resolved before being released — gating broken"

        release_big.set()
        big = await big_task
        assert big == b"X" * (4 << 20)
    finally:
        await client.close()
        await server.close()


@pytest.mark.asyncio
async def test_large_body_chunks_interleave_with_a_small_body():
    """Cooperative chunking interleaves: when a large body and a small body are
    sent concurrently on one channel, the small body's single frame lands in
    the writer queue BEFORE all of the large body's chunks — so the small
    reply is never head-of-line blocked behind the full large body.

    Inspects the framed queue directly (the writer is paused) so the property
    holds deterministically regardless of socket-buffer/drain timing.
    """

    s1, _s2 = socket.socketpair()
    r1, w1 = await asyncio.open_unix_connection(sock=s1)
    chan = IpcChannel(r1, w1, max_chunk_size=64)
    # Pause the writer so we can observe the queued frame order it would send.
    chan._writer_task.cancel()
    try:
        # Large body = many 64B chunks (request_id 100); small body = 1 frame
        # (request_id 200). Run them concurrently.
        big = bytes(64 * 20)  # 20 chunks
        await asyncio.gather(
            chan._send_body(100, 2, big),
            chan._send_body(200, 2, b"tiny"),
        )
        # Collect queued frames' request_ids in send order.
        ids = []
        while not chan._send_queue.empty():
            frame = chan._send_queue.get_nowait()
            if frame is None:
                continue
            # request_id is the u64 right after the u32 length prefix.
            ids.append(int.from_bytes(frame[4:12], "big"))
        assert 200 in ids, "small body never queued"
        # The small body's frame precedes the LAST big-body chunk: it interleaved.
        last_big = max(i for i, rid in enumerate(ids) if rid == 100)
        small_at = ids.index(200)
        assert small_at < last_big, f"small body queued behind the whole large body: {ids}"
    finally:
        chan._teardown(IpcChannelClosed("test done"))
        await chan.wait_closed()


@pytest.mark.asyncio
async def test_body_size_cap_fails_deterministically():
    """A reply body exceeding the client's max_body_size fails with IpcError
    (the reassembler rejects rather than buffering unbounded), and a late chunk
    after rejection does not crash the reader."""

    async def handler(_request_id: int, payload: bytes) -> bytes:
        return b"Y" * (2 << 20)  # 2 MiB reply

    s1, s2 = socket.socketpair()
    r1, w1 = await asyncio.open_unix_connection(sock=s1)
    r2, w2 = await asyncio.open_unix_connection(sock=s2)
    server = IpcChannel(r2, w2, request_handler=handler, max_chunk_size=64 << 10)
    # Client caps reassembled body well under the 2 MiB reply.
    client = IpcChannel(r1, w1, max_body_size=256 << 10)
    try:
        with pytest.raises((IpcError, IpcChannelClosed)):
            await client.request(b"go")
    finally:
        await client.close()
        await server.close()


@pytest.mark.asyncio
async def test_frame_size_cap_rejects_corrupt_length():
    """A length prefix beyond max_frame_size is rejected by the reader as a
    deterministic teardown rather than allocating a huge buffer."""

    import struct

    s1, s2 = socket.socketpair()
    r1, w1 = await asyncio.open_unix_connection(sock=s1)
    client = IpcChannel(r1, w1, max_frame_size=1024)
    try:
        fut = asyncio.create_task(client.request(b"x"))
        await asyncio.sleep(0)  # let the request enqueue
        # Peer writes a corrupt oversized length prefix straight onto the wire.
        s2.sendall(struct.pack(">I", 10_000_000))
        with pytest.raises(IpcChannelClosed):
            await fut
        assert client.closed
    finally:
        await client.close()
        s2.close()


@pytest.mark.asyncio
async def test_eof_fails_all_pending_futures_and_fires_on_close():
    """Peer death (EOF) must fail every pending request future deterministically
    and fire on_close exactly once (global fail-fast hook)."""

    closed_calls: list[BaseException | None] = []

    s1, s2 = socket.socketpair()
    r1, w1 = await asyncio.open_unix_connection(sock=s1)
    # Server never replies; it just dies.
    r2, w2 = await asyncio.open_unix_connection(sock=s2)
    server = IpcChannel(r2, w2, request_handler=lambda rid, p: asyncio.Future())
    client = IpcChannel(r1, w1, on_close=closed_calls.append)
    try:
        pendings = [asyncio.create_task(client.request(str(i).encode())) for i in range(5)]
        await asyncio.sleep(0.02)  # ensure all are in flight
        await server.close()  # peer EOF
        results = await asyncio.gather(*pendings, return_exceptions=True)
        assert all(isinstance(r, IpcChannelClosed) for r in results), results
        assert client.closed
        assert len(closed_calls) == 1, "on_close must fire exactly once"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_request_after_close_raises():
    """A request issued on an already-closed channel fails fast."""

    client, server = await _channel_pair(server_handler=lambda rid, p: asyncio.sleep(0, result=p))
    await server.close()
    await client.close()
    with pytest.raises(IpcChannelClosed):
        await client.request(b"x")


@pytest.mark.asyncio
async def test_handler_exception_returns_error_frame():
    """A handler raising returns a deterministic error frame to the caller as an
    IpcError, without tearing the channel down (later requests still work)."""

    calls = {"n": 0}

    async def handler(_request_id: int, payload: bytes) -> bytes:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom")
        return payload

    client, server = await _channel_pair(server_handler=handler)
    try:
        with pytest.raises(IpcError) as ei:
            await client.request(b"first")
        assert "boom" in str(ei.value)
        # Channel survived: a second request succeeds.
        assert await client.request(b"second") == b"second"
    finally:
        await client.close()
        await server.close()
