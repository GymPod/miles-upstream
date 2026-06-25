"""Framed, multiplexed asyncio IPC for the multi-process session server.

One :class:`IpcChannel` wraps a single connected socket (the router holds one
end, a worker holds the other) and gives both sides a request/reply RPC over
opaque ``bytes`` payloads:

* The CLIENT side (router) calls :meth:`IpcChannel.request` -> awaits the reply
  bytes. Each call mints a ``request_id`` and registers a future.
* The SERVER side (worker) sets a ``request_handler`` coroutine; the channel
  feeds it ``(request_id, payload)`` and ships back whatever bytes it returns
  via :meth:`IpcChannel.reply`.

Wire protocol — every frame is::

    [length:u32 big-endian][request_id:u64][frame_type:u8][flags:u8][payload...]

``length`` counts everything after itself (request_id + type + flags +
payload), so the reader frames deterministically off one u32 prefix. A logical
message (a request or a reply payload) is split into one or more ``CHUNK``
frames tagged with the same ``request_id``; the final chunk sets
``FLAG_LAST``. Because the single per-socket writer interleaves chunk frames
from different request_ids, a 100+ MiB reply cannot monopolize the stream and
block small replies (no head-of-line blocking).

Design invariants (see m3-design-contract §"IPC transport"):

1. Single writer per socket: all sends drain through one writer task off an
   ``asyncio.Queue``; length-prefix + payload of concurrent messages never
   interleave.
2. No HOL blocking: large payloads are chunked at ``MAX_CHUNK_SIZE`` and the
   writer round-robins frames, so small replies are not stuck behind a big one.
3. Size caps: a frame larger than ``MAX_FRAME_SIZE`` (corrupt length) or a
   reassembled body larger than ``max_body_size`` fails deterministically — no
   unbounded buffering.
4. Reader robustness: EOF / partial frame / corrupt length tears the channel
   down once, failing every pending request future and firing ``on_close`` so
   the owner can fail-fast globally.
5. Late / abandoned replies: a reply (or error) for an unknown or already
   settled request_id is dropped cleanly — never an ``InvalidStateError`` that
   kills the reader.

Stdlib only; importable by a headless worker or router without FastAPI.
"""

from __future__ import annotations

import asyncio
import json
import logging
import struct
from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)

# Frame header: u32 length prefix, then u64 request_id + u8 type + u8 flags.
_LEN = struct.Struct(">I")
_HEADER = struct.Struct(">QBB")  # request_id, frame_type, flags  (after the length prefix)
_HEADER_LEN = _HEADER.size

# Frame types.
FRAME_REQUEST = 1  # a CHUNK belonging to a request payload (client -> server)
FRAME_REPLY = 2  # a CHUNK belonging to a reply payload (server -> client)
FRAME_ERROR = 3  # a deterministic error for a request_id (payload = utf-8 message)

# Flag bits.
FLAG_LAST = 0x01  # this is the final chunk of its (request_id, direction) body

# Default caps. A frame caps the on-wire length prefix (reject corrupt/huge
# lengths before allocating); a body caps the reassembled total per request_id.
MAX_CHUNK_SIZE = 1 << 20  # 1 MiB payload per frame
MAX_FRAME_SIZE = MAX_CHUNK_SIZE + _HEADER_LEN + 4096  # header + slack
DEFAULT_MAX_BODY_SIZE = 512 << 20  # 512 MiB reassembled body
# Cap on the not-yet-sent bytes of all outbound bodies registered with the
# writer at once. The writer round-robins one chunk per active body per round,
# so this bounds queued memory; a new body's registration backpressures (awaits)
# until the budget frees, instead of growing without bound when the peer stalls.
DEFAULT_MAX_SEND_BUFFER_BYTES = 256 << 20  # 256 MiB of in-flight send payloads


class IpcError(Exception):
    """Raised on an IPC-level failure (channel closed, size cap, remote error)."""


class IpcChannelClosed(IpcError):
    """The channel is closed (peer EOF, teardown, or corruption)."""


class _Reassembler:
    """Accumulates CHUNK frames for one (request_id, direction) until FLAG_LAST.

    Enforces ``max_body_size`` across the accumulated chunks so a runaway peer
    cannot grow memory without bound.
    """

    __slots__ = ("parts", "size", "max_body_size")

    def __init__(self, max_body_size: int):
        self.parts: list[bytes] = []
        self.size = 0
        self.max_body_size = max_body_size

    def add(self, chunk: bytes) -> None:
        self.size += len(chunk)
        if self.size > self.max_body_size:
            raise IpcError(f"reassembled body {self.size} exceeds max_body_size {self.max_body_size}")
        self.parts.append(chunk)

    def take(self) -> bytes:
        return b"".join(self.parts)


class _OutboundBody:
    """A registered outbound message the writer emits one chunk at a time.

    Holds a REFERENCE to the original payload + a cursor; the per-chunk frame is
    built only when the writer is about to send it, so the queue never holds a
    second copy of the whole body. ``remaining`` drives the send-buffer budget.
    """

    __slots__ = ("request_id", "frame_type", "payload", "offset", "max_chunk_size", "done")

    def __init__(self, request_id: int, frame_type: int, payload: bytes, max_chunk_size: int):
        self.request_id = request_id
        self.frame_type = frame_type
        self.payload = payload
        self.offset = 0
        self.max_chunk_size = max_chunk_size
        self.done = False

    @property
    def remaining(self) -> int:
        return len(self.payload) - self.offset

    def next_frame(self) -> bytes:
        """Build the next chunk frame, advance the cursor, mark ``done`` on the
        final (FLAG_LAST) chunk. An empty payload still yields one FLAG_LAST."""
        n = len(self.payload)
        end = min(self.offset + self.max_chunk_size, n)
        flags = FLAG_LAST if end >= n else 0
        chunk = self.payload[self.offset : end]
        self.offset = end
        if flags & FLAG_LAST:
            self.done = True
        header = _HEADER.pack(self.request_id, self.frame_type, flags) + chunk
        return _LEN.pack(len(header)) + header


class IpcChannel:
    """Framed, multiplexed RPC over one connected socket.

    Use :meth:`request` on the client (router) side and set ``request_handler``
    on the server (worker) side. ``on_close`` (if given) is invoked exactly once
    when the channel tears down, after all pending futures are failed — the
    owner uses it to trigger global fail-fast.
    """

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        *,
        request_handler: Callable[[int, bytes], Awaitable[bytes]] | None = None,
        on_close: Callable[[BaseException | None], None] | None = None,
        max_chunk_size: int = MAX_CHUNK_SIZE,
        max_frame_size: int = MAX_FRAME_SIZE,
        max_body_size: int = DEFAULT_MAX_BODY_SIZE,
        max_send_buffer_bytes: int = DEFAULT_MAX_SEND_BUFFER_BYTES,
    ):
        self._reader = reader
        self._writer = writer
        self._request_handler = request_handler
        self._on_close = on_close
        self._max_chunk_size = max_chunk_size
        self._max_frame_size = max_frame_size
        self._max_body_size = max_body_size
        self._max_send_buffer_bytes = max_send_buffer_bytes

        self._next_request_id = 1
        self._pending: dict[int, asyncio.Future[bytes]] = {}
        # Per (request_id) reassembly buffers, one map per inbound direction.
        self._inbound_requests: dict[int, _Reassembler] = {}
        self._inbound_replies: dict[int, _Reassembler] = {}
        self._handler_tasks: set[asyncio.Task] = set()

        # Send path: tiny control frames jump a FIFO lane; data bodies register
        # in an active list the writer round-robins (one chunk each per round) so
        # no large body monopolizes the stream. Registration backpressures on the
        # send-buffer budget. _send_event wakes the writer; _send_budget wakes
        # senders parked on backpressure.
        self._control_frames: list[bytes] = []
        self._active_bodies: list[_OutboundBody] = []
        self._registered_send_bytes = 0
        self._send_event = asyncio.Event()
        self._send_budget = asyncio.Event()
        self._send_budget.set()

        self._closed = False
        self._close_exc: BaseException | None = None

        self._writer_task = asyncio.create_task(self._writer_loop())
        self._reader_task = asyncio.create_task(self._reader_loop())

    # ---- public API -----------------------------------------------------

    async def request(self, payload: bytes) -> bytes:
        """Send *payload* as a request and await the reply bytes.

        Raises :class:`IpcChannelClosed` if the channel is or becomes closed,
        :class:`IpcError` if the server returned an error frame.
        """
        if self._closed:
            raise IpcChannelClosed("channel closed") from self._close_exc
        request_id = self._next_request_id
        self._next_request_id += 1
        fut: asyncio.Future[bytes] = asyncio.get_event_loop().create_future()
        self._pending[request_id] = fut
        try:
            await self._send_body(request_id, FRAME_REQUEST, payload)
            return await fut
        finally:
            # Drop the future so a late/abandoned reply for this id is ignored.
            self._pending.pop(request_id, None)

    async def reply(self, request_id: int, payload: bytes) -> None:
        """Send a successful reply payload for *request_id* (server side)."""
        await self._send_body(request_id, FRAME_REPLY, payload)

    async def reply_error(self, request_id: int, message: str) -> None:
        """Send a deterministic error reply for *request_id* (server side)."""
        if self._closed:
            raise IpcChannelClosed("channel closed") from self._close_exc
        self._enqueue_control(request_id, FRAME_ERROR, FLAG_LAST, message.encode("utf-8")[: self._max_chunk_size])

    async def close(self, exc: BaseException | None = None) -> None:
        """Tear the channel down once and fail all pending requests."""
        self._teardown(exc or IpcChannelClosed("channel closed locally"))
        await self.wait_closed()

    async def wait_closed(self) -> None:
        for task in (self._reader_task, self._writer_task):
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

    @property
    def closed(self) -> bool:
        return self._closed

    # ---- send path (single writer) --------------------------------------

    async def _send_body(self, request_id: int, frame_type: int, payload: bytes) -> None:
        """Register *payload* as an outbound body for the round-robin writer.

        The whole body is NOT pre-sliced into queued frame copies; only a
        reference + cursor is registered, and the writer emits one chunk per
        active body per round so a large body never blocks small replies (no HOL
        blocking) and queued memory is the bodies themselves, not a second copy.
        Registration backpressures on the send-buffer budget so total in-flight
        send bytes stay bounded even if the peer/writer stalls.
        """
        if self._closed:
            raise IpcChannelClosed("channel closed") from self._close_exc
        n = len(payload)
        # Backpressure: wait until this body fits the send buffer (a body larger
        # than the whole budget is allowed in alone so it can still make progress).
        while (
            not self._closed
            and self._registered_send_bytes > 0
            and self._registered_send_bytes + n > self._max_send_buffer_bytes
        ):
            self._send_budget.clear()
            await self._send_budget.wait()
        if self._closed:
            raise IpcChannelClosed("channel closed") from self._close_exc
        self._active_bodies.append(_OutboundBody(request_id, frame_type, payload, self._max_chunk_size))
        self._registered_send_bytes += n
        self._send_event.set()

    def _enqueue_control(self, request_id: int, frame_type: int, flags: int, chunk: bytes) -> None:
        """Queue a single tiny control frame (error / no-handler) on the priority
        FIFO lane; non-blocking, callable from the sync reader path."""
        header = _HEADER.pack(request_id, frame_type, flags) + chunk
        self._control_frames.append(_LEN.pack(len(header)) + header)
        self._send_event.set()

    async def _writer_loop(self) -> None:
        try:
            while True:
                if not self._control_frames and not self._active_bodies:
                    self._send_event.clear()
                    if not self._control_frames and not self._active_bodies:
                        await self._send_event.wait()
                    continue
                # Control frames first (FIFO), then one chunk per active body
                # (round-robin) so no large body monopolizes the stream.
                while self._control_frames:
                    self._writer.write(self._control_frames.pop(0))
                    await self._writer.drain()
                freed = 0
                for body in list(self._active_bodies):
                    before = body.remaining
                    self._writer.write(body.next_frame())
                    await self._writer.drain()
                    freed += before - body.remaining
                    if body.done:
                        self._active_bodies.remove(body)
                if freed and self._registered_send_bytes > 0:
                    self._registered_send_bytes -= freed
                    self._send_budget.set()  # wake senders parked on backpressure
        except (ConnectionError, OSError) as exc:
            self._teardown(IpcChannelClosed(f"writer failed: {exc!r}"))
        except asyncio.CancelledError:
            raise
        finally:
            try:
                self._writer.close()
            except Exception:
                pass

    # ---- receive path (single reader) -----------------------------------

    async def _reader_loop(self) -> None:
        try:
            while True:
                header = await self._reader.readexactly(_LEN.size)
                (length,) = _LEN.unpack(header)
                if length < _HEADER_LEN or length > self._max_frame_size:
                    raise IpcError(f"frame length {length} out of bounds (max {self._max_frame_size})")
                body = await self._reader.readexactly(length)
                request_id, frame_type, flags = _HEADER.unpack_from(body, 0)
                chunk = body[_HEADER_LEN:]
                self._dispatch(request_id, frame_type, flags, chunk)
        except asyncio.IncompleteReadError as exc:
            # EOF (peer death) or a truncated frame: deterministic teardown.
            self._teardown(IpcChannelClosed(f"peer closed / partial frame: {exc!r}"))
        except (IpcError, struct.error, ConnectionError, OSError) as exc:
            self._teardown(IpcChannelClosed(f"reader failed: {exc!r}"))
        except asyncio.CancelledError:
            raise

    def _dispatch(self, request_id: int, frame_type: int, flags: int, chunk: bytes) -> None:
        if frame_type == FRAME_REQUEST:
            self._on_request_chunk(request_id, flags, chunk)
        elif frame_type == FRAME_REPLY:
            self._on_reply_chunk(request_id, flags, chunk)
        elif frame_type == FRAME_ERROR:
            self._settle(request_id, IpcError(chunk.decode("utf-8", "replace")))
        else:
            raise IpcError(f"unknown frame_type {frame_type}")

    def _on_request_chunk(self, request_id: int, flags: int, chunk: bytes) -> None:
        buf = self._inbound_requests.get(request_id)
        if buf is None:
            buf = self._inbound_requests[request_id] = _Reassembler(self._max_body_size)
        buf.add(chunk)
        if flags & FLAG_LAST:
            self._inbound_requests.pop(request_id, None)
            self._spawn_handler(request_id, buf.take())

    def _on_reply_chunk(self, request_id: int, flags: int, chunk: bytes) -> None:
        # An abandoned/unknown request_id: drop chunks cleanly.
        if request_id not in self._pending and request_id not in self._inbound_replies:
            return
        buf = self._inbound_replies.get(request_id)
        if buf is None:
            buf = self._inbound_replies[request_id] = _Reassembler(self._max_body_size)
        buf.add(chunk)
        if flags & FLAG_LAST:
            self._inbound_replies.pop(request_id, None)
            self._settle(request_id, buf.take())

    def _settle(self, request_id: int, result: bytes | BaseException) -> None:
        """Resolve a pending request future; a late/double settle is a no-op."""
        fut = self._pending.get(request_id)
        if fut is None or fut.done():
            return
        if isinstance(result, BaseException):
            fut.set_exception(result)
        else:
            fut.set_result(result)

    # ---- inbound request handling (server side) -------------------------

    def _spawn_handler(self, request_id: int, payload: bytes) -> None:
        if self._request_handler is None:
            # No server role on this side: tell the peer deterministically.
            self._enqueue_control(request_id, FRAME_ERROR, FLAG_LAST, b"")
            return
        task = asyncio.create_task(self._run_handler(request_id, payload))
        self._handler_tasks.add(task)
        task.add_done_callback(self._handler_tasks.discard)

    async def _run_handler(self, request_id: int, payload: bytes) -> None:
        try:
            result = await self._request_handler(request_id, payload)
            await self.reply(request_id, result)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("IPC request handler failed for request_id=%s", request_id)
            if not self._closed:
                try:
                    await self.reply_error(request_id, f"{type(exc).__name__}: {exc}")
                except IpcChannelClosed:
                    pass

    # ---- teardown -------------------------------------------------------

    def _teardown(self, exc: BaseException) -> None:
        if self._closed:
            return
        self._closed = True
        self._close_exc = exc
        # Fail every pending request future deterministically.
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(exc)
        self._pending.clear()
        self._inbound_requests.clear()
        self._inbound_replies.clear()
        # Drop queued sends and wake any sender parked on the send-buffer budget
        # so it observes the closed channel instead of hanging.
        self._active_bodies.clear()
        self._control_frames.clear()
        self._registered_send_bytes = 0
        self._send_event.set()
        self._send_budget.set()
        for task in (self._reader_task, self._writer_task):
            task.cancel()
        for task in list(self._handler_tasks):
            task.cancel()
        if self._on_close is not None:
            cb, self._on_close = self._on_close, None
            try:
                cb(exc)
            except Exception:
                logger.exception("IPC on_close callback raised")


async def open_unix_channel(sock, **kwargs) -> IpcChannel:
    """Build an :class:`IpcChannel` over an already-connected ``socket.socket``.

    The parent creates the pair with ``socket.socketpair()`` and hands one end
    to the router and the matching end to the worker; each side passes its end
    here. ``kwargs`` are forwarded to :class:`IpcChannel`.
    """
    reader, writer = await asyncio.open_unix_connection(sock=sock)
    return IpcChannel(reader, writer, **kwargs)


# ---------------------------------------------------------------------------
# Request / reply envelopes
#
# An op's metadata is small JSON; the request body and the reply body are raw
# bytes that may be large (a 100+ MiB GET-records reply). To avoid base64
# bloat, both envelopes are framed as ``[meta_len:u32][meta_json][raw_body]`` —
# the JSON header rides in front of the unaltered body bytes.
# ---------------------------------------------------------------------------


def encode_envelope(meta: dict, body: bytes) -> bytes:
    meta_bytes = json.dumps(meta, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return _LEN.pack(len(meta_bytes)) + meta_bytes + body


def decode_envelope(payload: bytes) -> tuple[dict, bytes]:
    (meta_len,) = _LEN.unpack_from(payload, 0)
    start = _LEN.size
    meta = json.loads(payload[start : start + meta_len])
    return meta, payload[start + meta_len :]
