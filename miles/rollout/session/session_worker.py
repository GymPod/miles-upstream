"""Headless session-server worker process (multi-process data plane).

A worker owns one shard of sessions: it runs its own asyncio loop, a private
:class:`SessionRegistry` + tokenizer, and a lightweight :class:`ProxyBackend`
(just ``do_proxy`` + an httpx client). The router dispatches each op to the
owning worker over an :class:`~miles.rollout.session.session_ipc.IpcChannel`;
this module decodes the op envelope, drives the matching :class:`SessionCore`
method, and ships the :class:`CoreResponse` back.

Concurrency (m3-design-contract §"Concurrency model"):

* Each inbound IPC request is handled as its own asyncio task, so different
  sessions' upstream ``await``\\s overlap inside one worker.
* The per-session in-flight gate stays in ``SessionCore`` (claim under
  ``session.lock``, release in ``finally``); a same-session second chat 409s
  BEFORE any parse-gate wait.
* A process-local ``asyncio.Semaphore`` (``parse_gate``) bounds concurrent CPU
  parse/validate for memory; it is never awaited under ``session.lock`` and is
  entered only after the chat slot is claimed.
* Per-worker backpressure caps in-flight requests and queued bytes; on exceed
  the worker replies 503 without touching session state.

Channel endpoint (for the router builder, teammate B): the parent creates one
``socket.socketpair()`` per worker under a multiprocessing **spawn** context,
passes the worker-end socket to :func:`run_worker` and the router-end to the
router. ``run_worker`` is the ``multiprocessing.Process`` target; the parent
MUST close the router-end in the worker process and the worker-end in the
parent/router so EOF is observable on peer death.
"""

from __future__ import annotations

import asyncio
import ctypes
import json
import logging
import signal
import socket
import sys

import httpx

from miles.rollout.session.linear_trajectory import SessionRegistry
from miles.rollout.session.session_core import CoreResponse, ProxyRequest, SessionCore
from miles.rollout.session.session_ipc import IpcChannelClosed, decode_envelope, encode_envelope, open_unix_channel

logger = logging.getLogger(__name__)

# Per-worker httpx pool. N workers must not multiply the upstream connection
# cap to N*1024; each worker holds a modest slice.
_PER_WORKER_MAX_CONNECTIONS = 256

# Backpressure defaults (per worker): in-flight request count and queued bytes.
DEFAULT_MAX_INFLIGHT = 256
DEFAULT_MAX_QUEUED_BYTES = 256 << 20  # 256 MiB of request payloads in flight
# Parse/validate memory bound (size 1-2): see contract §8.
DEFAULT_PARSE_CONCURRENCY = 2


class ProxyBackend:
    """Minimal upstream proxy for a worker: only ``do_proxy`` + an httpx client.

    Mirrors ``SessionServer.do_proxy`` byte-for-byte (it is the same upstream
    call) but carries no FastAPI app — a worker needs nothing else.
    """

    def __init__(
        self, backend_url: str, *, timeout: float = 600.0, max_connections: int = _PER_WORKER_MAX_CONNECTIONS
    ):
        self.backend_url = backend_url
        self.client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=max_connections),
            timeout=httpx.Timeout(timeout),
        )

    async def do_proxy(
        self,
        request: ProxyRequest,
        path: str,
        body: bytes | None = None,
        headers: dict | None = None,
    ) -> dict:
        url = f"{self.backend_url}/{path}"
        if request.query:
            url = f"{url}?{request.query}"
        headers = {
            k: v
            for k, v in (headers or {}).items()
            if k.lower() not in ("content-length", "transfer-encoding", "host")
        }
        try:
            response = await self.client.request(request.method, url, content=body, headers=headers)
        except httpx.TransportError as exc:
            logger.warning("Proxy transport error for %s %s: %s", request.method, path, exc)
            error_body = json.dumps({"error": f"backend transport error: {type(exc).__name__}: {exc}"}).encode()
            return {
                "request_body": body,
                "response_body": error_body,
                "status_code": 502,
                "headers": {"content-type": "application/json"},
            }
        content = await response.aread()
        return {
            "request_body": body,
            "response_body": content,
            "status_code": response.status_code,
            "headers": dict(response.headers),
        }

    async def aclose(self) -> None:
        await self.client.aclose()


def _encode_core_response(resp: CoreResponse) -> bytes:
    """Pack a CoreResponse into an IPC reply payload.

    The (possibly 100+ MiB) body rides raw after a small JSON header so it is
    not base64-bloated; the router rebuilds the HTTP response from these parts
    without re-parsing the body.
    """
    meta = {"status_code": resp.status_code, "headers": resp.headers, "media_type": resp.media_type}
    return encode_envelope(meta, resp.body)


def decode_core_response(payload: bytes) -> CoreResponse:
    """Router-side inverse of :func:`_encode_core_response`."""
    meta, body = decode_envelope(payload)
    return CoreResponse(
        status_code=meta["status_code"],
        headers=meta.get("headers") or {},
        body=body,
        media_type=meta.get("media_type"),
    )


# ---- request envelope ops (router builds these; worker dispatches) --------
OP_HEALTH = "health"
OP_CREATE = "create"  # no-arg create (workers=1 adapter)
OP_CREATE_ID = "create_id"  # router-supplied session_id
OP_GET = "get"
OP_DELETE = "delete"
OP_CHAT = "chat"
OP_PROXY = "proxy"


def encode_request(
    op: str,
    *,
    session_id: str | None = None,
    method: str = "",
    query: str = "",
    headers: dict[str, str] | None = None,
    path: str = "",
    body: bytes = b"",
) -> bytes:
    """Router-side helper to build an op request payload."""
    meta = {
        "op": op,
        "session_id": session_id,
        "method": method,
        "query": query,
        "headers": headers or {},
        "path": path,
    }
    return encode_envelope(meta, body)


class SessionWorker:
    """Owns one worker's SessionCore and dispatches IPC ops to it."""

    def __init__(self, args, backend_url: str, worker_index: int, n_worker: int):
        self.args = args
        self.worker_index = worker_index
        self.n_worker = n_worker

        from miles.utils.chat_template_utils import get_tito_tokenizer
        from miles.utils.processing_utils import load_tokenizer

        tokenizer = load_tokenizer(
            args.hf_checkpoint,
            chat_template_path=getattr(args, "chat_template_path", None),
            trust_remote_code=True,
        )
        tito_tokenizer = get_tito_tokenizer(
            tokenizer,
            tokenizer_type=getattr(args, "tito_model", "default"),
            chat_template_kwargs=getattr(args, "apply_chat_template_kwargs", None),
            allowed_append_roles=getattr(args, "tito_allowed_append_roles", None),
        )
        registry = SessionRegistry(args, tokenizer, tito_tokenizer=tito_tokenizer)
        self.backend = ProxyBackend(
            backend_url,
            timeout=getattr(args, "miles_router_timeout", 600.0),
            max_connections=getattr(args, "session_worker_max_connections", _PER_WORKER_MAX_CONNECTIONS),
        )
        self.core = SessionCore(self.backend, registry, args, getattr(args, "session_server_instance_id", None))

        self._parse_sem = asyncio.Semaphore(
            getattr(args, "session_worker_parse_concurrency", DEFAULT_PARSE_CONCURRENCY)
        )
        self._max_inflight = getattr(args, "session_worker_max_inflight", DEFAULT_MAX_INFLIGHT)
        self._max_queued_bytes = getattr(args, "session_worker_max_queued_bytes", DEFAULT_MAX_QUEUED_BYTES)
        self._inflight = 0
        self._queued_bytes = 0

    def _parse_gate(self):
        return self._parse_sem

    async def handle(self, request_id: int, payload: bytes) -> bytes:
        """IPC request handler: decode the op, drive the core, encode the reply.

        Backpressure is admission-checked here (count + bytes) BEFORE any work
        so an overloaded worker rejects with 503 rather than growing memory
        across futures + frames + tasks + httpx queue.
        """
        size = len(payload)
        if self._inflight >= self._max_inflight or self._queued_bytes + size > self._max_queued_bytes:
            return _encode_core_response(
                _json_core_response(
                    503,
                    {"error": "session worker overloaded (in-flight/queued-bytes cap reached)"},
                )
            )
        self._inflight += 1
        self._queued_bytes += size
        try:
            meta, body = decode_envelope(payload)
            return _encode_core_response(await self._dispatch(meta, body))
        finally:
            self._inflight -= 1
            self._queued_bytes -= size

    async def _dispatch(self, meta: dict, body: bytes) -> CoreResponse:
        op = meta["op"]
        session_id = meta.get("session_id")
        if op == OP_HEALTH:
            return await self.core.health()
        if op == OP_CREATE:
            return await self.core.create_session()
        if op == OP_CREATE_ID:
            return await self.core.create_session_with_id(session_id)
        if op == OP_GET:
            return await self.core.get_session(session_id, parse_gate=self._parse_gate)
        if op == OP_DELETE:
            return await self.core.delete_session(session_id)
        if op == OP_CHAT:
            return await self.core.chat_completions(
                session_id,
                method=meta["method"],
                query=meta["query"],
                headers=meta.get("headers") or {},
                body=body,
                parse_gate=self._parse_gate,
            )
        if op == OP_PROXY:
            return await self.core.proxy(
                session_id,
                meta["path"],
                method=meta["method"],
                query=meta["query"],
                headers=meta.get("headers") or {},
                body=body,
            )
        return _json_core_response(400, {"error": f"unknown op: {op}"})

    async def aclose(self) -> None:
        await self.backend.aclose()


def _json_core_response(status_code: int, payload) -> CoreResponse:
    return CoreResponse(
        status_code=status_code,
        headers={"content-type": "application/json"},
        body=json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
        media_type="application/json",
    )


def _set_pdeathsig() -> None:
    """On Linux, ask the kernel to SIGKILL this process if the parent dies.

    Prevents orphaned workers if the router/parent crashes without a clean
    teardown. No-op on non-Linux.
    """
    if sys.platform != "linux":
        return
    try:
        PR_SET_PDEATHSIG = 1
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0)
    except Exception:
        logger.warning("prctl(PR_SET_PDEATHSIG) failed; worker will not auto-die on parent death", exc_info=True)


async def _serve(args, backend_url: str, sock: socket.socket, worker_index: int, n_worker: int) -> None:
    worker = SessionWorker(args, backend_url, worker_index, n_worker)
    closed = asyncio.get_event_loop().create_future()

    def _on_close(exc: BaseException | None) -> None:
        if not closed.done():
            closed.set_result(exc)

    channel = await open_unix_channel(sock, request_handler=worker.handle, on_close=_on_close)
    logger.info("[session-worker %d/%d] ready, proxying to %s", worker_index, n_worker, backend_url)
    try:
        await closed  # runs until the router closes / the channel tears down (EOF)
    finally:
        await channel.close()
        await worker.aclose()
    logger.info("[session-worker %d/%d] channel closed, exiting", worker_index, n_worker)


def run_worker(args, backend_url: str, sock: socket.socket, worker_index: int) -> None:
    """``multiprocessing.Process`` target for one session-server worker.

    Parameters
    ----------
    args : the Miles args namespace (carries hf_checkpoint, tito_*, timeout, …).
    backend_url : the inference router URL the worker proxies to.
    sock : the worker-end of the parent's per-worker ``socket.socketpair()``.
    worker_index : this worker's index in ``range(session_server_workers)``.

    The parent passes the worker-end socket here (and the router-end to the
    router) under a spawn context, closing all other ends so EOF is observable.
    """
    import setproctitle

    setproctitle.setproctitle(f"miles-session-worker-{worker_index}")
    _set_pdeathsig()

    n_worker = getattr(args, "session_server_workers", 1)
    assert 0 <= worker_index < n_worker, f"worker_index {worker_index} out of range for n_worker {n_worker}"

    try:
        asyncio.run(_serve(args, backend_url, sock, worker_index, n_worker))
    except (KeyboardInterrupt, IpcChannelClosed):
        pass
