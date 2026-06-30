"""Thin client-facing router for the multi-process session server.

The router is the sole HTTP listener. It routes each request to the owning
worker by ``blake2b(session_id) % N`` over an IPC channel and relays the worker's
response bytes back verbatim — it never parses the body. ``POST /sessions`` mints
the id (so it can route by it) and dispatches a create to the owning worker.

Imports neither ``session_core`` nor ``session_worker``, so the router process
does not load the tokenizer/transformers stack — it only moves bytes.
"""

from __future__ import annotations

import asyncio
import json

import setproctitle
import uvicorn
from fastapi import FastAPI, Request
from starlette.responses import Response

from miles.rollout.session.routing import new_session_id, worker_index_for_session
from miles.rollout.session.session_ipc import (
    OP_CHAT,
    OP_CREATE,
    OP_DELETE,
    OP_GET,
    OP_HEALTH,
    OP_PROXY,
    IpcChannelClosed,
    IpcError,
    decode_envelope,
    encode_request,
    open_unix_channel,
    set_pdeathsig,
)

_JSON = "application/json"
_HEALTH_TIMEOUT = 5.0


def _err(status_code: int, message: str) -> Response:
    return Response(content=json.dumps({"error": message}).encode(), status_code=status_code, media_type=_JSON)


def _reply_to_response(reply: bytes) -> Response:
    meta, body = decode_envelope(reply)
    return Response(content=body, status_code=meta["status"], headers=meta["headers"])


class SessionRouter:
    """Routes requests to per-worker IPC channels by stable hash of session_id."""

    def __init__(self, channels: list, session_server_instance_id=None):
        if not channels:
            raise ValueError("SessionRouter requires at least one worker channel")
        self.channels = channels
        self.n_worker = len(channels)
        self.instance_id = session_server_instance_id

    def channel_for(self, session_id: str):
        return self.channels[worker_index_for_session(session_id, self.n_worker)]

    async def dispatch(self, channel, payload: bytes) -> Response:
        try:
            reply = await channel.request(payload)
        except IpcChannelClosed:
            return _err(503, "session worker unavailable")
        except IpcError as exc:
            return _err(502, f"session worker error: {exc}")
        return _reply_to_response(reply)

    async def healthy(self) -> bool:
        async def ping(channel) -> bool:
            meta, _ = decode_envelope(await channel.request(encode_request(OP_HEALTH)))
            return meta["status"] == 200

        try:
            results = await asyncio.wait_for(
                asyncio.gather(*(ping(ch) for ch in self.channels)), timeout=_HEALTH_TIMEOUT
            )
        except (asyncio.TimeoutError, IpcChannelClosed, IpcError):
            return False
        return all(results)


def build_router_app(channels: list, session_server_instance_id=None) -> FastAPI:
    router = SessionRouter(channels, session_server_instance_id)
    app = FastAPI()

    @app.get("/health")
    async def health():
        if not await router.healthy():
            return _err(503, "one or more session workers unhealthy")
        body = {"status": "ok"}
        if router.instance_id is not None:
            body["session_server_instance_id"] = router.instance_id
        return Response(content=json.dumps(body).encode(), status_code=200, media_type=_JSON)

    @app.post("/sessions")
    async def create_session():
        session_id = new_session_id()
        return await router.dispatch(router.channel_for(session_id), encode_request(OP_CREATE, session_id=session_id))

    @app.get("/sessions/{session_id}")
    async def get_session(session_id: str):
        return await router.dispatch(router.channel_for(session_id), encode_request(OP_GET, session_id=session_id))

    @app.delete("/sessions/{session_id}")
    async def delete_session(session_id: str):
        return await router.dispatch(router.channel_for(session_id), encode_request(OP_DELETE, session_id=session_id))

    @app.post("/sessions/{session_id}/v1/chat/completions")
    async def chat_completions(request: Request, session_id: str):
        body = await request.body()
        payload = encode_request(
            OP_CHAT,
            session_id=session_id,
            method=request.method,
            query=request.url.query,
            headers=dict(request.headers),
            body=body,
        )
        return await router.dispatch(router.channel_for(session_id), payload)

    @app.api_route("/sessions/{session_id}/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def session_proxy(request: Request, session_id: str, path: str):
        body = await request.body()
        payload = encode_request(
            OP_PROXY,
            session_id=session_id,
            path=path,
            method=request.method,
            query=request.url.query,
            headers=dict(request.headers),
            body=body,
        )
        return await router.dispatch(router.channel_for(session_id), payload)

    return app


async def _serve_router(args, router_ends: list, ip: str, port: int) -> None:
    channels = [await open_unix_channel(sock) for sock in router_ends]
    app = build_router_app(channels, getattr(args, "session_server_instance_id", None))
    await uvicorn.Server(uvicorn.Config(app, host=ip, port=port, log_level="info")).serve()


def run_router(args, router_ends: list, ip: str, port: int) -> None:
    """``multiprocessing.Process`` target: serve the router app over the worker sockets.

    Imports nothing from session_core/session_worker, so the router process never loads
    the tokenizer/transformers stack — it only moves bytes between clients and workers.
    """
    setproctitle.setproctitle("miles-session-router")
    set_pdeathsig()
    asyncio.run(_serve_router(args, router_ends, ip, port))
