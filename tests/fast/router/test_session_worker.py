"""Worker-over-channel integration test for the multi-process data plane.

Drives a real :class:`SessionWorker` (own ``SessionCore`` + tokenizer) in the
same event loop over an in-process socketpair :class:`IpcChannel`, with a mock
backend ``do_proxy``. Exercises the router-facing op surface end to end:

* create -> chat -> GET -> DELETE round-trip;
* a same-session second chat returns 409 WITHOUT hitting the backend (gate);
* two distinct sessions' chats overlap across simulated upstream latency.
"""

from __future__ import annotations

import asyncio
import json
import socket
import time
from types import SimpleNamespace

import pytest

from miles.rollout.session.routing import new_session_id
from miles.rollout.session.session_core import ProxyRequest
from miles.rollout.session.session_ipc import IpcChannel
from miles.rollout.session.session_worker import (
    OP_CHAT,
    OP_CREATE_ID,
    OP_DELETE,
    OP_GET,
    SessionWorker,
    decode_core_response,
    encode_request,
)

HF_CHECKPOINT = "Qwen/Qwen3-0.6B"


class _MockBackend:
    """Records requests and returns a valid chat-completion response.

    The completion token ids are taken from the prompt's own ``input_ids`` tail
    so the response is a legal first-turn continuation regardless of tokenizer.
    """

    def __init__(self, latency: float = 0.0):
        self.latency = latency
        self.request_log: list[dict] = []
        self._concurrency = 0
        self.max_concurrent = 0

    async def do_proxy(self, request: ProxyRequest, path: str, body=None, headers=None) -> dict:
        self.request_log.append({"path": path, "body": body})
        self._concurrency += 1
        self.max_concurrent = max(self.max_concurrent, self._concurrency)
        try:
            if self.latency:
                await asyncio.sleep(self.latency)
        finally:
            self._concurrency -= 1
        # Emit two completion tokens with valid (logprob, token_id) pairs.
        token_ids = [101, 102]
        output_token_logprobs = [[-0.1, tid, "tok"] for tid in token_ids]
        response = {
            "id": "chatcmpl-mock",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                    "meta_info": {
                        "output_token_logprobs": output_token_logprobs,
                        "completion_tokens": len(output_token_logprobs),
                        "routed_experts": [[1, 2]] * len(token_ids),
                    },
                }
            ],
        }
        return {
            "request_body": body,
            "response_body": json.dumps(response).encode(),
            "status_code": 200,
            "headers": {"content-type": "application/json"},
        }

    async def aclose(self) -> None:  # parity with ProxyBackend
        pass


async def _worker_and_client(latency: float = 0.0):
    """Build a SessionWorker (with a mock backend) wired to a client channel."""
    args = SimpleNamespace(
        hf_checkpoint=HF_CHECKPOINT,
        chat_template_path=None,
        tito_model="default",
        tito_allowed_append_roles=["tool", "system"],
        miles_router_timeout=30,
        session_server_workers=1,
        session_server_instance_id="test-instance",
    )
    worker = SessionWorker(args, backend_url="http://unused", worker_index=0, n_worker=1)
    backend = _MockBackend(latency=latency)
    # Swap the real httpx-backed ProxyBackend for the recording mock.
    await worker.backend.aclose()
    worker.backend = backend
    worker.core.backend = backend

    s1, s2 = socket.socketpair()
    r1, w1 = await asyncio.open_unix_connection(sock=s1)
    r2, w2 = await asyncio.open_unix_connection(sock=s2)
    server = IpcChannel(r2, w2, request_handler=worker.handle)
    client = IpcChannel(r1, w1)
    return worker, backend, client, server


async def _request(client: IpcChannel, payload: bytes):
    return decode_core_response(await client.request(payload))


@pytest.mark.asyncio
async def test_create_with_id_invalid_and_duplicate_mapped():
    """A router-supplied id that is not 32-hex maps to 400; a second create on
    the same id maps to 409 — never an unmapped 500 from the raw ValueError."""

    worker, backend, client, server = await _worker_and_client()
    try:
        bad = await _request(client, encode_request(OP_CREATE_ID, session_id="not-hex"))
        assert bad.status_code == 400

        sid = new_session_id()
        assert (await _request(client, encode_request(OP_CREATE_ID, session_id=sid))).status_code == 200
        dup = await _request(client, encode_request(OP_CREATE_ID, session_id=sid))
        assert dup.status_code == 409
    finally:
        await client.close()
        await server.close()
        await backend.aclose()


@pytest.mark.asyncio
async def test_worker_create_chat_get_delete_round_trip():
    worker, backend, client, server = await _worker_and_client()
    try:
        sid = new_session_id()
        # create-with-id
        resp = await _request(client, encode_request(OP_CREATE_ID, session_id=sid))
        assert resp.status_code == 200
        assert json.loads(resp.body)["session_id"] == sid

        # chat (first turn)
        chat_body = json.dumps({"messages": [{"role": "user", "content": "hi"}]}).encode()
        resp = await _request(
            client, encode_request(OP_CHAT, session_id=sid, method="POST", body=chat_body, headers={})
        )
        assert resp.status_code == 200, resp.body
        client_meta = json.loads(resp.body)["choices"][0]["meta_info"]
        # Uniform R3 strip on the client body.
        assert "routed_experts" not in client_meta
        assert client_meta["completion_tokens"] == 2
        assert len(backend.request_log) == 1

        # GET records: R3 retained for reconstruction.
        resp = await _request(client, encode_request(OP_GET, session_id=sid))
        assert resp.status_code == 200
        records = json.loads(resp.body)["records"]
        assert len(records) == 1
        assert records[0]["response"]["choices"][0]["meta_info"]["routed_experts"] == [[1, 2], [1, 2]]

        # DELETE -> 204; subsequent GET -> 404.
        resp = await _request(client, encode_request(OP_DELETE, session_id=sid))
        assert resp.status_code == 204
        resp = await _request(client, encode_request(OP_GET, session_id=sid))
        assert resp.status_code == 404
    finally:
        await client.close()
        await server.close()
        await backend.aclose()


@pytest.mark.asyncio
async def test_same_session_second_chat_409_without_backend():
    """While one chat is parked in the (latency-gated) backend, a second
    same-session chat must 409 from the gate WITHOUT reaching the backend."""

    worker, backend, client, server = await _worker_and_client(latency=0.3)
    try:
        sid = new_session_id()
        assert (await _request(client, encode_request(OP_CREATE_ID, session_id=sid))).status_code == 200

        chat_body = json.dumps({"messages": [{"role": "user", "content": "park"}]}).encode()
        owner = asyncio.create_task(
            _request(client, encode_request(OP_CHAT, session_id=sid, method="POST", body=chat_body, headers={}))
        )
        # Wait until the owner is parked in the backend (slot claimed).
        deadline = time.time() + 5.0
        while len(backend.request_log) < 1 and time.time() < deadline:
            await asyncio.sleep(0.005)
        assert len(backend.request_log) == 1

        contender = await _request(
            client, encode_request(OP_CHAT, session_id=sid, method="POST", body=chat_body, headers={})
        )
        assert contender.status_code == 409
        assert json.loads(contender.body)["error"] == "session already has an in-flight chat completion"
        # The contender never reached the backend.
        assert len(backend.request_log) == 1

        owner_resp = await owner
        assert owner_resp.status_code == 200
    finally:
        await client.close()
        await server.close()
        await backend.aclose()


@pytest.mark.asyncio
async def test_distinct_sessions_chats_overlap_in_one_worker():
    """Two distinct sessions' chats run as separate tasks in one worker, so
    their upstream awaits overlap (max_concurrent >= 2) — not serialized."""

    worker, backend, client, server = await _worker_and_client(latency=0.3)
    try:
        sids = [new_session_id() for _ in range(2)]
        for sid in sids:
            assert (await _request(client, encode_request(OP_CREATE_ID, session_id=sid))).status_code == 200

        chat_body = json.dumps({"messages": [{"role": "user", "content": "go"}]}).encode()
        results = await asyncio.gather(
            *(
                _request(client, encode_request(OP_CHAT, session_id=sid, method="POST", body=chat_body, headers={}))
                for sid in sids
            )
        )
        assert all(r.status_code == 200 for r in results), [r.status_code for r in results]
        assert backend.max_concurrent >= 2, f"sessions did not overlap: max_concurrent={backend.max_concurrent}"
    finally:
        await client.close()
        await server.close()
        await backend.aclose()
