"""AC-4 behavioral equivalence: the multi-process (router + IPC + N workers)
path must reproduce the SAME error / gate / passthrough / validation semantics
that ``test_session_race_conditions.py`` pins for the single-process path, plus
a client-disconnect stability regression.

Why a focused equivalence suite instead of fully parametrizing
``test_session_race_conditions.py`` over {workers=1, workers=N}:

* Most race-suite cases inject a failure with ``unittest.mock.patch.object`` on
  ``SessionServer`` / ``LinearTrajectory`` / ``SessionCore`` IN THE TEST PROCESS.
  The single-process server runs in a uvicorn THREAD in that same process, so
  the patch takes effect. The multi-process workers are SEPARATE spawned
  processes that re-import the modules fresh, so an in-process patch is invisible
  to them. Reproducing those injected-failure paths through real workers would
  require modifying session source (an injection hook) — out of scope here
  (test-only). So the patch-injected cases (prepare error, transport-502,
  NaN-via-do_proxy, invariant-mismatch, asyncio.CancelledError) stay
  single-process-only.
* The cases that ARE drivable end to end through real workers are exactly the
  ones driven from the CLIENT (malformed request JSON) or from the MOCK BACKEND
  (upstream non-200, invalid-200 body): the mock backend runs in the test
  process and the workers reach it over real HTTP, so a backend-side patch DOES
  cross the process boundary. Those are the invariants reproduced below at
  workers=N, end to end through the real router + workers + IPC:
    - closing (404) beats busy (409);
    - the in-flight slot is released on every error path so the session is
      reusable afterward (malformed request JSON -> 4xx/5xx then a normal chat
      200s; an upstream non-200 passes through UNRECORDED then next chat 200s;
      an invalid-200 body -> 502 with NOTHING committed to the session record);
    - response/header passthrough fidelity through IPC (status, content-type,
      stale-framing-header stripping) and the uniform R3 strip (client omits
      routed_experts/indexer_topk; GET-records retain them).

The client-disconnect stability invariant is pinned separately by the standalone
``test_session_multiprocess_disconnect`` in this directory; this file is the
error/gate/passthrough/validation equivalence suite only.

All tests reuse ``_start_supervisor`` + the thread-based mock SGLang backend
from ``test_session_multiprocess`` and speak plain HTTP to the spawned router.
"""

from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import requests
from fastapi.responses import JSONResponse

from miles.utils.test_utils.mock_sglang_server import MockSGLangServer, ProcessResult, with_mock_server

from .test_session_multiprocess import HF_CHECKPOINT, _args, _chat, _create, _default_process_fn, _start_supervisor

# ---- backend-side response patches (cross the spawn boundary over HTTP) -----


def _patch_mock_chat_missing_meta_info_first():
    """Mock backend strips meta_info from the FIRST chat response only.

    A 200 body with no meta_info makes the worker's response validation raise
    UpstreamResponseError -> 502, committing nothing. Later responses are valid.
    Patches the backend (in the test process), so the spawned worker sees it
    over real HTTP — unlike an in-process ``patch.object(SessionServer, ...)``.
    """
    original = MockSGLangServer._compute_chat_completions_response
    state = {"calls": 0}

    def patched(self, payload: dict) -> dict:
        response = original(self, payload)
        state["calls"] += 1
        if state["calls"] == 1:
            response["choices"][0].pop("meta_info", None)
        return response

    return patch.object(MockSGLangServer, "_compute_chat_completions_response", new=patched)


def _patch_mock_chat_bad_logprob_first():
    """Mock backend emits a non-numeric leading logprob on the FIRST chat only.

    A successful (200) body carrying a string logprob must be rejected (502)
    before any token id / logprob is committed to the trajectory; later
    responses are valid.
    """
    original = MockSGLangServer._compute_chat_completions_response
    state = {"calls": 0}

    def patched(self, payload: dict) -> dict:
        response = original(self, payload)
        state["calls"] += 1
        if state["calls"] == 1:
            otl = response["choices"][0]["meta_info"]["output_token_logprobs"]
            if otl:
                otl[0] = ("bad-logprob", otl[0][1])
        return response

    return patch.object(MockSGLangServer, "_compute_chat_completions_response", new=patched)


def _patch_mock_chat_with_r3():
    """Mock backend adds per-turn R3 replay blobs to every chat response.

    Mirrors the workers=1 R3 patch shape so the uniform-strip equivalence can be
    asserted end to end: the client chat body omits routed_experts/indexer_topk
    while the stored GET-records keep them.
    """
    original = MockSGLangServer._compute_chat_completions_response

    def patched(self, payload: dict) -> dict:
        response = original(self, payload)
        choice = response["choices"][0]
        otl = choice["meta_info"]["output_token_logprobs"]
        choice["meta_info"]["routed_experts"] = [[1, 2, 3]] * len(otl)
        choice["meta_info"]["indexer_topk"] = [[4, 5]] * len(otl)
        choice["meta_info"]["indexer_topk_num_layers"] = 1
        return response

    return patch.object(MockSGLangServer, "_compute_chat_completions_response", new=patched)


def _patch_mock_non_200_first():
    """Mock backend returns a real non-200 HTTP status on the FIRST chat only.

    The worker's ``do_proxy`` passes a non-200 straight through unrecorded, so
    the client sees that exact status/body and nothing is committed; later chats
    are normal 200s. Patches ``_handle_generate_like_request`` (not the compute
    fn) so the actual HTTP status code — not just the body — is non-200, which
    is what the "upstream non-200 passes through" contract requires.

    Reimplements the handler (rather than delegating to the original) because
    the first-call short-circuit must precede the latency/concurrency tracking.
    """
    state = {"calls": 0}

    async def patched(self, request, compute_fn):
        # Log the arrival exactly like the real path so request_log counts match.
        payload = await request.json()
        self.request_log.append(payload)
        state["calls"] += 1
        if state["calls"] == 1:
            return JSONResponse(content={"error": "context too long"}, status_code=400)
        with self._concurrency.track():
            if self.latency > 0:
                await asyncio.sleep(self.latency)
            response = compute_fn(payload)
        return JSONResponse(content=response)

    return patch.object(MockSGLangServer, "_handle_generate_like_request", new=patched)


# ---- equivalence fixtures (workers=N) ---------------------------------------


def _run_supervisor(process_fn, *, latency=0.0, response_patch=None, instance_id="mp-equiv"):
    """Context-style helper: spawn a workers=N supervisor against a mock backend.

    Mirrors ``test_session_race_conditions._router_env`` but over the real
    multi-process deployment. Returns a context manager yielding an env with
    ``url`` and ``backend``.
    """
    from contextlib import contextmanager

    @contextmanager
    def _cm():
        with with_mock_server(model_name=HF_CHECKPOINT, process_fn=process_fn, latency=latency) as backend:
            patch_cm = response_patch() if response_patch is not None else None
            if patch_cm is not None:
                patch_cm.__enter__()
            supervisor, url = _start_supervisor(_args(instance_id=instance_id), backend.url)
            try:
                yield SimpleNamespace(url=url, backend=backend, supervisor=supervisor)
            finally:
                supervisor.shutdown()
                if patch_cm is not None:
                    patch_cm.__exit__(None, None, None)

    return _cm()


class TestSlotReleaseEquivalenceWorkersN:
    """Slot-release equivalence at workers=N for the error paths reachable end
    to end through real workers (driven by the client or the mock backend).

    Each: inject a failure on the FIRST same-session chat, prove the slot is
    released (NEXT same-session chat 200s, not a stuck 409) and that the failing
    turn committed nothing.
    """

    def test_slot_released_after_malformed_request_json(self):
        """Malformed client JSON -> 5xx before the backend; slot released so the
        next normal chat on the SAME session 200s. The malformed body never
        reaches the backend (request_log stays 0)."""
        with _run_supervisor(_default_process_fn) as env:
            sid = _create(env.url)
            env.backend.reset_stats()

            bad = requests.post(
                f"{env.url}/sessions/{sid}/v1/chat/completions",
                data="{not json",
                headers={"content-type": "application/json"},
                timeout=20.0,
            )
            assert bad.status_code >= 500, bad.text
            assert len(env.backend.request_log) == 0

            good = _chat(env.url, sid, content="hi")
            assert good.status_code == 200, good.text
            assert len(env.backend.request_log) == 1

    def test_upstream_non_200_passes_through_unrecorded(self):
        """An upstream non-200 (real HTTP 400) passes through to the client
        unrecorded; the slot is released so the next normal chat 200s and the
        session committed exactly the one good turn."""
        with _run_supervisor(_default_process_fn, response_patch=_patch_mock_non_200_first) as env:
            sid = _create(env.url)
            env.backend.reset_stats()

            bad = _chat(env.url, sid, content="hi")
            assert bad.status_code == 400, bad.text
            assert bad.json()["error"] == "context too long"
            # Reached the backend (it produced the non-200), but nothing committed.
            assert len(env.backend.request_log) == 1
            state = requests.get(f"{env.url}/sessions/{sid}", timeout=10.0).json()
            assert state["records"] == []
            assert state["metadata"]["accumulated_token_ids"] == []

            good = _chat(env.url, sid, content="hi again")
            assert good.status_code == 200, good.text
            assert len(env.backend.request_log) == 2
            after = requests.get(f"{env.url}/sessions/{sid}", timeout=10.0).json()
            assert len(after["records"]) == 1

    def test_invalid_200_missing_meta_info_502_nothing_committed(self):
        """A 200 upstream body missing meta_info -> 502 with NOTHING committed;
        the slot is released so the next normal chat 200s."""
        with _run_supervisor(_default_process_fn, response_patch=_patch_mock_chat_missing_meta_info_first) as env:
            sid = _create(env.url)
            env.backend.reset_stats()

            bad = _chat(env.url, sid, content="hi")
            assert bad.status_code == 502, bad.text
            assert len(env.backend.request_log) == 1
            state = requests.get(f"{env.url}/sessions/{sid}", timeout=10.0).json()
            assert state["records"] == []
            assert state["metadata"]["accumulated_token_ids"] == []

            good = _chat(env.url, sid, content="hi again")
            assert good.status_code == 200, good.text
            assert len(env.backend.request_log) == 2

    def test_invalid_200_bad_logprob_502_nothing_committed(self):
        """A 200 upstream body with a non-numeric logprob value -> 502 with
        NOTHING committed (guards the bad logprob from flowing into
        rollout_log_probs downstream); slot released so the next chat 200s."""
        with _run_supervisor(_default_process_fn, response_patch=_patch_mock_chat_bad_logprob_first) as env:
            sid = _create(env.url)
            env.backend.reset_stats()

            bad = _chat(env.url, sid, content="hi")
            assert bad.status_code == 502, bad.text
            assert len(env.backend.request_log) == 1
            state = requests.get(f"{env.url}/sessions/{sid}", timeout=10.0).json()
            assert state["records"] == []
            assert state["metadata"]["accumulated_token_ids"] == []

            good = _chat(env.url, sid, content="hi again")
            assert good.status_code == 200, good.text
            assert len(after := requests.get(f"{env.url}/sessions/{sid}", timeout=10.0).json()["records"]) == 1
            assert after  # one committed turn


class TestClosingBeatsBusyWorkersN:
    """closing (404) beats busy (409) end to end through real workers.

    Park an owner chat in a latency-gated backend (holds the in-flight slot),
    fire a DELETE (sets closing=True), then a same-session chat: it must see
    closing FIRST and get 404, never the busy 409.
    """

    def test_closing_404_beats_busy_409(self):
        with _run_supervisor(_default_process_fn, latency=0.6) as env:
            sid = _create(env.url)
            env.backend.reset_stats()

            owner = {}

            def _owner():
                owner["resp"] = _chat(env.url, sid, content="park", timeout=30.0)

            t = threading.Thread(target=_owner)
            t.start()
            # Wait until the owner is parked in the backend holding the slot.
            deadline = time.time() + 10.0
            while len(env.backend.request_log) < 1 and time.time() < deadline:
                time.sleep(0.01)
            assert len(env.backend.request_log) == 1

            # DELETE sets closing=True (it blocks on the lock the owner's commit
            # will take, but closing is set synchronously before that wait).
            delete_t = {}

            def _delete():
                delete_t["resp"] = requests.delete(f"{env.url}/sessions/{sid}", timeout=30.0)

            dt = threading.Thread(target=_delete)
            dt.start()
            time.sleep(0.1)  # let DELETE set closing=True

            # Same-session chat now: closing must win -> 404 (not 409).
            contender = _chat(env.url, sid, content="contend", timeout=10.0)
            assert (
                contender.status_code == 404
            ), f"closing must beat busy: expected 404, got {contender.status_code}: {contender.text}"

            t.join(timeout=30.0)
            dt.join(timeout=30.0)
            assert owner["resp"].status_code == 200
            assert delete_t["resp"].status_code == 204
            # The contender never reached the backend (gate/closing rejected it).
            assert len(env.backend.request_log) == 1


class TestPassthroughFidelityWorkersN:
    """Response/header passthrough fidelity + uniform R3 strip through IPC."""

    def test_successful_response_passthrough_strips_stale_framing(self):
        def process_fn(prompt: str) -> ProcessResult:
            return ProcessResult(text="passthrough-ok", finish_reason="stop")

        with _run_supervisor(process_fn) as env:
            sid = _create(env.url)
            resp = _chat(env.url, sid, content="hi")
            assert resp.status_code == 200, resp.text

            body = resp.json()
            assert body["object"] == "chat.completion"
            assert body["id"].startswith("chatcmpl-")
            assert body["choices"][0]["message"]["content"] == "passthrough-ok"
            assert "meta_info" in body["choices"][0]

            # Stale upstream framing headers are stripped relaying over IPC; the
            # body is intact decodable JSON and content-type is preserved.
            lowered = {k.lower() for k in resp.headers.keys()}
            assert "transfer-encoding" not in lowered
            assert "content-encoding" not in lowered
            assert resp.headers.get("content-type", "").startswith("application/json")

    def test_uniform_r3_strip_client_omits_records_keep(self):
        with _run_supervisor(_default_process_fn, response_patch=_patch_mock_chat_with_r3) as env:
            sid = _create(env.url)
            resp = _chat(env.url, sid, content="hi")
            assert resp.status_code == 200, resp.text

            client_meta = resp.json()["choices"][0]["meta_info"]
            assert "output_token_logprobs" in client_meta
            assert client_meta["completion_tokens"] > 0
            assert "routed_experts" not in client_meta
            assert "indexer_topk" not in client_meta

            record_meta = requests.get(f"{env.url}/sessions/{sid}", timeout=10.0).json()["records"][0]["response"][
                "choices"
            ][0]["meta_info"]
            assert record_meta["routed_experts"] == [[1, 2, 3]] * client_meta["completion_tokens"]
            assert record_meta["indexer_topk"] == [[4, 5]] * client_meta["completion_tokens"]

    def test_generic_proxy_route_passes_through(self):
        """The catch-all /sessions/{id}/{path} proxy relays the body faithfully
        over IPC: /abort_request on the mock backend returns {"status":"ok"}."""
        with _run_supervisor(_default_process_fn) as env:
            sid = _create(env.url)
            resp = requests.post(f"{env.url}/sessions/{sid}/abort_request", timeout=10.0)
            assert resp.status_code == 200, resp.text
            assert resp.json() == {"status": "ok"}
