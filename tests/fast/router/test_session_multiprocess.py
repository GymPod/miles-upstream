"""End-to-end test of the REAL multi-process session server (workers=N).

Unlike ``test_session_worker`` (one in-process worker over a socketpair), this
drives the full launch path: a :class:`SessionServerSupervisor` spawns N real
worker processes + 1 real router process under a multiprocessing **spawn**
context, all talking to a thread-based mock SGLang backend over loopback TCP.
The test client speaks plain HTTP to the router, exactly as the rollout path
does.

Covers (m3-design-contract §"Verification"):

* create -> chat -> GET -> DELETE round-trips land consistently through the
  router + the owning worker;
* a same-session second concurrent chat -> 409 WITHOUT reaching the backend;
* two distinct sessions' chats overlap across simulated upstream latency;
* a large GET-records reply concurrent with a small DELETE/health does NOT stall
  the small one (head-of-line check, end to end over IPC chunking);
* the client chat body OMITS routed_experts/indexer_topk while GET-records keep
  them (uniform R3 strip);
* killing a worker triggers fail-fast (the supervisor records the failure and
  tears the group down) with no leftover worker/router processes.
"""

from __future__ import annotations

import os
import signal
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import requests

from miles.rollout.session.routing import worker_index_for_session
from miles.utils.http_utils import find_available_port
from miles.utils.test_utils.mock_sglang_server import ProcessResult, with_mock_server

HF_CHECKPOINT = "Qwen/Qwen3-0.6B"
N_WORKERS = 2


def _patch_mock_chat_response_with_r3():
    """Inject per-turn replay blobs (routed_experts / indexer_topk) into the
    mock's choice.meta_info so the uniform R3 strip can be checked end to end.
    """
    from miles.utils.test_utils.mock_sglang_server import MockSGLangServer

    original = MockSGLangServer._compute_chat_completions_response

    def patched(self, payload: dict) -> dict:
        response = original(self, payload)
        choice = response["choices"][0]
        otl = choice["meta_info"]["output_token_logprobs"]
        choice["meta_info"]["routed_experts"] = [[1, 2, 3]] * len(otl)
        choice["meta_info"]["indexer_topk"] = [[4, 5]] * len(otl)
        return response

    return patch.object(MockSGLangServer, "_compute_chat_completions_response", new=patched)


def _args(instance_id="mp-test", **overrides):
    base = dict(
        hf_checkpoint=HF_CHECKPOINT,
        chat_template_path=None,
        tito_model="default",
        tito_allowed_append_roles=["tool", "system"],
        apply_chat_template_kwargs=None,
        miles_router_timeout=30,
        session_server_workers=N_WORKERS,
        session_server_instance_id=instance_id,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _default_process_fn(prompt: str) -> ProcessResult:
    return ProcessResult(text="ok", finish_reason="stop")


def _start_supervisor(args, backend_url, *, response_patch=None):
    """Spawn the real supervisor (N workers + router) against ``backend_url``.

    Returns ``(supervisor, base_url)``. The caller must shut the supervisor down.
    The optional ``response_patch`` only affects the in-test mock server.
    """
    from miles.rollout.session.session_supervisor import SessionServerSupervisor

    ip = "127.0.0.1"
    port = find_available_port(32000)
    supervisor = SessionServerSupervisor(args, backend_url, ip, port)
    supervisor.start()
    return supervisor, f"http://{ip}:{port}"


@pytest.fixture
def server():
    """A running mock backend + a real spawned multi-process session server."""
    with with_mock_server(model_name=HF_CHECKPOINT, process_fn=_default_process_fn) as backend:
        with _patch_mock_chat_response_with_r3():
            supervisor, url = _start_supervisor(_args(), backend.url)
            try:
                yield SimpleNamespace(url=url, backend=backend, supervisor=supervisor)
            finally:
                supervisor.shutdown()


def _create(url):
    r = requests.post(f"{url}/sessions", timeout=10.0)
    assert r.status_code == 200, r.text
    return r.json()["session_id"]


def _chat(url, sid, content="hi", timeout=20.0):
    return requests.post(
        f"{url}/sessions/{sid}/v1/chat/completions",
        json={"messages": [{"role": "user", "content": content}]},
        timeout=timeout,
    )


def test_health_reports_instance_id(server):
    r = requests.get(f"{server.url}/health", timeout=10.0)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["session_server_instance_id"] == "mp-test"


def test_create_chat_get_delete_round_trip(server):
    sid = _create(server.url)

    resp = _chat(server.url, sid)
    assert resp.status_code == 200, resp.text
    client_meta = resp.json()["choices"][0]["meta_info"]
    # Uniform R3 strip on the client body; logprob meta retained.
    assert "output_token_logprobs" in client_meta
    assert "routed_experts" not in client_meta
    assert "indexer_topk" not in client_meta

    # GET records: full R3 retained for training-sample reconstruction.
    got = requests.get(f"{server.url}/sessions/{sid}", timeout=10.0)
    assert got.status_code == 200
    records = got.json()["records"]
    assert len(records) == 1
    rec_meta = records[0]["response"]["choices"][0]["meta_info"]
    assert "routed_experts" in rec_meta
    assert "indexer_topk" in rec_meta

    # DELETE -> 204; subsequent GET -> 404 (state lands on the owning worker).
    assert requests.delete(f"{server.url}/sessions/{sid}", timeout=10.0).status_code == 204
    assert requests.get(f"{server.url}/sessions/{sid}", timeout=10.0).status_code == 404


def test_sessions_spread_across_workers(server):
    """With N=2 workers, fresh ids should land on both workers (sticky routing
    is by id, so a round-trip on each is independently consistent)."""
    sids = [_create(server.url) for _ in range(12)]
    indices = {worker_index_for_session(s, N_WORKERS) for s in sids}
    assert indices == {0, 1}, f"sessions did not spread across both workers: {indices}"
    for sid in sids:
        assert _chat(server.url, sid).status_code == 200


def test_same_session_second_concurrent_chat_409_without_backend():
    """While one chat is parked in a latency-gated backend, a second same-session
    chat must 409 from the in-flight gate WITHOUT a second backend request."""
    with with_mock_server(model_name=HF_CHECKPOINT, process_fn=_default_process_fn, latency=0.6) as backend:
        supervisor, url = _start_supervisor(_args(), backend.url)
        try:
            sid = _create(url)
            backend.reset_stats()

            result = {}

            def _owner():
                result["owner"] = _chat(url, sid, content="park", timeout=20.0)

            t = threading.Thread(target=_owner)
            t.start()
            # Wait until the owner's request is parked in the backend.
            deadline = time.time() + 10.0
            while len(backend.request_log) < 1 and time.time() < deadline:
                time.sleep(0.01)
            assert len(backend.request_log) == 1

            contender = _chat(url, sid, content="contend", timeout=20.0)
            assert contender.status_code == 409
            assert contender.json()["error"] == "session already has an in-flight chat completion"
            # The 409 never reached the backend.
            assert len(backend.request_log) == 1

            t.join(timeout=20.0)
            assert result["owner"].status_code == 200
        finally:
            supervisor.shutdown()


def test_distinct_sessions_chats_overlap():
    """Two distinct sessions' chats overlap across the upstream latency window
    (max_concurrent >= 2 at the backend), i.e. the workers do not serialize."""
    with with_mock_server(model_name=HF_CHECKPOINT, process_fn=_default_process_fn, latency=0.6) as backend:
        supervisor, url = _start_supervisor(_args(), backend.url)
        try:
            sids = [_create(url) for _ in range(2)]
            backend.reset_stats()
            results = {}

            def _run(sid):
                results[sid] = _chat(url, sid, timeout=20.0)

            threads = [threading.Thread(target=_run, args=(s,)) for s in sids]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=20.0)

            assert all(results[s].status_code == 200 for s in sids), {s: results[s].status_code for s in sids}
            assert backend.max_concurrent >= 2, f"chats did not overlap: max_concurrent={backend.max_concurrent}"
        finally:
            supervisor.shutdown()


def test_large_get_records_does_not_stall_small_request():
    """A large GET-records reply (many big-blob turns) being relayed over IPC
    must NOT block a concurrent small /health on the same worker — the chunked
    multiplexed framing prevents head-of-line blocking end to end.
    """

    def _patch_big_r3():
        from miles.utils.test_utils.mock_sglang_server import MockSGLangServer

        original = MockSGLangServer._compute_chat_completions_response

        def patched(self, payload: dict) -> dict:
            response = original(self, payload)
            choice = response["choices"][0]
            # A single multi-MiB R3 blob: the stored record (hence GET-records
            # reply) spans many IPC chunks. One turn, so TITO stays clean.
            choice["meta_info"]["routed_experts"] = [list(range(256)) for _ in range(4096)]
            return response

        return patch.object(MockSGLangServer, "_compute_chat_completions_response", new=patched)

    with with_mock_server(model_name=HF_CHECKPOINT, process_fn=_default_process_fn) as backend:
        with _patch_big_r3():
            supervisor, url = _start_supervisor(_args(), backend.url)
            try:
                sid = _create(url)
                # One turn carrying a large R3 blob -> a big GET-records reply.
                assert _chat(url, sid, timeout=20.0).status_code == 200

                big_done = {}

                def _big_get():
                    t0 = time.time()
                    r = requests.get(f"{url}/sessions/{sid}", timeout=30.0)
                    big_done["status"] = r.status_code
                    big_done["bytes"] = len(r.content)
                    big_done["elapsed"] = time.time() - t0

                t = threading.Thread(target=_big_get)
                t.start()
                # The small /health must return promptly even while the big GET
                # is mid-flight over IPC (no HOL blocking).
                small_t0 = time.time()
                h = requests.get(f"{url}/health", timeout=10.0)
                small_elapsed = time.time() - small_t0
                assert h.status_code == 200
                assert small_elapsed < 5.0, f"small /health stalled behind big GET: {small_elapsed:.2f}s"

                t.join(timeout=30.0)
                assert big_done.get("status") == 200
                assert big_done.get("bytes", 0) > (1 << 20), "GET-records reply was not actually large"
            finally:
                supervisor.shutdown()


def test_large_get_serialization_does_not_block_same_worker():
    """The big GET-records serialization (model_dump + json.dumps, 10s of MiB)
    is offloaded off the event loop, so a SMALL GET on a DIFFERENT session that
    happens to live on the SAME worker still returns promptly while the big one
    serializes. Forces both sessions onto one worker so the small request can
    only be served by the loop that is busy producing the big body.
    """

    def _patch_big_r3():
        from miles.utils.test_utils.mock_sglang_server import MockSGLangServer

        original = MockSGLangServer._compute_chat_completions_response

        def patched(self, payload: dict) -> dict:
            response = original(self, payload)
            choice = response["choices"][0]
            # ~tens of MiB once serialized, so an inline dump would visibly block.
            choice["meta_info"]["routed_experts"] = [list(range(256)) for _ in range(20000)]
            return response

        return patch.object(MockSGLangServer, "_compute_chat_completions_response", new=patched)

    with with_mock_server(model_name=HF_CHECKPOINT, process_fn=_default_process_fn) as backend:
        with _patch_big_r3():
            supervisor, url = _start_supervisor(_args(), backend.url)
            try:
                # Two sessions on the SAME worker.
                big_sid = _create(url)
                small_sid = None
                for _ in range(64):
                    cand = _create(url)
                    if worker_index_for_session(cand, N_WORKERS) == worker_index_for_session(big_sid, N_WORKERS):
                        small_sid = cand
                        break
                assert small_sid is not None, "could not place two sessions on the same worker"

                # Give big_sid a large R3 turn; small_sid stays empty (tiny GET).
                assert _chat(url, big_sid, timeout=20.0).status_code == 200

                big_done = {}

                def _big_get():
                    r = requests.get(f"{url}/sessions/{big_sid}", timeout=30.0)
                    big_done["status"] = r.status_code
                    big_done["bytes"] = len(r.content)

                t = threading.Thread(target=_big_get)
                t.start()
                # Small GET on the SAME worker must return promptly even while the
                # big body serializes (proves the dump is off the loop).
                small_t0 = time.time()
                small = requests.get(f"{url}/sessions/{small_sid}", timeout=10.0)
                small_elapsed = time.time() - small_t0
                assert small.status_code == 200, small.text
                assert small_elapsed < 5.0, f"small GET stalled behind big-GET serialization: {small_elapsed:.2f}s"

                t.join(timeout=30.0)
                assert big_done.get("status") == 200
                assert big_done.get("bytes", 0) > (10 << 20), "big GET was not actually large"
            finally:
                supervisor.shutdown()


def test_worker_death_triggers_fail_fast():
    """Killing one worker process must make the supervisor fail fast (record the
    failure + tear the whole group down) with no leftover worker/router procs."""
    with with_mock_server(model_name=HF_CHECKPOINT, process_fn=_default_process_fn) as backend:
        supervisor, url = _start_supervisor(_args(), backend.url)
        try:
            # Sanity: server is up before we kill a worker.
            assert requests.get(f"{url}/health", timeout=10.0).status_code == 200

            victim = supervisor._workers[0]
            victim_pid = victim.pid
            router_pid = supervisor._router.pid
            all_pids = [w.pid for w in supervisor._workers] + [router_pid]

            os.kill(victim_pid, signal.SIGKILL)

            # The monitor must observe the death and fail fast.
            deadline = time.time() + 15.0
            while not supervisor.failed and time.time() < deadline:
                time.sleep(0.1)
            assert supervisor.failed, "supervisor did not record a fail-fast on worker death"
            with pytest.raises(RuntimeError):
                supervisor.check()

            # No leftover children: the group is torn down.
            deadline = time.time() + 15.0
            while time.time() < deadline and any(_pid_alive(p) for p in all_pids):
                time.sleep(0.1)
            leftovers = [p for p in all_pids if _pid_alive(p)]
            assert not leftovers, f"leftover session-server processes after fail-fast: {leftovers}"
        finally:
            supervisor.shutdown()


def _pid_alive(pid: int | None) -> bool:
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # A zombie (already reaped child) is not "alive" for our purposes; if it is
    # still our child, multiprocessing.join reaps it, so kill(0) succeeding means
    # the kernel still has the pid -> treat as alive until reaped.
    return True
