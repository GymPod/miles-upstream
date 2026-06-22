"""Event-loop-responsiveness micro-benchmark for the standalone SessionServer.

Why this exists (AC-6.1, directional / non-blocking):
The session server offloads stateless CPU work (large JSON parse/dump of the
chat request + response, including hundreds-of-KB `routed_experts` blobs) off the
single asyncio event loop onto a bounded `SessionServer.cpu_executor`. The claim
is that this keeps the one event loop responsive under heavy-response load so the
liveness probe `GET /health` (handled inline on the loop) stays fast. It does NOT
claim higher total CPU throughput: the GIL still serializes the Python JSON work,
so this measures latency/responsiveness, not aggregate CPU.

This file is named `bench_*` so pytest does NOT auto-collect it (no flaky timing
test in CI). Run it directly:  python tests/fast/router/bench_session_responsiveness.py

Method: fire K concurrent chats across K DISTINCT sessions (distinct sessions are
not gated, so they run in parallel), each producing a large response that forces a
CPU-heavy `json.loads` in `_parse_and_validate_response`. Concurrently, a separate
thread polls `GET /health` every ~10ms and records each round-trip latency. We
report chat throughput, response body size, and `/health` latency percentiles. All
timing is client-side `time.perf_counter`.
"""

from __future__ import annotations

import json
import logging
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import requests

# Quiet the uvicorn access logs from both servers so the summary block is the
# only thing on stdout (the per-request "GET /health 200 OK" lines bury it).
for _name in ("uvicorn", "uvicorn.access", "uvicorn.error"):
    logging.getLogger(_name).setLevel(logging.WARNING)

from miles.rollout.session.session_server import SessionServer
from miles.utils.http_utils import find_available_port
from miles.utils.test_utils.mock_sglang_server import MockSGLangServer, ProcessResult, with_mock_server
from miles.utils.test_utils.uvicorn_thread_server import UvicornThreadServer

# --- Tunable constants (kept modest so a run finishes well under a minute) ---
HF_CHECKPOINT = "Qwen/Qwen3-0.6B"
K_CHATS = 12  # concurrent chats, one per distinct (ungated) session
# The session server json.loads the WHOLE response body to validate it, but the
# `routed_experts` value is opaque to it (never decoded). To isolate the CPU
# parse cost the offload targets — without inflating the on-loop body I/O that
# scales with raw byte size — we make routed_experts a STRUCTURED blob (nested
# numeric arrays). Structured JSON parses ~7-8x slower per byte than one big
# string, so a modest ~2 MiB body costs ~15-30ms to parse. Run INLINE on the loop
# (pre-offload) that stalls every /health probe queued behind it; offloaded to
# cpu_executor it runs off-loop and the probe stays fast.
BLOB_ROWS = 900  # rows of BLOB_ROW_WIDTH floats -> ~1.5 MiB body, ~18ms parse/call
BLOB_ROW_WIDTH = 256
HEALTH_POLL_INTERVAL_S = 0.01  # ~10ms between /health probes
HEALTH_POLL_DURATION_S = 4.0  # how long the health poller runs (also caps the load window)
CHAT_TEXT = "responsiveness-ok"


def _make_large_blob() -> list:
    # Nested numeric arrays, like a routed_experts logits buffer: expensive to
    # json.loads per byte (many tokens), which is exactly the CPU work the
    # session server offloads off the event loop.
    row = [round(i * 0.001, 4) for i in range(BLOB_ROW_WIDTH)]
    return [list(row) for _ in range(BLOB_ROWS)]


def _patch_mock_chat_response_heavy(blob: list):
    """Patch the mock chat response into the shape the session server validates
    (output_token_logprobs as (logprob, token_id)), and inject a large STRUCTURED
    routed_experts blob into meta_info so the session server's json.loads of the
    response body is genuinely CPU-heavy. The blob is opaque to the session server
    (it only validates output_token_logprobs / message), matching production where
    routed_experts is an opaque buffer the session layer never decodes.
    """
    original_chat_response = MockSGLangServer._compute_chat_completions_response

    def patched_chat_response(self, payload: dict) -> dict:
        response = original_chat_response(self, payload)
        choice = response["choices"][0]
        logprobs_content = choice["logprobs"]["content"]
        output_token_logprobs = [
            (item["logprob"], self.tokenizer.convert_tokens_to_ids(item["token"])) for item in logprobs_content
        ]
        choice["meta_info"]["output_token_logprobs"] = output_token_logprobs
        choice["meta_info"]["completion_tokens"] = len(output_token_logprobs)
        choice["meta_info"]["routed_experts"] = blob
        return response

    return patch.object(MockSGLangServer, "_compute_chat_completions_response", new=patched_chat_response)


@contextmanager
def _router_env(process_fn, blob: list, *, latency: float = 0.0):
    with _patch_mock_chat_response_heavy(blob):
        with with_mock_server(model_name=HF_CHECKPOINT, process_fn=process_fn, latency=latency) as backend:
            args = SimpleNamespace(
                miles_router_timeout=30,
                hf_checkpoint=HF_CHECKPOINT,
                chat_template_path=None,
                trajectory_manager="linear_trajectory",
                tito_allowed_append_roles=["tool", "system"],
                # Make the session server request + echo a big routed_experts blob,
                # so the per-response json.loads is genuinely CPU-heavy.
                use_rollout_routing_replay=True,
            )
            server_obj = SessionServer(args, backend_url=backend.url)

            port = find_available_port(31000)
            server = UvicornThreadServer(server_obj.app, host="127.0.0.1", port=port)
            server.start()
            # uvicorn.Config(log_level="info") reconfigures these on startup, so
            # re-quiet them now that both servers are up (keeps stdout to the summary).
            for _name in ("uvicorn", "uvicorn.access", "uvicorn.error"):
                logging.getLogger(_name).setLevel(logging.WARNING)
            url = f"http://127.0.0.1:{port}"
            try:
                yield SimpleNamespace(url=url, backend=backend, server=server)
            finally:
                server.stop()


def _create_session(url: str) -> str:
    resp = requests.post(f"{url}/sessions", timeout=5.0)
    assert resp.status_code == 200, resp.text
    return resp.json()["session_id"]


def _chat(url: str, session_id: str) -> tuple[int, int]:
    """Returns (status_code, response_body_size_bytes)."""
    resp = requests.post(
        f"{url}/sessions/{session_id}/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "drive-load"}]},
        timeout=60.0,
    )
    return resp.status_code, len(resp.content)


class _HealthPoller(threading.Thread):
    """Polls GET /health on a fixed cadence and records each round-trip latency
    (seconds, client-side perf_counter). Runs until stop() or duration elapses."""

    def __init__(self, url: str, interval_s: float, max_duration_s: float):
        super().__init__(daemon=True)
        self.url = f"{url}/health"
        self.interval_s = interval_s
        self.max_duration_s = max_duration_s
        self.latencies_s: list[float] = []
        self.errors = 0
        self._stop_evt = threading.Event()
        self._session = requests.Session()

    def stop(self) -> None:
        self._stop_evt.set()

    def run(self) -> None:
        deadline = time.perf_counter() + self.max_duration_s
        while not self._stop_evt.is_set() and time.perf_counter() < deadline:
            t0 = time.perf_counter()
            try:
                resp = self._session.get(self.url, timeout=5.0)
                dt = time.perf_counter() - t0
                if resp.status_code == 200:
                    self.latencies_s.append(dt)
                else:
                    self.errors += 1
            except requests.RequestException:
                self.errors += 1
            time.sleep(self.interval_s)


def _pct(values_ms: list[float], q: float) -> float:
    if not values_ms:
        return float("nan")
    ordered = sorted(values_ms)
    idx = min(len(ordered) - 1, int(q * len(ordered)))
    return ordered[idx]


def run_bench() -> dict:
    blob = _make_large_blob()
    # The CPU cost the offload targets: one json.loads of a response carrying this
    # blob. Measured up front so the summary states how heavy the offloaded work is.
    sample_body = json.dumps({"choices": [{"meta_info": {"routed_experts": blob}}]}).encode()
    _t = time.perf_counter()
    json.loads(sample_body)
    parse_ms = (time.perf_counter() - _t) * 1000

    def process_fn(_prompt: str) -> ProcessResult:
        # routed_experts is injected by the response patch (kept off the str-typed
        # dataclass field); this just drives a small assistant message.
        return ProcessResult(text=CHAT_TEXT, finish_reason="stop")

    with _router_env(process_fn, blob) as env:
        session_ids = [_create_session(env.url) for _ in range(K_CHATS)]

        # Warm one chat so we can log the real response body size before the run.
        warm_status, warm_size = _chat(env.url, session_ids[0])
        assert warm_status == 200, f"warm-up chat failed: {warm_status}"

        # Baseline: /health latency with no chat load (single thread, short burst).
        baseline = _HealthPoller(env.url, HEALTH_POLL_INTERVAL_S, max_duration_s=0.7)
        baseline.start()
        baseline.join()
        baseline_ms = [s * 1000 for s in baseline.latencies_s]

        # Under load: poll /health while waves of K concurrent chats keep the
        # backend + cpu_executor busy. A single wave of K first-turn chats is
        # fast, so we fire back-to-back waves until the poll window elapses; this
        # sustains heavy-response load long enough to sample /health meaningfully.
        # Each chat uses its own fresh, distinct session (distinct sessions are
        # ungated, so they run in parallel).
        poller = _HealthPoller(env.url, HEALTH_POLL_INTERVAL_S, HEALTH_POLL_DURATION_S)
        poller.start()

        results: list[tuple[int, int]] = []
        waves = 0
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=K_CHATS) as pool:
            while time.perf_counter() - t0 < HEALTH_POLL_DURATION_S:
                fresh_ids = [_create_session(env.url) for _ in range(K_CHATS)]
                futures = [pool.submit(_chat, env.url, sid) for sid in fresh_ids]
                results.extend(f.result(timeout=120.0) for f in futures)
                waves += 1
        chat_wall_s = time.perf_counter() - t0

        poller.stop()
        poller.join()

        statuses = [s for s, _ in results]
        sizes = [sz for _, sz in results]
        ok = sum(1 for s in statuses if s == 200)
        load_ms = [s * 1000 for s in poller.latencies_s]

    total = len(results)
    return {
        "k_chats": K_CHATS,
        "waves": waves,
        "parse_ms": parse_ms,
        "total_chats": total,
        "ok_chats": ok,
        "failed_chats": total - ok,
        "response_body_bytes": warm_size if not sizes else max(sizes),
        "chat_wall_s": chat_wall_s,
        "chat_throughput_per_s": ok / chat_wall_s if chat_wall_s > 0 else float("nan"),
        "health_samples_under_load": len(load_ms),
        "health_errors_under_load": poller.errors,
        "health_baseline_samples": len(baseline_ms),
        "baseline_ms": baseline_ms,
        "load_ms": load_ms,
    }


def _fmt_block(r: dict) -> str:
    base = r["baseline_ms"]
    load = r["load_ms"]
    lines = [
        "=" * 64,
        "SessionServer event-loop responsiveness benchmark",
        "=" * 64,
        f"  concurrency / wave (K)      : {r['k_chats']}",
        f"  waves driven                : {r['waves']}  (total {r['total_chats']} chats)",
        f"  response body size (actual) : {r['response_body_bytes'] / 1024:.1f} KiB",
        f"  single-response parse cost  : {r['parse_ms']:.1f} ms  (the offloaded CPU work)",
        f"  chats ok / failed           : {r['ok_chats']} / {r['failed_chats']}",
        f"  chat wall-clock             : {r['chat_wall_s']:.3f} s",
        f"  chat throughput             : {r['chat_throughput_per_s']:.1f} chats/s",
        "-" * 64,
        f"  /health baseline (no load)  : n={r['health_baseline_samples']}, "
        f"p50={_pct(base, 0.50):.2f}ms  p95={_pct(base, 0.95):.2f}ms  "
        f"max={(max(base) if base else float('nan')):.2f}ms",
        f"  /health UNDER LOAD          : n={r['health_samples_under_load']}, "
        f"errors={r['health_errors_under_load']}",
        f"      p50 = {_pct(load, 0.50):.2f} ms",
        f"      p95 = {_pct(load, 0.95):.2f} ms",
        f"      p99 = {_pct(load, 0.99):.2f} ms",
        f"      max = {(max(load) if load else float('nan')):.2f} ms",
        f"      mean= {(statistics.mean(load) if load else float('nan')):.2f} ms",
        "=" * 64,
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    result = run_bench()
    print(_fmt_block(result))
