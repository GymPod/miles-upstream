---
title: Multi-process session server
description: Design for the opt-in multi-process session server — why it exists, what is kept vs cut, and how it is split into reviewable PRs.
---

# Multi-process session server

Status: PR-A (the `SessionCore` extraction) has landed as [#1510](https://github.com/radixark/miles/pull/1510). PR-B / PR-C (the multi-process mechanism and its activation) are planned and described here for review before implementation. Sections that describe PR-B/PR-C are design, not yet-shipped code.

Readers: maintainers of the rollout/session path who will review the multi-process PRs, and anyone later debugging or extending the session server. After reading you should be able to judge: why multi-process (not threads), why sticky hash routing, what was deliberately left out and why, and where each piece lands across PRs.

## Motivation

The standalone session server (`miles/rollout/session/`) tracks per-session TITO/trajectory state and proxies each chat turn to the inference backend. Under multi-turn agent rollouts with R3 (`routed_experts` / `indexer_topk`), its dominant cost is `json.loads` + validation of large (100+ MiB) inference response bodies. That work is pure-Python and runs under one process's GIL on a single asyncio event loop, so concurrent sessions' parses serialize against each other.

A bounded thread pool does not help: the parse is GIL-bound, so threads cannot raise aggregate parse throughput and they worsen tail latency under concurrency. The only way to add parse throughput is to add interpreters — i.e. shard sessions across OS processes, each with its own GIL.

Measured on the ported overhead benchmark (`tests/benchmark/bench_session_server_overhead.py`, mock R3 backend, production-shaped `sessions=32 turns=50 r3-scale=1000`, final-turn body ≈ 134 MiB): single-process vs 16 workers is **6.7× wall-time / throughput** and **~8–10× lower reply p50/p95/p99**. The serial in-process CPU floor (no HTTP) confirms the dominant per-turn cost is the large-body `json.loads`/validate, not transport.

Why now: the mechanism is proven by that benchmark; the only thing keeping it out of `main` is review/merge risk from a single large change. This design exists to land the same mechanism as a sequence of small, independently reviewable PRs.

## Constraints

Hard constraints (each traces back to the motivation or to existing behavior we must not break):

- The default path must be unchanged. `--session-server-workers=1` keeps today's single-process server, behavior-identical. Multi-process is strictly opt-in.
- Sessions are stateful (the TITO trajectory accumulates across turns), so every turn of a session must reach the same shard. Routing must be **process-stable**: the router and every worker must agree on the owning worker for a `session_id` without coordination.
- TITO correctness under concurrency must be preserved exactly as today: the per-session `asyncio.Lock`, the `closing` re-checks, and the `num_assistant`-mismatch skip path in `SessionCore.chat_completions` are the correctness gate and must remain.
- A worker death must surface as a hard error on the rollout path, never a silent hang. A dead worker owns a hash shard; without detection, every future request to that shard would fail forever while the rollout retries into a black hole.
- No orphan processes on crash or teardown.
- No contract change may be bundled in. The client-visible response stays as-is in PR-A/B/C; the R3 client-strip is a separate, explicitly-owned change.

Soft constraints (strongly preferred, tradeable):

- Minimal review surface. This is the whole point of the effort, so robustness features that are not load-bearing for correctness are cut by default and only restored with evidence.
- Keep the IPC layer minimal; defer latency optimizations until a benchmark shows they are needed.

Non-goals this round (explicitly out of scope):

- Consistent-hashing / rebalanceable routing. v1 uses simple modulo over a stable hash; resizing the worker count at runtime is not supported.
- A delta / incremental-R3 protocol (would need SGLang/trainer coordination).
- General backpressure / admission control / rate limiting.

## Vocabulary and layering

The design has four layers; keeping them distinct is what makes the PR split clean.

- **`SessionCore`** (`session_core.py`, landed in #1510) — the **logic**. Given a request's primitives (`method`, `query`, `headers`, `body: bytes`) it mutates session state, proxies upstream, and returns a Starlette `Response`. It knows nothing about processes, sockets, IPC, or HTTP servers, and holds one `SessionRegistry`. It is reused unchanged by both chassis below.
- **worker** (`session_worker.py`, PR-B) — a **process** owning one shard. It wraps exactly one `SessionCore` plus its own httpx `ProxyBackend`, speaks IPC, and does only transport: decode a request envelope off the socket, call its `SessionCore`, serialize the returned `Response` back over IPC.
- **router** (`session_router.py`, PR-B) — the single **client-facing HTTP listener**. It routes each request to the owning worker by hash and relays the opaque response body back; it never parses the body.
- **supervisor** (`session_supervisor.py`, PR-C) — **process lifecycle**. It spawns the N workers + 1 router, waits for readiness, monitors for any child death (fail-fast), and tears the group down without orphans.

Data flow when `--session-server-workers N` (N > 1):

```
client ──HTTP──▶ router ──(blake2b(session_id) % N)──▶ IPC ──▶ worker[i] ──▶ SessionCore ──httpx──▶ upstream
                   ▲                                                                                    │
                   └──────────────────────── opaque response body relayed back ◀───────────────────────┘
supervisor: spawns + monitors router and all workers; fail-fast on any death.
```

Because sessions are sticky-by-hash, IPC only ever carries a single turn's request/response — never accumulated session state.

## Proposal selection

The shape above follows from the constraints; the non-obvious choices:

- **Processes, not threads.** The motivation is GIL-bound parse throughput; only separate interpreters add throughput. (Hard constraint: raise aggregate parse throughput.)
- **Sticky modulo-over-stable-hash routing.** Sessions are stateful, so a turn must reach the shard holding its state. `hashlib.blake2b` (not the builtin `hash()`, which is salted per process by `PYTHONHASHSEED`) gives the router and every worker the same owner for a `session_id` with no coordination. Modulo is enough because resizing N is a non-goal.
- **A thin router that relays opaque bytes, not a second HTTP hop.** The router must not re-incur the parse cost we are trying to escape, so it never `json.loads` the body; it forwards a framed message over a UNIX socket and relays the worker's response bytes verbatim.
- **Reuse `SessionCore` instead of duplicating route logic.** #1510 already made the session logic transport-neutral and returning a Starlette `Response`. The worker therefore reduces to "drive the core over IPC and serialize its `Response`"; there is no second copy of the TITO/chat logic to keep in sync. This is why PR-A was done first and on its own.

### What is kept vs cut (the core of the minimal design)

The original prototype carried a large set of robustness features. They are evaluated against one yardstick: *is it load-bearing for correctness, or for the benchmarked win?* If neither, it is cut.

Kept (load-bearing):

- **`request_id` multiplexing** on the single per-worker socket. The router fans concurrent client requests onto one channel per worker; without multiplexing we would need many sockets per worker. ~15 lines, not ceremony.
- **Supervisor fail-fast** (monitor thread + `check()` polled on the rollout path). Without it, a dead worker's shard 503s forever and the rollout hangs silently (hard constraint).
- **`PR_SET_PDEATHSIG` + SIGTERM-then-SIGKILL teardown.** Prevents orphan processes (hard constraint). ~30 lines.
- **A single sanity frame-length cap** on the IPC reader. Without it a corrupt `u32` length (up to 4 GiB) triggers a 4 GiB `readexactly` allocation → OOM instead of a clean error. This is a corruption guard, not a configurable size feature.

Cut (not load-bearing for correctness; reintroduce only with evidence):

- IPC chunking + round-robin writer + no-head-of-line-blocking scheduling, the send-buffer budget, and the configurable frame/body size limits. Replaced by single-frame bodies. (Latency tradeoff is benchmark-gated — see below.)
- Per-worker 503 backpressure, `asyncio.shield` for client-disconnect, and the router's task-tracking set. The router awaits `channel.request(...)` directly; the request is cancel-safe (it pops its pending future in `finally`), and a late reply for an unknown `request_id` is dropped — so a client disconnect cannot corrupt state or hang a future.
- Worker-side `max_inflight` / `max_queued_bytes` admission and the parse-gate semaphore. These bound memory/admission, not correctness; the per-session lock remains the correctness gate. (A single worker's event loop already serializes its parses, so the parse-gate changed nothing functional.)
- The 409 in-flight gate and 500 invariant from the prototype, and the stricter response-validation. These were behavior changes, not part of the mechanism; the minimal version keeps `main`'s behavior (concurrent same-session turns serialize on the lock; a mid-flight state change is the existing warn-and-skip → 200 path).

## Design at the executable level

### Routing — `routing.py`

```python
def new_session_id() -> str: ...                              # uuid4().hex
def worker_index_for_session(session_id: str, n_worker: int) -> int:
    # int.from_bytes(blake2b(session_id, digest_size=8)) % n_worker  — process-stable
```

Stdlib only, so a headless worker/router can import it without FastAPI.

### IPC — `session_ipc.py`

- One UNIX socket per worker (`socket.socketpair()`), one `IpcChannel` per end.
- Frame = `u32 length` + payload; payload = `u64 request_id` + `u8 type` (REQUEST / REPLY / ERROR) + envelope. One whole message per frame (no chunking).
- Envelope = `u32 meta_len` + `meta_json` + `raw_body`. `meta` carries op / method / path / query / headers / status; `body` is the raw HTTP body bytes (no base64).
- `IpcChannel.request(payload) -> bytes`: assigns a `request_id`, registers a future, sends a REQUEST frame, awaits the reply. **Cancel-safe**: a `finally` pops the pending future, so a cancelled caller's late reply is dropped on arrival.
- Server side: a `request_handler` callback runs per inbound REQUEST in its own task; its return becomes the REPLY; an exception becomes an ERROR frame (raised as `IpcError` on the caller's future).
- One reader loop, one writer loop (drains a queue of whole frames). EOF (`IncompleteReadError`) → teardown: fail all pending futures with `IpcChannelClosed`, cancel handler tasks, invoke `on_close`.
- A single `_MAX_FRAME` sanity cap is checked before `readexactly`.

### Worker — `session_worker.py`

- `ProxyBackend`: holds an `httpx.AsyncClient`; `do_proxy(ProxyRequest, path, *, body, headers)` mirrors `SessionServer.do_proxy` (build URL from `backend_url + path (+query)`, drop `content-length`/`transfer-encoding`/`host` request headers, return `{request_body, response_body, status_code, headers}`, map `httpx.TransportError` → 502). No FastAPI dependency.
- The worker owns its shard's `SessionRegistry` + tokenizer + a `SessionCore(ProxyBackend, registry, args, instance_id)`.
- `request_handler` decodes the envelope, dispatches by op (`HEALTH` / `CREATE_ID` / `GET` / `DELETE` / `CHAT` / `PROXY`) to the core, reads the returned `Response.body` / `.status_code` / `dict(resp.headers)`, and encodes the REPLY. A `SessionError` raised by the core is caught and turned into the same `{"error": msg}` + status response the single-process FastAPI exception handler produces.
- `run_worker(args, backend_url, sock, worker_index)` is the `multiprocessing.Process` target: set process title + pdeathsig, build the core, open the channel, serve until the channel closes.

### Router — `session_router.py`

- A FastAPI app holding N `IpcChannel`s. Its routes mirror the session routes.
- Per request: pick the owner with `worker_index_for_session` (for `POST /sessions`, mint the id first then dispatch `CREATE_ID` to its owner so create lands on the same worker that later `get`/`chat`/`delete` will route to), `await channel.request(payload)`, and re-wrap the reply as `Response(content=body, status_code=..., headers=...)` — passing `dict(resp.headers)` and **not** a separate `media_type` (content-type is already in the headers, so Starlette will not duplicate `content-length`/`content-type`).
- `/health` pings all workers with one outer timeout. `IpcChannelClosed` → 503, `IpcError` → 502.

### Supervisor — `session_supervisor.py`

- `start()`: spawn context `"spawn"`; one `socketpair` per worker; spawn N workers + 1 router; the parent closes every socket end so a child death is observable as EOF on the peer; wait for readiness (poll the router `/health` until all-ok, or a child dies, or one deadline elapses); start a monitor thread that polls `is_alive()` and, on any death, records the failure and kills the group; register an `atexit` shutdown.
- `check()` raises if a failure was recorded — called on the rollout path.
- `shutdown()` sends SIGTERM, waits a grace period, then SIGKILL; idempotent.

### Registry change — `linear_trajectory.py`

- Add `SessionRegistry.create_session_with_id(session_id)`: create a `LinearTrajectory` under an explicit id (validated against `^[0-9a-f]{32}$`, raising on a bad shape or a collision). The router mints the id and the owning worker creates under it. The single-process path never calls this method, so its behavior is unchanged.

### Wiring (PR-C)

- New flag `--session-server-workers` (int, default 1; `1` = single-process, no router/IPC).
- `router_manager.start_session_server`: `workers == 1` keeps the current single-process spawn; `workers > 1` builds and starts a `SessionServerSupervisor` and returns it (held to prevent GC).
- `rollout_manager`: capture the supervisor and call `check()` at the start of **both** `generate()` and `eval()` (mirroring the prototype's `_check_session_server`). Calling it in only one would leave a generate-only or eval-only phase able to hang silently on a worker death.

## Accepted behavior notes

These are the only observable changes relative to the pre-refactor single-process server; both were introduced by PR-A (#1510) and confirmed (by a cross-model audit of the diff) to be the *only* changes beyond pure code-movement/extraction:

- **Body-read timing / lock scope.** The route now reads `await request.body()` before entering `SessionCore.chat_completions`, whereas before the body was read inside `session.lock` after the session lookup. JSON parsing, TITO mutation, and record writes remain under the lock; proxying remains outside it. Consequences: a chat to a missing/closing session consumes the full body before the error returns; in a DELETE-vs-chat race during upload the chat is more likely to be rejected (404). The race outcome was already non-deterministic and the race-condition tests pass. This timing is also inherent to multi-process, where the worker receives already-read bytes over IPC.
- **Debug request-logging removed.** The per-request `debug_request_logger` middleware (the `[session-server] REQUEST ARRIVED/DONE …` INFO lines and the `_inflight_chat` counter) and the two DELETE lock `debug` lines are gone. Logging-only; no contract impact.

## PR decomposition

The mechanism lands as a stack of small PRs so each is independently reviewable and the risky parts are isolated.

- **PR-A — extract `SessionCore`** ([#1510](https://github.com/radixark/miles/pull/1510), landed). Behavior-preserving single-process refactor; the enabler for everything below.
- **PR-B — data plane.** `routing.py` + `session_ipc.py` + `session_worker.py` + `session_router.py` + `SessionRegistry.create_session_with_id`, with tests (IPC units, worker end-to-end against a mock backend, routing determinism, router↔workers equivalence). The tests spawn workers and run the router directly, so the supervisor is not needed to exercise the mechanism. **PR-B adds no live behavior** — nothing wires it into launch, so the default path is untouched.
- **PR-C — control plane / activation.** `session_supervisor.py` + the `--session-server-workers` flag + the `router_manager` branch + the `rollout_manager` `check()` calls. This is what makes `--session-server-workers N` actually launch and integrates fail-fast into the rollout loop. The kill-a-worker → `check()` raises test lives here.
- **R3 client-strip** — a separate small commit, only if still wanted: strip `routed_experts`/`indexer_topk` from the client-facing chat response uniformly (records keep R3 for `GET /sessions`). It is a client-contract change orthogonal to multi-process, so it is reviewed on its own.

Splitting PR-C out keeps the riskiest part — process orchestration, fail-fast, and launch integration — as its own focused review, while PR-B is "new modules + tests, wired to nothing."

## Round-robin: a benchmark-gated decision

The cut of IPC chunking/round-robin is the one place where "too minimal" could matter, so it is decided by measurement, not assertion.

The headline 6.7× is a **chat-only** run ("no full-records GET"). On the chat path the client-facing response is small, so the bytes that cross IPC per turn are small; the large body is parsed *inside* the worker (the win) and is not sent over IPC. Large bodies only cross IPC on `GET /sessions/{id}` (full retained records), which the headline benchmark does not drive. Therefore round-robin / no-head-of-line-blocking is not exercised by the headline win at all.

Plan: ship single-frame IPC, then before merge run the benchmark both chat-only (expect the ~6.7× to hold) and with `--get-records` (measure chat reply p99 while large GETs run concurrently). Reintroduce round-robin **only if** the `--get-records` run shows chat p99 collapse **and** production actually drives concurrent large GETs alongside latency-sensitive chats on the same worker. Re-adding it touches only the IPC writer; the worker/router contract is unchanged.

## Risks and open questions

Accepted risks:

- **Single-frame large bodies.** A 100+ MiB `GET /sessions` body is sent/received as one frame and held in memory (the records are already in memory). Memory cost is acceptable; the latency cost is the round-robin question above, gated on the benchmark. Mitigated against corruption by the sanity frame cap.
- **Cutting client-disconnect shielding.** Relies on cancel-safe `request()` + late-reply drop rather than `asyncio.shield`. A worker handler is a separate task, so it still commits consistent state; verified by a deterministic disconnect/cancel test (not the prototype's timing-flaky one).

Open questions:

- **Does production overlap large `GET /sessions` with latency-sensitive chats on the same worker?** This is the precondition for round-robin mattering, and it is unverified. The round-robin decision is deferred to the `--get-records` benchmark result; until then it stays cut.
- **Biggest correctness risk to guard in tests: silent session mis-routing.** If `worker_index_for_session` were computed differently at create vs. chat vs. get, a session could be created on one worker and addressed on another — intermittent, load-dependent, and invisible at `workers=1`. PR-B must include a routing-determinism test (≥3 workers, several ids exercising ≥2 distinct workers, asserting create→chat→get→delete all hit the same worker) and the equivalence test must assert `status_code` + headers + raw body bytes are byte-identical between `workers=N` and `workers=1` (including 204-delete and the 502 transport-error path) plus that the TITO `get_session` metadata matches after a multi-turn chat through a worker.
