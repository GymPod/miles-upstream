# Session server 设计 —— 多进程重构

读者：`refactor/multi-process-session-server` 分支的 reviewer / maintainer，已经清楚 Miles session server 是做什么的（在多轮 agentic rollout 上做 TITO token 跟踪），想理解本分支新增的多进程层，以及为支持它而对请求逻辑做的重构。本文假设读者熟悉 TITO 和 linear-trajectory 模型，只在锚定下文决策所必需的程度上对它们做概述。

## 1. 这次重构为什么存在

session server 的职责是夹在 rollout 客户端与 inference router 之间，维护每个 session 的 *linear trajectory*：它用 Miles 自持的 `input_ids` 对每一轮 prompt 做预分词，累积实际被采样出来的 token IDs，并按轮存一条 `SessionRecord`，使得训练侧之后能从 `GET /sessions/{id}` 重建出 token 级精确的样本（logprobs、routed-experts / indexer replay）。这些领域逻辑都在 `linear_trajectory.py`（`LinearTrajectory`、`SessionRegistry`）里，本文当作既定前提。

重构前的 server（`session_server.py`、`run_session_server`）是一个故意只用**单个** worker 的 uvicorn 进程。`session_server.py:109-111` 的注释给出了原因：多开 uvicorn worker 会让每个进程各持一份 `SessionRegistry` + `asyncio.Lock`，于是某个 `session_id` 可能落到并不拥有它的进程上。session 的归属必须是 sticky 的，而保证它的廉价办法就是「一个进程，一份 registry」。

单进程意味着只有一个 asyncio event loop 服务所有请求。只要请求不在 loop 线程上做重的*同步* CPU，这就没问题。但有两类请求恰恰会：

- `GET /sessions/{id}` 要序列化整个 record 列表，可达 100+ MiB（`session_core.py:83`、`:294`）。`model_dump()` + `json.dumps()` 是同步 CPU。
- 每一轮 chat 都要 parse + validate 一个很大的 upstream 响应（`_parse_and_validate_response`，`session_core.py:162`），同样是同步 CPU。

当 loop 正忙于其中之一时，这条 loop 上的其它所有请求——包括 `/health` 和*其它* session 的小 chat 轮次——全都被卡住。单 event loop 就是吞吐天花板，而要抬高它，要么用线程（这类 JSON/CPU 活计受 GIL 限制），要么多开进程。重构前的代码已经把 GET 这一路的序列化甩到线程里缓解（`asyncio.to_thread`，`session_core.py:315`），但结构性天花板仍在。

**重构目标：** 让 session server 跨进程 / 跨核扩展，*同时*不丢掉 sticky session ownership，*并且*不把请求处理逻辑 fork 成会逐渐漂移的第二份实现。结果必须与单进程 server 行为一致（多进程 equivalence 测试钉住这一点）。

这是**可选开启**的：`--session-server-workers 1`（默认）保留原来的单进程路径，没有 router、没有 IPC。`--session-server-workers N`（N > 1）才激活下文描述的全部机制。本分支其实把三件事一起带上来了——transport-neutral 拆分、多进程机制、以及一个无关的 record 体积优化（丢弃被取代的 `routed_experts` / `indexer_topk` blob，见 §8）——但只有前两者对多进程是承重的。

## 2. 使能动作：transport-neutral 的 `SessionCore`

最硬的约束是「别 fork 逻辑」。同一套 create / get / delete / chat / proxy 行为要跑两种形态：进程内挂在 FastAPI 后面（单进程模式），以及进程外、运行在一个 headless worker 里、由一个 socket 驱动（多进程模式）。如果两种模式各自重写路由，迟早会漂移。

重构的答案是 `session_core.py`：每个 session 操作只在 `SessionCore` 里实现一次，面向 **primitive** 输入（`method`、`path`、`query`、`headers`、`body`，外加 `session_id`），返回一个带类型的 `CoreResponse`（status / headers / body-bytes / media-type）。`SessionCore` 不 import `fastapi.Request`、不 import `starlette.Response`、不碰任何 ASGI 机制。两个推论使它成为整个设计的支点：

- **`SessionError` 在 core *内部*被映射成 JSON `CoreResponse`**（`_error_response`，`session_core.py:503`），而不是靠 FastAPI 的 exception handler。于是一个非 FastAPI 的 adapter（worker）能免费拿到正确的 HTTP 状态码。
- **upstream proxy 被抽象成 `backend.do_proxy`**，接收一个 primitive 的 `ProxyRequest`（`session_core.py:58`），于是 core 自己从不持有 HTTP client。单进程提供 `SessionServer`，worker 提供 `ProxyBackend`。两份 `do_proxy` 实现刻意做到逐字节一致（`session_server.py:47` vs `session_worker.py:76`）。

于是每种部署形态都只是一个薄 adapter：

- **单进程** —— `sessions.setup_session_routes`（`sessions.py:37`）注册 FastAPI 路由，把请求读成 primitives，调对应的 `SessionCore` 方法，再用 `_to_starlette_response` 把 `CoreResponse` 渲染成 `starlette.Response`。
- **多进程** —— `SessionWorker._dispatch`（`session_worker.py:233`）把一个 op envelope 解码成同样的 primitives，调同样的 `SessionCore` 方法，把同样的 `CoreResponse` 经 IPC 编码回去。

因为两个 adapter 都收敛到同一个 `SessionCore`，单进程与多进程的行为是 by construction 一致的——状态码、headers、32-hex 的 `session_id` 形状、in-flight gate、upstream-body 透传，全都活在 core 里而不在 adapter 里。这正是 `test_session_multiprocess_equivalence.py` 断言的性质。

## 3. 宏观架构（多进程模式）

```
caller process (rollout / Ray actor)
└── SessionServerSupervisor            session_supervisor.py
      │  spawn 出子进程（multiprocessing "spawn"），每 worker 一对 socketpair
      │
      ├── router process                session_router.py  （唯一的 HTTP listener）
      │     SessionRouter
      │       client IpcChannel ──┐ （每 worker 一个）
      │                           │
      ├── worker 0  ◀── socketpair ┤   session_worker.py
      │     SessionWorker          │     SessionCore + SessionRegistry + tokenizer + httpx
      │                           │
      ├── worker 1  ◀── socketpair ┤
      │     …                      │
      └── worker N-1 ◀─ socketpair ┘
```

三个角色，职责严格分离：

- **Router**（`session_router.py`）—— 唯一绑定面向客户端 TCP 端口的进程。它**不持有** session 状态、**不持有** tokenizer。它每个 worker 持一个 client 侧 `IpcChannel`，把每个请求的 `session_id` 映射到其归属 worker，经 IPC 发一个 op envelope，再把 worker 的 `CoreResponse` 渲染回客户端，*而不重新 parse body*（chat 与 GET-records 的 body 都作为不透明字节中转）。
- **Worker**（`session_worker.py`）—— headless，拥有一个 session 分片。跑自己的 asyncio loop、一份私有的 `SessionRegistry` + tokenizer，以及一个最小的 `ProxyBackend`（只有 `do_proxy` + httpx）。它解码 op、驱动自己的 `SessionCore`、把 `CoreResponse` 发回。worker 之间从不互相通信。
- **Supervisor**（`session_supervisor.py`）—— 活在 caller 进程里。spawn 子进程、连好 socket、阻塞直到所有 worker 就绪，然后监控以做 fail-fast。不持有任何 session 状态。

入口与模式选择在 `router_manager.start_session_server`（`miles/ray/rollout/router_manager.py:88`）：`workers > 1` 构造一个 `SessionServerSupervisor` 并阻塞在 `supervisor.start()`；否则回退到单进程的 `run_session_server`。supervisor 实例被停在一个模块级 list 里（`_SESSION_SUPERVISORS`，`router_manager.py:20`），以免 rollout 还在跑时 GC 触发它的 `atexit` teardown。

接下来四节按依赖顺序走一遍决策：请求如何找到它的 worker（§4）、字节如何在 router 与 worker 间流动（§5）、并发与背压如何跨这次拆分被保住（§6）、进程组如何起与如何死（§7）。

## 4. 进程稳定的路由

sticky ownership —— 单进程 server 用「一个进程」买来的那个不变量 —— 改用一个确定性哈希重新建立。`routing.py` 就是全部契约，纯 stdlib，使得 headless worker 或 router 不依赖 FastAPI 即可 import：

- `new_session_id()` mint 一个 32-char hex id（`uuid4().hex`）。
- `worker_index_for_session(session_id, n_worker)` 通过 `blake2b(session_id) % n_worker` 把 id 映射到归属者。

非显然的选择是**用 `blake2b`，而非内置 `hash()`**（`routing.py:5-8`、`:33`）。`hash()` 被 `PYTHONHASHSEED` 加盐，因此逐进程不同——在 `spawn` context 下 router 和每个 worker 是各自独立的解释器，会对归属产生分歧。这个映射必须跨进程*且*跨 run 完全一致，所以用一个稳定的密码学摘要。

归属握手：

- **Create** —— router 自己 mint id 并算出它的归属者（`SessionRouter.create_session`，`session_router.py:165`），再把 `OP_CREATE_ID` 派发给那个 worker，于是 session 在 *router 选定*的 id 下创建，而非 worker 自选。worker 侧的 `SessionCore.create_session_with_id`（`session_core.py:261`）校验 32-hex 形状（畸形 → 400），并拒绝覆盖已存在的 id（碰撞 → 409，例如丢 ack 的重试）——两者都不会变成无映射的 500。（`OP_CREATE`，那个无参变体，在 `session_worker.py:138`，仅为完整性保留，router 从不使用；create 一律走 `OP_CREATE_ID`。）
- **之后每个 op** —— router 从同一个 `session_id` 重新推出归属者（`_dispatch_session`，`session_router.py:160`），发给那个已持有该 session 的 worker。没有目录表，没有跨 worker 查找。

因为路由是 `(session_id, n_worker)` 的纯函数，`n_worker` 在 server 生命周期内固定——中途改它会把已有 id 重路由到错误的 worker。见 §8。

## 5. IPC 传输

`session_ipc.py` 是一条单连接 socket（`socketpair`）上的分帧、多路复用 request/reply RPC。router 持一端，worker 持另一端；router 侧 `IpcChannel.request(payload)` await reply 字节，worker 侧设一个 `request_handler` 协程。纯 stdlib。

它的设计被一个事实主导：**reply body 可达 100+ MiB**（那个触发整次重构的 GET-records 载荷）。一个朴素的「长度前缀 socket」会让这一条 reply 独占整条流，把重构本想消掉的 head-of-line 停顿原样搬回来——这次是在 socket 上而非 event loop 上。帧协议的存在就是为防住它：

- **Wire frame** —— `[length:u32][request_id:u64][frame_type:u8][flags:u8][payload]`（`session_ipc.py:13-23`）。一条逻辑消息被切成 `MAX_CHUNK_SIZE`（1 MiB）的 chunk，打同一个 `request_id` 标记；最后一个置 `FLAG_LAST`。
- **每 socket 单 writer** —— 所有发送都汇流到一个 `_writer_loop`（`session_ipc.py:290`），从内存队列出帧，因此并发消息的字节绝不交错。
- **Round-robin，无 HOL 阻塞** —— writer 每轮对每个 active body *只发一个 chunk*（`session_ipc.py:304-310`）。大 body 和小 reply 一起推进，小 reply 不会卡在大 body 后面。微小的 control frame（错误）走一条优先 FIFO 通道，排在数据 body 之前。
- **不做第二份拷贝 / 发送背压** —— 一个出站 body 注册为 *引用 + 游标*（`_OutboundBody`，`session_ipc.py:112`），只有当 writer 即将发送时才物化出帧，于是队列里从不持有 100+ MiB body 的副本。当在途发送字节总量将超过 `DEFAULT_MAX_SEND_BUFFER_BYTES`（256 MiB）时，注册会 await，从而在 peer 卡住时也把内存兜住。
- **体积上限** —— 帧长超过 `MAX_FRAME_SIZE`（损坏的 length）或重组 body 超过 `max_body_size`（512 MiB）会确定性失败（`_Reassembler`，`session_ipc.py:88`）；没有无界缓冲。
- **确定性 teardown / fail-fast** —— EOF、半截帧、或损坏的 length 会把 channel 恰好拆一次（`_teardown`，`session_ipc.py:413`）：每个 pending request future 都被 `IpcChannelClosed` 置失败，并触发 `on_close` 回调，让 owner 能全局 fail-fast。对未知 / 已结清 `request_id` 的迟到或被弃 reply 会被干净丢弃（`_settle`，`session_ipc.py:376`），而不是抛异常把 reader 弄死。

op 与 `CoreResponse` 载荷本身用 `encode_envelope` / `decode_envelope`（`session_ipc.py:465`）分帧：`[meta_len:u32][meta_json][raw_body]`。小的 metadata（op 字段、status/headers/media-type）作为 JSON 骑在**未改动**的 body 字节前面——刻意*不*用 base64，于是 100+ MiB 的 body 零膨胀地过线（`session_worker.py:114`）。这也正是 router 能不透明中转 body 的原因：它根本不需要解码 body。

## 6. 并发模型 —— 是保住，不是重造

每个 session 的正确性规则没有为多进程重写；它们留在 `SessionCore` 里，因此两种模式下完全成立。多进程加上的是绕在它们外面的第二层、*进程局部*的准入层。

在 core 内部，一轮 chat 在 `session.lock`（每个 `LinearTrajectory` 一把 `asyncio.Lock`）下触碰三段短临界区，而慢的 upstream 调用持在锁*外*：

1. **抢 in-flight 槽**，在锁内（`session_core.py:384`）：`closing` → 404 压过 `chat_inflight` → 409。`chat_inflight` 标志强制*每 session 同时只有一轮 in-flight chat*；同 session 的第二个并发 chat 直接 409 fast-fail，根本不进 backend。
2. **准备 pretokenized prompt**，在锁内（`session_core.py:415`）—— 改 trajectory 状态（rollback 检测、prefix 合并）。
3. 为 upstream `do_proxy` await 释放锁 —— 这是最长的一段。在这里释放，正是让并发的 `DELETE` 能拿到锁推进下去（commit `0977e6ee0`，"Split session lock: release during proxy to unblock DELETE"）。
4. **提交状态**，在锁内（`session_core.py:447`），由一个不变量检查守住：`num_assistant` 在 gate 下没有移动（移动了就 `SessionInvariantError`——本应不可达）。

槽位在 `finally` 里以一次普通写清掉（`session_core.py:480`）；在单线程 loop 上这是原子的，并且即使被 cancel 也会执行。

仅 worker 才有的附加项，全都被设计成*不削弱*上面这套：

- **`parse_gate`**（`session_worker.py:198`，`_parse_sem`）—— 一个进程局部的 `asyncio.Semaphore`（默认 2），用于约束*并发 CPU parse/validate*，使 N 个重叠的大请求不会把 worker 内存撑爆。它被传进 `SessionCore.chat_completions` / `get_session`，并且**只在抢到 in-flight 槽之后**才进入（于是同 session 的竞争者仍然在任何 gate 等待之前就 409），且**绝不**在 `session.lock` 持有时进入（`session_core.py:349-360`）。单进程路径传 `parse_gate=None`（不做 gating）。这是 core 唯一吃一个多进程形状参数的地方，并且被小心地穿插好，使单进程行为不变。
- **每个入站 IPC 请求作为各自的 asyncio task 跑**（`_spawn_handler`，`session_ipc.py:388`），于是同一个 worker 内不同 session 的 upstream await 互相重叠——就是单进程 loop 本来有的那种重叠，现在按 worker 复现。
- **每 worker 背压**（`SessionWorker.handle`，`session_worker.py:209`）—— 在做任何工作*之前*准入检查：若 in-flight 计数 ≥ `max_inflight`（256），或排队字节将超过 `max_queued_bytes`（256 MiB），直接回 503，不碰 session 状态。

背压与取消也活在 **router** 前门（`session_router.py`）：

- **每 worker in-flight 上限**（`DEFAULT_ROUTER_MAX_INFLIGHT` = 512）：饱和的 worker 在 task 还没启动前就 503 fast-fail（`_dispatch`，`session_router.py:134`），于是 router 的 future 不会无界增长。
- **断连绝不能取消 worker 在跑的 chat。** IPC 请求作为一个 *router 自有* task 跑（`_spawn_ipc_request`，`session_router.py:112`），HTTP handler 在 `asyncio.shield` 下 await 它。客户端断连取消的是 handler 而非 IPC task，于是 worker 仍会把 reply 排干、并且仍可能提交（这与 race-test 的语义一致）。in-flight 槽在 task 的 **done-callback** 里释放——当 worker 的 reply 真正到达时——而不是在 handler 取消时，于是一个已断连的请求在它仍消耗 worker/httpx 资源期间继续计入上限，而不是把计数泄掉、或过早释放资源。
- **health 同时就是 readiness 信号**（`all_workers_healthy`，`session_router.py:230`）：`/health` 以有界超时 ping 每个 worker，全部应答才算 healthy。supervisor 复用的正是这个端点作为它的 readiness gate。

每 worker 的 httpx 连接池上限是 `_PER_WORKER_MAX_CONNECTIONS` = 256（`session_worker.py:51`），刻意*不是*单进程的 1024，于是 N 个 worker 不会把 upstream 连接上限乘成 N×1024。

## 7. 生命周期与失败处理

`SessionServerSupervisor`（`session_supervisor.py`）从 caller 进程里拥有整个进程组。

**启动**（`start`，`session_supervisor.py:68`）：每 worker 建一对 `socket.socketpair()`；在 `multiprocessing` **spawn** context 下 spawn 出 N 个 worker（各自拿到自己的 worker-end socket + index）和一个 router（拿到全部 router-end socket，按 worker-index 顺序）。然后——关键一步——parent 关掉它自己持有的*每一个* socket 端（`session_supervisor.py:98-103`）：子进程已收到各自 dup 出来的副本，而 parent 两端都不持有，这才使 peer 死亡时 EOF 可被观测到（IPC teardown 和 fail-fast 的触发条件）。子进程是 `daemon=False`，因为 daemon 进程不能 spawn 子进程，而且 teardown 由 supervisor 自己负责。

**就绪**（`_await_ready`，`session_supervisor.py:117`）：先等 router 的 TCP 端口，再轮询 router 的 `/health` 直到它报告所有 worker healthy，预算 600 s，这段也覆盖每个 worker 的 tokenizer/TITO 初始化。任何子进程在启动期间死掉都会中止等待。

**子进程死亡即 fail-fast**（`_monitor_loop`，`session_supervisor.py:162`）：一个后台线程轮询所有子进程的 `is_alive()`；*任一*死亡就记录 `self._failure` 并杀掉整组。这刻意*不是*优雅降级——一个半死的 server（缺一个分片）服务不了哈希到那个死 worker 上的 session，所以安全做法是把整组拿下并把失败暴露出去。因为线程无法 `raise` 进主 rollout 路径，失败通过 `check()`（`session_supervisor.py:180`）暴露，由 rollout 轮询；后续的 launch 也会观测到。（据项目 memory：一个 worker 在某 run 大约第 248 轮死掉，是 harness keep-alive 级联触发了这个 by-design 的 fail-fast，并非 worker 缺陷。）

**无孤儿**，纵深防御：

- 每个子进程调 `_set_pdeathsig()`（`session_worker.py:279`）—— Linux 上 `prctl(PR_SET_PDEATHSIG, SIGKILL)` —— 于是 parent 没干净 teardown 就崩了，子进程也会被回收。
- `shutdown()`（`session_supervisor.py:218`）幂等：对所有子进程 SIGTERM，以 `_TERM_GRACE`（5 s）期限 join，再对赖着的 SIGKILL。它接到 `atexit` 和 SIGTERM/SIGINT 上。
- 信号 handler（`_install_signal_handlers`，`session_supervisor.py:191`）**链回原 disposition**，使信号对 rollout 进程仍有其本来效果：原来的 callable 会跑；`SIG_DFL` 被恢复并重新发出，使进程真的死掉。没有这一步，一个 cluster/Ray 的 SIGTERM 会收掉子进程却让 rollout 进程活着。handler 只在主线程安装；Ray-actor-thread 启动则回退到 `atexit` + pdeathsig。

## 8. 边界、约束、以及容易改坏的地方

- **`n_worker` 在 launch 时固定。** 路由是 `blake2b(session_id) % n_worker`；改 worker 数会把已有 id 重路由到非归属 worker。没有 rebalancing，没有 session 迁移。死掉的 worker 其分片不会被接管——整组 fail-fast（§7）。
- **sticky ownership 是绝对的。** 一个 session 终其一生只在一个 worker 上。任何假设 session 状态在 worker 间共享的逻辑都是 by construction 错的。
- **失败模型是 fail-fast，不是降级。** 任一子进程死亡都拆掉整组。别在没同时解决分片恢复的前提下加「重启死 worker」——重启的 worker registry 是空的，答不了它死前创建的 session。
- **transport-neutral 边界是承重的。** `SessionCore` 必须保持不 import `fastapi`/`starlette`/ASGI，并保持在内部映射 `SessionError`；`routing.py` 和 `session_ipc.py` 必须保持纯 stdlib。把 FastAPI 依赖拉进它们任何一个，都会破坏 headless worker 的 import，以及 equivalence 测试所依赖的「行为单一来源」保证。
- **`parse_gate` 纪律。** 它绝不能在 `session.lock` 下被 await，且只能在抢到 in-flight 槽之后进入。任一处反转都会破坏「同 session 第二个 chat 快速 409」的保证，或带来持锁 CPU 停顿的风险。
- **可调项是代码默认值，不是 CLI flag。** `arguments.py` 里只定义了 `--use-session-server`、`--session-server-ip`、`--session-server-port`、`--session-server-workers` 和 `--tito-*`（`miles/utils/arguments.py:1769`）。背压 / parse-concurrency / health-timeout / 连接数这些旋钮（`session_worker_parse_concurrency`、`session_worker_max_inflight`、`session_worker_max_queued_bytes`、`session_worker_max_connections`、`session_router_health_timeout`、`miles_router_timeout`）都是 `getattr(args, ..., DEFAULT)` 读的，只有当有谁把它们挂到 args namespace 上才生效——目前没有命令行入口。
- **继承自领域层的约束**（来自 `linear_trajectory.py`，本分支未改）：`generate_multi_samples` 必须为 False（`SessionRegistry.__init__` 断言它，`linear_trajectory.py:264`），因为各轮通过 `merge_samples` last-wins 折叠；rollback 上限是一个 assistant step（`MAX_ASSISTANT_ROLLBACK_STEPS = 1`）；追加的 message role 必须在 `--tito-allowed-append-roles` 里。
- **record 体积优化（正交，同分支）。** `routed_experts` / `indexer_topk`（即 "R3" blob）在两种模式下都从面向客户端的 chat body 里剥掉（`_client_chat_response`，`session_core.py:128`），因为它们从 `GET /sessions/{id}` 重建；而一条更旧 record 的 blob 一旦再也回滚不到它就被丢弃（`append_record`，`linear_trajectory.py:52`），把保留体积维持在 O(prefix)。这与多进程工作相互独立，只是搭同一分支上来；它顺带也让 IPC reply body 更小。

## 关于出处的说明

若干模块 docstring 引用了一个 "m3-design-contract"（如 `session_ipc.py:25`、`session_worker.py:10`、`session_router.py:11`）。那份 contract 是重构期间的工作文档，并未入库；它捕获的不变量已内联进代码注释，本文把它们汇总在一起。本文与某条注释冲突时，以代码为准。
