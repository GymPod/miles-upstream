"""``RolloutManager`` cell dispatch + ``EnginesAndLock`` flow driven through
the production ``RolloutManager`` class.

We instantiate ``RolloutManager.__ray_actor_class__`` directly (the raw Python
class behind ``@ray.remote``) — that keeps the manager in the test process so
``monkeypatch`` reaches its dependencies, while the engines it spawns are
still real Ray actors (mocks). Methods are ``async`` and called with ``await``.

Pure routing/flag-flip helpers without Ray content live in
``tests/fast/ray/rollout/test_rollout_manager.py``."""

from __future__ import annotations

import asyncio
import textwrap
import time

import pytest
import ray
from tests.fast.ray.rollout.conftest import make_args, make_samples_grouped

from miles.ray.rollout.rollout_manager import RolloutManager
from miles.rollout.base_types import RolloutFnConstructorInput, RolloutFnEvalOutput, RolloutFnTrainOutput
from miles.rollout.rollout_session import (
    BatchRollbackReason,
    BatchRollbackUnsupportedError,
    RolloutSession,
    TrainBatchLease,
)


class _RollbackFailure(BaseException):
    pass


class _RecordingTrainBatchLease(TrainBatchLease):
    def __init__(
        self,
        *,
        rollout_id: int,
        output: RolloutFnTrainOutput,
        events: list[str],
        commit_error: BaseException | None,
        rollback_error: BaseException | None,
    ) -> None:
        super().__init__(rollout_id=rollout_id, output=output)
        self._events = events
        self._commit_error = commit_error
        self._rollback_error = rollback_error

    def _commit(self) -> None:
        self._events.append(f"commit:{self.rollout_id}")
        if self._commit_error is not None:
            raise self._commit_error

    def _rollback(self, reason: BatchRollbackReason) -> None:
        self._events.append(f"rollback:{reason.name}")
        if self._rollback_error is not None:
            raise self._rollback_error


class _CancelAfterCommitLease(_RecordingTrainBatchLease):
    def _commit(self) -> None:
        super()._commit()
        task = asyncio.current_task()
        assert task is not None
        asyncio.get_running_loop().call_soon(task.cancel)


class _RecordingRolloutSession(RolloutSession):
    def __init__(
        self,
        *,
        lease: TrainBatchLease | None,
        eval_output: RolloutFnEvalOutput,
        events: list[str],
    ) -> None:
        self._lease = lease
        self._eval_output = eval_output
        self._events = events
        self.train_loop: asyncio.AbstractEventLoop | None = None
        self.eval_loop: asyncio.AbstractEventLoop | None = None

    async def acquire_train_batch(self, rollout_id: int) -> TrainBatchLease:
        self.train_loop = asyncio.get_running_loop()
        self._events.append(f"acquire:{rollout_id}")
        if self._lease is None:
            raise AssertionError("Test session has no train batch lease.")
        return self._lease

    async def evaluate(self, rollout_id: int) -> RolloutFnEvalOutput:
        self.eval_loop = asyncio.get_running_loop()
        self._events.append(f"evaluate:{rollout_id}")
        return self._eval_output

    async def prepare_checkpoint(self, rollout_id: int) -> None:
        self._events.append(f"prepare_checkpoint:{rollout_id}")

    async def close(self) -> None:
        self._events.append("session_close")


class _BlockingCloseRolloutSession(_RecordingRolloutSession):
    def __init__(
        self,
        *,
        events: list[str],
        close_started: asyncio.Event,
        allow_close: asyncio.Event,
    ) -> None:
        super().__init__(
            lease=None,
            eval_output=RolloutFnEvalOutput(data={}, metrics=None),
            events=events,
        )
        self._close_started = close_started
        self._allow_close = allow_close

    async def close(self) -> None:
        self._events.append("session_close_started")
        self._close_started.set()
        await self._allow_close.wait()
        self._events.append("session_close_finished")


class _FailingOnceCloseRolloutSession(_RecordingRolloutSession):
    def __init__(self, *, events: list[str], close_error: BaseException) -> None:
        super().__init__(
            lease=None,
            eval_output=RolloutFnEvalOutput(data={}, metrics=None),
            events=events,
        )
        self._close_error: BaseException | None = close_error

    async def close(self) -> None:
        if self._close_error is not None:
            close_error = self._close_error
            self._close_error = None
            self._events.append("session_close_failed")
            raise close_error
        self._events.append("session_close_succeeded")


class _RecordingCloseableDataSource:
    def __init__(self, events: list[str]) -> None:
        self._events = events

    def close(self) -> None:
        self._events.append("data_source_close")


class _FailingCloseableDataSource(_RecordingCloseableDataSource):
    def __init__(self, events: list[str], close_error: BaseException) -> None:
        super().__init__(events)
        self._close_error = close_error

    def close(self) -> None:
        super().close()
        raise self._close_error


class _RecordingDisposable:
    def __init__(self, events: list[str], event: str) -> None:
        self._events = events
        self._event = event

    def dispose(self) -> None:
        self._events.append(self._event)


class _RecordingHealthMonitor:
    def __init__(self, events: list[str], event: str) -> None:
        self._events = events
        self._event = event

    def stop(self) -> None:
        self._events.append(self._event)


@pytest.fixture
def patch_low_level(monkeypatch):
    """Replace, in the test process:
    - ``SGLangEngine`` → ``MockSGLangEngine`` so created actors are mocks.
    - addr allocator → deterministic stub.
    - ``init_tracking`` / ``init_http_client`` / ``start_session_server`` /
      ``load_function`` / ``load_rollout_session`` → no-ops (the production
      defaults touch wandb / network / not-importable default function paths)."""
    import miles.ray.rollout.rollout_manager as rmgr
    import miles.ray.rollout.rollout_server as rsrv
    import miles.ray.rollout.server_group as sg
    from miles.ray.rollout.addr_allocator import PortCursors
    from miles.utils.test_utils.mock_sglang_engine import MockSGLangEngine

    monkeypatch.setattr(sg, "SGLangEngine", MockSGLangEngine.__ray_actor_class__)
    # multi-model tests would otherwise spawn a real router subprocess for
    # ``model_idx > 0`` (force_new=True bypasses the args.sglang_router_ip cache).
    monkeypatch.setattr(
        rsrv,
        "start_router",
        lambda args, **kw: (args.sglang_router_ip, args.sglang_router_port),
    )

    def _fake_alloc(*args, **kwargs):
        engines = kwargs["rollout_engines"]
        return (
            {
                rank: dict(
                    host="127.0.0.1",
                    port=30000 + rank,
                    nccl_port=31000 + rank,
                    engine_info_bootstrap_port=32000 + rank,
                    dist_init_addr=f"127.0.0.1:{33000 + rank}",
                )
                for rank, _ in engines
            },
            PortCursors(_values={0: 34000}),
        )

    monkeypatch.setattr(sg, "allocate_rollout_engine_addr_and_ports_normal", _fake_alloc)
    monkeypatch.setattr(rmgr, "init_tracking", lambda *a, **kw: None)
    monkeypatch.setattr(rmgr, "init_http_client", lambda args: None)
    monkeypatch.setattr(rmgr, "start_session_server", lambda args: None)
    monkeypatch.setattr(rmgr, "load_function", lambda path: lambda *a, **kw: None)
    monkeypatch.setattr(
        rmgr,
        "load_rollout_session",
        lambda *a, **kw: _RecordingRolloutSession(
            lease=None,
            eval_output=RolloutFnEvalOutput(data={}, metrics=None),
            events=[],
        ),
    )
    # generate()/eval() drive these — production hits wandb / tensorboard.
    monkeypatch.setattr(rmgr, "log_rollout_data", lambda *a, **kw: None)
    monkeypatch.setattr(rmgr, "log_eval_rollout_data", lambda *a, **kw: None)
    monkeypatch.setattr(rmgr, "save_debug_rollout_data", lambda *a, **kw: None)


def _make_manager(args, pg):
    return RolloutManager.__ray_actor_class__(args, pg)


def _make_train_output() -> RolloutFnTrainOutput:
    return RolloutFnTrainOutput(
        samples=[make_samples_grouped(n_groups=2, group_size=4)],
        metrics={"my_metric": 1.23},
    )


def _write_sglang_config(tmp_path, *, models: list[tuple[str, bool]]) -> str:
    """Write a multi-model sglang yaml — each entry ``(name, update_weights)``.
    Each model gets one regular group with 2 engines × 1 GPU = 2 GPUs. With N
    models, total GPUs = 2N; ``args.rollout_num_gpus`` must match."""
    lines = ["sglang:"]
    for name, update_weights in models:
        lines.extend(
            [
                f"  - name: {name}",
                f"    update_weights: {str(update_weights).lower()}",
                "    server_groups:",
                "      - worker_type: regular",
                "        num_gpus: 2",
                "        num_gpus_per_engine: 1",
            ]
        )
    cfg_path = tmp_path / "sglang.yaml"
    cfg_path.write_text(textwrap.dedent("\n".join(lines)) + "\n")
    return str(cfg_path)


def _make_test_args(tmp_path, *, models: list[tuple[str, bool]]):
    """Build args that drive ``RolloutManager.__init__`` →
    ``start_rollout_servers`` → N model servers each with 1 group of 2 mock
    engines."""
    cfg = _write_sglang_config(tmp_path, models=models)
    rollout_num_gpus = 2 * len(models)
    return make_args(
        sglang_config=cfg,
        rollout_num_gpus=rollout_num_gpus,
        # short-circuit start_router (returns early when ip+port already set)
        sglang_router_ip="127.0.0.1",
        sglang_router_port=30000,
        # disable everything else that would spawn subprocesses or hit network
        use_session_server=False,
        use_fault_tolerance=False,
        use_wandb=False,
        use_tensorboard=False,
        use_mlflow=False,
        use_distributed_post=False,
        sglang_server_concurrency=1,
    )


async def _assert_engine_dies(actor_handle, *, deadline_s: float = 15.0, poll_interval_s: float = 0.2) -> None:
    deadline = time.monotonic() + deadline_s
    while True:
        try:
            ray.get(actor_handle.health_generate.remote(timeout=1.0), timeout=5.0)
        except (ray.exceptions.RayActorError, ray.exceptions.RayTaskError):
            return
        except ray.exceptions.GetTimeoutError:
            pass
        if time.monotonic() >= deadline:
            pytest.fail(f"engine actor still alive {deadline_s}s after stop_cell")
        await asyncio.sleep(poll_interval_s)


@pytest.mark.asyncio
class TestRolloutManagerInit:
    async def test_init_constructs_one_rollout_session_with_train_and_eval_paths(
        self,
        ray_local_mode,
        placement_group_factory,
        tmp_path,
        patch_low_level,
        monkeypatch,
    ):
        import miles.ray.rollout.rollout_manager as rmgr

        args = _make_test_args(tmp_path, models=[("actor", True)])
        args.debug_train_only = True
        pg = placement_group_factory(2)
        events: list[str] = []
        session = _RecordingRolloutSession(
            lease=None,
            eval_output=RolloutFnEvalOutput(data={}, metrics=None),
            events=events,
        )
        captured: list[tuple[RolloutFnConstructorInput, str, str]] = []

        def fake_load_rollout_session(
            constructor_input: RolloutFnConstructorInput,
            *,
            train_path: str,
            eval_path: str,
        ) -> RolloutSession:
            captured.append((constructor_input, train_path, eval_path))
            return session

        monkeypatch.setattr(rmgr, "load_rollout_session", fake_load_rollout_session)

        manager = _make_manager(args, pg)

        assert manager.rollout_session is session
        assert captured == [
            (
                RolloutFnConstructorInput(args=args, data_source=manager.data_source),
                args.rollout_function_path,
                args.eval_function_path,
            )
        ]

    async def test_init_creates_live_mock_engines_via_real_start_rollout_servers(
        self,
        ray_local_mode,
        placement_group_factory,
        tmp_path,
        patch_low_level,
    ):
        """End-to-end smoke: production ``__init__`` + ``start_rollout_servers``
        runs against MockSGLangEngine; resulting engines are reachable as Ray
        actor handles via the public ``get_updatable_engines_and_lock``."""
        args = _make_test_args(tmp_path, models=[("actor", True)])
        pg = placement_group_factory(2)

        manager = _make_manager(args, pg)
        eal = await manager.get_updatable_engines_and_lock()
        assert len(eal.rollout_engines) == 2
        for h in eal.rollout_engines:
            assert isinstance(h, ray.actor.ActorHandle)
            assert ray.get(h.health_generate.remote(timeout=1.0)) is True


@pytest.mark.asyncio
class TestStartStopCell:
    async def test_stop_cell_kills_target_engine_only(
        self,
        ray_local_mode,
        placement_group_factory,
        tmp_path,
        patch_low_level,
    ):
        """``stop_cell(0)`` kills cell 0's actor; cell 1 untouched."""
        args = _make_test_args(tmp_path, models=[("actor", True)])
        pg = placement_group_factory(2)

        manager = _make_manager(args, pg)
        eal = await manager.get_updatable_engines_and_lock()
        actor0, actor1 = eal.rollout_engines

        await manager.stop_cell(0)

        await _assert_engine_dies(actor0)
        assert ray.get(actor1.health_generate.remote(timeout=1.0)) is True

    async def test_start_cell_recovers_after_stop_cell(
        self,
        ray_local_mode,
        placement_group_factory,
        tmp_path,
        patch_low_level,
    ):
        """stop_cell(0) → start_cell(0) drives a real ``recover()`` that spawns
        a fresh mock actor in place of the killed one."""
        args = _make_test_args(tmp_path, models=[("actor", True)])
        pg = placement_group_factory(2)

        manager = _make_manager(args, pg)
        eal_before = await manager.get_updatable_engines_and_lock()
        actor0_before = eal_before.rollout_engines[0]

        await manager.stop_cell(0)
        await manager.start_cell(0)

        eal_after = await manager.get_updatable_engines_and_lock()
        actor0_after = eal_after.rollout_engines[0]

        assert actor0_after is not actor0_before, "start_cell must produce a fresh actor"
        assert ray.get(actor0_after.health_generate.remote(timeout=1.0)) is True

    async def test_stop_cell_targets_high_id_correctly(
        self,
        ray_local_mode,
        placement_group_factory,
        tmp_path,
        patch_low_level,
    ):
        """``stop_cell(1)`` (not 0) must kill engine 1, leaving engine 0 alive —
        guards against off-by-one in ``get_cell_indexer_of_id_map``."""
        args = _make_test_args(tmp_path, models=[("actor", True)])
        pg = placement_group_factory(2)

        manager = _make_manager(args, pg)
        eal = await manager.get_updatable_engines_and_lock()
        actor0, actor1 = eal.rollout_engines

        await manager.stop_cell(1)

        assert ray.get(actor0.health_generate.remote(timeout=1.0)) is True
        await _assert_engine_dies(actor1)

    async def test_stop_cell_is_idempotent_on_already_stopped(
        self,
        ray_local_mode,
        placement_group_factory,
        tmp_path,
        patch_low_level,
    ):
        """Calling ``stop_cell(0)`` twice does not raise — production code logs
        and proceeds when the engine is already de-allocated."""
        args = _make_test_args(tmp_path, models=[("actor", True)])
        pg = placement_group_factory(2)

        manager = _make_manager(args, pg)
        await manager.get_updatable_engines_and_lock()  # ensure engines are alive

        await manager.stop_cell(0)
        await manager.stop_cell(0)  # must not raise


@pytest.mark.asyncio
class TestCellDispatchAcrossModels:
    async def test_cells_route_to_correct_model_by_sorted_srv_key(
        self,
        ray_local_mode,
        placement_group_factory,
        tmp_path,
        patch_low_level,
    ):
        """Cells are flattened in sorted-srv-key order: with models ("actor",
        "ref") the cells map (0,1)→actor, (2,3)→ref. Stopping cell 2 must hit
        ref's first engine and leave actor's engines untouched."""
        args = _make_test_args(tmp_path, models=[("actor", True), ("ref", False)])
        pg = placement_group_factory(4)

        manager = _make_manager(args, pg)
        actor_handles = [e.actor_handle for e in manager.servers["actor"].server_groups[0].engines]
        ref_handles = [e.actor_handle for e in manager.servers["ref"].server_groups[0].engines]

        await manager.stop_cell(2)

        # actor untouched
        for h in actor_handles:
            assert ray.get(h.health_generate.remote(timeout=1.0)) is True
        # ref engine 0 dead, ref engine 1 alive
        await _assert_engine_dies(ref_handles[0])
        assert ray.get(ref_handles[1].health_generate.remote(timeout=1.0)) is True


@pytest.mark.asyncio
class TestGetUpdatableEnginesAndLock:
    async def test_returns_only_updatable_servers_engines_in_multi_model_setup(
        self,
        ray_local_mode,
        placement_group_factory,
        tmp_path,
        patch_low_level,
    ):
        """With actor (update_weights=True) + ref (update_weights=False), the
        returned EnginesAndLock contains the actor's engines only."""
        args = _make_test_args(tmp_path, models=[("actor", True), ("ref", False)])
        pg = placement_group_factory(4)

        manager = _make_manager(args, pg)
        eal = await manager.get_updatable_engines_and_lock()
        assert len(eal.rollout_engines) == 2  # actor's 2, not ref's 2
        assert eal.engine_gpu_counts == [1, 1]
        assert all(isinstance(h, ray.actor.ActorHandle) for h in eal.rollout_engines)
        assert ray.get(eal.rollout_engines[0].health_generate.remote(timeout=1.0)) is True

    async def test_returns_empty_when_no_updatable_model(
        self,
        ray_local_mode,
        placement_group_factory,
        tmp_path,
        patch_low_level,
    ):
        """If every model has ``update_weights=False`` (e.g. inference-only
        deployment), the returned EnginesAndLock has empty engines list and
        the lock handle is still present (callers always need a lock)."""
        args = _make_test_args(tmp_path, models=[("ref", False)])
        pg = placement_group_factory(2)

        manager = _make_manager(args, pg)
        eal = await manager.get_updatable_engines_and_lock()
        assert eal.rollout_engines == []
        assert eal.engine_gpu_counts == []
        assert eal.has_new_engines is False
        assert eal.rollout_engine_lock is not None

    async def test_has_new_engines_flag_lifecycle(
        self,
        ray_local_mode,
        placement_group_factory,
        tmp_path,
        patch_low_level,
    ):
        """Lifecycle the trainer relies on: ``has_new_engines`` is True after
        init, False after ``clear_updatable_has_new_engines``, True again
        after ``start_cell`` spawns a fresh engine."""
        args = _make_test_args(tmp_path, models=[("actor", True)])
        pg = placement_group_factory(2)

        manager = _make_manager(args, pg)
        eal_init = await manager.get_updatable_engines_and_lock()
        assert eal_init.has_new_engines is True

        manager.clear_updatable_has_new_engines()
        eal_cleared = await manager.get_updatable_engines_and_lock()
        assert eal_cleared.has_new_engines is False

        await manager.stop_cell(0)
        await manager.start_cell(0)
        eal_recovered = await manager.get_updatable_engines_and_lock()
        assert eal_recovered.has_new_engines is True

    async def test_clear_does_not_affect_non_updatable_server(
        self,
        ray_local_mode,
        placement_group_factory,
        tmp_path,
        patch_low_level,
    ):
        """``clear_updatable_has_new_engines`` must touch only the updatable
        server's flag; non-updatable (ref) servers keep their flag intact."""
        args = _make_test_args(tmp_path, models=[("actor", True), ("ref", False)])
        pg = placement_group_factory(4)

        manager = _make_manager(args, pg)
        # Force ref's flag True so we can detect any erroneous clear.
        manager.servers["ref"].server_groups[0].has_new_engines = True

        manager.clear_updatable_has_new_engines()

        assert manager.servers["ref"].server_groups[0].has_new_engines is True
        assert manager.servers["actor"].server_groups[0].has_new_engines is False

    async def test_multiple_updatable_servers_raises_assertion(
        self,
        ray_local_mode,
        placement_group_factory,
        tmp_path,
        patch_low_level,
    ):
        """Production guards against misconfiguration where two models both set
        ``update_weights=True``; that's ambiguous for the trainer."""
        args = _make_test_args(tmp_path, models=[("actor1", True), ("actor2", True)])
        pg = placement_group_factory(4)

        manager = _make_manager(args, pg)
        with pytest.raises(ValueError, match="Multiple servers"):
            await manager.get_updatable_engines_and_lock()


@pytest.mark.asyncio
class TestCheckWeights:
    async def test_check_weights_targets_only_updatable_model(
        self,
        ray_local_mode,
        placement_group_factory,
        tmp_path,
        patch_low_level,
    ):
        """``check_weights`` targets only the updatable model. The snapshot/reset/
        compare round-trip is meaningless for a frozen model (restored from disk,
        never re-synced via update_weights), so it must be skipped there."""
        args = _make_test_args(tmp_path, models=[("actor", True), ("ref", False)])
        pg = placement_group_factory(4)

        manager = _make_manager(args, pg)
        await manager.get_updatable_engines_and_lock()  # wait for engines to be alive

        results = await manager.check_weights(action="pre_update")

        # Updatable server only: nested gather is [group][engine]; 1 group × 2 engines.
        assert len(results) == 1
        for per_group in results:
            assert len(per_group) == 2
            for engine_result in per_group:
                assert engine_result == {"_mock": True}

        # Frozen (non-updatable) servers must not have been touched.
        for srv in manager.servers.values():
            if srv.update_weights:
                continue
            for group in srv.server_groups:
                for engine in group.engines:
                    if not engine.is_allocated:
                        continue
                    calls = ray.get(engine.actor_handle.get_calls.remote())
                    assert not any(c[0] == "check_weights" for c in calls)


@pytest.mark.asyncio
class TestRecoverUpdatableEngines:
    async def test_skips_recovery_when_no_rollout_started(
        self,
        ray_local_mode,
        placement_group_factory,
        tmp_path,
        patch_low_level,
    ):
        """``recover_updatable_engines`` is a no-op while ``rollout_id == -1``
        (initial state) — the trainer hasn't issued a rollout yet, so even if
        a slot looks dead the manager must not pre-emptively recover."""
        args = _make_test_args(tmp_path, models=[("actor", True)])
        pg = placement_group_factory(2)

        manager = _make_manager(args, pg)
        eal_before = await manager.get_updatable_engines_and_lock()
        actor0_before = eal_before.rollout_engines[0]

        # Kill engine 0 directly + mark stopped (simulates a fault before any
        # rollout). recover_updatable_engines must not bring it back yet.
        ray.kill(actor0_before)
        manager.servers["actor"].server_groups[0].all_engines[0].mark_stopped()

        await manager.recover_updatable_engines()

        # Slot 0 is still de-allocated; recovery skipped because rollout_id=-1.
        assert not manager.servers["actor"].server_groups[0].all_engines[0].is_allocated

    async def test_recovers_dead_engine_after_rollout_started(
        self,
        ray_local_mode,
        placement_group_factory,
        tmp_path,
        patch_low_level,
    ):
        """Once ``rollout_id`` advances past -1 (mid-training), a dead slot on
        the updatable server is brought back by ``recover_updatable_engines``."""
        args = _make_test_args(tmp_path, models=[("actor", True)])
        pg = placement_group_factory(2)

        manager = _make_manager(args, pg)
        eal_before = await manager.get_updatable_engines_and_lock()
        actor0_before = eal_before.rollout_engines[0]

        ray.kill(actor0_before)
        manager.servers["actor"].server_groups[0].all_engines[0].mark_stopped()

        manager.rollout_id = 0  # simulates "rollout has started"
        await manager.recover_updatable_engines()

        slot0 = manager.servers["actor"].server_groups[0].all_engines[0]
        assert slot0.is_allocated
        assert slot0.actor_handle is not actor0_before
        assert ray.get(slot0.actor_handle.health_generate.remote(timeout=1.0)) is True


@pytest.mark.asyncio
class TestGenerate:
    """``generate(rollout_id)`` is the trainer's per-iteration rollout entry
    point. It must retain the session's lease through postprocessing,
    conversion, and Ray publication before settling the handoff."""

    async def test_commits_session_lease_after_real_dp_publication(
        self,
        ray_local_mode,
        placement_group_factory,
        tmp_path,
        patch_low_level,
        monkeypatch,
    ):
        args = _make_test_args(tmp_path, models=[("actor", True)])
        # global_batch_size = number of samples we'll produce (postprocess
        # trims to a multiple, so equality avoids losing samples).
        args.global_batch_size = 8
        args.debug_train_only = True
        pg = placement_group_factory(2)

        manager = _make_manager(args, pg)
        manager.train_parallel_config = {"dp_size": 2}
        events: list[str] = []
        lease = _RecordingTrainBatchLease(
            rollout_id=42,
            output=_make_train_output(),
            events=events,
            commit_error=None,
            rollback_error=None,
        )
        session = _RecordingRolloutSession(
            lease=lease,
            eval_output=RolloutFnEvalOutput(data={}, metrics=None),
            events=events,
        )
        manager.rollout_session = session
        original_ray_put = ray.put

        def recording_ray_put(value):
            object_ref = original_ray_put(value)
            events.append("ray_put")
            return object_ref

        monkeypatch.setattr(ray, "put", recording_ray_put)

        result = await manager.generate(rollout_id=42)

        assert manager.rollout_id == 42
        assert session.train_loop is asyncio.get_running_loop()
        assert events == ["acquire:42", "ray_put", "ray_put", "commit:42"]
        # generate returns {"sample_indices": ..., "data_ref": ...};
        # split_train_data_by_dp returns Box(ObjectRef) per dp rank
        assert set(result) == {"sample_indices", "data_ref"}
        data_refs = result["data_ref"]
        assert len(data_refs) == 2
        assert all(isinstance(box.inner, ray.ObjectRef) for box in data_refs)
        partitions = ray.get([box.inner for box in data_refs])
        for partition in partitions:
            assert "tokens" in partition
            assert "rewards" in partition
            assert "loss_masks" in partition
            # 8 samples / 2 dp = 4 per rank
            assert len(partition["tokens"]) == 4

    @pytest.mark.parametrize(
        "rollback_failure",
        [
            pytest.param(
                BatchRollbackUnsupportedError(
                    rollout_id=17,
                    reason=BatchRollbackReason.HANDOFF_FAILED,
                ),
                id="unsupported",
            ),
            pytest.param(_RollbackFailure("rollback failed"), id="base-exception"),
        ],
    )
    async def test_rolls_back_output_access_and_preserves_error_if_rollback_fails(
        self,
        ray_local_mode,
        placement_group_factory,
        tmp_path,
        patch_low_level,
        rollback_failure: BaseException,
    ):
        class HandoffFailure(BaseException):
            pass

        failure = HandoffFailure("output access failed")

        class OutputFailureLease(_RecordingTrainBatchLease):
            @property
            def output(self) -> RolloutFnTrainOutput:
                raise failure

        args = _make_test_args(tmp_path, models=[("actor", True)])
        args.global_batch_size = 8
        args.debug_train_only = True
        pg = placement_group_factory(2)
        manager = _make_manager(args, pg)
        manager.train_parallel_config = {"dp_size": 2}
        events: list[str] = []
        lease = OutputFailureLease(
            rollout_id=17,
            output=_make_train_output(),
            events=events,
            commit_error=None,
            rollback_error=rollback_failure,
        )
        manager.rollout_session = _RecordingRolloutSession(
            lease=lease,
            eval_output=RolloutFnEvalOutput(data={}, metrics=None),
            events=events,
        )

        with pytest.raises(HandoffFailure) as exc_info:
            await manager.generate(rollout_id=17)

        assert exc_info.value is failure
        assert exc_info.value.__cause__ is rollback_failure
        assert events == ["acquire:17", "rollback:HANDOFF_FAILED"]

    async def test_rolls_back_when_publication_is_cancelled(
        self,
        ray_local_mode,
        placement_group_factory,
        tmp_path,
        patch_low_level,
        monkeypatch,
    ):
        args = _make_test_args(tmp_path, models=[("actor", True)])
        args.global_batch_size = 8
        args.debug_train_only = True
        pg = placement_group_factory(2)
        manager = _make_manager(args, pg)
        manager.train_parallel_config = {"dp_size": 2}
        events: list[str] = []
        lease = _RecordingTrainBatchLease(
            rollout_id=23,
            output=_make_train_output(),
            events=events,
            commit_error=None,
            rollback_error=None,
        )
        manager.rollout_session = _RecordingRolloutSession(
            lease=lease,
            eval_output=RolloutFnEvalOutput(data={}, metrics=None),
            events=events,
        )
        original_ray_put = ray.put
        put_count = 0

        def cancel_second_ray_put(value):
            nonlocal put_count
            put_count += 1
            events.append(f"ray_put:{put_count}")
            if put_count == 2:
                raise asyncio.CancelledError
            return original_ray_put(value)

        monkeypatch.setattr(ray, "put", cancel_second_ray_put)

        with pytest.raises(asyncio.CancelledError):
            await manager.generate(rollout_id=23)

        assert events == ["acquire:23", "ray_put:1", "ray_put:2", "rollback:HANDOFF_FAILED"]

    async def test_does_not_roll_back_failed_commit(
        self,
        ray_local_mode,
        placement_group_factory,
        tmp_path,
        patch_low_level,
    ):
        class CommitFailure(BaseException):
            pass

        args = _make_test_args(tmp_path, models=[("actor", True)])
        args.global_batch_size = 8
        args.debug_train_only = True
        pg = placement_group_factory(2)
        manager = _make_manager(args, pg)
        manager.train_parallel_config = {"dp_size": 2}
        events: list[str] = []
        failure = CommitFailure("commit failed")
        lease = _RecordingTrainBatchLease(
            rollout_id=29,
            output=_make_train_output(),
            events=events,
            commit_error=failure,
            rollback_error=None,
        )
        manager.rollout_session = _RecordingRolloutSession(
            lease=lease,
            eval_output=RolloutFnEvalOutput(data={}, metrics=None),
            events=events,
        )

        with pytest.raises(CommitFailure) as exc_info:
            await manager.generate(rollout_id=29)

        assert exc_info.value is failure
        assert events == ["acquire:29", "commit:29"]

    async def test_has_no_suspension_after_commit(
        self,
        ray_local_mode,
        placement_group_factory,
        tmp_path,
        patch_low_level,
    ):
        args = _make_test_args(tmp_path, models=[("actor", True)])
        args.global_batch_size = 8
        args.delay_split_train_data_by_dp = True
        args.debug_train_only = True
        pg = placement_group_factory(2)
        manager = _make_manager(args, pg)
        manager.train_parallel_config = {"dp_size": 2}
        events: list[str] = []
        lease = _CancelAfterCommitLease(
            rollout_id=31,
            output=_make_train_output(),
            events=events,
            commit_error=None,
            rollback_error=None,
        )
        manager.rollout_session = _RecordingRolloutSession(
            lease=lease,
            eval_output=RolloutFnEvalOutput(data={}, metrics=None),
            events=events,
        )

        generate_task = asyncio.create_task(manager.generate(rollout_id=31))
        result = await generate_task

        assert generate_task.cancelled() is False
        assert events == ["acquire:31", "commit:31"]
        assert set(result) == {"sample_indices", "data_ref"}
        assert isinstance(result["data_ref"].inner, ray.ObjectRef)

    async def test_debug_data_bypasses_session_acquisition(
        self,
        ray_local_mode,
        placement_group_factory,
        tmp_path,
        patch_low_level,
        monkeypatch,
    ):
        import miles.ray.rollout.rollout_manager as rmgr

        args = _make_test_args(tmp_path, models=[("actor", True)])
        args.global_batch_size = 8
        args.load_debug_rollout_data = "unused_{rollout_id}.pt"
        args.debug_train_only = True
        pg = placement_group_factory(2)
        manager = _make_manager(args, pg)
        manager.train_parallel_config = {"dp_size": 2}
        events: list[str] = []
        manager.rollout_session = _RecordingRolloutSession(
            lease=None,
            eval_output=RolloutFnEvalOutput(data={}, metrics=None),
            events=events,
        )
        debug_samples = make_samples_grouped(n_groups=2, group_size=4)
        monkeypatch.setattr(
            rmgr,
            "load_debug_rollout_data",
            lambda args, rollout_id: (debug_samples, {}),
        )

        result = await manager.generate(rollout_id=37)

        assert events == []
        assert set(result) == {"sample_indices", "data_ref"}
        assert all(isinstance(box.inner, ray.ObjectRef) for box in result["data_ref"])


@pytest.mark.asyncio
class TestEval:
    async def test_delegates_to_session_on_manager_event_loop(
        self,
        ray_local_mode,
        placement_group_factory,
        tmp_path,
        patch_low_level,
    ):
        args = _make_test_args(tmp_path, models=[("actor", True)])
        args.debug_train_only = True
        pg = placement_group_factory(2)

        manager = _make_manager(args, pg)
        manager.args.debug_train_only = False
        events: list[str] = []
        session = _RecordingRolloutSession(
            lease=None,
            eval_output=RolloutFnEvalOutput(
                data={"my_dataset": {"rewards": [0.5, 1.0]}},
                metrics={},
            ),
            events=events,
        )
        manager.rollout_session = session

        await manager.eval(rollout_id=10)

        assert events == ["evaluate:10"]
        assert session.eval_loop is asyncio.get_running_loop()

    async def test_skipped_in_debug_train_only_mode(
        self,
        ray_local_mode,
        placement_group_factory,
        tmp_path,
        patch_low_level,
    ):
        """``debug_train_only=True`` must short-circuit ``eval`` before the
        rollout function is invoked — used by trainer-only debug runs that
        have no rollout cluster."""
        args = _make_test_args(tmp_path, models=[("actor", True)])
        args.debug_train_only = True
        pg = placement_group_factory(2)

        manager = _make_manager(args, pg)
        events: list[str] = []
        manager.rollout_session = _RecordingRolloutSession(
            lease=None,
            eval_output=RolloutFnEvalOutput(data={}, metrics=None),
            events=events,
        )

        await manager.eval(rollout_id=10)

        assert events == []


@pytest.mark.asyncio
class TestDispose:
    async def test_resource_failure_does_not_skip_later_cleanup(
        self,
        ray_local_mode,
        placement_group_factory,
        tmp_path,
        patch_low_level,
        monkeypatch,
    ):
        import miles.ray.rollout.rollout_manager as rmgr

        class DataSourceFailure(BaseException):
            pass

        class FailingDataSource:
            def close(self) -> None:
                events.append("data_source_close_failed")
                raise failure

        class RecordingMetricChecker:
            def dispose(self) -> None:
                events.append("metric_checker_dispose")

        class FailingMonitor:
            def stop(self) -> None:
                events.append("first_monitor_stop_failed")
                raise RuntimeError("monitor stop failed")

        class RecordingMonitor:
            def stop(self) -> None:
                events.append("second_monitor_stop")

        args = _make_test_args(tmp_path, models=[("actor", True)])
        args.debug_train_only = True
        pg = placement_group_factory(2)
        manager = _make_manager(args, pg)
        events: list[str] = []
        failure = DataSourceFailure("data source close failed")
        manager.rollout_session = _RecordingRolloutSession(
            lease=None,
            eval_output=RolloutFnEvalOutput(data={}, metrics=None),
            events=events,
        )
        manager.data_source = FailingDataSource()
        manager._metric_checker = RecordingMetricChecker()
        manager._health_monitors = [FailingMonitor(), RecordingMonitor()]
        monkeypatch.setattr(
            rmgr.event_analyzer,
            "run_analysis_from_args",
            lambda args: events.append("event_analysis"),
        )

        with pytest.raises(DataSourceFailure) as exc_info:
            await manager.dispose()

        assert exc_info.value is failure
        assert events == [
            "session_close",
            "data_source_close_failed",
            "event_analysis",
            "metric_checker_dispose",
            "first_monitor_stop_failed",
            "second_monitor_stop",
        ]

    async def test_close_failure_preserves_resources_for_dispose_retry(
        self,
        ray_local_mode,
        placement_group_factory,
        tmp_path,
        patch_low_level,
        monkeypatch,
    ):
        import miles.ray.rollout.rollout_manager as rmgr

        class CloseFailure(BaseException):
            pass

        args = _make_test_args(tmp_path, models=[("actor", True)])
        args.debug_train_only = True
        pg = placement_group_factory(2)
        manager = _make_manager(args, pg)
        events: list[str] = []
        failure = CloseFailure("session close failed")
        manager.rollout_session = _FailingOnceCloseRolloutSession(events=events, close_error=failure)
        manager.data_source = _RecordingCloseableDataSource(events)
        monkeypatch.setattr(
            rmgr.event_analyzer,
            "run_analysis_from_args",
            lambda args: events.append("event_analysis"),
        )

        with pytest.raises(CloseFailure) as exc_info:
            await manager.dispose()

        assert exc_info.value is failure
        assert events == ["session_close_failed"]

        await manager.dispose()

        assert events == [
            "session_close_failed",
            "session_close_succeeded",
            "data_source_close",
            "event_analysis",
        ]

    async def test_cleanup_failure_does_not_skip_remaining_resources(
        self,
        ray_local_mode,
        placement_group_factory,
        tmp_path,
        patch_low_level,
        monkeypatch,
    ):
        import miles.ray.rollout.rollout_manager as rmgr

        args = _make_test_args(tmp_path, models=[("actor", True)])
        args.debug_train_only = True
        pg = placement_group_factory(2)
        manager = _make_manager(args, pg)
        events: list[str] = []
        failure = RuntimeError("data source close failed")
        manager.rollout_session = _RecordingRolloutSession(
            lease=None,
            eval_output=RolloutFnEvalOutput(data={}, metrics=None),
            events=events,
        )
        manager.data_source = _FailingCloseableDataSource(events, failure)
        manager._metric_checker = _RecordingDisposable(events, "metric_checker_dispose")
        manager._health_monitors = [
            _RecordingHealthMonitor(events, "first_health_monitor_stop"),
            _RecordingHealthMonitor(events, "second_health_monitor_stop"),
        ]
        monkeypatch.setattr(
            rmgr.event_analyzer,
            "run_analysis_from_args",
            lambda args: events.append("event_analysis"),
        )

        with pytest.raises(RuntimeError) as exc_info:
            await manager.dispose()

        assert exc_info.value is failure
        assert events == [
            "session_close",
            "data_source_close",
            "event_analysis",
            "metric_checker_dispose",
            "first_health_monitor_stop",
            "second_health_monitor_stop",
        ]

    async def test_waits_for_session_close_before_remaining_cleanup(
        self,
        ray_local_mode,
        placement_group_factory,
        tmp_path,
        patch_low_level,
        monkeypatch,
    ):
        import miles.ray.rollout.rollout_manager as rmgr

        args = _make_test_args(tmp_path, models=[("actor", True)])
        args.debug_train_only = True
        pg = placement_group_factory(2)
        manager = _make_manager(args, pg)
        events: list[str] = []
        close_started = asyncio.Event()
        allow_close = asyncio.Event()
        manager.rollout_session = _BlockingCloseRolloutSession(
            events=events,
            close_started=close_started,
            allow_close=allow_close,
        )
        manager.data_source = _RecordingCloseableDataSource(events)
        monkeypatch.setattr(
            rmgr.event_analyzer,
            "run_analysis_from_args",
            lambda args: events.append("event_analysis"),
        )

        dispose_task = asyncio.create_task(manager.dispose())
        await close_started.wait()

        assert dispose_task.done() is False
        assert events == ["session_close_started"]

        allow_close.set()
        await dispose_task

        assert events == [
            "session_close_started",
            "session_close_finished",
            "data_source_close",
            "event_analysis",
        ]

    async def test_cancellation_waits_for_session_then_completes_cleanup(
        self,
        ray_local_mode,
        placement_group_factory,
        tmp_path,
        patch_low_level,
        monkeypatch,
    ):
        import miles.ray.rollout.rollout_manager as rmgr

        args = _make_test_args(tmp_path, models=[("actor", True)])
        args.debug_train_only = True
        pg = placement_group_factory(2)
        manager = _make_manager(args, pg)
        events: list[str] = []
        close_started = asyncio.Event()
        allow_close = asyncio.Event()
        manager.rollout_session = _BlockingCloseRolloutSession(
            events=events,
            close_started=close_started,
            allow_close=allow_close,
        )
        manager.data_source = _RecordingCloseableDataSource(events)
        monkeypatch.setattr(
            rmgr.event_analyzer,
            "run_analysis_from_args",
            lambda args: events.append("event_analysis"),
        )

        dispose_task = asyncio.create_task(manager.dispose())
        await close_started.wait()
        dispose_task.cancel()
        await asyncio.sleep(0)

        assert dispose_task.done() is False
        assert events == ["session_close_started"]

        allow_close.set()
        with pytest.raises(asyncio.CancelledError):
            await dispose_task

        assert events == [
            "session_close_started",
            "session_close_finished",
            "data_source_close",
            "event_analysis",
        ]
