import asyncio
from types import SimpleNamespace

import pytest

import train_async


class _RemoteMethod:
    def __init__(self, function):
        self._function = function

    def remote(self, *args, **kwargs):
        return self._function(*args, **kwargs)


class _RecordingRolloutManager:
    def __init__(self, events: list[str], *, supports_train_admission_control: bool) -> None:
        self._events = events
        self._supports_train_admission_control = supports_train_admission_control
        self.admission_quiescent = False
        self.generate = _RemoteMethod(self._generate)
        self.generate_and_quiesce_train_admission = _RemoteMethod(self._generate_and_quiesce_train_admission)
        self.supports_train_admission_control = _RemoteMethod(self._supports_admission_control)
        self.quiesce_train_admission = _RemoteMethod(self._quiesce_train_admission)
        self.resume_train_admission = _RemoteMethod(self._resume_train_admission)
        self.eval = _RemoteMethod(self._eval)
        self.save = _RemoteMethod(self._save)
        self.dispose = _RemoteMethod(self._dispose)

    def _generate(self, rollout_id: int):
        return self._start_generation(rollout_id, quiesce_after_handoff=False)

    def _generate_and_quiesce_train_admission(self, rollout_id: int):
        return self._start_generation(rollout_id, quiesce_after_handoff=True)

    def _start_generation(self, rollout_id: int, *, quiesce_after_handoff: bool):
        self._events.append(f"generate:{rollout_id}:quiesce={quiesce_after_handoff}")

        async def complete_generation():
            assert not self.admission_quiescent
            self._events.append(f"handoff:{rollout_id}")
            if quiesce_after_handoff:
                self.admission_quiescent = True
                self._events.append("quiesce")
            return f"data:{rollout_id}"

        return complete_generation()

    async def _supports_admission_control(self) -> bool:
        return self._supports_train_admission_control

    async def _quiesce_train_admission(self) -> None:
        assert self._supports_train_admission_control
        self.admission_quiescent = True
        self._events.append("quiesce")

    async def _resume_train_admission(self) -> None:
        assert self._supports_train_admission_control
        assert self.admission_quiescent
        self.admission_quiescent = False
        self._events.append("resume")

    async def _eval(self, rollout_id: int) -> None:
        raise AssertionError(f"unexpected eval for rollout {rollout_id}")

    async def _save(self, rollout_id: int) -> None:
        raise AssertionError(f"unexpected save for rollout {rollout_id}")

    async def _dispose(self) -> None:
        self._events.append("dispose")


class _BlockingBoundaryRolloutManager(_RecordingRolloutManager):
    def __init__(self, events: list[str]) -> None:
        super().__init__(events, supports_train_admission_control=True)
        self.boundary_lease_opened = asyncio.Event()
        self.release_boundary_handoff = asyncio.Event()
        self.open_lease_rollout_id: int | None = None

    def _start_generation(self, rollout_id: int, *, quiesce_after_handoff: bool):
        if not quiesce_after_handoff:
            return super()._start_generation(rollout_id, quiesce_after_handoff=False)

        self._events.append(f"generate:{rollout_id}:quiesce=True")
        assert self.open_lease_rollout_id is None
        self.open_lease_rollout_id = rollout_id
        self.boundary_lease_opened.set()

        async def complete_generation():
            await self.release_boundary_handoff.wait()
            self._events.append(f"handoff:{rollout_id}")
            self.admission_quiescent = True
            self._events.append("quiesce")
            self.open_lease_rollout_id = None
            return f"data:{rollout_id}"

        return asyncio.create_task(complete_generation())

    async def _save(self, rollout_id: int) -> None:
        assert self.open_lease_rollout_id is None
        self._events.append(f"save:{rollout_id}")


class _RecordingActorModel:
    def __init__(
        self,
        events: list[str],
        rollout_manager: _RecordingRolloutManager,
        failed_updates: set[int | None],
    ) -> None:
        self._events = events
        self._rollout_manager = rollout_manager
        self._failed_updates = failed_updates

    async def update_weights(self, rollout_id: int | None = None) -> None:
        if self._rollout_manager._supports_train_admission_control:
            assert self._rollout_manager.admission_quiescent
        else:
            assert not self._rollout_manager.admission_quiescent
        label = "initial" if rollout_id is None else str(rollout_id)
        self._events.append(f"update:{label}")
        if rollout_id in self._failed_updates:
            raise RuntimeError(f"update failed for {label}")

    async def train(self, rollout_id: int, rollout_data: str) -> None:
        assert rollout_data == f"data:{rollout_id}"
        self._events.append(f"train:{rollout_id}")

    async def save_model(self, rollout_id: int, *, force_sync: bool) -> None:
        raise AssertionError(f"unexpected save for rollout {rollout_id}, force_sync={force_sync}")


class _SavingActorModel(_RecordingActorModel):
    async def save_model(self, rollout_id: int, *, force_sync: bool) -> None:
        self._events.append(f"save_model:{rollout_id}:force_sync={force_sync}")


def _make_args(*, num_rollout: int, update_weights_interval: int) -> SimpleNamespace:
    return SimpleNamespace(
        check_weight_update_equal=False,
        colocate=False,
        control_server_port=None,
        debug_exit_after_rollout=None,
        eval_interval=None,
        ft_components=[],
        num_rollout=num_rollout,
        save_interval=None,
        save_trigger_sentinel=None,
        skip_eval_before_train=True,
        start_rollout_id=0,
        update_weights_interval=update_weights_interval,
        use_critic=False,
    )


def _patch_train_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    rollout_manager: _RecordingRolloutManager,
    actor_model: _RecordingActorModel,
) -> None:
    monkeypatch.setattr(train_async, "configure_logger", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_async, "maybe_start_periodic_pyspy_dump", lambda: None)
    monkeypatch.setattr(train_async, "create_placement_groups", lambda args: {"rollout": object()})
    monkeypatch.setattr(train_async, "init_tracking", lambda args: None)
    monkeypatch.setattr(train_async, "create_rollout_manager", lambda args, pg: (rollout_manager, 1))

    async def create_training_models(args, pgs, manager):
        assert manager is rollout_manager
        return actor_model, None

    monkeypatch.setattr(train_async, "create_training_models", create_training_models)
    monkeypatch.setattr(train_async, "maybe_start_mini_ft_controller", lambda args: None)
    monkeypatch.setattr(train_async, "should_run_periodic_action", lambda *args: False)


@pytest.mark.asyncio
async def test_update_interval_one_quiesces_every_weight_update_and_prefetch_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    rollout_manager = _RecordingRolloutManager(events, supports_train_admission_control=True)
    actor_model = _RecordingActorModel(events, rollout_manager, failed_updates=set())
    _patch_train_dependencies(monkeypatch, rollout_manager, actor_model)

    await train_async.train(_make_args(num_rollout=3, update_weights_interval=1))

    assert events == [
        "quiesce",
        "update:initial",
        "resume",
        "generate:0:quiesce=False",
        "handoff:0",
        "generate:1:quiesce=True",
        "train:0",
        "handoff:1",
        "quiesce",
        "update:0",
        "resume",
        "generate:2:quiesce=True",
        "train:1",
        "handoff:2",
        "quiesce",
        "update:1",
        "train:2",
        "update:2",
        "dispose",
    ]


@pytest.mark.asyncio
async def test_update_interval_greater_than_one_fences_non_final_and_final_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    rollout_manager = _RecordingRolloutManager(events, supports_train_admission_control=True)
    actor_model = _RecordingActorModel(events, rollout_manager, failed_updates=set())
    _patch_train_dependencies(monkeypatch, rollout_manager, actor_model)

    await train_async.train(_make_args(num_rollout=4, update_weights_interval=2))

    assert events == [
        "quiesce",
        "update:initial",
        "resume",
        "generate:0:quiesce=False",
        "handoff:0",
        "generate:1:quiesce=False",
        "train:0",
        "handoff:1",
        "generate:2:quiesce=True",
        "train:1",
        "handoff:2",
        "quiesce",
        "update:1",
        "resume",
        "generate:3:quiesce=False",
        "train:2",
        "handoff:3",
        "train:3",
        "quiesce",
        "update:3",
        "dispose",
    ]


@pytest.mark.asyncio
async def test_unsupported_session_preserves_legacy_async_training_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    rollout_manager = _RecordingRolloutManager(events, supports_train_admission_control=False)
    actor_model = _RecordingActorModel(events, rollout_manager, failed_updates=set())
    _patch_train_dependencies(monkeypatch, rollout_manager, actor_model)

    await train_async.train(_make_args(num_rollout=4, update_weights_interval=2))

    assert events == [
        "update:initial",
        "generate:0:quiesce=False",
        "handoff:0",
        "generate:1:quiesce=False",
        "train:0",
        "handoff:1",
        "generate:2:quiesce=False",
        "train:1",
        "handoff:2",
        "update:1",
        "generate:3:quiesce=False",
        "train:2",
        "handoff:3",
        "train:3",
        "update:3",
        "dispose",
    ]


@pytest.mark.asyncio
async def test_weight_update_failure_leaves_admission_quiescent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    rollout_manager = _RecordingRolloutManager(events, supports_train_admission_control=True)
    actor_model = _RecordingActorModel(events, rollout_manager, failed_updates={0})
    _patch_train_dependencies(monkeypatch, rollout_manager, actor_model)

    with pytest.raises(RuntimeError, match="update failed for 0"):
        await train_async.train(_make_args(num_rollout=2, update_weights_interval=1))

    assert rollout_manager.admission_quiescent is True
    assert events == [
        "quiesce",
        "update:initial",
        "resume",
        "generate:0:quiesce=False",
        "handoff:0",
        "generate:1:quiesce=True",
        "train:0",
        "handoff:1",
        "quiesce",
        "update:0",
    ]


@pytest.mark.asyncio
async def test_initial_weight_update_failure_never_resumes_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    rollout_manager = _RecordingRolloutManager(events, supports_train_admission_control=True)
    actor_model = _RecordingActorModel(events, rollout_manager, failed_updates={None})
    _patch_train_dependencies(monkeypatch, rollout_manager, actor_model)

    with pytest.raises(RuntimeError, match="update failed for initial"):
        await train_async.train(_make_args(num_rollout=2, update_weights_interval=1))

    assert rollout_manager.admission_quiescent is True
    assert events == ["quiesce", "update:initial"]


@pytest.mark.asyncio
async def test_checkpoint_waits_for_boundary_generation_and_preserves_its_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    rollout_manager = _BlockingBoundaryRolloutManager(events)
    actor_model = _SavingActorModel(events, rollout_manager, failed_updates=set())
    _patch_train_dependencies(monkeypatch, rollout_manager, actor_model)
    monkeypatch.setattr(
        train_async,
        "should_run_periodic_action",
        lambda rollout_id, interval, num_rollout_per_epoch, num_rollout=None: interval is not None,
    )
    args = _make_args(num_rollout=2, update_weights_interval=1)
    args.save_interval = 1

    train_task = asyncio.create_task(train_async.train(args))
    await rollout_manager.boundary_lease_opened.wait()
    rollout_manager.release_boundary_handoff.set()

    await train_task

    assert events == [
        "quiesce",
        "update:initial",
        "resume",
        "generate:0:quiesce=False",
        "handoff:0",
        "generate:1:quiesce=True",
        "train:0",
        "handoff:1",
        "quiesce",
        "save_model:0:force_sync=False",
        "save:0",
        "update:0",
        "train:1",
        "save_model:1:force_sync=True",
        "save:1",
        "update:1",
        "dispose",
    ]
