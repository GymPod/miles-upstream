import asyncio
import pickle
import threading
from argparse import Namespace
from collections.abc import Awaitable, Callable
from unittest.mock import MagicMock

import pytest

from miles.rollout.base_types import (
    RolloutFnConstructorInput,
    RolloutFnEvalOutput,
    RolloutFnInput,
    RolloutFnOutput,
    RolloutFnTrainInput,
    RolloutFnTrainOutput,
)
from miles.rollout.data_source import DataSource
from miles.rollout.inference_rollout.compatibility import _invoke_rollout_function, load_rollout_session
from miles.rollout.rollout_session import BatchRollbackReason, BatchRollbackUnsupportedError, TrainBatchLease
from miles.utils.misc import function_registry


class _RecordingTrainBatchLease(TrainBatchLease):
    def __init__(
        self,
        rollout_id: int,
        output: RolloutFnTrainOutput,
        on_commit: Callable[[], None],
        on_rollback: Callable[[BatchRollbackReason], None],
    ) -> None:
        super().__init__(rollout_id=rollout_id, output=output)
        self._on_commit = on_commit
        self._on_rollback = on_rollback

    def _commit(self) -> None:
        self._on_commit()

    def _rollback(self, reason: BatchRollbackReason) -> None:
        self._on_rollback(reason)


async def test_train_batch_lease_commits_exactly_once() -> None:
    commits: list[int] = []

    def record_commit() -> None:
        commits.append(7)

    def reject_rollback(reason: BatchRollbackReason) -> None:
        raise AssertionError(f"Unexpected rollback: {reason}")

    output = RolloutFnTrainOutput(samples=[], metrics={"source": "test"})
    lease = _RecordingTrainBatchLease(
        rollout_id=7,
        output=output,
        on_commit=record_commit,
        on_rollback=reject_rollback,
    )

    assert lease.rollout_id == 7
    assert lease.output == RolloutFnTrainOutput(samples=[], metrics={"source": "test"})

    lease.commit()

    assert commits == [7]
    with pytest.raises(RuntimeError) as exc_info:
        lease.commit()
    assert str(exc_info.value) == "Train batch lease for rollout 7 is already committed."


async def test_batch_rollback_unsupported_error_round_trips_through_pickle() -> None:
    error = BatchRollbackUnsupportedError(
        rollout_id=11,
        reason=BatchRollbackReason.HANDOFF_FAILED,
    )

    restored = pickle.loads(pickle.dumps(error))

    assert type(restored) is BatchRollbackUnsupportedError
    assert restored.rollout_id == 11
    assert restored.reason is BatchRollbackReason.HANDOFF_FAILED
    assert str(restored) == "Train batch lease for rollout 11 cannot restore ownership after HANDOFF_FAILED."


async def test_rollout_session_evaluates_with_its_separate_instance() -> None:
    instances: list[object] = []

    class StatefulRolloutFn:
        def __init__(self, constructor_input: RolloutFnConstructorInput) -> None:
            self.instance_id = len(instances)
            instances.append(self)

        async def __call__(self, rollout_input: RolloutFnInput) -> RolloutFnOutput:
            if rollout_input.evaluation:
                return RolloutFnEvalOutput(
                    data={"benchmark": {"instance_id": self.instance_id}},
                    metrics={"evaluation": True},
                )
            return RolloutFnTrainOutput(samples=[], metrics={"evaluation": False})

    constructor_input = RolloutFnConstructorInput(
        args=Namespace(),
        data_source=MagicMock(spec=DataSource),
    )
    with function_registry.temporary("test:stateful_rollout_evaluation", StatefulRolloutFn):
        session = load_rollout_session(
            constructor_input,
            train_path="test:stateful_rollout_evaluation",
            eval_path="test:stateful_rollout_evaluation",
        )
        output = await session.evaluate(rollout_id=5)

    assert len(instances) == 2
    assert output == RolloutFnEvalOutput(
        data={"benchmark": {"instance_id": 1}},
        metrics={"evaluation": True},
    )


async def test_train_batch_lease_rolls_back_exactly_once() -> None:
    rollbacks: list[BatchRollbackReason] = []

    def reject_commit() -> None:
        raise AssertionError("Unexpected commit")

    def record_rollback(reason: BatchRollbackReason) -> None:
        rollbacks.append(reason)

    lease = _RecordingTrainBatchLease(
        rollout_id=11,
        output=RolloutFnTrainOutput(samples=[], metrics=None),
        on_commit=reject_commit,
        on_rollback=record_rollback,
    )

    lease.rollback(BatchRollbackReason.HANDOFF_FAILED)

    assert rollbacks == [BatchRollbackReason.HANDOFF_FAILED]
    with pytest.raises(RuntimeError) as rollback_error:
        lease.rollback(BatchRollbackReason.HANDOFF_FAILED)
    assert str(rollback_error.value) == "Train batch lease for rollout 11 is already rolled back."
    with pytest.raises(RuntimeError) as commit_error:
        lease.commit()
    assert str(commit_error.value) == "Train batch lease for rollout 11 is already rolled back."


@pytest.mark.parametrize("settlement", ["commit", "rollback"])
async def test_train_batch_lease_rejects_retry_after_failed_settlement(settlement: str) -> None:
    commit_attempts = 0
    rollback_attempts = 0

    def commit() -> None:
        nonlocal commit_attempts
        commit_attempts += 1
        if commit_attempts == 1:
            raise RuntimeError("commit failed")

    def rollback(reason: BatchRollbackReason) -> None:
        nonlocal rollback_attempts
        rollback_attempts += 1
        if rollback_attempts == 1:
            raise RuntimeError("rollback failed")

    lease = _RecordingTrainBatchLease(
        rollout_id=12,
        output=RolloutFnTrainOutput(samples=[], metrics=None),
        on_commit=commit,
        on_rollback=rollback,
    )

    if settlement == "commit":
        with pytest.raises(RuntimeError) as settlement_error:
            lease.commit()
        assert str(settlement_error.value) == "commit failed"
        with pytest.raises(RuntimeError) as retry_error:
            lease.commit()
        assert str(retry_error.value) == "Train batch lease for rollout 12 is already commit failed."
        with pytest.raises(RuntimeError) as opposite_error:
            lease.rollback(BatchRollbackReason.HANDOFF_FAILED)
        assert str(opposite_error.value) == "Train batch lease for rollout 12 is already commit failed."
        assert (commit_attempts, rollback_attempts) == (1, 0)
    else:
        with pytest.raises(RuntimeError) as settlement_error:
            lease.rollback(BatchRollbackReason.HANDOFF_FAILED)
        assert str(settlement_error.value) == "rollback failed"
        with pytest.raises(RuntimeError) as retry_error:
            lease.rollback(BatchRollbackReason.HANDOFF_FAILED)
        assert str(retry_error.value) == "Train batch lease for rollout 12 is already rollback failed."
        with pytest.raises(RuntimeError) as opposite_error:
            lease.commit()
        assert str(opposite_error.value) == "Train batch lease for rollout 12 is already rollback failed."
        assert (commit_attempts, rollback_attempts) == (0, 1)


async def test_rollout_session_acquires_from_a_dedicated_train_instance() -> None:
    instances: list[object] = []
    actor_loop = asyncio.get_running_loop()
    call_loops: list[asyncio.AbstractEventLoop] = []

    class StatefulRolloutFn:
        def __init__(self, constructor_input: RolloutFnConstructorInput) -> None:
            self.instance_id = len(instances)
            instances.append(self)

        async def __call__(self, rollout_input: RolloutFnInput) -> RolloutFnTrainOutput:
            call_loops.append(asyncio.get_running_loop())
            return RolloutFnTrainOutput(
                samples=[],
                metrics={"instance_id": self.instance_id, "evaluation": rollout_input.evaluation},
            )

    constructor_input = RolloutFnConstructorInput(
        args=Namespace(),
        data_source=MagicMock(spec=DataSource),
    )
    with function_registry.temporary("test:stateful_rollout_session", StatefulRolloutFn):
        session = load_rollout_session(
            constructor_input,
            train_path="test:stateful_rollout_session",
            eval_path="test:stateful_rollout_session",
        )
        lease = await session.acquire_train_batch(rollout_id=3)

    assert len(instances) == 2
    assert instances[0] is not instances[1]
    assert call_loops == [actor_loop]
    assert lease.rollout_id == 3
    assert lease.output == RolloutFnTrainOutput(
        samples=[],
        metrics={"instance_id": 0, "evaluation": False},
    )

    lease.commit()


async def test_compatibility_session_reports_unsupported_rollback_after_settling_lease() -> None:
    def rollout_fn(
        args: Namespace,
        rollout_id: int,
        data_source: DataSource,
        evaluation: bool,
    ) -> RolloutFnOutput:
        if evaluation:
            return RolloutFnEvalOutput(data={})
        return RolloutFnTrainOutput(samples=[], metrics=None)

    constructor_input = RolloutFnConstructorInput(
        args=Namespace(),
        data_source=MagicMock(spec=DataSource),
    )
    with function_registry.temporary("test:unsupported_compatibility_rollback", rollout_fn):
        session = load_rollout_session(
            constructor_input,
            train_path="test:unsupported_compatibility_rollback",
            eval_path="test:unsupported_compatibility_rollback",
        )
        lease = await session.acquire_train_batch(rollout_id=5)

        with pytest.raises(BatchRollbackUnsupportedError) as error:
            lease.rollback(BatchRollbackReason.HANDOFF_FAILED)

        assert error.value.rollout_id == 5
        assert error.value.reason is BatchRollbackReason.HANDOFF_FAILED
        assert str(error.value) == "Train batch lease for rollout 5 cannot restore ownership after HANDOFF_FAILED."
        await session.prepare_checkpoint(rollout_id=5)
        await session.close()


async def test_rollout_session_runs_sync_classes_off_the_actor_loop() -> None:
    actor_thread_id = threading.get_ident()
    lifecycle_events: list[list[tuple[str, int]]] = []

    class SyncRolloutFn:
        def __init__(self, constructor_input: RolloutFnConstructorInput) -> None:
            self.events = [("construct", threading.get_ident())]
            lifecycle_events.append(self.events)

        def __call__(self, rollout_input: RolloutFnInput) -> RolloutFnOutput:
            self.events.append(("call", threading.get_ident()))
            if rollout_input.evaluation:
                return RolloutFnEvalOutput(data={}, metrics=None)
            return RolloutFnTrainOutput(samples=[], metrics={"rollout_id": rollout_input.rollout_id})

        def close(self) -> None:
            self.events.append(("close", threading.get_ident()))

    constructor_input = RolloutFnConstructorInput(
        args=Namespace(),
        data_source=MagicMock(spec=DataSource),
    )
    with function_registry.temporary("test:sync_class_rollout_session", SyncRolloutFn):
        session = load_rollout_session(
            constructor_input,
            train_path="test:sync_class_rollout_session",
            eval_path="test:sync_class_rollout_session",
        )
        lease = await session.acquire_train_batch(rollout_id=47)
        lease.commit()
        await session.evaluate(rollout_id=47)
        await session.close()

    assert [event for event, _ in lifecycle_events[0]] == ["construct", "call", "close"]
    assert [event for event, _ in lifecycle_events[1]] == ["construct", "call", "close"]
    for events in lifecycle_events:
        thread_ids = {thread_id for _, thread_id in events}
        assert len(thread_ids) == 1
        assert thread_ids != {actor_thread_id}
    assert lease.output == RolloutFnTrainOutput(samples=[], metrics={"rollout_id": 47})


async def test_sync_class_awaitables_resume_on_the_actor_loop() -> None:
    actor_loop = asyncio.get_running_loop()
    actor_thread_id = threading.get_ident()
    call_thread_ids: list[int] = []
    completion_loops: list[asyncio.AbstractEventLoop] = []

    class HybridRolloutFn:
        def __init__(self, constructor_input: RolloutFnConstructorInput) -> None:
            return

        def __call__(self, rollout_input: RolloutFnInput) -> Awaitable[RolloutFnOutput]:
            call_thread_ids.append(threading.get_ident())

            async def complete() -> RolloutFnOutput:
                completion_loops.append(asyncio.get_running_loop())
                return RolloutFnTrainOutput(samples=[], metrics={"rollout_id": rollout_input.rollout_id})

            return complete()

    constructor_input = RolloutFnConstructorInput(
        args=Namespace(),
        data_source=MagicMock(spec=DataSource),
    )
    with function_registry.temporary("test:hybrid_rollout_session", HybridRolloutFn):
        session = load_rollout_session(
            constructor_input,
            train_path="test:hybrid_rollout_session",
            eval_path="test:hybrid_rollout_session",
        )
        first = await session.acquire_train_batch(rollout_id=48)
        first.commit()
        second = await session.acquire_train_batch(rollout_id=49)
        second.commit()
        await session.close()

    assert len(set(call_thread_ids)) == 1
    assert call_thread_ids[0] != actor_thread_id
    assert completion_loops == [actor_loop, actor_loop]


async def test_cancelled_sync_call_settles_its_returned_awaitable() -> None:
    call_started = threading.Event()
    release_call = threading.Event()
    awaitable_completed = asyncio.Event()

    class HybridRolloutFn:
        def __init__(self, constructor_input: RolloutFnConstructorInput) -> None:
            return

        def __call__(self, rollout_input: RolloutFnInput) -> Awaitable[RolloutFnOutput]:
            call_started.set()
            assert release_call.wait(timeout=5)

            async def complete() -> RolloutFnOutput:
                await asyncio.sleep(0)
                awaitable_completed.set()
                return RolloutFnTrainOutput(samples=[], metrics=None)

            return complete()

    constructor_input = RolloutFnConstructorInput(
        args=Namespace(),
        data_source=MagicMock(spec=DataSource),
    )
    with function_registry.temporary("test:cancelled_hybrid_call", HybridRolloutFn):
        session = load_rollout_session(
            constructor_input,
            train_path="test:cancelled_hybrid_call",
            eval_path="test:cancelled_hybrid_call",
        )
        acquire_task = asyncio.create_task(session.acquire_train_batch(rollout_id=50))
        assert await asyncio.to_thread(call_started.wait, 1)

        acquire_task.cancel()
        release_call.set()
        with pytest.raises(asyncio.CancelledError):
            await acquire_task

        await session.close()

    assert awaitable_completed.is_set()


async def test_cancelled_legacy_sync_call_settles_its_returned_awaitable() -> None:
    call_started = threading.Event()
    release_call = threading.Event()
    awaitable_completed = asyncio.Event()

    def legacy_rollout_fn(rollout_input: RolloutFnInput) -> Awaitable[RolloutFnOutput]:
        call_started.set()
        assert release_call.wait(timeout=5)

        async def complete() -> RolloutFnOutput:
            await asyncio.sleep(0)
            awaitable_completed.set()
            return RolloutFnTrainOutput(samples=[], metrics=None)

        return complete()

    call_task = asyncio.create_task(
        _invoke_rollout_function(
            legacy_rollout_fn,
            RolloutFnTrainInput(rollout_id=51),
        )
    )
    assert await asyncio.to_thread(call_started.wait, 1)

    call_task.cancel()
    release_call.set()
    with pytest.raises(asyncio.CancelledError):
        await call_task

    assert awaitable_completed.is_set()


async def test_rollout_session_runs_sync_functions_off_the_actor_loop() -> None:
    actor_thread_id = threading.get_ident()
    call_thread_ids: list[int] = []

    def sync_rollout_fn(
        args: Namespace,
        rollout_id: int,
        data_source: DataSource,
        evaluation: bool,
    ) -> RolloutFnTrainOutput:
        call_thread_ids.append(threading.get_ident())
        return RolloutFnTrainOutput(
            samples=[],
            metrics={"rollout_id": rollout_id, "evaluation": evaluation},
        )

    constructor_input = RolloutFnConstructorInput(
        args=Namespace(),
        data_source=MagicMock(spec=DataSource),
    )
    with function_registry.temporary("test:sync_rollout_session", sync_rollout_fn):
        session = load_rollout_session(
            constructor_input,
            train_path="test:sync_rollout_session",
            eval_path="test:sync_rollout_session",
        )
        lease = await session.acquire_train_batch(rollout_id=13)

    assert len(call_thread_ids) == 1
    assert call_thread_ids[0] != actor_thread_id
    assert lease.output == RolloutFnTrainOutput(
        samples=[],
        metrics={"rollout_id": 13, "evaluation": False},
    )
    lease.commit()


async def test_rollout_session_rejects_checkpoint_with_an_open_batch_lease() -> None:
    def sync_rollout_fn(
        args: Namespace,
        rollout_id: int,
        data_source: DataSource,
        evaluation: bool,
    ) -> RolloutFnTrainOutput:
        return RolloutFnTrainOutput(samples=[], metrics={"rollout_id": rollout_id})

    constructor_input = RolloutFnConstructorInput(
        args=Namespace(),
        data_source=MagicMock(spec=DataSource),
    )
    with function_registry.temporary("test:checkpoint_rollout_session", sync_rollout_fn):
        session = load_rollout_session(
            constructor_input,
            train_path="test:checkpoint_rollout_session",
            eval_path="test:checkpoint_rollout_session",
        )
        lease = await session.acquire_train_batch(rollout_id=17)

        with pytest.raises(RuntimeError) as exc_info:
            await session.prepare_checkpoint(rollout_id=17)
        assert str(exc_info.value) == "Cannot prepare checkpoint 17 with open train batch leases: [17]."

        lease.commit()
        result = await session.prepare_checkpoint(rollout_id=17)

    assert result is None


async def test_duplicate_rollout_id_is_rejected_before_generation() -> None:
    call_rollout_ids: list[int] = []

    class RecordingRolloutFn:
        def __init__(self, constructor_input: RolloutFnConstructorInput) -> None:
            return

        async def __call__(self, rollout_input: RolloutFnInput) -> RolloutFnOutput:
            call_rollout_ids.append(rollout_input.rollout_id)
            if rollout_input.evaluation:
                return RolloutFnEvalOutput(data={}, metrics=None)
            return RolloutFnTrainOutput(samples=[], metrics=None)

    constructor_input = RolloutFnConstructorInput(
        args=Namespace(),
        data_source=MagicMock(spec=DataSource),
    )
    with function_registry.temporary("test:duplicate_rollout_id", RecordingRolloutFn):
        session = load_rollout_session(
            constructor_input,
            train_path="test:duplicate_rollout_id",
            eval_path="test:duplicate_rollout_id",
        )
        lease = await session.acquire_train_batch(rollout_id=53)

        with pytest.raises(RuntimeError) as exc_info:
            await session.acquire_train_batch(rollout_id=53)
        assert str(exc_info.value) == "Train batch lease for rollout 53 is already open."

        lease.commit()
        await session.close()

    assert call_rollout_ids == [53]


async def test_eval_constructor_failure_closes_the_train_instance() -> None:
    train_instances: list[object] = []
    closed_train_instances: list[int] = []

    class TrainRolloutFn:
        def __init__(self, constructor_input: RolloutFnConstructorInput) -> None:
            self.instance_id = len(train_instances)
            train_instances.append(self)

        async def __call__(self, rollout_input: RolloutFnInput) -> RolloutFnTrainOutput:
            return RolloutFnTrainOutput(samples=[], metrics=None)

        async def close(self) -> None:
            closed_train_instances.append(self.instance_id)

    class FailingEvalRolloutFn:
        def __init__(self, constructor_input: RolloutFnConstructorInput) -> None:
            raise RuntimeError("eval construction failed")

    constructor_input = RolloutFnConstructorInput(
        args=Namespace(),
        data_source=MagicMock(spec=DataSource),
    )
    with (
        function_registry.temporary("test:partial_train_constructor", TrainRolloutFn),
        function_registry.temporary("test:partial_eval_constructor", FailingEvalRolloutFn),
    ):
        session = load_rollout_session(
            constructor_input,
            train_path="test:partial_train_constructor",
            eval_path="test:partial_eval_constructor",
        )

        with pytest.raises(RuntimeError) as exc_info:
            await session.evaluate(rollout_id=59)
        assert str(exc_info.value) == "eval construction failed"

        await session.close()

    assert len(train_instances) == 1
    assert closed_train_instances == [0]


async def test_failed_partial_construction_cleanup_is_retried_by_close() -> None:
    close_attempts: list[int] = []

    class TrainRolloutFn:
        def __init__(self, constructor_input: RolloutFnConstructorInput) -> None:
            return

        async def __call__(self, rollout_input: RolloutFnInput) -> RolloutFnTrainOutput:
            return RolloutFnTrainOutput(samples=[], metrics=None)

        async def close(self) -> None:
            close_attempts.append(len(close_attempts) + 1)
            if len(close_attempts) == 1:
                raise RuntimeError("train close failed")

    class FailingEvalRolloutFn:
        def __init__(self, constructor_input: RolloutFnConstructorInput) -> None:
            raise RuntimeError("eval construction failed")

    constructor_input = RolloutFnConstructorInput(
        args=Namespace(),
        data_source=MagicMock(spec=DataSource),
    )
    with (
        function_registry.temporary("test:failed_partial_cleanup_train", TrainRolloutFn),
        function_registry.temporary("test:failed_partial_cleanup_eval", FailingEvalRolloutFn),
    ):
        session = load_rollout_session(
            constructor_input,
            train_path="test:failed_partial_cleanup_train",
            eval_path="test:failed_partial_cleanup_eval",
        )

        with pytest.raises(RuntimeError) as construction_error:
            await session.evaluate(rollout_id=61)
        assert str(construction_error.value) == "eval construction failed"
        assert isinstance(construction_error.value.__cause__, RuntimeError)
        assert str(construction_error.value.__cause__) == "train close failed"

        await session.close()

    assert close_attempts == [1, 2]


async def test_cancelled_partial_construction_waits_for_cleanup() -> None:
    close_started = asyncio.Event()
    release_close = asyncio.Event()
    close_completed = asyncio.Event()

    class TrainRolloutFn:
        def __init__(self, constructor_input: RolloutFnConstructorInput) -> None:
            return

        async def __call__(self, rollout_input: RolloutFnInput) -> RolloutFnTrainOutput:
            return RolloutFnTrainOutput(samples=[], metrics=None)

        async def close(self) -> None:
            close_started.set()
            await release_close.wait()
            close_completed.set()

    class FailingEvalRolloutFn:
        def __init__(self, constructor_input: RolloutFnConstructorInput) -> None:
            raise RuntimeError("eval construction failed")

    constructor_input = RolloutFnConstructorInput(
        args=Namespace(),
        data_source=MagicMock(spec=DataSource),
    )
    with (
        function_registry.temporary("test:cancelled_partial_cleanup_train", TrainRolloutFn),
        function_registry.temporary("test:cancelled_partial_cleanup_eval", FailingEvalRolloutFn),
    ):
        session = load_rollout_session(
            constructor_input,
            train_path="test:cancelled_partial_cleanup_train",
            eval_path="test:cancelled_partial_cleanup_eval",
        )
        evaluate_task = asyncio.create_task(session.evaluate(rollout_id=67))
        await close_started.wait()

        evaluate_task.cancel()
        await asyncio.sleep(0)

        finished_before_release = evaluate_task.done()
        release_close.set()
        with pytest.raises(asyncio.CancelledError):
            await evaluate_task

        await session.close()

    assert not finished_before_release
    assert close_completed.is_set()


async def test_rollout_session_closes_both_instances_exactly_once() -> None:
    instances: list[object] = []
    closed_instance_ids: list[int] = []

    class ClosableRolloutFn:
        def __init__(self, constructor_input: RolloutFnConstructorInput) -> None:
            self.instance_id = len(instances)
            instances.append(self)

        async def __call__(self, rollout_input: RolloutFnInput) -> RolloutFnOutput:
            if rollout_input.evaluation:
                return RolloutFnEvalOutput(data={}, metrics=None)
            return RolloutFnTrainOutput(samples=[], metrics=None)

        async def close(self) -> None:
            closed_instance_ids.append(self.instance_id)

    constructor_input = RolloutFnConstructorInput(
        args=Namespace(),
        data_source=MagicMock(spec=DataSource),
    )
    with function_registry.temporary("test:closable_rollout_session", ClosableRolloutFn):
        session = load_rollout_session(
            constructor_input,
            train_path="test:closable_rollout_session",
            eval_path="test:closable_rollout_session",
        )

        await session.evaluate(rollout_id=0)
        await session.close()
        await session.close()

    assert len(instances) == 2
    assert closed_instance_ids == [0, 1]


async def test_rollout_session_rejects_close_with_an_open_batch_lease() -> None:
    def sync_rollout_fn(
        args: Namespace,
        rollout_id: int,
        data_source: DataSource,
        evaluation: bool,
    ) -> RolloutFnTrainOutput:
        return RolloutFnTrainOutput(samples=[], metrics={"rollout_id": rollout_id})

    constructor_input = RolloutFnConstructorInput(
        args=Namespace(),
        data_source=MagicMock(spec=DataSource),
    )
    with function_registry.temporary("test:close_with_lease", sync_rollout_fn):
        session = load_rollout_session(
            constructor_input,
            train_path="test:close_with_lease",
            eval_path="test:close_with_lease",
        )
        lease = await session.acquire_train_batch(rollout_id=19)

        with pytest.raises(RuntimeError) as exc_info:
            await session.close()
        assert str(exc_info.value) == "Cannot close rollout session with open train batch leases: [19]."

        lease.commit()
        await session.close()


async def test_rollout_session_rejects_operations_after_close() -> None:
    calls: list[RolloutFnInput] = []

    async def rollout_fn(rollout_input: RolloutFnInput) -> RolloutFnOutput:
        calls.append(rollout_input)
        if rollout_input.evaluation:
            return RolloutFnEvalOutput(data={}, metrics=None)
        return RolloutFnTrainOutput(samples=[], metrics=None)

    constructor_input = RolloutFnConstructorInput(
        args=Namespace(),
        data_source=MagicMock(spec=DataSource),
    )
    with function_registry.temporary("test:closed_rollout_session", rollout_fn):
        session = load_rollout_session(
            constructor_input,
            train_path="test:closed_rollout_session",
            eval_path="test:closed_rollout_session",
        )
        await session.close()

        with pytest.raises(RuntimeError) as acquire_error:
            await session.acquire_train_batch(rollout_id=23)
        assert str(acquire_error.value) == "Rollout session is closed."

        with pytest.raises(RuntimeError) as eval_error:
            await session.evaluate(rollout_id=23)
        assert str(eval_error.value) == "Rollout session is closed."

        with pytest.raises(RuntimeError) as checkpoint_error:
            await session.prepare_checkpoint(rollout_id=23)
        assert str(checkpoint_error.value) == "Rollout session is closed."

    assert calls == []


async def test_cancelled_close_waits_for_both_instances_to_close() -> None:
    instances: list[object] = []
    close_started = asyncio.Event()
    release_close = asyncio.Event()
    closed_instance_ids: list[int] = []

    class SlowCloseRolloutFn:
        def __init__(self, constructor_input: RolloutFnConstructorInput) -> None:
            self.instance_id = len(instances)
            instances.append(self)

        async def __call__(self, rollout_input: RolloutFnInput) -> RolloutFnOutput:
            if rollout_input.evaluation:
                return RolloutFnEvalOutput(data={}, metrics=None)
            return RolloutFnTrainOutput(samples=[], metrics=None)

        async def close(self) -> None:
            if self.instance_id == 0:
                close_started.set()
                await release_close.wait()
            closed_instance_ids.append(self.instance_id)

    constructor_input = RolloutFnConstructorInput(
        args=Namespace(),
        data_source=MagicMock(spec=DataSource),
    )
    with function_registry.temporary("test:cancelled_close", SlowCloseRolloutFn):
        session = load_rollout_session(
            constructor_input,
            train_path="test:cancelled_close",
            eval_path="test:cancelled_close",
        )
        await session.evaluate(rollout_id=0)
        close_task = asyncio.create_task(session.close())
        await close_started.wait()

        close_task.cancel()
        await asyncio.sleep(0)

        finished_before_release = close_task.done()
        release_close.set()
        with pytest.raises(asyncio.CancelledError):
            await close_task

        await session.close()

    assert not finished_before_release
    assert len(instances) == 2
    assert closed_instance_ids == [0, 1]


async def test_close_attempts_both_instances_and_retries_only_failures() -> None:
    instances: list[object] = []
    close_attempts: list[int] = []

    class FailingCloseRolloutFn:
        def __init__(self, constructor_input: RolloutFnConstructorInput) -> None:
            self.instance_id = len(instances)
            instances.append(self)

        async def __call__(self, rollout_input: RolloutFnInput) -> RolloutFnOutput:
            if rollout_input.evaluation:
                return RolloutFnEvalOutput(data={}, metrics=None)
            return RolloutFnTrainOutput(samples=[], metrics=None)

        async def close(self) -> None:
            close_attempts.append(self.instance_id)
            if self.instance_id == 0:
                raise RuntimeError("train close failed")

    constructor_input = RolloutFnConstructorInput(
        args=Namespace(),
        data_source=MagicMock(spec=DataSource),
    )
    with function_registry.temporary("test:failing_close", FailingCloseRolloutFn):
        session = load_rollout_session(
            constructor_input,
            train_path="test:failing_close",
            eval_path="test:failing_close",
        )

        await session.evaluate(rollout_id=0)
        with pytest.raises(RuntimeError) as first_error:
            await session.close()
        assert str(first_error.value) == "train close failed"
        assert close_attempts == [0, 1]

        with pytest.raises(RuntimeError) as acquire_error:
            await session.acquire_train_batch(rollout_id=43)
        assert str(acquire_error.value) == "Rollout session is closed."

        with pytest.raises(RuntimeError) as second_error:
            await session.close()
        assert str(second_error.value) == "train close failed"

    assert len(instances) == 2
    assert close_attempts == [0, 1, 0]
