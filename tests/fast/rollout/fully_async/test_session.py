import asyncio
from argparse import Namespace
from collections.abc import Sequence
from dataclasses import replace
from typing import cast

import pytest

from miles.rollout.data_source import DataSource, SourceReservation, SourceReservationId
from miles.rollout.fully_async.execution import (
    FullyAsyncExecution,
    FullyAsyncExecutionFailure,
    FullyAsyncExecutionOutcome,
    FullyAsyncExecutionRetry,
    FullyAsyncExecutionSuccess,
    FullyAsyncExecutor,
    FullyAsyncRetryReason,
    FullyAsyncTerminalPendingError,
)
from miles.rollout.fully_async.ownership import ReservationExecutorReceipt
from miles.rollout.fully_async.scheduler import _FullyAsyncScheduler
from miles.rollout.fully_async.session import FullyAsyncRolloutSession
from miles.rollout.rollout_session import BatchRollbackReason
from miles.utils.types import Sample


class _RecordingDataSource(DataSource):
    def __init__(self) -> None:
        self.next_reservation_id = 0
        self.replay_ids: list[SourceReservationId] = []
        self.issued: list[SourceReservation] = []
        self.outstanding: dict[SourceReservationId, SourceReservation] = {}
        self.acknowledged: list[tuple[list[SourceReservation], int]] = []
        self.requeued: list[list[SourceReservation]] = []
        self.requeue_error: BaseException | None = None

    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        raise AssertionError("Bounded scheduling must reserve source groups.")

    def add_samples(self, samples: list[list[Sample]]) -> None:
        raise AssertionError("Bounded scheduling must settle source reservations.")

    def save(self, rollout_id: int) -> None:
        return

    def load(self, rollout_id: int | None = None) -> None:
        return

    def reserve_samples(self, num_groups: int) -> list[SourceReservation]:
        reservations = []
        for _ in range(num_groups):
            if self.replay_ids:
                reservation_id = self.replay_ids.pop(0)
            else:
                reservation_id = SourceReservationId(self.next_reservation_id)
                self.next_reservation_id += 1
            reservation = SourceReservation(
                reservation_id=reservation_id,
                samples=[
                    Sample(
                        group_index=int(reservation_id),
                        index=int(reservation_id),
                        prompt=f"prompt-{reservation_id}",
                    )
                ],
            )
            self.issued.append(reservation)
            self.outstanding[reservation_id] = reservation
            reservations.append(reservation)
        return reservations

    def acknowledge_reservations(
        self,
        reservations: Sequence[SourceReservation],
        *,
        rollout_id: int,
    ) -> None:
        attempts = list(reservations)
        self._settle(attempts)
        self.acknowledged.append((attempts, rollout_id))

    def requeue_reservations(self, reservations: Sequence[SourceReservation]) -> None:
        attempts = list(reservations)
        if self.requeue_error is not None:
            raise self.requeue_error
        self._settle(attempts)
        self.replay_ids.extend(reservation.reservation_id for reservation in attempts)
        self.requeued.append(attempts)

    def _settle(self, reservations: list[SourceReservation]) -> None:
        for reservation in reservations:
            if self.outstanding.get(reservation.reservation_id) is not reservation:
                raise RuntimeError(f"Reservation {reservation.reservation_id} is not outstanding.")
        for reservation in reservations:
            del self.outstanding[reservation.reservation_id]


class _ControlledExecution(FullyAsyncExecution):
    def __init__(
        self,
        reservation: SourceReservation,
        receipt: ReservationExecutorReceipt,
        *,
        cancel_completes: bool,
    ) -> None:
        self.reservation = reservation
        self.receipt = receipt
        self.cancel_completes = cancel_completes
        self.cancellation_requests = 0
        self.result: asyncio.Future[FullyAsyncExecutionOutcome] = asyncio.get_running_loop().create_future()

    def request_cancellation(self) -> None:
        self.cancellation_requests += 1
        if self.cancel_completes and not self.result.done():
            self.result.set_result(
                FullyAsyncExecutionRetry(
                    executor_receipt=self.receipt,
                    reason=FullyAsyncRetryReason.CANCELLATION_REQUESTED,
                )
            )

    async def wait_terminal(self) -> FullyAsyncExecutionOutcome:
        return await self.result

    def succeed(
        self,
        *,
        receipt: ReservationExecutorReceipt | None = None,
        samples: list[Sample] | None = None,
    ) -> None:
        self.result.set_result(
            FullyAsyncExecutionSuccess(
                executor_receipt=self.receipt if receipt is None else receipt,
                samples=self.reservation.samples if samples is None else samples,
            )
        )

    def fail(self, error: BaseException) -> None:
        self.result.set_result(
            FullyAsyncExecutionFailure(
                executor_receipt=self.receipt,
                error=error,
            )
        )

    def retry(self, reason: FullyAsyncRetryReason) -> None:
        self.result.set_result(
            FullyAsyncExecutionRetry(
                executor_receipt=self.receipt,
                reason=reason,
            )
        )


class _ControlledExecutor(FullyAsyncExecutor):
    def __init__(self, *, cancel_completes: bool = True) -> None:
        self.cancel_completes = cancel_completes
        self.executions: list[_ControlledExecution] = []
        self.close_calls = 0

    def submit(
        self,
        reservation: SourceReservation,
        receipt: ReservationExecutorReceipt,
    ) -> FullyAsyncExecution:
        execution = _ControlledExecution(
            reservation,
            receipt,
            cancel_completes=self.cancel_completes,
        )
        self.executions.append(execution)
        return execution

    async def close(self) -> None:
        self.close_calls += 1


class _RetryableTerminalExecution(_ControlledExecution):
    def __init__(
        self,
        reservation: SourceReservation,
        receipt: ReservationExecutorReceipt,
    ) -> None:
        super().__init__(reservation, receipt, cancel_completes=False)
        self.observation_attempts = 0
        self.retry_result: asyncio.Future[FullyAsyncExecutionOutcome] = asyncio.get_running_loop().create_future()

    async def wait_terminal(self) -> FullyAsyncExecutionOutcome:
        self.observation_attempts += 1
        if self.observation_attempts == 1:
            while self.cancellation_requests == 0:
                await asyncio.sleep(0)
            raise FullyAsyncTerminalPendingError("terminal proof is pending")
        return await self.retry_result

    def complete_retry(self) -> None:
        self.retry_result.set_result(
            FullyAsyncExecutionRetry(
                executor_receipt=self.receipt,
                reason=FullyAsyncRetryReason.CANCELLATION_REQUESTED,
            )
        )


class _RetryableTerminalExecutor(_ControlledExecutor):
    def __init__(self) -> None:
        super().__init__(cancel_completes=False)

    def submit(
        self,
        reservation: SourceReservation,
        receipt: ReservationExecutorReceipt,
    ) -> FullyAsyncExecution:
        execution = _RetryableTerminalExecution(reservation, receipt)
        self.executions.append(execution)
        return execution


def _scheduler(
    *,
    groups_per_batch: int,
    execution_samples: int,
    retained_groups: int,
    completed_groups: int,
) -> tuple[_FullyAsyncScheduler, _RecordingDataSource, _ControlledExecutor]:
    data_source = _RecordingDataSource()
    executor = _ControlledExecutor()
    scheduler = _FullyAsyncScheduler(
        data_source=data_source,
        executor=executor,
        samples_per_group=1,
        groups_per_batch=groups_per_batch,
        max_execution_samples=execution_samples,
        max_retained_groups=retained_groups,
        max_completed_groups=completed_groups,
    )
    return scheduler, data_source, executor


async def _wait_for_executions(executor: _ControlledExecutor, count: int) -> None:
    async def wait() -> None:
        while len(executor.executions) < count:
            await asyncio.sleep(0)

    await asyncio.wait_for(wait(), timeout=1)


async def _wait_for_requeued_groups(data_source: _RecordingDataSource, count: int) -> None:
    async def wait() -> None:
        while len(data_source.requeued) < count:
            await asyncio.sleep(0)

    await asyncio.wait_for(wait(), timeout=1)


def _session(
    *,
    batch_groups: int = 1,
    execution_samples: int = 1,
    retained_groups: int = 1,
    completed_groups: int = 1,
    cancel_completes: bool = True,
) -> tuple[FullyAsyncRolloutSession, _RecordingDataSource, _ControlledExecutor]:
    data_source = _RecordingDataSource()
    executor = _ControlledExecutor(cancel_completes=cancel_completes)
    session = FullyAsyncRolloutSession(
        args=Namespace(
            rollout_batch_size=batch_groups,
            n_samples_per_prompt=1,
            fully_async_max_execution_samples=execution_samples,
            fully_async_max_retained_groups=retained_groups,
            fully_async_max_completed_prefetch_groups=completed_groups,
            async_max_concurrent_samples=None,
        ),
        data_source=data_source,
        executor=executor,
    )
    return session, data_source, executor


def test_execution_outcomes_bind_terminal_results_to_exact_receipts() -> None:
    receipt = cast(ReservationExecutorReceipt, object())
    error = RuntimeError("execution failed")

    success = FullyAsyncExecutionSuccess(executor_receipt=receipt, samples=[])
    failure = FullyAsyncExecutionFailure(executor_receipt=receipt, error=error)
    retry = FullyAsyncExecutionRetry(
        executor_receipt=receipt,
        reason=FullyAsyncRetryReason.EXECUTION_ABORTED,
    )

    assert success == (receipt, [])
    assert failure == (receipt, error)
    assert retry == (receipt, FullyAsyncRetryReason.EXECUTION_ABORTED)
    assert list(FullyAsyncRetryReason) == [
        FullyAsyncRetryReason.EXECUTION_ABORTED,
        FullyAsyncRetryReason.CANCELLATION_REQUESTED,
    ]


def test_execution_interfaces_require_submit_terminal_cancellation_and_close() -> None:
    with pytest.raises(TypeError):
        FullyAsyncExecution()
    with pytest.raises(TypeError):
        FullyAsyncExecutor()
    assert issubclass(FullyAsyncTerminalPendingError, RuntimeError)


async def test_successful_batch_is_sorted_and_retained_until_commit() -> None:
    scheduler, data_source, executor = _scheduler(
        groups_per_batch=2,
        execution_samples=2,
        retained_groups=2,
        completed_groups=2,
    )
    acquire_task = asyncio.create_task(scheduler.acquire_batch())
    await _wait_for_executions(executor, 2)

    executor.executions[1].succeed()
    executor.executions[0].succeed()
    batch = await acquire_task

    assert [[sample.index for sample in group] for group in batch.samples] == [[0], [1]]
    assert data_source.acknowledged == []
    assert data_source.requeued == []

    scheduler.commit_batch(batch, rollout_id=7)

    assert data_source.acknowledged == [(data_source.issued[:2], 7)]
    await scheduler.close()


async def test_execution_capacity_limits_concurrent_submissions() -> None:
    scheduler, _, executor = _scheduler(
        groups_per_batch=2,
        execution_samples=1,
        retained_groups=2,
        completed_groups=2,
    )
    acquire_task = asyncio.create_task(scheduler.acquire_batch())
    await _wait_for_executions(executor, 1)
    await asyncio.sleep(0)
    assert len(executor.executions) == 1

    executor.executions[0].succeed()
    await _wait_for_executions(executor, 2)
    executor.executions[1].succeed()
    batch = await acquire_task
    scheduler.rollback_batch(batch)
    await scheduler.close()


async def test_retained_capacity_is_held_until_batch_settlement() -> None:
    scheduler, _, executor = _scheduler(
        groups_per_batch=1,
        execution_samples=3,
        retained_groups=2,
        completed_groups=2,
    )
    acquire_task = asyncio.create_task(scheduler.acquire_batch())
    await _wait_for_executions(executor, 1)
    executor.executions[0].succeed()
    batch = await acquire_task
    await _wait_for_executions(executor, 2)
    await asyncio.sleep(0)
    assert len(executor.executions) == 2

    scheduler.commit_batch(batch, rollout_id=8)
    await _wait_for_executions(executor, 3)
    await scheduler.close()


async def test_completed_capacity_halts_admission_until_batch_settlement() -> None:
    scheduler, _, executor = _scheduler(
        groups_per_batch=1,
        execution_samples=1,
        retained_groups=3,
        completed_groups=1,
    )
    first_acquire = asyncio.create_task(scheduler.acquire_batch())
    await _wait_for_executions(executor, 1)
    executor.executions[0].succeed()
    first_batch = await first_acquire

    await asyncio.sleep(0)
    assert len(executor.executions) == 1

    scheduler.commit_batch(first_batch, rollout_id=9)
    await _wait_for_executions(executor, 2)
    executor.executions[1].succeed()
    second_batch = await scheduler.acquire_batch()
    scheduler.rollback_batch(second_batch)
    await scheduler.close()


async def test_completed_capacity_requeues_terminal_overflow_without_cancelling_siblings() -> None:
    scheduler, data_source, executor = _scheduler(
        groups_per_batch=1,
        execution_samples=2,
        retained_groups=2,
        completed_groups=1,
    )
    first_acquire = asyncio.create_task(scheduler.acquire_batch())
    await _wait_for_executions(executor, 2)

    executor.executions[0].succeed()
    first_batch = await first_acquire
    executor.executions[1].succeed()

    async def wait_for_requeue() -> None:
        while not data_source.requeued:
            await asyncio.sleep(0)

    await asyncio.wait_for(wait_for_requeue(), timeout=1)

    assert data_source.requeued == [[data_source.issued[1]]]
    assert [execution.cancellation_requests for execution in executor.executions] == [0, 0]

    scheduler.commit_batch(first_batch, rollout_id=10)
    await scheduler.close()


async def test_managed_session_commits_a_train_batch_lease() -> None:
    session, data_source, executor = _session()
    acquire_task = asyncio.create_task(session.acquire_train_batch(rollout_id=21))
    await _wait_for_executions(executor, 1)
    executor.executions[0].succeed()
    lease = await acquire_task

    assert lease.output.samples == [executor.executions[0].reservation.samples]
    assert lease.output.metrics is None

    lease.commit()
    assert data_source.acknowledged == [([data_source.issued[0]], 21)]
    await session.close()


async def test_checkpoint_rejects_an_open_lease_then_succeeds_after_settlement() -> None:
    session, _, executor = _session()
    acquire_task = asyncio.create_task(session.acquire_train_batch(rollout_id=22))
    await _wait_for_executions(executor, 1)
    executor.executions[0].succeed()
    lease = await acquire_task

    with pytest.raises(RuntimeError) as checkpoint_error:
        await session.prepare_checkpoint(rollout_id=22)
    assert str(checkpoint_error.value) == "Cannot prepare checkpoint 22 with open train batch leases: [22]."

    lease.rollback(BatchRollbackReason.HANDOFF_FAILED)
    assert await session.prepare_checkpoint(rollout_id=22) is None
    await session.close()


async def test_close_rejects_an_open_lease_then_retries_after_settlement() -> None:
    session, data_source, executor = _session()
    acquire_task = asyncio.create_task(session.acquire_train_batch(rollout_id=23))
    await _wait_for_executions(executor, 1)
    executor.executions[0].succeed()
    lease = await acquire_task

    with pytest.raises(RuntimeError) as close_error:
        await session.close()
    assert str(close_error.value) == "Cannot close rollout session with open train batch leases: [23]."

    lease.rollback(BatchRollbackReason.HANDOFF_FAILED)
    await session.close()
    assert data_source.requeued == [[data_source.issued[0]]]


async def test_execution_failure_requeues_the_attempt_and_fails_loudly() -> None:
    session, data_source, executor = _session()
    acquire_task = asyncio.create_task(session.acquire_train_batch(rollout_id=24))
    await _wait_for_executions(executor, 1)
    failure = RuntimeError("execution failed")
    executor.executions[0].fail(failure)

    with pytest.raises(RuntimeError) as acquisition_error:
        await acquire_task
    assert acquisition_error.value is failure
    assert data_source.requeued == [[data_source.issued[0]]]
    await session.close()


async def test_terminal_retry_requeues_and_keeps_producing() -> None:
    session, data_source, executor = _session()
    acquire_task = asyncio.create_task(session.acquire_train_batch(rollout_id=25))
    await _wait_for_executions(executor, 1)
    first_execution = executor.executions[0]

    first_execution.retry(FullyAsyncRetryReason.EXECUTION_ABORTED)
    await _wait_for_executions(executor, 2)
    replay_execution = executor.executions[1]

    assert data_source.requeued == [[data_source.issued[0]]]
    assert replay_execution.reservation.reservation_id == first_execution.reservation.reservation_id
    assert replay_execution.reservation is not first_execution.reservation

    replay_execution.succeed()
    lease = await acquire_task
    lease.commit()
    await session.close()


async def test_close_cancels_active_execution_before_requeue_and_executor_close() -> None:
    session, data_source, executor = _session()
    acquire_task = asyncio.create_task(session.acquire_train_batch(rollout_id=26))
    await _wait_for_executions(executor, 1)
    execution = executor.executions[0]

    await session.close()

    assert execution.cancellation_requests == 1
    assert data_source.requeued == [[data_source.issued[0]]]
    assert executor.close_calls == 1
    with pytest.raises(RuntimeError) as acquisition_error:
        await acquire_task
    assert str(acquisition_error.value) == "Rollout session is closed."


async def test_close_retries_terminal_observation_without_releasing_ownership() -> None:
    data_source = _RecordingDataSource()
    executor = _RetryableTerminalExecutor()
    session = FullyAsyncRolloutSession(
        args=Namespace(
            rollout_batch_size=1,
            n_samples_per_prompt=1,
            fully_async_max_execution_samples=1,
            fully_async_max_retained_groups=1,
            fully_async_max_completed_prefetch_groups=1,
            async_max_concurrent_samples=None,
        ),
        data_source=data_source,
        executor=executor,
    )
    acquire_task = asyncio.create_task(session.acquire_train_batch(rollout_id=27))
    await _wait_for_executions(executor, 1)
    execution = cast(_RetryableTerminalExecution, executor.executions[0])

    with pytest.raises(FullyAsyncTerminalPendingError) as first_close_error:
        await session.close()
    assert str(first_close_error.value) == "terminal proof is pending"
    assert execution.observation_attempts == 1
    assert data_source.requeued == []
    assert data_source.outstanding == {data_source.issued[0].reservation_id: data_source.issued[0]}

    execution.complete_retry()
    await session.close()

    assert execution.observation_attempts == 2
    assert data_source.requeued == [[data_source.issued[0]]]
    assert executor.close_calls == 1
    with pytest.raises(RuntimeError):
        await acquire_task


async def test_mismatched_terminal_receipt_poison_retains_source_ownership() -> None:
    session, data_source, executor = _session()
    acquire_task = asyncio.create_task(session.acquire_train_batch(rollout_id=28))
    await _wait_for_executions(executor, 1)
    execution = executor.executions[0]
    execution.succeed(receipt=replace(execution.receipt))

    with pytest.raises(RuntimeError) as acquisition_error:
        await acquire_task
    with pytest.raises(RuntimeError) as close_error:
        await session.close()

    expected_error = "Execution receipt 0 did not return its exact terminal receipt."
    assert str(acquisition_error.value) == expected_error
    assert str(close_error.value) == expected_error
    assert data_source.requeued == []
    assert data_source.outstanding == {data_source.issued[0].reservation_id: data_source.issued[0]}
    assert executor.close_calls == 0


async def test_shutdown_retries_requeue_then_preserves_generation_failure() -> None:
    session, data_source, executor = _session(cancel_completes=False)
    acquire_task = asyncio.create_task(session.acquire_train_batch(rollout_id=29))
    await _wait_for_executions(executor, 1)
    execution = executor.executions[0]
    generation_error = RuntimeError("generation failed during shutdown")
    requeue_error = RuntimeError("terminal requeue failed")
    data_source.requeue_error = requeue_error

    first_close = asyncio.create_task(session.close())
    while execution.cancellation_requests == 0:
        await asyncio.sleep(0)
    execution.fail(generation_error)

    with pytest.raises(RuntimeError) as first_close_error:
        await first_close
    assert first_close_error.value is requeue_error
    assert data_source.requeued == []

    data_source.requeue_error = None
    with pytest.raises(RuntimeError) as retry_close_error:
        await session.close()
    assert retry_close_error.value is generation_error
    assert data_source.requeued == [[data_source.issued[0]]]
    assert executor.close_calls == 1

    await session.close()
    with pytest.raises(RuntimeError):
        await acquire_task


@pytest.mark.parametrize(
    ("execution_samples", "retained_groups"),
    [
        pytest.param(2, 1, id="before-retained-capacity"),
        pytest.param(1, 2, id="after-retained-capacity"),
    ],
)
async def test_quiesce_drains_admitted_execution_without_admitting_capacity_waiter(
    execution_samples: int,
    retained_groups: int,
) -> None:
    session, data_source, executor = _session(
        execution_samples=execution_samples,
        retained_groups=retained_groups,
        completed_groups=retained_groups,
    )
    acquire_task = asyncio.create_task(session.acquire_train_batch(rollout_id=31))
    await _wait_for_executions(executor, 1)

    quiesce_task = asyncio.create_task(session.quiesce_train_admission())
    await asyncio.sleep(0)

    assert quiesce_task.done() is False

    executor.executions[0].succeed()
    await quiesce_task
    lease = await acquire_task

    for _ in range(10):
        await asyncio.sleep(0)
    assert len(executor.executions) == 1
    assert list(data_source.outstanding.values()) == [executor.executions[0].reservation]
    assert lease.output.samples == [executor.executions[0].reservation.samples]

    lease.rollback(BatchRollbackReason.HANDOFF_FAILED)
    await session.close()


async def test_quiesce_and_resume_are_idempotent_admission_transitions() -> None:
    session, _, executor = _session(
        execution_samples=1,
        retained_groups=2,
        completed_groups=2,
    )
    first_acquire = asyncio.create_task(session.acquire_train_batch(rollout_id=32))
    await _wait_for_executions(executor, 1)

    quiesce_task = asyncio.create_task(session.quiesce_train_admission())
    executor.executions[0].succeed()
    await quiesce_task
    await session.quiesce_train_admission()
    first_lease = await first_acquire

    await session.resume_train_admission()
    await session.resume_train_admission()
    await _wait_for_executions(executor, 2)
    executor.executions[1].succeed()
    second_lease = await session.acquire_train_batch(rollout_id=33)

    assert first_lease.output.samples == [executor.executions[0].reservation.samples]
    assert second_lease.output.samples == [executor.executions[1].reservation.samples]

    first_lease.rollback(BatchRollbackReason.HANDOFF_FAILED)
    second_lease.rollback(BatchRollbackReason.HANDOFF_FAILED)
    await session.close()


async def test_concurrent_quiesce_callers_share_the_admission_transition() -> None:
    session, _, executor = _session()
    acquire_task = asyncio.create_task(session.acquire_train_batch(rollout_id=39))
    await _wait_for_executions(executor, 1)

    first_quiesce = asyncio.create_task(session.quiesce_train_admission())
    second_quiesce = asyncio.create_task(session.quiesce_train_admission())
    await asyncio.sleep(0)

    assert (first_quiesce.done(), second_quiesce.done()) == (False, False)

    executor.executions[0].succeed()
    assert await asyncio.gather(first_quiesce, second_quiesce) == [None, None]
    lease = await acquire_task
    lease.rollback(BatchRollbackReason.HANDOFF_FAILED)
    await session.close()


async def test_resume_during_quiescence_waits_then_reuses_released_capacity() -> None:
    session, _, executor = _session()
    first_acquire = asyncio.create_task(session.acquire_train_batch(rollout_id=40))
    await _wait_for_executions(executor, 1)

    quiesce_task = asyncio.create_task(session.quiesce_train_admission())
    resume_task = asyncio.create_task(session.resume_train_admission())
    await asyncio.sleep(0)

    assert (quiesce_task.done(), resume_task.done()) == (False, False)

    executor.executions[0].succeed()
    assert await asyncio.gather(quiesce_task, resume_task) == [None, None]
    first_lease = await first_acquire
    await asyncio.sleep(0)
    assert len(executor.executions) == 1

    first_lease.rollback(BatchRollbackReason.HANDOFF_FAILED)
    await _wait_for_executions(executor, 2)
    executor.executions[1].succeed()
    second_lease = await session.acquire_train_batch(rollout_id=41)

    assert second_lease.output.samples == [executor.executions[1].reservation.samples]

    second_lease.rollback(BatchRollbackReason.HANDOFF_FAILED)
    await session.close()


async def test_quiesce_retains_completed_group_until_close() -> None:
    session, data_source, executor = _session()
    acquire_task = asyncio.create_task(session.acquire_train_batch(rollout_id=34))
    await _wait_for_executions(executor, 1)
    acquire_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await acquire_task

    quiesce_task = asyncio.create_task(session.quiesce_train_admission())
    executor.executions[0].succeed()
    await quiesce_task

    assert data_source.requeued == []
    assert list(data_source.outstanding.values()) == [executor.executions[0].reservation]

    await session.close()

    assert data_source.requeued == [[executor.executions[0].reservation]]
    assert executor.close_calls == 1


async def test_quiesce_propagates_fatal_scheduler_error() -> None:
    session, data_source, executor = _session()
    acquire_task = asyncio.create_task(session.acquire_train_batch(rollout_id=35))
    await _wait_for_executions(executor, 1)
    failure = RuntimeError("execution failed while quiescing")

    quiesce_task = asyncio.create_task(session.quiesce_train_admission())
    executor.executions[0].fail(failure)

    with pytest.raises(RuntimeError) as quiesce_error:
        await quiesce_task
    with pytest.raises(RuntimeError) as acquisition_error:
        await acquire_task

    assert quiesce_error.value is failure
    assert acquisition_error.value is failure
    assert data_source.requeued == [[executor.executions[0].reservation]]
    await session.close()


async def test_fatal_quiescence_rolls_back_previously_completed_groups() -> None:
    session, data_source, executor = _session(
        batch_groups=2,
        execution_samples=2,
        retained_groups=2,
        completed_groups=2,
    )
    acquire_task = asyncio.create_task(session.acquire_train_batch(rollout_id=42))
    await _wait_for_executions(executor, 2)
    quiesce_task = asyncio.create_task(session.quiesce_train_admission())

    executor.executions[0].succeed()
    await asyncio.sleep(0)
    failure = RuntimeError("second execution failed while quiescing")
    executor.executions[1].fail(failure)

    with pytest.raises(RuntimeError) as quiesce_error:
        await quiesce_task
    with pytest.raises(RuntimeError) as acquisition_error:
        await acquire_task

    assert quiesce_error.value is failure
    assert acquisition_error.value is failure
    assert data_source.requeued == [
        [executor.executions[1].reservation],
        [executor.executions[0].reservation],
    ]
    assert data_source.outstanding == {}
    await session.close()


async def test_fatal_quiescence_rolls_back_ready_groups_after_watchers_finish(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, data_source, executor = _session(
        batch_groups=2,
        execution_samples=2,
        retained_groups=2,
        completed_groups=2,
    )
    scheduler = session._scheduler
    original_drain = scheduler._drain_admitted_executions
    drain_started = asyncio.Event()
    allow_drain = asyncio.Event()

    async def delayed_drain() -> None:
        drain_started.set()
        await allow_drain.wait()
        await original_drain()

    monkeypatch.setattr(scheduler, "_drain_admitted_executions", delayed_drain)
    acquire_task = asyncio.create_task(session.acquire_train_batch(rollout_id=47))
    await _wait_for_executions(executor, 2)
    quiesce_task = asyncio.create_task(session.quiesce_train_admission())
    await drain_started.wait()

    executor.executions[0].succeed()
    failure = RuntimeError("watchers finished before quiescence drain")
    executor.executions[1].fail(failure)
    while scheduler._watcher_tasks:
        await asyncio.sleep(0)

    allow_drain.set()
    with pytest.raises(RuntimeError) as quiesce_error:
        await quiesce_task
    with pytest.raises(RuntimeError) as acquisition_error:
        await acquire_task

    assert quiesce_error.value is failure
    assert acquisition_error.value is failure
    assert data_source.requeued == [
        [executor.executions[1].reservation],
        [executor.executions[0].reservation],
    ]
    assert data_source.outstanding == {}
    await session.close()


async def test_close_preempts_an_inflight_quiescence_drain() -> None:
    session, data_source, executor = _session()
    acquire_task = asyncio.create_task(session.acquire_train_batch(rollout_id=36))
    await _wait_for_executions(executor, 1)

    quiesce_task = asyncio.create_task(session.quiesce_train_admission())
    await asyncio.sleep(0)
    close_task = asyncio.create_task(session.close())

    await asyncio.wait_for(asyncio.gather(quiesce_task, close_task), timeout=1)

    assert executor.executions[0].cancellation_requests == 1
    assert data_source.requeued == [[executor.executions[0].reservation]]
    assert executor.close_calls == 1
    with pytest.raises(RuntimeError) as acquisition_error:
        await acquire_task
    assert str(acquisition_error.value) == "Rollout session is closed."


async def test_close_joins_quiescence_after_its_caller_is_cancelled() -> None:
    session, _, executor = _session(cancel_completes=False)
    acquire_task = asyncio.create_task(session.acquire_train_batch(rollout_id=43))
    await _wait_for_executions(executor, 1)

    quiesce_caller = asyncio.create_task(session.quiesce_train_admission())
    await asyncio.sleep(0)
    quiesce_caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await quiesce_caller

    close_task = asyncio.create_task(session.close())
    while executor.executions[0].cancellation_requests == 0:
        await asyncio.sleep(0)
    executor.executions[0].succeed()
    await close_task

    current_task = asyncio.current_task()
    pending_task_names = sorted(
        task.get_name() for task in asyncio.all_tasks() if task is not current_task and not task.done()
    )
    assert "fully-async-admission-quiescence" not in pending_task_names
    with pytest.raises(RuntimeError) as acquisition_error:
        await acquire_task
    assert str(acquisition_error.value) == "Rollout session is closed."


async def test_close_surfaces_unobserved_quiescence_failure_after_callers_are_cancelled() -> None:
    session, data_source, executor = _session()
    acquire_caller = asyncio.create_task(session.acquire_train_batch(rollout_id=44))
    await _wait_for_executions(executor, 1)

    quiesce_caller = asyncio.create_task(session.quiesce_train_admission())
    await asyncio.sleep(0)
    acquire_caller.cancel()
    quiesce_caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await acquire_caller
    with pytest.raises(asyncio.CancelledError):
        await quiesce_caller

    failure = RuntimeError("unobserved execution failure while quiescing")
    executor.executions[0].fail(failure)
    await _wait_for_requeued_groups(data_source, 1)

    with pytest.raises(RuntimeError) as close_error:
        await session.close()

    assert close_error.value is failure
    assert data_source.requeued == [[executor.executions[0].reservation]]
    assert data_source.outstanding == {}
    assert executor.close_calls == 1
    await session.close()


async def test_close_does_not_repeat_quiescence_failure_observed_by_acquisition() -> None:
    session, data_source, executor = _session()
    acquire_caller = asyncio.create_task(session.acquire_train_batch(rollout_id=45))
    await _wait_for_executions(executor, 1)

    quiesce_caller = asyncio.create_task(session.quiesce_train_admission())
    await asyncio.sleep(0)
    quiesce_caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await quiesce_caller

    failure = RuntimeError("execution failure observed by acquisition")
    executor.executions[0].fail(failure)

    with pytest.raises(RuntimeError) as acquisition_error:
        await acquire_caller
    assert acquisition_error.value is failure

    await session.close()

    assert data_source.requeued == [[executor.executions[0].reservation]]
    assert executor.close_calls == 1


async def test_close_does_not_repeat_quiescence_failure_observed_by_resume() -> None:
    session, data_source, executor = _session()
    acquire_caller = asyncio.create_task(session.acquire_train_batch(rollout_id=46))
    await _wait_for_executions(executor, 1)

    quiesce_caller = asyncio.create_task(session.quiesce_train_admission())
    resume_caller = asyncio.create_task(session.resume_train_admission())
    await asyncio.sleep(0)
    acquire_caller.cancel()
    quiesce_caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await acquire_caller
    with pytest.raises(asyncio.CancelledError):
        await quiesce_caller

    failure = RuntimeError("execution failure observed by resume")
    executor.executions[0].fail(failure)

    with pytest.raises(RuntimeError) as resume_error:
        await resume_caller
    assert resume_error.value is failure

    await session.close()

    assert data_source.requeued == [[executor.executions[0].reservation]]
    assert executor.close_calls == 1


async def test_cancelled_quiesce_caller_does_not_complete_the_transition() -> None:
    session, _, executor = _session()
    acquire_task = asyncio.create_task(session.acquire_train_batch(rollout_id=37))
    await _wait_for_executions(executor, 1)

    cancelled_quiesce = asyncio.create_task(session.quiesce_train_admission())
    await asyncio.sleep(0)
    cancelled_quiesce.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled_quiesce

    retry_quiesce = asyncio.create_task(session.quiesce_train_admission())
    await asyncio.sleep(0)
    assert retry_quiesce.done() is False

    executor.executions[0].succeed()
    await retry_quiesce
    lease = await acquire_task
    lease.rollback(BatchRollbackReason.HANDOFF_FAILED)
    await session.close()


async def test_resume_starts_admission_requested_during_initial_quiescence() -> None:
    session, _, executor = _session()
    await session.quiesce_train_admission()

    acquire_task = asyncio.create_task(session.acquire_train_batch(rollout_id=38))
    await asyncio.sleep(0)
    assert executor.executions == []

    await session.resume_train_admission()
    await _wait_for_executions(executor, 1)
    executor.executions[0].succeed()
    lease = await acquire_task

    assert lease.output.samples == [executor.executions[0].reservation.samples]

    lease.rollback(BatchRollbackReason.HANDOFF_FAILED)
    await session.close()


async def test_fully_async_session_supports_train_admission_control() -> None:
    session, _, _ = _session()

    assert session.supports_train_admission_control is True

    await session.close()
