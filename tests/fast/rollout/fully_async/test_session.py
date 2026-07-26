import asyncio
from argparse import Namespace
from collections.abc import Callable, Sequence
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
        self.acknowledge_error: BaseException | None = None
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
                reservation_id = SourceReservationId(str(self.next_reservation_id))
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
        self._validate_outstanding(attempts)
        if self.acknowledge_error is not None:
            raise self.acknowledge_error
        self._settle(attempts)
        self.acknowledged.append((attempts, rollout_id))

    def requeue_reservations(self, reservations: Sequence[SourceReservation]) -> None:
        attempts = list(reservations)
        self._validate_outstanding(attempts)
        if self.requeue_error is not None:
            raise self.requeue_error
        self._settle(attempts)
        self.replay_ids.extend(reservation.reservation_id for reservation in attempts)
        self.requeued.append(attempts)

    def _validate_outstanding(self, reservations: list[SourceReservation]) -> None:
        for reservation in reservations:
            if self.outstanding.get(reservation.reservation_id) is not reservation:
                raise RuntimeError(f"Reservation {reservation.reservation_id} is not outstanding.")

    def _settle(self, reservations: list[SourceReservation]) -> None:
        self._validate_outstanding(reservations)
        for reservation in reservations:
            del self.outstanding[reservation.reservation_id]


class _PerReservationRequeueDataSource(_RecordingDataSource):
    def __init__(self) -> None:
        super().__init__()
        self.requeue_errors: dict[SourceReservationId, BaseException] = {}

    def requeue_reservations(self, reservations: Sequence[SourceReservation]) -> None:
        attempts = list(reservations)
        self._validate_outstanding(attempts)
        for reservation in attempts:
            if error := self.requeue_errors.get(reservation.reservation_id):
                raise error
        super().requeue_reservations(attempts)


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


class _BlockingCloseExecutor(_ControlledExecutor):
    def __init__(self) -> None:
        super().__init__()
        self.close_started = asyncio.Event()
        self.allow_close = asyncio.Event()

    async def close(self) -> None:
        self.close_calls += 1
        self.close_started.set()
        await self.allow_close.wait()


class _FailingOnceCloseExecutor(_ControlledExecutor):
    def __init__(self, close_error: BaseException) -> None:
        super().__init__()
        self.close_error = close_error

    async def close(self) -> None:
        self.close_calls += 1
        if self.close_calls == 1:
            raise self.close_error


class _FailingCancellationExecution(_ControlledExecution):
    def __init__(
        self,
        reservation: SourceReservation,
        receipt: ReservationExecutorReceipt,
        *,
        cancellation_error: BaseException,
    ) -> None:
        super().__init__(reservation, receipt, cancel_completes=True)
        self.cancellation_error = cancellation_error

    def request_cancellation(self) -> None:
        self.cancellation_requests += 1
        if self.cancellation_requests == 1:
            raise self.cancellation_error
        if not self.result.done():
            self.retry(FullyAsyncRetryReason.CANCELLATION_REQUESTED)


class _FailingCancellationExecutor(_ControlledExecutor):
    def __init__(self, cancellation_error: BaseException) -> None:
        super().__init__()
        self.cancellation_error = cancellation_error

    def submit(
        self,
        reservation: SourceReservation,
        receipt: ReservationExecutorReceipt,
    ) -> FullyAsyncExecution:
        execution = _FailingCancellationExecution(
            reservation,
            receipt,
            cancellation_error=self.cancellation_error,
        )
        self.executions.append(execution)
        return execution


class _CoordinatedRetryCancellationExecution(_ControlledExecution):
    def __init__(
        self,
        reservation: SourceReservation,
        receipt: ReservationExecutorReceipt,
        *,
        cancellation_error: BaseException,
        all_executions: list[_ControlledExecution],
    ) -> None:
        super().__init__(reservation, receipt, cancel_completes=False)
        self.cancellation_error = cancellation_error
        self.all_executions = all_executions

    def request_cancellation(self) -> None:
        self.cancellation_requests += 1
        if self.cancellation_requests == 1:
            raise self.cancellation_error
        if all(execution.cancellation_requests >= 2 for execution in self.all_executions):
            for execution in self.all_executions:
                if not execution.result.done():
                    execution.retry(FullyAsyncRetryReason.CANCELLATION_REQUESTED)


class _CoordinatedRetryCancellationExecutor(_ControlledExecutor):
    def __init__(self) -> None:
        super().__init__(cancel_completes=False)
        self.cancellation_errors: list[RuntimeError] = []

    def submit(
        self,
        reservation: SourceReservation,
        receipt: ReservationExecutorReceipt,
    ) -> FullyAsyncExecution:
        cancellation_error = RuntimeError(f"cancellation request {len(self.executions)} failed")
        self.cancellation_errors.append(cancellation_error)
        execution = _CoordinatedRetryCancellationExecution(
            reservation,
            receipt,
            cancellation_error=cancellation_error,
            all_executions=self.executions,
        )
        self.executions.append(execution)
        return execution

    def complete_all(self) -> None:
        for execution in self.executions:
            if not execution.result.done():
                execution.retry(FullyAsyncRetryReason.CANCELLATION_REQUESTED)


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

    def complete_success(self, samples: list[Sample]) -> None:
        self.retry_result.set_result(
            FullyAsyncExecutionSuccess(
                executor_receipt=self.receipt,
                samples=samples,
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


async def _wait_until(predicate: Callable[[], bool]) -> None:
    async def wait() -> None:
        while not predicate():
            await asyncio.sleep(0)

    await asyncio.wait_for(wait(), timeout=1)


async def _wait_for_executions(executor: _ControlledExecutor, count: int) -> None:
    await _wait_until(lambda: len(executor.executions) >= count)


async def _wait_for_requeued_groups(data_source: _RecordingDataSource, count: int) -> None:
    await _wait_until(lambda: len(data_source.requeued) >= count)


def _session(
    *,
    execution_samples: int = 1,
    retained_groups: int = 1,
    completed_groups: int = 1,
    cancel_completes: bool = True,
) -> tuple[FullyAsyncRolloutSession, _RecordingDataSource, _ControlledExecutor]:
    data_source = _RecordingDataSource()
    executor = _ControlledExecutor(cancel_completes=cancel_completes)
    session = FullyAsyncRolloutSession(
        args=_session_args(
            execution_samples=execution_samples,
            retained_groups=retained_groups,
            completed_groups=completed_groups,
        ),
        data_source=data_source,
        executor=executor,
    )
    return session, data_source, executor


def _session_args(
    *,
    execution_samples: int = 1,
    retained_groups: int = 1,
    completed_groups: int = 1,
) -> Namespace:
    return Namespace(
        rollout_batch_size=1,
        n_samples_per_prompt=1,
        fully_async_max_execution_samples=execution_samples,
        fully_async_max_retained_groups=retained_groups,
        fully_async_max_completed_prefetch_groups=completed_groups,
        async_max_concurrent_samples=None,
    )


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
    execution_type = cast(type[object], FullyAsyncExecution)
    executor_type = cast(type[object], FullyAsyncExecutor)
    with pytest.raises(TypeError):
        execution_type()
    with pytest.raises(TypeError):
        executor_type()
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
    await session.prepare_checkpoint(rollout_id=22)
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


async def test_cancelled_batch_acquisition_does_not_consume_a_ready_group() -> None:
    session, _, executor = _session()
    cancelled_acquire = asyncio.create_task(session.acquire_train_batch(rollout_id=30))
    await _wait_for_executions(executor, 1)
    cancelled_acquire.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled_acquire

    executor.executions[0].succeed()
    lease = await session.acquire_train_batch(rollout_id=31)

    assert lease.output.samples == [executor.executions[0].reservation.samples]
    lease.rollback(BatchRollbackReason.HANDOFF_FAILED)
    await session.close()


async def test_prepare_checkpoint_waits_for_an_inflight_batch_acquisition() -> None:
    session, _, executor = _session()
    acquire_task = asyncio.create_task(session.acquire_train_batch(rollout_id=32))
    await _wait_for_executions(executor, 1)

    checkpoint_task = asyncio.create_task(session.prepare_checkpoint(rollout_id=32))
    await asyncio.sleep(0)

    assert checkpoint_task.done() is False

    executor.executions[0].succeed()
    lease = await acquire_task
    with pytest.raises(RuntimeError) as checkpoint_error:
        await checkpoint_task
    assert str(checkpoint_error.value) == "Cannot prepare checkpoint 32 with open train batch leases: [32]."

    lease.rollback(BatchRollbackReason.HANDOFF_FAILED)
    await session.close()


@pytest.mark.parametrize("settlement", ["commit", "rollback"])
async def test_failed_lease_settlement_is_terminal_and_retains_capacity(settlement: str) -> None:
    session, data_source, executor = _session()
    acquire_task = asyncio.create_task(session.acquire_train_batch(rollout_id=33))
    await _wait_for_executions(executor, 1)
    executor.executions[0].succeed()
    lease = await acquire_task

    failure = RuntimeError(f"{settlement} failed")
    if settlement == "commit":
        data_source.acknowledge_error = failure
        with pytest.raises(RuntimeError) as error:
            lease.commit()
        assert error.value is failure
        with pytest.raises(RuntimeError) as retry_error:
            lease.commit()
        assert str(retry_error.value) == "Train batch lease for rollout 33 is already commit failed."
        with pytest.raises(RuntimeError) as opposite_error:
            lease.rollback(BatchRollbackReason.HANDOFF_FAILED)
        assert str(opposite_error.value) == "Train batch lease for rollout 33 is already commit failed."
    else:
        data_source.requeue_error = failure
        with pytest.raises(RuntimeError) as error:
            lease.rollback(BatchRollbackReason.HANDOFF_FAILED)
        assert error.value is failure
        with pytest.raises(RuntimeError) as retry_error:
            lease.rollback(BatchRollbackReason.HANDOFF_FAILED)
        assert str(retry_error.value) == "Train batch lease for rollout 33 is already rollback failed."
        with pytest.raises(RuntimeError) as opposite_error:
            lease.commit()
        assert str(opposite_error.value) == "Train batch lease for rollout 33 is already rollback failed."

    await asyncio.sleep(0)
    assert len(executor.executions) == 1

    with pytest.raises(RuntimeError) as close_error:
        await session.close()
    assert str(close_error.value) == "Cannot close rollout session with open train batch leases: [33]."
    assert executor.close_calls == 0


async def test_mismatched_terminal_sample_identity_requeues_the_original_reservation() -> None:
    session, data_source, executor = _session()
    acquire_task = asyncio.create_task(session.acquire_train_batch(rollout_id=34))
    await _wait_for_executions(executor, 1)
    executor.executions[0].succeed(
        samples=[
            Sample(
                group_index=99,
                index=99,
                prompt="wrong prompt",
            )
        ]
    )

    with pytest.raises(ValueError) as acquisition_error:
        await acquire_task

    assert (
        str(acquisition_error.value) == "Execution receipt 0 returned sample identities [(99, 99)]; expected [(0, 0)]."
    )
    assert data_source.requeued == [[data_source.issued[0]]]
    assert data_source.outstanding == {}
    await session.close()


async def test_non_sample_terminal_payload_requeues_the_original_reservation() -> None:
    session, data_source, executor = _session()
    acquire_task = asyncio.create_task(session.acquire_train_batch(rollout_id=35))
    await _wait_for_executions(executor, 1)
    executor.executions[0].succeed(
        samples=cast(
            list[Sample],
            [Namespace(group_index=0, index=0)],
        )
    )

    with pytest.raises(ValueError) as acquisition_error:
        await acquire_task

    assert str(acquisition_error.value) == "Execution receipt 0 returned non-Sample values at positions [0]."
    assert data_source.requeued == [[data_source.issued[0]]]
    assert data_source.outstanding == {}
    await session.close()


async def test_mismatched_terminal_sample_count_requeues_the_original_reservation() -> None:
    session, data_source, executor = _session()
    acquire_task = asyncio.create_task(session.acquire_train_batch(rollout_id=36))
    await _wait_for_executions(executor, 1)
    executor.executions[0].succeed(samples=[])

    with pytest.raises(ValueError) as acquisition_error:
        await acquire_task

    assert str(acquisition_error.value) == "Execution receipt 0 returned 0 samples; expected 1."
    assert data_source.requeued == [[data_source.issued[0]]]
    assert data_source.outstanding == {}
    await session.close()


async def test_executor_cannot_mutate_the_source_reservation_requeued_for_replay() -> None:
    session, data_source, executor = _session()
    acquire_task = asyncio.create_task(session.acquire_train_batch(rollout_id=37))
    await _wait_for_executions(executor, 1)
    execution = executor.executions[0]
    expected_reservation = SourceReservation(
        reservation_id=SourceReservationId("0"),
        samples=[Sample(group_index=0, index=0, prompt="prompt-0")],
    )
    execution.reservation.samples[0].group_index = 99
    execution.reservation.samples[0].index = 99
    execution.reservation.samples[0].prompt = "mutated"
    execution.reservation.samples[0].metadata["attempt"] = "mutated"
    execution.succeed()

    with pytest.raises(ValueError) as acquisition_error:
        await acquire_task
    assert (
        str(acquisition_error.value) == "Execution receipt 0 returned sample identities [(99, 99)]; expected [(0, 0)]."
    )

    await session.close()

    assert data_source.requeued == [[expected_reservation]]
    assert data_source.outstanding == {}


async def test_unsolicited_cancellation_retry_fails_loudly() -> None:
    session, data_source, executor = _session()
    acquire_task = asyncio.create_task(session.acquire_train_batch(rollout_id=38))
    await _wait_for_executions(executor, 1)
    execution = executor.executions[0]

    execution.retry(FullyAsyncRetryReason.CANCELLATION_REQUESTED)

    with pytest.raises(RuntimeError) as acquisition_error:
        await acquire_task
    assert (
        str(acquisition_error.value)
        == "Execution receipt 0 reported a cancellation retry before the scheduler requested it."
    )
    assert data_source.requeued == [[data_source.issued[0]]]
    await session.close()


async def test_unsolicited_execution_cancellation_fails_and_cancels_siblings() -> None:
    session, data_source, executor = _session(
        execution_samples=2,
        retained_groups=2,
        completed_groups=2,
    )
    acquire_task = asyncio.create_task(session.acquire_train_batch(rollout_id=39))
    await _wait_for_executions(executor, 2)
    executor.executions[0].fail(asyncio.CancelledError())

    with pytest.raises(RuntimeError) as acquisition_error:
        await asyncio.wait_for(acquire_task, timeout=1)
    assert (
        str(acquisition_error.value) == "Execution receipt 0 reported cancellation before the scheduler requested it."
    )
    await _wait_until(lambda: executor.executions[1].cancellation_requests > 0)
    await _wait_for_requeued_groups(data_source, 2)
    assert data_source.requeued == [
        [data_source.issued[0]],
        [data_source.issued[1]],
    ]
    await session.close()


async def test_close_retries_a_failed_cancellation_request_before_settlement() -> None:
    data_source = _RecordingDataSource()
    cancellation_error = RuntimeError("cancellation request failed")
    executor = _FailingCancellationExecutor(cancellation_error)
    session = FullyAsyncRolloutSession(
        args=_session_args(),
        data_source=data_source,
        executor=executor,
    )
    acquire_task = asyncio.create_task(session.acquire_train_batch(rollout_id=40))
    await _wait_for_executions(executor, 1)
    execution = executor.executions[0]

    with pytest.raises(RuntimeError) as first_close_error:
        await session.close()
    assert first_close_error.value is cancellation_error
    assert execution.cancellation_requests == 1
    assert data_source.requeued == []
    assert executor.close_calls == 0

    await session.close()

    assert execution.cancellation_requests == 2
    assert data_source.requeued == [[data_source.issued[0]]]
    assert executor.close_calls == 1
    with pytest.raises(RuntimeError) as acquisition_error:
        await acquire_task
    assert str(acquisition_error.value) == "Rollout session is closed."


async def test_close_retries_all_cancellation_requests_before_waiting_for_receipts() -> None:
    data_source = _RecordingDataSource()
    executor = _CoordinatedRetryCancellationExecutor()
    session = FullyAsyncRolloutSession(
        args=_session_args(
            execution_samples=2,
            retained_groups=2,
            completed_groups=2,
        ),
        data_source=data_source,
        executor=executor,
    )
    acquire_task = asyncio.create_task(session.acquire_train_batch(rollout_id=41))
    await _wait_for_executions(executor, 2)

    with pytest.raises(RuntimeError) as first_close_error:
        await session.close()
    assert first_close_error.value in executor.cancellation_errors
    assert [execution.cancellation_requests for execution in executor.executions] == [1, 1]

    retry_close = asyncio.create_task(session.close())
    try:
        await _wait_until(lambda: all(execution.cancellation_requests == 2 for execution in executor.executions))
        assert [execution.cancellation_requests for execution in executor.executions] == [2, 2]
    finally:
        executor.complete_all()
        await asyncio.gather(retry_close, acquire_task, return_exceptions=True)

    assert retry_close.result() is None
    assert data_source.requeued == [[data_source.issued[0]], [data_source.issued[1]]]
    assert data_source.outstanding == {}
    assert executor.close_calls == 1


async def test_close_reports_only_an_error_that_remains_unresolved() -> None:
    data_source = _PerReservationRequeueDataSource()
    executor = _ControlledExecutor()
    session = FullyAsyncRolloutSession(
        args=_session_args(
            execution_samples=2,
            retained_groups=2,
            completed_groups=2,
        ),
        data_source=data_source,
        executor=executor,
    )
    acquire_task = asyncio.create_task(session.acquire_train_batch(rollout_id=42))
    await _wait_for_executions(executor, 2)
    first_error = RuntimeError("first requeue failed")
    second_error = RuntimeError("second requeue failed")
    errors_by_reservation = {
        data_source.issued[0].reservation_id: first_error,
        data_source.issued[1].reservation_id: second_error,
    }
    data_source.requeue_errors.update(errors_by_reservation)

    with pytest.raises(RuntimeError) as initial_close_error:
        await session.close()
    assert initial_close_error.value in (first_error, second_error)
    resolved_reservation_id = next(
        reservation_id for reservation_id, error in errors_by_reservation.items() if error is initial_close_error.value
    )
    del data_source.requeue_errors[resolved_reservation_id]

    with pytest.raises(RuntimeError) as retry_close_error:
        await session.close()
    remaining_error = next(iter(data_source.requeue_errors.values()))
    assert retry_close_error.value is remaining_error
    assert executor.close_calls == 0

    data_source.requeue_errors.clear()
    await session.close()

    assert data_source.outstanding == {}
    assert executor.close_calls == 1
    with pytest.raises(RuntimeError) as acquisition_error:
        await acquire_task
    assert str(acquisition_error.value) == "Rollout session is closed."


async def test_cancelled_close_caller_still_waits_for_terminal_teardown() -> None:
    session, data_source, executor = _session(cancel_completes=False)
    acquire_task = asyncio.create_task(session.acquire_train_batch(rollout_id=43))
    await _wait_for_executions(executor, 1)

    close_task = asyncio.create_task(session.close())
    await _wait_until(lambda: executor.executions[0].cancellation_requests > 0)
    close_task.cancel()
    await asyncio.sleep(0)

    assert close_task.done() is False
    assert data_source.requeued == []

    executor.executions[0].succeed()
    with pytest.raises(asyncio.CancelledError):
        await close_task

    assert data_source.requeued == [[data_source.issued[0]]]
    assert executor.close_calls == 1
    with pytest.raises(RuntimeError) as acquisition_error:
        await acquire_task
    assert str(acquisition_error.value) == "Rollout session is closed."
    await session.close()
    assert executor.close_calls == 1


async def test_concurrent_close_callers_share_one_teardown() -> None:
    data_source = _RecordingDataSource()
    executor = _BlockingCloseExecutor()
    session = FullyAsyncRolloutSession(
        args=_session_args(),
        data_source=data_source,
        executor=executor,
    )

    first_close = asyncio.create_task(session.close())
    await asyncio.wait_for(executor.close_started.wait(), timeout=1)
    second_close = asyncio.create_task(session.close())
    await asyncio.sleep(0)

    assert executor.close_calls == 1

    executor.allow_close.set()
    assert await asyncio.gather(first_close, second_close) == [None, None]
    assert executor.close_calls == 1


async def test_malformed_execution_success_during_shutdown_fails_loudly() -> None:
    session, data_source, executor = _session(cancel_completes=False)
    acquire_task = asyncio.create_task(session.acquire_train_batch(rollout_id=44))
    await _wait_for_executions(executor, 1)
    execution = executor.executions[0]

    close_task = asyncio.create_task(session.close())
    await _wait_until(lambda: execution.cancellation_requests > 0)
    execution.succeed(samples=[])

    try:
        with pytest.raises(ValueError) as close_error:
            await close_task
        assert str(close_error.value) == "Execution receipt 0 returned 0 samples; expected 1."
    finally:
        await asyncio.gather(close_task, acquire_task, return_exceptions=True)

    assert data_source.requeued == [[data_source.issued[0]]]
    assert data_source.outstanding == {}
    assert executor.close_calls == 1
    await session.close()
    assert executor.close_calls == 1


async def test_close_validates_success_after_retrying_terminal_observation() -> None:
    data_source = _RecordingDataSource()
    executor = _RetryableTerminalExecutor()
    session = FullyAsyncRolloutSession(
        args=_session_args(),
        data_source=data_source,
        executor=executor,
    )
    acquire_task = asyncio.create_task(session.acquire_train_batch(rollout_id=45))
    await _wait_for_executions(executor, 1)
    execution = cast(_RetryableTerminalExecution, executor.executions[0])

    with pytest.raises(FullyAsyncTerminalPendingError) as first_close_error:
        await session.close()
    assert str(first_close_error.value) == "terminal proof is pending"

    execution.complete_success([])
    try:
        with pytest.raises(ValueError) as close_error:
            await session.close()
        assert str(close_error.value) == "Execution receipt 0 returned 0 samples; expected 1."
    finally:
        await asyncio.gather(acquire_task, return_exceptions=True)

    with pytest.raises(RuntimeError) as acquisition_error:
        await acquire_task
    assert str(acquisition_error.value) == "Rollout session is closed."
    assert execution.observation_attempts == 2
    assert data_source.requeued == [[data_source.issued[0]]]
    assert data_source.outstanding == {}
    assert executor.close_calls == 1
    await session.close()
    assert executor.close_calls == 1


async def test_close_retries_executor_close_after_a_transient_failure() -> None:
    data_source = _RecordingDataSource()
    close_error = RuntimeError("executor close failed")
    executor = _FailingOnceCloseExecutor(close_error)
    session = FullyAsyncRolloutSession(
        args=_session_args(),
        data_source=data_source,
        executor=executor,
    )

    with pytest.raises(RuntimeError) as first_close_error:
        await session.close()
    assert first_close_error.value is close_error
    assert executor.close_calls == 1
    assert data_source.outstanding == {}
    assert data_source.acknowledged == []
    assert data_source.requeued == []

    await session.close()

    assert executor.close_calls == 2
    assert data_source.outstanding == {}
    assert data_source.acknowledged == []
    assert data_source.requeued == []

    await session.close()
    assert executor.close_calls == 2
