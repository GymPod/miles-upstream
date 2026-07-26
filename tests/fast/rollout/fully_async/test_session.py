import asyncio
from collections.abc import Sequence
from typing import cast

import pytest

from miles.rollout.data_source import DataSource, SourceReservation, SourceReservationId
from miles.rollout.fully_async.execution import (
    FullyAsyncExecution,
    FullyAsyncExecutionFailure,
    FullyAsyncExecutionSuccess,
    FullyAsyncExecutor,
)
from miles.rollout.fully_async.ownership import ReservationExecutorReceipt
from miles.rollout.fully_async.scheduler import _FullyAsyncScheduler
from miles.utils.types import Sample


class _RecordingDataSource(DataSource):
    def __init__(self) -> None:
        self.next_reservation_id = 0
        self.outstanding: dict[SourceReservationId, SourceReservation] = {}
        self.acknowledged: list[tuple[list[SourceReservation], int]] = []
        self.requeued: list[list[SourceReservation]] = []

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
        self._settle(attempts)
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
    ) -> None:
        self.reservation = reservation
        self.receipt = receipt
        self.result: asyncio.Future[
            FullyAsyncExecutionSuccess | FullyAsyncExecutionFailure
        ] = asyncio.get_running_loop().create_future()

    def request_cancellation(self) -> None:
        if not self.result.done():
            self.result.set_result(
                FullyAsyncExecutionFailure(
                    executor_receipt=self.receipt,
                    error=asyncio.CancelledError(),
                )
            )

    async def wait_terminal(self) -> FullyAsyncExecutionSuccess | FullyAsyncExecutionFailure:
        return await self.result

    def succeed(self) -> None:
        self.result.set_result(
            FullyAsyncExecutionSuccess(
                executor_receipt=self.receipt,
                samples=self.reservation.samples,
            )
        )


class _ControlledExecutor(FullyAsyncExecutor):
    def __init__(self) -> None:
        self.executions: list[_ControlledExecution] = []

    def submit(
        self,
        reservation: SourceReservation,
        receipt: ReservationExecutorReceipt,
    ) -> FullyAsyncExecution:
        execution = _ControlledExecution(reservation, receipt)
        self.executions.append(execution)
        return execution

    async def close(self) -> None:
        return


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


def test_execution_outcomes_bind_terminal_results_to_exact_receipts() -> None:
    receipt = cast(ReservationExecutorReceipt, object())
    error = RuntimeError("execution failed")

    success = FullyAsyncExecutionSuccess(executor_receipt=receipt, samples=[])
    failure = FullyAsyncExecutionFailure(executor_receipt=receipt, error=error)

    assert success == (receipt, [])
    assert failure == (receipt, error)


def test_execution_interfaces_require_submit_terminal_cancellation_and_close() -> None:
    with pytest.raises(TypeError):
        FullyAsyncExecution()
    with pytest.raises(TypeError):
        FullyAsyncExecutor()


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

    assert data_source.acknowledged == [
        ([executor.executions[0].reservation, executor.executions[1].reservation], 7)
    ]
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

    await _wait_for_executions(executor, 2)
    executor.executions[1].succeed()
    await asyncio.sleep(0)
    assert len(executor.executions) == 2

    scheduler.commit_batch(first_batch, rollout_id=9)
    await _wait_for_executions(executor, 3)
    second_batch = await scheduler.acquire_batch()
    scheduler.rollback_batch(second_batch)
    await scheduler.close()
