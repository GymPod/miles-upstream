import asyncio
from copy import deepcopy
from dataclasses import dataclass

from miles.rollout.data_source import DataSource, SourceReservation
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
from miles.rollout.fully_async.ownership import (
    ReservationExecutorReceipt,
    ReservationOwnership,
    ReservationReceiptId,
    ReservationStageId,
    ReservationTerminalDisposition,
    ReservationTerminalReceipt,
)
from miles.utils.types import Sample


@dataclass
class _ExecutionRecord:
    reservation: SourceReservation
    source_sample_identities: tuple[tuple[int | None, int | None], ...]
    receipt: ReservationExecutorReceipt
    execution: FullyAsyncExecution | None
    terminal_task: asyncio.Task[FullyAsyncExecutionOutcome] | None
    cancellation_requested: bool = False
    terminal_observed: bool = False


@dataclass(frozen=True)
class _ReadyGroup:
    receipt: ReservationTerminalReceipt
    samples: list[Sample]


@dataclass(frozen=True)
class _SchedulerBatch:
    groups: tuple[_ReadyGroup, ...]

    @property
    def samples(self) -> list[list[Sample]]:
        return [group.samples for group in self.groups]


class _FullyAsyncScheduler:
    def __init__(
        self,
        *,
        data_source: DataSource,
        executor: FullyAsyncExecutor,
        samples_per_group: int,
        groups_per_batch: int,
        max_execution_samples: int,
        max_retained_groups: int,
        max_completed_groups: int,
    ) -> None:
        self._ownership = ReservationOwnership(data_source)
        self._executor = executor
        self._samples_per_group = samples_per_group
        self._groups_per_batch = groups_per_batch
        self._execution_slots = asyncio.BoundedSemaphore(max_execution_samples // samples_per_group)
        self._retained_slots = asyncio.BoundedSemaphore(max_retained_groups)
        self._completed_slots = asyncio.BoundedSemaphore(max_completed_groups)
        self._completed_capacity_available = asyncio.Event()
        self._completed_capacity_available.set()
        self._active: dict[ReservationReceiptId, _ExecutionRecord] = {}
        self._ready: list[_ReadyGroup] = []
        self._ready_changed = asyncio.Event()
        self._acquire_lock = asyncio.Lock()
        self._admission_transition_lock = asyncio.Lock()
        self._quiesce_task: asyncio.Task[None] | None = None
        self._quiesce_failure_reported = False
        self._producer_task: asyncio.Task[None] | None = None
        self._producer_requested = False
        self._watcher_tasks: set[asyncio.Task[None]] = set()
        self._next_stage_id = 0
        self._accepting = True
        self._closing = False
        self._fatal_error: BaseException | None = None
        self._shutdown_failure: BaseException | None = None
        self._cleanup_error: BaseException | None = None
        self._pending_acquisition_capacity = False
        self._pending_reserved_rollbacks: list[SourceReservation] = []

    async def acquire_batch(self) -> _SchedulerBatch:
        async with self._acquire_lock:
            self._ensure_started()
            while True:
                self.raise_if_failed()
                if self._closing:
                    raise RuntimeError("Fully async scheduler is closing.")
                if len(self._ready) >= self._groups_per_batch:
                    groups = tuple(
                        sorted(
                            self._ready[: self._groups_per_batch],
                            key=lambda group: group.samples[0].index,
                        )
                    )
                    del self._ready[: self._groups_per_batch]
                    return _SchedulerBatch(groups=groups)
                self._ready_changed.clear()
                await self._ready_changed.wait()

    def commit_batch(self, batch: _SchedulerBatch, *, rollout_id: int) -> None:
        self._ownership.commit_batch(
            [group.receipt for group in batch.groups],
            rollout_id=rollout_id,
        )
        self._release_settled_batch(batch)

    def rollback_batch(self, batch: _SchedulerBatch) -> None:
        self._ownership.rollback_batch([group.receipt for group in batch.groups])
        self._release_settled_batch(batch)

    def raise_if_failed(self) -> None:
        if self._fatal_error is not None:
            if self._quiesce_task is not None:
                self._quiesce_failure_reported = True
            raise self._fatal_error

    @property
    def has_active_executions(self) -> bool:
        return bool(self._active)

    async def quiesce_train_admission(self) -> None:
        async with self._admission_transition_lock:
            self.raise_if_failed()
            if self._closing:
                raise RuntimeError("Fully async scheduler is closing.")

            quiesce_task = self._quiesce_task
            if quiesce_task is None:
                self._accepting = False
                producer_task = self._producer_task
                if producer_task is not None and not producer_task.done():
                    producer_task.cancel()
                quiesce_task = asyncio.create_task(
                    self._finish_quiescence(producer_task),
                    name="fully-async-admission-quiescence",
                )
                self._quiesce_task = quiesce_task
                self._quiesce_failure_reported = False

        await self._await_quiescence(quiesce_task)

    async def resume_train_admission(self) -> None:
        while True:
            async with self._admission_transition_lock:
                self.raise_if_failed()
                if self._closing:
                    raise RuntimeError("Fully async scheduler is closing.")
                quiesce_task = self._quiesce_task
                if quiesce_task is not None and quiesce_task.done():
                    quiesce_task.result()
                    self._accepting = True
                    self._quiesce_task = None
                    if self._producer_requested:
                        self._start_producer()
                    return
                if quiesce_task is None:
                    return

            await self._await_quiescence(quiesce_task)

    async def close(self) -> None:
        async with self._admission_transition_lock:
            self._closing = True
            self._accepting = False
            self._ready_changed.set()

        producer_task = self._producer_task
        if producer_task is not None and not producer_task.done():
            producer_was_running = True
            producer_task.cancel()
        else:
            producer_was_running = False
        if producer_task is not None:
            await _await_task_completion(producer_task)

        if not producer_was_running:
            self._cleanup_error = None
        await self._stop_watcher_tasks()
        quiesce_task = self._quiesce_task
        quiesce_failure: BaseException | None = None
        if quiesce_task is not None:
            try:
                await _await_task_completion(quiesce_task)
            except BaseException as error:
                quiesce_failure = error
        self._retry_pending_acquisition_rollback()
        self._retry_reserved_rollbacks()
        if not producer_was_running:
            await self._retry_active_shutdown()
        self._rollback_ready()

        if self._pending_acquisition_capacity or self._pending_reserved_rollbacks or self._active or self._ready:
            if self._cleanup_error is not None:
                raise self._cleanup_error
            raise RuntimeError("Fully async scheduler cleanup did not settle all retained groups.")
        self._cleanup_error = None
        if self._shutdown_failure is not None:
            shutdown_failure = self._shutdown_failure
            self._shutdown_failure = None
            if quiesce_failure is shutdown_failure:
                self._quiesce_failure_reported = True
            raise shutdown_failure
        if quiesce_failure is not None and not self._quiesce_failure_reported:
            self._quiesce_failure_reported = True
            raise quiesce_failure

    def _ensure_started(self) -> None:
        if self._closing:
            raise RuntimeError("Fully async scheduler is closing.")
        self._producer_requested = True
        if self._accepting and (self._producer_task is None or self._producer_task.done()):
            self._start_producer()

    def _start_producer(self) -> None:
        self._producer_task = asyncio.create_task(
            self._run_producer(),
            name="fully-async-rollout-producer",
        )

    async def _run_producer(self) -> None:
        try:
            while self._accepting:
                await self._admit_one()
        except asyncio.CancelledError as cancellation:
            if self._accepting and not self._closing and self._fatal_error is None:
                self._record_fatal(cancellation)
        except BaseException as error:
            if self._fatal_error is None:
                self._record_fatal(error)
        finally:
            if self._closing or self._fatal_error is not None:
                await self._stop_watcher_tasks()
                try:
                    self._rollback_ready()
                except BaseException as error:
                    self._record_cleanup_error(error)
            self._ready_changed.set()

    async def _admit_one(self) -> None:
        await self._completed_capacity_available.wait()
        await self._retained_slots.acquire()
        try:
            await self._execution_slots.acquire()
        except BaseException:
            self._retained_slots.release()
            raise

        try:
            await self._admission_transition_lock.acquire()
        except BaseException:
            self._execution_slots.release()
            self._retained_slots.release()
            raise

        try:
            if not self._accepting or not self._completed_capacity_available.is_set():
                self._execution_slots.release()
                self._retained_slots.release()
                return

            try:
                [reservation] = self._ownership.reserve_samples(1)
            except BaseException:
                if self._ownership.has_pending_acquisition_rollback:
                    self._pending_acquisition_capacity = True
                else:
                    self._execution_slots.release()
                    self._retained_slots.release()
                raise
        finally:
            self._admission_transition_lock.release()

        if len(reservation.samples) != self._samples_per_group:
            try:
                self._ownership.rollback_reserved([reservation])
            except BaseException as rollback_error:
                self._pending_reserved_rollbacks.append(reservation)
                self._record_cleanup_error(rollback_error)
            else:
                self._execution_slots.release()
                self._retained_slots.release()
            raise ValueError(
                f"Reservation {reservation.reservation_id} contains {len(reservation.samples)} samples; "
                f"expected {self._samples_per_group}."
            )

        try:
            source_sample_identities = tuple((sample.group_index, sample.index) for sample in reservation.samples)
        except BaseException:
            try:
                self._ownership.rollback_reserved([reservation])
            except BaseException as rollback_error:
                self._pending_reserved_rollbacks.append(reservation)
                self._record_cleanup_error(rollback_error)
            else:
                self._execution_slots.release()
                self._retained_slots.release()
            raise

        stage_id = ReservationStageId(f"execution-{self._next_stage_id}")
        self._next_stage_id += 1
        [receipt] = self._ownership.begin_execution([reservation], stage_id=stage_id)
        record = _ExecutionRecord(
            reservation=reservation,
            source_sample_identities=source_sample_identities,
            receipt=receipt,
            execution=None,
            terminal_task=None,
        )
        self._active[receipt.receipt_id] = record

        try:
            # Ownership retains the pristine attempt while execution mutates this copy.
            execution = self._executor.submit(deepcopy(reservation), receipt)
        except BaseException as submit_error:
            record.terminal_observed = True
            settlement_error = self._settle_nontrainable(record)
            if settlement_error is not None:
                raise submit_error from settlement_error
            raise

        record.execution = execution
        record.terminal_task = asyncio.create_task(
            execution.wait_terminal(),
            name=f"fully-async-execution-{receipt.receipt_id}",
        )
        watcher_task = asyncio.create_task(
            self._watch_execution(record),
            name=f"fully-async-owner-{receipt.receipt_id}",
        )
        self._watcher_tasks.add(watcher_task)
        watcher_task.add_done_callback(self._watcher_done)

    async def _drain_admitted_executions(self) -> None:
        while self._watcher_tasks and self._fatal_error is None:
            await asyncio.wait(tuple(self._watcher_tasks), return_when=asyncio.FIRST_COMPLETED)
        if self._fatal_error is None:
            return
        await self._stop_watcher_tasks()
        try:
            self._rollback_ready()
        except BaseException:
            # _rollback_ready records cleanup failure while the fatal error stays primary.
            pass

    async def _finish_quiescence(self, producer_task: asyncio.Task[None] | None) -> None:
        if producer_task is not None:
            await _await_task_completion(producer_task)
        await self._drain_admitted_executions()
        if self._fatal_error is not None:
            raise self._fatal_error

    async def _await_quiescence(self, quiesce_task: asyncio.Task[None]) -> None:
        try:
            await asyncio.shield(quiesce_task)
        except asyncio.CancelledError:
            raise
        except BaseException:
            self._quiesce_failure_reported = True
            raise

    async def _watch_execution(self, record: _ExecutionRecord) -> None:
        terminal_task = record.terminal_task
        if terminal_task is None:
            raise RuntimeError(f"Execution receipt {record.receipt.receipt_id} has no terminal task.")

        try:
            outcome = await asyncio.shield(terminal_task)
        except asyncio.CancelledError as cancellation:
            request_error = self._request_cancellation(record)
            if request_error is not None:
                self._record_cleanup_error(request_error)
                raise request_error from cancellation
            terminal_outcome, observation_error = await _await_terminal_outcome(terminal_task)
            if observation_error is not None:
                self._record_cleanup_error(observation_error)
                raise observation_error from cancellation
            if terminal_outcome is None:
                outcome_error = RuntimeError(f"Execution receipt {record.receipt.receipt_id} has no terminal outcome.")
                self._record_cleanup_error(outcome_error)
                raise outcome_error from cancellation
            receipt_error = _validate_terminal_outcome(record, terminal_outcome)
            if receipt_error is not None:
                self._record_cleanup_error(receipt_error)
                raise receipt_error from cancellation
            record.terminal_observed = True
            payload_error = _validate_terminal_payload(
                record,
                terminal_outcome,
                samples_per_group=self._samples_per_group,
            )
            if payload_error is not None:
                self._record_execution_failure(payload_error)
            elif isinstance(terminal_outcome, FullyAsyncExecutionFailure) and not isinstance(
                terminal_outcome.error, asyncio.CancelledError
            ):
                self._record_execution_failure(terminal_outcome.error)
            settlement_error = self._settle_nontrainable(record)
            if settlement_error is not None:
                self._record_cleanup_error(settlement_error)
                raise settlement_error from cancellation
            if payload_error is not None:
                raise payload_error from cancellation
            raise
        except BaseException as observation_error:
            self._record_cleanup_error(observation_error)
            raise

        receipt_error = _validate_terminal_outcome(record, outcome)
        if receipt_error is not None:
            self._record_cleanup_error(receipt_error)
            raise receipt_error
        record.terminal_observed = True

        payload_error = _validate_terminal_payload(
            record,
            outcome,
            samples_per_group=self._samples_per_group,
        )
        if payload_error is not None:
            settlement_error = self._settle_nontrainable(record)
            self._record_execution_failure(payload_error)
            if settlement_error is not None:
                raise payload_error from settlement_error
            raise payload_error

        if isinstance(outcome, FullyAsyncExecutionRetry):
            settlement_error = self._settle_nontrainable(record)
            if settlement_error is not None:
                raise settlement_error
            return

        if isinstance(outcome, FullyAsyncExecutionFailure):
            if not isinstance(outcome.error, asyncio.CancelledError):
                self._record_execution_failure(outcome.error)
            settlement_error = self._settle_nontrainable(record)
            if settlement_error is not None:
                raise outcome.error from settlement_error
            if isinstance(outcome.error, asyncio.CancelledError):
                return
            raise outcome.error

        if self._completed_slots.locked():
            settlement_error = self._settle_nontrainable(record)
            if settlement_error is not None:
                raise settlement_error
            return

        await self._completed_slots.acquire()
        if self._completed_slots.locked():
            self._completed_capacity_available.clear()

        try:
            [terminal_receipt] = self._ownership.record_terminal(
                [record.receipt],
                stage_id=record.receipt.stage_id,
            )
        except BaseException as error:
            self._release_completed_capacity()
            self._record_cleanup_error(error)
            raise
        if terminal_receipt.disposition is not ReservationTerminalDisposition.TRAINABLE:
            self._release_completed_capacity()
            disposition_error = RuntimeError(
                f"Execution receipt {record.receipt.receipt_id} completed with unexpected "
                f"{terminal_receipt.disposition.name.lower()} disposition."
            )
            self._record_cleanup_error(disposition_error)
            raise disposition_error

        del self._active[record.receipt.receipt_id]
        self._ready.append(_ReadyGroup(receipt=terminal_receipt, samples=outcome.samples))
        self._execution_slots.release()
        self._ready_changed.set()

    def _request_cancellation(self, record: _ExecutionRecord) -> BaseException | None:
        if not record.cancellation_requested:
            try:
                self._ownership.request_cancellation(
                    [record.receipt],
                    stage_id=record.receipt.stage_id,
                )
            except BaseException as error:
                return error
            record.cancellation_requested = True
        terminal_task = record.terminal_task
        if record.execution is None:
            return None
        if terminal_task is not None and terminal_task.done() and not _terminal_observation_is_pending(terminal_task):
            return None
        try:
            record.execution.request_cancellation()
        except BaseException as error:
            return error
        return None

    def _watcher_done(self, task: asyncio.Task[None]) -> None:
        self._watcher_tasks.discard(task)
        if task.cancelled():
            return
        error = task.exception()
        if error is None:
            return
        self._record_fatal(error)
        if self._producer_task is not None and not self._producer_task.done() and not self._closing:
            self._producer_task.cancel()

    async def _stop_watcher_tasks(self) -> None:
        tasks = tuple(self._watcher_tasks)
        for task in tasks:
            if not task.done():
                task.cancel()
        for task in tasks:
            try:
                await _await_task_completion(task)
            except BaseException as error:
                self._record_cleanup_error(error)
        self._watcher_tasks.difference_update(tasks)

    def _settle_nontrainable(self, record: _ExecutionRecord) -> BaseException | None:
        if not record.terminal_observed:
            observation_error = RuntimeError(
                f"Execution receipt {record.receipt.receipt_id} cannot settle before terminal observation."
            )
            self._record_cleanup_error(observation_error)
            return observation_error
        try:
            # Mark ownership nontrainable. Terminal tasks skip the executor's
            # cancellation hook in _request_cancellation.
            request_error = self._request_cancellation(record)
            if request_error is not None:
                raise request_error
            [terminal_receipt] = self._ownership.record_terminal(
                [record.receipt],
                stage_id=record.receipt.stage_id,
            )
        except BaseException as error:
            self._record_cleanup_error(error)
            return error
        if terminal_receipt.disposition is not ReservationTerminalDisposition.CANCELLED:
            disposition_error = RuntimeError(
                f"Execution receipt {record.receipt.receipt_id} settled with unexpected "
                f"{terminal_receipt.disposition.name.lower()} disposition."
            )
            self._record_cleanup_error(disposition_error)
            return disposition_error
        self._release_active(record)
        return None

    def _release_active(self, record: _ExecutionRecord) -> None:
        current = self._active.get(record.receipt.receipt_id)
        if current is not record:
            raise RuntimeError(f"Execution receipt {record.receipt.receipt_id} is not active.")
        del self._active[record.receipt.receipt_id]
        self._execution_slots.release()
        self._retained_slots.release()
        self._ready_changed.set()

    def _release_settled_batch(self, batch: _SchedulerBatch) -> None:
        for _ in batch.groups:
            self._release_completed_capacity()
            self._retained_slots.release()
        self._ready_changed.set()

    def _release_completed_capacity(self) -> None:
        self._completed_slots.release()
        self._completed_capacity_available.set()

    def _retry_pending_acquisition_rollback(self) -> None:
        if not self._pending_acquisition_capacity:
            return
        try:
            self._ownership.retry_failed_acquisition_rollback()
        except BaseException as error:
            self._record_cleanup_error(error)
            return
        self._pending_acquisition_capacity = False
        self._execution_slots.release()
        self._retained_slots.release()

    def _retry_reserved_rollbacks(self) -> None:
        while self._pending_reserved_rollbacks:
            reservation = self._pending_reserved_rollbacks[0]
            try:
                self._ownership.rollback_reserved([reservation])
            except BaseException as error:
                self._record_cleanup_error(error)
                return
            del self._pending_reserved_rollbacks[0]
            self._execution_slots.release()
            self._retained_slots.release()

    async def _retry_active_shutdown(self) -> None:
        cancellable_records: list[_ExecutionRecord] = []
        for record in list(self._active.values()):
            request_error = self._request_cancellation(record)
            if request_error is not None:
                self._record_cleanup_error(request_error)
                continue
            cancellable_records.append(record)

        for record in cancellable_records:
            if not record.terminal_observed:
                terminal_task = record.terminal_task
                if terminal_task is None:
                    terminal_error = RuntimeError(
                        f"Execution receipt {record.receipt.receipt_id} has no terminal task."
                    )
                    self._record_cleanup_error(terminal_error)
                    continue
                if _terminal_observation_is_pending(terminal_task):
                    execution = record.execution
                    if execution is None:
                        execution_error = RuntimeError(
                            f"Execution receipt {record.receipt.receipt_id} has no execution handle."
                        )
                        self._record_cleanup_error(execution_error)
                        continue
                    terminal_task = asyncio.create_task(
                        execution.wait_terminal(),
                        name=f"fully-async-execution-retry-{record.receipt.receipt_id}",
                    )
                    record.terminal_task = terminal_task
                outcome, observation_error = await _await_terminal_outcome(terminal_task)
                if observation_error is not None:
                    self._record_cleanup_error(observation_error)
                    continue
                if outcome is None:
                    outcome_error = RuntimeError(
                        f"Execution receipt {record.receipt.receipt_id} has no terminal outcome."
                    )
                    self._record_cleanup_error(outcome_error)
                    continue
                receipt_error = _validate_terminal_outcome(record, outcome)
                if receipt_error is not None:
                    self._record_cleanup_error(receipt_error)
                    continue
                record.terminal_observed = True
                payload_error = _validate_terminal_payload(
                    record,
                    outcome,
                    samples_per_group=self._samples_per_group,
                )
                if payload_error is not None:
                    self._record_execution_failure(payload_error)
                elif isinstance(outcome, FullyAsyncExecutionFailure) and not isinstance(
                    outcome.error, asyncio.CancelledError
                ):
                    self._record_execution_failure(outcome.error)

            settlement_error = self._settle_nontrainable(record)
            if settlement_error is not None:
                continue

    def _rollback_ready(self) -> None:
        if not self._ready:
            return
        groups = tuple(self._ready)
        try:
            self._ownership.rollback_batch([group.receipt for group in groups])
        except BaseException as error:
            self._record_cleanup_error(error)
            raise
        self._ready.clear()
        for _ in groups:
            self._release_completed_capacity()
            self._retained_slots.release()

    def _record_fatal(self, error: BaseException) -> None:
        if self._fatal_error is None:
            self._fatal_error = error
        self._accepting = False
        self._ready_changed.set()

    def _record_execution_failure(self, error: BaseException) -> None:
        self._record_fatal(error)
        if self._closing and self._shutdown_failure is None:
            self._shutdown_failure = error

    def _record_cleanup_error(self, error: BaseException) -> None:
        if self._cleanup_error is None:
            self._cleanup_error = error
        self._record_fatal(error)


async def _await_terminal_outcome(
    task: asyncio.Task[FullyAsyncExecutionOutcome],
) -> tuple[FullyAsyncExecutionOutcome | None, BaseException | None]:
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
        except BaseException:
            break
    try:
        return task.result(), None
    except BaseException as error:
        return None, error


def _terminal_observation_is_pending(task: asyncio.Task[FullyAsyncExecutionOutcome]) -> bool:
    if not task.done() or task.cancelled():
        return False
    return isinstance(task.exception(), FullyAsyncTerminalPendingError)


def _validate_terminal_outcome(
    record: _ExecutionRecord,
    outcome: FullyAsyncExecutionOutcome,
) -> RuntimeError | None:
    if not isinstance(
        outcome,
        (FullyAsyncExecutionSuccess, FullyAsyncExecutionFailure, FullyAsyncExecutionRetry),
    ):
        return RuntimeError(
            f"Execution receipt {record.receipt.receipt_id} returned unsupported terminal outcome "
            f"{type(outcome).__name__}."
        )
    if outcome.executor_receipt is not record.receipt:
        return RuntimeError(
            f"Execution receipt {record.receipt.receipt_id} did not return its exact terminal receipt."
        )
    return None


def _validate_execution_samples(
    record: _ExecutionRecord,
    samples: list[Sample],
    *,
    samples_per_group: int,
) -> ValueError | None:
    if len(samples) != samples_per_group:
        return ValueError(
            f"Execution receipt {record.receipt.receipt_id} returned {len(samples)} samples; "
            f"expected {samples_per_group}."
        )
    invalid_positions = [position for position, sample in enumerate(samples) if not isinstance(sample, Sample)]
    if invalid_positions:
        return ValueError(
            f"Execution receipt {record.receipt.receipt_id} returned non-Sample values "
            f"at positions {invalid_positions}."
        )
    expected_identity = list(record.source_sample_identities)
    actual_identity = [(sample.group_index, sample.index) for sample in samples]
    if actual_identity != expected_identity:
        return ValueError(
            f"Execution receipt {record.receipt.receipt_id} returned sample identities {actual_identity}; "
            f"expected {expected_identity}."
        )
    return None


def _normalize_execution_failure(record: _ExecutionRecord, error: BaseException) -> BaseException:
    if not isinstance(error, BaseException):
        return RuntimeError(f"Execution receipt {record.receipt.receipt_id} returned invalid failure {error!r}.")
    if isinstance(error, asyncio.CancelledError) and not record.cancellation_requested:
        return RuntimeError(
            f"Execution receipt {record.receipt.receipt_id} reported cancellation "
            "before the scheduler requested it."
        )
    return error


def _normalize_execution_retry(
    record: _ExecutionRecord,
    outcome: FullyAsyncExecutionRetry,
) -> RuntimeError | None:
    if not isinstance(outcome.reason, FullyAsyncRetryReason):
        return RuntimeError(
            f"Execution receipt {record.receipt.receipt_id} returned unsupported retry reason {outcome.reason!r}."
        )
    if outcome.reason is FullyAsyncRetryReason.CANCELLATION_REQUESTED and not record.cancellation_requested:
        return RuntimeError(
            f"Execution receipt {record.receipt.receipt_id} reported a cancellation retry "
            "before the scheduler requested it."
        )
    return None


def _validate_terminal_payload(
    record: _ExecutionRecord,
    outcome: FullyAsyncExecutionOutcome,
    *,
    samples_per_group: int,
) -> BaseException | None:
    if isinstance(outcome, FullyAsyncExecutionRetry):
        return _normalize_execution_retry(record, outcome)
    if isinstance(outcome, FullyAsyncExecutionFailure):
        failure_error = _normalize_execution_failure(record, outcome.error)
        return None if failure_error is outcome.error else failure_error
    return _validate_execution_samples(
        record,
        outcome.samples,
        samples_per_group=samples_per_group,
    )


async def _await_task_completion(task: asyncio.Task[None]) -> None:
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
    try:
        task.result()
    except asyncio.CancelledError:
        return
