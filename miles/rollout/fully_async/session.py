from __future__ import annotations

import asyncio
from argparse import Namespace

from miles.rollout.base_types import RolloutFnEvalOutput, RolloutFnTrainOutput
from miles.rollout.data_source import DataSource
from miles.rollout.fully_async.execution import FullyAsyncExecutor
from miles.rollout.fully_async.scheduler import _FullyAsyncScheduler, _SchedulerBatch
from miles.rollout.rollout_session import BatchRollbackReason, RolloutSession, TrainBatchLease
from miles.utils.arguments import resolve_fully_async_limits


class _FullyAsyncTrainBatchLease(TrainBatchLease):
    def __init__(
        self,
        *,
        rollout_id: int,
        batch: _SchedulerBatch,
        session: FullyAsyncRolloutSession,
    ) -> None:
        super().__init__(
            rollout_id=rollout_id,
            output=RolloutFnTrainOutput(samples=batch.samples, metrics=None),
        )
        self._batch = batch
        self._session = session

    def _commit(self) -> None:
        self._session._commit_lease(self)

    def _rollback(self, reason: BatchRollbackReason) -> None:
        self._session._rollback_lease(self)


class FullyAsyncRolloutSession(RolloutSession):
    """Own bounded fully asynchronous rollout production and batch handoff.

    Args:
        args: Parsed Miles arguments. The session resolves all fully
            asynchronous capacity defaults.
        data_source: Durable source used for exact prompt reservations.
        executor: Executor whose attempts implement terminal cancellation.
    """

    def __init__(
        self,
        *,
        args: Namespace,
        data_source: DataSource,
        executor: FullyAsyncExecutor,
    ) -> None:
        resolve_fully_async_limits(args)
        self._scheduler = _FullyAsyncScheduler(
            data_source=data_source,
            executor=executor,
            samples_per_group=args.n_samples_per_prompt,
            groups_per_batch=args.rollout_batch_size,
            max_execution_samples=args.fully_async_max_execution_samples,
            max_retained_groups=args.fully_async_max_retained_groups,
            max_completed_groups=args.fully_async_max_completed_prefetch_groups,
        )
        self._executor = executor
        self._open_leases: dict[int, _FullyAsyncTrainBatchLease] = {}
        self._acquire_lock = asyncio.Lock()
        self._accepting_operations = True
        self._executor_closed = False
        self._closed = False
        self._close_task: asyncio.Task[None] | None = None

    async def acquire_train_batch(self, rollout_id: int) -> TrainBatchLease:
        self._ensure_open()
        async with self._acquire_lock:
            self._ensure_open()
            if rollout_id in self._open_leases:
                raise RuntimeError(f"Train batch lease for rollout {rollout_id} is already open.")
            try:
                batch = await self._scheduler.acquire_batch()
            except BaseException as error:
                if not self._accepting_operations:
                    raise RuntimeError("Rollout session is closed.") from error
                raise
            lease = _FullyAsyncTrainBatchLease(
                rollout_id=rollout_id,
                batch=batch,
                session=self,
            )
            self._open_leases[rollout_id] = lease
            return lease

    async def evaluate(self, rollout_id: int) -> RolloutFnEvalOutput:
        self._ensure_open()
        raise RuntimeError("Fully asynchronous evaluation requires a separate quiescence integration.")

    async def prepare_checkpoint(self, rollout_id: int) -> None:
        self._ensure_open()
        async with self._acquire_lock:
            self._ensure_open()
            self._scheduler.raise_if_failed()
            if self._open_leases:
                open_rollout_ids = sorted(self._open_leases)
                raise RuntimeError(
                    f"Cannot prepare checkpoint {rollout_id} with open train batch leases: {open_rollout_ids}."
                )

    async def close(self) -> None:
        close_task = self._close_task
        if close_task is None:
            close_task = asyncio.create_task(self._close())
            self._close_task = close_task
        try:
            await asyncio.shield(close_task)
        except asyncio.CancelledError as cancelled:
            try:
                await _await_task_terminal(close_task)
            except BaseException as terminal_error:
                raise cancelled from terminal_error
            raise
        finally:
            if self._close_task is close_task and close_task.done() and not self._closed:
                self._close_task = None

    async def _close(self) -> None:
        if self._closed:
            return
        self._accepting_operations = False

        scheduler_error: BaseException | None = None
        try:
            await self._scheduler.close()
        except BaseException as error:
            scheduler_error = error

        executor_error: BaseException | None = None
        if not self._open_leases and not self._scheduler.has_active_executions and not self._executor_closed:
            try:
                await self._executor.close()
            except BaseException as error:
                executor_error = error
            else:
                self._executor_closed = True

        if self._open_leases:
            open_rollout_ids = sorted(self._open_leases)
            lease_error = RuntimeError(
                f"Cannot close rollout session with open train batch leases: {open_rollout_ids}."
            )
            if scheduler_error is not None:
                raise lease_error from scheduler_error
            if executor_error is not None:
                raise lease_error from executor_error
            raise lease_error

        if scheduler_error is not None:
            if executor_error is not None:
                raise scheduler_error from executor_error
            raise scheduler_error

        if executor_error is not None:
            raise executor_error

        self._closed = self._executor_closed

    def _commit_lease(self, lease: _FullyAsyncTrainBatchLease) -> None:
        self._require_lease(lease)
        self._scheduler.commit_batch(lease._batch, rollout_id=lease.rollout_id)
        del self._open_leases[lease.rollout_id]

    def _rollback_lease(self, lease: _FullyAsyncTrainBatchLease) -> None:
        self._require_lease(lease)
        self._scheduler.rollback_batch(lease._batch)
        del self._open_leases[lease.rollout_id]

    def _require_lease(self, lease: _FullyAsyncTrainBatchLease) -> None:
        if self._open_leases.get(lease.rollout_id) is not lease:
            raise RuntimeError(f"Train batch lease for rollout {lease.rollout_id} is not owned by this session.")

    def _ensure_open(self) -> None:
        if not self._accepting_operations:
            raise RuntimeError("Rollout session is closed.")


async def _await_task_terminal(task: asyncio.Task[None]) -> None:
    while True:
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            if task.done():
                task.result()
                return
        else:
            return
