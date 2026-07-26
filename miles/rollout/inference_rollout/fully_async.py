import asyncio
from typing import TypeVar, cast

from miles.rollout.base_types import RolloutFnConstructorInput
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
from miles.rollout.fully_async.ownership import ReservationExecutorReceipt
from miles.rollout.fully_async.session import FullyAsyncRolloutSession
from miles.rollout.inference_rollout.inference_rollout_common import GenerateState, generate_and_rm_group
from miles.rollout.inference_rollout.inference_rollout_train import request_abort
from miles.utils.types import Sample

_T = TypeVar("_T")


class _InferenceCancellationCoordinator:
    def __init__(self, state: GenerateState) -> None:
        self._state = state
        self._timeout = state.args.rollout_health_check_timeout
        self._task: asyncio.Task[None] | None = None

    def request(self) -> asyncio.Task[None]:
        task = self._task
        retry_failed_request = task is not None and task.done() and (task.cancelled() or task.exception() is not None)
        if task is None or retry_failed_request:
            self._state.aborted = True
            task = asyncio.create_task(
                asyncio.wait_for(
                    request_abort(self._state.args),
                    timeout=self._timeout,
                ),
                name="fully-async-inference-abort",
            )
            self._task = task
        return task

    def allow_retry_after_terminal_pending(self, task: asyncio.Task[None]) -> None:
        if self._task is task:
            self._task = None

    async def close(self) -> None:
        if self._task is not None:
            await asyncio.gather(self._task, return_exceptions=True)


class _InferenceFullyAsyncExecution(FullyAsyncExecution):
    def __init__(
        self,
        *,
        cancellation: _InferenceCancellationCoordinator,
        cancellation_timeout: float,
        task: asyncio.Task[list[Sample]],
        receipt: ReservationExecutorReceipt,
    ) -> None:
        self._cancellation = cancellation
        self._cancellation_timeout = cancellation_timeout
        self._task = task
        self._receipt = receipt
        self._cancellation_task: asyncio.Task[None] | None = None
        self._cancellation_started = asyncio.Event()
        self._cancellation_requested = False

    def request_cancellation(self) -> None:
        self._cancellation_requested = True
        self._cancellation_task = self._cancellation.request()
        self._cancellation_started.set()

    async def wait_terminal(self) -> FullyAsyncExecutionOutcome:
        cancellation_started = await _await_task_terminal(
            asyncio.create_task(
                _wait_for_cancellation_or_terminal(
                    self._task,
                    self._cancellation_started,
                )
            )
        )
        cancellation_error: BaseException | None = None
        cancellation_task: asyncio.Task[None] | None = None
        if cancellation_started:
            cancellation_task = self._cancellation_task
            if cancellation_task is None:
                raise RuntimeError("Fully async inference cancellation has no abort task.")
            try:
                await _await_task_terminal(cancellation_task)
            except BaseException as error:
                cancellation_error = error
        try:
            if cancellation_started:
                samples = await _await_task_terminal_with_timeout(
                    self._task,
                    timeout=self._cancellation_timeout,
                )
            else:
                samples = await _await_task_terminal(self._task)
        except TimeoutError as error:
            if cancellation_task is not None:
                self._cancellation.allow_retry_after_terminal_pending(cancellation_task)
            terminal_error = FullyAsyncTerminalPendingError(
                "Fully async inference cancellation did not make the submitted group "
                f"terminal within {self._cancellation_timeout} seconds."
            )
            if cancellation_error is not None:
                raise terminal_error from cancellation_error
            raise terminal_error from error
        except BaseException as error:
            return FullyAsyncExecutionFailure(
                executor_receipt=self._receipt,
                error=error,
            )
        replay_statuses = (Sample.Status.PENDING, Sample.Status.ABORTED)
        terminal_statuses = (Sample.Status.COMPLETED, Sample.Status.TRUNCATED, Sample.Status.FAILED)
        supported_statuses = replay_statuses + terminal_statuses
        for sample in samples:
            if sample.status not in supported_statuses:
                return FullyAsyncExecutionFailure(
                    executor_receipt=self._receipt,
                    error=ValueError(
                        f"Fully async inference returned sample {sample.index} "
                        f"with unsupported status {sample.status!r}."
                    ),
                )
        if self._cancellation_requested:
            return FullyAsyncExecutionRetry(
                executor_receipt=self._receipt,
                reason=FullyAsyncRetryReason.CANCELLATION_REQUESTED,
            )
        if any(sample.status in replay_statuses for sample in samples):
            return FullyAsyncExecutionRetry(
                executor_receipt=self._receipt,
                reason=FullyAsyncRetryReason.EXECUTION_ABORTED,
            )
        return FullyAsyncExecutionSuccess(
            executor_receipt=self._receipt,
            samples=samples,
        )


class InferenceFullyAsyncExecutor(FullyAsyncExecutor):
    """Execute bounded inference groups on the caller's event loop.

    Args:
        state: Shared inference generation state.
    """

    def __init__(self, state: GenerateState) -> None:
        self._state = state
        self._cancellation = _InferenceCancellationCoordinator(state)
        self._tasks: set[asyncio.Task[list[Sample]]] = set()
        self._closed = False

    def submit(
        self,
        reservation: SourceReservation,
        receipt: ReservationExecutorReceipt,
    ) -> FullyAsyncExecution:
        """Submit one reserved group for in-process inference.

        Args:
            reservation: Exact source reservation to execute.
            receipt: Exact ownership receipt for this attempt.

        Returns:
            An execution handle for the submitted group.

        Raises:
            RuntimeError: If the executor is closed.
        """
        if self._closed:
            raise RuntimeError("Fully async inference executor is closed.")
        task = asyncio.create_task(
            _execute_group(
                self._state,
                reservation.samples,
            )
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return _InferenceFullyAsyncExecution(
            cancellation=self._cancellation,
            cancellation_timeout=self._state.args.rollout_health_check_timeout,
            task=task,
            receipt=receipt,
        )

    async def close(self) -> None:
        """Stop accepting groups after all submitted inference becomes terminal."""
        self._closed = True
        if self._tasks:
            tasks = tuple(self._tasks)
            done, pending = await asyncio.wait(
                tasks,
                timeout=self._state.args.rollout_health_check_timeout,
            )
            await asyncio.gather(*done, return_exceptions=True)
            if pending:
                raise FullyAsyncTerminalPendingError(
                    "Fully async inference executor still has "
                    f"{len(pending)} nonterminal group after "
                    f"{self._state.args.rollout_health_check_timeout} seconds."
                )
        await self._cancellation.close()


async def _execute_group(state: GenerateState, samples: list[Sample]) -> list[Sample]:
    generated = await generate_and_rm_group(
        state,
        samples,
        sampling_params=state.sampling_params.copy(),
        evaluation=False,
    )
    raw_samples = cast(list[object], generated)
    if not all(isinstance(sample, Sample) for sample in raw_samples):
        raise ValueError("Fully async inference does not support one-to-many generation output.")
    generated = [sample for sample in raw_samples if isinstance(sample, Sample)]
    return generated


class InferenceFullyAsyncRolloutSession(FullyAsyncRolloutSession):
    """Run bounded inference rollout through the managed fully async session.

    Args:
        input: Rollout arguments and durable prompt source.
    """

    def __init__(self, input: RolloutFnConstructorInput) -> None:
        unsupported_options = (
            (
                "--dynamic-sampling-filter-path",
                getattr(input.args, "dynamic_sampling_filter_path", None) is not None,
            ),
            (
                "--rollout-sample-filter-path",
                getattr(input.args, "rollout_sample_filter_path", None) is not None,
            ),
            (
                "--rollout-all-samples-process-path",
                getattr(input.args, "rollout_all_samples_process_path", None) is not None,
            ),
            (
                "--recompute-logprobs-via-prefill",
                getattr(input.args, "recompute_logprobs_via_prefill", False),
            ),
            ("--partial-rollout", getattr(input.args, "partial_rollout", False)),
            (
                "--max-weight-staleness",
                getattr(input.args, "max_weight_staleness", None) is not None,
            ),
            ("--eval-interval", getattr(input.args, "eval_interval", None) is not None),
            ("--generate-multi-samples", getattr(input.args, "generate_multi_samples", False)),
            (
                "--over-sampling-batch-size",
                getattr(input.args, "over_sampling_batch_size", input.args.rollout_batch_size)
                not in (None, input.args.rollout_batch_size),
            ),
            ("--colocate", getattr(input.args, "colocate", False)),
            ("--offload-rollout", getattr(input.args, "offload_rollout", False)),
            ("--dumper-enable", getattr(input.args, "dumper_enable", False)),
            ("--dumper-inference", getattr(input.args, "dumper_inference", None) is not None),
            (
                "--dumper-source-patcher-config-inference",
                getattr(input.args, "dumper_source_patcher_config_inference", None) is not None,
            ),
        )
        for option, configured in unsupported_options:
            if configured:
                raise ValueError(
                    f"Fully async inference does not support {option}; "
                    "remove it or use the compatibility rollout path."
                )
        required_source_methods = (
            "reserve_samples",
            "acknowledge_reservations",
            "requeue_reservations",
        )
        missing_source_methods = [
            method_name
            for method_name in required_source_methods
            if getattr(getattr(input.data_source, method_name), "__func__", None) is getattr(DataSource, method_name)
        ]
        if missing_source_methods:
            raise ValueError(
                "Fully async inference requires durable reservation methods; "
                f"{type(input.data_source).__name__} does not override: {', '.join(missing_source_methods)}."
            )
        try:
            reservation_probe = input.data_source.reserve_samples(0)
        except Exception as error:
            raise ValueError(
                "Fully async inference requires a data source with durable reservations; configure "
                "--data-source-path miles.rollout.data_source.RolloutDataSource or another reservation-aware source."
            ) from error
        if reservation_probe != []:
            validation_error = ValueError(
                "A durable data source must return no reservations when asked to reserve zero groups; "
                f"got {len(reservation_probe)}."
            )
            try:
                input.data_source.requeue_reservations(reservation_probe)
            except Exception as requeue_error:
                raise validation_error from requeue_error
            raise validation_error
        state = GenerateState(input.args)
        super().__init__(
            args=input.args,
            data_source=input.data_source,
            executor=InferenceFullyAsyncExecutor(state),
        )


async def _await_task_terminal_with_timeout(
    task: asyncio.Task[_T],
    *,
    timeout: float,
) -> _T:
    timeout_task = asyncio.create_task(asyncio.wait_for(asyncio.shield(task), timeout=timeout))
    return await _await_task_terminal(timeout_task)


async def _wait_for_cancellation_or_terminal(
    task: asyncio.Task[object],
    cancellation_started: asyncio.Event,
) -> bool:
    cancellation_task = asyncio.create_task(cancellation_started.wait())
    try:
        done, _ = await asyncio.wait(
            (task, cancellation_task),
            return_when=asyncio.FIRST_COMPLETED,
        )
        return cancellation_task in done or cancellation_started.is_set()
    finally:
        cancellation_task.cancel()
        await asyncio.gather(cancellation_task, return_exceptions=True)


async def _await_task_terminal(task: asyncio.Task[_T]) -> _T:
    while True:
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            if task.done():
                return task.result()
