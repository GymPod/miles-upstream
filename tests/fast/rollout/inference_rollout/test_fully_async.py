import asyncio
from argparse import Namespace
from typing import cast
from unittest.mock import MagicMock

import pytest

import miles.rollout.inference_rollout.fully_async as fully_async_module
from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput, RolloutFnConstructorInput, RolloutFnTrainOutput
from miles.rollout.data_source import DataSource, RolloutDataSourceWithBuffer, SourceReservation, SourceReservationId
from miles.rollout.fully_async.execution import (
    FullyAsyncExecutionFailure,
    FullyAsyncExecutionRetry,
    FullyAsyncExecutionSuccess,
    FullyAsyncRetryReason,
)
from miles.rollout.fully_async.ownership import ReservationExecutorReceipt
from miles.rollout.inference_rollout.compatibility import load_rollout_session
from miles.rollout.inference_rollout.fully_async import InferenceFullyAsyncExecutor, InferenceFullyAsyncRolloutSession
from miles.rollout.inference_rollout.inference_rollout_common import GenerateState
from miles.utils.types import Sample


class _CopyTrackedPayload:
    def __init__(self, copy_count: list[int]) -> None:
        self.copy_count = copy_count

    def __deepcopy__(self, memo: dict[int, object]) -> "_CopyTrackedPayload":
        self.copy_count[0] += 1
        copied = _CopyTrackedPayload(self.copy_count)
        memo[id(self)] = copied
        return copied


def _generate_state() -> GenerateState:
    async def generate(input: GenerateFnInput) -> GenerateFnOutput:
        sample = cast(Sample, input.sample)
        sample.status = Sample.Status.COMPLETED
        sample.reward = 1.0
        return GenerateFnOutput(samples=sample)

    state = GenerateState.__new__(GenerateState)
    state.args = Namespace(
        group_rm=False,
        mask_offpolicy_in_partial_rollout=False,
        partial_rollout=False,
        rollout_batch_size=1,
        n_samples_per_prompt=1,
        fully_async_max_execution_samples=1,
        fully_async_max_retained_groups=1,
        fully_async_max_completed_prefetch_groups=1,
        async_max_concurrent_samples=None,
        custom_rm_path=None,
        dynamic_sampling_filter_path=None,
        max_weight_staleness=None,
        recompute_logprobs_via_prefill=False,
        rollout_health_check_timeout=0.1,
        rollout_all_samples_process_path=None,
        rollout_sample_filter_path=None,
        sglang_enable_deterministic_inference=False,
        sglang_router_ip="router",
        sglang_router_port=8000,
        sglang_router_policy="random",
    )
    state.generate_fn_semaphore = asyncio.Semaphore(1)
    state.sampling_params = {}
    state.generate_function = generate
    state.aborted = False
    return state


@pytest.mark.asyncio
async def test_executor_consumes_its_execution_reservation_without_another_copy() -> None:
    state = _generate_state()
    executor = InferenceFullyAsyncExecutor(state)
    copy_count = [0]
    payload = _CopyTrackedPayload(copy_count)
    reservation = SourceReservation(
        reservation_id=SourceReservationId("group-0"),
        samples=[
            Sample(
                group_index=0,
                index=0,
                prompt="prompt",
                multimodal_inputs={"images": [payload]},
            )
        ],
    )
    receipt = cast(ReservationExecutorReceipt, object())

    execution = executor.submit(reservation, receipt)
    outcome = await execution.wait_terminal()

    assert outcome == FullyAsyncExecutionSuccess(
        executor_receipt=receipt,
        samples=[
            Sample(
                group_index=0,
                index=0,
                prompt="prompt",
                multimodal_inputs={"images": [payload]},
                reward=1.0,
                status=Sample.Status.COMPLETED,
            )
        ],
    )
    assert isinstance(outcome, FullyAsyncExecutionSuccess)
    assert outcome.samples[0] is reservation.samples[0]
    assert copy_count == [0]
    await executor.close()


@pytest.mark.asyncio
async def test_executor_retries_a_group_submitted_after_global_abort() -> None:
    state = _generate_state()
    state.aborted = True
    executor = InferenceFullyAsyncExecutor(state)
    reservation = SourceReservation(
        reservation_id=SourceReservationId("group-0"),
        samples=[Sample(group_index=0, index=0, prompt="prompt")],
    )
    receipt = cast(ReservationExecutorReceipt, object())

    execution = executor.submit(reservation, receipt)
    outcome = await execution.wait_terminal()

    assert outcome == FullyAsyncExecutionRetry(
        executor_receipt=receipt,
        reason=FullyAsyncRetryReason.EXECUTION_ABORTED,
    )
    await executor.close()


@pytest.mark.asyncio
async def test_executor_reports_one_to_many_output_as_receipt_bound_failure() -> None:
    async def generate(input: GenerateFnInput) -> GenerateFnOutput:
        sample = cast(Sample, input.sample)
        sample.status = Sample.Status.COMPLETED
        sample.reward = 1.0
        child = Sample(
            group_index=sample.group_index,
            index=1,
            prompt="child",
            status=Sample.Status.COMPLETED,
            reward=1.0,
        )
        return GenerateFnOutput(samples=[sample, child])

    state = _generate_state()
    state.generate_function = generate
    executor = InferenceFullyAsyncExecutor(state)
    reservation = SourceReservation(
        reservation_id=SourceReservationId("group-0"),
        samples=[Sample(group_index=0, index=0, prompt="prompt")],
    )
    receipt = cast(ReservationExecutorReceipt, object())

    execution = executor.submit(reservation, receipt)
    outcome = await execution.wait_terminal()

    assert isinstance(outcome, FullyAsyncExecutionFailure)
    assert outcome.executor_receipt is receipt
    assert type(outcome.error) is ValueError
    assert str(outcome.error) == "Fully async inference does not support one-to-many generation output."
    await executor.close()


@pytest.mark.asyncio
async def test_loader_runs_the_inference_fully_async_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _generate_state()
    reservations: list[SourceReservation] = []

    def reserve_samples(num_groups: int) -> list[SourceReservation]:
        if num_groups == 0:
            return []
        assert num_groups == 1
        index = len(reservations)
        reservation = SourceReservation(
            reservation_id=SourceReservationId(f"group-{index}"),
            samples=[Sample(group_index=index, index=index, prompt=f"prompt-{index}")],
        )
        reservations.append(reservation)
        return [reservation]

    data_source = MagicMock(spec=DataSource)
    data_source.reserve_samples.side_effect = reserve_samples
    monkeypatch.setattr(fully_async_module, "GenerateState", lambda _args: state)
    constructor_input = RolloutFnConstructorInput(args=state.args, data_source=data_source)

    session = load_rollout_session(
        constructor_input,
        train_path="miles.rollout.inference_rollout.fully_async.InferenceFullyAsyncRolloutSession",
        eval_path="miles.rollout.inference_rollout.fully_async.InferenceFullyAsyncRolloutSession",
    )
    lease = await session.acquire_train_batch(rollout_id=9)

    assert lease.output == RolloutFnTrainOutput(
        samples=[
            [
                Sample(
                    group_index=0,
                    index=0,
                    prompt="prompt-0",
                    reward=1.0,
                    status=Sample.Status.COMPLETED,
                )
            ]
        ],
        metrics=None,
    )
    assert reservations == [
        SourceReservation(
            reservation_id=SourceReservationId("group-0"),
            samples=[Sample(group_index=0, index=0, prompt="prompt-0")],
        )
    ]
    lease.commit()
    data_source.acknowledge_reservations.assert_called_once_with([reservations[0]], rollout_id=9)
    await session.close()


def test_session_rejects_a_source_without_durable_reservations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _generate_state()
    state_constructions: list[Namespace] = []

    def construct_state(args: Namespace) -> GenerateState:
        state_constructions.append(args)
        return state

    monkeypatch.setattr(fully_async_module, "GenerateState", construct_state)
    data_source = RolloutDataSourceWithBuffer.__new__(RolloutDataSourceWithBuffer)
    constructor_input = RolloutFnConstructorInput(args=state.args, data_source=data_source)

    with pytest.raises(ValueError) as error:
        InferenceFullyAsyncRolloutSession(constructor_input)

    assert str(error.value) == (
        "Fully async inference requires a data source with durable reservations; configure "
        "--data-source-path miles.rollout.data_source.RolloutDataSource or another reservation-aware source."
    )
    assert isinstance(error.value.__cause__, RuntimeError)
    assert str(error.value.__cause__) == (
        "RolloutDataSourceWithBuffer does not support durable source reservations "
        "because they would bypass its retry buffer."
    )
    assert state_constructions == []


def test_session_rejects_a_source_without_durable_settlement_methods(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ReserveOnlyDataSource(DataSource):
        def get_samples(self, num_samples: int) -> list[list[Sample]]:
            raise AssertionError(f"Unexpected sample request: {num_samples}")

        def add_samples(self, samples: list[list[Sample]]) -> None:
            raise AssertionError(f"Unexpected sample return: {samples}")

        def save(self, rollout_id: int) -> None:
            raise AssertionError(f"Unexpected save: {rollout_id}")

        def load(self, rollout_id: int | None = None) -> None:
            raise AssertionError(f"Unexpected load: {rollout_id}")

        def reserve_samples(self, num_groups: int) -> list[SourceReservation]:
            assert num_groups == 0
            return []

    state = _generate_state()
    monkeypatch.setattr(fully_async_module, "GenerateState", lambda _args: state)
    data_source = ReserveOnlyDataSource()
    constructor_input = RolloutFnConstructorInput(args=state.args, data_source=data_source)

    with pytest.raises(ValueError) as error:
        InferenceFullyAsyncRolloutSession(constructor_input)

    assert str(error.value) == (
        "Fully async inference requires durable reservation methods; "
        "ReserveOnlyDataSource does not override: acknowledge_reservations, requeue_reservations."
    )


def test_session_requeues_reservations_returned_by_the_zero_group_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _generate_state()
    reservation = SourceReservation(
        reservation_id=SourceReservationId("unexpected"),
        samples=[Sample(group_index=0, index=0, prompt="prompt")],
    )
    data_source = MagicMock(spec=DataSource)
    data_source.reserve_samples.return_value = [reservation]
    monkeypatch.setattr(fully_async_module, "GenerateState", lambda _args: state)
    constructor_input = RolloutFnConstructorInput(args=state.args, data_source=data_source)

    with pytest.raises(ValueError) as error:
        InferenceFullyAsyncRolloutSession(constructor_input)

    assert str(error.value) == (
        "A durable data source must return no reservations when asked to reserve zero groups; got 1."
    )
    data_source.requeue_reservations.assert_called_once_with([reservation])


def test_session_preserves_a_failed_zero_group_probe_requeue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _generate_state()
    reservation = SourceReservation(
        reservation_id=SourceReservationId("unexpected"),
        samples=[Sample(group_index=0, index=0, prompt="prompt")],
    )
    requeue_error = RuntimeError("requeue failed")
    data_source = MagicMock(spec=DataSource)
    data_source.reserve_samples.return_value = [reservation]
    data_source.requeue_reservations.side_effect = requeue_error
    monkeypatch.setattr(fully_async_module, "GenerateState", lambda _args: state)
    constructor_input = RolloutFnConstructorInput(args=state.args, data_source=data_source)

    with pytest.raises(ValueError) as error:
        InferenceFullyAsyncRolloutSession(constructor_input)

    assert str(error.value) == (
        "A durable data source must return no reservations when asked to reserve zero groups; got 1."
    )
    assert error.value.__cause__ is requeue_error
    data_source.requeue_reservations.assert_called_once_with([reservation])


@pytest.mark.parametrize(
    ("attribute", "value", "option"),
    [
        ("dynamic_sampling_filter_path", "package.dynamic_filter", "--dynamic-sampling-filter-path"),
        ("rollout_sample_filter_path", "package.sample_filter", "--rollout-sample-filter-path"),
        (
            "rollout_all_samples_process_path",
            "package.process_all",
            "--rollout-all-samples-process-path",
        ),
        ("recompute_logprobs_via_prefill", True, "--recompute-logprobs-via-prefill"),
        ("partial_rollout", True, "--partial-rollout"),
        ("max_weight_staleness", 1, "--max-weight-staleness"),
        ("eval_interval", 1, "--eval-interval"),
        ("generate_multi_samples", True, "--generate-multi-samples"),
        ("over_sampling_batch_size", 2, "--over-sampling-batch-size"),
        ("colocate", True, "--colocate"),
        ("offload_rollout", True, "--offload-rollout"),
        ("dumper_enable", True, "--dumper-enable"),
        ("dumper_inference", ["enable=true"], "--dumper-inference"),
        (
            "dumper_source_patcher_config_inference",
            "source-patcher.json",
            "--dumper-source-patcher-config-inference",
        ),
    ],
)
def test_session_rejects_unsupported_batch_behavior(
    monkeypatch: pytest.MonkeyPatch,
    attribute: str,
    value: object,
    option: str,
) -> None:
    state = _generate_state()
    setattr(state.args, attribute, value)
    data_source = MagicMock(spec=DataSource)
    data_source.reserve_samples.return_value = []
    monkeypatch.setattr(fully_async_module, "GenerateState", lambda _args: state)
    constructor_input = RolloutFnConstructorInput(args=state.args, data_source=data_source)

    with pytest.raises(ValueError) as error:
        InferenceFullyAsyncRolloutSession(constructor_input)

    assert str(error.value) == (
        f"Fully async inference does not support {option}; remove it or use the compatibility rollout path."
    )


@pytest.mark.asyncio
async def test_executor_retries_a_terminal_aborted_group() -> None:
    async def generate(input: GenerateFnInput) -> GenerateFnOutput:
        sample = cast(Sample, input.sample)
        sample.tokens = [101, 102]
        sample.response = "discarded"
        sample.response_length = 1
        sample.reward = 0.0
        sample.metadata["attempt"] = "mutated"
        sample.status = Sample.Status.ABORTED
        return GenerateFnOutput(samples=sample)

    state = _generate_state()
    state.generate_function = generate
    executor = InferenceFullyAsyncExecutor(state)
    reservation = SourceReservation(
        reservation_id=SourceReservationId("group-0"),
        samples=[
            Sample(
                group_index=0,
                index=0,
                prompt="prompt",
                metadata={"source": "preserved"},
            )
        ],
    )
    receipt = cast(ReservationExecutorReceipt, object())

    execution = executor.submit(reservation, receipt)
    outcome = await execution.wait_terminal()

    assert outcome == FullyAsyncExecutionRetry(
        executor_receipt=receipt,
        reason=FullyAsyncRetryReason.EXECUTION_ABORTED,
    )
    assert reservation == SourceReservation(
        reservation_id=SourceReservationId("group-0"),
        samples=[
            Sample(
                group_index=0,
                index=0,
                prompt="prompt",
                tokens=[101, 102],
                response="discarded",
                response_length=1,
                reward=0.0,
                metadata={"source": "preserved", "attempt": "mutated"},
                status=Sample.Status.ABORTED,
            )
        ],
    )
    await executor.close()


@pytest.mark.asyncio
async def test_executor_returns_a_terminal_failed_group_for_training() -> None:
    async def generate(input: GenerateFnInput) -> GenerateFnOutput:
        sample = cast(Sample, input.sample)
        sample.status = Sample.Status.FAILED
        sample.reward = 0.0
        return GenerateFnOutput(samples=sample)

    state = _generate_state()
    state.generate_function = generate
    executor = InferenceFullyAsyncExecutor(state)
    reservation = SourceReservation(
        reservation_id=SourceReservationId("group-0"),
        samples=[Sample(group_index=0, index=0, prompt="prompt")],
    )
    receipt = cast(ReservationExecutorReceipt, object())

    execution = executor.submit(reservation, receipt)
    outcome = await execution.wait_terminal()

    assert outcome == FullyAsyncExecutionSuccess(
        executor_receipt=receipt,
        samples=[
            Sample(
                group_index=0,
                index=0,
                prompt="prompt",
                reward=0.0,
                status=Sample.Status.FAILED,
            )
        ],
    )
    assert isinstance(outcome, FullyAsyncExecutionSuccess)
    assert outcome.samples[0] is reservation.samples[0]
    await executor.close()


@pytest.mark.asyncio
async def test_executor_returns_receipt_bound_failure_for_unsupported_sample_status() -> None:
    async def generate(input: GenerateFnInput) -> GenerateFnOutput:
        sample = cast(Sample, input.sample)
        sample.status = cast(Sample.Status, None)
        sample.reward = 0.0
        return GenerateFnOutput(samples=sample)

    state = _generate_state()
    state.generate_function = generate
    executor = InferenceFullyAsyncExecutor(state)
    reservation = SourceReservation(
        reservation_id=SourceReservationId("group-0"),
        samples=[Sample(group_index=0, index=0, prompt="prompt")],
    )
    receipt = cast(ReservationExecutorReceipt, object())

    execution = executor.submit(reservation, receipt)
    outcome = await execution.wait_terminal()

    assert isinstance(outcome, FullyAsyncExecutionFailure)
    assert outcome.executor_receipt is receipt
    assert type(outcome.error) is ValueError
    assert str(outcome.error) == "Fully async inference returned sample 0 with unsupported status None."
    await executor.close()


@pytest.mark.asyncio
async def test_executor_rejects_unsupported_status_before_replaying_a_group() -> None:
    async def generate(input: GenerateFnInput) -> GenerateFnOutput:
        sample = cast(Sample, input.sample)
        sample.status = Sample.Status.PENDING if sample.index == 0 else cast(Sample.Status, None)
        sample.reward = 0.0
        return GenerateFnOutput(samples=sample)

    state = _generate_state()
    state.generate_function = generate
    executor = InferenceFullyAsyncExecutor(state)
    reservation = SourceReservation(
        reservation_id=SourceReservationId("group-0"),
        samples=[
            Sample(group_index=0, index=0, prompt="first"),
            Sample(group_index=0, index=1, prompt="second"),
        ],
    )
    receipt = cast(ReservationExecutorReceipt, object())

    execution = executor.submit(reservation, receipt)
    outcome = await execution.wait_terminal()

    assert isinstance(outcome, FullyAsyncExecutionFailure)
    assert outcome.executor_receipt is receipt
    assert type(outcome.error) is ValueError
    assert str(outcome.error) == "Fully async inference returned sample 1 with unsupported status None."
    await executor.close()


@pytest.mark.asyncio
async def test_executor_returns_receipt_bound_terminal_failure() -> None:
    generation_error = RuntimeError("generation failed")

    async def generate(input: GenerateFnInput) -> GenerateFnOutput:
        raise generation_error

    state = _generate_state()
    state.generate_function = generate
    executor = InferenceFullyAsyncExecutor(state)
    reservation = SourceReservation(
        reservation_id=SourceReservationId("group-0"),
        samples=[Sample(group_index=0, index=0, prompt="prompt")],
    )
    receipt = cast(ReservationExecutorReceipt, object())

    execution = executor.submit(reservation, receipt)
    outcome = await execution.wait_terminal()

    assert outcome == FullyAsyncExecutionFailure(
        executor_receipt=receipt,
        error=generation_error,
    )
    await executor.close()


@pytest.mark.asyncio
async def test_cancellation_requests_remote_abort_and_waits_for_terminal_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generation_started = asyncio.Event()
    generation_finished = asyncio.Event()
    release_generation = asyncio.Event()
    abort_requested = asyncio.Event()
    release_abort = asyncio.Event()

    async def generate(input: GenerateFnInput) -> GenerateFnOutput:
        generation_started.set()
        await release_generation.wait()
        sample = cast(Sample, input.sample)
        sample.status = Sample.Status.COMPLETED
        sample.reward = 1.0
        generation_finished.set()
        return GenerateFnOutput(samples=sample)

    state = _generate_state()
    state.generate_function = generate

    async def request_abort(args: Namespace) -> None:
        assert args is state.args
        assert state.aborted is True
        abort_requested.set()
        release_generation.set()
        await release_abort.wait()

    monkeypatch.setattr(fully_async_module, "request_abort", request_abort, raising=False)
    executor = InferenceFullyAsyncExecutor(state)
    reservation = SourceReservation(
        reservation_id=SourceReservationId("group-0"),
        samples=[Sample(group_index=0, index=0, prompt="prompt")],
    )
    receipt = cast(ReservationExecutorReceipt, object())
    execution = executor.submit(reservation, receipt)
    terminal_task = asyncio.create_task(execution.wait_terminal())
    await generation_started.wait()

    execution.request_cancellation()
    try:
        await asyncio.wait_for(abort_requested.wait(), timeout=0.1)
        await generation_finished.wait()
        for _ in range(10):
            await asyncio.sleep(0)
        assert terminal_task.done() is False
        release_abort.set()
        outcome = await terminal_task
    finally:
        release_generation.set()
        release_abort.set()
        await asyncio.gather(terminal_task, return_exceptions=True)

    assert outcome == FullyAsyncExecutionRetry(
        executor_receipt=receipt,
        reason=FullyAsyncRetryReason.CANCELLATION_REQUESTED,
    )
    await executor.close()
