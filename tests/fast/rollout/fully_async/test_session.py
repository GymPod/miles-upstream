from typing import cast

import pytest

from miles.rollout.fully_async.execution import (
    FullyAsyncExecution,
    FullyAsyncExecutionFailure,
    FullyAsyncExecutionSuccess,
    FullyAsyncExecutor,
)
from miles.rollout.fully_async.ownership import ReservationExecutorReceipt


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
