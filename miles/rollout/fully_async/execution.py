import abc
from typing import NamedTuple

from miles.rollout.data_source import SourceReservation
from miles.rollout.fully_async.ownership import ReservationExecutorReceipt
from miles.utils.types import Sample


class FullyAsyncExecutionSuccess(NamedTuple):
    """Report successful terminal execution for one exact attempt.

    Attributes:
        executor_receipt: Exact submitted attempt that reached terminal state.
        samples: Completed samples for that attempt.
    """

    executor_receipt: ReservationExecutorReceipt
    samples: list[Sample]


class FullyAsyncExecutionFailure(NamedTuple):
    """Report failed terminal execution for one exact attempt.

    Attributes:
        executor_receipt: Exact submitted attempt that reached terminal state.
        error: Failure observed after that attempt became terminal.
    """

    executor_receipt: ReservationExecutorReceipt
    error: BaseException


FullyAsyncExecutionOutcome = FullyAsyncExecutionSuccess | FullyAsyncExecutionFailure


class FullyAsyncExecution(abc.ABC):
    """Represent one submitted prompt-group execution.

    Cancellation is a two-phase protocol. ``request_cancellation`` only sends
    the request. ``wait_terminal`` must not return or raise until all execution
    owned by this attempt is terminal.
    """

    @abc.abstractmethod
    def request_cancellation(self) -> None:
        """Request cancellation without claiming terminal completion.

        Repeated calls must be idempotent. A call that raises must remain safe
        to retry because it does not prove whether the request reached the
        executor.
        """

    @abc.abstractmethod
    async def wait_terminal(self) -> FullyAsyncExecutionOutcome:
        """Wait for terminal execution and return its exact receipt.

        Returns:
            A receipt-bound success or failure outcome.

        Raises:
            BaseException: If terminal state cannot be proved. Execution
                failures must use ``FullyAsyncExecutionFailure``.
        """


class FullyAsyncExecutor(abc.ABC):
    """Submit and close executions used by a fully asynchronous session."""

    @abc.abstractmethod
    def submit(
        self,
        reservation: SourceReservation,
        receipt: ReservationExecutorReceipt,
    ) -> FullyAsyncExecution:
        """Submit one exact reservation attempt.

        Args:
            reservation: Source-owned prompt group to execute.
            receipt: Exact ownership receipt for this execution.

        Returns:
            Execution handle whose terminal result belongs to ``receipt``.

        If this method raises, it must not accept or start any execution.
        """

    @abc.abstractmethod
    async def close(self) -> None:
        """Close executor resources after every submitted attempt is terminal."""
