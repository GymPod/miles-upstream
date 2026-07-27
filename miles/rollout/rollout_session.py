import abc
from enum import Enum, auto

from miles.rollout.base_types import RolloutFnEvalOutput, RolloutFnTrainOutput


class BatchRollbackReason(Enum):
    """Reason that a manager could not hand a leased batch to training."""

    HANDOFF_FAILED = auto()


class BatchRollbackUnsupportedError(RuntimeError):
    """Report that a lease cannot restore ownership after a failed handoff.

    Args:
        rollout_id: Training rollout whose leased batch could not be restored.
        reason: Why the manager requested ownership restoration.
    """

    def __init__(self, rollout_id: int, reason: BatchRollbackReason) -> None:
        self.rollout_id = rollout_id
        self.reason = reason
        super().__init__(rollout_id, reason)

    def __str__(self) -> str:
        return f"Train batch lease for rollout {self.rollout_id} cannot restore ownership after {self.reason.name}."


class _LeaseState(Enum):
    OPEN = auto()
    COMMITTING = auto()
    COMMITTED = auto()
    COMMIT_FAILED = auto()
    ROLLING_BACK = auto()
    ROLLED_BACK = auto()
    ROLLBACK_FAILED = auto()


class TrainBatchLease(abc.ABC):
    """Own a rollout batch until its train-data handoff is settled.

    Args:
        rollout_id: Training rollout that requested the batch.
        output: Completed rollout output held by this lease.

    A successful ``commit`` records that downstream train data now owns the
    batch. Settlement may be attempted only once, including when its
    implementation raises.
    """

    def __init__(self, rollout_id: int, output: RolloutFnTrainOutput) -> None:
        self._rollout_id = rollout_id
        self._output = output
        self._state = _LeaseState.OPEN

    @property
    def rollout_id(self) -> int:
        """Return the rollout that acquired this batch."""
        return self._rollout_id

    @property
    def output(self) -> RolloutFnTrainOutput:
        """Return the completed rollout output owned by this lease."""
        return self._output

    def commit(self) -> None:
        """Transfer batch ownership to downstream train data.

        Raises:
            RuntimeError: If any settlement was already attempted.
        """
        if self._state is not _LeaseState.OPEN:
            self._raise_already_settled()

        self._state = _LeaseState.COMMITTING
        try:
            self._commit()
        except BaseException:
            self._state = _LeaseState.COMMIT_FAILED
            raise
        self._state = _LeaseState.COMMITTED

    @abc.abstractmethod
    def _commit(self) -> None:
        """Implement the ownership transfer after settlement is claimed."""

    def rollback(self, reason: BatchRollbackReason) -> None:
        """Return ownership after the manager fails to publish train data.

        Args:
            reason: Why the manager could not complete the handoff.

        Raises:
            RuntimeError: If any settlement was already attempted.
        """
        if self._state is not _LeaseState.OPEN:
            self._raise_already_settled()

        self._state = _LeaseState.ROLLING_BACK
        try:
            self._rollback(reason)
        except BaseException:
            self._state = _LeaseState.ROLLBACK_FAILED
            raise
        self._state = _LeaseState.ROLLED_BACK

    @abc.abstractmethod
    def _rollback(self, reason: BatchRollbackReason) -> None:
        """Implement ownership recovery after settlement is claimed."""

    def _raise_already_settled(self) -> None:
        descriptions = {
            _LeaseState.COMMITTING: "committing",
            _LeaseState.COMMITTED: "committed",
            _LeaseState.COMMIT_FAILED: "commit failed",
            _LeaseState.ROLLING_BACK: "rolling back",
            _LeaseState.ROLLED_BACK: "rolled back",
            _LeaseState.ROLLBACK_FAILED: "rollback failed",
        }
        raise RuntimeError(f"Train batch lease for rollout {self.rollout_id} is already {descriptions[self._state]}.")


class RolloutSession(abc.ABC):
    """Own the lifecycle and batch handoff for one train/eval rollout pair.

    Explicit session classes loaded from rollout function paths must accept one
    ``RolloutFnConstructorInput`` positional argument.
    """

    @abc.abstractmethod
    async def acquire_train_batch(self, rollout_id: int) -> TrainBatchLease:
        """Acquire one training batch without releasing its ownership.

        Args:
            rollout_id: Training rollout requesting the batch.

        Returns:
            A lease that must be committed or rolled back exactly once.

        Raises:
            RuntimeError: If this rollout already has an open batch lease.
        """

    @abc.abstractmethod
    async def evaluate(self, rollout_id: int) -> RolloutFnEvalOutput:
        """Run evaluation through state isolated from training.

        Args:
            rollout_id: Training rollout associated with the evaluation.

        Returns:
            Evaluation data and metrics.
        """

    @abc.abstractmethod
    async def prepare_checkpoint(self, rollout_id: int) -> None:
        """Prepare rollout-owned state for a matched checkpoint.

        Args:
            rollout_id: Rollout identifier that the checkpoint will publish.

        Raises:
            RuntimeError: If an acquired train batch remains unsettled.
        """

    @property
    def supports_train_admission_control(self) -> bool:
        """Return whether training admission can be quiesced and resumed."""
        return False

    async def quiesce_train_admission(self) -> None:
        """Stop admitting new training work after draining admitted work.

        Completed training work remains owned by the session for later batch
        acquisition.

        Raises:
            RuntimeError: If this session does not support admission control.
        """
        raise RuntimeError(f"{type(self).__name__} does not support train admission quiescence.")

    async def resume_train_admission(self) -> None:
        """Resume admission of new training work.

        Raises:
            RuntimeError: If this session does not support admission control.
        """
        raise RuntimeError(f"{type(self).__name__} does not support train admission resumption.")

    @abc.abstractmethod
    async def close(self) -> None:
        """Close train and evaluation rollout resources.

        Repeated calls have no effect after the first successful close.
        """
