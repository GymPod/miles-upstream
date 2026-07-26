import pickle
from collections.abc import Callable

import pytest

from miles.rollout.base_types import RolloutFnTrainOutput
from miles.rollout.rollout_session import BatchRollbackReason, BatchRollbackUnsupportedError, TrainBatchLease


class _RecordingTrainBatchLease(TrainBatchLease):
    def __init__(
        self,
        rollout_id: int,
        output: RolloutFnTrainOutput,
        on_commit: Callable[[], None],
        on_rollback: Callable[[BatchRollbackReason], None],
    ) -> None:
        super().__init__(rollout_id=rollout_id, output=output)
        self._on_commit = on_commit
        self._on_rollback = on_rollback

    def _commit(self) -> None:
        self._on_commit()

    def _rollback(self, reason: BatchRollbackReason) -> None:
        self._on_rollback(reason)


async def test_train_batch_lease_commits_exactly_once() -> None:
    commits: list[int] = []

    def record_commit() -> None:
        commits.append(7)

    def reject_rollback(reason: BatchRollbackReason) -> None:
        raise AssertionError(f"Unexpected rollback: {reason}")

    output = RolloutFnTrainOutput(samples=[], metrics={"source": "test"})
    lease = _RecordingTrainBatchLease(
        rollout_id=7,
        output=output,
        on_commit=record_commit,
        on_rollback=reject_rollback,
    )

    assert lease.rollout_id == 7
    assert lease.output == RolloutFnTrainOutput(samples=[], metrics={"source": "test"})

    lease.commit()

    assert commits == [7]
    with pytest.raises(RuntimeError) as exc_info:
        lease.commit()
    assert str(exc_info.value) == "Train batch lease for rollout 7 is already committed."


async def test_batch_rollback_unsupported_error_round_trips_through_pickle() -> None:
    error = BatchRollbackUnsupportedError(
        rollout_id=11,
        reason=BatchRollbackReason.HANDOFF_FAILED,
    )

    restored = pickle.loads(pickle.dumps(error))

    assert type(restored) is BatchRollbackUnsupportedError
    assert restored.rollout_id == 11
    assert restored.reason is BatchRollbackReason.HANDOFF_FAILED
    assert str(restored) == "Train batch lease for rollout 11 cannot restore ownership after HANDOFF_FAILED."


async def test_train_batch_lease_rolls_back_exactly_once() -> None:
    rollbacks: list[BatchRollbackReason] = []

    def reject_commit() -> None:
        raise AssertionError("Unexpected commit")

    def record_rollback(reason: BatchRollbackReason) -> None:
        rollbacks.append(reason)

    lease = _RecordingTrainBatchLease(
        rollout_id=11,
        output=RolloutFnTrainOutput(samples=[], metrics=None),
        on_commit=reject_commit,
        on_rollback=record_rollback,
    )

    lease.rollback(BatchRollbackReason.HANDOFF_FAILED)

    assert rollbacks == [BatchRollbackReason.HANDOFF_FAILED]
    with pytest.raises(RuntimeError) as rollback_error:
        lease.rollback(BatchRollbackReason.HANDOFF_FAILED)
    assert str(rollback_error.value) == "Train batch lease for rollout 11 is already rolled back."
    with pytest.raises(RuntimeError) as commit_error:
        lease.commit()
    assert str(commit_error.value) == "Train batch lease for rollout 11 is already rolled back."


@pytest.mark.parametrize("settlement", ["commit", "rollback"])
async def test_train_batch_lease_rejects_retry_after_failed_settlement(settlement: str) -> None:
    commit_attempts = 0
    rollback_attempts = 0

    def commit() -> None:
        nonlocal commit_attempts
        commit_attempts += 1
        if commit_attempts == 1:
            raise RuntimeError("commit failed")

    def rollback(reason: BatchRollbackReason) -> None:
        nonlocal rollback_attempts
        rollback_attempts += 1
        if rollback_attempts == 1:
            raise RuntimeError("rollback failed")

    lease = _RecordingTrainBatchLease(
        rollout_id=12,
        output=RolloutFnTrainOutput(samples=[], metrics=None),
        on_commit=commit,
        on_rollback=rollback,
    )

    if settlement == "commit":
        with pytest.raises(RuntimeError) as settlement_error:
            lease.commit()
        assert str(settlement_error.value) == "commit failed"
        with pytest.raises(RuntimeError) as retry_error:
            lease.commit()
        assert str(retry_error.value) == "Train batch lease for rollout 12 is already commit failed."
        with pytest.raises(RuntimeError) as opposite_error:
            lease.rollback(BatchRollbackReason.HANDOFF_FAILED)
        assert str(opposite_error.value) == "Train batch lease for rollout 12 is already commit failed."
        assert (commit_attempts, rollback_attempts) == (1, 0)
    else:
        with pytest.raises(RuntimeError) as settlement_error:
            lease.rollback(BatchRollbackReason.HANDOFF_FAILED)
        assert str(settlement_error.value) == "rollback failed"
        with pytest.raises(RuntimeError) as retry_error:
            lease.rollback(BatchRollbackReason.HANDOFF_FAILED)
        assert str(retry_error.value) == "Train batch lease for rollout 12 is already rollback failed."
        with pytest.raises(RuntimeError) as opposite_error:
            lease.commit()
        assert str(opposite_error.value) == "Train batch lease for rollout 12 is already rollback failed."
        assert (commit_attempts, rollback_attempts) == (0, 1)
