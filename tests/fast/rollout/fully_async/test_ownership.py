import threading
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

import miles.rollout.fully_async.ownership as ownership_module
from miles.rollout.data_source import DataSource, SourceReservation, SourceReservationId
from miles.rollout.fully_async.ownership import (
    ReservationAcquisitionRollbackError,
    ReservationIdentityConflictError,
    ReservationOwnership,
    ReservationOwnershipPoisonedError,
    ReservationStageId,
    ReservationTerminalDisposition,
    ReservationTerminalReceipt,
)
from miles.utils.types import Sample


class _RecordingDataSource(DataSource):
    def __init__(self, reservations: list[SourceReservation]) -> None:
        self.reservations = reservations
        self.acknowledged: list[tuple[list[SourceReservation], int]] = []
        self.requeued: list[list[SourceReservation]] = []

    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        raise NotImplementedError

    def add_samples(self, samples: list[list[Sample]]) -> None:
        raise NotImplementedError

    def save(self, rollout_id: int) -> None:
        raise NotImplementedError

    def load(self, rollout_id: int | None = None) -> None:
        raise NotImplementedError

    def reserve_samples(self, num_groups: int) -> list[SourceReservation]:
        reservations = self.reservations[:num_groups]
        self.reservations = self.reservations[num_groups:]
        return reservations

    def acknowledge_reservations(
        self,
        reservations: Sequence[SourceReservation],
        *,
        rollout_id: int,
    ) -> None:
        self.acknowledged.append((list(reservations), rollout_id))

    def requeue_reservations(self, reservations: Sequence[SourceReservation]) -> None:
        self.requeued.append(list(reservations))


class _FailingSettlementDataSource(_RecordingDataSource):
    def __init__(
        self,
        reservations: list[SourceReservation],
        acknowledge_error: BaseException | None,
        requeue_error: BaseException | None,
    ) -> None:
        super().__init__(reservations)
        self.acknowledge_error = acknowledge_error
        self.requeue_error = requeue_error

    def acknowledge_reservations(
        self,
        reservations: Sequence[SourceReservation],
        *,
        rollout_id: int,
    ) -> None:
        if self.acknowledge_error is not None:
            raise self.acknowledge_error
        super().acknowledge_reservations(reservations, rollout_id=rollout_id)

    def requeue_reservations(self, reservations: Sequence[SourceReservation]) -> None:
        if self.requeue_error is not None:
            raise self.requeue_error
        super().requeue_reservations(reservations)


class _AcknowledgingThenInterruptingDataSource(_RecordingDataSource):
    def __init__(self, reservations: list[SourceReservation], interrupt: BaseException) -> None:
        super().__init__(reservations)
        self.interrupt = interrupt

    def acknowledge_reservations(
        self,
        reservations: Sequence[SourceReservation],
        *,
        rollout_id: int,
    ) -> None:
        super().acknowledge_reservations(reservations, rollout_id=rollout_id)
        raise self.interrupt


class _InterruptingReservationList(list[SourceReservation]):
    def __init__(self, reservations: list[SourceReservation], interrupt: BaseException) -> None:
        super().__init__(reservations)
        self.interrupt = interrupt

    def __iter__(self) -> Iterator[SourceReservation]:
        raise self.interrupt


class _InterruptingAcquisitionResolutionDataSource(_RecordingDataSource):
    def __init__(self, reservations: list[SourceReservation], interrupt: BaseException) -> None:
        super().__init__(reservations)
        self.interrupt = interrupt

    def reserve_samples(self, num_groups: int) -> list[SourceReservation]:
        reservations = super().reserve_samples(num_groups)
        return _InterruptingReservationList(reservations, self.interrupt)


class _InterruptingRecordMap(dict[SourceReservationId, ownership_module._ReservationRecord]):
    def __init__(
        self,
        records: dict[SourceReservationId, ownership_module._ReservationRecord],
        interrupt: BaseException,
    ) -> None:
        super().__init__(records)
        self.interrupt = interrupt

    def __delitem__(self, reservation_id: SourceReservationId) -> None:
        super().__delitem__(reservation_id)
        raise self.interrupt


class _InterruptingExitLock:
    def __init__(self, interrupt: BaseException) -> None:
        self._lock = threading.RLock()
        self._interrupt: BaseException | None = interrupt

    def __enter__(self) -> "_InterruptingExitLock":
        self._lock.acquire()
        return self

    def __exit__(self, *exception: object) -> None:
        self._lock.release()
        if self._interrupt is not None:
            interrupt = self._interrupt
            self._interrupt = None
            raise interrupt


class _CoordinatedInterruptingExitLock:
    def __init__(
        self,
        interrupt: BaseException,
        exit_started: threading.Event,
        continue_exit: threading.Event,
    ) -> None:
        self._lock = threading.RLock()
        self._interrupt: BaseException | None = interrupt
        self._exit_started = exit_started
        self._continue_exit = continue_exit

    def __enter__(self) -> "_CoordinatedInterruptingExitLock":
        self._lock.acquire()
        return self

    def __exit__(self, *exception: object) -> None:
        self._lock.release()
        if self._interrupt is not None:
            interrupt = self._interrupt
            self._interrupt = None
            self._exit_started.set()
            if not self._continue_exit.wait(timeout=5):
                raise TimeoutError("Timed out waiting to finish the interrupted lock exit.")
            raise interrupt


def _reservation(reservation_id: int) -> SourceReservation:
    return SourceReservation(
        reservation_id=SourceReservationId(reservation_id),
        samples=[Sample(group_index=reservation_id, index=reservation_id)],
    )


def test_short_source_acquisition_requeues_every_returned_reservation() -> None:
    source_reservation = _reservation(0)
    data_source = _RecordingDataSource([source_reservation])
    ownership = ReservationOwnership(data_source)

    with pytest.raises(RuntimeError) as error:
        ownership.reserve_samples(2)

    assert str(error.value) == "Data source returned 1 reservations for a request of 2."
    assert data_source.acknowledged == []
    assert data_source.requeued == [[source_reservation]]


def test_duplicate_source_acquisition_requeues_one_exact_attempt_per_identity() -> None:
    source_reservation = _reservation(0)
    data_source = _RecordingDataSource([source_reservation, source_reservation])
    ownership = ReservationOwnership(data_source)

    with pytest.raises(ValueError) as error:
        ownership.reserve_samples(2)

    assert str(error.value) == "Data source returned duplicate reservation identities: [0, 0]."
    assert data_source.acknowledged == []
    assert data_source.requeued == [[source_reservation]]


def test_conflicting_exact_acquisition_does_not_release_existing_ownership() -> None:
    source_reservation = _reservation(0)
    data_source = _RecordingDataSource([source_reservation])
    ownership = ReservationOwnership(data_source)
    [reservation] = ownership.reserve_samples(1)
    data_source.reservations = [source_reservation]

    with pytest.raises(RuntimeError) as error:
        ownership.reserve_samples(1)

    stage_id = ReservationStageId("weights-0")
    [receipt] = ownership.begin_execution([reservation], stage_id=stage_id)
    assert str(error.value) == "Source reservations are already owned: [0]."
    assert receipt.reservation_id == source_reservation.reservation_id
    assert data_source.acknowledged == []
    assert data_source.requeued == []


def test_distinct_same_identity_attempts_poison_ownership_without_compensation() -> None:
    first_attempt = _reservation(0)
    second_attempt = _reservation(0)
    data_source = _RecordingDataSource([first_attempt, second_attempt])
    ownership = ReservationOwnership(data_source)

    with pytest.raises(ReservationIdentityConflictError) as error:
        ownership.reserve_samples(2)

    assert str(error.value) == (
        "Reservation ownership is poisoned by distinct attempts with the same identities: [0]."
    )
    assert data_source.acknowledged == []
    assert data_source.requeued == []

    with pytest.raises(ReservationIdentityConflictError) as poisoned_error:
        ownership.reserve_samples(1)
    assert str(poisoned_error.value) == str(error.value)


def test_distinct_conflict_with_existing_attempt_poison_ownership() -> None:
    first_attempt = _reservation(0)
    data_source = _RecordingDataSource([first_attempt])
    ownership = ReservationOwnership(data_source)
    [reservation] = ownership.reserve_samples(1)
    data_source.reservations = [_reservation(0)]

    with pytest.raises(ReservationIdentityConflictError) as error:
        ownership.reserve_samples(1)

    with pytest.raises(ReservationIdentityConflictError) as poisoned_error:
        ownership.begin_execution([reservation], stage_id=ReservationStageId("weights-0"))
    assert str(error.value) == (
        "Reservation ownership is poisoned by distinct attempts with the same identities: [0]."
    )
    assert str(poisoned_error.value) == str(error.value)
    assert data_source.acknowledged == []
    assert data_source.requeued == []


def test_failed_acquisition_rollback_is_retained_until_explicit_retry() -> None:
    source_reservation = _reservation(0)
    requeue_error = RuntimeError("acquisition requeue failed")
    data_source = _FailingSettlementDataSource(
        [source_reservation],
        acknowledge_error=None,
        requeue_error=requeue_error,
    )
    ownership = ReservationOwnership(data_source)
    assert not ownership.has_pending_acquisition_rollback

    with pytest.raises(ReservationAcquisitionRollbackError) as error:
        ownership.reserve_samples(2)

    assert str(error.value) == "Failed to requeue reservations after invalid acquisition: acquisition requeue failed"
    assert str(error.value.validation_error) == "Data source returned 1 reservations for a request of 2."
    assert error.value.rollback_error is requeue_error
    assert error.value.__cause__ is requeue_error
    assert data_source.acknowledged == []
    assert data_source.requeued == []
    assert ownership.has_pending_acquisition_rollback

    with pytest.raises(RuntimeError) as blocked_error:
        ownership.reserve_samples(1)
    assert str(blocked_error.value) == "Cannot reserve samples while an acquisition rollback is pending."

    data_source.requeue_error = None
    ownership.retry_failed_acquisition_rollback()
    assert data_source.requeued == [[source_reservation]]
    assert not ownership.has_pending_acquisition_rollback


def test_acquisition_interrupt_poison_ownership_and_preserves_interrupt() -> None:
    source_reservation = _reservation(0)
    interrupt = KeyboardInterrupt()
    data_source = _FailingSettlementDataSource(
        [source_reservation],
        acknowledge_error=None,
        requeue_error=interrupt,
    )
    ownership = ReservationOwnership(data_source)

    with pytest.raises(KeyboardInterrupt) as error:
        ownership.reserve_samples(2)

    assert error.value is interrupt
    assert data_source.acknowledged == []
    assert data_source.requeued == []

    with pytest.raises(ReservationOwnershipPoisonedError) as poisoned_error:
        ownership.retry_failed_acquisition_rollback()
    assert str(poisoned_error.value) == (
        "Reservation ownership is poisoned because transition "
        "'compensate invalid source acquisition' may have partially completed after KeyboardInterrupt(). "
        "Recover from a durable checkpoint."
    )
    assert poisoned_error.value.__cause__ is interrupt


def test_post_acquisition_resolution_interrupt_poison_ownership() -> None:
    source_reservation = _reservation(1)
    interrupt = KeyboardInterrupt()
    data_source = _InterruptingAcquisitionResolutionDataSource([source_reservation], interrupt)
    ownership = ReservationOwnership(data_source)

    with pytest.raises(KeyboardInterrupt) as error:
        ownership.reserve_samples(1)

    assert error.value is interrupt
    assert data_source.reservations == []
    assert data_source.acknowledged == []
    assert data_source.requeued == []

    with pytest.raises(ReservationOwnershipPoisonedError) as poisoned_error:
        ownership.reserve_samples(1)
    assert str(poisoned_error.value) == (
        "Reservation ownership is poisoned because transition "
        "'resolve acquired source reservations' may have partially completed after KeyboardInterrupt(). "
        "Recover from a durable checkpoint."
    )
    assert poisoned_error.value.__cause__ is interrupt


def test_acquisition_return_interrupt_poison_ownership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_reservation = _reservation(2)
    data_source = _RecordingDataSource([source_reservation])
    ownership = ReservationOwnership(data_source)
    interrupt = KeyboardInterrupt()
    monkeypatch.setattr(ownership, "_lock", _InterruptingExitLock(interrupt))

    with pytest.raises(KeyboardInterrupt) as error:
        ownership.reserve_samples(1)

    assert error.value is interrupt
    assert data_source.reservations == []
    assert data_source.acknowledged == []
    assert data_source.requeued == []

    with pytest.raises(ReservationOwnershipPoisonedError) as poisoned_error:
        ownership.reserve_samples(1)
    assert str(poisoned_error.value) == (
        "Reservation ownership is poisoned because transition "
        "'resolve acquired source reservations' may have partially completed after KeyboardInterrupt(). "
        "Recover from a durable checkpoint."
    )
    assert poisoned_error.value.__cause__ is interrupt


def test_cancellation_request_does_not_release_source_ownership() -> None:
    source_reservation = _reservation(1)
    data_source = _RecordingDataSource([source_reservation])
    ownership = ReservationOwnership(data_source)
    [reservation] = ownership.reserve_samples(1)
    stage_id = ReservationStageId("weights-7")
    [receipt] = ownership.begin_execution([reservation], stage_id=stage_id)

    ownership.request_cancellation([receipt], stage_id=stage_id)

    assert reservation is source_reservation
    assert data_source.acknowledged == []
    assert data_source.requeued == []


def test_completion_after_cancellation_requeues_once_and_is_not_trainable() -> None:
    source_reservation = _reservation(2)
    data_source = _RecordingDataSource([source_reservation])
    ownership = ReservationOwnership(data_source)
    [reservation] = ownership.reserve_samples(1)
    stage_id = ReservationStageId("weights-8")
    [executor_receipt] = ownership.begin_execution([reservation], stage_id=stage_id)
    ownership.request_cancellation([executor_receipt], stage_id=stage_id)

    terminal_receipts = ownership.record_terminal([executor_receipt], stage_id=stage_id)
    with pytest.raises(RuntimeError) as duplicate_error:
        ownership.record_terminal([executor_receipt], stage_id=stage_id)

    assert terminal_receipts == [
        ReservationTerminalReceipt(
            executor_receipt=executor_receipt,
            disposition=ReservationTerminalDisposition.CANCELLED,
        )
    ]
    assert str(duplicate_error.value) == (
        "Cannot record terminal callback for executor receipt 0; reservation is cancelled."
    )
    assert data_source.acknowledged == []
    assert data_source.requeued == [[source_reservation]]


def test_trainable_completion_commits_exact_source_reservation() -> None:
    source_reservation = _reservation(3)
    data_source = _RecordingDataSource([source_reservation])
    ownership = ReservationOwnership(data_source)
    [reservation] = ownership.reserve_samples(1)
    stage_id = ReservationStageId("weights-9")
    [executor_receipt] = ownership.begin_execution([reservation], stage_id=stage_id)
    [terminal_receipt] = ownership.record_terminal([executor_receipt], stage_id=stage_id)

    ownership.commit_batch([terminal_receipt], rollout_id=11)
    with pytest.raises(RuntimeError) as duplicate_terminal_error:
        ownership.record_terminal([executor_receipt], stage_id=stage_id)
    with pytest.raises(RuntimeError) as duplicate_commit_error:
        ownership.commit_batch([terminal_receipt], rollout_id=12)
    with pytest.raises(RuntimeError) as commit_rollback_error:
        ownership.rollback_batch([terminal_receipt])

    assert terminal_receipt == ReservationTerminalReceipt(
        executor_receipt=executor_receipt,
        disposition=ReservationTerminalDisposition.TRAINABLE,
    )
    assert str(duplicate_terminal_error.value) == (
        "Cannot record terminal callback for executor receipt 0; reservation is committed."
    )
    assert str(duplicate_commit_error.value) == (
        "Cannot commit terminal receipt 0; receipt is not exact trainable ownership."
    )
    assert str(commit_rollback_error.value) == (
        "Cannot roll back terminal receipt 0; receipt is not exact trainable ownership."
    )
    assert data_source.acknowledged == [([source_reservation], 11)]
    assert data_source.requeued == []


def test_trainable_completion_rolls_back_exact_source_reservation() -> None:
    source_reservation = _reservation(4)
    data_source = _RecordingDataSource([source_reservation])
    ownership = ReservationOwnership(data_source)
    [reservation] = ownership.reserve_samples(1)
    stage_id = ReservationStageId("weights-10")
    [executor_receipt] = ownership.begin_execution([reservation], stage_id=stage_id)
    [terminal_receipt] = ownership.record_terminal([executor_receipt], stage_id=stage_id)
    reservation.samples[0].response = "generated output must not replace source replay"

    ownership.rollback_batch([terminal_receipt])
    with pytest.raises(RuntimeError) as duplicate_terminal_error:
        ownership.record_terminal([executor_receipt], stage_id=stage_id)
    with pytest.raises(RuntimeError) as duplicate_rollback_error:
        ownership.rollback_batch([terminal_receipt])
    with pytest.raises(RuntimeError) as rollback_commit_error:
        ownership.commit_batch([terminal_receipt], rollout_id=12)

    assert terminal_receipt == ReservationTerminalReceipt(
        executor_receipt=executor_receipt,
        disposition=ReservationTerminalDisposition.TRAINABLE,
    )
    assert str(duplicate_terminal_error.value) == (
        "Cannot record terminal callback for executor receipt 0; reservation is rolled_back."
    )
    assert str(duplicate_rollback_error.value) == (
        "Cannot roll back terminal receipt 0; receipt is not exact trainable ownership."
    )
    assert str(rollback_commit_error.value) == (
        "Cannot commit terminal receipt 0; receipt is not exact trainable ownership."
    )
    assert data_source.acknowledged == []
    assert data_source.requeued == [[source_reservation]]
    assert data_source.requeued[0][0] is source_reservation


def test_reserved_groups_can_roll_back_before_executor_admission() -> None:
    source_reservations = [_reservation(5), _reservation(6)]
    data_source = _RecordingDataSource(source_reservations)
    ownership = ReservationOwnership(data_source)
    reservations = ownership.reserve_samples(2)

    ownership.rollback_reserved(reservations)

    assert reservations == source_reservations
    assert data_source.acknowledged == []
    assert data_source.requeued == [source_reservations]


def test_failed_reserved_rollback_leaves_reservation_retryable() -> None:
    source_reservation = _reservation(7)
    requeue_error = RuntimeError("reserved requeue failed")
    data_source = _FailingSettlementDataSource(
        [source_reservation],
        acknowledge_error=None,
        requeue_error=requeue_error,
    )
    ownership = ReservationOwnership(data_source)
    [reservation] = ownership.reserve_samples(1)

    with pytest.raises(RuntimeError) as error:
        ownership.rollback_reserved([reservation])

    assert error.value is requeue_error
    assert data_source.acknowledged == []
    assert data_source.requeued == []

    data_source.requeue_error = None
    ownership.rollback_reserved([reservation])
    assert data_source.requeued == [[source_reservation]]


def test_invalid_stage_id_does_not_start_reserved_execution() -> None:
    source_reservation = _reservation(8)
    data_source = _RecordingDataSource([source_reservation])
    ownership = ReservationOwnership(data_source)
    [reservation] = ownership.reserve_samples(1)

    with pytest.raises(ValueError) as error:
        ownership.begin_execution([reservation], stage_id=ReservationStageId(""))

    stage_id = ReservationStageId("weights-11")
    [receipt] = ownership.begin_execution([reservation], stage_id=stage_id)
    assert str(error.value) == "stage_id must be a nonempty string, got ''."
    assert receipt.reservation_id == source_reservation.reservation_id
    assert data_source.acknowledged == []
    assert data_source.requeued == []


def test_executor_receipt_return_interrupt_poison_ownership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_reservation = _reservation(9)
    data_source = _RecordingDataSource([source_reservation])
    ownership = ReservationOwnership(data_source)
    [reservation] = ownership.reserve_samples(1)
    stage_id = ReservationStageId("weights-12")
    interrupt = KeyboardInterrupt()
    monkeypatch.setattr(ownership, "_lock", _InterruptingExitLock(interrupt))

    with pytest.raises(KeyboardInterrupt) as error:
        ownership.begin_execution([reservation], stage_id=stage_id)

    assert error.value is interrupt
    assert data_source.acknowledged == []
    assert data_source.requeued == []

    with pytest.raises(ReservationOwnershipPoisonedError) as poisoned_error:
        ownership.reserve_samples(1)
    assert str(poisoned_error.value) == (
        "Reservation ownership is poisoned because transition "
        "'begin reservation execution' may have partially completed after KeyboardInterrupt(). "
        "Recover from a durable checkpoint."
    )
    assert poisoned_error.value.__cause__ is interrupt


def test_transition_gate_return_interrupt_poison_ownership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_reservation = _reservation(10)
    data_source = _RecordingDataSource([source_reservation])
    ownership = ReservationOwnership(data_source)
    [reservation] = ownership.reserve_samples(1)
    stage_id = ReservationStageId("weights-13")
    interrupt = KeyboardInterrupt()
    monkeypatch.setattr(ownership, "_transition_lock", _InterruptingExitLock(interrupt))

    with pytest.raises(KeyboardInterrupt) as error:
        ownership.begin_execution([reservation], stage_id=stage_id)

    assert error.value is interrupt
    assert data_source.acknowledged == []
    assert data_source.requeued == []

    with pytest.raises(ReservationOwnershipPoisonedError) as poisoned_error:
        ownership.reserve_samples(1)
    assert str(poisoned_error.value) == (
        "Reservation ownership is poisoned because transition "
        "'return from begin_execution' may have partially completed after KeyboardInterrupt(). "
        "Recover from a durable checkpoint."
    )
    assert poisoned_error.value.__cause__ is interrupt


def test_mixed_stage_cancellation_fails_before_changing_valid_receipts() -> None:
    source_reservations = [_reservation(7), _reservation(8)]
    data_source = _RecordingDataSource(source_reservations)
    ownership = ReservationOwnership(data_source)
    first, second = ownership.reserve_samples(2)
    first_stage = ReservationStageId("weights-11")
    second_stage = ReservationStageId("weights-12")
    [first_receipt] = ownership.begin_execution([first], stage_id=first_stage)
    [second_receipt] = ownership.begin_execution([second], stage_id=second_stage)

    with pytest.raises(RuntimeError) as error:
        ownership.request_cancellation([first_receipt, second_receipt], stage_id=first_stage)

    first_terminal = ownership.record_terminal([first_receipt], stage_id=first_stage)
    second_terminal = ownership.record_terminal([second_receipt], stage_id=second_stage)
    assert str(error.value) == (
        "Cannot request cancellation for executor receipt 1; " "receipt is not owned by stage 'weights-11'."
    )
    assert first_terminal == [ReservationTerminalReceipt(first_receipt, ReservationTerminalDisposition.TRAINABLE)]
    assert second_terminal == [ReservationTerminalReceipt(second_receipt, ReservationTerminalDisposition.TRAINABLE)]
    assert data_source.acknowledged == []
    assert data_source.requeued == []


def test_equal_reservation_clone_cannot_replace_exact_source_attempt() -> None:
    source_reservation = _reservation(9)
    data_source = _RecordingDataSource([source_reservation])
    ownership = ReservationOwnership(data_source)
    [reservation] = ownership.reserve_samples(1)
    cloned_reservation = SourceReservation(
        reservation_id=reservation.reservation_id,
        samples=reservation.samples,
    )
    stage_id = ReservationStageId("weights-13")

    with pytest.raises(RuntimeError) as error:
        ownership.begin_execution([cloned_reservation], stage_id=stage_id)

    [executor_receipt] = ownership.begin_execution([reservation], stage_id=stage_id)
    assert str(error.value) == "Source reservation 9 is not an exact reserved attempt owned here."
    assert executor_receipt.reservation_id == SourceReservationId(9)
    assert data_source.acknowledged == []
    assert data_source.requeued == []


def test_foreign_and_copied_executor_receipts_cannot_settle_live_execution() -> None:
    source_reservation = _reservation(10)
    data_source = _RecordingDataSource([source_reservation])
    ownership = ReservationOwnership(data_source)
    foreign_ownership = ReservationOwnership(_RecordingDataSource([]))
    [reservation] = ownership.reserve_samples(1)
    stage_id = ReservationStageId("weights-14")
    [executor_receipt] = ownership.begin_execution([reservation], stage_id=stage_id)
    copied_receipt = replace(executor_receipt)

    with pytest.raises(RuntimeError) as foreign_error:
        foreign_ownership.record_terminal([executor_receipt], stage_id=stage_id)
    with pytest.raises(RuntimeError) as copied_error:
        ownership.record_terminal([copied_receipt], stage_id=stage_id)

    terminal_receipts = ownership.record_terminal([executor_receipt], stage_id=stage_id)
    expected_error = (
        "Cannot record terminal callback for executor receipt 0; " "receipt is not owned by stage 'weights-14'."
    )
    assert str(foreign_error.value) == expected_error
    assert str(copied_error.value) == expected_error
    assert terminal_receipts == [
        ReservationTerminalReceipt(executor_receipt, ReservationTerminalDisposition.TRAINABLE)
    ]
    assert data_source.acknowledged == []
    assert data_source.requeued == []


def test_batch_commit_preflights_duplicate_and_copied_terminal_receipts() -> None:
    source_reservations = [_reservation(11), _reservation(12)]
    data_source = _RecordingDataSource(source_reservations)
    ownership = ReservationOwnership(data_source)
    reservations = ownership.reserve_samples(2)
    stage_id = ReservationStageId("weights-15")
    executor_receipts = ownership.begin_execution(reservations, stage_id=stage_id)
    terminal_receipts = ownership.record_terminal(executor_receipts, stage_id=stage_id)
    copied_second = ReservationTerminalReceipt(
        executor_receipt=terminal_receipts[1].executor_receipt,
        disposition=terminal_receipts[1].disposition,
    )

    with pytest.raises(ValueError) as duplicate_error:
        ownership.commit_batch([terminal_receipts[0], terminal_receipts[0]], rollout_id=12)
    with pytest.raises(RuntimeError) as copied_error:
        ownership.commit_batch([terminal_receipts[0], copied_second], rollout_id=12)

    ownership.commit_batch(terminal_receipts, rollout_id=12)
    assert str(duplicate_error.value) == "Terminal receipt batch contains duplicate identities: [0, 0]."
    assert str(copied_error.value) == ("Cannot commit terminal receipt 1; receipt is not exact trainable ownership.")
    assert data_source.acknowledged == [(source_reservations, 12)]
    assert data_source.requeued == []


def test_terminal_batch_rejects_duplicates_before_requeueing_any_reservation() -> None:
    source_reservations = [_reservation(13), _reservation(14)]
    data_source = _RecordingDataSource(source_reservations)
    ownership = ReservationOwnership(data_source)
    reservations = ownership.reserve_samples(2)
    stage_id = ReservationStageId("weights-16")
    executor_receipts = ownership.begin_execution(reservations, stage_id=stage_id)
    ownership.request_cancellation(executor_receipts, stage_id=stage_id)

    with pytest.raises(ValueError) as error:
        ownership.record_terminal([executor_receipts[0], executor_receipts[0]], stage_id=stage_id)

    assert str(error.value) == "Executor receipt batch contains duplicate identities: [0, 0]."
    assert data_source.acknowledged == []
    assert data_source.requeued == []

    terminal_receipts = ownership.record_terminal(executor_receipts, stage_id=stage_id)
    assert terminal_receipts == [
        ReservationTerminalReceipt(executor_receipts[0], ReservationTerminalDisposition.CANCELLED),
        ReservationTerminalReceipt(executor_receipts[1], ReservationTerminalDisposition.CANCELLED),
    ]
    assert data_source.requeued == [source_reservations]


def test_cancellation_after_trainable_completion_requires_batch_rollback() -> None:
    source_reservation = _reservation(15)
    data_source = _RecordingDataSource([source_reservation])
    ownership = ReservationOwnership(data_source)
    [reservation] = ownership.reserve_samples(1)
    stage_id = ReservationStageId("weights-17")
    [executor_receipt] = ownership.begin_execution([reservation], stage_id=stage_id)
    [terminal_receipt] = ownership.record_terminal([executor_receipt], stage_id=stage_id)

    with pytest.raises(RuntimeError) as error:
        ownership.request_cancellation([executor_receipt], stage_id=stage_id)

    assert str(error.value) == (
        "Cannot request cancellation for executor receipt 0; "
        "expected stage 'weights-17' in states executing, cancellation_requested."
    )
    assert data_source.acknowledged == []
    assert data_source.requeued == []

    ownership.rollback_batch([terminal_receipt])
    assert data_source.requeued == [[source_reservation]]


def test_failed_acknowledgement_leaves_trainable_receipt_retryable() -> None:
    source_reservation = _reservation(16)
    acknowledgement_error = RuntimeError("source acknowledgement failed")
    data_source = _FailingSettlementDataSource(
        [source_reservation],
        acknowledge_error=acknowledgement_error,
        requeue_error=None,
    )
    ownership = ReservationOwnership(data_source)
    [reservation] = ownership.reserve_samples(1)
    stage_id = ReservationStageId("weights-18")
    [executor_receipt] = ownership.begin_execution([reservation], stage_id=stage_id)
    [terminal_receipt] = ownership.record_terminal([executor_receipt], stage_id=stage_id)

    with pytest.raises(RuntimeError) as error:
        ownership.commit_batch([terminal_receipt], rollout_id=13)

    assert error.value is acknowledgement_error
    assert data_source.acknowledged == []
    assert data_source.requeued == []

    data_source.acknowledge_error = None
    ownership.commit_batch([terminal_receipt], rollout_id=13)
    assert data_source.acknowledged == [([source_reservation], 13)]


def test_failed_cancelled_requeue_leaves_terminal_callback_retryable() -> None:
    source_reservation = _reservation(17)
    requeue_error = RuntimeError("source requeue failed")
    data_source = _FailingSettlementDataSource(
        [source_reservation],
        acknowledge_error=None,
        requeue_error=requeue_error,
    )
    ownership = ReservationOwnership(data_source)
    [reservation] = ownership.reserve_samples(1)
    stage_id = ReservationStageId("weights-19")
    [executor_receipt] = ownership.begin_execution([reservation], stage_id=stage_id)
    ownership.request_cancellation([executor_receipt], stage_id=stage_id)

    with pytest.raises(RuntimeError) as error:
        ownership.record_terminal([executor_receipt], stage_id=stage_id)

    assert error.value is requeue_error
    assert data_source.acknowledged == []
    assert data_source.requeued == []

    data_source.requeue_error = None
    terminal_receipts = ownership.record_terminal([executor_receipt], stage_id=stage_id)
    assert terminal_receipts == [
        ReservationTerminalReceipt(executor_receipt, ReservationTerminalDisposition.CANCELLED)
    ]
    assert data_source.requeued == [[source_reservation]]


def test_stale_receipt_from_requeued_attempt_is_rejected_without_touching_new_attempt() -> None:
    first_attempt = _reservation(18)
    data_source = _RecordingDataSource([first_attempt])
    ownership = ReservationOwnership(data_source)
    [first_reservation] = ownership.reserve_samples(1)
    first_stage = ReservationStageId("weights-20")
    [first_executor_receipt] = ownership.begin_execution([first_reservation], stage_id=first_stage)
    ownership.request_cancellation([first_executor_receipt], stage_id=first_stage)
    ownership.record_terminal([first_executor_receipt], stage_id=first_stage)

    second_attempt = _reservation(18)
    data_source.reservations = [second_attempt]
    [second_reservation] = ownership.reserve_samples(1)
    second_stage = ReservationStageId("weights-21")
    [second_executor_receipt] = ownership.begin_execution([second_reservation], stage_id=second_stage)

    with pytest.raises(RuntimeError) as stale_error:
        ownership.record_terminal([first_executor_receipt], stage_id=first_stage)
    second_terminal_receipts = ownership.record_terminal([second_executor_receipt], stage_id=second_stage)
    ownership.commit_batch(second_terminal_receipts, rollout_id=14)

    assert str(stale_error.value) == (
        "Cannot record terminal callback for executor receipt 0; reservation is cancelled."
    )
    assert second_terminal_receipts == [
        ReservationTerminalReceipt(second_executor_receipt, ReservationTerminalDisposition.TRAINABLE)
    ]
    assert first_reservation is first_attempt
    assert second_reservation is second_attempt
    assert data_source.requeued == [[first_attempt]]
    assert data_source.acknowledged == [([second_attempt], 14)]


def test_failed_batch_rollback_leaves_trainable_receipt_retryable() -> None:
    source_reservation = _reservation(19)
    requeue_error = RuntimeError("batch requeue failed")
    data_source = _FailingSettlementDataSource(
        [source_reservation],
        acknowledge_error=None,
        requeue_error=requeue_error,
    )
    ownership = ReservationOwnership(data_source)
    [reservation] = ownership.reserve_samples(1)
    stage_id = ReservationStageId("weights-22")
    [executor_receipt] = ownership.begin_execution([reservation], stage_id=stage_id)
    [terminal_receipt] = ownership.record_terminal([executor_receipt], stage_id=stage_id)

    with pytest.raises(RuntimeError) as error:
        ownership.rollback_batch([terminal_receipt])

    assert error.value is requeue_error
    assert data_source.acknowledged == []
    assert data_source.requeued == []

    data_source.requeue_error = None
    ownership.rollback_batch([terminal_receipt])
    assert data_source.requeued == [[source_reservation]]


def test_settled_then_interrupted_acknowledgement_poison_ownership() -> None:
    source_reservation = _reservation(20)
    interrupt = KeyboardInterrupt()
    data_source = _AcknowledgingThenInterruptingDataSource([source_reservation], interrupt)
    ownership = ReservationOwnership(data_source)
    [reservation] = ownership.reserve_samples(1)
    stage_id = ReservationStageId("weights-23")
    [executor_receipt] = ownership.begin_execution([reservation], stage_id=stage_id)
    [terminal_receipt] = ownership.record_terminal([executor_receipt], stage_id=stage_id)

    with pytest.raises(KeyboardInterrupt) as error:
        ownership.commit_batch([terminal_receipt], rollout_id=15)

    assert error.value is interrupt
    assert data_source.acknowledged == [([source_reservation], 15)]
    assert data_source.requeued == []

    with pytest.raises(ReservationOwnershipPoisonedError) as poisoned_error:
        ownership.rollback_batch([terminal_receipt])
    assert str(poisoned_error.value) == (
        "Reservation ownership is poisoned because transition "
        "'commit trainable reservations' may have partially completed after KeyboardInterrupt(). "
        "Recover from a durable checkpoint."
    )
    assert poisoned_error.value.__cause__ is interrupt


def test_settlement_return_interrupt_poison_ownership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_reservation = _reservation(21)
    data_source = _RecordingDataSource([source_reservation])
    ownership = ReservationOwnership(data_source)
    [reservation] = ownership.reserve_samples(1)
    stage_id = ReservationStageId("weights-24")
    [executor_receipt] = ownership.begin_execution([reservation], stage_id=stage_id)
    [terminal_receipt] = ownership.record_terminal([executor_receipt], stage_id=stage_id)
    interrupt = KeyboardInterrupt()
    monkeypatch.setattr(ownership, "_lock", _InterruptingExitLock(interrupt))

    with pytest.raises(KeyboardInterrupt) as error:
        ownership.commit_batch([terminal_receipt], rollout_id=16)

    assert error.value is interrupt
    assert data_source.acknowledged == [([source_reservation], 16)]
    assert data_source.requeued == []

    with pytest.raises(ReservationOwnershipPoisonedError) as poisoned_error:
        ownership.reserve_samples(1)
    assert str(poisoned_error.value) == (
        "Reservation ownership is poisoned because transition "
        "'commit trainable reservations' may have partially completed after KeyboardInterrupt(). "
        "Recover from a durable checkpoint."
    )
    assert poisoned_error.value.__cause__ is interrupt


def test_transition_gate_blocks_competing_work_until_interrupt_is_poisoned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    committed_reservation = _reservation(22)
    untouched_reservation = _reservation(23)
    data_source = _RecordingDataSource([committed_reservation, untouched_reservation])
    ownership = ReservationOwnership(data_source)
    [reservation] = ownership.reserve_samples(1)
    stage_id = ReservationStageId("weights-25")
    [executor_receipt] = ownership.begin_execution([reservation], stage_id=stage_id)
    [terminal_receipt] = ownership.record_terminal([executor_receipt], stage_id=stage_id)
    interrupt = KeyboardInterrupt()
    exit_started = threading.Event()
    continue_exit = threading.Event()
    competing_started = threading.Event()
    competing_finished = threading.Event()
    monkeypatch.setattr(
        ownership,
        "_lock",
        _CoordinatedInterruptingExitLock(
            interrupt=interrupt,
            exit_started=exit_started,
            continue_exit=continue_exit,
        ),
    )

    def reserve_competing_work() -> list[SourceReservation]:
        competing_started.set()
        try:
            return ownership.reserve_samples(1)
        finally:
            competing_finished.set()

    with ThreadPoolExecutor(max_workers=2) as executor:
        commit_future = executor.submit(ownership.commit_batch, [terminal_receipt], rollout_id=17)
        assert exit_started.wait(timeout=5)
        competing_future = executor.submit(reserve_competing_work)
        assert competing_started.wait(timeout=5)
        assert not competing_finished.wait(timeout=0.2)

        continue_exit.set()
        with pytest.raises(KeyboardInterrupt) as error:
            commit_future.result(timeout=5)
        with pytest.raises(ReservationOwnershipPoisonedError) as poisoned_error:
            competing_future.result(timeout=5)

    assert error.value is interrupt
    assert str(poisoned_error.value) == (
        "Reservation ownership is poisoned because transition "
        "'commit trainable reservations' may have partially completed after KeyboardInterrupt(). "
        "Recover from a durable checkpoint."
    )
    assert poisoned_error.value.__cause__ is interrupt
    assert data_source.reservations == [untouched_reservation]
    assert data_source.acknowledged == [([committed_reservation], 17)]
    assert data_source.requeued == []


def test_post_settlement_bookkeeping_interrupt_poison_ownership() -> None:
    source_reservation = _reservation(21)
    data_source = _RecordingDataSource([source_reservation])
    ownership = ReservationOwnership(data_source)
    [reservation] = ownership.reserve_samples(1)
    stage_id = ReservationStageId("weights-24")
    [executor_receipt] = ownership.begin_execution([reservation], stage_id=stage_id)
    [terminal_receipt] = ownership.record_terminal([executor_receipt], stage_id=stage_id)
    interrupt = KeyboardInterrupt()
    ownership._records = _InterruptingRecordMap(ownership._records, interrupt)

    with pytest.raises(KeyboardInterrupt) as error:
        ownership.commit_batch([terminal_receipt], rollout_id=16)

    assert error.value is interrupt
    assert data_source.acknowledged == [([source_reservation], 16)]
    assert data_source.requeued == []

    with pytest.raises(ReservationOwnershipPoisonedError) as poisoned_error:
        ownership.reserve_samples(1)
    assert str(poisoned_error.value) == (
        "Reservation ownership is poisoned because transition "
        "'commit trainable reservations' may have partially completed after KeyboardInterrupt(). "
        "Recover from a durable checkpoint."
    )
    assert poisoned_error.value.__cause__ is interrupt


def test_cancellation_and_terminal_callback_race_is_serialized() -> None:
    source_reservation = _reservation(22)
    data_source = _RecordingDataSource([source_reservation])
    ownership = ReservationOwnership(data_source)
    [reservation] = ownership.reserve_samples(1)
    stage_id = ReservationStageId("weights-25")
    [executor_receipt] = ownership.begin_execution([reservation], stage_id=stage_id)
    start = threading.Barrier(2)

    def request_cancellation() -> RuntimeError | None:
        start.wait(timeout=5)
        try:
            ownership.request_cancellation([executor_receipt], stage_id=stage_id)
        except RuntimeError as error:
            return error
        return None

    def record_terminal() -> ReservationTerminalReceipt:
        start.wait(timeout=5)
        [terminal_receipt] = ownership.record_terminal([executor_receipt], stage_id=stage_id)
        return terminal_receipt

    with ThreadPoolExecutor(max_workers=2) as executor:
        cancellation_future = executor.submit(request_cancellation)
        terminal_future = executor.submit(record_terminal)
        cancellation_error = cancellation_future.result(timeout=5)
        terminal_receipt = terminal_future.result(timeout=5)

    if terminal_receipt.disposition is ReservationTerminalDisposition.CANCELLED:
        assert cancellation_error is None
        assert terminal_receipt == ReservationTerminalReceipt(
            executor_receipt,
            ReservationTerminalDisposition.CANCELLED,
        )
        assert data_source.requeued == [[source_reservation]]
    else:
        assert isinstance(cancellation_error, RuntimeError)
        assert str(cancellation_error) == (
            "Cannot request cancellation for executor receipt 0; "
            "expected stage 'weights-25' in states executing, cancellation_requested."
        )
        assert terminal_receipt == ReservationTerminalReceipt(
            executor_receipt,
            ReservationTerminalDisposition.TRAINABLE,
        )
        assert data_source.requeued == []
    assert data_source.acknowledged == []


def test_duplicate_terminal_callback_race_records_one_result_and_rejects_the_duplicate() -> None:
    source_reservation = _reservation(23)
    data_source = _RecordingDataSource([source_reservation])
    ownership = ReservationOwnership(data_source)
    [reservation] = ownership.reserve_samples(1)
    stage_id = ReservationStageId("weights-26")
    [executor_receipt] = ownership.begin_execution([reservation], stage_id=stage_id)
    start = threading.Barrier(2)

    def record_terminal() -> ReservationTerminalReceipt | RuntimeError:
        start.wait(timeout=5)
        try:
            [terminal_receipt] = ownership.record_terminal([executor_receipt], stage_id=stage_id)
            return terminal_receipt
        except RuntimeError as error:
            return error

    with ThreadPoolExecutor(max_workers=2) as executor:
        first_future = executor.submit(record_terminal)
        second_future = executor.submit(record_terminal)
        outcomes = [
            first_future.result(timeout=5),
            second_future.result(timeout=5),
        ]

    terminal_receipts = [outcome for outcome in outcomes if isinstance(outcome, ReservationTerminalReceipt)]
    duplicate_errors = [outcome for outcome in outcomes if isinstance(outcome, RuntimeError)]
    assert terminal_receipts == [
        ReservationTerminalReceipt(executor_receipt, ReservationTerminalDisposition.TRAINABLE),
    ]
    assert [str(error) for error in duplicate_errors] == [
        "Cannot record terminal callback for executor receipt 0; reservation is trainable."
    ]
    assert data_source.acknowledged == []
    assert data_source.requeued == []
