import threading
from collections.abc import Iterator, Sequence
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
        "Reservation ownership is poisoned because transition 'compensate invalid source acquisition' may have partially completed after KeyboardInterrupt(). Recover from a durable checkpoint."
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
        "Reservation ownership is poisoned because transition 'resolve acquired source reservations' may have partially completed after KeyboardInterrupt(). Recover from a durable checkpoint."
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
        "Reservation ownership is poisoned because transition 'resolve acquired source reservations' may have partially completed after KeyboardInterrupt(). Recover from a durable checkpoint."
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
        "Reservation ownership is poisoned because transition 'begin reservation execution' may have partially completed after KeyboardInterrupt(). Recover from a durable checkpoint."
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
        "Reservation ownership is poisoned because transition 'return from begin_execution' may have partially completed after KeyboardInterrupt(). Recover from a durable checkpoint."
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

    first_record = ownership._records[first.reservation_id]
    second_record = ownership._records[second.reservation_id]
    assert str(error.value) == (
        "Cannot request cancellation for executor receipt 1; receipt is not owned by stage 'weights-11'."
    )
    assert first_record.state is ownership_module._ReservationState.EXECUTING
    assert second_record.state is ownership_module._ReservationState.EXECUTING
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


def test_foreign_and_copied_executor_receipts_cannot_cancel_live_execution() -> None:
    source_reservation = _reservation(10)
    data_source = _RecordingDataSource([source_reservation])
    ownership = ReservationOwnership(data_source)
    foreign_ownership = ReservationOwnership(_RecordingDataSource([]))
    [reservation] = ownership.reserve_samples(1)
    stage_id = ReservationStageId("weights-14")
    [executor_receipt] = ownership.begin_execution([reservation], stage_id=stage_id)
    copied_receipt = replace(executor_receipt)

    with pytest.raises(RuntimeError) as foreign_error:
        foreign_ownership.request_cancellation([executor_receipt], stage_id=stage_id)
    with pytest.raises(RuntimeError) as copied_error:
        ownership.request_cancellation([copied_receipt], stage_id=stage_id)

    ownership.request_cancellation([executor_receipt], stage_id=stage_id)
    expected_error = "Cannot request cancellation for executor receipt 0; receipt is not owned by stage 'weights-14'."
    assert str(foreign_error.value) == expected_error
    assert str(copied_error.value) == expected_error
    assert (
        ownership._records[reservation.reservation_id].state
        is ownership_module._ReservationState.CANCELLATION_REQUESTED
    )
    assert data_source.acknowledged == []
    assert data_source.requeued == []
