from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import wraps
from typing import Concatenate, NamedTuple, NewType, ParamSpec, TypeVar

from miles.rollout.data_source import DataSource, SourceReservation, SourceReservationId

ReservationStageId = NewType("ReservationStageId", str)
ReservationReceiptId = NewType("ReservationReceiptId", int)
_TransitionParameters = ParamSpec("_TransitionParameters")
_TransitionResult = TypeVar("_TransitionResult")


class ReservationAcquisitionRollbackError(RuntimeError):
    """Report a retained source rollback after invalid acquisition.

    Args:
        validation_error: Error that made the acquired batch invalid.
        rollback_error: Source error that prevented compensating requeue.

    Attributes:
        validation_error: Error that made the acquired batch invalid.
        rollback_error: Source error that prevented compensating requeue.
    """

    def __init__(
        self,
        validation_error: ValueError | RuntimeError,
        rollback_error: BaseException,
    ) -> None:
        super().__init__(f"Failed to requeue reservations after invalid acquisition: {rollback_error}")
        self.validation_error = validation_error
        self.rollback_error = rollback_error


class ReservationIdentityConflictError(RuntimeError):
    """Report ambiguous source attempts that permanently poison ownership."""


class ReservationOwnershipPoisonedError(RuntimeError):
    """Report ownership that requires recovery from a durable checkpoint."""


class _ReservationState(Enum):
    RESERVED = auto()
    EXECUTING = auto()
    CANCELLATION_REQUESTED = auto()
    ROLLED_BACK = auto()


@dataclass
class _ReservationRecord:
    reservation: SourceReservation
    state: _ReservationState
    stage_id: ReservationStageId | None
    executor_receipt: ReservationExecutorReceipt | None


@dataclass(frozen=True, eq=False)
class ReservationExecutorReceipt:
    """Identify one exact reservation execution attempt.

    Attributes:
        receipt_id: Owner-local identity for this execution attempt.
        reservation_id: Stable source identity for the prompt group.
        stage_id: Policy stage that submitted the execution.
    """

    receipt_id: ReservationReceiptId
    reservation_id: SourceReservationId
    stage_id: ReservationStageId
    _owner_token: object = field(repr=False)
    _record: _ReservationRecord = field(repr=False)


class _PoisonedTransition(NamedTuple):
    operation: str
    error: BaseException


@dataclass
class _TransitionProgress:
    operation: str
    source_started: bool
    state_may_have_changed: bool


def _serialize_transition(
    method: Callable[
        Concatenate[ReservationOwnership, _TransitionParameters],
        _TransitionResult,
    ],
) -> Callable[
    Concatenate[ReservationOwnership, _TransitionParameters],
    _TransitionResult,
]:
    @wraps(method)
    def serialized(
        ownership: ReservationOwnership,
        *args: _TransitionParameters.args,
        **kwargs: _TransitionParameters.kwargs,
    ) -> _TransitionResult:
        method_completed = False
        try:
            # Keep competitors out until method-level poisoning finishes after state-lock exit.
            with ownership._transition_lock:
                result = method(ownership, *args, **kwargs)
                method_completed = True
            return result
        except BaseException as error:
            if method_completed or not isinstance(error, Exception):
                ownership._poison_transition(f"return from {method.__name__}", error)
            raise

    return serialized


class ReservationOwnership:
    """Track exact source reservations across synchronous ownership transitions.

    A ``BaseException`` escaping a transition is fatal to the current rollout
    session. Callers must stop dispatching work and recover from a durable
    checkpoint; concurrent continuation during exception unwinding is not
    supported.

    Args:
        data_source: Source that owns durable reservation settlement.
    """

    def __init__(self, data_source: DataSource) -> None:
        self._data_source = data_source
        self._transition_lock = threading.RLock()
        self._lock = threading.RLock()
        self._owner_token = object()
        self._next_receipt_id = 0
        self._records: dict[SourceReservationId, _ReservationRecord] = {}
        self._pending_acquisition_rollback: list[SourceReservation] | None = None
        self._poisoned_identity_conflicts: tuple[SourceReservationId, ...] | None = None
        self._poisoned_transition: _PoisonedTransition | None = None

    @_serialize_transition
    def reserve_samples(self, num_groups: int) -> list[SourceReservation]:
        """Reserve and register exact source-owned prompt groups.

        Args:
            num_groups: Number of prompt groups to reserve.

        Returns:
            Exact source reservation attempts in source order.
        """
        progress = _TransitionProgress(
            operation="acquire source reservations",
            source_started=False,
            state_may_have_changed=False,
        )
        try:
            with self._lock:
                self._ensure_usable()
                if self._pending_acquisition_rollback is not None:
                    raise RuntimeError("Cannot reserve samples while an acquisition rollback is pending.")
                progress.source_started = True
                reservations = self._data_source.reserve_samples(num_groups)
                progress.operation = "resolve acquired source reservations"
                progress.state_may_have_changed = True
                reservation_ids = [reservation.reservation_id for reservation in reservations]
                conflicts = [reservation_id for reservation_id in reservation_ids if reservation_id in self._records]
                first_attempt_by_id: dict[SourceReservationId, SourceReservation] = {}
                ambiguous_conflicts: set[SourceReservationId] = set()
                for reservation in reservations:
                    first_attempt = first_attempt_by_id.setdefault(reservation.reservation_id, reservation)
                    existing = self._records.get(reservation.reservation_id)
                    if first_attempt is not reservation or (
                        existing is not None and existing.reservation is not reservation
                    ):
                        ambiguous_conflicts.add(reservation.reservation_id)
                if ambiguous_conflicts:
                    self._poisoned_identity_conflicts = tuple(sorted(ambiguous_conflicts))
                    progress.source_started = False
                    progress.state_may_have_changed = False
                    self._ensure_usable()

                validation_error: ValueError | RuntimeError | None = None
                if len(reservations) != num_groups:
                    validation_error = RuntimeError(
                        f"Data source returned {len(reservations)} reservations for a request of {num_groups}."
                    )
                elif len(reservation_ids) != len(set(reservation_ids)):
                    validation_error = ValueError(
                        f"Data source returned duplicate reservation identities: {reservation_ids}."
                    )
                elif conflicts:
                    validation_error = RuntimeError(f"Source reservations are already owned: {conflicts}.")

                if validation_error is not None:
                    rollback_reservations: list[SourceReservation] = []
                    seen_ids: set[SourceReservationId] = set()
                    for reservation in reservations:
                        existing = self._records.get(reservation.reservation_id)
                        if existing is not None and existing.reservation is reservation:
                            continue
                        if reservation.reservation_id in seen_ids:
                            continue
                        seen_ids.add(reservation.reservation_id)
                        rollback_reservations.append(reservation)
                    if rollback_reservations:
                        try:
                            self._data_source.requeue_reservations(rollback_reservations)
                        except BaseException as rollback_error:
                            self._pending_acquisition_rollback = rollback_reservations
                            if not isinstance(rollback_error, Exception):
                                self._poison_transition("compensate invalid source acquisition", rollback_error)
                                raise
                            progress.source_started = False
                            progress.state_may_have_changed = False
                            raise ReservationAcquisitionRollbackError(
                                validation_error=validation_error,
                                rollback_error=rollback_error,
                            ) from rollback_error
                    progress.source_started = False
                    progress.state_may_have_changed = False
                    raise validation_error

                records = {
                    reservation.reservation_id: _ReservationRecord(
                        reservation=reservation,
                        state=_ReservationState.RESERVED,
                        stage_id=None,
                        executor_receipt=None,
                    )
                    for reservation in reservations
                }
                self._records.update(records)
                return reservations
        except BaseException as error:
            self._poison_interrupted_transition(progress, error)
            raise

    @_serialize_transition
    def retry_failed_acquisition_rollback(self) -> None:
        """Retry source requeue retained after invalid acquisition.

        Raises:
            RuntimeError: If no failed acquisition rollback is pending.
        """
        progress = _TransitionProgress(
            operation="retry invalid acquisition rollback",
            source_started=False,
            state_may_have_changed=False,
        )
        try:
            with self._lock:
                self._ensure_usable()
                if self._pending_acquisition_rollback is None:
                    raise RuntimeError("No failed acquisition rollback is pending.")
                pending_reservations = self._pending_acquisition_rollback

                def clear_pending_rollback() -> None:
                    self._pending_acquisition_rollback = None

                self._apply_source_transition(
                    progress=progress,
                    source_transition=lambda: self._data_source.requeue_reservations(pending_reservations),
                    local_transition=clear_pending_rollback,
                )
            return
        except BaseException as error:
            self._poison_interrupted_transition(progress, error)
            raise

    @property
    def has_pending_acquisition_rollback(self) -> bool:
        """Return whether failed compensation still retains reservations."""
        with self._lock:
            return self._pending_acquisition_rollback is not None

    @_serialize_transition
    def begin_execution(
        self,
        reservations: Sequence[SourceReservation],
        *,
        stage_id: ReservationStageId,
    ) -> list[ReservationExecutorReceipt]:
        """Transfer reserved groups into stage-fenced executor ownership.

        Args:
            reservations: Exact reserved attempts to execute.
            stage_id: Policy stage that will execute the groups.

        Returns:
            Identity-sensitive receipts for terminal executor callbacks.
        """
        attempts = list(reservations)
        self._validate_stage_id(stage_id)
        progress = _TransitionProgress(
            operation="begin reservation execution",
            source_started=False,
            state_may_have_changed=False,
        )
        try:
            with self._lock:
                self._ensure_usable()
                records = self._require_reservations(attempts, expected_state=_ReservationState.RESERVED)
                receipts = [
                    ReservationExecutorReceipt(
                        receipt_id=ReservationReceiptId(self._next_receipt_id + offset),
                        reservation_id=record.reservation.reservation_id,
                        stage_id=stage_id,
                        _owner_token=self._owner_token,
                        _record=record,
                    )
                    for offset, record in enumerate(records)
                ]

                def start_execution() -> None:
                    self._next_receipt_id += len(receipts)
                    for record, receipt in zip(records, receipts, strict=True):
                        record.state = _ReservationState.EXECUTING
                        record.stage_id = stage_id
                        record.executor_receipt = receipt

                self._apply_local_transition(
                    progress=progress,
                    transition=start_execution,
                )
                return receipts
        except BaseException as error:
            self._poison_interrupted_transition(progress, error)
            raise

    @_serialize_transition
    def rollback_reserved(self, reservations: Sequence[SourceReservation]) -> None:
        """Requeue groups that never entered executor ownership.

        Args:
            reservations: Exact reserved attempts to return for pristine
                replay.
        """
        attempts = list(reservations)
        progress = _TransitionProgress(
            operation="roll back reserved reservations",
            source_started=False,
            state_may_have_changed=False,
        )
        try:
            with self._lock:
                self._ensure_usable()
                records = self._require_reservations(attempts, expected_state=_ReservationState.RESERVED)
                source_reservations = [record.reservation for record in records]

                def release_records() -> None:
                    for record in records:
                        record.state = _ReservationState.ROLLED_BACK
                        del self._records[record.reservation.reservation_id]

                self._apply_source_transition(
                    progress=progress,
                    source_transition=lambda: self._data_source.requeue_reservations(source_reservations),
                    local_transition=release_records,
                )
            return
        except BaseException as error:
            self._poison_interrupted_transition(progress, error)
            raise

    @_serialize_transition
    def request_cancellation(
        self,
        receipts: Sequence[ReservationExecutorReceipt],
        *,
        stage_id: ReservationStageId,
    ) -> None:
        """Request cancellation without releasing source ownership.

        Args:
            receipts: Exact active executor receipts to cancel.
            stage_id: Policy stage that owns the executions.
        """
        attempts = list(receipts)
        self._validate_stage_id(stage_id)
        progress = _TransitionProgress(
            operation="request reservation cancellation",
            source_started=False,
            state_may_have_changed=False,
        )
        try:
            with self._lock:
                self._ensure_usable()
                records = self._require_executor_receipts(
                    attempts,
                    stage_id=stage_id,
                    allowed_states=(
                        _ReservationState.EXECUTING,
                        _ReservationState.CANCELLATION_REQUESTED,
                    ),
                    operation="request cancellation for",
                )

                def mark_cancellation_requested() -> None:
                    for record in records:
                        record.state = _ReservationState.CANCELLATION_REQUESTED

                self._apply_local_transition(
                    progress=progress,
                    transition=mark_cancellation_requested,
                )
            return
        except BaseException as error:
            self._poison_interrupted_transition(progress, error)
            raise

    def _require_reservations(
        self,
        reservations: list[SourceReservation],
        *,
        expected_state: _ReservationState,
    ) -> list[_ReservationRecord]:
        reservation_ids = [reservation.reservation_id for reservation in reservations]
        if len(reservation_ids) != len(set(reservation_ids)):
            raise ValueError(f"Reservation batch contains duplicate identities: {reservation_ids}.")

        records: list[_ReservationRecord] = []
        for reservation in reservations:
            record = self._records.get(reservation.reservation_id)
            if record is None or record.reservation is not reservation or record.state is not expected_state:
                raise RuntimeError(
                    f"Source reservation {reservation.reservation_id} is not an exact {expected_state.name.lower()} attempt owned here."
                )
            records.append(record)
        return records

    def _require_executor_receipts(
        self,
        receipts: list[ReservationExecutorReceipt],
        *,
        stage_id: ReservationStageId,
        allowed_states: tuple[_ReservationState, ...],
        operation: str,
    ) -> list[_ReservationRecord]:
        self._validate_unique_executor_receipts(receipts)

        records: list[_ReservationRecord] = []
        for receipt in receipts:
            record = self._require_executor_receipt_authority(
                receipt,
                stage_id=stage_id,
                operation=operation,
            )
            if record.state not in allowed_states:
                expected_states = ", ".join(state.name.lower() for state in allowed_states)
                raise RuntimeError(
                    f"Cannot {operation} executor receipt {receipt.receipt_id}; expected stage {stage_id!r} in states {expected_states}."
                )
            records.append(record)
        return records

    def _require_executor_receipt_authority(
        self,
        receipt: ReservationExecutorReceipt,
        *,
        stage_id: ReservationStageId,
        operation: str,
    ) -> _ReservationRecord:
        record = receipt._record
        if (
            receipt._owner_token is not self._owner_token
            or record.executor_receipt is not receipt
            or receipt.stage_id != stage_id
            or record.stage_id != stage_id
        ):
            raise RuntimeError(
                f"Cannot {operation} executor receipt {receipt.receipt_id}; receipt is not owned by stage {stage_id!r}."
            )
        return record

    @staticmethod
    def _validate_unique_executor_receipts(receipts: list[ReservationExecutorReceipt]) -> None:
        receipt_ids = [receipt.receipt_id for receipt in receipts]
        if len(receipt_ids) != len(set(receipt_ids)):
            raise ValueError(f"Executor receipt batch contains duplicate identities: {receipt_ids}.")

    def _ensure_usable(self) -> None:
        if self._poisoned_identity_conflicts is not None:
            raise ReservationIdentityConflictError(
                f"Reservation ownership is poisoned by distinct attempts with the same identities: {list(self._poisoned_identity_conflicts)}."
            )
        if self._poisoned_transition is not None:
            transition = self._poisoned_transition
            raise ReservationOwnershipPoisonedError(
                f"Reservation ownership is poisoned because transition {transition.operation!r} may have partially completed after {transition.error!r}. Recover from a durable checkpoint."
            ) from transition.error

    def _apply_local_transition(
        self,
        *,
        progress: _TransitionProgress,
        transition: Callable[[], None],
    ) -> None:
        progress.state_may_have_changed = True
        transition()

    def _apply_source_transition(
        self,
        *,
        progress: _TransitionProgress,
        source_transition: Callable[[], None],
        local_transition: Callable[[], None],
    ) -> None:
        progress.source_started = True
        source_transition()
        progress.state_may_have_changed = True
        local_transition()

    def _poison_interrupted_transition(
        self,
        progress: _TransitionProgress,
        error: BaseException,
    ) -> None:
        if progress.state_may_have_changed or (progress.source_started and not isinstance(error, Exception)):
            self._poison_transition(progress.operation, error)

    def _poison_transition(self, operation: str, error: BaseException) -> None:
        with self._lock:
            if self._poisoned_transition is None:
                self._poisoned_transition = _PoisonedTransition(operation=operation, error=error)

    @staticmethod
    def _validate_stage_id(stage_id: ReservationStageId) -> None:
        if not isinstance(stage_id, str) or not stage_id:
            raise ValueError(f"stage_id must be a nonempty string, got {stage_id!r}.")
