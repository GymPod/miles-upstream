from pathlib import Path

from miles.utils.event_logger.logger import read_events
from miles.utils.event_logger.models import CellReconfigureEvent
from miles.utils.pydantic_utils import FrozenStrictBaseModel


class ExpectedReconfigure(FrozenStrictBaseModel):
    rollout_id: int
    src_cell_index: int | None
    healed_cell_indices: list[int]
    alive_cell_indices_after: list[int]


def assert_reconfigure_events(event_dir: Path, *, expected: list[ExpectedReconfigure]) -> None:
    actual = [_shape_of(event) for event in load_reconfigure_events(event_dir)]
    assert actual == expected, (
        f"CellReconfigureEvent sequence mismatch in {event_dir}:\n" f"  expected: {expected}\n" f"  actual:   {actual}"
    )


def assert_soak_reconfigure_events(
    event_dir: Path, *, num_successful_injections: int, num_cells: int, final_rollout_id: int
) -> None:
    events = load_reconfigure_events(event_dir)
    healings = [event for event in events if event.healed_cell_indices]

    if num_successful_injections > 0:
        assert healings, (
            f"Healing witness failed in {event_dir}: the fault injector reported "
            f"{num_successful_injections} successful injection(s) but no healing event was emitted "
            f"(reconfigure events: {[_shape_of(event) for event in events]})"
        )

    full_membership = list(range(num_cells))
    checked_events = list(events)
    if checked_events and _is_tolerated_trailing_shrink(
        checked_events[-1], full_membership=full_membership, final_rollout_id=final_rollout_id
    ):
        checked_events = checked_events[:-1]

    if checked_events:
        assert checked_events[-1].alive_cell_indices_after == full_membership, (
            f"Soak run must end fully healed (modulo at most one trailing shrink at final rollout "
            f"{final_rollout_id}): last checked reconfigure event in {event_dir} left alive cells "
            f"{checked_events[-1].alive_cell_indices_after}, expected {full_membership} "
            f"(full reconfigure sequence: {[_shape_of(event) for event in events]})"
        )

    print(
        f"Soak reconfigure witness assertion passed: {len(events)} reconfigure event(s) "
        f"({len(healings)} healing(s)) for {num_successful_injections} successful injection(s) in {event_dir}"
    )


def load_reconfigure_events(event_dir: Path) -> list[CellReconfigureEvent]:
    return [event for event in read_events(event_dir) if isinstance(event, CellReconfigureEvent)]


def _is_tolerated_trailing_shrink(
    event: CellReconfigureEvent, *, full_membership: list[int], final_rollout_id: int
) -> bool:
    return (
        event.alive_cell_indices_after != full_membership
        and not event.healed_cell_indices
        and event.rollout_id == final_rollout_id
    )


def _shape_of(event: CellReconfigureEvent) -> ExpectedReconfigure:
    return ExpectedReconfigure(
        rollout_id=event.rollout_id,
        src_cell_index=event.src_cell_index,
        healed_cell_indices=event.healed_cell_indices,
        alive_cell_indices_after=event.alive_cell_indices_after,
    )
