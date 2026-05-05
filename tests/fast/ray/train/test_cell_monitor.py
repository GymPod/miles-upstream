from unittest.mock import MagicMock

import pytest
import ray

from miles.ray.train.cell_monitor import compute_cell_status
from miles.ray.train.cell_state import (
    StateAllocatedAlive,
    StateAllocatedErrored,
    StateAllocatedUninitialized,
    StatePending,
    StateStopped,
)
from miles.utils.control_server.models import TriState
from miles.utils.indep_dp import IndepDPInfo


def _make_actor_handle_mock() -> MagicMock:
    return MagicMock(spec=ray.actor.ActorHandle)


def _make_indep_dp_info() -> IndepDPInfo:
    return IndepDPInfo(
        cell_index=0,
        num_cells=1,
        alive_rank=0,
        alive_size=1,
        quorum_id=1,
        alive_cell_indices=[0],
    )


def _make_alive_state() -> StateAllocatedAlive:
    return StateAllocatedAlive(actor_handles=[_make_actor_handle_mock()], indep_dp_info=_make_indep_dp_info())


def _find_condition(status, type_: str):
    matches = [c for c in status.conditions if c.type == type_]
    assert len(matches) == 1, f"expected exactly one {type_!r} condition, got {len(matches)}"
    return matches[0]


class TestComputeCellStatusAlive:
    def test_health_true_reports_healthy_true(self):
        result = compute_cell_status(_make_alive_state(), TriState.TRUE)

        assert result.phase == "Running"
        healthy = _find_condition(result, "Healthy")
        assert healthy.status == TriState.TRUE
        assert healthy.reason is None

    def test_health_false_reports_healthy_false_with_failed_reason(self):
        result = compute_cell_status(_make_alive_state(), TriState.FALSE)

        healthy = _find_condition(result, "Healthy")
        assert healthy.status == TriState.FALSE
        assert healthy.reason == "HealthCheckFailed"

    def test_health_unknown_reports_healthy_unknown_not_translated_to_true(self):
        """Regression: paused health checker reports UNKNOWN; previously this was
        silently translated to Healthy=TRUE, hiding the transient state from observers."""
        result = compute_cell_status(_make_alive_state(), TriState.UNKNOWN)

        healthy = _find_condition(result, "Healthy")
        assert healthy.status == TriState.UNKNOWN
        assert healthy.reason == "HealthCheckUnknown"


class TestComputeCellStatusOtherStates:
    @pytest.mark.parametrize("health_status", [TriState.TRUE, TriState.FALSE, TriState.UNKNOWN])
    def test_uninitialized_ignores_health_checker(self, health_status: TriState):
        state = StateAllocatedUninitialized(actor_handles=[_make_actor_handle_mock()])

        result = compute_cell_status(state, health_status)

        assert result.phase == "Running"
        healthy = _find_condition(result, "Healthy")
        assert healthy.status == TriState.TRUE

    @pytest.mark.parametrize("health_status", [TriState.TRUE, TriState.FALSE, TriState.UNKNOWN])
    def test_errored_always_reports_unhealthy(self, health_status: TriState):
        state = StateAllocatedErrored(actor_handles=[_make_actor_handle_mock()], indep_dp_info=_make_indep_dp_info())

        result = compute_cell_status(state, health_status)

        healthy = _find_condition(result, "Healthy")
        assert healthy.status == TriState.FALSE
        assert healthy.reason == "ExecutionErrored"

    def test_pending_reports_allocated_false_no_healthy_condition(self):
        result = compute_cell_status(StatePending(), TriState.UNKNOWN)

        assert result.phase == "Pending"
        allocated = _find_condition(result, "Allocated")
        assert allocated.status == TriState.FALSE
        assert all(c.type != "Healthy" for c in result.conditions)

    def test_stopped_reports_suspended_phase_no_healthy_condition(self):
        result = compute_cell_status(StateStopped(), TriState.UNKNOWN)

        assert result.phase == "Suspended"
        assert all(c.type != "Healthy" for c in result.conditions)
