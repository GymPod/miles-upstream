import pytest

from miles.utils import distributed_utils


def test_run_on_local_rank_serialized_runs_on_local_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCAL_RANK", "2")

    gloo_group = object()
    events: list[tuple[str, object | None]] = []
    expected_value = object()

    monkeypatch.setattr(distributed_utils, "get_gloo_group", lambda: gloo_group)
    monkeypatch.setattr(
        distributed_utils.dist,
        "barrier",
        lambda *, group: events.append(("barrier", group)),
    )

    def operation() -> object:
        events.append(("operation", None))
        return expected_value

    result = distributed_utils.run_on_local_rank_serialized(operation, num_local_ranks=4)

    assert result is expected_value
    assert events == [
        ("barrier", gloo_group),
        ("barrier", gloo_group),
        ("operation", None),
        ("barrier", gloo_group),
        ("barrier", gloo_group),
    ]


def test_run_on_local_rank_serialized_rejects_out_of_range_rank(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCAL_RANK", "4")

    with pytest.raises(ValueError, match=r"LOCAL_RANK 4 must be in \[0, 4\)"):
        distributed_utils.run_on_local_rank_serialized(lambda: None, num_local_ranks=4)
