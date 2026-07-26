import copy
import json
from argparse import Namespace
from pathlib import Path

import pytest

import miles.rollout.data_source as data_source_module
from miles.rollout.data_source import (
    RolloutDataSource,
    RolloutDataSourceWithBuffer,
    SourceReservation,
    SourceReservationId,
)
from miles.utils.types import Sample


@pytest.fixture(autouse=True)
def patch_processors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(data_source_module, "load_tokenizer", lambda *args, **kwargs: object())
    monkeypatch.setattr(data_source_module, "load_processor", lambda *args, **kwargs: None)


def _make_args(tmp_path: Path, *, rollout_shuffle: bool) -> Namespace:
    prompt_path = tmp_path / "prompts.jsonl"
    prompt_path.write_text(
        "\n".join(json.dumps({"prompt": prompt}) for prompt in ("alpha", "bravo", "charlie", "delta")),
        encoding="utf-8",
    )
    return Namespace(
        rollout_global_dataset=True,
        hf_checkpoint="unused",
        chat_template_path=None,
        dump_details=None,
        prompt_data=str(prompt_path),
        rollout_max_prompt_len=None,
        input_key="prompt",
        multimodal_keys=None,
        label_key=None,
        metadata_key="metadata",
        tool_key=None,
        apply_chat_template=False,
        apply_chat_template_kwargs=None,
        rollout_seed=100,
        rollout_shuffle=rollout_shuffle,
        n_samples_per_prompt=2,
        save=str(tmp_path),
        load=str(tmp_path),
        buffer_filter_path=None,
    )


def _reservation(
    reservation_id: int,
    *,
    prompt: str,
    first_sample_index: int,
) -> SourceReservation:
    sample_indices = (first_sample_index, first_sample_index + 1)
    return SourceReservation(
        reservation_id=SourceReservationId(str(reservation_id)),
        samples=[
            Sample(group_index=reservation_id, index=sample_index, prompt=prompt) for sample_index in sample_indices
        ],
    )


def test_requeue_replays_pristine_holes_in_source_order(tmp_path: Path) -> None:
    source = RolloutDataSource(_make_args(tmp_path, rollout_shuffle=False))

    first, second, third = source.reserve_samples(3)
    assert [first, second, third] == [
        _reservation(0, prompt="alpha", first_sample_index=0),
        _reservation(1, prompt="bravo", first_sample_index=2),
        _reservation(2, prompt="charlie", first_sample_index=4),
    ]

    source.requeue_reservations([third])
    source.requeue_reservations([first])
    source.acknowledge_reservations([second], rollout_id=7)
    first.samples[0].response = "mutated after reservation"
    third.samples[0].prompt = "also mutated"

    replayed = source.reserve_samples(2)
    assert replayed == [
        _reservation(0, prompt="alpha", first_sample_index=0),
        _reservation(2, prompt="charlie", first_sample_index=4),
    ]

    source.acknowledge_reservations(replayed, rollout_id=8)
    assert source.reserve_samples(1) == [_reservation(3, prompt="delta", first_sample_index=6)]


def test_invalid_settlement_leaves_every_reservation_outstanding(tmp_path: Path) -> None:
    source = RolloutDataSource(_make_args(tmp_path, rollout_shuffle=False))
    first, second = source.reserve_samples(2)

    with pytest.raises(
        ValueError,
        match=r"Reservation settlement contains duplicate identities: \['0', '0'\]\.",
    ):
        source.acknowledge_reservations([first, first], rollout_id=0)

    unknown = SourceReservation(
        reservation_id=SourceReservationId("99"),
        samples=[Sample(group_index=99, index=99, prompt="unknown")],
    )
    with pytest.raises(
        RuntimeError,
        match=r"Source reservations are not the current outstanding attempts: \['99'\]\.",
    ):
        source.requeue_reservations([first, unknown])

    source.acknowledge_reservations([first, second], rollout_id=0)
    assert source.reserve_samples(1) == [_reservation(2, prompt="charlie", first_sample_index=4)]


def test_reissued_reservation_rejects_late_settlement_from_old_attempt(
    tmp_path: Path,
) -> None:
    source = RolloutDataSource(_make_args(tmp_path, rollout_shuffle=False))
    [first_attempt] = source.reserve_samples(1)
    source.requeue_reservations([first_attempt])

    [second_attempt] = source.reserve_samples(1)
    assert second_attempt == first_attempt
    assert second_attempt is not first_attempt

    with pytest.raises(
        RuntimeError,
        match=r"Source reservations are not the current outstanding attempts: \['0'\]\.",
    ):
        source.acknowledge_reservations([first_attempt], rollout_id=0)

    source.acknowledge_reservations([second_attempt], rollout_id=0)


def test_materialization_failure_does_not_advance_the_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = RolloutDataSource(_make_args(tmp_path, rollout_shuffle=False))
    original_deepcopy = copy.deepcopy
    copy_count = 0

    def fail_during_second_group(value: object) -> object:
        nonlocal copy_count
        copy_count += 1
        if copy_count == 3:
            raise RuntimeError("injected materialization failure")
        return original_deepcopy(value)

    monkeypatch.setattr(data_source_module.copy, "deepcopy", fail_during_second_group)
    with pytest.raises(RuntimeError, match="injected materialization failure"):
        source.reserve_samples(2)

    monkeypatch.setattr(data_source_module.copy, "deepcopy", original_deepcopy)
    assert source.reserve_samples(2) == [
        _reservation(0, prompt="alpha", first_sample_index=0),
        _reservation(1, prompt="bravo", first_sample_index=2),
    ]


def test_non_persistent_source_rejects_durable_reservations(tmp_path: Path) -> None:
    args = _make_args(tmp_path, rollout_shuffle=False)
    args.rollout_global_dataset = False
    source = RolloutDataSource(args)

    with pytest.raises(
        RuntimeError,
        match=(
            "RolloutDataSource does not support durable source reservations "
            "when rollout_global_dataset is disabled."
        ),
    ):
        source.reserve_samples(1)

    assert source.get_samples(1) == [_reservation(0, prompt="", first_sample_index=0).samples]


def test_legacy_get_samples_returns_empty_for_an_empty_dataset(tmp_path: Path) -> None:
    args = _make_args(tmp_path, rollout_shuffle=False)
    Path(args.prompt_data).write_text("", encoding="utf-8")
    source = RolloutDataSource(args)

    assert source.get_samples(1) == []
    with pytest.raises(ValueError, match="Cannot reserve samples from an empty rollout dataset."):
        source.reserve_samples(1)


def test_buffered_source_does_not_bypass_retry_buffer_for_reservations(tmp_path: Path) -> None:
    source = RolloutDataSourceWithBuffer(_make_args(tmp_path, rollout_shuffle=False))
    retry_group = _reservation(7, prompt="retry", first_sample_index=14).samples
    source.add_samples([retry_group])

    with pytest.raises(
        RuntimeError,
        match=(
            "RolloutDataSourceWithBuffer does not support durable source reservations "
            "because they would bypass its retry buffer."
        ),
    ):
        source.reserve_samples(1)

    assert source.get_samples(1) == [retry_group]
    assert source.get_buffer_length() == 0
