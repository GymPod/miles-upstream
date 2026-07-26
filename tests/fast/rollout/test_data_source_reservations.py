import copy
import json
import threading
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy
import pytest
import torch
from PIL import Image

import miles.rollout.data_source as data_source_module
import miles.utils.processing_utils as processing_utils
from miles.rollout.data_source import (
    RolloutDataSource,
    RolloutDataSourceWithBuffer,
    SourceReservation,
    SourceReservationId,
)
from miles.utils import chat_template_utils
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


def test_restart_replays_pristine_holes_before_advancing_source(tmp_path: Path) -> None:
    args = _make_args(tmp_path, rollout_shuffle=False)
    source = RolloutDataSource(args)

    first, second, third = source.reserve_samples(3)
    assert [first, second, third] == [
        _reservation(0, prompt="alpha", first_sample_index=0),
        _reservation(1, prompt="bravo", first_sample_index=2),
        _reservation(2, prompt="charlie", first_sample_index=4),
    ]

    source.acknowledge_reservations([second], rollout_id=7)
    source.requeue_reservations([third])
    first.samples[0].response = "mutated after reservation"
    third.samples[0].prompt = "also mutated"
    source.save(rollout_id=7)

    restored = RolloutDataSource(args)
    restored.load(rollout_id=7)

    replayed = restored.reserve_samples(2)
    assert replayed == [
        _reservation(0, prompt="alpha", first_sample_index=0),
        _reservation(2, prompt="charlie", first_sample_index=4),
    ]

    restored.acknowledge_reservations(replayed, rollout_id=8)
    assert restored.reserve_samples(1) == [_reservation(3, prompt="delta", first_sample_index=6)]


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


def test_checkpoint_replays_only_work_after_checkpoint_rollout(tmp_path: Path) -> None:
    args = _make_args(tmp_path, rollout_shuffle=False)
    source = RolloutDataSource(args)
    first, second, third = source.reserve_samples(3)
    source.acknowledge_reservations([first], rollout_id=5)
    source.acknowledge_reservations([second], rollout_id=6)

    first.samples[0].prompt = "mutated"
    second.samples[0].prompt = "mutated"
    third.samples[0].prompt = "mutated"
    source.save(rollout_id=5)

    restored = RolloutDataSource(args)
    restored.load(rollout_id=5)

    assert restored.reserve_samples(2) == [
        _reservation(1, prompt="bravo", first_sample_index=2),
        _reservation(2, prompt="charlie", first_sample_index=4),
    ]


def test_shuffled_multi_epoch_reservations_reconstruct_exact_groups(tmp_path: Path) -> None:
    args = _make_args(tmp_path, rollout_shuffle=True)
    source = RolloutDataSource(args)
    reservations = source.reserve_samples(10)
    expected = copy.deepcopy(reservations)

    for reservation in reservations:
        reservation.samples[0].prompt = "mutated"
    source.save(rollout_id=11)

    restored = RolloutDataSource(args)
    restored.load(rollout_id=11)

    assert restored.reserve_samples(10) == expected


def test_legacy_get_samples_is_immediately_acknowledged(tmp_path: Path) -> None:
    args = _make_args(tmp_path, rollout_shuffle=False)
    source = RolloutDataSource(args)

    assert source.get_samples(2) == [
        _reservation(0, prompt="alpha", first_sample_index=0).samples,
        _reservation(1, prompt="bravo", first_sample_index=2).samples,
    ]
    source.save(rollout_id=13)

    restored = RolloutDataSource(args)
    restored.load(rollout_id=13)

    assert restored.reserve_samples(1) == [_reservation(2, prompt="charlie", first_sample_index=4)]


def test_load_accepts_legacy_checkpoint_without_reservation_state(tmp_path: Path) -> None:
    args = _make_args(tmp_path, rollout_shuffle=False)
    checkpoint_path = tmp_path / "rollout" / "global_dataset_state_dict_15.pt"
    checkpoint_path.parent.mkdir()
    torch.save(
        {
            "sample_offset": 2,
            "epoch_id": 0,
            "sample_group_index": 2,
            "sample_index": 4,
            "metadata": {"legacy": True},
        },
        checkpoint_path,
    )

    restored = RolloutDataSource(args)
    restored.load(rollout_id=15)

    assert restored.reserve_samples(1) == [_reservation(2, prompt="charlie", first_sample_index=4)]
    assert restored.metadata == {"legacy": True}


def test_load_normalizes_legacy_cursor_from_group_frontier(tmp_path: Path) -> None:
    args = _make_args(tmp_path, rollout_shuffle=False)
    checkpoint_path = tmp_path / "rollout" / "global_dataset_state_dict_16.pt"
    checkpoint_path.parent.mkdir()
    torch.save(
        {
            "sample_offset": 6,
            "epoch_id": 1,
            "sample_group_index": 8,
            "sample_index": 16,
            "metadata": {},
        },
        checkpoint_path,
    )

    restored = RolloutDataSource(args)
    restored.load(rollout_id=16)

    assert restored.reserve_samples(1) == [_reservation(8, prompt="alpha", first_sample_index=16)]


def test_load_rejects_different_source_configuration(tmp_path: Path) -> None:
    args = _make_args(tmp_path, rollout_shuffle=False)
    source = RolloutDataSource(args)
    source.reserve_samples(1)
    source.save(rollout_id=17)

    incompatible_args = _make_args(tmp_path, rollout_shuffle=False)
    incompatible_args.n_samples_per_prompt = 3
    restored = RolloutDataSource(incompatible_args)

    with pytest.raises(
        ValueError,
        match="Source reservation checkpoint configuration does not match the current data source.",
    ):
        restored.load(rollout_id=17)


def test_load_rejects_different_prompt_processing_configuration(tmp_path: Path) -> None:
    args = _make_args(tmp_path, rollout_shuffle=False)
    source = RolloutDataSource(args)
    source.reserve_samples(1)
    source.save(rollout_id=19)

    incompatible_args = _make_args(tmp_path, rollout_shuffle=False)
    incompatible_args.hf_checkpoint = "different-tokenizer"
    restored = RolloutDataSource(incompatible_args)

    with pytest.raises(
        ValueError,
        match="Source reservation checkpoint configuration does not match the current data source.",
    ):
        restored.load(rollout_id=19)


def test_load_rejects_changed_dataset_contents_at_same_path(tmp_path: Path) -> None:
    args = _make_args(tmp_path, rollout_shuffle=False)
    source = RolloutDataSource(args)
    source.reserve_samples(1)
    source.save(rollout_id=20)

    Path(args.prompt_data).write_text(
        "\n".join(json.dumps({"prompt": prompt}) for prompt in ("omega", "bravo", "charlie", "delta")),
        encoding="utf-8",
    )
    restored = RolloutDataSource(args)

    with pytest.raises(
        ValueError,
        match="Source reservation checkpoint configuration does not match the current data source.",
    ):
        restored.load(rollout_id=20)


def test_load_rejects_changed_multimodal_descriptor_hidden_by_template(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _make_args(tmp_path, rollout_shuffle=False)
    args.multimodal_keys = {"image": "images"}
    args.apply_chat_template = True
    monkeypatch.setattr(chat_template_utils, "apply_chat_template", lambda *args, **kwargs: "<image>inspect")
    Path(args.prompt_data).write_text(
        json.dumps({"prompt": "<image>inspect", "images": ["first.png"]}),
        encoding="utf-8",
    )
    source = RolloutDataSource(args)
    source.reserve_samples(1)
    source.save(rollout_id=20)

    Path(args.prompt_data).write_text(
        json.dumps({"prompt": "<image>inspect", "images": ["second.png"]}),
        encoding="utf-8",
    )
    restored = RolloutDataSource(args)

    with pytest.raises(
        ValueError,
        match="Source reservation checkpoint configuration does not match the current data source.",
    ):
        restored.load(rollout_id=20)


def test_multimodal_source_fingerprint_is_stable_across_fresh_instances(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _make_args(tmp_path, rollout_shuffle=False)
    args.multimodal_keys = {"video": "videos"}
    Path(args.prompt_data).write_text(
        json.dumps({"prompt": "<video>inspect", "videos": ["unused"]}),
        encoding="utf-8",
    )

    monkeypatch.setattr(data_source_module, "load_processor", lambda *args, **kwargs: object())

    def process_vision_info(prompt: object, processor: object) -> dict[str, list[object]]:
        return {
            "images": [Image.fromarray(numpy.array([[0, 1], [2, 3]], dtype=numpy.uint8))],
            "videos": [
                torch.tensor([[1, 2], [3, 4]]),
                numpy.array([[5, 6], [7, 8]], dtype=numpy.int16),
            ],
        }

    monkeypatch.setattr(processing_utils, "process_vision_info", process_vision_info)

    source = RolloutDataSource(args)
    [first] = source.reserve_samples(1)
    source.save(rollout_id=20)

    restored = RolloutDataSource(args)
    restored.load(rollout_id=20)
    [replayed] = restored.reserve_samples(1)

    assert replayed.reservation_id == first.reservation_id
    assert replayed.samples[0].prompt == first.samples[0].prompt


def test_source_fingerprint_does_not_read_decoded_multimodal_payload_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _make_args(tmp_path, rollout_shuffle=False)
    args.multimodal_keys = {"image": "images"}
    Path(args.prompt_data).write_text(
        json.dumps({"prompt": "<image>inspect", "images": ["unused"]}),
        encoding="utf-8",
    )
    image = Image.fromarray(numpy.array([[0, 1], [2, 3]], dtype=numpy.uint8))

    def reject_payload_read() -> bytes:
        raise AssertionError("source fingerprint must not read decoded multimodal bytes")

    monkeypatch.setattr(image, "tobytes", reject_payload_read)
    monkeypatch.setattr(data_source_module, "load_processor", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        processing_utils,
        "process_vision_info",
        lambda prompt, processor: {"images": [image], "videos": []},
    )

    source = RolloutDataSource(args)

    assert source.reserve_samples(0) == []


def test_load_rejects_unsupported_reservation_schema(tmp_path: Path) -> None:
    args = _make_args(tmp_path, rollout_shuffle=False)
    source = RolloutDataSource(args)
    source.reserve_samples(1)
    source.save(rollout_id=21)

    checkpoint_path = tmp_path / "rollout" / "global_dataset_state_dict_21.pt"
    state = torch.load(checkpoint_path)
    state["source_reservations"]["schema_version"] = 2
    torch.save(state, checkpoint_path)

    restored = RolloutDataSource(args)
    with pytest.raises(ValueError):
        restored.load(rollout_id=21)


@pytest.mark.parametrize("missing_field", ["schema_version", "source_config_hash", "replay"])
def test_load_rejects_incomplete_versioned_reservation_state(tmp_path: Path, missing_field: str) -> None:
    args = _make_args(tmp_path, rollout_shuffle=False)
    source = RolloutDataSource(args)
    source.reserve_samples(1)
    source.save(rollout_id=22)

    checkpoint_path = tmp_path / "rollout" / "global_dataset_state_dict_22.pt"
    state = torch.load(checkpoint_path)
    del state["source_reservations"][missing_field]
    torch.save(state, checkpoint_path)

    restored = RolloutDataSource(args)
    with pytest.raises(ValueError):
        restored.load(rollout_id=22)


@pytest.mark.parametrize("missing_field", ["sample_offset", "epoch_id", "sample_group_index", "sample_index"])
def test_load_rejects_missing_source_cursor_field(tmp_path: Path, missing_field: str) -> None:
    args = _make_args(tmp_path, rollout_shuffle=False)
    source = RolloutDataSource(args)
    source.reserve_samples(1)
    source.save(rollout_id=22)

    checkpoint_path = tmp_path / "rollout" / "global_dataset_state_dict_22.pt"
    state = torch.load(checkpoint_path)
    del state[missing_field]
    torch.save(state, checkpoint_path)

    restored = RolloutDataSource(args)
    with pytest.raises(
        ValueError,
        match=rf"Checkpoint is missing source cursor fields: \['{missing_field}'\]\.",
    ):
        restored.load(rollout_id=22)


def test_load_rejects_replay_at_saved_group_frontier(tmp_path: Path) -> None:
    args = _make_args(tmp_path, rollout_shuffle=False)
    source = RolloutDataSource(args)
    source.reserve_samples(1)
    source.save(rollout_id=23)

    checkpoint_path = tmp_path / "rollout" / "global_dataset_state_dict_23.pt"
    state = torch.load(checkpoint_path)
    [record] = state["source_reservations"]["replay"]
    record["reservation_id"] = "1"
    record["group_index"] = 1
    torch.save(state, checkpoint_path)

    restored = RolloutDataSource(args)
    with pytest.raises(
        ValueError,
        match="Checkpoint replay reservation 1 is not behind group frontier 1.",
    ):
        restored.load(rollout_id=23)


def test_load_rejects_replay_at_saved_sample_frontier(tmp_path: Path) -> None:
    args = _make_args(tmp_path, rollout_shuffle=False)
    source = RolloutDataSource(args)
    source.reserve_samples(1)
    source.save(rollout_id=23)

    checkpoint_path = tmp_path / "rollout" / "global_dataset_state_dict_23.pt"
    state = torch.load(checkpoint_path)
    [record] = state["source_reservations"]["replay"]
    record["sample_indices"] = (2, 3)
    torch.save(state, checkpoint_path)

    restored = RolloutDataSource(args)
    with pytest.raises(
        ValueError,
        match=r"Checkpoint replay reservation 0 has sample indices \(2, 3\) that are not behind sample frontier 2\.",
    ):
        restored.load(rollout_id=23)


def test_load_rejects_replay_without_dataset_identity(tmp_path: Path) -> None:
    args = _make_args(tmp_path, rollout_shuffle=False)
    source = RolloutDataSource(args)
    source.reserve_samples(1)
    source.save(rollout_id=24)

    checkpoint_path = tmp_path / "rollout" / "global_dataset_state_dict_24.pt"
    state = torch.load(checkpoint_path)
    [record] = state["source_reservations"]["replay"]
    record["dataset_index"] = None
    torch.save(state, checkpoint_path)

    restored = RolloutDataSource(args)
    with pytest.raises(
        ValueError,
        match="Checkpoint replay reservation 0 has no dataset identity.",
    ):
        restored.load(rollout_id=24)


def test_load_rejects_replay_with_wrong_dataset_identity(tmp_path: Path) -> None:
    args = _make_args(tmp_path, rollout_shuffle=True)
    source = RolloutDataSource(args)
    source.reserve_samples(5)
    source.save(rollout_id=25)

    checkpoint_path = tmp_path / "rollout" / "global_dataset_state_dict_25.pt"
    state = torch.load(checkpoint_path)
    record = state["source_reservations"]["replay"][4]
    assert record["group_index"] == 4
    assert record["epoch_id"] == 1
    assert record["epoch_offset"] == 0
    assert record["dataset_index"] == 0
    record["dataset_index"] = 1
    torch.save(state, checkpoint_path)

    restored = RolloutDataSource(args)
    with pytest.raises(
        ValueError,
        match=("Checkpoint replay reservation 4 has dataset index 1, " "expected 0 for epoch 1 offset 0."),
    ):
        restored.load(rollout_id=25)


def test_load_rejects_replay_with_wrong_sampling_seeds(tmp_path: Path) -> None:
    args = _make_args(tmp_path, rollout_shuffle=False)
    source = RolloutDataSource(args)
    source.reserve_samples(1)
    source.save(rollout_id=26)

    checkpoint_path = tmp_path / "rollout" / "global_dataset_state_dict_26.pt"
    state = torch.load(checkpoint_path)
    [record] = state["source_reservations"]["replay"]
    record["sampling_seeds"] = (101, 102)
    torch.save(state, checkpoint_path)

    restored = RolloutDataSource(args)
    with pytest.raises(
        ValueError,
        match=r"Checkpoint replay reservation 0 has sampling seeds \(101, 102\), expected \(100, 101\)\.",
    ):
        restored.load(rollout_id=26)


def test_load_rejects_cursor_position_that_disagrees_with_group_frontier(tmp_path: Path) -> None:
    args = _make_args(tmp_path, rollout_shuffle=False)
    source = RolloutDataSource(args)
    source.reserve_samples(1)
    source.save(rollout_id=27)

    checkpoint_path = tmp_path / "rollout" / "global_dataset_state_dict_27.pt"
    state = torch.load(checkpoint_path)
    state["epoch_id"] = 1
    torch.save(state, checkpoint_path)

    restored = RolloutDataSource(args)
    with pytest.raises(
        ValueError,
        match=("Checkpoint group frontier 1 does not match dataset cursor " "at epoch 1 offset 1 for dataset size 4."),
    ):
        restored.load(rollout_id=27)


def test_load_accepts_cold_start_sentinel_without_advancing_source(tmp_path: Path) -> None:
    source = RolloutDataSource(_make_args(tmp_path, rollout_shuffle=False))

    source.load(rollout_id=-1)

    assert source.reserve_samples(1) == [_reservation(0, prompt="alpha", first_sample_index=0)]


@pytest.mark.parametrize("rollout_id", ["1", True, -2])
def test_load_rejects_invalid_rollout_id(tmp_path: Path, rollout_id: object) -> None:
    restored = RolloutDataSource(_make_args(tmp_path, rollout_shuffle=False))

    with pytest.raises(
        ValueError,
        match=rf"rollout_id must be a nonnegative integer, got {rollout_id!r}\.",
    ):
        restored.load(rollout_id=rollout_id)


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


def test_save_rejects_rollout_id_regression_after_pruning_acknowledgements(tmp_path: Path) -> None:
    args = _make_args(tmp_path, rollout_shuffle=False)
    source = RolloutDataSource(args)
    [first] = source.reserve_samples(1)
    source.acknowledge_reservations([first], rollout_id=6)
    source.save(rollout_id=6)

    with pytest.raises(
        ValueError,
        match="Source checkpoint rollout_id must not move backward from 6 to 5.",
    ):
        source.save(rollout_id=5)

    restored = RolloutDataSource(args)
    restored.load(rollout_id=6)
    assert restored.reserve_samples(1) == [_reservation(1, prompt="bravo", first_sample_index=2)]


def test_acknowledge_rejects_rollout_at_published_checkpoint_frontier(tmp_path: Path) -> None:
    args = _make_args(tmp_path, rollout_shuffle=False)
    source = RolloutDataSource(args)
    source.save(rollout_id=6)
    [first] = source.reserve_samples(1)

    with pytest.raises(
        ValueError,
        match="Reservation rollout_id 6 must be newer than published checkpoint 6.",
    ):
        source.acknowledge_reservations([first], rollout_id=6)

    source.requeue_reservations([first])
    source.save(rollout_id=7)
    restored = RolloutDataSource(args)
    restored.load(rollout_id=7)
    assert restored.reserve_samples(1) == [first]


def test_save_is_linearized_with_reservation_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _make_args(tmp_path, rollout_shuffle=False)
    source = RolloutDataSource(args)
    [first] = source.reserve_samples(1)

    save_started = threading.Event()
    allow_save = threading.Event()
    reserve_started = threading.Event()
    reserve_finished = threading.Event()
    original_torch_save = torch.save

    def blocking_save(obj: object, path: str) -> None:
        save_started.set()
        assert allow_save.wait(timeout=5)
        original_torch_save(obj, path)

    def reserve_after_save_starts() -> SourceReservation:
        reserve_started.set()
        [reservation] = source.reserve_samples(1)
        reserve_finished.set()
        return reservation

    monkeypatch.setattr(data_source_module.torch, "save", blocking_save)
    with ThreadPoolExecutor(max_workers=2) as executor:
        save_future = executor.submit(source.save, 23)
        assert save_started.wait(timeout=5)
        reserve_future = executor.submit(reserve_after_save_starts)
        assert reserve_started.wait(timeout=5)
        assert not reserve_finished.wait(timeout=0.05)
        allow_save.set()
        save_future.result(timeout=5)
        second = reserve_future.result(timeout=5)

    assert second == _reservation(1, prompt="bravo", first_sample_index=2)

    restored = RolloutDataSource(args)
    restored.load(rollout_id=23)
    [replayed] = restored.reserve_samples(1)
    assert replayed == first
    restored.acknowledge_reservations([replayed], rollout_id=24)
    assert restored.reserve_samples(1) == [second]
