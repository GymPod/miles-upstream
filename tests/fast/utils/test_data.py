import datetime
import json
from pathlib import Path

import numpy
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from miles.utils import chat_template_utils, processing_utils
from miles.utils.data import Dataset


def _write_multimodal_row(path: Path, *, image: str) -> None:
    path.write_text(
        json.dumps(
            {
                "prompt": "<image>inspect",
                "images": [image],
            }
        ),
        encoding="utf-8",
    )


def test_dataset_fingerprint_includes_pre_template_multimodal_descriptors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "prompts.jsonl"
    monkeypatch.setattr(chat_template_utils, "apply_chat_template", lambda *args, **kwargs: "<image>inspect")
    _write_multimodal_row(path, image="first.png")
    first = Dataset(
        str(path),
        tokenizer=object(),
        processor=None,
        max_length=None,
        prompt_key="prompt",
        multimodal_keys={"image": "images"},
        apply_chat_template=True,
    )

    _write_multimodal_row(path, image="second.png")
    second = Dataset(
        str(path),
        tokenizer=object(),
        processor=None,
        max_length=None,
        prompt_key="prompt",
        multimodal_keys={"image": "images"},
        apply_chat_template=True,
    )

    assert first.origin_samples == second.origin_samples
    assert first.fingerprint != second.fingerprint


def test_dataset_fingerprint_ignores_json_object_order_and_formatting(tmp_path: Path) -> None:
    path = tmp_path / "prompts.jsonl"
    path.write_text('{"prompt":"inspect","metadata":{"a":1,"b":2}}\n', encoding="utf-8")
    first = Dataset(
        str(path),
        tokenizer=object(),
        processor=None,
        max_length=None,
        prompt_key="prompt",
    )

    path.write_text(
        '{ "metadata": { "b": 2, "a": 1 }, "prompt": "inspect" }\n',
        encoding="utf-8",
    )
    second = Dataset(
        str(path),
        tokenizer=object(),
        processor=None,
        max_length=None,
        prompt_key="prompt",
    )

    assert first.origin_samples == second.origin_samples
    assert first.fingerprint == second.fingerprint


def test_dataset_fingerprint_does_not_read_decoded_multimodal_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "prompts.jsonl"
    _write_multimodal_row(path, image="image.png")
    image = Image.fromarray(numpy.array([[0, 1], [2, 3]], dtype=numpy.uint8))

    def reject_payload_read() -> bytes:
        raise AssertionError("dataset fingerprint must not read decoded multimodal bytes")

    monkeypatch.setattr(image, "tobytes", reject_payload_read)
    monkeypatch.setattr(
        processing_utils,
        "process_vision_info",
        lambda prompt, processor: {"images": [image], "videos": []},
    )

    dataset = Dataset(
        str(path),
        tokenizer=object(),
        processor=object(),
        max_length=None,
        prompt_key="prompt",
        multimodal_keys={"image": "images"},
    )

    assert dataset.fingerprint.startswith("miles-dataset-v1:")


def test_dataset_fingerprint_supports_parquet_duration_values(tmp_path: Path) -> None:
    path = tmp_path / "prompts.parquet"
    table = pa.Table.from_pylist(
        [
            {
                "prompt": "inspect",
                "metadata": {"delay": datetime.timedelta(seconds=5)},
            }
        ]
    )
    pq.write_table(table, path)

    first = Dataset(
        str(path),
        tokenizer=object(),
        processor=None,
        max_length=None,
        prompt_key="prompt",
    )
    second = Dataset(
        str(path),
        tokenizer=object(),
        processor=None,
        max_length=None,
        prompt_key="prompt",
    )

    assert first.fingerprint == second.fingerprint


def test_dataset_fingerprint_preserves_parquet_duration_nanoseconds(tmp_path: Path) -> None:
    def write_duration(path: Path, nanoseconds: int) -> None:
        metadata = pa.StructArray.from_arrays(
            [pa.array([nanoseconds], type=pa.duration("ns"))],
            names=["delay"],
        )
        table = pa.Table.from_arrays(
            [pa.array(["inspect"]), metadata],
            names=["prompt", "metadata"],
        )
        pq.write_table(table, path)

    first_path = tmp_path / "first.parquet"
    second_path = tmp_path / "second.parquet"
    write_duration(first_path, 1)
    write_duration(second_path, 2)

    first = Dataset(
        str(first_path),
        tokenizer=object(),
        processor=None,
        max_length=None,
        prompt_key="prompt",
    )
    second = Dataset(
        str(second_path),
        tokenizer=object(),
        processor=None,
        max_length=None,
        prompt_key="prompt",
    )

    assert first.fingerprint != second.fingerprint
