"""Tests for test_utils.comparisons.compare_engine_checksums."""

from pathlib import Path
from typing import Any

import pytest

from miles.utils.event_logger.logger import EventLogger
from miles.utils.event_logger.models import EngineWeightChecksumEvent
from miles.utils.process_identity import MainProcessIdentity
from miles.utils.test_utils.comparisons import compare_engine_checksums


def _write_engine_events(side_dir: Path, partials: list[dict[str, Any]]) -> None:
    events_dir = side_dir / "events"
    event_logger = EventLogger(log_dir=events_dir, source=MainProcessIdentity())
    for partial in partials:
        event_logger.log(EngineWeightChecksumEvent, partial, print_log=False)
    event_logger.close()


def _partial(*, rollout_id: int, engine_index: int, checksums: dict[str, str]) -> dict[str, Any]:
    return dict(rollout_id=rollout_id, engine_index=engine_index, checksums=checksums)


class TestCompareEngineChecksums:
    def test_identical_passes(self, tmp_path: Path) -> None:
        """Baseline and target with identical per-engine checksums pass."""
        partials = [
            _partial(rollout_id=1, engine_index=0, checksums={"rank0/w": "aaa"}),
            _partial(rollout_id=1, engine_index=1, checksums={"rank0/w": "bbb"}),
        ]
        _write_engine_events(tmp_path / "baseline", partials)
        _write_engine_events(tmp_path / "target", partials)

        compare_engine_checksums(str(tmp_path / "baseline"), str(tmp_path / "target"))

    def test_single_tensor_difference_fails(self, tmp_path: Path) -> None:
        """One differing tensor on one engine fails and names rollout/engine/tensor."""
        _write_engine_events(
            tmp_path / "baseline", [_partial(rollout_id=1, engine_index=0, checksums={"rank0/w": "aaa"})]
        )
        _write_engine_events(
            tmp_path / "target", [_partial(rollout_id=1, engine_index=0, checksums={"rank0/w": "zzz"})]
        )

        with pytest.raises(AssertionError, match=r"rollout 1 engine 0 tensor rank0/w"):
            compare_engine_checksums(str(tmp_path / "baseline"), str(tmp_path / "target"))

    def test_missing_engine_fails(self, tmp_path: Path) -> None:
        """A (rollout, engine) present in baseline but absent in target fails closed."""
        _write_engine_events(
            tmp_path / "baseline",
            [
                _partial(rollout_id=1, engine_index=0, checksums={"rank0/w": "aaa"}),
                _partial(rollout_id=1, engine_index=1, checksums={"rank0/w": "bbb"}),
            ],
        )
        _write_engine_events(
            tmp_path / "target", [_partial(rollout_id=1, engine_index=0, checksums={"rank0/w": "aaa"})]
        )

        with pytest.raises(AssertionError, match="engine_index. sets differ"):
            compare_engine_checksums(str(tmp_path / "baseline"), str(tmp_path / "target"))

    def test_missing_rollout_fails(self, tmp_path: Path) -> None:
        """A rollout present only on one side fails closed."""
        _write_engine_events(
            tmp_path / "baseline",
            [
                _partial(rollout_id=1, engine_index=0, checksums={"rank0/w": "aaa"}),
                _partial(rollout_id=2, engine_index=0, checksums={"rank0/w": "ccc"}),
            ],
        )
        _write_engine_events(
            tmp_path / "target", [_partial(rollout_id=1, engine_index=0, checksums={"rank0/w": "aaa"})]
        )

        with pytest.raises(AssertionError, match="sets differ"):
            compare_engine_checksums(str(tmp_path / "baseline"), str(tmp_path / "target"))

    def test_tensor_name_set_mismatch_fails(self, tmp_path: Path) -> None:
        """An engine whose tensor-name set differs fails before per-value comparison."""
        _write_engine_events(
            tmp_path / "baseline",
            [_partial(rollout_id=1, engine_index=0, checksums={"rank0/w": "aaa", "rank0/b": "bbb"})],
        )
        _write_engine_events(
            tmp_path / "target", [_partial(rollout_id=1, engine_index=0, checksums={"rank0/w": "aaa"})]
        )

        with pytest.raises(AssertionError, match="tensor-name sets differ"):
            compare_engine_checksums(str(tmp_path / "baseline"), str(tmp_path / "target"))

    def test_empty_baseline_fails(self, tmp_path: Path) -> None:
        """No baseline events fails closed rather than vacuously passing."""
        _write_engine_events(tmp_path / "baseline", [])
        _write_engine_events(
            tmp_path / "target", [_partial(rollout_id=1, engine_index=0, checksums={"rank0/w": "aaa"})]
        )

        with pytest.raises(AssertionError, match="No EngineWeightChecksumEvents found in baseline"):
            compare_engine_checksums(str(tmp_path / "baseline"), str(tmp_path / "target"))
