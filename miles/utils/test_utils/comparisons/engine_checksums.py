from pathlib import Path

from miles.utils.event_logger.logger import read_events
from miles.utils.event_logger.models import EngineWeightChecksumEvent


def compare_engine_checksums(baseline_dir: str, target_dir: str) -> None:
    """Assert baseline and target pushed bitwise-identical weights to every rollout engine.

    Aligns EngineWeightChecksumEvents by rollout_id, then compares the per-engine checksum
    lists element-wise. Hashes are bitwise by construction, so any difference is a real
    weight-sync divergence. Fail-closed: the rollout_id sets, the per-rollout engine counts,
    and the per-engine tensor-name sets must be equal and non-empty on both sides.
    """
    baseline = _index_engine_checksums(_read_engine_checksum_events(Path(baseline_dir)))
    target = _index_engine_checksums(_read_engine_checksum_events(Path(target_dir)))

    assert baseline, f"No EngineWeightChecksumEvents found in baseline dir: {baseline_dir}"
    assert target, f"No EngineWeightChecksumEvents found in target dir: {target_dir}"

    assert baseline.keys() == target.keys(), (
        f"Engine checksum rollout_id sets differ: "
        f"baseline={sorted(baseline.keys())} vs target={sorted(target.keys())}"
    )

    issues: list[str] = []
    for rollout_id in sorted(baseline.keys()):
        b_engines = baseline[rollout_id]
        t_engines = target[rollout_id]
        if len(b_engines) != len(t_engines):
            issues.append(
                f"rollout {rollout_id}: engine count differs (baseline={len(b_engines)} vs target={len(t_engines)})"
            )
            continue
        for engine_index, (b_checksums, t_checksums) in enumerate(zip(b_engines, t_engines, strict=True)):
            if b_checksums.keys() != t_checksums.keys():
                issues.append(
                    f"rollout {rollout_id} engine {engine_index}: tensor-name sets differ "
                    f"(baseline-only={sorted(set(b_checksums) - set(t_checksums))}, "
                    f"target-only={sorted(set(t_checksums) - set(b_checksums))})"
                )
                continue
            for name in sorted(b_checksums.keys()):
                if b_checksums[name] != t_checksums[name]:
                    issues.append(
                        f"rollout {rollout_id} engine {engine_index} tensor {name}: "
                        f"baseline={b_checksums[name]} != target={t_checksums[name]}"
                    )

    assert not issues, (
        "Engine weight checksum baseline-vs-target comparison found "
        + f"{len(issues)} issue(s):\n"
        + "\n".join(f"  - {i}" for i in issues)
    )
    print(f"Engine weight checksum comparison passed: {len(baseline)} rollout(s) compared")


def _index_engine_checksums(events: list[EngineWeightChecksumEvent]) -> dict[int, list[dict[str, str]]]:
    indexed: dict[int, list[dict[str, str]]] = {}
    for event in events:
        assert event.rollout_id not in indexed, f"Duplicate EngineWeightChecksumEvent for rollout {event.rollout_id}"
        indexed[event.rollout_id] = event.engine_checksums
    return indexed


def _read_engine_checksum_events(dump_dir: Path) -> list[EngineWeightChecksumEvent]:
    """Read all EngineWeightChecksumEvents from the events directory."""
    events_dir: Path = dump_dir / "events"
    if not events_dir.exists():
        return []
    all_events = read_events(events_dir)
    return [e for e in all_events if isinstance(e, EngineWeightChecksumEvent)]
