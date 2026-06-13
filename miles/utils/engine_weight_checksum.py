"""Flatten the nested ``RolloutManager.check_weights("checksum")`` response into
per-engine merged checksum dicts, ready to log as ``EngineWeightChecksumEvent``.

The response is nested servers -> server_groups -> engines. Each engine returns
the raw HTTP body ``{"success": bool, "message": str, "ranks": [ChecksumInfo...]}``;
a multi-node engine's non-zero node ranks return ``None`` (the ``_make_request``
early-return), which we drop before assigning a stable ``engine_index``.
"""

from typing import Any

EngineChecksums = dict[str, str]


def flatten_engine_checksums(check_weights_result: Any) -> list[EngineChecksums]:
    """Flatten servers->groups->engines, drop None engines, merge each engine's ranks.

    Returns one merged ``{name: hash}`` dict per surviving engine, in flattened
    order. Fails loud if every engine was filtered out (a None-only result means
    the checksum action silently did nothing).
    """
    engine_bodies = _flatten_to_engine_bodies(check_weights_result)
    surviving = [body for body in engine_bodies if body is not None]
    assert surviving, (
        f"check_weights('checksum') returned no non-None engine bodies "
        f"(got {len(engine_bodies)} entries, all None): {check_weights_result!r}"
    )
    return [_merge_engine_ranks(body) for body in surviving]


def _flatten_to_engine_bodies(check_weights_result: Any) -> list[Any]:
    """servers -> server_groups -> engines, yielding each engine's raw body (or None)."""
    engine_bodies: list[Any] = []
    for server in check_weights_result:
        for server_group in server:
            for engine_body in server_group:
                engine_bodies.append(engine_body)
    return engine_bodies


def _merge_engine_ranks(engine_body: dict[str, Any]) -> EngineChecksums:
    """Merge one engine's per-rank ChecksumInfo dicts into a single flat dict.

    Ranks arrive in non-deterministic (zmq) order under TP>1, so sort by
    ``parallelism_info.rank`` and prefix each tensor name with ``rank{r}/`` to
    keep distinct shards' identically-named tensors from clobbering one another.
    """
    assert engine_body.get("success", False), f"check_weights engine reported failure: {engine_body!r}"
    ranks: list[dict[str, Any]] = engine_body.get("ranks", []) or []
    assert ranks, f"check_weights engine body has no ranks: {engine_body!r}"

    ranks_sorted = sorted(ranks, key=lambda r: r["parallelism_info"]["rank"])

    merged: EngineChecksums = {}
    for rank_info in ranks_sorted:
        rank = rank_info["parallelism_info"]["rank"]
        for name, value in rank_info["checksums"].items():
            merged[f"rank{rank}/{name}"] = value
    return merged
