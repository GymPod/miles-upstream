import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_INITIAL_DUMP_NAME = "initial"


class EngineChecksumDumper:
    """Dump per-engine weight checksums after each update_weights (CI verification only).

    Layout: ``<dump_dir>/rollout_<rollout_id>/engine_<i>.json`` (``<dump_dir>/initial/...``
    for the weight sync before the first rollout). Each file is the raw response of the
    sglang ``weights_checker`` "checksum" action for one engine:
    ``{"success": bool, "message": str, "ranks": [{"checksums": {name: hash}, "parallelism_info": {...}}]}``.

    Engines are flattened across servers and server groups in a stable order, so
    ``engine_<i>`` matches between two runs with the same engine topology. If engine-side
    faults are ever injected (engine set differs between runs), alignment must switch to
    ``parallelism_info`` instead of the flat index.
    """

    def __init__(self, *, dump_dir: Path, rollout_manager: object) -> None:
        self._dump_dir = dump_dir
        self._rollout_manager = rollout_manager

    @staticmethod
    def from_args(args: object, *, rollout_manager: object | None) -> "EngineChecksumDumper | None":
        if args.ci_dump_engine_weight_checksums is None or rollout_manager is None:
            return None
        return EngineChecksumDumper(
            dump_dir=Path(args.ci_dump_engine_weight_checksums),
            rollout_manager=rollout_manager,
        )

    async def dump(self, *, rollout_id: int | None) -> None:
        """Checksum all engines and write one JSON per engine.

        Must be called only after update_weights() fully completed (all ranks finished
        pushing), so the checksums reflect the post-sync engine weights.
        """
        # Nesting: servers -> server groups -> engines. Multi-node engines return None
        # from non-zero node ranks (no HTTP server there); drop those entries.
        nested = await self._rollout_manager.check_weights.remote(action="checksum")
        engine_responses: list[dict] = [
            response
            for per_server in nested
            for per_group in per_server
            for response in per_group
            if response is not None
        ]
        assert engine_responses, "check_weights('checksum') returned no engine responses"

        rollout_dir = self._dump_dir / (f"rollout_{rollout_id}" if rollout_id is not None else _INITIAL_DUMP_NAME)
        rollout_dir.mkdir(parents=True, exist_ok=True)
        for engine_index, response in enumerate(engine_responses):
            path = rollout_dir / f"engine_{engine_index}.json"
            path.write_text(json.dumps(response, indent=2, sort_keys=True))
        logger.info(
            "Dumped engine weight checksums for %d engine(s) to %s",
            len(engine_responses),
            rollout_dir,
        )
