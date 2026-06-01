from __future__ import annotations

import asyncio
import dataclasses
import enum
import logging
import shutil
from argparse import Namespace
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from sglang.srt.debug_utils.dumper import DumperConfig, _get_rank, dumper

from miles.backends.training_utils.parallel import get_parallel_state

logger = logging.getLogger(__name__)


class DumperPhase(enum.Enum):
    INFERENCE = "inference"
    FWD_ONLY = "fwd_only"
    FWD_BWD = "fwd_bwd"


# ------------------------------- SGLang -------------------------------------


def get_sglang_env(args: Namespace) -> dict[str, str]:
    if not _is_phase_enabled(args, DumperPhase.INFERENCE):
        return {}

    env: dict[str, str] = {"DUMPER_SERVER_PORT": "reuse"}
    overrides = _get_phase_override_configs(args, DumperPhase.INFERENCE)

    # SGLang registers non-intrusive hooks while loading the model. Configs that
    # affect hook registration must be present in the actor environment; the
    # later HTTP configure call only controls active dumping/output location.
    if non_intrusive_mode := overrides.get("non_intrusive_mode"):
        env["DUMPER_NON_INTRUSIVE_MODE"] = str(non_intrusive_mode)

    if source_patcher_config := args.dumper_source_patcher_config_inference:
        env["DUMPER_SOURCE_PATCHER_CONFIG"] = source_patcher_config
    elif source_patcher_config := overrides.get("source_patcher_config"):
        env["DUMPER_SOURCE_PATCHER_CONFIG"] = str(source_patcher_config)

    return env


async def configure_sglang(args: Namespace) -> None:
    if not _is_phase_enabled(args, DumperPhase.INFERENCE):
        return

    from miles.rollout.inference_rollout.inference_rollout_train import get_worker_urls
    from miles.utils.http_utils import post

    worker_urls = await get_worker_urls(args)
    overrides = _get_phase_override_configs(args, DumperPhase.INFERENCE)

    engines_dir: Path = _get_dir(args) / "engines"
    _cleanup_dump_dir(engines_dir)

    coros = []
    for i, url in enumerate(worker_urls):
        body = {
            "enable": True,
            "dir": str(_get_dir(args)),
            "exp_name": f"engines/engine_{i}",
            **overrides,
        }
        coros.append(post(f"{url}/dumper/configure", body))

    await asyncio.gather(*coros)
    logger.info("Configured dumper on %d SGLang engines", len(worker_urls))


# ------------------------------- Megatron -------------------------------------


class DumperMegatronUtil:
    def __init__(self, args: Namespace, model: Sequence[torch.nn.Module], phase: DumperPhase) -> None:
        self.phase = phase
        self.overrides = _get_phase_override_configs(args, phase)
        self.enabled = self._configure(args, phase, self.overrides)
        if self.enabled:
            dumper.register_non_intrusive_dumper(self._extract_model(model))

    def wrap_forward_step(self, forward_step_func: Callable) -> Callable:
        if not self.enabled:
            return forward_step_func

        return _wrap_forward_step_with_stepping(forward_step_func)

    def finalize(self, model: Sequence[torch.nn.Module]) -> None:
        if not self.enabled:
            return

        extracted_model = self._extract_model(model)
        if self.phase is DumperPhase.FWD_BWD and self.overrides.get("enable_model_grad"):
            _log_model_grad_coverage(extracted_model)
            # With the distributed optimizer, grads are reduce-scattered over the
            # dp_cp group, so each rank's main_grad holds the reduced gradient only
            # on its own shard. Re-gather the shards so the dumped gradient is the
            # full reduced gradient (the same global gradient the optimizer steps
            # with) — making dumps comparable across different DP topologies.
            _reconstruct_full_grads_inplace(extracted_model)
        dumper.dump_model(extracted_model)
        dumper.step()
        dumper.configure(enable=False)

    @staticmethod
    def _extract_model(model: Sequence[torch.nn.Module]) -> torch.nn.Module:
        assert (
            len(model) == 1
        ), f"Dumper does not yet support virtual pipeline parallelism (got {len(model)} model chunks)"
        return model[0]

    @staticmethod
    def _configure(args: Namespace, phase: DumperPhase, overrides: dict[str, Any] | None = None) -> bool:
        if overrides is None:
            overrides = _get_phase_override_configs(args, phase)
        if not overrides.get("enable"):
            return False

        merged = {
            "dir": str(_get_dir(args)),
            "exp_name": phase.value,
            "enable_output_console": False,
            **overrides,
        }

        # Only write dump files on effective DP rank 0 (covers both intra-DP
        # and indep-DP). Other DP ranks still participate in dumper collectives
        # (barrier, broadcast, allgather) but don't produce output files.
        # TODO: optimize — non-DP-rank-0 ranks currently run full dumper logic
        # (forward hooks, model iteration) without producing output.
        if get_parallel_state().effective_dp.rank != 0:
            merged["enable_output_file"] = False
            merged["enable_output_console"] = False

        full_config = DumperConfig(**merged)
        dumper.reset()
        _cleanup_dump_dir(Path(merged["dir"]) / merged["exp_name"])
        dumper.configure(**dataclasses.asdict(full_config))
        return True


def _reconstruct_full_grads_inplace(model_chunk: torch.nn.Module) -> None:
    """All-gather the distributed-optimizer grad shards back into the full buffer.

    With ``use_distributed_optimizer=True`` Megatron reduce-scatters gradients
    over the dp_cp group: each rank's ``grad_data`` (and the ``main_grad`` views
    into it) holds the fully-reduced gradient only on its own 1/dp_cp shard, with
    stale local partials elsewhere. The raw dump is therefore not comparable
    across DP topologies (a non-FT baseline shards over dp4×cp2=8 while an
    indep-DP target shards over a cell's 1×cp2=2 then sums across cells).

    All-gathering the shards (mirroring Megatron's own ``start_param_sync`` but
    for grads) materializes the complete reduced gradient on every rank — exactly
    the global gradient the optimizer steps with, which is identical across the
    two topologies (the post-step weights are bit-identical). The comparator then
    checks the real, full gradient at full tolerance.

    In-place is safe: ``all_gather_into_tensor`` writes each rank's own shard back
    unchanged and only overwrites the (otherwise unused) non-owned regions; the
    optimizer reads only its owned shard, and ``zero_grad_buffer`` wipes the
    buffer after the step. All dp_cp ranks run the dumper, so the collective is
    symmetric.
    """
    bucket_groups = list(getattr(model_chunk, "bucket_groups", [])) + list(
        getattr(model_chunk, "expert_parallel_bucket_groups", [])
    )
    for bucket_group in bucket_groups:
        if not bucket_group.ddp_config.use_distributed_optimizer:
            continue
        group = bucket_group.intra_distributed_optimizer_instance_group
        instance_size = bucket_group.intra_distributed_optimizer_instance_size
        instance_rank = bucket_group.intra_distributed_optimizer_instance_rank
        if instance_size <= 1:
            continue
        for bucket in bucket_group.buckets:
            local_shard = _shard_grad_buffer(bucket.grad_data, instance_size)[instance_rank]
            dist.all_gather_into_tensor(bucket.grad_data, local_shard, group=group)


def _shard_grad_buffer(buffer: torch.Tensor, world_size: int) -> list[torch.Tensor]:
    shard_size = buffer.numel() // world_size
    return [buffer[r * shard_size : (r + 1) * shard_size] for r in range(world_size)]


def _log_model_grad_coverage(model: torch.nn.Module) -> None:
    missing: list[str] = []
    with_grad = 0
    total = 0

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        total += 1
        grad = param.grad if param.grad is not None else getattr(param, "main_grad", None)
        if grad is None:
            missing.append(name)
        else:
            with_grad += 1

    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else _get_rank()
    logger.info(
        "Dumper fwd_bwd model grad coverage rank=%s with_grad=%d total=%d missing=%d missing_names=%s",
        rank,
        with_grad,
        total,
        len(missing),
        missing[:20],
    )


def _wrap_forward_step_with_stepping(forward_step_func: Callable) -> Callable:
    is_first_call = True

    def _wrapped(*args: Any, **kwargs: Any) -> Any:
        nonlocal is_first_call
        if not is_first_call:
            dumper.step()
        is_first_call = False
        return forward_step_func(*args, **kwargs)

    return _wrapped


# ------------------------------- Common -------------------------------------


def _cleanup_dump_dir(dump_dir: Path) -> None:
    # Only cell 0's rank 0 deletes — avoids race when multiple cells' rank 0
    # all see _get_rank()==0 and try to rmtree the same directory.
    # Best-effort: stale handles from a peer that crashed (NFS .nfsXXXX stubs)
    # can make rmtree fail with "Directory not empty"; we don't want that to
    # propagate up and mark the (healthy) cell as errored.
    indep_dp = get_parallel_state().indep_dp
    if (_get_rank() == 0) and (indep_dp.rank == 0) and dump_dir.is_dir():
        try:
            shutil.rmtree(dump_dir)
        except OSError:
            logger.warning("dump dir cleanup failed; continuing", exc_info=True)
    if dist.is_initialized():
        dist.barrier()
    if indep_dp.group is not None:
        indep_dp.group.barrier()


def _get_phase_override_configs(args: Namespace, phase: DumperPhase) -> dict[str, Any]:
    raw = getattr(args, f"dumper_{phase.value}")
    return {"enable": args.dumper_enable, **DumperConfig._kv_pairs_to_dict(raw)}


def _is_phase_enabled(args: Namespace, phase: DumperPhase) -> bool:
    return _get_phase_override_configs(args, phase).get("enable", False)


def _get_dir(args: Namespace) -> Path:
    return Path(args.dumper_dir)
