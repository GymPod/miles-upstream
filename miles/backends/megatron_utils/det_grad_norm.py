"""Partition-independent deterministic grad norm (--debug-deterministic-collective).

Megatron's get_grad_norm sums squared norms of the distributed optimizer's local
shard fragments, so its bracketing depends on how many ranks shard the grad buffer
(dp_cp=8 in the plain-DP baseline vs 2 inside an FT cell) and on bucket padding.
Instead: gather the reduced grad buffer per bucket (pure data movement), square-sum
each parameter's full-shape gradient (a fixed-shape kernel, identical regardless of
sharding), and fold the per-parameter sums in the fixed traversal order. The
model-parallel combine runs on det_nccl groups, i.e. the deterministic fold.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.tensor_parallel import param_is_not_tensor_parallel_duplicate
from megatron.core.transformer.module import param_is_not_shared

if TYPE_CHECKING:
    from megatron.core.distributed import DistributedDataParallel as DDP


def deterministic_grad_norm(model: Sequence["DDP"]) -> float:
    """Sharding-independent L2 norm over the same parameters Megatron's norm counts.

    Call after grad sync and before optimizer.step() (clipping may rescale grads).
    """
    total_sq: torch.Tensor | None = None

    for model_chunk in model:
        for bucket_group in model_chunk.bucket_groups + model_chunk.expert_parallel_bucket_groups:
            group = bucket_group.intra_distributed_optimizer_instance_group
            world_size = dist.get_world_size(group)
            rank = dist.get_rank(group)
            for bucket in bucket_group.buckets:
                shard_numel = bucket.grad_data.numel() // world_size
                gathered = torch.empty_like(bucket.grad_data)
                dist.all_gather_into_tensor(
                    gathered, bucket.grad_data[rank * shard_numel : (rank + 1) * shard_numel], group=group
                )
                # TEMP DEBUG PROBE
                slot_sqs = [
                    float(gathered[i * shard_numel : (i + 1) * shard_numel].float().pow(2).sum().item())
                    for i in range(world_size)
                ]
                own_sq = float(
                    bucket.grad_data[rank * shard_numel : (rank + 1) * shard_numel].float().pow(2).sum().item()
                )
                import logging as _logging

                _logging.getLogger(__name__).info(
                    "DETNORM-PROBE bucket=%d numel=%d world=%d rank=%d own_sq=%.6e slot_sqs=%s group_type=%s",
                    bucket.bucket_id,
                    bucket.grad_data.numel(),
                    world_size,
                    rank,
                    own_sq,
                    [f"{x:.3e}" for x in slot_sqs],
                    type(group).__name__,
                )
                for param in bucket.params_list:
                    if getattr(param, "_is_witness_param", False):
                        # FT witness params carry verification data, not gradients;
                        # Megatron's grad norm excludes them the same way.
                        continue
                    if not (param_is_not_shared(param) and param_is_not_tensor_parallel_duplicate(param)):
                        continue
                    start, end = bucket.param_to_index[param]
                    param_sq = gathered[start:end].float().pow(2).sum()
                    total_sq = param_sq if total_sq is None else total_sq + param_sq

    if total_sq is None:
        total_sq = torch.zeros((), dtype=torch.float32, device=torch.cuda.current_device())

    # dp/cp is covered by the gathers; combine the model-parallel planes.
    dist.all_reduce(total_sq, group=parallel_state.get_model_parallel_group())
    return total_sq.item() ** 0.5
