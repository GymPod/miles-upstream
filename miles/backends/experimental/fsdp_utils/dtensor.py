"""DTensor materialization for FSDP2 weight export.

FSDP2 holds each parameter as a sharded ``DTensor``; the rollout engine needs the full
(unsharded) tensor. The sync and async weight-sync paths gather a shard the same way:
move to CUDA first, then all-gather to ``Replicate`` via ``redistribute``. This module is
the single home for that idiom and the two collective-backend caveats it has to respect:

  * ``full_tensor()`` on a *CPU* DTensor picks the wrong collective backend, so the move to
    CUDA must happen before the gather.
  * ``redistribute`` on a 1-rank mesh trips ``assert compute_mesh is not None``, so
    ``world_size == 1`` falls back to ``full_tensor()``.
"""

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Replicate


def gather_full_param(param: torch.Tensor, *, async_op: bool = False) -> torch.Tensor:
    """Materialize a (possibly FSDP2-sharded) param to a full local tensor on CUDA.

    Non-DTensor inputs are returned moved to CUDA, unchanged. A sharded DTensor is
    all-gathered to ``Replicate`` across every mesh dim and returned as a plain local
    tensor.

    With ``async_op=True`` the all-gather is issued asynchronously and the returned tensor
    carries a ``.wait()`` the caller must drain before use; dtype casts and any other
    consumption must happen post-wait.
    """
    full = param.cuda()
    if not isinstance(full, DTensor):
        return full
    if dist.get_world_size() == 1:
        # redistribute on a 1-rank mesh trips `assert compute_mesh is not None`
        return full.full_tensor()
    return full.redistribute(
        placements=[Replicate()] * full.device_mesh.ndim,
        async_op=async_op,
    ).to_local()
