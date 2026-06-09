import os
from argparse import Namespace
from pathlib import Path

import torch
import torch.distributed as dist

from miles.utils.types import RolloutBatch

_POLICY_LOSS_DUMP_COUNTER = 0


def maybe_dump_policy_loss_debug(
    *,
    args: Namespace,
    batch: RolloutBatch,
    train_log_probs: list[torch.Tensor],
    old_log_probs: list[torch.Tensor],
    rollout_log_probs: list[torch.Tensor] | None,
    advantages: list[torch.Tensor],
    local_loss_masks: list[torch.Tensor],
    ppo_kl: torch.Tensor,
    pg_loss: torch.Tensor,
) -> None:
    dump_dir = getattr(args, "dump_details", None)
    if dump_dir is None and os.environ.get("MILES_DUMP_PL"):
        # Per-phase fallback: save_debug_event_data is "{phase_dump_dir}/events", so its
        # parent puts the policy-loss dump under each phase's own dir (baseline vs target).
        ev = getattr(args, "save_debug_event_data", None)
        dump_dir = str(Path(ev).parent) if ev else os.environ["MILES_DUMP_PL"]
    if dump_dir is None:
        return

    global _POLICY_LOSS_DUMP_COUNTER
    counter = _POLICY_LOSS_DUMP_COUNTER
    _POLICY_LOSS_DUMP_COUNTER += 1

    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    # Under FT each cell uses cell-local ranks (0..3), so both cells would write the same
    # rank_X file and overwrite each other. Tag with the indep_dp (cell) rank to keep them
    # distinct, so a diagnostic can compare both cells' tensors for the same step.
    from miles.backends.training_utils.parallel import get_parallel_state

    cell = get_parallel_state().indep_dp.rank
    path = Path(dump_dir) / "policy_loss_debug" / f"cell_{cell}_rank_{rank}_call_{counter}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)

    def to_cpu(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.detach().float().cpu()

    samples = []
    for index, train_lp in enumerate(train_log_probs):
        sample = {
            "index": index,
            "total_length": batch["total_lengths"][index],
            "response_length": batch["response_lengths"][index],
            "train_log_probs": to_cpu(train_lp),
            "old_log_probs": to_cpu(old_log_probs[index]),
            "advantages": to_cpu(advantages[index]),
            "local_loss_mask": to_cpu(local_loss_masks[index]),
        }
        if rollout_log_probs is not None:
            sample["rollout_log_probs"] = to_cpu(rollout_log_probs[index])
            if train_lp.shape == rollout_log_probs[index].shape:
                sample["train_rollout_abs_diff"] = to_cpu((train_lp - rollout_log_probs[index]).abs())
        samples.append(sample)

    torch.save(
        {
            "rank": rank,
            "call": counter,
            "samples": samples,
            "ppo_kl": to_cpu(ppo_kl),
            "pg_loss": to_cpu(pg_loss),
            "finite": {
                "ppo_kl": torch.isfinite(ppo_kl).all().item(),
                "pg_loss": torch.isfinite(pg_loss).all().item(),
                "train_log_probs": all(torch.isfinite(t).all().item() for t in train_log_probs),
                "old_log_probs": all(torch.isfinite(t).all().item() for t in old_log_probs),
                "advantages": all(torch.isfinite(t).all().item() for t in advantages),
            },
        },
        path,
    )
