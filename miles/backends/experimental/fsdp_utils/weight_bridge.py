"""WeightBridge: the explicit train->rollout parameter contract for the FSDP backend.

The training side (stock HF / FSDP) and the SGLang rollout loader do not always agree on
parameter *names and shapes*. Each (model_type) that disagrees needs a transform that rewrites
the streamed ``(name, tensor)`` into what the rollout loader expects — e.g. transformers>=5.6
stores qwen3_moe experts as one batched ``experts.gate_up_proj`` / ``experts.down_proj`` tensor,
but SGLang's qwen3_moe loader expects per-expert ``experts.{i}.{gate,up,down}_proj.weight``.

This module makes that contract a first-class, registered object instead of a hardcoded
``if model_type == ...`` branch. It is the FSDP analogue of the Megatron ``megatron_to_hf``
converter registry, and the seam the backend refactor (DESIGN_NOTES §4 / v2-FIX of the WeightBridge)
generalizes: a new arch registers a ``ParamTransform`` rather than editing the sync loop.

A transform is a pair of callables:
  * ``matches(name, param) -> bool`` — does this transform apply to this param?
  * ``expand(name, full) -> Iterable[(name, tensor)]`` — pure tensor logic producing the rollout
    stream from the *materialized* (unsharded, on-device) tensor.
Keeping ``expand`` pure (no DTensor/device handling) makes every transform unit-testable on CPU;
the device/DTensor materialization stays in the one place that owns it (update_weight_utils).
"""

from collections.abc import Callable, Iterable

import torch

# model_type -> list of (matches, expand)
_REGISTRY: dict[str, list[tuple[Callable, Callable]]] = {}


def register_param_transform(model_type: str, matches: Callable, expand: Callable) -> None:
    _REGISTRY.setdefault(model_type, []).append((matches, expand))


def get_param_transform(name: str, param, model_type: str):
    """Return the ``expand`` fn for the transform matching this param, or None (passthrough)."""
    for matches, expand in _REGISTRY.get(model_type, ()):
        if matches(name, param):
            return expand
    return None


# ------------------------------------------------------------------------------------------------
# qwen3_moe: split transformers>=5.6 batched experts back into the per-expert names SGLang expects.
# ------------------------------------------------------------------------------------------------
def _qwen3_moe_matches(name: str, param) -> bool:
    return getattr(param, "dim", lambda: 0)() == 3 and (
        name.endswith(".experts.gate_up_proj") or name.endswith(".experts.down_proj")
    )


def _qwen3_moe_expand(name: str, full: torch.Tensor) -> Iterable[tuple[str, torch.Tensor]]:
    """experts.gate_up_proj [E,2I,H] -> experts.{i}.{gate,up}_proj.weight; down_proj [E,H,I] -> per-expert."""
    prefix = name.rsplit(".", 1)[0]  # ...mlp.experts
    num_experts = full.shape[0]
    if name.endswith(".gate_up_proj"):
        half = full.shape[1] // 2  # fused rows are [gate | up]
        for i in range(num_experts):
            yield f"{prefix}.{i}.gate_proj.weight", full[i, :half, :].contiguous()
            yield f"{prefix}.{i}.up_proj.weight", full[i, half:, :].contiguous()
    else:  # .down_proj
        for i in range(num_experts):
            yield f"{prefix}.{i}.down_proj.weight", full[i].contiguous()


register_param_transform("qwen3_moe", _qwen3_moe_matches, _qwen3_moe_expand)

# glm4_moe_lite (GLM-4.7-Flash) uses the IDENTICAL batched-expert layout — Glm4MoeLiteNaiveMoe stores
# experts.gate_up_proj [E, 2I, H] (rows [gate|up]) + experts.down_proj [E, H, I] — while sglang's loader
# expects per-expert names (as on disk). Reuse the same split transform.
register_param_transform("glm4_moe_lite", _qwen3_moe_matches, _qwen3_moe_expand)
