"""Unit tests for the experimental-FSDP HF-compat fixes (CPU-only, no GPU/sglang).

Covers:
  * F9 weight-sync: batched MoE expert params are split into the per-expert names
    SGLang expects, with the correct gate/up row split, and only for the right
    model types.
  * F8: the legacy qwen3_moe MoE graph patch no-ops (does not crash) on the
    transformers>=5.6 batched structure.
"""

import torch

from miles.backends.experimental.fsdp_utils.update_weight_utils import (
    _is_batched_expert_param,
    _iter_sync_named_params,
    _split_batched_moe_expert,
)


def test_split_gate_up_proj_rows_and_names():
    # [E=2, 2I=6, H=4]: fused rows are [gate(:3) | up(3:)]
    E, I, H = 2, 3, 4
    full = torch.arange(E * 2 * I * H, dtype=torch.float32).reshape(E, 2 * I, H)
    out = dict(_split_batched_moe_expert("model.layers.0.mlp.experts.gate_up_proj", full))

    assert set(out) == {
        "model.layers.0.mlp.experts.0.gate_proj.weight",
        "model.layers.0.mlp.experts.0.up_proj.weight",
        "model.layers.0.mlp.experts.1.gate_proj.weight",
        "model.layers.0.mlp.experts.1.up_proj.weight",
    }
    for e in range(E):
        g = out[f"model.layers.0.mlp.experts.{e}.gate_proj.weight"]
        u = out[f"model.layers.0.mlp.experts.{e}.up_proj.weight"]
        assert g.shape == (I, H) and u.shape == (I, H)
        torch.testing.assert_close(g, full[e, :I, :])
        torch.testing.assert_close(u, full[e, I:, :])
        assert g.is_contiguous() and u.is_contiguous()


def test_split_down_proj():
    E, H, I = 2, 4, 3
    full = torch.arange(E * H * I, dtype=torch.float32).reshape(E, H, I)
    out = dict(_split_batched_moe_expert("model.layers.5.mlp.experts.down_proj", full))
    assert set(out) == {
        "model.layers.5.mlp.experts.0.down_proj.weight",
        "model.layers.5.mlp.experts.1.down_proj.weight",
    }
    for e in range(E):
        d = out[f"model.layers.5.mlp.experts.{e}.down_proj.weight"]
        assert d.shape == (H, I)
        torch.testing.assert_close(d, full[e])


def test_is_batched_expert_param_gating():
    gate_up = torch.zeros(2, 6, 4)
    name = "model.layers.0.mlp.experts.gate_up_proj"
    # only for model types whose SGLang loader expects per-expert weights
    assert _is_batched_expert_param(name, gate_up, "qwen3_moe")
    assert not _is_batched_expert_param(name, gate_up, "qwen3_5_moe")  # consumes batched directly
    assert not _is_batched_expert_param(name, gate_up, "qwen3")  # dense
    # non-expert params are never split
    assert not _is_batched_expert_param("model.layers.0.self_attn.q_proj.weight", torch.zeros(4, 4), "qwen3_moe")
    # 2D tensor named like an expert param is not the batched layout
    assert not _is_batched_expert_param(name, torch.zeros(6, 4), "qwen3_moe")


def test_iter_passthrough_for_non_expert():
    p = torch.zeros(4, 4)
    out = list(_iter_sync_named_params("model.embed_tokens.weight", p, "qwen3_moe"))
    assert len(out) == 1 and out[0][0] == "model.embed_tokens.weight" and out[0][1] is p
    # expert-named param under a model type that consumes batched layout -> passthrough
    g = torch.zeros(2, 6, 4)
    out = list(_iter_sync_named_params("model.layers.0.mlp.experts.gate_up_proj", g, "qwen3_5_moe"))
    assert len(out) == 1 and out[0][1] is g


def test_qwen3_moe_patch_noops_on_batched_structure():
    # On transformers>=5.6 the patch must not replace the forward (batched experts).
    from transformers.models.qwen3_moe import modeling_qwen3_moe

    from miles.backends.experimental.fsdp_utils.models.qwen3_moe_hf import apply_fsdp_moe_patch

    original_forward = modeling_qwen3_moe.Qwen3MoeSparseMoeBlock.forward
    apply_fsdp_moe_patch()  # must not raise
    if hasattr(modeling_qwen3_moe, "Qwen3MoeExperts") or hasattr(modeling_qwen3_moe, "Qwen3MoeTopKRouter"):
        assert modeling_qwen3_moe.Qwen3MoeSparseMoeBlock.forward is original_forward


def test_is_mamba_hybrid_gating():
    # The clobber-reload only runs for Mamba/SSM-hybrid archs (NemotronH _init_weights
    # re-inits dt_bias + out_proj post-load); it must be a no-op gate for everything else.
    from types import SimpleNamespace

    from miles.backends.experimental.fsdp_utils.hf_compat_patches import _is_mamba_hybrid

    assert _is_mamba_hybrid(SimpleNamespace(model_type="nemotron_h"))
    assert _is_mamba_hybrid(SimpleNamespace(model_type="mamba2"))
    # detected via layer_types even when model_type doesn't say "mamba"
    assert _is_mamba_hybrid(SimpleNamespace(model_type="hybrid", layer_types=["mamba", "attention"]))
    # dense / non-mamba archs must NOT trigger the reload
    assert not _is_mamba_hybrid(SimpleNamespace(model_type="qwen3"))
    assert not _is_mamba_hybrid(SimpleNamespace(model_type="qwen3_moe"))
    assert not _is_mamba_hybrid(SimpleNamespace(model_type="llama", layer_types=["attention"]))


def test_weight_bridge_registry():
    # The WeightBridge registry is the train->rollout param-name/shape contract: a model type with
    # a registered transform gets its params rewritten; unregistered types stream verbatim.
    import torch

    from miles.backends.experimental.fsdp_utils.weight_bridge import (
        get_param_transform,
        register_param_transform,
    )

    # qwen3_moe is registered (batched experts -> per-expert); a 3D experts param matches.
    g = torch.zeros(2, 6, 4)
    assert get_param_transform("model.layers.0.mlp.experts.gate_up_proj", g, "qwen3_moe") is not None
    # unregistered model type -> no transform (passthrough)
    assert get_param_transform("model.layers.0.mlp.experts.gate_up_proj", g, "qwen3_5_moe") is None
    # registering a new transform routes matching params through it
    register_param_transform(
        "_test_arch",
        matches=lambda name, p: name.endswith(".foo"),
        expand=lambda name, full: [(name.replace(".foo", ".bar"), full)],
    )
    fn = get_param_transform("x.foo", g, "_test_arch")
    assert fn is not None and list(fn("x.foo", g))[0][0] == "x.bar"
    assert get_param_transform("x.baz", g, "_test_arch") is None


def test_model_patch_registry_gating():
    # The ModelPatchHook registry replaces the hardcoded per-arch dispatch in apply_hf_compat_patches.
    # Verify the predicates gate correctly (s_aux always; config-checks need a config; GDN needs gated-deltanet).
    from types import SimpleNamespace

    from miles.backends.experimental.fsdp_utils.hf_compat_patches import _MODEL_PATCH_HOOKS

    by_name = {h.name: h for h in _MODEL_PATCH_HOOKS}
    # the four expected hooks are registered, in order
    assert [h.name for h in _MODEL_PATCH_HOOKS][:4] == [
        "flash_attn_saux_guard",
        "fp8_checkpoint_guard",
        "dsa_train_infer_warn",
        "gated_deltanet_packing",
    ]
    # s_aux guard runs even without a config; the others don't
    assert by_name["flash_attn_saux_guard"].applies_to(None)
    assert not by_name["fp8_checkpoint_guard"].applies_to(None)
    assert not by_name["gated_deltanet_packing"].applies_to(None)
    # GDN packing fires for a gated-deltanet arch, not for a dense one
    gdn = SimpleNamespace(model_type="qwen3_5_moe", layer_types=["linear_attention", "full_attention"])
    dense = SimpleNamespace(model_type="qwen3", layer_types=["full_attention"])
    assert by_name["gated_deltanet_packing"].applies_to(gdn)
    assert not by_name["gated_deltanet_packing"].applies_to(dense)
