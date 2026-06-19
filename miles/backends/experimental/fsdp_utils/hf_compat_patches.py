"""HuggingFace-version compatibility patches for the experimental FSDP backend.

The FSDP backend trains the stock HuggingFace model, so it is sensitive to
transformers-version drift. These patches keep the training forward runnable and warn
on architectures whose forward diverges from the SGLang rollout. All are idempotent and
no-op when the underlying issue is absent.
"""

import inspect
import logging
import textwrap

import torch

logger = logging.getLogger(__name__)


def apply_flash_attn_saux_guard() -> bool:
    """Guard ``s_aux`` (attention sink) against ``None`` in flash_attention_forward.

    transformers 5.6.0 does ``s_aux.to(query.dtype)`` unconditionally; for models without
    sinks (Qwen3, Qwen3.5) ``s_aux`` is None and the first training forward raises. Recompile
    the function with a None-guard and re-register it. Returns True if patched.
    """
    try:
        import transformers.integrations.flash_attention as fa
    except Exception:  # pragma: no cover
        return False
    try:
        src = inspect.getsource(fa.flash_attention_forward)
    except (OSError, TypeError):
        return False

    BUG = "s_aux=s_aux.to(query.dtype)"
    if "if s_aux is not None" in src or BUG not in src:
        return False  # already guarded, or an unrecognized layout

    new_src = textwrap.dedent(src).replace(
        BUG, "s_aux=(s_aux.to(query.dtype) if s_aux is not None else None)"
    )
    ns = vars(fa)
    try:
        exec(compile(new_src, fa.__file__, "exec"), ns)  # noqa: S102 - controlled recompile
    except Exception as e:  # pragma: no cover
        logger.warning(f"[fsdp hf_compat] s_aux guard compile failed: {e}")
        return False
    patched = ns["flash_attention_forward"]
    patched._saux_guarded = True
    fa.flash_attention_forward = patched

    try:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS as A

        for key in list(A.valid_keys()):
            try:
                cur = A[key]
            except Exception:
                continue
            if getattr(cur, "__name__", None) == "flash_attention_forward":
                try:
                    A[key] = patched
                except Exception:
                    try:
                        A.register(key, patched, exist_ok=True)
                    except Exception:
                        pass
    except Exception as e:  # pragma: no cover
        logger.warning(f"[fsdp hf_compat] s_aux guard re-register skipped: {e}")

    logger.info("[fsdp hf_compat] applied flash-attention s_aux None-guard")
    return True


def check_train_infer_consistency(hf_config) -> None:
    """Warn when an arch's training forward diverges structurally from the SGLang rollout.

    DeepSeek-V3.2/DSA: the sparse-attention indexer is absent from HF training modeling, so
    its weights load UNEXPECTED and are never trained (train is dense, rollout is sparse).
    """
    model_type = str(getattr(hf_config, "model_type", "") or "")
    is_dsa = (
        "deepseek_v3" in model_type
        or bool(getattr(hf_config, "index_topk", None))
        or getattr(hf_config, "attn_module_list_cfg", None) is not None
    )
    if is_dsa:
        logger.warning(
            "[fsdp hf_compat] DeepSeek sparse-attention (DSA) detected (model_type=%s): the HF "
            "training forward has no indexer, so it is dropped and train attention is DENSE while "
            "the rollout is SPARSE. RL on DSA via FSDP is not currently consistent.",
            model_type,
        )


def check_fp8_checkpoint(hf_config) -> None:
    """Fail fast on native-fp8 checkpoints (the actor has no inline dequant)."""
    qc = getattr(hf_config, "quantization_config", None)
    if not qc:
        return
    method = qc.get("quant_method") if isinstance(qc, dict) else getattr(qc, "quant_method", None)
    if str(method or "").lower() == "fp8":
        raise ValueError(
            "FSDP backend cannot train from an fp8-quantized checkpoint "
            "(quantization_config.quant_method='fp8'). Convert to bf16 first:\n"
            "  python tools/fp8_cast_bf16.py --input-fp8-hf-path <src> --output-bf16-hf-path <dst>\n"
            "then copy config/tokenizer (dropping quantization_config) into <dst> and point "
            "--hf-checkpoint at it."
        )


def _is_mamba_hybrid(hf_config) -> bool:
    """True for Mamba/SSM-hybrid archs whose HF `_init_weights` clobbers loaded weights post-load."""
    model_type = str(getattr(hf_config, "model_type", "") or "").lower()
    if "nemotron_h" in model_type or "mamba" in model_type:
        return True
    tc = getattr(hf_config, "get_text_config", lambda: hf_config)()
    layer_types = getattr(tc, "layer_types", None) or getattr(hf_config, "layer_types", None)
    return bool(layer_types) and any("mamba" in str(t).lower() for t in layer_types)


def reload_clobbered_checkpoint_params(model, ckpt_path, hf_config, tol=1e-3) -> int:
    """Re-assert the checkpoint over params that ``from_pretrained`` silently clobbered post-load.

    transformers' NemotronH (Mamba2) ``_init_weights`` runs AFTER weight loading and re-initializes
    Mamba special-init params — observed: every layer's ``mixer.dt_bias`` (Mamba dt init, off by ~40)
    AND ``mixer.out_proj.weight`` (residual-rescaled init). The FSDP backend then trains on these wrong
    values and syncs them to sglang, producing garbage rollouts (logprob_abs_diff ~1.2). The on-disk
    checkpoint is ground truth (standalone sglang loads it and generates coherently), so reload every
    param whose materialized value differs from disk by > ``tol``.

    Gated to Mamba/hybrid archs so it never reverts an intended from_pretrained transform elsewhere.
    Runs only where weights are materialized (rank-0 CPU load); meta-device ranks are skipped and get
    the corrected value via the rank-0 broadcast in ``_fsdp2_load_full_state_dict``. Returns the count.
    """
    if not _is_mamba_hybrid(hf_config):
        return 0
    import glob
    import json
    import os

    try:
        from safetensors import safe_open
    except Exception:  # pragma: no cover
        return 0
    files = sorted(glob.glob(os.path.join(ckpt_path, "*.safetensors")))
    if not files:
        return 0
    index = os.path.join(ckpt_path, "model.safetensors.index.json")
    shard_of = json.load(open(index))["weight_map"] if os.path.exists(index) else {}

    reloaded = 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.device.type == "meta":
                continue
            shards = [os.path.join(ckpt_path, shard_of[name])] if name in shard_of else files
            for f in shards:
                try:
                    with safe_open(f, framework="pt") as sf:
                        if name not in sf.keys():
                            continue
                        disk = sf.get_tensor(name)
                except Exception:
                    continue
                if disk.shape == param.shape:
                    disk = disk.to(param.dtype)
                    if (param.detach() - disk).abs().max().item() > tol:
                        param.copy_(disk)
                        reloaded += 1
                break
    if reloaded:
        logger.info("[fsdp hf_compat] re-asserted %d checkpoint param(s) that from_pretrained clobbered "
                    "post-load (Mamba _init_weights)", reloaded)
    return reloaded


def _is_gated_deltanet(hf_config) -> bool:
    model_type = str(getattr(hf_config, "model_type", "") or "")
    tc = getattr(hf_config, "get_text_config", lambda: hf_config)()
    layer_types = getattr(tc, "layer_types", None) or getattr(hf_config, "layer_types", None)
    return (layer_types is not None and "linear_attention" in layer_types) or "qwen3_5" in model_type


class ModelPatchHook:
    """A config-time HF-compat patch: an ``applies_to`` predicate + an ``apply`` action.

    Makes the per-arch dispatch a first-class registry
    instead of a hardcoded ``if _is_gated_deltanet(...)`` chain. New archs register a hook rather than
    editing ``apply_hf_compat_patches``. ``applies_to``/``apply`` both take the hf_config (or None);
    self-gating checks (fp8 fail-fast, DSA warn) use ``applies_to=_has_config`` and gate internally.
    """

    def __init__(self, name, applies_to, apply):
        self.name = name
        self.applies_to = applies_to
        self.apply = apply


_MODEL_PATCH_HOOKS: list[ModelPatchHook] = []


def register_model_patch(hook: ModelPatchHook) -> None:
    _MODEL_PATCH_HOOKS.append(hook)


def _always(hf_config) -> bool:
    return True


def _has_config(hf_config) -> bool:
    return hf_config is not None


# Order preserved from the original hardcoded sequence (s_aux guard, fp8 guard, DSA warn, GDN packing).
register_model_patch(ModelPatchHook("flash_attn_saux_guard", _always, lambda cfg: apply_flash_attn_saux_guard()))
register_model_patch(ModelPatchHook("fp8_checkpoint_guard", _has_config, check_fp8_checkpoint))
register_model_patch(ModelPatchHook("dsa_train_infer_warn", _has_config, check_train_infer_consistency))


def apply_hf_compat_patches(hf_config=None) -> None:
    """Apply all registered FSDP HF-compat patches. Safe to call once at actor init.

    (The post-load Mamba clobber-reload is applied separately in the actor, since it needs the
    instantiated model, not just the config — see ``reload_clobbered_checkpoint_params``.)
    """
    for hook in _MODEL_PATCH_HOOKS:
        if hook.applies_to(hf_config):
            hook.apply(hf_config)
    # Config-lifetime packed-sequence layout patches (GatedDeltaNet, ...) go through the unified
    # packing registry; new packing archs register a PackingPatch there instead of a ModelPatchHook.
    from .packing import apply_packing

    apply_packing(None, hf_config, "config")
