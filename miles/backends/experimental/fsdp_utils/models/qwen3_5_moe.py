"""GatedDeltaNet packed-sequence fix for the FSDP backend (explicit cu_seqlens threading).

The FSDP backend packs multiple (prompt+response) documents into one forward
(``--use-dynamic-batch-size``). Softmax-attention layers handle this via varlen
flash-attn + position_ids, but the stock-HF GatedDeltaNet runs its TWO stateful ops
over the whole packed row without any sequence boundaries:

  * the linear-attention recurrence (fla ``chunk_gated_delta_rule`` / ``fused_recurrent_*``), and
  * the ``causal_conv1d`` short convolution.

Both bleed across document boundaries, so every token after the first packed document
gets a wrong hidden state -> the train/rollout logprob abs-diff inflates. (The SGLang
rollout runs each sequence separately, so it has no bleed -- the FSDP train side is the
odd one out; Megatron passes cu_seqlens to its GDN and resets per document.)

Fix (explicit, no global/thread-local state): the decoder-layer forward already receives
the packed ``position_ids``; it derives ``cu_seqlens`` (recurrence) and ``seq_idx`` (conv)
from them (the shared ``packed_seq_context`` derivation) and stashes them on its own
GatedDeltaNet submodule. The GatedDeltaNet forward then injects them into the fla /
causal_conv1d kernels so both states reset per document. Both run inside the
gradient-checkpointed decoder layer, so the boundaries are recomputed identically on the
backward pass (setting them at the outer model forward would break activation
checkpointing, since that forward is not re-run during recomputation).

This only affects the THD packing case (batch==1, >1 document) of GatedDeltaNet models;
every other shape / model is left untouched.
"""

import functools
import logging

from ..packing.boundaries import packed_seq_context

logger = logging.getLogger(__name__)


def _inject_kwarg(fn, key, value):
    """Wrap a kernel callable to default a kwarg (cu_seqlens / seq_idx) when unset."""

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        if kwargs.get(key) is None:
            kwargs[key] = value
        return fn(*args, **kwargs)

    return wrapped


def _patch_gdn_forward(gdn_cls):
    orig = gdn_cls.forward
    if getattr(orig, "_gdn_packing", False):
        return

    # The recurrence / conv kernels are stored as *instance* attributes on the module
    # (self.chunk_gated_delta_rule, self.causal_conv1d_fn, ...). We temporarily rebind
    # them for the duration of the forward to inject the per-document boundaries, then
    # restore -- so nothing leaks across modules or forwards.
    _INJECT = (
        ("chunk_gated_delta_rule", "cu_seqlens"),
        ("recurrent_gated_delta_rule", "cu_seqlens"),
        ("causal_conv1d_fn", "seq_idx"),
    )

    @functools.wraps(orig)
    def forward(self, *args, **kwargs):
        cu = getattr(self, "_gdn_cu_seqlens", None)
        si = getattr(self, "_gdn_seq_idx", None)
        if cu is None and si is None:
            return orig(self, *args, **kwargs)
        saved = {}
        for attr, key in _INJECT:
            value = cu if key == "cu_seqlens" else si
            fn = getattr(self, attr, None)
            if fn is not None and value is not None:
                saved[attr] = fn
                setattr(self, attr, _inject_kwarg(fn, key, value))
        try:
            return orig(self, *args, **kwargs)
        finally:
            for attr, fn in saved.items():
                setattr(self, attr, fn)

    forward._gdn_packing = True
    gdn_cls.forward = forward


def _patch_decoder_forward(dl_cls, gdn_cls):
    orig = dl_cls.forward
    if getattr(orig, "_gdn_packing", False):
        return

    @functools.wraps(orig)
    def forward(self, *args, **kwargs):
        ctx = packed_seq_context(kwargs.get("position_ids"))
        if ctx is not None:
            for module in self.modules():
                if isinstance(module, gdn_cls):
                    module._gdn_cu_seqlens = ctx.cu_seqlens
                    module._gdn_seq_idx = ctx.seq_idx
        return orig(self, *args, **kwargs)

    forward._gdn_packing = True
    dl_cls.forward = forward


def _find_class(mod, suffix):
    for name in dir(mod):
        if name.endswith(suffix):
            return getattr(mod, name)
    return None


def apply_gateddeltanet_packing_patch():
    """Reset GatedDeltaNet recurrence + conv state at packed-document boundaries.

    Idempotent; patches every GatedDeltaNet hybrid arch present (Qwen3.5/3.6 MoE,
    Qwen3-Next, ...). Returns True if anything was patched.
    """
    patched = False
    for mod_name in ("qwen3_5_moe", "qwen3_next"):
        try:
            mod = __import__(f"transformers.models.{mod_name}.modeling_{mod_name}", fromlist=["x"])
        except Exception:
            continue
        gdn_cls = _find_class(mod, "GatedDeltaNet")
        dl_cls = _find_class(mod, "DecoderLayer")
        if gdn_cls is None or dl_cls is None:
            continue
        _patch_gdn_forward(gdn_cls)
        _patch_decoder_forward(dl_cls, gdn_cls)
        patched = True

    if patched:
        logger.info(
            "[fsdp] GatedDeltaNet packing fix applied: cu_seqlens/seq_idx reset the "
            "linear-attn recurrence and causal-conv state per packed document"
        )
    return patched
