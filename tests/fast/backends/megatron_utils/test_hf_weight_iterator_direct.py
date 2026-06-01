import sys
import types
from argparse import Namespace

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu", labels=[])

import pytest
import torch

from miles.utils.types import ParamInfo


def _install_import_stubs(monkeypatch):
    triton = types.ModuleType("triton")
    triton.jit = lambda fn: fn
    triton.cdiv = lambda x, y: (x + y - 1) // y
    tl = types.ModuleType("triton.language")
    tl.constexpr = int
    monkeypatch.setitem(sys.modules, "triton", triton)
    monkeypatch.setitem(sys.modules, "triton.language", tl)

    for name in [
        "sglang",
        "sglang.srt",
        "sglang.srt.utils",
        "sglang.srt.utils.patch_torch",
        "sglang.srt.weight_sync",
        "sglang.srt.weight_sync.tensor_bucket",
        "sglang.srt.layers",
        "sglang.srt.layers.quantization",
        "sglang.srt.layers.quantization.fp8_utils",
    ]:
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))

    sys.modules["sglang.srt.utils"].MultiprocessingSerializer = object
    sys.modules["sglang.srt.utils.patch_torch"].monkey_patch_torch_reductions = lambda: None
    sys.modules["sglang.srt.weight_sync.tensor_bucket"].FlattenedTensorBucket = object
    fp8_utils = sys.modules["sglang.srt.layers.quantization.fp8_utils"]
    fp8_utils.mxfp8_group_quantize = lambda *args, **kwargs: None
    fp8_utils.quant_weight_ue8m0 = lambda *args, **kwargs: None
    fp8_utils.transform_scale_ue8m0 = lambda x, **kwargs: x

    ray = types.ModuleType("ray")
    ray_actor = types.ModuleType("ray.actor")
    ray_actor.ActorHandle = object
    monkeypatch.setitem(sys.modules, "ray", ray)
    monkeypatch.setitem(sys.modules, "ray.actor", ray_actor)

    for name in [
        "megatron",
        "megatron.core",
        "megatron.core.transformer",
        "megatron.core.transformer.transformer_layer",
    ]:
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    sys.modules["megatron.core.transformer.transformer_layer"].get_transformer_layer_offset = lambda *args: 0


@pytest.fixture
def direct_module(monkeypatch):
    module_names = [
        "miles.backends.megatron_utils.sglang",
        "miles.backends.megatron_utils.megatron_to_hf",
        "miles.backends.megatron_utils.megatron_to_hf.processors",
        "miles.backends.megatron_utils.megatron_to_hf.processors.quantizer_fp8",
        "miles.backends.megatron_utils.megatron_to_hf.processors.quantizer_mxfp8",
        "miles.backends.megatron_utils.update_weight.common",
        "miles.backends.megatron_utils.update_weight.hf_weight_iterator_direct",
    ]
    saved_modules = {name: sys.modules.get(name) for name in module_names}
    for name in module_names:
        sys.modules.pop(name, None)

    _install_import_stubs(monkeypatch)

    from miles.backends.megatron_utils.update_weight import hf_weight_iterator_direct

    yield hf_weight_iterator_direct

    for name, module in saved_modules.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


def _param(name: str, size: int) -> ParamInfo:
    return ParamInfo(
        name=name,
        dtype=torch.float32,
        shape=torch.Size([size]),
        attrs={},
        size=size,
        src_rank=0,
    )


def test_atomic_group_is_single_update_unit_and_packed_together(direct_module, monkeypatch):
    from miles.backends.megatron_utils.megatron_to_hf import AtomicUpdateGroup

    params = [_param("a", 4), _param("b", 4), _param("c", 4)]
    monkeypatch.setattr(direct_module, "_get_param_full_size", lambda info: info.size)
    monkeypatch.setattr(
        direct_module,
        "get_atomic_update_groups",
        lambda model_name, param_infos: [AtomicUpdateGroup("pair", ("b", "c"))],
    )

    update_units = direct_module._get_update_units("test-model", params)
    assert [[param.name for param in unit.params] for unit in update_units] == [["a"], ["b", "c"]]

    buckets = direct_module._pack_update_units(Namespace(update_weight_buffer_size=6), update_units)
    assert [[param.name for param in bucket] for bucket in buckets] == [["a"], ["b", "c"]]


def test_atomic_group_specs_raise_explicit_errors(direct_module, monkeypatch):
    from miles.backends.megatron_utils.megatron_to_hf import AtomicUpdateGroup

    params = [_param("a", 4), _param("b", 4)]
    monkeypatch.setattr(direct_module, "_get_param_full_size", lambda info: info.size)

    invalid_groups = [
        ([AtomicUpdateGroup("empty", ())], "Atomic update group empty has no params"),
        ([AtomicUpdateGroup("missing", ("c",))], "Atomic update group missing references unknown param c"),
        (
            [AtomicUpdateGroup("left", ("a",)), AtomicUpdateGroup("right", ("a",))],
            "Param a appears in multiple atomic update groups",
        ),
        (
            [AtomicUpdateGroup("duplicate", ("a",)), AtomicUpdateGroup("duplicate", ("b",))],
            "Duplicate atomic update group: duplicate",
        ),
    ]

    for groups, error in invalid_groups:
        monkeypatch.setattr(
            direct_module,
            "get_atomic_update_groups",
            lambda model_name, param_infos, groups=groups: groups,
        )
        with pytest.raises(RuntimeError, match=error):
            direct_module._get_update_units("test-model", params)
