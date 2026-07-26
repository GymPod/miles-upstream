import datetime

import numpy
import pytest
import torch
from PIL import Image

from miles.utils.source_fingerprint import canonical_source_digest


def test_canonical_source_digest_preserves_structured_numpy_dtype() -> None:
    first = numpy.array([(1, 2)], dtype=[("left", "<i4"), ("right", "<i4")])
    second = numpy.array([(1, 2)], dtype=[("first", "<i4"), ("second", "<i4")])

    assert first.tobytes() == second.tobytes()
    assert first.dtype.str == second.dtype.str
    assert canonical_source_digest(first) != canonical_source_digest(second)


def test_canonical_source_digest_does_not_hash_numpy_object_pointers() -> None:
    first = numpy.array([({"value": 1},)], dtype=[("payload", object)])[0]
    second = numpy.array([({"value": 1},)], dtype=[("payload", object)])[0]

    assert first.tobytes() != second.tobytes()
    assert canonical_source_digest(first) == canonical_source_digest(second)


def test_canonical_source_digest_includes_masked_array_masks() -> None:
    first = numpy.ma.array([1, 2], mask=[False, True])
    second = numpy.ma.array([1, 2], mask=[True, False])

    assert canonical_source_digest(first) != canonical_source_digest(second)


def test_canonical_source_digest_ignores_structured_array_padding() -> None:
    dtype = numpy.dtype(
        {
            "names": ["value"],
            "formats": ["u1"],
            "offsets": [0],
            "itemsize": 4,
        }
    )
    first = numpy.empty(1, dtype=dtype)
    second = numpy.empty(1, dtype=dtype)
    first.view(numpy.uint8)[:] = 0
    second.view(numpy.uint8)[:] = 255
    first["value"] = 7
    second["value"] = 7

    assert first.tolist() == second.tolist()
    assert first.tobytes() != second.tobytes()
    assert canonical_source_digest(first) == canonical_source_digest(second)


@pytest.mark.parametrize("dtype", [numpy.longdouble, numpy.clongdouble])
def test_canonical_source_digest_ignores_extended_numpy_padding(dtype: type[numpy.generic]) -> None:
    first = numpy.array([dtype(1.25)], dtype=dtype)
    second = first.copy()
    original = second[0]

    for index in range(second.itemsize):
        candidate = first.copy()
        candidate.view(numpy.uint8)[index] ^= 255
        if candidate[0] == original:
            second = candidate
            break
    else:
        pytest.skip(f"{second.dtype} has no observable padding bytes on this platform")

    assert first.tobytes() != second.tobytes()
    assert first.tolist() == second.tolist()
    assert canonical_source_digest(first) == canonical_source_digest(second)


def test_canonical_source_digest_supports_timedelta_nanoseconds() -> None:
    class NanosecondTimedelta(datetime.timedelta):
        nanoseconds = 1

    first = NanosecondTimedelta(seconds=5)
    second = datetime.timedelta(seconds=5)

    assert canonical_source_digest(first) != canonical_source_digest(second)


def test_canonical_source_digest_normalizes_tensor_views() -> None:
    tensor = torch.arange(6, dtype=torch.int64).reshape(2, 3)
    noncontiguous = tensor.t().contiguous().t()

    assert not noncontiguous.is_contiguous()
    assert canonical_source_digest(tensor) == canonical_source_digest(noncontiguous)


def test_canonical_source_digest_resolves_conjugate_tensor_views() -> None:
    tensor = torch.tensor([1 + 2j], dtype=torch.complex64)

    assert canonical_source_digest(tensor.conj()) == canonical_source_digest(torch.tensor([1 - 2j]))


def test_canonical_source_digest_rejects_quantized_tensors() -> None:
    tensor = torch.quantize_per_tensor(torch.tensor([1.0, 2.0]), scale=0.1, zero_point=10, dtype=torch.quint8)

    with pytest.raises(TypeError, match="Cannot fingerprint quantized tensors"):
        canonical_source_digest(tensor)


def test_canonical_source_digest_includes_palette_transparency() -> None:
    first = Image.new("P", (1, 1))
    second = Image.new("P", (1, 1))
    palette = [0, 0, 0, 255, 255, 255] + [0] * 762
    first.putpalette(palette)
    second.putpalette(palette)
    first.info["transparency"] = 0
    second.info["transparency"] = 1

    assert first.convert("RGBA").tobytes() != second.convert("RGBA").tobytes()
    assert canonical_source_digest(first) != canonical_source_digest(second)
