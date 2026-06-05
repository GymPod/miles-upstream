from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=180, suite="stage-c-4-gpu-h200", labels=[])

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.distributed_c10d import _coalescing_manager

from miles.utils.det_process_group import register_det_nccl_backend

_WORLD_SIZE = 4
_NUMEL = 1_048_576
_SEED = 1234


def _order_sensitive_input(rank: int, seed: int = _SEED) -> torch.Tensor:
    """Per-rank input whose cross-rank sum catastrophically cancels (~1e-4 from +-0.5)."""
    shared = torch.randn(_NUMEL, generator=torch.Generator().manual_seed(seed), dtype=torch.float32)
    own = torch.randn(_NUMEL, generator=torch.Generator().manual_seed(seed + 1 + rank), dtype=torch.float32)
    sign = -1.0 if rank % 2 else 1.0
    return (sign * 0.5 * shared + 1e-4 * own).cuda()


def _manual_tree_sum(partials: list[torch.Tensor]) -> torch.Tensor:
    running = list(partials)
    while len(running) > 1:
        running = [running[i] + running[i + 1] for i in range(0, len(running), 2)]
    return running[0]


def _fixed_tree_reference(x: torch.Tensor) -> torch.Tensor:
    """Gather every rank's tensor (data movement only) and fold in the fixed tree order."""
    gathered = [torch.empty_like(x) for _ in range(_WORLD_SIZE)]
    dist.all_gather(gathered, x)
    return _manual_tree_sum(gathered)


def _assert_bitwise(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    if torch.equal(actual, expected):
        return
    mismatch = int((actual != expected).sum().item())
    max_abs = float((actual - expected).abs().max().item())
    raise AssertionError(f"{name}: mismatch_elems={mismatch}/{actual.numel()} max_abs={max_abs:.3e}")


def _shard_of(full: torch.Tensor, rank: int) -> torch.Tensor:
    shard_numel = full.numel() // _WORLD_SIZE
    return full[rank * shard_numel : (rank + 1) * shard_numel]


def _check_allreduce(rank: int, det1: dist.ProcessGroup, det2: dist.ProcessGroup, x, tree) -> None:
    a = x.clone()
    dist.all_reduce(a, op=dist.ReduceOp.SUM, group=det1)
    _assert_bitwise("allreduce SUM vs fixed tree", a, tree)

    b = x.clone()
    dist.all_reduce(b, op=dist.ReduceOp.SUM, group=det2)
    _assert_bitwise("allreduce bitwise across communicator instances", b, tree)

    averaged = x.clone()
    dist.all_reduce(averaged, op=dist.ReduceOp.AVG, group=det1)
    _assert_bitwise("allreduce AVG == SUM/world", averaged, tree / _WORLD_SIZE)

    max_det = x.clone()
    dist.all_reduce(max_det, op=dist.ReduceOp.MAX, group=det1)
    max_native = x.clone()
    dist.all_reduce(max_native, op=dist.ReduceOp.MAX)
    _assert_bitwise("allreduce MAX delegates to native", max_det, max_native)


def _check_reduce_scatter_vs_allreduce(rank: int, det1: dist.ProcessGroup, x, tree) -> None:
    expected_shard = _shard_of(tree, rank)

    rs = torch.empty_like(expected_shard)
    dist.reduce_scatter_tensor(rs, x.clone(), op=dist.ReduceOp.SUM, group=det1)
    _assert_bitwise("reduce_scatter_tensor == slice of allreduce", rs, expected_shard)

    # Megatron distributed-optimizer style: the output shard is a view of the input.
    buf = x.clone()
    shard_view = buf.view(_WORLD_SIZE, -1)[rank]
    dist.reduce_scatter_tensor(shard_view, buf, op=dist.ReduceOp.SUM, group=det1)
    _assert_bitwise("aliased reduce_scatter_tensor", shard_view, expected_shard.view(shard_view.shape))

    # List variant: rank r's output = sum over ranks of their r-th chunk
    # (= the same slice of the full fold, since slicing commutes with elementwise sums).
    inputs = [chunk.contiguous() for chunk in x.clone().chunk(_WORLD_SIZE)]
    out = torch.empty_like(expected_shard)
    dist.reduce_scatter(out, inputs, op=dist.ReduceOp.SUM, group=det1)
    _assert_bitwise("reduce_scatter (list) == slice of allreduce", out, expected_shard)


def _check_coalescing_manager(rank: int, det1: dist.ProcessGroup, x, tree, x2, tree2) -> None:
    device = torch.device("cuda", torch.cuda.current_device())

    ar1, ar2 = x.clone(), x2.clone()
    with _coalescing_manager(group=det1, device=device):
        dist.all_reduce(ar1, op=dist.ReduceOp.SUM, group=det1)
        dist.all_reduce(ar2, op=dist.ReduceOp.SUM, group=det1)
    _assert_bitwise("coalescing_manager AR (1st)", ar1, tree)
    _assert_bitwise("coalescing_manager AR (2nd)", ar2, tree2)

    rs1 = torch.empty_like(_shard_of(tree, rank))
    rs2 = torch.empty_like(_shard_of(tree2, rank))
    with _coalescing_manager(group=det1, device=device):
        dist.reduce_scatter_tensor(rs1, x.clone(), op=dist.ReduceOp.SUM, group=det1)
        dist.reduce_scatter_tensor(rs2, x2.clone(), op=dist.ReduceOp.SUM, group=det1)
    _assert_bitwise("coalescing_manager RS (1st)", rs1, _shard_of(tree, rank))
    _assert_bitwise("coalescing_manager RS (2nd)", rs2, _shard_of(tree2, rank))

    shard_in = torch.full((128,), float(rank), device=device)
    full_out = torch.empty(128 * _WORLD_SIZE, device=device)
    with _coalescing_manager(group=det1, device=device):
        dist.all_gather_into_tensor(full_out, shard_in, group=det1)
    expected = torch.cat([torch.full((128,), float(r), device=device) for r in range(_WORLD_SIZE)])
    _assert_bitwise("coalescing_manager all_gather_into_tensor", full_out, expected)


def _check_non_contiguous(rank: int, det1: dist.ProcessGroup) -> None:
    base = torch.randn(64, 64, generator=torch.Generator().manual_seed(_SEED + 100 + rank)).cuda()
    non_contiguous = base.t()
    assert not non_contiguous.is_contiguous()

    gathered = [torch.empty_like(base) for _ in range(_WORLD_SIZE)]
    dist.all_gather(gathered, base)
    expected = _manual_tree_sum([g.t().contiguous() for g in gathered])

    dist.all_reduce(non_contiguous, op=dist.ReduceOp.SUM, group=det1)
    _assert_bitwise("non-contiguous allreduce", non_contiguous.contiguous(), expected)


def _check_delegation(rank: int, det1: dist.ProcessGroup) -> None:
    device = torch.device("cuda", torch.cuda.current_device())

    broadcasted = torch.full((8,), float(rank), device=device)
    dist.broadcast(broadcasted, src=0, group=det1)
    _assert_bitwise("broadcast", broadcasted, torch.zeros(8, device=device))

    piece = torch.full((4,), float(rank), device=device)
    pieces = [torch.empty_like(piece) for _ in range(_WORLD_SIZE)]
    dist.all_gather(pieces, piece, group=det1)
    for source_rank, gathered_piece in enumerate(pieces):
        _assert_bitwise(
            f"all_gather piece {source_rank}", gathered_piece, torch.full((4,), float(source_rank), device=device)
        )

    full = torch.empty(4 * _WORLD_SIZE, device=device)
    dist.all_gather_into_tensor(full, piece, group=det1)
    expected = torch.cat([torch.full((4,), float(r), device=device) for r in range(_WORLD_SIZE)])
    _assert_bitwise("all_gather_into_tensor", full, expected)

    reduced = torch.full((4,), float(rank), device=device)
    dist.reduce(reduced, dst=0, op=dist.ReduceOp.MAX, group=det1)
    if rank == 0:
        _assert_bitwise("reduce MAX to dst", reduced, torch.full((4,), float(_WORLD_SIZE - 1), device=device))

    # rank r receives element r from every rank q: value = r + 10*q at position q
    scatter_in = torch.arange(_WORLD_SIZE, dtype=torch.float32, device=device) + rank * 10
    a2a_out = torch.empty(_WORLD_SIZE, device=device)
    dist.all_to_all_single(a2a_out, scatter_in, group=det1)
    expected_a2a = torch.tensor([float(rank + 10 * q) for q in range(_WORLD_SIZE)], device=device)
    _assert_bitwise("all_to_all_single", a2a_out, expected_a2a)

    peer = rank + 1 if rank % 2 == 0 else rank - 1
    outgoing = torch.full((4,), float(rank), device=device)
    incoming = torch.empty(4, device=device)
    if rank % 2 == 0:
        dist.send(outgoing, dst=peer, group=det1)
        dist.recv(incoming, src=peer, group=det1)
    else:
        dist.recv(incoming, src=peer, group=det1)
        dist.send(outgoing, dst=peer, group=det1)
    _assert_bitwise("send/recv", incoming, torch.full((4,), float(peer), device=device))

    dist.barrier(group=det1)

    assert dist.get_backend(det1) == "det_nccl", f"unexpected backend name {dist.get_backend(det1)}"


def _worker(rank: int, world_size: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    register_det_nccl_backend()
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    det1 = dist.new_group(list(range(world_size)), backend="det_nccl")
    det2 = dist.new_group(list(range(world_size)), backend="det_nccl")

    x = _order_sensitive_input(rank)
    x2 = _order_sensitive_input(rank, seed=_SEED + 50)
    tree = _fixed_tree_reference(x)
    tree2 = _fixed_tree_reference(x2)

    _check_allreduce(rank, det1, det2, x, tree)
    _check_reduce_scatter_vs_allreduce(rank, det1, x, tree)
    _check_coalescing_manager(rank, det1, x, tree, x2, tree2)
    _check_non_contiguous(rank, det1)
    _check_delegation(rank, det1)

    dist.barrier()
    dist.destroy_process_group()


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("localhost", 0))
        return sock.getsockname()[1]


def test_det_process_group_multi_gpu():
    """det_nccl backend: bitwise fixed-order SUM/AVG (allreduce + reduce_scatter, incl. under
    the coalescing manager) and faithful delegation of every other collective, on 4 GPUs."""
    if torch.cuda.device_count() < _WORLD_SIZE:
        pytest.skip(f"requires {_WORLD_SIZE} GPUs, found {torch.cuda.device_count()}")

    mp.spawn(_worker, args=(_WORLD_SIZE, _free_port()), nprocs=_WORLD_SIZE, join=True)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
