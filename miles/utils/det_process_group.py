"""Process group with bitwise-deterministic SUM reductions.

``DetProcessGroup`` wraps an inner c10d NCCL group. Every collective delegates to
the inner group except the order-sensitive reductions — ``allreduce`` and
``reduce_scatter`` — which are computed as all-gather (pure data movement, no
arithmetic) plus a fixed local fold: a pairwise
tree for power-of-two world sizes, an ascending-rank fold otherwise. The summation
order is therefore independent of the NCCL version, topology, communicator
instance, or buffer layout, and reduce-scatter takes its shard from the same full
fold, so reduce-scatter and all-reduce agree bitwise by construction.

Debug/test use only: the fold trades bandwidth and synchrony for determinism.
"""

import logging
from collections.abc import Callable

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup as BaseProcessGroup
from torch.distributed import Work
from torch.distributed.distributed_c10d import AllgatherOptions

logger = logging.getLogger(__name__)


_BACKEND_NAME = "det_nccl"
_backend_registered = False

# Cap on the gather buffer so the fold never allocates world_size x the full tensor
# (a 30B MoE grad buffer x 8 ranks is >100 GiB). Chunking is bitwise-neutral: each
# output element is still folded over the same per-rank operands in the same order.
_GATHER_BUFFER_CAP_BYTES = 1 << 30


def register_det_nccl_backend() -> None:
    """Register the ``det_nccl`` torch.distributed backend (DetProcessGroup over NCCL).

    After registration, groups created with ``backend="det_nccl"`` (via
    ``init_process_group`` or ``new_group``) run SUM/AVG reductions through the
    deterministic fold. Idempotent.
    """
    global _backend_registered
    if _backend_registered:
        return
    dist.Backend.register_backend(_BACKEND_NAME, _create_det_nccl_backend, extended_api=True, devices=["cuda"])
    _backend_registered = True
    logger.info("Registered torch.distributed backend %s", _BACKEND_NAME)


def _create_det_nccl_backend(dist_backend_opts: object, pg_options: object) -> "DetProcessGroup":
    from torch.distributed import ProcessGroupNCCL

    inner = ProcessGroupNCCL(
        dist_backend_opts.store,
        dist_backend_opts.group_rank,
        dist_backend_opts.group_size,
        ProcessGroupNCCL.Options(),
    )
    return DetProcessGroup(inner)


class DetProcessGroup(BaseProcessGroup):
    """Wrapper process group whose SUM/AVG reductions use a fixed-order fold."""

    def __init__(self, inner: dist.ProcessGroup) -> None:
        super().__init__(inner.rank(), inner.size())
        self._inner = inner
        # Register the cuda backend as torchft's ProcessGroupNCCL._create_pg does, so
        # _device_types reports cuda and object collectives pick cuda over cpu. Only
        # feeds device/backend lookups; collectives still dispatch to our methods.
        self._set_default_backend(BaseProcessGroup.BackendType.CUSTOM)
        self._register_backend(torch.device("cuda"), BaseProcessGroup.BackendType.CUSTOM, self._inner)

    # ------------------------------------------------------------------ #
    # Deterministic reductions
    # ------------------------------------------------------------------ #

    def allreduce(self, tensors: list[torch.Tensor], opts: object) -> Work:
        reduce_op = _reduce_op_of(opts)
        if reduce_op == dist.ReduceOp.MAX or reduce_op == dist.ReduceOp.MIN:
            return self._inner.allreduce(tensors, opts)

        for tensor in tensors:
            det_all_reduce(
                tensor,
                world_size=self.size(),
                gather_fn=lambda output, input: self._inner._allgather_base(output, input, AllgatherOptions()).wait(),
            )
            if reduce_op == dist.ReduceOp.AVG:
                tensor.div_(self.size())
        return _CompletedWork()

    def allreduce_coalesced(self, tensors: list[torch.Tensor], opts: object) -> Work:
        return self.allreduce(tensors, opts)

    def _reduce_scatter_base(self, output: torch.Tensor, input: torch.Tensor, opts: object) -> Work:
        reduce_op = _reduce_op_of(opts)
        if reduce_op == dist.ReduceOp.MAX or reduce_op == dist.ReduceOp.MIN:
            return self._inner._reduce_scatter_base(output, input, opts)

        flat = input.contiguous().view(-1)
        out_flat = output.view(-1) if output.is_contiguous() else torch.empty_like(output).view(-1)
        _det_chunked_fold(
            flat,
            out_flat,
            out_offset=self.rank() * output.numel(),
            world_size=self.size(),
            gather_fn=lambda output, input: self._inner._allgather_base(output, input, AllgatherOptions()).wait(),
        )
        if not output.is_contiguous():
            output.copy_(out_flat.view(output.shape))
        if reduce_op == dist.ReduceOp.AVG:
            output.div_(self.size())
        return _CompletedWork()

    def reduce_scatter(
        self, output_tensors: list[torch.Tensor], input_tensors: list[list[torch.Tensor]], opts: object
    ) -> Work:
        for output, inputs in zip(output_tensors, input_tensors, strict=True):
            self._reduce_scatter_base(output, torch.cat([t.contiguous().view(-1) for t in inputs]), opts)
        return _CompletedWork()

    # ------------------------------------------------------------------ #
    # Plain delegation
    # ------------------------------------------------------------------ #

    def allgather(
        self, output_tensors: list[list[torch.Tensor]], input_tensors: list[torch.Tensor], opts: object
    ) -> Work:
        return self._inner.allgather(output_tensors, input_tensors, opts)

    def allgather_into_tensor_coalesced(
        self, output_tensors: list[torch.Tensor], input_tensors: list[torch.Tensor], opts: object = None
    ) -> Work:
        # The coalescing manager's flush passes no opts; inner lacks the coalesced form.
        effective_opts = opts if opts is not None else AllgatherOptions()
        for output, input in zip(output_tensors, input_tensors, strict=True):
            self._inner._allgather_base(output, input, effective_opts).wait()
        return _CompletedWork()

    def _allgather_base(self, output: torch.Tensor, input: torch.Tensor, opts: object) -> Work:
        return self._inner._allgather_base(output, input, opts)

    def barrier(self, opts: object) -> Work:
        return self._inner.barrier(opts)

    def broadcast(self, tensor_list: list[torch.Tensor], opts: object) -> Work:
        return self._inner.broadcast(tensor_list, opts)

    def reduce(self, tensors: list[torch.Tensor], opts: object) -> Work:
        reduce_op = _reduce_op_of(opts)
        if reduce_op == dist.ReduceOp.MAX or reduce_op == dist.ReduceOp.MIN:
            return self._inner.reduce(tensors, opts)
        # SUM/AVG reduce is order-sensitive too; fold on every rank (the root gets
        # the required result, non-root buffers are unspecified by the contract).
        return self.allreduce(tensors, opts)

    def reduce_scatter_tensor_coalesced(
        self, output_tensors: list[torch.Tensor], input_tensors: list[torch.Tensor], opts: object
    ) -> Work:
        for output, input in zip(output_tensors, input_tensors, strict=True):
            self._reduce_scatter_base(output, input, opts)
        return _CompletedWork()

    def alltoall_base(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        output_split_sizes: list[int],
        input_split_sizes: list[int],
        opts: object,
    ) -> Work:
        return self._inner.alltoall_base(output_tensor, input_tensor, output_split_sizes, input_split_sizes, opts)

    def send(self, tensors: list[torch.Tensor], dst_rank: int, tag: int) -> Work:
        return self._inner.send(tensors, dst_rank, tag)

    def recv(self, tensors: list[torch.Tensor], src_rank: int, tag: int) -> Work:
        return self._inner.recv(tensors, src_rank, tag)

    def _start_coalescing(self, device: torch.device) -> None:
        # Ops queue at the Python level and flush via the *_coalesced methods.
        return None

    def _end_coalescing(self, device: torch.device) -> Work:
        return _CompletedWork()

    def getBackendName(self) -> str:
        return "det_nccl"


def det_all_reduce(
    tensor: torch.Tensor, *, world_size: int, gather_fn: Callable[[torch.Tensor, torch.Tensor], None]
) -> None:
    """SUM ``tensor`` across ranks in-place with the fixed fold; the all-gather is injected.

    ``gather_fn(output, input)`` follows the ``_allgather_base`` calling convention: fill
    the flat ``output`` (``world_size * input.numel()``) with every rank's ``input``.
    Pure data movement -- the local fold defines the (shared) summation order.
    """
    if not tensor.is_contiguous():
        work = tensor.contiguous()
        det_all_reduce(work, world_size=world_size, gather_fn=gather_fn)
        tensor.copy_(work)
        return

    flat = tensor.view(-1)
    _det_chunked_fold(flat, flat, out_offset=0, world_size=world_size, gather_fn=gather_fn)


def _det_chunked_fold(
    flat_input: torch.Tensor,
    out_flat: torch.Tensor,
    *,
    out_offset: int,
    world_size: int,
    gather_fn: Callable[[torch.Tensor, torch.Tensor], None],
) -> None:
    """Fold ``flat_input`` across ranks chunk by chunk; write the summed elements
    covering ``[out_offset, out_offset + out_flat.numel())`` into ``out_flat``.

    All ranks gather the same input range per chunk (a collective requirement), and
    each keeps only the part overlapping its output window -- allreduce passes the
    full window, reduce-scatter its shard. Chunk boundaries cannot change bits:
    every output element is folded over the same operands in the same order.
    ``out_flat`` may alias ``flat_input`` (allreduce, Megatron's aliased shards):
    writes land only in the already-gathered chunk, never ahead of a read.
    """
    total = flat_input.numel()
    chunk_numel = max(1, min(total, _GATHER_BUFFER_CAP_BYTES // (world_size * flat_input.element_size())))
    gather_buf = torch.empty(world_size * chunk_numel, dtype=flat_input.dtype, device=flat_input.device)
    out_end = out_offset + out_flat.numel()

    for start in range(0, total, chunk_numel):
        count = min(chunk_numel, total - start)
        lo = max(start, out_offset)
        hi = min(start + count, out_end)
        buf = gather_buf[: world_size * count]
        gather_fn(buf, flat_input[start : start + count])
        if lo < hi:
            folded = _fold_gathered_sum(list(buf.view(world_size, count).unbind(dim=0)))
            out_flat[lo - out_offset : hi - out_offset].copy_(folded[lo - start : hi - start])


def _reduce_op_of(opts: object) -> object:
    """Extract the ReduceOp from an options object (or pass a bare ReduceOp through)."""
    return opts.reduceOp if hasattr(opts, "reduceOp") else opts


class _CompletedWork(Work):
    """Work handle for an operation that already completed synchronously."""

    def wait(self, timeout: object = None) -> bool:
        return True

    def get_future(self) -> torch.futures.Future:
        future: torch.futures.Future = torch.futures.Future()
        future.set_result(None)
        return future


def _fold_gathered_sum(gathered: list[torch.Tensor]) -> torch.Tensor:
    """Sum a per-rank gathered list in a fixed order (pairwise tree for power-of-two).

    May reuse (mutate) the gathered buffers as accumulators.
    """
    world_size = len(gathered)
    if world_size > 0 and (world_size & (world_size - 1)) == 0:
        partials = gathered
        while len(partials) > 1:
            partials = [partials[i] + partials[i + 1] for i in range(0, len(partials), 2)]
        return partials[0]

    running = gathered[0]
    for index in range(1, world_size):
        running += gathered[index]
    return running
