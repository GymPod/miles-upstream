"""Process group with bitwise-deterministic SUM reductions.

``DetProcessGroup`` wraps an inner c10d NCCL group. Every collective delegates to
the inner group except the
SUM/AVG reductions — ``allreduce`` and ``reduce_scatter`` — which are computed as
all-gather (pure data movement, no arithmetic) plus a fixed local fold: a pairwise
tree for power-of-two world sizes, an ascending-rank fold otherwise. The summation
order is therefore independent of the NCCL version, topology, communicator
instance, or buffer layout, and reduce-scatter takes its shard from the same full
fold, so reduce-scatter and all-reduce agree bitwise by construction.

Debug/test use only: the fold trades bandwidth and synchrony for determinism.
"""

import logging

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup as BaseProcessGroup
from torch.distributed import Work
from torch.distributed.distributed_c10d import AllgatherOptions

logger = logging.getLogger(__name__)


_BACKEND_NAME = "det_nccl"
_backend_registered = False


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


def _reduce_op_of(opts: object) -> object:
    """Extract the ReduceOp from an options object (or pass a bare ReduceOp through)."""
    return opts.reduceOp if hasattr(opts, "reduceOp") else opts


def _is_deterministic_op(reduce_op: object) -> bool:
    """Whether the op is order-sensitive and folded deterministically (SUM/AVG).

    Compared with explicit ``==``: ``ReduceOp.__eq__`` handles the RedOpType enum,
    but tuple containment (``in``) does not, so ``op in (SUM, AVG)`` is always False.
    """
    return reduce_op == dist.ReduceOp.SUM or reduce_op == dist.ReduceOp.AVG


class DetProcessGroup(BaseProcessGroup):
    """Wrapper process group whose SUM/AVG reductions use a fixed-order fold."""

    def __init__(self, inner: dist.ProcessGroup) -> None:
        super().__init__(inner.rank(), inner.size())
        self._inner = inner

    # ------------------------------------------------------------------ #
    # Deterministic reductions
    # ------------------------------------------------------------------ #

    def allreduce(self, tensors: list[torch.Tensor], opts: object) -> Work:
        reduce_op = _reduce_op_of(opts)
        if not _is_deterministic_op(reduce_op):
            # MAX/MIN and friends are exactly associative-commutative: order cannot
            # change the bits, so the native implementation is already deterministic.
            return self._inner.allreduce(tensors, opts)

        for tensor in tensors:
            work = tensor if tensor.is_contiguous() else tensor.contiguous()
            flat = work.view(-1)
            flat.copy_(self._det_full_sum(flat))
            if reduce_op == dist.ReduceOp.AVG:
                flat.div_(self.size())
            if work is not tensor:
                tensor.copy_(work)
        return _CompletedWork()

    def allreduce_coalesced(self, tensors: list[torch.Tensor], opts: object) -> Work:
        return self.allreduce(tensors, opts)

    def _reduce_scatter_base(self, output: torch.Tensor, input: torch.Tensor, opts: object) -> Work:
        reduce_op = _reduce_op_of(opts)
        if not _is_deterministic_op(reduce_op):
            return self._inner._reduce_scatter_base(output, input, opts)

        flat = input.contiguous().view(-1)
        folded = self._det_full_sum(flat)
        shard_numel = output.numel()
        shard = folded[self.rank() * shard_numel : (self.rank() + 1) * shard_numel]
        output.copy_(shard.view(output.shape))
        if reduce_op == dist.ReduceOp.AVG:
            output.div_(self.size())
        return _CompletedWork()

    def reduce_scatter(
        self, output_tensors: list[torch.Tensor], input_tensors: list[list[torch.Tensor]], opts: object
    ) -> Work:
        reduce_op = _reduce_op_of(opts)
        if not _is_deterministic_op(reduce_op):
            return self._inner.reduce_scatter(output_tensors, input_tensors, opts)

        for output, inputs in zip(output_tensors, input_tensors, strict=True):
            # output on rank r = fold over ranks q of inputs_q[r]; fold every slot in
            # the same fixed order and keep this rank's one.
            for slot, slot_input in enumerate(inputs):
                folded = self._det_full_sum(slot_input.contiguous().view(-1))
                if slot == self.rank():
                    output.copy_(folded.view(output.shape))
                    if reduce_op == dist.ReduceOp.AVG:
                        output.div_(self.size())
        return _CompletedWork()

    def _det_full_sum(self, flat: torch.Tensor) -> torch.Tensor:
        """Return the fixed-order cross-rank sum of a contiguous 1-D tensor."""
        world_size = self.size()
        gathered_flat = torch.empty(world_size * flat.numel(), dtype=flat.dtype, device=flat.device)
        self._inner._allgather_base(gathered_flat, flat, AllgatherOptions()).wait()
        return fold_gathered_sum(list(gathered_flat.view(world_size, -1).unbind(dim=0)))

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
        # Called by the coalescing manager's flush (without an opts argument). The
        # inner NCCL backend object has no coalesced allgather; gather pair by pair.
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
        return self._inner.reduce(tensors, opts)

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
        # The coalescing manager queues ops at the Python level and flushes them
        # through the *_coalesced methods above; there is no backend-level
        # coalescing to start for this wrapper.
        return None

    def _end_coalescing(self, device: torch.device) -> Work:
        return _CompletedWork()

    def getBackendName(self) -> str:
        return "det_nccl"


class _CompletedWork(Work):
    """Work handle for an operation that already completed synchronously."""

    def wait(self, timeout: object = None) -> bool:
        return True

    def get_future(self) -> torch.futures.Future:
        future: torch.futures.Future = torch.futures.Future()
        future.set_result(None)
        return future


def fold_gathered_sum(gathered: list[torch.Tensor]) -> torch.Tensor:
    """Sum a per-rank gathered list in a fixed order (pairwise tree for power-of-two).

    Shared with indep_dp's cross-cell reduction so both sides use one bracketing.
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
