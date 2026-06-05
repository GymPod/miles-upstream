"""Experiment: do pinned NCCL flags give a bitwise-identical reduction order across comms?

NOT a CI test (needs a Ray cluster with >= world-size GPUs; intentionally not ``test_*.py``).

Context: the deterministic FT test wants baseline (normal DP, Megatron -> c10d NCCL) and
target (indep_dp, torchft -> c10d NCCL) gradients to match bitwise. Both stacks sum the
same per-rank gradients; the only degree of freedom is the floating-point summation
order, which for >= 3 operands is decided by NCCL internals. Hypothesis under test: NCCL
ring allreduce order is a pure function of (participants, ring order, buffer size,
channels, protocol), so pinning

    NCCL_ALGO=Ring  NCCL_PROTO=Simple  NCCL_MIN_NCHANNELS=NCCL_MAX_NCHANNELS=1
    NCCL_NVLS_ENABLE=0

makes every NCCL reduction over the same ranks produce identical bits — across separate
communicator instances, across c10d vs torchft, and across a torchft quorum rebuild
(the FT recovery path). If true, deterministic-FT bitwise needs NO custom tree
reduction: plain Megatron + plain torchft + these env vars suffice.

Each rank reduces the SAME input through several paths and the driver compares bitwise:

    c10d_default        all_reduce on the default c10d PG            (Megatron baseline analog)
    c10d_newgroup       all_reduce on a fresh dist.new_group         (cross-comm-instance, same lib)
    torchft_q0          all_reduce on a torchft ProcessGroupNCCL     (indep_dp analog)
    torchft_q1          all_reduce after shutdown + reconfigure      (FT recovery rebuild analog)
    c10d_reduce_scatter reduce_scatter_tensor + all_gather           (Megatron distributed-optimizer analog)
    fold_ascending      all_gather + local g0+g1+g2+g3               (reference fixed order)
    fold_tree           all_gather + local (g0+g1)+(g2+g3)           (reference fixed tree)

Inputs are order-sensitive: a catastrophic-cancellation block (per-rank +-0.5 values
whose cross-rank sum is ~1e-4 — the starved-MoE-expert gradient regime where the FT
comparison actually fails) plus mixed-magnitude random noise.

Verdict reading:
  - c10d_default == c10d_newgroup == torchft_q0 == torchft_q1
        -> pinned flags make NCCL order comm-instance-invariant: the flag-only approach
           works; no tree reduction needed.
  - additionally == c10d_reduce_scatter
        -> even the distributed-optimizer path aligns; no test-config change needed.
  - any mismatch among the NCCL paths
        -> NCCL order is not comm-invariant even when pinned; fall back to a fixed
           tree/fold reduction on both sides.

Usage:
    python tests/e2e/external/nccl_reduction_order_reference.py
    python tests/e2e/external/nccl_reduction_order_reference.py --no-pin-env   # control run
    python tests/e2e/external/nccl_reduction_order_reference.py --world-size 2 --numel 1000000
"""

import hashlib
import logging
import os
from datetime import timedelta
from typing import Annotated

import ray
import typer

logger = logging.getLogger(__name__)

PINNED_NCCL_ENV: dict[str, str] = {
    "NCCL_ALGO": "Ring",
    "NCCL_PROTO": "Simple",
    "NCCL_MIN_NCHANNELS": "1",
    "NCCL_MAX_NCHANNELS": "1",
    "NCCL_NVLS_ENABLE": "0",
}

# First _CANCEL_NUMEL elements use the catastrophic-cancellation pattern.
_CANCEL_NUMEL: int = 1_000_000


@ray.remote(num_gpus=1)
class _ReduceWorker:
    """Ray actor (1 GPU) that reduces the same input through several NCCL paths."""

    def setup(
        self,
        *,
        rank: int,
        world_size: int,
        store_host: str,
        store_port: int,
        torchft_store_addr: str,
        seed: int,
        numel: int,
        timeout_s: float,
        pin_env: bool,
    ) -> dict:
        # Must happen before the first NCCL communicator is created in this process.
        if pin_env:
            os.environ.update(PINNED_NCCL_ENV)

        import torch
        import torch.distributed as dist
        from torch.distributed import TCPStore

        self._rank = rank
        self._world_size = world_size
        self._timeout_s = timeout_s
        self._torchft_store_addr = torchft_store_addr
        self._device = torch.device(f"cuda:{torch.cuda.current_device()}")
        self._results: dict[str, torch.Tensor] = {}

        store = TCPStore(host_name=store_host, port=store_port, is_master=False, wait_for_workers=False)
        dist.init_process_group(
            backend="nccl",
            store=dist.PrefixStore("native", store),
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=timeout_s),
        )

        self._input = self._build_input(seed=seed, numel=numel)

        return {
            "rank": rank,
            "nccl_version": torch.cuda.nccl.version(),
            "device": str(self._device),
            "pinned_env": {k: os.environ.get(k) for k in PINNED_NCCL_ENV},
        }

    def _build_input(self, *, seed: int, numel: int):
        """Order-sensitive input: cancellation block + mixed-magnitude noise.

        Cancellation block (first _CANCEL_NUMEL elems): rank r holds
        ``(-1)^r * 0.5 * shared + 1e-4 * own`` so cross-rank sums cancel +-0.5 operands
        down to ~1e-4 — the rounding-sensitive regime of starved-MoE-expert grads.
        """
        import torch

        cancel_numel = min(_CANCEL_NUMEL, numel)
        shared_gen = torch.Generator().manual_seed(seed)  # same on every rank
        own_gen = torch.Generator().manual_seed(seed + 1 + self._rank)

        shared = torch.randn(cancel_numel, generator=shared_gen, dtype=torch.float32)
        own = torch.randn(cancel_numel, generator=own_gen, dtype=torch.float32)
        sign = -1.0 if self._rank % 2 else 1.0
        cancel_block = sign * 0.5 * shared + 1e-4 * own

        noise_numel = numel - cancel_numel
        base = torch.randn(noise_numel, generator=own_gen, dtype=torch.float32)
        exponent = torch.randint(-6, 1, (noise_numel,), generator=own_gen).float()
        noise_block = base * torch.pow(10.0, exponent)

        return torch.cat([cancel_block, noise_block]).to(self._device)

    def run_c10d_default(self) -> None:
        import torch.distributed as dist

        x = self._input.clone()
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        self._results["c10d_default"] = x

    def run_c10d_newgroup(self) -> None:
        """A second, freshly created c10d communicator over the same ranks."""
        import torch.distributed as dist

        group = dist.new_group(ranks=list(range(self._world_size)), backend="nccl")
        x = self._input.clone()
        dist.all_reduce(x, op=dist.ReduceOp.SUM, group=group)
        self._results["c10d_newgroup"] = x

    def run_torchft_q0(self) -> None:
        import torch.distributed as dist
        from torchft.process_group import ProcessGroupNCCL

        self._torchft_pg = ProcessGroupNCCL(timeout=timedelta(seconds=self._timeout_s))
        self._torchft_pg.configure(
            store_addr=f"{self._torchft_store_addr}/q0",
            replica_id=str(self._rank),
            rank=self._rank,
            world_size=self._world_size,
            quorum_id=0,
        )
        x = self._input.clone()
        opts = dist.AllreduceOptions()
        opts.reduceOp = dist.ReduceOp.SUM
        assert self._torchft_pg.allreduce([x], opts).wait()
        self._results["torchft_q0"] = x

    def run_torchft_q1_rebuilt(self) -> None:
        """Shutdown the q0 PG and build a fresh one (the FT recovery rebuild analog)."""
        import torch.distributed as dist
        from torchft.process_group import ProcessGroupNCCL

        self._torchft_pg.shutdown()
        self._torchft_pg = ProcessGroupNCCL(timeout=timedelta(seconds=self._timeout_s))
        self._torchft_pg.configure(
            store_addr=f"{self._torchft_store_addr}/q1",
            replica_id=str(self._rank),
            rank=self._rank,
            world_size=self._world_size,
            quorum_id=1,
        )
        x = self._input.clone()
        opts = dist.AllreduceOptions()
        opts.reduceOp = dist.ReduceOp.SUM
        assert self._torchft_pg.allreduce([x], opts).wait()
        self._results["torchft_q1"] = x

    def run_c10d_reduce_scatter(self) -> None:
        """reduce_scatter + all_gather: the Megatron distributed-optimizer analog."""
        import torch
        import torch.distributed as dist

        x = self._input.clone()
        shard = torch.empty(x.numel() // self._world_size, device=self._device, dtype=x.dtype)
        dist.reduce_scatter_tensor(shard, x, op=dist.ReduceOp.SUM)
        full = torch.empty_like(x)
        dist.all_gather_into_tensor(full, shard)
        self._results["c10d_reduce_scatter"] = full

    def run_reference_folds(self) -> None:
        """Exact local reductions in two fixed orders (order ground truths)."""
        import torch
        import torch.distributed as dist

        gathered = [torch.empty_like(self._input) for _ in range(self._world_size)]
        dist.all_gather(gathered, self._input)

        ascending = gathered[0].clone()
        for other in gathered[1:]:
            ascending += other
        self._results["fold_ascending"] = ascending

        partials = list(gathered)
        while len(partials) > 1:
            partials = [partials[i] + partials[i + 1] for i in range(0, len(partials), 2)]
        self._results["fold_tree"] = partials[0]

    def checksums(self) -> dict[str, str]:
        """Per-path sha256 of the result bytes (for cross-rank + cross-path comparison)."""
        return {
            name: hashlib.sha256(tensor.cpu().numpy().tobytes()).hexdigest()[:16]
            for name, tensor in self._results.items()
        }

    def pairwise_mismatch(self) -> dict[str, dict]:
        """For every path pair: #bitwise-mismatching elements and max abs diff."""
        out: dict[str, dict] = {}
        names = sorted(self._results)
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                ta, tb = self._results[a], self._results[b]
                mismatch = int((ta != tb).sum().item())
                max_abs = float((ta - tb).abs().max().item()) if mismatch else 0.0
                out[f"{a} vs {b}"] = {"mismatch_elems": mismatch, "max_abs_diff": max_abs}
        return out


def _run_experiment(*, world_size: int, numel: int, seed: int, timeout_s: float, pin_env: bool) -> None:
    from torch.distributed import TCPStore

    store = TCPStore(host_name="localhost", port=0, is_master=True, wait_for_workers=False)
    torchft_store_addr = f"localhost:{store.port}/nccl_order"

    workers = [_ReduceWorker.remote() for _ in range(world_size)]
    init = ray.get(
        [
            w.setup.remote(
                rank=i,
                world_size=world_size,
                store_host="localhost",
                store_port=store.port,
                torchft_store_addr=torchft_store_addr,
                seed=seed,
                numel=numel,
                timeout_s=timeout_s,
                pin_env=pin_env,
            )
            for i, w in enumerate(workers)
        ]
    )
    print(f"setup: {init}")

    for method in (
        "run_c10d_default",
        "run_c10d_newgroup",
        "run_torchft_q0",
        "run_torchft_q1_rebuilt",
        "run_c10d_reduce_scatter",
        "run_reference_folds",
    ):
        ray.get([getattr(w, method).remote() for w in workers])
        print(f"done: {method}")

    # Sanity: every rank must hold identical bits for each path (else the path itself
    # is broken, not just order-divergent).
    checksums = ray.get([w.checksums.remote() for w in workers])
    for name in checksums[0]:
        per_rank = [c[name] for c in checksums]
        consistent = len(set(per_rank)) == 1
        print(f"cross-rank consistency [{name}]: {'OK' if consistent else 'BROKEN ' + str(per_rank)}")
        assert consistent, f"path {name} disagrees across ranks: {per_rank}"

    print(f"\n{'=' * 78}\n  PAIRWISE BITWISE COMPARISON (rank 0)  pin_env={pin_env}\n{'=' * 78}")
    pairwise = ray.get(workers[0].pairwise_mismatch.remote())
    for pair, stats in pairwise.items():
        equal = stats["mismatch_elems"] == 0
        mark = (
            "BITWISE-EQUAL"
            if equal
            else f"DIFF ({stats['mismatch_elems']} elems, max_abs={stats['max_abs_diff']:.3e})"
        )
        print(f"  {pair:<42} {mark}")

    nccl_paths = ["c10d_default", "c10d_newgroup", "torchft_q0", "torchft_q1"]
    cross_comm_equal = all(pairwise[f"{a} vs {b}"]["mismatch_elems"] == 0 for a, b in zip(nccl_paths, nccl_paths[1:]))
    rs_equal = pairwise["c10d_default vs c10d_reduce_scatter"]["mismatch_elems"] == 0
    print(f"\n  VERDICT: allreduce comm-instance-invariant (c10d/newgroup/torchft/rebuilt) = {cross_comm_equal}")
    print(f"  VERDICT: reduce_scatter+gather matches allreduce                          = {rs_equal}")

    for w in workers:
        try:
            ray.kill(w, no_restart=True)
        except Exception:
            pass
    del store


def main(
    world_size: Annotated[int, typer.Option(help="Ranks / GPUs (the FT case is 4)")] = 4,
    numel: Annotated[int, typer.Option(help="Elements per tensor (fp32)")] = 32_000_000,
    seed: Annotated[int, typer.Option(help="Input seed")] = 42,
    timeout_s: Annotated[float, typer.Option(help="PG timeout")] = 60.0,
    pin_env: Annotated[
        bool, typer.Option(help="Pin NCCL_ALGO=Ring/PROTO=Simple/1-channel/NVLS-off (off = control run)")
    ] = True,
) -> None:
    """Run the NCCL reduction-order experiment (prints verdicts; no pass/fail assert)."""
    assert numel % world_size == 0, "numel must be divisible by world_size (reduce_scatter)"
    ray.init(ignore_reinit_error=True)
    _run_experiment(world_size=world_size, numel=numel, seed=seed, timeout_s=timeout_s, pin_env=pin_env)


if __name__ == "__main__":
    typer.run(main)
