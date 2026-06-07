"""Reproduce the FT abort hang with a GPU-wedged peer (NOT a sleep-wedged one).

NOT a CI test (needs a Ray GPU cluster). Mirrors the causal chain proven by
py-spy --native on the real pp2 with_failure run (see agent-context
2026-06-07-abort-hang-bisection-report):

    D (W's cell sibling) crashes -> W wedges inside a REAL native NCCL
    collective (its kernel spins on the GPU forever; this is cell1_rank1's
    state, stuck in cudaStreamSynchronize) -> the survivor S has an in-flight
    cross-cell allreduce to W -> S calls errored() (bug A: full-device cuda
    synchronize, indep_dp.py:115) or abort() (bug B: c10d abort ->
    waitForFutureOrTimeout, indep_dp.py:77) BEFORE the 120s userspace timer.

Every earlier attempt used a sleep-wedged peer: process stuck but GPU idle and
comms drained, so abort returned in <1s. The differentiator under test is the
peer's GPU state.

Scenarios (fresh actors each; env mirrors the real run:
CUDA_DEVICE_MAX_CONNECTIONS=1, NCCL_NVLS_ENABLE=1, NCCL_ALGO=Ring):
    wedged_abort    S aborts the cross PG while W is GPU-wedged and S's cross
                    allreduce is in-flight. EXPECT (claim): abort blocks; then
                    killing W unblocks it (the dead-peer law).
    wedged_errored  Same setup but S calls cross_pg.errored() instead: its
                    internal synchronize must block on S's own in-flight kernel
                    (bug A's shape). EXPECT: blocks; unblocked by killing W.
    sleep_control   Old (false-negative) shape: W sleep-wedged, GPU idle.
                    EXPECT: abort returns fast -- proving the GPU wedge is the
                    differentiator.

Usage:
    python tests/e2e/external/torchft_wedged_peer_abort_experiment.py
    python tests/e2e/external/torchft_wedged_peer_abort_experiment.py --only wedged_abort
"""

import logging
import os
import time
from datetime import timedelta
from typing import Annotated

import ray
import typer

logger = logging.getLogger(__name__)

_VERDICT_S = 75.0  # < cross timeout (120s) so the userspace timer cannot self-heal first
_SETTLE_S = 3.0
_REAL_RUN_ENV = {
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "NCCL_NVLS_ENABLE": "1",
    "NCCL_ALGO": "Ring",
    "NCCL_DEBUG": "WARN",
}


@ray.remote(num_gpus=1, max_concurrency=4)
class _Worker:
    """One rank. max_concurrency=4 so probes run while another call blocks forever."""

    def init(self, *, name: str) -> dict:
        import torch

        self._name = name
        self._device = torch.device(f"cuda:{torch.cuda.current_device()}")
        return {"name": name, "device": str(self._device)}

    def build_cross_pg(self, *, store_addr: str, rank: int, timeout_s: float) -> dict:
        """torchft cross-cell NCCL PG (S<->W), warmed up. Mirrors create_indep_dp_group."""
        import torch
        import torch.distributed as dist
        from torchft.process_group import ProcessGroupNCCL

        self._cross = ProcessGroupNCCL(timeout=timedelta(seconds=timeout_s))
        self._cross.configure(store_addr=store_addr, replica_id=self._name, rank=rank, world_size=2, quorum_id=0)

        opts = dist.AllreduceOptions()
        opts.reduceOp = dist.ReduceOp.SUM
        t = torch.ones(8, device=self._device)
        assert self._cross.allreduce([t], opts).wait(), f"{self._name}: cross warmup failed"
        return {"name": self._name, "use_abort": self._cross._use_abort}

    def build_native_cell_pg(self, *, store_addr: str, rank: int, long_timeout_s: float = 3600.0) -> dict:
        """RAW c10d NCCL PG for the W<->D 'cell' (no torchft timer): the wedge medium."""
        import torch
        from torch.distributed import PrefixStore, TCPStore
        from torch.distributed import ProcessGroupNCCL as BaseProcessGroupNCCL

        host, port = store_addr.split(":")
        store = TCPStore(host_name=host, port=int(port), is_master=False)
        prefixed = PrefixStore("wedged_peer_cell", store)

        opts = BaseProcessGroupNCCL.Options()
        opts._timeout = timedelta(seconds=long_timeout_s)
        self._cell = BaseProcessGroupNCCL(prefixed, rank, 2, opts)

        t = torch.ones(8, device=self._device)
        self._cell.allreduce([t]).wait()
        return {"name": self._name, "cell_warmup": float(t[0].item())}

    def die(self) -> None:
        os._exit(1)

    def wedge_sleep(self) -> None:
        time.sleep(1_000_000)

    def wedge_native_collective(self) -> None:
        """Block forever inside a real cell collective whose peer (D) is dead:
        a spinning NCCL kernel on this GPU -- cell1_rank1's wedge state."""
        import torch

        t = torch.ones(1 << 20, device=self._device)
        self._cell.allreduce([t]).wait()  # D never joins; raw c10d, huge timeout

    def enqueue_inflight_cross(self, *, numel: int) -> dict:
        """S posts the cross allreduce W never joins (the in-flight grad allreduce
        at crash time). wait() is non-blocking for the nonblocking comm; do NOT
        touch errored() here -- that is bug A, measured separately."""
        import torch
        import torch.distributed as dist

        t = torch.ones(numel, device=self._device)
        opts = dist.AllreduceOptions()
        opts.reduceOp = dist.ReduceOp.SUM
        start = time.monotonic()
        work = self._cross.allreduce([t], opts)
        ok = work.wait()
        self._inflight_tensor = t
        return {"name": self._name, "wait_ok": bool(ok), "wait_s": round(time.monotonic() - start, 3)}

    def abort_cross(self) -> dict:
        """Bug B shape: reconfigure_indep_dp_group's g.abort(errored=False)."""
        start = time.monotonic()
        self._cross.abort(errored=False)
        return {"name": self._name, "abort_s": round(time.monotonic() - start, 2)}

    def errored_cross(self) -> dict:
        """Bug A shape: _allreduce_grads_across_replicas' pg.errored() (internally
        a full torch.cuda.synchronize())."""
        start = time.monotonic()
        e = self._cross.errored()
        return {"name": self._name, "errored_s": round(time.monotonic() - start, 2), "errored": str(e)[:120]}

    def cuda_probe(self) -> dict:
        """Second-thread torch.cuda.synchronize() timing: GPU-wedged vs idle."""
        import torch

        start = time.monotonic()
        torch.cuda.synchronize()
        return {"name": self._name, "synchronize_s": round(time.monotonic() - start, 3)}

    def ping(self) -> str:
        return f"{self._name} alive at {time.monotonic():.1f}"


def _spawn(name: str) -> object:
    worker = _Worker.options(runtime_env={"env_vars": dict(_REAL_RUN_ENV)}).remote()
    ray.get(worker.init.remote(name=name), timeout=60)
    return worker


def _confirm_gpu_wedged(worker: object, *, who: str) -> str:
    try:
        probe = ray.get(worker.cuda_probe.remote(), timeout=10)
        return f"{who} GPU NOT wedged (synchronize returned: {probe})"
    except ray.exceptions.GetTimeoutError:
        return f"{who} GPU WEDGED (synchronize blocked >10s) -- faithful to cell1_rank1"


def _kill_then_await(ref: object, *, victim: object, what: str) -> str:
    """The dead-peer law check: kill the wedged peer, see if the blocked call returns."""
    ray.kill(victim, no_restart=True)
    start = time.monotonic()
    try:
        out = ray.get(ref, timeout=120)
        return f"UNBLOCKED {round(time.monotonic() - start, 1)}s after killing W: {out}"
    except ray.exceptions.GetTimeoutError:
        return f"STILL STUCK 120s after killing W ({what})"
    except Exception as e:  # noqa: BLE001 - blocked call may surface the abort as an error
        return f"UNBLOCKED-with-exception {round(time.monotonic() - start, 1)}s after killing W: {type(e).__name__}: {str(e)[:200]}"


def _run(exp: str, *, store_base: str, cell_store_addr: str, timeout_s: float, numel: int) -> str:
    print(f"== experiment {exp} ==")
    surv = _spawn("S")
    wedged = _spawn("W")
    sibling = _spawn("D")

    try:
        # Cross pair (S rank0, W rank1) + W<->D native cell PG, all warmed up.
        cross_store = f"{store_base}/{exp}/cross"
        ray.get(
            [
                surv.build_cross_pg.remote(store_addr=cross_store, rank=0, timeout_s=timeout_s),
                wedged.build_cross_pg.remote(store_addr=cross_store, rank=1, timeout_s=timeout_s),
            ],
            timeout=120,
        )
        ray.get(
            [
                wedged.build_native_cell_pg.remote(store_addr=cell_store_addr, rank=0),
                sibling.build_native_cell_pg.remote(store_addr=cell_store_addr, rank=1),
            ],
            timeout=120,
        )
        print("  cross pair + native cell PG built and warmed")

        # The real failure shape: D dies, then W wedges in a real cell collective.
        sibling.die.remote()
        time.sleep(_SETTLE_S)
        if exp == "sleep_control":
            wedged.wedge_sleep.remote()
            time.sleep(_SETTLE_S)
            print(f"  {_confirm_gpu_wedged(wedged, who='W')} (control expects NOT wedged)")
        else:
            wedged.wedge_native_collective.remote()
            time.sleep(_SETTLE_S)
            print(f"  {_confirm_gpu_wedged(wedged, who='W')}")

        inflight = ray.get(surv.enqueue_inflight_cross.remote(numel=numel), timeout=30)
        print(f"  S in-flight cross allreduce (numel={numel}): {inflight}")

        target = surv.errored_cross if exp == "wedged_errored" else surv.abort_cross
        what = "errored()" if exp == "wedged_errored" else "abort()"
        ref = target.remote()
        try:
            out = ray.get(ref, timeout=_VERDICT_S)
            return f"{what} returned in <{_VERDICT_S}s: {out} (NOT reproduced)"
        except ray.exceptions.GetTimeoutError:
            print(f"  *** {what} BLOCKED >{_VERDICT_S}s (REPRODUCED the hang) *** now testing the dead-peer law")
            return f"*** {what} BLOCKED >{_VERDICT_S}s (REPRODUCED) *** then: {_kill_then_await(ref, victim=wedged, what=what)}"
    finally:
        for w in [surv, wedged, sibling]:
            try:
                ray.kill(w, no_restart=True)
            except Exception:  # noqa: BLE001 - idempotent cleanup
                pass


def main(
    timeout_s: Annotated[float, typer.Option(help="torchft cross PG timeout (miles uses 120)")] = 120.0,
    numel: Annotated[int, typer.Option(help="in-flight cross allreduce size (grad bucket scale)")] = 1 << 22,
    only: Annotated[str | None, typer.Option(help="run a single experiment")] = None,
) -> None:
    """Run the wedged-peer abort reproduction matrix; one RESULT line per experiment."""
    ray.init(ignore_reinit_error=True)

    from torch.distributed import TCPStore

    store = TCPStore(host_name="localhost", port=0, is_master=True, wait_for_workers=False)
    store_base = f"localhost:{store.port}/wedged_peer"

    experiments = ["wedged_abort", "wedged_errored", "sleep_control"]
    if only is not None:
        experiments = [only]

    results: dict[str, str] = {}
    for i, exp in enumerate(experiments):
        cell_store = TCPStore(host_name="localhost", port=0, is_master=True, wait_for_workers=False)
        results[exp] = _run(
            exp,
            store_base=store_base,
            cell_store_addr=f"localhost:{cell_store.port}",
            timeout_s=timeout_s,
            numel=numel,
        )
        print(f"RESULT {exp}: {results[exp]}\n")
        del cell_store
        time.sleep(3)

    print("==== SUMMARY ====")
    for exp, res in results.items():
        print(f"RESULT {exp}: {res}")
    del store


if __name__ == "__main__":
    typer.run(main)
