"""Reproduce the 卡② fatal hang: graveyard-leaked in-flight kernel poisons the next quorum.

NOT a CI test (needs a Ray GPU cluster). This is the targeted reproduction of the
r19 fatal hang that NONE of the abort-matrix rounds caught, because they all
aborted-then-ended; none of them PARKED a zombie in-flight collective (the
``_retired_pgs`` graveyard in ``reconfigure_indep_dp_group``) AND THEN built a new
quorum and ran a collective on the same survivor GPU.

Precise hypothesis under test (see agent-context 2026-06-07-hang-latest-understanding):
    The survivor's cross-cell allreduce on quorum_0 is enqueued against an
    alive-but-wedged peer (the peer crashed-its-sibling and is stuck in its own
    cell-internal collective, so it never joins the cross allreduce). Its kernel
    sits in-flight on the survivor's CUDA stream. Reconfigure fires (triggered by
    the OTHER pair's dead-peer detection) BEFORE this collective hits its 120s
    userspace timeout, so ``errored() is None`` and the PG is parked in the
    graveyard WITHOUT aborting -> the zombie kernel keeps occupying the stream.
    The new quorum_2 collective (with a fresh, healthy peer) enqueues BEHIND the
    zombie -> never runs -> times out -> the abort cannot drain the device ->
    the 600s hang.

Topology (minimal): one cross-cell pair (survivor S <-> wedged peer W) on quorum_0,
plus a fresh peer N for quorum_2.
    q0 pair:  (W rank0, S rank1)   -- W wedges; S enqueues the zombie cross allreduce
    q2 pair:  (N rank0, S rank1)   -- N is healthy; S must run a collective here

Scenarios:
    park_then_q2     S parks the in-flight q0 PG (graveyard, no abort) then builds
                     q2 with N and runs collective_bool_and's shape. EXPECT (hypothesis):
                     q2 collective HANGS because the zombie q0 kernel occupies the stream.
    abort_then_q2    CONTROL / A-plan: kill W first (peer dies), THEN abort the q0 PG
                     (fast, peer dead), THEN build q2 + collective. EXPECT: no hang.
    shutdown_then_q2 CONTROL: like park_then_q2 but S calls shutdown() on the q0 PG
                     instead of leaking it (reference's reconfigure path). EXPECT: ?

Usage:
    python tests/e2e/external/torchft_graveyard_repro_experiment.py
    python tests/e2e/external/torchft_graveyard_repro_experiment.py --only park_then_q2 --timeout-s 20
"""

import logging
import os
import time
from datetime import timedelta
from typing import Annotated

import ray
import typer

logger = logging.getLogger(__name__)

_HANG_VERDICT_TIMEOUT_S = 60.0
_SETTLE_S = 3.0


@ray.remote(num_gpus=1, max_concurrency=3)
class _Worker:
    """One rank holding torchft cross-cell PGs. max_concurrency=3 so cuda_probe /
    ping can run on a second/third actor thread while a collective blocks."""

    def setup(self, *, store_addr: str, name: str, rank: int, quorum_id: int, timeout_s: float) -> dict:
        """Build ONE torchft NCCL+Gloo cross pair PG (world_size=2) and warm it up."""
        import torch
        import torch.distributed as dist
        from torchft.process_group import ProcessGroupGloo, ProcessGroupNCCL

        self._name = name
        self._timeout_s = timeout_s
        self._device = torch.device(f"cuda:{torch.cuda.current_device()}")
        self._graveyard: list[object] = []

        self._nccl = ProcessGroupNCCL(timeout=timedelta(seconds=timeout_s))
        self._nccl.configure(store_addr=f"{store_addr}/nccl", replica_id=name, rank=rank, world_size=2, quorum_id=quorum_id)
        self._gloo = ProcessGroupGloo(timeout=timedelta(seconds=timeout_s))
        self._gloo.configure(store_addr=f"{store_addr}/gloo", replica_id=name, rank=rank, world_size=2, quorum_id=quorum_id)

        opts = dist.AllreduceOptions()
        opts.reduceOp = dist.ReduceOp.SUM
        t = torch.ones(8, device=self._device)
        assert self._nccl.allreduce([t], opts).wait(), f"{name}: nccl warmup failed"
        tg = torch.ones(8)
        assert self._gloo.allreduce([tg], opts).wait(), f"{name}: gloo warmup failed"
        return {"name": name, "nccl": torch.cuda.nccl.version(), "use_abort": self._nccl._use_abort}

    def init_fresh(self, *, name: str, timeout_s: float) -> dict:
        """Minimal init for a brand-new peer that only ever joins quorum_2 (no q0 PG)."""
        import torch

        self._name = name
        self._timeout_s = timeout_s
        self._device = torch.device(f"cuda:{torch.cuda.current_device()}")
        self._graveyard = []
        return {"name": name, "device": str(self._device)}

    def wedge_sleep(self) -> None:
        """Alive-but-wedged peer: never joins any further collective."""
        time.sleep(1_000_000)

    def die(self) -> None:
        os._exit(1)

    def enqueue_inflight_cross(self) -> dict:
        """Survivor enqueues the cross allreduce (collective_bool_and shape: numel=1
        MIN) against the wedged peer, calls torchft's non-blocking wait() so the
        userspace timer is armed, but does NOT synchronize. The kernel is now
        in-flight on this rank's stream, errored() still None (timer not fired)."""
        import torch
        import torch.distributed as dist

        t = torch.tensor([1.0], dtype=torch.float32, device=self._device)
        opts = dist.AllreduceOptions()
        opts.reduceOp = dist.ReduceOp.MIN
        start = time.monotonic()
        work = self._nccl.allreduce([t], opts)
        ok = work.wait()  # non-blocking / stream-ordered for nonblocking comm
        self._inflight_tensor = t
        return {
            "name": self._name,
            "wait_ok": bool(ok),
            "wait_s": round(time.monotonic() - start, 3),
            "errored": str(self._nccl.errored())[:80],
        }

    def retire_old(self, *, how: str) -> dict:
        """Dispose of the q0 cross PG the way reconfigure_indep_dp_group would.

        how='graveyard' -> append to a kept-alive list, NO abort (miles' wedged-peer
                           path: errored() is None so it is leaked).
        how='abort'      -> abort(errored=False) (miles' errored path / A-plan after
                           the peer is dead).
        how='shutdown'   -> shutdown() both (the reference reconfigure path).
        """
        start = time.monotonic()
        if how == "graveyard":
            self._graveyard.append(self._nccl)
            self._graveyard.append(self._gloo)
        elif how == "abort":
            self._nccl.abort(errored=False)
            self._gloo.abort(errored=False)
        elif how == "shutdown":
            self._nccl.shutdown()
            self._gloo.shutdown()
        else:
            raise ValueError(how)
        return {"name": self._name, "how": how, "retire_s": round(time.monotonic() - start, 3)}

    def build_q2(self, *, store_addr: str, rank: int, timeout_s: float) -> dict:
        """Build a FRESH quorum_2 cross PG (new wrapper, as create_indep_dp_group does)."""
        from torchft.process_group import ProcessGroupGloo, ProcessGroupNCCL

        start = time.monotonic()
        self._nccl_q2 = ProcessGroupNCCL(timeout=timedelta(seconds=timeout_s))
        self._nccl_q2.configure(store_addr=f"{store_addr}/nccl", replica_id=self._name, rank=rank, world_size=2, quorum_id=2)
        self._gloo_q2 = ProcessGroupGloo(timeout=timedelta(seconds=timeout_s))
        self._gloo_q2.configure(store_addr=f"{store_addr}/gloo", replica_id=self._name, rank=rank, world_size=2, quorum_id=2)
        return {"name": self._name, "build_q2_s": round(time.monotonic() - start, 3)}

    def collective_on_q2(self) -> dict:
        """Run collective_bool_and's exact shape on quorum_2 and block on .item().

        If the graveyard zombie kernel occupies this rank's stream, this collective
        enqueues behind it and never runs -> the 600s hang we are hunting."""
        import torch
        import torch.distributed as dist

        t = torch.tensor([1.0], dtype=torch.float32, device=self._device)
        opts = dist.AllreduceOptions()
        opts.reduceOp = dist.ReduceOp.MIN
        start = time.monotonic()
        work = self._nccl_q2.allreduce([t], opts)
        wait_ok = work.wait()
        wait_s = round(time.monotonic() - start, 3)
        value = None
        err = None
        try:
            value = t.item()
        except Exception as e:  # noqa: BLE001 - the expected abort/timeout error
            err = f"{type(e).__name__}: {str(e)[:200]}"
        return {
            "name": self._name,
            "wait_ok": bool(wait_ok),
            "wait_s": wait_s,
            "item_s": round(time.monotonic() - start, 2),
            "value": value,
            "err": err,
            "q2_errored": str(self._nccl_q2.errored())[:80],
        }

    def cuda_probe(self) -> dict:
        """Time torch.cuda.synchronize() on another actor thread: GPU-blocked vs host-blocked."""
        import torch

        start = time.monotonic()
        torch.cuda.synchronize()
        return {"name": self._name, "synchronize_s": round(time.monotonic() - start, 3)}

    def ping(self) -> str:
        return f"{self._name} alive at {time.monotonic():.1f}"


def _peer_setup(peer: object, *, store_addr: str, name: str, rank: int, quorum_id: int, timeout_s: float) -> object:
    return peer.setup.remote(store_addr=store_addr, name=name, rank=rank, quorum_id=quorum_id, timeout_s=timeout_s)


def _run(exp: str, *, store_base: str, timeout_s: float) -> str:
    print(f"== experiment {exp} ==")
    survivor = _Worker.remote()
    wedged = _Worker.remote()

    q0 = f"{store_base}/{exp}/q0"
    # q0 pair: wedged=rank0, survivor=rank1
    ray.get(
        [
            _peer_setup(wedged, store_addr=q0, name="W", rank=0, quorum_id=0, timeout_s=timeout_s),
            _peer_setup(survivor, store_addr=q0, name="S", rank=1, quorum_id=0, timeout_s=timeout_s),
        ],
        timeout=120,
    )
    print("  q0 pair built + warmed")

    # Wedged peer stops responding; survivor enqueues the in-flight cross allreduce.
    wedged.wedge_sleep.remote()
    time.sleep(_SETTLE_S)
    inflight = ray.get(survivor.enqueue_inflight_cross.remote(), timeout=30)
    print(f"  survivor in-flight q0 allreduce: {inflight}")

    # Confirm the zombie kernel is STILL occupying the survivor's stream (it must
    # not have been timer-aborted yet -- that is why we run with timeout_s=120 and
    # rebuild q2 within seconds, exactly like real miles reconfigure-after-crash).
    try:
        zprobe = ray.get(survivor.cuda_probe.remote(), timeout=10)
        print(f"  zombie-stream probe (expect HANG/blocked if kernel in-flight): {zprobe}")
    except ray.exceptions.GetTimeoutError:
        print("  zombie-stream probe: BLOCKED >10s -> zombie kernel IS occupying the stream (good)")

    # Retire q0 the way the chosen path does.
    if exp == "abort_then_q2":
        # A-plan: peer dies first, so the abort can complete fast.
        ray.kill(wedged, no_restart=True)
        time.sleep(_SETTLE_S)
        how = "abort"
    elif exp == "shutdown_then_q2":
        how = "shutdown"
    else:
        how = "graveyard"

    try:
        retire = ray.get(survivor.retire_old.remote(how=how), timeout=_HANG_VERDICT_TIMEOUT_S)
        print(f"  retire: {retire}")
    except ray.exceptions.GetTimeoutError:
        return f"RETIRE ITSELF HUNG ({how}) >{_HANG_VERDICT_TIMEOUT_S}s -- this is 卡① (abort/shutdown on wedged peer)"

    # Build quorum_2 with a FRESH healthy peer N. Both ranks must configure
    # CONCURRENTLY (configure is a 2-rank rendezvous). If the survivor's stream is
    # poisoned by the graveyard zombie, the survivor's build_q2 itself may hang.
    fresh = _Worker.remote()
    ray.get(fresh.init_fresh.remote(name="N", timeout_s=timeout_s), timeout=60)
    q2 = f"{store_base}/{exp}/q2"
    try:
        ray.get(
            [
                fresh.build_q2.remote(store_addr=q2, rank=0, timeout_s=timeout_s),
                survivor.build_q2.remote(store_addr=q2, rank=1, timeout_s=timeout_s),
            ],
            timeout=timeout_s + _HANG_VERDICT_TIMEOUT_S,
        )
        print("  q2 pair built (survivor + fresh peer)")
    except ray.exceptions.GetTimeoutError:
        alive = _try_ping(survivor)
        _cleanup([survivor, wedged, fresh])
        return f"*** q2 BUILD/configure HUNG (REPRODUCED 卡② at quorum rebuild) *** survivor_ping={alive}"

    # Fresh peer issues its matching q2 collective so the pair CAN complete -- unless
    # the survivor's stream is poisoned by the graveyard zombie.
    fresh_ref = fresh.collective_on_q2.remote()
    surv_ref = survivor.collective_on_q2.remote()
    time.sleep(8)
    probe = None
    try:
        probe = ray.get(survivor.cuda_probe.remote(), timeout=_HANG_VERDICT_TIMEOUT_S)
    except ray.exceptions.GetTimeoutError:
        probe = "cuda_probe HUNG (survivor GPU/stream occupied)"
    print(f"  survivor cuda_probe during q2 collective: {probe}")

    try:
        surv_out = ray.get(surv_ref, timeout=timeout_s + _HANG_VERDICT_TIMEOUT_S)
        fresh_out = ray.get(fresh_ref, timeout=_HANG_VERDICT_TIMEOUT_S)
        _cleanup([survivor, wedged, fresh])
        return f"q2 OK: survivor={surv_out} fresh={fresh_out} probe={probe}"
    except ray.exceptions.GetTimeoutError:
        alive = _try_ping(survivor)
        _cleanup([survivor, wedged, fresh])
        return (
            f"*** q2 COLLECTIVE HUNG >{timeout_s + _HANG_VERDICT_TIMEOUT_S}s (REPRODUCED 卡②) *** "
            f"survivor_ping={alive} probe={probe}"
        )


def _try_ping(worker: object) -> str:
    try:
        return str(ray.get(worker.ping.remote(), timeout=10))
    except Exception as e:  # noqa: BLE001
        return f"ping failed: {type(e).__name__}"


def _cleanup(workers: list) -> None:
    for w in workers:
        try:
            ray.kill(w, no_restart=True)
        except Exception:  # noqa: BLE001 - idempotent cleanup
            pass


def main(
    timeout_s: Annotated[float, typer.Option(help="torchft PG timeout (userspace abort deadline)")] = 20.0,
    only: Annotated[str | None, typer.Option(help="run a single experiment")] = None,
) -> None:
    """Run the graveyard-poisoning reproduction matrix."""
    ray.init(ignore_reinit_error=True)

    from torch.distributed import TCPStore

    store = TCPStore(host_name="localhost", port=0, is_master=True, wait_for_workers=False)
    store_base = f"localhost:{store.port}/graveyard_repro"

    experiments = ["park_then_q2", "abort_then_q2", "shutdown_then_q2"]
    if only is not None:
        experiments = [only]

    results: dict[str, str] = {}
    for exp in experiments:
        results[exp] = _run(exp, store_base=store_base, timeout_s=timeout_s)
        print(f"RESULT {exp}: {results[exp]}\n")
        time.sleep(3)

    print("==== SUMMARY ====")
    for exp, res in results.items():
        print(f"RESULT {exp}: {res}")
    del store


if __name__ == "__main__":
    typer.run(main)
