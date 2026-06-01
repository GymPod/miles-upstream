"""Explore torchft ProcessGroup behavior under various failure modes.

Mirrors how torchft Manager uses ProcessGroupNCCL/Gloo internally:
  - pg.allreduce([tensor], opts) → work
  - work.get_future() → fut.then(callback)
  - pg.errored() checked before/after operations
  - pg.configure() for setup

Run inside a Ray cluster with GPUs.

Usage:
    python tests/e2e/external/test_torchft_process_group.py run --failure-mode os-exit
    python tests/e2e/external/test_torchft_process_group.py run-all
"""

import logging
import os
import signal
import time
from datetime import timedelta
from enum import Enum
from typing import Annotated

import ray
import typer

logger = logging.getLogger(__name__)


class FailureMode(str, Enum):
    OS_EXIT = "os-exit"
    RAY_KILL = "ray-kill"
    SHUTDOWN = "shutdown"
    SIGTERM = "sigterm"


@ray.remote(num_gpus=1)
class _PGWorker:
    """Ray actor that holds a torchft PG and runs collectives the same way Manager does."""

    def init(
        self,
        *,
        store_addr: str,
        rank: int,
        world_size: int,
        backend: str,
        timeout_s: float,
    ) -> dict:
        import torch
        from torchft.process_group import ProcessGroupGloo, ProcessGroupNCCL

        self._rank = rank
        self._backend = backend
        self._device = torch.device(f"cuda:{torch.cuda.current_device()}")

        pg_cls = ProcessGroupNCCL if backend == "nccl" else ProcessGroupGloo
        self._pg = pg_cls(timeout=timedelta(seconds=timeout_s))
        self._pg.configure(
            store_addr=store_addr,
            replica_id=str(rank),
            rank=rank,
            world_size=world_size,
            quorum_id=0,
        )

        return {"rank": rank, "backend": backend, "device": str(self._device)}

    def run_allreduce_manager_style(self) -> dict:
        """Run allreduce exactly like torchft Manager.allreduce() does.

        Ref: torchft/manager.py Manager.allreduce() lines ~435-493
        Pattern:
          1. if self.errored(): return noop
          2. opts = AllreduceOptions(); work = pg.allreduce([tensor], opts)
          3. managed_work wraps work, calls work.get_future().then(callback)
          4. on exception: report_error, return noop
        """
        import torch
        import torch.distributed as dist

        tensor = torch.tensor([self._rank + 1.0], device=self._device)
        start = time.monotonic()

        # Step 1: check errored (Manager line 435-436)
        if (e := self._pg.errored()) is not None:
            return {
                "rank": self._rank,
                "status": "skipped_errored",
                "error": str(e),
                "elapsed_s": 0,
            }

        # Step 2: allreduce + get_future (Manager lines 466-484)
        try:
            opts = dist.AllreduceOptions()
            opts.reduceOp = dist.ReduceOp.SUM
            work = self._pg.allreduce([tensor], opts)

            # Manager wraps in _ManagedWork and calls get_future().then(callback)
            fut = work.get_future()
            # Block on the future (simulates what happens when optimizer.step waits)
            fut.wait()

            elapsed = time.monotonic() - start
            errored_after = self._pg.errored()
            return {
                "rank": self._rank,
                "status": "ok",
                "value": tensor.item(),
                "elapsed_s": round(elapsed, 2),
                "errored_after": str(errored_after),
            }
        except Exception as e:
            # Manager lines 487-493: report_error, return DummyWork
            elapsed = time.monotonic() - start
            errored_after = None
            try:
                errored_after = self._pg.errored()
            except Exception:
                errored_after = "errored() failed"
            return {
                "rank": self._rank,
                "status": "exception",
                "error": f"{type(e).__name__}: {e}",
                "elapsed_s": round(elapsed, 2),
                "errored_after": str(errored_after),
            }

    def run_allreduce_blocking_wait(self) -> dict:
        """Run allreduce with blocking work.wait() — how miles indep_dp uses it."""
        import torch
        import torch.distributed as dist

        tensor = torch.tensor([self._rank + 1.0], device=self._device)
        start = time.monotonic()

        try:
            opts = dist.AllreduceOptions()
            opts.reduceOp = dist.ReduceOp.SUM
            work = self._pg.allreduce([tensor], opts)
            success = work.wait()
            elapsed = time.monotonic() - start
            return {
                "rank": self._rank,
                "status": "ok" if success else "wait_returned_false",
                "value": tensor.item(),
                "elapsed_s": round(elapsed, 2),
                "errored_after": str(self._pg.errored()),
            }
        except Exception as e:
            elapsed = time.monotonic() - start
            errored_after = None
            try:
                errored_after = self._pg.errored()
            except Exception:
                errored_after = "errored() failed"
            return {
                "rank": self._rank,
                "status": "exception",
                "error": f"{type(e).__name__}: {e}",
                "elapsed_s": round(elapsed, 2),
                "errored_after": str(errored_after),
            }

    def run_allreduce_poll(self, timeout_s: float = 15.0) -> dict:
        """Run allreduce with is_completed() polling — experimental approach."""
        import torch
        import torch.distributed as dist

        tensor = torch.tensor([self._rank + 1.0], device=self._device)
        start = time.monotonic()

        try:
            opts = dist.AllreduceOptions()
            opts.reduceOp = dist.ReduceOp.SUM
            work = self._pg.allreduce([tensor], opts)

            inner_work = getattr(work, "_work", work)
            deadline = time.monotonic() + timeout_s
            while inner_work is not None and not inner_work.is_completed():
                if time.monotonic() > deadline:
                    return {
                        "rank": self._rank,
                        "status": "poll_timeout",
                        "elapsed_s": round(time.monotonic() - start, 2),
                        "errored_after": str(self._pg.errored()),
                    }
                time.sleep(0.05)

            success = work.wait()
            elapsed = time.monotonic() - start
            return {
                "rank": self._rank,
                "status": "ok" if success else "wait_returned_false",
                "value": tensor.item(),
                "elapsed_s": round(elapsed, 2),
            }
        except Exception as e:
            return {
                "rank": self._rank,
                "status": "exception",
                "error": f"{type(e).__name__}: {e}",
                "elapsed_s": round(time.monotonic() - start, 2),
            }

    def run_continuous_allreduce(self, *, tensor_size: int, duration_s: float) -> dict:
        """Run continuous allreduce for duration_s, then return status.

        Used to keep both ranks in-flight during crash tests.
        """
        import torch
        import torch.distributed as dist

        tensor = torch.ones(tensor_size, device=self._device) * (self._rank + 1.0)
        opts = dist.AllreduceOptions()
        opts.reduceOp = dist.ReduceOp.SUM
        start = time.monotonic()
        count = 0

        try:
            while time.monotonic() - start < duration_s:
                work = self._pg.allreduce([tensor], opts)
                work.wait()
                count += 1

            elapsed = time.monotonic() - start
            return {
                "rank": self._rank,
                "status": "completed",
                "count": count,
                "elapsed_s": round(elapsed, 2),
                "errored_after": str(self._pg.errored()),
            }
        except Exception as e:
            elapsed = time.monotonic() - start
            errored_after = None
            try:
                errored_after = self._pg.errored()
            except Exception:
                errored_after = "errored() failed"
            return {
                "rank": self._rank,
                "status": "exception",
                "error": f"{type(e).__name__}: {e}",
                "count": count,
                "elapsed_s": round(elapsed, 2),
                "errored_after": str(errored_after),
            }

    def run_allreduce_then_die(self, *, tensor_size: int, die_after_s: float) -> None:
        """Start continuous allreduce, then os._exit after a delay.

        This simulates a crash DURING an in-flight allreduce — the key scenario
        that causes ncclCommAbort to hang in production (NVLink DMA residuals).
        """
        import threading

        import torch
        import torch.distributed as dist

        def _delayed_exit() -> None:
            time.sleep(die_after_s)
            logger.warning("rank %d: os._exit after %.1fs (mid-allreduce kill)", self._rank, die_after_s)
            os._exit(1)

        threading.Thread(target=_delayed_exit, daemon=True).start()

        tensor = torch.ones(tensor_size, device=self._device) * (self._rank + 1.0)
        opts = dist.AllreduceOptions()
        opts.reduceOp = dist.ReduceOp.SUM
        while True:
            work = self._pg.allreduce([tensor], opts)
            work.wait()

    # ------------------------------------------------------------------ #
    # Recovery experiment: reproduce the miles indep_dp teardown hang and
    # compare candidate teardown strategies. See the `run_recovery` command.
    # ------------------------------------------------------------------ #

    def init_recovery(
        self,
        *,
        store_addr_q0: str,
        store_addr_q1: str,
        rank: int,
        world_size: int,
        timeout_s: float,
        nonblocking_timeout_override: str | None = None,
        prime_native: bool = False,
        store_host: str | None = None,
        store_port: int | None = None,
        with_gloo: bool = False,
    ) -> dict:
        """Configure a torchft NCCL PG (quorum 0) and report the abort-capability flags.

        Mirrors miles ``create_indep_dp_group``: a fresh ``ProcessGroupNCCL``
        wrapper + ``configure``. ``store_addr_q1`` is kept for the later
        reconfigure-to-singleton step (what a lone surviving cell does).

        Two knobs reproduce the production context the minimal repro lacks:
          - ``nonblocking_timeout_override``: set ``TORCH_NCCL_NONBLOCKING_TIMEOUT``
            in THIS actor before constructing the PG. torchft only sets it if unset,
            so a pre-set large value (e.g. "600") is what governs ncclCommAbort's wait.
          - ``prime_native``: first create a *native* (blocking) ``ProcessGroupNCCL``
            and run an allreduce — like megatron initializing NCCL before indep_dp.
            Tests whether torch caches the nonblocking timeout process-globally at the
            first PG, making torchft's later env-set ineffective.
        """
        import torch
        from torch.distributed import TCPStore

        self._rank = rank
        self._world_size = world_size
        self._timeout_s = timeout_s
        self._store_addr_q1 = store_addr_q1
        self._device = torch.device(f"cuda:{torch.cuda.current_device()}")

        env_before = os.environ.get("TORCH_NCCL_NONBLOCKING_TIMEOUT")
        if nonblocking_timeout_override is not None:
            os.environ["TORCH_NCCL_NONBLOCKING_TIMEOUT"] = nonblocking_timeout_override

        primed = False
        if prime_native and store_host is not None and store_port is not None:
            import torch.distributed as dist

            native_store = TCPStore(host_name=store_host, port=store_port, is_master=False, wait_for_workers=False)
            dist.init_process_group(
                backend="nccl",
                store=dist.PrefixStore("native_prime", native_store),
                rank=rank,
                world_size=world_size,
            )
            t = torch.ones(1024, device=self._device) * (rank + 1.0)
            dist.all_reduce(t)
            torch.cuda.synchronize()
            primed = True

        from torchft.process_group import ProcessGroupGloo, ProcessGroupNCCL

        self._pg = ProcessGroupNCCL(timeout=timedelta(seconds=timeout_s))
        self._pg.configure(
            store_addr=store_addr_q0,
            replica_id=str(rank),
            rank=rank,
            world_size=world_size,
            quorum_id=0,
        )

        # Mirror miles create_indep_dp_group, which also builds a Gloo PG and tears
        # both down in reconfigure. Gloo has no ncclCommAbort; its destructor can
        # block on a TCP socket to a dead peer.
        self._gloo_pg = None
        if with_gloo:
            self._gloo_pg = ProcessGroupGloo(timeout=timedelta(seconds=timeout_s))
            self._gloo_pg.configure(
                store_addr=f"{store_addr_q0}/gloo",
                replica_id=str(rank),
                rank=rank,
                world_size=world_size,
                quorum_id=0,
            )

        return {
            "rank": rank,
            "nccl_version": torch.cuda.nccl.version(),
            "use_abort": self._pg._use_abort,
            "with_gloo": with_gloo,
            "env_before": env_before,
            "nonblocking_timeout_env": os.environ.get("TORCH_NCCL_NONBLOCKING_TIMEOUT"),
            "primed_native": primed,
            "device": str(self._device),
        }

    def survivor_detect(self, *, tensor_size: int, max_iters: int = 100_000) -> dict:
        """Loop allreduce until the peer's death is detected (the torchft-idiomatic path).

        ``ProcessGroupNCCL.allreduce`` returns a ``_WorkAcceleratorTimeout`` whose
        ``wait()`` arms a userspace ``context_timeout`` that fires ``pg.abort()``
        after ``timeout_s``. So when the peer dies mid-flight, the in-flight comm
        is aborted *here*, during the failed wait — not at teardown. This is the
        crux of why torchft's reconfigure can be fast.
        """
        import torch
        import torch.distributed as dist

        tensor = torch.ones(tensor_size, device=self._device) * (self._rank + 1.0)
        opts = dist.AllreduceOptions()
        opts.reduceOp = dist.ReduceOp.SUM

        start = time.monotonic()
        count = 0
        status = "still_running"
        err = None
        try:
            for _ in range(max_iters):
                work = self._pg.allreduce([tensor], opts)
                if not work.wait():
                    status = "wait_false"
                    break
                count += 1
            else:
                status = "exhausted_iters"
        except Exception as e:
            status = "exception"
            err = f"{type(e).__name__}: {e}"

        elapsed = time.monotonic() - start
        try:
            errored_after = str(self._pg.errored())
        except Exception as e:
            errored_after = f"errored() raised {type(e).__name__}: {e}"

        return {
            "rank": self._rank,
            "status": status,
            "count": count,
            "error": err,
            "detect_elapsed_s": round(elapsed, 2),
            "errored_after": errored_after,
        }

    def survivor_launch_inflight(self, *, tensor_size: int) -> dict:
        """Launch an allreduce WITHOUT waiting, leaving the collective in-flight.

        Used for the no-abort variant: the peer dies while this kernel is still
        enqueued and no ``wait()`` ever fires the userspace abort, so a subsequent
        teardown is forced to drain a stuck-on-NVLink collective itself.
        """
        import torch
        import torch.distributed as dist

        tensor = torch.ones(tensor_size, device=self._device) * (self._rank + 1.0)
        opts = dist.AllreduceOptions()
        opts.reduceOp = dist.ReduceOp.SUM
        self._inflight_work = self._pg.allreduce([tensor], opts)
        self._inflight_tensor = tensor
        return {"rank": self._rank, "status": "launched"}

    def survivor_teardown(self, *, strategy: str, daemon_join_s: float = 20.0) -> dict:
        """Tear down the dead-peer PG and reconfigure to a singleton quorum.

        Strategies mirror the candidate miles fixes:
          - ``manager_idiomatic``: reuse the SAME wrapper, call ``configure(q1, world_size=1)``
            (the torchft ``Manager`` pattern; ``configure`` internally ``abort(errored=False)``).
          - ``shutdown_new``: ``old.shutdown()`` then a fresh wrapper (miles pre-fix; the
            blocking C++ ``~ProcessGroupNCCL`` runs when the last ref drops).
          - ``abort_new``: ``old.abort(errored=False)`` then a fresh wrapper.
          - ``daemon_shutdown``: ``old.shutdown()`` on a daemon thread with a join
            timeout, then a fresh wrapper (the current miles fix).
        """
        import threading

        from torchft.process_group import ProcessGroupNCCL

        start = time.monotonic()
        note = None

        def _fresh_singleton() -> None:
            self._pg = ProcessGroupNCCL(timeout=timedelta(seconds=self._timeout_s))
            self._pg.configure(
                store_addr=self._store_addr_q1,
                replica_id=str(self._rank),
                rank=0,
                world_size=1,
                quorum_id=1,
            )

        if strategy == "manager_idiomatic":
            self._pg.configure(
                store_addr=self._store_addr_q1,
                replica_id=str(self._rank),
                rank=0,
                world_size=1,
                quorum_id=1,
            )
        elif strategy == "shutdown_new":
            old = self._pg
            old.shutdown()
            del old
            _fresh_singleton()
        elif strategy == "abort_new":
            old = self._pg
            old.abort(errored=False)
            del old
            _fresh_singleton()
        elif strategy == "daemon_shutdown":
            old = self._pg
            t = threading.Thread(target=old.shutdown, name="pg-shutdown", daemon=True)
            t.start()
            t.join(daemon_join_s)
            if t.is_alive():
                note = f"old shutdown still blocked after {daemon_join_s}s; abandoned"
            del old
            _fresh_singleton()
        else:
            raise ValueError(f"unknown strategy {strategy}")

        nccl_elapsed = time.monotonic() - start

        # Tear down the Gloo PG the same way reconfigure_indep_dp_group does.
        gloo_elapsed = None
        if getattr(self, "_gloo_pg", None) is not None:
            gloo_start = time.monotonic()
            if strategy == "daemon_shutdown":
                gt = threading.Thread(target=self._gloo_pg.shutdown, name="gloo-shutdown", daemon=True)
                gt.start()
                gt.join(daemon_join_s)
                if gt.is_alive():
                    note = (note or "") + f" gloo shutdown still blocked after {daemon_join_s}s"
            elif strategy == "abort_new":
                self._gloo_pg.abort(errored=False)
            else:
                self._gloo_pg.shutdown()
            self._gloo_pg = None
            gloo_elapsed = round(time.monotonic() - gloo_start, 2)

        return {
            "rank": self._rank,
            "strategy": strategy,
            "teardown_elapsed_s": round(nccl_elapsed, 2),
            "gloo_teardown_s": gloo_elapsed,
            "note": note,
        }

    def verify_singleton(self) -> dict:
        """Confirm the survivor's new singleton (world_size=1) PG is functional."""
        import torch
        import torch.distributed as dist

        tensor = torch.ones(8, device=self._device) * 7.0
        opts = dist.AllreduceOptions()
        opts.reduceOp = dist.ReduceOp.SUM
        start = time.monotonic()
        ok = self._pg.allreduce([tensor], opts).wait()
        return {
            "rank": self._rank,
            "ok": bool(ok),
            "value": tensor[0].item(),
            "elapsed_s": round(time.monotonic() - start, 2),
        }

    def die_os_exit(self) -> None:
        os._exit(1)

    def die_shutdown(self) -> None:
        self._pg.shutdown()

    def die_sigterm(self) -> None:
        os.kill(os.getpid(), signal.SIGTERM)

    def get_status(self) -> dict:
        errored = None
        try:
            errored = self._pg.errored()
        except Exception as e:
            errored = f"errored() failed: {e}"
        return {"rank": self._rank, "alive": True, "errored": str(errored)}


class _WaitStyle(str, Enum):
    MANAGER = "manager"
    BLOCKING = "blocking"
    POLL = "poll"


def _run_test(
    *,
    failure_mode: FailureMode,
    backend: str,
    timeout_s: float,
    wait_style: _WaitStyle,
    poll_timeout_s: float,
    world_size: int,
) -> None:
    from torch.distributed import TCPStore

    store = TCPStore(
        host_name="localhost",
        port=0,
        is_master=True,
        wait_for_workers=False,
    )
    store_addr = f"localhost:{store.port}/test"

    print(f"\n{'='*70}")
    print(
        f"  backend={backend}  failure={failure_mode.value}  timeout={timeout_s}s"
        f"  wait={wait_style.value}  world_size={world_size}"
    )
    print(f"{'='*70}\n")

    workers = [_PGWorker.remote() for _ in range(world_size)]

    init_results = ray.get(
        [
            w.init.remote(
                store_addr=store_addr,
                rank=i,
                world_size=world_size,
                backend=backend,
                timeout_s=timeout_s,
            )
            for i, w in enumerate(workers)
        ]
    )
    for r in init_results:
        print(f"  init: {r}")

    # Step 1: sanity check — normal allreduce
    print("\n--- Step 1: Normal allreduce (sanity) ---")
    results = ray.get([w.run_allreduce_manager_style.remote() for w in workers])
    for r in results:
        print(f"  {r}")

    # Step 2: kill rank 0
    victim = workers[0]
    survivors = workers[1:]
    print(f"\n--- Step 2: Kill rank 0 ({failure_mode.value}) ---")

    if failure_mode == FailureMode.OS_EXIT:
        try:
            ray.get(victim.die_os_exit.remote(), timeout=5)
        except Exception as e:
            print(f"  rank 0 died: {type(e).__name__}")
    elif failure_mode == FailureMode.RAY_KILL:
        ray.kill(victim, no_restart=True)
        print("  rank 0 killed via ray.kill")
    elif failure_mode == FailureMode.SHUTDOWN:
        ray.get(victim.die_shutdown.remote())
        print("  rank 0 shutdown PG gracefully")
    elif failure_mode == FailureMode.SIGTERM:
        try:
            ray.get(victim.die_sigterm.remote(), timeout=5)
        except Exception as e:
            print(f"  rank 0 sigterm: {type(e).__name__}")

    time.sleep(1)

    # Step 3: survivors try allreduce
    print(f"\n--- Step 3: Survivor allreduce (wait_style={wait_style.value}) ---")
    start = time.monotonic()

    if wait_style == _WaitStyle.MANAGER:
        refs = [s.run_allreduce_manager_style.remote() for s in survivors]
    elif wait_style == _WaitStyle.BLOCKING:
        refs = [s.run_allreduce_blocking_wait.remote() for s in survivors]
    else:
        refs = [s.run_allreduce_poll.remote(timeout_s=poll_timeout_s) for s in survivors]

    for ref in refs:
        try:
            result = ray.get(ref, timeout=timeout_s + 90)
            print(f"  {result}")
        except ray.exceptions.GetTimeoutError:
            elapsed = time.monotonic() - start
            print(f"  TIMEOUT: ray.get timed out after {elapsed:.0f}s — actor likely dead or hung")
        except ray.exceptions.RayActorError as e:
            elapsed = time.monotonic() - start
            print(f"  ACTOR DIED after {elapsed:.0f}s: {e}")

    # Step 4: check survivor status
    print("\n--- Step 4: Survivor status ---")
    for s in survivors:
        try:
            status = ray.get(s.get_status.remote(), timeout=10)
            print(f"  {status}")
        except Exception as e:
            print(f"  Cannot reach survivor: {type(e).__name__}: {e}")

    # Cleanup: kill all remaining actors to free GPU memory for next test
    for w in survivors:
        try:
            ray.kill(w, no_restart=True)
        except Exception:
            pass
    del store

    print()


app = typer.Typer()


@app.command()
def run(
    failure_mode: Annotated[FailureMode, typer.Option(help="How to kill rank 0")] = FailureMode.OS_EXIT,
    backend: Annotated[str, typer.Option(help="nccl or gloo")] = "nccl",
    timeout_s: Annotated[float, typer.Option(help="torchft PG timeout in seconds")] = 30.0,
    wait_style: Annotated[_WaitStyle, typer.Option(help="manager / blocking / poll")] = _WaitStyle.MANAGER,
    poll_timeout_s: Annotated[float, typer.Option(help="Poll timeout (only with --wait-style poll)")] = 15.0,
    world_size: Annotated[int, typer.Option(help="Number of ranks")] = 2,
) -> None:
    """Run a single failure-mode test."""
    ray.init(ignore_reinit_error=True)
    _run_test(
        failure_mode=failure_mode,
        backend=backend,
        timeout_s=timeout_s,
        wait_style=wait_style,
        poll_timeout_s=poll_timeout_s,
        world_size=world_size,
    )


@app.command()
def run_all(
    timeout_s: Annotated[float, typer.Option(help="torchft PG timeout in seconds")] = 30.0,
    world_size: Annotated[int, typer.Option(help="Number of ranks")] = 2,
) -> None:
    """Run all failure modes × backends × wait styles."""
    ray.init(ignore_reinit_error=True)

    for backend in ["gloo", "nccl"]:
        for failure_mode in [FailureMode.SHUTDOWN, FailureMode.OS_EXIT, FailureMode.RAY_KILL]:
            for wait_style in [_WaitStyle.MANAGER, _WaitStyle.BLOCKING, _WaitStyle.POLL]:
                try:
                    _run_test(
                        failure_mode=failure_mode,
                        backend=backend,
                        timeout_s=timeout_s,
                        wait_style=wait_style,
                        poll_timeout_s=min(15.0, timeout_s),
                        world_size=world_size,
                    )
                except Exception as e:
                    print(f"  TEST FAILED: {e}\n")


@app.command()
def run_inflight(
    timeout_s: Annotated[float, typer.Option(help="torchft PG timeout in seconds")] = 30.0,
    tensor_size: Annotated[int, typer.Option(help="Tensor size for allreduce (larger = longer)")] = 100_000_000,
    die_after_s: Annotated[float, typer.Option(help="Kill rank 0 after this many seconds")] = 2.0,
    wait_style: Annotated[_WaitStyle, typer.Option(help="How survivor waits")] = _WaitStyle.BLOCKING,
) -> None:
    """Test in-flight allreduce crash — simulates the miles production scenario.

    Both ranks start a continuous allreduce loop with a large tensor.
    Rank 0 os._exit's after die_after_s while allreduce is in progress.
    This tests whether ncclCommAbort hangs when NCCL kernels are in-flight on NVLink.
    """
    from torch.distributed import TCPStore

    ray.init(ignore_reinit_error=True)

    store = TCPStore(
        host_name="localhost",
        port=0,
        is_master=True,
        wait_for_workers=False,
    )
    store_addr = f"localhost:{store.port}/inflight"

    print(f"\n{'='*70}")
    print(
        f"  IN-FLIGHT CRASH TEST: tensor_size={tensor_size}  die_after={die_after_s}s"
        f"  timeout={timeout_s}s  wait={wait_style.value}"
    )
    print(f"{'='*70}\n")

    workers = [_PGWorker.remote() for _ in range(2)]
    init_results = ray.get(
        [
            w.init.remote(
                store_addr=store_addr,
                rank=i,
                world_size=2,
                backend="nccl",
                timeout_s=timeout_s,
            )
            for i, w in enumerate(workers)
        ]
    )
    for r in init_results:
        print(f"  init: {r}")

    # Step 1: sanity allreduce
    print("\n--- Step 1: Sanity allreduce ---")
    results = ray.get([w.run_allreduce_blocking_wait.remote() for w in workers])
    for r in results:
        print(f"  {r}")

    # Step 2: start concurrent continuous allreduce on BOTH ranks
    # Rank 0 will die mid-flight; rank 1 keeps going and should detect the failure
    print("\n--- Step 2: Both ranks start continuous allreduce ---")
    print(f"  Rank 0 will os._exit after {die_after_s}s")
    print(f"  Rank 1 runs for {die_after_s + timeout_s + 30}s (covers timeout + abort)")
    start = time.monotonic()

    survivor_duration = die_after_s + timeout_s + 30
    victim_ref = workers[0].run_allreduce_then_die.remote(
        tensor_size=tensor_size,
        die_after_s=die_after_s,
    )
    survivor_ref = workers[1].run_continuous_allreduce.remote(
        tensor_size=tensor_size,
        duration_s=survivor_duration,
    )

    # Wait for victim to confirm death
    try:
        ray.get(victim_ref, timeout=die_after_s + 10)
    except Exception as e:
        elapsed = time.monotonic() - start
        print(f"  Rank 0 confirmed dead after {elapsed:.1f}s: {type(e).__name__}")

    # Step 3: wait for survivor result
    print(f"\n--- Step 3: Waiting for survivor (timeout={timeout_s + 90}s) ---")
    try:
        result = ray.get(survivor_ref, timeout=timeout_s + 90)
        elapsed = time.monotonic() - start
        print(f"  Survivor result after {elapsed:.1f}s: {result}")
    except ray.exceptions.GetTimeoutError:
        elapsed = time.monotonic() - start
        print(f"  TIMEOUT: survivor hung for {elapsed:.0f}s — ABORT HANG CONFIRMED")
    except ray.exceptions.RayActorError as e:
        elapsed = time.monotonic() - start
        print(f"  ACTOR DIED after {elapsed:.0f}s: {e}")

    # Step 4: check survivor
    print("\n--- Step 4: Survivor status ---")
    try:
        status = ray.get(workers[1].get_status.remote(), timeout=10)
        print(f"  {status}")
    except Exception as e:
        print(f"  Cannot reach survivor: {type(e).__name__}: {e}")

    # Cleanup
    for w in workers:
        try:
            ray.kill(w, no_restart=True)
        except Exception:
            pass
    del store
    print()


@app.command()
def run_p2p(
    timeout_s: Annotated[float, typer.Option(help="torchft PG timeout")] = 30.0,
) -> None:
    """Test P2P send/recv on torchft ProcessGroupGloo — used by PGTransport for ckpt transfer."""
    from torch.distributed import TCPStore

    ray.init(ignore_reinit_error=True)

    store = TCPStore(host_name="localhost", port=0, is_master=True, wait_for_workers=False)
    store_addr = f"localhost:{store.port}/p2p"

    print(f"\n{'='*70}")
    print(f"  P2P SEND/RECV TEST (Gloo)  timeout={timeout_s}s")
    print(f"{'='*70}\n")

    @ray.remote(num_gpus=1)
    class _P2PWorker:
        def init(self, *, store_addr: str, rank: int, timeout_s: float) -> dict:
            from torchft.process_group import ProcessGroupGloo

            self._rank = rank
            self._pg = ProcessGroupGloo(timeout=timedelta(seconds=timeout_s))
            self._pg.configure(store_addr=store_addr, replica_id=str(rank), rank=rank, world_size=2, quorum_id=0)
            return {"rank": rank, "status": "configured"}

        def send_tensor(self, dst_rank: int) -> dict:
            import torch

            t = torch.tensor([42.0, 43.0])
            start = time.monotonic()
            try:
                work = self._pg.send([t], dst_rank, tag=100)
                work.wait()
                return {"rank": self._rank, "status": "sent", "elapsed_s": round(time.monotonic() - start, 2)}
            except Exception as e:
                return {
                    "rank": self._rank,
                    "status": "send_error",
                    "error": str(e),
                    "elapsed_s": round(time.monotonic() - start, 2),
                }

        def recv_tensor(self, src_rank: int) -> dict:
            import torch

            t = torch.zeros(2)
            start = time.monotonic()
            try:
                work = self._pg.recv([t], src_rank, tag=100)
                work.wait()
                return {
                    "rank": self._rank,
                    "status": "received",
                    "value": t.tolist(),
                    "elapsed_s": round(time.monotonic() - start, 2),
                }
            except Exception as e:
                return {
                    "rank": self._rank,
                    "status": "recv_error",
                    "error": str(e),
                    "elapsed_s": round(time.monotonic() - start, 2),
                }

    workers = [_P2PWorker.remote() for _ in range(2)]
    results = ray.get(
        [w.init.remote(store_addr=store_addr, rank=i, timeout_s=timeout_s) for i, w in enumerate(workers)]
    )
    for r in results:
        print(f"  {r}")

    print("\n--- P2P: rank 0 sends to rank 1 ---")
    send_ref = workers[0].send_tensor.remote(dst_rank=1)
    recv_ref = workers[1].recv_tensor.remote(src_rank=0)
    for ref, label in [(send_ref, "send"), (recv_ref, "recv")]:
        try:
            result = ray.get(ref, timeout=timeout_s + 30)
            print(f"  {label}: {result}")
        except Exception as e:
            print(f"  {label}: FAILED — {e}")

    for w in workers:
        try:
            ray.kill(w, no_restart=True)
        except Exception:
            pass
    del store
    print()


@app.command()
def run_p2p_after_reconfig(
    timeout_s: Annotated[float, typer.Option(help="torchft PG timeout")] = 30.0,
) -> None:
    """Test P2P after PG reconfigure — simulates healing ckpt transfer scenario."""
    from torch.distributed import TCPStore

    ray.init(ignore_reinit_error=True)

    store = TCPStore(host_name="localhost", port=0, is_master=True, wait_for_workers=False)
    store_addr = f"localhost:{store.port}/p2p_reconfig"

    print(f"\n{'='*70}")
    print(f"  P2P AFTER RECONFIG TEST (Gloo)  timeout={timeout_s}s")
    print(f"{'='*70}\n")

    @ray.remote(num_gpus=1)
    class _ReconfigP2PWorker:
        def init(self, *, store_addr: str, rank: int, timeout_s: float) -> dict:
            from torchft.process_group import ProcessGroupGloo

            self._rank = rank
            self._store_addr = store_addr
            self._timeout_s = timeout_s
            self._pg = ProcessGroupGloo(timeout=timedelta(seconds=timeout_s))
            self._pg.configure(
                store_addr=f"{store_addr}/q0", replica_id=str(rank), rank=rank, world_size=2, quorum_id=0
            )
            return {"rank": rank, "status": "configured_q0"}

        def do_allreduce(self) -> dict:
            import torch
            import torch.distributed as dist

            t = torch.tensor([self._rank + 1.0])
            opts = dist.AllreduceOptions()
            opts.reduceOp = dist.ReduceOp.SUM
            work = self._pg.allreduce([t], opts)
            work.wait()
            return {"rank": self._rank, "value": t.item()}

        def shutdown_and_reconfigure(self) -> dict:
            from torchft.process_group import ProcessGroupGloo

            self._pg.shutdown()
            self._pg = ProcessGroupGloo(timeout=timedelta(seconds=self._timeout_s))
            self._pg.configure(
                store_addr=f"{self._store_addr}/q1",
                replica_id=str(self._rank),
                rank=self._rank,
                world_size=2,
                quorum_id=1,
            )
            return {"rank": self._rank, "status": "reconfigured_q1"}

        def send_tensor(self, dst_rank: int) -> dict:
            import torch

            t = torch.tensor([99.0, 100.0])
            start = time.monotonic()
            try:
                work = self._pg.send([t], dst_rank, tag=200)
                work.wait()
                return {"rank": self._rank, "status": "sent", "elapsed_s": round(time.monotonic() - start, 2)}
            except Exception as e:
                return {
                    "rank": self._rank,
                    "status": "send_error",
                    "error": str(e),
                    "elapsed_s": round(time.monotonic() - start, 2),
                }

        def recv_tensor(self, src_rank: int) -> dict:
            import torch

            t = torch.zeros(2)
            start = time.monotonic()
            try:
                work = self._pg.recv([t], src_rank, tag=200)
                work.wait()
                return {
                    "rank": self._rank,
                    "status": "received",
                    "value": t.tolist(),
                    "elapsed_s": round(time.monotonic() - start, 2),
                }
            except Exception as e:
                return {
                    "rank": self._rank,
                    "status": "recv_error",
                    "error": str(e),
                    "elapsed_s": round(time.monotonic() - start, 2),
                }

    workers = [_ReconfigP2PWorker.remote() for _ in range(2)]

    print("--- Step 1: Configure (quorum 0) ---")
    results = ray.get(
        [w.init.remote(store_addr=store_addr, rank=i, timeout_s=timeout_s) for i, w in enumerate(workers)]
    )
    for r in results:
        print(f"  {r}")

    print("\n--- Step 2: Normal allreduce ---")
    results = ray.get([w.do_allreduce.remote() for w in workers])
    for r in results:
        print(f"  {r}")

    print("\n--- Step 3: Shutdown + reconfigure (quorum 1) ---")
    results = ray.get([w.shutdown_and_reconfigure.remote() for w in workers])
    for r in results:
        print(f"  {r}")

    print("\n--- Step 4: P2P send/recv after reconfig ---")
    send_ref = workers[0].send_tensor.remote(dst_rank=1)
    recv_ref = workers[1].recv_tensor.remote(src_rank=0)
    for ref, label in [(send_ref, "send"), (recv_ref, "recv")]:
        try:
            result = ray.get(ref, timeout=timeout_s + 30)
            print(f"  {label}: {result}")
        except Exception as e:
            print(f"  {label}: FAILED — {e}")

    for w in workers:
        try:
            ray.kill(w, no_restart=True)
        except Exception:
            pass
    del store
    print()


def _run_recovery_once(
    *,
    strategy: str,
    pre_detect: bool,
    timeout_s: float,
    tensor_size: int,
    die_after_s: float,
    teardown_budget_s: float,
    nonblocking_timeout_override: str | None = None,
    prime_native: bool = False,
    with_gloo: bool = False,
) -> dict:
    """One recovery trial: 2-rank NVLink PG, peer dies mid-allreduce, survivor recovers.

    ``pre_detect=True`` lets the survivor's ``wait()`` fire torchft's userspace abort
    during the failed collective (the idiomatic path) before tearing down.
    ``pre_detect=False`` leaves the collective in-flight (no abort) so teardown must
    drain it — the regime that hangs on NVLink. Returns a verdict dict; a teardown
    that exceeds ``teardown_budget_s`` is reported as a HANG (not waited out).
    """
    from torch.distributed import TCPStore

    store = TCPStore(host_name="localhost", port=0, is_master=True, wait_for_workers=False)
    base = f"localhost:{store.port}/recovery"
    store_addr_q0 = f"{base}/q0"
    store_addr_q1 = f"{base}/q1"

    label = f"strategy={strategy} pre_detect={pre_detect} override={nonblocking_timeout_override} prime={prime_native}"
    print(f"\n{'='*70}\n  RECOVERY: {label}  timeout={timeout_s}s die_after={die_after_s}s\n{'='*70}\n")

    workers = [_PGWorker.remote() for _ in range(2)]
    init = ray.get(
        [
            w.init_recovery.remote(
                store_addr_q0=store_addr_q0,
                store_addr_q1=store_addr_q1,
                rank=i,
                world_size=2,
                timeout_s=timeout_s,
                nonblocking_timeout_override=nonblocking_timeout_override,
                prime_native=prime_native,
                store_host="localhost",
                store_port=store.port,
                with_gloo=with_gloo,
            )
            for i, w in enumerate(workers)
        ]
    )
    for r in init:
        print(f"  init: {r}")

    victim, survivor = workers[0], workers[1]
    verdict = {
        "strategy": strategy,
        "pre_detect": pre_detect,
        "nccl_version": init[0]["nccl_version"],
        "use_abort": init[0]["use_abort"],
    }

    # Peer launches continuous allreduce then os._exit's mid-flight.
    victim_ref = victim.run_allreduce_then_die.remote(tensor_size=tensor_size, die_after_s=die_after_s)

    if pre_detect:
        # Survivor loops allreduce; the failed wait() fires torchft's userspace abort.
        detect_start = time.monotonic()
        try:
            detect = ray.get(
                survivor.survivor_detect.remote(tensor_size=tensor_size),
                timeout=die_after_s + timeout_s + 60,
            )
            print(f"  detect: {detect}")
            verdict["detect"] = detect
        except ray.exceptions.GetTimeoutError:
            verdict["detect"] = f"DETECT HANG > {time.monotonic()-detect_start:.0f}s"
            print(f"  detect: {verdict['detect']}")
    else:
        # Leave a collective in-flight with no wait → no userspace abort fires.
        print(f"  launch: {ray.get(survivor.survivor_launch_inflight.remote(tensor_size=tensor_size))}")
        time.sleep(die_after_s + 3)  # let the peer die while the collective is stuck

    try:
        ray.get(victim_ref, timeout=die_after_s + 10)
    except Exception as e:
        print(f"  victim dead: {type(e).__name__}")

    # The decisive step: tear down the dead-peer comm + reconfigure to singleton.
    teardown_start = time.monotonic()
    try:
        teardown = ray.get(
            survivor.survivor_teardown.remote(strategy=strategy),
            timeout=teardown_budget_s,
        )
        print(f"  teardown: {teardown}")
        verdict["teardown"] = teardown

        verify = ray.get(survivor.verify_singleton.remote(), timeout=30)
        print(f"  verify: {verify}")
        verdict["verify"] = verify
        verdict["result"] = "RECOVERED" if verify.get("ok") else "TEARDOWN_OK_VERIFY_FAILED"
    except ray.exceptions.GetTimeoutError:
        hung_for = time.monotonic() - teardown_start
        verdict["teardown"] = f"HANG > {hung_for:.0f}s (budget {teardown_budget_s}s)"
        verdict["result"] = "HANG"
        print(f"  teardown: {verdict['teardown']}  <<< TEARDOWN HANG CONFIRMED")
    except Exception as e:
        verdict["teardown"] = f"{type(e).__name__}: {e}"
        verdict["result"] = "ERROR"
        print(f"  teardown: {verdict['teardown']}")

    for w in workers:
        try:
            ray.kill(w, no_restart=True)
        except Exception:
            pass
    del store
    print(f"  => {verdict.get('result')}\n")
    return verdict


@app.command()
def run_recovery(
    strategy: Annotated[
        str, typer.Option(help="manager_idiomatic | shutdown_new | abort_new | daemon_shutdown")
    ] = "manager_idiomatic",
    pre_detect: Annotated[bool, typer.Option(help="Let userspace abort fire during the failed wait first")] = True,
    timeout_s: Annotated[float, typer.Option(help="torchft PG timeout (also the userspace abort deadline)")] = 20.0,
    tensor_size: Annotated[
        int, typer.Option(help="allreduce size; larger keeps the collective in-flight longer")
    ] = 100_000_000,
    die_after_s: Annotated[float, typer.Option(help="peer os._exit's this long after starting")] = 2.0,
    teardown_budget_s: Annotated[float, typer.Option(help="ray.get timeout for teardown; exceeding = HANG")] = 90.0,
    nonblocking_timeout_override: Annotated[
        str | None, typer.Option(help="Pre-set TORCH_NCCL_NONBLOCKING_TIMEOUT in the actor (e.g. '600')")
    ] = None,
    prime_native: Annotated[
        bool, typer.Option(help="Create a native NCCL PG + allreduce first (mimics megatron init)")
    ] = False,
    with_gloo: Annotated[
        bool, typer.Option(help="Also create + tear down a torchft Gloo PG (mirrors indep_dp gloo_group)")
    ] = False,
) -> None:
    """Run a single recovery trial."""
    ray.init(ignore_reinit_error=True)
    _run_recovery_once(
        strategy=strategy,
        pre_detect=pre_detect,
        timeout_s=timeout_s,
        tensor_size=tensor_size,
        die_after_s=die_after_s,
        teardown_budget_s=teardown_budget_s,
        nonblocking_timeout_override=nonblocking_timeout_override,
        prime_native=prime_native,
        with_gloo=with_gloo,
    )


@app.command()
def run_recovery_all(
    timeout_s: Annotated[float, typer.Option()] = 20.0,
    tensor_size: Annotated[int, typer.Option()] = 100_000_000,
    die_after_s: Annotated[float, typer.Option()] = 2.0,
    teardown_budget_s: Annotated[float, typer.Option()] = 90.0,
) -> None:
    """Run the full strategy × {abort-fired, in-flight-no-abort} matrix and print a verdict table."""
    ray.init(ignore_reinit_error=True)

    strategies = ["manager_idiomatic", "shutdown_new", "abort_new", "daemon_shutdown"]
    verdicts = []
    for pre_detect in [True, False]:
        for strategy in strategies:
            try:
                verdicts.append(
                    _run_recovery_once(
                        strategy=strategy,
                        pre_detect=pre_detect,
                        timeout_s=timeout_s,
                        tensor_size=tensor_size,
                        die_after_s=die_after_s,
                        teardown_budget_s=teardown_budget_s,
                    )
                )
            except Exception as e:
                print(f"  TRIAL FAILED ({strategy}, pre_detect={pre_detect}): {e}\n")
                verdicts.append({"strategy": strategy, "pre_detect": pre_detect, "result": f"TRIAL_ERROR: {e}"})

    print(f"\n{'='*70}\n  RECOVERY VERDICT TABLE\n{'='*70}")
    print(f"  {'strategy':<20} {'pre_detect':<11} {'result':<26} teardown")
    for v in verdicts:
        td = v.get("teardown")
        td_s = td.get("teardown_elapsed_s") if isinstance(td, dict) else td
        print(f"  {v.get('strategy',''):<20} {str(v.get('pre_detect','')):<11} {str(v.get('result','')):<26} {td_s}")
    print()


if __name__ == "__main__":
    app()
