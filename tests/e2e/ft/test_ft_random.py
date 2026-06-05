# NOTE: You MUST read tests/e2e/ft/README.md as source-of-truth and documentations
# WARNING: Do NOT relax any assert logic in this file. All assertions must remain strict.

import logging
import random
import sys
import threading
import time
from pathlib import Path
from typing import Annotated

import requests

logger = logging.getLogger(__name__)

_MILES_ROOT: Path = Path(__file__).resolve().parents[3]
_miles_root_str = str(_MILES_ROOT)
if _miles_root_str in sys.path:
    sys.path.remove(_miles_root_str)
sys.path.insert(0, _miles_root_str)

import typer
from tests.ci.ci_register import register_cuda_ci
from tests.e2e.ft.conftest_ft.app import resolve_dump_dir
from tests.e2e.ft.conftest_ft.execution import get_common_train_args, get_ft_args, prepare, run_training
from tests.e2e.ft.conftest_ft.modes import FTTestMode, resolve_mode

from miles.utils.test_utils.fault_injector import FailureMode

register_cuda_ci(est_time=1800, suite="stage-c-8-gpu-h200", labels=["ft"])

app: typer.Typer = typer.Typer()

# The mode CI runs (bare `python3 test_ft_random.py`). Single soak mode; others run
# manually via `python tests/e2e/ft/test_ft_random.py run --mode <x>`.
_CI_MODE: str = "dp2_cp2_tp2_ep2"

_CONTROL_SERVER_PORT: int = 18080
_MEAN_INTERVAL_SECONDS: float = 60.0
# Hard floor between consecutive injections so the FT controller has time to
# spawn the replacement actor and let it rejoin before the next crash. Without
# this, the exponential delay can produce several injections within a few
# seconds, causing the all-cells-dead cascade.
_MIN_GAP_BETWEEN_INJECTIONS_SECONDS: float = 30.0
_FAILURE_MODES: list[FailureMode] = [FailureMode.SIGKILL, FailureMode.EXIT, FailureMode.SEGFAULT]


def _run_fault_injection_loop(
    *,
    base_url: str,
    seed: int,
    mean_interval_seconds: float,
    stop_event: threading.Event,
) -> None:
    rng = random.Random(seed)
    last_injection_at: float = 0.0

    while not stop_event.is_set():
        delay = rng.expovariate(1.0 / mean_interval_seconds)
        if stop_event.wait(timeout=delay):
            break

        elapsed = time.monotonic() - last_injection_at
        if elapsed < _MIN_GAP_BETWEEN_INJECTIONS_SECONDS:
            logger.info(
                "Skipping injection: only %.1fs since last, need %.1fs",
                elapsed,
                _MIN_GAP_BETWEEN_INJECTIONS_SECONDS,
            )
            continue

        try:
            resp = requests.get(f"{base_url}/api/v1/cells", timeout=5)
            resp.raise_for_status()
            cells = resp.json()["items"]
        except Exception:
            logger.info("Failed to list cells from control server", exc_info=True)
            continue

        # A cell is "alive" iff its Healthy condition is TRUE. Note: phase=="Running"
        # is also true for StateAllocatedErrored (cell crashed mid-step but not yet
        # cleaned up), so phase alone is too permissive.
        def _is_alive(cell: dict) -> bool:
            return any(
                cond["type"] == "Healthy" and cond["status"] == "True"
                for cond in cell["status"]["conditions"]
            )

        alive = [c for c in cells if _is_alive(c)]
        # Skip injection only when killing one more would leave us with no
        # redundancy left (≤1 alive). Otherwise inject — even if some peers
        # are still mid-recovery, we tolerate further reductions because dp
        # still has spare cells.
        if len(alive) <= 1:
            logger.info(
                "Skipping injection: %d/%d cells alive (need >1 to keep redundancy)",
                len(alive),
                len(cells),
            )
            continue

        target = rng.choice(alive)
        cell_name = target["metadata"]["name"]
        mode = rng.choice(_FAILURE_MODES)

        try:
            resp = requests.post(
                f"{base_url}/api/v1/cells/{cell_name}/inject-fault",
                json={"mode": mode.value, "sub_index": 0},
                timeout=5,
            )
            resp.raise_for_status()
            last_injection_at = time.monotonic()
        except Exception:
            logger.info("Failed to inject fault into %s", cell_name, exc_info=True)


@app.command()
def run(
    mode: Annotated[str, typer.Option(help="Test mode variant")],
    seed: Annotated[int, typer.Option(help="Random seed for fault injection")] = 42,
    num_steps: Annotated[int, typer.Option(help="Number of train() calls")] = 30,
    crash_probability: Annotated[float, typer.Option(help="Per-step crash probability per cell")] = 0.1,
) -> None:
    """Random failure soak test.

    Starts a background thread that injects faults at random intervals
    via the control server HTTP API. The mini FT controller auto-recovers.
    """
    ft_mode: FTTestMode = resolve_mode(mode)
    dump_dir: str = resolve_dump_dir(Path(__file__).stem)
    print(f"Dump directory: {dump_dir}")
    mean_interval: float = _MEAN_INTERVAL_SECONDS / max(crash_probability, 0.01)
    print(f"Seed: {seed}, Steps: {num_steps}, Mean injection interval: {mean_interval:.1f}s")

    prepare(ft_mode)

    train_args = (
        get_common_train_args(ft_mode, dump_dir=dump_dir, num_steps=num_steps)
        + get_ft_args(ft_mode)
        + f"--control-server-port {_CONTROL_SERVER_PORT} "
        + "--mini-ft-controller-enable "
    )

    base_url = f"http://localhost:{_CONTROL_SERVER_PORT}"
    stop_event = threading.Event()
    injector_thread = threading.Thread(
        target=_run_fault_injection_loop,
        kwargs={"base_url": base_url, "seed": seed, "mean_interval_seconds": mean_interval, "stop_event": stop_event},
        daemon=True,
        name="ft-random-fault-injector",
    )
    injector_thread.start()

    try:
        run_training(train_args=train_args, mode=ft_mode)
    finally:
        stop_event.set()
        injector_thread.join(timeout=5)

    print(f"Random failure soak test PASSED (seed={seed}, steps={num_steps})")


if __name__ == "__main__":
    # CUDA CI runs this file as bare `python3 <file>`, so run the soak mode directly
    # (exit code = pass/fail) instead of dispatching to the typer app.
    if len(sys.argv) > 1:
        app()
    else:
        run(mode=_CI_MODE)
