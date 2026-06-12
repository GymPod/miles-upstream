# NOTE: You MUST read tests/e2e/ft/README.md as source-of-truth and documentations

import logging
import random
import threading
import time

import requests

from miles.utils.test_utils.fault_injector import FailureMode

logger = logging.getLogger(__name__)

CONTROL_SERVER_PORT: int = 18080
MEAN_INTERVAL_SECONDS: float = 60.0
# Hard floor between consecutive injections so the FT controller has time to
# spawn the replacement actor and let it rejoin before the next crash. Without
# this, the exponential delay can produce several injections within a few
# seconds, causing the all-cells-dead cascade.
MIN_GAP_BETWEEN_INJECTIONS_SECONDS: float = 30.0
FAILURE_MODES: list[FailureMode] = [FailureMode.SIGKILL, FailureMode.EXIT, FailureMode.SEGFAULT]


def run_fault_injection_loop(
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
        if elapsed < MIN_GAP_BETWEEN_INJECTIONS_SECONDS:
            logger.info(
                "Skipping injection: only %.1fs since last, need %.1fs",
                elapsed,
                MIN_GAP_BETWEEN_INJECTIONS_SECONDS,
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
            return any(cond["type"] == "Healthy" and cond["status"] == "True" for cond in cell["status"]["conditions"])

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
        mode = rng.choice(FAILURE_MODES)

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


def spawn_fault_injector(*, seed: int, mean_interval_seconds: float) -> tuple[threading.Event, threading.Thread]:
    base_url = f"http://localhost:{CONTROL_SERVER_PORT}"
    stop_event = threading.Event()
    injector_thread = threading.Thread(
        target=run_fault_injection_loop,
        kwargs={
            "base_url": base_url,
            "seed": seed,
            "mean_interval_seconds": mean_interval_seconds,
            "stop_event": stop_event,
        },
        daemon=True,
        name="ft-random-fault-injector",
    )
    injector_thread.start()
    return stop_event, injector_thread
