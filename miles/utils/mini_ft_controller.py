from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

import httpx

from miles.utils.control_server.models import Cell, CellList, CellPatch, CellPatchSpec, TriState
from miles.utils.pydantic_utils import StrictBaseModel

logger = logging.getLogger(__name__)


# ------------------------ entrypoint ------------------------


def maybe_start_mini_ft_controller(args: Any) -> None:
    if not args.mini_ft_controller_enable:
        return

    runner = _MiniFTControllerRunner(
        control_server_url=f"http://127.0.0.1:{args.control_server_port}",
        poll_interval=args.mini_ft_controller_poll_interval,
        resume_delay=args.mini_ft_controller_resume_delay,
    )

    def _run() -> None:
        asyncio.run(runner.run())

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    logger.info("Started mini FT controller on daemon thread")


# ------------------------ HTTP transport + thread runner ------------------------


class _MiniFTControllerRunner:
    def __init__(
        self,
        *,
        control_server_url: str,
        poll_interval: float,
        resume_delay: float,
    ) -> None:
        url = control_server_url.rstrip("/")
        self._client = httpx.AsyncClient(base_url=url, timeout=30.0)
        self._controller = _MiniFTController(
            get_cells=self._get_cells,
            suspend_cell=self._suspend_cell,
            resume_cell=self._resume_cell,
            poll_interval=poll_interval,
            resume_delay=resume_delay,
        )

    async def run(self) -> None:
        try:
            await self._controller.run()
        finally:
            await self._client.aclose()

    async def _get_cells(self) -> list[_CellSnapshot]:
        resp = await self._client.get("/api/v1/cells")
        resp.raise_for_status()
        cell_list = CellList.model_validate(resp.json())
        return [_compute_cell_snapshot(cell) for cell in cell_list.items]

    async def _suspend_cell(self, name: str) -> None:
        await self._patch_cell_suspend(name=name, suspend=True)

    async def _resume_cell(self, name: str) -> None:
        await self._patch_cell_suspend(name=name, suspend=False)

    async def _patch_cell_suspend(self, *, name: str, suspend: bool) -> None:
        patch = CellPatch(spec=CellPatchSpec(suspend=suspend))
        resp = await self._client.patch(
            f"/api/v1/cells/{name}",
            content=patch.model_dump_json(),
            headers={"Content-Type": "application/json"},
        )
        resp.raise_for_status()


def _compute_cell_snapshot(cell: Cell) -> _CellSnapshot:
    healthy_conditions = [c for c in cell.status.conditions if c.type == "Healthy"]
    if not healthy_conditions:
        status = CellHealthStatus.NOT_APPLICABLE
    elif any(c.status == TriState.FALSE for c in healthy_conditions):
        status = CellHealthStatus.UNHEALTHY
    else:
        status = CellHealthStatus.HEALTHY
    return _CellSnapshot(name=cell.metadata.name, status=status)


# ------------------------ data models ------------------------


class CellHealthStatus(str, Enum):
    HEALTHY = "Healthy"
    UNHEALTHY = "Unhealthy"
    NOT_APPLICABLE = "NotApplicable"


class _CellSnapshot(StrictBaseModel):
    name: str
    status: CellHealthStatus


@dataclass
class _CellBackoff:
    consecutive_failures: int = 0
    next_attempt_at: float = 0.0


# ------------------------ core controller (pure async, no HTTP) ------------------------


class _MiniFTController:
    def __init__(
        self,
        *,
        get_cells: Callable[[], Awaitable[list[_CellSnapshot]]],
        suspend_cell: Callable[[str], Awaitable[None]],
        resume_cell: Callable[[str], Awaitable[None]],
        poll_interval: float,
        resume_delay: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._get_cells = get_cells
        self._suspend_cell = suspend_cell
        self._resume_cell = resume_cell
        self._poll_interval = poll_interval
        self._resume_delay = resume_delay
        self._clock = clock

        self._running: bool = False
        self._cell_backoffs: dict[str, _CellBackoff] = {}

    async def run(self) -> None:
        try:
            self._running = True
            while self._running:
                start = self._clock()
                await self._poll_and_heal()
                elapsed = self._clock() - start
                await asyncio.sleep(max(0.0, self._poll_interval - elapsed))
        except Exception:
            logger.error("Error in run", exc_info=True)
            raise

    def request_stop(self) -> None:
        self._running = False

    async def _poll_and_heal(self) -> None:
        try:
            cells = await self._get_cells()

            unhealthy_names: set[str] = set()
            for cell in cells:
                if cell.status != CellHealthStatus.UNHEALTHY:
                    continue

                unhealthy_names.add(cell.name)
                backoff = self._cell_backoffs.setdefault(cell.name, _CellBackoff())

                now = self._clock()
                if now < backoff.next_attempt_at:
                    continue

                await self._heal(cell_name=cell.name, backoff=backoff)

            stale_keys = set(self._cell_backoffs) - unhealthy_names
            for key in stale_keys:
                del self._cell_backoffs[key]
        except Exception:
            logger.error("Error in _poll_and_heal", exc_info=True)

    async def _heal(self, *, cell_name: str, backoff: _CellBackoff) -> None:
        try:
            logger.info("Healing cell %s: suspending", cell_name)
            await self._suspend_cell(cell_name)

            logger.info("Healing cell %s: sleeping for resume_delay seconds", cell_name)
            await asyncio.sleep(self._resume_delay)

            logger.info("Healing cell %s: resuming", cell_name)
            await self._resume_cell(cell_name)

            backoff.consecutive_failures = 0
            backoff.next_attempt_at = self._clock() + self._resume_delay
            logger.info("Successfully healed cell %s, cooldown until %.0f", cell_name, backoff.next_attempt_at)
        except Exception:
            backoff.consecutive_failures += 1
            delay = min(5 * (2**backoff.consecutive_failures), 300)
            backoff.next_attempt_at = self._clock() + delay
            logger.warning(
                "Failed to heal cell %s (attempt %d), next attempt in %.0fs",
                cell_name,
                backoff.consecutive_failures,
                delay,
                exc_info=True,
            )
