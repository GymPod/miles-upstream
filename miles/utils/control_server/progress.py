from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainingProgress:
    """Mutable holder for the rollout currently being trained.

    Lives in the same process as training, so the control server reads an
    authoritative current rollout id when deciding whether to allow fault
    injection. ``None`` until the first ``train()`` sets it.
    """

    current_rollout_id: int | None = None
