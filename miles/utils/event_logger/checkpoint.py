"""Snapshot/restore the event directory alongside model checkpoints.

Event files are append-only and shared by every process of a run, and they can run ahead
of the last saved checkpoint (a crash after rollout N with the last checkpoint at rollout
M < N leaves events for rollouts the resumed run will re-execute). Versioning a snapshot
of the event directory with each checkpoint and restoring it on load keeps the event
history exactly consistent with the loaded model state, so event consumers (the event
analyzer, the witness id allocator) behave as if the run was never interrupted.

Snapshots live under ``{save}/debug_events/iter_<iteration>/``, next to (not inside)
Megatron's ``iter_<iteration>`` checkpoint directories.
"""

import logging
import shutil
from argparse import Namespace
from pathlib import Path

logger = logging.getLogger(__name__)

_TRACKER_FILENAME = "latest_checkpointed_iteration.txt"


def snapshot_events(args: Namespace, iteration: int) -> None:
    """Copy the live event dir into the checkpoint tree. Called after a checkpoint save."""
    if args.save_debug_event_data is None or args.save is None:
        return

    src = Path(args.save_debug_event_data)
    if not src.is_dir():
        return

    dst = _snapshot_dir(Path(args.save), iteration)
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)
    logger.info("Snapshotted event dir %s -> %s", src, dst)


def restore_events(args: Namespace) -> None:
    """Replace the live event dir with the loaded checkpoint's snapshot.

    Must run before any process of the run opens an event file (event files are opened
    in append mode and kept open). No-op when not resuming or when the checkpoint
    predates event snapshots.
    """
    if args.save_debug_event_data is None or args.load is None:
        return

    iteration = _read_tracker_iteration(Path(args.load))
    if iteration is None:
        return

    src = _snapshot_dir(Path(args.load), iteration)
    if not src.is_dir():
        return

    # Write through the existing files instead of replacing the directory: processes that
    # already opened their event file keep an O_APPEND handle, which stays valid across an
    # in-place truncate (every append re-seeks to EOF) but not across an unlink.
    dst = Path(args.save_debug_event_data)
    dst.mkdir(parents=True, exist_ok=True)
    snapshot_files = {p.name: p for p in src.iterdir() if p.is_file()}
    live_files = {p.name for p in dst.iterdir() if p.is_file()}
    for name, path in snapshot_files.items():
        (dst / name).write_bytes(path.read_bytes())
    for name in live_files - set(snapshot_files):
        (dst / name).write_bytes(b"")
    logger.info("Restored event dir %s <- %s", dst, src)


def _snapshot_dir(checkpoint_root: Path, iteration: int) -> Path:
    return checkpoint_root / "debug_events" / f"iter_{iteration:07d}"


def _read_tracker_iteration(checkpoint_root: Path) -> int | None:
    tracker = checkpoint_root / _TRACKER_FILENAME
    if not tracker.is_file():
        return None

    content = tracker.read_text().strip()
    if not content.isdigit():
        return None
    return int(content)
