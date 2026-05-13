def should_run_periodic_action(
    rollout_id: int,
    interval: int | None,
    num_rollout_per_epoch: int | None = None,
    num_rollout: int | None = None,
) -> bool:
    """
    Return True when a periodic action (eval/save/checkpoint) should run.

    Args:
        rollout_id: The current rollout index (0-based).
        interval: Desired cadence; disables checks when None.
        num_rollout_per_epoch: Optional epoch boundary to treat as a trigger.
    """
    if interval is None:
        return False

    if num_rollout is not None and rollout_id == num_rollout - 1:
        return True

    step = rollout_id + 1
    return (step % interval == 0) or (num_rollout_per_epoch is not None and step % num_rollout_per_epoch == 0)
