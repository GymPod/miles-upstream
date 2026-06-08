import logging
import math
import re
import subprocess
import sys
from pathlib import Path

from miles.utils.event_logger.logger import read_events
from miles.utils.event_logger.models import MetricEvent

logger = logging.getLogger(__name__)

_REQUIRED_METRIC_KEYS: list[str] = ["train/grad_norm", "train/loss"]

# Shared regexes for model-input / metadata tensors that are not weights or grads to
# compare. Exposed as named constants (not as defaults) so each test passes them
# explicitly -- every pass/fail knob is visible at the call site, nothing is implicit.
INPUT_TENSORS_SKIP_PATTERN: str = "input_ids|positions|cu_seqlens_q|cu_seqlens_kv|qkv_format|.*witness.*"
INPUT_TENSORS_ALLOW_FAILED_PATTERN: str = "input_ids|positions|cu_seqlens_q|cu_seqlens_kv|qkv_format"

# Per-tensor dump pass predicates for the tolerance comparison on the NORMAL (non-deterministic)
# path -- indep_dp vs normal DP run with different collective reduction orders, so this is not
# bitwise. Weights stay strict (rel <= 0.0085). Grads additionally pass when their absolute diff
# is tiny (max_abs <= 1e-3): near-zero grads (starved MoE experts, some layernorm grads) get
# their relative diff amplified by the differing reduction order (the diverging set varies run to
# run -> FP noise; observed max_abs ~1e-5..few e-4). Real grads (~1e-2) exceed the 1e-3 floor, so
# a real gradient bug still fails. An unmatched tensor is a fail-closed error (catch-all '.*').
TOLERANCE_DIFF_THRESHOLDS: list[tuple[str, str]] = [
    (r"grad__.*", "rel <= 0.0085 or max_abs <= 1e-3"),
    (".*", "rel <= 0.0085"),
]


def compare_dumps(
    baseline_dir: str,
    target_dir: str,
    *,
    diff_thresholds: list[tuple[str, str]],
    allow_skipped_pattern: str,
    allow_failed_pattern: str,
    phase_subdir: str | None = None,
    grouping_skip_keys: list[str] | None = None,
    extra_args: list[str] | None = None,
) -> None:
    subdir = phase_subdir or ""
    baseline_root = Path(baseline_dir) / "dumps" / subdir
    target_root = Path(target_dir) / "dumps" / subdir

    assert baseline_root.exists(), f"Baseline dump dir does not exist: {baseline_root}"
    assert target_root.exists(), f"Target dump dir does not exist: {target_root}"

    # Dumps are segmented into leaf dirs (e.g. fwd_bwd/rollout_<id>), each a flat set of
    # .pt files with its own per-leaf step numbering. The sglang comparator compares one
    # flat dir at a time, so compare each matching leaf pair independently.
    baseline_leaves = _find_leaf_dump_dirs(baseline_root)
    target_leaves = _find_leaf_dump_dirs(target_root)

    assert baseline_leaves, f"No .pt dump files found under {baseline_root}"
    assert baseline_leaves == target_leaves, (
        f"Dump leaf-dir mismatch: baseline={baseline_leaves} vs target={target_leaves} "
        f"(under {baseline_root} vs {target_root})"
    )

    failed_leaves: list[str] = []
    for leaf in baseline_leaves:
        # FT retries re-run a rollout in the same process: the failed attempt leaves
        # partial dumps at a lower dumper step and the successful (final) attempt lands
        # at the highest step. Compare only the final attempt against baseline's single
        # clean attempt -- the dump analog of compare_metrics._keep_only_final_attempt.
        target_steps = _distinct_dump_steps(target_root / leaf)
        final_step: int | None = None
        if len(target_steps) > 1:
            baseline_steps = _distinct_dump_steps(baseline_root / leaf)
            assert len(baseline_steps) == 1, (
                f"Baseline leaf {leaf} unexpectedly holds multiple dump steps "
                f"{sorted(baseline_steps)}; baseline has no FT retries (one attempt)."
            )
            final_step = max(target_steps)

        result = _run_comparator(
            baseline_path=baseline_root / leaf,
            target_path=target_root / leaf,
            diff_thresholds=diff_thresholds,
            allow_skipped_pattern=allow_skipped_pattern,
            allow_failed_pattern=allow_failed_pattern,
            final_step=final_step,
            grouping_skip_keys=grouping_skip_keys,
            extra_args=extra_args,
        )
        if result.returncode != 0:
            failed_leaves.append(leaf)

    assert not failed_leaves, (
        f"Dump comparator failed (rc!=0) for {len(failed_leaves)}/{len(baseline_leaves)} leaf dir(s): "
        f"{failed_leaves} (baseline {baseline_root} vs target {target_root}). The comparator applies the "
        f"per-tensor predicates ({diff_thresholds}) and the allow/skip patterns itself; see "
        f"comparator_report.jsonl under {target_root}/<leaf> for the offending tensors."
    )
    print(f"Dump comparison passed: {len(baseline_leaves)} leaf dir(s) under {baseline_root} vs {target_root}")


def _find_leaf_dump_dirs(root: Path) -> list[str]:
    leaves: set[str] = {str(p.parent.relative_to(root)) for p in root.rglob("*.pt")}
    return sorted(leaves)


def _distinct_dump_steps(leaf_dir: Path) -> set[int]:
    """Distinct dumper ``step`` values among the .pt files directly in ``leaf_dir``."""
    steps: set[int] = set()
    for p in leaf_dir.glob("*.pt"):
        m = re.search(r"step=(\d+)", p.name)
        if m:
            steps.add(int(m.group(1)))
    return steps


def compare_metrics(
    baseline_dir: str,
    target_dir: str,
    *,
    rtol: float,
    atol: float,
    key_prefixes: list[str],
    exclude_keys: list[str],
) -> None:
    baseline_events = _read_metric_events(Path(baseline_dir))
    target_events = _read_metric_events(Path(target_dir))

    # FT retries (healing path) leave events from earlier failed attempts. Only
    # the highest-attempt events per rollout_id reflect the successful run.
    baseline_events = _keep_only_final_attempt(baseline_events)
    target_events = _keep_only_final_attempt(target_events)

    issues: list[str] = []
    issues += _check_event_counts(baseline_events, target_events, baseline_dir, target_dir)

    if not issues:
        for step_idx, (b_event, t_event) in enumerate(zip(baseline_events, target_events, strict=True)):
            _print_step_comparison_table(step_idx, b_event, t_event, key_prefixes, exclude_keys=exclude_keys)
            issues += _check_step_metrics(
                step_idx, b_event, t_event, key_prefixes, rtol, atol=atol, exclude_keys=exclude_keys
            )

    issues += _check_required_keys_exist(baseline_events)

    assert not issues, f"MetricEvent comparison found {len(issues)} issue(s):\n" + "\n".join(
        f"  - {i}" for i in issues
    )
    print(f"MetricEvent comparison passed: {len(baseline_events)} steps compared")


def _keep_only_final_attempt(events: list[MetricEvent]) -> list[MetricEvent]:
    """Keep only events from the highest-attempt for each rollout_id.

    During FT healing, a crashed rollout is retried at attempt+1; events from
    the failed attempt are partial and should be discarded for comparison.

    Rollout-side metrics (e.g. RolloutManager log_rollout_metrics) have
    attempt=None — they are not part of the FT retry stream, so we treat them
    as a single attempt (normalized to 0).
    """

    def _attempt(e: MetricEvent) -> int:
        return e.attempt if e.attempt is not None else 0

    max_attempt_by_rollout: dict[int, int] = {
        rollout_id: max(_attempt(e) for e in events if e.rollout_id == rollout_id)
        for rollout_id in {e.rollout_id for e in events}
    }
    return [e for e in events if _attempt(e) == max_attempt_by_rollout[e.rollout_id]]


def _check_event_counts(
    baseline: list[MetricEvent],
    target: list[MetricEvent],
    baseline_dir: str,
    target_dir: str,
) -> list[str]:
    issues: list[str] = []
    if len(baseline) == 0:
        issues.append(f"No MetricEvents found in baseline dir: {baseline_dir}")
    if len(target) == 0:
        issues.append(f"No MetricEvents found in target dir: {target_dir}")
    if len(baseline) > 0 and len(target) > 0 and len(baseline) != len(target):
        issues.append(f"MetricEvent count mismatch: baseline={len(baseline)}, target={len(target)}")
    return issues


def _check_step_metrics(
    step_idx: int,
    baseline_event: MetricEvent,
    target_event: MetricEvent,
    key_prefixes: list[str],
    rtol: float,
    *,
    atol: float,
    exclude_keys: list[str] | None = None,
) -> list[str]:
    issues: list[str] = []
    for key in baseline_event.metrics:
        if not any(key.startswith(prefix) for prefix in key_prefixes):
            continue
        if exclude_keys and key in exclude_keys:
            continue

        if key not in target_event.metrics:
            issues.append(f"Step {step_idx}: metric '{key}' present in baseline but missing in target")
            continue

        issues += _check_single_metric(
            step_idx, key, baseline_event.metrics[key], target_event.metrics[key], rtol, atol=atol
        )
    return issues


def _check_single_metric(
    step_idx: int,
    key: str,
    baseline_val: object,
    target_val: object,
    rtol: float,
    atol: float,
) -> list[str]:
    if not isinstance(baseline_val, (int, float)) or not isinstance(target_val, (int, float)):
        return []

    if math.isnan(baseline_val) or math.isnan(target_val):
        return [f"Step {step_idx}, metric '{key}': NaN detected (baseline={baseline_val}, target={target_val})"]
    if math.isinf(baseline_val) or math.isinf(target_val):
        if baseline_val != target_val:
            return [f"Step {step_idx}, metric '{key}': inf mismatch (baseline={baseline_val}, target={target_val})"]
        return []

    if baseline_val == 0.0 and target_val == 0.0:
        return []

    abs_diff = abs(baseline_val - target_val)
    if abs_diff <= atol:
        return []

    rel_diff = abs_diff / max(abs(baseline_val), abs(target_val), 1e-12)
    if rel_diff > rtol:
        return [
            f"Step {step_idx}, metric '{key}': baseline={baseline_val}, target={target_val}, "
            f"rel_diff={rel_diff:.6f} > rtol={rtol}"
        ]
    return []


def _print_step_comparison_table(
    step_idx: int,
    baseline_event: MetricEvent,
    target_event: MetricEvent,
    key_prefixes: list[str],
    *,
    exclude_keys: list[str] | None = None,
) -> None:
    import polars as pl
    from sglang.srt.debug_utils.comparator.display import _render_polars_as_text

    rows: list[dict[str, str]] = []
    for key in sorted(baseline_event.metrics):
        if not any(key.startswith(p) for p in key_prefixes):
            continue
        b_val = baseline_event.metrics[key]
        t_val = target_event.metrics.get(key)
        if not isinstance(b_val, (int, float)) or t_val is None or not isinstance(t_val, (int, float)):
            continue
        excluded = "(excluded)" if exclude_keys and key in exclude_keys else ""
        abs_diff = abs(b_val - t_val)
        denom = max(abs(b_val), abs(t_val), 1e-12)
        rel_diff = abs_diff / denom
        rows.append(
            {
                "metric": key,
                "baseline": f"{b_val:.6e}",
                "target": f"{t_val:.6e}",
                "abs_diff": f"{abs_diff:.2e}",
                "rel_diff": f"{rel_diff:.4%}{excluded}",
            }
        )

    if not rows:
        return
    df = pl.DataFrame(rows)
    print(_render_polars_as_text(df, title=f"Step {step_idx} metric comparison"))


def _check_required_keys_exist(events: list[MetricEvent]) -> list[str]:
    all_keys: set[str] = set()
    for event in events:
        all_keys.update(event.metrics.keys())

    issues: list[str] = []
    for required in _REQUIRED_METRIC_KEYS:
        if required not in all_keys:
            issues.append(
                f"Required metric '{required}' not found in any baseline MetricEvent. "
                f"Available keys: {sorted(all_keys)}"
            )
    return issues


def _run_comparator(
    *,
    baseline_path: Path,
    target_path: Path,
    diff_thresholds: list[tuple[str, str]],
    allow_skipped_pattern: str,
    allow_failed_pattern: str,
    final_step: int | None = None,
    grouping_skip_keys: list[str] | None = None,
    extra_args: list[str] | None,
) -> subprocess.CompletedProcess[str]:
    # Skip 'rank' when grouping bundles: under FT (target) and non-FT (baseline) the same
    # logical (pp_rank, cp_rank, ep_rank, tp_rank) coordinate maps to a different absolute
    # rank ID (e.g. baseline rank=4 vs target cell0 rank=2 for PP=1, CP=0). Without skipping
    # 'rank' the comparator gets `baseline_load_failed` for every tensor and fails with rc=1.
    # Callers may pass extra keys (e.g. no_failure skips 'dp'/'edp' too). When restricting to
    # a final FT-retry step (below), also skip 'step': the target's final attempt lands at a
    # higher step than baseline's single attempt, so step must not be a grouping key.
    # (Grouping is a comparator-matching detail, not a pass/fail threshold.)
    skip_keys: list[str] = list(grouping_skip_keys) if grouping_skip_keys is not None else ["rank"]
    if final_step is not None and "step" not in skip_keys:
        skip_keys.append("step")

    cmd: list[str] = [
        sys.executable,
        "-m",
        "sglang.srt.debug_utils.comparator",
        "--baseline-path",
        str(baseline_path),
        "--target-path",
        str(target_path),
        "--output-format",
        "json",
        "--grouping-skip-keys",
        *skip_keys,
        "--allow-skipped-pattern",
        allow_skipped_pattern,
        "--allow-failed-pattern",
        allow_failed_pattern,
    ]
    if final_step is not None:
        # _read_df filters only the target side by step; restrict it to the final attempt
        # so the failed attempt's partial dumps are dropped. Baseline (single attempt) is
        # unfiltered and aligns via the skipped 'step' grouping key above.
        cmd.extend(["--start-step", str(final_step), "--end-step", str(final_step)])
    if extra_args:
        cmd.extend(extra_args)
    # Keep --diff-threshold strictly last: its nargs="*" greedily consumes every
    # following token, so no flag with a bare value may come after it.
    cmd.append("--diff-threshold")
    for pattern, predicate in diff_thresholds:
        cmd.extend([pattern, predicate])

    result: subprocess.CompletedProcess[str] = subprocess.run(
        cmd,
        text=True,
    )

    return result


def _read_metric_events(dump_dir: Path) -> list[MetricEvent]:
    """Read all MetricEvents from the events directory."""
    events_dir: Path = dump_dir / "events"
    if not events_dir.exists():
        return []
    all_events = read_events(events_dir)
    return [e for e in all_events if isinstance(e, MetricEvent)]
