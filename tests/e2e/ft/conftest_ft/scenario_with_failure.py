# NOTE: You MUST read tests/e2e/ft/README.md as source-of-truth and documentations
# WARNING: Do NOT relax any assert logic in this file. All assertions must remain strict.

import json
from pathlib import Path

from tests.e2e.ft.conftest_ft.app import create_comparison_app_and_run_ci
from tests.e2e.ft.conftest_ft.execution import get_common_train_args, get_ft_args
from tests.e2e.ft.conftest_ft.modes import FTTestMode

from miles.utils.test_utils.comparisons import (
    INPUT_TENSORS_ALLOW_FAILED_PATTERN,
    INPUT_TENSORS_SKIP_PATTERN,
    compare_dumps,
    compare_metrics,
)

# Both phases execute the same operations (one shared builder, parameterized only by the
# phase's start rollout id): NUM_ROLLOUTS_PER_PHASE rollouts, the same relative fault
# timeline, and a ckpt save after every rollout. Only the start regime differs: phase_a
# cold-starts (no --load => start_rollout_id=0), phase_b resumes from phase_a's last ckpt
# (start_rollout_id = loaded_rollout_id + 1 = NUM_ROLLOUTS_PER_PHASE).
#
# Rollout ids are global across phases: phase_a executes [0, NUM_ROLLOUTS_PER_PHASE) and
# phase_b executes [NUM_ROLLOUTS_PER_PHASE, TOTAL_NUM_ROLLOUTS). Off-by-one convention:
# --num-rollout is the exclusive end rollout id (TOTAL_NUM_ROLLOUTS for both phases), NOT
# a per-run count; each phase instead stops after its NUM_ROLLOUTS_PER_PHASE rollouts via
# --debug-exit-after-rollout, which counts rollouts executed in the current run and fires
# after that rollout's ckpt save.
NUM_ROLLOUTS_PER_PHASE: int = 3
TOTAL_NUM_ROLLOUTS: int = 2 * NUM_ROLLOUTS_PER_PHASE
PHASE_START_ROLLOUT_IDS: dict[str, int] = {"phase_a": 0, "phase_b": NUM_ROLLOUTS_PER_PHASE}

# First rollout id whose target-side state carries the fault-inherent ulp drift of a
# degraded-quorum commit: the phase_a fault hits rollout 1 (= phase_a start + 1), so the
# target's weights drift from rollout 2 onward, and that drift persists into phase_b
# through the ckpt (phase_b resumes from the drift-carrying phase_a ckpt).
_FIRST_POST_FAULT_ROLLOUT_ID: int = PHASE_START_ROLLOUT_IDS["phase_a"] + 2

# Per-tensor pass predicates. A few specific near-zero grads diverge under the
# crash-recovery (solo / degraded-quorum) collective's reduction order while their
# weights stay bit-identical:
#   - starved MoE expert grads (grad__...mlp.experts.*), max_abs ~1e-5..4e-4;
#   - attention q/k_layernorm grads (grad__...q_layernorm.* / ...k_layernorm.*, the two
#     QK-norm weights), measured rel ~1.1-1.5%, max_abs_diff ~6e-5..2.3e-4, failing
#     layer varies per rollout -> FP noise.
# Both are cancellation-dominated near-zero grads, so they also tolerate
# max_abs <= 1e-3 (well below real grads ~1e-2): a real diff there still fails, and
# everything else stays strict via the catch-all (an unmatched tensor is fail-closed).
_DIFF_THRESHOLDS: list[tuple[str, str]] = [
    (r"grad__.*\.mlp\.experts\..*", "rel <= 0.0085 or max_abs <= 1e-3"),
    (r"grad__.*\.[qk]_layernorm\..*", "rel <= 0.0085 or max_abs <= 1e-3"),
    (".*", "rel <= 0.0085"),
]

# Post-fault (injected) rollouts in the real_rollout mode: training data is injected to be
# bitwise-identical, but the target's weights carry the fault-inherent ulp drift of the
# degraded-quorum commit. On the converged dense model that drift lands in the
# cancellation-dominated near-zero grads of the decoder-layer norms and attention/MLP
# matrices as absolute noise measured <= 2.8e-3 (40 tensors, 2026-06-12; q_layernorm up to
# rel 20% at max_abs 2.6e-3) while real grads sit at ~1e-2 — only those measured families
# get a 3e-3 floor. Everything else (embeddings, output layer, final norm, all
# activations/values) stays strict, and all passed at rel <= 0.85% in the same run.
_POST_FAULT_DIFF_THRESHOLDS: list[tuple[str, str]] = [
    (r"grad__.*\.[qk]_layernorm\..*", "rel <= 0.0085 or max_abs <= 3e-3"),
    (r"grad__.*\.layer_norm_weight", "rel <= 0.0085 or max_abs <= 3e-3"),
    (r"grad__.*\.self_attention\.linear_qkv\.weight", "rel <= 0.0085 or max_abs <= 3e-3"),
    (r"grad__.*\.self_attention\.linear_proj\.weight", "rel <= 0.0085 or max_abs <= 3e-3"),
    (r"grad__.*\.mlp\.linear_fc[12]\.weight", "rel <= 0.0085 or max_abs <= 3e-3"),
    (".*", "rel <= 0.0085"),
]


def _build_actions(phase_start_rollout_id: int) -> list[dict]:
    # The fault hits the phase's second rollout: crash at attempt 0 -> degraded-quorum
    # commit at attempt 1, then stop/start at the end of that rollout so healing executes
    # at the start of the phase's third (last) rollout, which trains with the healed cell.
    fault_rollout_id: int = phase_start_rollout_id + 1
    return [
        {
            "at_rollout": fault_rollout_id,
            "action": "crash_before_allreduce",
            "cell_index": -1,
            "rank": 0,
            "attempt": 0,
        },
        {"at_rollout": fault_rollout_id, "action": "stop_cell_at_end", "cell_index": -1},
        {"at_rollout": fault_rollout_id, "action": "start_cell_at_end", "cell_index": -1},
    ]


def _build_phase_args(mode: FTTestMode, dump_dir: str, *, is_target: bool, enable_dumper: bool = True) -> str:
    phase_name: str = dump_dir.rsplit("/", 1)[-1]
    phase_start_rollout_id: int = PHASE_START_ROLLOUT_IDS[phase_name]

    base = get_common_train_args(mode, dump_dir=dump_dir, num_steps=TOTAL_NUM_ROLLOUTS, enable_dumper=enable_dumper)

    if is_target:
        base += get_ft_args(mode)

    base += f"--save {dump_dir}/ckpt --save-interval 1 "
    base += f"--debug-exit-after-rollout {NUM_ROLLOUTS_PER_PHASE} "
    if phase_name != "phase_a":
        phase_a_dir = dump_dir.replace("/phase_b", "/phase_a")
        base += f"--load {phase_a_dir}/ckpt "

    if is_target:
        base += f"--ci-ft-test-actions '{json.dumps(_build_actions(phase_start_rollout_id))}' "
        if mode.has_real_rollout:
            # Post-fault rollouts inject the baseline's recorded data (see README). The
            # baseline records every phase's rollout data under its own dump dir, so the
            # injection path mirrors this phase's dump dir on the baseline side. In
            # phase_a injection starts right after the fault; in phase_b every rollout is
            # injected because all of phase_b's generation runs on drift-carrying weights.
            baseline_dump_dir = dump_dir.replace("/target/", "/baseline/")
            inject_start_rollout_id = max(phase_start_rollout_id, _FIRST_POST_FAULT_ROLLOUT_ID)
            base += (
                f"--ci-inject-rollout-data-path {baseline_dump_dir}/rollout_data/{{rollout_id}}.pt "
                f"--ci-inject-rollout-data-start-rollout-id {inject_start_rollout_id} "
                "--ci-inject-rollout-data-min-match-ratio 0.5 "
            )

    return base


def _build_baseline_args(mode: FTTestMode, dump_dir: str, enable_dumper: bool = True) -> str:
    return _build_phase_args(mode, dump_dir, is_target=False, enable_dumper=enable_dumper)


def _build_target_args(mode: FTTestMode, dump_dir: str, enable_dumper: bool = True) -> str:
    return _build_phase_args(mode, dump_dir, is_target=True, enable_dumper=enable_dumper)


def _assert_dump_leaves(side_dir: str, side: str, phase: str, expected_leaves: set[str]) -> None:
    dumps_root = Path(f"{side_dir}/dumps")
    actual_leaves = {str(p.parent.relative_to(dumps_root)) for p in dumps_root.rglob("*.pt")}
    assert actual_leaves == expected_leaves, (
        f"{phase} {side}: dump leaves {actual_leaves} do not match the per-rollout comparison loop "
        f"{expected_leaves}; an extra or missing leaf would silently skip comparison"
    )


def _compare(dump_dir: str, mode: FTTestMode) -> None:
    # Both phases are compared: phase_a exercises fault + healing on a cold-started run,
    # phase_b exercises the same timeline after resuming from phase_a's post-FT ckpt.
    for phase, phase_start_rollout_id in PHASE_START_ROLLOUT_IDS.items():
        baseline_dir = f"{dump_dir}/baseline/{phase}"
        target_dir = f"{dump_dir}/target/{phase}"
        compare_metrics(
            baseline_dir=baseline_dir,
            target_dir=target_dir,
            rtol=5e-2,
            atol=1e-7,
            key_prefixes=["train/"],
            exclude_keys=[],
        )

        phase_rollout_ids = range(phase_start_rollout_id, phase_start_rollout_id + NUM_ROLLOUTS_PER_PHASE)
        expected_leaves = {f"fwd_bwd/rollout_{rollout_id}" for rollout_id in phase_rollout_ids}
        _assert_dump_leaves(side_dir=baseline_dir, side="baseline", phase=phase, expected_leaves=expected_leaves)
        _assert_dump_leaves(side_dir=target_dir, side="target", phase=phase, expected_leaves=expected_leaves)

        for rollout_id in phase_rollout_ids:
            is_post_fault = mode.has_real_rollout and rollout_id >= _FIRST_POST_FAULT_ROLLOUT_ID
            compare_dumps(
                baseline_dir=baseline_dir,
                target_dir=target_dir,
                diff_thresholds=_POST_FAULT_DIFF_THRESHOLDS if is_post_fault else _DIFF_THRESHOLDS,
                allow_skipped_pattern=INPUT_TENSORS_SKIP_PATTERN,
                allow_failed_pattern=INPUT_TENSORS_ALLOW_FAILED_PATTERN,
                phase_subdir=f"fwd_bwd/rollout_{rollout_id}",
            )
    print("With-failure comparison test PASSED")


TEST_NAME: str = "trainer_ft_with_failure"
PHASES: list[str] = list(PHASE_START_ROLLOUT_IDS)


app, run_ci = create_comparison_app_and_run_ci(
    test_name=TEST_NAME,
    build_baseline_args=_build_baseline_args,
    build_target_args=_build_target_args,
    compare_fn=_compare,
    phases=PHASES,
)

if __name__ == "__main__":
    app()
