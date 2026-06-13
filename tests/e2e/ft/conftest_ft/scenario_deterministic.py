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
from miles.utils.test_utils.reconfigure_assertions import ExpectedReconfigure, assert_reconfigure_events

NUM_PHASE_A_STEPS: int = 1
# --num-rollout value; phase_b resumes from the phase_a ckpt and executes rollouts
# [NUM_PHASE_A_STEPS, NUM_PHASE_B_STEPS). With 4, rollouts 1..3 run: stop/start fires
# at the end of rollout 2, so healing executes at the start of rollout 3 and rollout 3
# trains with the healed cell. With 3, healing would never run (nothing after rollout 2).
NUM_PHASE_B_STEPS: int = 4

_DETERMINISTIC_ENV_VARS: str = (
    '--train-env-vars \'{"NCCL_ALGO": "Ring", '
    '"NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0", '
    '"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}\' '
)

# rollout_id in phase_b starts from NUM_PHASE_A_STEPS (ckpt resume offset)
_DETERMINISTIC_ACTIONS: list[dict] = [
    {"at_rollout": NUM_PHASE_A_STEPS + 1, "action": "stop_cell_at_end", "cell_index": -1},
    {"at_rollout": NUM_PHASE_A_STEPS + 1, "action": "start_cell_at_end", "cell_index": -1},
]


# Pinned to the current phase timeline (FT actions only run on target/phase_b); update if
# it changes. The stop+start is absorbed by one _refresh_cells, so exactly one healing and
# no standalone shrink.
def _expected_reconfigures(*, is_target: bool, phase: str, num_cells: int) -> list[ExpectedReconfigure]:
    if not (is_target and phase == "phase_b"):
        return []
    return [
        ExpectedReconfigure(
            rollout_id=NUM_PHASE_A_STEPS + 2,
            src_cell_index=0,
            healed_cell_indices=[num_cells - 1],
            alive_cell_indices_after=list(range(num_cells)),
        ),
    ]


def _build_phase_args(mode: FTTestMode, dump_dir: str, *, is_target: bool, enable_dumper: bool = True) -> str:
    is_phase_a: bool = dump_dir.endswith("phase_a")
    base = get_common_train_args(mode, dump_dir=dump_dir, num_steps=NUM_PHASE_B_STEPS, enable_dumper=enable_dumper)
    base += "--deterministic-mode " + _DETERMINISTIC_ENV_VARS
    base += "--debug-deterministic-collective "

    if is_target:
        base += get_ft_args(mode)

    if is_phase_a:
        base += f"--save {dump_dir}/ckpt --save-interval 1 "
        base += f"--debug-exit-after-rollout {NUM_PHASE_A_STEPS} "
    else:
        phase_a_dir = dump_dir.replace("/phase_b", "/phase_a")
        base += f"--load {phase_a_dir}/ckpt "
        if is_target:
            base += f"--ci-ft-test-actions '{json.dumps(_DETERMINISTIC_ACTIONS)}' "

    return base


def _build_baseline_args(mode: FTTestMode, dump_dir: str, enable_dumper: bool = True) -> str:
    return _build_phase_args(mode, dump_dir, is_target=False, enable_dumper=enable_dumper)


def _build_target_args(mode: FTTestMode, dump_dir: str, enable_dumper: bool = True) -> str:
    return _build_phase_args(mode, dump_dir, is_target=True, enable_dumper=enable_dumper)


def _compare(dump_dir: str, mode: FTTestMode) -> None:
    # Bitwise (zero-tolerance) comparison. The deterministic healing test exists to
    # prove that state pulled from another replica during healing is reconstructed
    # *bit-for-bit*: a state-copy bug is trivial to introduce and an approximate
    # ("looks close") check would silently miss it. So every assertion is exact --
    # all metrics must be equal (rtol=atol=0) and every dumped tensor must match
    # bitwise (predicate "rel <= 0" for every tensor, no near-zero tolerance).
    # Sole exception: train/grad_norm sums squared shard fragments, so its
    # bracketing depends on the distributed-optimizer shard count (8 in the flat
    # baseline vs 2 per FT cell) -- a few fp32 ulps of drift are inherent to
    # comparing different shardings, and the grads themselves are still compared
    # bitwise by compare_dumps below. It gets a tight non-zero gate instead.
    #
    # This requires the run to be fully deterministic on both sides.
    # Any divergence is a real bug and must be fixed at the source, never hidden by
    # loosening these thresholds.
    # Witness the FT path executed before comparing numerics, so two fault-free runs cannot pass.
    for side in ["baseline", "target"]:
        for phase in PHASES:
            assert_reconfigure_events(
                Path(f"{dump_dir}/{side}/{phase}/events"),
                expected=_expected_reconfigures(is_target=side == "target", phase=phase, num_cells=mode.num_cells),
            )

    grad_norm_key = "train/grad_norm"
    compare_metrics(
        baseline_dir=f"{dump_dir}/baseline/phase_b",
        target_dir=f"{dump_dir}/target/phase_b",
        rtol=0.0,
        atol=0.0,
        key_prefixes=["train/"],
        exclude_keys=[grad_norm_key],
    )
    compare_metrics(
        baseline_dir=f"{dump_dir}/baseline/phase_b",
        target_dir=f"{dump_dir}/target/phase_b",
        rtol=1e-6,
        atol=0.0,
        key_prefixes=[grad_norm_key],
        exclude_keys=[],
    )
    phase_b_rollout_ids = range(NUM_PHASE_A_STEPS, NUM_PHASE_B_STEPS)
    expected_leaves = {f"fwd_bwd/rollout_{rollout_id}" for rollout_id in phase_b_rollout_ids}
    actual_leaves = {
        str(p.parent.relative_to(Path(f"{dump_dir}/baseline/phase_b/dumps")))
        for p in Path(f"{dump_dir}/baseline/phase_b/dumps").rglob("*.pt")
    }
    assert actual_leaves == expected_leaves, (
        f"Dump leaves {actual_leaves} do not match the expected phase_b rollouts {expected_leaves}; "
        f"the post-healing rollout must be present or healing was never exercised"
    )

    compare_dumps(
        baseline_dir=f"{dump_dir}/baseline/phase_b",
        target_dir=f"{dump_dir}/target/phase_b",
        diff_thresholds=[(".*", "rel <= 0")],
        allow_skipped_pattern=INPUT_TENSORS_SKIP_PATTERN,
        allow_failed_pattern=INPUT_TENSORS_ALLOW_FAILED_PATTERN,
    )
    print("Deterministic healing comparison test PASSED")


TEST_NAME: str = "trainer_ft_deterministic"
PHASES: list[str] = ["phase_a", "phase_b"]


app, run_ci = create_comparison_app_and_run_ci(
    test_name=TEST_NAME,
    build_baseline_args=_build_baseline_args,
    build_target_args=_build_target_args,
    compare_fn=_compare,
    phases=PHASES,
)

if __name__ == "__main__":
    app()
