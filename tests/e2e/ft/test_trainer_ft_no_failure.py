# NOTE: You MUST read tests/e2e/ft/README.md as source-of-truth and documentations
# WARNING: Do NOT relax any assert logic in this file. All assertions must remain strict.

import sys
from pathlib import Path

_MILES_ROOT: Path = Path(__file__).resolve().parents[3]
_miles_root_str = str(_MILES_ROOT)
if _miles_root_str in sys.path:
    sys.path.remove(_miles_root_str)
sys.path.insert(0, _miles_root_str)

from tests.e2e.ft.conftest_ft.app import create_comparison_app
from tests.e2e.ft.conftest_ft.execution import get_common_train_args, get_ft_args
from tests.e2e.ft.conftest_ft.modes import FTTestMode

from miles.utils.test_utils.comparisons import (
    INPUT_TENSORS_ALLOW_FAILED_PATTERN,
    INPUT_TENSORS_SKIP_PATTERN,
    compare_dumps,
    compare_metrics,
)

NUM_STEPS: int = 2


def _build_baseline_args(mode: FTTestMode, dump_dir: str, enable_dumper: bool = True) -> str:
    return get_common_train_args(mode, dump_dir=dump_dir, num_steps=NUM_STEPS, enable_dumper=enable_dumper)


def _build_target_args(mode: FTTestMode, dump_dir: str, enable_dumper: bool = True) -> str:
    return get_common_train_args(
        mode, dump_dir=dump_dir, num_steps=NUM_STEPS, enable_dumper=enable_dumper
    ) + get_ft_args(mode)


def _compare(dump_dir: str, mode: FTTestMode) -> None:
    compare_metrics(
        baseline_dir=f"{dump_dir}/baseline",
        target_dir=f"{dump_dir}/target",
        rtol=1e-2,
        atol=1e-8,
        key_prefixes=["train/"],
        exclude_keys=[],
    )

    # Match by parallel identity (pp_rank, tp_rank, cp_rank, ep_rank) instead
    # of global rank, since baseline and target have different world sizes.
    compare_dumps(
        baseline_dir=f"{dump_dir}/baseline",
        target_dir=f"{dump_dir}/target",
        diff_thresholds=[(".*", "rel <= 0.0085")],
        allow_skipped_pattern=INPUT_TENSORS_SKIP_PATTERN,
        allow_failed_pattern=INPUT_TENSORS_ALLOW_FAILED_PATTERN,
        extra_args=["--grouping-skip-keys", "rank", "dp", "edp"],
    )
    print("No-failure comparison test PASSED")


app = create_comparison_app(
    test_name=Path(__file__).stem,
    build_baseline_args=_build_baseline_args,
    build_target_args=_build_target_args,
    compare_fn=_compare,
)

if __name__ == "__main__":
    app()
