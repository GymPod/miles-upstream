import logging
from pathlib import Path

import torch

from miles.utils.types import Sample

logger = logging.getLogger(__name__)


def load_debug_rollout_data(args, rollout_id: int):
    path = Path(args.load_debug_rollout_data.format(rollout_id=rollout_id))
    if not path.exists():
        # The recorded debug rollouts are a finite set (one .pt per rollout). A soak that runs more
        # steps than there are files (e.g. the random-failure test) reuses them cyclically; the
        # rollout content is irrelevant to what a soak asserts (FT survival across crashes), so wrap
        # the index by the number of available files.
        available = sorted(int(p.stem) for p in path.parent.glob("*.pt") if p.stem.isdigit())
        if not available:
            raise FileNotFoundError(f"No debug rollout data files found in {path.parent}")
        path = path.with_name(f"{available[rollout_id % len(available)]}.pt")
    data = torch.load(path, weights_only=False)["samples"]
    data = [Sample.from_dict(sample) for sample in data]
    if (ratio := args.load_debug_rollout_data_subsample) is not None:
        original_num_rows = len(data)
        rough_subsample_num_rows = int(original_num_rows * ratio)
        data = data[: rough_subsample_num_rows // 2] + data[-rough_subsample_num_rows // 2 :]
        logger.info(
            f"Subsample loaded debug rollout data using {ratio=} and change num rows {original_num_rows} -> {len(data)}"
        )
    return data


def save_debug_rollout_data(args, data, rollout_id, evaluation: bool):
    # TODO to be refactored (originally Buffer._set_data)
    if (path_template := args.save_debug_rollout_data) is not None:
        path = Path(path_template.format(rollout_id=("eval_" if evaluation else "") + str(rollout_id)))
        logger.info(f"Save debug rollout data to {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

        # TODO may improve the format
        if evaluation:
            dump_data = dict(
                samples=[sample.to_dict() for dataset_name, info in data.items() for sample in info["samples"]]
            )
        else:
            dump_data = dict(
                samples=[sample.to_dict() for sample in data],
            )

        torch.save(dict(rollout_id=rollout_id, **dump_data), path)
