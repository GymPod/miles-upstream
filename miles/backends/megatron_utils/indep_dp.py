import logging
import os
from collections.abc import Sequence
from datetime import timedelta
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from miles.utils.distributed_utils import get_gloo_group
from miles.utils.indep_dp import IndepDPInfo
from miles.utils.process_group_utils import GeneralPGUtil, GroupInfo, collective_bool_and

from ..training_utils.parallel import ParallelState

if TYPE_CHECKING:
    from megatron.core.distributed import DistributedDataParallel as DDP

logger = logging.getLogger(__name__)


def create_indep_dp_group(
    store_addr: str | None,
    indep_dp_info: IndepDPInfo,
    megatron_rank: int,
    megatron_world_size: int,
) -> GroupInfo:
    if indep_dp_info.alive_size <= 1:
        return GroupInfo(rank=0, size=1, group=None)

    try:
        from torchft.process_group import ProcessGroupGloo, ProcessGroupNCCL
    except ImportError as e:
        raise ImportError("torchft is required for indep_dp. Install with: pip install torchft") from e

    _TIMEOUT = timedelta(seconds=120)

    def _create(pg_cls: type, backend_name: str) -> dist.ProcessGroup:
        pg = pg_cls(timeout=_TIMEOUT)
        pg.configure(
            store_addr=f"{store_addr}/indep_dp/{backend_name}/{indep_dp_info.quorum_id}/{megatron_rank}",
            replica_id=str(indep_dp_info.cell_index),
            rank=indep_dp_info.alive_rank,
            world_size=indep_dp_info.alive_size,
            quorum_id=indep_dp_info.quorum_id,
            group_rank=megatron_rank,
            group_world_size=megatron_world_size,
        )
        return pg

    nccl_pg = _create(ProcessGroupNCCL, "nccl")
    gloo_pg = _create(ProcessGroupGloo, "gloo")
    logger.info(
        f"Configured independent DP PG: {indep_dp_info}, "
        f"megatron_rank={megatron_rank}, megatron_world_size={megatron_world_size}"
    )
    _debug_membership_probe(
        pg=nccl_pg, expected_members=indep_dp_info.alive_size, cell_rank=indep_dp_info.alive_rank, where="postcreate"
    )
    return GroupInfo(rank=indep_dp_info.alive_rank, size=indep_dp_info.alive_size, group=nccl_pg, gloo_group=gloo_pg)


def reconfigure_indep_dp_group(
    parallel_state: ParallelState,
    store_addr: str | None,
    indep_dp_info: IndepDPInfo,
    megatron_rank: int,
    megatron_world_size: int,
) -> None:
    """Abort old indep_dp PGs and create new ones with a fresh quorum_id."""
    old = parallel_state.indep_dp
    for g in [old.group, old.gloo_group]:
        if g is not None:
            g.abort(errored=False)

    parallel_state.indep_dp = create_indep_dp_group(
        store_addr=store_addr,
        indep_dp_info=indep_dp_info,
        megatron_rank=megatron_rank,
        megatron_world_size=megatron_world_size,
    )
    logger.info(f"Reconfigured indep_dp PG with quorum_id={indep_dp_info.quorum_id}")


def _allreduce_grads_across_replicas(args, model: Sequence["DDP"], parallel_state: ParallelState) -> bool:
    assert not args.calculate_per_token_loss, "calculate_per_token_loss is not supported with indep_dp yet"
    assert parallel_state.intra_dp.size == 1, (
        f"indep_dp requires intra_dp.size == 1, got {parallel_state.intra_dp.size}. "
        "Simultaneous intra and indep DP is not supported."
    )

    pg = parallel_state.indep_dp.group
    util = GeneralPGUtil.create(pg)

    _debug_membership_probe(
        pg=pg,
        expected_members=parallel_state.indep_dp.size,
        cell_rank=parallel_state.indep_dp.rank,
        where="pre_grad_reduce",
    )

    allreduce_success = True
    try:
        for model_chunk in model:
            # mimic: DistributedDataParallel.start_grad_sync
            for bucket_group in model_chunk.bucket_groups + model_chunk.expert_parallel_bucket_groups:
                for bucket in bucket_group.buckets:
                    util.all_reduce(bucket.grad_data, pg, op=dist.ReduceOp.SUM)
    except Exception:
        allreduce_success = False
        logger.exception(
            "indep_dp cross-cell gradient allreduce raised (cell_rank=%d, expected_members=%d)",
            parallel_state.indep_dp.rank,
            parallel_state.indep_dp.size,
        )

    # pg.errored() can force a CUDA/stream sync, so call it exactly once per step here -- do NOT
    # sprinkle extra errored() probes. When it does report an async error it MUST be logged loudly:
    # a swallowed cross-cell error means an un-reduced (wrong) gradient would be applied silently.
    if (e := pg.errored()) is not None:
        allreduce_success = False
        logger.error(
            "indep_dp cross-cell PG async error (cell_rank=%d, expected_members=%d): %s",
            parallel_state.indep_dp.rank,
            parallel_state.indep_dp.size,
            e,
        )

    # Intra-cell consensus: if ANY rank's allreduce failed, ALL ranks discard.
    # get_gloo_group() is cell-local (created from the default world PG).
    return collective_bool_and(value=allreduce_success, group=get_gloo_group())


# Debug-only membership probe, gated off by default. A degraded (single-member) cross-cell
# communicator turns every all_reduce into a legal no-op: it neither raises nor sets
# pg.errored(), so the loud-logging paths above can never fire for it and an un-reduced
# gradient gets applied silently. The probe all_reduces ones(1) and compares the result with
# the expected member count, which catches that silent degradation at the exact step it
# happens. The .item() readback is a CUDA sync per call, which is why this must stay off
# (zero-cost) outside debug runs.
_DEBUG_MEMBERSHIP_PROBE_ENV_VAR = "MILES_FT_DEBUG_INDEP_DP_PROBE"


def _debug_membership_probe(pg: dist.ProcessGroup, expected_members: int, cell_rank: int, where: str) -> None:
    if not bool(int(os.environ.get(_DEBUG_MEMBERSHIP_PROBE_ENV_VAR, "0"))):
        return

    try:
        probe = torch.ones(1, dtype=torch.float32, device="cuda")
        GeneralPGUtil.create(pg).all_reduce(probe, pg, op=dist.ReduceOp.SUM)
        observed = probe.item()
    except Exception:
        logger.exception(
            "indep_dp membership probe raised (where=%s, cell_rank=%d, expected_members=%d)",
            where,
            cell_rank,
            expected_members,
        )
        return

    if observed == float(expected_members):
        logger.info(
            "indep_dp membership probe ok (where=%s, cell_rank=%d, members=%d)", where, cell_rank, expected_members
        )
        return

    # The extra errored() call (and its potential CUDA sync) is acceptable here: this whole
    # path only runs in env-gated debug mode and the comm is already known-degraded.
    logger.error(
        "indep_dp membership probe DEGRADED (where=%s, cell_rank=%d, expected_members=%d, observed_sum=%s, "
        "pg.errored()=%s)",
        where,
        cell_rank,
        expected_members,
        observed,
        pg.errored(),
    )
