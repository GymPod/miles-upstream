import logging
from collections.abc import Sequence
from datetime import timedelta
from typing import TYPE_CHECKING

import torch.distributed as dist

from miles.utils.distributed_utils import get_gloo_group
from miles.utils.indep_dp import IndepDPInfo
from miles.utils.process_group_utils import GeneralPGUtil, GroupInfo, collective_bool_and

from ..training_utils.parallel import ParallelState

if TYPE_CHECKING:
    from megatron.core.distributed import DistributedDataParallel as DDP

logger = logging.getLogger(__name__)

# Args to recreate the current cross-cell comm, set only by reconfigure_indep_dp_group. A
# reconfigured (post-recovery) comm degrades to single-member during the compute-interleaved
# forward (an NCCL 2.28 issue, not a timeout -- it degrades even well within the comm timeout), so
# maybe_refresh_reconfigured_comm recreates it before each reduction. The initial comm is never
# degraded and is never recreated, which also keeps its short crash-recovery timeout intact.
_INDEP_DP_RECREATE_ARGS: dict | None = None


def create_indep_dp_group(
    store_addr: str | None,
    indep_dp_info: IndepDPInfo,
    megatron_rank: int,
    megatron_world_size: int,
    timeout_s: float = 120,
    reuse_gloo: "dist.ProcessGroup | None" = None,
) -> GroupInfo:
    if indep_dp_info.alive_size <= 1:
        return GroupInfo(rank=0, size=1, group=None)

    try:
        from torchft.process_group import ProcessGroupGloo, ProcessGroupNCCL
    except ImportError as e:
        raise ImportError("torchft is required for indep_dp. Install with: pip install torchft") from e

    _TIMEOUT = timedelta(seconds=timeout_s)

    if __import__("os").environ.get("MILES_SANITY_INDEPDP"):
        logger.warning(
            "INDEPDP_RENDEZVOUS base_store=%r quorum=%s mr=%s alive_rank=%s alive_size=%s cell=%s",
            store_addr,
            indep_dp_info.quorum_id,
            megatron_rank,
            indep_dp_info.alive_rank,
            indep_dp_info.alive_size,
            indep_dp_info.cell_index,
        )

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

    # Create the gloo PG first and synchronize all alive cells BEFORE the NCCL rendezvous.
    # torchft's NCCL PG is nonblocking (opts.config.blocking=False) and eager-connects, so if
    # the cells reach NCCL init at different times (e.g. the healing cell is slower to rebuild
    # after a rejoin) the comm forms single-member (nranks=1, confirmed via NCCL_DEBUG) and
    # every cross-cell collective silently no-ops -- the rejoin-step gradient/metric reduction
    # is then skipped and a wrong (un-reduced) gradient is applied. A blocking gloo all_reduce
    # gates the NCCL rendezvous until every alive cell is present, so the NCCL comm forms with
    # all members. This is always entered collectively by all alive cells (alive_size > 1) and
    # cannot deadlock on a mid-crash peer (crashed cells are killed before this reconfigure).
    # Recreating the gloo PG on every post-recovery reduction leaks CPU memory and churns the store
    # (it crashed the actor with an OOM/EOF). The gloo PG is CPU-based and does NOT degrade during
    # the GPU forward, so callers that only need to refresh the corrupted NCCL comm pass the existing
    # gloo PG via reuse_gloo to keep it across recreations.
    gloo_pg = reuse_gloo if reuse_gloo is not None else _create(ProcessGroupGloo, "gloo")
    _barrier_via_gloo(gloo_pg)
    nccl_pg = _create(ProcessGroupNCCL, "nccl")
    if __import__("os").environ.get("MILES_SANITY_INDEPDP"):
        import torch

        g = torch.ones(1)
        GeneralPGUtil.create(gloo_pg).all_reduce(g, gloo_pg, op=dist.ReduceOp.SUM)
        n = torch.ones(1, device="cuda")
        GeneralPGUtil.create(nccl_pg).all_reduce(n, nccl_pg, op=dist.ReduceOp.SUM)
        logger.warning(
            "INDEPDP_POSTCREATE quorum=%s mr=%s alive_size=%s gloo_size=%s nccl_size=%s gloo_ar=%s nccl_ar=%s",
            indep_dp_info.quorum_id,
            megatron_rank,
            indep_dp_info.alive_size,
            gloo_pg.size(),
            nccl_pg.size(),
            g.item(),
            n.item(),
        )
    logger.info(
        f"Configured independent DP PG: {indep_dp_info}, "
        f"megatron_rank={megatron_rank}, megatron_world_size={megatron_world_size}"
    )
    return GroupInfo(rank=indep_dp_info.alive_rank, size=indep_dp_info.alive_size, group=nccl_pg, gloo_group=gloo_pg)


def _barrier_via_gloo(gloo_pg: dist.ProcessGroup) -> None:
    import torch

    GeneralPGUtil.create(gloo_pg).all_reduce(torch.ones(1), gloo_pg, op=dist.ReduceOp.SUM)


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

    # A reconfigured comm is always created during recovery, where a cell that just respawned runs a
    # cold torch.compile of its first forward (~180-300s, variable) while the survivor waits on this
    # comm. Use a long timeout so that wait cannot expire (which would silently degrade the comm to
    # single-member and apply an un-reduced gradient). Only the initial comm (create_indep_dp_group's
    # short default) needs the fast timeout that makes a crash recoverable -- crashes happen during
    # normal training on the initial comm, never on a reconfigured one mid-recovery.
    parallel_state.indep_dp = create_indep_dp_group(
        store_addr=store_addr,
        indep_dp_info=indep_dp_info,
        megatron_rank=megatron_rank,
        megatron_world_size=megatron_world_size,
        timeout_s=1500,
    )
    global _INDEP_DP_RECREATE_ARGS
    _INDEP_DP_RECREATE_ARGS = dict(
        store_addr=store_addr,
        indep_dp_info=indep_dp_info,
        megatron_rank=megatron_rank,
        megatron_world_size=megatron_world_size,
    )
    logger.info(f"Reconfigured indep_dp PG with quorum_id={indep_dp_info.quorum_id}")


def maybe_refresh_reconfigured_comm(parallel_state: ParallelState, rollout_id: int, attempt: int) -> None:
    """Recreate a reconfigured cross-cell comm right before it is used for a reduction.

    A reconfigured comm degrades to single-member during the compute-interleaved forward (NCCL 2.28;
    verified not a timeout -- it degrades even when the step is well within the comm timeout), which
    would silently skip the cross-cell gradient/metric reduction and apply an un-reduced gradient. A
    freshly-created comm survives a single reduction, so recreate it here; the step's later metric
    all_reduce reuses this fresh comm too (no forward runs in between). The store sub-key is derived
    from controller-synced (quorum_id, rollout_id, attempt) so all alive cells rendezvous on the same
    key. Only reconfigured comms are refreshed (the initial comm is never degraded). The long timeout
    tolerates a respawned cell still finishing its (cache-warmed ~120s) recompile.
    """
    args = _INDEP_DP_RECREATE_ARGS
    if args is None:
        return

    old = parallel_state.indep_dp
    quorum_id = args["indep_dp_info"].quorum_id
    # Reuse the persistent gloo PG (CPU, does not degrade) and recreate ONLY the NCCL comm. Creating
    # a fresh gloo every step leaked memory and crashed the actor (OOM/EOF) after several rejoins.
    fresh = create_indep_dp_group(
        store_addr=f"{args['store_addr']}/recreate/{quorum_id}_{rollout_id}_{attempt}",
        indep_dp_info=args["indep_dp_info"],
        megatron_rank=args["megatron_rank"],
        megatron_world_size=args["megatron_world_size"],
        timeout_s=1500,
        reuse_gloo=old.gloo_group,
    )

    parallel_state.indep_dp = fresh
    # Abort only the old NCCL comm; the gloo PG is reused (kept alive in `fresh`). Then free the
    # aborted comm's references promptly so repeated rejoins do not accumulate GPU/CPU memory.
    if old.group is not None:
        old.group.abort(errored=False)
    import gc

    gc.collect()


def _allreduce_grads_across_replicas(
    args, model: Sequence["DDP"], parallel_state: ParallelState, rollout_id: int, attempt: int
) -> bool:
    assert not args.calculate_per_token_loss, "calculate_per_token_loss is not supported with indep_dp yet"
    assert parallel_state.intra_dp.size == 1, (
        f"indep_dp requires intra_dp.size == 1, got {parallel_state.intra_dp.size}. "
        "Simultaneous intra and indep DP is not supported."
    )

    maybe_refresh_reconfigured_comm(parallel_state, rollout_id, attempt)

    pg = parallel_state.indep_dp.group
    util = GeneralPGUtil.create(pg)

    if __import__("os").environ.get("MILES_SANITY_INDEPDP"):
        import torch

        _s = torch.ones(1, device="cuda")
        util.all_reduce(_s, pg, op=dist.ReduceOp.SUM)
        logger.warning(
            "INDEPDP_SANITY mr=%s size=%s rank=%s all_reduce(ones)=%s (expect alive_size)",
            dist.get_rank(),
            parallel_state.indep_dp.size,
            parallel_state.indep_dp.rank,
            _s.item(),
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
        logger.exception("Gradient allreduce across replicas failed")

    if (e := pg.errored()) is not None:
        allreduce_success = False
        logger.error("indep_dp PG has async error: %s", e)

    # Intra-cell consensus: if ANY rank's allreduce failed, ALL ranks discard.
    # get_gloo_group() is cell-local (created from the default world PG).
    return collective_bool_and(value=allreduce_success, group=get_gloo_group())
