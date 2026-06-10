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

# Per-process args needed to recreate the current reconfigured cross-cell comm (see
# maybe_refresh_reconfigured_comm). Set whenever a >1-member indep_dp comm is created.
_INDEP_DP_RECREATE_ARGS: dict | None = None


def create_indep_dp_group(
    store_addr: str | None,
    indep_dp_info: IndepDPInfo,
    megatron_rank: int,
    megatron_world_size: int,
    recreate_tag: str = "",
) -> GroupInfo:
    if indep_dp_info.alive_size <= 1:
        return GroupInfo(rank=0, size=1, group=None)

    try:
        from torchft.process_group import ProcessGroupGloo, ProcessGroupNCCL
    except ImportError as e:
        raise ImportError("torchft is required for indep_dp. Install with: pip install torchft") from e

    _TIMEOUT = timedelta(seconds=120)

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
            store_addr=f"{store_addr}{recreate_tag}/indep_dp/{backend_name}/{indep_dp_info.quorum_id}/{megatron_rank}",
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
    gloo_pg = _create(ProcessGroupGloo, "gloo")
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
    global _INDEP_DP_RECREATE_ARGS
    _INDEP_DP_RECREATE_ARGS = dict(
        store_addr=store_addr,
        indep_dp_info=indep_dp_info,
        megatron_rank=megatron_rank,
        megatron_world_size=megatron_world_size,
    )
    logger.info(
        f"Configured independent DP PG: {indep_dp_info}, "
        f"megatron_rank={megatron_rank}, megatron_world_size={megatron_world_size}"
    )
    return GroupInfo(rank=indep_dp_info.alive_rank, size=indep_dp_info.alive_size, group=nccl_pg, gloo_group=gloo_pg)


def _barrier_via_gloo(gloo_pg: dist.ProcessGroup) -> None:
    import torch

    GeneralPGUtil.create(gloo_pg).all_reduce(torch.ones(1), gloo_pg, op=dist.ReduceOp.SUM)


def maybe_refresh_reconfigured_comm(parallel_state: ParallelState, rollout_id: int, attempt: int) -> None:
    """Recreate a reconfigured cross-cell comm right before it is used for a reduction.

    NCCL 2.28 workaround: a cross-cell torchft comm that was reconfigured after a rejoin
    (quorum_id > 0) is created correctly (verified nranks=2 right after create) but silently
    degrades to single-member DURING the rejoin step's compute-interleaved forward/backward --
    its subsequent all_reduce then no-ops (each cell reduces with itself), so the cross-cell
    gradient and metric reductions are skipped and a wrong un-reduced gradient is applied. The
    initial quorum_0 comm is unaffected, so this only refreshes reconfigured comms.

    A freshly-created comm survives a single reduction, so we recreate it immediately before the
    grad reduction; the later metric all_reduce in the same step reuses this fresh comm too (no
    forward runs in between). The store sub-key is derived from controller-assigned, cell-synced
    (quorum_id, rollout_id, attempt) so every alive cell rendezvous on the same key (a per-process
    counter would desync across a cell stop/start and hang the rendezvous).
    """
    args = _INDEP_DP_RECREATE_ARGS
    if args is None or args["indep_dp_info"].quorum_id == 0:
        return

    old = parallel_state.indep_dp
    for g in [old.group, old.gloo_group]:
        if g is not None:
            g.abort(errored=False)

    quorum_id = args["indep_dp_info"].quorum_id
    parallel_state.indep_dp = create_indep_dp_group(
        store_addr=args["store_addr"],
        indep_dp_info=args["indep_dp_info"],
        megatron_rank=args["megatron_rank"],
        megatron_world_size=args["megatron_world_size"],
        recreate_tag=f"/recreate/{quorum_id}_{rollout_id}_{attempt}",
    )


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


def _allreduce_grads_across_replicas(
    args, model: Sequence["DDP"], parallel_state: ParallelState, rollout_id: int, attempt: int
) -> bool:
    assert not args.calculate_per_token_loss, "calculate_per_token_loss is not supported with indep_dp yet"
    assert parallel_state.intra_dp.size == 1, (
        f"indep_dp requires intra_dp.size == 1, got {parallel_state.intra_dp.size}. "
        "Simultaneous intra and indep DP is not supported."
    )

    # NCCL 2.28 workaround: refresh a reconfigured (post-rejoin) cross-cell comm that the forward
    # pass corrupted to single-member, so this grad reduction (and the step's later metric reduce)
    # run over an uncorrupted comm. No-op for the unaffected initial quorum_0 comm.
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
