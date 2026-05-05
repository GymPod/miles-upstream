import logging
from collections.abc import Sequence
from datetime import timedelta

import torch

try:
    from torchft.checkpointing.pg_transport import PGTransport
except ImportError:
    PGTransport = None

from megatron.core.dist_checkpointing.tensor_aware_state_dict import MCoreTensorAwareStateDict

from miles.backends.megatron_utils.in_memory_checkpoint import InMemoryCheckpointManager, save_to_memory
from miles.utils.process_group_utils import GroupInfo

logger = logging.getLogger(__name__)

# Must accommodate receiver's model init time (can take minutes for large models)
_DEFAULT_TIMEOUT = timedelta(seconds=600)


def send_ckpt(
    *,
    indep_dp: GroupInfo,
    model: Sequence,
    optimizer: object,
    opt_param_scheduler: object,
    iteration: int,
    dst_rank: int,
    timeout: timedelta = _DEFAULT_TIMEOUT,
) -> None:
    """Send in-memory checkpoint to a destination cell via torchft PGTransport.

    Args:
        indep_dp: Independent DP group info (provides the torchft PG).
        model: Megatron model chunks.
        optimizer: Megatron optimizer.
        opt_param_scheduler: LR scheduler.
        iteration: Current training iteration / rollout_id.
        dst_rank: Destination alive_rank in the indep_dp process group.
        timeout: Timeout for the NCCL send operation.
    """
    state_dict = save_to_memory(
        iteration=iteration,
        model=model,
        optimizer=optimizer,
        opt_param_scheduler=opt_param_scheduler,
    )

    payload = _serialize_for_transport(state_dict=state_dict, iteration=iteration)

    transport = _create_transport(indep_dp, timeout)
    transport.send_checkpoint(
        dst_ranks=[dst_rank],
        step=0,
        state_dict=payload,
        timeout=timeout,
    )
    transport.disallow_checkpoint()
    logger.info(f"Sent checkpoint (iteration={iteration}) to alive_rank={dst_rank}")


def recv_ckpt(
    *,
    indep_dp: GroupInfo,
    src_rank: int,
    timeout: timedelta = _DEFAULT_TIMEOUT,
) -> InMemoryCheckpointManager:
    """Receive checkpoint from a healthy cell via torchft PGTransport.

    Returns an InMemoryCheckpointManager containing the received state_dict,
    ready to be passed to initialize_model_and_optimizer.

    Args:
        indep_dp: Independent DP group info (provides the torchft PG).
        src_rank: Source alive_rank in the indep_dp process group.
        timeout: Timeout for the NCCL recv operation.

    Returns:
        InMemoryCheckpointManager with state_dict loaded, ready for
        initialize_model_and_optimizer to consume.
    """
    transport = _create_transport(indep_dp, timeout)
    payload = transport.recv_checkpoint(
        src_rank=src_rank,
        metadata=transport.metadata(),
        step=0,
        timeout=timeout,
    )

    iteration, state_dict = _deserialize_from_transport(payload)
    logger.info(f"Received checkpoint (iteration={iteration}) from alive_rank={src_rank}")

    manager = InMemoryCheckpointManager()
    manager.save(state_dict, iteration=iteration)
    return manager


def _serialize_for_transport(
    *,
    state_dict: MCoreTensorAwareStateDict,
    iteration: int,
) -> dict[str, object]:
    tensors: list[torch.Tensor] = state_dict.pop_tensors()
    # torchft PGTransport._cast_tensor uses `type(t) is torch.Tensor` (strict),
    # which rejects torch.nn.Parameter. Detach into plain Tensors that share
    # storage but pass the type check.
    tensors = [t.detach() if type(t) is not torch.Tensor else t for t in tensors]
    total_bytes = sum(t.nbytes for t in tensors)
    logger.info(
        "[OOM_DEBUG] _serialize_for_transport: %d tensors, total %.2f GB",
        len(tensors),
        total_bytes / 1024 / 1024 / 1024,
    )
    return {
        "tensors": tensors,
        "hollow_state_dict": state_dict,
        "iteration": iteration,
    }


def _deserialize_from_transport(
    payload: dict[str, object],
) -> tuple[int, MCoreTensorAwareStateDict]:
    iteration: int = payload["iteration"]
    hollow_state_dict: MCoreTensorAwareStateDict = payload["hollow_state_dict"]
    tensors: list[torch.Tensor] = payload["tensors"]

    hollow_state_dict.insert_tensors(tensors)
    return iteration, hollow_state_dict


def _create_transport(indep_dp: GroupInfo, timeout: timedelta) -> PGTransport:
    return PGTransport(
        pg=indep_dp.group,
        timeout=timeout,
        device=torch.device("cuda"),
    )
