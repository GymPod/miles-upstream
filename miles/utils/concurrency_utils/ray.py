import re

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from ray.util.state import list_nodes

from miles.utils.shell import exec_command


class Box:
    def __init__(self, inner):
        self._inner = inner

    @property
    def inner(self):
        return self._inner


def compute_ray_pin_head_options():
    head_node_id = _get_head_node_id()
    return {
        "scheduling_strategy": NodeAffinitySchedulingStrategy(
            node_id=head_node_id,
            soft=False,
        )
    }


def _get_head_node_id() -> str:
    for node in list_nodes():
        if node.is_head_node:
            return node.node_id
    raise RuntimeError("Could not find a head node in the Ray cluster")


def get_current_node_ip():
    address = ray._private.services.get_node_ip_address()
    # strip ipv6 address
    address = address.strip("[]")
    return address


@ray.remote(num_cpus=0.001)
def _exec_command_on_node(cmd: str, capture_output: bool) -> str | None:
    return exec_command(f"unset CUDA_VISIBLE_DEVICES; {cmd}", capture_output=capture_output)


def exec_command_all_ray_node(
    cmd: str, capture_output: bool = False, num_nodes: int | None = None
) -> list[str | None]:
    """Execute a shell command on every alive Ray node in parallel.

    Supported placeholders in `cmd` (replaced per-node before execution):
        {{node_rank}}   - 0-based index of the node
        {{nnodes}}      - total number of alive nodes (or num_nodes if specified)
        {{master_addr}} - NodeManagerAddress of the first node
        {{node_ip}}     - NodeManagerAddress of the current node

    Args:
        num_nodes: If set, only use the first `num_nodes` nodes instead of all alive nodes.
    """
    ray.init(address="auto")
    try:
        current_ip = get_current_node_ip()
        nodes = sorted(
            [n for n in ray.nodes() if n.get("Alive")],
            key=lambda n: (n["NodeManagerAddress"] != current_ip, n["NodeManagerAddress"]),
        )
        assert len(nodes) > 0

        if num_nodes is not None:
            assert num_nodes <= len(nodes), f"Requested {num_nodes} nodes but only {len(nodes)} alive nodes available."
            nodes = nodes[:num_nodes]

        master_addr = nodes[0]["NodeManagerAddress"]
        nnodes = str(len(nodes))

        placeholder_pattern = re.compile(
            "|".join(map(re.escape, ["{{node_rank}}", "{{nnodes}}", "{{master_addr}}", "{{node_ip}}"]))
        )

        refs = []
        for rank, node in enumerate(nodes):
            substitutions = {
                "{{node_rank}}": str(rank),
                "{{nnodes}}": nnodes,
                "{{master_addr}}": master_addr,
                "{{node_ip}}": node["NodeManagerAddress"],
            }
            node_cmd = placeholder_pattern.sub(lambda m, s=substitutions: s[m.group(0)], cmd)
            refs.append(
                _exec_command_on_node.options(
                    scheduling_strategy=NodeAffinitySchedulingStrategy(
                        node_id=node["NodeID"],
                        soft=False,
                    ),
                ).remote(node_cmd, capture_output=capture_output)
            )
        return ray.get(refs)
    finally:
        ray.shutdown()
