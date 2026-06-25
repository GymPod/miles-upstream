"""
Phase-1 OpenEnv smoke adapter (HuggingFace OpenEnv <-> miles).

Drop-in replacement for ``swe-agent-v2/swe_agent_function.py``: a custom agent
function for ``miles.rollout.generate_hub.agentic_tool_call.generate`` selected
via ``--custom-agent-function-path``.

Instead of POSTing to a Harbor agent server, this drives an OpenEnv environment
(the hosted ``echo_env``) directly from inside the agent loop:

  1. reset() the OpenEnv environment.
  2. Ask the *trained policy* (served by miles, reachable at ``base_url/v1``) to
     produce a message. These tokens flow through the session server, so miles
     captures token ids + logprobs + loss masks natively -- no re-tokenization.
  3. step() the env with the model's output; read the env-provided reward.
  4. Return env metadata; the generate layer merges it into sample.metadata and
     the task-agnostic reward_func in ``swe-agent-v2/generate.py`` reads
     metadata["reward"].

This validates the integration *seam* end to end. Echo's reward is trivial, so
this proves the wiring (steps advance, tokens captured, reward flows) -- it does
NOT exercise learning. Phase 2 swaps Echo for a real env (e.g. the Coding env in
local Docker) using the same adapter shape.

Env vars:
  OPENENV_ENV_URL       base_url of the OpenEnv environment server
                        (default: hosted Echo space)
  AGENT_MODEL_NAME      model name to send to the policy (default: "model")
  MILES_ROUTER_EXTERNAL_HOST  optional host rewrite for off-cluster agents
                        (same semantics as swe_agent_function.py)
"""

import asyncio
import logging
import os
import random
from typing import Any
from urllib.parse import urlparse, urlunparse

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

_DEFAULT_ECHO_URL = "https://openenv-echo-env.hf.space"

# The hosted Echo space serves a single MCP session at a time
# (CAPACITY_REACHED). Episodes hold the session only for the tiny reset+step
# round-trip, so retrying with jittered backoff serializes them cheaply.
_CAPACITY_MAX_WAIT_S = 180.0
_CAPACITY_BACKOFF_S = (0.25, 1.5)


def _resolve_session_url(base_url: str) -> str:
    """Build the OpenAI-compatible policy URL, rewriting host for off-cluster agents."""
    session_url = f"{base_url}/v1"
    external_host = os.getenv("MILES_ROUTER_EXTERNAL_HOST")
    if external_host:
        parsed = urlparse(session_url)
        netloc = f"{external_host}:{parsed.port}" if parsed.port else external_host
        session_url = urlunparse(parsed._replace(netloc=netloc))
    return session_url


def _extract_messages(prompt: Any) -> list[dict[str, str]]:
    """Accept either a chat-message list or a raw string prompt."""
    if isinstance(prompt, list):
        return prompt
    return [{"role": "user", "content": str(prompt)}]


async def _run_episode(EchoEnv: Any, CallToolAction: Any, env_url: str, action_text: str) -> float:
    """reset()+step() the Echo env, retrying while the single session is busy."""
    deadline = asyncio.get_event_loop().time() + _CAPACITY_MAX_WAIT_S
    while True:
        try:
            async with EchoEnv(base_url=env_url) as env:
                await env.reset()
                result = await env.step(
                    CallToolAction(tool_name="echo_message", arguments={"message": action_text})
                )
                return float(getattr(result, "reward", 0.0) or 0.0)
        except Exception as e:
            if "CAPACITY_REACHED" in str(e) and asyncio.get_event_loop().time() < deadline:
                await asyncio.sleep(random.uniform(*_CAPACITY_BACKOFF_S))
                continue
            raise


async def run(
    base_url: str,
    prompt: Any,
    request_kwargs: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    **kwargs,
) -> dict[str, Any] | None:
    """Run one OpenEnv Echo episode driven by the trained policy."""
    # Imported lazily so the file is importable without the env client installed.
    from echo_env import CallToolAction, EchoEnv

    request_kwargs = request_kwargs or {}
    metadata = metadata or {}

    session_url = _resolve_session_url(base_url)
    model_name = os.getenv("AGENT_MODEL_NAME", os.getenv("SWE_AGENT_MODEL_NAME", "model"))
    env_url = os.getenv("OPENENV_ENV_URL", _DEFAULT_ECHO_URL)

    policy = AsyncOpenAI(base_url=session_url, api_key="EMPTY")

    try:
        # Generate the policy action FIRST, with no env session held -- so the
        # (slow) LLM calls for every episode run concurrently through sglang.
        # miles' sampling params (top_k, min_tokens, ...) are sglang-specific and
        # not accepted as top-level kwargs by the OpenAI client, so forward them
        # via extra_body (merged into the request body sglang reads).
        messages = _extract_messages(prompt)
        completion = await policy.chat.completions.create(
            model=model_name,
            messages=messages,
            extra_body=request_kwargs,
        )
        action_text = completion.choices[0].message.content or ""

        # Only the tiny reset+step round-trip holds the single Echo session.
        reward = await _run_episode(EchoEnv, CallToolAction, env_url, action_text)
    except Exception as e:
        logger.error(f"OpenEnv Echo episode failed: {e}", exc_info=True)
        return None

    return {
        "reward": reward,
        "exit_status": "completed",
        "eval_report": {},
        "agent_metrics": {"turns": 1, "tool_calls": 1},
    }
