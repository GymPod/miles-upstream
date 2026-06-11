from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

logger = logging.getLogger(__name__)

JsonDict = dict[str, Any]
KwargsDict = dict[str, str | int | float | None]


def _get_instruction_dict():
    """Return IFEvalG's instruction registry, imported lazily on first scoring.

    The registry pulls in nltk/langdetect/immutabledict at import time; those are
    optional extras (examples/eval_multi_task/requirements_ifeval.txt), so importing
    this module for its pure helpers — or under fast unit tests that lack the extras —
    must not trigger that import. Python caches the module, so the deferral is free
    after the first call. The package is vendored under IFEvalG/, so there is no
    network or pip side effect (unlike the previous official-checkout loader)."""

    from .IFEvalG import instructions_registry

    return instructions_registry.INSTRUCTION_DICT


def _remove_thinking_section(text: str) -> str:
    """Strip reasoning scaffolding so only the user-visible answer is verified.

    Reasoning-model rollouts wrap the answer in <think>...</think> and sometimes
    <answer> tags; the IFEval constraints must be checked against the answer a user
    would see, not the chain-of-thought that precedes it."""

    text = text.replace("<|assistant|>", "").strip()
    text = text.split("</think>")[-1]
    text = text.replace("<answer>", "").replace("</answer>", "")
    return text.strip()


def _normalize_instruction_ids(raw_ids: Sequence[Any]) -> list[str]:
    """Ensure instruction identifiers are clean strings."""

    normalized: list[str] = []
    for entry in raw_ids or []:
        if entry is None:
            continue
        text = str(entry).strip()
        if not text:
            continue
        normalized.append(text)
    return normalized


def _coerce_kwargs_list(raw_kwargs: Any, num_instructions: int) -> list[KwargsDict]:
    """Convert stored kwargs into the per-instruction list IFEvalG expects.

    None values are dropped so that ``build_description(**kwargs)`` only receives the
    keys a given instruction actually declares; the dataset stores every possible
    kwarg key per row with unused ones set to null."""

    if isinstance(raw_kwargs, list):
        processed: list[KwargsDict] = []
        for entry in raw_kwargs:
            if isinstance(entry, dict):
                processed.append(dict(entry))
            else:
                processed.append({})
    elif isinstance(raw_kwargs, dict):
        processed = [dict(raw_kwargs) for _ in range(num_instructions)]
    else:
        processed = [{} for _ in range(num_instructions)]

    if len(processed) < num_instructions:
        tail = processed[-1] if processed else {}
        processed.extend([dict(tail) for _ in range(num_instructions - len(processed))])
    elif len(processed) > num_instructions:
        processed = processed[:num_instructions]

    sanitized: list[KwargsDict] = []
    for entry in processed:
        sanitized.append({k: v for k, v in entry.items() if v is not None})
    return sanitized


def _loose_response_variants(answer: str) -> list[str]:
    """Response variants for the official IFEval ``loose`` criterion.

    Loose tolerates boilerplate that brackets an otherwise-compliant answer: a lead-in
    or sign-off line, and surrounding ``*`` markdown emphasis. An instruction passes if
    any variant satisfies it. This mirrors instruction_following_eval.evaluation_lib;
    IFEvalG itself ships only the per-instruction checkers, so we reproduce the variant
    set here to keep the loose score aligned with the official benchmark."""

    lines = answer.split("\n")
    remove_first = "\n".join(lines[1:]).strip()
    remove_last = "\n".join(lines[:-1]).strip()
    remove_both = "\n".join(lines[1:-1]).strip()
    return [
        answer,
        answer.replace("*", ""),
        remove_first,
        remove_last,
        remove_both,
        remove_first.replace("*", ""),
        remove_last.replace("*", ""),
        remove_both.replace("*", ""),
    ]


def _instruction_satisfied(instruction, candidates: Sequence[str]) -> bool:
    """Whether the instruction is followed by any candidate response variant.

    A checker that raises on pathological model output is treated as "not followed"
    (with the traceback logged), not propagated, so a single bad rollout cannot crash
    reward scoring for the whole batch."""

    for candidate in candidates:
        if not candidate.strip():
            continue
        try:
            if instruction.check_following(candidate):
                return True
        except Exception:
            logger.exception("IFEval checker %s raised; treating as not followed", instruction)
    return False


def compute_ifeval_reward(
    response: str, label: Any, metadata: JsonDict | None = None, *, strict: bool = True
) -> float:
    """Score a model response using IFEvalG rules (open-instruct IFEvalVerifier style).

    Sibling of compute_ifbench_reward. Constraints are read from ``metadata``
    (``instruction_id_list`` / ``kwargs``), the same schema the IFBench reward uses;
    ``label`` is accepted for interface parity but unused. Scoring is all-or-nothing:
    every instruction must be followed for a reward of 1.0. ``strict`` selects the
    criterion exposed as the ``ifeval_strict`` / ``ifeval_loose`` reward types — strict
    checks the raw answer, loose retries against formatting-stripped variants."""

    if metadata is None:
        logger.debug("No metadata provided for IFEval scoring.")
        return 0.0
    if not response:
        return 0.0

    instruction_ids = _normalize_instruction_ids(metadata.get("instruction_id_list") or [])
    if not instruction_ids:
        logger.debug("Missing instruction identifiers in metadata: %s", metadata)
        return 0.0

    kwargs_list = _coerce_kwargs_list(metadata.get("kwargs"), len(instruction_ids))

    answer = _remove_thinking_section(str(response))
    if not answer:
        return 0.0
    candidates = [answer] if strict else _loose_response_variants(answer)

    instruction_dict = _get_instruction_dict()
    # _coerce_kwargs_list pads/truncates kwargs to len(instruction_ids), so they pair 1:1.
    for instruction_id, kwargs in zip(instruction_ids, kwargs_list, strict=True):
        instruction_cls = instruction_dict.get(instruction_id)
        if instruction_cls is None:
            logger.warning("Unknown IFEval instruction id: %s", instruction_id)
            return 0.0
        instruction = instruction_cls(instruction_id)
        instruction.build_description(**kwargs)
        if not _instruction_satisfied(instruction, candidates):
            return 0.0
    return 1.0
