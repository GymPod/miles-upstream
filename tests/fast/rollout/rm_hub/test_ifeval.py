"""Tests for the IFEval reward wrapper (miles.rollout.rm_hub.ifeval).

Two layers:

* Pure-helper tests (metadata parsing, thinking-section stripping, loose-variant
  generation) run everywhere with no external deps — they pin down the wrapper logic
  this module owns.
* The ``@requires_ifevalg`` cases drive the real vendored IFEvalG checkers end to end
  (strict / loose scoring, dispatch routing). They are skipped only when IFEvalG's
  extras (nltk / langdetect / immutabledict) are not installed, since fast CI may lack
  them; on a fully-provisioned env they are the real correctness check.
"""

from __future__ import annotations

import functools
import importlib.util
from unittest.mock import MagicMock

import pytest

from miles.rollout.rm_hub import async_rm, ifeval
from miles.utils.async_utils import run
from miles.utils.types import Sample


class TestNormalizeInstructionIds:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            (["keywords:existence", "punctuation:no_comma"], ["keywords:existence", "punctuation:no_comma"]),
            (
                [" keywords:existence ", "", None, "punctuation:no_comma"],
                ["keywords:existence", "punctuation:no_comma"],
            ),
            ([1, 2], ["1", "2"]),
            ([], []),
            (None, []),
        ],
    )
    def test_normalize(self, raw, expected):
        assert ifeval._normalize_instruction_ids(raw) == expected


class TestCoerceKwargsList:
    def test_list_passthrough(self):
        assert ifeval._coerce_kwargs_list([{"keywords": ["a"]}], 1) == [{"keywords": ["a"]}]

    def test_dict_broadcast(self):
        assert ifeval._coerce_kwargs_list({"num_words": 3}, 2) == [{"num_words": 3}, {"num_words": 3}]

    def test_none_or_other_yields_empty_dicts(self):
        assert ifeval._coerce_kwargs_list(None, 2) == [{}, {}]
        assert ifeval._coerce_kwargs_list("nonsense", 1) == [{}]

    def test_non_dict_entries_become_empty(self):
        assert ifeval._coerce_kwargs_list([{"a": 1}, "bad"], 2) == [{"a": 1}, {}]

    def test_padding_repeats_tail(self):
        assert ifeval._coerce_kwargs_list([{"a": 1}], 3) == [{"a": 1}, {"a": 1}, {"a": 1}]

    def test_truncation(self):
        assert ifeval._coerce_kwargs_list([{"a": 1}, {"b": 2}, {"c": 3}], 2) == [{"a": 1}, {"b": 2}]

    def test_none_values_dropped(self):
        assert ifeval._coerce_kwargs_list([{"keep": 1, "drop": None}], 1) == [{"keep": 1}]


class TestRemoveThinkingSection:
    """The constraints must be checked against the user-visible answer, so reasoning
    scaffolding (<think>, <answer>, <|assistant|>) is stripped before verification."""

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("<think>reasoning, with a comma</think>\nFinal answer", "Final answer"),
            ("<answer>hello there</answer>", "hello there"),
            ("<|assistant|>plain text", "plain text"),
            ("no tags at all", "no tags at all"),
            # Only content after the *last* </think> survives.
            ("<think>a</think>mid<think>b</think>tail", "tail"),
        ],
    )
    def test_strip(self, raw, expected):
        assert ifeval._remove_thinking_section(raw) == expected


class TestLooseResponseVariants:
    def test_includes_raw_and_markdown_stripped(self):
        variants = ifeval._loose_response_variants("*bold answer*")
        assert "*bold answer*" in variants
        assert "bold answer" in variants

    def test_includes_first_and_last_line_removed(self):
        variants = ifeval._loose_response_variants("intro line\nmiddle\nsign-off")
        assert "middle\nsign-off" in variants  # first line removed
        assert "intro line\nmiddle" in variants  # last line removed
        assert "middle" in variants  # both removed


# --- Wrapper guards (no IFEvalG import needed: they return before scoring) ----------


class TestComputeIfevalRewardGuards:
    def test_none_metadata_returns_zero(self):
        assert ifeval.compute_ifeval_reward("anything", None, metadata=None) == 0.0

    def test_none_response_returns_zero(self):
        metadata = {"instruction_id_list": ["keywords:existence"]}
        assert ifeval.compute_ifeval_reward(None, None, metadata=metadata) == 0.0

    def test_missing_instruction_ids_returns_zero(self):
        assert ifeval.compute_ifeval_reward("resp", None, metadata={"instruction_id_list": []}) == 0.0


# --- Integration against the real vendored IFEvalG checkers -------------------------

_IFEVALG_DEP_MODULES = ("nltk", "langdetect", "immutabledict")


@functools.lru_cache(maxsize=1)
def _ifevalg_available() -> bool:
    if any(importlib.util.find_spec(name) is None for name in _IFEVALG_DEP_MODULES):
        return False
    try:
        ifeval._get_instruction_dict()
        return True
    except Exception:
        return False


requires_ifevalg = pytest.mark.skipif(
    not _ifevalg_available(),
    reason="IFEvalG extras (nltk/langdetect/immutabledict) not installed",
)


@requires_ifevalg
class TestComputeIfevalRewardScoring:
    """End-to-end against the real checkers. All chosen instructions are regex/string
    based (no nltk punkt data, no langdetect), so results are deterministic."""

    @pytest.mark.parametrize(
        "instruction_id,kwargs,response,expected_strict,expected_loose",
        [
            # keyword existence: both required keywords present vs. one missing
            (
                "keywords:existence",
                {"keywords": ["banana", "umbrella"]},
                "I bought a banana and an umbrella today.",
                1.0,
                1.0,
            ),
            ("keywords:existence", {"keywords": ["banana", "umbrella"]}, "I bought a banana today.", 0.0, 0.0),
            # no comma allowed
            ("punctuation:no_comma", {}, "I have no commas here", 1.0, 1.0),
            ("punctuation:no_comma", {}, "I have a comma, right here", 0.0, 0.0),
            # response must end with an exact phrase
            ("startend:end_checker", {"end_phrase": "That is all."}, "Here is my answer. That is all.", 1.0, 1.0),
            ("startend:end_checker", {"end_phrase": "That is all."}, "Here is my answer.", 0.0, 0.0),
            # word count: count_words uses a regex tokenizer, so no punkt data needed
            (
                "length_constraints:number_words",
                {"num_words": 5, "relation": "at least"},
                "one two three four five six",
                1.0,
                1.0,
            ),
            ("length_constraints:number_words", {"num_words": 5, "relation": "at least"}, "one two three", 0.0, 0.0),
            # mode-sensitive: the comma sits on the first line, so loose passes after
            # stripping it while strict still sees it.
            ("punctuation:no_comma", {}, "Listen, this is the intro.\nThis line is clean", 0.0, 1.0),
        ],
    )
    def test_strict_and_loose(self, instruction_id, kwargs, response, expected_strict, expected_loose):
        metadata = {"instruction_id_list": [instruction_id], "kwargs": [kwargs]}
        assert ifeval.compute_ifeval_reward(response, None, metadata=metadata, strict=True) == expected_strict
        assert ifeval.compute_ifeval_reward(response, None, metadata=metadata, strict=False) == expected_loose

    def test_all_or_nothing_across_constraints(self):
        # Two constraints: full credit only when both hold; partial compliance scores 0.
        metadata = {
            "instruction_id_list": ["keywords:existence", "punctuation:no_comma"],
            "kwargs": [{"keywords": ["banana"]}, {}],
        }
        assert ifeval.compute_ifeval_reward("a banana with no commas", None, metadata=metadata) == 1.0
        assert ifeval.compute_ifeval_reward("a banana, yes", None, metadata=metadata) == 0.0  # comma fails
        assert ifeval.compute_ifeval_reward("no fruit here", None, metadata=metadata) == 0.0  # keyword fails

    def test_thinking_section_is_stripped_before_checking(self):
        # The only comma lives inside <think>; stripping it lets no_comma pass.
        metadata = {"instruction_id_list": ["punctuation:no_comma"], "kwargs": [{}]}
        response = "<think>hmm, let me reason</think>\nNo commas in the answer"
        assert ifeval.compute_ifeval_reward(response, None, metadata=metadata, strict=True) == 1.0

    def test_unknown_instruction_returns_zero(self):
        metadata = {"instruction_id_list": ["does_not:exist"], "kwargs": [{}]}
        assert ifeval.compute_ifeval_reward("anything", None, metadata=metadata) == 0.0


@requires_ifevalg
class TestIfevalRmTypeDispatch:
    """The rm_type token must select the right criterion: bare ``ifeval`` and
    ``ifeval_strict`` route to strict, ``ifeval_loose`` to loose. Exercised through the
    full async_rm dispatch against the real checkers, using the mode-sensitive comma
    case so strict (0.0) and loose (1.0) diverge."""

    @pytest.mark.parametrize(
        "rm_type,expected_reward",
        [
            ("ifeval", 0.0),
            ("ifeval_strict", 0.0),
            ("ifeval_loose", 1.0),
        ],
    )
    def test_rm_type_routes_to_mode(self, rm_type, expected_reward):
        args = MagicMock()
        args.custom_rm_path = None
        args.rm_type = rm_type
        sample = Sample(
            prompt="",
            response="Listen, this is the intro.\nThis line is clean",
            label=None,
            metadata={"instruction_id_list": ["punctuation:no_comma"], "kwargs": [{}]},
        )
        assert run(async_rm(args, sample)) == expected_reward
