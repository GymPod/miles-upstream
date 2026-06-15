import logging

from miles.utils.structured_log import _to_logfmt, log_structured, prune_for_log


class TestToLogfmt:
    def test_scalars_and_bools_lowercased(self):
        """Scalars render bare and bools render as lowercase true/false."""
        assert _to_logfmt({"cell": 1, "ok": True, "bad": False}) == "cell=1 ok=true bad=false"

    def test_list_is_comma_joined_without_spaces(self):
        """Lists render comma-joined with no spaces so a space-split parser keeps them one token."""
        assert _to_logfmt({"alive": [0, 1]}) == "alive=0,1"

    def test_empty_list_and_none_render_as_empty_value(self):
        """Empty list and None both render as an empty logfmt value."""
        assert _to_logfmt({"pending": [], "x": None}) == "pending= x="

    def test_value_with_spaces_is_quoted(self):
        """A value containing spaces is double-quoted so it stays a single token."""
        assert _to_logfmt({"reason": "survivors normal"}) == 'reason="survivors normal"'


class TestPruneForLog:
    def test_small_payload_kept_verbatim(self):
        """A payload under the cap is returned unchanged."""
        payload = {"quorum_id": 1, "healed": [0]}
        assert prune_for_log(payload, cap=160) == payload

    def test_large_list_field_summarized_small_siblings_kept(self):
        """An oversized list field becomes a length summary while small siblings stay inline."""
        pruned = prune_for_log({"quorum_id": 1, "checksums": list(range(500))}, cap=80)
        assert pruned["quorum_id"] == 1
        assert pruned["checksums"] == "<list len=500>"

    def test_large_string_field_summarized(self):
        """An oversized string field becomes a char-count summary."""
        assert prune_for_log({"blob": "x" * 1000}, cap=80)["blob"] == "<str 1000 chars>"


class TestLogStructured:
    def test_emits_ft_prefixed_logfmt_line_via_given_log_fn(self, caplog):
        """log_structured emits one 'ft '-prefixed logfmt line through the passed logger method."""
        logger = logging.getLogger("test_structured_log")
        with caplog.at_level(logging.INFO, logger="test_structured_log"):
            log_structured(logger.info, op="execute", phase="start", cell=1, fn="train")
        assert caplog.messages == ["ft op=execute phase=start cell=1 fn=train"]
