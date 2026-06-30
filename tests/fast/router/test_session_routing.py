"""Unit tests for process-stable session routing (miles/rollout/session/routing.py)."""

import os
import re
import subprocess
import sys

import pytest

from miles.rollout.session.routing import new_session_id, worker_index_for_session


def test_new_session_id_shape():
    sid = new_session_id()
    assert re.fullmatch(r"[0-9a-f]{32}", sid)
    assert new_session_id() != new_session_id()  # fresh each call


def test_worker_index_in_range_and_deterministic():
    sid = new_session_id()
    for n in (1, 2, 4, 16):
        idx = worker_index_for_session(sid, n)
        assert 0 <= idx < n
        # deterministic: repeated calls agree
        assert worker_index_for_session(sid, n) == idx


def test_worker_index_distributes_across_workers():
    n = 8
    seen = {worker_index_for_session(new_session_id(), n) for _ in range(2000)}
    # with 2000 ids over 8 workers, every worker should be hit
    assert seen == set(range(n))


def test_worker_index_rejects_bad_n():
    with pytest.raises(ValueError):
        worker_index_for_session(new_session_id(), 0)


def test_mapping_is_process_stable_across_pythonhashseed():
    """The mapping must not depend on PYTHONHASHSEED (i.e. must not use builtin hash())."""
    sid = "0123456789abcdef0123456789abcdef"
    n = 7
    expected = worker_index_for_session(sid, n)

    def index_in_subproc(seed: str) -> int:
        code = (
            "from miles.rollout.session.routing import worker_index_for_session;"
            f"print(worker_index_for_session({sid!r}, {n}))"
        )
        out = subprocess.check_output([sys.executable, "-c", code], env={**os.environ, "PYTHONHASHSEED": seed})
        return int(out.strip())

    assert index_in_subproc("0") == expected
    assert index_in_subproc("12345") == expected
