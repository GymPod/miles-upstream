"""Unit tests for process-stable session routing and explicit-id session creation.

Covers ``miles.rollout.session.routing`` (worker index determinism, cross-process
stability, range guarantee, n_worker guard) and
``SessionRegistry.create_session_with_id`` (valid id, malformed id, duplicate id).
"""

import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

# Same stub the linear_trajectory unit tests use, imported to build a registry
# without a real tokenizer.
from tests.fast.router.test_linear_trajectory import _MockTITOTokenizer

from miles.rollout.session.linear_trajectory import SessionRegistry
from miles.rollout.session.routing import new_session_id, worker_index_for_session


def _make_registry() -> SessionRegistry:
    args = SimpleNamespace()
    mock_tito = _MockTITOTokenizer(
        tokenizer=None, assistant_start_str="<|im_start|>assistant", allowed_append_roles=None
    )
    return SessionRegistry(args, tokenizer=None, tito_tokenizer=mock_tito)


class TestWorkerIndexForSession:
    def test_deterministic(self):
        sid = new_session_id()
        assert worker_index_for_session(sid, 4) == worker_index_for_session(sid, 4)

    def test_in_range(self):
        for n in (1, 2, 3, 4, 7, 16):
            for _ in range(50):
                idx = worker_index_for_session(new_session_id(), n)
                assert 0 <= idx < n

    def test_n_worker_one_always_zero(self):
        assert worker_index_for_session(new_session_id(), 1) == 0

    def test_n_worker_guard(self):
        with pytest.raises(ValueError):
            worker_index_for_session(new_session_id(), 0)
        with pytest.raises(ValueError):
            worker_index_for_session(new_session_id(), -1)

    def test_matches_precomputed_blake2b(self):
        # Pin the exact mapping for a fixed id so a future change to the hash
        # function (which would silently break router/worker agreement) fails here.
        # Expected derived independently via hashlib.blake2b(..., digest_size=8).
        import hashlib

        sid = "0123456789abcdef0123456789abcdef"
        digest = hashlib.blake2b(sid.encode("utf-8"), digest_size=8).digest()
        for n in (2, 4, 8, 13):
            expected = int.from_bytes(digest, "big") % n
            assert worker_index_for_session(sid, n) == expected

    def test_stable_across_pythonhashseed(self):
        # The mapping must NOT depend on PYTHONHASHSEED (which salts builtin
        # hash()); run the same id+n in subprocesses with seed 0 vs 1 and compare.
        sid = "0123456789abcdef0123456789abcdef"
        n = 8
        script = (
            "from miles.rollout.session.routing import worker_index_for_session;"
            f"print(worker_index_for_session({sid!r}, {n}))"
        )

        def run(seed: str) -> str:
            env = {**os.environ, "PYTHONHASHSEED": seed}
            out = subprocess.run(
                [sys.executable, "-c", script],
                env=env,
                capture_output=True,
                text=True,
                check=True,
            )
            return out.stdout.strip()

        idx0 = run("0")
        idx1 = run("1")
        assert idx0 == idx1
        assert idx0 == str(worker_index_for_session(sid, n))


class TestCreateSessionWithId:
    def test_accepts_valid_hex(self):
        registry = _make_registry()
        sid = new_session_id()
        assert registry.create_session_with_id(sid) == sid
        assert sid in registry.sessions

    def test_rejects_non_hex(self):
        registry = _make_registry()
        for bad in (
            "not-hex",
            "0123456789ABCDEF0123456789ABCDEF",  # uppercase
            "0123456789abcdef0123456789abcde",  # 31 chars
            "0123456789abcdef0123456789abcdef0",  # 33 chars
            "0123456789abcdef0123456789abcdeg",  # non-hex char
            "",
        ):
            with pytest.raises(ValueError):
                registry.create_session_with_id(bad)
        assert registry.sessions == {}

    def test_rejects_duplicate(self):
        registry = _make_registry()
        sid = new_session_id()
        registry.create_session_with_id(sid)
        with pytest.raises(ValueError):
            registry.create_session_with_id(sid)

    def test_create_session_delegates(self):
        # create_session() should still produce a valid, registered 32-hex id.
        registry = _make_registry()
        sid = registry.create_session()
        assert len(sid) == 32
        assert sid in registry.sessions
