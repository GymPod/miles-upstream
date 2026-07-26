import logging
import os
from argparse import Namespace

import pytest

from miles.utils.misc import FunctionRegistry, call_agent_abort_hook, filter_keys, function_registry, load_function


def _fn_a():
    return "a"


def _fn_b():
    return "b"


class TestFunctionRegistry:
    def test_register_and_get(self):
        registry = FunctionRegistry()
        with registry.temporary("my_fn", _fn_a):
            assert registry.get("my_fn") is _fn_a

    def test_register_duplicate_raises(self):
        registry = FunctionRegistry()
        with registry.temporary("my_fn", _fn_a):
            with pytest.raises(AssertionError):
                with registry.temporary("my_fn", _fn_b):
                    pass

    def test_unregister(self):
        registry = FunctionRegistry()
        with registry.temporary("my_fn", _fn_a):
            assert registry.get("my_fn") is _fn_a
        assert registry.get("my_fn") is None

    def test_temporary_cleanup_on_exception(self):
        registry = FunctionRegistry()
        with pytest.raises(RuntimeError):
            with registry.temporary("temp_fn", _fn_a):
                raise RuntimeError("test")
        assert registry.get("temp_fn") is None


class TestLoadFunction:
    def test_load_from_module(self):
        import os.path

        assert load_function("os.path.join") is os.path.join

    def test_load_none_returns_none(self):
        assert load_function(None) is None

    def test_load_from_registry(self):
        with function_registry.temporary("test:my_fn", _fn_a):
            assert load_function("test:my_fn") is _fn_a

    def test_registry_takes_precedence(self):
        with function_registry.temporary("os.path.join", _fn_b):
            assert load_function("os.path.join") is _fn_b
        assert load_function("os.path.join") is os.path.join


class TestCallAgentAbortHook:
    @pytest.mark.asyncio
    async def test_returns_when_no_agent_function_is_configured(self) -> None:
        args = Namespace(custom_agent_function_path=None)

        await call_agent_abort_hook(args)

    @pytest.mark.asyncio
    async def test_returns_when_agent_path_has_no_module(self) -> None:
        args = Namespace(custom_agent_function_path="generate")

        await call_agent_abort_hook(args)

    @pytest.mark.asyncio
    async def test_returns_when_agent_module_has_no_abort_hook(self) -> None:
        args = Namespace(custom_agent_function_path="pathlib.Path")

        await call_agent_abort_hook(args)

    @pytest.mark.asyncio
    async def test_returns_when_registry_only_agent_has_no_abort_hook(self) -> None:
        args = Namespace(custom_agent_function_path="test_agent.generate")

        async def generate() -> None:
            return None

        with function_registry.temporary("test_agent.generate", generate):
            await call_agent_abort_hook(args)

    @pytest.mark.asyncio
    async def test_propagates_missing_module_for_nonregistered_agent(self) -> None:
        args = Namespace(custom_agent_function_path="miles_nonexistent_agent_module_for_test.generate")

        with pytest.raises(ModuleNotFoundError) as error:
            await call_agent_abort_hook(args)

        assert error.value.name == "miles_nonexistent_agent_module_for_test"

    @pytest.mark.asyncio
    async def test_propagates_registry_lookup_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        args = Namespace(custom_agent_function_path="test_agent.generate")
        registry_error = RuntimeError("registry lookup failed")

        def fail_registry_lookup(_path: str) -> object:
            raise registry_error

        monkeypatch.setattr(function_registry, "get", fail_registry_lookup)
        with pytest.raises(RuntimeError) as error:
            await call_agent_abort_hook(args)

        assert error.value is registry_error

    @pytest.mark.asyncio
    async def test_registry_agent_does_not_hide_abort_hook_dependency_failure(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        args = Namespace(custom_agent_function_path="test_agent.generate")
        import_error = ModuleNotFoundError(
            "No module named 'missing_abort_dependency'",
            name="missing_abort_dependency",
        )

        async def generate() -> None:
            return None

        def fail_abort_hook_import(_path: str) -> object:
            raise import_error

        monkeypatch.setattr("miles.utils.misc.importlib.import_module", fail_abort_hook_import)
        with function_registry.temporary("test_agent.generate", generate):
            with pytest.raises(ModuleNotFoundError) as error:
                await call_agent_abort_hook(args)

        assert error.value is import_error

    @pytest.mark.asyncio
    async def test_propagates_attribute_error_raised_during_module_import(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        args = Namespace(custom_agent_function_path="test_agent.generate")
        import_error = AttributeError("module initialization failed")

        def fail_module_import(_path: str) -> object:
            raise import_error

        monkeypatch.setattr("miles.utils.misc.importlib.import_module", fail_module_import)
        with pytest.raises(AttributeError) as error:
            await call_agent_abort_hook(args)

        assert error.value is import_error

    @pytest.mark.asyncio
    async def test_invokes_registered_abort_hook_with_arguments(self) -> None:
        args = Namespace(custom_agent_function_path="test_agent.generate")
        calls: list[Namespace] = []

        async def abort_hook(hook_args: Namespace) -> None:
            calls.append(hook_args)

        with function_registry.temporary("test_agent.abort", abort_hook):
            await call_agent_abort_hook(args)

        assert calls == [args]

    @pytest.mark.asyncio
    async def test_propagates_registered_abort_hook_failure(self) -> None:
        args = Namespace(custom_agent_function_path="test_agent.generate")
        hook_error = RuntimeError("agent abort failed")

        async def abort_hook(_args: Namespace) -> None:
            raise hook_error

        with function_registry.temporary("test_agent.abort", abort_hook):
            with pytest.raises(RuntimeError) as error:
                await call_agent_abort_hook(args)

        assert error.value is hook_error

    @pytest.mark.asyncio
    async def test_propagates_imported_abort_hook_failure(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        args = Namespace(custom_agent_function_path="test_agent.generate")
        hook_error = RuntimeError("imported agent abort failed")

        async def abort_hook(_args: Namespace) -> None:
            raise hook_error

        monkeypatch.setattr(
            "miles.utils.misc.importlib.import_module",
            lambda _path: Namespace(abort=abort_hook),
        )
        with pytest.raises(RuntimeError) as error:
            await call_agent_abort_hook(args)

        assert error.value is hook_error


class TestFilterKeys:
    def test_projects_dict_by_keys(self):
        """filter_keys returns only the requested keys with their values."""
        d = {"a": 1, "b": 2, "c": 3}
        assert filter_keys(d, ["a", "c"]) == {"a": 1, "c": 3}

    def test_empty_interest_keys_returns_empty_dict(self):
        """An empty interest list yields an empty dict regardless of input."""
        assert filter_keys({"a": 1, "b": 2}, []) == {}

    def test_preserves_interest_keys_order(self):
        """Result key order follows interest_keys, not the source dict order."""
        d = {"a": 1, "b": 2, "c": 3}
        assert list(filter_keys(d, ["c", "a"]).keys()) == ["c", "a"]

    def test_full_subset_returns_all_entries(self):
        """Requesting every key returns the whole projection."""
        d = {"x": 10, "y": 20}
        assert filter_keys(d, ["x", "y"]) == {"x": 10, "y": 20}

    def test_duplicate_interest_key_collapses_to_single_entry(self):
        """A repeated interest key produces a single dict entry."""
        d = {"a": 1, "b": 2}
        assert filter_keys(d, ["a", "a"]) == {"a": 1}

    def test_missing_key_raises_key_error_and_logs(self, caplog):
        """A missing key raises KeyError and logs the error with context."""
        d = {"a": 1}
        with caplog.at_level(logging.ERROR, logger="miles.utils.misc"):
            with pytest.raises(KeyError):
                filter_keys(d, ["a", "missing"])
        assert any("filter_keys" in record.message for record in caplog.records)
