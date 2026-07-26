import asyncio
from argparse import Namespace

import pytest

import miles.rollout.inference_rollout.inference_rollout_train as inference_rollout_train
from miles.rollout.inference_rollout.inference_rollout_common import GenerateState
from miles.utils.misc import function_registry
from miles.utils.types import Sample


@pytest.mark.asyncio
async def test_abort_drains_pending_generations_before_reraising_request_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = Namespace(partial_rollout=False)
    state = object.__new__(GenerateState)
    state.args = args
    state.aborted = False
    abort_error = RuntimeError("abort request failed")
    generation_started = asyncio.Event()
    release_generation = asyncio.Event()
    generation_finished = asyncio.Event()

    async def request_abort(_args: Namespace) -> None:
        raise abort_error

    async def generate() -> list[Sample]:
        generation_started.set()
        await release_generation.wait()
        generation_finished.set()
        return []

    monkeypatch.setattr(inference_rollout_train, "request_abort", request_abort)
    pending = asyncio.create_task(generate())
    await generation_started.wait()
    abort_task = asyncio.create_task(inference_rollout_train.abort(state, {pending}, rollout_id=7))
    try:
        await asyncio.sleep(0)
        assert abort_task.done() is False

        release_generation.set()
        with pytest.raises(RuntimeError) as exc_info:
            await abort_task
    finally:
        release_generation.set()
        await asyncio.gather(pending, abort_task, return_exceptions=True)

    assert exc_info.value is abort_error
    assert generation_finished.is_set()
    assert state.aborted is True


@pytest.mark.asyncio
async def test_request_abort_starts_and_settles_every_operation_before_reraising_first_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = Namespace()
    first_worker_error = RuntimeError("first worker abort failed")
    hook_error = ValueError("agent abort failed")
    first_worker_failed = asyncio.Event()
    sibling_started = asyncio.Event()
    release_sibling = asyncio.Event()
    sibling_finished = asyncio.Event()
    hook_started = asyncio.Event()
    release_hook = asyncio.Event()
    hook_finished = asyncio.Event()

    async def get_worker_urls(_args: Namespace) -> list[str]:
        return ["http://worker-0", "http://worker-1"]

    async def post(url: str, payload: dict[str, bool]) -> None:
        assert payload == {"abort_all": True}
        if url == "http://worker-0/abort_request":
            first_worker_failed.set()
            raise first_worker_error

        assert url == "http://worker-1/abort_request"
        sibling_started.set()
        await release_sibling.wait()
        sibling_finished.set()

    async def call_agent_abort_hook(_args: Namespace) -> None:
        hook_started.set()
        await release_hook.wait()
        hook_finished.set()
        raise hook_error

    monkeypatch.setattr(inference_rollout_train, "get_worker_urls", get_worker_urls)
    monkeypatch.setattr(inference_rollout_train, "post", post)
    monkeypatch.setattr(inference_rollout_train, "call_agent_abort_hook", call_agent_abort_hook)

    abort_task = asyncio.create_task(inference_rollout_train.request_abort(args))
    try:
        await asyncio.wait_for(first_worker_failed.wait(), timeout=1)
        await asyncio.wait_for(sibling_started.wait(), timeout=1)
        await asyncio.wait_for(hook_started.wait(), timeout=1)
        request_waited_for_sibling = not abort_task.done()

        release_sibling.set()
        release_hook.set()
        with pytest.raises(RuntimeError) as exc_info:
            await abort_task
        await asyncio.wait_for(sibling_finished.wait(), timeout=1)
    finally:
        release_sibling.set()
        release_hook.set()
        await asyncio.gather(abort_task, return_exceptions=True)

    assert exc_info.value is first_worker_error
    assert (
        request_waited_for_sibling,
        sibling_finished.is_set(),
        hook_finished.is_set(),
    ) == (True, True, True)


@pytest.mark.asyncio
async def test_request_abort_propagates_configured_agent_hook_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = Namespace(custom_agent_function_path="test_agent.generate")
    hook_error = RuntimeError("agent abort failed")

    async def get_worker_urls(_args: Namespace) -> list[str]:
        return []

    async def abort_hook(_args: Namespace) -> None:
        raise hook_error

    monkeypatch.setattr(inference_rollout_train, "get_worker_urls", get_worker_urls)

    with function_registry.temporary("test_agent.abort", abort_hook):
        with pytest.raises(RuntimeError) as error:
            await inference_rollout_train.request_abort(args)

    assert error.value is hook_error
