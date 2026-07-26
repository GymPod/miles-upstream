from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from typing import NoReturn, TypeVar, cast

from miles.rollout.base_types import (
    GenerateFnInput,
    GenerateFnOutput,
    RolloutFnConstructorInput,
    RolloutFnEvalInput,
    RolloutFnEvalOutput,
    RolloutFnInput,
    RolloutFnOutput,
    RolloutFnTrainInput,
    RolloutFnTrainOutput,
)
from miles.rollout.rollout_session import (
    BatchRollbackReason,
    BatchRollbackUnsupportedError,
    RolloutSession,
    TrainBatchLease,
)
from miles.utils.async_utils import run
from miles.utils.misc import load_function

RolloutCallable = Callable[[RolloutFnInput], RolloutFnOutput | Awaitable[RolloutFnOutput]]
RolloutDefinition = Callable[..., object]
_T = TypeVar("_T")


class LegacyRolloutFnAdapter:
    def __init__(self, input: RolloutFnConstructorInput, fn: Callable):
        self.args = input.args
        self.data_source = input.data_source
        self.fn = fn

    def __call__(self, input: RolloutFnInput) -> RolloutFnOutput:
        output = self.fn(self.args, input.rollout_id, self.data_source, evaluation=input.evaluation)

        # compatibility for legacy version
        if not isinstance(output, (RolloutFnTrainOutput, RolloutFnEvalOutput)):
            output = RolloutFnEvalOutput(data=output) if input.evaluation else RolloutFnTrainOutput(samples=output)

        return output


def _construct_rollout_class(
    input: RolloutFnConstructorInput,
    definition: RolloutDefinition,
) -> RolloutCallable:
    constructor = cast(Callable[[RolloutFnConstructorInput], object], definition)
    rollout_function = constructor(input)
    if not callable(rollout_function):
        raise TypeError(f"Rollout class {definition!r} produced a non-callable instance.")
    return cast(RolloutCallable, rollout_function)


def _create_rollout_function(input: RolloutFnConstructorInput, definition: RolloutDefinition) -> RolloutCallable:
    if inspect.isclass(definition):
        return _construct_rollout_class(input, definition)
    return LegacyRolloutFnAdapter(input, definition)


def _run_thread_affine_call(
    fn: Callable[..., object],
    *args: object,
) -> object:
    return fn(*args)


class _ThreadAffineRolloutCallable:
    """Run one synchronous rollout instance on its construction thread."""

    def __init__(self, fn: RolloutCallable, executor: ThreadPoolExecutor) -> None:
        self._fn = fn
        self._executor = executor
        self._closed = False

    @classmethod
    async def create(
        cls,
        input: RolloutFnConstructorInput,
        definition: RolloutDefinition,
        retain_on_close_failure: Callable[[_ThreadAffineRolloutCallable], None],
    ) -> _ThreadAffineRolloutCallable:
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="miles-rollout")
        loop = asyncio.get_running_loop()
        construction = asyncio.ensure_future(
            loop.run_in_executor(executor, _construct_rollout_class, input, definition)
        )
        try:
            fn = await asyncio.shield(construction)
        except asyncio.CancelledError as cancelled:
            try:
                fn = await _await_task_terminal(construction)
            except BaseException as terminal_error:
                executor.shutdown(wait=True)
                raise cancelled from terminal_error
            wrapper = cls(fn=fn, executor=executor)
            close_task = asyncio.create_task(wrapper.close())
            try:
                await asyncio.shield(close_task)
            except asyncio.CancelledError:
                try:
                    await _await_task_terminal(close_task)
                except BaseException as close_error:
                    retain_on_close_failure(wrapper)
                    raise cancelled from close_error
                raise cancelled from None
            except BaseException as close_error:
                retain_on_close_failure(wrapper)
                raise cancelled from close_error
            raise
        except BaseException:
            executor.shutdown(wait=True)
            raise
        return cls(fn=fn, executor=executor)

    async def __call__(self, input: RolloutFnInput) -> RolloutFnOutput:
        if self._closed:
            raise RuntimeError("Rollout function is closed.")
        output = await self._run(self._fn, input)
        if inspect.isawaitable(output):
            output = await output
        return cast(RolloutFnOutput, output)

    async def close(self) -> None:
        if self._closed:
            return
        close = getattr(self._fn, "close", None)
        if close is not None:
            if not callable(close):
                raise TypeError(f"Rollout close attribute is not callable: {close!r}.")
            result = await self._run(cast(Callable[..., object], close))
            if inspect.isawaitable(result):
                await result
        self._shutdown_executor()
        self._closed = True

    async def _run(self, fn: Callable[..., object], *args: object) -> object:
        loop = asyncio.get_running_loop()
        call = asyncio.ensure_future(loop.run_in_executor(self._executor, _run_thread_affine_call, fn, *args))
        try:
            return await asyncio.shield(call)
        except asyncio.CancelledError as cancelled:
            try:
                result = await _await_task_terminal(call)
                if inspect.isawaitable(result):
                    await _await_task_terminal(asyncio.ensure_future(result))
            except BaseException as terminal_error:
                raise cancelled from terminal_error
            raise

    def _shutdown_executor(self) -> None:
        self._executor.shutdown(wait=True)


async def _create_session_rollout_function(
    input: RolloutFnConstructorInput,
    definition: RolloutDefinition,
    retain_on_close_failure: Callable[[_ThreadAffineRolloutCallable], None],
) -> RolloutCallable:
    call = inspect.getattr_static(definition, "__call__", None)
    if inspect.isclass(definition) and not inspect.iscoroutinefunction(call):
        return await _ThreadAffineRolloutCallable.create(
            input=input,
            definition=definition,
            retain_on_close_failure=retain_on_close_failure,
        )
    return _create_rollout_function(input, definition)


def load_rollout_function(input: RolloutFnConstructorInput, path: str) -> RolloutCallable:
    return _create_rollout_function(input, load_function(path))


def call_rollout_function(fn, input: RolloutFnInput) -> RolloutFnOutput:
    output = fn(input)

    if inspect.iscoroutine(output):
        output = run(output)

    return output


class _CompatibilityTrainBatchLease(TrainBatchLease):
    def __init__(
        self,
        rollout_id: int,
        output: RolloutFnTrainOutput,
        settle: Callable[[_CompatibilityTrainBatchLease], None],
    ) -> None:
        super().__init__(rollout_id=rollout_id, output=output)
        self._settle = settle

    def _commit(self) -> None:
        self._settle(self)

    def _rollback(self, reason: BatchRollbackReason) -> None:
        self._settle(self)
        raise BatchRollbackUnsupportedError(rollout_id=self.rollout_id, reason=reason)


class _CompatibilityRolloutSession(RolloutSession):
    def __init__(
        self,
        input: RolloutFnConstructorInput,
        train_definition: RolloutDefinition,
        eval_definition: RolloutDefinition,
    ) -> None:
        self._constructor_input = input
        self._train_definition = train_definition
        self._eval_definition = eval_definition
        self._train_fn: RolloutCallable | None = None
        self._eval_fn: RolloutCallable | None = None
        self._open_leases: dict[int, _CompatibilityTrainBatchLease] = {}
        self._accepting_operations = True
        self._closed = False
        self._train_closed = False
        self._eval_closed = False
        self._initialization_lock = asyncio.Lock()
        self._train_lock = asyncio.Lock()
        self._eval_lock = asyncio.Lock()

    async def acquire_train_batch(self, rollout_id: int) -> TrainBatchLease:
        self._ensure_open()
        async with self._train_lock:
            self._ensure_open()
            if rollout_id in self._open_leases:
                raise RuntimeError(f"Train batch lease for rollout {rollout_id} is already open.")
            train_fn, _ = await self._get_rollout_functions()
            output = await _invoke_rollout_function(train_fn, RolloutFnTrainInput(rollout_id=rollout_id))
            if not isinstance(output, RolloutFnTrainOutput):
                raise TypeError(f"Train rollout returned {type(output).__name__}, expected RolloutFnTrainOutput.")
            lease = _CompatibilityTrainBatchLease(
                rollout_id=rollout_id,
                output=output,
                settle=self._settle_lease,
            )
            self._open_leases[rollout_id] = lease
            return lease

    async def evaluate(self, rollout_id: int) -> RolloutFnEvalOutput:
        self._ensure_open()
        async with self._eval_lock:
            self._ensure_open()
            _, eval_fn = await self._get_rollout_functions()
            output = await _invoke_rollout_function(eval_fn, RolloutFnEvalInput(rollout_id=rollout_id))
            if not isinstance(output, RolloutFnEvalOutput):
                raise TypeError(f"Eval rollout returned {type(output).__name__}, expected RolloutFnEvalOutput.")
            return output

    async def prepare_checkpoint(self, rollout_id: int) -> None:
        self._ensure_open()
        async with self._train_lock:
            async with self._eval_lock:
                self._ensure_open()
                if self._open_leases:
                    open_rollout_ids = sorted(self._open_leases)
                    raise RuntimeError(
                        f"Cannot prepare checkpoint {rollout_id} with open train batch leases: {open_rollout_ids}."
                    )

    async def close(self) -> None:
        close_task = asyncio.create_task(self._close())
        try:
            await asyncio.shield(close_task)
        except asyncio.CancelledError as cancelled:
            try:
                await _await_task_terminal(close_task)
            except BaseException as terminal_error:
                raise cancelled from terminal_error
            raise

    async def _close(self) -> None:
        if self._closed:
            return
        async with self._train_lock:
            async with self._eval_lock:
                if self._closed:
                    return
                if self._open_leases:
                    open_rollout_ids = sorted(self._open_leases)
                    raise RuntimeError(
                        f"Cannot close rollout session with open train batch leases: {open_rollout_ids}."
                    )
                self._accepting_operations = False
                close_errors: list[BaseException] = []
                if self._train_fn is None:
                    self._train_closed = True
                elif not self._train_closed:
                    try:
                        await _close_rollout_function(self._train_fn)
                    except BaseException as error:
                        close_errors.append(error)
                    else:
                        self._train_closed = True
                if self._eval_fn is None:
                    self._eval_closed = True
                elif not self._eval_closed:
                    try:
                        await _close_rollout_function(self._eval_fn)
                    except BaseException as error:
                        close_errors.append(error)
                    else:
                        self._eval_closed = True
                self._closed = self._train_closed and self._eval_closed
                if close_errors:
                    raise close_errors[0]

    async def _get_rollout_functions(self) -> tuple[RolloutCallable, RolloutCallable]:
        self._ensure_open()
        if self._train_fn is not None and self._eval_fn is not None:
            return self._train_fn, self._eval_fn

        async with self._initialization_lock:
            self._ensure_open()
            if self._train_fn is not None and self._eval_fn is not None:
                return self._train_fn, self._eval_fn

            train_fn = await _create_session_rollout_function(
                self._constructor_input,
                self._train_definition,
                self._retain_partially_initialized_train,
            )
            try:
                eval_fn = await _create_session_rollout_function(
                    self._constructor_input,
                    self._eval_definition,
                    self._retain_partially_initialized_eval,
                )
            except BaseException as construction_error:
                await self._close_after_construction_failure(train_fn, construction_error)

            self._train_fn = train_fn
            self._eval_fn = eval_fn
            return train_fn, eval_fn

    async def _close_after_construction_failure(
        self,
        train_fn: RolloutCallable,
        construction_error: BaseException,
    ) -> NoReturn:
        close_task = asyncio.create_task(_close_rollout_function(train_fn))
        try:
            await asyncio.shield(close_task)
        except asyncio.CancelledError as cancelled:
            try:
                await _await_task_terminal(close_task)
            except BaseException as close_error:
                self._retain_partially_initialized_train(train_fn)
                raise cancelled from close_error
            raise
        except BaseException as close_error:
            self._retain_partially_initialized_train(train_fn)
            raise construction_error from close_error
        raise construction_error

    def _retain_partially_initialized_train(self, train_fn: RolloutCallable) -> None:
        self._train_fn = train_fn
        self._accepting_operations = False

    def _retain_partially_initialized_eval(self, eval_fn: RolloutCallable) -> None:
        self._eval_fn = eval_fn
        self._accepting_operations = False

    def _settle_lease(self, lease: _CompatibilityTrainBatchLease) -> None:
        current_lease = self._open_leases.get(lease.rollout_id)
        if current_lease is not lease:
            raise RuntimeError(f"Train batch lease for rollout {lease.rollout_id} is not owned by this session.")
        del self._open_leases[lease.rollout_id]

    def _ensure_open(self) -> None:
        if not self._accepting_operations:
            raise RuntimeError("Rollout session is closed.")


async def _invoke_rollout_function(fn: RolloutCallable, input: RolloutFnInput) -> RolloutFnOutput:
    invoke = cast(Callable[[RolloutFnInput], object], fn)
    call = inspect.getattr_static(fn, "__call__")
    if inspect.iscoroutinefunction(fn) or inspect.iscoroutinefunction(call):
        output = invoke(input)
    else:
        call_task = asyncio.create_task(asyncio.to_thread(invoke, input))
        try:
            output = await asyncio.shield(call_task)
        except asyncio.CancelledError as cancelled:
            try:
                output = await _await_task_terminal(call_task)
                if inspect.isawaitable(output):
                    await _await_task_terminal(asyncio.ensure_future(output))
            except BaseException as terminal_error:
                raise cancelled from terminal_error
            raise

    if inspect.isawaitable(output):
        output = await output
    return cast(RolloutFnOutput, output)


async def _await_task_terminal(task: asyncio.Future[_T]) -> _T:
    while True:
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            if task.done():
                return task.result()


async def _close_rollout_function(fn: RolloutCallable) -> None:
    close = getattr(fn, "close", None)
    if close is None:
        return
    if inspect.iscoroutinefunction(close):
        await close()
        return

    result = close()
    if inspect.isawaitable(result):
        await result


def load_rollout_session(
    input: RolloutFnConstructorInput,
    *,
    train_path: str,
    eval_path: str,
) -> RolloutSession:
    """Resolve isolated train and eval rollout functions behind one session.

    Args:
        input: Shared constructor dependencies for both rollout functions.
        train_path: Import path for the training rollout function.
        eval_path: Import path for the evaluation rollout function.

    Returns:
        An explicitly configured session, or a compatibility session that
        constructs independent train and eval instances together on its first
        asynchronous operation.
    """
    train_definition = load_function(train_path)
    eval_definition = load_function(eval_path)
    train_is_session = inspect.isclass(train_definition) and issubclass(train_definition, RolloutSession)
    eval_is_session = inspect.isclass(eval_definition) and issubclass(eval_definition, RolloutSession)
    if train_is_session or eval_is_session:
        if not (train_is_session and eval_is_session and train_definition is eval_definition):
            raise ValueError(
                "Explicit RolloutSession definitions must use the same class for train and eval; got "
                f"{train_path!r} and {eval_path!r}."
            )
        try:
            inspect.signature(train_definition).bind(input)
        except TypeError as error:
            raise TypeError(
                f"Rollout session {train_path!r} must accept one RolloutFnConstructorInput positional argument."
            ) from error
        constructor = cast(Callable[[RolloutFnConstructorInput], RolloutSession], train_definition)
        session = constructor(input)
        if not isinstance(session, RolloutSession):
            raise TypeError(
                f"Rollout session {train_path!r} produced {type(session).__name__}, expected RolloutSession."
            )
        return session
    return _CompatibilityRolloutSession(
        input=input,
        train_definition=train_definition,
        eval_definition=eval_definition,
    )


class LegacyGenerateFnAdapter:
    def __init__(self, fn: Callable):
        self.fn = fn
        self._has_evaluation_param = "evaluation" in inspect.signature(fn).parameters

    async def __call__(self, input: GenerateFnInput) -> GenerateFnOutput:
        if self._has_evaluation_param:
            output = await self.fn(input.args, input.sample, input.sampling_params, evaluation=input.evaluation)
        else:
            output = await self.fn(input.args, input.sample, input.sampling_params)

        if not isinstance(output, GenerateFnOutput):
            output = GenerateFnOutput(samples=output)

        return output


def load_generate_function(path: str):
    fn = load_function(path)
    if fn is None:
        return None

    if inspect.isclass(fn):
        return fn()
    elif _is_legacy_generate_fn(fn):
        return LegacyGenerateFnAdapter(fn)
    else:
        return fn


def _is_legacy_generate_fn(fn: Callable) -> bool:
    sig = inspect.signature(fn)
    params = list(sig.parameters.keys())
    return len(params) >= 3 and params[0] != "input"
