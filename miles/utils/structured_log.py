import functools
import inspect
import json
import logging
import time
from collections.abc import Callable
from typing import Any

_PRUNE_CAP = 160


def log_structured(log_fn: Callable[..., None], *, exc_info: bool = False, **fields: Any) -> None:
    log_fn("ft " + _to_logfmt(fields), stacklevel=2, exc_info=exc_info)


def with_logs(name: str | None = None) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Bracket a method with one `ft op=actor phase=start` and one `phase=end` line.

    The end line carries elapsed_s and ok; on exception ok=false and the traceback is
    logged before re-raising. Per-rank entry/exit lines make a hang visible as a
    phase=start with no matching phase=end (e.g. a cell wedged inside train). The
    function's own module logger is used so each rank's identity tag is preserved.
    """

    def decorate(func: Callable[..., Any]) -> Callable[..., Any]:
        fn_name = name or func.__name__
        method_logger = logging.getLogger(func.__module__)

        def log_start() -> float:
            log_structured(method_logger.info, op="actor", phase="start", fn=fn_name)
            return time.monotonic()

        def log_end(start: float) -> None:
            log_structured(method_logger.info, op="actor", phase="end", fn=fn_name, ok=True, elapsed_s=_elapsed(start))

        def log_fail(start: float) -> None:
            log_structured(
                method_logger.error, op="actor", phase="end", fn=fn_name, ok=False, elapsed_s=_elapsed(start), exc_info=True
            )

        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                start = log_start()
                try:
                    result = await func(*args, **kwargs)
                except BaseException:
                    log_fail(start)
                    raise
                log_end(start)
                return result

            return async_wrapper

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = log_start()
            try:
                result = func(*args, **kwargs)
            except BaseException:
                log_fail(start)
                raise
            log_end(start)
            return result

        return wrapper

    return decorate


def _elapsed(start: float) -> float:
    return round(time.monotonic() - start, 1)


def _to_logfmt(fields: dict[str, Any]) -> str:
    return " ".join(f"{key}={_format_value(value)}" for key, value in fields.items())


def _format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (list, tuple)):
        return _maybe_quote(",".join(_format_scalar(item) for item in value))
    if isinstance(value, dict):
        return _quote(json.dumps(value, separators=(",", ":"), default=str))
    return _maybe_quote(str(value))


def _format_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _maybe_quote(text: str) -> str:
    if text and any(ch in text for ch in (" ", "=", '"')):
        return _quote(text)
    return text


def _quote(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def prune_for_log(value: Any, cap: int = _PRUNE_CAP) -> Any:
    if len(_compact_json(value)) <= cap:
        return value
    if isinstance(value, dict):
        return {key: prune_for_log(item, cap) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return f"<list len={len(value)}>"
    if isinstance(value, str):
        return f"<str {len(value)} chars>"
    return f"<{type(value).__name__}>"


def _compact_json(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), default=str)
