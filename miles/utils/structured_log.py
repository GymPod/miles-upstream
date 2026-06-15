import json
from collections.abc import Callable
from typing import Any

_PRUNE_CAP = 160


def log_structured(log_fn: Callable[..., None], *, exc_info: bool = False, **fields: Any) -> None:
    log_fn("ft " + _to_logfmt(fields), stacklevel=2, exc_info=exc_info)


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
