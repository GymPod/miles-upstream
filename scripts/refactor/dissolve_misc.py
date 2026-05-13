#!/usr/bin/env python3
"""Rewrite every `from miles.utils.misc import X[, Y]` to the new homes.

Symbol → module mapping after misc.py is dissolved:
  FunctionRegistry, function_registry, load_function  → miles.utils.registry
  SingletonMeta                                       → miles.utils.singleton
  exec_command                                        → miles.utils.shell
  _exec_command_on_node, exec_command_all_ray_node,
    get_current_node_ip                               → miles.utils.concurrency_utils  (re-exports)
  get_free_port                                       → miles.utils.net_utils
  should_run_periodic_action                          → miles.utils.periodic_utils
  as_completed_async                                  → miles.utils.iter_utils
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

SYMBOL_HOME = {
    "FunctionRegistry": "miles.utils.registry",
    "function_registry": "miles.utils.registry",
    "load_function": "miles.utils.registry",
    "SingletonMeta": "miles.utils.singleton",
    "exec_command": "miles.utils.shell",
    "_exec_command_on_node": "miles.utils.concurrency_utils",
    "exec_command_all_ray_node": "miles.utils.concurrency_utils",
    "get_current_node_ip": "miles.utils.concurrency_utils",
    "get_free_port": "miles.utils.net_utils",
    "should_run_periodic_action": "miles.utils.periodic_utils",
    "as_completed_async": "miles.utils.iter_utils",
}

IMPORT_RE = re.compile(
    r"^(?P<indent>[ \t]*)from\s+miles\.utils\.misc\s+import\s+(?P<symbols>[^\n#]+?)(?P<trailing>\s*(?:#.*)?)$",
    re.MULTILINE,
)


def split_symbols(s: str) -> list[str]:
    s = s.strip().strip("()").strip()
    return [tok.strip() for tok in s.split(",") if tok.strip()]


def rewrite_one(match: re.Match[str]) -> str:
    indent = match.group("indent")
    raw = match.group("symbols").strip()
    trailing = match.group("trailing") or ""
    symbols = split_symbols(raw)
    by_module: dict[str, list[str]] = {}
    for sym in symbols:
        # handle "as alias"
        base = sym.split(" as ")[0].strip()
        home = SYMBOL_HOME.get(base)
        if home is None:
            # Unknown symbol — bail out by leaving original line alone.
            return match.group(0)
        by_module.setdefault(home, []).append(sym)
    lines = [f"{indent}from {mod} import {', '.join(syms)}{trailing if i == 0 else ''}"
             for i, (mod, syms) in enumerate(by_module.items())]
    return "\n".join(lines)


def walk() -> list[Path]:
    skip = {".git", "__pycache__", "node_modules"}
    out: list[Path] = []
    for p in REPO.rglob("*.py"):
        if any(part in skip for part in p.parts):
            continue
        out.append(p)
    return out


def main() -> int:
    total = 0
    changed = 0
    for path in walk():
        text = path.read_text()
        new, n = IMPORT_RE.subn(rewrite_one, text)
        if n:
            path.write_text(new)
            changed += 1
            total += n
            print(f"{path.relative_to(REPO)}: {n} import statements rewritten")
    print(f"-- {changed} files, {total} statements --")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
