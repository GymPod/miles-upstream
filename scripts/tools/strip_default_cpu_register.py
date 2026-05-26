"""One-shot codemod: strip default-form `register_cpu_ci` calls from tests/fast/.

After the directory-based implicit CPU registration landed in
`tests/ci/ci_register.py`, any `tests/fast/**/*.py` file calling
`register_cpu_ci(est_time=10, suite="stage-a-cpu", labels=[])` is redundant
with the collector's implicit fallback. This script removes those calls
plus the now-orphan `from tests.ci.ci_register import register_cpu_ci`
imports.

The script is intentionally conservative:
- Only matches the *exact* default form: kwargs `est_time=10`, `suite="stage-a-cpu"`, `labels=[]`. Any deviation (different `est_time`, non-empty `labels`, extra kwargs, positional args) leaves the call in place as an explicit override.
- Only removes the import when no remaining `register_cpu_ci` / `register_cuda_ci` symbol reference survives in the file.
- Only touches files under `tests/fast/`; other subtrees are off-limits.

Run from repo root: `python scripts/tools/strip_default_cpu_register.py`.
Pass `--dry-run` to print what would change without writing.
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

TARGET_SUBTREE = Path("tests/fast")
DEFAULT_EST_TIME = 10
DEFAULT_SUITE = "stage-a-cpu"


def _is_default_form_register_cpu_ci(call: ast.Call) -> bool:
    """True iff this Call is exactly `register_cpu_ci(est_time=10, suite="stage-a-cpu", labels=[])`.

    Any positional args, missing kwargs, deviating values, or extra kwargs
    disqualify the call.
    """
    if not isinstance(call.func, ast.Name) or call.func.id != "register_cpu_ci":
        return False
    if call.args:
        return False
    expected_keywords = {"est_time", "suite", "labels"}
    if {kw.arg for kw in call.keywords} != expected_keywords:
        return False
    for kw in call.keywords:
        if not isinstance(kw.value, (ast.Constant, ast.List)):
            return False
        if kw.arg == "est_time":
            if not isinstance(kw.value, ast.Constant) or kw.value.value != DEFAULT_EST_TIME:
                return False
        elif kw.arg == "suite":
            if not isinstance(kw.value, ast.Constant) or kw.value.value != DEFAULT_SUITE:
                return False
        elif kw.arg == "labels":
            if not isinstance(kw.value, ast.List) or kw.value.elts:
                return False
    return True


def _find_orphan_import_lines(tree: ast.Module, file_text: str) -> set[int]:
    """Return 1-based line numbers of imports that bring in `register_cpu_ci`
    but become unused if the matching default-form calls are removed.

    Heuristic: after all default-form `register_cpu_ci` Expr(Call) nodes are
    removed, scan the entire AST for any other Name/Attribute reference to
    `register_cpu_ci`. If none remain, the import is orphan.

    Similar for `register_cuda_ci`: never auto-removed (it stays explicit in
    tests/fast-gpu/), so we only consider `register_cpu_ci` imports here.
    """
    to_remove: set[int] = set()
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.module != "tests.ci.ci_register":
            continue
        names = [alias.name for alias in node.names]
        if "register_cpu_ci" not in names:
            continue
        # Only mark the import for removal if `register_cpu_ci` is the only
        # name being imported from `tests.ci.ci_register` -- otherwise we
        # would silently drop e.g. `register_cuda_ci` imports too.
        if len(node.names) != 1:
            continue
        # Confirm no surviving non-removed reference to `register_cpu_ci`
        # exists anywhere in the file (Names, Attributes, or strings already
        # excluded because Constant strings aren't Name nodes).
        for sub in ast.walk(tree):
            if isinstance(sub, ast.Name) and sub.id == "register_cpu_ci":
                # Skip the import alias itself (it's not a Name node, it's
                # ast.alias inside ImportFrom). Any Name is a real use.
                # But if this Name belongs to a default-form Call we're
                # about to delete, it doesn't count. We approximate by
                # bailing out of the orphan claim only when the Name
                # appears outside such a call.
                parent = _find_parent_call_expr(tree, sub)
                if parent is None or not _is_default_form_register_cpu_ci(parent):
                    break
        else:
            for line in range(node.lineno, (node.end_lineno or node.lineno) + 1):
                to_remove.add(line)
    return to_remove


def _find_parent_call_expr(tree: ast.Module, target: ast.AST) -> ast.Call | None:
    """Return the enclosing ast.Call whose `.func` is `target`, or None.

    Used to decide whether a `Name('register_cpu_ci')` belongs to a
    default-form call (which we are about to remove) or some other use
    (which keeps the import alive).
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and node.func is target:
            return node
    return None


def _process_one_file(path: Path, dry_run: bool) -> tuple[bool, int]:
    """Process a single file. Returns (changed, removed_call_count)."""
    text = path.read_text()
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError:
        return False, 0

    target_call_lines: set[int] = set()
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            if _is_default_form_register_cpu_ci(node.value):
                for line in range(node.lineno, (node.end_lineno or node.lineno) + 1):
                    target_call_lines.add(line)

    if not target_call_lines:
        return False, 0

    orphan_import_lines = _find_orphan_import_lines(tree, text)
    lines_to_remove = target_call_lines | orphan_import_lines

    src_lines = text.splitlines(keepends=True)
    kept: list[str] = []
    for idx, line in enumerate(src_lines, start=1):
        if idx in lines_to_remove:
            continue
        kept.append(line)

    # Collapse runs of blank lines that result from the deletions: keep at
    # most one consecutive blank line.
    cleaned: list[str] = []
    last_was_blank = False
    for line in kept:
        is_blank = line.strip() == ""
        if is_blank and last_was_blank:
            continue
        cleaned.append(line)
        last_was_blank = is_blank

    # Trim leading blank lines that resulted from removing an early import.
    while cleaned and cleaned[0].strip() == "":
        cleaned.pop(0)

    new_text = "".join(cleaned)
    if new_text == text:
        return False, 0

    if not dry_run:
        path.write_text(new_text)
    return True, len(target_call_lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print files that would change without writing.",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=".",
        help="Repo root from which `tests/fast/` is resolved (default: cwd).",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    target_dir = root / TARGET_SUBTREE
    if not target_dir.is_dir():
        print(f"error: {target_dir} is not a directory", file=sys.stderr)
        return 2

    changed_files = 0
    total_calls_removed = 0
    for py in sorted(target_dir.rglob("*.py")):
        changed, removed = _process_one_file(py, args.dry_run)
        if changed:
            changed_files += 1
            total_calls_removed += removed
            rel = py.relative_to(root)
            print(f"{'would strip' if args.dry_run else 'stripped'} {removed} call(s) in {rel}")

    print(
        f"{'would change' if args.dry_run else 'changed'} {changed_files} file(s); removed {total_calls_removed} default-form register_cpu_ci call(s)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
