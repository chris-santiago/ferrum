#!/usr/bin/env python
"""Audit ferrum docstrings for *parameter drift* against actual signatures.

Lint (ruff D-rules) and ``tests/test_docstring_coverage.py`` enforce docstring
*presence and structure*, but neither catches **drift**: a NumPy ``Parameters``
section that lists a param the signature no longer has, or omits one the
signature gained. This script closes that gap deterministically.

For every function / method / class with a NumPy-style docstring containing a
``Parameters`` section, it compares:

* the real signature params (from ``ast``; ``self``/``cls``/``*args``/``**kwargs``
  excluded), and
* the param *names* documented in the ``Parameters`` section,

and splits findings into two tiers:

* **HIGH-CONFIDENCE** — a named param undocumented (``MISSING``), or a documented
  name absent from the signature on a function with no ``**kwargs`` sink
  (``STALE``). These are real drift.
* **LOW-CONFIDENCE** — a documented name absent from the signature but a
  ``**kwargs`` sink exists, so it is plausibly a forwarded-kwarg doc (the common,
  legitimate ferrum pattern for ``mark_*``/``encode``/``Theme``). Reported for
  eyeballing, not failed on.

Private symbols (leading underscore) are skipped: per the ferrum-docstrings
convention they are prose-only and not expected to carry a complete
``Parameters`` block. Classes are handled per the same convention — the **class**
docstring owns the ``__init__`` ``Parameters``, so a class's documented params
are diffed against its ``__init__`` signature.

Usage::

    uv run --no-sync python scripts/audit_docstring_drift.py [paths...]

Defaults to scanning ``src/ferrum``. Exit code is non-zero only when
HIGH-CONFIDENCE drift is found, so it can double as a CI gate.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

# Params that are never documented even when present in a signature.
_SKIP_PARAMS = {"self", "cls"}

# A NumPy Parameters entry header. Either "name[, name] : type" or — using the
# type-less form NumPy also permits — just "name[, name]" on its own line.
# Only the names (before " : ") matter; they must be bare identifiers so prose
# lines that slip to indent 0 are not mistaken for params.
_NAMES = re.compile(r"^[*\w][\w, *]*$")


def _sig_params(node: ast.FunctionDef | ast.AsyncFunctionDef):
    """Return (named_params, vararg_kwarg_names, has_var_kwargs).

    ``named_params``: positional + keyword-only, sans self/cls.
    ``vararg_kwarg_names``: the ``*args``/``**kwargs`` identifiers (documenting
    these by name is legitimate, so they must not count as stale).
    ``has_var_kwargs``: whether a ``**kwargs`` sink exists — when it does, an
    "extra" documented param is plausibly a forwarded kwarg, not drift.
    """
    a = node.args
    names = [
        arg.arg for arg in (*a.posonlyargs, *a.args, *a.kwonlyargs) if arg.arg not in _SKIP_PARAMS
    ]
    var_names: set[str] = set()
    if a.vararg:
        var_names.add(a.vararg.arg)
    if a.kwarg:
        var_names.add(a.kwarg.arg)
    return names, var_names, a.kwarg is not None


def _doc_params(docstring: str) -> list[str] | None:
    """Param names from a NumPy ``Parameters`` section, or None if no section.

    ``ast.get_docstring`` returns the docstring already dedented (``cleandoc``),
    so section content sits at indent 0: a param entry is an indent-0
    ``name : type`` line, its description lines are indented beneath it, and the
    section ends at the next indent-0 header underlined by dashes.
    """
    lines = docstring.splitlines()
    # Find the "Parameters" header followed by an "----" underline.
    start = None
    for i in range(len(lines) - 1):
        if lines[i].strip() == "Parameters" and set(lines[i + 1].strip()) == {"-"}:
            start = i + 2
            break
    if start is None:
        return None

    names: list[str] = []
    for j in range(start, len(lines)):
        line = lines[j]
        stripped = line.strip()
        if not stripped:
            continue
        indent = len(line) - len(line.lstrip())
        if indent > 0:
            continue  # description line beneath a param entry
        # Indent-0: either the next section header (followed by dashes) or a param.
        if j + 1 < len(lines) and set(lines[j + 1].strip()) == {"-"}:
            break
        # Strip an optional " : type" suffix, leaving just the name list.
        head = stripped.split(" : ", 1)[0].strip()
        if _NAMES.fullmatch(head):
            for raw in head.split(","):
                name = raw.strip().lstrip("*").strip()
                if name and name not in _SKIP_PARAMS:
                    names.append(name)
    return names


def _check(node, doc_owner_name: str, sig_node, path: Path, findings: list) -> None:
    """Diff documented params (on node) against signature params (on sig_node)."""
    docstring = ast.get_docstring(node)
    if not docstring:
        return
    documented = _doc_params(docstring)
    if documented is None:
        return  # no Parameters section -> prose-only, nothing to drift
    sig, var_names, has_var_kwargs = _sig_params(sig_node)
    sig_set, doc_set = set(sig), set(documented)
    missing = [p for p in sig if p not in doc_set]
    # A documented name that isn't a real param. If a **kwargs sink exists it is
    # plausibly a forwarded kwarg (legitimate), so classify separately. The
    # *args/**kwargs identifiers themselves are never "stale".
    extra = [p for p in documented if p not in sig_set and p not in var_names]
    stale = [] if has_var_kwargs else extra
    forwarded = extra if has_var_kwargs else []
    if missing or stale or forwarded:
        findings.append(
            {
                "symbol": doc_owner_name,
                "loc": f"{path}:{node.lineno}",
                "missing": missing,
                "stale": stale,
                "forwarded": forwarded,
            }
        )


def audit_file(path: Path, findings: list) -> None:
    """Append param-drift findings for every public symbol in ``path``."""
    tree = ast.parse(path.read_text(), filename=str(path))

    def walk(scope: ast.AST) -> None:
        for child in ast.iter_child_nodes(scope):
            # Private symbols (leading underscore) are internal helpers; per the
            # ferrum-docstrings convention they are prose-only and not expected
            # to carry a complete NumPy Parameters block, so skip them.
            name = getattr(child, "name", "")
            if name.startswith("_"):
                continue
            if isinstance(child, ast.ClassDef):
                # Class docstring owns __init__ Parameters (ferrum convention).
                init = next(
                    (
                        n
                        for n in child.body
                        if isinstance(n, ast.FunctionDef) and n.name == "__init__"
                    ),
                    None,
                )
                if init is not None:
                    _check(child, child.name, init, path, findings)
                # Recurse for public methods (documented on themselves). Private
                # and dunder methods are skipped (prose-only by convention).
                for n in child.body:
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if n.name.startswith("_"):
                            continue
                        _check(n, f"{child.name}.{n.name}", n, path, findings)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                _check(child, child.name, child, path, findings)
                walk(child)  # nested defs are rare but cheap to cover

    walk(tree)


def main(argv: list[str]) -> int:
    """Scan the given paths (default ``src/ferrum``) and report param drift."""
    roots = [Path(p) for p in argv[1:]] or [Path("src/ferrum")]
    files: list[Path] = []
    for root in roots:
        files.extend(sorted(root.rglob("*.py")) if root.is_dir() else [root])

    findings: list = []
    for f in files:
        if f.name == "_core.pyi":
            continue
        audit_file(f, findings)

    if not findings:
        print("No docstring parameter drift found.")
        return 0

    findings.sort(key=lambda d: d["loc"])
    high = [d for d in findings if d["missing"] or d["stale"]]
    low = [d for d in findings if d["forwarded"] and not (d["missing"] or d["stale"])]

    print(f"=== HIGH-CONFIDENCE DRIFT ({len(high)} symbols) ===")
    print(
        "(MISSING: named param undocumented. STALE: documented param absent and no **kwargs sink.)\n"
    )
    for d in high:
        print(f"  {d['symbol']}  ({d['loc']})")
        if d["missing"]:
            print(f"      MISSING from docstring: {', '.join(d['missing'])}")
        if d["stale"]:
            print(f"      STALE in docstring: {', '.join(d['stale'])}")

    print(
        f"\n=== LOW-CONFIDENCE ({len(low)} symbols) — documented names not in signature, but **kwargs present ==="
    )
    print("(Likely forwarded-kwarg docs — legitimate — but verify none are truly stale.)\n")
    for d in low:
        print(f"  {d['symbol']}  ({d['loc']}): {', '.join(d['forwarded'])}")
    # Only high-confidence drift is a failure; forwarded-kwarg docs are expected.
    return 1 if high else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
