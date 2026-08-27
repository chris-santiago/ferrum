"""AST-level regression guard: no ``desugar_*`` function may silently
drop kwargs.

Every ``desugar_<name>`` function in ``src/ferrum/marks/`` must satisfy
ONE of:

1. **No ``**kwargs`` parameter at all** — Python's TypeError for unknown
   kwargs is the enforcement (the cleanest fix, used by most desugars).

2. **Calls ``validate_user_mark_kwargs(...)``** in its body — explicitly
   validates user-supplied kwargs against the renderer-level allowlist
   and raises ``TypeError`` on unknown keys. Used by the four 10e
   desugars that *also* forward known visual kwargs into their layers.

If a future contributor adds a new ``desugar_<name>(*, **kwargs)`` that
silently drops unknown keys, this test fails. The fix is to either
remove ``**kwargs`` (preferred) or wire up validation via
``ferrum.marks._mark_kwargs.validate_user_mark_kwargs``.

The guarantee applies to every module that contributes desugars: every
``*.py`` directly under ``src/ferrum/marks/`` AND every ``*.py`` directly
under ``src/ferrum/marks/diagnostic/`` (both globbed, not a hardcoded file
list or an underscore-prefix filter — a future ``marks/<new_family>.py`` or
``marks/diagnostic/<new_family>.py`` is picked up automatically regardless
of naming convention; files with no top-level ``desugar_*`` function are
simply harmless to include).

A second, broader guard (below, "Finding P9") polices a different blind
spot: a ``desugar_*`` function can accept a *declared* parameter (not a
stray kwarg) and never give it any effect — whether by explicitly
``del``eting it or simply never referencing it in the body — discarding
whatever the caller passed with no error, no warning, and no effect (the
latter shape is the trivial way to defeat a `del`-only guard: delete the
`del` line). Every declared parameter of a ``desugar_*`` function must be
either genuinely used, or justified by exactly one of (design spec
``.claude/output/specs/2026-08-27-findings-remediation-design.md`` §6,
"Amended 2026-08-27 during execution (Task 14, closing the P9 AST
guard)"):

(a) **Same-method ``data_transform`` wiring** — the mixin method of the
    same name (``desugar_roc`` <-> ``Chart.mark_roc``) passes
    ``data_transform=<callable>`` to ``_set_composite_mark``, and that
    callable (inline lambda, or a locally-defined nested function
    referenced by name — the ``top_k`` pattern) references the deleted
    parameter as a free variable. Verified directly from the mixin
    method's AST, not asserted by a hardcoded allowlist.
(b) **Informational-kwargs registry membership, load-bearing** — the
    parameter is listed in
    ``ferrum.marks._informational_kwargs.INFORMATIONAL_KWARGS`` under
    the mark's name, AND the correspondingly-named mixin method calls
    ``ferrum.marks._informational_kwargs.warn_informational_kwarg(mark,
    param, message)`` — the sole runtime consumer of the registry, which
    itself raises if ``(mark, param)`` is not registered. Both halves
    are verified here: registry membership by dict lookup
    (``_justify_del_name``), and the call site by walking the mixin
    method's AST for a ``warn_informational_kwarg`` call whose first two
    literal arguments match ``(mark, param)``
    (``_mixin_calls_informational_warn``). Registry membership *alone*,
    with no matching call, is NOT sufficient — that gap (a one-line
    registry addition silencing this guard with no warning ever firing)
    is exactly the drift branch (b) exists to prevent; see
    ``test_ast_guard_rejects_registry_membership_without_a_warn_call``.
(c) **Dispatcher-contract exemption (``x_field``/``y_field`` only)** —
    not a per-mark disposition chosen ad hoc like (a)/(b), but the
    guard's structural expression of the plan's pre-existing "Dispatcher
    contract preserved" invariant (design spec §7): every ``desugar_*``
    is invoked positionally as ``(x_field, y_field, ...)`` by
    ``_split_style_kwargs``/``_resolve_pending`` regardless of whether
    the mark has genuine user-driven x/y fields, so composite/diagnostic
    marks without real x/y fields discard them structurally. Verified
    positionally — the deleted name must be one of the desugar's
    *literal first two positional parameters*, and both of those must be
    named exactly ``x_field``/``y_field`` — not by name-matching alone,
    so a same-named parameter used for something else in a hypothetical
    future desugar would not accidentally qualify.

Any declared parameter not genuinely used and not covered by (a), (b), or
(c) fails the suite — whether or not it is explicitly ``del``eted.
"genuinely used" here means "referenced" (see ``_is_referenced_in_body``'s
docstring for what that does and does not prove), not "read and given some
observable effect".
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from ferrum.marks._informational_kwargs import INFORMATIONAL_KWARGS

_MARKS_DIR = Path(__file__).parent.parent / "src" / "ferrum" / "marks"


def _find_desugar_functions() -> list[tuple[str, ast.FunctionDef]]:
    """Walk every marks/*.py and marks/diagnostic/*.py module and yield
    (qualname, FunctionDef) pairs for every top-level function named
    ``desugar_*``.

    Both directories are globbed rather than enumerated by a hardcoded file
    list or filtered by a naming convention (e.g. an underscore prefix): a
    hardcoded list or a convention-based filter silently stops covering a
    future ``marks/<new_family>.py`` / ``marks/diagnostic/<new_family>.py``
    that doesn't happen to follow today's naming pattern (this is exactly
    how the previous ``diagnostic.py`` module-docstring reference went
    stale once the diagnostic desugars moved into a ``diagnostic/``
    subpackage, and how a prior version of this function only globbed
    ``diagnostic/_*.py``). Filtering by the ``desugar_`` name prefix below
    is what makes it safe to glob every ``*.py`` file in both directories,
    including mixins/helpers/``__init__.py`` that declare no desugars at
    all.
    """
    out: list[tuple[str, ast.FunctionDef]] = []
    # Top-level mark modules.
    for path in sorted(_MARKS_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name.startswith("desugar_"):
                out.append((f"{path.name}::{node.name}", node))
    # Diagnostic subpackage domain modules.
    diag_dir = _MARKS_DIR / "diagnostic"
    if diag_dir.is_dir():
        for path in sorted(diag_dir.glob("*.py")):
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in tree.body:
                if isinstance(node, ast.FunctionDef) and node.name.startswith("desugar_"):
                    out.append((f"diagnostic/{path.name}::{node.name}", node))
    return out


def _has_var_kwargs(fn: ast.FunctionDef) -> bool:
    return fn.args.kwarg is not None


def _calls_validator(fn: ast.FunctionDef) -> bool:
    """True if the function body contains a call to
    ``validate_user_mark_kwargs`` (or its imported alias).
    """
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        target = node.func
        # Direct call: validate_user_mark_kwargs(...) or alias _validate(...).
        if isinstance(target, ast.Name) and target.id in {
            "validate_user_mark_kwargs",
            "_validate",
        }:
            return True
        # Attribute call: _mark_kwargs.validate_user_mark_kwargs(...).
        if isinstance(target, ast.Attribute) and target.attr == "validate_user_mark_kwargs":
            return True
    return False


_DESUGARS = _find_desugar_functions()


def test_marks_dir_discovered():
    """Sanity: the AST walker found a non-trivial set of desugars."""
    assert len(_DESUGARS) >= 25, (
        f"Expected ≥ 25 desugar functions across the marks package; "
        f"found {len(_DESUGARS)}. The discovery glob may be broken."
    )


@pytest.mark.parametrize("qualname,fn", _DESUGARS, ids=[q for q, _ in _DESUGARS])
def test_desugar_does_not_silently_drop_kwargs(qualname: str, fn: ast.FunctionDef):
    """Each desugar must either reject unknown kwargs at the call boundary
    (no ``**kwargs``) or explicitly validate them via
    ``validate_user_mark_kwargs``.
    """
    if not _has_var_kwargs(fn):
        return  # Path 1: Python's TypeError is the enforcement. Pass.
    if _calls_validator(fn):
        return  # Path 2: explicit validation. Pass.
    pytest.fail(
        f"{qualname}: accepts **{fn.args.kwarg.arg} but does not call "
        f"validate_user_mark_kwargs. Either drop the **kwargs parameter "
        f"(preferred) so Python raises TypeError on unknown keys at the "
        f"call boundary, or import "
        f"`ferrum.marks._mark_kwargs.validate_user_mark_kwargs` and call "
        f"it on the kwargs. Silent-drop is forbidden per the Phase 9+ "
        f"no-defer principle."
    )


def test_meta_test_rejects_silent_drop_function():
    """Negative control: feed the meta-test logic a synthetic desugar
    that accepts **kwargs without validating and verify it would fail.

    Without this, the meta-test could pass on a vacuously-empty rule
    (e.g. someone refactors away the `_calls_validator` branch and the
    test would silently let everything through). Verifying both arms of
    the discriminator on a controlled input keeps the test honest.
    """
    silent_drop = ast.parse("def desugar_evil(*, x, **kwargs):\n    return ()\n").body[0]
    explicit_validate = ast.parse(
        "def desugar_good(*, x, **kwargs):\n    _validate('good', kwargs)\n    return ()\n"
    ).body[0]
    no_kwargs = ast.parse("def desugar_clean(*, x):\n    return ()\n").body[0]

    assert _has_var_kwargs(silent_drop) and not _calls_validator(silent_drop)
    assert _has_var_kwargs(explicit_validate) and _calls_validator(explicit_validate)
    assert not _has_var_kwargs(no_kwargs)


# ---------------------------------------------------------------------------
# Finding P9: every `del <param>` in a desugar must be justified.
# ---------------------------------------------------------------------------

_MIXIN_FILES = tuple(sorted((_MARKS_DIR / "_chart_mixins").glob("*.py"))) + (
    _MARKS_DIR / "_chart_methods_statistical.py",
)


def _find_mixin_methods() -> dict[str, ast.FunctionDef]:
    """Return every top-level ``mark_*`` method defined on a mixin class,
    across ``marks/_chart_mixins/*.py`` and
    ``marks/_chart_methods_statistical.py`` (the two homes of the
    ``mark_<name>()`` counterpart to a ``desugar_<name>`` function),
    keyed by method name.
    """
    out: dict[str, ast.FunctionDef] = {}
    for path in _MIXIN_FILES:
        if not path.exists():
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name.startswith("mark_"):
                    out[item.name] = item
    return out


def _positional_param_names(fn: ast.FunctionDef) -> list[str]:
    return [a.arg for a in (*fn.args.posonlyargs, *fn.args.args)]


def _declared_param_names(fn: ast.FunctionDef) -> list[str]:
    """Every parameter name a desugar declares in its signature —
    positional-or-keyword and keyword-only — excluding ``**mark_kwargs``
    (a var-keyword catch-all cannot be "unreferenced": it exists precisely
    to accept arbitrary caller kwargs, and its own contents are policed by
    ``test_desugar_does_not_silently_drop_kwargs`` above) and any
    ``*args``.
    """
    return [a.arg for a in (*fn.args.posonlyargs, *fn.args.args, *fn.args.kwonlyargs)]


def _is_referenced_in_body(fn: ast.FunctionDef, name: str) -> bool:
    """True if *name* is read (``Load`` context) anywhere in the function's
    body statements. Restricted to ``fn.body`` (not the whole ``FunctionDef``
    subtree, which would also include the signature's default-value
    expressions) so a same-named default on some other parameter can't
    count as a "use". A ``del <name>`` reference does not count — its
    ``ast.Name`` node carries ``ctx=Del``, not ``Load``.

    This is a syntactic check, not a semantic one: it proves a reference
    exists, not that the reference has any effect. ``_ = palette`` or
    ``if palette: pass`` both satisfy it while still discarding the
    caller's value. Closing that gap requires knowing what "effect" means
    for arbitrary code and is out of reach for an AST guard; treat a
    passing result here as "not silently dropped by omission", not as
    "proven to do something".
    """
    for stmt in fn.body:
        for node in ast.walk(stmt):
            if isinstance(node, ast.Name) and node.id == name and isinstance(node.ctx, ast.Load):
                return True
    return False


def _is_dispatcher_contract_param(fn: ast.FunctionDef, name: str) -> bool:
    """True for ``x_field``/``y_field`` when they are literally the
    desugar's first two positional parameters — the uniform calling
    convention every ``desugar_*`` function is invoked through, not a
    per-mark disposition. Positional, not just name-matched, so a
    same-named parameter used for something else in a hypothetical future
    desugar would not accidentally qualify.
    """
    first_two = _positional_param_names(fn)[:2]
    return first_two == ["x_field", "y_field"] and name in first_two


def _data_transform_references_param(mixin_fn: ast.FunctionDef, param: str) -> bool:
    """True if some ``data_transform=<callable>`` keyword argument passed
    to a call inside *mixin_fn* references *param* as a free variable —
    either directly (an inline lambda body) or one level removed (a bare
    ``Name`` referencing a nested function defined earlier in the same
    method body, the ``top_k`` pattern: ``data_transform=_roc_prep`` where
    ``_roc_prep`` closes over the mixin method's parameter).
    """
    nested_defs = {node.name: node for node in mixin_fn.body if isinstance(node, ast.FunctionDef)}
    for node in ast.walk(mixin_fn):
        if not isinstance(node, ast.Call):
            continue
        for kw in node.keywords:
            if kw.arg != "data_transform" or kw.value is None:
                continue
            names = {n.id for n in ast.walk(kw.value) if isinstance(n, ast.Name)}
            if param in names:
                return True
            for referenced_name in names:
                nested = nested_defs.get(referenced_name)
                if nested is not None and any(
                    isinstance(n, ast.Name) and n.id == param for n in ast.walk(nested)
                ):
                    return True
    return False


def _mixin_calls_informational_warn(mixin_fn: ast.FunctionDef, mark_name: str, param: str) -> bool:
    """True if *mixin_fn* calls
    ``ferrum.marks._informational_kwargs.warn_informational_kwarg`` with
    *mark_name* and *param* as its first two (string-literal) positional
    arguments — the runtime half of the informational-kwargs contract, and
    what makes registry membership (branch (b)) load-bearing rather than
    documentation: a registry entry with no matching call here is rejected
    (see ``test_ast_guard_rejects_registry_membership_without_a_warn_call``).
    """
    for node in ast.walk(mixin_fn):
        if not isinstance(node, ast.Call):
            continue
        target = node.func
        is_name_match = isinstance(target, ast.Name) and target.id == "warn_informational_kwarg"
        is_attr_match = (
            isinstance(target, ast.Attribute) and target.attr == "warn_informational_kwarg"
        )
        if not (is_name_match or is_attr_match):
            continue
        if len(node.args) < 2:
            continue
        first, second = node.args[0], node.args[1]
        if (
            isinstance(first, ast.Constant)
            and first.value == mark_name
            and isinstance(second, ast.Constant)
            and second.value == param
        ):
            return True
    return False


def _justify_del_name(
    name: str,
    fn: ast.FunctionDef,
    mixin_fn: ast.FunctionDef | None,
    mark_suffix: str,
    informational: frozenset[str],
) -> bool:
    """True if ``del <name>`` inside desugar *fn* is justified by the
    dispatcher-contract exemption (c), same-method ``data_transform``
    wiring (a), or informational-registry membership *plus* a matching
    ``warn_informational_kwarg`` call (b).
    """
    if _is_dispatcher_contract_param(fn, name):
        return True
    if mixin_fn is not None and _data_transform_references_param(mixin_fn, name):
        return True
    if name not in informational:
        return False
    return mixin_fn is not None and _mixin_calls_informational_warn(mixin_fn, mark_suffix, name)


_MIXIN_METHODS = _find_mixin_methods()


@pytest.mark.parametrize("qualname,fn", _DESUGARS, ids=[q for q, _ in _DESUGARS])
def test_desugar_params_are_used_or_justified(qualname: str, fn: ast.FunctionDef):
    """Every declared parameter of a desugar must be genuinely used, or
    justified by the dispatcher contract, same-method ``data_transform``
    wiring, or the load-bearing informational-kwargs registry — see the
    module docstring for the three-way contract.

    This covers BOTH shapes of the P9 defect with one predicate
    (``_justify_del_name``, reused verbatim regardless of which shape is
    present): a parameter explicitly ``del``eted with no justification,
    and a parameter that is simply never referenced at all (no ``del``,
    no use) — the latter is the trivial way to defeat a ``del``-only
    guard, since deleting the `del` line alone reproduces the identical
    silent-discard defect.
    """
    mark_suffix = fn.name[len("desugar_") :]
    mixin_fn = _MIXIN_METHODS.get(f"mark_{mark_suffix}")
    informational = INFORMATIONAL_KWARGS.get(mark_suffix, frozenset())

    for name in _declared_param_names(fn):
        if _is_dispatcher_contract_param(fn, name):
            continue  # (c) -- x_field/y_field are exempt regardless of use.
        if _is_referenced_in_body(fn, name):
            continue  # Genuinely used -- nothing to justify.
        if _justify_del_name(name, fn, mixin_fn, mark_suffix, informational):
            continue
        pytest.fail(
            f"{qualname}: parameter `{name}` is never referenced in the "
            f"function body (whether or not it is explicitly `del`eted) "
            f"and is not justified. It must be either (a) wired via "
            f"`data_transform=` in the same-named mixin method "
            f"(`mark_{mark_suffix}`), (b) listed in "
            f"`ferrum.marks._informational_kwargs.INFORMATIONAL_KWARGS"
            f"[{mark_suffix!r}]` AND wired to a matching "
            f"`warn_informational_kwarg({mark_suffix!r}, {name!r}, ...)` "
            f"call in `mark_{mark_suffix}`, or (c) `x_field`/`y_field` as "
            f"the desugar's first two positional parameters (the "
            f"dispatcher-contract exemption). Silent parameter-drop is "
            f"forbidden per finding P9 / the Phase 9+ no-defer principle."
        )


def test_ast_guard_rejects_injected_unjustified_del():
    """Self-test (spec §9.12): parse a synthetic desugar with a `del` on a
    parameter justified by none of (a)/(b)/(c) through the same checker
    the parametrized test above uses, and verify it is rejected. Proves
    the guard is not vacuously true.
    """
    src = (
        "def desugar_evil(x_field, y_field, *, secret=None, **kwargs):\n"
        "    del x_field, y_field\n"
        "    del secret\n"
        "    return ()\n"
    )
    fn = ast.parse(src).body[0]
    assert _is_dispatcher_contract_param(fn, "x_field")
    assert _is_dispatcher_contract_param(fn, "y_field")
    assert not _justify_del_name(
        "secret", fn, mixin_fn=None, mark_suffix="evil", informational=frozenset()
    )


def test_ast_guard_rejects_never_referenced_param():
    """Self-test for the extended guard: a declared parameter that is
    neither ``del``eted nor ever read anywhere in the body -- the shape
    that defeats a `del`-only guard by simply deleting the `del` line --
    is rejected by ``_declared_param_names`` + ``_is_referenced_in_body``
    + ``_justify_del_name`` exactly like an unjustified `del` would be.
    This is what closed the guard's own P9-class gap (design spec §6
    amendment, Task 14 extension): three real desugar parameters
    (`desugar_boxen(palette)`, `desugar_confusion(normalize)`,
    `desugar_pdp(center)`) were in this exact state -- declared, never
    referenced, never `del`eted -- and passed the pre-extension guard.
    """
    src = "def desugar_evil(x_field, y_field, *, secret=None, **kwargs):\n    return ()\n"
    fn = ast.parse(src).body[0]
    assert "secret" in _declared_param_names(fn)
    assert not _is_referenced_in_body(fn, "secret")
    assert not _justify_del_name(
        "secret", fn, mixin_fn=None, mark_suffix="evil", informational=frozenset()
    )


def test_ast_guard_accepts_data_transform_justified_del():
    """Positive control (a): a `del` on a parameter the same-named mixin
    method wires through `data_transform=` — either as a bare name
    referencing a nested function (the `top_k` pattern) or an inline
    lambda — is accepted.
    """
    desugar_src = (
        "def desugar_good(x_field, y_field, *, average=None, **kwargs):\n"
        "    del x_field, y_field\n"
        "    del average\n"
        "    return ()\n"
    )
    fn = ast.parse(desugar_src).body[0]

    nested_fn_mixin_src = (
        "def mark_good(self, *, average=None, **kwargs):\n"
        "    def _prep(df):\n"
        "        return df.filter(average)\n"
        "    return self._set_composite_mark('good', desugar_good, {}, "
        "data_transform=_prep)\n"
    )
    nested_fn_mixin = ast.parse(nested_fn_mixin_src).body[0]
    assert _justify_del_name("average", fn, nested_fn_mixin, "good", frozenset())

    lambda_mixin_src = (
        "def mark_good(self, *, average=None, **kwargs):\n"
        "    return self._set_composite_mark('good', desugar_good, {}, "
        "data_transform=(lambda df: df.filter(average)))\n"
    )
    lambda_mixin = ast.parse(lambda_mixin_src).body[0]
    assert _justify_del_name("average", fn, lambda_mixin, "good", frozenset())

    unrelated_mixin_src = (
        "def mark_good(self, *, average=None, **kwargs):\n"
        "    return self._set_composite_mark('good', desugar_good, {})\n"
    )
    unrelated_mixin = ast.parse(unrelated_mixin_src).body[0]
    assert not _justify_del_name("average", fn, unrelated_mixin, "good", frozenset())


def test_ast_guard_accepts_registry_justified_del():
    """Positive control (b): a `del` on a parameter listed in the
    informational-kwargs registry AND wired to a matching
    ``warn_informational_kwarg(mark_suffix, param, ...)`` call in the
    same-named mixin method is accepted.
    """
    desugar_src = (
        "def desugar_good(x_field, y_field, *, flag=False, **kwargs):\n"
        "    del x_field, y_field\n"
        "    del flag\n"
        "    return ()\n"
    )
    fn = ast.parse(desugar_src).body[0]

    wired_mixin_src = (
        "def mark_good(self, *, flag=False, **kwargs):\n"
        "    if flag:\n"
        "        warn_informational_kwarg('good', 'flag', 'no effect here')\n"
        "    return self._set_composite_mark('good', desugar_good, {})\n"
    )
    wired_mixin = ast.parse(wired_mixin_src).body[0]
    assert _justify_del_name("flag", fn, wired_mixin, "good", informational=frozenset({"flag"}))
    assert _mixin_calls_informational_warn(wired_mixin, "good", "flag")


def test_ast_guard_rejects_registry_membership_without_a_warn_call():
    """Negative control: the exact drift branch (b) exists to prevent. A
    parameter listed in the registry but with no matching
    `warn_informational_kwarg` call anywhere in the mixin -- or no mixin
    method at all, or a call naming a different mark/param -- is rejected.
    Registry membership alone is never sufficient; without this control a
    future contributor could silence the guard for any parameter by adding
    one registry line, with no warning ever firing.
    """
    desugar_src = (
        "def desugar_good(x_field, y_field, *, flag=False, **kwargs):\n"
        "    del x_field, y_field\n"
        "    del flag\n"
        "    return ()\n"
    )
    fn = ast.parse(desugar_src).body[0]
    informational = frozenset({"flag"})

    unwired_mixin_src = (
        "def mark_good(self, *, flag=False, **kwargs):\n"
        "    return self._set_composite_mark('good', desugar_good, {})\n"
    )
    unwired_mixin = ast.parse(unwired_mixin_src).body[0]
    assert not _justify_del_name("flag", fn, unwired_mixin, "good", informational)

    assert not _justify_del_name("flag", fn, None, "good", informational)

    wrong_param_mixin_src = (
        "def mark_good(self, *, flag=False, **kwargs):\n"
        "    warn_informational_kwarg('good', 'other_flag', 'no effect here')\n"
        "    return self._set_composite_mark('good', desugar_good, {})\n"
    )
    wrong_param_mixin = ast.parse(wrong_param_mixin_src).body[0]
    assert not _justify_del_name("flag", fn, wrong_param_mixin, "good", informational)

    wrong_mark_mixin_src = (
        "def mark_good(self, *, flag=False, **kwargs):\n"
        "    warn_informational_kwarg('other_mark', 'flag', 'no effect here')\n"
        "    return self._set_composite_mark('good', desugar_good, {})\n"
    )
    wrong_mark_mixin = ast.parse(wrong_mark_mixin_src).body[0]
    assert not _justify_del_name("flag", fn, wrong_mark_mixin, "good", informational)


def test_informational_kwargs_registry_entries_wire_a_warn_once_call():
    """The structural half of the load-bearing link (spec §6 branch (b)):
    every registered ``(mark, param)`` pair must have a matching
    ``warn_informational_kwarg(mark, param, ...)`` call in the
    correspondingly-named mixin method. This is the direct assertion the
    spec review asked for -- checked independently of
    ``test_desugar_params_are_used_or_justified`` (which only exercises
    this path when a matching unused/deleted parameter exists) so the
    linkage is verified for the registry's own sake, not as a side effect
    of the desugar sweep.
    """
    for mark_suffix, params in INFORMATIONAL_KWARGS.items():
        mixin_fn = _MIXIN_METHODS.get(f"mark_{mark_suffix}")
        assert mixin_fn is not None, (
            f"INFORMATIONAL_KWARGS[{mark_suffix!r}] has no corresponding "
            f"mark_{mark_suffix} mixin method to wire the warning into."
        )
        for param in params:
            assert _mixin_calls_informational_warn(mixin_fn, mark_suffix, param), (
                f"mark_{mark_suffix} does not call warn_informational_kwarg"
                f"({mark_suffix!r}, {param!r}, ...); "
                f"INFORMATIONAL_KWARGS[{mark_suffix!r}] lists {param!r} but "
                f"nothing warns about it."
            )


def test_informational_kwargs_registry_entries_correspond_to_real_desugars():
    """Every registry key must name a mark that still has a desugar (and
    every listed parameter must still appear in that desugar's
    signature) — otherwise the registry is documenting a parameter that
    no longer exists, which the guard would accept without ever being
    exercised.
    """
    desugars_by_name = {fn.name: fn for _, fn in _DESUGARS}
    for mark_suffix, params in INFORMATIONAL_KWARGS.items():
        desugar_name = f"desugar_{mark_suffix}"
        fn = desugars_by_name.get(desugar_name)
        assert fn is not None, (
            f"INFORMATIONAL_KWARGS[{mark_suffix!r}] names a mark with no "
            f"matching {desugar_name}; remove the stale entry."
        )
        param_names = {a.arg for a in (*fn.args.posonlyargs, *fn.args.args, *fn.args.kwonlyargs)}
        for param in params:
            assert param in param_names, (
                f"INFORMATIONAL_KWARGS[{mark_suffix!r}] lists {param!r}, "
                f"which is not a parameter of {desugar_name}."
            )


def test_warn_informational_kwarg_raises_for_unregistered_pair():
    """The runtime half of the load-bearing link, as a tested contract
    rather than only a module doctest (doctests are not collected by the
    default ``pytest`` run here -- no ``--doctest-modules`` in ``addopts``).
    """
    from ferrum.marks._informational_kwargs import warn_informational_kwarg

    with pytest.raises(ValueError, match="not registered"):
        warn_informational_kwarg("no_such_mark", "x", "unused")


def test_validator_helper_is_importable():
    """The shared validation helper exists at the expected path so future
    desugars have one canonical place to import from.
    """
    from ferrum.marks._mark_kwargs import (
        apply_user_mark_kwargs,
        validate_user_mark_kwargs,
    )

    assert callable(validate_user_mark_kwargs)
    assert callable(apply_user_mark_kwargs)
    # Round-trip: empty dict passes through; unknown raises.
    assert validate_user_mark_kwargs("foo", {}) == {}
    with pytest.raises(TypeError, match="not_a_real_kwarg"):
        validate_user_mark_kwargs("foo", {"not_a_real_kwarg": 1})
