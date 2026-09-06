"""Batch-D T13: drift guard for `lmplot`'s `_VALID_METHODS`.

Closes the T12 spec-review carry (`.sdd/task-12-spec-reviewer.md` cycle 5's
"⚠️ `_VALID_METHODS` widening blind spot"): T12 shipped `_VALID_METHODS` as a
hand-picked, routing-coupled subset (`{"lm", "logistic", "glm", "loess",
"robust"}`) that refused every other `transform_registry("smooth")["methods"]`
spelling ("quadratic"/"cubic"/"sqrt"/"log"/"gam") even though `mark_smooth`
already rendered all of them -- a widening in the registry could never be
detected because nothing derived from it. T13 routes the remaining smooth
methods through `lmplot` (see `regression.py`'s `_SMOOTH_ROUTED_METHODS`
module comment for the full "why glm stays dedicated" reasoning) and DERIVES
`_VALID_METHODS` from the registry instead of hand-listing it, so this test
module now pins the *shape* of that derivation rather than a frozen literal
that would itself go stale the next time the registry grows a method.

Two self-declared tautologies lived here until 2026-09-06 -- one restating
`regression.py`'s own subtraction formula, one vacuous given it. Their real
duty (a new registry smooth method must not slip in unnoticed) is discharged
by `tests/test_lmplot_routing.py::test_widened_methods_are_exactly_the_new_
registry_smooth_names`, which hard-lists the widened names. What they are
replaced by is the guard nothing had: the *prose* enumeration in `lmplot`'s
docstring, checked against the derived set, since derived code beside a
hand-listed docstring is the same mirror class this batch set out to kill.
"""

from __future__ import annotations

import re

from ferrum import _core
from ferrum.plots.regression import (
    _DEDICATED_FIT_METHODS,
    _SMOOTH_ROUTED_METHODS,
    _VALID_METHODS,
    lmplot,
)


def test_valid_methods_is_exactly_the_smooth_routed_and_dedicated_split():
    assert _VALID_METHODS == _SMOOTH_ROUTED_METHODS | _DEDICATED_FIT_METHODS


def test_dedicated_fit_methods_are_real_registry_kinds():
    # Each must resolve without raising -- an unknown kind raises ValueError.
    for kind in _DEDICATED_FIT_METHODS:
        entry = _core.transform_registry(kind)
        assert "outputs" in entry and "params" in entry


def test_dedicated_fit_methods_share_the_fit_output_schema():
    """`logistic`/`glm`/`robust` all conform to the shared `x, y, ci_lower,
    ci_upper` fit-family schema (spec §4.7/M6c) -- the schema
    `_fit_chart_from_named_transform` reads via `registry_output_columns`."""
    for kind in _DEDICATED_FIT_METHODS:
        outputs = dict(_core.transform_registry(kind)["outputs"])
        assert set(outputs) == {"x", "y", "ci_lower", "ci_upper"}


def _method_entry() -> str:
    """`lmplot`'s ``method`` parameter entry, the one prose enumeration.

    Sliced from the docstring at the entry header and ended at the next
    parameter header, so an enumeration added to a *different* entry cannot
    be mistaken for this one.
    """
    doc = lmplot.__doc__ or ""
    entry = doc[doc.index("    method : str") :]
    end = re.search(r"\n    \w+ : ", entry[1:])
    return entry[: end.start() + 1] if end else entry


def _documented_methods() -> set[str]:
    """Every double-backticked quoted name in that entry."""
    return set(re.findall(r'``"([a-z_]+)"``', _method_entry()))


def test_the_docstring_enumeration_matches_the_derived_method_set():
    """The prose list and the derived set are one claim, checked.

    `_VALID_METHODS` is derived from the registry, but `lmplot`'s docstring
    spells the accepted values out for the reader -- so a sixth registry
    smooth method would widen the code and leave the documentation stale
    with nothing failing. (`"linear"`, the alias exclusion, appears only
    inside the entry's subtraction formula, which is one backticked span
    rather than a quoted name, so it is not part of this enumeration.)
    """
    assert _documented_methods() == _VALID_METHODS


def test_the_docstring_enumeration_is_not_vacuous():
    """The extraction above must actually find the entry -- an empty or
    mis-sliced parse would make the comparison meaningless."""
    entry = _method_entry()
    assert entry.startswith("    method : str")
    assert len(_documented_methods()) >= 10


def test_the_order_entry_gives_every_routed_method_a_degree_disposition():
    """The one place a per-method list is content rather than duplication.

    `order` means something different for each method it reaches -- a
    polynomial degree, a fixed no-op, a LOESS local-fit degree, or a refusal
    -- so its entry documents an acceptance matrix. That matrix has to cover
    every method `order` can reach, or a newly-routed method arrives with
    undocumented `order` behavior.
    """
    doc = lmplot.__doc__ or ""
    start = doc.index("    order : int or None")
    matrix = doc[start:]
    undocumented = sorted(name for name in _SMOOTH_ROUTED_METHODS if f'``"{name}"``' not in matrix)
    assert not undocumented, (
        f"lmplot's `order` entry documents no degree disposition for {undocumented}"
    )


def test_the_internal_constant_name_is_not_leaked_into_user_facing_prose():
    """`_SMOOTH_ROUTED_METHODS` is a private module constant; a docstring
    that points the reader at it instead of naming the methods is telling a
    library user to go read the source."""
    assert "_SMOOTH_ROUTED_METHODS" not in (lmplot.__doc__ or "")


def test_glm_is_excluded_from_the_derived_smooth_set_not_forgotten():
    """`"glm"` is a real registry `smooth` method but is deliberately kept out
    of `_SMOOTH_ROUTED_METHODS` -- `lmplot`'s `"glm"` name stays mapped to the
    standalone `Glm` transform (`_DEDICATED_FIT_METHODS`), not
    `Smooth(method="glm")`. Confirms the exclusion is a live registry member,
    not a name that already doesn't exist (which would make the exclusion
    vacuous)."""
    registry_smooth_methods = set(_core.transform_registry("smooth")["methods"])
    assert "glm" in registry_smooth_methods
    assert "glm" not in _SMOOTH_ROUTED_METHODS
    assert "glm" in _DEDICATED_FIT_METHODS
