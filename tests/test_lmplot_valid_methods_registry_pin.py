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
"""

from __future__ import annotations

from ferrum import _core
from ferrum.plots.regression import _DEDICATED_FIT_METHODS, _SMOOTH_ROUTED_METHODS, _VALID_METHODS


def test_valid_methods_is_exactly_the_smooth_routed_and_dedicated_split():
    assert _VALID_METHODS == _SMOOTH_ROUTED_METHODS | _DEDICATED_FIT_METHODS


def test_smooth_routed_methods_is_derived_from_the_registry_not_hand_listed():
    """The widening this test guards against: `_SMOOTH_ROUTED_METHODS` must
    equal the *live* registry set (minus the two documented exclusions), so a
    new registry smooth method automatically becomes an accepted `lmplot`
    `method=` value instead of silently staying refused.

    Honesty note (py-quality round-1, non-blocking): this assertion is a
    tautology by construction -- `regression.py` defines
    `_SMOOTH_ROUTED_METHODS` with this exact formula
    (`registry_smooth_methods - {"linear", "glm"}`), so this test can only
    fail if the two literal exclusion sets (here vs. `regression.py`) drift
    apart, not if the registry itself grows a method the source formula
    would then still auto-widen to include. It is a guard against the
    *exclusion set* changing unnoticed, not evidence that "a new registry
    smooth method auto-widens `lmplot`" -- that duty is actually discharged
    by `tests/test_lmplot_routing.py::test_widened_methods_are_exactly_the_
    new_registry_smooth_names`, which hard-lists the five widened names and
    fails the moment the registry (and therefore `_SMOOTH_ROUTED_METHODS`)
    grows a sixth."""
    registry_smooth_methods = set(_core.transform_registry("smooth")["methods"])
    assert _SMOOTH_ROUTED_METHODS == registry_smooth_methods - {"linear", "glm"}


def test_smooth_routed_methods_all_render_through_mark_smooth():
    """Every derived name must be a real, currently-accepted `Smooth` method
    -- guards against the derived set ever including a stale/renamed entry
    that `Smooth`'s own parser would reject."""
    registry_smooth_methods = set(_core.transform_registry("smooth")["methods"])
    assert _SMOOTH_ROUTED_METHODS <= registry_smooth_methods


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
