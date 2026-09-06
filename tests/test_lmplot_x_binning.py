"""`lmplot(x_bins=, x_estimator=)` reaches every method it is accepted for.

Before R2 the two parameters forwarded for `method="lm"` alone and were
dropped in silence for the other nine methods `lmplot` accepts, justified
in-source by the claim that the widened set "don't accept binning the way `lm`
does". `crates/ferrum-core/src/transform/smooth.rs` contradicts that:
`apply_one_group` aggregates the x bins BEFORE it dispatches on method, so
binning is method-agnostic in the engine.

Two things are pinned per method: that a binned fit is a *different* fit
(the parameter reached the engine), and that `x_estimator` picks the
aggregate (two estimators over the same bins disagree). Both are compared
render against render, so a re-drop would fail rather than pass on the
absence of an assertion.
"""

from __future__ import annotations

import polars as pl
import pytest

import ferrum as fm
from ferrum.plots.regression import _SMOOTH_ROUTED_METHODS

#: Every method `lmplot` routes through `mark_smooth`. Derived, so a registry
#: widening is covered the day it lands rather than the day someone edits a
#: list here.
_ROUTED = sorted(_SMOOTH_ROUTED_METHODS)

#: The methods whose own fit transform takes no binning parameters at all.
_DEDICATED = ["logistic", "glm", "robust"]


def _frame(n: int = 90) -> pl.DataFrame:
    """Repeated x values, so binning genuinely aggregates rows."""
    return pl.DataFrame(
        {
            "x": [float(i % 15) + 1.0 for i in range(n)],
            "y": [float(i % 15) * 1.7 + (i % 4) * 3.0 for i in range(n)],
            "g": ["a" if i % 2 else "b" for i in range(n)],
        }
    )


def test_the_routed_set_is_not_empty():
    """The parametrisation below is derived; an empty set would make every
    case vacuous."""
    assert len(_ROUTED) >= 7, _ROUTED


@pytest.mark.parametrize("method", _ROUTED)
def test_x_bins_changes_the_fit_for_every_smooth_routed_method(method):
    frame = _frame()
    unbinned = fm.lmplot(frame, x="x", y="y", method=method).to_svg()
    binned = fm.lmplot(frame, x="x", y="y", method=method, x_bins=4).to_svg()
    assert unbinned != binned, f"x_bins was dropped for method={method!r}"


@pytest.mark.parametrize("method", _ROUTED)
def test_x_bins_changes_the_fit_under_hue_too(method):
    """The `hue=` branch carried its own copy of the carve-out, so it needs
    its own pin."""
    frame = _frame()
    unbinned = fm.lmplot(frame, x="x", y="y", method=method, hue="g").to_svg()
    binned = fm.lmplot(frame, x="x", y="y", method=method, hue="g", x_bins=4).to_svg()
    assert unbinned != binned, f"x_bins was dropped for method={method!r} with hue="


@pytest.mark.parametrize("method", _ROUTED)
def test_x_estimator_picks_the_aggregate_for_every_smooth_routed_method(method):
    frame = _frame()
    mean = fm.lmplot(frame, x="x", y="y", method=method, x_bins=4, x_estimator="mean").to_svg()
    maximum = fm.lmplot(frame, x="x", y="y", method=method, x_bins=4, x_estimator="max").to_svg()
    assert mean != maximum, f"x_estimator was dropped for method={method!r}"


@pytest.mark.parametrize("method", _DEDICATED)
def test_the_dedicated_fit_methods_still_ignore_binning(method):
    """`Logistic`/`Glm`/`Robust` expose no binning parameters, so `lmplot`
    has nothing to forward -- documented in its `x_bins` entry. Pinned so the
    difference between the two families stays deliberate."""
    frame = _frame()
    y = "g_int" if method == "logistic" else "y"
    if method == "logistic":
        frame = frame.with_columns((pl.col("x") % 2).alias("g_int"))
    plain = fm.lmplot(frame, x="x", y=y, method=method).to_svg()
    binned = fm.lmplot(frame, x="x", y=y, method=method, x_bins=4, x_estimator="mean").to_svg()
    assert plain == binned


def test_an_unknown_x_estimator_is_refused_by_the_engine():
    with pytest.raises(ValueError, match="unknown x_estimator"):
        fm.lmplot(_frame(), x="x", y="y", method="loess", x_bins=4, x_estimator="bogus").to_svg()
