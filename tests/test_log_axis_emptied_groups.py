"""A group a log axis empties keeps its place and is reported by the log warning alone.

On a log positional axis a statistic computes in the axis's space, where a group whose every value
is zero or negative has nothing to read. Each of the eight statistics that run in log space (bin,
kernel density, 2-d kernel density, contour, loess and least-squares smooths, a general linear
fit, a robust fit and a logistic fit) keeps the group's legend entry and draws nothing for it; the
log warning that counts the rows the axis cannot place is the only report, so no fit or interval
warning names the group a second time. A group the axis leaves alone but whose own fit degrades
keeps that fit's warning.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
import numpy as np
import polars as pl
import pytest

import ferrum as fm

_ROWS = 24


def _frame() -> pl.DataFrame:
    """Group ``p`` holds positive values; group ``q`` holds only negative ones, so a log ``x``
    empties ``q``. ``b`` is a binary response whose two outcomes both occur at every ``x`` of ``p``."""
    rng = np.random.default_rng(1)
    positive = np.exp(rng.normal(1.0, 0.5, _ROWS))
    negative = -np.abs(rng.normal(1.0, 0.3, _ROWS))
    return pl.DataFrame(
        {
            "x": np.concatenate([positive, negative]),
            "w": rng.normal(0.0, 1.0, 2 * _ROWS),
            "b": np.concatenate([(rng.random(_ROWS) > 0.5), (rng.random(_ROWS) > 0.5)]).astype(
                float
            ),
            "g": ["p"] * _ROWS + ["q"] * _ROWS,
        }
    )


_LOG_X = fm.X("x", scale=fm.LogScale())


def _smooth(method: str, y: str = "w") -> Callable[[pl.DataFrame], fm.Chart]:
    return lambda df: fm.Chart(df).mark_smooth(method=method).encode(x=_LOG_X, y=y, color="g")


_DOORS: dict[str, Callable[[pl.DataFrame], fm.Chart]] = {
    "bin": lambda df: fm.Chart(df).mark_histogram().encode(x=_LOG_X, color="g"),
    "kde": lambda df: fm.Chart(df).mark_density().encode(x=_LOG_X, color="g"),
    "kde_2d": lambda df: fm.Chart(df).mark_density().encode(x=_LOG_X, y="w", color="g"),
    "contour": lambda df: fm.Chart(df).mark_contour().encode(x=_LOG_X, y="w", color="g"),
    "loess": _smooth("loess"),
    "lm": _smooth("lm"),
    "glm": _smooth("glm"),
    "robust": lambda df: fm.regplot(df, x="x", y="w", hue="g", method="robust", logx=True),
    "logistic": lambda df: fm.regplot(df, x="x", y="b", hue="g", method="logistic", logx=True),
}

_OFF_SIDE = (
    f"x scale: {_ROWS} row(s) of 'x' are zero or negative, which a LogScale cannot place, so they "
    "are not drawn; filter them, or use fm.SymlogScale() for data that reaches zero"
)


def _render(chart: fm.Chart) -> tuple[str, list[str]]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        svg = chart.to_svg()
    return svg, [str(w.message) for w in caught]


@pytest.mark.parametrize("door", sorted(_DOORS))
def test_a_group_the_log_axis_empties_keeps_its_entry_and_only_the_log_warning_reports_it(
    door: str,
):
    """The chart over both groups warns exactly the log sentence; the twin over group ``p`` alone
    warns nothing and has no ``q`` entry, so the entry is the emptied group's own."""
    svg, shown = _render(_DOORS[door](_frame()))
    twin_svg, twin_shown = _render(_DOORS[door](_frame().filter(pl.col("g") == "p")))
    assert shown == [_OFF_SIDE]
    assert ">q<" in svg
    assert twin_shown == []
    assert ">q<" not in twin_svg


_ONE_ROW_GROUP = pl.DataFrame({"x": [3.0], "w": [0.5], "b": [1.0], "g": ["r"]})
_POINTS = "fewer than 2 usable (non-null, finite) x/y points"
_VALUES = "fewer than 2 usable (non-null, finite) values"
_OWN_FIT = {
    "lm": ("Smooth", _POINTS),
    "glm": ("Smooth", _POINTS),
    "loess": ("Smooth", _POINTS),
    "robust": ("Robust", _POINTS),
    "logistic": ("Logistic", _POINTS),
    "kde": ("Kde", _VALUES),
}


@pytest.mark.parametrize("door", sorted(_OWN_FIT))
def test_a_group_the_axis_leaves_alone_keeps_its_own_fit_warning_beside_the_log_warning(door: str):
    """Group ``r`` holds one positive row, so its fit degrades for its own reason: that warning
    stays, before the log warning, and ``q`` (emptied by the axis) is named by neither."""
    statistic, reason = _OWN_FIT[door]
    _, shown = _render(_DOORS[door](pl.concat([_frame(), _ONE_ROW_GROUP])))
    assert shown == [
        f"{statistic}: could not fit group g=r ({reason}); it degraded to an empty/flat output "
        "instead of failing the whole chart",
        _OFF_SIDE,
    ]
