"""A ``Logistic`` or ``Glm`` fit that cannot be computed warns and draws, as ``Smooth`` and ``Robust`` do.

A facet panel, or a whole unfaceted call, whose data the fit cannot use draws its scatter and an
empty fit, and is named once by the shared fit warning in the shared usable wording. The data a
fit cannot use is: fewer than two usable (non-null, finite) points, no variance in ``x``, one
response class, a singular design, a perfectly separated response, a diverged iteration, and a log
axis that empties it. A panel the log axis empties is reported by the log warning alone, and a
whole frame it empties is refused by the axis. A response the method cannot read at all (``y``
outside 0 or 1 for ``Logistic``, ``y`` outside a ``Glm`` family) still refuses the genuine whole
frame with its typed error, and degrades a panel or a hue group. A fit that computes is
byte-identical and warns nothing. ``Robust`` and ``lm`` are the sibling controls: every row runs
on them as well, so a wrong expectation fails on a fit that already degrades.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Callable

import polars as pl
import pytest

import ferrum as fm
from tests.test_domain_rows_left_out import _domain_warning
from tests.test_log_axis_statistics import _no_value_on_side_refusal
from tests.test_null_group_warning import (
    FIT_DIVERGED,
    FIT_FEW_POINTS,
    FIT_NOT_BINARY,
    FIT_SEPARATED,
    FIT_SEPARATED_DIVERGED,
    FIT_SINGULAR,
    FIT_ZERO_VARIANCE,
    _all_keys_null_fallback_message,
    _facet_key_dropped_message,
    _fit_message,
    _layer_message,
    _list_message,
    _log_message,
    _null_key_dropped_message,
    _panel_fit_message,
    _panel_key_fallback_message,
    _panel_message,
    _panel_units,
    _render_capturing,
    _slot_message,
    _whole_fit_message,
)

_NAN = float("nan")
_INF = float("inf")

#: The method each door names, and the statistic that names its warnings.
_STATISTIC = {"logistic": "Logistic", "glm": "Glm", "robust": "Robust", "lm": "Smooth"}
#: The two fits that used to refuse.
_REPAIRED = ["logistic", "glm"]
#: The repaired fits and the sibling that already degrades.
_FITS = [*_REPAIRED, "robust"]

# Panel ``p`` fits for every method: eight rows whose binary response overlaps along ``x``.
_P_X = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
_P_Y = [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0]
_P_ROWS = len(_P_X)


def _panel_frame(
    x: list[Any],
    y: list[Any],
    *,
    g: list[Any] | None = None,
    x_dtype: Any = pl.Float64,
    y_dtype: Any = pl.Float64,
    p_y: list[float] | None = None,
) -> pl.DataFrame:
    """Panel ``p`` (a fit every method computes) and panel ``q`` holding ``x`` and ``y``.

    ``g`` is ``q``'s hue column (panel ``p`` is level ``a``) when given. ``p_y`` replaces panel
    ``p``'s response with one a family outside the binary can read.
    """
    q_rows = len(x)
    columns: dict[str, Any] = {
        "f": ["p"] * _P_ROWS + ["q"] * q_rows,
        "x": pl.Series(_P_X + x, dtype=pl.Float64).cast(x_dtype),
        "y": pl.Series((p_y or _P_Y) + y, dtype=pl.Float64).cast(y_dtype),
    }
    if g is not None:
        columns["g"] = pl.Series(["a"] * _P_ROWS + g, dtype=pl.Utf8)
    return pl.DataFrame(columns)


def _q_rows(frame: pl.DataFrame) -> pl.DataFrame:
    """The rows of panel ``q`` alone: the same data as one unfaceted call."""
    return frame.filter(pl.col("f") == "q")


# ---------------------------------------------------------------------------
# What every method says about each panel shape
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Outcome:
    """What panel ``q`` (or a whole call) of one shape draws and reports.

    ``reason`` is the fit's reason when it cannot be computed (only on the methods the shape
    lists in ``degrades``); ``census`` the channel the scatter names as unplaced; ``log_rows`` the
    rows the log axis cannot place; ``domain_rows`` the rows a log axis's explicit domain cuts (an
    infinite value); ``drawn`` the scatter circles of ``q``; ``axis_emptied`` whether the log axis
    leaves ``q`` no on-side value, so no fit is drawn for it; ``not_lm`` the methods whose reason
    differs from the shared one (``Smooth`` words a zero variance its own way).
    """

    reason: str | None = None
    census: str | None = None
    log_rows: int = 0
    domain_rows: int = 0
    drawn: int = 0
    axis_emptied: bool = False
    not_lm: bool = False


@dataclass(frozen=True)
class Shape:
    """The rows of panel ``q`` and its outcome on a linear and on a log x axis."""

    x: list[Any]
    y: list[Any]
    lin: Outcome
    log: Outcome | None
    degrades: tuple[str, ...] = ("logistic", "glm", "robust", "lm")


_BINARY = [0.0, 1.0, 0.0, 1.0]

_SHAPES: dict[str, Shape] = {
    "null_x": Shape(
        [None] * 4, _BINARY, Outcome(FIT_FEW_POINTS, census="x"), Outcome(FIT_FEW_POINTS, "x")
    ),
    "nan_x": Shape(
        [_NAN] * 4, _BINARY, Outcome(FIT_FEW_POINTS, census="x"), Outcome(FIT_FEW_POINTS, "x")
    ),
    "inf_x": Shape(
        [_INF] * 4,
        _BINARY,
        Outcome(FIT_FEW_POINTS, census="x"),
        Outcome(FIT_FEW_POINTS, domain_rows=4),
    ),
    "null_y": Shape(
        [1.0, 2.0, 3.0, 4.0],
        [None] * 4,
        Outcome(FIT_FEW_POINTS, census="y"),
        Outcome(FIT_FEW_POINTS, census="y"),
    ),
    "nan_y": Shape(
        [1.0, 2.0, 3.0, 4.0],
        [_NAN] * 4,
        Outcome(FIT_FEW_POINTS, census="y"),
        Outcome(FIT_FEW_POINTS, census="y"),
    ),
    "inf_y": Shape(
        [1.0, 2.0, 3.0, 4.0],
        [_INF] * 4,
        Outcome(FIT_FEW_POINTS, census="y"),
        Outcome(FIT_FEW_POINTS, census="y"),
    ),
    "one_row": Shape(
        [2.0], [1.0], Outcome(FIT_FEW_POINTS, drawn=1), Outcome(FIT_FEW_POINTS, drawn=1)
    ),
    "zero_variance": Shape(
        [3.0] * 4,
        _BINARY,
        Outcome(FIT_ZERO_VARIANCE, drawn=4, not_lm=True),
        Outcome(FIT_ZERO_VARIANCE, drawn=4, not_lm=True),
    ),
    "one_class": Shape(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        [1.0] * 6,
        Outcome(FIT_SINGULAR, drawn=6),
        None,
        degrades=("logistic",),
    ),
    "separated": Shape(
        [-100.0, -90.0, -80.0, 80.0, 90.0, 100.0],
        [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        Outcome(FIT_SEPARATED, drawn=6),
        None,
        degrades=("logistic",),
    ),
    "y_is_2": Shape(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        [0.0, 1.0, 2.0, 0.0, 1.0, 0.0],
        Outcome(FIT_NOT_BINARY, drawn=6),
        None,
        degrades=("logistic",),
    ),
    "diverged_small_scale": Shape(
        [0.0, 0.001, 0.002, 0.003, 0.004, 0.005],
        [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        Outcome(FIT_SEPARATED_DIVERGED, drawn=6),
        None,
        degrades=("logistic",),
    ),
    "diverged_large_offset": Shape(
        [1e6, 1e6 + 1, 1e6 + 2, 1e6 + 3],
        [0.0, 0.0, 1.0, 1.0],
        Outcome(FIT_SEPARATED_DIVERGED, drawn=4),
        None,
        degrades=("logistic",),
    ),
    "off_side_varied": Shape(
        [-1.0, -2.0, -3.0, -4.0],
        _BINARY,
        Outcome(drawn=4),
        Outcome(log_rows=4, axis_emptied=True),
    ),
    "off_side_repeated": Shape(
        [-3.0] * 4,
        _BINARY,
        Outcome(FIT_ZERO_VARIANCE, drawn=4, not_lm=True),
        Outcome(log_rows=4, axis_emptied=True),
    ),
    "one_on_side": Shape(
        [2.0, -1.0, -2.0, -3.0],
        _BINARY,
        Outcome(drawn=4),
        Outcome(FIT_FEW_POINTS, log_rows=3, drawn=1),
    ),
}


def _outcome(shape: Shape, *, logx: bool) -> Outcome | None:
    return shape.log if logx else shape.lin


def _cases(methods: list[str]) -> list[Any]:
    """Every (shape, method, logx) the table defines; ``lm`` only where its reason is shared."""
    cases = []
    for name, shape in _SHAPES.items():
        for logx in (False, True):
            outcome = _outcome(shape, logx=logx)
            if outcome is None:
                continue
            for method in methods:
                if method == "lm" and outcome.not_lm:
                    continue
                cases.append(
                    pytest.param(
                        name, method, logx, id=f"{name}-{method}-{'log' if logx else 'lin'}"
                    )
                )
    return cases


def _reason_of(shape: Shape, outcome: Outcome, method: str) -> str | None:
    """The reason ``method`` gives for the shape, or ``None`` when its fit computes."""
    return outcome.reason if method in shape.degrades else None


def _expected(method: str, shape: Shape, outcome: Outcome, *, scope: str) -> list[str]:
    """The ordered messages a ``method`` render of the shape raises, panel or whole call."""
    statistic = _STATISTIC[method]
    panel = scope == "panel"
    said: list[str] = []
    reason = _reason_of(shape, outcome, method)
    if reason is not None:
        said.append(
            _panel_fit_message(statistic, reason)
            if panel
            else _whole_fit_message(statistic, reason)
        )
    if outcome.census is not None:
        said.append(
            _panel_message("lmplot", outcome.census)
            if panel
            else _layer_message("regplot", outcome.census)
        )
    if outcome.log_rows:
        said.append(_log_message("x", "x", outcome.log_rows))
    if outcome.domain_rows:
        said.append(_domain_warning("x", "x", outcome.domain_rows))
    return said


def _fit_drawn(shape: Shape, outcome: Outcome, method: str) -> bool:
    """Whether ``q`` draws a fit: it computes, and the log axis leaves it a value."""
    return _reason_of(shape, outcome, method) is None and not outcome.axis_emptied


def _polylines(svg: str) -> int:
    return svg.count("<polyline")


def _circles(svg: str) -> int:
    return svg.count("<circle")


# ---------------------------------------------------------------------------
# A facet panel the fit cannot use warns and draws
# ---------------------------------------------------------------------------


def _lmplot(frame: pl.DataFrame, method: str, *, logx: bool = False, **kwargs: Any) -> fm.Chart:
    return fm.lmplot(
        frame, x="x", y="y", method=method, ci=None, show_metrics=False, logx=logx, **kwargs
    )


def _regplot(frame: pl.DataFrame, method: str, *, logx: bool = False, **kwargs: Any) -> fm.Chart:
    return fm.regplot(
        frame, x="x", y="y", method=method, ci=None, show_metrics=False, logx=logx, **kwargs
    )


@pytest.mark.parametrize(("name", "method", "logx"), _cases([*_FITS, "lm"]))
def test_a_panel_the_fit_cannot_use_is_named_once_and_drawn_empty(name, method, logx):
    """Panel ``q`` draws its scatter and no fit, panel ``p`` fits, and one sentence names ``q``.

    Rejects the whole chart refusing for one panel's data, a panel named twice, and a panel a log
    axis empties named by the fit as well as by the axis.
    """
    shape = _SHAPES[name]
    outcome = _outcome(shape, logx=logx)
    assert outcome is not None
    frame = _panel_frame(shape.x, shape.y)
    svg, messages = _render_capturing(lambda: _lmplot(frame, method, logx=logx, col="f").to_svg())
    assert messages == _expected(method, shape, outcome, scope="panel")
    assert _polylines(svg) == 1 + int(_fit_drawn(shape, outcome, method))
    assert _circles(svg) == _P_ROWS + outcome.drawn


@pytest.mark.parametrize(
    ("name", "method", "logx"),
    [
        case
        for case in _cases([*_FITS, "lm"])
        if not (_outcome(_SHAPES[case.values[0]], logx=case.values[2]) or Outcome()).axis_emptied
        # A y of 2 is a response `Logistic` cannot read: the whole call refuses (see the
        # y-of-two refusal test below), only a panel of it degrades.
        and not (case.values[0] == "y_is_2" and case.values[1] == "logistic")
    ],
)
def test_a_whole_call_the_fit_cannot_use_is_named_once_and_drawn_empty(name, method, logx):
    """One unfaceted call with the same rows draws its scatter and no fit, named once.

    Rejects a whole-frame refusal where the siblings warn and draw.
    """
    shape = _SHAPES[name]
    outcome = _outcome(shape, logx=logx)
    assert outcome is not None
    frame = _q_rows(_panel_frame(shape.x, shape.y))
    svg, messages = _render_capturing(lambda: _regplot(frame, method, logx=logx).to_svg())
    assert messages == _expected(method, shape, outcome, scope="whole")
    assert _polylines(svg) == int(_fit_drawn(shape, outcome, method))
    assert _circles(svg) == outcome.drawn


@pytest.mark.parametrize("method", [*_FITS, "lm"])
@pytest.mark.parametrize("name", ["off_side_varied", "off_side_repeated"])
def test_a_whole_frame_a_log_axis_empties_is_refused_by_the_axis(name, method):
    """A call whose every ``x`` is off the log axis's side is refused in the axis's voice.

    Rejects the fit refusing first (``need at least 2 ...`` or ``zero x variance``).
    """
    shape = _SHAPES[name]
    frame = _q_rows(_panel_frame(shape.x, shape.y))
    with pytest.raises(ValueError) as refused:
        _regplot(frame, method, logx=True).to_svg()
    expected = _no_value_on_side_refusal("x", "x", len(shape.x), min(shape.x), max(shape.x))
    assert str(refused.value) == expected


@pytest.mark.parametrize("method", [*_FITS, "lm"])
def test_every_panel_the_fit_cannot_use_draws_the_frame_and_names_each(method):
    """Every panel unfittable: the frame and axes draw, and each panel is named in panel order."""
    frame = _panel_frame([None] * 4, _BINARY).with_columns(x=pl.lit(None, dtype=pl.Float64))
    svg, messages = _render_capturing(lambda: _lmplot(frame, method, col="f").to_svg())
    statistic = _STATISTIC[method]
    assert messages == [
        _panel_fit_message(statistic, FIT_FEW_POINTS, "f=p"),
        _panel_fit_message(statistic, FIT_FEW_POINTS, "f=q"),
        _list_message("lmplot", _panel_units(["f=p", "f=q"]), "x"),
    ]
    assert "<svg" in svg
    assert _polylines(svg) == 0
    assert _circles(svg) == 0


@pytest.mark.parametrize("method", [*_FITS, "lm"])
def test_three_panels_the_fit_cannot_use_are_listed_as_the_engine_lists_them(method):
    """Three emptied panels share one scatter sentence joined ``A, B, and C``, each named by the
    fit in panel order."""
    statistic = _STATISTIC[method]
    frame = pl.DataFrame(
        {
            "f": ["p", "p", "q", "q", "r", "r"],
            "x": pl.Series([None] * 6, dtype=pl.Float64),
            "y": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        }
    )
    svg, messages = _render_capturing(lambda: _lmplot(frame, method, col="f").to_svg())
    assert messages == [
        *[_panel_fit_message(statistic, FIT_FEW_POINTS, f"f={panel}") for panel in "pqr"],
        _list_message("lmplot", _panel_units(["f=p", "f=q", "f=r"]), "x"),
    ]
    assert _polylines(svg) == 0


@pytest.mark.parametrize("method", [*_FITS, "lm"])
def test_two_grid_panels_the_fit_cannot_use_are_listed_with_the_comma_label_joiner(method):
    """Row-and-column panels carry a comma in their label, so the scatter sentence joins them
    with ``;``."""
    statistic = _STATISTIC[method]
    base = _panel_frame([None] * 4, _BINARY)
    frame = pl.concat(
        [base.with_columns(r=pl.lit("r1")), base.with_columns(r=pl.lit("r2"))]
    ).with_columns(x=pl.when(pl.col("f") == "q").then(None).otherwise(pl.col("x")).cast(pl.Float64))
    svg, messages = _render_capturing(lambda: _lmplot(frame, method, col="f", row="r").to_svg())
    assert messages == [
        _panel_fit_message(statistic, FIT_FEW_POINTS, "f=q, r=r1"),
        _panel_fit_message(statistic, FIT_FEW_POINTS, "f=q, r=r2"),
        _list_message("lmplot", _panel_units(["f=q, r=r1", "f=q, r=r2"]), "x"),
    ]
    assert _polylines(svg) == 2


@pytest.mark.parametrize("method", [*_FITS, "lm"])
def test_a_zero_row_frame_warns_the_fit_and_draws_the_frame(method):
    """A zero-row frame, faceted and not, is named by the fit's own sentence and draws its frame."""
    statistic = _STATISTIC[method]
    empty = _panel_frame([], [])[:0]
    svg, messages = _render_capturing(lambda: _lmplot(empty, method).to_svg())
    assert messages == [_whole_fit_message(statistic, FIT_FEW_POINTS)]
    assert "<svg" in svg
    svg, messages = _render_capturing(lambda: _lmplot(empty, method, col="f").to_svg())
    assert messages == [_panel_fit_message(statistic, FIT_FEW_POINTS, "f=")]
    assert "<svg" in svg


@pytest.mark.parametrize("method", [*_FITS, "lm"])
def test_a_facet_whose_key_is_null_on_every_row_is_named_after_the_drop(method):
    """Every facet key null: the rows belong to no panel and the one empty panel is named."""
    frame = _panel_frame([1.0, 2.0, 3.0, 4.0], _BINARY).with_columns(f=pl.lit(None, dtype=pl.Utf8))
    svg, messages = _render_capturing(lambda: _lmplot(frame, method, col="f").to_svg())
    assert messages == [
        _facet_key_dropped_message(12, "f"),
        _panel_fit_message(_STATISTIC[method], FIT_FEW_POINTS, "f="),
    ]
    assert "<svg" in svg


# ---------------------------------------------------------------------------
# The same on a column of every width and integer type
# ---------------------------------------------------------------------------

_DTYPE_CASES = [
    pytest.param(pl.Float32, [_NAN] * 4, FIT_FEW_POINTS, "x", id="float32-nan"),
    pytest.param(pl.Float32, [_INF] * 4, FIT_FEW_POINTS, "x", id="float32-inf"),
    pytest.param(pl.Float32, [None] * 4, FIT_FEW_POINTS, "x", id="float32-null"),
    pytest.param(pl.Int64, [None] * 4, FIT_FEW_POINTS, "x", id="int64-null"),
    pytest.param(pl.Int64, [3] * 4, FIT_ZERO_VARIANCE, None, id="int64-constant"),
    pytest.param(pl.Int32, [None] * 4, FIT_FEW_POINTS, "x", id="int32-null"),
]


@pytest.mark.parametrize("scope", ["panel", "whole"])
@pytest.mark.parametrize("method", _FITS)
@pytest.mark.parametrize(("dtype", "values", "reason", "census"), _DTYPE_CASES)
def test_a_column_of_any_numeric_type_the_fit_cannot_use_is_named(
    dtype, values, reason, census, method, scope
):
    """``x`` of each float width and integer type, wholly unusable or constant, is named the same.

    Rejects a refusal that only the Float64 path was repaired to avoid.
    """
    frame = _panel_frame(values, _BINARY, x_dtype=dtype)
    statistic = _STATISTIC[method]
    if scope == "panel":
        render = lambda: _lmplot(frame, method, col="f").to_svg()  # noqa: E731
        expected = [_panel_fit_message(statistic, reason)]
        if census:
            expected.append(_panel_message("lmplot", census))
    else:
        render = lambda: _regplot(_q_rows(frame), method).to_svg()  # noqa: E731
        expected = [_whole_fit_message(statistic, reason)]
        if census:
            expected.append(_layer_message("regplot", census))
    svg, messages = _render_capturing(render)
    assert messages == expected
    assert "<svg" in svg


@pytest.mark.parametrize("scope", ["panel", "whole"])
@pytest.mark.parametrize("method", _FITS)
@pytest.mark.parametrize(
    ("column", "dtype", "text"),
    [
        pytest.param("y", pl.Boolean, "column 'y' must be numeric, got Boolean", id="boolean-y"),
        pytest.param(
            "x", pl.Decimal(10, 2), "column 'x' must be numeric, got Decimal", id="decimal-x"
        ),
    ],
)
def test_a_column_that_is_not_numeric_still_refuses(column, dtype, text, method, scope):
    """A column the fit cannot read as numbers is refused with a typed error on every call."""
    frame = _panel_frame([1.0, 2.0, 3.0, 4.0], _BINARY).with_columns(
        pl.col(column).cast(dtype, strict=False)
    )
    statistic = _STATISTIC[method]
    with pytest.raises(ValueError, match=re.escape(f"{statistic}: {text}")):
        if scope == "panel":
            _lmplot(frame, method, col="f").to_svg()
        else:
            _regplot(_q_rows(frame), method).to_svg()


# ---------------------------------------------------------------------------
# Every door
# ---------------------------------------------------------------------------


def _step(method: str) -> Any:
    return {
        "logistic": fm.Logistic("x", "y", n_grid=20),
        "glm": fm.Glm("x", "y", family="gaussian", n_grid=20),
        "robust": fm.Robust("x", "y", n_grid=20),
    }[method]


def _scatter(frame: pl.DataFrame) -> fm.Chart:
    return fm.Chart(frame).mark_point().encode(x="x", y="y")


def _fit(frame: pl.DataFrame, method: str) -> fm.Chart:
    return fm.Chart(frame).transform(_step(method)).mark_line().encode(x="x", y="y")


@dataclass(frozen=True)
class Door:
    """One way to draw the fit: how it builds the chart and which entry names its scatter.

    ``build`` takes the frame, the method and whether the x axis is logarithmic; ``on_log`` says
    whether the door reads that flag (the figure functions do, the hand-written steps do not).
    ``census`` is the entry the scatter's own sentence starts with (``None`` for a door that draws
    no scatter); ``census_first`` whether a whole call's scatter sentence comes before the fit's.
    """

    build: Callable[[pl.DataFrame, str, bool], Any]
    census: str | None
    census_first: bool = False
    panel: str = "f=q"
    on_log: bool = False


_PANEL_DOORS: dict[str, Door] = {
    "lmplot": Door(lambda f, m, lg: _lmplot(f, m, logx=lg, col="f"), "lmplot", on_log=True),
    "lmplot-ci": Door(
        lambda f, m, lg: fm.lmplot(
            f, x="x", y="y", method=m, col="f", ci=95, show_metrics=False, logx=lg
        ),
        "lmplot",
        on_log=True,
    ),
    "lmplot-no-scatter": Door(
        lambda f, m, lg: _lmplot(f, m, logx=lg, col="f", scatter=False), None, on_log=True
    ),
    "lmplot-grid": Door(
        lambda f, m, lg: _lmplot(f.with_columns(r=pl.lit("u")), m, logx=lg, col="f", row="r"),
        "lmplot",
        panel="f=q, r=u",
        on_log=True,
    ),
    "hand-written": Door(lambda f, m, lg: _fit(f, m).facet(col="f"), None),
    "plus": Door(lambda f, m, lg: (_scatter(f) + _fit(f, m)).facet(col="f"), "layer 0: mark_point"),
    "plus-faceted-members": Door(
        lambda f, m, lg: _scatter(f).facet(col="f") + _fit(f, m).facet(col="f"),
        "layer 0: mark_point",
    ),
    "chart-layer": Door(
        lambda f, m, lg: _fit(f, m).facet(col="f").layer(fm.Layer(mark="point")), None
    ),
    "layerchart": Door(
        lambda f, m, lg: fm.LayerChart(_scatter(f).facet(col="f"), _fit(f, m).facet(col="f")),
        "layer 0: mark_point",
    ),
}
_WHOLE_DOORS: dict[str, Door] = {
    "lmplot": Door(lambda f, m, lg: _lmplot(f, m, logx=lg), "lmplot", on_log=True),
    "regplot": Door(lambda f, m, lg: _regplot(f, m, logx=lg), "regplot", on_log=True),
    "regplot-ci": Door(
        lambda f, m, lg: fm.regplot(f, x="x", y="y", method=m, ci=95, show_metrics=False, logx=lg),
        "regplot",
        on_log=True,
    ),
    "hand-written": Door(lambda f, m, lg: _fit(f, m), None),
    "plus": Door(lambda f, m, lg: _scatter(f) + _fit(f, m), "layer 0: mark_point"),
    "chart-layer": Door(lambda f, m, lg: _fit(f, m).layer(fm.Layer(mark="point")), None),
    "layerchart": Door(
        lambda f, m, lg: fm.LayerChart(_scatter(f), _fit(f, m)), "mark_point", census_first=True
    ),
}
_DOOR_SHAPES = ["null_x", "null_y", "one_row", "zero_variance"]
#: The log-axis shapes the figure-function doors take: the log warning alone, the same off-side
#: value repeated, one value on the side, and an unusable column.
_LOG_DOOR_SHAPES = ["off_side_varied", "off_side_repeated", "one_on_side", "null_x"]


def _door_expected(
    door: Door, method: str, shape: Shape, *, scope: str, logx: bool = False
) -> list[str]:
    statistic = _STATISTIC[method]
    outcome = _outcome(shape, logx=logx)
    assert outcome is not None
    reason = _reason_of(shape, outcome, method)
    fit = (
        []
        if reason is None
        else [
            _panel_fit_message(statistic, reason, door.panel)
            if scope == "panel"
            else _whole_fit_message(statistic, reason)
        ]
    )
    census: list[str] = []
    if door.census is not None and outcome.census is not None:
        census = [
            _panel_message(door.census, outcome.census, door.panel)
            if scope == "panel"
            else _layer_message(door.census, outcome.census)
        ]
    axis: list[str] = []
    if outcome.log_rows:
        axis.append(_log_message("x", "x", outcome.log_rows))
    if outcome.domain_rows:
        axis.append(_domain_warning("x", "x", outcome.domain_rows))
    if scope == "whole" and door.census_first:
        return census + fit + axis
    return fit + census + axis


def _door_cases(doors: dict[str, Door], shapes: list[str], *, logx: bool) -> list[Any]:
    """Every (door, shape, method) of ``doors`` on one axis; only the doors that read the axis
    flag run on the log axis."""
    return [
        pytest.param(door, shape, method, id=f"{door}-{shape}-{method}")
        for door, spec in doors.items()
        if spec.on_log or not logx
        for shape in shapes
        for method in _FITS
    ]


def _panel_door_render(
    door: Door, method: str, shape: Shape, *, logx: bool
) -> tuple[str, list[str]]:
    frame = _panel_frame(shape.x, shape.y)
    return _render_capturing(lambda: door.build(frame, method, logx).to_svg())


@pytest.mark.parametrize(
    ("door_name", "shape_name", "method"), _door_cases(_PANEL_DOORS, _DOOR_SHAPES, logx=False)
)
def test_every_door_names_a_panel_the_fit_cannot_use_and_draws_the_rest(
    door_name, shape_name, method
):
    """The panel is named once on each door, and panel ``p``'s fit is the one line drawn.

    Rejects a door whose route to the fit still refuses, or names the panel twice.
    """
    door, shape = _PANEL_DOORS[door_name], _SHAPES[shape_name]
    svg, messages = _panel_door_render(door, method, shape, logx=False)
    assert messages == _door_expected(door, method, shape, scope="panel")
    assert _polylines(svg) == 1


@pytest.mark.parametrize(
    ("door_name", "shape_name", "method"), _door_cases(_PANEL_DOORS, _LOG_DOOR_SHAPES, logx=True)
)
def test_every_figure_door_names_a_panel_on_a_log_axis_as_the_axis_and_the_fit_say(
    door_name, shape_name, method
):
    """On a log x axis a panel the axis empties is reported by the log warning alone, a panel with
    one value on the side is named by the fit beside it, and ``p``'s fit is the one line drawn.

    Rejects a door whose route to the log seam still refuses, or names an emptied panel twice.
    """
    door, shape = _PANEL_DOORS[door_name], _SHAPES[shape_name]
    svg, messages = _panel_door_render(door, method, shape, logx=True)
    assert messages == _door_expected(door, method, shape, scope="panel", logx=True)
    assert _polylines(svg) == 1


@pytest.mark.parametrize(
    ("door_name", "shape_name", "method"), _door_cases(_WHOLE_DOORS, _DOOR_SHAPES, logx=False)
)
def test_every_door_names_a_whole_call_the_fit_cannot_use(door_name, shape_name, method):
    """One unfaceted call is named once on each door and draws no fit."""
    door, shape = _WHOLE_DOORS[door_name], _SHAPES[shape_name]
    frame = _q_rows(_panel_frame(shape.x, shape.y))
    svg, messages = _render_capturing(lambda: door.build(frame, method, False).to_svg())
    assert messages == _door_expected(door, method, shape, scope="whole")
    assert _polylines(svg) == 0


@pytest.mark.parametrize(
    ("door_name", "shape_name", "method"),
    _door_cases(_WHOLE_DOORS, ["one_on_side", "null_x", "zero_variance"], logx=True),
)
def test_every_figure_door_names_a_whole_call_on_a_log_axis(door_name, shape_name, method):
    """The same call on a log x axis is named by the fit and the axis, once each."""
    door, shape = _WHOLE_DOORS[door_name], _SHAPES[shape_name]
    frame = _q_rows(_panel_frame(shape.x, shape.y))
    svg, messages = _render_capturing(lambda: door.build(frame, method, True).to_svg())
    assert messages == _door_expected(door, method, shape, scope="whole", logx=True)
    assert _polylines(svg) == 0


def _line_nodes(scene: dict) -> list[int]:
    """Each panel's line-batch node count, in panel order."""
    return [
        sum(len(batch["nodes"]) for batch in panel["marks"] if batch["kind"] == "line")
        for panel in scene["panels"]
    ]


@pytest.mark.parametrize("method", _FITS)
@pytest.mark.parametrize("shape_name", _DOOR_SHAPES)
@pytest.mark.parametrize("scope", ["panel", "whole"])
def test_the_interactive_route_takes_the_static_outcome(scope, shape_name, method):
    """``.interactive()`` names the panel or call once and draws the panels that fit."""
    shape = _SHAPES[shape_name]
    frame = _panel_frame(shape.x, shape.y)
    if scope == "panel":
        chart = _fit(frame, method).facet(col="f")
    else:
        chart = _fit(_q_rows(frame), method)
    scene_json, messages = _render_capturing(lambda: chart.interactive().scene_json)
    assert messages == _door_expected(_PANEL_DOORS["hand-written"], method, shape, scope=scope)
    assert _line_nodes(json.loads(scene_json)) == ([1, 0] if scope == "panel" else [0])


def _interactive_cases() -> list[Any]:
    """Every shape on both axes, as a facet panel and as a whole call; a whole call the log axis
    empties is the axis's refusal and has no scene."""
    cases = []
    for name, shape in _SHAPES.items():
        for logx in (False, True):
            outcome = _outcome(shape, logx=logx)
            if outcome is None:
                continue
            for scope in ("panel", "whole"):
                if scope == "whole" and outcome.axis_emptied:
                    continue
                for method in _FITS:
                    if scope == "whole" and name == "y_is_2" and method == "logistic":
                        continue
                    cases.append(
                        pytest.param(
                            name, scope, logx, method, id=f"{name}-{scope}-{logx}-{method}"
                        )
                    )
    return cases


@pytest.mark.parametrize(("name", "scope", "logx", "method"), _interactive_cases())
def test_the_interactive_figure_route_takes_the_static_outcome_on_both_axes(
    name, scope, logx, method
):
    """``lmplot`` and ``regplot`` ``.interactive()`` give the static messages and one fit line per
    panel that fits, on a linear and on a log axis.

    Rejects the interactive route refusing, or naming a panel twice, where the static one warns.
    """
    shape = _SHAPES[name]
    outcome = _outcome(shape, logx=logx)
    assert outcome is not None
    frame = _panel_frame(shape.x, shape.y)
    if scope == "panel":
        build = lambda: _lmplot(frame, method, logx=logx, col="f")  # noqa: E731
    else:
        build = lambda: _regplot(_q_rows(frame), method, logx=logx)  # noqa: E731
    scene_json, messages = _render_capturing(lambda: build().interactive().scene_json)
    assert messages == _expected(method, shape, outcome, scope=scope)
    fits = int(_fit_drawn(shape, outcome, method))
    assert _line_nodes(json.loads(scene_json)) == ([1, fits] if scope == "panel" else [fits])


@pytest.mark.parametrize("method", _FITS)
def test_an_hconcat_leaf_the_fit_cannot_use_is_named_and_both_leaves_draw(method):
    """A composite leaf is a whole call: its fit is named, and the leaf beside it still draws."""
    frame = _panel_frame([None] * 4, _BINARY)
    good, bad = frame.filter(pl.col("f") == "p"), _q_rows(frame)
    chart = fm.hconcat(_fit(good, method), _fit(bad, method))
    svg, messages = _render_capturing(chart.to_svg)
    assert messages == [_whole_fit_message(_STATISTIC[method], FIT_FEW_POINTS)]
    assert _polylines(svg) == 1


# ---------------------------------------------------------------------------
# A hue group, and a panel whose every hue key is null
# ---------------------------------------------------------------------------

_NULL_KEYS = [None] * 4


@pytest.mark.parametrize("logx", [False, True], ids=["linear", "log"])
@pytest.mark.parametrize("method", [*_FITS, "lm"])
def test_a_panel_with_every_key_and_every_x_null_names_the_panel_the_keys_and_the_scatter(
    method, logx
):
    """The panel's pooled fit has nothing to use: the panel, the keys and the scatter each speak once.

    Rejects the pooled fallback refusing the chart where its siblings degrade.
    """
    statistic = _STATISTIC[method]
    frame = _panel_frame([None] * 4, _BINARY, g=_NULL_KEYS)
    svg, messages = _render_capturing(
        lambda: _lmplot(frame, method, logx=logx, col="f", hue="g").to_svg()
    )
    assert messages == [
        _panel_fit_message(statistic, FIT_FEW_POINTS),
        _panel_key_fallback_message(statistic, "g", "f=q"),
        _slot_message("lmplot", "x", "null", "g", panel="f=q"),
    ]
    assert _polylines(svg) == 1


@pytest.mark.parametrize("logx", [False, True], ids=["linear", "log"])
@pytest.mark.parametrize("method", _FITS)
def test_the_interactive_route_names_a_hue_fallback_panel_as_the_static_one_does(method, logx):
    """``lmplot(col=, hue=).interactive()`` over a panel whose every key and every ``x`` is null
    gives the static messages and draws panel ``p``'s group fit alone."""
    statistic = _STATISTIC[method]
    frame = _panel_frame([None] * 4, _BINARY, g=_NULL_KEYS)
    chart = _lmplot(frame, method, logx=logx, col="f", hue="g")
    scene, messages = _render_capturing(lambda: chart.interactive().scene_json)
    assert messages == [
        _panel_fit_message(statistic, FIT_FEW_POINTS),
        _panel_key_fallback_message(statistic, "g", "f=q"),
        _slot_message("lmplot", "x", "null", "g", panel="f=q"),
    ]
    assert _line_nodes(json.loads(scene)) == [1, 0]


@pytest.mark.parametrize("method", [*_FITS, "lm"])
def test_a_panel_with_every_key_null_and_every_x_off_the_log_side_names_the_keys_and_the_axis(
    method,
):
    """The axis empties the panel: the keys and the axis speak, and the fit does not."""
    statistic = _STATISTIC[method]
    frame = _panel_frame([-1.0, -2.0, -3.0, -4.0], _BINARY, g=_NULL_KEYS)
    svg, messages = _render_capturing(
        lambda: _lmplot(frame, method, logx=True, col="f", hue="g").to_svg()
    )
    assert messages == [
        _panel_key_fallback_message(statistic, "g", "f=q"),
        _null_key_dropped_message("lmplot", 4, "g"),
        _log_message("x", "x", 4),
    ]
    assert _polylines(svg) == 1


@pytest.mark.parametrize("method", [*_FITS, "lm"])
def test_a_panel_with_every_key_null_and_a_usable_x_draws_the_pooled_fit(method):
    """Every key null with a usable ``x``: the pooled fit draws, the panel's keys are named."""
    statistic = _STATISTIC[method]
    frame = _panel_frame([1.0, 2.0, 3.0, 4.0], _BINARY, g=_NULL_KEYS)
    svg, messages = _render_capturing(lambda: _lmplot(frame, method, col="f", hue="g").to_svg())
    assert messages == [
        _panel_key_fallback_message(statistic, "g", "f=q"),
        _null_key_dropped_message("lmplot", 4, "g"),
    ]
    assert _polylines(svg) == 2


@pytest.mark.parametrize("method", [*_FITS, "lm"])
def test_an_unfaceted_call_with_every_key_null_and_every_x_null_names_the_keys_the_fit_and_the_scatter(
    method,
):
    """No panel: the whole batch is one ungrouped fit with nothing to use, and each part speaks once."""
    statistic = _STATISTIC[method]
    frame = _q_rows(_panel_frame([None] * 4, _BINARY, g=_NULL_KEYS))
    svg, messages = _render_capturing(lambda: _regplot(frame, method, hue="g").to_svg())
    assert messages == [
        _all_keys_null_fallback_message(statistic, "g"),
        _whole_fit_message(statistic, FIT_FEW_POINTS),
        _slot_message("regplot", "x", "null", "g"),
    ]
    assert _polylines(svg) == 0


@pytest.mark.parametrize("method", [*_FITS, "lm"])
def test_a_hue_group_the_fit_cannot_use_is_named_by_its_group(method):
    """A hue group of one row is named by its group in a panel and in a whole call."""
    statistic = _STATISTIC[method]
    frame = _panel_frame([1.0, 2.0, 3.0, 4.0], _BINARY, g=["a", "a", "a", "b"])
    svg, messages = _render_capturing(lambda: _lmplot(frame, method, col="f", hue="g").to_svg())
    assert messages == [_fit_message(statistic, "g=b", FIT_FEW_POINTS, panel="f=q")]
    assert _polylines(svg) == 2
    svg, messages = _render_capturing(lambda: _regplot(_q_rows(frame), method, hue="g").to_svg())
    assert messages == [_fit_message(statistic, "g=b", FIT_FEW_POINTS)]
    assert _polylines(svg) == 1


# ---------------------------------------------------------------------------
# A hand-written step: the reasons only it can reach
# ---------------------------------------------------------------------------


def _family_frame(p_y: list[float], q_y: list[float]) -> pl.DataFrame:
    return _panel_frame([1.0, 2.0, 3.0, 4.0][: len(q_y)], q_y, p_y=p_y)


_BINOMIAL_OUTSIDE = "binomial family requires binary y (0 or 1); found 2"
_POISSON_NEGATIVE = "poisson family requires non-negative integer y; found -1"
_POISSON_FRACTION = "poisson family requires non-negative integer y; found 1.5"
_GAMMA_ZERO = "gamma family requires strictly positive y; found 0"
_INVERSE_GAUSSIAN_ZERO = "inverse_gaussian family requires strictly positive y; found 0"
_LOGISTIC_NOT_BINARY = "Logistic: y must be binary (only 0 or 1); found value 2"
_COUNTS = [1.0, 2.0, 2.0, 3.0, 4.0, 5.0, 7.0, 9.0]
_FAMILIES = [
    pytest.param(
        "binomial",
        _P_Y,
        [0.0, 2.0, 1.0, 0.0],
        "binomial family requires binary y (0 or 1); found 2",
        id="binomial",
    ),
    pytest.param(
        "poisson",
        _COUNTS,
        [1.0, -1.0, 2.0, 3.0],
        _POISSON_NEGATIVE,
        id="poisson",
    ),
    pytest.param(
        "gamma",
        _COUNTS,
        [1.0, 0.0, 2.0, 3.0],
        _GAMMA_ZERO,
        id="gamma",
    ),
    pytest.param(
        "inverse_gaussian",
        _COUNTS,
        [1.0, 0.0, 2.0, 3.0],
        _INVERSE_GAUSSIAN_ZERO,
        id="inverse_gaussian",
    ),
    pytest.param(
        "poisson",
        _COUNTS,
        [1.0, 1.5, 2.0, 3.0],
        _POISSON_FRACTION,
        id="poisson-fraction",
    ),
]


@pytest.mark.parametrize(("family", "p_y", "q_y", "reason"), _FAMILIES)
def test_a_glm_panel_with_a_response_outside_its_family_is_named_and_the_others_fit(
    family, p_y, q_y, reason
):
    """A response outside the family degrades the panel that holds it; the other panel fits."""
    frame = _family_frame(p_y, q_y)
    chart = (
        fm.Chart(frame)
        .transform(fm.Glm("x", "y", family=family, n_grid=20))
        .mark_line()
        .encode(x="x", y="y")
        .facet(col="f")
    )
    svg, messages = _render_capturing(chart.to_svg)
    assert messages == [_panel_fit_message("Glm", reason)]
    assert _polylines(svg) == 1


@pytest.mark.parametrize(("family", "p_y", "q_y", "reason"), _FAMILIES)
def test_a_glm_whole_frame_with_a_response_outside_its_family_still_refuses(
    family, p_y, q_y, reason
):
    """The same rows as one unfaceted call are a caller's mistake: refused with the family text."""
    frame = _q_rows(_family_frame(p_y, q_y))
    chart = (
        fm.Chart(frame)
        .transform(fm.Glm("x", "y", family=family, n_grid=20))
        .mark_line()
        .encode(x="x", y="y")
    )
    with pytest.raises(ValueError, match=re.escape(f"Glm: {reason}")):
        chart.to_svg()


_DIVERGING = ([0.0, 1.0, 2.0], [1e308, -1e308, 1e308])


def test_a_diverging_glm_is_named_on_a_panel_and_on_a_whole_call():
    """An iteration that diverges is data the fit cannot use: it degrades on every call."""
    qx, qy = _DIVERGING
    frame = pl.DataFrame(
        {
            "f": ["p"] * _P_ROWS + ["q"] * 3,
            "x": _P_X + qx,
            "y": [1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0] + qy,
        }
    )
    chart = _fit(frame, "glm").facet(col="f")
    svg, messages = _render_capturing(chart.to_svg)
    assert messages == [_panel_fit_message("Glm", FIT_DIVERGED)]
    assert _polylines(svg) == 1
    svg, messages = _render_capturing(_fit(_q_rows(frame), "glm").to_svg)
    assert messages == [_whole_fit_message("Glm", FIT_DIVERGED)]
    assert _polylines(svg) == 0


_SINGULAR_ROWS = ([0.0, 1e-9, 2e-9], [0.0, 1.0, 0.0])


@pytest.mark.parametrize("method", _REPAIRED)
def test_a_singular_weighted_design_is_named_on_a_panel_and_on_a_whole_call(method):
    """A design the weights make singular degrades the panel and the whole call alike."""
    qx, qy = _SINGULAR_ROWS
    frame = pl.DataFrame({"f": ["p"] * _P_ROWS + ["q"] * 3, "x": _P_X + qx, "y": _P_Y + qy})
    statistic = _STATISTIC[method]
    svg, messages = _render_capturing(_fit(frame, method).facet(col="f").to_svg)
    assert messages == [_panel_fit_message(statistic, FIT_SINGULAR)]
    assert _polylines(svg) == 1
    svg, messages = _render_capturing(_fit(_q_rows(frame), method).to_svg)
    assert messages == [_whole_fit_message(statistic, FIT_SINGULAR)]
    assert _polylines(svg) == 0


@pytest.mark.parametrize("name", ["one_class", "separated"])
def test_a_logistic_whole_call_with_one_class_or_separated_classes_warns_and_draws(name):
    """The data a logistic fit cannot use degrades an unfaceted call, as ``Robust`` is never refused."""
    shape = _SHAPES[name]
    frame = _q_rows(_panel_frame(shape.x, shape.y))
    svg, messages = _render_capturing(_fit(frame, "logistic").to_svg)
    assert shape.lin.reason is not None
    assert messages == [_whole_fit_message("Logistic", shape.lin.reason)]
    assert _polylines(svg) == 0


@pytest.mark.parametrize("door", ["hand-written", "regplot", "lmplot"])
def test_a_logistic_whole_call_with_a_y_of_two_still_refuses(door):
    """A response ``Logistic`` cannot read, as one unfaceted call, is refused with its typed error."""
    shape = _SHAPES["y_is_2"]
    frame = _q_rows(_panel_frame(shape.x, shape.y))
    build = {
        "hand-written": lambda: _fit(frame, "logistic"),
        "regplot": lambda: _regplot(frame, "logistic"),
        "lmplot": lambda: _lmplot(frame, "logistic"),
    }[door]
    with pytest.raises(ValueError, match=re.escape(_LOGISTIC_NOT_BINARY)):
        build().to_svg()


@pytest.mark.parametrize("method", _REPAIRED)
@pytest.mark.parametrize(
    ("knobs", "text"),
    [
        pytest.param({"n_grid": 0}, "n_grid must be > 0", id="n_grid"),
        pytest.param({"ci": 2.0}, "ci must be in (0, 1)", id="ci"),
        pytest.param({"max_iter": 0}, "max_iter must be > 0", id="max_iter"),
        pytest.param({"tol": 0.0}, "tol must be finite and > 0", id="tol"),
    ],
)
def test_a_refused_parameter_is_refused_when_the_step_is_built(method, knobs, text):
    """A knob the fit cannot take is refused by the step's constructor, before any row is read."""
    with pytest.raises(ValueError, match=re.escape(f"{_STATISTIC[method]}: {text}")):
        {"logistic": fm.Logistic, "glm": fm.Glm}[method]("x", "y", **knobs)


def test_a_glm_family_and_link_that_do_not_match_are_refused_when_the_step_is_built():
    """A link the family cannot take is refused by the constructor."""
    with pytest.raises(
        ValueError, match=re.escape("Glm: family 'poisson' incompatible with link 'logit'")
    ):
        fm.Glm("x", "y", family="poisson", link="logit")


# ---------------------------------------------------------------------------
# A response outside the method: every storage type, and a frame of one usable row
# ---------------------------------------------------------------------------

_Y_DTYPES = [
    pytest.param(pl.Int64, id="int64"),
    pytest.param(pl.Int32, id="int32"),
    pytest.param(pl.UInt8, id="uint8"),
    pytest.param(pl.Float32, id="float32"),
]
#: Per fit: the step over ``x`` and ``y`` that reads a 0/1 response, the reason it gives a panel
#: whose response holds a 2, and the text it refuses a whole call with.
_BINARY_FITS = {
    "logistic": (
        lambda: fm.Logistic("x", "y", n_grid=20),
        FIT_NOT_BINARY,
        _LOGISTIC_NOT_BINARY,
    ),
    "glm": (
        lambda: fm.Glm("x", "y", family="binomial", n_grid=20),
        _BINOMIAL_OUTSIDE,
        f"Glm: {_BINOMIAL_OUTSIDE}",
    ),
}


@pytest.mark.parametrize("scope", ["panel", "whole"])
@pytest.mark.parametrize("method", _REPAIRED)
@pytest.mark.parametrize("dtype", _Y_DTYPES)
def test_a_response_of_any_numeric_type_with_no_usable_value_is_named(dtype, method, scope):
    """A 0/1 response stored as an integer or a ``Float32``, null on every row of a panel or of a
    whole call, is named by the fit and by the scatter.

    Rejects a degrade that only the ``Float64`` response path was repaired to reach.
    """
    frame = _panel_frame([1.0, 2.0, 3.0, 4.0], [None] * 4, y_dtype=dtype)
    statistic = _STATISTIC[method]
    if scope == "panel":
        render = lambda: _lmplot(frame, method, col="f").to_svg()  # noqa: E731
        expected = [_panel_fit_message(statistic, FIT_FEW_POINTS), _panel_message("lmplot", "y")]
    else:
        render = lambda: _regplot(_q_rows(frame), method).to_svg()  # noqa: E731
        expected = [_whole_fit_message(statistic, FIT_FEW_POINTS), _layer_message("regplot", "y")]
    svg, messages = _render_capturing(render)
    assert messages == expected
    assert _polylines(svg) == (1 if scope == "panel" else 0)


@pytest.mark.parametrize("method", _REPAIRED)
@pytest.mark.parametrize("dtype", _Y_DTYPES)
def test_a_response_of_any_numeric_type_holding_a_two_degrades_a_panel_and_refuses_a_call(
    dtype, method
):
    """A 0/1 response with a 2 in it, of any storage type: the panel that holds it is named, the
    whole call is refused with the fit's own text."""
    step, reason, refusal = _BINARY_FITS[method]
    frame = _panel_frame([1.0, 2.0, 3.0, 4.0], [0.0, 1.0, 2.0, 0.0], y_dtype=dtype)

    def chart(rows: pl.DataFrame) -> fm.Chart:
        return fm.Chart(rows).transform(step()).mark_line().encode(x="x", y="y")

    svg, messages = _render_capturing(chart(frame).facet(col="f").to_svg)
    assert messages == [_panel_fit_message(_STATISTIC[method], reason)]
    assert _polylines(svg) == 1
    with pytest.raises(ValueError, match=re.escape(refusal)):
        chart(_q_rows(frame)).to_svg()


#: A response outside the family on a single usable row: (family, panel p's response, the row's y,
#: the family's reason).
_ONE_ROW_OUTSIDE = [
    pytest.param(
        "poisson",
        _COUNTS,
        -1.0,
        _POISSON_NEGATIVE,
        id="poisson",
    ),
    pytest.param(
        "poisson",
        _COUNTS,
        1.5,
        _POISSON_FRACTION,
        id="poisson-fraction",
    ),
    pytest.param("binomial", _P_Y, 2.0, _BINOMIAL_OUTSIDE, id="binomial"),
    pytest.param("gamma", _COUNTS, 0.0, _GAMMA_ZERO, id="gamma"),
]


@pytest.mark.parametrize("beside_null_x", [False, True], ids=["alone", "beside-a-null-x-row"])
@pytest.mark.parametrize(("family", "p_y", "value", "reason"), _ONE_ROW_OUTSIDE)
def test_a_glm_whole_frame_whose_one_usable_row_is_outside_its_family_refuses(
    family, p_y, value, reason, beside_null_x
):
    """A response the family cannot read is the caller's mistake whatever the row count, as it is
    for ``Logistic``: the genuine whole frame refuses it, where the count of usable rows is not the
    first thing read.

    Rejects the fewer-than-two warning masking a response outside the family.
    """
    qx, qy = ([None, 1.0], [0.0, value]) if beside_null_x else ([1.0], [value])
    frame = _q_rows(_panel_frame(qx, qy, p_y=p_y))
    chart = (
        fm.Chart(frame)
        .transform(fm.Glm("x", "y", family=family, n_grid=20))
        .mark_line()
        .encode(x="x", y="y")
    )
    with pytest.raises(ValueError, match=re.escape(f"Glm: {reason}")):
        chart.to_svg()


@pytest.mark.parametrize(("family", "p_y", "value", "reason"), _ONE_ROW_OUTSIDE)
def test_a_glm_panel_of_one_usable_row_outside_its_family_is_named_not_refused(
    family, p_y, value, reason
):
    """The same one row inside a facet panel degrades that panel: only the genuine whole frame
    refuses a response the family cannot read."""
    frame = _panel_frame([1.0], [value], p_y=p_y)
    chart = (
        fm.Chart(frame)
        .transform(fm.Glm("x", "y", family=family, n_grid=20))
        .mark_line()
        .encode(x="x", y="y")
        .facet(col="f")
    )
    svg, messages = _render_capturing(chart.to_svg)
    assert messages == [_panel_fit_message("Glm", FIT_FEW_POINTS)]
    assert _polylines(svg) == 1


def test_a_logistic_whole_frame_whose_one_usable_row_is_a_two_refuses_and_a_panel_is_named():
    """``Logistic`` reads the response before it counts rows: one row of ``y=2`` refuses a whole
    call and degrades a panel."""
    frame = _panel_frame([1.0], [2.0])
    chart = lambda rows: (
        fm.Chart(rows).transform(fm.Logistic("x", "y", n_grid=20)).mark_line().encode(x="x", y="y")
    )  # noqa: E731
    with pytest.raises(ValueError, match=re.escape(_LOGISTIC_NOT_BINARY)):
        chart(_q_rows(frame)).to_svg()
    svg, messages = _render_capturing(chart(frame).facet(col="f").to_svg)
    assert messages == [_panel_fit_message("Logistic", FIT_NOT_BINARY)]
    assert _polylines(svg) == 1


@pytest.mark.parametrize("output", ["static", "interactive"])
@pytest.mark.parametrize("name", ["diverged_small_scale", "diverged_large_offset"])
def test_a_logistic_whose_coefficients_diverge_is_named_on_a_panel_and_a_whole_call(name, output):
    """Two separated shapes reach the divergence guard, not the convergence check: the fit is
    named on a panel and on a whole call, static and interactive.

    Rejects the divergence guard refusing a whole frame.
    """
    shape = _SHAPES[name]
    frame = _panel_frame(shape.x, shape.y)

    def render(rows: pl.DataFrame, *, facet: bool) -> tuple[Any, list[str]]:
        chart = _fit(rows, "logistic")
        chart = chart.facet(col="f") if facet else chart
        if output == "static":
            return _render_capturing(chart.to_svg)
        scene, said = _render_capturing(lambda: chart.interactive().scene_json)
        return json.loads(scene), said

    drawn, said = render(frame, facet=True)
    assert said == [_panel_fit_message("Logistic", FIT_SEPARATED_DIVERGED)]
    assert (_polylines(drawn) if output == "static" else _line_nodes(drawn)) == (
        1 if output == "static" else [1, 0]
    )
    drawn, said = render(_q_rows(frame), facet=False)
    assert said == [_whole_fit_message("Logistic", FIT_SEPARATED_DIVERGED)]
    assert (_polylines(drawn) if output == "static" else _line_nodes(drawn)) == (
        0 if output == "static" else [0]
    )


# ---------------------------------------------------------------------------
# A fit that computes is unchanged
# ---------------------------------------------------------------------------

# sha256 of the output of every door a computing ``Logistic`` or ``Glm`` fit draws through,
# static and interactive. The five digests in ``_RECORDED_BEFORE`` were recorded from the build
# that computed these fits before the degrade rule changed; the others were recorded after it from
# doors that read the same rows, and guard later drift. After an intended render change that moves
# them, run ``uv run python -m tests.test_regression_fit_degenerate_panels`` to print the new
# digests, inspect the renders it names (rasterize with ``scripts/snapshot-goldens.py`` style
# tooling) and paste them in.
_COMPUTING = _panel_frame([1.0, 2.0, 3.0, 4.0], _BINARY)


def _computed_renders() -> dict[str, Callable[[], str]]:
    """Every door, both output kinds, for each repaired fit over rows both panels can fit."""
    whole = _q_rows(_COMPUTING)
    renders: dict[str, Callable[[], str]] = {}
    for method in _REPAIRED:
        for name, door in _PANEL_DOORS.items():
            renders[f"{method}-panel-{name}"] = lambda d=door, m=method: str(
                d.build(_COMPUTING, m, False).to_svg()
            )
        for name, door in _WHOLE_DOORS.items():
            renders[f"{method}-whole-{name}"] = lambda d=door, m=method: str(
                d.build(whole, m, False).to_svg()
            )
        renders[f"{method}-interactive-panel"] = lambda m=method: str(
            _fit(_COMPUTING, m).facet(col="f").interactive().scene_json
        )
        renders[f"{method}-interactive-whole"] = lambda m=method: str(
            _fit(whole, m).interactive().scene_json
        )
        renders[f"{method}-interactive-lmplot"] = lambda m=method: str(
            _lmplot(_COMPUTING, m, col="f").interactive().scene_json
        )
        renders[f"{method}-interactive-regplot"] = lambda m=method: str(
            _regplot(whole, m).interactive().scene_json
        )
    return renders


_COMPUTED = _computed_renders()
_RECORDED_BEFORE = {
    "logistic-panel-lmplot": "048595cfa6b01f62f2e75942b81e7cb089d339cc9e624f9f09778fb271f39c2a",
    "glm-panel-lmplot": "f63b0309d45f02972944ca65adadf6d0d5a503b5770dbe8393ecae367ba4acfe",
    "logistic-whole-regplot": "0170091ebbb8e213e44cc4af28692b9966b4c383a68f082690384befee5331b1",
    "glm-panel-hand-written": "8eb7ac7efa881f133396446eb46d6e246bbd595ec9bb61f9d6044c435b8f6283",
    "logistic-interactive-panel": "e02d7fdb8232336075010f1d69185708cbabf4c3b681360d7d83c1c082f296a6",
}
_COMPUTED_DIGESTS = {
    **_RECORDED_BEFORE,
    "logistic-panel-lmplot-ci": "25be1c74db079ee96fc34f55ea68e889f9680981220b1d680696b7d133532f10",
    "logistic-panel-lmplot-no-scatter": "c6738ed64922a7f1a76cb1ac7a5e1b27853040db89cf1b25cb45777bffb4eab1",
    "logistic-panel-lmplot-grid": "d0217b3ce2190fef0840dafb2bd900cf06cc78fa4172b2c8979780ccca90f6d1",
    "logistic-panel-hand-written": "d8bacde9af415aece25ec7cefad51289393453e75d61e7a13d6bef3752399ce8",
    "logistic-panel-plus": "512651717c6da84a6ed63ad296cf07cf6b35a3f3f2d6569a949437d8341c88cc",
    "logistic-panel-plus-faceted-members": "512651717c6da84a6ed63ad296cf07cf6b35a3f3f2d6569a949437d8341c88cc",
    "logistic-panel-chart-layer": "8add15bc06c67c87a64cce7946a9b2a7118a81fb0df5d00ddf8ed5d1824f1409",
    "logistic-panel-layerchart": "512651717c6da84a6ed63ad296cf07cf6b35a3f3f2d6569a949437d8341c88cc",
    "logistic-whole-lmplot": "0170091ebbb8e213e44cc4af28692b9966b4c383a68f082690384befee5331b1",
    "logistic-whole-regplot-ci": "8ccdff3d16c0d65a4238d079eeff953c3661840af1b72093349d4db1408b0e17",
    "logistic-whole-hand-written": "1a7d4557da1d22f743ea13b36a13c31a67006bf231a07dedf369f19ee5fe48b9",
    "logistic-whole-plus": "808ef9c6c3fb02e6e146821f41ba6214ba1b65a44daa3d8c36b9e16c8c697a31",
    "logistic-whole-chart-layer": "5658e659f7e572cb2aa93f735bf8b95242711eeee8bdce991f8f2644c1a6ec05",
    "logistic-whole-layerchart": "4f2a497ec19fc0eb691f6d52d8094ab84328d22d1933464ccafcd5081df60b8c",
    "logistic-interactive-whole": "6a0c5822fa81fdc32e8e4776dd0ae441fdb038eecdc59a89277f7f0ae1999f89",
    "logistic-interactive-lmplot": "2660930dee9d07542e7c79054067a962e601c0648fc0f65f36a62ce407e80b1a",
    "logistic-interactive-regplot": "e6aafdc4ab6554d1c2b9201cd80c689ff652f9b21b48ac9afc5c528eb20f1678",
    "glm-panel-lmplot-ci": "17b0b45fb99f69bb279280ec3bb63a8029b1842156505626484e11f00b696f32",
    "glm-panel-lmplot-no-scatter": "3cf1336d18a47cda43c3dbda9d20e0bae356e8bdbcd209fd14a0fccdc3f75884",
    "glm-panel-lmplot-grid": "f6cc1b5ac0225eb847cb39d853e5e6baf949442cc36641c84d290f326320f0b2",
    "glm-panel-plus": "a60683ad7367f8d1a1e0710811f6ae4b31a9aab9789694a1470bf3333f25a722",
    "glm-panel-plus-faceted-members": "a60683ad7367f8d1a1e0710811f6ae4b31a9aab9789694a1470bf3333f25a722",
    "glm-panel-chart-layer": "1200c55580ef50eec9c3cb2fb0e09d145f03ddf13131824294ff5ba721d28090",
    "glm-panel-layerchart": "a60683ad7367f8d1a1e0710811f6ae4b31a9aab9789694a1470bf3333f25a722",
    "glm-whole-lmplot": "8cab52c4406e562c246be335d79664e04ad5842095b8dee17b52c7c36562f49b",
    "glm-whole-regplot": "8cab52c4406e562c246be335d79664e04ad5842095b8dee17b52c7c36562f49b",
    "glm-whole-regplot-ci": "ba8c2c0c92a4d3a92d8a6192ab260049667ea16138cbfee3d251c4dbf64d786a",
    "glm-whole-hand-written": "ebee01cc974941a042ca48b5385d6bb49c22d7bd7755e2d27ffdaeafdd8a8447",
    "glm-whole-plus": "bac1aeb708449a52b848f633eae4ae9f5c35a5b3e5f633c9730555d4fac3cc94",
    "glm-whole-chart-layer": "af36401b1f90c4c4c75f4ee7456032c514f97d1b6fdd586bc528a88aa3ffd53f",
    "glm-whole-layerchart": "b31051d1a2cb041966e92ea00dc2512f70b960bb2a7d08c65fb44d1c37a5397e",
    "glm-interactive-panel": "aa5d353d4b05709a44976f41b37f39febcbed38a83a34f7324cba7c9c92e17ef",
    "glm-interactive-whole": "7a28ff1f79330fbd4d00386a461e699d88b875de263a6d641af9a5582140f96f",
    "glm-interactive-lmplot": "fe7d672f303979337476740e8234086c0b176afe8d1757e40528067e2e8ba738",
    "glm-interactive-regplot": "120065c75c14d2e76863bf32d38c68c2941eeb6074f547c0b42be3723cc80e2b",
}


def _digest(name: str) -> str:
    return hashlib.sha256(_COMPUTED[name]().encode()).hexdigest()


@pytest.mark.parametrize("name", list(_COMPUTED))
def test_a_fit_that_computes_is_byte_identical_and_warns_nothing(name):
    """A fit that computes renders the same bytes on every door and output kind, and warns nothing.

    Only the five digests in ``_RECORDED_BEFORE`` were recorded from the build that preceded the
    degrade rule, so only those five prove the bytes did not move across it. The other 35 were
    recorded from the changed build over the same rows: they guard later drift and are no evidence
    of identity across the change. After an intended render change, regenerate with
    ``uv run python -m tests.test_regression_fit_degenerate_panels`` and inspect the renders first.
    """
    out, messages = _render_capturing(_COMPUTED[name])
    assert messages == []
    assert hashlib.sha256(out.encode()).hexdigest() == _COMPUTED_DIGESTS[name]


if __name__ == "__main__":
    for _name in _COMPUTED:
        print(f'    "{_name}": "{_digest(_name)}",')
