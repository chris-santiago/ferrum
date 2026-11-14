"""Every remedy a log-axis message names is executed, and the rows it promises are drawn.

A message that tells the caller to pass ``fm.LogScale(domain=[a, b])``, ``clamp=True`` or
``fm.SymlogScale()`` is followed here as written: the suggested call is parsed out of the
message text, evaluated, and handed to the door that refused or warned. A remedy the library
itself refuses (a domain whose endpoints are equal, a non-finite or zero endpoint), or one
that leaves the rows outside the domain it names, fails here: each test counts the marks
drawn after following the remedy and compares the full ordered warning list (under
``action="always"``) with ``==``. The constant-column remedies are pinned for every base a
``LogScale`` accepts (above and below one), for integer and float columns, at the float
extremes, and as printed text for exact powers of the base.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any, Callable

import polars as pl
import pytest

import ferrum as fm
from tests.test_log_axis_statistics import _no_value_on_side_refusal
from tests.test_null_group_warning import (
    FIT_FEW_POINTS,
    FIT_LM_UNCOMPUTED,
    FIT_SINGULAR,
    FIT_ZERO_VARIANCE,
    _infinite_warning,
    _log_message,
    _log_positive_side_message,
    _log_zero_or_negative_refusal,
    _render_capturing,
    _whole_fit_message,
)

_INF = float("inf")
_DOMAIN_CALL = re.compile(r"fm\.LogScale\(domain=\[[^\]]+\]\)")
_STATISTIC = {"lm": "Smooth", "robust": "Robust", "logistic": "Logistic", "glm": "Glm"}
#: What the canvas says of the ``-5e-324`` remedy's tick labels: the decade the suggestion names
#: is twenty-three orders of magnitude of printed zeros wide.
_TICK_OVERFLOW = (
    "x tick labels overflow the canvas on the left by 28595.8px: padding.auto declined the "
    "margin, since containing them would take more than half the plot or drop x tick labels; "
    "widen the chart, set the x axis label_angle or label_overlap, or set "
    "configure_padding(left=...)."
)


def _binary(n: int) -> list[float]:
    return [float(i % 2) for i in range(n)]


def _static(chart: fm.Chart) -> Any:
    return chart.to_svg()


def _interactive(chart: fm.Chart) -> Any:
    return chart.interactive().scene_json


_KINDS: dict[str, Callable[[fm.Chart], Any]] = {"static": _static, "interactive": _interactive}


def _drawn(output: str, kind: str) -> int:
    """The marks a render of ``kind`` drew: circles of the svg, or nodes of the scene."""
    if kind == "static":
        return output.count("<circle")
    scene = json.loads(output)
    return sum(len(batch["nodes"]) for panel in scene["panels"] for batch in panel["marks"])


def _render(chart: fm.Chart, kind: str = "static") -> tuple[int, list[str]]:
    """The marks ``chart`` draws on ``kind`` and every warning, under ``action="always"``."""
    output, shown = _render_capturing(lambda: _KINDS[kind](chart))
    return _drawn(output, kind), shown


def _refusal(build: Callable[[], fm.Chart], kind: str = "static") -> str:
    with pytest.raises(ValueError) as refused:
        _KINDS[kind](build())
    return str(refused.value)


def _suggested_domain(message: str) -> tuple[str, fm.LogScale]:
    """The ``fm.LogScale(domain=[a, b])`` call *message* names, as its text and its value."""
    found = _DOMAIN_CALL.findall(message)
    assert len(found) == 1, message
    return found[0], eval(found[0], {"fm": fm})  # noqa: S307 -- the message's own example call


def _frame(xs: list[float], dtype: Any = pl.Float64) -> pl.DataFrame:
    return pl.DataFrame(
        {"x": pl.Series(xs, dtype=dtype, strict=False), "y": [float(i + 1) for i in range(len(xs))]}
    )


def _point(xs: list[float], scale: Any, dtype: Any = pl.Float64) -> fm.Chart:
    return fm.Chart(_frame(xs, dtype)).mark_point().encode(x=fm.X("x", scale=scale), y="y")


def _lmplot_refused(xs: list[float], method: str) -> fm.Chart:
    frame = pl.DataFrame({"x": xs, "y": _binary(len(xs))})
    return fm.lmplot(frame, x="x", y="y", method=method, ci=None, show_metrics=False, logx=True)


def _lmplot_followed(xs: list[float], method: str, scale: Any) -> fm.Chart:
    frame = pl.DataFrame({"x": xs, "y": _binary(len(xs))})
    x = fm.X("x", scale=scale)
    return fm.lmplot(frame, x=x, y="y", method=method, ci=None, show_metrics=False)


def _endpoints(scale: fm.LogScale) -> tuple[float, float]:
    lo, hi = scale.domain
    return float(lo), float(hi)


# ---------------------------------------------------------------------------
# A constant off-side column: the suggested domain renders, and draws every row
# ---------------------------------------------------------------------------

#: (rows, base, the endpoints the refusal suggests): every ``-3`` and every ``-10`` on base 10,
#: a lone row, base 2, and the bases between 0 and 1 (a decade in base ``b`` is a decade in base
#: ``1 / b``).
_CONSTANT_COLUMNS = {
    "every -3": ([-3.0] * 4, 10.0, (-10.0, -1.0)),
    "one row -3": ([-3.0], 10.0, (-10.0, -1.0)),
    "every -10": ([-10.0] * 3, 10.0, (-100.0, -1.0)),
    "base 2 every -3": ([-3.0] * 3, 2.0, (-4.0, -2.0)),
    "base 0.5 every -3": ([-3.0] * 3, 0.5, (-4.0, -2.0)),
    "base 0.1 every -3": ([-3.0] * 3, 0.1, (-10.0, -1.0)),
}


@pytest.mark.parametrize("kind", sorted(_KINDS))
@pytest.mark.parametrize("name", list(_CONSTANT_COLUMNS))
def test_the_domain_suggested_for_a_constant_negative_column_renders_when_followed(name, kind):
    xs, base, (lo, hi) = _CONSTANT_COLUMNS[name]
    message = _refusal(lambda: _point(xs, fm.LogScale(base=base)), kind)
    assert message == _no_value_on_side_refusal("x", "x", len(xs), lo, hi)
    _, suggested = _suggested_domain(message)
    assert _endpoints(suggested) == (lo, hi)
    followed = fm.LogScale(domain=list(_endpoints(suggested)), base=base)
    assert _render(_point(xs, followed), kind) == (len(xs), [])


@pytest.mark.parametrize("dtype", [pl.Int64, pl.Int32, pl.Float32], ids=lambda d: str(d))
def test_the_domain_suggested_for_a_constant_negative_column_of_any_numeric_dtype_renders(dtype):
    xs = [-3.0] * 3
    message = _refusal(lambda: _point(xs, fm.LogScale(), dtype))
    assert message == _no_value_on_side_refusal("x", "x", 3, -10.0, -1.0)
    _, suggested = _suggested_domain(message)
    assert _render(_point(xs, suggested, dtype)) == (3, [])


#: The call a refusal prints for a value that is an exact power of the base: the endpoints are
#: the neighbouring powers, printed exactly, not as a quotient's representation noise.
_EXACT_POWERS = {
    -1e-5: "fm.LogScale(domain=[-0.0001, -0.000001])",
    -1e-7: "fm.LogScale(domain=[-0.000001, -0.00000001])",
    -1e3: "fm.LogScale(domain=[-10000, -100])",
}


@pytest.mark.parametrize("value", list(_EXACT_POWERS))
def test_the_domain_suggested_around_an_exact_power_of_the_base_prints_exact_endpoints(value):
    xs = [value] * 3
    call, suggested = _suggested_domain(_refusal(lambda: _point(xs, fm.LogScale())))
    assert call == _EXACT_POWERS[value]
    assert _render(_point(xs, suggested)) == (3, [])


@pytest.mark.parametrize("method", list(_STATISTIC))
@pytest.mark.parametrize("name", ["every -3", "one row -3"])
def test_the_domain_suggested_to_a_regression_door_renders_when_followed(name, method):
    xs = _CONSTANT_COLUMNS[name][0]
    message = _refusal(lambda: _lmplot_refused(xs, method))
    assert message == _no_value_on_side_refusal("x", "x", len(xs), -10.0, -1.0)
    _, suggested = _suggested_domain(message)
    drawn, shown = _render(_lmplot_followed(xs, method, suggested))
    reason = FIT_FEW_POINTS if len(xs) == 1 else _constant_column_reason(method)
    assert drawn == len(xs)
    assert shown == [_whole_fit_message(_STATISTIC[method], reason)]


def _constant_column_reason(method: str) -> str:
    """Why the fit cannot be computed over a constant ``x``."""
    return FIT_LM_UNCOMPUTED if method == "lm" else FIT_ZERO_VARIANCE


def test_the_domain_suggested_beside_a_zero_renders_and_warns_the_zero_row_off_side():
    xs = [0.0, -3.0]
    message = _refusal(lambda: _point(xs, fm.LogScale()))
    assert message == _log_zero_or_negative_refusal("x", "x", 2, "fm.LogScale(domain=[-10, -1])")
    _, suggested = _suggested_domain(message)
    assert _render(_point(xs, suggested)) == (1, [_log_positive_side_message("x", "x", 1)])


@pytest.mark.parametrize("kind", sorted(_KINDS))
def test_the_domain_suggested_beside_a_zero_renders_on_both_output_kinds(kind):
    xs = [0.0, -3.0, -3.0]
    message = _refusal(lambda: _point(xs, fm.LogScale()), kind)
    assert message == _log_zero_or_negative_refusal("x", "x", 3, "fm.LogScale(domain=[-10, -1])")
    _, suggested = _suggested_domain(message)
    assert _render(_point(xs, suggested), kind) == (2, [_log_positive_side_message("x", "x", 1)])


@pytest.mark.parametrize("method", list(_STATISTIC))
def test_the_domain_suggested_beside_a_zero_to_a_regression_door_renders_when_followed(method):
    xs = [0.0, -3.0, 0.0, -3.0]
    message = _refusal(lambda: _lmplot_refused(xs, method))
    assert message == _log_zero_or_negative_refusal("x", "x", 4, "fm.LogScale(domain=[-10, -1])")
    _, suggested = _suggested_domain(message)
    drawn, shown = _render(_lmplot_followed(xs, method, suggested))
    assert drawn == 2
    assert shown == [
        _whole_fit_message(_STATISTIC[method], _constant_column_reason(method)),
        _log_positive_side_message("x", "x", 2),
    ]


#: Per float extreme: the warnings following its remedy raises. The decade around the largest
#: magnitude has no representable upper power, so the remedy names the value itself as one
#: bound; the smallest magnitude's remedy spans twenty-three printed zeros of tick labels.
_EXTREMES = {-1.7e308: [], -5e-324: [_TICK_OVERFLOW]}


@pytest.mark.parametrize("value", list(_EXTREMES), ids=["largest-magnitude", "smallest-magnitude"])
def test_the_domain_suggested_at_the_float_extremes_holds_the_value_and_draws_every_row(value):
    xs = [value] * 3
    message = _refusal(lambda: _point(xs, fm.LogScale()))
    _, suggested = _suggested_domain(message)
    lo, hi = _endpoints(suggested)
    assert math.isfinite(lo) and math.isfinite(hi), (lo, hi)
    assert lo != 0.0 and hi != 0.0 and lo != hi, (lo, hi)
    assert lo <= value <= hi, "the suggested domain holds the value"
    assert _render(_point(xs, suggested)) == (3, _EXTREMES[value])


def test_a_negative_span_keeps_its_own_extent_in_the_suggestion_and_renders():
    xs = [-30.0, -10.0, -3.0]
    message = _refusal(lambda: _point(xs, fm.LogScale()))
    assert message == _no_value_on_side_refusal("x", "x", 3, -30.0, -3.0)
    _, suggested = _suggested_domain(message)
    assert _render(_point(xs, suggested)) == (3, [])


@pytest.mark.parametrize("kind", sorted(_KINDS))
def test_a_column_of_zeros_suggests_a_symlog_scale_and_it_renders(kind):
    xs = [0.0, 0.0]
    message = _refusal(lambda: _point(xs, fm.LogScale()), kind)
    assert message == _log_zero_or_negative_refusal("x", "x", 2, None)
    assert _render(_point(xs, fm.SymlogScale()), kind) == (2, [])


# ---------------------------------------------------------------------------
# An infinity: clamp=True draws it, and filtering it leaves a chart with nothing to report
# ---------------------------------------------------------------------------

_INFINITE_X = [1.0, 2.0, _INF, _INF]


@pytest.mark.parametrize("kind", sorted(_KINDS))
@pytest.mark.parametrize(
    "scale",
    [fm.LogScale(), fm.LogScale(domain=[1, 1000])],
    ids=["no-domain", "written-domain"],
)
def test_the_remedies_the_infinite_sentence_names_render(scale, kind):
    assert _render(_point(_INFINITE_X, scale), kind) == (2, [_infinite_warning("x", "x", 2)])
    clamped = fm.LogScale(domain=scale.domain, clamp=True)
    assert _render(_point(_INFINITE_X, clamped), kind) == (4, [])
    finite = [x for x in _INFINITE_X if math.isfinite(x)]
    assert _render(_point(finite, scale), kind) == (2, [])


@pytest.mark.parametrize("method", list(_STATISTIC))
def test_the_clamp_the_infinite_sentence_names_renders_on_a_regression_door(method):
    """The infinite rows draw at the axis edge and nothing says they are left out; ``Logistic``
    alone still says its own design is singular over them."""
    clamped = _lmplot_followed(_INFINITE_X, method, fm.LogScale(clamp=True))
    drawn, shown = _render(clamped)
    assert drawn == 4
    assert shown == ([_whole_fit_message("Logistic", FIT_SINGULAR)] if method == "logistic" else [])


@pytest.mark.parametrize("kind", sorted(_KINDS))
def test_the_off_side_sentence_still_names_a_remedy_that_renders(kind):
    """``-inf`` is off the positive side, and its sentence names ``fm.SymlogScale()``."""
    xs = [1.0, 2.0, -_INF]
    assert _render(_point(xs, fm.LogScale()), kind) == (2, [_log_message("x", "x", 1)])
    followed_drawn, shown = _render(_point(xs, fm.SymlogScale()), kind)
    assert followed_drawn == 2
    assert shown == []
