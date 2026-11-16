"""A coordinate with no position is refused where the caller writes it.

NaN and the infinities cannot cross the JSON wire, and a placeholder position is not one the
caller named, so every annotation primitive, every coordinate-taking ``annotate_*`` helper and
the ``fm.NormCoord`` and ``fm.PixelCoord`` wrappers refuse a non-finite number at construction,
whatever axis it would later meet. ``annotate_abline`` takes no coordinate, so it refuses a slope
and intercept that give no finite line.
"""

from __future__ import annotations

import datetime as dt
from typing import Any, Callable

import numpy as np
import polars as pl
import pytest

import ferrum as fm
import ferrum.annotation as ann

_NAN = float("nan")
_INF = float("inf")
_NEG_INF = float("-inf")
_SRC = "data:image/png;base64,iVBORw0KGgo="


def _coordinate_message(entry: str, field: str, value: str, axis: str) -> str:
    return (
        f"{entry}: {field}={value} is not a finite number, so it has no position; "
        f"pass a value the {axis} axis holds, fm.NormCoord or fm.PixelCoord"
    )


_PRIMITIVES: list[tuple[Callable[[], Any], str]] = [
    (lambda: ann.text(_NAN, 2.0, "t"), _coordinate_message("annotation.text", "x", "nan", "x")),
    (lambda: ann.text(2.0, _INF, "t"), _coordinate_message("annotation.text", "y", "inf", "y")),
    (
        lambda: ann.text(_NEG_INF, 2.0, "t"),
        _coordinate_message("annotation.text", "x", "-inf", "x"),
    ),
    (
        lambda: ann.text(np.float64("nan"), 2.0, "t"),
        _coordinate_message("annotation.text", "x", "nan", "x"),
    ),
    (
        lambda: ann.arrow(1.0, 1.0, _NAN, 2.0),
        _coordinate_message("annotation.arrow", "x2", "nan", "x"),
    ),
    (
        lambda: ann.arrow(1.0, 1.0, 2.0, _INF),
        _coordinate_message("annotation.arrow", "y2", "inf", "y"),
    ),
    (
        lambda: ann.rect(_NAN, 1.0, 2.0, 3.0),
        _coordinate_message("annotation.rect", "x1", "nan", "x"),
    ),
    (
        lambda: ann.rect(1.0, 1.0, 2.0, _NEG_INF),
        _coordinate_message("annotation.rect", "y2", "-inf", "y"),
    ),
    (
        lambda: ann.line(1.0, _NAN, 2.0, 3.0),
        _coordinate_message("annotation.line", "y1", "nan", "y"),
    ),
    (
        lambda: ann.span("x", _NAN, 2.0),
        _coordinate_message("annotation.span", "start", "nan", "x"),
    ),
    (
        lambda: ann.span("x", 1.0, _INF),
        _coordinate_message("annotation.span", "end", "inf", "x"),
    ),
    (
        lambda: ann.span("y", _NEG_INF, 2.0),
        _coordinate_message("annotation.span", "start", "-inf", "y"),
    ),
    (
        lambda: ann.bracket(1.0, 2.0, _NAN, 2.0, label="r"),
        _coordinate_message("annotation.bracket", "x2", "nan", "x"),
    ),
    (
        lambda: ann.callout(_INF, 1.0, "n"),
        _coordinate_message("annotation.callout", "x", "inf", "x"),
    ),
    (
        lambda: ann.callout(1.0, 1.0, "n", text_x=2.0, text_y=_NAN),
        _coordinate_message("annotation.callout", "text_y", "nan", "y"),
    ),
    (
        lambda: ann.image(_NAN, 1.0, _SRC, width=10, height=10),
        _coordinate_message("annotation.image", "x", "nan", "x"),
    ),
]


@pytest.mark.parametrize("build,message", _PRIMITIVES, ids=range(len(_PRIMITIVES)))
def test_a_non_finite_primitive_coordinate_is_refused_where_written(build, message):
    with pytest.raises(ValueError) as refused:
        build()
    assert str(refused.value) == message


def test_dates_strings_and_positions_pass_the_coordinate_check():
    """The negative control: only a number can be non-finite."""
    ann.text(dt.date(2020, 1, 2), "b", "t")
    ann.text("b", 2.0, "t")
    ann.text(fm.NormCoord(0.5), fm.PixelCoord(40.0), "t")
    ann.span("x", dt.date(2020, 1, 1), "2020-01-02")
    ann.callout(1.0, 1.0, "n", text_x="b", text_y=None)


def test_a_span_checks_its_axis_before_its_ends():
    with pytest.raises(ValueError) as refused:
        ann.span("z", _NAN, 1.0)
    assert str(refused.value) == "AnnotationSpan.axis: axis must be one of ['x', 'y']; got 'z'"


_HELPERS: list[tuple[Callable[[], Any], str]] = [
    (lambda: fm.annotate_hline(_NAN), _coordinate_message("annotate_hline", "y", "nan", "y")),
    (lambda: fm.annotate_hline(_INF), _coordinate_message("annotate_hline", "y", "inf", "y")),
    (lambda: fm.annotate_hline(_NEG_INF), _coordinate_message("annotate_hline", "y", "-inf", "y")),
    (
        lambda: fm.annotate_hline(np.float64("nan")),
        _coordinate_message("annotate_hline", "y", "nan", "y"),
    ),
    (lambda: fm.annotate_vline(_NAN), _coordinate_message("annotate_vline", "x", "nan", "x")),
    (lambda: fm.annotate_vline(_INF), _coordinate_message("annotate_vline", "x", "inf", "x")),
    (
        lambda: fm.annotate_rect(_NAN, 3.0, 2.0, 6.0),
        _coordinate_message("annotate_rect", "x1", "nan", "x"),
    ),
    (
        lambda: fm.annotate_rect(1.0, 3.0, 2.0, _INF),
        _coordinate_message("annotate_rect", "y2", "inf", "y"),
    ),
    (
        lambda: fm.annotate_text(_NEG_INF, 2.0, "t"),
        _coordinate_message("annotate_text", "x", "-inf", "x"),
    ),
    (
        lambda: fm.annotate_text(2.0, _NAN, "t"),
        _coordinate_message("annotate_text", "y", "nan", "y"),
    ),
    (
        lambda: fm.annotate_arrow(1.0, 1.0, _NAN, 2.0),
        _coordinate_message("annotate_arrow", "x2", "nan", "x"),
    ),
    (
        lambda: fm.annotate_arrow(1.0, 1.0, 2.0, _INF, label="L"),
        _coordinate_message("annotate_arrow", "y2", "inf", "y"),
    ),
]


@pytest.mark.parametrize("build,message", _HELPERS, ids=range(len(_HELPERS)))
def test_a_non_finite_helper_coordinate_is_refused_naming_the_helper_and_its_parameter(
    build, message
):
    with pytest.raises(ValueError) as refused:
        build()
    assert str(refused.value) == message


@pytest.mark.parametrize(
    "slope,intercept,message",
    [
        (_NAN, 0.0, "annotate_abline: slope=nan and intercept=0.0 give no finite line; "),
        (1.0, _INF, "annotate_abline: slope=1.0 and intercept=inf give no finite line; "),
        (1e303, 0.0, "annotate_abline: slope=1e+303 and intercept=0.0 give no finite line; "),
    ],
    ids=["nan-slope", "inf-intercept", "overflowing-end"],
)
def test_annotate_abline_refuses_a_slope_and_intercept_that_give_no_finite_line(
    slope, intercept, message
):
    with pytest.raises(ValueError) as refused:
        fm.annotate_abline(slope, intercept)
    assert str(refused.value) == message + "pass a finite slope and intercept"


def test_a_finite_abline_is_unchanged():
    assert fm.annotate_abline(1.0, 0.0)._annotation_primitive.x1 == -1e6


@pytest.mark.parametrize("value", [_NAN, _INF])
def test_a_non_finite_coordinate_is_refused_before_any_axis_is_read(value):
    """The band-axis chart is never rendered: the refusal fires at ``ann.text``."""
    bars = fm.Chart(pl.DataFrame({"c": ["a", "b"], "v": [1.0, 2.0]})).mark_bar()
    bars = bars.encode(x="c", y="v")
    with pytest.raises(ValueError) as refused:
        bars + ann.text(value, 2.0, "t")
    assert str(refused.value) == _coordinate_message("annotation.text", "x", str(value), "x")


@pytest.mark.parametrize(
    "build,message",
    [
        (
            lambda: fm.NormCoord(_NAN),
            "fm.NormCoord: value=nan is not a finite number; "
            "pass a fraction of the plot area, such as 0.5",
        ),
        (
            lambda: fm.norm(_INF),
            "fm.NormCoord: value=inf is not a finite number; "
            "pass a fraction of the plot area, such as 0.5",
        ),
        (
            lambda: fm.PixelCoord(_NEG_INF),
            "fm.PixelCoord: value=-inf is not a finite number; "
            "pass a pixel offset from the plot origin, such as 40",
        ),
        (
            lambda: fm.px(np.float64("nan")),
            "fm.PixelCoord: value=nan is not a finite number; "
            "pass a pixel offset from the plot origin, such as 40",
        ),
    ],
    ids=["norm-nan", "norm-factory-inf", "pixel-neg-inf", "pixel-factory-nan"],
)
def test_a_non_finite_norm_or_pixel_coordinate_is_refused_at_construction(build, message):
    with pytest.raises(ValueError) as refused:
        build()
    assert str(refused.value) == message


def test_the_coordinate_is_checked_before_the_colour():
    with pytest.raises(ValueError) as both:
        ann.text(_NAN, 2.0, "t", color="zzz")
    assert str(both.value) == _coordinate_message("annotation.text", "x", "nan", "x")
    with pytest.raises(ValueError) as colour_only:
        ann.text(1.0, 2.0, "t", color="zzz")
    assert str(colour_only.value).startswith("annotation.text: color='zzz' is not a valid color")


_NOT_A_FLOAT_NONE = "float() argument must be a string or a real number, not 'NoneType'"
_NOT_A_FLOAT_LIST = "float() argument must be a string or a real number, not 'list'"


@pytest.mark.parametrize(
    "build,error,text",
    [
        (
            lambda: fm.annotate_abline("a", 0),
            TypeError,
            "can't multiply sequence by non-int of type 'float'",
        ),
        (
            lambda: fm.annotate_abline(None, 0),
            TypeError,
            "unsupported operand type(s) for *: 'NoneType' and 'float'",
        ),
        (lambda: fm.annotate_abline(True, 0), None, None),
        (lambda: fm.annotate_hline(None), TypeError, _NOT_A_FLOAT_NONE),
        (lambda: fm.annotate_hline([1]), TypeError, _NOT_A_FLOAT_LIST),
        (lambda: fm.annotate_hline("b"), None, None),
        (lambda: fm.annotate_hline(True), None, None),
        (lambda: ann.text(None, 1.0, "t"), None, None),
        (lambda: ann.text(True, 1.0, "t"), None, None),
        (lambda: ann.text([1], 1.0, "t"), None, None),
        (lambda: ann.text("b", 1.0, "t"), None, None),
        (lambda: fm.NormCoord("a"), None, None),
        (lambda: fm.NormCoord(None), None, None),
        (lambda: fm.PixelCoord(True), None, None),
        (lambda: fm.PixelCoord([1]), None, None),
    ],
    ids=[
        "abline-str",
        "abline-none",
        "abline-bool",
        "hline-none",
        "hline-list",
        "hline-str",
        "hline-bool",
        "text-none",
        "text-bool",
        "text-list",
        "text-str",
        "norm-str",
        "norm-none",
        "pixel-bool",
        "pixel-list",
    ],
)
def test_a_value_that_is_not_a_number_keeps_the_outcome_it_had(build, error, text):
    """The finite-number check speaks only for a real number: a string, None, a bool or a list
    is refused (or accepted) exactly as it was before the check existed."""
    if error is None:
        build()
        return
    with pytest.raises(error) as refused:
        build()
    assert str(refused.value) == text
