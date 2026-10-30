"""Endpoint direct labels never sit on the line they name.

``_direct_label_endpoint`` places one text label per series at that
series' own endpoint, exactly 4px past it (``"end"``) or before it
(``"start"``), and must clear every drawn line in the plot -- not just
its own series, but every sibling series and a reference line like
``gain_chart``'s "Baseline" too -- even though the label's color (and no
legend) makes it read as one line's own name.

A label's box occupies one side of its SVG anchor only, decided by
``text-anchor``: ``start`` (``align="left"``) extends rightward from the
anchor, ``end`` (``align="right"``) extends leftward. So a box provably
clears a line's own x-range without measuring glyph widths: an ``end``
(last-point) label's anchor at or past a line's own rightmost x, with
``text-anchor="start"``, puts its whole box (extending further right) at
or past that point; a ``start`` (first-point) label mirrors this to the
left of a line's own leftmost x with ``text-anchor="end"``. These checks
read the real rendered polyline points and the real rendered ``<text>``
attributes -- never a hand-built glyph shape.
"""

from __future__ import annotations

import re

import numpy as np
import polars as pl
import pytest
from sklearn.linear_model import LogisticRegression, Ridge

import ferrum as fm
from ferrum import Chart
from ferrum._direct_label import _direct_label_endpoint
from tests._svg_extents import text_elements
from tests._svg_rects import polylines

#: The helper's own documented offset (module consumers: gain/lift/
#: learning-curve/validation-curve), pinned directly so a dx=0 or
#: wrong-magnitude placement fails even when it happens to still clear
#: every drawn line.
_LABEL_OFFSET_PX = 4.0

#: The helper's own documented vertical stagger: 5% of the label field's
#: full data-column y range.
_STAGGER_FRACTION = 0.05


def _attr(attrs: str, name: str) -> str | None:
    m = re.search(rf'{name}="([^"]*)"', attrs)
    return m.group(1) if m else None


def _label_anchor(svg: str, text: str, fill: str) -> tuple[float, str]:
    """``(x, text-anchor)`` of the one colored ``<text>`` reading ``text`` filled ``fill``."""
    matches = [
        (float(_attr(attrs, "x")), _attr(attrs, "text-anchor"))
        for attrs, body in text_elements(svg)
        if body == text and _attr(attrs, "fill") == fill
    ]
    assert matches, f"no label {text!r} filled {fill!r} found in the rendered SVG"
    assert len(set(matches)) == 1, f"ambiguous label {text!r} filled {fill!r}: {matches}"
    return matches[0]


def _own_line_extreme_x(svg: str, stroke: str, *, end: str) -> float:
    """The max (``end="max"``) or min (``end="min"``) x the one polyline drawn in ``stroke`` reaches."""
    xs = [x for s, points in polylines(svg) if s == stroke for x, _ in points]
    assert xs, f"no polyline drawn with stroke {stroke!r} found in the rendered SVG"
    return max(xs) if end == "max" else min(xs)


def _every_line_extreme_x(svg: str, *, end: str) -> float:
    """The max (``end="max"``) or min (``end="min"``) x ANY drawn polyline reaches.

    Every series' own line and any reference line (e.g. ``gain_chart``'s
    "Baseline") count -- not just the one line a label's own color draws --
    so a label that clears its own line but still sits on a sibling's is
    caught.
    """
    xs = [x for _, points in polylines(svg) for x, _ in points]
    assert xs, "no polyline drawn in the rendered SVG"
    return max(xs) if end == "max" else min(xs)


def _assert_end_label_clear(svg: str, text: str, color: str) -> None:
    """An "end" label starts exactly 4px right of its own line's last point,
    and that placement clears every drawn line in the plot, not only its own."""
    x, anchor = _label_anchor(svg, text, color)
    assert anchor == "start", (
        f"{text!r} label is anchored {anchor!r}, not left-aligned outside its line's end"
    )
    own_end_x = _own_line_extreme_x(svg, color, end="max")
    assert x == pytest.approx(own_end_x + _LABEL_OFFSET_PX, abs=1e-6), (
        f"{text!r} label box starts at x={x}, expected exactly "
        f"{_LABEL_OFFSET_PX}px right of its own line's end at x={own_end_x}"
    )
    every_end_x = _every_line_extreme_x(svg, end="max")
    assert x >= every_end_x - 1e-6, (
        f"{text!r} label box starts at x={x}, left of the plot's rightmost "
        f"drawn line at x={every_end_x} (a sibling line or reference line, "
        "not necessarily its own)"
    )


def _assert_start_label_clear(svg: str, text: str, color: str) -> None:
    """A "start" label ends exactly 4px left of its own line's first point,
    and that placement clears every drawn line in the plot, not only its own."""
    x, anchor = _label_anchor(svg, text, color)
    assert anchor == "end", (
        f"{text!r} label is anchored {anchor!r}, not right-aligned outside its line's start"
    )
    own_start_x = _own_line_extreme_x(svg, color, end="min")
    assert x == pytest.approx(own_start_x - _LABEL_OFFSET_PX, abs=1e-6), (
        f"{text!r} label box ends at x={x}, expected exactly "
        f"{_LABEL_OFFSET_PX}px left of its own line's start at x={own_start_x}"
    )
    every_start_x = _every_line_extreme_x(svg, end="min")
    assert x <= every_start_x + 1e-6, (
        f"{text!r} label box ends at x={x}, right of the plot's leftmost "
        f"drawn line at x={every_start_x} (a sibling line or reference "
        "line, not necessarily its own)"
    )


# ---------------------------------------------------------------------------
# Production figures: gain, lift, learning-curve, validation-curve.
# ---------------------------------------------------------------------------


def _binary_classifier_fixture():
    rng = np.random.default_rng(11)
    y = rng.integers(0, 2, size=200)
    X = rng.normal(size=(200, 3))
    X[:, 0] += y * 1.5
    clf = LogisticRegression().fit(X, y)
    return clf, X, y


def _learning_curve_source():
    rng = np.random.default_rng(12)
    X = rng.normal(size=(80, 4))
    y = X[:, 0] + 2 * X[:, 1] + rng.normal(scale=0.5, size=80)
    return Ridge(), X, y


def test_gain_chart_binary_unnamed_labels_clear_their_lines():
    """Bare "0"/"1" digit labels stay, but sit right of every drawn line's end."""
    clf, X, y = _binary_classifier_fixture()
    svg = fm.gain_chart(clf, X, y).to_svg()
    for text, color in {"0": "#2563eb", "1": "#dc2626", "Baseline": "#d4a017"}.items():
        _assert_end_label_clear(svg, text, color)


def test_lift_chart_binary_labels_clear_their_lines():
    clf, X, y = _binary_classifier_fixture()
    svg = fm.lift_chart(clf, X, y).to_svg()
    for text, color in {"0": "#2563eb", "1": "#dc2626", "Baseline": "#d4a017"}.items():
        _assert_end_label_clear(svg, text, color)


def test_gain_chart_named_classes_label_clears_its_line():
    """A named class ("Baseline" plus a real class name) reads clear of every line too."""
    clf, X, y = _binary_classifier_fixture()
    source = fm.ModelSource(clf, X, y, class_names=["negative", "positive"])
    svg = fm.gain_chart(source).to_svg()
    for text, color in {
        "negative": "#2563eb",
        "positive": "#dc2626",
        "Baseline": "#d4a017",
    }.items():
        _assert_end_label_clear(svg, text, color)


def test_learning_curve_chart_labels_clear_their_lines():
    model, X, y = _learning_curve_source()
    svg = fm.learning_curve_chart(model, X, y, cv=3).to_svg()
    for text, color in {
        "Training Score": "#dc2626",
        "Cross-Validation Score": "#2563eb",
    }.items():
        _assert_end_label_clear(svg, text, color)


def test_validation_curve_chart_labels_clear_their_lines():
    model, X, y = _learning_curve_source()
    svg = fm.validation_curve_chart(
        model, X, y, param="alpha", values=[0.1, 1.0, 10.0], cv=3
    ).to_svg()
    for text, color in {
        "Training Score": "#dc2626",
        "Cross-Validation Score": "#2563eb",
    }.items():
        _assert_end_label_clear(svg, text, color)


# ---------------------------------------------------------------------------
# Helper-level: both endpoint sides, validation, and the properties that
# must not move.
# ---------------------------------------------------------------------------


def _two_series_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "x": [1, 2, 3, 4, 1, 2, 3, 4],
            "y": [0.1, 0.3, 0.5, 0.6, 0.4, 0.3, 0.2, 0.1],
            "split": ["a", "a", "a", "a", "b", "b", "b", "b"],
        }
    )


def test_end_position_label_box_right_of_the_last_point():
    df = _two_series_df()
    base = Chart(df).encode(x="x", y="y", color="split").mark_line()
    chart = _direct_label_endpoint(base, label_field="split", color_field="split")
    svg = chart.to_svg()
    for text, color in {"a": "#2563eb", "b": "#dc2626"}.items():
        _assert_end_label_clear(svg, text, color)


def test_start_position_label_box_left_of_the_first_point():
    """No production caller passes ``position="start"`` today; the helper's
    own contract for that side is pinned directly."""
    df = _two_series_df()
    base = Chart(df).encode(x="x", y="y", color="split").mark_line()
    chart = _direct_label_endpoint(base, label_field="split", color_field="split", position="start")
    svg = chart.to_svg()
    for text, color in {"a": "#2563eb", "b": "#dc2626"}.items():
        _assert_start_label_clear(svg, text, color)


def test_position_outside_end_or_start_is_refused():
    df = _two_series_df()
    base = Chart(df).encode(x="x", y="y", color="split").mark_line()
    with pytest.raises(ValueError, match=r"position must be one of .*'end'.*'start'"):
        _direct_label_endpoint(base, label_field="split", position="middle")


def test_label_color_still_matches_its_own_line():
    df = _two_series_df()
    base = Chart(df).encode(x="x", y="y", color="split").mark_line()
    chart = _direct_label_endpoint(base, label_field="split", color_field="split")
    svg = chart.to_svg()
    _, anchor_a = _label_anchor(svg, "a", "#2563eb")
    _, anchor_b = _label_anchor(svg, "b", "#dc2626")
    assert anchor_a == "start"
    assert anchor_b == "start"


def test_vertical_stagger_is_five_percent_of_the_y_range():
    """Two series ending at the identical point must stagger by exactly 5%
    of the label field's full data-column y range, not merely by some
    nonzero amount."""
    df = pl.DataFrame(
        {
            "x": [1, 2, 3, 1, 2, 3],
            "y": [0.1, 0.2, 0.3, 0.1, 0.2, 0.3],
            "split": ["a", "a", "a", "b", "b", "b"],
        }
    )
    expected_data_gap = _STAGGER_FRACTION * (df["y"].max() - df["y"].min())

    base = Chart(df).encode(x="x", y="y", color="split").mark_line()
    chart = _direct_label_endpoint(base, label_field="split", color_field="split")
    svg = chart.to_svg()

    # The chart's own rendered line carries two points of known data y (row
    # order 0.1, 0.2, 0.3 at x=1, 2, 3), at their own rendered pixel y, so
    # the scale's own px-per-unit slope is read off the SAME render the
    # labels come from -- not off the axis tick ladder, whose own
    # text-inset gutter can reserve a different pixel span than the one the
    # line (and the label sharing its y scale) draws through.
    points = next(pts for s, pts in polylines(svg) if s == "#2563eb")
    (_, y_px_at_0_1), (_, y_px_at_0_2), (_, _y_px_at_0_3) = points
    px_per_unit = abs((y_px_at_0_1 - y_px_at_0_2) / (0.2 - 0.1))

    ys = {
        body: float(_attr(attrs, "y"))
        for attrs, body in text_elements(svg)
        if body in ("a", "b") and _attr(attrs, "fill") in ("#2563eb", "#dc2626")
    }
    observed_pixel_gap = abs(ys["a"] - ys["b"])
    expected_pixel_gap = expected_data_gap * px_per_unit
    assert observed_pixel_gap == pytest.approx(expected_pixel_gap, rel=1e-3), (
        f"stagger gap {observed_pixel_gap}px, expected {expected_pixel_gap}px "
        "(5% of the y range, converted through the line's own scale)"
    )
