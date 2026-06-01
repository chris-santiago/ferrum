"""Regression tests for defect D5 — silent failures should be observable.

Four sub-defects:

  5a — Window frame sign is INVERTED (correctness bug).
       ``frame=(-k, 0)`` in the Vega/Altair convention means "k preceding rows
       through the current row" (a trailing window).  Today the sign is inverted
       in ``crates/ferrum-core/src/transform/data_window.rs``  ``compute_window_op``
       lines 356–363: ``start = (pos - preceding).max(0)`` uses the literal
       ``preceding`` value, so ``frame=(-2, 0)`` computes
       ``start = pos - (-2) = pos + 2``, which looks *forward* rather than backward
       and produces an out-of-bounds slice → all NaN.

  5b — Unsupported/dropped kwarg should WARN (not silently no-op or corrupt output).
       ``Y(stack='normalize')`` is in ``_honored_kwargs`` so it never fires
       ``warn_once``.  On ``mark_line`` the value is forwarded to the Rust render
       pipeline which interprets it as a stacking request; because ``mark_line`` has
       no stacking code path, the marks are silently dropped entirely (the clip-group
       is emitted but contains no polyline elements).  No warning is emitted.

  5c — Empty render partition should WARN, identifying the partition key.
       When a grid facet is requested (``row=`` + ``col=``) and data does not cover
       all ``(row, col)`` combinations, the missing panel is silently dropped.  The
       existing warning ``Layout(PanelsDropped { count: N })`` names the *count* but
       not the identity of the missing partition (e.g. ``row='r2', col='c2'``).

  5d — Annotation ``label_position`` is honored or warns.
       The ``span()`` annotation factory accepts ``label_position='top'``,
       ``'middle'``, or ``'bottom'``.  In
       ``crates/ferrum-core/src/render/annotation.rs`` ``emit_span`` the field is
       marked ``#[allow(dead_code)]`` and the label is always placed at the
       vertical center (``y + h * 0.5``) regardless of the requested position.
       No warning is emitted.

All tests in this file are RED today and will become GREEN once the fixes land.

Root causes (file:line):
  5a — Rust  ``crates/ferrum-core/src/transform/data_window.rs:356``
  5b — Python ``src/ferrum/encoding/positional.py:8-28``   (``_RENDERED_HONORED``
       lists ``stack`` unconditionally; mark-level filtering needed) and/or
       Rust ``crates/ferrum-core/src/render/position.rs`` (line-stacking path)
  5c — Rust  ``crates/ferrum-core/src/layout/mod.rs:699``
       (``LayoutWarning::PanelsDropped`` — needs partition key in the message)
  5d — Rust  ``crates/ferrum-core/src/render/annotation.rs:118-119``
       (``#[allow(dead_code)] label_position`` field; ``emit_span`` hardcodes center)

Fix layers:
  5a — Rust only: negate ``preceding`` before the frame-start calculation.
  5b — Python + Rust: either remove ``stack`` from ``_RENDERED_HONORED`` for
       channels on marks that never stack, or have the Rust renderer emit a
       warning when it receives a ``stack`` request it cannot fulfill.
  5c — Rust: add the dropped partition key(s) to the ``PanelsDropped`` warning
       message; Python: surface the key as part of the UserWarning text.
  5d — Rust: implement the three ``label_position`` cases (top/middle/bottom) in
       ``emit_span``; until then, emit a Python-level warning when it is set
       to a non-default value.
"""

from __future__ import annotations

import re
import warnings

import polars as pl
import pytest

import ferrum as fm
import ferrum.annotation.primitives as ann_prims
from ferrum._warn import reset_warnings
from ferrum.annotation import Annotate
from ferrum.encoding import Text


# ---------------------------------------------------------------------------
# SVG helpers (shared)
# ---------------------------------------------------------------------------


def _mark_text_values(svg: str) -> list[str]:
    """Return text content of mark-text elements rendered inside the plot area.

    Mark-text elements produced by ``mark_text()`` use ``text-anchor="middle"``
    and have their ``y`` coordinate within the plot area (y < 430 in a standard
    480px-height chart).  Axis tick labels also use ``text-anchor="middle"`` but
    appear below the plot area (y ≈ 444px for a 480px chart), so they are
    excluded by a y-position threshold.

    The format string ``.2f`` is used in tests, so mark-text values contain a
    decimal point (e.g. ``"2.00"``) while x-axis integer ticks do not.
    Both heuristics are applied: y < 435 AND content contains ``"."``.
    """
    texts_with_attrs = re.findall(r"<text([^>]+)>([^<]+)</text>", svg)
    values = []
    for attrs, content in texts_with_attrs:
        # Skip y-axis tick labels (text-anchor="end").
        if 'text-anchor="end"' in attrs:
            continue
        # Skip elements that don't use text-anchor="middle" at all
        # (would be text-anchor="start" for legend labels etc.).
        if 'text-anchor="middle"' not in attrs:
            continue
        stripped = content.strip()
        # Must look like a floating-point number with a decimal point
        # (mark-text with format='.2f' → "2.00", "3.50", etc.).
        if "." not in stripped:
            continue
        try:
            float(stripped)
        except ValueError:
            continue
        # Must be inside the plot area (y < 435) to exclude x-axis tick labels
        # which appear at approximately y=444 in a 480px chart.
        y_match = re.search(r'\by="([^"]+)"', attrs)
        if y_match:
            try:
                y_val = float(y_match.group(1))
                if y_val < 435:
                    values.append(stripped)
            except ValueError:
                pass
    return values


def _y_axis_max(svg: str) -> float | None:
    """Return the maximum numeric value on the y-axis tick labels."""
    texts = re.findall(r"<text\s([^>]+)>([^<]+)</text>", svg)
    values = []
    for attrs, content in texts:
        if 'text-anchor="end"' in attrs:
            try:
                values.append(float(content.strip()))
            except ValueError:
                pass
    return max(values) if values else None


def _mark_polylines(svg: str) -> list[str]:
    """Return all <polyline> point-strings from the SVG (line mark segments)."""
    return re.findall(r'<polyline points="([^"]+)"', svg)


def _find_span_label_y(svg: str, label_text: str) -> float | None:
    """Return the SVG y-coordinate of a text element whose content is *label_text*.

    Returns ``None`` if no such element is found.
    """
    texts = re.findall(r"<text([^>]+)>([^<]+)</text>", svg)
    for attrs, content in texts:
        if content.strip() == label_text:
            y_match = re.search(r'\by="([^"]+)"', attrs)
            if y_match:
                return float(y_match.group(1))
    return None


# ---------------------------------------------------------------------------
# 5a — Window frame sign is INVERTED
# ---------------------------------------------------------------------------


@pytest.fixture
def window_df() -> pl.DataFrame:
    """Five-row ordered series for rolling-mean tests.

    ``val = [1, 2, 3, 4, 5]``, ``idx = [0, 1, 2, 3, 4]``.

    With a 3-row trailing window (``frame=(-2, 0)`` in Vega convention —
    current row plus 2 preceding), the expected rolling means are:

    ====  ===  ==========================  ========
    row   val  window rows (partition pos)  mean
    ====  ===  ==========================  ========
    0     1    [0]                         1.0
    1     2    [0, 1]                      1.5
    2     3    [0, 1, 2]                   2.0
    3     4    [1, 2, 3]                   3.0
    4     5    [2, 3, 4]                   4.0
    ====  ===  ==========================  ========

    Under the current bug ``frame=(-2, 0)`` computes
    ``start = pos - preceding = pos - (-2) = pos + 2`` which is out of
    bounds for most rows, so every result is NaN.
    """
    return pl.DataFrame(
        {
            "val": [1.0, 2.0, 3.0, 4.0, 5.0],
            "idx": [0.0, 1.0, 2.0, 3.0, 4.0],
        }
    )


def test_window_frame_negative_preceding_is_not_all_nan(window_df: pl.DataFrame) -> None:
    """``transform_window`` with ``frame=(-2, 0)`` must produce non-NaN trailing means.

    The Vega/Altair convention documented in ``transforms.py`` defines
    ``frame=(preceding, following)`` where ``preceding`` is NEGATIVE to count
    rows *before* the current row.  ``frame=(-2, 0)`` therefore means
    "2 preceding rows + current row" — a 3-row trailing window.

    For ``val=[1,2,3,4,5]`` sorted by ``idx``, the expected outputs are
    ``[1.0, 1.5, 2.0, 3.0, 4.0]``.

    Under the current bug, ``data_window.rs compute_window_op`` treats
    ``preceding=-2`` literally: ``start = pos - (-2) = pos + 2``.  At
    ``pos=0`` this gives ``start=2, end=1`` (empty slice) → NaN.  Every
    row produces NaN, so no mark-text elements appear and the y-axis
    scale collapses to [0, 1] rather than [1.0, 4.0].

    Red indicator: ``_mark_text_values()`` returns an empty list (no
    non-NaN mark-text renders); y-axis max ≤ 1.0 rather than ≥ 4.0.
    """
    t = fm.transform_window(
        {"op": "mean", "field": "val", "as": "rolling_mean"},
        sort=["idx"],
        frame=(-2, 0),
    )
    svg = (
        fm.Chart(window_df)
        .mark_text()
        .encode(
            x="idx:Q",
            y="rolling_mean:Q",
            text=Text("rolling_mean", format=".2f"),
        )
        .transform(t)
        .show_svg()
    )

    mark_values = _mark_text_values(svg)
    assert mark_values, (
        "transform_window(frame=(-2, 0)) produced no rendered mark-text values — "
        "all rolling_mean values are NaN.  The frame sign is inverted: "
        "frame=(-2, 0) with preceding=-2 computes start = pos - (-2) = pos + 2 "
        "(forward-looking) rather than pos - 2 (backward-looking).  "
        "Expected mark values: ['1.00', '1.50', '2.00', '3.00', '4.00'].  "
        "Current behavior: all NaN → empty mark group, y-axis scale 0–1."
    )


def test_window_frame_negative_preceding_values_are_correct(window_df: pl.DataFrame) -> None:
    """``frame=(-2, 0)`` trailing means for ``val=[1,2,3,4,5]`` must be exact.

    Expected: ``[1.00, 1.50, 2.00, 3.00, 4.00]`` (formatted to 2dp).

    Row-by-row derivation:
    - idx=0: only row 0 in window → mean([1]) = 1.00
    - idx=1: rows 0,1 → mean([1,2]) = 1.50
    - idx=2: rows 0,1,2 → mean([1,2,3]) = 2.00
    - idx=3: rows 1,2,3 → mean([2,3,4]) = 3.00
    - idx=4: rows 2,3,4 → mean([3,4,5]) = 4.00

    Under the bug all five are NaN and ``_mark_text_values()`` returns ``[]``.
    """
    t = fm.transform_window(
        {"op": "mean", "field": "val", "as": "rolling_mean"},
        sort=["idx"],
        frame=(-2, 0),
    )
    svg = (
        fm.Chart(window_df)
        .mark_text()
        .encode(
            x="idx:Q",
            y="rolling_mean:Q",
            text=Text("rolling_mean", format=".2f"),
        )
        .transform(t)
        .show_svg()
    )

    mark_values = sorted(_mark_text_values(svg))
    expected = sorted(["1.00", "1.50", "2.00", "3.00", "4.00"])

    assert mark_values == expected, (
        f"transform_window(frame=(-2, 0)) produced mark values {mark_values!r} "
        f"but expected {expected!r}.  "
        "The Vega convention states frame=(-k, 0) is a k-preceding trailing window; "
        "the Rust data_window.rs currently treats the literal negative value as a "
        "forward offset, producing NaN for most rows.  "
        "Under the bug, mark_values is empty [] because all values are NaN."
    )


def test_window_frame_negative_preceding_yaxis_spans_data_range(
    window_df: pl.DataFrame,
) -> None:
    """When ``frame=(-2, 0)`` produces valid means, the y-axis must span them.

    The five rolling means (1.0–4.0) must appear on the y-axis.  Under the
    bug the y-axis spans [0, 1] (the NaN fallback domain) — its maximum tick
    is 1.0, not 4.0 or higher.

    This test guards the y-axis domain even if the mark-text assertion above
    passes for a different reason (e.g. text elements rendered elsewhere).
    """
    t = fm.transform_window(
        {"op": "mean", "field": "val", "as": "rolling_mean"},
        sort=["idx"],
        frame=(-2, 0),
    )
    svg = (
        fm.Chart(window_df)
        .mark_point()
        .encode(x="idx:Q", y="rolling_mean:Q")
        .transform(t)
        .show_svg()
    )

    y_max = _y_axis_max(svg)
    assert y_max is not None, "Could not extract y-axis tick labels from SVG."
    assert y_max >= 3.5, (
        f"y-axis maximum tick is {y_max:.2f} — expected ≥ 3.5 (data range is 1.0–4.0 "
        "for a valid trailing mean).  Under the frame-sign bug the y-axis max is ~1.0 "
        "(the NaN-collapsed domain fallback).  "
        "transform_window(frame=(-2, 0)) is producing all-NaN values."
    )


# ---------------------------------------------------------------------------
# 5b — Unsupported/dropped kwarg should WARN
# ---------------------------------------------------------------------------


def test_y_stack_normalize_on_mark_line_emits_warning() -> None:
    """``Y(stack='normalize')`` on ``mark_line`` must emit a UserWarning.

    ``mark_line`` has no stacking code path.  When ``stack='normalize'`` is
    forwarded to the Rust renderer, the line marks are silently dropped —
    the clipping group is emitted but contains no ``<polyline>`` elements.
    No warning is emitted today.

    After the fix: a warning naming ``stack`` (or ``stack='normalize'``) must
    be emitted, and the marks must NOT be silently dropped.

    Current behavior (RED):
    - No ``UserWarning`` is raised.
    - The mark group contains no ``<polyline>`` elements (marks vanish).

    The test asserts a warning is emitted; the marks-vanish symptom is
    covered by ``test_y_stack_normalize_on_mark_line_does_not_drop_marks``.
    """
    reset_warnings()
    df = pl.DataFrame(
        {
            "x": [1.0, 2.0, 1.0, 2.0],
            "grp": ["a", "a", "b", "b"],
            "val": [3.0, 4.0, 2.0, 5.0],
        }
    )

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = (
            fm.Chart(df)
            .mark_line()
            .encode(x="x:Q", y=fm.Y("val:Q", stack="normalize"), color="grp:N")
            .show_svg()
        )

    stack_warnings = [
        warning
        for warning in w
        if "stack" in str(warning.message).lower() or "normalize" in str(warning.message).lower()
    ]
    assert stack_warnings, (
        "Y(stack='normalize') on mark_line emitted no warning.  "
        "The stack kwarg is accepted silently (it is in Y._honored_kwargs) and "
        "forwarded to the Rust renderer, where mark_line has no stacking path.  "
        "The result is that line marks are silently dropped from the SVG with "
        "no diagnostic.  A UserWarning naming 'stack' or 'normalize' must be "
        "emitted when stack= is supplied on a mark that cannot honor it.  "
        f"Warnings actually received: {[str(ww.message) for ww in w]!r}"
    )


def test_y_stack_normalize_on_mark_line_does_not_drop_marks() -> None:
    """``Y(stack='normalize')`` on ``mark_line`` must not silently drop line marks.

    Today, forwarding ``stack='normalize'`` to ``mark_line`` causes the Rust
    renderer to emit an empty clipping group — all ``<polyline>`` elements
    disappear from the SVG.  This is data loss with no diagnostic.

    After the fix: either the warning is emitted AND the marks remain (the
    kwarg is ignored with a warning), or the marks are stacked correctly.

    This test asserts that the rendered SVG contains at least one
    ``<polyline>`` element (meaning the mark rendered, even if the stack
    param is silently no-op'd with a warning).

    Current behavior (RED):
    - The ``<g clip-path=...>`` element is empty — no ``<polyline>`` children.
    - The y-axis scale and legend still render (headers present), but data is gone.
    """
    df = pl.DataFrame(
        {
            "x": [1.0, 2.0, 1.0, 2.0],
            "grp": ["a", "a", "b", "b"],
            "val": [3.0, 4.0, 2.0, 5.0],
        }
    )

    svg = (
        fm.Chart(df)
        .mark_line()
        .encode(x="x:Q", y=fm.Y("val:Q", stack="normalize"), color="grp:N")
        .show_svg()
    )

    polylines = _mark_polylines(svg)
    assert polylines, (
        "Y(stack='normalize') on mark_line produced no <polyline> elements — "
        "all line marks were silently dropped.  The clip group is emitted but "
        "empty.  Current SVG mark group: <g clip-path=...></g> (no children).  "
        "After the fix, marks must render (possibly without stacking, with a "
        "warning) rather than disappearing silently.  "
        f"Polyline count: {len(polylines)}"
    )


def test_y_stack_normalize_on_mark_line_differs_from_no_stack() -> None:
    """Post-fix: ``Y(stack='normalize')`` on ``mark_line`` renders the same marks.

    After the D5b fix, ``stack='normalize'`` is stripped with a warning for
    non-stackable marks.  Both the no-stack and with-stack charts should
    produce polylines.  The two SVGs may differ in minor details (the
    with-stack render triggered a warning and had the kwarg stripped) but
    both must contain line marks.

    This replaces the former bug-documentation sentinel that asserted marks
    were absent when stack= was forwarded to a non-stackable mark renderer.
    """
    df = pl.DataFrame(
        {
            "x": [1.0, 2.0, 1.0, 2.0],
            "grp": ["a", "a", "b", "b"],
            "val": [3.0, 4.0, 2.0, 5.0],
        }
    )

    svg_no_stack = fm.Chart(df).mark_line().encode(x="x:Q", y="val:Q", color="grp:N").show_svg()
    svg_with_stack = (
        fm.Chart(df)
        .mark_line()
        .encode(x="x:Q", y=fm.Y("val:Q", stack="normalize"), color="grp:N")
        .show_svg()
    )

    polylines_no_stack = _mark_polylines(svg_no_stack)
    polylines_with_stack = _mark_polylines(svg_with_stack)

    assert polylines_no_stack, (
        "mark_line without stack= produced no polylines — baseline is broken.  "
        f"SVG length: {len(svg_no_stack)}"
    )
    assert polylines_with_stack, (
        "mark_line with Y(stack='normalize') produced no polylines — marks were "
        "silently dropped.  After the D5b fix, the stack kwarg is stripped with "
        "a warning and marks render normally.  "
        f"SVG length: {len(svg_with_stack)}"
    )


def test_layered_y_stack_normalize_on_mark_line_emits_warning() -> None:
    """Layered ``+`` path: ``Y(stack='normalize')`` on a ``mark_line`` layer must warn.

    The single-chart D5b fix guards ``_build_encoding_specs``.  The layered
    path ``Chart._build_layers_list`` re-serializes each layer's encoding
    independently.  Before this fix, ``stack`` was forwarded to Rust with no
    guard, so line marks were silently dropped and the warning was misleading
    (it fired from single-chart resolution but the layered path still emitted
    ``stack`` to Rust).

    This test builds ``(line with stack) + (point without stack)`` and asserts:
    (a) a warning naming ``stack`` or ``normalize`` is emitted, AND
    (b) the line layer's polylines are NOT silently dropped.

    It must FAIL against unfixed code (marks vanish, rendering is broken) and
    PASS after the fix (stack stripped with warning, marks render).
    """
    reset_warnings()
    df = pl.DataFrame(
        {
            "x": [1.0, 2.0, 1.0, 2.0],
            "grp": ["a", "a", "b", "b"],
            "val": [3.0, 4.0, 2.0, 5.0],
        }
    )

    line_layer = (
        fm.Chart(df).mark_line().encode(x="x:Q", y=fm.Y("val:Q", stack="normalize"), color="grp:N")
    )
    point_layer = fm.Chart(df).mark_point().encode(x="x:Q", y="val:Q", color="grp:N")
    layered = line_layer + point_layer

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        svg = layered.show_svg()

    # (a) A warning naming ``stack`` or ``normalize`` must be emitted.
    stack_warnings = [
        warning
        for warning in w
        if "stack" in str(warning.message).lower() or "normalize" in str(warning.message).lower()
    ]
    assert stack_warnings, (
        "Layered chart with Y(stack='normalize') on mark_line layer emitted no warning.  "
        "The _build_layers_list path forwarded stack= to Rust without the D5b guard.  "
        "A UserWarning naming 'stack' or 'normalize' must be emitted.  "
        f"Warnings actually received: {[str(ww.message) for ww in w]!r}"
    )

    # (b) The line layer's polylines must not be silently dropped.
    polylines = _mark_polylines(svg)
    assert polylines, (
        "Layered chart with Y(stack='normalize') on mark_line layer produced no "
        "<polyline> elements — line marks were silently dropped via the layered path.  "
        "After the fix, _build_layers_list strips stack= before emitting to Rust, "
        "so line marks render normally.  "
        f"SVG length: {len(svg)}"
    )


def _mark_rects(svg: str) -> list[str]:
    """Return all <rect> elements from the SVG (bar mark segments)."""
    return re.findall(r"<rect\s[^/]*/?>", svg)


def test_layered_bar_stack_guard_does_not_fire() -> None:
    """Guard: the D5b guard in ``_build_layers_list`` must NOT fire for ``mark_bar``.

    ``mark_bar`` is in ``_STACKABLE_MARKS``, so passing ``Y(stack='normalize')``
    on a bar layer must never emit the "stack ignored" warning.  This test
    ensures the guard condition (``layer_mark not in _STACKABLE_MARKS``) is
    correctly evaluated and bar marks still render when ``stack=`` is set.
    """
    df = pl.DataFrame(
        {
            "x": [1.0, 2.0, 1.0, 2.0],
            "grp": ["a", "a", "b", "b"],
            "val": [3.0, 4.0, 2.0, 5.0],
        }
    )

    bar_layer = (
        fm.Chart(df).mark_bar().encode(x="x:Q", y=fm.Y("val:Q", stack="normalize"), color="grp:N")
    )
    point_layer = fm.Chart(df).mark_point().encode(x="x:Q", y="val:Q", color="grp:N")
    layered = bar_layer + point_layer

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        svg = layered.show_svg()

    # The guard must not fire for a stackable mark: no "stack ignored" warning.
    stack_dropped_warnings = [
        warning
        for warning in w
        if ("stack" in str(warning.message).lower() or "normalize" in str(warning.message).lower())
        and "ignored" in str(warning.message).lower()
    ]
    assert not stack_dropped_warnings, (
        "mark_bar layer with Y(stack='normalize') in a layered chart emitted a "
        "'stack ignored' warning — the D5b guard incorrectly fired for a stackable mark.  "
        f"Warnings: {[str(ww.message) for ww in w]!r}"
    )

    # Bar marks must still render (rect elements present in SVG).
    rects = _mark_rects(svg)
    assert rects, (
        "mark_bar layer in a layered chart produced no <rect> elements — bar marks "
        "were silently dropped.  The D5b guard must not affect stackable marks.  "
        f"SVG length: {len(svg)}"
    )


# ---------------------------------------------------------------------------
# 5c — Empty render partition should WARN, identifying the partition key
# ---------------------------------------------------------------------------


@pytest.fixture
def sparse_grid_df() -> pl.DataFrame:
    """A 2-row × 2-col grid facet dataset with one combination absent.

    The data covers (r1, c1), (r1, c2), and (r2, c1) but NOT (r2, c2).
    When faceted by ``row='row_cat'`` and ``col='col_cat'``, the renderer
    must warn about the missing ``(row_cat='r2', col_cat='c2')`` panel.

    With the current ``partition_batch_by_field`` implementation, only
    row×col combinations that appear in the data are partitioned; the
    missing combination is left as a "dropped" panel in the grid layout.
    """
    return pl.DataFrame(
        {
            "x": [1.0, 2.0, 3.0],
            "y": [1.0, 2.0, 3.0],
            "row_cat": ["r1", "r1", "r2"],
            "col_cat": ["c1", "c2", "c1"],
        }
    )


def test_empty_facet_partition_emits_warning(sparse_grid_df: pl.DataFrame) -> None:
    """A grid facet with a missing (row, col) combination must emit a UserWarning.

    The current behavior emits ``Layout(PanelsDropped { count: 1 })`` — this
    IS a warning but it does not identify which partition is missing.

    After the fix: the warning message must identify the dropped partition,
    e.g. include ``row_cat='r2'`` or ``col_cat='c2'`` or ``(r2, c2)``.

    This test asserts the warning message includes at least one of the missing
    partition key values (``r2`` or ``c2``), which the current message does not.

    Current behavior (RED):
    - A ``UserWarning`` IS emitted: ``Layout(PanelsDropped { count: 1 })``.
    - The message does NOT contain ``r2``, ``c2``, ``row_cat``, or ``col_cat``.
    """
    reset_warnings()

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = (
            fm.Chart(sparse_grid_df)
            .mark_point()
            .encode(x="x:Q", y="y:Q")
            .facet(row="row_cat", col="col_cat")
            .show_svg()
        )

    # At least one warning must identify the missing partition.
    partition_identifying_warnings = [
        warning
        for warning in w
        if any(
            key in str(warning.message)
            for key in ["r2", "c2", "row_cat", "col_cat", "r2, c2", "(r2, c2)"]
        )
    ]

    assert partition_identifying_warnings, (
        "Grid facet with missing (r2, c2) combination emitted no warning that "
        "identifies the dropped partition.  "
        "Current warning: 'Layout(PanelsDropped { count: 1 })' — this counts "
        "dropped panels but does not name the partition key.  "
        "After the fix, the warning message must include the partition identity "
        "(e.g. row_cat='r2', col_cat='c2') so the user can diagnose the data gap.  "
        f"All warnings received: {[str(ww.message) for ww in w]!r}"
    )


def test_empty_facet_partition_warning_is_emitted_at_all(
    sparse_grid_df: pl.DataFrame,
) -> None:
    """Guard: a grid facet with a missing combination must emit SOME warning.

    This test documents the current behavior: the existing
    ``Layout(PanelsDropped { count: 1 })`` warning IS already emitted.
    This guard ensures that the existing warning mechanism is not accidentally
    removed when fixing 5c (the fix should EXPAND the message, not remove it).

    If this test fails, the existing PanelsDropped warning was removed and
    must be restored along with the partition-identifying extension.
    """
    reset_warnings()

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = (
            fm.Chart(sparse_grid_df)
            .mark_point()
            .encode(x="x:Q", y="y:Q")
            .facet(row="row_cat", col="col_cat")
            .show_svg()
        )

    dropped_warnings = [
        warning
        for warning in w
        if "drop" in str(warning.message).lower()
        or "panel" in str(warning.message).lower()
        or "PanelsDropped" in str(warning.message)
    ]

    assert dropped_warnings, (
        "Grid facet with missing panel emitted NO dropped-panel warning at all.  "
        "Expected at least 'Layout(PanelsDropped { count: 1 })' or equivalent.  "
        f"All warnings received: {[str(ww.message) for ww in w]!r}"
    )


# ---------------------------------------------------------------------------
# 5d — Annotation ``label_position`` honored or warns
# ---------------------------------------------------------------------------


@pytest.fixture
def annotation_chart_df() -> pl.DataFrame:
    """Minimal DataFrame for annotation rendering tests."""
    return pl.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "y": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )


def _render_span_with_label_position(
    df: pl.DataFrame,
    label_position: str,
) -> str:
    """Render a chart with a y-span annotation with the given label_position."""
    span = ann_prims.span(
        "y",
        start=2.0,
        end=4.0,
        fill="#aaa",
        label="target zone",
        label_position=label_position,
    )
    return fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q").__add__(Annotate([span])).show_svg()


def test_span_label_position_changes_rendered_y_or_warns() -> None:
    """``span(..., label_position='top')`` must either move the label upward
    relative to ``label_position='bottom'``, OR emit a warning.

    The spec allows two possible outcomes after the fix:
    1. The label y-coordinate for ``'top'`` is strictly less than for
       ``'bottom'`` (in SVG, lower y = higher on screen).
    2. A ``UserWarning`` naming ``label_position`` is emitted.

    Currently (RED): both ``'top'`` and ``'bottom'`` produce the same y-position
    (the vertical center of the span) and no warning is emitted.

    Observable root cause: in ``annotation.rs emit_span`` the label is always
    placed at ``y + h * 0.5`` regardless of ``label_position``, which carries
    ``#[allow(dead_code)]`` on the Rust struct field.
    """
    reset_warnings()
    df = pl.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "y": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        svg_top = _render_span_with_label_position(df, "top")
        svg_bottom = _render_span_with_label_position(df, "bottom")

    label_position_warnings = [
        warning
        for warning in w
        if "label_position" in str(warning.message).lower()
        or "label" in str(warning.message).lower()
    ]

    y_top = _find_span_label_y(svg_top, "target zone")
    y_bottom = _find_span_label_y(svg_bottom, "target zone")

    assert y_top is not None, (
        "span annotation with label='target zone' and label_position='top' "
        "produced no text element.  The span annotation may have failed to render."
    )
    assert y_bottom is not None, (
        "span annotation with label='target zone' and label_position='bottom' "
        "produced no text element.  The span annotation may have failed to render."
    )

    # Either the label is repositioned OR a warning is emitted.
    position_changes = y_top < y_bottom  # SVG: lower y = higher on screen
    warns = bool(label_position_warnings)

    assert position_changes or warns, (
        f"span(label_position='top') y={y_top:.1f}, "
        f"span(label_position='bottom') y={y_bottom:.1f}: "
        "the label y-position did not change (label_position is silently ignored) "
        "AND no warning was emitted naming 'label_position'.  "
        "After the fix, either the position changes (top < bottom in SVG coords) "
        "or a UserWarning naming 'label_position' is raised when the value cannot "
        "be honored.  Currently label_position is always placed at the vertical "
        "center of the span (y + h/2) with #[allow(dead_code)] in Rust.  "
        f"Warnings received: {[str(ww.message) for ww in w]!r}"
    )


def test_span_label_position_top_less_than_middle_less_than_bottom() -> None:
    """label_position='top' must render the label above 'middle', 'bottom' below.

    In SVG coordinates lower y = higher on screen, so:
        y_top < y_mid < y_bot

    This replaces the former bug-confirmation sentinel
    ``test_span_label_position_top_equals_middle_equals_bottom_today``.
    """
    df = pl.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "y": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )

    svg_top = _render_span_with_label_position(df, "top")
    svg_mid = _render_span_with_label_position(df, "middle")
    svg_bot = _render_span_with_label_position(df, "bottom")

    y_top = _find_span_label_y(svg_top, "target zone")
    y_mid = _find_span_label_y(svg_mid, "target zone")
    y_bot = _find_span_label_y(svg_bot, "target zone")

    assert y_top is not None and y_mid is not None and y_bot is not None, (
        "One or more span labels were not found in the SVG.  "
        f"y_top={y_top}, y_mid={y_mid}, y_bot={y_bot}"
    )

    assert y_top < y_mid, (
        f"label_position='top' y={y_top:.1f} must be less than 'middle' y={y_mid:.1f} "
        "(SVG: lower y = higher on screen)."
    )
    assert y_mid < y_bot, (
        f"label_position='middle' y={y_mid:.1f} must be less than 'bottom' y={y_bot:.1f}."
    )


def test_span_label_position_default_renders_label() -> None:
    """Guard: a span with ``label='text'`` renders a label element (sanity check).

    This test passes today and must continue to pass after the fix.  It ensures
    the basic span-label rendering is not broken by the 5d changes.
    """
    df = pl.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "y": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )

    svg = _render_span_with_label_position(df, "middle")
    y = _find_span_label_y(svg, "target zone")

    assert y is not None, (
        "span(label='target zone', label_position='middle') produced no text "
        "element with content 'target zone'.  Basic span label rendering is broken."
    )
