"""A merged legend's swatches carry every channel folded into it (GH #111).

Wave 5 task 4 (spec ``.claude/output/specs/2026-09-18-wave5-coord-scale-mark-design.md``
§4.6). When ``stroke_dash`` or ``shape`` encodes the same field as ``color``,
that channel gets no legend block of its own — the color legend already
enumerates the field. What shipped was a color legend enumerating it with
three identical solid circles while the marks drew three dash patterns (or
three glyphs), so the suppression cost the reader exactly the information the
legend exists to give.

What is pinned here:

- **Dash.** Each merged entry draws a LINE swatch whose ``stroke-dasharray``
  is the one its own category's polyline draws. The swatch KIND is the load-
  bearing half: ``dash_pattern`` is read only on the renderer's line arm, so a
  circle carrying a dash renders byte-identically to one without — which is
  why the issue's own prescribed "thread the pattern through" fix measured as
  a no-op.
- **By label, never by index.** The color domain and the dash domain are both
  built over the same field but each with its OWN ``sort=``, so
  ``StrokeDash(..., sort="descending")`` yields two correct domains in
  opposite orders. Every dash test here uses that spec: an index-paired
  implementation transposes the patterns and fails.
- **Shape.** The same merge for ``shape=``: circle / square / cross, matching
  the glyphs the marks draw.
- **The negative control.** A duplicate column (``stroke_dash="h"``, a
  different field holding the same values) is NOT the same field, so nothing
  is folded: the color legend keeps its circles and the dash channel gets its
  own block. This is what separates "the suppression path learned to carry
  identity" from "every legend swatch became a dashed line".
- **The two adjacent behaviors (§4.6).** A folded channel's explicit ``title=``
  fills the title slot color left at its field-name default instead of being
  silently discarded, and disagreeing explicit titles keep color's with a
  warning naming the other. ``Color(..., legend=None)`` plus a same-field
  ``stroke_dash`` renders the dash block on its own: suppression exists to
  avoid a duplicate legend, and with no color legend there is nothing to
  duplicate.
"""

from __future__ import annotations

import warnings

import polars as pl
import pytest

import ferrum as fm
from _snapshots import legend_texts
from _svg_extents import ordered_text_entries
from _svg_rects import svg_elements

CATEGORIES = ["a", "b", "c"]

#: Every SVG tag a legend swatch can be drawn with: a color/size circle, a
#: square swatch or square glyph, and the line segments a dash swatch or a
#: stroke-drawn point glyph (cross, triangle outline, ...) emits.
SWATCH_TAGS = ("circle", "rect", "line")


@pytest.fixture
def df() -> pl.DataFrame:
    """Three categories, plus ``h`` — a DIFFERENT field holding the same values.

    ``h`` is the negative control's whole point: identical data, so a fix that
    keyed on the values rather than on the field would pass the same-field
    tests and still be wrong.
    """
    return pl.DataFrame(
        {
            "x": [1.0, 2.0, 3.0] * 3,
            "y": [1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0],
            "g": ["a"] * 3 + ["b"] * 3 + ["c"] * 3,
            "h": ["a"] * 3 + ["b"] * 3 + ["c"] * 3,
        }
    )


def render(chart) -> tuple[str, list[str]]:
    """``(svg, warning_messages)`` for one chart, with warnings captured."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        svg = chart.to_svg()
    return svg, [str(w.message) for w in caught]


def _center_x(attrs: dict[str, str]) -> float:
    if "cx" in attrs:
        return float(attrs["cx"])
    if "x1" in attrs:
        return (float(attrs["x1"]) + float(attrs["x2"])) / 2.0
    return float(attrs["x"]) + float(attrs["width"]) / 2.0


def _center_y(attrs: dict[str, str]) -> float:
    if "cy" in attrs:
        return float(attrs["cy"])
    if "y1" in attrs:
        return (float(attrs["y1"]) + float(attrs["y2"])) / 2.0
    return float(attrs["y"]) + float(attrs["height"]) / 2.0


def legend_rows(svg: str) -> list[tuple[str, list[dict[str, str]]]]:
    """``(label, swatch elements)`` for every legend entry, in document order.

    An entry is located from its own label ``<text>`` rather than from a plot-
    rect boundary: the swatch column sits immediately left of the label and is
    centered on the same row, which is true for the color legend and for every
    aux block regardless of where the legend strip was placed. Pairing swatch
    to label positionally is what lets a test say "the swatch standing for
    category ``b``" instead of "the second swatch" — the index reading whose
    silent transposition this file exists to catch.

    Repeated labels (a color block and an aux block both enumerating the same
    field) yield one row each, in document order, so a caller can assert on
    block count as well as on content.
    """
    labels = [(x, y, text) for x, y, text in ordered_text_entries(svg) if text in CATEGORIES]
    elements = [
        attrs | {"__tag__": tag}
        for tag in SWATCH_TAGS
        for attrs in svg_elements(svg, tag, lambda a: True)
    ]
    rows = []
    for lx, ly, label in labels:
        # The swatch column: within 20px to the LEFT of the label anchor (the
        # swatch box is ~12px wide with a 4px gap), on the same row. The 6px
        # vertical tolerance covers the aux blocks' label baseline offset
        # (`label_font_size * 0.35`) while staying far inside the ~17px row
        # pitch, so a row can never absorb its neighbour's swatch.
        row = [
            e for e in elements if lx - 20.0 <= _center_x(e) < lx and abs(_center_y(e) - ly) <= 6.0
        ]
        rows.append((label, row))
    return rows


def swatch_dasharray(row: list[dict[str, str]]) -> str:
    """The dasharray of a one-element line swatch, or a marker for the wrong shape.

    Returns the swatch's KIND when it is not a single line, so a failure
    message says "this entry drew a circle" rather than "expected '2,3', got
    None" — the circle is the defect, the missing dasharray is its symptom.
    """
    if len(row) != 1:
        return f"<{len(row)} elements: {[e['__tag__'] for e in row]}>"
    (swatch,) = row
    if swatch["__tag__"] != "line":
        return f"<{swatch['__tag__']} swatch>"
    return swatch.get("stroke-dasharray", "solid")


def mark_dasharrays(svg: str) -> dict[str, str]:
    """``stroke`` color → ``stroke-dasharray`` for every drawn polyline mark."""
    return {
        a["stroke"]: a.get("stroke-dasharray", "solid")
        for a in svg_elements(svg, "polyline", lambda a: "stroke" in a)
    }


def swatch_color(row: list[dict[str, str]]) -> str | None:
    """The one color a legend row's swatch elements are painted in."""
    colors = {e.get("fill") or e.get("stroke") for e in row}
    return colors.pop() if len(colors) == 1 else None


# ---------------------------------------------------------------------------
# stroke_dash folded into the color legend
# ---------------------------------------------------------------------------


def test_same_field_dash_swatches_draw_the_patterns_their_marks_draw(df):
    """Each merged entry is a line swatch dashed exactly like its own category's line.

    Pairing is by COLOR: a legend entry's swatch color comes from the color
    scale keyed on its label, and a polyline's comes from the same scale keyed
    on its category's value, so matching the two maps compares "what the
    legend promises for category X" against "what X's mark draws" without
    either side's ordering entering into it.

    RED before the fix in both halves: every entry drew a solid circle, so the
    swatch kind was wrong AND the pattern was absent.
    """
    chart = (
        fm.Chart(df)
        .mark_line()
        .encode(
            x="x:Q",
            y="y:Q",
            color="g:N",
            stroke_dash=fm.StrokeDash("g", sort="descending"),
        )
    )
    svg, _ = render(chart)

    rows = legend_rows(svg)
    assert [label for label, _ in rows] == CATEGORIES, (
        f"expected ONE merged legend block enumerating {CATEGORIES} once each; "
        f"got {[label for label, _ in rows]}"
    )
    legend = {swatch_color(row): swatch_dasharray(row) for label, row in rows}
    assert legend == mark_dasharrays(svg), (
        f"legend swatches {legend} disagree with the marks {mark_dasharrays(svg)}"
    )
    # Three genuinely different patterns, so the agreement above cannot be the
    # degenerate "everything is solid on both sides".
    assert len(set(legend.values())) == 3, legend
    assert "solid" in legend.values(), legend


def test_dash_patterns_follow_the_label_not_the_domain_position(df):
    """``sort="descending"`` on the dash channel does not transpose the swatches.

    The color domain sorts ascending (a, b, c) and the dash domain descending
    (c, b, a), so the two agree on the field and disagree on order. The dash
    palette hands its solid slot to the FIRST category of its own domain — ``c``
    — which is therefore the entry that must draw solid. An implementation
    that zipped the two domains by index would hand solid to ``a`` and pass
    every "the legend has three distinct patterns" check while showing each
    category another category's dash.
    """
    chart = (
        fm.Chart(df)
        .mark_line()
        .encode(
            x="x:Q",
            y="y:Q",
            color="g:N",
            stroke_dash=fm.StrokeDash("g", sort="descending"),
        )
    )
    svg, _ = render(chart)
    by_label = {label: swatch_dasharray(row) for label, row in legend_rows(svg)}
    assert by_label["c"] == "solid", by_label
    assert by_label["a"] != "solid" and by_label["b"] != "solid", by_label
    assert by_label["a"] != by_label["b"], by_label


def test_a_duplicate_column_dash_leaves_the_color_legend_alone(df):
    """NEGATIVE CONTROL: same values, different field → no merge, two blocks.

    ``h`` holds exactly the data ``g`` holds, so every value-level signal is
    identical and only the FIELD differs. The color legend must keep its plain
    circle swatches and the dash channel must get its own block. A fix that
    made every categorical swatch a dashed line — or one that keyed the merge
    on the domains matching rather than on the field — passes the same-field
    tests above and fails here.
    """
    chart = (
        fm.Chart(df)
        .mark_line()
        .encode(
            x="x:Q",
            y="y:Q",
            color="g:N",
            stroke_dash=fm.StrokeDash("h", sort="descending"),
        )
    )
    svg, _ = render(chart)

    rows = legend_rows(svg)
    # Two blocks, each in its OWN channel's domain order: color sorts ascending
    # by default, the dash channel was asked for descending.
    assert [label for label, _ in rows] == CATEGORIES + CATEGORIES[::-1], (
        f"expected a color block AND a dash block, each enumerating {CATEGORIES}; "
        f"got {[label for label, _ in rows]}"
    )
    color_block, dash_block = rows[:3], rows[3:]
    assert [e["__tag__"] for _, row in color_block for e in row] == ["circle"] * 3, (
        "the color legend's swatches must stay circles when nothing was folded in: "
        f"{[e['__tag__'] for _, row in color_block for e in row]}"
    )
    assert all("stroke-dasharray" not in e for _, row in color_block for e in row), (
        "an unmerged color swatch carries no dash"
    )
    assert [e["__tag__"] for _, row in dash_block for e in row] == ["line"] * 3, (
        f"{[e['__tag__'] for _, row in dash_block for e in row]}"
    )
    assert {swatch_dasharray(row) for _, row in dash_block} == set(mark_dasharrays(svg).values())


# ---------------------------------------------------------------------------
# shape folded into the color legend
# ---------------------------------------------------------------------------


def test_same_field_shape_swatches_draw_the_glyphs_their_marks_draw(df):
    """Each merged entry draws its own category's point glyph, not a third circle.

    The shape palette's first three slots are circle, square, cross; the marks
    draw them in that order over the domain (a, b, c). The swatch for each
    label must be the same glyph, which in SVG terms is a ``<circle>``, a
    ``<rect>``, and the cross's two ``<line>`` strokes. RED before the fix:
    three identical ``<circle>`` swatches.
    """
    chart = (
        fm.Chart(df)
        .mark_point()
        .encode(x="x:Q", y="y:Q", color="g:N", shape=fm.Shape("g", sort="descending"))
    )
    svg, _ = render(chart)

    rows = legend_rows(svg)
    assert [label for label, _ in rows] == CATEGORIES, (
        f"expected ONE merged legend block; got {[label for label, _ in rows]}"
    )
    glyphs = {label: sorted(e["__tag__"] for e in row) for label, row in rows}
    # `sort="descending"` runs the shape domain c, b, a, so the palette's first
    # slot (circle) lands on `c` — the by-label pin, mirroring the dash case.
    assert glyphs == {
        "c": ["circle"],
        "b": ["rect"],
        "a": ["line", "line"],
    }, glyphs


# ---------------------------------------------------------------------------
# Adjacent behavior 1 — the folded channel's title
# ---------------------------------------------------------------------------


def test_a_folded_channels_explicit_title_is_not_discarded(df):
    """``StrokeDash(..., title=)`` beside a title-less ``color`` titles the merged legend.

    The color channel states nothing, so its title is the field name ``g`` — a
    default. The dash channel states ``"Style"``. A default must not outrank a
    statement, and the statement's own block no longer exists to carry it, so
    the merged legend takes it. Measured before the fix: the legend read ``g``
    and ``"Style"`` reached nothing.
    """
    chart = (
        fm.Chart(df)
        .mark_line()
        .encode(
            x="x:Q",
            y="y:Q",
            color="g:N",
            stroke_dash=fm.StrokeDash("g", title="Style", sort="descending"),
        )
    )
    svg, msgs = render(chart)
    drawn = legend_texts(svg)
    assert "Style" in drawn, drawn
    assert "g" not in drawn, f"the field-name default must not survive the merge: {drawn}"
    assert not [m for m in msgs if "merged legend" in m], (
        f"one stated title is agreement, not conflict: {msgs}"
    )


def test_two_stated_titles_keep_colors_and_name_the_discarded_one(df):
    """Both channels state a title → color's stands and the other is reported.

    There is one title slot and two statements, so one has to go; what must
    not happen is it going silently, which is indistinguishable from the
    parameter having been ignored.
    """
    chart = (
        fm.Chart(df)
        .mark_line()
        .encode(
            x="x:Q",
            y="y:Q",
            color=fm.Color("g:N", title="Species"),
            stroke_dash=fm.StrokeDash("g", title="Style", sort="descending"),
        )
    )
    svg, msgs = render(chart)
    drawn = legend_texts(svg)
    assert "Species" in drawn and "Style" not in drawn, drawn
    assert any("Style" in m and "could not be shown" in m for m in msgs), msgs


# ---------------------------------------------------------------------------
# Adjacent behavior 2 — a disabled color legend
# ---------------------------------------------------------------------------


def test_disabling_the_color_legend_lets_the_dash_block_render(df):
    """``Color(legend=None)`` + same-field ``stroke_dash`` no longer erases both.

    Suppression exists to avoid duplicating an enumeration the color legend
    already draws. With the color legend off there is no enumeration and
    nothing to duplicate, so the dash block is the chart's only legend and
    renders on its own. Before the fix the chart had no legend at all — the
    dash channel was suppressed against a block that was never drawn.
    """
    chart = (
        fm.Chart(df)
        .mark_line()
        .encode(
            x="x:Q",
            y="y:Q",
            color=fm.Color("g:N", legend=None),
            stroke_dash=fm.StrokeDash("g", sort="descending"),
        )
    )
    svg, _ = render(chart)

    rows = legend_rows(svg)
    # The dash channel's own block, so its own descending domain order — not
    # the color legend's ascending one, which is not being drawn at all.
    assert [label for label, _ in rows] == CATEGORIES[::-1], (
        f"the dash block is the chart's only legend and must draw; got {rows}"
    )
    assert [e["__tag__"] for _, row in rows for e in row] == ["line"] * 3
    assert {label: swatch_dasharray(row) for label, row in rows}["c"] == "solid", rows


# ---------------------------------------------------------------------------
# color + shape + stroke_dash on one field — two folds, one swatch
# ---------------------------------------------------------------------------


def test_a_glyph_and_a_dash_folded_together_draw_the_glyph(df):
    """All three channels on one field: the swatches show glyphs, not dashes.

    One swatch cannot be a point glyph and a dash pattern at once, so the fold
    has to choose. The glyph wins because that is what a standalone ``shape``
    aux block already draws — a folded legend agrees with an unfolded one
    instead of drifting from it — and because the legend's glyphs are painted
    fill-only, so a dasharray on them would paint nothing on ``circle``/
    ``square`` and merely fray a 5px cross arm.

    This pins the RENDERED precedence, which is stable across the fix: the
    renderer already drew glyphs here. What the fix changed is beneath the
    SVG — the losing pattern is no longer written onto the entry (so it cannot
    reach the scene for a swatch nothing paints, where the SVG and WASM walkers
    could disagree about it) and the loss is now reported. Those two halves are
    pinned in Rust, at
    ``render::prepare::legend::tests::a_glyph_and_a_dash_folded_onto_one_entry_keep_the_glyph_and_report_the_dash``.

    The mark assertion is load-bearing, not scene-setting: it establishes that
    the chart genuinely has three distinct patterns to lose. Without it a
    future change that stopped dashing the marks entirely would leave this test
    passing while asserting nothing.
    """
    chart = (
        fm.Chart(df).mark_line().encode(x="x:Q", y="y:Q", color="g:N", shape="g", stroke_dash="g")
    )
    svg, _ = render(chart)

    assert len(set(mark_dasharrays(svg).values())) == 3, (
        "the marks must draw three distinct patterns, or there is nothing for "
        f"the legend to fail to show: {mark_dasharrays(svg)}"
    )

    rows = legend_rows(svg)
    assert [label for label, _ in rows] == CATEGORIES, (
        f"both folds land on ONE merged block; got {[label for label, _ in rows]}"
    )
    glyphs = {label: sorted(e["__tag__"] for e in row) for label, row in rows}
    assert glyphs == {
        "a": ["circle"],
        "b": ["rect"],
        # The cross glyph's two stroked arms — NOT a dash swatch, which is the
        # misreading this shape invites: counting `<line>` elements, or
        # counting `stroke-dasharray` occurrences chart-wide, cannot tell the
        # two apart.
        "c": ["line", "line"],
    }, glyphs
    assert all("stroke-dasharray" not in e for _, row in rows for e in row), (
        f"no swatch may claim a dash it does not draw: {[e for _, row in rows for e in row]}"
    )
