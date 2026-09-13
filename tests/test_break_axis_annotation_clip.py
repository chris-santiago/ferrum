"""End-to-end coverage for the #228 stratified annotation clip (wave-2 T7).

An annotation with a data-space extent on a broken axis is no longer dropped
whole the moment a gap touches it. What happens instead depends on what the
geometry MEANS, which is the whole of the #228 contract (spec §3.12's
2026-09-12 note, design spec §4):

* ``rect`` and ``span`` clip into one piece per visible segment, silently,
  exactly as the equivalent MARK geometry (a bar of the same extent) always
  did — and a ``span``'s label anchors on the midpoint of the largest
  surviving piece rather than on the pre-clip midpoint, which for a straddling
  span lies inside the gap;
* ``line`` and ``arrow`` shafts split per visible piece through the same
  vertex splitter ``mark_line``/``mark_segment`` use, with an arrow's head
  riding the piece that reaches its tip;
* ``bracket`` stays suppress-whole PERMANENTLY (its endpoint tips admit no
  honest split);
* the point-anchored kinds (``text``, ``callout``, ``image``) keep the plain
  membership test, and a callout's background rect — a text box, not data — is
  never clipped;
* suppression (with the existing per-axis rollup warning) is now the narrower
  fact it should always have been: nothing of the annotation survives, or its
  kind admits no split.

This module carries the net-new behavior (including the Span mechanism, which
had no Python coverage at all before this batch). ``test_break_axis_geometry.py``
keeps the older suppression/remap pins.
"""

from __future__ import annotations

import re
import warnings

import polars as pl
import pytest

import ferrum as fm
import ferrum.annotation as ann

# One chart shape for every case: three bars on an ordinal x axis, with the y
# scale PINNED to [0, 1000] so a gap of (150, 850) is hand-checkable and the
# pixel arithmetic below does not move with the data.
_DF = pl.DataFrame({"category": ["A", "B", "C"], "value": [72.0, 900.0, 500.0]})
_WIDE_Y = fm.Y("value:Q", scale=fm.LinearScale(domain=[0.0, 1000.0]))

# The break's indicator band, in pixels, for the (150, 850) gap on that chart:
# the plot's y range 16.0…420.961 loses `break_size` (12) to the indicator and
# splits the rest evenly, so the two segments are 196.48px each and the band
# between them runs 212.48…224.48.
_BAND = (212.48, 224.48)
_SEGMENT_PX = 196.48


def _render(*extras: object) -> tuple[str, list[str]]:
    """Render the shared chart plus `extras`, returning the SVG and warnings."""
    chart = fm.Chart(_DF).mark_bar().encode(x="category:N", y=_WIDE_Y)
    for extra in extras:
        chart = chart + extra
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        svg = chart.to_svg()
    return svg, [str(w.message) for w in caught]


def _bands(svg: str, fill: str) -> list[tuple[float, float]]:
    """`(y, height)` of every `<rect>` painted in `fill`, in paint order."""
    pattern = (
        r'<rect x="[-\d.]+" y="([-\d.]+)" width="[-\d.]+" height="([-\d.]+)"[^>]*' + re.escape(fill)
    )
    return [(float(m.group(1)), float(m.group(2))) for m in re.finditer(pattern, svg)]


def _strokes(svg: str, color: str) -> list[tuple[float, float, float, float]]:
    """`(x1, y1, x2, y2)` of every `<line>` stroked in `color`, in paint order."""
    pattern = (
        r'<line x1="([-\d.]+)" y1="([-\d.]+)" x2="([-\d.]+)" y2="([-\d.]+)" stroke="'
        + re.escape(color)
        + '"'
    )
    return [tuple(float(g) for g in m.groups()) for m in re.finditer(pattern, svg)]


def _head_tip_and_size(svg: str, color: str) -> tuple[tuple[float, float], float]:
    """An arrowhead's tip and its drawn size (tip → base midpoint distance)."""
    paths = re.findall(
        r'<path d="M([-\d.]+) ([-\d.]+) L([-\d.]+) ([-\d.]+) L([-\d.]+) ([-\d.]+) Z" fill="'
        + re.escape(color),
        svg,
    )
    assert len(paths) == 1, f"expected exactly one arrowhead in {color}; got {paths}"
    tx, ty, lx, ly, rx, ry = (float(v) for v in paths[0])
    base = ((lx + rx) / 2.0, (ly + ry) / 2.0)
    return (tx, ty), ((tx - base[0]) ** 2 + (ty - base[1]) ** 2) ** 0.5


def _approx(got: list[tuple[float, float]], want: list[tuple[float, float]]) -> None:
    assert got == pytest.approx(want, abs=1e-3), f"got {got}, want {want}"


# --------------------------------------------------------------------------
# rect: the intent-ledger headline
# --------------------------------------------------------------------------


def test_straddling_rect_clips_into_one_piece_per_visible_segment():
    """`annotate_rect(y1=0, y2=1000)` under `gap=(150, 850)` renders two pieces.

    Pre-#228 this rect vanished with an `AnnotationSpansBreakGap` warning while
    a BAR of the same extent clipped per segment — one chart answering the same
    question two ways. The clip is silent, exactly as the mark clip is.
    """
    rect = fm.Annotate(ann.rect(x1=fm.norm(0), y1=0.0, x2=fm.norm(1), y2=1000.0, fill="#ffcc00"))
    svg, messages = _render(fm.BreakAxis(axis="y", gap=(150, 850)), rect)

    _approx(_bands(svg, "#ffcc00"), [(224.48, _SEGMENT_PX), (16.0, _SEGMENT_PX)])
    assert not messages, f"a successful clip is silent; got {messages}"

    # The pieces stop AT the gap edges: the band between them is the indicator's
    # own 12px, and together they cover the unbroken rect minus that band.
    unbroken_svg, _ = _render(rect)
    [(_, unbroken_h)] = _bands(unbroken_svg, "#ffcc00")
    assert _SEGMENT_PX * 2 == pytest.approx(unbroken_h - 12.0, abs=1e-2)


def test_rect_with_an_in_gap_endpoint_clips_to_its_visible_portion():
    """The retired endpoint-first shortcut: an in-gap endpoint no longer drops.

    `annotate_rect(y1=100, y2=400)` has its upper bound inside the gap, which
    pre-#228 suppressed the whole rect. Its visible 100–150 portion now renders,
    cut at the gap edge (224.48, the lower segment's top pixel).
    """
    rect = fm.Annotate(ann.rect(x1=fm.norm(0), y1=100.0, x2=fm.norm(1), y2=400.0, fill="#ffcc00"))
    svg, messages = _render(fm.BreakAxis(axis="y", gap=(150, 850)), rect)

    _approx(_bands(svg, "#ffcc00"), [(224.48, 65.493)])
    assert not messages, f"clipping is silent; got {messages}"


def test_rect_wholly_inside_a_gap_is_suppressed_and_rolled_up():
    """Nothing survives → dropped whole, with the existing per-axis rollup."""
    rect = fm.Annotate(ann.rect(x1=fm.norm(0), y1=300.0, x2=fm.norm(1), y2=700.0, fill="#ffcc00"))
    svg, messages = _render(fm.BreakAxis(axis="y", gap=(150, 850)), rect)

    assert _bands(svg, "#ffcc00") == [], "an invisible rect paints nothing"
    assert messages == [
        "1 annotation element (rect) on the y axis fell inside a BreakAxis gap and was suppressed"
    ]


def test_straddling_rect_with_corner_radius_squares_only_the_cut_edge():
    """#230-2, the annotation half: `ann.rect(corner_radius=...)` straddling a
    gap inherits the mark-side fix, since both route through the same
    `clip_nodes` -> `clip_node_into` machinery.

    The rect renders as two `<path>` pieces (never falling back to a plain
    `<rect rx ry>`), each rounding only the corners its own uncut edges gave
    it.
    """
    rect = fm.Annotate(
        ann.rect(x1=fm.norm(0), y1=0.0, x2=fm.norm(1), y2=1000.0, fill="#ffcc00", corner_radius=8.0)
    )
    svg, messages = _render(fm.BreakAxis(axis="y", gap=(150, 850)), rect)
    assert not messages, f"a successful clip is silent; got {messages}"
    assert 'rx="' not in svg, "a cut rounded annotation rect must never fall back to a plain rect"

    paths = re.findall(r'<path d="([^"]*)" fill="#ffcc00"', svg)
    assert len(paths) == 2, f"one path per visible segment; got {paths}"
    for d in paths:
        arcs = re.findall(r"A([\d.]+) ([\d.]+) [\d.]+ \d \d [-\d.]+ [-\d.]+", d)
        assert len(arcs) == 2, f"only the two uncut corners round; got {d}"
        for rx, ry in arcs:
            assert float(rx) == pytest.approx(8.0) and float(ry) == pytest.approx(8.0), (
                f"an uncut corner keeps the full requested radius; got {d}"
            )


def test_rect_with_a_pixel_bound_drops_whole_when_its_data_bound_is_in_the_gap():
    """`AnnotationSuppressedByBreak`'s third population (quality-review cycle 2).

    `rect` routes `Cut`, but a bound that is pixel-space rather than data (here
    ``y2=fm.px(...)``) means the broken axis has only ONE data bound to test —
    there is no interval left to clip. An in-gap value there drops the whole
    rect exactly as a point-anchored kind's coordinate would, even though
    `rect` itself is a clip-routed kind.
    """
    rect = fm.Annotate(
        ann.rect(x1=fm.norm(0), y1=500.0, x2=fm.norm(1), y2=fm.px(50.0), fill="#ffcc00")
    )
    svg, messages = _render(fm.BreakAxis(axis="y", gap=(150, 850)), rect)

    assert _bands(svg, "#ffcc00") == [], "an unreadable interval cannot be clipped"
    assert messages == [
        "1 annotation element (rect) on the y axis fell inside a BreakAxis gap and was suppressed"
    ]


# --------------------------------------------------------------------------
# span: the mechanism with no end-to-end coverage before this batch
# --------------------------------------------------------------------------


def test_span_clips_and_its_label_rides_the_largest_surviving_piece():
    """A span's band clips, and its label anchors on the largest piece.

    Gap `(200, 600)` splits the band unevenly (130.987px below, 261.974px
    above), so "largest" is discriminating: the label sits at the UPPER piece's
    midpoint (16 + 261.974/2), not the first piece's and not the pre-clip
    band's.
    """
    span = fm.Annotate(
        ann.span("y", 0.0, 1000.0, fill="#eeeeee", label="band", label_position="middle")
    )
    svg, messages = _render(fm.BreakAxis(axis="y", gap=(200, 600)), span)

    _approx(_bands(svg, "#eeeeee"), [(289.974, 130.987), (16.0, 261.974)])
    assert not messages, f"clipping is silent; got {messages}"

    labels = [
        float(m.group(1))
        for m in re.finditer(r'<text x="[-\d.]+" y="([-\d.]+)"[^>]*>band</text>', svg)
    ]
    assert labels == pytest.approx([146.987], abs=1e-3), (
        f"one label, on the larger (upper) piece's midpoint; got {labels}"
    )


def test_a_clipped_spans_label_never_lands_in_the_gap_band():
    """The same rule's sharp edge: the label is never drawn in the gap.

    With the gap centred on the axis the two pieces are equal, the label takes
    the first of them, and the pre-clip midpoint it would otherwise have used
    (the middle of the full 16…420.961 band) falls inside the indicator band.
    """
    span = fm.Annotate(
        ann.span("y", 0.0, 1000.0, fill="#eeeeee", label="band", label_position="middle")
    )
    svg, _ = _render(fm.BreakAxis(axis="y", gap=(150, 850)), span)

    [label_y] = [
        float(m.group(1))
        for m in re.finditer(r'<text x="[-\d.]+" y="([-\d.]+)"[^>]*>band</text>', svg)
    ]
    assert label_y == pytest.approx(322.721, abs=1e-3), (
        f"first of two equal pieces: 224.48 + 196.48/2; got {label_y}"
    )
    assert not _BAND[0] <= label_y <= _BAND[1], (
        f"the label must never sit in the invisible gap band {_BAND}; got {label_y}"
    )


# The Span mechanism's DROP case (a span with nothing outside the gap) is
# pinned by `test_break_axis_geometry.py::test_span_inside_a_gap_is_suppressed`,
# rewritten in this batch for the new trigger — kept there rather than copied
# here, so one behavior has one pin.


# --------------------------------------------------------------------------
# line / arrow: the vertex splitter, shared with mark geometry
# --------------------------------------------------------------------------


def test_annotation_line_splits_at_the_gap_edges():
    """A diagonal annotation line becomes one sub-line per visible segment.

    Each piece ends ON the gap edge at the x it actually crosses (the crossing's
    cross-axis coordinate is interpolated), so the stroke never spans the break
    band and never spikes toward the hidden sentinel.
    """
    line = fm.Annotate(ann.line(fm.norm(0.0), 0.0, fm.norm(1.0), 1000.0, stroke="#00aa00"))
    svg, messages = _render(fm.BreakAxis(axis="y", gap=(150, 850)), line)

    pieces = _strokes(svg, "#00aa00")
    assert len(pieces) == 2, f"one piece per visible segment; got {pieces}"
    assert pieces[0][3] == pytest.approx(_BAND[1], abs=1e-3), "first piece ends at the gap edge"
    assert pieces[1][1] == pytest.approx(_BAND[0], abs=1e-3), "second starts at the far edge"
    assert pieces[0][2] < pieces[1][0], (
        f"the crossing x is interpolated along the line, so the pieces do not "
        f"share an x; got {pieces}"
    )
    assert not messages, f"splitting is silent; got {messages}"


def test_arrow_shaft_splits_and_its_head_rides_the_tip_piece():
    """An arrow's shaft splits; the head follows the piece that reaches the tip.

    The head keeps its drawn size (`head_size=12`) and is rebuilt on the tip
    piece's own direction, so it points along the piece a reader sees it
    attached to.
    """
    arrow = fm.Annotate(
        ann.arrow(fm.norm(0.1), 0.0, fm.norm(0.9), 1000.0, stroke="#0000ff", head_size=12)
    )
    svg, messages = _render(fm.BreakAxis(axis="y", gap=(150, 850)), arrow)

    shafts = _strokes(svg, "#0000ff")
    assert len(shafts) == 2, f"the shaft splits in two; got {shafts}"
    tip, size = _head_tip_and_size(svg, "#0000ff")
    assert tip == pytest.approx(shafts[1][2:], abs=1e-3), (
        "the head sits on the tip piece's end, not the first piece's"
    )
    assert size == pytest.approx(12.0, abs=1e-3), f"drawn size preserved; got {size}"
    assert not messages, f"splitting is silent; got {messages}"


def test_arrow_wholly_inside_a_gap_is_suppressed_and_warns():
    """No piece of the shaft survives, so there is no head either."""
    arrow = fm.Annotate(ann.arrow(fm.norm(0.1), 300.0, fm.norm(0.9), 700.0, stroke="#0000ff"))
    svg, messages = _render(fm.BreakAxis(axis="y", gap=(150, 850)), arrow)

    assert "#0000ff" not in svg
    assert messages == [
        "1 annotation element (arrow) on the y axis fell inside a BreakAxis gap and was suppressed"
    ]


def test_arrow_with_an_in_gap_tip_drops_only_the_head():
    """A visible tail with an in-gap TIP: only the head drops, silently.

    Orchestrator adjudication (quality-review cycle 2): the CODE behavior
    stands — a tip inside a gap leaves no piece to terminate the head on, so
    the head alone drops, the same fate `mark_segment(arrow=True)`'s shaft
    gives an in-gap tip (#230-1). The shaft's own visible piece still renders,
    truncated at the gap edge it ends on. This is clip-fate geometry (a
    faithful rendering of what is visible), not a silent drop of the
    annotation — the ferrum-spec.md note that called this "the whole arrow
    drops" was wrong, not the code.
    """
    arrow = fm.Annotate(
        ann.arrow(fm.norm(0.1), 0.0, fm.norm(0.9), 500.0, stroke="#0000ff", head_size=12)
    )
    svg, messages = _render(fm.BreakAxis(axis="y", gap=(150, 850)), arrow)

    shafts = _strokes(svg, "#0000ff")
    assert len(shafts) == 1, f"the visible tail survives as one shaft piece; got {shafts}"
    assert shafts[0][3] == pytest.approx(_BAND[1], abs=1e-3), "the piece ends at the gap edge"

    heads = re.findall(
        r'<path d="M[-\d.]+ [-\d.]+ L[-\d.]+ [-\d.]+ L[-\d.]+ [-\d.]+ Z" fill="#0000ff"', svg
    )
    assert heads == [], f"the head drops with its hidden tip; got {heads}"
    assert not messages, f"the drop is silent, same as mark_segment(arrow=True)'s; got {messages}"


# --------------------------------------------------------------------------
# bracket: suppress-whole, permanently
# --------------------------------------------------------------------------


def test_bracket_straddling_a_gap_is_suppressed_whole_permanently():
    """`bracket` is the ONE extent kind that stays suppress-whole (spec §4, D1).

    Its endpoint tips are what say where the bracketed range begins and ends, so
    cutting the baseline at a gap edge would either strand them at positions the
    data never had or invent new tips at the cut. This is the permanent contract,
    not a residual: a rect of the same extent clips (see above), a bracket does
    not.
    """
    bracket = fm.Annotate(ann.bracket(fm.norm(0.2), 100.0, fm.norm(0.8), 900.0, label="range"))
    svg, messages = _render(fm.BreakAxis(axis="y", gap=(150, 850)), bracket)

    assert "range" not in svg, "the bracket and its label are dropped together"
    assert messages == [
        "1 annotation element (bracket) on the y axis spans a BreakAxis gap and was suppressed"
    ]


# --------------------------------------------------------------------------
# point-anchored kinds: unchanged, and the callout box is never clipped
# --------------------------------------------------------------------------


def test_callout_box_is_never_clipped_even_across_the_gap_band():
    """A callout's background rect is a text box — drawn size, never data.

    The anchor at y=132 places its 22px box at 207.058…229.058 — straight across
    the 212.48…224.48 indicator band, so a clip would have cut it into two
    pieces. It stays ONE rect at full height: the clip's reach is decided by what
    the geometry encodes, not by where it happens to land.
    """
    callout = fm.Annotate(ann.callout(fm.norm(0.5), 132.0, text="note", background="#ff00ff"))
    svg, messages = _render(fm.BreakAxis(axis="y", gap=(150, 850)), callout)

    boxes = _bands(svg, "#ff00ff")
    assert len(boxes) == 1, f"the box is never split; got {boxes}"
    assert boxes[0][1] == pytest.approx(22.0), "and never resized"
    assert boxes[0][0] < _BAND[0] < _BAND[1] < boxes[0][0] + boxes[0][1], (
        f"this box must actually straddle the band for the test to bite; got {boxes}"
    )
    assert not messages


# --------------------------------------------------------------------------
# desugar fan-out and byte-stability
# --------------------------------------------------------------------------


def test_hline_and_its_label_share_one_fate_under_a_y_break():
    """`annotate_hline(label=...)` emits a `line` plus a `text` at the same y.

    Both are at one y value, so on a y break the pair shares one fate: inside
    the gap they are dropped together and roll into ONE warning naming both
    primitive kinds; outside it both render. There is nothing to clip here — a
    zero-length extent is either in a gap or not.
    """
    dropped_svg, dropped_messages = _render(
        fm.BreakAxis(axis="y", gap=(150, 850)), fm.annotate_hline(y=500.0, label="inside")
    )
    assert "inside" not in dropped_svg
    assert dropped_messages == [
        "2 annotation elements (line, text) on the y axis fell inside a "
        "BreakAxis gap and were suppressed"
    ]

    kept_svg, kept_messages = _render(
        fm.BreakAxis(axis="y", gap=(150, 850)), fm.annotate_hline(y=100.0, label="below")
    )
    assert "below" in kept_svg, "a visible hline keeps its label"
    assert not kept_messages


def test_annotations_under_an_ineffective_gap_stay_byte_identical():
    """A break that resolves to no effective gap changes no annotation byte.

    The C1 contract, carried by an annotation the clip WOULD have cut had the
    gap landed: routing is reached only through a live remap, so the
    no-effective-break render is the plain build's, byte for byte.
    """
    rect = fm.Annotate(ann.rect(x1=fm.norm(0), y1=0.0, x2=fm.norm(1), y2=1000.0, fill="#ffcc00"))
    plain_svg, plain_messages = _render(rect)
    ineffective_svg, ineffective_messages = _render(fm.BreakAxis(axis="y", gap=(5000, 6000)), rect)

    assert ineffective_svg == plain_svg
    assert not plain_messages
    assert ineffective_messages == [
        "BreakAxis on the y axis had no effect: no gap overlaps the axis domain (0, 1000)"
    ]
