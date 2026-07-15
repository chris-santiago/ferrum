import polars as pl
import pytest

from ferrum import Chart
from ferrum.annotations import annotate_hline, annotate_vline, annotate_rect, annotate_text


def test_annotate_hline_returns_chart_with_rule_mark():
    h = annotate_hline(0)
    assert h._mark == "rule"


def test_annotate_vline_returns_chart_with_rule_mark():
    v = annotate_vline(5)
    assert v._mark == "rule"


def test_annotate_rect_returns_chart_with_rect_mark():
    r = annotate_rect(0, 1, 0, 1, opacity=0.1)
    assert r._mark == "rect"


def test_annotate_text_returns_chart_with_text_mark():
    t = annotate_text(1.0, 2.0, "hi")
    assert t._mark == "text"


def test_annotate_rect_encodes_x2_y2():
    """BUG-2 regression: annotate_rect must encode x2 and y2 channels."""
    r = annotate_rect(1.0, 3.0, 2.0, 4.0)
    enc = r._encoding
    assert "x2" in enc, "annotate_rect must encode x2"
    assert "y2" in enc, "annotate_rect must encode y2"


def test_annotate_text_encodes_text_channel():
    """BUG-3 regression: annotate_text must encode the text channel."""
    t = annotate_text(1.0, 2.0, "hello")
    enc = t._encoding
    assert "text" in enc, "annotate_text must encode the text channel"


def test_annotate_hline_can_be_added_to_scatter():
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    scatter = Chart(df).mark_point().encode(x="a", y="b")
    # + always layers now — annotate_hline's different data is auto null-pad
    # merged into the scatter's DataFrame.
    composed = scatter + annotate_hline(5)
    assert composed is not None
    assert composed._layers is not None


# ---------------------------------------------------------------------------
# GH #82 — annotate_hline / annotate_vline label= was a documented no-op.
#
# The label must render both standalone (``.to_svg()``) AND when the
# reference line is layered onto another chart with ``+`` — the primary use
# case. ``Chart.__add__``'s annotate_* dispatch (chart.py) discards the
# helper chart's mark layers and reads only ``_annotation_primitive``, so a
# labelled line carries BOTH the line and its label through
# ``_annotation_primitive`` as a list (``ferrum.annotations.
# _attach_line_with_label`` / ``ferrum.chart._annotations_from_primitive``).
# A labelled line therefore stays a plain ``Chart`` (never a VConcatChart)
# so it remains ``+``-overlay compatible, matching every other annotate_*
# helper's return shape.
# ---------------------------------------------------------------------------


def test_annotate_hline_label_renders_in_svg():
    svg = annotate_hline(y=0.0, label="baseline").to_svg()
    assert "NaN" not in svg
    assert ">baseline<" in svg


def test_annotate_vline_label_renders_in_svg():
    svg = annotate_vline(x=1.0, label="cutoff").to_svg()
    assert "NaN" not in svg
    assert ">cutoff<" in svg


def test_annotate_hline_without_label_returns_plain_chart():
    """No label → unchanged return shape (a Chart, not a composite)."""
    h = annotate_hline(0)
    assert isinstance(h, Chart)
    assert h._mark == "rule"


def test_annotate_vline_without_label_returns_plain_chart():
    v = annotate_vline(5)
    assert isinstance(v, Chart)
    assert v._mark == "rule"


def test_annotate_hline_with_label_still_returns_plain_chart():
    """A labelled reference line MUST remain a plain Chart — not a composite
    like VConcatChart — so it stays overlay-compatible with `+`."""
    h = annotate_hline(0, label="baseline")
    assert isinstance(h, Chart)
    assert h._mark == "rule"


def test_annotate_vline_with_label_still_returns_plain_chart():
    v = annotate_vline(5, label="cutoff")
    assert isinstance(v, Chart)
    assert v._mark == "rule"


def test_annotate_hline_label_survives_plus_overlay():
    """Primary use case: `chart + annotate_hline(y, label=...)` must not
    raise, and the resulting SVG must contain both the label text and the
    reference rule (a line element)."""
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    scatter = Chart(df).mark_point().encode(x="a", y="b")
    composed = scatter + annotate_hline(y=0, label="baseline")
    svg = composed.to_svg()
    assert ">baseline<" in svg
    assert "<line" in svg


def test_annotate_vline_label_survives_plus_overlay():
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    scatter = Chart(df).mark_point().encode(x="a", y="b")
    composed = scatter + annotate_vline(x=1, label="cutoff")
    svg = composed.to_svg()
    assert ">cutoff<" in svg
    assert "<line" in svg


def test_annotate_hline_unlabeled_plus_overlay_annotation_shape_unaffected():
    """Unlabelled `chart + annotate_hline(y)` must still produce exactly one
    annotation entry (the reference line, type "line") — the label=None path
    is unaffected by the `_annotation_primitive`-list generalization added
    for the labelled case."""
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    scatter = Chart(df).mark_point().encode(x="a", y="b")
    composed = scatter + annotate_hline(5)
    ann_dicts = [d for annotate in composed._annotations for d in annotate.to_dict_list()]
    assert len(ann_dicts) == 1
    assert ann_dicts[0]["type"] == "line"


def test_annotate_vline_unlabeled_plus_overlay_annotation_shape_unaffected():
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    scatter = Chart(df).mark_point().encode(x="a", y="b")
    composed = scatter + annotate_vline(1)
    ann_dicts = [d for annotate in composed._annotations for d in annotate.to_dict_list()]
    assert len(ann_dicts) == 1
    assert ann_dicts[0]["type"] == "line"


def test_both_labeled_helpers_overlay_does_not_double_render_lhs_label():
    """S4 regression: `annotate_hline(label=) + annotate_vline(label=)` — the
    BOTH-operands-are-annotate-helpers branch of Chart.__add__ — must not
    double-render the LHS label.  `new = lhs._clone()` carries the labelled
    LHS's dormant `_annotations` (its own label Annotate); that branch must
    rebuild `_annotations` from the two primitive expansions only, not add
    to the carried-over value, or the LHS label is emitted twice."""
    composed = annotate_hline(y=0, label="base") + annotate_vline(x=5, label="cut")
    ann_dicts = [d for annotate in composed._annotations for d in annotate.to_dict_list()]
    assert len(ann_dicts) == 4  # 2 lines + 2 labels
    svg = composed.to_svg()
    assert svg.count(">base<") == 1
    assert svg.count(">cut<") == 1


def test_both_unlabeled_helpers_overlay_annotation_count_byte_identity_anchor():
    """Unlabelled both-helper overlay stays at exactly 2 entries (1 line
    each) — the byte-identity anchor for the fix above: before the fix this
    was `[] + [line_h] + [line_v]`, after it's `[line_h] + [line_v]`, which
    is identical when the dropped prefix (`new._annotations`) is empty."""
    composed = annotate_hline(0) + annotate_vline(5)
    ann_dicts = [d for annotate in composed._annotations for d in annotate.to_dict_list()]
    assert len(ann_dicts) == 2
    assert {d["type"] for d in ann_dicts} == {"line"}


def test_annotations_from_primitive_single_primitive_matches_prior_wrapping():
    """`_annotations_from_primitive` for a single (non-list) primitive must
    produce the exact prior `[Annotate(primitive)]` wrapping — the
    byte-identical guarantee for every existing single-primitive annotate_*
    caller (annotate_rect/abline/text/arrow, and unlabelled hline/vline)."""
    from ferrum.annotation.container import Annotate
    from ferrum.annotation.coords import norm
    from ferrum.annotation.primitives import AnnotationLine
    from ferrum.chart import _annotations_from_primitive

    prim = AnnotationLine(
        x1=norm(0), y1=0.0, x2=norm(1), y2=0.0, stroke="#333", stroke_width=1, dash=None
    )
    result = _annotations_from_primitive(prim)
    assert len(result) == 1
    assert isinstance(result[0], Annotate)
    assert result[0].items == [prim]


# ---------------------------------------------------------------------------
# COMP-06 — annotate_text anchor/align vocabulary reconciliation
#
# `anchor` is the canonical SVG keyword (start/middle/end, matching
# annotation.text); `align` is the back-compat alias (left/center/right).
# These guard the additive `anchor=` kwarg, the alias equivalence, the
# behavior-preserving default, and the double-supply error.
# ---------------------------------------------------------------------------


def test_annotate_text_anchor_sets_primitive_anchor():
    """anchor='start' (SVG vocab) must reach the primitive anchor directly."""
    t = annotate_text(1.0, 2.0, "hi", anchor="start")
    assert t._annotation_primitive.anchor == "start"
    # The mark speaks the left/center/right vocab; anchor='start' maps to 'left'.
    assert t._mark_kwargs.get("align") == "left"


def test_annotate_text_align_alias_matches_anchor():
    """align='left' (alias) must yield the same output as anchor='start'."""
    by_align = annotate_text(1.0, 2.0, "hi", align="left")
    by_anchor = annotate_text(1.0, 2.0, "hi", anchor="start")
    assert by_align._annotation_primitive.anchor == by_anchor._annotation_primitive.anchor
    assert by_align._annotation_primitive.anchor == "start"
    assert by_align._mark_kwargs.get("align") == by_anchor._mark_kwargs.get("align")
    assert by_align._mark_kwargs.get("align") == "left"


def test_annotate_text_align_center_maps_to_middle():
    """align='center' (alias) maps to anchor 'middle' (the historical mapping)."""
    t = annotate_text(1.0, 2.0, "hi", align="center")
    assert t._annotation_primitive.anchor == "middle"


def test_annotate_text_default_anchor_is_middle():
    """Neither anchor nor align supplied → anchor 'middle' (behavior-preserving).

    This is the old `align='center'` default; the rendered output must be
    identical to supplying align='center' explicitly.
    """
    default = annotate_text(1.0, 2.0, "hi")
    explicit = annotate_text(1.0, 2.0, "hi", align="center")
    assert default._annotation_primitive.anchor == "middle"
    assert default._annotation_primitive.anchor == explicit._annotation_primitive.anchor
    assert default._mark_kwargs.get("align") == explicit._mark_kwargs.get("align")
    assert default._mark_kwargs.get("align") == "center"


def test_annotate_text_both_anchor_and_align_raises():
    """Supplying both the canonical anchor and the alias align raises ValueError."""
    with pytest.raises(ValueError, match="both 'anchor' and 'align'"):
        annotate_text(1.0, 2.0, "hi", anchor="start", align="left")


# ---------------------------------------------------------------------------
# COMP-01 — coordinate coercion is single-sourced in annotation/coords.py
#
# The annotations-path (annotate_*) and the primitives-path (to_dict via
# _coord) must agree on one classification.  These import the coercion from
# its single home and assert both paths see the same result for each input
# kind (datetime / ISO-string / ordinal / PixelCoord / NormCoord).
# ---------------------------------------------------------------------------


def test_coerce_coord_single_home_importable():
    """The coercion functions live in (and import from) annotation.coords."""
    from ferrum.annotation.coords import _coerce_coord, _coerce_coord_to_numeric, _coord

    # primitives.py re-exports _coord from coords.py — same object, not a copy.
    from ferrum.annotation import primitives

    assert primitives._coord is _coord
    # annotations.py imports the coercion from coords.py — same object.
    from ferrum import annotations as _ann

    assert _ann._coerce_coord is _coerce_coord
    assert _ann._coerce_coord_to_numeric is _coerce_coord_to_numeric


def test_coerce_and_serialize_agree_across_inputs():
    """annotations-path coercion + primitives-path serialization agree on one classification."""
    import datetime as dt

    from ferrum.annotation.coords import (
        OrdinalCategoryCoord,
        _coerce_coord,
        _coord,
        temporal_coord_to_epoch_ms,
    )
    from ferrum.annotation.coords import px, norm

    # datetime → epoch-ms float, identical from both directions.
    d = dt.date(2020, 6, 1)
    assert _coerce_coord(d) == temporal_coord_to_epoch_ms(d)
    assert _coord(_coerce_coord(d)) == temporal_coord_to_epoch_ms(d)

    # ISO-8601 string → epoch-ms float.
    iso = "2020-06-01"
    assert _coerce_coord(iso) == temporal_coord_to_epoch_ms(iso)
    assert _coord(_coerce_coord(iso)) == temporal_coord_to_epoch_ms(iso)

    # Non-ISO string → OrdinalCategoryCoord → {"category": ...}.
    coerced = _coerce_coord("cat_a")
    assert isinstance(coerced, OrdinalCategoryCoord)
    assert _coord(coerced) == {"category": "cat_a"}

    # PixelCoord / NormCoord pass through coercion and serialize to px/norm.
    assert _coerce_coord(px(50)) == px(50)
    assert _coord(_coerce_coord(px(50))) == {"px": 50}
    assert _coerce_coord(norm(0.5)) == norm(0.5)
    assert _coord(_coerce_coord(norm(0.5))) == {"norm": 0.5}
