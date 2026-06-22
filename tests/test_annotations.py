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
