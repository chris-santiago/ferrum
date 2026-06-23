"""FA-6: hue-split violin box-inner layers must carry consistent color encoding.

The T10 fix colored the violin body and quartile/point inner layers when a hue
field is present.  The box-inner path delegates to ``desugar_boxplot``, whose
layers previously never added a ``color`` encoding — producing a visible asymmetry
where the IQR box, whiskers, caps, and median stayed a single color while the
violin body was hue-split.

These tests pin the corrected behavior:
- A hue-split violin with ``inner="box"`` → every box layer (whisker, lower_cap,
  upper_cap, box, median) carries the color encoding for the hue field.
- A violin/boxplot WITHOUT hue → box layers are unchanged (no spurious color).
- Standalone ``mark_boxplot`` without hue is byte-stable (guard test).
"""

from __future__ import annotations

import json

import polars as pl

import ferrum as fm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _spec_dict(chart: fm.Chart) -> dict:
    return json.loads(chart.to_spec().to_json())


def _transform(spec: dict, ttype: str) -> dict:
    for t in spec["transforms"]:
        if t["type"] == ttype:
            return t
    raise AssertionError(
        f"transform {ttype!r} not found in {[t['type'] for t in spec['transforms']]}"
    )


def _layers_by_mark(spec: dict, mark: str) -> list[dict]:
    return [lyr for lyr in spec["layers"] if lyr["mark"] == mark]


def _box_inner_layers(spec: dict) -> list[dict]:
    """All non-polygon layers — the box-inner layers in a violin spec."""
    return [lyr for lyr in spec["layers"] if lyr["mark"] != "polygon"]


def _make_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "g": ["a"] * 40 + ["b"] * 40,
            "h": (["x"] * 20 + ["y"] * 20) * 2,
            "v": (
                [float(i) for i in range(20)]
                + [50.0 + float(i) * 0.2 for i in range(20)]
                + [float(i) * 0.3 for i in range(20)]
                + [10.0 + float(i) * 2 for i in range(20)]
            ),
        }
    )


# ---------------------------------------------------------------------------
# FA-6 core: box-inner layers carry color when hue is present
# ---------------------------------------------------------------------------


def test_violin_hue_box_inner_layers_all_carry_color():
    """Every box-inner layer (rule/tick/rect) must have color=hue when hue is set."""
    df = _make_df()
    chart = fm.Chart(df).mark_violin(inner="box").encode(x="g:N", y="v:Q", color="h:N")
    spec = _spec_dict(chart)

    inner_layers = _box_inner_layers(spec)
    assert inner_layers, "expected box-inner layers in spec"

    for lyr in inner_layers:
        enc = lyr.get("encoding", {})
        assert "color" in enc, (
            f"box-inner layer {lyr['mark']!r} missing color encoding; "
            f"encoding keys: {list(enc.keys())}"
        )
        assert enc["color"].get("field") == "h", (
            f"box-inner layer {lyr['mark']!r} color field is {enc['color'].get('field')!r}, "
            f"expected 'h'"
        )


def test_violin_hue_whisker_layer_carries_color():
    """Specifically the whisker (rule) layer must carry the hue color encoding."""
    df = _make_df()
    chart = fm.Chart(df).mark_violin(inner="box").encode(x="g:N", y="v:Q", color="h:N")
    spec = _spec_dict(chart)

    rules = _layers_by_mark(spec, "rule")
    assert rules, "expected at least one rule (whisker) layer"
    whisker = rules[0]
    enc = whisker.get("encoding", {})
    assert "color" in enc, f"whisker rule missing color; encoding: {enc}"
    assert enc["color"].get("field") == "h"


def test_violin_hue_box_rect_layer_carries_color():
    """The IQR rect (box) layer must carry the hue color encoding."""
    df = _make_df()
    chart = fm.Chart(df).mark_violin(inner="box").encode(x="g:N", y="v:Q", color="h:N")
    spec = _spec_dict(chart)

    rects = _layers_by_mark(spec, "rect")
    assert rects, "expected at least one rect (IQR box) layer"
    box_rect = rects[0]
    enc = box_rect.get("encoding", {})
    assert "color" in enc, f"IQR rect missing color; encoding: {enc}"
    assert enc["color"].get("field") == "h"


def test_violin_hue_median_layer_carries_color():
    """The median tick must carry the hue color encoding."""
    df = _make_df()
    chart = fm.Chart(df).mark_violin(inner="box").encode(x="g:N", y="v:Q", color="h:N")
    spec = _spec_dict(chart)

    # Find the tick layer named "median" — it's the one with stroke_width=2 in mark_kwargs
    ticks = _layers_by_mark(spec, "tick")
    assert ticks, "expected tick layers"
    for tick in ticks:
        enc = tick.get("encoding", {})
        assert "color" in enc, (
            f"tick layer missing color; encoding: {enc}; mark_kwargs: {tick.get('mark_kwargs')}"
        )
        assert enc["color"].get("field") == "h"


# ---------------------------------------------------------------------------
# No-hue path: box-inner layers must NOT gain spurious color encoding
# ---------------------------------------------------------------------------


def test_violin_no_hue_box_inner_no_color():
    """Without hue the box-inner layers must have no color encoding."""
    df = _make_df()
    chart = fm.Chart(df).mark_violin(inner="box").encode(x="g:N", y="v:Q")
    spec = _spec_dict(chart)

    inner_layers = _box_inner_layers(spec)
    for lyr in inner_layers:
        enc = lyr.get("encoding", {})
        assert "color" not in enc, (
            f"box-inner layer {lyr['mark']!r} gained spurious color; encoding: {enc}"
        )


def test_violin_no_hue_box_stats_groupby_unchanged():
    """Without hue the BoxStats groupby must be [x_field] only."""
    df = _make_df()
    chart = fm.Chart(df).mark_violin(inner="box").encode(x="g:N", y="v:Q")
    spec = _spec_dict(chart)
    box = _transform(spec, "box_stats")
    assert box["groupby"] == ["g"], box["groupby"]


# ---------------------------------------------------------------------------
# Guard: standalone mark_boxplot without hue is unaffected
# ---------------------------------------------------------------------------


def test_standalone_boxplot_no_hue_no_color():
    """Standalone mark_boxplot without color= must not gain color encoding on layers."""
    df = _make_df()
    chart = fm.Chart(df).mark_boxplot().encode(x="g:N", y="v:Q")
    spec = _spec_dict(chart)

    for lyr in spec["layers"]:
        enc = lyr.get("encoding", {})
        assert "color" not in enc, (
            f"standalone boxplot layer {lyr['mark']!r} gained spurious color; enc: {enc}"
        )


def test_standalone_boxplot_with_hue_carries_color():
    """Standalone mark_boxplot WITH color= must have color encoding on all layers."""
    df = _make_df()
    chart = fm.Chart(df).mark_boxplot().encode(x="g:N", y="v:Q", color="h:N")
    spec = _spec_dict(chart)

    # When hue is present the box_stats groupby should include the hue field
    box_t = _transform(spec, "box_stats")
    assert "h" in box_t["groupby"], f"hue not in boxplot groupby: {box_t['groupby']}"

    # Every layer should carry color
    for lyr in spec["layers"]:
        enc = lyr.get("encoding", {})
        assert "color" in enc, f"boxplot-with-hue layer {lyr['mark']!r} missing color; enc: {enc}"
        assert enc["color"].get("field") == "h"


# ---------------------------------------------------------------------------
# MARK-08: boxplot/violin-body suppress the redundant color==categorical-axis
# encoding, matching the errorbar/errorband siblings (gate on split_hue).
# ---------------------------------------------------------------------------


def _layer_has_color(spec: dict, mark: str | None = None) -> bool:
    """True if any layer (optionally filtered by mark) carries a color encoding."""
    layers = spec["layers"] if mark is None else _layers_by_mark(spec, mark)
    return any("color" in lyr.get("encoding", {}) for lyr in layers)


def test_boxplot_color_equals_x_suppresses_layer_color():
    """color==x (split_hue=False) → no box layer carries a color encoding.

    When the hue column equals the categorical axis the per-layer color
    encoding is redundant with the axis, so it is suppressed.
    """
    df = _make_df()
    chart = fm.Chart(df).mark_boxplot().encode(x="g:N", y="v:Q", color="g:N")
    spec = _spec_dict(chart)

    for lyr in spec["layers"]:
        enc = lyr.get("encoding", {})
        assert "color" not in enc, (
            f"boxplot layer {lyr['mark']!r} kept a redundant color==x encoding; enc: {enc}"
        )


def test_violin_body_color_equals_x_suppresses_body_color():
    """color==x → the violin body polygon carries no color encoding."""
    df = _make_df()
    chart = fm.Chart(df).mark_violin().encode(x="g:N", y="v:Q", color="g:N")
    spec = _spec_dict(chart)

    poly_layers = _layers_by_mark(spec, "polygon")
    assert poly_layers, "expected a violin body polygon layer"
    for body in poly_layers:
        enc = body.get("encoding", {})
        assert "color" not in enc, f"violin body kept a redundant color==x encoding; enc: {enc}"


def test_boxplot_and_errorbar_agree_on_color_equals_x():
    """Sibling consistency: for color==x neither boxplot nor errorbar attach color."""
    df = _make_df()
    box = _spec_dict(fm.Chart(df).mark_boxplot().encode(x="g:N", y="v:Q", color="g:N"))
    err = _spec_dict(fm.Chart(df).mark_errorbar().encode(x="g:N", y="v:Q", color="g:N"))

    assert not _layer_has_color(box), "boxplot attached color for color==x"
    assert not _layer_has_color(err), "errorbar attached color for color==x"


def test_boxplot_and_errorbar_agree_on_color_not_equals_x():
    """Sibling consistency: for color!=x both boxplot and errorbar attach color."""
    df = _make_df()
    box = _spec_dict(fm.Chart(df).mark_boxplot().encode(x="g:N", y="v:Q", color="h:N"))
    err = _spec_dict(fm.Chart(df).mark_errorbar().encode(x="g:N", y="v:Q", color="h:N"))

    assert _layer_has_color(box), "boxplot dropped color for a distinct hue"
    assert _layer_has_color(err), "errorbar dropped color for a distinct hue"


def test_violin_quartile_color_equals_x_full_internal_consistency():
    """color==x → NEITHER the body polygon NOR the q1/median/q3 rule layers carry color.

    The quartile rules read the split "quart" BoxStats output, exactly like the
    body polygon, so both gate on split_hue. When the hue equals the categorical
    axis (split_hue=False) the per-layer color is redundant with the axis, so the
    body and all three quartile rules must agree by suppressing it — full
    intra-violin consistency, matching the errorbar/errorband siblings.
    """
    df = _make_df()
    chart = fm.Chart(df).mark_violin(inner="quartile").encode(x="g:N", y="v:Q", color="g:N")
    spec = _spec_dict(chart)

    poly_layers = _layers_by_mark(spec, "polygon")
    assert poly_layers, "expected a violin body polygon layer"
    for body in poly_layers:
        enc = body.get("encoding", {})
        assert "color" not in enc, f"violin body kept a redundant color==x encoding; enc: {enc}"

    rules = _layers_by_mark(spec, "rule")
    assert rules, "expected q1/median/q3 quartile rule layers"
    for rule in rules:
        enc = rule.get("encoding", {})
        assert "color" not in enc, f"quartile rule kept a redundant color==x encoding; enc: {enc}"


# ---------------------------------------------------------------------------
# Consistency with existing T10 tests: quartile inner and body still work
# ---------------------------------------------------------------------------


def test_violin_hue_quartile_body_and_quartile_layers_consistent():
    """Quartile inner: body AND quartile rule layers should both carry color."""
    df = _make_df()
    chart = fm.Chart(df).mark_violin(inner="quartile").encode(x="g:N", y="v:Q", color="h:N")
    spec = _spec_dict(chart)

    # Body (polygon) layer
    poly_layers = _layers_by_mark(spec, "polygon")
    assert poly_layers
    body_enc = poly_layers[0].get("encoding", {})
    assert "color" in body_enc, f"violin body missing color; enc: {body_enc}"

    # Rule (quartile) layers
    rules = _layers_by_mark(spec, "rule")
    assert rules
    for rule in rules:
        enc = rule.get("encoding", {})
        assert "color" in enc, f"quartile rule missing color; enc: {enc}"
