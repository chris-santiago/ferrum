"""Pins spec §4.4 / §6.5's annotation-contrast guarantee for the five figure
functions whose cells draw a per-cell ink instead of one fixed theme ink:
``heatmap(annot=True)``, ``confusion_matrix_chart``, ``rank2d_chart``,
``classification_report_chart`` and ``class_prediction_error_chart``'s
stacked-bar count labels.

Each of these figures' text/annotation layer binds its own ``color`` channel
to the same field its cell (rect or bar segment) draws, and carries the
internal ``"theme:contrast"`` fill sentinel
(``crates/ferrum-core/src/render/draw.rs``), so every annotated cell's label
reads at least 4.5:1 WCAG 2.x contrast against its own rendered cell, on
every theme and color ramp -- UNLESS a caller painted the label itself (a
figure's ``mark={<label>: {"fill": ...}}``, a composite mark's catch-all
``fill=``, or ``fill="none"``/the ``"color"`` alias), in which case the
caller's paint wins and the label draws no contrast ink at all (spec §6.5,
§4.4's "text with explicit fill= ... unchanged"). This module asserts both
outcomes from the rendered SVG using an independent WCAG implementation
(``tests/_wcag.py``), not by replaying ferrum's own contrast-resolution
logic -- a test built that way would just restate the implementation as its
own oracle.
"""

from __future__ import annotations

import json

import polars as pl
import pytest

import ferrum as fm
from ferrum import themes
from tests._hue_probe import legend_labels
from tests._svg_rects import figure_elements
from tests._wcag import (
    contrast_ratio as _contrast_ratio,
    opaque_rgb as _opaque_rgb,
    parse_paint as _parse_paint,
)

_WCAG_AA = 4.5


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    r, g, b, _ = _parse_paint(value)
    return (r, g, b)


def _background_rgb(theme_obj) -> tuple[int, int, int]:
    return _hex_to_rgb(theme_obj.to_spec_dict()["background_color"])


def _font_rgb(theme_obj) -> tuple[int, int, int]:
    return _hex_to_rgb(theme_obj.to_spec_dict()["font_color"])


def _font_hex(theme_obj) -> str:
    return theme_obj.to_spec_dict()["font_color"].lower()


# ---------------------------------------------------------------------------
# Themes under test -- spec §4.4's named set.
# ---------------------------------------------------------------------------

_THEMES = {
    "paper_ink": themes.paper_ink,
    "slate_citrus": themes.slate_citrus,
    "arctic_signal": themes.arctic_signal,
}


def _cell_label_pairs(
    svg: str, background_rgb: tuple[int, int, int], *, y_tolerance: float = 2.5
) -> list[tuple[tuple[int, int, int], str, str]]:
    """Pair each data cell (rect or bar segment) with the text centered on it.

    Matched purely by geometry (``figure_elements`` resolves every coordinate
    into one shared figure space): a stacked bar's count label is centered on
    its segment's midpoint (``anchor="mid"``), which needs a slightly looser
    y tolerance than a plain rect+text heatmap pairing, hence the parameter.
    A cell with no matching label (a null/zero-count segment ferrum draws no
    text for) is simply omitted, not an error.
    """
    elements = figure_elements(svg)
    rects = [
        e
        for e in elements
        if e["tag"] == "rect"
        and e.get("fill")
        and e["panel"] is not None
        and not e["fill"].startswith("url(")
    ]
    texts = [e for e in elements if e["tag"] == "text"]
    pairs: list[tuple[tuple[int, int, int], str, str]] = []
    for r in rects:
        cx, cy = r["x"] + r["w"] / 2, r["y"] + r["h"] / 2
        match = next(
            (t for t in texts if abs(t["x"] - cx) < 0.5 and abs(t["y"] - cy) < y_tolerance),
            None,
        )
        if match is not None:
            pairs.append((_opaque_rgb(r["fill"], background_rgb), match["fill"], match["text"]))
    return pairs


def _assert_all_pairs_meet_aa(
    pairs: list[tuple[tuple[int, int, int], str, str]],
    background_rgb: tuple[int, int, int],
    *,
    context: str,
) -> None:
    assert pairs, f"{context}: no cell/label pairs matched -- the geometry probe found nothing"
    for cell_rgb, ink, text in pairs:
        ink_rgb = _opaque_rgb(ink, background_rgb)
        ratio = _contrast_ratio(ink_rgb, cell_rgb)
        assert ratio >= _WCAG_AA, (
            f"{context}: label {text!r} ink {ink!r} ({ink_rgb}) against cell {cell_rgb} "
            f"only reaches {ratio:.2f}:1, below WCAG AA {_WCAG_AA}:1"
        )


# ---------------------------------------------------------------------------
# Fixtures -- small, deterministic datasets with enough value spread to
# exercise both light and dark cells under every scheme.
# ---------------------------------------------------------------------------


def _heatmap_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": ["r0", "r1", "r2", "r3"],
            "m0": [-10.0, -3.0, 2.0, 9.0],
            "m1": [-6.0, 1.0, 5.0, -9.0],
            "m2": [10.0, -8.0, 0.0, 4.0],
        }
    )


def _rank2d_df() -> pl.DataFrame:
    features = ["a", "b", "c"]
    correlations = {
        ("a", "a"): 1.0,
        ("a", "b"): 0.2,
        ("a", "c"): -0.9,
        ("b", "a"): 0.2,
        ("b", "b"): 1.0,
        ("b", "c"): 0.45,
        ("c", "a"): -0.9,
        ("c", "b"): 0.45,
        ("c", "c"): 1.0,
    }
    rows = [
        {
            "feature_x": fx,
            "feature_y": fy,
            "correlation": correlations[(fx, fy)],
            "correlation_fmt": f"{correlations[(fx, fy)]:.2f}",
        }
        for fx in features
        for fy in features
    ]
    return pl.DataFrame(rows)


def _rank2d_wide_df() -> pl.DataFrame:
    """A plain wide numeric frame for ``rank2d_chart``, which computes its own correlations."""
    return pl.DataFrame(
        {"f0": [1.0, 2.0, 3.0, 4.0], "f1": [4.0, 3.0, 2.0, 1.0], "f2": [2.0, 2.0, 3.0, 1.0]}
    )


#: Precomputed classification labels -- varied enough that confusion / report
#: / class-prediction-error cells span both the low and high ends of each
#: color domain (near-0, mid, and near-1 proportions; a mix of correct and
#: confused predictions).
_Y_TRUE = ["a"] * 10 + ["b"] * 10 + ["c"] * 10
_Y_PRED = (["a"] * 9 + ["b"] * 1) + (["b"] * 5 + ["c"] * 5) + (["a"] * 1 + ["c"] * 9)


# ---------------------------------------------------------------------------
# Case 1: heatmap(annot=True) -- rdbu and the theme ramp, every theme.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("theme_name", list(_THEMES))
@pytest.mark.parametrize(
    ("cmap", "center"), [(None, None), ("rdbu", 0.0)], ids=["theme_ramp", "rdbu"]
)
def test_heatmap_annotation_contrast_meets_wcag_aa(theme_name, cmap, center) -> None:
    theme_obj = _THEMES[theme_name]
    chart = fm.heatmap(_heatmap_df(), annot=True, cmap=cmap, center=center, theme=theme_obj)
    svg = chart.to_svg()
    pairs = _cell_label_pairs(svg, _background_rgb(theme_obj), y_tolerance=0.5)
    _assert_all_pairs_meet_aa(
        pairs, _background_rgb(theme_obj), context=f"heatmap/{theme_name}/{cmap}"
    )


# ---------------------------------------------------------------------------
# Case 2: confusion_matrix_chart -- continuous proportion lookup, every theme.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("theme_name", list(_THEMES))
def test_confusion_matrix_annotation_contrast_meets_wcag_aa(theme_name) -> None:
    theme_obj = _THEMES[theme_name]
    chart = fm.confusion_matrix_chart(y_true=_Y_TRUE, y_pred=_Y_PRED, theme=theme_obj)
    svg = chart.to_svg()
    pairs = _cell_label_pairs(svg, _background_rgb(theme_obj), y_tolerance=0.5)
    _assert_all_pairs_meet_aa(pairs, _background_rgb(theme_obj), context=f"confusion/{theme_name}")


# ---------------------------------------------------------------------------
# Case 3: rank2d_chart -- diverging correlation lookup, every theme and ramp.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("theme_name", list(_THEMES))
@pytest.mark.parametrize("cmap", [None, "rdbu"], ids=["theme_ramp", "rdbu"])
def test_rank2d_annotation_contrast_meets_wcag_aa(theme_name, cmap) -> None:
    theme_obj = _THEMES[theme_name]
    # `rank2d_chart` (the figure wrapper) does not expose `cmap=`; the
    # composite mark it desugars to (`Chart.mark_rank2d`) does, and is the
    # same `desugar_rank2d` code path the figure resolves to.
    chart = fm.Chart(_rank2d_df()).mark_rank2d(cmap=cmap).theme(theme_obj)
    svg = chart.to_svg()
    pairs = _cell_label_pairs(svg, _background_rgb(theme_obj), y_tolerance=0.5)
    _assert_all_pairs_meet_aa(
        pairs, _background_rgb(theme_obj), context=f"rank2d/{theme_name}/{cmap}"
    )


# ---------------------------------------------------------------------------
# Case 4: classification_report_chart -- continuous metric lookup, every theme.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("theme_name", list(_THEMES))
def test_classification_report_annotation_contrast_meets_wcag_aa(theme_name) -> None:
    theme_obj = _THEMES[theme_name]
    chart = fm.classification_report_chart(y_true=_Y_TRUE, y_pred=_Y_PRED, theme=theme_obj)
    svg = chart.to_svg()
    pairs = _cell_label_pairs(svg, _background_rgb(theme_obj), y_tolerance=0.5)
    _assert_all_pairs_meet_aa(
        pairs, _background_rgb(theme_obj), context=f"classification_report/{theme_name}"
    )


# ---------------------------------------------------------------------------
# Case 5: class_prediction_error_chart -- CATEGORICAL lookup, every theme.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("theme_name", list(_THEMES))
def test_class_prediction_error_count_label_contrast_meets_wcag_aa(theme_name) -> None:
    theme_obj = _THEMES[theme_name]
    chart = fm.class_prediction_error_chart(y_true=_Y_TRUE, y_pred=_Y_PRED, theme=theme_obj)
    # The categorical lookup: the bar's (and label's) `color` binds `predicted`,
    # a nominal field, not a continuous domain -- pin that it is read that way.
    spec = chart.to_spec()
    wire = json.loads(spec.to_json())
    bar_layer = next(ly for ly in wire["layers"] if ly["mark"] == "bar")
    assert bar_layer["encoding"]["color"].get("type") == "nominal", (
        "class_prediction_error's color binding must be the categorical (nominal) lookup, "
        "not an inferred continuous scale"
    )

    svg = chart.to_svg()
    pairs = _cell_label_pairs(svg, _background_rgb(theme_obj), y_tolerance=2.5)
    _assert_all_pairs_meet_aa(
        pairs, _background_rgb(theme_obj), context=f"class_prediction_error/{theme_name}"
    )


# ---------------------------------------------------------------------------
# Negative control: prove the oracle discriminates. A single constant ink
# (the theme's own font color, the pre-fix behavior) must FAIL against at
# least one of these real rendered cells, while the sentinel-driven ink
# (asserted above) passes the same cell. This is read off the actual
# rendered cell color, never a re-derivation of ferrum's own ink logic.
# ---------------------------------------------------------------------------


def test_a_single_constant_ink_would_fail_where_the_sentinel_passes() -> None:
    theme_obj = _THEMES["slate_citrus"]
    background_rgb = _background_rgb(theme_obj)
    font_rgb = _font_rgb(theme_obj)
    chart = fm.heatmap(_heatmap_df(), annot=True, theme=theme_obj)
    svg = chart.to_svg()
    pairs = _cell_label_pairs(svg, background_rgb, y_tolerance=0.5)
    assert pairs, "setup failure: no cell/label pairs found"

    would_fail = [
        (cell_rgb, text)
        for cell_rgb, _, text in pairs
        if _contrast_ratio(font_rgb, cell_rgb) < _WCAG_AA
    ]
    assert would_fail, (
        "test setup must include at least one cell where the theme's constant font ink "
        "fails WCAG AA -- otherwise this fixture cannot discriminate the fix from the defect"
    )
    # And the actual (sentinel-driven) rendered ink for that same cell clears AA.
    failing_cell_rgb, failing_text = would_fail[0]
    actual_ink = next(ink for cell_rgb, ink, text in pairs if text == failing_text)
    actual_ink_rgb = _opaque_rgb(actual_ink, background_rgb)
    assert _contrast_ratio(actual_ink_rgb, failing_cell_rgb) >= _WCAG_AA, (
        "the sentinel-driven ink must clear AA on a cell where the constant font ink fails"
    )
    assert actual_ink_rgb != font_rgb, (
        "the sentinel must not have drawn the same constant ink that fails this cell"
    )


# ---------------------------------------------------------------------------
# The contrast binding's "binds color, draws nothing" rule (spec §6.5): a
# label's own color binding resolves correctly, warns nothing, and adds no
# second scale on an overlay (`+`) route and a faceted route, reached here
# through `mark_confusion` composed with another layer and through
# `.facet(...)`.
# ---------------------------------------------------------------------------


def _manual_confusion_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "actual": ["a", "a", "a", "b", "b", "b", "c", "c", "c"],
            "predicted": ["a", "b", "c", "a", "b", "c", "a", "b", "c"],
            "value": [0.95, 0.03, 0.02, 0.4, 0.5, 0.1, 0.05, 0.05, 0.9],
            "value_fmt": ["0.95", "0.03", "0.02", "0.40", "0.50", "0.10", "0.05", "0.05", "0.90"],
            "fold": ["f0"] * 9,
        }
    )


def test_contrast_label_resolves_on_an_overlay_route_without_a_second_scale() -> None:
    import warnings

    df = _manual_confusion_df()
    base = fm.Chart(df).mark_confusion()
    marker = (
        fm.Chart(df.unique(["actual", "predicted"])).mark_point().encode(x="predicted", y="actual")
    )
    chart = base + marker
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        svg = chart.to_svg()
    channel_warnings = [str(w.message) for w in caught if "ChannelNotDrawn" in str(w.message)]
    assert not channel_warnings, (
        "the label's own (non-drawing) color binding must not warn ChannelNotDrawn when the "
        f"cell layer already draws color; got {channel_warnings}"
    )
    pairs = _cell_label_pairs(svg, _background_rgb(_THEMES["paper_ink"]), y_tolerance=0.5)
    _assert_all_pairs_meet_aa(
        pairs, _background_rgb(_THEMES["paper_ink"]), context="confusion-overlay"
    )
    assert svg.count("<linearGradient") == 1, (
        "the label's own color binding must add no second colorbar on the overlay route"
    )


def test_contrast_label_resolves_on_a_faceted_route_per_panel() -> None:
    import warnings

    df = _manual_confusion_df()
    extra = df.with_columns(pl.lit("f1").alias("fold"))
    both_folds = pl.concat([df, extra])
    chart = fm.Chart(both_folds).mark_confusion().facet("fold")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        svg = chart.to_svg()
    channel_warnings = [str(w.message) for w in caught if "ChannelNotDrawn" in str(w.message)]
    assert not channel_warnings, (
        "the label's own (non-drawing) color binding must not warn ChannelNotDrawn on a "
        f"faceted route when the cell layer already draws color; got {channel_warnings}"
    )
    pairs = _cell_label_pairs(svg, _background_rgb(_THEMES["paper_ink"]), y_tolerance=0.5)
    _assert_all_pairs_meet_aa(
        pairs, _background_rgb(_THEMES["paper_ink"]), context="confusion-faceted"
    )
    # Two facet panels' worth of cells, not just one panel's.
    assert len(pairs) >= 2 * 9 - 4, (
        f"expected cell/label pairs from both facet panels, got {len(pairs)}"
    )


# ---------------------------------------------------------------------------
# Wire shape: each of the five text/annotation layers binds its own `color`
# to the same field its cell draws, and carries the sentinel.
# ---------------------------------------------------------------------------


def _label_layer(wire: dict, name: str) -> dict:
    return next(ly for ly in wire["layers"] if ly.get("name") == name)


@pytest.mark.parametrize(
    ("build_chart", "label_name", "color_field"),
    [
        (lambda: fm.heatmap(_heatmap_df(), annot=True), "label", "value"),
        (lambda: fm.confusion_matrix_chart(y_true=_Y_TRUE, y_pred=_Y_PRED), "label", "value"),
        (lambda: fm.Chart(_rank2d_df()).mark_rank2d(), "label", "correlation"),
        (
            lambda: fm.classification_report_chart(y_true=_Y_TRUE, y_pred=_Y_PRED),
            "annotation",
            "value",
        ),
        (
            lambda: fm.class_prediction_error_chart(y_true=_Y_TRUE, y_pred=_Y_PRED),
            "label",
            "predicted",
        ),
    ],
    ids=["heatmap", "confusion", "rank2d", "classification_report", "class_prediction_error"],
)
def test_each_label_layer_binds_its_own_color_and_carries_the_sentinel(
    build_chart, label_name, color_field
) -> None:
    chart = build_chart()
    wire = json.loads(chart.to_spec().to_json())
    layer = _label_layer(wire, label_name)
    color_binding = layer["encoding"].get("color")
    assert color_binding is not None, (
        f"{label_name} layer must bind its own color channel to the cell field {color_field!r}"
    )
    assert color_binding["field"] == color_field, (
        f"{label_name} layer must bind its own color to the cell field {color_field!r}"
    )
    assert layer.get("mark_style", {}).get("fill") == "theme:contrast", (
        f"{label_name} layer must carry the theme:contrast fill sentinel"
    )


# ---------------------------------------------------------------------------
# Legends unchanged: the label layer's own (non-drawing) color binding adds
# no second colorbar / no doubled categorical swatch set.
# ---------------------------------------------------------------------------


def test_heatmap_annotation_adds_no_second_colorbar() -> None:
    svg = fm.heatmap(_heatmap_df(), annot=True).to_svg()
    assert svg.count("<linearGradient") == 1


def test_confusion_matrix_annotation_adds_no_second_colorbar() -> None:
    svg = fm.confusion_matrix_chart(y_true=_Y_TRUE, y_pred=_Y_PRED).to_svg()
    assert svg.count("<linearGradient") == 1


def test_rank2d_annotation_adds_no_second_colorbar() -> None:
    svg = fm.Chart(_rank2d_df()).mark_rank2d().to_svg()
    assert svg.count("<linearGradient") == 1


def test_classification_report_annotation_adds_no_second_colorbar() -> None:
    svg = fm.classification_report_chart(y_true=_Y_TRUE, y_pred=_Y_PRED).to_svg()
    assert svg.count("<linearGradient") == 1


def test_class_prediction_error_label_adds_no_second_legend() -> None:
    svg = fm.class_prediction_error_chart(y_true=_Y_TRUE, y_pred=_Y_PRED).to_svg()
    # Three predicted classes -> three swatches, not six (one set, not one
    # per color-binding layer).
    assert sorted(set(legend_labels(svg))) == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# Unchanged: clustermap (no annotation layer at all).
# ---------------------------------------------------------------------------


def test_clustermap_carries_no_contrast_sentinel() -> None:
    df = pl.DataFrame(
        {
            "id": ["r0", "r1", "r2"],
            "m0": [1.0, 2.0, 3.0],
            "m1": [3.0, 2.0, 1.0],
            "m2": [2.0, 2.0, 2.0],
        }
    )
    chart = fm.clustermap(df)
    svg = chart.to_svg()
    assert "theme:contrast" not in svg


# ---------------------------------------------------------------------------
# Unchanged: a plain user rect+text chart (no bound color on the text layer)
# still draws the constant theme font ink; explicit fill=/fill="none" still
# wins exactly as before. None of these routes carry the sentinel or an own
# color binding, so this batch cannot have touched them.
# ---------------------------------------------------------------------------


def _plain_matrix_df() -> pl.DataFrame:
    return pl.DataFrame(
        {"col": ["q0", "q0", "q1"], "row": ["p0", "p1", "p0"], "v": [1.0, 2.0, 3.0]}
    )


def test_user_rect_plus_text_with_no_own_color_draws_the_constant_theme_ink() -> None:
    theme_obj = themes.slate_citrus
    df = _plain_matrix_df()
    chart = (
        fm.Chart(df)
        .mark_rect()
        .encode(x="col", y="row", color="v")
        .layer(fm.Layer(mark="text", encoding={"x": "col", "y": "row", "text": "v"}))
        .theme(theme_obj)
    )
    svg = chart.to_svg()
    label_fills = {
        t["fill"]
        for t in figure_elements(svg)
        if t["tag"] == "text" and t["text"] in {"1", "2", "3"}
    }
    # Every label fill is the one constant (theme font color); no per-row
    # variation, because this text layer binds no color of its own.
    assert label_fills == {_font_hex(theme_obj)}


def test_explicit_literal_fill_on_a_plain_text_layer_still_wins() -> None:
    df = _plain_matrix_df()
    chart = fm.Chart(df).mark_text(fill="#112233").encode(x="col", y="row", text="v")
    svg = chart.to_svg()
    assert "#112233" in svg


def test_explicit_fill_none_on_a_plain_text_layer_still_clears() -> None:
    df = _plain_matrix_df()
    chart = fm.Chart(df).mark_text(fill="none").encode(x="col", y="row", text="v")
    svg = chart.to_svg()
    label_fills = {
        t["fill"]
        for t in figure_elements(svg)
        if t["tag"] == "text" and t["text"] in {"1", "2", "3"}
    }
    assert label_fills == {"none"}


# ---------------------------------------------------------------------------
# A caller fill must win over the contrast ink, on every one of the five
# label layers, through every door that can paint one: the figure's
# `mark={<label>: {...}}` override (a literal fill, the "color" alias, and
# `fill="none"`) and the composite mark's catch-all `fill=`. The label's own
# `color` binding exists only to pick a contrast ink when nothing else
# painted the layer; once a caller does, that binding must never draw a hue
# of its own (spec §6.5, §4.4's "text with explicit fill= ... unchanged").
# ---------------------------------------------------------------------------

_FILL_HEX = "#112233"

#: (figure name, a builder taking one `mark=`-style dict for the label layer,
#: the label layer's own name) for each of the five sites.
_LABEL_SITES = [
    (
        "heatmap",
        lambda style: fm.heatmap(_heatmap_df(), annot=True, mark={"label": style}),
        "label",
    ),
    (
        "confusion",
        lambda style: fm.confusion_matrix_chart(
            y_true=_Y_TRUE, y_pred=_Y_PRED, mark={"label": style}
        ),
        "label",
    ),
    (
        "rank2d",
        lambda style: fm.rank2d_chart(_rank2d_wide_df(), mark={"label": style}),
        "label",
    ),
    (
        "classification_report",
        lambda style: fm.classification_report_chart(
            y_true=_Y_TRUE, y_pred=_Y_PRED, mark={"annotation": style}
        ),
        "annotation",
    ),
    (
        "class_prediction_error",
        lambda style: fm.class_prediction_error_chart(
            y_true=_Y_TRUE, y_pred=_Y_PRED, mark={"label": style}
        ),
        "label",
    ),
]

_LABEL_SITE_IDS = [name for name, _, _ in _LABEL_SITES]


def _label_inks(svg: str, *, y_tolerance: float = 2.5) -> list[str]:
    """The raw (unconverted) ``fill`` of every text geometrically centered on a data cell.

    Like :func:`_cell_label_pairs`, but returns the SVG fill attribute text
    as-is rather than resolving it to an opaque RGB triple -- needed for a
    non-color paint such as ``"none"``, which :func:`_opaque_rgb` cannot
    parse (by design: it is not a color).
    """
    elements = figure_elements(svg)
    rects = [
        e
        for e in elements
        if e["tag"] == "rect"
        and e.get("fill")
        and e["panel"] is not None
        and not e["fill"].startswith("url(")
    ]
    texts = [e for e in elements if e["tag"] == "text"]
    inks = []
    for r in rects:
        cx, cy = r["x"] + r["w"] / 2, r["y"] + r["h"] / 2
        match = next(
            (t for t in texts if abs(t["x"] - cx) < 0.5 and abs(t["y"] - cy) < y_tolerance),
            None,
        )
        if match is not None:
            inks.append(match["fill"])
    return inks


@pytest.mark.parametrize(
    ("fig_name", "build_chart", "label_name"), _LABEL_SITES, ids=_LABEL_SITE_IDS
)
def test_caller_literal_fill_wins_over_the_contrast_ink(fig_name, build_chart, label_name) -> None:
    chart = build_chart({"fill": _FILL_HEX})
    inks = _label_inks(chart.to_svg())
    assert inks, f"{fig_name}: no labels matched"
    assert set(inks) == {_FILL_HEX}, (
        f"{fig_name}: a caller fill= on the {label_name!r} layer must paint every label "
        f"{_FILL_HEX}, not a per-cell contrast ink; got {set(inks)}"
    )


@pytest.mark.parametrize(
    ("fig_name", "build_chart", "label_name"), _LABEL_SITES, ids=_LABEL_SITE_IDS
)
def test_caller_color_alias_wins_over_the_contrast_ink(fig_name, build_chart, label_name) -> None:
    chart = build_chart({"color": _FILL_HEX})
    inks = _label_inks(chart.to_svg())
    assert inks, f"{fig_name}: no labels matched"
    assert set(inks) == {_FILL_HEX}, (
        f"{fig_name}: mark={{{label_name!r}: {{'color': ...}}}} (the fill alias) on the "
        f"{label_name!r} layer must paint every label {_FILL_HEX}; got {set(inks)}"
    )


@pytest.mark.parametrize(
    ("fig_name", "build_chart", "label_name"), _LABEL_SITES, ids=_LABEL_SITE_IDS
)
def test_fill_none_drops_the_labels_color_binding(fig_name, build_chart, label_name) -> None:
    chart = build_chart({"fill": "none"})
    wire = json.loads(chart.to_spec().to_json())
    layer = _label_layer(wire, label_name)
    assert "color" not in layer["encoding"], (
        f"{fig_name}: fill='none' on the {label_name!r} layer must drop its color binding "
        f"entirely (no contrast reading once the caller cleared the paint); "
        f"got encoding={layer['encoding']!r}"
    )
    inks = _label_inks(chart.to_svg())
    assert inks, f"{fig_name}: no labels matched"
    assert set(inks) == {"none"}, f"{fig_name}: fill='none' must still clear every label"


def test_catch_all_fill_on_a_composite_mark_wins_over_the_contrast_ink() -> None:
    """``mark_confusion(fill=...)``: the catch-all style door, not the figure's `mark=`."""
    df = _manual_confusion_df()
    chart = fm.Chart(df).mark_confusion(fill=_FILL_HEX)
    inks = _label_inks(chart.to_svg())
    assert inks, "no labels matched"
    assert set(inks) == {_FILL_HEX}, (
        f"mark_confusion(fill=...) must paint every label {_FILL_HEX}, not a per-cell "
        f"contrast ink; got {set(inks)}"
    )


# ---------------------------------------------------------------------------
# heatmap's own **encode_kwargs (opacity=, size=, ...) bind the cells only;
# the label is a `+` sibling, not a `chart.layer(...)` host-dependent, so it
# must not inherit them.
# ---------------------------------------------------------------------------


def test_heatmap_opacity_encode_kwarg_stays_off_the_label() -> None:
    chart = fm.heatmap(_heatmap_df(), annot=True, opacity="value")
    inks = _label_inks(chart.to_svg())
    assert inks, "no labels matched"
    for ink in inks:
        assert ink.startswith("#"), (
            f"opacity='value' must not reach the label layer (full-ink hex expected); got {ink!r}"
        )


def test_heatmap_size_encode_kwarg_stays_off_the_label_and_warns() -> None:
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        chart = fm.heatmap(_heatmap_df(), annot=True, size="value")
        svg = chart.to_svg()
    messages = [str(w.message) for w in caught]
    assert any("size" in m and "has no effect" in m for m in messages), (
        f"size='value' on the cells (a rect mark, which draws no size) must warn "
        f"'has no effect' exactly as it did before the label had its own color binding; "
        f"got {messages}"
    )
    assert 'font-size="2"' not in svg, (
        "size='value' must not reach the label layer (it must stay at the default font size)"
    )


# ---------------------------------------------------------------------------
# heatmap(annot=True, color=<field>): the label must read the SAME binding
# the cells draw (choice 3), not the literal "value" field, so a caller's
# color= override still clears WCAG AA against the hue it actually paints.
# ---------------------------------------------------------------------------


def test_heatmap_color_override_annotation_contrast_meets_wcag_aa() -> None:
    theme_obj = _THEMES["paper_ink"]
    df = pl.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "x": [1.0, 2.0, 3.0, 4.0],
            "y": [3.0, -1.0, 2.0, 0.5],
        }
    )
    chart = fm.heatmap(df, annot=True, color="id", theme=theme_obj)
    svg = chart.to_svg()
    pairs = _cell_label_pairs(svg, _background_rgb(theme_obj), y_tolerance=0.5)
    _assert_all_pairs_meet_aa(pairs, _background_rgb(theme_obj), context="heatmap/color=id")


# ---------------------------------------------------------------------------
# The mask= (melted_data) branch shares the same construction as the default
# branch (choice 2: one rows chart, reused for both the cells and the
# label), so it must isolate the cells' own **encode_kwargs from the label
# exactly as the default branch does.
# ---------------------------------------------------------------------------


def test_heatmap_mask_with_opacity_keeps_the_label_at_full_ink() -> None:
    df = _heatmap_df()
    chart = fm.heatmap(df, annot=True, mask="upper", opacity="value")
    inks = _label_inks(chart.to_svg())
    assert inks, "no labels matched"
    for ink in inks:
        assert ink.startswith("#"), (
            f"opacity='value' must not reach the label layer on the mask= path either "
            f"(full-ink hex expected); got {ink!r}"
        )
