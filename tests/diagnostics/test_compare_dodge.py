"""Dodge-by-model ``compare=`` layout (GH #42).

For dodge-eligible diagnostics, a ``compare=`` call renders a single
shared-axis panel with marks grouped (dodged) by model instead of the #35
small-multiples panels. This module holds the discriminating contract tests:
single ``Chart`` (not ``ConcatChart``), one bar per (feature, model) at
distinct sub-band positions, a model legend, both orientations, and dodged
error rules / value labels. Each assertion fails on the pre-#42
small-multiples layout.

Task 3 covers ``importance_chart``; ``shap_bar_chart`` (Task 4) and
``cv_scores_chart`` (Task 5, ``kind="box"``/``"strip"`` only -- ``kind="bar"``
keeps the small-multiples layout per spec D3) extend this file as they land.
"""

from __future__ import annotations

import re
from collections import Counter

import polars as pl

import ferrum
from ferrum.composition import ConcatChart
from ferrum.coord import CoordFlip
from ferrum.position import Dodge
from tests.fixtures import load_dataset, load_fixture


_REG_FEATURES = ["f0", "f1", "f2", "f3", "f4"]


def _reg_setup():
    df = load_dataset("regression")
    X = df.select(_REG_FEATURES)
    y = df["y"]
    base = load_fixture("regression_ridge")
    alt = load_fixture("regression_rf")
    return X, y, base, alt


def _bar_rects(svg: str) -> list[tuple[float, float, str]]:
    """Return ``(x, width, fill)`` for every categorical-palette bar rect.

    Bars carry one of the (few) categorical fills, each appearing once per
    feature band; the single full-width panel background carries a fill that
    appears exactly once. Filtering to fills that occur more than once keeps
    the bars and drops the background, theme-independently.
    """
    rects = re.findall(r"<rect[^>]*>", svg)

    def attr(r: str, a: str) -> str | None:
        m = re.search(a + r'="([^"]+)"', r)
        return m.group(1) if m else None

    hexes = [
        (attr(r, "x"), attr(r, "width"), attr(r, "fill"))
        for r in rects
        if attr(r, "fill") and re.fullmatch(r"#[0-9a-fA-F]{6}", attr(r, "fill") or "")
    ]
    fill_counts = Counter(f for _, _, f in hexes)
    return [
        (float(x), float(w), f)
        for x, w, f in hexes
        if x is not None and w is not None and fill_counts[f] > 1
    ]


def _bar_geoms(svg: str) -> list[tuple[float, float, float, float, str]]:
    """Return ``(x, y, width, height, fill)`` for every palette bar rect.

    Same background-vs-bar discrimination as :func:`_bar_rects`, but keeps the
    full geometry so band-axis (y) positions can be checked under CoordFlip.
    """
    rects = re.findall(r"<rect[^>]*>", svg)

    def attr(r: str, a: str) -> str | None:
        m = re.search(a + r'="([^"]+)"', r)
        return m.group(1) if m else None

    hexes = [
        (attr(r, "x"), attr(r, "y"), attr(r, "width"), attr(r, "height"), attr(r, "fill"))
        for r in rects
        if attr(r, "fill") and re.fullmatch(r"#[0-9a-fA-F]{6}", attr(r, "fill") or "")
    ]
    fill_counts = Counter(h[-1] for h in hexes)
    return [
        (float(x), float(y), float(w), float(h), f)
        for x, y, w, h, f in hexes
        if None not in (x, y, w, h) and fill_counts[f] > 1
    ]


def _error_rule_x_positions(svg: str) -> list[float]:
    """Return the x1 (== x2) position of every dashed error-rule ``<line>``.

    The importance error-rule layer always renders as a dashed
    ``stroke="#9ca3af"`` ``stroke-dasharray="4,4"`` line regardless of theme
    (verified against ``tests/goldens/phase_10/importance_chart_*.svg``),
    distinguishing it from axis ticks/gridlines (solid, different colors) and
    from the dodged bars (filled rects).
    """
    lines = re.findall(r"<line[^>]*>", svg)

    def attr(ln: str, a: str) -> str | None:
        m = re.search(a + r'="([^"]+)"', ln)
        return m.group(1) if m else None

    positions: list[float] = []
    for ln in lines:
        if attr(ln, "stroke") == "#9ca3af" and attr(ln, "stroke-dasharray") == "4,4":
            x1 = attr(ln, "x1")
            if x1 is not None:
                positions.append(float(x1))
    return positions


def _non_background_rect_geoms(svg: str) -> list[tuple[float, float, float, float, str]]:
    """Return ``(x, y, width, height, fill)`` for every palette rect, excluding
    the full-canvas background.

    ``_bar_rects``/``_bar_geoms`` discriminate bars from the background by fill
    frequency (a fill used more than once); that heuristic degenerates to zero
    matches when a ``split=`` filter leaves only one box per model (each fill
    then appears exactly once). This helper instead excludes only the
    known-position full-canvas background rect (rendered at ``x="0" y="0"``),
    so it stays correct regardless of how many bars share a fill.
    """
    rects = re.findall(r"<rect[^>]*>", svg)

    def attr(r: str, a: str) -> str | None:
        m = re.search(a + r'="([^"]+)"', r)
        return m.group(1) if m else None

    out: list[tuple[float, float, float, float, str]] = []
    for r in rects:
        fill = attr(r, "fill")
        x, y, w, h = attr(r, "x"), attr(r, "y"), attr(r, "width"), attr(r, "height")
        if not (fill and x and y and w and h):
            continue
        if not re.fullmatch(r"#[0-9a-fA-F]{6}", fill):
            continue
        if x == "0" and y == "0":
            continue  # full-canvas background rect
        out.append((float(x), float(y), float(w), float(h), fill))
    return out


def _point_geoms(svg: str) -> list[tuple[float, float, str]]:
    """Return ``(cx, cy, fill)`` for every data point-mark circle.

    Point marks and legend swatches both render as ``<circle>`` elements
    (unlike bar/box rects, whose legend swatches are always circles too --
    see ``_bar_rects``'s docstring), so fill-count discrimination doesn't
    separate them here: a data point's fill also appears on its legend
    swatch. Legend swatches instead carry a fixed ``r="4"`` regardless of
    the mark's own point size (observed on every colour-legend chart in
    this suite), distinct from a data point's size-derived radius -- for
    ``cv_scores_chart``'s default point size that is ``r="3.385"``.
    Filtering out ``r="4"`` isolates the actual data marks.
    """
    circles = re.findall(r"<circle[^>]*/>", svg)

    def attr(c: str, a: str) -> str | None:
        m = re.search(a + r'="([^"]+)"', c)
        return m.group(1) if m else None

    out: list[tuple[float, float, str]] = []
    for c in circles:
        fill = attr(c, "fill")
        r = attr(c, "r")
        cx = attr(c, "cx")
        cy = attr(c, "cy")
        if fill and r != "4" and re.fullmatch(r"#[0-9a-fA-F]{6}", fill) and cx and cy:
            out.append((float(cx), float(cy), fill))
    return out


def _text_labels(svg: str) -> list[tuple[float, str]]:
    """Return ``(x, text)`` for every ``<text>`` element carrying an x anchor."""
    out: list[tuple[float, str]] = []
    for m in re.finditer(r"<text[^>]*x=\"([^\"]+)\"[^>]*>([^<]*)</text>", svg):
        out.append((float(m.group(1)), m.group(2)))
    return out


# ---------------------------------------------------------------------------
# importance_chart — dodge-by-model contract
# ---------------------------------------------------------------------------


def test_importance_compare_returns_single_chart():
    """compare= returns a single Chart, not a small-multiples ConcatChart."""
    X, y, base, alt = _reg_setup()
    result = ferrum.importance_chart(base, X, y, compare={"alt": alt})
    from ferrum.chart import Chart

    assert isinstance(result, Chart)
    assert not isinstance(result, ConcatChart)
    assert "<svg" in result.to_svg()


def test_importance_compare_data_is_stacked_by_model():
    """The combined frame carries a model column with one row per (feature, model)."""
    X, y, base, alt = _reg_setup()
    result = ferrum.importance_chart(base, X, y, top_k=3, compare={"alt": alt})
    data = result._data
    assert "model" in data.columns
    # Registration order: base first, alt second, repeated per feature band.
    assert data["model"].to_list() == ["base", "alt"] * 3
    assert data.height == 2 * 3  # n_models * top_k


def test_importance_compare_global_top_k_shared_feature_set():
    """Exactly top_k distinct features, shared by every model (spec D4/§6)."""
    X, y, base, alt = _reg_setup()
    result = ferrum.importance_chart(base, X, y, top_k=3, compare={"alt": alt})
    data = result._data
    features = data["feature"].unique().to_list()
    assert len(features) == 3
    # Every model exposes the same feature set.
    for model in ("base", "alt"):
        model_feats = set(data.filter(pl.col("model") == model)["feature"].to_list())
        assert model_feats == set(features)


def test_importance_compare_ranks_by_absolute_magnitude():
    """Global feature ranking must be magnitude-based, not signed-mean.

    RED-proof: a signed ``mean(importance)`` ranking silently drops a
    large-magnitude *negative* importance (e.g. from
    ``method="permutation"``) in favor of small positive ones, because a
    large negative value sorts to the bottom of a descending signed sort.
    ``neg_big`` (|~0.85|, dominant magnitude) must survive a top_k=2 cut
    over ``{neg_big, pos_small, pos_mid}`` alongside ``pos_mid`` (the next
    largest magnitude); ``pos_small`` (~0.045) must be dropped.
    """
    from ferrum.plots.explanation import _importance_chart_compare_from_source

    def _frame(rows: dict[str, float]) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "feature": list(rows.keys()),
                "importance": list(rows.values()),
                "std": [0.0] * len(rows),
                "rank": list(range(1, len(rows) + 1)),
            }
        )

    class _FakeModelSource:
        def __init__(self, rows: dict[str, float]):
            self._df = _frame(rows)

        def importances(self, *, method, random_state):
            del method, random_state
            return self._df

    class _FakeComparedSource:
        def __init__(self, sources: dict[str, "_FakeModelSource"]):
            self._sources = sources

        @property
        def model_names(self) -> list[str]:
            return list(self._sources.keys())

        def items(self):
            return list(self._sources.items())

    source = _FakeComparedSource(
        {
            "base": _FakeModelSource({"neg_big": -0.9, "pos_small": 0.05, "pos_mid": 0.3}),
            "alt": _FakeModelSource({"neg_big": -0.8, "pos_small": 0.04, "pos_mid": 0.28}),
        }
    )

    chart = _importance_chart_compare_from_source(source, top_k=2, show_values=False)
    features = set(chart._data["feature"].unique().to_list())
    assert features == {"neg_big", "pos_mid"}, (
        f"expected the two largest-|importance| features {{'neg_big', 'pos_mid'}}, "
        f"got {features} -- signed-mean ranking would drop neg_big"
    )


def test_importance_compare_declares_color_and_dodge():
    """color='model' + Dodge(by='model') at the chart level; layers inherit it."""
    X, y, base, alt = _reg_setup()
    result = ferrum.importance_chart(base, X, y, top_k=3, compare={"alt": alt})
    assert result._position == Dodge(by="model")
    resolved = result._resolve_pending()
    layer_names = {ly.name for ly in resolved._layers}
    assert {"bar", "errorbar", "value_text"} <= layer_names
    bar = next(ly for ly in resolved._layers if ly.name == "bar")
    assert bar.encoding.get("color") == "model"


def test_importance_compare_vertical_bars_are_dodged():
    """Vertical orient: one bar per (feature, model) at distinct band positions.

    A non-dodged (overlapping) layout would place both models' bars at the same
    band center, giving only top_k distinct centers.
    """
    X, y, base, alt = _reg_setup()
    result = ferrum.importance_chart(base, X, y, top_k=3, orient="vertical", compare={"alt": alt})
    bars = _bar_rects(result.to_svg())
    assert len(bars) == 2 * 3  # n_models * top_k bars
    centers = {round(x + w / 2, 1) for x, w, _ in bars}
    assert len(centers) == 2 * 3  # every bar at its own sub-band position
    # Two distinct model fills, each used once per feature band.
    fills = Counter(f for _, _, f in bars)
    assert len(fills) == 2
    assert set(fills.values()) == {3}


def test_importance_compare_horizontal_uses_coordflip():
    """Horizontal orient renders the dodged layout via the vertical form + CoordFlip."""
    X, y, base, alt = _reg_setup()
    result = ferrum.importance_chart(base, X, y, top_k=3, orient="horizontal", compare={"alt": alt})
    assert isinstance(result._coord, CoordFlip)
    bars = _bar_rects(result.to_svg())
    assert len(bars) == 2 * 3
    fills = Counter(f for _, _, f in bars)
    assert set(fills.values()) == {3}


def test_importance_compare_horizontal_bars_dodge_on_band():
    """Horizontal orient must offset each model's bar within its feature band.

    Discriminating contract for the visual dodge: with CoordFlip the feature
    band renders on the vertical axis, so each feature band must contain
    n_models bars at distinct band-axis (y-center) positions -- not
    n_models bars stacked at the same position (the pre-fix overlap bug,
    GH #42 Rust apply_dodge coord_flipped gap, fixed upstream).

    Verifies the *banding* structure, not just "6 distinct y values": bars are
    sorted by y-center and chunked into consecutive groups of ``n_models``;
    each group's two bars must carry distinct model fills, and the gap
    between bars *within* a feature band must be smaller than the gap
    *between* feature bands -- i.e. bars cluster into n_features tight
    sub-band pairs rather than being spread arbitrarily along the axis.
    """
    n_models, top_k = 2, 3
    X, y, base, alt = _reg_setup()
    svg = ferrum.importance_chart(
        base,
        X,
        y,
        top_k=top_k,
        orient="horizontal",
        error_bars=False,
        show_values=False,
        compare={"alt": alt},
    ).to_svg()
    bars = _bar_geoms(svg)
    assert len(bars) == n_models * top_k

    ordered = sorted(bars, key=lambda b: b[1] + b[3] / 2)
    centers = [by + bh / 2 for _, by, _, bh, _ in ordered]
    assert len({round(c, 1) for c in centers}) == n_models * top_k

    bands = [ordered[i : i + n_models] for i in range(0, len(ordered), n_models)]
    for band in bands:
        fills = {fill for *_, fill in band}
        assert len(fills) == n_models, "each feature band must show one bar per model"

    gaps = [b - a for a, b in zip(centers, centers[1:])]
    intra_band_gaps = gaps[0::n_models]
    inter_band_gaps = gaps[n_models - 1 :: n_models]
    assert max(intra_band_gaps) < min(inter_band_gaps), (
        "bars did not cluster into tight per-feature bands; "
        "expected within-band spacing < between-band spacing"
    )


def test_importance_compare_has_model_legend():
    """A model legend labels each dodged group by its registration name."""
    X, y, base, alt = _reg_setup()
    svg = ferrum.importance_chart(base, X, y, top_k=3, compare={"alt": alt}).to_svg()
    texts = {t.strip() for _, t in _text_labels(svg)}
    assert "base" in texts
    assert "alt" in texts


def test_importance_compare_value_labels_dodge():
    """show_values labels track their dodged bars (one per bar, distinct x)."""
    X, y, base, alt = _reg_setup()
    svg = ferrum.importance_chart(
        base, X, y, top_k=3, orient="vertical", show_values=True, compare={"alt": alt}
    ).to_svg()
    value_labels = [
        (x, t) for x, t in _text_labels(svg) if re.fullmatch(r"-?\d+\.\d{3}", t.strip())
    ]
    assert len(value_labels) == 2 * 3  # one label per (feature, model) bar
    xs = {round(x, 1) for x, _ in value_labels}
    assert len(xs) == 2 * 3  # labels dodged to their bar's sub-band


def test_importance_compare_no_values_no_labels():
    """show_values=False drops the value-label layer entirely."""
    X, y, base, alt = _reg_setup()
    result = ferrum.importance_chart(base, X, y, top_k=3, show_values=False, compare={"alt": alt})
    resolved = result._resolve_pending()
    assert "_value_text" not in result._data.columns
    assert all(ly.name != "value_text" for ly in resolved._layers)


def test_importance_compare_no_error_bars_drops_layer():
    """error_bars=False drops the dodged error-rule layer."""
    X, y, base, alt = _reg_setup()
    result = ferrum.importance_chart(base, X, y, top_k=3, error_bars=False, compare={"alt": alt})
    resolved = result._resolve_pending()
    assert all(ly.name != "errorbar" for ly in resolved._layers)


def test_importance_compare_error_rules_dodge_to_distinct_positions():
    """Error rules must offset per model, tracking their bar's sub-band.

    Discriminating geometry check for the errorbar layer (spec: "error rules
    dodge alongside their bars") -- a non-dodged errorbar layer would place
    both models' rules at the same top_k x-positions (one per feature), not
    n_models*top_k distinct positions. Matched against the dodged bar centers
    to confirm each rule tracks its own bar, not just "is somewhere new".
    """
    X, y, base, alt = _reg_setup()
    svg = ferrum.importance_chart(
        base, X, y, top_k=3, orient="vertical", show_values=False, compare={"alt": alt}
    ).to_svg()

    error_xs = _error_rule_x_positions(svg)
    assert len(error_xs) == 2 * 3  # one rule per (feature, model)
    assert len({round(x, 1) for x in error_xs}) == 2 * 3  # every rule at its own sub-band x

    bar_centers = {round(x + w / 2, 1) for x, w, _ in _bar_rects(svg)}
    assert {round(x, 1) for x in error_xs} == bar_centers, (
        "error-rule x positions did not match the dodged bar centers"
    )


def test_importance_compare_deterministic():
    """Repeated compare renders are byte-identical (registration + global rank)."""
    X, y, base, alt = _reg_setup()
    svgs = {ferrum.importance_chart(base, X, y, compare={"alt": alt}).to_svg() for _ in range(3)}
    assert len(svgs) == 1


def test_importance_compare_none_byte_identical():
    """compare=None stays byte-identical to the no-compare single-model path."""
    X, y, base, _ = _reg_setup()
    with_kwarg = ferrum.importance_chart(base, X, y, compare=None).to_svg()
    without_kwarg = ferrum.importance_chart(base, X, y).to_svg()
    assert with_kwarg == without_kwarg


# ---------------------------------------------------------------------------
# shap_bar_chart (per_class=False) — dodge-by-model contract
# ---------------------------------------------------------------------------


def test_shap_bar_compare_returns_single_chart():
    """compare= (per_class=False, the default) returns a single Chart, not a
    small-multiples ConcatChart."""
    X, y, base, alt = _reg_setup()
    result = ferrum.shap_bar_chart(base, X, y, compare={"alt": alt})
    from ferrum.chart import Chart

    assert isinstance(result, Chart)
    assert not isinstance(result, ConcatChart)
    assert "<svg" in result.to_svg()


def test_shap_bar_compare_data_is_stacked_by_model():
    """The combined frame carries a model column with one row per (feature, model)."""
    X, y, base, alt = _reg_setup()
    result = ferrum.shap_bar_chart(base, X, y, max_display=3, compare={"alt": alt})
    data = result._data
    assert set(data.columns) >= {"feature", "abs_mean_shap", "model"}
    assert set(data["model"].unique().to_list()) == {"base", "alt"}
    assert data.height == 2 * 3  # n_models * max_display


def test_shap_bar_compare_global_max_display_shared_feature_set():
    """Exactly max_display distinct features, shared by every model (spec D4/§6)."""
    X, y, base, alt = _reg_setup()
    result = ferrum.shap_bar_chart(base, X, y, max_display=3, compare={"alt": alt})
    data = result._data
    features = data["feature"].unique().to_list()
    assert len(features) == 3
    for model in ("base", "alt"):
        model_feats = set(data.filter(pl.col("model") == model)["feature"].to_list())
        assert model_feats == set(features)


def test_shap_bar_compare_declares_color_and_dodge():
    """color='model' + Dodge(by='model') at the chart level."""
    X, y, base, alt = _reg_setup()
    result = ferrum.shap_bar_chart(base, X, y, max_display=3, compare={"alt": alt})
    assert result._position == Dodge(by="model")
    resolved = result._resolve_pending()
    bar = next(ly for ly in resolved._layers if ly.name == "bar")
    assert bar.encoding.get("color") == "model"


def test_shap_bar_compare_uses_coordflip():
    """The always-horizontal shap_bar layout renders via the vertical desugar
    form + CoordFlip under compare= (spec D2, mirrors importance)."""
    X, y, base, alt = _reg_setup()
    result = ferrum.shap_bar_chart(base, X, y, max_display=3, compare={"alt": alt})
    assert isinstance(result._coord, CoordFlip)


def test_shap_bar_compare_bars_are_dodged():
    """One bar per (feature, model) at distinct sub-band positions.

    A non-dodged (overlapping) layout would place both models' bars at the
    same band center, giving only max_display distinct band positions.
    """
    n_models, max_display = 2, 3
    X, y, base, alt = _reg_setup()
    svg = ferrum.shap_bar_chart(base, X, y, max_display=max_display, compare={"alt": alt}).to_svg()
    bars = _bar_geoms(svg)
    assert len(bars) == n_models * max_display

    ordered = sorted(bars, key=lambda b: b[1] + b[3] / 2)
    centers = [by + bh / 2 for _, by, _, bh, _ in ordered]
    assert len({round(c, 1) for c in centers}) == n_models * max_display

    bands = [ordered[i : i + n_models] for i in range(0, len(ordered), n_models)]
    for band in bands:
        fills = {fill for *_, fill in band}
        assert len(fills) == n_models, "each feature band must show one bar per model"


def test_shap_bar_compare_has_model_legend():
    """A model legend labels each dodged group by its registration name."""
    X, y, base, alt = _reg_setup()
    svg = ferrum.shap_bar_chart(base, X, y, max_display=3, compare={"alt": alt}).to_svg()
    texts = {t.strip() for _, t in _text_labels(svg)}
    assert "base" in texts
    assert "alt" in texts


def test_shap_bar_compare_deterministic():
    """Repeated compare renders are byte-identical (registration + global rank)."""
    X, y, base, alt = _reg_setup()
    svgs = {ferrum.shap_bar_chart(base, X, y, compare={"alt": alt}).to_svg() for _ in range(3)}
    assert len(svgs) == 1


def test_shap_bar_compare_none_byte_identical():
    """compare=None stays byte-identical to the no-compare single-model path."""
    X, y, base, _ = _reg_setup()
    with_kwarg = ferrum.shap_bar_chart(base, X, y, compare=None).to_svg()
    without_kwarg = ferrum.shap_bar_chart(base, X, y).to_svg()
    assert with_kwarg == without_kwarg


def test_shap_bar_compare_per_class_true_keeps_small_multiples():
    """per_class=True still returns the pre-#42 small-multiples ConcatChart,
    even under compare= (competing class-facet dimension, spec D6)."""
    X, y, base, alt = _reg_setup()
    result = ferrum.shap_bar_chart(base, X, y, per_class=True, compare={"alt": alt})
    assert isinstance(result, ConcatChart)
    assert len(result.charts) == 2


# ---------------------------------------------------------------------------
# cv_scores_chart (kind="box"/"strip") — dodge-by-model contract (spec D3)
# ---------------------------------------------------------------------------


def test_cv_scores_compare_kind_box_returns_single_chart():
    """compare= with kind="box" (default) returns a single Chart, not a
    small-multiples ConcatChart."""
    from ferrum.chart import Chart

    X, y, base, alt = _reg_setup()
    result = ferrum.cv_scores_chart(base, X, y, cv=3, compare={"alt": alt})
    assert isinstance(result, Chart)
    assert not isinstance(result, ConcatChart)
    assert "<svg" in result.to_svg()


def test_cv_scores_compare_kind_strip_returns_single_chart():
    """compare= with kind="strip" also returns a single dodge-by-model Chart."""
    from ferrum.chart import Chart

    X, y, base, alt = _reg_setup()
    result = ferrum.cv_scores_chart(base, X, y, cv=3, kind="strip", compare={"alt": alt})
    assert isinstance(result, Chart)
    assert not isinstance(result, ConcatChart)
    assert "<svg" in result.to_svg()


def test_cv_scores_compare_kind_bar_still_small_multiples():
    """kind="bar" keeps the #35 small-multiples layout under compare= (spec
    D3 carve-out: fold x split x model is three grouping dimensions, no
    coherent single dodge)."""
    X, y, base, alt = _reg_setup()
    result = ferrum.cv_scores_chart(base, X, y, cv=3, kind="bar", compare={"alt": alt})
    assert isinstance(result, ConcatChart)
    assert len(result.charts) == 2
    assert "<svg" in result.to_svg()


def test_cv_scores_compare_declares_color_and_dodge():
    """color='model' + Dodge(by='model') at the chart level; the box's IQR
    layer inherits the colour encoding."""
    X, y, base, alt = _reg_setup()
    result = ferrum.cv_scores_chart(base, X, y, cv=3, compare={"alt": alt})
    assert result._position == Dodge(by="model")
    resolved = result._resolve_pending()
    box_layer = next(ly for ly in resolved._layers if ly.name == "box")
    assert box_layer.encoding.get("color") == "model"


def test_cv_scores_compare_box_dodged_per_split_band():
    """One box per (split, model) at distinct sub-band x positions.

    A non-dodged (overlapping) layout would place both models' boxes at the
    same split-band center, giving only n_splits distinct centers.
    """
    n_models, n_splits = 2, 2
    X, y, base, alt = _reg_setup()
    svg = ferrum.cv_scores_chart(base, X, y, cv=3, compare={"alt": alt}).to_svg()
    boxes = _bar_geoms(svg)
    assert len(boxes) == n_models * n_splits

    ordered = sorted(boxes, key=lambda b: b[0] + b[2] / 2)
    centers = [b[0] + b[2] / 2 for b in ordered]
    assert len({round(c, 1) for c in centers}) == n_models * n_splits

    bands = [ordered[i : i + n_models] for i in range(0, len(ordered), n_models)]
    for band in bands:
        fills = {b[4] for b in band}
        assert len(fills) == n_models, "each split band must show one box per model"


def test_cv_scores_compare_strip_dodged_per_split_band():
    """One strip sub-band per (split, model); the ``cv`` fold points inside a
    sub-band collapse onto that sub-band's dodged x (no jitter -- spec D3
    single-position rule, verified separately in the jitter-drop test below).
    """
    n_models, n_splits, cv = 2, 2, 3
    X, y, base, alt = _reg_setup()
    svg = ferrum.cv_scores_chart(base, X, y, cv=cv, kind="strip", compare={"alt": alt}).to_svg()
    points = _point_geoms(svg)
    assert len(points) == n_models * n_splits * cv

    by_x: dict[float, list[str]] = {}
    for x, _, fill in points:
        by_x.setdefault(round(x, 1), []).append(fill)
    assert len(by_x) == n_models * n_splits  # one dodged sub-band per (split, model)
    for fills in by_x.values():
        assert len(set(fills)) == 1  # each sub-band is single-model
        assert len(fills) == cv  # every fold point shares its group's x


def test_cv_scores_compare_strip_drops_jitter():
    """Discriminating: a jittered layout scatters a (split, model) group's
    ``cv`` fold points across a range of x positions; dropping jitter under
    dodge (spec D3) collapses them onto exactly one x per group instead of
    up to ``cv`` distinct positions per group.
    """
    n_models, n_splits, cv = 2, 2, 3
    X, y, base, alt = _reg_setup()
    svg = ferrum.cv_scores_chart(base, X, y, cv=cv, kind="strip", compare={"alt": alt}).to_svg()
    points = _point_geoms(svg)
    assert len(points) == n_models * n_splits * cv
    distinct_x = {round(x, 1) for x, _, _ in points}
    assert len(distinct_x) == n_models * n_splits, (
        "expected exactly one x per (split, model) group -- jitter would spread "
        "the group's fold points across additional x positions"
    )


def test_cv_scores_compare_has_model_legend():
    """A model legend labels each dodged group by its registration name."""
    X, y, base, alt = _reg_setup()
    svg = ferrum.cv_scores_chart(base, X, y, cv=3, compare={"alt": alt}).to_svg()
    texts = {t.strip() for _, t in _text_labels(svg)}
    assert "base" in texts
    assert "alt" in texts


def test_cv_scores_compare_split_filter_train():
    """split="train" filters to a single split band under compare=, matching
    the single-model split= contract; only one dodged band (n_models boxes)
    remains."""
    X, y, base, alt = _reg_setup()
    result = ferrum.cv_scores_chart(base, X, y, cv=3, split="train", compare={"alt": alt})
    assert set(result._data["split"].unique().to_list()) == {"train"}
    boxes = _non_background_rect_geoms(result.to_svg())
    assert len(boxes) == 2  # n_models, one split band only


def test_cv_scores_compare_split_filter_test():
    """split="test" likewise filters to a single split band under compare=."""
    X, y, base, alt = _reg_setup()
    result = ferrum.cv_scores_chart(base, X, y, cv=3, split="test", compare={"alt": alt})
    assert set(result._data["split"].unique().to_list()) == {"test"}
    boxes = _non_background_rect_geoms(result.to_svg())
    assert len(boxes) == 2  # n_models, one split band only


def test_cv_scores_compare_deterministic():
    """Repeated compare renders are byte-identical (registration + row order)."""
    X, y, base, alt = _reg_setup()
    svgs = {
        ferrum.cv_scores_chart(base, X, y, cv=3, compare={"alt": alt}).to_svg() for _ in range(3)
    }
    assert len(svgs) == 1


def test_cv_scores_compare_strip_deterministic():
    """Repeated kind="strip" compare renders are byte-identical."""
    X, y, base, alt = _reg_setup()
    svgs = {
        ferrum.cv_scores_chart(base, X, y, cv=3, kind="strip", compare={"alt": alt}).to_svg()
        for _ in range(3)
    }
    assert len(svgs) == 1


def test_cv_scores_compare_none_byte_identical():
    """compare=None stays byte-identical to the no-compare single-model path
    (kind="box", the default)."""
    X, y, base, _ = _reg_setup()
    with_kwarg = ferrum.cv_scores_chart(base, X, y, cv=3, compare=None).to_svg()
    without_kwarg = ferrum.cv_scores_chart(base, X, y, cv=3).to_svg()
    assert with_kwarg == without_kwarg


def test_cv_scores_compare_strip_none_byte_identical():
    """compare=None stays byte-identical to the no-compare path for
    kind="strip" too -- the jitter-drop only fires when color_field is set."""
    X, y, base, _ = _reg_setup()
    with_kwarg = ferrum.cv_scores_chart(base, X, y, cv=3, kind="strip", compare=None).to_svg()
    without_kwarg = ferrum.cv_scores_chart(base, X, y, cv=3, kind="strip").to_svg()
    assert with_kwarg == without_kwarg
