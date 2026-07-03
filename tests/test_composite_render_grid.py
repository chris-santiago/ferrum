"""Grid/wrap composite cutover — ConcatChart routed through the Rust composite path.

Sibling of ``test_composite_render_linear.py`` (HConcat/VConcat).  ConcatChart
lowers to the ``wrap`` layout of the one-call Rust composite entry
(:func:`ferrum.composition._lower_composite`); because ``_compose_compare``
emits a ConcatChart, every ``compare=`` diagnostic whose per-model panels lower
cleanly rides the new path automatically.

The observable proxy for "these panels share (or don't share) an axis domain"
is the set of rendered ``<text>`` tick labels, parsed via the shared
``tests/_svg_extents`` helpers (no test re-copies the ``<text>`` regex).

What is RED-provable vs parity here
-----------------------------------
- **Parity** — ``cv_scores_chart(compare=)`` panels are single ``Chart``s, so
  they already shared via the old ConcatChart ``_apply_resolve`` injection.  The
  test locks that the *new* wrap path keeps the shared-y union.
- **GREEN (GH #45)** — position-wise sharing across *composite* children.
  ``residuals`` compare panels are titled *composites* (each a ``VConcat`` with
  ``child.properties(title=name)``).  The composite wire now lowers a non-root
  composite's figure title to a per-child ``"label"`` (Task 5d), so such children
  ride the new path and pair position-wise: the same grid position shares its axis
  across models while distinct positions keep distinct domains.
  (``pdp`` compare panels, by contrast, are single *faceted Charts* whose title
  lands on the leaf spec, so pdp cuts over with ``x: independent``.)
"""

from __future__ import annotations

import numpy as np
import polars as pl

import ferrum as fm
from ferrum.composition import ConcatChart, VConcatChart, _lower_composite
from tests._svg_extents import (
    extents_all_equal,
    numeric_text_entries,
    y_axis_extents,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _titled_vconcat(xlo: float, xhi: float, *, title: str | None = None) -> VConcatChart:
    """A 2-row VConcat whose panels bind ``x`` over ``[xlo, xhi]``.

    Mirrors the shape of a ``residuals``/``pdp`` compare panel (a titled,
    multi-leaf composite child) with a controllable x-domain so the
    shared-vs-independent distinction is unambiguous.
    """
    df = pl.DataFrame(
        {"x": [float(xlo), (float(xlo) + float(xhi)) / 2.0, float(xhi)], "y": [1.0, 2.0, 3.0]}
    )
    top = fm.Chart(df).mark_point().encode(x="x", y="y")
    bottom = fm.Chart(df).mark_point().encode(x="x", y="y")
    vc = VConcatChart([top, bottom])
    return vc.properties(title=title) if title is not None else vc


def _high_value_tick_bands(svg: str, *, value_threshold: float = 50.0, gap: float = 150.0) -> int:
    """Count spatially-separated clusters of high-value x-axis tick labels.

    When two disjoint-domain panels ([0, 1] and [100, 110]) share an axis, BOTH
    panels render tick labels reaching the union's high end, at two well-separated
    x-bands → 2 clusters.  When they are independent, only the high-domain panel
    reaches those values → 1 cluster.  Clustering by an x-gap makes the signal
    layout-agnostic (no reliance on which half of the canvas a panel occupies).
    """
    xs = sorted({round(x) for x, _, v in numeric_text_entries(svg) if v >= value_threshold})
    if not xs:
        return 0
    bands = 1
    for prev, cur in zip(xs, xs[1:]):
        if cur - prev > gap:
            bands += 1
    return bands


def _bottom_row_x_bands(
    svg: str, *, gap: float = 80.0, min_entries: int = 6
) -> list[tuple[float, float]]:
    """Return per-panel (lo, hi) x-tick extents for the bottom-most axis row.

    The residuals compare= grid stacks two 2x2 model panels side by side, so its
    bottom diagnostic row carries several panels whose x-domains do NOT reset
    monotonically (a low-domain leverage panel sits left of a higher-domain
    scale-location panel), which the shared value-reset splitter in
    ``_svg_extents`` cannot separate.  Here the panels are split by an x-gap
    instead — layout-agnostic and robust to non-monotone cross-panel domains.
    Bands are returned left-to-right.
    """
    from collections import defaultdict

    rows: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for x, y, val in numeric_text_entries(svg):
        rows[round(y)].append((x, val))
    candidates = [(y, ents) for y, ents in rows.items() if len(ents) >= min_entries]
    if not candidates:
        return []
    _, axis_row = max(candidates, key=lambda r: r[0])
    axis_row.sort()
    bands: list[list[float]] = [[axis_row[0][1]]]
    for (prev_x, _), (cur_x, cur_v) in zip(axis_row, axis_row[1:]):
        if cur_x - prev_x > gap:
            bands.append([cur_v])
        else:
            bands[-1].append(cur_v)
    return [(min(b), max(b)) for b in bands]


def _regression_models():
    """Two fitted regressors on the same X, y for compare= diagnostics."""
    from sklearn.linear_model import LinearRegression, Ridge

    rng = np.random.RandomState(0)
    X = rng.rand(80, 3)
    y = X[:, 0] * 2 + rng.rand(80)
    return LinearRegression().fit(X, y), Ridge().fit(X, y), X, y


# ---------------------------------------------------------------------------
# Lowering mechanism — ConcatChart → wrap
# ---------------------------------------------------------------------------


def test_concat_lowers_to_wrap_layout_with_ncols():
    charts = [
        fm.Chart(pl.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})).mark_point().encode(x="x", y="y")
        for _ in range(4)
    ]
    lowered = _lower_composite(ConcatChart(*charts, columns=2), auto_tooltips=False)
    assert lowered is not None
    assert lowered.tree["layout"] == "wrap"
    assert lowered.tree["ncols"] == 2
    assert len(lowered.tree["children"]) == 4
    assert all(c["kind"] == "leaf" for c in lowered.tree["children"])


def test_concat_columns_none_lowers_to_single_row():
    charts = [
        fm.Chart(pl.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})).mark_point().encode(x="x", y="y")
        for _ in range(3)
    ]
    lowered = _lower_composite(ConcatChart(*charts), auto_tooltips=False)
    assert lowered is not None
    assert lowered.tree["ncols"] == 3  # defaults to len(charts): one row


def test_concat_wrap_geometry_grows_height_with_rows():
    """columns=2 over 4 panels stacks two rows → taller than the 4-in-a-row layout."""
    charts = [
        fm.Chart(pl.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})).mark_point().encode(x="x", y="y")
        for _ in range(4)
    ]
    import re

    def _height(svg: str) -> float:
        m = re.search(r'viewBox="0 0 [0-9.]+ ([0-9.]+)"', svg)
        assert m is not None
        return float(m.group(1))

    single_row = _height(ConcatChart(*charts, columns=4).to_svg())
    two_rows = _height(ConcatChart(*charts, columns=2).to_svg())
    assert two_rows > single_row


# ---------------------------------------------------------------------------
# compare= parity — cv_scores (single-Chart children) shares y via the new path
# ---------------------------------------------------------------------------


def test_cv_scores_compare_lowers_and_shares_y():
    """PARITY: cv_scores compare panels are single Charts → lower to wrap leaves,
    and the shared-y union renders equal per-panel y extents.
    """
    base, alt, X, y = _regression_models()
    chart = fm.cv_scores_chart(base, X, y, cv=3, compare={"alt": alt})

    lowered = _lower_composite(chart, auto_tooltips=False)
    assert lowered is not None, "single-Chart compare panels must lower to the wrap path"
    assert lowered.tree["layout"] == "wrap"
    assert lowered.tree["resolve"] == {"x": "shared", "y": "shared"}
    assert all(c["kind"] == "leaf" for c in lowered.tree["children"])

    extents = y_axis_extents(chart.to_svg())
    assert len(extents) >= 2
    assert extents_all_equal(extents), f"cv_scores compare y not shared across panels: {extents}"


def test_cv_scores_compare_preserves_model_titles():
    """The per-model chart title (base/alt) survives the wrap-path cutover."""
    base, alt, X, y = _regression_models()
    svg = fm.cv_scores_chart(base, X, y, cv=3, compare={"alt": alt}).to_svg()
    assert "base" in svg and "alt" in svg


# ---------------------------------------------------------------------------
# Non-congruent children — sharing is skipped, render still succeeds
# ---------------------------------------------------------------------------


def test_non_congruent_children_skip_sharing_without_error():
    """A leaf child paired with a composite child is non-congruent: the Rust
    resolve pass skips the shared group (no cross-layout pairing) and each child
    keeps its own y-domain, yet the composition still renders.
    """
    leaf = (
        fm.Chart(pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [0.0, 0.5, 1.0]}))
        .mark_point()
        .encode(x="x", y="y")
    )
    composite = _titled_vconcat(1.0, 3.0)  # y is 1..3 in each stacked panel
    concat = ConcatChart(leaf, composite, columns=2, resolve={"y": "shared"})

    lowered = _lower_composite(concat, auto_tooltips=False)
    assert lowered is not None  # both children lower; congruence is judged in Rust
    child_kinds = {c["kind"] for c in lowered.tree["children"]}
    assert child_kinds == {"leaf", "composite"}  # non-congruent by construction

    extents = y_axis_extents(concat.to_svg())
    # The leaf's y-domain (max ~1) is NOT pulled up to the composite's (max ~3):
    # a forced shared union would make every panel's hi equal.
    assert not extents_all_equal(extents), (
        f"non-congruent children must not be force-shared; extents={extents}"
    )


# ---------------------------------------------------------------------------
# compare= over COMPOSITE children (residuals/pdp) — Rust-blocked, gated to old path
# ---------------------------------------------------------------------------


def test_pdp_compare_lowers_but_keeps_independent_feature_x():
    """pdp compare panels are single faceted Charts (leaves, not titled
    composites), so the composition lowers to the wrap path with an
    ``x: independent`` resolve field — the per-feature x axes stay independent
    (the disjoint-range gap is tick-free), matching the pre-cutover behavior
    guarded by ``test_compare.py::test_pdp_chart_compare_keeps_independent_feature_x``.
    """
    import re

    from sklearn.ensemble import RandomForestRegressor

    rng = np.random.default_rng(0)
    x_np = np.column_stack([rng.uniform(-2, 2, 200), rng.uniform(100, 110, 200)])
    y_np = x_np[:, 0] * 3 - (x_np[:, 1] - 105) * 2 + rng.standard_normal(200) * 0.1
    X = pl.DataFrame({"f0": x_np[:, 0], "f1": x_np[:, 1]})
    y = pl.Series("y", y_np)
    base = RandomForestRegressor(n_estimators=20, random_state=0).fit(x_np, y_np)
    alt = RandomForestRegressor(n_estimators=20, random_state=1).fit(x_np, y_np)

    chart = fm.pdp_chart(base, X, y, features=["f0", "f1"], compare={"alt": alt})
    lowered = _lower_composite(chart, auto_tooltips=False)
    assert lowered is not None  # faceted-Chart panels lower to the new wrap path
    assert lowered.tree["resolve"]["x"] == "independent"

    # Independent per-feature x → no ticks land in the empty gap between the two
    # disjoint feature ranges ([-2,2] and [100,110]); a collapsed shared x would.
    svg = chart.to_svg()
    ticks = [float(t) for t in re.findall(r">(-?\d+\.?\d*)</text>", svg)]
    assert not [t for t in ticks if 30 < t < 90], "per-feature x axes collapsed onto a shared range"


def test_residuals_compare_lowers_and_shares_position_wise():
    """GH #45 headline: residuals compare panels are titled composites (each a
    2x2 diagnostic grid).  Their figure title now lowers to a per-child label
    (Task 5d), so the composition rides the new path and pairs position-wise —
    each grid position shares its x-axis across the two models, while distinct
    positions keep distinct x-domains (no global collapse).
    """
    base, alt, X, y = _regression_models()
    chart = fm.residuals_chart(base, X, y, compare={"alt": alt})

    # (a) the titled-composite compare= tree now lowers to the Rust composite path.
    lowered = _lower_composite(chart, auto_tooltips=False)
    assert lowered is not None
    labels = {c.get("label") for c in lowered.tree["children"]}
    assert labels == {"base", "alt"}, f"per-model labels not on the tree: {labels}"

    svg = chart.to_svg()
    # (b) the per-model labels render in the SVG.
    assert "base" in svg and "alt" in svg

    # (c) position-wise sharing evidence.  The bottom diagnostic row is
    # [base-left, base-right, alt-left, alt-right]; the same grid position
    # shares its x-domain across models, but the two positions differ.
    bands = _bottom_row_x_bands(svg)
    assert len(bands) == 4, f"expected 4 bottom-row panels, got {bands}"
    assert bands[0] == bands[2], f"left-column x-domain not shared across models: {bands}"
    assert bands[1] == bands[3], f"right-column x-domain not shared across models: {bands}"
    assert bands[0] != bands[1], f"distinct grid positions collapsed onto one x-domain: {bands}"


def test_untitled_composite_children_share_axis_position_wise():
    """Permanent guard for the Rust congruence pairing (spec §6): UNTITLED
    congruent composite children lower to the new path today, and shared-x
    unions position-wise across the children — the low-domain panel's axis
    extends to the union, so the high-value tick band appears in both panels
    (two spatially-separated bands). If Rust-side congruence pairing for
    multi-leaf composite children regresses, this test catches it green-side
    (the titled variant below stays xfail until the per-child label lands)."""
    concat = ConcatChart(
        _titled_vconcat(0.0, 1.0),
        _titled_vconcat(100.0, 110.0),
        columns=2,
        resolve={"x": "shared"},
    )
    assert _lower_composite(concat, auto_tooltips=False) is not None
    assert _high_value_tick_bands(concat.to_svg()) >= 2


def test_titled_composite_children_share_axis_position_wise():
    """RED→GREEN target: two titled composite panels with disjoint x-domains,
    concatenated with ``resolve={'x': 'shared'}``, must pair position-wise so the
    low-domain panel's axis extends to the union (its high-value tick band appears
    alongside the high-domain panel's → two spatially-separated bands).
    """
    concat = ConcatChart(
        _titled_vconcat(0.0, 1.0, title="A"),
        _titled_vconcat(100.0, 110.0, title="B"),
        columns=2,
        resolve={"x": "shared"},
    )
    assert _high_value_tick_bands(concat.to_svg()) >= 2
