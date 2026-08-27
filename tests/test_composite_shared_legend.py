"""Tests for ``fm.Resolve(scale=, legend=)`` — GH #16 Task 4.

Ships the user-facing control for composite figure-level shared legends: a
new ``Resolve`` value class accepted everywhere ``resolve=`` is accepted
today, plus the ``"legend"`` sub-object emitted on the composite node's wire
``resolve`` field (spec §6). Legend resolution defaults to following scale
resolution and can be forced back to per-panel with
``legend={"color": "independent"}``.

Task 4 (above) covers ``Resolve`` normalization, the wire shape
``_lower_composite`` produces, the validation matrix's typed ``ValueError``s,
and back-compat with the flat-dict ``resolve=`` form.

Task 5 adds the end-to-end regression suite for the flagship outcome: the
Rust-side figure-level legend band is now landed and ``pairplot(hue=)`` /
``jointplot(hue=)`` render exactly ONE legend. The tests below assert on
rendered SVG text-node counts (see ``legend_texts`` in ``tests/_snapshots.py``,
imported below as ``_legend_texts``) to discriminate
one-figure-legend output from N-per-panel-legend output, covering spec §9.1
(pairplot/jointplot one-legend), §9.5 (opt-out still shows N per-panel
legends — this also documents pre-feature/main behavior), §9.7 (explicit
leaf scale keeps its own legend), §9.8 (``legend=None`` leaf handling), and
§9.12 (``markers=`` shape collapse into the single color legend).
"""

from __future__ import annotations

import polars as pl
import pytest

import ferrum as fm
from ferrum.composition import (
    HConcatChart,
    LayerChart,
    RepeatChart,
    Resolve,
    _composite_resolve_field,
    _lower_composite,
)
from tests._snapshots import legend_texts as _legend_texts

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def df_a():
    return pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0], "g": ["a", "b", "a"]})


@pytest.fixture
def df_b():
    return pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0], "g": ["a", "b", "b"]})


@pytest.fixture
def chart_a(df_a):
    return fm.Chart(df_a).mark_point().encode(x="x", y="y", color="g")


@pytest.fixture
def chart_b(df_b):
    return fm.Chart(df_b).mark_point().encode(x="x", y="y", color="g")


# ---------------------------------------------------------------------------
# fm.Resolve export + normalization
# ---------------------------------------------------------------------------


def test_resolve_is_exported_from_top_level():
    assert fm.Resolve is Resolve


def test_resolve_default_fields_are_none():
    r = Resolve()
    assert r.scale is None
    assert r.legend is None


def test_resolve_is_a_value_object_equality():
    assert Resolve(scale={"color": "shared"}) == Resolve(scale={"color": "shared"})
    assert Resolve(scale={"color": "shared"}) != Resolve(scale={"color": "independent"})


# ---------------------------------------------------------------------------
# Accepted everywhere resolve= is accepted today.
# ---------------------------------------------------------------------------


def test_hconcat_accepts_resolve_value_class(chart_a, chart_b):
    combo = HConcatChart([chart_a, chart_b], resolve=Resolve(scale={"color": "shared"}))
    assert combo._resolve == Resolve(scale={"color": "shared"})


def test_layerchart_accepts_resolve_value_class(chart_a, chart_b):
    combo = LayerChart(chart_a, chart_b, resolve=Resolve(scale={"color": "independent"}))
    assert combo._resolve == Resolve(scale={"color": "independent"})


def test_repeatchart_accepts_resolve_value_class():
    template = (
        fm.Chart(pl.DataFrame({"m1": [1.0, 2.0], "m2": [3.0, 4.0]}))
        .mark_point()
        .encode(x=fm.Repeat.column, y="m1")
    )
    grid = fm.RepeatChart(template, column=["m1", "m2"], resolve=Resolve(scale={"x": "shared"}))
    assert grid.resolve == Resolve(scale={"x": "shared"})


def test_concatchart_accepts_resolve_value_class(chart_a, chart_b):
    combo = fm.ConcatChart(chart_a, chart_b, resolve=Resolve(scale={"y": "shared"}))
    assert combo._resolve == Resolve(scale={"y": "shared"})


def test_hconcat_sugar_accepts_resolve_value_class(chart_a, chart_b):
    combo = fm.hconcat(chart_a, chart_b, resolve=Resolve(scale={"color": "shared"}))
    assert isinstance(combo, HConcatChart)


def test_layer_sugar_accepts_resolve_value_class(chart_a, chart_b):
    combo = fm.layer(chart_a, chart_b, resolve=Resolve(scale={"color": "independent"}))
    assert isinstance(combo, LayerChart)


# ---------------------------------------------------------------------------
# Flat dict remains valid and means Resolve(scale=dict) — back-compat.
# ---------------------------------------------------------------------------


def test_flat_dict_and_resolve_scale_lower_identically(chart_a, chart_b):
    via_dict = HConcatChart([chart_a, chart_b], resolve={"color": "shared"})
    via_resolve = HConcatChart([chart_a, chart_b], resolve=Resolve(scale={"color": "shared"}))
    lowered_dict = _lower_composite(via_dict, auto_tooltips=False)
    lowered_resolve = _lower_composite(via_resolve, auto_tooltips=False)
    assert lowered_dict.tree["resolve"] == lowered_resolve.tree["resolve"]
    assert lowered_dict.tree["resolve"] == {"color": "shared"}


def test_flat_dict_renders_byte_identical_to_resolve_scale_equivalent(chart_a, chart_b):
    """Byte-stability guard: no legend= means the wire is scale-only, unchanged."""
    via_dict = HConcatChart([chart_a, chart_b], resolve={"color": "shared"}).to_svg()
    via_resolve = HConcatChart(
        [chart_a, chart_b], resolve=Resolve(scale={"color": "shared"})
    ).to_svg()
    assert via_dict == via_resolve


def test_no_resolve_still_renders_byte_identical_to_main(chart_a, chart_b):
    """Independent-resolve (no resolve= at all) composites are untouched."""
    plain = HConcatChart([chart_a, chart_b]).to_svg()
    explicit_independent = HConcatChart(
        [chart_a, chart_b], resolve=Resolve(scale={"color": "independent"})
    ).to_svg()
    assert "<svg" in plain
    assert plain == explicit_independent


# ---------------------------------------------------------------------------
# Wire emission: the "legend" sub-object (spec §6 wire contract).
# ---------------------------------------------------------------------------


def test_wire_omits_legend_key_when_resolve_has_no_legend(chart_a, chart_b):
    combo = HConcatChart([chart_a, chart_b], resolve={"color": "shared"})
    lowered = _lower_composite(combo, auto_tooltips=False)
    assert "legend" not in lowered.tree["resolve"]


def test_wire_emits_legend_independent_over_shared_scale(chart_a, chart_b):
    combo = HConcatChart(
        [chart_a, chart_b],
        resolve=Resolve(scale={"color": "shared"}, legend={"color": "independent"}),
    )
    lowered = _lower_composite(combo, auto_tooltips=False)
    assert lowered.tree["resolve"] == {
        "color": "shared",
        "legend": {"color": "independent"},
    }


def test_wire_emits_legend_shared_matching_shared_scale(chart_a, chart_b):
    combo = HConcatChart(
        [chart_a, chart_b],
        resolve=Resolve(scale={"color": "shared"}, legend={"color": "shared"}),
    )
    lowered = _lower_composite(combo, auto_tooltips=False)
    assert lowered.tree["resolve"] == {
        "color": "shared",
        "legend": {"color": "shared"},
    }


def test_composite_resolve_field_directly_scale_and_legend(chart_a, chart_b):
    field = _composite_resolve_field(
        Resolve(scale={"color": "shared", "size": "shared"}, legend={"size": "independent"}),
        kind="TestKind",
    )
    assert field == {
        "color": "shared",
        "size": "shared",
        "legend": {"size": "independent"},
    }


def test_composite_resolve_field_accepts_flat_dict():
    field = _composite_resolve_field({"x": "shared"}, kind="TestKind")
    assert field == {"x": "shared"}


def test_composite_resolve_field_accepts_none():
    assert _composite_resolve_field(None, kind="TestKind") == {}


# ---------------------------------------------------------------------------
# Validation matrix — typed ValueError, never a silent fallback.
# ---------------------------------------------------------------------------


def test_legend_shared_without_scale_shared_raises_at_lowering(chart_a, chart_b):
    """Acceptance criterion 6: legend='shared' with no shared scale is an error."""
    combo = HConcatChart([chart_a, chart_b], resolve=Resolve(legend={"color": "shared"}))
    with pytest.raises(ValueError, match="color"):
        _lower_composite(combo, auto_tooltips=False)


def test_legend_shared_over_independent_scale_raises_at_lowering(chart_a, chart_b):
    combo = HConcatChart(
        [chart_a, chart_b],
        resolve=Resolve(scale={"color": "independent"}, legend={"color": "shared"}),
    )
    with pytest.raises(ValueError, match="color"):
        _lower_composite(combo, auto_tooltips=False)


def test_legend_shared_error_names_both_modes(chart_a, chart_b):
    combo = HConcatChart(
        [chart_a, chart_b],
        resolve=Resolve(scale={"color": "independent"}, legend={"color": "shared"}),
    )
    with pytest.raises(ValueError) as excinfo:
        _lower_composite(combo, auto_tooltips=False)
    message = str(excinfo.value)
    assert "shared" in message
    assert "independent" in message


def test_legend_shared_over_shared_other_channel_scale_raises(chart_a, chart_b):
    """legend={'color': 'shared'} with only size shared (not color) still errors."""
    combo = HConcatChart(
        [chart_a, chart_b],
        resolve=Resolve(scale={"size": "shared"}, legend={"color": "shared"}),
    )
    with pytest.raises(ValueError, match="color"):
        _lower_composite(combo, auto_tooltips=False)


def test_legend_unsupported_channel_raises_at_construction(chart_a, chart_b):
    with pytest.raises(ValueError, match="legend"):
        HConcatChart([chart_a, chart_b], resolve=Resolve(legend={"x": "shared"}))


def test_legend_unsupported_channel_shape_raises(chart_a, chart_b):
    with pytest.raises(ValueError, match="legend"):
        HConcatChart([chart_a, chart_b], resolve=Resolve(legend={"shape": "shared"}))


def test_legend_invalid_mode_string_raises_at_construction(chart_a, chart_b):
    with pytest.raises(
        ValueError, match=r"HConcatChart: resolve\.legend\['color'\] must be one of"
    ):
        HConcatChart([chart_a, chart_b], resolve=Resolve(legend={"color": "loud"}))


def test_resolve_invalid_type_raises():
    chart = fm.Chart(pl.DataFrame({"x": [1]})).mark_point().encode(x="x")
    with pytest.raises(ValueError, match="dict, Resolve, or None"):
        fm.HConcatChart([chart, chart], resolve=42)


def test_resolve_legend_wrong_type_raises(chart_a, chart_b):
    with pytest.raises(ValueError, match="resolve.legend"):
        HConcatChart([chart_a, chart_b], resolve=Resolve(legend="color"))


# ---------------------------------------------------------------------------
# Explicit legend=independent over shared scale: today's rendering (spec §4).
# ---------------------------------------------------------------------------


def test_legend_independent_over_shared_scale_lowers_without_error(chart_a, chart_b):
    combo = HConcatChart(
        [chart_a, chart_b],
        resolve=Resolve(scale={"color": "shared"}, legend={"color": "independent"}),
    )
    lowered = _lower_composite(combo, auto_tooltips=False)
    assert lowered.tree["resolve"]["color"] == "shared"
    assert lowered.tree["resolve"]["legend"] == {"color": "independent"}


def test_legend_independent_over_shared_scale_renders(chart_a, chart_b):
    """Acceptance criterion 5 (partial — wire only, rendering lands separately).

    An end-to-end ``to_svg()`` smoke call: the Rust side already round-trips
    an accepted legend override (Task 1), so this must not raise and must
    still produce valid SVG with both panels.
    """
    svg = HConcatChart(
        [chart_a, chart_b],
        resolve=Resolve(scale={"color": "shared"}, legend={"color": "independent"}),
    ).to_svg()
    assert "<svg" in svg


# ---------------------------------------------------------------------------
# share_scale() sugar: scale-only merge, existing legend carried through.
# ---------------------------------------------------------------------------


def test_share_scale_preserves_existing_legend_field(chart_a, chart_b):
    base = HConcatChart(
        [chart_a, chart_b],
        resolve=Resolve(scale={"color": "shared"}, legend={"color": "independent"}),
    )
    merged = base.share_scale(size="shared")
    assert isinstance(merged._resolve, Resolve)
    assert merged._resolve.scale == {"color": "shared", "size": "shared"}
    assert merged._resolve.legend == {"color": "independent"}


def test_share_scale_without_legend_normalizes_to_resolve(chart_a, chart_b):
    """No legend ever set: share_scale still normalizes to one stable type.

    share_scale() always constructs the merged resolve as a ``Resolve``
    (never the flat-dict form), so ``_resolve``'s type is stable across
    every share_scale() call regardless of whether a legend override was
    ever set. Wire-equivalence between the flat-dict and Resolve(scale=)
    forms is already proven elsewhere (byte-identical SVG), so this has no
    rendering effect.
    """
    base = HConcatChart([chart_a, chart_b], resolve={"color": "shared"})
    merged = base.share_scale(size="shared")
    assert merged._resolve == Resolve(scale={"color": "shared", "size": "shared"})


def test_share_scale_still_raises_for_jointchart():
    joint = fm.jointplot(pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0]}), x="x", y="y")
    with pytest.raises(ValueError, match="resolve="):
        joint.share_scale(x="shared")


# ---------------------------------------------------------------------------
# Interaction with LayerChart's x-always-shared / y-independent invariants.
# ---------------------------------------------------------------------------


def test_layerchart_resolve_value_class_x_independent_still_raises(chart_a, chart_b):
    with pytest.raises(ValueError, match="GH #55"):
        LayerChart(chart_a, chart_b, resolve=Resolve(scale={"x": "independent"}))


def test_layerchart_resolve_value_class_y_independent_still_supported(chart_a, chart_b):
    layered = LayerChart(chart_a, chart_b, resolve=Resolve(scale={"y": "independent"}))
    svg = layered.to_svg()
    assert "<svg" in svg


# ---------------------------------------------------------------------------
# RepeatChart.spec introspection stays JSON-serializable with Resolve inputs.
# ---------------------------------------------------------------------------


def test_repeatchart_spec_serializable_with_resolve_value_class(df_a):
    import json

    template = fm.Chart(df_a).mark_point().encode(x="x", y="y", color="g")
    rc = fm.RepeatChart(
        template,
        row=["x", "y"],
        column=["x", "y"],
        resolve=Resolve(scale={"color": "shared"}, legend={"color": "independent"}),
    )
    spec = rc.spec
    json.dumps(spec)  # must not raise: spec is a serialization surface
    assert spec["resolve"] == {"color": "shared", "legend": {"color": "independent"}}


def test_repeatchart_spec_flat_dict_resolve_unchanged(df_a):
    template = fm.Chart(df_a).mark_point().encode(x="x", y="y", color="g")
    flat = {"color": "shared"}
    rc = fm.RepeatChart(template, row=["x", "y"], column=["x", "y"], resolve=flat)
    assert rc.spec["resolve"] == {"color": "shared"}
    assert rc.spec["resolve"] is flat  # back-compat: pass-through, not a copy


def test_repeatchart_spec_resolve_matches_lowered_composite_tree_resolve(df_a):
    """Divergence regression (design-review remediation, GH #16).

    ``RepeatChart.spec`` (introspection, via ``_resolve_wire_dict``) and the
    lowered composite tree's resolve field (render-time lowering, via
    ``_composite_resolve_field``) now share one wire-assembly core
    (``_assemble_resolve_wire``), so a ``Resolve(scale=..., legend=...)``
    value must flatten to the exact same dict on both surfaces -- they can
    never again drift apart on which channels survive.
    """
    template = fm.Chart(df_a).mark_point().encode(x="x", y="y", color="g")
    rc = fm.RepeatChart(
        template,
        row=["x", "y"],
        column=["x", "y"],
        resolve=Resolve(scale={"color": "shared", "x": "shared"}, legend={"color": "independent"}),
    )
    lowered = rc._composite_tree(auto_tooltips=False)
    assert rc.spec["resolve"] == lowered.tree["resolve"]


# ---------------------------------------------------------------------------
# Task 5 fixture — small hue-bearing DataFrame for jointplot.
#
# ``pairplot_df`` is defined in ``tests/conftest.py`` (shared with
# ``test_composite_shared_legend_goldens.py``).
# ---------------------------------------------------------------------------


@pytest.fixture
def joint_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "y": [2.0, 1.0, 4.0, 3.0, 6.0, 5.0],
            "grp": ["a", "b", "a", "b", "a", "b"],
        }
    )


# ---------------------------------------------------------------------------
# §9.1 — pairplot(hue=)/jointplot(hue=) render exactly ONE figure legend.
# ---------------------------------------------------------------------------


def test_pairplot_hue_renders_exactly_one_legend_title(pairplot_df):
    """Acceptance criterion 1: one legend group, not one per panel.

    Symmetric grid over 3 vars is 3x3=9 panels (6 off-diagonal + 3 diagonal);
    every panel carries the ``color`` channel on the shared ``grp`` field.
    Pre-feature (see the opt-out test below), this rendered 9 separate
    ``grp`` legend titles — one per panel. The figure-level legend band
    collapses that to exactly 1.
    """
    svg = fm.pairplot(pairplot_df, vars=["x", "y", "z"], hue="grp").to_svg()
    texts = _legend_texts(svg)
    assert texts.count("grp") == 1, f"expected exactly one 'grp' legend title; got texts: {texts}"


def test_pairplot_hue_legend_swatches_render_once_not_per_panel(pairplot_df):
    """Per-category swatch labels appear once, not once per participating panel."""
    svg = fm.pairplot(pairplot_df, vars=["x", "y", "z"], hue="grp").to_svg()
    texts = _legend_texts(svg)
    assert texts.count("a") == 1, f"expected exactly one 'a' swatch label; got texts: {texts}"
    assert texts.count("b") == 1, f"expected exactly one 'b' swatch label; got texts: {texts}"


def test_pairplot_corner_hue_renders_exactly_one_legend_title(pairplot_df):
    """Regression for the hole-cell fix: corner=True leaves a Hole in the
    upper-right triangle (pairplot's empty corner). The grid union/band
    placement must still collapse to one legend, not raise or duplicate.
    """
    svg = fm.pairplot(pairplot_df, vars=["x", "y", "z"], hue="grp", corner=True).to_svg()
    texts = _legend_texts(svg)
    assert texts.count("grp") == 1, (
        f"corner=True must still collapse to one 'grp' legend title; got texts: {texts}"
    )


def test_jointplot_hue_renders_exactly_one_legend_title(joint_df):
    """Acceptance criterion 9: jointplot(hue=) renders one legend.

    jointplot's grid has a Hole in the empty top-right corner (no cell
    between the top and right marginals) — the same hole-cell fix this
    regression covers for pairplot's corner=True.
    """
    svg = fm.jointplot(joint_df, x="x", y="y", hue="grp").to_svg()
    texts = _legend_texts(svg)
    assert texts.count("grp") == 1, f"expected exactly one 'grp' legend title; got texts: {texts}"
    assert texts.count("a") == 1, f"expected exactly one 'a' swatch label; got texts: {texts}"
    assert texts.count("b") == 1, f"expected exactly one 'b' swatch label; got texts: {texts}"


# ---------------------------------------------------------------------------
# §9.5 — opt-out (legend=independent) still shows N per-panel legends.
#
# This is the RED-proof: the feature ships as committed Rust+Python, so a
# regression test can't be run against unpatched Rust directly. Instead, the
# opt-out path exercises the *pre-feature* rendering contract (shared scale,
# independent legend) that pairplot/jointplot used to always produce, and
# which the default-legend-follows-scale behavior above replaces. If the
# figure-legend band regressed to always-suppress (over-eager) or
# always-per-panel (under-eager), one of this pair of tests would catch it.
# ---------------------------------------------------------------------------


def test_pairplot_legend_independent_opt_out_shows_per_panel_titles(pairplot_df):
    """Acceptance criterion 5, applied to pairplot's grid.

    Builds the equivalent RepeatChart directly (same template/diagonal/row/
    column/corner that ``pairplot(hue=)`` produces) but with
    ``Resolve(scale={"color": "shared"}, legend={"color": "independent"})``
    — the explicit opt-out. Every one of the 9 panels keeps its own 'grp'
    legend title: this is the pre-feature (main) rendering contract,
    documented here as a permanent regression guard.
    """
    shared = fm.pairplot(pairplot_df, vars=["x", "y", "z"], hue="grp")
    n_panels = len(shared.row) * len(shared.column)
    opt_out = RepeatChart(
        shared.template,
        row=shared.row,
        column=shared.column,
        diagonal=shared.diagonal,
        corner=shared.corner,
        resolve=Resolve(scale={"color": "shared"}, legend={"color": "independent"}),
    )
    svg = opt_out.to_svg()
    texts = _legend_texts(svg)
    assert texts.count("grp") == n_panels, (
        f"legend=independent must render one 'grp' title per panel "
        f"({n_panels} panels); got texts: {texts}"
    )


# ---------------------------------------------------------------------------
# §9.7 — explicit leaf Color(..., scale=...) keeps its own legend; others dedupe.
# ---------------------------------------------------------------------------


def test_leaf_with_explicit_scale_keeps_own_legend_others_dedupe():
    """Two shared-scale leaves dedupe into 1 figure legend; the third leaf,
    which carries an explicit ``Color(..., scale=OrdinalScale(...))``, is
    excluded from the domain union (spec §4/§7) and keeps its own panel
    legend — so the total 'grp' title count is 2, not 1 and not 3.
    """
    df_a = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0], "grp": ["p", "q", "p"]})
    df_b = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0], "grp": ["p", "q", "q"]})
    df_c = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [7.0, 8.0, 9.0], "grp": ["p", "q", "p"]})

    a = fm.Chart(df_a).mark_point().encode(x="x", y="y", color="grp")
    b = fm.Chart(df_b).mark_point().encode(x="x", y="y", color="grp")
    c = (
        fm.Chart(df_c)
        .mark_point()
        .encode(
            x="x",
            y="y",
            color=fm.Color("grp", scale=fm.OrdinalScale(domain=["p", "q"], range=["red", "blue"])),
        )
    )

    combo = fm.HConcatChart([a, b, c], resolve={"color": "shared"})
    texts = _legend_texts(combo.to_svg())
    assert texts.count("grp") == 2, (
        f"expected 1 figure legend (a+b) + 1 leaf legend (c, explicit scale); got texts: {texts}"
    )


# ---------------------------------------------------------------------------
# §9.8 — legend=None leaf: participates in the domain union, never captured;
# all-disabled leaves emit no figure legend.
# ---------------------------------------------------------------------------


def test_leaf_legend_none_participates_in_union_but_not_capture():
    """One leaf disables its legend; the other (still enabled) leaf's bundle
    is captured for the single figure legend, so the title still appears
    exactly once (not zero, not two).
    """
    df_a = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0], "grp": ["p", "q", "p"]})
    df_b = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0], "grp": ["p", "q", "q"]})

    a = fm.Chart(df_a).mark_point().encode(x="x", y="y", color="grp")
    b_disabled = (
        fm.Chart(df_b).mark_point().encode(x="x", y="y", color=fm.Color("grp", legend=None))
    )

    combo = fm.HConcatChart([a, b_disabled], resolve={"color": "shared"})
    texts = _legend_texts(combo.to_svg())
    assert texts.count("grp") == 1, (
        f"one enabled + one legend=None leaf must still render exactly one "
        f"'grp' legend title (from the enabled leaf); got texts: {texts}"
    )


def test_all_leaves_legend_none_emits_no_figure_legend():
    """All participating leaves disabled -> no figure legend at all."""
    df_a = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0], "grp": ["p", "q", "p"]})
    df_b = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0], "grp": ["p", "q", "q"]})

    a_disabled = (
        fm.Chart(df_a).mark_point().encode(x="x", y="y", color=fm.Color("grp", legend=None))
    )
    b_disabled = (
        fm.Chart(df_b).mark_point().encode(x="x", y="y", color=fm.Color("grp", legend=None))
    )

    combo = fm.HConcatChart([a_disabled, b_disabled], resolve={"color": "shared"})
    svg = combo.to_svg()
    texts = _legend_texts(svg)
    assert "grp" not in texts, f"all-disabled leaves must emit no figure legend; got texts: {texts}"
    assert "p" not in texts and "q" not in texts, (
        f"all-disabled leaves must render no swatch labels either; got texts: {texts}"
    )


# ---------------------------------------------------------------------------
# §9.12 — pairplot(hue=, markers=): shape glyphs collapse into the color legend.
# ---------------------------------------------------------------------------


def test_pairplot_hue_with_markers_collapses_shape_into_color_legend(pairplot_df):
    """``markers=`` maps hue levels to point shapes via the same ``grp`` field
    as ``color``. Same-field color+shape collapse (already covered per-panel
    by the flexibility campaign) must hold at the figure level too: still
    exactly one 'grp' legend title, not a second one for the shape channel.
    """
    svg = fm.pairplot(
        pairplot_df, vars=["x", "y"], hue="grp", markers=["circle", "square"]
    ).to_svg()
    texts = _legend_texts(svg)
    assert texts.count("grp") == 1, (
        f"shape must collapse into the single color legend, not add a second "
        f"'grp' title; got texts: {texts}"
    )
    # Sanity: the markers actually varied per hue level (not silently ignored).
    assert svg.count("<rect") > 0 and svg.count("<circle") > 0, (
        "expected a mix of circle and square (rect) point glyphs from markers="
    )


# ---------------------------------------------------------------------------
# Task 5 item 6 — jointplot public surface: the private resolve= wiring
# JointChart uses internally (``_resolve=``, renamed from ``_internal_resolve``
# for GH #16 design-review remediation) stays private.
# ---------------------------------------------------------------------------


def test_jointplot_with_hue_share_scale_still_raises(joint_df):
    """JointChart's internal ``_resolve=`` wiring must not have broken the
    ``_unsupported_resolve_error`` gate: ``share_scale()`` on a hue-colored
    ``jointplot`` still raises (JointChart's class-level
    ``_supports_user_resolve`` stays False regardless of this internal field).
    """
    joint = fm.jointplot(joint_df, x="x", y="y", hue="grp")
    with pytest.raises(ValueError, match="resolve="):
        joint.share_scale(x="shared")


def test_internal_resolve_not_in_jointplot_public_docstring():
    assert "_resolve" not in (fm.jointplot.__doc__ or "")


def test_internal_resolve_not_in_jointchart_public_docstring():
    assert "_resolve" not in (fm.JointChart.__doc__ or "")


def test_internal_resolve_not_in_jointplot_signature():
    import inspect

    sig = inspect.signature(fm.jointplot)
    assert "_resolve" not in sig.parameters
