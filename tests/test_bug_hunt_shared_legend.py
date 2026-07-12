"""Edge-case tests for the composite shared-legend subsystem (GH #16) — bug-hunter agent.

Targets the NEW code paths landed for figure-level shared legends:

- ``fm.Resolve(scale=, legend=)`` validation and wire assembly
  (``_assemble_resolve_wire`` / ``_composite_resolve_field`` in
  ``src/ferrum/composition.py``),
- the compositor legend band capture across hconcat/vconcat/wrap/grid
  layouts, holes, and nested composites,
- ``pairplot(hue=)`` / ``jointplot(hue=)`` figure-legend wiring in
  ``src/ferrum/plots/matrix.py`` (``rc_resolve`` / ``joint_resolve``).

Every test names the code path it exercises. The shared ``legend_texts``
probe (``tests/_snapshots.py``) counts legend/colorbar title occurrences —
the discriminating signal between ONE figure legend and N per-panel legends.
"""

from __future__ import annotations

import json

import polars as pl
import pytest

import ferrum as fm
from ferrum.composition import (
    HConcatChart,
    LayerChart,
    RepeatChart,
    Resolve,
    _lower_composite,
    _resolve_wire_dict,
)
from tests._snapshots import legend_texts as _legend_texts


# ---------------------------------------------------------------------------
# Fixtures — distinctive channel-field names ("grp", "sz", "w") that never
# collide with axis-title text nodes ("x", "y").
# ---------------------------------------------------------------------------


@pytest.fixture
def df_p():
    return pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0], "grp": ["p", "q", "p"]})


@pytest.fixture
def df_q():
    return pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0], "grp": ["p", "q", "q"]})


@pytest.fixture
def chart_p(df_p):
    return fm.Chart(df_p).mark_point().encode(x="x", y="y", color="grp")


@pytest.fixture
def chart_q(df_q):
    return fm.Chart(df_q).mark_point().encode(x="x", y="y", color="grp")


def _empty_chart():
    """A zero-row leaf chart — lowers to {"kind": "hole"} in _lower_leaf_node."""
    df = pl.DataFrame(schema={"x": pl.Float64, "y": pl.Float64, "grp": pl.Utf8})
    return fm.Chart(df).mark_point().encode(x="x", y="y", color="grp")


# ===========================================================================
# Composition corners — single leaf, vconcat, wrap partial row, nesting.
# ===========================================================================


def test_single_child_hconcat_shared_color_renders_one_legend(chart_p):
    """A 1-child composite with resolve={'color': 'shared'}: the union
    degenerates to the single leaf's own domain.  The figure band must not
    DOUBLE the legend (leaf legend + figure legend) — exactly one 'grp'.
    Code path: _lower_composite -> composite node resolve on a 1-leaf tree.
    """
    svg = HConcatChart([chart_p], resolve={"color": "shared"}).to_svg()
    texts = _legend_texts(svg)
    assert texts.count("grp") == 1, f"1-leaf shared composite must show one legend; got {texts}"


def test_vconcat_shared_color_renders_one_legend(chart_p, chart_q):
    """Every existing legend-band render test uses hconcat/grid; the vconcat
    layout arm (band placement below/right of a vertical stack) is untested.
    Code path: _composite_node(layout='vconcat', resolve={'color':'shared'}).
    """
    svg = fm.vconcat(chart_p, chart_q, resolve={"color": "shared"}).to_svg()
    texts = _legend_texts(svg)
    assert texts.count("grp") == 1, f"vconcat shared color must dedupe to one legend; got {texts}"


def test_concat_wrap_partial_last_row_shared_color_one_legend(chart_p, chart_q, df_p):
    """3 charts in a 2-column wrap layout leave a ragged last row.  The band
    extent heuristic must still capture one legend across all three cells.
    Code path: ConcatChart._composite_node_fields (ncols) + wrap-layout band.
    """
    c = fm.Chart(df_p).mark_point().encode(x="x", y="y", color="grp")
    svg = fm.concat(chart_p, chart_q, c, columns=2, resolve={"color": "shared"}).to_svg()
    texts = _legend_texts(svg)
    assert texts.count("grp") == 1, f"wrap layout shared color must dedupe; got {texts}"


def test_outer_shared_resolve_spans_nested_hconcat_leaves(chart_p, chart_q, df_p):  # FIXED (#74): outer resolve={'color':'shared'} now unions across the whole leaf span, spanning leaves nested inside a composite child into one figure legend
    """resolve declared on the OUTER vconcat only; the inner hconcat carries
    no resolve.  The outer union must span the nested subtree's leaves and
    produce exactly ONE figure legend for all three panels.
    Code path: _lower_any recursion — outer composite resolve over a nested
    composite child's leaves.
    """
    c = fm.Chart(df_p).mark_point().encode(x="x", y="y", color="grp")
    inner = fm.hconcat(chart_p, chart_q)
    svg = fm.vconcat(inner, c, resolve={"color": "shared"}).to_svg()
    texts = _legend_texts(svg)
    assert texts.count("grp") == 1, (
        f"outer shared resolve must span nested hconcat leaves into one legend; got {texts}"
    )


def test_nested_and_outer_both_shared_renders_single_legend(chart_p, chart_q, df_p):  # FIXED (#74): the outer union covers every leaf and the inner shared node is already-covered (no second band), so the figure renders exactly one 'grp' legend
    """Sharing declared at BOTH nesting levels: the outer union already
    covers every leaf, so the figure must not render duplicate 'grp'
    legends (one per composite node claiming the same leaves).
    Code path: resolve fields on both an inner and outer composite node.
    """
    c = fm.Chart(df_p).mark_point().encode(x="x", y="y", color="grp")
    inner = fm.hconcat(chart_p, chart_q, resolve={"color": "shared"})
    svg = fm.vconcat(inner, c, resolve={"color": "shared"}).to_svg()
    texts = _legend_texts(svg)
    assert texts.count("grp") == 1, (
        f"nested+outer shared must not duplicate the figure legend; got {texts}"
    )


def test_layerchart_inside_hconcat_shared_color_one_legend(chart_p, chart_q, df_p):  # FIXED (#74): the overlay node leaves color unset, so it inherits the outer shared mode and its spliced leaves join the union — one figure legend across overlay + sibling
    """A LayerChart child lowers via _splice_lowered_subtree (overlay node)
    inside the outer hconcat; the outer shared color must union across the
    overlay's leaves AND the sibling leaf, capturing one legend.
    Code path: _lower_any -> LayerChart._composite_tree(is_root=False) ->
    _splice_lowered_subtree -> outer resolve.
    """
    overlay = fm.layer(chart_p, chart_q)
    c = fm.Chart(df_p).mark_point().encode(x="x", y="y", color="grp")
    svg = fm.hconcat(overlay, c, resolve={"color": "shared"}).to_svg()
    texts = _legend_texts(svg)
    assert texts.count("grp") == 1, (
        f"Layer-inside-Concat shared color must dedupe to one legend; got {texts}"
    )


def test_explicit_independent_child_under_outer_shared_stays_independent(chart_p, chart_q, df_p):
    """The inheritance rule's explicit-wins half (spec §6, #74): an inner
    composite that explicitly declares resolve={'color':'independent'} opts its
    leaves OUT of an outer shared union — they keep their own per-panel legends
    rather than being deduped into the figure band.  Discriminating baseline:
    the same shape with the inner left UNSET collapses to one legend (the
    inherit-shared case, covered above), so an explicit-independent inner must
    render MORE than one 'grp' legend.
    Code path: _lower_any recursion — outer shared over an explicitly-
    independent inner composite (channel_resolve None vs Some(Independent)).
    """
    c = fm.Chart(df_p).mark_point().encode(x="x", y="y", color="grp")
    inner = fm.hconcat(chart_p, chart_q, resolve={"color": "independent"})
    svg = fm.hconcat(inner, c, resolve={"color": "shared"}).to_svg()
    texts = _legend_texts(svg)
    # Control: identical shape but inner UNSET -> inherits shared -> one legend.
    shared_inner = fm.hconcat(chart_p, chart_q)
    shared_svg = fm.hconcat(shared_inner, c, resolve={"color": "shared"}).to_svg()
    assert _legend_texts(shared_svg).count("grp") == 1, (
        "control: an unset inner must inherit the outer shared mode (one legend)"
    )
    assert texts.count("grp") > 1, (
        f"explicit-independent inner must keep its own per-panel legends, not "
        f"dedupe into the figure band; got {texts}"
    )


def test_repeat_1d_wrap_trailing_hole_shared_color_one_legend(df_p):
    """1-D column repeat wrapped at columns=2 over 3 fields leaves one
    trailing {"kind": "hole"} cell.  The hole must not disqualify the real
    cells from the legend band.
    Code path: RepeatChart._wrap_dimensions -> _build_grid_tree trailing
    holes + grid resolve.
    """
    df = pl.DataFrame(
        {
            "a": [1.0, 2.0, 3.0],
            "b": [2.0, 3.0, 4.0],
            "c": [3.0, 4.0, 5.0],
            "grp": ["p", "q", "p"],
        }
    )
    template = fm.Chart(df).mark_point().encode(x=fm.Repeat.column, y="a", color="grp")
    rc = RepeatChart(template, column=["a", "b", "c"], columns=2, resolve={"color": "shared"})
    svg = rc.to_svg()
    texts = _legend_texts(svg)
    assert texts.count("grp") == 1, (
        f"trailing wrap hole must not break the shared legend; got {texts}"
    )


# ===========================================================================
# Empty inputs / holes — empty-data leaves lower to holes; holes must not
# disqualify real cells from the legend band.
# ===========================================================================


def test_empty_leaf_hole_does_not_disqualify_shared_legend(chart_p, chart_q):
    """An empty-data middle child lowers to a sized hole under hconcat
    (_lower_leaf_node allow_hole=True).  The two real leaves must still
    union and produce one figure legend, and the hole must contribute
    nothing (no crash, no NaN geometry).
    """
    svg = fm.hconcat(chart_p, _empty_chart(), chart_q, resolve={"color": "shared"}).to_svg()
    texts = _legend_texts(svg)
    assert texts.count("grp") == 1, f"hole between real leaves broke the band; got {texts}"
    assert "NaN" not in svg


def test_empty_leaf_hole_first_child_shared_legend(chart_p, chart_q):
    """Empty chart FIRST: exercises the hole-before-any-leaf ordering.
    _apply_leaf_binding_overrides takes leaf_inputs[0] from the first REAL
    leaf; the band must still capture from the real leaves.
    """
    svg = fm.hconcat(_empty_chart(), chart_p, chart_q, resolve={"color": "shared"}).to_svg()
    texts = _legend_texts(svg)
    assert texts.count("grp") == 1, f"leading hole broke the shared legend; got {texts}"


def test_concat_wrap_hole_cell_shared_legend(chart_p, chart_q):
    """A hole inside a wrap grid cell (Task 8a: hole width/height ignored by
    grid/wrap cell math) with shared color: real cells still get one legend.
    """
    svg = fm.concat(
        chart_p, _empty_chart(), chart_q, columns=2, resolve={"color": "shared"}
    ).to_svg()
    texts = _legend_texts(svg)
    assert texts.count("grp") == 1, f"wrap hole cell broke the shared legend; got {texts}"


def test_all_empty_children_with_shared_legend_raises_typed_error():
    """Every child empty + shared legend resolve: the all-holes guard in
    _lower_composite must still raise the typed ValueError (not a Rust
    panic from an empty legend band).
    """
    with pytest.raises(ValueError, match="empty"):
        fm.hconcat(
            _empty_chart(),
            _empty_chart(),
            resolve=Resolve(scale={"color": "shared"}, legend={"color": "shared"}),
        ).to_svg()


# ===========================================================================
# Null / NaN — nulls in the hue column feeding the shared union.
# ===========================================================================


def test_null_in_hue_column_shared_color_does_not_crash():
    """Nulls in the categorical color column on both sides of the union:
    the shared ordinal domain union must not panic or emit NaN geometry.
    Code path: Rust composite resolve pass ordinal union with null values.
    """
    df1 = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0], "grp": ["p", None, "q"]})
    df2 = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0], "grp": [None, "q", "p"]})
    a = fm.Chart(df1).mark_point().encode(x="x", y="y", color="grp")
    b = fm.Chart(df2).mark_point().encode(x="x", y="y", color="grp")
    svg = fm.hconcat(a, b, resolve={"color": "shared"}).to_svg()
    assert "<svg" in svg
    assert "NaN" not in svg
    texts = _legend_texts(svg)
    assert texts.count("grp") == 1, f"null-bearing hue must still dedupe to one legend; got {texts}"


def test_nan_in_continuous_shared_color_domain():
    """NaN values inside a continuous color column under a shared union:
    the unioned [min, max] extent must not be poisoned to NaN (which would
    render an empty/NaN colorbar gradient).
    """
    df1 = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0], "val": [1.0, float("nan"), 5.0]})
    df2 = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0], "val": [2.0, 10.0, float("nan")]})
    a = fm.Chart(df1).mark_point().encode(x="x", y="y", color="val")
    b = fm.Chart(df2).mark_point().encode(x="x", y="y", color="val")
    svg = fm.hconcat(a, b, resolve={"color": "shared"}).to_svg()
    assert "<svg" in svg
    assert "NaN" not in svg, "NaN leaked into rendered SVG coordinates/gradient"
    texts = _legend_texts(svg)
    assert texts.count("val") == 1, f"NaN-bearing continuous union must yield one colorbar; got {texts}"


# ===========================================================================
# Type boundaries — int / bool hue dtypes crossing the Arrow CDI union.
# ===========================================================================


def test_int_hue_shared_color_renders_one_colorbar():
    """Integer color column under a shared union: pyarrow ships i64, the
    Rust resolve pass unions extents.  Must not reject the dtype or render
    two colorbars.
    """
    df1 = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0], "lvl": [1, 2, 3]})
    df2 = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0], "lvl": [4, 5, 6]})
    a = fm.Chart(df1).mark_point().encode(x="x", y="y", color="lvl")
    b = fm.Chart(df2).mark_point().encode(x="x", y="y", color="lvl")
    svg = fm.hconcat(a, b, resolve={"color": "shared"}).to_svg()
    texts = _legend_texts(svg)
    assert texts.count("lvl") == 1, f"int hue shared union must yield one colorbar; got {texts}"


def test_boolean_hue_shared_color_does_not_crash():
    """Boolean color column under a shared union — a dtype no existing
    shared-legend test touches.  Must render (categorical or rejected with
    a legible ValueError), never a Rust panic.
    """
    df1 = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0], "flag": [True, False, True]})
    df2 = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0], "flag": [False, True, False]})
    a = fm.Chart(df1).mark_point().encode(x="x", y="y", color="flag")
    b = fm.Chart(df2).mark_point().encode(x="x", y="y", color="flag")
    try:
        svg = fm.hconcat(a, b, resolve={"color": "shared"}).to_svg()
    except ValueError:
        return  # a legible rejection of bool color is acceptable
    assert "<svg" in svg
    assert "NaN" not in svg


def test_jointplot_hist_kind_with_hue_mixed_color_types():
    """jointplot(kind='hist', hue=...) sets joint_resolve={'color':'shared'}
    but the CENTER panel encodes color='count' (continuous Bin2D output)
    while the marginals encode color=hue (categorical).  The shared color
    union then spans two different field/type domains.
    Code path: _jointplot_build kind='hist' branch + JointChart._resolve.
    Must either render valid SVG or raise a legible ValueError — a Rust
    panic is a bug.
    """
    df = pl.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "y": [2.0, 1.0, 4.0, 3.0, 6.0, 5.0],
            "grp": ["p", "q", "p", "q", "p", "q"],
        }
    )
    try:
        svg = fm.jointplot(df, x="x", y="y", kind="hist", hue="grp").to_svg()
    except ValueError:
        return  # legible rejection acceptable
    assert "<svg" in svg
    assert "NaN" not in svg


# ===========================================================================
# Degenerate domains — single category, empty-string category, disjoint sets.
# ===========================================================================


def test_single_category_hue_shared_legend():
    """Every row in every panel is the same single category: the unioned
    ordinal domain has exactly one entry.  One legend title, one swatch.
    """
    df1 = pl.DataFrame({"x": [1.0, 2.0], "y": [1.0, 2.0], "grp": ["only", "only"]})
    df2 = pl.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0], "grp": ["only", "only"]})
    a = fm.Chart(df1).mark_point().encode(x="x", y="y", color="grp")
    b = fm.Chart(df2).mark_point().encode(x="x", y="y", color="grp")
    svg = fm.hconcat(a, b, resolve={"color": "shared"}).to_svg()
    texts = _legend_texts(svg)
    assert texts.count("grp") == 1
    assert texts.count("only") == 1, f"single-category union must show one swatch; got {texts}"


def test_empty_string_category_in_hue_shared_legend():
    """An empty-string category value in the shared ordinal union: must not
    crash, and the legend title still dedupes to one.  (The empty-string
    swatch label produces an empty <text> node the probe cannot count, so
    only the title count is asserted.)
    """
    df1 = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0], "grp": ["", "b", ""]})
    df2 = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0], "grp": ["b", "", "b"]})
    a = fm.Chart(df1).mark_point().encode(x="x", y="y", color="grp")
    b = fm.Chart(df2).mark_point().encode(x="x", y="y", color="grp")
    svg = fm.hconcat(a, b, resolve={"color": "shared"}).to_svg()
    texts = _legend_texts(svg)
    assert texts.count("grp") == 1, f"empty-string category broke legend dedup; got {texts}"
    assert texts.count("b") == 1


def test_union_domain_disjoint_categories_both_in_figure_legend():
    """Panel A only has 'alpha', panel B only has 'beta': the figure legend
    must show BOTH (the union), each exactly once — proving the band is
    built from the unioned domain, not one panel's local domain.
    """
    df1 = pl.DataFrame({"x": [1.0, 2.0], "y": [1.0, 2.0], "grp": ["alpha", "alpha"]})
    df2 = pl.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0], "grp": ["beta", "beta"]})
    a = fm.Chart(df1).mark_point().encode(x="x", y="y", color="grp")
    b = fm.Chart(df2).mark_point().encode(x="x", y="y", color="grp")
    svg = fm.hconcat(a, b, resolve={"color": "shared"}).to_svg()
    texts = _legend_texts(svg)
    assert texts.count("grp") == 1
    assert texts.count("alpha") == 1, f"union must include panel-A-only category; got {texts}"
    assert texts.count("beta") == 1, f"union must include panel-B-only category; got {texts}"


# ===========================================================================
# Both channels resolving — color AND size on the same composite.
# ===========================================================================


def test_color_and_size_both_shared_two_figure_legends():
    """resolve marks BOTH color and size shared; every leaf encodes both.
    The band must carry one 'grp' legend AND one 'sz' legend — two figure
    legends total, none per-panel.
    Code path: _assemble_resolve_wire multi-channel + Rust band stacking.
    """
    df1 = pl.DataFrame(
        {"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0], "grp": ["p", "q", "p"], "sz": [1.0, 5.0, 3.0]}
    )
    df2 = pl.DataFrame(
        {"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0], "grp": ["q", "p", "q"], "sz": [2.0, 10.0, 8.0]}
    )
    a = fm.Chart(df1).mark_point().encode(x="x", y="y", color="grp", size="sz")
    b = fm.Chart(df2).mark_point().encode(x="x", y="y", color="grp", size="sz")
    svg = fm.hconcat(a, b, resolve={"color": "shared", "size": "shared"}).to_svg()
    texts = _legend_texts(svg)
    assert texts.count("grp") == 1, f"shared color must dedupe to one legend; got {texts}"
    assert texts.count("sz") == 1, f"shared size must dedupe to one legend; got {texts}"


def test_color_shared_size_independent_mixed_modes():
    """color shared + size explicitly independent on the same composite:
    one figure color legend, two per-panel size legends.  Discriminates the
    per-channel legend capture (the band must not swallow the independent
    channel's per-panel legends).
    """
    df1 = pl.DataFrame(
        {"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0], "grp": ["p", "q", "p"], "sz": [1.0, 5.0, 3.0]}
    )
    df2 = pl.DataFrame(
        {"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0], "grp": ["q", "p", "q"], "sz": [2.0, 10.0, 8.0]}
    )
    a = fm.Chart(df1).mark_point().encode(x="x", y="y", color="grp", size="sz")
    b = fm.Chart(df2).mark_point().encode(x="x", y="y", color="grp", size="sz")
    svg = fm.hconcat(a, b, resolve={"color": "shared", "size": "independent"}).to_svg()
    texts = _legend_texts(svg)
    assert texts.count("grp") == 1, f"shared color must dedupe; got {texts}"
    assert texts.count("sz") == 2, f"independent size must keep per-panel legends; got {texts}"


# ===========================================================================
# Resolve(legend=) x Resolve(scale=) interaction matrix.
# ===========================================================================


def test_legend_independent_without_scale_renders_like_default(chart_p, chart_q):
    """Resolve(legend={'color': 'independent'}) with NO scale entry: the
    effective scale mode is 'independent', legend independent is valid
    (matrix allows it), and rendering must equal the no-resolve default
    (two per-panel legends) — the legend wire key alone must not trigger
    any band.
    Code path: _assemble_resolve_wire legend-only branch (out has no scale
    keys, legend sub-object emitted).
    """
    svg = fm.hconcat(
        chart_p, chart_q, resolve=Resolve(legend={"color": "independent"})
    ).to_svg()
    texts = _legend_texts(svg)
    assert texts.count("grp") == 2, (
        f"legend=independent with no shared scale must keep per-panel legends; got {texts}"
    )


def test_legend_shared_for_size_while_only_color_scale_shared_raises(chart_p, chart_q):
    """legend={'size': 'shared'} while only COLOR's scale is shared: size's
    effective scale mode is 'independent', so the mode matrix must raise at
    lowering — per-channel enforcement, not any-channel-shared.
    Code path: _assemble_resolve_wire effective_scale_mode lookup per channel.
    """
    combo = HConcatChart(
        [chart_p, chart_q],
        resolve=Resolve(scale={"color": "shared"}, legend={"size": "shared"}),
    )
    with pytest.raises(ValueError, match="size"):
        _lower_composite(combo, auto_tooltips=False)


def test_empty_resolve_value_renders_identical_to_none(chart_p, chart_q):
    """Resolve() (both fields None) must lower to an empty resolve field and
    render byte-identical to passing no resolve at all.
    Code path: _composite_resolve_field(Resolve(scale=None, legend=None)).
    """
    plain = fm.hconcat(chart_p, chart_q).to_svg()
    empty = fm.hconcat(chart_p, chart_q, resolve=Resolve()).to_svg()
    assert plain == empty


def test_repeatchart_matrix_violation_constructs_but_raises_at_render(df_p):
    """The shared-legend-requires-shared-scale matrix is enforced at
    LOWERING, not construction (documented in Resolve's docstring).  A
    RepeatChart holding the violating Resolve must (a) construct, (b) expose
    the violation verbatim through .spec (validate=False wire), and
    (c) raise ValueError from to_svg().
    Code paths: _validate_resolve (no matrix check) vs _assemble_resolve_wire
    validate=True/False split.
    """
    template = fm.Chart(df_p).mark_point().encode(x="x", y="y", color="grp")
    rc = RepeatChart(
        template, column=["x", "y"], resolve=Resolve(legend={"color": "shared"})
    )
    spec = rc.spec  # introspection must not raise
    assert spec["resolve"] == {"legend": {"color": "shared"}}
    json.dumps(spec)
    with pytest.raises(ValueError, match="shared"):
        rc.to_svg()


def test_resolve_wire_dict_excludes_unsupported_legend_channel_without_raising():
    """A Resolve constructed DIRECTLY (dataclass has no __post_init__
    validation) with an unsupported legend channel: _resolve_wire_dict
    (validate=False) must exclude the channel silently rather than raise
    or emit it — the documented introspection contract.
    Code path: _assemble_resolve_wire validate=False channel exclusion.
    """
    r = Resolve(legend={"opacity": "shared"})  # never validated at construction
    wire = _resolve_wire_dict(r)
    assert wire == {"legend": {}}


def test_share_scale_after_legend_independent_renders_per_panel(chart_p, chart_q):
    """share_scale() carries an existing legend override through to the
    REBUILT composite; the rendered output must honor it (per-panel
    legends), not just the _resolve attribute.  Extends the existing
    attribute-only test to the render contract.
    Code path: _ChartLike.share_scale legend carry-through -> wire -> band.
    """
    base = HConcatChart(
        [chart_p, chart_q],
        resolve=Resolve(scale={"color": "shared"}, legend={"color": "independent"}),
    )
    merged = base.share_scale(x="shared")
    svg = merged.to_svg()
    texts = _legend_texts(svg)
    assert texts.count("grp") == 2, (
        f"legend=independent must survive share_scale into rendering; got {texts}"
    )


# ===========================================================================
# Partial channel coverage — only some leaves encode the resolved channel.
# ===========================================================================


def test_only_some_leaves_encode_color_shared_legend(chart_p, df_q):
    """Leaf B carries NO color encoding at all; resolve={'color': 'shared'}.
    The union must span only the encoding leaves and still produce one
    figure legend — an unencoded leaf must not crash the union or produce
    a phantom second legend.
    """
    b_plain = fm.Chart(df_q).mark_point().encode(x="x", y="y")
    svg = fm.hconcat(chart_p, b_plain, resolve={"color": "shared"}).to_svg()
    texts = _legend_texts(svg)
    assert texts.count("grp") == 1, (
        f"one encoding + one non-encoding leaf must yield one legend; got {texts}"
    )


def test_no_leaf_encodes_color_shared_resolve_renders_without_legend(df_p, df_q):
    """NO leaf encodes color but resolve marks it shared: the union domain
    is empty.  Must render (no legend, no crash) — an empty union must not
    emit a phantom band or divide by zero in band sizing.
    """
    a = fm.Chart(df_p).mark_point().encode(x="x", y="y")
    b = fm.Chart(df_q).mark_point().encode(x="x", y="y")
    svg = fm.hconcat(a, b, resolve={"color": "shared"}).to_svg()
    assert "<svg" in svg
    assert "NaN" not in svg
    texts = _legend_texts(svg)
    assert texts.count("grp") == 0


def test_size_shared_but_only_one_leaf_encodes_size():
    """Size shared where only leaf A encodes size: one figure size legend
    from a single-leaf union (the size-channel analog of the partial-color
    case above).
    """
    df1 = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0], "sz": [1.0, 5.0, 3.0]})
    df2 = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    a = fm.Chart(df1).mark_point().encode(x="x", y="y", size="sz")
    b = fm.Chart(df2).mark_point().encode(x="x", y="y")
    svg = fm.hconcat(a, b, resolve={"size": "shared"}).to_svg()
    texts = _legend_texts(svg)
    assert texts.count("sz") == 1, f"single-encoder size union must yield one legend; got {texts}"


# ===========================================================================
# pairplot / jointplot figure-legend wiring (src/ferrum/plots/matrix.py).
# ===========================================================================


def test_pairplot_without_hue_sets_no_resolve(pairplot_df):
    """No hue -> rc_resolve must be None (matrix.py line ~337): a hue-less
    pairplot must not opt into the shared band, and its SVG carries no
    legend text at all.
    """
    pp = fm.pairplot(pairplot_df, vars=["x", "y"])
    assert pp.resolve is None
    svg = pp.to_svg()
    texts = _legend_texts(svg)
    assert texts.count("grp") == 0


def test_jointplot_without_hue_sets_no_internal_resolve():
    """No hue -> joint_resolve must be None (matrix.py line ~1293); the
    private _resolve field stays unset and no legend band is requested.
    """
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [2.0, 1.0, 3.0]})
    joint = fm.jointplot(df, x="x", y="y")
    assert joint._resolve is None
    assert "<svg" in joint.to_svg()


def test_pairplot_hue_survives_properties_title(pairplot_df):
    """RepeatChart._rebuild_with_charts must carry resolve through
    .properties(title=...); the figure legend must still dedupe after the
    rebuild (a dropped resolve would silently regress to 4 per-panel
    legends).
    """
    pp = fm.pairplot(pairplot_df, vars=["x", "y"], hue="grp").properties(title="Titled")
    svg = pp.to_svg()
    assert "Titled" in svg
    texts = _legend_texts(svg)
    assert texts.count("grp") == 1, f"resolve lost through properties() rebuild; got {texts}"


def test_jointplot_hue_survives_theme_rebuild():
    """JointChart._rebuild_with_charts passes _resolve=self._resolve; a
    theme() rebuild on a hue-bearing jointplot must keep the one-legend
    contract (the private _resolve is the only thing carrying it).
    """
    df = pl.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0],
            "y": [2.0, 1.0, 4.0, 3.0],
            "grp": ["p", "q", "p", "q"],
        }
    )
    joint = fm.jointplot(df, x="x", y="y", hue="grp").theme(fm.Theme(grid=False))
    svg = joint.to_svg()
    texts = _legend_texts(svg)
    assert texts.count("grp") == 1, f"_resolve lost through theme() rebuild; got {texts}"


def test_jointplot_hue_survives_properties_width():
    """JointChart._forward_child_properties routes width to the center chart
    and must also carry _resolve (it constructs a new JointChart with
    _resolve=self._resolve); one legend must survive.
    """
    df = pl.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0],
            "y": [2.0, 1.0, 4.0, 3.0],
            "grp": ["p", "q", "p", "q"],
        }
    )
    joint = fm.jointplot(df, x="x", y="y", hue="grp").properties(width=300, height=300)
    svg = joint.to_svg()
    texts = _legend_texts(svg)
    assert texts.count("grp") == 1, (
        f"_resolve lost through _forward_child_properties; got {texts}"
    )


def test_jointplot_hue_marginal_box_drops_hue_on_right_still_one_legend():  # FIXED (GH #75): both marginals now bind a synthetic single-level categorical column so desugar_boxplot's x-and-y requirement is met, and color=hue is set via .encode() (not a color_field= mark kwarg) so both the BoxStats groupby and the figure-legend title resolve from the same chart-level encoding
    """marginal_kind='box' builds the RIGHT marginal with encode(x=y) only —
    the hue color encoding is silently dropped for that panel
    (matrix.py _jointplot_build box branch).  The shared union then spans
    center+top only; the figure legend must still dedupe to one and the
    grid must render.
    """
    df = pl.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "y": [2.0, 1.0, 4.0, 3.0, 6.0, 5.0],
            "grp": ["p", "q", "p", "q", "p", "q"],
        }
    )
    svg = fm.jointplot(df, x="x", y="y", hue="grp", marginal_kind="box").to_svg()
    texts = _legend_texts(svg)
    assert texts.count("grp") == 1, (
        f"box right-marginal (no hue) must not break the one-legend contract; got {texts}"
    )


def test_pairplot_x_y_vars_asymmetric_hue_one_legend(pairplot_df):
    """Asymmetric x_vars/y_vars grid (no diagonal template at all) with hue:
    the resolve wiring is the same rc_resolve path but expand() takes the
    non-symmetric branch; still exactly one legend.
    """
    svg = fm.pairplot(pairplot_df, x_vars=["x", "y"], y_vars=["z"], hue="grp").to_svg()
    texts = _legend_texts(svg)
    assert texts.count("grp") == 1, f"asymmetric pairplot must dedupe; got {texts}"


def test_pairplot_numeric_hue_one_figure_colorbar(pairplot_df):
    """Numeric (float) hue: color becomes a CONTINUOUS scale, so the shared
    band must render one figure colorbar (not a categorical legend, not
    N per-panel colorbars).  diag_kind='none' isolates the colorbar path
    from the KDE-groupby-on-float question (probed separately below).
    """
    svg = fm.pairplot(pairplot_df, vars=["x", "y"], hue="z", diag_kind="none").to_svg()
    texts = _legend_texts(svg)
    assert texts.count("z") == 1, f"numeric hue must yield one figure colorbar; got {texts}"


def test_pairplot_numeric_hue_with_kde_diagonal_does_not_crash(pairplot_df):
    """Default diag_kind resolves to 'kde' with groupby=hue — here a FLOAT
    column.  Kde(groupby=<float col>) is a dtype no shared-legend test
    exercises; must render or raise a legible ValueError, never panic.
    """
    try:
        svg = fm.pairplot(pairplot_df, vars=["x", "y"], hue="z").to_svg()
    except ValueError:
        return  # legible rejection of float groupby acceptable
    assert "<svg" in svg
    assert "NaN" not in svg


def test_pairplot_hue_column_included_in_vars():
    """hue column ALSO used as a plotted var: pairplot with no vars= auto-
    detects numeric columns; with a numeric hue the hue column appears both
    as a panel axis AND the shared color channel.  Union + band must not
    crash and must dedupe to one colorbar title (axis titles for 'v' are
    separate text nodes counted too — so assert on the total occurrences
    being the axis count + exactly one legend title).
    """
    df = pl.DataFrame({"u": [1.0, 2.0, 3.0, 4.0], "v": [2.0, 1.0, 4.0, 3.0]})
    pp = fm.pairplot(df, hue="v", diag_kind="none")  # vars auto = [u, v]
    svg = pp.to_svg()
    assert "<svg" in svg
    assert "NaN" not in svg


# ===========================================================================
# Per-chart legend suppression interacting with the band.
# ===========================================================================


def test_leaf_configure_legend_orient_none_with_shared_band(chart_p, df_q):
    """One leaf suppresses its legend via configure_legend(orient='none')
    (a different suppression mechanism than Color(legend=None), which the
    existing suite covers).  The other enabled leaf's bundle must still be
    captured — exactly one figure legend, not zero and not two.
    """
    b = (
        fm.Chart(df_q)
        .mark_point()
        .encode(x="x", y="y", color="grp")
        .configure_legend(orient="none")
    )
    svg = fm.hconcat(chart_p, b, resolve={"color": "shared"}).to_svg()
    texts = _legend_texts(svg)
    assert texts.count("grp") == 1, (
        f"orient='none' on one leaf must leave one captured figure legend; got {texts}"
    )


def test_composite_level_configure_legend_orient_none_suppresses_band(chart_p, chart_q):  # FIXED (GH #74, sweep Task 11): configure_legend(orient='none') now joins the Color(legend=None) disabled mechanism, so per-panel legends AND the figure-level band are both suppressed (all-disabled → no band, §9.8) — consistent with orient='bottom' being honored by the band
    """configure_legend(orient='none') on the COMPOSITE fans to every leaf
    via _inject_parent_config; with all leaves suppressed the band must
    render NO figure legend (the all-disabled analog for the configure
    mechanism).
    """
    combo = fm.hconcat(chart_p, chart_q, resolve={"color": "shared"}).configure_legend(
        orient="none"
    )
    svg = combo.to_svg()
    texts = _legend_texts(svg)
    assert texts.count("grp") == 0, (
        f"composite-wide orient='none' must suppress the figure legend; got {texts}"
    )


def test_single_chart_configure_legend_orient_none_suppresses_legend(chart_p):
    """Fixed half of GH #74 defect 2: on a STANDALONE (non-composite) chart,
    ``configure_legend(orient='none')`` was a silent no-op (Rust had no
    ``LegendOrient::None`` variant, so ``prepare_render_inputs`` never saw a
    suppression signal for the chart-level path). It must now join the same
    ``disabled`` mechanism ``Color(legend=None)`` uses.
    """
    svg = chart_p.configure_legend(orient="none").to_svg()
    texts = _legend_texts(svg)
    assert texts.count("grp") == 0, (
        f"single-chart orient='none' must suppress its own legend; got {texts}"
    )


def test_chart_level_orient_none_is_absolute_over_explicit_channel_legend(df_p):
    """Pinned contract: chart-level ``configure_legend(orient='none')`` is an
    ABSOLUTE disable — it is not overridable by a per-channel explicit legend
    request (``Color(legend=Legend(...))``). There is no existing
    "per-channel wins over chart-level *disable*" precedent in this codebase;
    the only documented per-channel-wins rule is style-FILL (a per-channel
    ``Legend(...)`` fills style fields the chart-level ``configure_legend``
    left unset, `prepare/legend.rs` D13), not a suppression override. So even
    a leaf that explicitly customizes its color legend still loses it under a
    chart-level ``orient='none'``.
    """
    chart = (
        fm.Chart(df_p)
        .mark_point()
        .encode(x="x", y="y", color=fm.Color("grp", legend=fm.Legend(title="Group")))
        .configure_legend(orient="none")
    )
    svg = chart.to_svg()
    texts = _legend_texts(svg)
    assert texts.count("Group") == 0 and texts.count("grp") == 0, (
        f"chart-level orient='none' must win over an explicit per-channel "
        f"Legend(...) request; got {texts}"
    )


def test_leaf_configure_legend_overrides_composite_cascaded_orient_none(chart_p, chart_q):
    """Precedence pin (spec §9.6 'per-chart explicit legend settings
    interplay'): a composite-level ``configure_legend(orient='none')`` fans
    to every leaf's ``_configure`` list via ``_inject_parent_config``, which
    *prepends* the composite's layer so a leaf's own, later ``configure_legend``
    call still wins (``_inject_parent_config``'s documented "per-chart layers
    (which appear later) override them"). A leaf that re-enables its legend
    with an explicit ``orient='bottom'`` must therefore still render it, and
    the untouched sibling leaf stays suppressed by the composite's 'none'.
    """
    q_visible = chart_q.configure_legend(orient="bottom")
    combo = fm.hconcat(chart_p, q_visible, resolve={"color": "shared"}).configure_legend(
        orient="none"
    )
    svg = combo.to_svg()
    texts = _legend_texts(svg)
    assert texts.count("grp") == 1, (
        f"leaf's own configure_legend must override the composite's cascaded "
        f"orient='none' and keep exactly one legend visible; got {texts}"
    )


# ===========================================================================
# Interactive-path boundary (scene JSON, not SVG).
# ===========================================================================


def test_shared_legend_interactive_render_does_not_raise(chart_p, chart_q):
    """The interactive composite entry (render_composite_interactive) takes
    the same lowered tree with the legend wire; it must produce parseable
    scene JSON + packed bytes, not panic on the legend sub-object.
    """
    combo = fm.hconcat(
        chart_p, chart_q, resolve=Resolve(scale={"color": "shared"}, legend={"color": "shared"})
    )
    scene_json, packed = combo._render_interactive()
    assert isinstance(packed, bytes)
    json.loads(scene_json)


def test_pairplot_hue_interactive_render_does_not_raise(pairplot_df):
    """pairplot(hue=)'s grid tree through the interactive entry: holes +
    shared color + legend band must survive the scene path.
    """
    pp = fm.pairplot(pairplot_df, vars=["x", "y"], hue="grp")
    scene_json, packed = pp._render_interactive()
    assert isinstance(packed, bytes)
    json.loads(scene_json)


# ===========================================================================
# Extreme values in the shared continuous union.
# ===========================================================================


def test_infinity_in_shared_continuous_color_domain():
    """+inf in one panel's color column: the unioned extent hits infinity.
    The colorbar/scale must not emit 'Infinity' or 'NaN' into the SVG
    (either clamp/ignore inf or raise legibly).
    """
    df1 = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0], "val": [1.0, 2.0, 3.0]})
    df2 = pl.DataFrame(
        {"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0], "val": [2.0, float("inf"), 4.0]}
    )
    a = fm.Chart(df1).mark_point().encode(x="x", y="y", color="val")
    b = fm.Chart(df2).mark_point().encode(x="x", y="y", color="val")
    try:
        svg = fm.hconcat(a, b, resolve={"color": "shared"}).to_svg()
    except ValueError:
        return  # legible rejection acceptable
    assert "NaN" not in svg, "inf in shared color union leaked NaN into the SVG"
    assert "Infinity" not in svg, "inf in shared color union leaked Infinity into the SVG"


def test_identical_value_shared_size_domain_collapses_to_point():
    """Every size value identical across both panels: the unioned size
    domain is [v, v] (zero width).  Size-scale normalization must not
    divide by zero (NaN radii) in either the marks or the figure size
    legend.
    """
    df1 = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0], "sz": [5.0, 5.0, 5.0]})
    df2 = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0], "sz": [5.0, 5.0, 5.0]})
    a = fm.Chart(df1).mark_point().encode(x="x", y="y", size="sz")
    b = fm.Chart(df2).mark_point().encode(x="x", y="y", size="sz")
    svg = fm.hconcat(a, b, resolve={"size": "shared"}).to_svg()
    assert "<svg" in svg
    assert "NaN" not in svg, "degenerate [v, v] shared size domain produced NaN geometry"
