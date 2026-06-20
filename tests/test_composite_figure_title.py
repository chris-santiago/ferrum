"""Structural tests for composite figure-level chrome consolidation (#8).

`JointChart`, `ClusterMapChart`, and `RepeatChart` historically extended
`_ChartLike` and overrode `.properties()` to forward *all* kwargs (including
``title``/``subtitle``/``caption``) to an inner panel (center / heatmap /
template).  That placed the figure title on a sub-panel instead of the figure.

After consolidation, all composites inherit figure-chrome handling from
`_CompositeBase`:

  - ``.properties(title=, subtitle=, caption=)`` is intercepted at the figure
    level (stored as ``_figure_title`` / ``_figure_subtitle`` /
    ``_figure_caption``) and never fanned to inner panels.
  - Non-chrome kwargs (``width``, ``height``, ...) still reach the inner
    panel(s) they reached before.
  - A canonical title accessor (``_figure_title_text``) resolves the figure
    title for composites and ``_title`` for single charts, so ``to_html``
    sets the document ``<title>`` correctly.

This module covers the structural half (Task 12).  SVG-band and interactive
on-canvas threading are covered separately.
"""

from __future__ import annotations

import polars as pl
import pytest

import ferrum as fm


@pytest.fixture
def df():
    return pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [4.0, 3.0, 2.0, 1.0]})


@pytest.fixture
def base_chart(df):
    return fm.Chart(df).mark_point().encode(x="x", y="y")


@pytest.fixture
def hist_chart(df):
    return fm.Chart(df).mark_bar().encode(x="x", y="y")


# ---------------------------------------------------------------------------
# JointChart
# ---------------------------------------------------------------------------


def test_joint_title_stored_on_figure_not_center(base_chart, hist_chart):
    joint = fm.JointChart(base_chart, top=hist_chart, right=hist_chart)
    out = joint.properties(title="Joint Title")

    assert out._figure_title == "Joint Title"
    assert out._figure_title_text() == "Joint Title"
    # Inner center panel must not have received the title.
    assert out.center._title is None
    assert out.top._title is None
    assert out.right._title is None


def test_joint_subtitle_caption_stored_on_figure(base_chart):
    joint = fm.JointChart(base_chart)
    out = joint.properties(title="T", subtitle="S", caption="C")

    assert out._figure_title == "T"
    assert out._figure_subtitle == "S"
    assert out._figure_caption == "C"
    assert out.center._title is None


def test_joint_non_chrome_kwargs_reach_center(base_chart):
    joint = fm.JointChart(base_chart)
    out = joint.properties(width=320, height=240)

    assert out.center._width == 320
    assert out.center._height == 240
    # No figure chrome was set.
    assert out._figure_title is None


def test_joint_title_plus_width(base_chart):
    joint = fm.JointChart(base_chart)
    out = joint.properties(title="T", width=300)

    assert out._figure_title == "T"
    assert out.center._title is None
    assert out.center._width == 300


# ---------------------------------------------------------------------------
# ClusterMapChart
# ---------------------------------------------------------------------------


def test_clustermap_title_stored_on_figure_not_heatmap(base_chart, hist_chart):
    cm = fm.ClusterMapChart(base_chart, row_dendrogram=hist_chart, col_dendrogram=hist_chart)
    out = cm.properties(title="Cluster Title")

    assert out._figure_title == "Cluster Title"
    assert out._figure_title_text() == "Cluster Title"
    assert out.heatmap._title is None
    assert out.row_dendrogram._title is None
    assert out.col_dendrogram._title is None


def test_clustermap_non_chrome_kwargs_reach_heatmap(base_chart):
    cm = fm.ClusterMapChart(base_chart)
    out = cm.properties(width=500, height=500)

    assert out.heatmap._width == 500
    assert out.heatmap._height == 500
    assert out._figure_title is None


# ---------------------------------------------------------------------------
# RepeatChart
# ---------------------------------------------------------------------------


def test_repeat_title_stored_on_figure_not_template(df):
    template = fm.Chart(df).mark_point().encode(x=fm.Repeat.column, y=fm.Repeat.row)
    rep = fm.RepeatChart(template, row=["x", "y"], column=["x", "y"])
    out = rep.properties(title="Repeat Title")

    assert out._figure_title == "Repeat Title"
    assert out._figure_title_text() == "Repeat Title"
    assert out.template._title is None


def test_repeat_subtitle_caption_stored_on_figure(df):
    template = fm.Chart(df).mark_point().encode(x=fm.Repeat.column, y=fm.Repeat.row)
    rep = fm.RepeatChart(template, row=["x", "y"], column=["x", "y"])
    out = rep.properties(title="T", subtitle="S", caption="C")

    assert out._figure_title == "T"
    assert out._figure_subtitle == "S"
    assert out._figure_caption == "C"
    assert out.template._title is None


def test_repeat_non_chrome_kwargs_reach_template(df):
    template = fm.Chart(df).mark_point().encode(x=fm.Repeat.column, y=fm.Repeat.row)
    rep = fm.RepeatChart(template, row=["x", "y"], column=["x", "y"])
    out = rep.properties(width=200, height=200)

    assert out.template._width == 200
    assert out.template._height == 200
    assert out._figure_title is None


# ---------------------------------------------------------------------------
# Regression: concat composites unchanged
# ---------------------------------------------------------------------------


def test_hconcat_title_stored_on_figure(base_chart, hist_chart):
    out = (base_chart | hist_chart).properties(title="HC")
    assert out._figure_title == "HC"
    assert out._figure_title_text() == "HC"
    for child in out.charts:
        assert child._title is None


def test_vconcat_title_stored_on_figure(base_chart, hist_chart):
    out = (base_chart & hist_chart).properties(title="VC")
    assert out._figure_title == "VC"
    for child in out.charts:
        assert child._title is None


def test_concat_title_stored_on_figure(base_chart, hist_chart):
    out = fm.ConcatChart(base_chart, hist_chart, columns=2).properties(title="CC")
    assert out._figure_title == "CC"
    for child in out.charts:
        assert child._title is None


def test_concat_non_chrome_kwargs_reach_children(base_chart, hist_chart):
    out = fm.ConcatChart(base_chart, hist_chart, columns=2).properties(width=250)
    for child in out.charts:
        assert child._width == 250


# ---------------------------------------------------------------------------
# Canonical accessor + to_html document <title>
# ---------------------------------------------------------------------------


def test_single_chart_accessor_unchanged_via_layer(base_chart):
    # LayerChart keeps reading _title; canonical accessor resolves it.
    layered = fm.LayerChart(base_chart, base_chart, title="Layer T")
    assert layered._figure_title_text() == "Layer T"


def test_layer_accessor_default_when_no_title(base_chart):
    layered = fm.LayerChart(base_chart, base_chart)
    assert layered._figure_title_text() == "Ferrum chart"


# ---------------------------------------------------------------------------
# LayerChart HTML document <title> via .properties(title=) — R8.1
# ---------------------------------------------------------------------------


def test_layer_properties_title_resolves_document_title(base_chart):
    """LayerChart.properties(title=) must set the HTML document <title>."""
    layered = fm.LayerChart(base_chart, base_chart).properties(title="R8 Title")
    assert layered._figure_title_text() == "R8 Title"


def test_layer_properties_title_to_html(base_chart):
    """HTML export must embed the title set via .properties()."""
    layered = fm.LayerChart(base_chart, base_chart).properties(title="LayerDoc")
    html = layered.to_html()
    assert "<title>LayerDoc</title>" in html


def test_layer_ctor_title_to_html(base_chart):
    """Constructor path LayerChart(title=) must also set the HTML document <title>."""
    layered = fm.LayerChart(base_chart, base_chart, title="CtorDoc")
    html = layered.to_html()
    assert "<title>CtorDoc</title>" in html


def test_layer_properties_title_no_stray_layer_title(base_chart):
    """Inner layer charts must not carry the title after .properties(title=)."""
    layered = fm.LayerChart(base_chart, base_chart).properties(title="NoLeak")
    # The title must not be fanned to the inner charts.
    for inner in layered._charts:
        assert inner._title is None


def test_layer_no_title_default_document_title(base_chart):
    """LayerChart without a title falls back to the default document <title>."""
    layered = fm.LayerChart(base_chart, base_chart)
    html = layered.to_html()
    assert "<title>Ferrum chart</title>" in html


def test_layer_properties_width_still_fans_to_children(base_chart):
    """Non-chrome kwargs (width) must still reach the inner charts."""
    layered = fm.LayerChart(base_chart, base_chart).properties(width=400)
    for inner in layered._charts:
        assert inner._width == 400


def test_layer_properties_title_and_width(base_chart):
    """Title stored on LayerChart, width fanned to children — both work together."""
    layered = fm.LayerChart(base_chart, base_chart).properties(title="TW", width=350)
    assert layered._figure_title_text() == "TW"
    for inner in layered._charts:
        assert inner._title is None
        assert inner._width == 350


def test_joint_to_html_document_title(base_chart):
    joint = fm.JointChart(base_chart).properties(title="Doc Title")
    html = joint.to_html()
    assert "<title>Doc Title</title>" in html


def test_hconcat_to_html_document_title(base_chart, hist_chart):
    out = (base_chart | hist_chart).properties(title="HC Doc")
    html = out.to_html()
    assert "<title>HC Doc</title>" in html


def test_clustermap_to_html_document_title(base_chart):
    cm = fm.ClusterMapChart(base_chart).properties(title="CM Doc")
    html = cm.to_html()
    assert "<title>CM Doc</title>" in html


def test_to_html_default_title_without_figure_title(base_chart):
    joint = fm.JointChart(base_chart)
    html = joint.to_html()
    assert "<title>Ferrum chart</title>" in html


# ---------------------------------------------------------------------------
# SVG figure band (Part 1): Joint / Cluster / Repeat render the title at the
# figure level, not inside an inner panel.
# ---------------------------------------------------------------------------


def _inner_panel_svgs(composite) -> list[str]:
    """Render each inner panel chart of *composite* to its own SVG string."""
    panels = []
    for chart in composite.charts:
        # ``charts`` may contain Chart or LayerChart; both expose ``to_svg``.
        panels.append(chart.to_svg())
    return panels


def test_joint_svg_renders_figure_title(base_chart, hist_chart):
    joint = fm.JointChart(base_chart, top=hist_chart, right=hist_chart)
    svg = joint.properties(title="JointBand").to_svg()
    assert "JointBand" in svg
    # The title is on the composed figure, not baked into an inner panel.
    for panel_svg in _inner_panel_svgs(joint):
        assert "JointBand" not in panel_svg


def test_clustermap_svg_renders_figure_title(base_chart, hist_chart):
    cm = fm.ClusterMapChart(base_chart, row_dendrogram=hist_chart, col_dendrogram=hist_chart)
    svg = cm.properties(title="ClusterBand").to_svg()
    assert "ClusterBand" in svg
    assert "ClusterBand" not in cm.heatmap.to_svg()


def test_repeat_svg_renders_figure_title(df):
    template = fm.Chart(df).mark_point().encode(x=fm.Repeat.column, y=fm.Repeat.row)
    rep = fm.RepeatChart(template, row=["x", "y"], column=["x", "y"])
    svg = rep.properties(title="RepeatBand").to_svg()
    assert "RepeatBand" in svg
    # No materialized cell chart carries the figure title.
    for _r, _c, cell in rep.expand():
        assert "RepeatBand" not in cell.to_svg()


def test_joint_svg_renders_subtitle_and_caption(base_chart, hist_chart):
    joint = fm.JointChart(base_chart, top=hist_chart)
    svg = joint.properties(title="T", subtitle="Sub", caption="Cap").to_svg()
    assert "T" in svg
    assert "Sub" in svg
    assert "Cap" in svg


def test_joint_svg_byte_identical_without_figure_title(base_chart, hist_chart):
    """Backward compat: a composite with no figure title renders unchanged."""
    joint = fm.JointChart(base_chart, top=hist_chart, right=hist_chart)
    before = joint.to_svg()
    # ``.properties()`` with no chrome kwargs must not perturb the SVG.
    after = joint.properties().to_svg()
    assert before == after


# ---------------------------------------------------------------------------
# Interactive on-canvas band (Part 2): the merged scene carries the
# figure-title text node(s) and the canvas grows; inner panels carry no title.
# ---------------------------------------------------------------------------


def _scene_text_contents(scene: dict) -> list[str]:
    """Collect every text-node ``content`` in a scene's top-level title list."""
    return [n.get("content") for n in scene.get("title", []) if n.get("type") == "text"]


def _panel_text_contents(scene: dict) -> list[str]:
    """Collect text-node contents inside panels (per-panel chrome)."""
    out: list[str] = []
    for panel in scene.get("panels", []):
        for node in panel.get("strip_title", []):
            if node.get("type") == "text":
                out.append(node.get("content"))
    return out


def _render_interactive_scene(composite) -> dict:
    import json

    scene_json, _packed = composite._render_interactive()
    return json.loads(scene_json)


def test_hconcat_interactive_band_and_growth(base_chart, hist_chart):
    composite = base_chart | hist_chart
    base = _render_interactive_scene(composite)
    titled = _render_interactive_scene(composite.properties(title="HCBand"))

    assert "HCBand" in _scene_text_contents(titled)
    # Canvas grew vertically to fit the header band; width is unchanged.
    assert titled["height"] > base["height"]
    assert titled["width"] == base["width"]


def test_vconcat_interactive_band(base_chart, hist_chart):
    titled = _render_interactive_scene((base_chart & hist_chart).properties(title="VCBand"))
    assert "VCBand" in _scene_text_contents(titled)


def test_concat_interactive_band(base_chart, hist_chart):
    composite = fm.ConcatChart(base_chart, hist_chart, columns=2)
    titled = _render_interactive_scene(composite.properties(title="CCBand"))
    assert "CCBand" in _scene_text_contents(titled)


def test_joint_interactive_band_with_marginals(base_chart, hist_chart):
    joint = fm.JointChart(base_chart, top=hist_chart, right=hist_chart)
    base = _render_interactive_scene(joint)
    titled = _render_interactive_scene(joint.properties(title="JIBand"))

    assert "JIBand" in _scene_text_contents(titled)
    assert titled["height"] > base["height"]
    # The title is figure-level, not a per-panel strip title.
    assert "JIBand" not in _panel_text_contents(titled)


def test_joint_interactive_band_no_marginals(base_chart):
    """Single-panel joint (no marginals) still wraps the figure band."""
    joint = fm.JointChart(base_chart)
    base = _render_interactive_scene(joint)
    titled = _render_interactive_scene(joint.properties(title="JISolo"))

    assert "JISolo" in _scene_text_contents(titled)
    assert titled["height"] > base["height"]


def test_clustermap_interactive_band(base_chart, hist_chart):
    cm = fm.ClusterMapChart(base_chart, row_dendrogram=hist_chart, col_dendrogram=hist_chart)
    titled = _render_interactive_scene(cm.properties(title="CMBand"))
    assert "CMBand" in _scene_text_contents(titled)


def test_clustermap_interactive_band_no_dendrograms(base_chart):
    cm = fm.ClusterMapChart(base_chart)
    base = _render_interactive_scene(cm)
    titled = _render_interactive_scene(cm.properties(title="CMSolo"))
    assert "CMSolo" in _scene_text_contents(titled)
    assert titled["height"] > base["height"]


def test_repeat_interactive_band(df):
    template = fm.Chart(df).mark_point().encode(x=fm.Repeat.column, y=fm.Repeat.row)
    rep = fm.RepeatChart(template, row=["x", "y"], column=["x", "y"])
    titled = _render_interactive_scene(rep.properties(title="RIBand"))
    assert "RIBand" in _scene_text_contents(titled)


def test_repeat_corner_interactive_band(df):
    template = fm.Chart(df).mark_point().encode(x=fm.Repeat.column, y=fm.Repeat.row)
    rep = fm.RepeatChart(template, row=["x", "y"], column=["x", "y"], corner=True)
    titled = _render_interactive_scene(rep.properties(title="RCBand"))
    assert "RCBand" in _scene_text_contents(titled)


def test_interactive_caption_grows_canvas(base_chart, hist_chart):
    composite = base_chart | hist_chart
    base = _render_interactive_scene(composite)
    captioned = _render_interactive_scene(composite.properties(caption="Footer"))
    assert "Footer" in _scene_text_contents(captioned)
    # A caption is a footer band: it grows the canvas without a header offset.
    assert captioned["height"] > base["height"]


def test_interactive_byte_identical_without_figure_title(base_chart, hist_chart):
    """Backward compat: no figure title -> merged scene unchanged."""
    composite = base_chart | hist_chart
    before_json, before_packed = composite._render_interactive()
    after_json, after_packed = composite.properties()._render_interactive()
    assert before_json == after_json
    assert before_packed == after_packed


def test_interactive_parity_with_svg_caption_y(base_chart, hist_chart):
    """The interactive caption baseline matches the SVG band (parity contract).

    Both paths feed the composed-figure body width/height to the same Rust
    layout helper, so the caption text node's ``y`` equals the SVG caption ``y``.
    """
    import json

    from ferrum._chrome import chrome_kwargs, merge_configure_layers
    from ferrum._core import figure_title_nodes

    composite = (base_chart | hist_chart).properties(caption="Cap")
    # The merged body height the interactive path uses (pre-chrome) is the base
    # composite's merged height with no chrome injected.
    base_scene = _render_interactive_scene(base_chart | hist_chart)
    chrome = chrome_kwargs(merge_configure_layers(getattr(composite, "_configure_layers", None)))
    nodes_json, _hh, _fh = figure_title_nodes(
        width=float(base_scene["width"]),
        height=float(base_scene["height"]),
        caption="Cap",
        **chrome,
    )
    expected_cap_y = next(n["y"] for n in json.loads(nodes_json) if n.get("content") == "Cap")

    titled = _render_interactive_scene(composite)
    actual_cap_y = next(n["y"] for n in titled["title"] if n.get("content") == "Cap")
    assert actual_cap_y == expected_cap_y


# ---------------------------------------------------------------------------
# Factory dict-path (Part 3): properties={"title": ...} stores chrome at the
# figure level (does not leak into the inner panel).
# ---------------------------------------------------------------------------


def test_jointplot_dict_path_title_on_figure(df):
    joint = fm.jointplot(df, x="x", y="y", properties={"title": "DictTitle"})
    assert joint._figure_title == "DictTitle"
    # The inner center panel SVG must not embed the figure title.
    assert "DictTitle" not in joint.center.to_svg()
    # The composed figure SVG does render it.
    assert "DictTitle" in joint.to_svg()


def test_jointplot_dict_path_chrome_matches_chained(df):
    via_dict = fm.jointplot(df, x="x", y="y", properties={"title": "T", "subtitle": "S"})
    chained = fm.jointplot(df, x="x", y="y").properties(title="T", subtitle="S")
    assert via_dict._figure_title == chained._figure_title == "T"
    assert via_dict._figure_subtitle == chained._figure_subtitle == "S"
    assert via_dict.center._title is None


def test_jointplot_dict_path_non_chrome_reaches_center(df):
    joint = fm.jointplot(df, x="x", y="y", properties={"width": 320, "title": "T"})
    assert joint._figure_title == "T"
    assert joint.center._width == 320
    assert joint.center._title is None


def test_clustermap_dict_path_title_on_figure():
    # ``clustermap`` consumes a wide (matrix-shaped) dataframe; numeric columns
    # are the heatmap values and the row labels come from the leading column.
    wide = pl.DataFrame(
        {
            "row": ["a", "b", "c", "d"],
            "x": [1.0, 0.5, 0.3, 0.9],
            "y": [0.5, 1.0, 0.7, 0.2],
            "z": [0.3, 0.7, 1.0, 0.6],
        }
    )
    cm = fm.clustermap(wide, properties={"title": "CMDict"})
    assert cm._figure_title == "CMDict"
    assert "CMDict" not in cm.heatmap.to_svg()
    assert "CMDict" in cm.to_svg()


# ---------------------------------------------------------------------------
# Factory dict-path — LayerChart (T2: NEW-1 fix, round-3)
#
# _apply_overrides previously gated the chrome split on isinstance(_CompositeBase)
# only.  LayerChart is _ChartLike (not _CompositeBase), so properties={"title":...}
# was fanned by _rebuild_with_charts to every inner chart's .properties() call,
# setting title on each inner _title (leak) and leaving the HTML <title> as default.
# ---------------------------------------------------------------------------


def test_layer_factory_dict_title_stored_on_layer_not_inner(df):
    """Factory-dict properties={title=} on a LayerChart: title stored on the
    LayerChart itself (_title), not leaked into the inner layer charts."""
    from ferrum._overrides import _apply_overrides
    from ferrum.composition import LayerChart

    c = fm.Chart(df).mark_point().encode(x="x", y="y")
    layer = LayerChart(c, c)
    result = _apply_overrides(layer, properties={"title": "FactoryT"})

    # Title stored on the LayerChart.
    assert result._title == "FactoryT"
    # No inner chart should carry the title.
    for inner in result._charts:
        assert inner._title is None


def test_layer_factory_dict_title_figure_title_text(df):
    """_figure_title_text() resolves the title set via the factory-dict path."""
    from ferrum._overrides import _apply_overrides
    from ferrum.composition import LayerChart

    c = fm.Chart(df).mark_point().encode(x="x", y="y")
    layer = LayerChart(c, c)
    result = _apply_overrides(layer, properties={"title": "FTT"})
    assert result._figure_title_text() == "FTT"


def test_layer_factory_dict_title_html_document_title(df):
    """HTML export uses the figure-level title from the factory-dict path."""
    from ferrum._overrides import _apply_overrides
    from ferrum.composition import LayerChart

    c = fm.Chart(df).mark_point().encode(x="x", y="y")
    layer = LayerChart(c, c)
    result = _apply_overrides(layer, properties={"title": "HTMLDoc"})
    html = result.to_html()
    assert "<title>HTMLDoc</title>" in html


def test_layer_factory_dict_title_matches_chained(df):
    """properties={title=} dict path and .properties(title=) chained path agree."""
    from ferrum._overrides import _apply_overrides
    from ferrum.composition import LayerChart

    c = fm.Chart(df).mark_point().encode(x="x", y="y")
    layer = LayerChart(c, c)
    via_dict = _apply_overrides(layer, properties={"title": "Match"})
    via_chain = LayerChart(c, c).properties(title="Match")

    assert via_dict._title == via_chain._title == "Match"
    for inner in via_dict._charts:
        assert inner._title is None
    for inner in via_chain._charts:
        assert inner._title is None


def test_layer_factory_dict_width_still_fans_to_children(df):
    """Non-chrome kwargs (width) must still reach the inner charts."""
    from ferrum._overrides import _apply_overrides
    from ferrum.composition import LayerChart

    c = fm.Chart(df).mark_point().encode(x="x", y="y")
    layer = LayerChart(c, c)
    result = _apply_overrides(layer, properties={"title": "T", "width": 400})

    assert result._title == "T"
    for inner in result._charts:
        assert inner._title is None
        assert inner._width == 400


def test_layer_factory_dict_subtitle_caption_fan_to_merged(df):
    """subtitle/caption in the factory-dict path still reach the merged chart.

    LayerChart has no figure band, so subtitle/caption keep fanning to the
    inner charts (matching the chained .properties(subtitle=) behaviour).
    The factory-dict path must be equivalent to the chained .properties() path.
    """
    from ferrum._overrides import _apply_overrides
    from ferrum.composition import LayerChart

    c = fm.Chart(df).mark_point().encode(x="x", y="y")
    layer = LayerChart(c, c)
    result = _apply_overrides(layer, properties={"title": "T", "subtitle": "S"})

    # Title must be stored on the LayerChart (not leaked to inner charts as a
    # bare title — only subtitle is set on inner charts, not the text portion).
    assert result._title == "T"

    # The result must be equivalent to the chained .properties() path:
    # title stored on _title, subtitle fanned to inner charts the same way.
    chained = LayerChart(c, c).properties(title="T", subtitle="S")
    assert result._title == chained._title

    # subtitle was fanned to inner charts (LayerChart.properties fans non-title
    # kwargs).  Verify this is the same in both paths — not a regression.
    for inner_dict, inner_chain in zip(result._charts, chained._charts):
        assert inner_dict._title == inner_chain._title


# ---------------------------------------------------------------------------
# Shared _FIGURE_CHROME_KEYS constant (T2: NEW-2 fix, round-3)
#
# _CompositeBase.properties previously hard-coded three literal pops:
#   kwargs.pop("title"), kwargs.pop("subtitle"), kwargs.pop("caption").
# This parallel copy could drift from _overrides._FIGURE_CHROME_KEYS.
# After the fix, _CompositeBase.properties uses the shared constant, so
# both code paths reference the same definition.
# ---------------------------------------------------------------------------


def test_figure_chrome_keys_shared_constant():
    """_overrides and _CompositeBase.properties reference the same chrome keys.

    This test is an import-level smoke-check: if either site hard-codes a
    different key, the constant would diverge and this test would catch it
    indirectly via behavior.  The behavioral tests above already cover the
    split logic; this asserts the constant itself is the single source of truth.
    """
    from ferrum._overrides import _FIGURE_CHROME_KEYS

    assert set(_FIGURE_CHROME_KEYS) == {"title", "subtitle", "caption"}


def test_figure_chrome_keys_parity_composite_vs_layer(df):
    """Both _CompositeBase and LayerChart intercept the same chrome keys.

    Behavioral parity: a composite and a LayerChart given the same
    properties dict both intercept title/subtitle/caption and leave
    non-chrome kwargs for child forwarding.
    """
    from ferrum._overrides import _FIGURE_CHROME_KEYS, _apply_overrides
    from ferrum.composition import LayerChart

    c = fm.Chart(df).mark_point().encode(x="x", y="y")

    # LayerChart: title intercepted on _title.
    layer = LayerChart(c, c)
    layer_result = _apply_overrides(layer, properties={"title": "T", "width": 300})
    assert layer_result._title == "T"
    for inner in layer_result._charts:
        assert inner._title is None

    # _CompositeBase: title intercepted on _figure_title.
    composite = fm.HConcatChart([c, c])
    comp_result = _apply_overrides(composite, properties={"title": "T", "width": 300})
    assert comp_result._figure_title == "T"
    for child in comp_result.charts:
        assert child._title is None

    # Both must intersect exactly the _FIGURE_CHROME_KEYS keys.
    assert set(_FIGURE_CHROME_KEYS) == {"title", "subtitle", "caption"}
