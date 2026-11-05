import warnings

import polars as pl
import pytest
import ferrum as fe


@pytest.fixture
def df():
    return pl.DataFrame(
        {
            "t": [1.0, 2.0, 3.0, 4.0, 5.0],
            "lo": [0.5, 1.5, 2.5, 3.5, 4.5],
            "hi": [1.5, 2.5, 3.5, 4.5, 5.5],
        }
    )


def test_ribbon_basic(df):
    """A bare mark_ribbon lowers to one unlayered ribbon mark, as mark_area does."""
    chart = fe.Chart(df).mark_ribbon().encode(x="t", y="lo", y2="hi")
    spec = chart._build_spec()
    assert spec.layers is None
    assert spec.mark == "ribbon"


def test_ribbon_missing_y2_raises(df):
    chart = fe.Chart(df).mark_ribbon().encode(x="t", y="lo")
    with pytest.raises(ValueError, match="y2"):
        chart._build_spec()


def test_ribbon_opacity_threaded(df):
    chart = fe.Chart(df).mark_ribbon(opacity=0.5).encode(x="t", y="lo", y2="hi")
    json_str = chart._build_spec().to_json()
    assert "0.5" in json_str  # opacity in mark_style


def test_ribbon_render_smoke(df):
    chart = fe.Chart(df).mark_ribbon().encode(x="t", y="lo", y2="hi")
    svg = chart.to_svg()
    assert "<svg" in svg
    assert "<path" in svg  # ribbon emits a path


def test_ribbon_y2_field_propagates(df):
    chart = fe.Chart(df).mark_ribbon().encode(x="t", y="lo", y2="hi")
    json_str = chart._build_spec().to_json()
    assert '"y2"' in json_str
    assert '"hi"' in json_str  # y2 field name


# ---------------------------------------------------------------------------
# F-L02-03: interpolate= forwarding (batch E, T14)
# ---------------------------------------------------------------------------


def test_ribbon_interpolate_forwarded_to_mark_kwargs(df):
    """desugar_ribbon must put ``interpolate`` in the emitted layer's
    mark_kwargs, not discard it after validating."""
    chart = fe.Chart(df).mark_ribbon(interpolate="cardinal").encode(x="t", y="lo", y2="hi")
    json_str = chart._build_spec().to_json()
    assert '"interpolate":"cardinal"' in json_str


def test_ribbon_cardinal_differs_from_linear_bytes(df):
    """A non-linear interpolate= must change the rendered path bytes —
    the T1 Rust curve module renders curves for ribbon, so once forwarded
    a cardinal boundary is geometrically different from a linear one."""
    linear_svg = (
        fe.Chart(df).mark_ribbon(interpolate="linear").encode(x="t", y="lo", y2="hi").to_svg()
    )
    cardinal_svg = (
        fe.Chart(df).mark_ribbon(interpolate="cardinal").encode(x="t", y="lo", y2="hi").to_svg()
    )
    assert linear_svg != cardinal_svg


def test_ribbon_cardinal_emits_curve_command(df):
    """A curved ribbon boundary must emit an SVG cubic-bezier ("C") path
    command; a linear boundary uses only "M"/"L"."""
    svg = fe.Chart(df).mark_ribbon(interpolate="cardinal").encode(x="t", y="lo", y2="hi").to_svg()
    assert " C" in svg


def test_ribbon_default_renders_identically_to_prefix_spec_without_interpolate(df):
    """Byte-stability: a genuinely pre-fix spec (no ``interpolate`` key at
    all in the flat ribbon mark's ``mark_style``, reconstructed by stripping
    the key from the post-fix JSON and re-parsing via ``ChartSpec.from_json``)
    must render identically to the post-fix default.  This is the Python
    side of the same property Rust pins directly in
    ``ribbon_linear_output_is_byte_identical_to_pre_change``
    (crates/ferrum-core/src/render/marks/ribbon.rs) — forwarding
    ``interpolate`` must not perturb rendering when it is absent/linear."""
    import json

    from ferrum._core import ChartSpec, render_svg

    chart = fe.Chart(df).mark_ribbon().encode(x="t", y="lo", y2="hi")
    spec, data, viewport, theme_dict, chart_config_dict = chart._render_inputs()
    spec_dict = json.loads(spec.to_json())
    layer_style = spec_dict["mark_style"]
    assert "interpolate" in layer_style  # forwarding did add the key post-fix
    del layer_style["interpolate"]
    prefix_spec = ChartSpec.from_json(json.dumps(spec_dict))
    prefix_svg = render_svg(
        prefix_spec,
        data,
        viewport=viewport,
        theme=theme_dict,
        chart_config=chart_config_dict or None,
    )
    assert prefix_svg == chart.to_svg()


def test_ribbon_unknown_interpolate_raises_value_error(df):
    """An out-of-vocabulary interpolate= value now refuses at declaration,
    through ``mark_ribbon``'s own ``validate_choice`` gate (wave 6B,
    #137/#176) — not deferred to Rust's ``RenderError::UnknownInterpolate``
    at render time, which this test previously pinned. ``mark_ribbon`` is a
    composite-mark constructor parameter, not forwarded through ``MarkBase``,
    so it needed its own gate rather than inheriting one."""
    with pytest.raises(ValueError, match="smooth"):
        fe.Chart(df).mark_ribbon(interpolate="smooth")


# ---------------------------------------------------------------------------
# A bare mark_ribbon() is unlayered: every message about it, and every
# option its own x/y/y2 bindings carry, reaches it the way mark_area's do.
# ---------------------------------------------------------------------------


def test_a_bare_ribbons_other_messages_drop_the_layer_index_too(df):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fe.Chart(df).mark_ribbon().encode(x="t", y="lo", y2="hi", size="lo").to_svg()
    messages = [str(w.message) for w in caught]
    assert len(messages) == 1
    assert messages[0].startswith("mark_ribbon: size 'lo' has no effect")

    with pytest.raises(ValueError) as excinfo:
        fe.Chart(df).mark_ribbon().encode(x="t", y="lo", y2="hi", color="nope").to_svg()
    assert str(excinfo.value) == "unknown column 'nope' referenced by an encoding"


def test_plus_mark_line_mark_ribbon_partial_ribbon_keeps_its_own_layer_index():
    drawn = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0], "g": ["a", "a", "a"]})
    partial = pl.DataFrame(
        {"x": [1.0, 2.0, 1.0], "lo": [1.0, 2.0, 1.0], "hi": [3.0, 4.0, 3.0], "g": ["a", "a", "b"]}
    )
    line = fe.Chart(drawn).mark_line().encode(x="x", y="y", color="g")
    ribbon = fe.Chart(partial).mark_ribbon().encode(x="x", y="lo", y2="hi", color="g")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        (line + ribbon).to_svg()
    messages = [str(w.message) for w in caught]
    expected = (
        "layer 1: mark_ribbon: mark_ribbon() spans two or more points per band, "
        "so it did not draw g=b (one point)"
    )
    assert messages == [expected]


def test_ribbon_aggregate_renders_identically_to_the_pre_aggregated_frame():
    """mark_ribbon's y/y2 honor aggregate=, as mark_area's do, since the
    chart's own encoding reaches the mark unrebuilt."""
    raw = pl.DataFrame(
        {
            "x": [1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
            "lo": [0.0, 1.0, 1.0, 2.0, 2.0, 4.0],
            "hi": [2.0, 3.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    pre_aggregated = raw.group_by("x", maintain_order=True).agg(
        pl.col("lo").mean(), pl.col("hi").mean()
    )
    aggregated_svg = (
        fe.Chart(raw)
        .mark_ribbon()
        .encode(x="x", y=fe.Y("lo", aggregate="mean", title="lo"), y2=fe.Y2("hi", aggregate="mean"))
        .to_svg()
    )
    pre_aggregated_svg = (
        fe.Chart(pre_aggregated)
        .mark_ribbon()
        .encode(x="x", y=fe.Y("lo", title="lo"), y2="hi")
        .to_svg()
    )
    assert aggregated_svg == pre_aggregated_svg


def test_ribbon_interactive_tooltip_carries_caller_titles_and_color():
    """The interactive scene of a titled, colored ribbon carries the
    caller's axis titles and a color entry, as mark_area's does.

    Asserts on the tooltip's own structured field entries
    (``{"name":"XT",...}``), not a bare substring search: the scene embeds
    a WASM binary blob (base64) in which short needles like "XT"/"YT"/"g"
    turn up by sheer coincidence, matching whether or not the real tooltip
    carries them.
    """
    rows = pl.DataFrame(
        {
            "x": [1.0, 2.0, 1.0, 2.0],
            "lo": [1.0, 2.0, 1.0, 2.0],
            "hi": [2.0, 3.0, 2.0, 3.0],
            "g": ["a", "a", "b", "b"],
        }
    )
    chart = (
        fe.Chart(rows)
        .mark_ribbon()
        .encode(x=fe.X("x", title="XT"), y=fe.Y("lo", title="YT"), y2="hi", color="g")
    )
    html = chart.interactive().to_html()
    for needle in ('"name":"XT"', '"name":"YT"', '"name":"g"'):
        assert needle in html, needle
