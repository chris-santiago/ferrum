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
    chart = fe.Chart(df).mark_ribbon().encode(x="t", y="lo", y2="hi")
    spec = chart._build_spec()
    assert len(spec.layers) == 1


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
    all in the ribbon layer's ``mark_style``, reconstructed by stripping the
    key from the post-fix JSON and re-parsing via ``ChartSpec.from_json``)
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
    layer_style = spec_dict["layers"][0]["mark_style"]
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
    """An out-of-vocabulary interpolate= value must surface Rust's typed
    RenderError::UnknownInterpolate as a Python ValueError — desugar_ribbon
    must NOT duplicate the vocabulary check in Python."""
    chart = fe.Chart(df).mark_ribbon(interpolate="smooth").encode(x="t", y="lo", y2="hi")
    with pytest.raises(ValueError, match="smooth"):
        chart.to_svg()
