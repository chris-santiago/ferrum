import polars as pl
import pytest
import ferrum as fe


@pytest.fixture
def df():
    return pl.DataFrame(
        {
            "group": ["a", "a", "a", "b", "b", "b"],
            "value": [1.0, 2.0, 100.0, 4.0, 5.0, 6.0],
        }
    )


def test_boxplot_smoke_6_layers(df):
    chart = fe.Chart(df).mark_boxplot().encode(x="group", y="value")
    spec = chart._build_spec()
    assert spec.layers is not None
    assert len(spec.layers) == 6  # rule + lower cap + upper cap + rect + tick + outliers point


def test_boxplot_no_outliers_5_layers(df):
    chart = fe.Chart(df).mark_boxplot(outliers=False).encode(x="group", y="value")
    spec = chart._build_spec()
    assert len(spec.layers) == 5  # rule + lower cap + upper cap + rect + tick


def test_boxplot_extent_min_max(df):
    chart = fe.Chart(df).mark_boxplot(extent="min-max").encode(x="group", y="value")
    json_str = chart._build_spec().to_json()
    assert "min-max" in json_str


def test_boxplot_horizontal_swaps_x_y(df):
    chart = fe.Chart(df).mark_boxplot(horizontal=True).encode(x="value", y="group")
    spec = chart._build_spec()
    json_str = spec.to_json()
    assert "lower_whisker" in json_str  # whisker layer present
    assert '"q1"' in json_str  # box stats present


def test_boxplot_render_smoke(df):
    chart = fe.Chart(df).mark_boxplot().encode(x="group", y="value")
    svg = chart.to_svg()
    assert svg.startswith("<?xml") or svg.startswith("<svg")


def test_boxplot_width_sets_box_band_size(df):
    """mark_boxplot(width=0.3) should control box width (maps to size/band_size)."""
    # width=0.3 should produce a narrower box than the default (0.6).
    chart_narrow = fe.Chart(df).mark_boxplot(width=0.3).encode(x="group", y="value")
    chart_default = fe.Chart(df).mark_boxplot().encode(x="group", y="value")

    # Should render without error.
    svg_narrow = chart_narrow.to_svg()
    svg_default = chart_default.to_svg()
    assert "<svg" in svg_narrow or svg_narrow.startswith("<?xml"), (
        "width=0.3 should produce valid SVG"
    )

    # The box layer uses band_size; narrow (0.3) should produce smaller box rects
    # than the default (0.6). Compare the minimum width of <rect> elements.
    import re

    def _rect_widths(svg: str) -> list:
        return [float(m) for m in re.findall(r'<rect[^>]+width="([^"]+)"', svg)]

    widths_narrow = _rect_widths(svg_narrow)
    widths_default = _rect_widths(svg_default)
    assert widths_narrow, "narrow boxplot should contain <rect> elements"
    assert widths_default, "default boxplot should contain <rect> elements"
    assert min(widths_narrow) < min(widths_default), (
        f"width=0.3 should produce narrower boxes than default (0.6); "
        f"narrow_min={min(widths_narrow):.2f}, default_min={min(widths_default):.2f}"
    )
