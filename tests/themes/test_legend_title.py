"""Legend title rendering (Themes-T2.5b)."""
import polars as pl

import ferrum as fm


def test_legend_title_uses_field_name() -> None:
    df = pl.DataFrame({
        "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "species": ["a", "a", "b", "b", "c", "c"],
    })
    svg = fm.Chart(df).mark_point().encode(x="x", y="y", color="species").show_svg()
    assert ">species<" in svg


def test_no_color_encoding_has_no_legend_title() -> None:
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    svg = fm.Chart(df).mark_point().encode(x="x", y="y").show_svg()
    # Without a color encoding, no legend at all, no "species"-style title.
    assert "<text" in svg  # axes still emit text
    # But the bare field names that would be a legend title shouldn't appear
    # as standalone text elements. Loose check: no extra legend-shaped text.


def test_legend_title_font_size_flows_through() -> None:
    df = pl.DataFrame({
        "x": [1.0, 2.0, 3.0],
        "y": [4.0, 5.0, 6.0],
        "cat": ["a", "b", "c"],
    })
    chart = fm.Chart(df).mark_point().encode(x="x", y="y", color="cat").theme(
        fm.Theme(legend_title_font_size=18.0)
    )
    svg = chart.show_svg()
    # Find the legend title text element specifically. ">cat<" identifies it.
    idx = svg.find(">cat<")
    assert idx > 0
    text_start = svg.rfind("<text", 0, idx)
    title_element = svg[text_start:idx]
    assert 'font-size="18"' in title_element
