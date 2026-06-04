"""Chart-level title rendering (Themes-T2.5a)."""

import polars as pl

import ferrum as fm


def test_title_renders_as_text_element() -> None:
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    chart = fm.Chart(df).mark_point().encode(x="x", y="y").properties(title="My Title")
    svg = chart.to_svg()
    assert ">My Title<" in svg


def test_no_title_renders_no_extra_text() -> None:
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    svg_with = fm.Chart(df).mark_point().encode(x="x", y="y").properties(title="X").to_svg()
    svg_no = fm.Chart(df).mark_point().encode(x="x", y="y").to_svg()
    # Title-bearing SVG has more <text> elements.
    assert svg_with.count("<text") > svg_no.count("<text")


def test_title_anchor_start_emits_text_anchor_start() -> None:
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    svg = (
        fm.Chart(df)
        .mark_point()
        .encode(x="x", y="y")
        .properties(title="Hi")
        .theme(fm.Theme(title_anchor="start"))
        .to_svg()
    )
    # Find the title text element specifically: it's the one containing 'Hi'.
    # The text-anchor attr on that element should be 'start'.
    # Loose check: SVG contains both the title and 'text-anchor="start"' close to it.
    assert ">Hi<" in svg
    # The title text is emitted with text-anchor="start" when anchor=start.
    # We check the substring around 'Hi'.
    idx = svg.find(">Hi<")
    # Look backwards for the most recent <text element.
    text_start = svg.rfind("<text", 0, idx)
    assert text_start >= 0
    title_element = svg[text_start:idx]
    assert 'text-anchor="start"' in title_element


def test_title_anchor_middle_emits_text_anchor_middle() -> None:
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    svg = (
        fm.Chart(df)
        .mark_point()
        .encode(x="x", y="y")
        .properties(title="Hi")
        .theme(fm.Theme(title_anchor="middle"))
        .to_svg()
    )
    idx = svg.find(">Hi<")
    text_start = svg.rfind("<text", 0, idx)
    title_element = svg[text_start:idx]
    assert 'text-anchor="middle"' in title_element


def test_title_uses_title_color() -> None:
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    svg = (
        fm.Chart(df)
        .mark_point()
        .encode(x="x", y="y")
        .properties(title="Hi")
        .theme(fm.Theme(title_color="#ff0000"))
        .to_svg()
    )
    idx = svg.find(">Hi<")
    text_start = svg.rfind("<text", 0, idx)
    title_element = svg[text_start:idx]
    assert "#ff0000" in title_element.lower()


def test_title_uses_title_font_weight_when_not_normal() -> None:
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    svg = (
        fm.Chart(df)
        .mark_point()
        .encode(x="x", y="y")
        .properties(title="Hi")
        .theme(fm.Theme(title_font_weight="bold"))
        .to_svg()
    )
    idx = svg.find(">Hi<")
    text_start = svg.rfind("<text", 0, idx)
    title_element = svg[text_start:idx]
    assert 'font-weight="bold"' in title_element
