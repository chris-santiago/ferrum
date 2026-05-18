from ferrum.themes import Theme


def test_theme_default_has_no_props():
    t = Theme()
    assert t._props == {}


def test_theme_with_kwargs_stores_them():
    t = Theme(background="#000", font_family="Inter")
    assert t._props == {"background": "#000", "font_family": "Inter"}


def test_theme_omits_none_values():
    t = Theme(background="#000", font_family=None)
    assert t._props == {"background": "#000"}


def test_theme_update_returns_new_theme_with_merged_props():
    t1 = Theme(background="#000")
    t2 = t1.update(font_family="Inter")
    assert t1._props == {"background": "#000"}
    assert t2._props == {"background": "#000", "font_family": "Inter"}
    assert t1 is not t2


def test_theme_update_overrides_existing_prop():
    t1 = Theme(background="#000")
    t2 = t1.update(background="#fff")
    assert t1._props == {"background": "#000"}
    assert t2._props == {"background": "#fff"}


def test_theme_eq_when_props_match():
    t1 = Theme(background="#000", font_family="Inter")
    t2 = Theme(font_family="Inter", background="#000")
    assert t1 == t2


def test_theme_to_spec_dict_passes_through_props():
    # "background" is normalised to "background_color" for the Rust binding.
    t = Theme(background="#1a1a2e", font_color="#e6e6e6")
    d = t.to_spec_dict()
    assert d["background_color"] == "#1a1a2e", "background must be renamed to background_color"
    assert "background" not in d, "raw 'background' key must not remain after normalisation"
    assert d["font_color"] == "#e6e6e6"


def test_theme_hashable():
    t = Theme(background="#000")
    s = {t}
    assert t in s


def test_eight_builtins_exist():
    from ferrum.themes import (
        default,
        minimal,
        dark,
        publication,
        economist,
        fivethirtyeight,
        solarized_light,
        solarized_dark,
    )

    for t in (
        default,
        minimal,
        dark,
        publication,
        economist,
        fivethirtyeight,
        solarized_light,
        solarized_dark,
    ):
        assert isinstance(t, Theme)


def test_default_theme_has_no_props():
    from ferrum.themes import default

    assert default._props == {}


def test_dark_theme_has_dark_background():
    from ferrum.themes import dark

    assert dark._props["background"] == "#1a1a2e"


def test_set_default_theme_sets_process_default():
    import ferrum
    from ferrum.themes import dark, get_default_theme
    from ferrum.themes._defaults import _default_theme

    # Save and restore to avoid bleed-through
    original = get_default_theme()
    try:
        ferrum.set_default_theme(dark)
        assert get_default_theme() is dark
    finally:
        _default_theme.set(original)


def test_set_default_theme_returns_context_manager():
    import ferrum
    from ferrum.themes import get_default_theme

    original = get_default_theme()
    with ferrum.set_default_theme(ferrum.themes.dark):
        assert get_default_theme() is ferrum.themes.dark
    assert get_default_theme() is original


def test_nested_set_default_theme_restores_correctly():
    import ferrum
    from ferrum.themes import get_default_theme

    original = get_default_theme()
    with ferrum.set_default_theme(ferrum.themes.dark):
        with ferrum.set_default_theme(ferrum.themes.minimal):
            assert get_default_theme() is ferrum.themes.minimal
        assert get_default_theme() is ferrum.themes.dark
    assert get_default_theme() is original


def test_chart_theme_attaches_theme_to_chart():
    import polars as pl
    from ferrum import Chart
    from ferrum.themes import dark

    df = pl.DataFrame({"a": [1], "b": [2]})
    c = Chart(df).mark_point().encode(x="a", y="b").theme(dark)
    assert c._theme is dark


def test_chart_theme_per_chart_overrides_default():
    import polars as pl
    from ferrum import Chart, set_default_theme
    from ferrum.themes import dark, minimal, get_default_theme

    df = pl.DataFrame({"a": [1], "b": [2]})
    with set_default_theme(dark):
        c = Chart(df).mark_point().encode(x="a", y="b").theme(minimal)
        # When show_svg is called, c's theme (minimal) is used, not dark.
        assert c._theme is minimal
        assert get_default_theme() is dark


def test_set_default_theme_affects_rendered_svg():
    """W14 regression: set_default_theme must change rendered SVG output."""
    import polars as pl
    from ferrum import Chart, set_default_theme
    from ferrum.themes import Theme

    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    bg_color = "#1a1a2e"
    with set_default_theme(Theme(background=bg_color)):
        svg = Chart(df).mark_point().encode(x="a", y="b").show_svg()
    assert bg_color in svg, (
        f"Expected background color {bg_color!r} in SVG from set_default_theme "
        "but it was not found. set_default_theme is disconnected from rendering."
    )


def test_theme_context_affects_rendered_svg():
    """W13 regression: theme_context must change rendered SVG output."""
    import polars as pl
    from ferrum import Chart
    from ferrum.themes import Theme, theme_context

    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    bg_color = "#2d2d44"
    with theme_context(Theme(background=bg_color)):
        svg = Chart(df).mark_point().encode(x="a", y="b").show_svg()
    assert bg_color in svg, (
        f"Expected background color {bg_color!r} in SVG from theme_context "
        "but it was not found. theme_context is disconnected from rendering."
    )


def test_per_chart_theme_overrides_default_in_svg():
    """Cascade: per-chart .theme() must override set_default_theme in SVG."""
    import polars as pl
    from ferrum import Chart, set_default_theme
    from ferrum.themes import Theme

    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    default_bg = "#1a1a2e"
    chart_bg = "#ffffff"
    with set_default_theme(Theme(background=default_bg)):
        svg = (
            Chart(df).mark_point().encode(x="a", y="b").theme(Theme(background=chart_bg)).show_svg()
        )
    assert chart_bg in svg, "Per-chart theme background must appear in SVG"
    assert default_bg not in svg, "Default theme background must be overridden"


def test_theme_strip_text_size_accepted():
    """B2: strip_text_size must be a valid Theme key."""
    from ferrum.themes import Theme

    t = Theme(strip_text_size=16.0)
    assert t._props == {"strip_text_size": 16.0}


def test_theme_strip_padding_accepted():
    """B3: strip_padding must be a valid Theme key."""
    from ferrum.themes import Theme

    t = Theme(strip_padding=10.0)
    assert t._props == {"strip_padding": 10.0}


def test_strip_text_size_reaches_rust():
    """B2 end-to-end: strip_text_size must not be rejected by Rust serde."""
    import polars as pl
    from ferrum import Chart
    from ferrum.themes import Theme

    df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
    svg = Chart(df).mark_point().encode(x="a", y="b").theme(Theme(strip_text_size=16.0)).show_svg()
    assert svg.startswith("<svg")


def test_strip_padding_reaches_rust():
    """B3 end-to-end: strip_padding must not be rejected by Rust serde."""
    import polars as pl
    from ferrum import Chart
    from ferrum.themes import Theme

    df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
    svg = Chart(df).mark_point().encode(x="a", y="b").theme(Theme(strip_padding=10.0)).show_svg()
    assert svg.startswith("<svg")


def test_theme_background_appears_in_rendered_svg():
    """BUG-1 regression: Theme.background must not be silently dropped by Rust."""
    import polars as pl
    from ferrum import Chart
    from ferrum.themes import Theme

    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    bg_color = "#1a1a2e"
    svg = Chart(df).mark_point().encode(x="a", y="b").theme(Theme(background=bg_color)).show_svg()
    assert bg_color in svg, (
        f"Expected background color {bg_color!r} in SVG but it was not found. "
        "Theme.background is being silently dropped by the Rust renderer."
    )
