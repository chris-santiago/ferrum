import pytest

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
        # When to_svg is called, c's theme (minimal) is used, not dark.
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
        svg = Chart(df).mark_point().encode(x="a", y="b").to_svg()
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
        svg = Chart(df).mark_point().encode(x="a", y="b").to_svg()
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
        svg = Chart(df).mark_point().encode(x="a", y="b").theme(Theme(background=chart_bg)).to_svg()
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
    svg = Chart(df).mark_point().encode(x="a", y="b").theme(Theme(strip_text_size=16.0)).to_svg()
    assert svg.startswith("<svg")


def test_strip_padding_reaches_rust():
    """B3 end-to-end: strip_padding must not be rejected by Rust serde."""
    import polars as pl
    from ferrum import Chart
    from ferrum.themes import Theme

    df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
    svg = Chart(df).mark_point().encode(x="a", y="b").theme(Theme(strip_padding=10.0)).to_svg()
    assert svg.startswith("<svg")


def test_theme_background_appears_in_rendered_svg():
    """BUG-1 regression: Theme.background must not be silently dropped by Rust."""
    import polars as pl
    from ferrum import Chart
    from ferrum.themes import Theme

    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    bg_color = "#1a1a2e"
    svg = Chart(df).mark_point().encode(x="a", y="b").theme(Theme(background=bg_color)).to_svg()
    assert bg_color in svg, (
        f"Expected background color {bg_color!r} in SVG but it was not found. "
        "Theme.background is being silently dropped by the Rust renderer."
    )


# --- T2.1 (D-THEME-1): Python↔Rust theme-key contract is one source of truth ---


def _sample_for(key: str):
    """Return a type- and domain-valid sample value for a theme key.

    Mirrors the Rust ``sample_for`` type categories (binding.rs) so the value
    survives both ``Theme(...)`` construction and the Rust ``apply_theme_overrides``
    arm (enum keys get a valid variant, not the generic ``"x"`` placeholder).
    """
    dash_keys = {"grid_dash", "major_grid_dash", "minor_grid_dash", "reference_line_dash"}
    bool_keys = {"axis_line", "grid", "minor"}
    f64_keys = {
        "padding",
        "font_size",
        "title_font_size",
        "title_offset",
        "axis_line_width",
        "tick_size",
        "tick_width",
        "grid_width",
        "grid_opacity",
        "major_grid_width",
        "minor_grid_width",
        "major_grid_opacity",
        "minor_grid_opacity",
        "point_size",
        "point_opacity",
        "line_stroke_width",
        "bar_corner_radius",
        "area_opacity",
        "opacity",
        "strip_text_size",
        "strip_padding",
        "legend_title_font_size",
        "axis_title_padding",
        "column_padding",
        "row_padding",
    }
    # Enum-valued keys validated by apply_theme_overrides — need real variants.
    enum_samples = {
        "title_anchor": "middle",
        "legend_orient": "right",
        "legend_direction": "vertical",
        "color_scheme": "tableau10",
        "sequential_scheme": "viridis",
        "diverging_scheme": "rdbu",
    }
    from ferrum._core import theme_color_keys

    if key in dash_keys:
        return [2.0, 1.0]
    if key in bool_keys:
        return True
    if key == "cull_threshold":
        return 7
    if key in f64_keys:
        return 1.5
    if key in enum_samples:
        return enum_samples[key]
    if key in frozenset(theme_color_keys()):
        return "#123456"
    return "x"


def test_known_keys_derived_from_rust_manifest():
    """THEME-01: ``Theme._KNOWN_KEYS`` is the Rust ``theme_known_keys()`` set."""
    from ferrum._core import theme_known_keys

    assert Theme._KNOWN_KEYS == frozenset(theme_known_keys())


@pytest.mark.parametrize("key", sorted(Theme._KNOWN_KEYS))
def test_every_rust_known_key_accepted_by_theme(key):
    """THEME-01 parity: every ``theme_known_keys()`` key constructs a ``Theme``."""
    t = Theme(**{key: _sample_for(key)})
    assert key in t._props


def test_unknown_theme_key_rejected():
    """THEME-01: a key not in the Rust manifest still raises ValueError."""
    with pytest.raises(ValueError, match="Unknown Theme key"):
        Theme(definitely_not_a_theme_key="x")


@pytest.mark.parametrize(
    "key",
    [
        "major_grid_color",
        "minor_grid_color",
        "major_grid_width",
        "minor_grid_width",
        "major_grid_dash",
        "minor_grid_dash",
        "major_grid_opacity",
        "minor_grid_opacity",
        "minor",
    ],
)
def test_per_level_grid_keys_valid_in_theme(key):
    """THEME-01 drift fix: per-level grid keys now validate in ``Theme(...)``.

    Before deriving ``_KNOWN_KEYS`` from Rust these raised ``ValueError`` in
    Python even though Rust accepted them.
    """
    t = Theme(**{key: _sample_for(key)})
    assert t._props[key] == _sample_for(key)


def test_every_known_key_reaches_rust():
    """THEME-01 end-to-end: every theme key renders without a Rust serde rejection."""
    import polars as pl
    from ferrum import Chart

    props = {key: _sample_for(key) for key in Theme._KNOWN_KEYS}
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    svg = Chart(df).mark_point().encode(x="a", y="b").theme(Theme(**props)).to_svg()
    assert svg.startswith("<svg")


def test_theme_color_keys_subset_of_known():
    """THEME-05: the Rust color-key subset stays inside the known-key set."""
    from ferrum._core import theme_color_keys

    assert frozenset(theme_color_keys()) <= Theme._KNOWN_KEYS


def test_css_shorthand_hex_forwarded_to_rust_unchanged():
    """THEME-05: Python no longer expands shorthand hex; Rust handles it.

    ``to_spec_dict`` forwards the color string verbatim (the renderer's
    ``from_hex_str`` expands ``#rgb`` / ``#rgba``).
    """
    d = Theme(background="#abc", font_color="#1234").to_spec_dict()
    assert d["background_color"] == "#abc"
    assert d["font_color"] == "#1234"


def test_css_shorthand_hex_renders_correctly():
    """THEME-05 end-to-end: shorthand hex theme color reaches a full-length fill."""
    import polars as pl
    from ferrum import Chart

    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    svg = Chart(df).mark_point().encode(x="a", y="b").theme(Theme(background="#abc")).to_svg()
    # Rust from_hex_str expands #abc -> #aabbcc.
    assert "#aabbcc" in svg


# --- THEME-07: the title/label -> body-text fallback chain is complete ---


def test_fallbacks_cover_every_title_label_key_with_body_counterpart():
    """THEME-07: every ``title_*`` / ``label_*`` key with a body-text counterpart
    has a ``_FALLBACKS`` entry, and only those do."""
    expected = {
        "title_color": "font_color",
        "label_color": "font_color",
        "title_font_family": "font_family",
        "label_font_family": "font_family",
        "title_font_weight": "font_weight",
        "title_font_size": "font_size",
    }
    assert Theme._FALLBACKS == expected


@pytest.mark.parametrize(
    "derived,source,value",
    [
        ("title_color", "font_color", "#123456"),
        ("label_color", "font_color", "#123456"),
        ("title_font_family", "font_family", "Serif"),
        ("label_font_family", "font_family", "Serif"),
        ("title_font_weight", "font_weight", "bold"),
        ("title_font_size", "font_size", 18.0),
    ],
)
def test_unset_title_label_key_falls_back_to_body_counterpart(derived, source, value):
    """THEME-07: an unset title/label key resolves to its body-text counterpart."""
    d = Theme(**{source: value}).to_spec_dict()
    assert d[derived] == value


def test_explicit_title_label_key_wins_over_fallback():
    """THEME-07: an explicit title/label value is not overwritten by the fallback."""
    d = Theme(font_weight="bold", title_font_weight="normal").to_spec_dict()
    assert d["title_font_weight"] == "normal"


def test_theme_docstring_fallback_list_matches_table():
    """THEME-07: the docstring's fallback pair list is generated from ``_FALLBACKS``."""
    doc = Theme.__doc__ or ""
    assert "%(fallbacks)s" not in doc, "docstring placeholder was not filled at import"
    for derived, source in Theme._FALLBACKS.items():
        assert f"``{derived}`` → ``{source}``" in doc
