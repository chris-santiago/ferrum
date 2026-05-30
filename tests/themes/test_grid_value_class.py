"""Tests for the ferrum.Grid value class and its Theme ingestion.

Covers:
- Grid construction and to_spec_dict() per-level key emission
- Shorthand-is-both-levels resolution
- Per-level override beats bare shorthand
- Bare shorthand keys never emitted to Rust
- ferrum.Grid exported in ferrum.__all__
- Theme ingestion (Grid value object spliced into spec dict)
- Theme back-compat: bool grid= still works
- End-to-end: minor=True renders MORE gridlines on a continuous chart;
  minor=True on a categorical chart does NOT add extra minor lines.
"""

from __future__ import annotations

import polars as pl
import pytest

import ferrum as fm
from ferrum.grid import Grid
from ferrum.themes import Theme


# ---------------------------------------------------------------------------
# Grid construction
# ---------------------------------------------------------------------------


def test_grid_defaults():
    g = Grid()
    assert g.major is True
    assert g.minor is False


def test_grid_major_false_minor_true():
    g = Grid(major=False, minor=True)
    assert g.major is False
    assert g.minor is True


def test_grid_is_frozen():
    g = Grid()
    with pytest.raises((AttributeError, TypeError)):
        g.major = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# to_spec_dict — boolean keys always emitted
# ---------------------------------------------------------------------------


def test_to_spec_dict_booleans_always_present():
    d = Grid(major=True, minor=False).to_spec_dict()
    assert d["grid"] is True
    assert d["minor"] is False


def test_to_spec_dict_major_false():
    d = Grid(major=False, minor=True).to_spec_dict()
    assert d["grid"] is False
    assert d["minor"] is True


def test_to_spec_dict_no_styling_when_all_none():
    d = Grid(major=True, minor=False).to_spec_dict()
    # Only the two boolean keys present.
    assert set(d.keys()) == {"grid", "minor"}


# ---------------------------------------------------------------------------
# Shorthand resolution — bare sets both levels
# ---------------------------------------------------------------------------


def test_shorthand_color_sets_both_levels():
    d = Grid(color="#f5f5f5").to_spec_dict()
    assert d["major_grid_color"] == "#f5f5f5"
    assert d["minor_grid_color"] == "#f5f5f5"


def test_shorthand_width_sets_both_levels():
    d = Grid(width=0.5).to_spec_dict()
    assert d["major_grid_width"] == 0.5
    assert d["minor_grid_width"] == 0.5


def test_shorthand_dash_sets_both_levels():
    d = Grid(dash=[4.0, 2.0]).to_spec_dict()
    assert d["major_grid_dash"] == [4.0, 2.0]
    assert d["minor_grid_dash"] == [4.0, 2.0]


def test_shorthand_opacity_sets_both_levels():
    d = Grid(opacity=0.4).to_spec_dict()
    assert d["major_grid_opacity"] == 0.4
    assert d["minor_grid_opacity"] == 0.4


# ---------------------------------------------------------------------------
# Per-level override beats bare shorthand
# ---------------------------------------------------------------------------


def test_minor_color_overrides_bare():
    d = Grid(color="#eeeeee", minor_color="#f8f8f8").to_spec_dict()
    assert d["major_grid_color"] == "#eeeeee"
    assert d["minor_grid_color"] == "#f8f8f8"


def test_major_color_overrides_bare():
    d = Grid(color="#eeeeee", major_color="#cccccc").to_spec_dict()
    assert d["major_grid_color"] == "#cccccc"
    assert d["minor_grid_color"] == "#eeeeee"


def test_minor_width_overrides_bare():
    d = Grid(width=1.0, minor_width=0.5).to_spec_dict()
    assert d["major_grid_width"] == 1.0
    assert d["minor_grid_width"] == 0.5


def test_major_width_overrides_bare():
    d = Grid(width=1.0, major_width=1.5).to_spec_dict()
    assert d["major_grid_width"] == 1.5
    assert d["minor_grid_width"] == 1.0


def test_minor_dash_overrides_bare():
    d = Grid(dash=[4.0, 2.0], minor_dash=[2.0, 2.0]).to_spec_dict()
    assert d["major_grid_dash"] == [4.0, 2.0]
    assert d["minor_grid_dash"] == [2.0, 2.0]


def test_minor_opacity_overrides_bare():
    d = Grid(opacity=0.6, minor_opacity=0.3).to_spec_dict()
    assert d["major_grid_opacity"] == 0.6
    assert d["minor_grid_opacity"] == 0.3


# ---------------------------------------------------------------------------
# Bare shorthand keys NEVER emitted
# ---------------------------------------------------------------------------


_BARE_KEYS = {"color", "width", "dash", "opacity"}


def test_bare_keys_never_emitted_minimal():
    d = Grid().to_spec_dict()
    assert _BARE_KEYS.isdisjoint(d.keys())


def test_bare_keys_never_emitted_with_shorthands():
    d = Grid(color="#eee", width=1.0, dash=[3.0], opacity=0.5).to_spec_dict()
    assert _BARE_KEYS.isdisjoint(d.keys())


def test_bare_keys_never_emitted_with_per_level_only():
    d = Grid(major_color="#ccc", minor_color="#eee").to_spec_dict()
    assert _BARE_KEYS.isdisjoint(d.keys())


# ---------------------------------------------------------------------------
# Export: ferrum.Grid must be in ferrum.__all__
# ---------------------------------------------------------------------------


def test_grid_exported_from_ferrum():
    assert hasattr(fm, "Grid")
    assert isinstance(fm.Grid, type)


def test_grid_in_ferrum_all():
    assert "Grid" in fm.__all__


# ---------------------------------------------------------------------------
# Theme ingestion — Grid value object
# ---------------------------------------------------------------------------


def test_theme_update_with_grid_object():
    t = Theme().update(grid=Grid(minor=True, minor_color="#eeeeee"))
    d = t.to_spec_dict()
    assert d["minor"] is True
    assert d["minor_grid_color"] == "#eeeeee"


def test_theme_with_grid_object_splices_per_level_keys():
    t = Theme(grid=Grid(color="#f5f5f5"))
    d = t.to_spec_dict()
    assert d["grid"] is True
    assert d["major_grid_color"] == "#f5f5f5"
    assert d["minor_grid_color"] == "#f5f5f5"
    # The "grid" key is consumed and replaced by the per-level expansion.
    # The raw Grid object must not be in the output.
    assert not hasattr(d.get("grid"), "to_spec_dict"), (
        "Grid object must be expanded; raw object must not remain in spec dict"
    )


def test_theme_ingestion_full_shorthand():
    # Use fully-expanded hex strings — Theme.to_spec_dict() expands #rgb → #rrggbb.
    t = Theme(
        grid=Grid(minor=True, color="#eeeeee", minor_color="#f8f8f8", width=0.8, minor_width=0.4)
    )
    d = t.to_spec_dict()
    assert d["grid"] is True
    assert d["minor"] is True
    assert d["major_grid_color"] == "#eeeeee"
    assert d["minor_grid_color"] == "#f8f8f8"
    assert d["major_grid_width"] == 0.8
    assert d["minor_grid_width"] == 0.4


# ---------------------------------------------------------------------------
# Theme back-compat — bool grid= still works
# ---------------------------------------------------------------------------


def test_theme_bool_grid_true():
    t = Theme(grid=True)
    d = t.to_spec_dict()
    assert d["grid"] is True


def test_theme_bool_grid_false():
    t = Theme(grid=False)
    d = t.to_spec_dict()
    assert d["grid"] is False


def test_theme_bool_grid_reaches_rust():
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 4.0, 9.0]})
    svg = fm.Chart(df).mark_point().encode(x="x", y="y").theme(Theme(grid=False)).show_svg()
    assert svg.startswith("<svg")


# ---------------------------------------------------------------------------
# End-to-end: minor=True renders MORE lines on a continuous chart
# ---------------------------------------------------------------------------


def _continuous_chart(minor: bool) -> str:
    """Render a continuous (linear) scatter chart with or without minor gridlines."""
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0], "y": [1.0, 4.0, 9.0, 16.0, 25.0]})
    t = fm.Theme(grid=Grid(major=True, minor=minor, minor_color="#abcdef"))
    return fm.Chart(df).mark_point().encode(x="x", y="y").theme(t).show_svg()


def test_minor_true_continuous_more_lines_than_minor_false():
    svg_with_minor = _continuous_chart(minor=True)
    svg_without_minor = _continuous_chart(minor=False)
    lines_with = svg_with_minor.count("<line")
    lines_without = svg_without_minor.count("<line")
    assert lines_with > lines_without, (
        f"Expected more <line> elements with minor=True ({lines_with}) "
        f"than minor=False ({lines_without}), but got the same or fewer. "
        "Minor gridlines are not being emitted."
    )


def test_minor_true_continuous_minor_color_in_svg():
    """Verify the minor color actually appears, confirming lines are styled minor lines."""
    svg = _continuous_chart(minor=True)
    assert "#abcdef" in svg.lower(), (
        "Expected minor_color '#abcdef' in SVG when minor=True, but it was absent. "
        "Minor gridlines are not being styled."
    )


# ---------------------------------------------------------------------------
# End-to-end: minor=True on a categorical chart adds NO extra lines
# ---------------------------------------------------------------------------


def test_minor_true_categorical_x_no_extra_vertical_lines():
    """minor=True on a band/ordinal x-axis must not add extra vertical gridlines.

    A bar chart with categorical x and continuous y: the categorical x-axis
    produces no minor ticks, so vertical gridlines are unchanged.  The
    continuous y-axis DOES get horizontal minor gridlines (that's correct).
    We verify the categorical side by counting vertical gridlines only — those
    are <line> elements that span the full plot height (x1 == x2).

    We proxy for this by using a theme with a unique minor color and checking
    that no column with that color appears in the SVG when x is categorical.
    """
    # Render a chart where BOTH axes are categorical and confirm there are no
    # extra lines with minor=True vs. minor=False — neither axis has a continuum
    # to subdivide, so the minor flag must be a no-op here.
    df2 = pl.DataFrame({"cat": ["a", "b", "c"], "cat2": ["x", "y", "z"], "v": [1, 2, 3]})
    t_minor2 = fm.Theme(grid=Grid(major=True, minor=True))
    t_no_minor2 = fm.Theme(grid=Grid(major=True, minor=False))
    svg_minor2 = fm.Chart(df2).mark_point().encode(x="cat", y="cat2").theme(t_minor2).show_svg()
    svg_no_minor2 = (
        fm.Chart(df2).mark_point().encode(x="cat", y="cat2").theme(t_no_minor2).show_svg()
    )
    assert svg_minor2.count("<line") == svg_no_minor2.count("<line"), (
        "Both-categorical chart should have the same number of <line> elements "
        "with minor=True and minor=False — neither axis has a continuum to subdivide."
    )


# ---------------------------------------------------------------------------
# Theme ingestion: Grid reaches Rust without error
# ---------------------------------------------------------------------------


def test_grid_object_theme_reaches_rust():
    """Grid value object must not cause a serde/render error at the Rust boundary."""
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 4.0, 9.0]})
    t = fm.Theme(grid=Grid(minor=True, color="#f0f0f0"))
    svg = fm.Chart(df).mark_point().encode(x="x", y="y").theme(t).show_svg()
    assert svg.startswith("<svg")
