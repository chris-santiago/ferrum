"""Feature tests for ``clustermap``'s cell border (#125).

``clustermap``'s center heatmap had no ``linewidths=``/``linecolor=``
parameters at all and drew its cells with a bare ``.mark_rect()`` -- no
border, ever -- while ``heatmap`` drew its own cells with a white 0.5
default border. #125 is that drift: the two siblings disagreed about a
default neither owned on its own. The fix is one helper,
``ferrum.plots.matrix._resolve_cell_border``, that owns the default (0.5,
white), the ``None``-means-default rule, and the validation voice for both
figures; ``heatmap``'s own tests live in ``test_heatmap_appearance.py``.

This module pins:

- the default border (white, 0.5) landing on every center cell, with the
  stroked-cell count equal to the cell count;
- ``linewidths=0`` disabling the border entirely;
- explicit ``linewidths=``/``linecolor=`` values reaching the center cells;
- ``None`` (the declared default on both parameters) resolving to the same
  default as leaving them unset, per cell, including each parameter left
  ``None`` alone;
- malformed ``linewidths=``/``linecolor=`` refusing with the exact text
  ``heatmap`` refuses with, voiced as ``clustermap`` and naming the
  argument, never ``stroke``;
- the ``"none"``/``"transparent"``/``"theme:label"`` paint sentinels
  ``linecolor=`` accepts -- the same vocabulary ``mark_rect(stroke=)``
  does (#125);
- an explicit drawing value made inert by the other switch's off state
  (a clearing ``linecolor`` with an explicit ``linewidths``, or an
  explicit drawing ``linecolor`` with ``linewidths=0``) warning with the
  exact text, naming both arguments; agreeing off switches and any
  ``None`` warning nothing;
- ``mark=`` still overriding the border default on the cells, including
  over a clearing ``linecolor``;
- the colorbar frame and the dendrogram lines staying unaffected by every
  variation above -- the border is a center-cell-only concern.
"""

from __future__ import annotations

import sys
import warnings

import polars as pl
import pytest

import ferrum as fm
from tests._svg_rects import svg_elements


def _wide_df() -> pl.DataFrame:
    return pl.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0], "c": [7.0, 8.0, 9.0]})


def _cell_rects(svg: str) -> list[dict[str, str]]:
    """Return the center heatmap's own cell ``<rect>``s: solid-fill rects
    carrying their own stroke, excluding the colorbar frame (a
    ``url(...)`` gradient fill, stroked independently of this helper)."""
    return svg_elements(
        svg, "rect", lambda a: "stroke" in a and not a.get("fill", "").startswith("url(")
    )


def _colorbar_frame(svg: str) -> dict[str, str]:
    frames = svg_elements(svg, "rect", lambda a: a.get("fill", "").startswith("url("))
    assert len(frames) == 1, "expected exactly one colorbar frame rect"
    return frames[0]


def _dendrogram_lines(svg: str) -> list[dict[str, str]]:
    return svg_elements(svg, "line", lambda _a: True)


def _theme_axis_color(svg: str) -> str:
    """Read the theme's resolved axis/label color from this render's own
    axis chrome, never hard-coded -- mirrors
    ``tests/test_heatmap_appearance.py``'s helper of the same name. The
    center heatmap's axis/tick lines render at ``stroke-width="1"``,
    distinct from the ``"0.5"`` gridlines and dendrogram lines."""
    axis_strokes = {
        a["stroke"] for a in svg_elements(svg, "line", lambda a: a.get("stroke-width") == "1")
    }
    assert len(axis_strokes) == 1, f"expected exactly one axis chrome color, got {axis_strokes}"
    return axis_strokes.pop()


# ---------------------------------------------------------------------------
# Default border
# ---------------------------------------------------------------------------


def test_clustermap_default_borders_every_cell_white_half_pixel():
    df = _wide_df()
    svg = fm.clustermap(df).to_svg()
    cells = _cell_rects(svg)
    assert len(cells) == 9  # 3 rows x 3 columns
    assert all(c["stroke"] == "#ffffff" for c in cells)
    assert all(c["stroke-width"] == "0.5" for c in cells)


def test_clustermap_default_border_leaves_colorbar_and_dendrograms_unaffected():
    df = _wide_df()
    svg = fm.clustermap(df).to_svg()
    frame = _colorbar_frame(svg)
    assert frame["stroke"] == "#6b7280"
    assert frame["stroke-width"] == "0.5"
    assert len(_dendrogram_lines(svg)) == 32


# ---------------------------------------------------------------------------
# linewidths=0 disables the border
# ---------------------------------------------------------------------------


def test_clustermap_linewidths_zero_disables_cell_border():
    df = _wide_df()
    svg = fm.clustermap(df, linewidths=0).to_svg()
    assert _cell_rects(svg) == []
    # The colorbar frame carries its own, unrelated stroke and is untouched.
    frame = _colorbar_frame(svg)
    assert frame["stroke"] == "#6b7280"


def test_clustermap_linewidths_zero_leaves_dendrograms_and_colorbar_unchanged():
    """Disabling the cell border must not perturb the dendrogram lines or
    the colorbar frame -- the border is a center-cell-only concern."""
    df = _wide_df()
    with_border = fm.clustermap(df).to_svg()
    borderless = fm.clustermap(df, linewidths=0).to_svg()
    assert _dendrogram_lines(with_border) == _dendrogram_lines(borderless)
    assert _colorbar_frame(with_border) == _colorbar_frame(borderless)


# ---------------------------------------------------------------------------
# Explicit linewidths=/linecolor=
# ---------------------------------------------------------------------------


def test_clustermap_explicit_linewidths_and_linecolor_reach_the_cells():
    df = _wide_df()
    svg = fm.clustermap(df, linewidths=1.5, linecolor="black").to_svg()
    cells = _cell_rects(svg)
    assert len(cells) == 9
    assert all(c["stroke"] == "#000000" for c in cells)
    assert all(c["stroke-width"] == "1.5" for c in cells)


# ---------------------------------------------------------------------------
# None (the declared default) and malformed values -- same rule as heatmap,
# voiced as clustermap
# ---------------------------------------------------------------------------


def test_clustermap_none_linewidths_and_linecolor_resolve_to_the_default():
    df = _wide_df()
    svg = fm.clustermap(df, linewidths=None, linecolor=None).to_svg()
    cells = _cell_rects(svg)
    assert len(cells) == 9
    assert all(c["stroke"] == "#ffffff" for c in cells)
    assert all(c["stroke-width"] == "0.5" for c in cells)


def test_clustermap_linewidths_none_alone_resolves_to_default_width():
    """`linewidths=None` passed alone (`linecolor` left at its own default,
    not even passed) resolves to the 0.5 default width -- pinned per cell,
    separately from the both-None case above."""
    df = _wide_df()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        svg = fm.clustermap(df, linewidths=None).to_svg()
    cells = _cell_rects(svg)
    assert len(cells) == 9
    assert all(c["stroke"] == "#ffffff" for c in cells)
    assert all(c["stroke-width"] == "0.5" for c in cells)


def test_clustermap_linecolor_none_with_explicit_width_resolves_to_default_color():
    """`linecolor=None` combined with an explicit `linewidths` resolves to
    the white default at the caller's explicit width, with no warning
    (`None` is unset, never inert) -- the sibling of
    ``test_heatmap_linecolor_none_with_explicit_width_resolves_to_default_color``."""
    df = _wide_df()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        svg = fm.clustermap(df, linewidths=2.0, linecolor=None).to_svg()
    cells = _cell_rects(svg)
    assert len(cells) == 9
    assert all(c["stroke"] == "#ffffff" for c in cells)
    assert all(c["stroke-width"] == "2" for c in cells)


# ---------------------------------------------------------------------------
# linecolor paint sentinels ("none"/"transparent"/"theme:label") -- the same
# vocabulary `mark_rect(stroke=)` accepts, mirroring
# ``test_heatmap_appearance.py``'s coverage for ``clustermap`` (#125).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spelling", ["none", "transparent", " NONE ", "Transparent"])
def test_clustermap_clearing_linecolor_matches_linewidths_zero_and_warns_nothing(spelling):
    df = _wide_df()
    zero = fm.clustermap(df, linewidths=0).to_svg()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        svg = fm.clustermap(df, linecolor=spelling).to_svg()
    assert svg == zero
    assert caught == []


def test_clustermap_theme_label_linecolor_renders_the_theme_axis_color():
    """`linecolor="theme:label"` is forwarded unchanged to the mark, which
    resolves it at render time -- pinned against this same render's own
    axis chrome (`_theme_axis_color`), never a hard-coded hex."""
    df = _wide_df()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        svg = fm.clustermap(df, linecolor="theme:label").to_svg()
    expected = _theme_axis_color(svg)
    cells = _cell_rects(svg)
    assert len(cells) == 9
    assert all(c["stroke"] == expected for c in cells)
    assert all(c["stroke-width"] == "0.5" for c in cells)


# ---------------------------------------------------------------------------
# Inert pairs -- an explicit drawing value made inert by the other switch's
# off state warns, naming both arguments. `None` on either side is unset,
# never inert; two off switches that agree do not warn.
# ---------------------------------------------------------------------------


def test_clustermap_explicit_linewidths_inert_under_clearing_linecolor_warns_exact_text():
    df = _wide_df()
    with pytest.warns(UserWarning) as record:
        fm.clustermap(df, linewidths=2, linecolor="none")
    assert len(record) == 1
    assert str(record[0].message) == (
        "clustermap: linewidths=2.0 has no effect when linecolor='none' draws no "
        "cell border; it was ignored"
    )


def test_clustermap_explicit_linecolor_inert_under_linewidths_zero_warns_exact_text():
    df = _wide_df()
    with pytest.warns(UserWarning) as record:
        fm.clustermap(df, linewidths=0, linecolor="red")
    assert len(record) == 1
    assert str(record[0].message) == (
        "clustermap: linecolor='red' has no effect when linewidths=0.0 draws no "
        "cell border; it was ignored"
    )


def test_clustermap_agreeing_off_switches_warn_nothing():
    df = _wide_df()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fm.clustermap(df, linewidths=0, linecolor="none")
        fm.clustermap(df, linewidths=0, linecolor=None)


def test_clustermap_theme_label_inert_under_linewidths_zero_warns_exact_text():
    """`linecolor="theme:label"` is a drawing value (not a clearing
    sentinel), so it is inert under `linewidths=0` -- the clustermap
    sibling of ``test_heatmap_theme_label_inert_under_linewidths_zero_warns_exact_text``."""
    df = _wide_df()
    with pytest.warns(UserWarning) as record:
        fm.clustermap(df, linewidths=0, linecolor="theme:label")
    assert len(record) == 1
    assert str(record[0].message) == (
        "clustermap: linecolor='theme:label' has no effect when linewidths=0.0 "
        "draws no cell border; it was ignored"
    )


# ---------------------------------------------------------------------------
# Warning attribution -- both inert-pair warnings must name the user's own
# call site, not an internal ferrum frame. Mirrors
# ``test_heatmap_appearance.py``'s "Warning attribution" section; see that
# module's section header for the `@_figure_entry` frame-depth background.
# ---------------------------------------------------------------------------


def test_clustermap_inert_linewidths_warning_names_the_users_line():
    df = _wide_df()
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        line = sys._getframe().f_lineno + 1  # the next line is the call
        fm.clustermap(df, linewidths=2, linecolor="none")
    matching = [w for w in record if "linewidths=2.0 has no effect" in str(w.message)]
    assert [(w.filename, w.lineno) for w in matching] == [(__file__, line)]


def test_clustermap_inert_linecolor_warning_names_the_users_line():
    df = _wide_df()
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        line = sys._getframe().f_lineno + 1
        fm.clustermap(df, linewidths=0, linecolor="red")
    matching = [w for w in record if "linecolor='red' has no effect" in str(w.message)]
    assert [(w.filename, w.lineno) for w in matching] == [(__file__, line)]


# ---------------------------------------------------------------------------
# Malformed linewidths=/linecolor= -- refused naming clustermap and the
# argument, with the exact refusal text, never "stroke" (#125)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("bad_linewidths", "expected_message"),
    [
        (-1, "clustermap(linewidths=...) must be non-negative; received -1"),
        (
            float("nan"),
            "clustermap(linewidths=...) must be a finite numeric pixel value; received nan",
        ),
        (
            True,
            "clustermap(linewidths=...) must be a numeric pixel value (int or float), "
            "not bool; received True",
        ),
        (
            "a",
            "clustermap(linewidths=...) must be a numeric pixel value (int or float), "
            "not str; received 'a'",
        ),
    ],
)
def test_clustermap_malformed_linewidths_refuses_with_exact_text(bad_linewidths, expected_message):
    df = _wide_df()
    with pytest.raises(ValueError) as excinfo:
        fm.clustermap(df, linewidths=bad_linewidths)
    assert str(excinfo.value) == expected_message


@pytest.mark.parametrize(
    ("bad_linecolor", "expected_message"),
    [
        (
            "notacolor",
            "clustermap(linecolor=...)='notacolor' is not a valid color (invalid color "
            "'notacolor': expected a CSS color name, #rrggbb[aa]/#rgb[a] hex, or "
            "rgb()/rgba())",
        ),
        (
            "theme:foo",
            "clustermap(linecolor=...)='theme:foo' is not a valid color (invalid color "
            "'theme:foo': expected a CSS color name, #rrggbb[aa]/#rgb[a] hex, or "
            "rgb()/rgba())",
        ),
        (
            "Theme:Label",
            "clustermap(linecolor=...)='Theme:Label' is not a valid color (invalid color "
            "'Theme:Label': expected a CSS color name, #rrggbb[aa]/#rgb[a] hex, or "
            "rgb()/rgba())",
        ),
        (
            " theme:label",
            "clustermap(linecolor=...)=' theme:label' is not a valid color (invalid color "
            "'theme:label': expected a CSS color name, #rrggbb[aa]/#rgb[a] hex, or "
            "rgb()/rgba())",
        ),
    ],
)
def test_clustermap_malformed_linecolor_refuses_with_exact_text_not_stroke(
    bad_linecolor, expected_message
):
    """``"theme:foo"`` (a non-sentinel ``theme:*`` token) and a case or
    whitespace variant of the sentinel (``"Theme:Label"``, ``" theme:label"``)
    are malformed colors like any other -- ``_is_paint_sentinel`` matches
    ``"theme:label"`` by exact string equality. Mirrors
    ``test_heatmap_appearance.py``'s coverage."""
    df = _wide_df()
    with pytest.raises(ValueError) as excinfo:
        fm.clustermap(df, linecolor=bad_linecolor)
    assert str(excinfo.value) == expected_message
    assert "stroke" not in expected_message


def test_clustermap_non_str_linecolor_refuses_with_exact_text():
    df = _wide_df()
    with pytest.raises(ValueError) as excinfo:
        fm.clustermap(df, linecolor=3)
    assert str(excinfo.value) == "clustermap(linecolor=...)=3 must be a color string, not int"


def test_clustermap_malformed_linecolor_refuses_even_with_linewidths_zero():
    """Validation precedes the off-switch check: a malformed `linecolor` is
    refused even when `linewidths=0` would otherwise make it inert -- the
    clustermap sibling of ``test_heatmap_malformed_linecolor_refuses_even_with_linewidths_zero``."""
    df = _wide_df()
    with pytest.raises(ValueError) as excinfo:
        fm.clustermap(df, linewidths=0, linecolor="notacolor")
    assert str(excinfo.value) == (
        "clustermap(linecolor=...)='notacolor' is not a valid color (invalid color "
        "'notacolor': expected a CSS color name, #rrggbb[aa]/#rgb[a] hex, or "
        "rgb()/rgba())"
    )


# ---------------------------------------------------------------------------
# mark= still wins on the cells
# ---------------------------------------------------------------------------


def test_clustermap_mark_override_wins_over_the_border_default():
    df = _wide_df()
    svg = fm.clustermap(df, mark={"stroke": "red", "stroke_width": 2}).to_svg()
    cells = _cell_rects(svg)
    assert len(cells) == 9
    assert all(c["stroke"] == "#ff0000" for c in cells)
    assert all(c["stroke-width"] == "2" for c in cells)


def test_clustermap_mark_override_wins_over_a_clearing_linecolor():
    """`mark=` still reaches the cells even when `linecolor=` would clear
    the border on its own -- the helper returns `{}`, and `mark=` merges
    its own stroke in over that empty base."""
    df = _wide_df()
    svg = fm.clustermap(df, linecolor="none", mark={"stroke": "red", "stroke_width": 2}).to_svg()
    cells = _cell_rects(svg)
    assert len(cells) == 9
    assert all(c["stroke"] == "#ff0000" for c in cells)
    assert all(c["stroke-width"] == "2" for c in cells)
