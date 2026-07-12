"""SVG golden tests for the composite figure-level shared legend (#16).

These goldens pin the visual contract of the compositor legend band: when a
composite (concat/grid) resolves ``color`` or ``size`` as shared, the
compositor renders exactly one figure-level legend (or colorbar) built from
the unioned domain, placed outside the panel grid, instead of one legend per
participating panel. Seven representative cases are covered (spec §9):

- ``pairplot_hue_categorical``: the flagship — ``pairplot(hue=)`` collapses
  what used to be one legend per panel into a single figure legend (§9.1).
- ``hconcat_shared_continuous_color``: two panels sharing a continuous
  ``color`` scale render one figure colorbar with the unioned extent (§9.3).
- ``hconcat_shared_size``: ``resolve={"size": "shared"}`` renders one figure
  size legend (§9.4).
- ``shared_scale_independent_legend_optout``: ``Resolve(scale={"color":
  "shared"}, legend={"color": "independent"})`` keeps the unified domain but
  reverts to per-panel legends (§9.5) — today's pre-feature rendering.
- ``nested_hconcat_in_vconcat_inner_share``: sharing declared on an inner
  ``HConcatChart`` node inside an outer ``VConcatChart`` attaches the figure
  legend to that subtree only; the outer sibling panel keeps its own legend
  (§9.10).
- ``title_and_shared_legend``: a composite-level ``.properties(title=...)``
  co-renders with a shared legend — title band above, legend correctly
  offset, no overlap (§9.11).
- ``shared_colorbar_orient_bottom``: a shared continuous colorbar reoriented
  via ``.configure_legend(orient="bottom")`` — the band-geometry heuristic's
  known-risk area (it assumes a vertical colorbar by default).

Run with ``FERRUM_UPDATE_GOLDENS=1`` to regenerate the on-disk goldens (via
``tests._snapshots.regen_and_verify``, which also rasterizes an inspection
PNG); the default behavior is byte-identical comparison. A missing golden is
an explicit failure, not a silent skip.
"""

from __future__ import annotations

import os
from pathlib import Path

import polars as pl
import pytest

import ferrum as fm
from ferrum.composition import Resolve
from tests._snapshots import assert_svg_eq, legend_texts as _legend_texts, regen_and_verify


GOLDENS_DIR = Path(__file__).parent / "goldens" / "composite_shared_legend"
UPDATE = os.environ.get("FERRUM_UPDATE_GOLDENS") == "1"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_or_update(name: str, svg: str) -> None:
    """Compare ``svg`` to the on-disk golden ``name``.

    With ``FERRUM_UPDATE_GOLDENS=1`` the golden is (re)written via
    ``regen_and_verify``, which also rasterizes an inspection PNG under
    ``tests/snapshots/`` and prints both paths. Otherwise asserts
    byte-for-byte equality. A missing golden is an explicit test failure
    rather than a silent skip.
    """
    path = GOLDENS_DIR / name
    if UPDATE:
        regen_and_verify(path, svg)
        return
    if not path.exists():
        pytest.fail(
            f"golden {name!r} does not exist; rerun with FERRUM_UPDATE_GOLDENS=1 to generate it"
        )
    expected = path.read_text()
    assert_svg_eq(
        svg,
        expected,
        name=name,
        regen_hint="rerun with FERRUM_UPDATE_GOLDENS=1 to refresh",
    )


# ---------------------------------------------------------------------------
# Deterministic fixtures
#
# ``pairplot_df`` is defined in ``tests/conftest.py`` (shared with
# ``test_composite_shared_legend.py``).
# ---------------------------------------------------------------------------


@pytest.fixture
def continuous_color_charts():
    """Two panels with non-overlapping continuous ``val`` ranges.

    Panel 1: val in [1, 5].  Panel 2: val in [2, 10].  The unioned extent
    (spec §9.3) is [1, 10] — detectably different from either panel alone.
    """
    df1 = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0], "val": [1.0, 5.0, 3.0]})
    df2 = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0], "val": [10.0, 2.0, 8.0]})
    a = fm.Chart(df1).mark_point().encode(x="x", y="y", color="val")
    b = fm.Chart(df2).mark_point().encode(x="x", y="y", color="val")
    return a, b


@pytest.fixture
def continuous_size_charts():
    """Two panels with non-overlapping continuous ``sz`` ranges (unioned [1, 10])."""
    df1 = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0], "sz": [1.0, 5.0, 3.0]})
    df2 = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0], "sz": [10.0, 2.0, 8.0]})
    a = fm.Chart(df1).mark_point().encode(x="x", y="y", size="sz")
    b = fm.Chart(df2).mark_point().encode(x="x", y="y", size="sz")
    return a, b


@pytest.fixture
def categorical_color_charts():
    """Two panels sharing a categorical ``grp`` field (values 'p'/'q')."""
    df1 = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0], "grp": ["p", "q", "p"]})
    df2 = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0], "grp": ["p", "q", "q"]})
    a = fm.Chart(df1).mark_point().encode(x="x", y="y", color="grp")
    b = fm.Chart(df2).mark_point().encode(x="x", y="y", color="grp")
    return a, b


# ---------------------------------------------------------------------------
# Golden 1 (§9.1): pairplot(hue=) categorical — the flagship.
# ---------------------------------------------------------------------------


def test_pairplot_hue_categorical_golden(pairplot_df):
    """Exactly one 'grp' legend group, right of the 3x3 panel grid.

    Visual check: all 9 panels equal-sized; colors consistent panel-to-panel;
    one legend outside the grid, not one per panel.
    """
    chart = fm.pairplot(pairplot_df, vars=["x", "y", "z"], hue="grp")
    svg = chart.to_svg()

    texts = _legend_texts(svg)
    assert texts.count("grp") == 1, f"expected exactly one 'grp' legend title; got {texts}"
    assert texts.count("a") == 1
    assert texts.count("b") == 1

    _check_or_update("pairplot_hue_categorical.svg", svg)


# ---------------------------------------------------------------------------
# Golden 2 (§9.3): shared continuous color -> one figure colorbar.
# ---------------------------------------------------------------------------


def test_hconcat_shared_continuous_color_golden(continuous_color_charts):
    """One figure colorbar with the unioned [1, 10] extent, not one per panel.

    Visual check: single colorbar right of both panels; both panels' points
    are colored against the *same* (unioned) scale, not their own local
    range.
    """
    a, b = continuous_color_charts
    chart = fm.hconcat(a, b, resolve={"color": "shared"})
    svg = chart.to_svg()

    texts = _legend_texts(svg)
    assert texts.count("val") == 1, f"expected exactly one 'val' colorbar title; got {texts}"
    # Sanity: a bare hconcat (no resolve) would render two colorbars.
    default_svg = fm.hconcat(a, b).to_svg()
    assert _legend_texts(default_svg).count("val") == 2

    _check_or_update("hconcat_shared_continuous_color.svg", svg)


# ---------------------------------------------------------------------------
# Golden 3 (§9.4): resolve={"size": "shared"} -> one figure size legend.
# ---------------------------------------------------------------------------


def test_hconcat_shared_size_golden(continuous_size_charts):
    """One figure size legend with the unioned [1, 10] extent.

    Visual check: single size legend right of both panels; point sizes are
    comparable across panels (unioned scale), not scaled per-panel.
    """
    a, b = continuous_size_charts
    chart = fm.hconcat(a, b, resolve={"size": "shared"})
    svg = chart.to_svg()

    texts = _legend_texts(svg)
    assert texts.count("sz") == 1, f"expected exactly one 'sz' legend title; got {texts}"
    default_svg = fm.hconcat(a, b).to_svg()
    assert _legend_texts(default_svg).count("sz") == 2

    _check_or_update("hconcat_shared_size.svg", svg)


# ---------------------------------------------------------------------------
# Golden 4 (§9.5): shared scale + Resolve(legend=independent) opt-out.
# ---------------------------------------------------------------------------


def test_shared_scale_independent_legend_optout_golden(categorical_color_charts):
    """Unified domain, per-panel legends — today's (pre-feature) rendering.

    Visual check: two separate 'grp' legends, one per panel, each showing
    the same unioned {'p', 'q'} domain (both panels use identical colors for
    matching categories even though the legend itself is duplicated).
    """
    a, b = categorical_color_charts
    chart = fm.hconcat(
        a, b, resolve=Resolve(scale={"color": "shared"}, legend={"color": "independent"})
    )
    svg = chart.to_svg()

    texts = _legend_texts(svg)
    assert texts.count("grp") == 2, f"expected 2 per-panel 'grp' legend titles; got {texts}"

    _check_or_update("shared_scale_independent_legend_optout.svg", svg)


# ---------------------------------------------------------------------------
# Golden 5 (§9.10): nested composite — sharing on an inner subtree only.
# ---------------------------------------------------------------------------


def test_nested_hconcat_in_vconcat_inner_share_golden(categorical_color_charts):
    """An HConcat child inside a VConcat parent shares color only internally.

    The outer VConcat is: [inner HConcat(a, b, resolve=shared-color), c].
    Visual check: the inner HConcat's two panels dedupe into ONE legend
    attached just below/right of that subtree; the outer sibling panel ``c``
    keeps its own independent 'grp' legend, untouched by the inner sharing.
    """
    a, b = categorical_color_charts
    df_c = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [7.0, 8.0, 9.0], "grp": ["p", "q", "p"]})
    c = fm.Chart(df_c).mark_point().encode(x="x", y="y", color="grp")

    inner = fm.hconcat(a, b, resolve={"color": "shared"})
    chart = fm.vconcat(inner, c)
    svg = chart.to_svg()

    texts = _legend_texts(svg)
    assert texts.count("grp") == 2, (
        f"expected 1 figure legend (inner a+b) + 1 panel legend (outer c); got {texts}"
    )

    _check_or_update("nested_hconcat_in_vconcat_inner_share.svg", svg)


# ---------------------------------------------------------------------------
# Golden 6 (§9.11): figure title + shared legend co-render.
# ---------------------------------------------------------------------------


def test_title_and_shared_legend_golden(categorical_color_charts):
    """Composite-level title stacks above the legend band, no overlap.

    Visual check: title text above both panels; shared legend to the right,
    vertically offset below the title band's height (not overlapping it).
    """
    a, b = categorical_color_charts
    chart = fm.hconcat(a, b, resolve={"color": "shared"}).properties(title="Shared Legend Figure")
    svg = chart.to_svg()

    assert "Shared Legend Figure" in svg
    texts = _legend_texts(svg)
    assert texts.count("grp") == 1, f"expected exactly one 'grp' legend title; got {texts}"

    _check_or_update("title_and_shared_legend.svg", svg)


# ---------------------------------------------------------------------------
# Golden 7: shared continuous colorbar reoriented via configure_legend(orient="bottom").
#
# Known-risk area: the compositor's legend-band extent heuristic assumes a
# vertical colorbar (right/left orient); this golden exercises the
# horizontal (top/bottom) orientation explicitly.
# ---------------------------------------------------------------------------


def test_shared_colorbar_orient_bottom_golden(continuous_color_charts):
    """A shared colorbar reoriented to the bottom edge still renders once,
    below both panels, without corrupting panel layout.

    Visual check: single horizontal colorbar band below both panels (not to
    the right); no overlap with panel content; panels remain equal-sized.
    """
    a, b = continuous_color_charts
    chart = fm.hconcat(a, b, resolve={"color": "shared"}).configure_legend(orient="bottom")
    svg = chart.to_svg()

    texts = _legend_texts(svg)
    assert texts.count("val") == 1, f"expected exactly one 'val' colorbar title; got {texts}"

    _check_or_update("shared_colorbar_orient_bottom.svg", svg)
