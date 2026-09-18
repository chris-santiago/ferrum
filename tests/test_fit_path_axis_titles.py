"""Every fit path titles its axes with the caller's own columns (#199).

The fit-family layers (``marks/_fit_layers.build_fit_layers``) and
``residplot``'s residual layers are bound to their TRANSFORM's output column
names -- ``x``, ``y``, ``ci_lower``, ``residual``. The renderer resolves each
axis title from layer 0 only (spec-level channel ``title``, then layer-0
channel ``title``, then layer-0 field name --
``build_axis_input``, ``crates/ferrum-core/src/render/prepare/mod.rs``), so
any chart whose layer 0 is one of those announced internal names on its axes.

Measured pre-fix, with ``x="bill", y="tip"``:

* ``fm.Chart(df).mark_smooth(method="lm", ci=0.95).encode(x="bill", y="tip")``
  -> axis titles ``ci_lower`` and ``x`` -- the value axis labelled with the
  confidence band's LOWER BOUND, not merely an internal spelling;
* ``fm.residplot(df, x="bill", y="tip")`` -> ``residual`` and ``x``, on the
  DEFAULT path with no flag involved;
* ``fm.regplot(df, x="bill", y="tip", scatter=False)`` -> ``ci_lower`` and
  ``x`` (``lmplot``'s own matrix is pinned in
  ``tests/test_lmplot_scatter_data_source.py``, where the scatter-vs-fit data
  claims already live).

``lmplot(scatter=True)`` was correct by accident: its layer 0 is the scatter,
bound to the caller's columns. The contract is now carried by the fit layers
themselves, so it no longer depends on a sibling layer being present.

Net-new coverage for a contract that spans a mark and two figure functions
gets its own module, per the test-homing convention in CLAUDE.md.
"""

from __future__ import annotations

import re

import numpy as np
import polars as pl
import pytest

import ferrum as fm

#: Transform output column names that must never reach an axis title. Both are
#: real columns of the fit output, and `ci_lower` is the wrong QUANTITY for the
#: value axis, not just the wrong spelling.
INTERNAL_NAMES = ("ci_lower", "ci_upper")


@pytest.fixture
def df() -> pl.DataFrame:
    """Columns deliberately named neither ``x`` nor ``y``: those two spellings
    collide with the transform's own output names, which is what let the defect
    read as correct in the goldens that happen to use them."""
    rng = np.random.default_rng(0)
    bill = np.linspace(2.0, 8.0, 60)
    return pl.DataFrame(
        {
            "bill": bill,
            "tip": 1.5 * bill + rng.normal(0.0, 0.4, 60),
            "day": rng.choice(["mon", "tue"], 60),
        }
    )


def rendered_text(svg: str) -> list[str]:
    return re.findall(r"<text[^>]*>([^<]*)</text>", svg)


def assert_titles(svg: str, *, expected: tuple[str, ...], context: str) -> None:
    text = rendered_text(svg)
    for title in expected:
        assert title in text, f"{context}: expected axis title {title!r}; rendered text was {text}"
    for internal in INTERNAL_NAMES:
        assert internal not in text, (
            f"{context}: rendered {internal!r} as a label -- that is a fit-transform "
            "output column, and for 'ci_lower' the confidence band's lower bound"
        )


class TestMarkSmooth:
    """The public mark, with no figure function involved."""

    @pytest.mark.parametrize("ci", [0.95, None])
    @pytest.mark.parametrize("show_metrics", [True, False])
    def test_mark_smooth_titles_its_axes_with_the_encoded_columns(self, df, ci, show_metrics):
        chart = fm.Chart(df).mark_smooth(method="lm", ci=ci, show_metrics=show_metrics)
        svg = chart.encode(x="bill", y="tip").to_svg()
        assert_titles(
            svg,
            expected=("bill", "tip"),
            context=f"mark_smooth(ci={ci!r}, show_metrics={show_metrics!r})",
        )

    def test_a_chart_level_title_still_outranks_the_layer_title(self, df):
        """The layer title is a FALLBACK. An explicit channel ``title=`` sits
        one priority above it and must still win, or this fix would have
        stolen the caller's own override."""
        svg = (
            fm.Chart(df)
            .mark_smooth(method="lm", ci=0.95)
            .encode(x=fm.X("bill", title="Total bill (USD)"), y="tip")
            .to_svg()
        )
        text = rendered_text(svg)
        assert "Total bill (USD)" in text
        assert "bill" not in text


class TestResidplot:
    """``residplot``'s default path -- no flag needed to reach the defect."""

    @pytest.mark.parametrize(
        "kwargs",
        [
            {},
            {"lowess": True},
            {"robust": True},
            {"zero_line": False, "show_metrics": False},
            {"color": "day"},
        ],
        ids=["default", "lowess", "robust", "flat", "grouped"],
    )
    def test_residplot_names_the_predictor_and_the_residualized_response(self, df, kwargs):
        """The value axis names the column being residualized: ``residual``
        alone leaves no way back to which response it came from."""
        svg = fm.residplot(df, x="bill", y="tip", **kwargs).to_svg()
        assert_titles(svg, expected=("bill", "tip residual"), context=f"residplot(**{kwargs})")

    def test_an_explicit_encode_title_still_wins(self, df):
        svg = fm.residplot(
            df, x="bill", y="tip", encode={"y": fm.Y("residual", title="Deviation")}
        ).to_svg()
        text = rendered_text(svg)
        assert "Deviation" in text
        assert "tip residual" not in text


class TestRegplot:
    """Pure delegation to ``lmplot``, and so inherits the fix."""

    @pytest.mark.parametrize("ci", [95, None])
    def test_regplot_scatter_false_titles_the_axes(self, df, ci):
        svg = fm.regplot(df, x="bill", y="tip", ci=ci, scatter=False).to_svg()
        assert_titles(svg, expected=("bill", "tip"), context=f"regplot(ci={ci!r}, scatter=False)")


def band_elements(svg: str) -> list[str]:
    """Every rendered ``<path>``/``<polygon>`` element, attributes included.

    The CI ribbon renders as a ``<path>``; the fit line is a ``<polyline>``,
    so these two element sets are disjoint. Whole elements rather than one
    attribute, because a style that leaks onto the band can land on EITHER
    its ``fill`` or its ``stroke`` -- and the leak this module guards against
    is a ``stroke``, which an assertion over fills alone cannot see.
    """
    return [m.group(0) for m in re.finditer(r"<(?:path|polygon)[^>]*/>", svg)]


def smooth_svg(df, **mark_kwargs) -> str:
    return fm.Chart(df).mark_smooth(method="lm", **mark_kwargs).encode(x="bill", y="tip").to_svg()


class TestLineStyling:
    """``line_kwargs`` styles the fit line WITHOUT repainting the CI band.

    This is why the fit line needs its own slot rather than riding the mark
    kwargs ``_resolve_pending`` spreads over every emitted layer.
    """

    def test_a_broad_mark_kwarg_does_leak_onto_the_band(self, df):
        """The control that makes the next test discriminate.

        ``**mark_kwargs`` reach every emitted layer by design, so a bare
        ``stroke=`` lands on the ribbon as well as the line. Measured, the
        band renders ``<path ... fill="rgba(37,99,235,0.302)"
        stroke="#dc143c"/>`` -- the leak is on the band's STROKE, and its
        fill is byte-identical to the unstyled case. Pinned here because the
        first version of the test below looked only at band fills and so
        passed on this exact shape.
        """
        leaked = [
            el
            for el in band_elements(smooth_svg(df, ci=0.95, stroke="#dc143c"))
            if "#dc143c" in el.lower()
        ]
        assert leaked, (
            "expected a bare stroke= to reach the CI band (that is what "
            "line_kwargs exists to avoid); band elements carried no crimson"
        )
        assert all('stroke="#dc143c"' in el for el in leaked), (
            f"the leak is expected on the band's stroke attribute; got {leaked}"
        )

    def test_line_kwargs_paint_the_line_and_leave_the_band_alone(self, df):
        svg = smooth_svg(df, ci=0.95, line_kwargs={"stroke": "#dc143c"})
        strokes = re.findall(r'<polyline[^>]*stroke="([^"]+)"', svg)
        assert any(s.lower() == "#dc143c" for s in strokes), (
            f"the fit line must render crimson; polyline strokes were {set(strokes)}"
        )
        bands = band_elements(svg)
        assert bands, "the ci=0.95 band must render, or there is nothing to leave alone"
        leaked = [el for el in bands if "#dc143c" in el.lower()]
        assert not leaked, (
            "line_kwargs must not reach the confidence band, on its stroke or "
            f"its fill; these band elements carried the line's color: {leaked}"
        )

    def test_line_kwargs_reach_the_single_line_shortcut(self, df):
        """``ci=None, show_metrics=False`` returns a single-mark result with no
        layer to carry mark kwargs, so the styling rides
        ``MarkDesugarResult.mark_kwargs`` instead."""
        svg = smooth_svg(df, line_kwargs={"stroke": "#dc143c"})
        strokes = re.findall(r'<polyline[^>]*stroke="([^"]+)"', svg)
        assert any(s.lower() == "#dc143c" for s in strokes), (
            f"the fit line must render crimson; polyline strokes were {set(strokes)}"
        )

    @pytest.mark.parametrize("ci", [0.95, None], ids=["layered", "shortcut"])
    def test_line_kwargs_outrank_a_broad_mark_kwarg(self, df, ci):
        """The narrower instruction wins: specificity, not authorship.

        ``line_kwargs`` names ONE layer; ``**mark_kwargs`` names the mark as a
        whole, and the dispatcher spreads it over every emitted layer. When
        both name the same key, the one aimed at the line wins -- a parameter
        that exists solely to target the line would be pointless if the
        catch-all could silently override it.

        Pinned on both shapes because they reach the rule through different
        code (``_apply_style_to_layer`` over ``_Layer.caller_style_keys`` vs
        ``_merge_desugar_mark_kwargs`` over
        ``MarkDesugarResult.caller_style_keys``), which is exactly where the
        two could drift apart.

        This flips the ordering shipped in the first #199 cycle, where the
        catch-all won. Nothing external depended on it -- ``line_kwargs`` is
        new API introduced by that same change, and ``lmplot``'s ``line_kws``
        path never reaches the collision.
        """
        svg = smooth_svg(df, ci=ci, stroke="#0000ff", line_kwargs={"stroke": "#ff0000"})
        strokes = {s.lower() for s in re.findall(r'<polyline[^>]*stroke="([^"]+)"', svg)}
        assert strokes == {"#ff0000"}, (
            "line_kwargs['stroke'] must outrank the broad stroke= on the fit "
            f"line; polyline strokes were {strokes}"
        )

    def test_the_broad_kwarg_still_paints_what_nothing_narrower_named(self, df):
        """The catch-all is not disabled, only outranked where they collide.

        The band has no ``line_kwargs`` of its own, so it still takes the
        broad ``stroke=`` -- the same spread this module's leak control pins.
        Without this, "line_kwargs wins" could be satisfied by dropping the
        catch-all entirely, which is a different and wrong behavior.
        """
        svg = smooth_svg(df, ci=0.95, stroke="#0000ff", line_kwargs={"stroke": "#ff0000"})
        painted = [el for el in band_elements(svg) if 'stroke="#0000ff"' in el]
        assert painted, (
            "the CI band must still take the broad stroke= (nothing narrower "
            f"named it); band elements were {band_elements(svg)}"
        )

    def test_a_key_only_the_broad_kwarg_names_still_reaches_the_line(self, df):
        """Specificity is per KEY, not per call.

        Passing ``line_kwargs`` must not make the line opt out of the
        catch-all wholesale -- only the keys the caller actually named there
        are protected.
        """
        svg = smooth_svg(
            df, ci=0.95, stroke="#0000ff", stroke_width=7.0, line_kwargs={"stroke": "#ff0000"}
        )
        fit_line = re.findall(r"<polyline[^>]*/>", svg)
        assert any('stroke="#ff0000"' in el and 'stroke-width="7"' in el for el in fit_line), (
            "the fit line must take line_kwargs' stroke AND the broad "
            f"stroke_width, which line_kwargs never named; polylines were {fit_line}"
        )
