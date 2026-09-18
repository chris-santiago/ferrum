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
from tests._svg_rects import polylines, svg_elements

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


class TestTheTitlesCannotGoMissingByTypeHole:
    """The fit layers' ``x_title``/``y_title`` are typed ``str`` and were fed
    ``_field_name(x)``, which returns ``str | None``.

    Nothing enforced the annotation, so a ``None`` reaching it did not raise:
    it retitled the value axis with the transform's own output column --
    ``ci_lower``, the confidence band's lower bound -- which is the defect
    this whole module pins, reintroduced through the type rather than through
    a code path. ``lmplot`` now narrows both names once, at its own entry, so
    the refusal names ``lmplot`` and the ``str`` downstream is true by
    construction.
    """

    @pytest.mark.parametrize("param", ["x", "y"])
    @pytest.mark.parametrize("method", ["lm", "robust"], ids=["smooth-routed", "dedicated"])
    def test_lmplot_refuses_a_missing_fit_column_in_its_own_voice(self, df, param, method):
        """Typed, and in ``lmplot``'s name, on both fit routes.

        Measured pre-fix, the same call raised an untyped ``TypeError`` from
        whichever consumer happened to receive the ``None`` first: ``encode(
        x=...) expects str, X instance, or Repeat placeholder`` on the
        smooth-routed methods, and the raw PyO3 ``argument 'x': 'None' is not
        an instance of 'str'`` on the dedicated-transform methods -- a
        binding-layer message escaping a figure-level call, which is the
        shape this wave refuses everywhere.
        """
        kwargs = {"x": "bill", "y": "tip", "method": method, param: None}
        with pytest.raises(ValueError) as excinfo:
            fm.lmplot(df, **kwargs).to_svg()
        message = str(excinfo.value)
        assert f"lmplot({param}=None)" in message, (
            f"the refusal must name lmplot and the parameter; got {message!r}"
        )
        assert "is not an instance of" not in message, (
            f"a raw PyO3 binding message must not reach the caller; got {message!r}"
        )

    @pytest.mark.parametrize("param", ["x", "y"])
    def test_residplot_refuses_a_missing_fit_column_in_its_own_voice(self, df, param):
        """The same hole at the sibling entry, which fed polars rather than PyO3.

        ``residplot`` passes ``x``/``y`` to ``drop_nulls(subset=...)`` and then
        into the residual transform, and puts them in the axis TITLES
        (``X(_resid_x, title=x)``) -- so the same ``str | None`` reached both
        a third-party binding and #199's own defect site. Measured pre-fix:
        ``TypeError: cannot turn 'NoneType' into selector``, polars'
        internals speaking for a ferrum figure function.
        """
        kwargs = {"x": "bill", "y": "tip", param: None}
        with pytest.raises(ValueError) as excinfo:
            fm.residplot(df, **kwargs).to_svg()
        message = str(excinfo.value)
        assert f"residplot({param}=None)" in message, (
            f"the refusal must name residplot and the parameter; got {message!r}"
        )
        assert "selector" not in message, (
            f"polars' own binding message must not reach the caller; got {message!r}"
        )


class TestAChannelObjectIsAcceptedNotLeaked:
    """``x=fm.X("bill")`` is a spelling this family takes, on every path.

    It always worked on ``lmplot``'s seven ``mark_smooth``-routed methods --
    ``.encode()`` takes a channel object -- while the three that construct
    their transform directly raised the raw PyO3 ``argument 'x': 'X' object
    is not an instance of 'str'``, and ``residplot`` raised polars' ``cannot
    turn 'X' into selector``. Since the column NAME is all those consumers
    ever wanted, the answer is to hand them the name rather than to refuse a
    spelling seven paths already honored: refusing would have been the
    reflexive answer and a regression for those seven.
    """

    ALL_METHODS = ["lm", "loess", "quadratic", "cubic", "log", "sqrt", "gam", "glm", "robust"]

    @pytest.mark.parametrize("method", ALL_METHODS)
    def test_every_lmplot_method_takes_a_channel_object(self, df, method):
        svg = fm.lmplot(df, x=fm.X("bill"), y=fm.Y("tip"), method=method).to_svg()
        assert_titles(svg, expected=("bill", "tip"), context=f"lmplot(x=X(), method={method!r})")

    def test_the_callers_own_channel_options_survive(self, df):
        """The channel still reaches ``.encode()``; only its NAME is extracted
        for the transform. So a ``title=`` on it wins, exactly as it does on
        the ``mark_smooth`` path -- the fix must not flatten the channel into
        a bare column name."""
        svg = fm.lmplot(
            df, x=fm.X("bill", title="Total bill (USD)"), y="tip", method="robust"
        ).to_svg()
        text = rendered_text(svg)
        assert "Total bill (USD)" in text
        assert "bill" not in text, "the caller's channel was flattened to its column name"

    def test_residplot_takes_one_too(self, df):
        svg = fm.residplot(df, x=fm.X("bill"), y=fm.Y("tip")).to_svg()
        assert_titles(svg, expected=("bill", "tip residual"), context="residplot(x=X(), y=Y())")

    def test_logx_composes_with_a_channel_object(self, df):
        """``logx`` used to build ``X(x, scale=log)`` unconditionally, which
        with a channel object raised ``X: field must be str ... got X`` -- so
        accepting the spelling everywhere else would have left one flag that
        still broke it. The rebuild keeps the caller's own kwargs."""
        svg = fm.lmplot(df, x=fm.X("bill", title="Bill"), y="tip", logx=True).to_svg()
        assert "Bill" in rendered_text(svg)

    def test_logx_refuses_a_channel_that_already_carries_a_scale(self, df):
        """Two instructions for one scale slot: refused, not silently
        overruled in either direction."""
        with pytest.raises(ValueError) as excinfo:
            fm.lmplot(df, x=fm.X("bill", scale={"type": "sqrt"}), y="tip", logx=True).to_svg()
        message = str(excinfo.value)
        assert "lmplot(logx=True" in message
        assert "scale" in message


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


def polyline_strokes(svg: str) -> set[str]:
    """Lower-cased ``stroke`` of every rendered ``<polyline>``.

    Reads through ``tests/_svg_rects``' shared parser rather than a local
    regex: this module used to roll four of its own, in the same change that
    landed the shared one.
    """
    return {stroke.lower() for stroke, _points in polylines(svg) if stroke is not None}


class TestLineStyling:
    """``layer_kwargs={"line": ...}`` styles the fit line WITHOUT repainting
    the CI band.

    This is why the fit line needs a per-layer door rather than riding the
    mark kwargs ``_resolve_pending`` spreads over every emitted layer. The
    door itself -- shared by every composite mark, validated against
    ``LAYER_NAME_CATALOG`` -- is pinned in
    ``tests/test_layer_kwargs_door.py``; what is pinned here is the fit
    line's own use of it.
    """

    def test_a_broad_mark_kwarg_does_leak_onto_the_band(self, df):
        """The control that makes the next test discriminate.

        ``**mark_kwargs`` reach every emitted layer by design, so a bare
        ``stroke=`` lands on the ribbon as well as the line. Measured, the
        band renders ``<path ... fill="rgba(37,99,235,0.302)"
        stroke="#dc143c"/>`` -- the leak is on the band's STROKE, and its
        fill is byte-identical to the unstyled case. Pinned here because the
        first version of the test below looked only at band fills and so
        passed on this exact shape. This is what ``layer_kwargs`` exists to
        avoid.
        """
        leaked = [
            el
            for el in band_elements(smooth_svg(df, ci=0.95, stroke="#dc143c"))
            if "#dc143c" in el.lower()
        ]
        assert leaked, (
            "expected a bare stroke= to reach the CI band (that is what "
            "layer_kwargs exists to avoid); band elements carried no crimson"
        )
        assert all('stroke="#dc143c"' in el for el in leaked), (
            f"the leak is expected on the band's stroke attribute; got {leaked}"
        )

    def test_the_per_layer_door_paints_the_line_and_leaves_the_band_alone(self, df):
        svg = smooth_svg(df, ci=0.95, layer_kwargs={"line": {"stroke": "#dc143c"}})
        strokes = polyline_strokes(svg)
        assert "#dc143c" in strokes, (
            f"the fit line must render crimson; polyline strokes were {strokes}"
        )
        bands = band_elements(svg)
        assert bands, "the ci=0.95 band must render, or there is nothing to leave alone"
        leaked = [el for el in bands if "#dc143c" in el.lower()]
        assert not leaked, (
            "layer_kwargs['line'] must not reach the confidence band, on its "
            f"stroke or its fill; these band elements carried the line's color: {leaked}"
        )

    def test_the_per_layer_door_reaches_the_single_line_shortcut(self, df):
        """``ci=None, show_metrics=False`` returns a single-mark result with no
        layer to carry mark kwargs, so the styling rides
        ``MarkDesugarResult.mark_kwargs`` instead -- reached through the
        ``layer_name`` that result declares."""
        svg = smooth_svg(df, layer_kwargs={"line": {"stroke": "#dc143c"}})
        strokes = polyline_strokes(svg)
        assert "#dc143c" in strokes, (
            f"the fit line must render crimson; polyline strokes were {strokes}"
        )

    @pytest.mark.parametrize("ci", [0.95, None], ids=["layered", "shortcut"])
    def test_the_per_layer_door_outranks_a_broad_mark_kwarg(self, df, ci):
        """The narrower instruction wins: specificity, not authorship.

        ``layer_kwargs`` names ONE layer; ``**mark_kwargs`` names the mark as
        a whole, and the dispatcher spreads it over every emitted layer. When
        both name the same key, the one aimed at the line wins -- a door that
        exists solely to target one layer would be pointless if the catch-all
        could silently override it.

        Pinned on both shapes because they reach the rule through different
        code (``_aim_layer_kwargs`` + ``apply_catch_all_style`` over
        ``_Layer.caller_style_keys`` vs ``_aim_single_mark_kwargs`` +
        ``_merge_desugar_mark_kwargs`` over
        ``MarkDesugarResult.caller_style_keys``), which is exactly where the
        two could drift apart.

        This flips the ordering shipped in the first #199 cycle, where the
        catch-all won. Nothing external depended on it -- the per-layer door
        is new API introduced by that same change, and ``lmplot``'s
        ``line_kws`` path never reaches the collision.
        """
        svg = smooth_svg(df, ci=ci, stroke="#0000ff", layer_kwargs={"line": {"stroke": "#ff0000"}})
        strokes = polyline_strokes(svg)
        assert strokes == {"#ff0000"}, (
            "layer_kwargs['line']['stroke'] must outrank the broad stroke= on "
            f"the fit line; polyline strokes were {strokes}"
        )

    def test_the_broad_kwarg_still_paints_what_nothing_narrower_named(self, df):
        """The catch-all is not disabled, only outranked where they collide.

        The band was named in no ``layer_kwargs`` entry of its own, so it
        still takes the broad ``stroke=`` -- the same spread this module's
        leak control pins. Without this, "the per-layer door wins" could be
        satisfied by dropping the catch-all entirely, which is a different
        and wrong behavior.
        """
        svg = smooth_svg(
            df, ci=0.95, stroke="#0000ff", layer_kwargs={"line": {"stroke": "#ff0000"}}
        )
        painted = [el for el in band_elements(svg) if 'stroke="#0000ff"' in el]
        assert painted, (
            "the CI band must still take the broad stroke= (nothing narrower "
            f"named it); band elements were {band_elements(svg)}"
        )

    def test_a_key_only_the_broad_kwarg_names_still_reaches_the_line(self, df):
        """Specificity is per KEY, not per call.

        Passing ``layer_kwargs`` must not make the line opt out of the
        catch-all wholesale -- only the keys the caller actually named there
        are protected.
        """
        svg = smooth_svg(
            df,
            ci=0.95,
            stroke="#0000ff",
            stroke_width=7.0,
            layer_kwargs={"line": {"stroke": "#ff0000"}},
        )
        # Attribute dicts, not whole elements: `stroke-width` is the half of
        # this assertion a stroke-only parser cannot see.
        fit_line = svg_elements(svg, "polyline", lambda a: "points" in a)
        assert any(
            a.get("stroke") == "#ff0000" and a.get("stroke-width") == "7" for a in fit_line
        ), (
            "the fit line must take the per-layer stroke AND the broad "
            f"stroke_width, which layer_kwargs never named; polylines were {fit_line}"
        )
