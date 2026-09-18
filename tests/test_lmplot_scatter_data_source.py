"""``lmplot``/``regplot``: the scatter layer keeps the observations (F-L10-02).

The fit chart and the scatter chart are composed by ``_merge_layers``, which
pools both charts' transforms onto the merged chart. A transform with no
``name=`` is CHAIN-ADVANCING: it replaces the data every layer reads. So a fit
whose layer is bound to a chain-advancing ``Smooth`` does not merely draw
itself -- it also overwrites the scatter's rows with its own evaluation grid.

``mark_smooth`` produces exactly that shape for ``ci=None`` and
``show_metrics=False`` (its single-line shortcut), and ``lmplot`` used it
unconditionally. **RED, measured on the pre-fix tree**, with a 200-row frame:

* columns named ``x``/``y`` -- ``lmplot(df, x="x", y="y", ci=None,
  show_metrics=False)`` rendered 200 "observations" sitting EXACTLY on the fit
  polyline's 200 vertices (a string-exact match, 0.000 px spread about a
  perfect line), for ``lm``, ``loess``, ``quadratic``, ``gam``, ``log`` and
  ``sqrt``, in both ``lmplot`` and ``regplot`` -- 12 of 12 combinations;
* columns named anything else -- the same call raised ``ValueError: unknown
  column 'sepal_length' referenced by an encoding``, because the fit grid does
  not carry the user's column names.

Batch D's ``data_source`` routing closed this for the CI path only, since only
that path was already layered. The fix routes the single-mark case through the
same named-transform + ``build_fit_layers`` shape the CI path and the
dedicated fit methods (``logistic``/``glm``/``robust``) use.

Assertions read the RENDERED SVG and convert it back into data units through
the axis ticks: the defect was visible geometry that every ``to_svg()``
does-not-raise test in the suite passed straight through (the intent gate
found it against `tests/test_bug_hunt_figure_functions.py`'s own coverage of
this exact combination). Net-new coverage for a figure function gets its own
module per the test-homing convention in CLAUDE.md.
"""

from __future__ import annotations

import re

import numpy as np
import polars as pl
import pytest

import ferrum as fm
from ferrum.plots.regression import _DEDICATED_FIT_METHODS, _SMOOTH_ROUTED_METHODS

from _svg_extents import y_axis_units_per_pixel

#: Every method `lmplot` routes through `mark_smooth`, i.e. every method whose
#: fit chart could collapse to the single-line shortcut.
#:
#: DERIVED from the production constant, not restated: it is itself
#: `transform_registry("smooth")["methods"] - {"linear", "glm"}`, so a new
#: registry smooth method joins this matrix the day it lands. The hand-written
#: literal this replaced is exactly how `cubic` stayed uncovered through a
#: batch close -- a comment naming the derivation is not a derivation. Same
#: import as `tests/test_lmplot_valid_methods_registry_pin.py`, which pins the
#: two constants against the registry itself.
SMOOTH_ROUTED = sorted(_SMOOTH_ROUTED_METHODS)
#: The methods with their own fit-family transform, which `lmplot` builds
#: directly rather than through `mark_smooth`. Controls for the scatter-data
#: claims above; full participants in the axis-title / `line_kws` claims below,
#: which are properties of the fit layers whoever built them. Derived on the
#: same terms.
DEDICATED = sorted(_DEDICATED_FIT_METHODS)
#: `robust` joins the scatter-data claims as the dedicated-path control: that
#: path was already correct, so it must stay correct.
FIT_METHODS = SMOOTH_ROUTED + ["robust"]

#: Deliberately NOT 200 (`Smooth`'s default grid size) and deliberately not
#: named x/y: an equal count would let a corrupted scatter pass a count
#: assertion, and x/y column names are what made the pre-fix corruption
#: silent instead of raising.
_N_ROWS = 60


@pytest.fixture
def obs_df() -> pl.DataFrame:
    """60 noisy observations of a monotone relationship, plus a hue column.

    ``binary`` is the response ``method="logistic"`` needs (its transform
    refuses anything but 0/1); every other method reads ``response``. See
    :func:`response_column`.
    """
    rng = np.random.default_rng(0)
    predictor = np.linspace(2.0, 8.0, _N_ROWS)
    return pl.DataFrame(
        {
            "predictor": predictor,
            "response": 1.5 * predictor + 3.0 + rng.normal(0, 0.4, _N_ROWS),
            # Drawn from a logistic model rather than thresholded, so the two
            # classes overlap: a clean split makes `Logistic` refuse outright
            # ("perfect separation detected").
            "binary": (rng.random(_N_ROWS) < 1.0 / (1.0 + np.exp(-(predictor - 5.0)))).astype(
                float
            ),
            "grp": rng.choice(["a", "b"], _N_ROWS),
        }
    )


def response_column(method: str) -> str:
    """The response column *method* can be fit against (see ``obs_df``)."""
    return "binary" if method == "logistic" else "response"


def circle_centers(svg: str) -> list[tuple[float, float]]:
    """``(cx, cy)`` for every rendered ``<circle>``."""
    centers = []
    for match in re.finditer(r"<circle([^/]+)/>", svg):
        attrs = dict(re.findall(r'([\w-]+)="([^"]+)"', match.group(1)))
        if "cx" in attrs and "cy" in attrs:
            centers.append((float(attrs["cx"]), float(attrs["cy"])))
    return centers


def polyline_vertices(svg: str) -> list[tuple[float, float]]:
    """Every vertex of every rendered ``<polyline>`` (the fit curve)."""
    vertices = []
    for points in re.findall(r"<polyline[^>]*points=\"([^\"]+)\"", svg):
        for pair in re.finditer(r"(-?\d+\.?\d*)[ ,](-?\d+\.?\d*)", points):
            vertices.append((float(pair.group(1)), float(pair.group(2))))
    return vertices


def observed_response_ladder(svg: str) -> list[float]:
    """The scatter circles' response values, in data units, lowest-first at 0.

    Ascending, and expressed as distances from the smallest one, because the
    axis ticks fix the scale exactly but the offset only approximately (see
    ``y_axis_units_per_pixel``). A shift is not what a corrupted scatter does
    to this ladder: replacing 60 observations with a 200-point fit grid
    changes its length, its spacing, and its total range.
    """
    units = y_axis_units_per_pixel(svg)
    pixels = sorted((cy for _, cy in circle_centers(svg)), reverse=True)
    return [(pixels[0] - pixel) * units for pixel in pixels]


def value_ladder(values) -> list[float]:
    """The same ladder for raw data values: ascending, lowest-first at 0."""
    ordered = sorted(values)
    return [value - ordered[0] for value in ordered]


@pytest.mark.parametrize("figure", [fm.lmplot, fm.regplot], ids=["lmplot", "regplot"])
@pytest.mark.parametrize("method", FIT_METHODS)
def test_scatter_draws_the_observations_not_the_fit_grid(obs_df, figure, method):
    """The strongest form of the claim: the circles ARE the data.

    Not "the circles are not on the line" (a shifted grid would pass that) --
    every rendered circle, converted back to data units through the axis, is
    one of the frame's own response values.
    """
    svg = figure(
        obs_df, x="predictor", y="response", method=method, ci=None, show_metrics=False
    ).to_svg()
    observed = observed_response_ladder(svg)
    expected = value_ladder(obs_df["response"].to_list())
    assert len(observed) == _N_ROWS, (
        f"{figure.__name__}(method={method!r}) must draw one circle per row; "
        f"got {len(observed)} for {_N_ROWS} rows (200 would be the fit grid)"
    )
    # A generous tolerance: each value is recovered through a rendered pixel
    # position, so it carries the axis's own rounding. It is far tighter than
    # any wrong-data failure mode -- a fit grid's values differ from the
    # observations by the residuals (sd 0.4 here), not by a rounding step.
    assert observed == pytest.approx(expected, abs=0.05)


@pytest.mark.parametrize("figure", [fm.lmplot, fm.regplot], ids=["lmplot", "regplot"])
@pytest.mark.parametrize("method", FIT_METHODS)
def test_scatter_keeps_its_spread_about_the_fit(obs_df, figure, method):
    """The observations scatter about the fit; they do not lie on it.

    This is the pre-fix signature in its rawest form: 200/200 circles at a
    polyline vertex, 0.000 px deviation. Both halves are asserted -- no circle
    sits on a vertex, AND the fit curve is still drawn (a fix that merely
    dropped the fit would pass the first half alone).
    """
    svg = figure(
        obs_df, x="predictor", y="response", method=method, ci=None, show_metrics=False
    ).to_svg()
    vertices = polyline_vertices(svg)
    assert len(vertices) > 50, f"the {method!r} fit curve must still render; got {vertices[:3]}"

    rounded_vertices = {(round(vx, 3), round(vy, 3)) for vx, vy in vertices}
    on_the_curve = [
        c for c in circle_centers(svg) if (round(c[0], 3), round(c[1], 3)) in rounded_vertices
    ]
    assert not on_the_curve, (
        f"{figure.__name__}(method={method!r}): {len(on_the_curve)} scatter circles sit exactly "
        "on a fit vertex -- the fit's evaluation grid has replaced the observations"
    )

    # The scatter's own spread about the fit, in data units: the pre-fix
    # render collapsed it to 0.000 (every point ON the curve), while the
    # observations carry the fixture's injected noise (sd 0.4).
    units = y_axis_units_per_pixel(svg)
    fit = np.array(sorted(vertices))
    centers = np.array(circle_centers(svg))
    fit_y_at_each_circle = np.interp(centers[:, 0], fit[:, 0], fit[:, 1])
    deviations = np.abs(centers[:, 1] - fit_y_at_each_circle) * units
    assert float(np.mean(deviations)) > 0.1, (
        f"{figure.__name__}(method={method!r}): the observations must scatter about the fit; "
        f"mean deviation {np.mean(deviations):.4f} data units against an injected sd of 0.4"
    )


@pytest.mark.parametrize("method", SMOOTH_ROUTED + DEDICATED)
@pytest.mark.parametrize("ci", [95, None])
@pytest.mark.parametrize("show_metrics", [True, False])
def test_scatter_false_titles_the_axes_with_the_callers_columns(obs_df, method, ci, show_metrics):
    """With no scatter layer there is nothing to corrupt, and nothing to lose.

    The fit IS the chart here, so its own layers are what the renderer reads
    the axis titles off -- and those layers are bound to the TRANSFORM's
    output columns (``x``, ``y``, ``ci_lower``). Measured pre-fix over this
    whole 40-cell matrix (10 methods × 2 ``ci`` × 2 ``show_metrics``), only
    7 cells titled the axes with the caller's columns -- the seven
    ``mark_smooth``-routed methods at ``ci=None, show_metrics=False``, where
    the single-line shortcut keeps the chart-level encoding and so never had
    the problem. The other 33 announced their value axis as ``ci_lower``
    (the CI band's lower bound) or as ``y``, and their predictor axis as
    ``x`` (#199).

    Parameterized over the full method × ci × show_metrics product precisely
    because the earlier version of this test pinned ONE of those seven lucky
    cells, and the defect survived a batch close behind it.
    """
    y_col = response_column(method)
    svg = fm.lmplot(
        obs_df,
        x="predictor",
        y=y_col,
        method=method,
        ci=ci,
        show_metrics=show_metrics,
        scatter=False,
    ).to_svg()
    assert len(polyline_vertices(svg)) > 50
    assert not circle_centers(svg)
    titles = re.findall(r"<text[^>]*>([^<]*)</text>", svg)
    assert "predictor" in titles and y_col in titles, (
        f"lmplot(method={method!r}, ci={ci!r}, show_metrics={show_metrics!r}, "
        f"scatter=False) must title its axes with the caller's own columns; "
        f"rendered text was {titles}"
    )
    for internal in ("ci_lower", "ci_upper"):
        assert internal not in titles, (
            f"lmplot(method={method!r}, ci={ci!r}, scatter=False) titled an "
            f"axis {internal!r} -- a transform output column, and for "
            "'ci_lower' the wrong quantity entirely"
        )


@pytest.mark.parametrize("method", SMOOTH_ROUTED + DEDICATED)
@pytest.mark.parametrize("ci", [95, None])
def test_scatter_false_honors_line_kws(obs_df, method, ci):
    """``line_kws`` reaches the rendered fit line when the fit is alone.

    All seven smooth-routed methods dropped it here -- 28 of the 40
    ``scatter=False`` cells, measured pre-fix: ``lmplot`` used to apply
    ``line_kws`` by rewriting the finished chart's ``_layers``, which on this
    branch are still empty -- the chart carries a pending stat mark whose
    desugar does not run until render. The three dedicated methods (which
    forward at construction) were unaffected on all 12 of theirs, which is
    what made the gap look like a method quirk rather than a branch one
    (#199).

    Asserted on the rendered stroke, not on the chart object: the point is
    that the caller's color reaches the SVG.
    """
    svg = fm.lmplot(
        obs_df,
        x="predictor",
        y=response_column(method),
        method=method,
        ci=ci,
        scatter=False,
        line_kws={"stroke": "#dc143c", "stroke_width": 9},
    ).to_svg()
    fit_strokes = re.findall(r'<polyline[^>]*stroke="([^"]+)"', svg)
    assert any(stroke.lower() == "#dc143c" for stroke in fit_strokes), (
        f"lmplot(method={method!r}, ci={ci!r}, scatter=False, "
        f"line_kws={{'stroke': '#dc143c'}}) must paint the fit line crimson; "
        f"rendered polyline strokes were {set(fit_strokes)}"
    )


@pytest.mark.parametrize("figure", [fm.lmplot, fm.regplot], ids=["lmplot", "regplot"])
@pytest.mark.parametrize("method", SMOOTH_ROUTED)
def test_scatter_layer_present_still_shows_metrics(obs_df, figure, method):
    """The R² annotation survives on the named-transform (scatter-present) branch.

    ``ci=None`` is what pushes ``mark_smooth`` toward its chain-advancing
    single-line shortcut for ``show_metrics=False`` -- which is why the rest
    of this module fixes ``ci=None``. With ``show_metrics=True`` (the
    default) that shortcut is never taken, but the fit is still built through
    ``_fit_chart_from_named_transform`` on the ``scatter_layer is not None``
    branch, which appends the metrics text layer only when it is called with
    ``metrics=metrics_applied`` -- not a hardcoded ``False``. This is the
    exact branch and trigger combination the mutation gate found under-tested
    (batch-F close, F-L10-02): killing ``metrics=metrics_applied`` ->
    ``metrics=False`` there left the whole suite in this module green.
    """
    svg = figure(
        obs_df, x="predictor", y="response", method=method, ci=None, show_metrics=True
    ).to_svg()
    assert "R²" in svg, (
        f"{figure.__name__}(method={method!r}, ci=None, show_metrics=True): "
        "expected the R² metrics annotation on the scatter+fit chart"
    )


@pytest.mark.parametrize("ci", [95, None])
@pytest.mark.parametrize("show_metrics", [True, False])
def test_hue_still_fits_and_colors_per_group(obs_df, ci, show_metrics):
    """The per-group fit path keeps working across the shortcut boundary."""
    svg = fm.lmplot(
        obs_df, x="predictor", y="response", hue="grp", ci=ci, show_metrics=show_metrics
    ).to_svg()
    strokes = set(re.findall(r'<polyline[^>]*stroke="([^"]+)"', svg))
    assert len(strokes) >= 2, f"one fit line per hue level expected; got strokes {strokes}"


class TestIntegerPredictorWithJitter:
    """``x_jitter`` on a non-Float64 predictor.

    ``apply_jitter`` writes a Float64 noise array back into the column it
    offsets, so an Int64 predictor failed Arrow's schema check: ``Jitter: ...
    expected Int64 but found Float64``. Measured on the pre-fix tree, that
    raised for 3 of the 4 ``ci``/``show_metrics`` combinations; the 4th
    "passed" only because the scatter layer was reading the fit's own Float64
    grid there -- the very corruption above. Fixing that exposed the dtype gap
    on all four, so the predictor is now widened before either chart is built.
    """

    @pytest.fixture
    def int_df(self) -> pl.DataFrame:
        return pl.DataFrame({"predictor": list(range(1, 21)), "response": list(range(2, 22))})

    @pytest.mark.parametrize("ci", [95, None])
    @pytest.mark.parametrize("show_metrics", [True, False])
    def test_integer_predictor_with_x_jitter_renders(self, int_df, ci, show_metrics):
        svg = fm.lmplot(
            int_df,
            x="predictor",
            y="response",
            x_jitter=0.3,
            ci=ci,
            show_metrics=show_metrics,
        ).to_svg()
        assert "<svg" in svg and "NaN" not in svg
        # Jitter can push an edge observation just outside the data-derived
        # domain, where the renderer drops it -- a separate, documented gap
        # (`NS-1`, the one `catplot(native_scale=True)` pads around). The
        # claim here is that the call renders its observations at all.
        assert len(circle_centers(svg)) >= len(int_df) - 2

    def test_float_predictor_is_untouched_by_the_widening(self, obs_df):
        """The cast is a no-op for a column that is already Float64."""
        with_jitter = fm.lmplot(
            obs_df, x="predictor", y="response", x_jitter=0.3, ci=None, show_metrics=False
        ).to_svg()
        assert "<svg" in with_jitter and "NaN" not in with_jitter

    def test_no_jitter_leaves_an_integer_predictor_alone(self, int_df):
        """Without ``x_jitter`` there is no adjustment, so no widening."""
        svg = fm.lmplot(int_df, x="predictor", y="response", ci=None, show_metrics=False).to_svg()
        assert "<svg" in svg and "NaN" not in svg
