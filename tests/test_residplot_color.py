"""``residplot(color=)``: the two documented halves, and how they compose.

``color=`` has always been documented as "column name or constant color"
and always raised ``unknown column`` for both (finding F-L10-04). This
module is the feature coverage for the implemented contract:

- a **column** groups the residuals -- they are refit per group (Smooth
  *and* Robust), coloured by that column, and a ``lowess=True`` overlay
  smooths each group separately;
- anything else ferrum's color parser accepts is a **literal**, painted
  onto the points through the mark's own ``fill``;
- membership wins, so a column literally named ``"red"`` groups;
- a value that is neither raises, naming both readings.

Geometry claims are asserted from the parsed SVG rather than the spec
dict, because "the residuals were refit per group" is a statement about
rendered point positions that a spec-level ``groupby=[...]`` assertion
cannot make.
"""

from __future__ import annotations

import collections
import json
import re

import numpy as np
import polars as pl
import pytest

import ferrum as fm

# ---------------------------------------------------------------------------
# SVG parsing helpers
# ---------------------------------------------------------------------------

_CIRCLE = re.compile(r"<circle[^>]*/>")
_ATTR = r'{}="([^"]+)"'


def _mark_circles(svg: str) -> list[tuple[float, float, str]]:
    """Return ``(cx, cy, fill)`` for the data-point circles of *svg*.

    Legend swatches are circles too; they are drawn at a fixed radius that
    differs from the point mark's, so the modal radius selects the data
    points without depending on either number.
    """
    parsed = []
    for tag in _CIRCLE.findall(svg):
        parsed.append(
            (
                float(re.search(_ATTR.format("cx"), tag).group(1)),
                float(re.search(_ATTR.format("cy"), tag).group(1)),
                re.search(_ATTR.format("r"), tag).group(1),
                re.search(_ATTR.format("fill"), tag).group(1),
            )
        )
    modal_r = collections.Counter(p[2] for p in parsed).most_common(1)[0][0]
    return [(cx, cy, fill) for cx, cy, r, fill in parsed if r == modal_r]


def _points_by_fill(svg: str) -> dict[str, list[float]]:
    """Map each point fill colour to its list of ``cy`` values."""
    by_fill: dict[str, list[float]] = collections.defaultdict(list)
    for _cx, cy, fill in _mark_circles(svg):
        by_fill[fill].append(cy)
    return dict(by_fill)


def _polylines(svg: str) -> list[str]:
    return re.findall(r"<polyline[^>]*/?>", svg)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def offset_groups() -> pl.DataFrame:
    """Two groups on the same slope, separated by a large intercept shift.

    The design is chosen to discriminate a per-group refit from a pooled
    one: pooled, one line splits the difference and each group's residuals
    sit ~20 units to one side, so the two clouds are disjoint; refit per
    group, both clouds are the same noise around zero and interleave. The
    groups alternate along ``x`` so both span the same predictor range --
    with the groups segregated in ``x`` a pooled fit could absorb the shift
    into its slope and the two designs would look alike.
    """
    rng = np.random.default_rng(1)
    n = 24
    x = np.linspace(0.0, 10.0, n)
    group = np.where(np.arange(n) % 2 == 0, "a", "b")
    y = np.where(group == "a", 2.0 * x, 2.0 * x + 40.0) + rng.normal(0.0, 0.3, n)
    return pl.DataFrame({"x": x.tolist(), "y": y.tolist(), "g": group.tolist()})


@pytest.fixture
def simple_frame() -> pl.DataFrame:
    rng = np.random.default_rng(7)
    n = 20
    x = np.linspace(0.0, 5.0, n)
    return pl.DataFrame({"x": x.tolist(), "y": (3.0 * x + rng.normal(0.0, 0.4, n)).tolist()})


def _assert_interleaved(by_fill: dict[str, list[float]]) -> None:
    """Assert two point clouds overlap -- the per-group-refit signature.

    Each group's median must fall inside the other group's vertical extent.
    A pooled fit on ``offset_groups`` puts the two clouds on opposite sides
    of zero with no overlap at all, so this fails loudly if the grouping
    column stops reaching the transform's ``groupby``.
    """
    assert len(by_fill) == 2, f"expected two coloured point sets, got {sorted(by_fill)}"
    (_fill_a, ys_a), (_fill_b, ys_b) = sorted(by_fill.items())
    median_a = float(np.median(ys_a))
    median_b = float(np.median(ys_b))
    assert min(ys_b) <= median_a <= max(ys_b), (
        f"group A's median residual {median_a} lies outside group B's extent "
        f"[{min(ys_b)}, {max(ys_b)}] -- the residuals look pooled, not refit per group"
    )
    assert min(ys_a) <= median_b <= max(ys_a), (
        f"group B's median residual {median_b} lies outside group A's extent "
        f"[{min(ys_a)}, {max(ys_a)}] -- the residuals look pooled, not refit per group"
    )


# ---------------------------------------------------------------------------
# The column half
# ---------------------------------------------------------------------------


class TestColorColumn:
    def test_column_refits_residuals_per_group(self, offset_groups):
        svg = fm.residplot(
            offset_groups, x="x", y="y", color="g", show_metrics=False, zero_line=False
        ).to_svg()
        by_fill = _points_by_fill(svg)
        assert sorted(len(v) for v in by_fill.values()) == [12, 12]
        _assert_interleaved(by_fill)

    def test_pooled_control_separates_the_same_data(self, offset_groups):
        """The same frame without ``color=`` is one cloud with a visible gap.

        Pins the discrimination the test above relies on: the fixture really
        does render two disjoint bands when the fit is pooled.
        """
        svg = fm.residplot(
            offset_groups, x="x", y="y", show_metrics=False, zero_line=False
        ).to_svg()
        ys = sorted(cy for _cx, cy, _fill in _mark_circles(svg))
        assert len(_points_by_fill(svg)) == 1
        gaps = [b - a for a, b in zip(ys, ys[1:])]
        assert max(gaps) > 0.5 * (ys[-1] - ys[0]), (
            "pooled residuals should fall in two bands separated by most of the "
            "plot height; the fixture no longer discriminates"
        )

    def test_column_threads_into_the_transform_groupby(self, offset_groups):
        spec = json.loads(fm.residplot(offset_groups, x="x", y="y", color="g").to_spec().to_json())
        smooth = next(t for t in spec["transforms"] if t.get("type") == "smooth")
        assert smooth["groupby"] == ["g"]
        assert spec["encoding"]["color"]["field"] == "g"

    def test_column_composes_with_label(self, offset_groups):
        spec = json.loads(
            fm.residplot(offset_groups, x="x", y="y", color="g", label="fold 1").to_spec().to_json()
        )
        smooth = next(t for t in spec["transforms"] if t.get("type") == "smooth")
        assert smooth["groupby"] == ["_label", "g"]
        # The caller-named discriminator wins the colour scale; `_label`
        # rides along as a group key so it survives the transform.
        assert spec["encoding"]["color"]["field"] == "g"

    def test_grouping_column_is_typed_nominal(self, offset_groups):
        """An integer group column must not infer a Continuous colour scale.

        It drives ``groupby=``, so it is a discriminator by construction.
        """
        frame = offset_groups.with_columns((pl.col("g") == "b").cast(pl.Int64).alias("gi"))
        spec = json.loads(fm.residplot(frame, x="x", y="y", color="gi").to_spec().to_json())
        assert spec["encoding"]["color"]["type"] == "nominal"

    def test_encoding_object_is_honored(self, offset_groups):
        spec = json.loads(
            fm.residplot(offset_groups, x="x", y="y", color=fm.Color("g", legend=False))
            .to_spec()
            .to_json()
        )
        smooth = next(t for t in spec["transforms"] if t.get("type") == "smooth")
        assert smooth["groupby"] == ["g"]
        assert spec["encoding"]["color"]["field"] == "g"


# ---------------------------------------------------------------------------
# The literal half
# ---------------------------------------------------------------------------


class TestColorLiteral:
    def test_literal_paints_the_points(self, simple_frame):
        svg = fm.residplot(
            simple_frame, x="x", y="y", color="red", show_metrics=False, zero_line=False
        ).to_svg()
        fills = {fill for _cx, _cy, fill in _mark_circles(svg)}
        assert fills == {"#ff0000"}

    def test_literal_paints_the_layered_point_layer_too(self, simple_frame):
        """The overlay branch builds its own point layer, which must own the
        paint rather than inherit it: a per-layer ``mark=`` override or any
        future change to chart-level style inheritance would otherwise drop
        the colour on every ``zero_line``/``lowess``/metrics chart."""
        chart = fm.residplot(
            simple_frame, x="x", y="y", color="red", show_metrics=False, zero_line=True
        )
        svg = chart.to_svg()
        fills = {fill for _cx, _cy, fill in _mark_circles(svg)}
        assert fills == {"#ff0000"}
        assert "#8a8a8a" in svg  # the zero rule still renders
        spec = json.loads(chart.to_spec().to_json())
        point_layer = next(layer for layer in spec["layers"] if layer["name"] == "point")
        assert point_layer["mark_style"]["fill"] == "#ff0000"
        reference = next(layer for layer in spec["layers"] if layer["name"] == "reference")
        assert "fill" not in reference["mark_style"], "the paint leaked onto the zero rule"

    def test_literal_does_not_fabricate_a_colour_encoding(self, simple_frame):
        spec = json.loads(fm.residplot(simple_frame, x="x", y="y", color="red").to_spec().to_json())
        assert "color" not in spec["encoding"]
        smooth = next(t for t in spec["transforms"] if t.get("type") == "smooth")
        assert "groupby" not in smooth or smooth["groupby"] == []

    def test_literal_keeps_the_metrics_corner(self, simple_frame):
        """A literal paints; it does not group, so the metrics gate must not
        fire. The gate keys on ``color_arg.group_field``, and every other
        literal test opts out of metrics — without this, widening the gate to
        "any ``color=`` suppresses" would take R²/RMSE/MAE off a plain
        ``residplot(..., color="red")`` with nothing failing."""
        chart = fm.residplot(simple_frame, x="x", y="y", color="red")
        assert "R²" in chart.to_svg()
        spec = json.loads(chart.to_spec().to_json())
        smooth = next(t for t in spec["transforms"] if t.get("type") == "smooth")
        assert smooth["inject_metrics"] is True

    def test_literal_accepts_the_full_colour_vocabulary(self, simple_frame):
        for value, expected in (
            ("steelblue", "#4682b4"),
            ("#0f0", "#00ff00"),
            ("rgb(1, 2, 3)", "#010203"),
            ((1.0, 0.0, 0.0), "#ff0000"),
        ):
            svg = fm.residplot(
                simple_frame,
                x="x",
                y="y",
                color=value,
                show_metrics=False,
                zero_line=False,
            ).to_svg()
            fills = {fill for _cx, _cy, fill in _mark_circles(svg)}
            assert fills == {expected}, f"color={value!r}"


# ---------------------------------------------------------------------------
# Dispatch order and refusals
# ---------------------------------------------------------------------------


class TestColorDispatch:
    def test_colour_named_column_groups_rather_than_paints(self, offset_groups):
        """Membership is primary: a column named ``"red"`` is a column."""
        frame = offset_groups.rename({"g": "red"})
        svg = fm.residplot(
            frame, x="x", y="y", color="red", show_metrics=False, zero_line=False
        ).to_svg()
        by_fill = _points_by_fill(svg)
        assert "#ff0000" not in by_fill, "the column was painted instead of grouped"
        assert sorted(len(v) for v in by_fill.values()) == [12, 12]
        _assert_interleaved(by_fill)

    def test_neither_column_nor_colour_refuses_naming_both(self, simple_frame):
        with pytest.raises(ValueError) as excinfo:
            fm.residplot(simple_frame, x="x", y="y", color="nosuchthing")
        message = str(excinfo.value)
        assert "not a column of the data" in message
        assert "parseable color" in message
        assert "'nosuchthing'" in message

    def test_non_numeric_tuple_refuses_with_the_same_typed_error(self, simple_frame):
        """The third leg of the dispatch is a typed ``ValueError`` for *any*
        unusable value. A non-numeric tuple makes the colour parser raise
        ``TypeError`` from inside its own component comparison; that must not
        reach the caller as an unexplained internal error."""
        with pytest.raises(ValueError) as excinfo:
            fm.residplot(simple_frame, x="x", y="y", color=("a", "b", "c"))
        message = str(excinfo.value)
        assert "not a column of the data" in message
        assert "parseable color" in message

    def test_existing_label_column_refuses_rather_than_colliding(self, simple_frame):
        """``label=`` needs ``_label`` for its own group key. A caller who
        already has that column would have their values silently overwritten
        by the literal, and ``groupby`` would carry the name twice -- which
        Rust tolerates, so the whole collision renders quietly."""
        frame = simple_frame.with_columns(pl.lit("mine").alias("_label"))
        with pytest.raises(ValueError, match="_label"):
            fm.residplot(frame, x="x", y="y", label="series A")
        # Without label=, the caller's own column is untouched and usable.
        spec = json.loads(fm.residplot(frame, x="x", y="y", color="_label").to_spec().to_json())
        smooth = next(t for t in spec["transforms"] if t.get("type") == "smooth")
        assert smooth["groupby"] == ["_label"]

    def test_literal_with_label_colours_both_points_and_swatch(self, simple_frame):
        """Both arguments are honored: the literal paints the points, and the
        one-entry legend ``label=`` creates takes that same colour rather than
        a palette colour that would repaint the marks."""
        chart = fm.residplot(
            simple_frame,
            x="x",
            y="y",
            color="red",
            label="series A",
            show_metrics=False,
            zero_line=False,
        )
        svg = chart.to_svg()
        assert {fill for _cx, _cy, fill in _mark_circles(svg)} == {"#ff0000"}
        assert ">series A<" in svg, "the label legend entry is missing"
        swatches = [tag for tag in re.findall(r"<circle[^>]*/>", svg) if 'r="4"' in tag]
        assert swatches, "no legend swatch rendered"
        assert all('fill="#ff0000"' in tag for tag in swatches), (
            f"the legend swatch does not carry the requested colour: {swatches}"
        )

    def test_literal_with_label_carries_through_to_the_lowess_overlay(self, simple_frame):
        """The overlay binds the same pinned channel, so the curve cannot come
        out in a palette colour beside literal-coloured points."""
        chart = fm.residplot(
            simple_frame,
            x="x",
            y="y",
            color="red",
            label="series A",
            lowess=True,
            show_metrics=False,
            zero_line=False,
        )
        svg = chart.to_svg()
        curves = _polylines(svg)
        assert len(curves) == 1
        assert re.search(_ATTR.format("stroke"), curves[0]).group(1) == "#ff0000"
        # The rendered stroke alone does not prove the *overlay* binds the
        # pinned channel: a layer naming a bare ``_label`` resolves against the
        # chart-level scale and comes out the same colour. Pin the layer's own
        # wire spec, mirroring the column case above.
        spec = json.loads(chart.to_spec().to_json())
        lowess_layer = next(layer for layer in spec["layers"] if layer["name"] == "lowess")
        overlay_color = lowess_layer["encoding"]["color"]
        assert overlay_color.get("scale", {}).get("range") == ["#ff0000"], (
            f"the overlay layer does not carry the pinned scale: {overlay_color}"
        )

    def test_literal_with_label_pins_the_scale_to_the_literal(self, simple_frame):
        """The swatch colour comes from the label scale's single range slot,
        not from the palette -- pinned at the spec so the mechanism, not just
        today's palette, is what the rendering assertion above rests on."""
        spec = json.loads(
            fm.residplot(simple_frame, x="x", y="y", color="red", label="series A")
            .to_spec()
            .to_json()
        )
        color = spec["encoding"]["color"]
        assert color["field"] == "_label"
        assert color["scale"]["type"] == "ordinal"
        assert color["scale"]["domain"] == ["series A"]
        assert color["scale"]["range"] == ["#ff0000"]


# ---------------------------------------------------------------------------
# Compositions: robust, lowess, metrics
# ---------------------------------------------------------------------------


class TestColorCompositions:
    def test_robust_refits_per_group(self, offset_groups):
        svg = fm.residplot(
            offset_groups,
            x="x",
            y="y",
            color="g",
            robust=True,
            show_metrics=False,
            zero_line=False,
        ).to_svg()
        by_fill = _points_by_fill(svg)
        assert sorted(len(v) for v in by_fill.values()) == [12, 12]
        _assert_interleaved(by_fill)

    def test_robust_transform_carries_the_groupby(self, offset_groups):
        spec = json.loads(
            fm.residplot(offset_groups, x="x", y="y", color="g", robust=True).to_spec().to_json()
        )
        robust = next(t for t in spec["transforms"] if t.get("type") == "robust")
        assert robust["groupby"] == ["g"]

    def test_robust_with_label_renders_the_label(self, simple_frame):
        """The former silent no-op: ``Robust`` dropped ``_label``, so the
        colour encoding referenced a column the output did not have."""
        chart = fm.residplot(
            simple_frame,
            x="x",
            y="y",
            label="robust series",
            robust=True,
            show_metrics=False,
            zero_line=False,
        )
        svg = chart.to_svg()
        assert ">robust series<" in svg
        spec = json.loads(chart.to_spec().to_json())
        robust = next(t for t in spec["transforms"] if t.get("type") == "robust")
        assert robust["groupby"] == ["_label"]
        assert spec["encoding"]["color"]["field"] == "_label"

    def test_lowess_draws_one_curve_per_group(self, offset_groups):
        chart = fm.residplot(
            offset_groups,
            x="x",
            y="y",
            color="g",
            lowess=True,
            show_metrics=False,
            zero_line=False,
        )
        svg = chart.to_svg()
        curves = _polylines(svg)
        assert len(curves) == 2, f"expected one lowess curve per group, got {len(curves)}"
        # Equal to the POINT colours, not merely two of something: an overlay
        # whose palette diverged from the points' would still be "two distinct
        # strokes" while drawing each group's curve in the other group's
        # colour -- the exact failure the explicit layer channel prevents.
        strokes = {re.search(_ATTR.format("stroke"), c).group(1) for c in curves}
        assert strokes == set(_points_by_fill(svg)), (
            f"overlay strokes {sorted(strokes)} do not match the point fills "
            f"{sorted(_points_by_fill(svg))}"
        )
        # The overlay layer must carry the grouping channel itself. A line mark
        # draws ONE polyline through the rows it is handed, and what splits
        # these two is the colour channel; today a layer that names none
        # inherits the chart-level one, but that inheritance (#106) is
        # deliberately left in flux by this batch, and the overlay reads a
        # different batch than the chart-level encoding was written against.
        spec = json.loads(chart.to_spec().to_json())
        lowess_layer = next(layer for layer in spec["layers"] if layer["name"] == "lowess")
        assert lowess_layer["encoding"]["color"]["field"] == "g"

    def test_lowess_ungrouped_still_draws_one_curve(self, offset_groups):
        svg = fm.residplot(
            offset_groups, x="x", y="y", lowess=True, show_metrics=False, zero_line=False
        ).to_svg()
        assert len(_polylines(svg)) == 1

    def test_bare_literal_leaves_the_lowess_curve_at_the_default_stroke(self, simple_frame):
        """A deliberate divergence from seaborn, pinned so it stays deliberate.

        seaborn's ``color=`` tints the lowess line as well as the points.
        Spec §4 scopes the paint to the point layer's mark kwargs, so a bare
        literal (no ``label=``) leaves the curve at the theme's stroke.
        Without this assertion, tinting the curve would look like a bug fix
        rather than the contract change it would be. (The literal + ``label=``
        combination is different and deliberately does tint the curve: there
        the colour rides the label's own scale, which the overlay shares.)
        """
        kwargs = dict(x="x", y="y", lowess=True, show_metrics=False, zero_line=False)
        painted = _polylines(fm.residplot(simple_frame, color="red", **kwargs).to_svg())
        default = _polylines(fm.residplot(simple_frame, **kwargs).to_svg())
        assert len(painted) == len(default) == 1
        painted_stroke = re.search(_ATTR.format("stroke"), painted[0]).group(1)
        assert painted_stroke != "#ff0000", "the bare literal tinted the lowess curve"
        assert painted_stroke == re.search(_ATTR.format("stroke"), default[0]).group(1)

    def test_grouped_show_metrics_suppresses_the_corner(self, offset_groups):
        grouped = fm.residplot(offset_groups, x="x", y="y", color="g").to_svg()
        assert "R²" not in grouped
        pooled = fm.residplot(offset_groups, x="x", y="y").to_svg()
        assert "R²" in pooled

    def test_labelled_show_metrics_keeps_the_corner(self, simple_frame):
        """``label=`` groups, but into exactly ONE level (``pl.lit``), so the
        transform emits one correct corner and the per-group rationale for
        suppression does not apply. Gating on any grouping would take
        R²/RMSE/MAE away from a default ``residplot(..., label="…")`` call for
        no benefit; the gate reads `color_arg.group_field is None`, which is
        the same distinction ``lmplot`` draws on a real multi-level ``hue``.
        """
        chart = fm.residplot(simple_frame, x="x", y="y", label="L")
        svg = chart.to_svg()
        assert "R²" in svg
        assert ">L<" in svg, "the label legend entry should still render"
        spec = json.loads(chart.to_spec().to_json())
        smooth = next(t for t in spec["transforms"] if t.get("type") == "smooth")
        assert smooth["groupby"] == ["_label"]
        assert smooth["inject_metrics"] is True

    def test_colour_column_suppresses_even_beside_a_label(self, offset_groups):
        """The gate keys on the ``color=`` grouping, so adding a label to a
        colour-grouped chart does not resurrect a corner that has no single
        value to show."""
        chart = fm.residplot(offset_groups, x="x", y="y", color="g", label="L")
        assert "R²" not in chart.to_svg()

    def test_grouped_metrics_transform_flag_is_off(self, offset_groups):
        spec = json.loads(fm.residplot(offset_groups, x="x", y="y", color="g").to_spec().to_json())
        smooth = next(t for t in spec["transforms"] if t.get("type") == "smooth")
        assert smooth.get("inject_metrics") in (False, None)


# ---------------------------------------------------------------------------
# The untouched default
# ---------------------------------------------------------------------------


class TestColorNone:
    def test_default_chart_is_untouched(self, simple_frame):
        """``color=None`` must not perturb the pre-existing default path: no
        grouping, no colour scale, no mark-level paint, metrics corner kept."""
        chart = fm.residplot(simple_frame, x="x", y="y")
        spec = json.loads(chart.to_spec().to_json())
        smooth = next(t for t in spec["transforms"] if t.get("type") == "smooth")
        assert "groupby" not in smooth or smooth["groupby"] == []
        assert smooth["inject_metrics"] is True
        svg = chart.to_svg()
        assert "R²" in svg
        assert len(_points_by_fill(svg)) == 1
        assert chart.to_svg() == fm.residplot(simple_frame, x="x", y="y", color=None).to_svg()
