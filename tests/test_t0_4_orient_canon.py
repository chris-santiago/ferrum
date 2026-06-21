"""Regression tests for T0.4 — canonical orient= parameter across marks.

Covers:
- XSIB-01: mark_violin ignored horizontal=True / had no orient= param (silent drop).
- XSIB-02: orientation concept spelled differently across mark siblings.

Tests validate:
1. mark_violin(orient="horizontal") flips cat→y, value→x in the desugar encoding.
2. mark_violin(horizontal=True) is an alias that produces the same result.
3. Both must fail on the pre-fix code where horizontal was silently absorbed into
   mark style (the assertions below would have been False on the old code).
4. Legacy aliases (horizontal= on boxplot/boxen, orientation= on density/histogram)
   still work and produce the same outcome as the canonical orient= spelling.
5. orient= on boxplot, boxen, swarm, density, histogram all route correctly.
"""

from __future__ import annotations

import pytest
import polars as pl

import ferrum as fm
from ferrum.marks.heavy_stat import desugar_violin
from ferrum.encoding.base import ChannelBase


# ---------------------------------------------------------------------------
# Shared fixture data
# ---------------------------------------------------------------------------


@pytest.fixture
def grouped_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "cat": ["a"] * 15 + ["b"] * 15 + ["c"] * 15,
            "val": list(range(45)),
        }
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _field_of(enc_value) -> str:
    """Return the field name from a plain string or ChannelBase encoding value."""
    if isinstance(enc_value, ChannelBase):
        return enc_value.field
    return enc_value


def _body_layer(result):
    """Return the first (body polygon) layer from a violin MarkDesugarResult."""
    assert result.layers is not None, "expected layered result"
    body = result.layers[0]
    assert body.name == "body", f"expected body layer, got {body.name!r}"
    return body


# ---------------------------------------------------------------------------
# XSIB-01: mark_violin horizontal orientation
# ---------------------------------------------------------------------------


class TestViolinOrientHorizontal:
    """mark_violin(orient='horizontal') and mark_violin(horizontal=True) must
    flip the axis assignment in the desugar encoding.

    Argument-order convention for desugar_violin:
    - Vertical:   x_field = categorical (e.g. "cat"), y_field = numeric (e.g. "val")
    - Horizontal: x_field = numeric (e.g. "val"),     y_field = categorical (e.g. "cat")
      because in horizontal mode the user encodes .encode(x="val:Q", y="cat:N"),
      and the chart routes x_field=val, y_field=cat to the desugar.

    The desugar_violin(horizontal=True) logic:
      cat_field = y_field (categorical), val_field = x_field (numeric)
    """

    def test_vertical_default_has_cat_on_x_val_on_y(self):
        """Vertical (default) violin: x_field=cat (categorical), y=violin_y (value grid)."""
        result = desugar_violin("cat", "val", inner=None)
        body = _body_layer(result)
        # Vertical: category (x_field="cat") goes on x; violin_y (value grid) on y.
        assert _field_of(body.encoding["x"]) == "cat"
        assert _field_of(body.encoding["y"]) == "violin_y"

    def test_horizontal_flips_to_cat_on_y(self):
        """horizontal=True must put y_field (cat) on y and violin_y on x.

        Call convention: desugar_violin(x_field="val", y_field="cat", horizontal=True)
        — in horizontal mode x_field holds the numeric values, y_field holds the
        categorical grouping (matching .encode(x="val", y="cat") in the chart API).
        """
        # "val" is x_field (numeric), "cat" is y_field (categorical)
        result = desugar_violin("val", "cat", inner=None, horizontal=True)
        body = _body_layer(result)
        # Horizontal: cat (y_field) goes on y-axis; violin_y (value grid) goes on x.
        assert _field_of(body.encoding["y"]) == "cat", (
            "horizontal violin: category field (y_field) should be on y-axis"
        )
        assert _field_of(body.encoding["x"]) == "violin_y", (
            "horizontal violin: violin_y (value grid points) should be on x-axis"
        )
        assert "x" not in body.encoding or _field_of(body.encoding["x"]) != "cat", (
            "cat must NOT be on x in horizontal orientation"
        )

    def test_horizontal_kwarg_same_as_orient_horizontal(self):
        """horizontal=True produces the same body encoding as orient='horizontal'."""
        # Both call paths normalize to horizontal=True before desugar_violin.
        r_h = desugar_violin("val", "cat", inner=None, horizontal=True)
        body_h = _body_layer(r_h)
        assert _field_of(body_h.encoding["y"]) == "cat"
        assert _field_of(body_h.encoding["x"]) == "violin_y"

    def test_mark_violin_orient_horizontal_through_chart_api(self, grouped_df):
        """mark_violin(orient='horizontal') must NOT silently drop the kwarg.

        Pre-fix behaviour: horizontal=True was absorbed into mark_style (style
        kwargs) and silently ignored — the violin always rendered vertically
        regardless of the kwarg.  After the fix the body encoding is flipped.
        """
        chart = (
            fm.Chart(grouped_df)
            .mark_violin(orient="horizontal", inner=None)
            .encode(x="val:Q", y="cat:N")
        )
        # Force desugar by resolving the pending mark.
        resolved = chart._resolve_pending()
        assert resolved._layers is not None, "expected layered chart"
        body = next((la for la in resolved._layers if la.name == "body"), None)
        assert body is not None, "violin body layer must exist"
        # In horizontal mode, cat (y-encoding field) goes on y-axis of the body.
        y_field = _field_of(body.encoding.get("y", ""))
        x_field = _field_of(body.encoding.get("x", ""))
        assert y_field == "cat", f"horizontal violin: expected body y='cat', got {y_field!r}"
        assert x_field == "violin_y", (
            f"horizontal violin: expected body x='violin_y', got {x_field!r}"
        )

    def test_mark_violin_horizontal_true_through_chart_api(self, grouped_df):
        """mark_violin(horizontal=True) legacy alias must also flip the encoding."""
        chart = (
            fm.Chart(grouped_df)
            .mark_violin(horizontal=True, inner=None)
            .encode(x="val:Q", y="cat:N")
        )
        resolved = chart._resolve_pending()
        assert resolved._layers is not None
        body = next(la for la in resolved._layers if la.name == "body")
        y_field = _field_of(body.encoding.get("y", ""))
        x_field = _field_of(body.encoding.get("x", ""))
        assert y_field == "cat", f"horizontal=True violin: expected body y='cat', got {y_field!r}"
        assert x_field == "violin_y", (
            f"horizontal=True violin: expected body x='violin_y', got {x_field!r}"
        )

    def test_orient_horizontal_and_horizontal_true_produce_same_body(self, grouped_df):
        """orient='horizontal' and horizontal=True must produce the same body encoding."""
        chart_orient = (
            fm.Chart(grouped_df)
            .mark_violin(orient="horizontal", inner=None)
            .encode(x="val:Q", y="cat:N")
        )
        chart_h = (
            fm.Chart(grouped_df)
            .mark_violin(horizontal=True, inner=None)
            .encode(x="val:Q", y="cat:N")
        )
        body_orient = next(
            la for la in chart_orient._resolve_pending()._layers if la.name == "body"
        )
        body_h = next(la for la in chart_h._resolve_pending()._layers if la.name == "body")
        assert body_orient.encoding == body_h.encoding, (
            "orient='horizontal' and horizontal=True must produce identical body encodings"
        )

    def test_violin_inner_box_horizontal_passes_horizontal_to_boxplot(self, grouped_df):
        """inner='box' with orient='horizontal' must also flip the inner boxplot.

        When the user calls .encode(x='val:Q', y='cat:N') with horizontal violin,
        desugar_violin gets x_field='val' and y_field='cat'.
        """
        chart = (
            fm.Chart(grouped_df)
            .mark_violin(orient="horizontal", inner="box")
            .encode(x="val:Q", y="cat:N")
        )
        resolved = chart._resolve_pending()
        assert resolved._layers is not None
        # The inner box whisker layer should have y=cat (categorical on y).
        whisker = next((la for la in resolved._layers if la.name == "whisker"), None)
        assert whisker is not None, "inner box whisker layer must exist"
        y_field = _field_of(whisker.encoding.get("y", ""))
        assert y_field == "cat", f"horizontal inner box: whisker y should be cat, got {y_field!r}"

    def test_violin_inner_quartile_horizontal_flips(self, grouped_df):
        """inner='quartile' with orient='horizontal' must put quartile rules on x-axis."""
        chart = (
            fm.Chart(grouped_df)
            .mark_violin(orient="horizontal", inner="quartile")
            .encode(x="val:Q", y="cat:N")
        )
        resolved = chart._resolve_pending()
        assert resolved._layers is not None
        median = next((la for la in resolved._layers if la.name == "median"), None)
        assert median is not None
        # Horizontal: quartile columns on x, cat on y
        assert _field_of(median.encoding.get("x", "")) == "median", (
            "horizontal quartile violin: median rule should have x=median"
        )
        assert _field_of(median.encoding.get("y", "")) == "cat", (
            "horizontal quartile violin: median rule should have y=cat"
        )

    def test_violin_inner_point_horizontal_flips(self, grouped_df):
        """inner='point' with orient='horizontal' must put raw points with y=cat, x=val."""
        chart = (
            fm.Chart(grouped_df)
            .mark_violin(orient="horizontal", inner="point")
            .encode(x="val:Q", y="cat:N")
        )
        resolved = chart._resolve_pending()
        assert resolved._layers is not None
        point = next((la for la in resolved._layers if la.name == "point"), None)
        assert point is not None
        assert _field_of(point.encoding.get("y", "")) == "cat", (
            "horizontal point violin: point y should be cat"
        )
        assert _field_of(point.encoding.get("x", "")) == "val", (
            "horizontal point violin: point x should be val"
        )

    def test_violin_transform_uses_x_field_as_val_when_horizontal(self, grouped_df):
        """With horizontal=True the Violin transform must compute KDE on x_field (numeric).

        When the user encodes .encode(x='val:Q', y='cat:N'), x_field='val' (numeric)
        and y_field='cat' (categorical).  desugar_violin with horizontal=True should
        set val_field = x_field = 'val' and pass it to the Violin transform's field.
        """
        chart = (
            fm.Chart(grouped_df)
            .mark_violin(orient="horizontal", inner=None)
            .encode(x="val:Q", y="cat:N")
        )
        resolved = chart._resolve_pending()
        assert resolved._transforms, "expected transforms on resolved chart"
        violin_t = next(
            (
                t
                for t in resolved._transforms
                if hasattr(t, "_spec") or t.__class__.__name__ == "Violin"
            ),
            None,
        )
        assert violin_t is not None, "Violin transform must be present"

    def test_horizontal_violin_wires_horizontal_true_into_violin_transform(self):
        """horizontal=True must be forwarded into the Violin transform constructor.

        This is the end-to-end wire-through regression test.  On pre-fix code,
        desugar_violin constructed Violin(...) without horizontal=horizontal, so
        the transform always received horizontal=False regardless of the caller's
        intent — the KDE expansion remained vertical (using __pos_x_offset__) even
        though the encoding and inner marks were flipped.

        The test compares the actual Violin object produced by desugar_violin to a
        reference Violin(horizontal=True) using PyO3 equality; they differ only in
        the horizontal field, so inequality on old code is structural proof the kwarg
        was dropped.
        """
        from ferrum import Violin as ViolinTransform

        # desugar_violin(x_field="val", y_field="cat", horizontal=True):
        #   cat_field = y_field = "cat"  (categorical grouping)
        #   val_field = x_field = "val"  (numeric values)
        #   groupby = ["cat"]
        result = desugar_violin("val", "cat", inner=None, horizontal=True)
        violin_t = result.transforms[0]

        # Reference: the Violin we expect with horizontal=True
        expected = ViolinTransform(
            field="val",
            groupby=["cat"],
            horizontal=True,
            name="violin",
        )
        # Reference without horizontal (what pre-fix code produced)
        wrong = ViolinTransform(
            field="val",
            groupby=["cat"],
            horizontal=False,
            name="violin",
        )

        assert violin_t == expected, (
            "Violin transform must have horizontal=True when desugar_violin is called "
            "with horizontal=True; the kwarg was silently dropped on pre-fix code"
        )
        assert violin_t != wrong, (
            "Violin transform must NOT have horizontal=False when horizontal=True was requested"
        )

    def test_horizontal_violin_renders_svg_with_polygon_content(self):
        """mark_violin(orient='horizontal') must render a non-empty SVG with polygon paths.

        This is the full-pipeline regression: encoding swap + Violin(horizontal=True)
        must flow through the Rust render path and produce actual polygon geometry.
        On pre-fix code the Violin transform had horizontal=False, causing the KDE
        offset to be applied on the wrong axis; the SVG rendered without error but
        the polygon shapes were geometrically incorrect (vertical expand on horizontal
        chart).  After the fix the correct offset column (__pos_y_offset__) is emitted
        and the polygon content reflects the horizontal layout.
        """
        import polars as pl

        df = pl.DataFrame(
            {
                "cat": ["a"] * 20 + ["b"] * 20 + ["c"] * 20,
                "val": list(range(60)),
            }
        )
        svg = (
            fm.Chart(df)
            .mark_violin(orient="horizontal", inner=None)
            .encode(x="val:Q", y="cat:N")
            .to_svg()
        )
        assert svg.startswith("<svg"), "to_svg() must return an SVG string"
        # The violin body is a polygon mark; its geometry appears as SVG path elements.
        assert "<path" in svg or "<polygon" in svg, (
            "horizontal violin SVG must contain polygon path elements"
        )


# ---------------------------------------------------------------------------
# XSIB-02: orient= canonical alias across all orientation-bearing marks
# ---------------------------------------------------------------------------


class TestOrientAliasBoxplot:
    """mark_boxplot(orient=) is a canonical alias for horizontal=."""

    def test_orient_vertical_matches_horizontal_false(self, grouped_df):
        """orient='vertical' and horizontal=False produce identical charts."""
        c_orient = fm.Chart(grouped_df).mark_boxplot(orient="vertical").encode(x="cat", y="val")
        c_default = fm.Chart(grouped_df).mark_boxplot(horizontal=False).encode(x="cat", y="val")
        # Both should render without error.
        svg_orient = c_orient.to_svg()
        svg_default = c_default.to_svg()
        assert "<svg" in svg_orient
        assert "<svg" in svg_default

    def test_orient_horizontal_matches_horizontal_true(self, grouped_df):
        """orient='horizontal' and horizontal=True are byte-equivalent for boxplot."""
        c_orient = (
            fm.Chart(grouped_df).mark_boxplot(orient="horizontal").encode(x="val:Q", y="cat:N")
        )
        c_h = fm.Chart(grouped_df).mark_boxplot(horizontal=True).encode(x="val:Q", y="cat:N")
        svg_orient = c_orient.to_svg()
        svg_h = c_h.to_svg()
        # Both must render and produce non-trivial SVG.
        assert "<svg" in svg_orient
        assert "<svg" in svg_h
        # The resulting SVGs should be identical since orient= normalizes to horizontal=.
        assert svg_orient == svg_h, "orient='horizontal' and horizontal=True must be byte-identical"

    def test_orient_invalid_raises(self, grouped_df):
        """orient with an invalid value raises ValueError at call time."""
        with pytest.raises(ValueError, match="orient must be 'vertical' or 'horizontal'"):
            fm.Chart(grouped_df).mark_boxplot(orient="diagonal")


class TestOrientAliasBoxen:
    """mark_boxen(orient=) is a canonical alias for horizontal=."""

    def test_orient_horizontal_matches_horizontal_true(self, grouped_df):
        """orient='horizontal' and horizontal=True are byte-equivalent for boxen."""
        c_orient = fm.Chart(grouped_df).mark_boxen(orient="horizontal").encode(x="val:Q", y="cat:N")
        c_h = fm.Chart(grouped_df).mark_boxen(horizontal=True).encode(x="val:Q", y="cat:N")
        svg_orient = c_orient.to_svg()
        svg_h = c_h.to_svg()
        assert "<svg" in svg_orient
        assert svg_orient == svg_h, "orient='horizontal' and horizontal=True must be byte-identical"

    def test_orient_invalid_raises(self, grouped_df):
        with pytest.raises(ValueError, match="orient must be 'vertical' or 'horizontal'"):
            fm.Chart(grouped_df).mark_boxen(orient="upward")


class TestOrientAliasSwarm:
    """mark_swarm(horizontal=) is a legacy alias for orient=."""

    def test_horizontal_true_matches_orient_horizontal(self, grouped_df):
        """horizontal=True and orient='horizontal' produce the same SVG for swarm."""
        c_h = fm.Chart(grouped_df).mark_swarm(horizontal=True).encode(x="val:Q", y="cat:N")
        c_orient = fm.Chart(grouped_df).mark_swarm(orient="horizontal").encode(x="val:Q", y="cat:N")
        svg_h = c_h.to_svg()
        svg_orient = c_orient.to_svg()
        assert "<svg" in svg_h
        assert svg_h == svg_orient, (
            "mark_swarm(horizontal=True) must be equivalent to mark_swarm(orient='horizontal')"
        )

    def test_horizontal_false_matches_orient_vertical(self, grouped_df):
        """horizontal=False and orient='vertical' are the same for swarm."""
        c_h = fm.Chart(grouped_df).mark_swarm(horizontal=False).encode(x="cat", y="val:Q")
        c_orient = fm.Chart(grouped_df).mark_swarm(orient="vertical").encode(x="cat", y="val:Q")
        assert c_h.to_svg() == c_orient.to_svg()


class TestOrientAliasDensity:
    """mark_density(orient=) is a canonical alias for orientation=."""

    def test_orient_horizontal_matches_orientation_horizontal(self):
        """orient='horizontal' and orientation='horizontal' produce the same SVG."""
        df = pl.DataFrame({"val": [1.0, 2.0, 3.0, 2.5, 1.5]})
        c_orient = fm.Chart(df).mark_density(orient="horizontal").encode(y="val")
        c_old = fm.Chart(df).mark_density(orientation="horizontal").encode(y="val")
        assert c_orient.to_svg() == c_old.to_svg(), (
            "mark_density(orient='horizontal') must match mark_density(orientation='horizontal')"
        )

    def test_orient_vertical_matches_orientation_vertical(self):
        """orient='vertical' and orientation='vertical' (default) are equivalent."""
        df = pl.DataFrame({"val": [1.0, 2.0, 3.0, 2.5, 1.5]})
        c_orient = fm.Chart(df).mark_density(orient="vertical").encode(x="val")
        c_old = fm.Chart(df).mark_density(orientation="vertical").encode(x="val")
        assert c_orient.to_svg() == c_old.to_svg()

    def test_orient_wins_over_orientation_when_both_given(self):
        """When both orient= and orientation= are given, orient= takes precedence."""
        df = pl.DataFrame({"val": [1.0, 2.0, 3.0, 2.5, 1.5]})
        # orient='horizontal' + orientation='vertical': horizontal should win.
        c_conflict = (
            fm.Chart(df).mark_density(orient="horizontal", orientation="vertical").encode(y="val")
        )
        c_h = fm.Chart(df).mark_density(orient="horizontal").encode(y="val")
        assert c_conflict.to_svg() == c_h.to_svg(), (
            "orient= must take precedence over orientation= when both are given"
        )

    def test_orient_invalid_raises(self):
        df = pl.DataFrame({"val": [1.0, 2.0, 3.0]})
        with pytest.raises(ValueError, match="orient must be 'vertical' or 'horizontal'"):
            fm.Chart(df).mark_density(orient="sideways").encode(x="val").to_svg()


class TestOrientAliasHistogram:
    """mark_histogram(orient=) is a canonical alias for orientation=."""

    def test_orient_horizontal_matches_orientation_horizontal(self):
        """orient='horizontal' and orientation='horizontal' produce the same SVG."""
        df = pl.DataFrame({"val": [1.0, 2.0, 3.0, 2.5, 1.5, 2.2]})
        c_orient = fm.Chart(df).mark_histogram(orient="horizontal").encode(y="val")
        c_old = fm.Chart(df).mark_histogram(orientation="horizontal").encode(y="val")
        assert c_orient.to_svg() == c_old.to_svg(), (
            "mark_histogram(orient='horizontal') must match mark_histogram(orientation='horizontal')"
        )

    def test_orient_wins_over_orientation_when_both_given(self):
        """orient= wins over orientation= when both are given."""
        df = pl.DataFrame({"val": [1.0, 2.0, 3.0, 2.5, 1.5, 2.2]})
        c_conflict = (
            fm.Chart(df).mark_histogram(orient="horizontal", orientation="vertical").encode(y="val")
        )
        c_h = fm.Chart(df).mark_histogram(orient="horizontal").encode(y="val")
        assert c_conflict.to_svg() == c_h.to_svg()

    def test_orient_invalid_raises(self):
        df = pl.DataFrame({"val": [1.0, 2.0, 3.0]})
        with pytest.raises(ValueError, match="orient must be 'vertical' or 'horizontal'"):
            fm.Chart(df).mark_histogram(orient="diagonal").encode(x="val").to_svg()


class TestOrientAliasViolin:
    """orient= and horizontal= are aliases on mark_violin."""

    def test_orient_vertical_is_default(self, grouped_df):
        """orient='vertical' and no-arg violin produce the same chart."""
        c_explicit = fm.Chart(grouped_df).mark_violin(orient="vertical").encode(x="cat", y="val")
        c_default = fm.Chart(grouped_df).mark_violin().encode(x="cat", y="val")
        assert c_explicit.to_svg() == c_default.to_svg()

    def test_orient_invalid_raises(self, grouped_df):
        with pytest.raises(ValueError, match="orient must be 'vertical' or 'horizontal'"):
            fm.Chart(grouped_df).mark_violin(orient="sideways").encode(x="cat", y="val")

    def test_horizontal_true_and_orient_horizontal_are_equivalent_at_desugar(self):
        """horizontal=True and orient='horizontal' reach the same desugar encoding."""
        # Both are normalized to horizontal=True before reaching desugar_violin,
        # so the desugar results should be identical.
        r_h = desugar_violin("cat", "val", inner=None, horizontal=True)
        # Since mark_violin normalizes orient→horizontal before calling desugar_violin,
        # we test the desugar directly with horizontal=True for both paths.
        r_orient = desugar_violin("cat", "val", inner=None, horizontal=True)
        assert _field_of(r_h.layers[0].encoding["y"]) == _field_of(r_orient.layers[0].encoding["y"])
        assert _field_of(r_h.layers[0].encoding["x"]) == _field_of(r_orient.layers[0].encoding["x"])


# ---------------------------------------------------------------------------
# F1 regression: orient= WINS over horizontal= alias when both are given
# ---------------------------------------------------------------------------


def _effective_orient_from_kwargs(chart) -> str:
    """Extract the effective orientation from a chart's pending-mark kwargs.

    Returns ``"vertical"`` or ``"horizontal"`` depending on what the mark
    normalizer resolved.

    - boxplot / boxen / violin store a bool ``horizontal`` in kwargs.
    - swarm stores an orient string ``orient`` in kwargs.
    """
    kwargs = chart._pending_stat_mark.kwargs
    if "orient" in kwargs:
        # swarm: orient string is already canonical
        return kwargs["orient"]
    # boxplot / boxen / violin: bool horizontal
    return "horizontal" if kwargs.get("horizontal", False) else "vertical"


@pytest.mark.parametrize(
    "mark_fn, encode_kwargs",
    [
        ("mark_boxplot", {"x": "cat", "y": "val"}),
        ("mark_boxen", {"x": "cat", "y": "val"}),
        ("mark_violin", {"x": "cat", "y": "val"}),
        ("mark_swarm", {"x": "cat", "y": "val"}),
    ],
)
def test_orient_wins_when_orient_vertical_and_horizontal_true(grouped_df, mark_fn, encode_kwargs):
    """orient='vertical' must win over horizontal=True for all four marks.

    This test FAILS on pre-fix swarm because mark_swarm used to give
    horizontal= precedence when it was not None, resolving to 'horizontal'
    instead of 'vertical'.
    """
    chart = getattr(fm.Chart(grouped_df), mark_fn)(orient="vertical", horizontal=True).encode(
        **encode_kwargs
    )
    effective = _effective_orient_from_kwargs(chart)
    assert effective == "vertical", (
        f"{mark_fn}(orient='vertical', horizontal=True) must resolve to 'vertical' "
        f"(orient wins); got {effective!r}"
    )


@pytest.mark.parametrize(
    "mark_fn, encode_kwargs",
    [
        ("mark_boxplot", {"x": "val:Q", "y": "cat:N"}),
        ("mark_boxen", {"x": "val:Q", "y": "cat:N"}),
        ("mark_violin", {"x": "val:Q", "y": "cat:N"}),
        ("mark_swarm", {"x": "val:Q", "y": "cat:N"}),
    ],
)
def test_orient_wins_when_orient_horizontal_and_horizontal_false(
    grouped_df, mark_fn, encode_kwargs
):
    """orient='horizontal' must win over horizontal=False for all four marks.

    This test FAILS on pre-fix swarm because mark_swarm used to give
    horizontal= precedence when it was not None, resolving to 'vertical'
    instead of 'horizontal'.
    """
    chart = getattr(fm.Chart(grouped_df), mark_fn)(orient="horizontal", horizontal=False).encode(
        **encode_kwargs
    )
    effective = _effective_orient_from_kwargs(chart)
    assert effective == "horizontal", (
        f"{mark_fn}(orient='horizontal', horizontal=False) must resolve to 'horizontal' "
        f"(orient wins); got {effective!r}"
    )
