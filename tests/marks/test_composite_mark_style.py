"""Tests for constant mark-style kwargs on single-mark composite desugars.

Covers the three-tuple desugars (mark_density, mark_histogram) and the
hex layered desugar (mark_hex), which were the gap not covered by
tests/test_composite_kwargs.py (which covers desugar_errorbar, desugar_boxplot,
desugar_errorband, desugar_ribbon, and desugar_boxen).

Each test verifies one of:
- Style lands on _mark_kwargs (single-mark path) or layer mark_kwargs (layered path).
- Aliases (color->fill, alpha->opacity) are resolved by MarkBase before storage.
- Collision: a kwarg that IS a declared desugar param stays on the transform side.
- Unknown kwargs raise TypeError.
- Byte-identity: no style kwargs => _mark_kwargs is empty (no style injected).
"""

from __future__ import annotations

import re

import polars as pl
import pytest

import ferrum as fm


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def density_df() -> pl.DataFrame:
    """Minimal DataFrame for 1D density tests."""
    return pl.DataFrame(
        {
            "value": [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5],
        }
    )


@pytest.fixture()
def density_group_df() -> pl.DataFrame:
    """Two-group DataFrame for grouped density tests."""
    return pl.DataFrame(
        {
            "value": [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 4.5, 5.0, 5.5, 6.0],
            "group": ["a"] * 5 + ["b"] * 5,
        }
    )


@pytest.fixture()
def hex_df() -> pl.DataFrame:
    """Minimal two-column DataFrame for hex bin tests."""
    return pl.DataFrame(
        {
            "x": [1.0, 2.0, 1.5, 3.0, 2.5, 1.0, 2.0, 3.5, 4.0, 2.5],
            "y": [4.0, 5.0, 4.5, 3.0, 4.0, 3.5, 4.5, 5.0, 4.0, 3.0],
        }
    )


@pytest.fixture()
def smooth_df() -> pl.DataFrame:
    """20-row DataFrame for smooth/CI tests."""
    return pl.DataFrame(
        {
            "x": [float(i) for i in range(20)],
            "y": [float(i) * 0.8 + 0.5 for i in range(20)],
        }
    )


# ---------------------------------------------------------------------------
# mark_density — single-mark path
# ---------------------------------------------------------------------------


class TestMarkDensityStyle:
    def test_opacity_lands_on_mark_kwargs(self, density_df):
        resolved = (
            fm.Chart(density_df).mark_density(opacity=0.5).encode(x="value:Q")._resolve_pending()
        )
        assert resolved._mark_kwargs == {"opacity": 0.5}

    def test_mark_is_area_by_default(self, density_df):
        resolved = (
            fm.Chart(density_df).mark_density(opacity=0.5).encode(x="value:Q")._resolve_pending()
        )
        assert resolved._mark == "area"

    def test_stroke_width_lands_on_mark_kwargs(self, density_df):
        resolved = (
            fm.Chart(density_df)
            .mark_density(stroke_width=2)
            .encode(x="value:Q")
            ._resolve_pending()
        )
        assert resolved._mark_kwargs == {"stroke_width": 2}

    def test_alias_color_resolves_to_fill(self, density_df):
        """color= is aliased to fill= by MarkBase."""
        resolved = (
            fm.Chart(density_df)
            .mark_density(color="#888888")
            .encode(x="value:Q")
            ._resolve_pending()
        )
        assert resolved._mark_kwargs.get("fill") == "#888888"
        assert "color" not in resolved._mark_kwargs

    def test_alias_alpha_resolves_to_opacity(self, density_df):
        """alpha= is aliased to opacity= by MarkBase."""
        resolved = (
            fm.Chart(density_df)
            .mark_density(alpha=0.3)
            .encode(x="value:Q")
            ._resolve_pending()
        )
        assert resolved._mark_kwargs.get("opacity") == pytest.approx(0.3)
        assert "alpha" not in resolved._mark_kwargs

    def test_collision_fill_stays_on_transform(self, density_df):
        """fill is a declared desugar_density param; fill=False routes to transform, not style.

        The resulting mark is 'line' (not 'area'), confirming transform routing.
        """
        resolved = (
            fm.Chart(density_df).mark_density(fill=False).encode(x="value:Q")._resolve_pending()
        )
        assert resolved._mark == "line"
        # fill is consumed by the transform; it must NOT appear in _mark_kwargs
        assert "fill" not in resolved._mark_kwargs

    def test_collision_multiple_stays_on_transform(self, density_df):
        """multiple= is a declared desugar_density param and must not appear in _mark_kwargs."""
        resolved = (
            fm.Chart(density_df)
            .mark_density(multiple="stack")
            .encode(x="value:Q")
            ._resolve_pending()
        )
        assert "multiple" not in resolved._mark_kwargs
        # Stack position is set (confirms transform routing)
        assert resolved._position is not None

    def test_typo_raises_type_error(self, density_df):
        """An unknown kwarg raises TypeError at resolve time."""
        chart = fm.Chart(density_df).mark_density(notakwarg=1).encode(x="value:Q")
        with pytest.raises(TypeError, match="notakwarg"):
            chart._resolve_pending()

    def test_byte_identity_no_style(self, density_df):
        """No style kwargs => _mark_kwargs is empty (byte-identity invariant)."""
        resolved = (
            fm.Chart(density_df).mark_density().encode(x="value:Q")._resolve_pending()
        )
        assert not resolved._mark_kwargs  # empty dict or falsy

    def test_rendered_svg_contains_rgba_fill(self, density_group_df):
        """opacity=0.4 renders as rgba() fill colour in the SVG (ferrum bakes opacity into rgba).

        Asserts that the alpha component 0.4 appears in at least one rgba() call,
        allowing for float formatting like 0.400, 0.4, etc.
        """
        svg = (
            fm.Chart(density_group_df)
            .mark_density(multiple="layer", opacity=0.4)
            .encode(x="value:Q", color="group:N")
            .properties(width=400, height=250)
            .show_svg()
        )
        assert "rgba(" in svg, "Expected rgba() fill in SVG but got none"
        # Match alpha component 0.4 (may be formatted as 0.400, 0.4, etc.)
        rgba_values = re.findall(r"rgba\([^)]+\)", svg)
        assert any(
            re.search(r"0\.4\d*\)", v) for v in rgba_values
        ), f"Expected alpha ~0.4 in rgba() calls; found: {rgba_values[:5]}"


# ---------------------------------------------------------------------------
# mark_histogram — single-mark path
# ---------------------------------------------------------------------------


class TestMarkHistogramStyle:
    def test_opacity_lands_on_mark_kwargs(self, density_df):
        resolved = (
            fm.Chart(density_df)
            .mark_histogram(opacity=0.5)
            .encode(x="value:Q")
            ._resolve_pending()
        )
        assert resolved._mark_kwargs == {"opacity": 0.5}

    def test_mark_is_bar(self, density_df):
        resolved = (
            fm.Chart(density_df)
            .mark_histogram(opacity=0.5)
            .encode(x="value:Q")
            ._resolve_pending()
        )
        assert resolved._mark == "bar"

    def test_stroke_lands_on_mark_kwargs(self, density_df):
        resolved = (
            fm.Chart(density_df)
            .mark_histogram(stroke="red")
            .encode(x="value:Q")
            ._resolve_pending()
        )
        assert resolved._mark_kwargs.get("stroke") == "red"

    def test_collision_bin_count_stays_on_transform(self, density_df):
        """bin_count= is a declared desugar_histogram param; must not appear in _mark_kwargs."""
        resolved = (
            fm.Chart(density_df)
            .mark_histogram(bin_count=10)
            .encode(x="value:Q")
            ._resolve_pending()
        )
        assert "bin_count" not in resolved._mark_kwargs

    def test_typo_raises_type_error(self, density_df):
        chart = fm.Chart(density_df).mark_histogram(notakwarg=1).encode(x="value:Q")
        with pytest.raises(TypeError, match="notakwarg"):
            chart._resolve_pending()

    def test_byte_identity_no_style(self, density_df):
        resolved = (
            fm.Chart(density_df).mark_histogram().encode(x="value:Q")._resolve_pending()
        )
        assert not resolved._mark_kwargs  # empty dict or falsy


# ---------------------------------------------------------------------------
# mark_hex — layered path
# ---------------------------------------------------------------------------


class TestMarkHexStyle:
    def test_opacity_applied_to_polygon_layer(self, hex_df):
        """opacity= (not a desugar_hex param) routes to style and merges into the polygon layer."""
        resolved = (
            fm.Chart(hex_df).mark_hex(opacity=0.5).encode(x="x", y="y")._resolve_pending()
        )
        assert resolved._layers is not None
        poly_layer = next(l for l in resolved._layers if l.mark == "polygon")
        assert poly_layer.mark_kwargs.get("opacity") == 0.5

    def test_collision_cmap_stays_on_transform(self):
        """cmap= is a declared desugar_hex param; it must NOT appear as a mark-style key.

        After routing, cmap IS present in the polygon layer's mark_kwargs because
        desugar_hex stores it there explicitly — that is correct transform behavior.
        The test guards against cmap being treated as a style override (which would
        duplicate or overwrite the transform's value).
        """
        from ferrum.chart import _split_style_kwargs
        from ferrum.marks.heavy_stat import desugar_hex

        _, style_kwargs = _split_style_kwargs(desugar_hex, {"cmap": "viridis", "opacity": 0.7})
        assert "cmap" not in style_kwargs, "cmap must not appear in style_kwargs (it's a transform param)"
        assert "opacity" in style_kwargs

    def test_cmap_resolves_without_error(self, hex_df):
        """mark_hex(cmap='viridis') resolves without raising."""
        resolved = (
            fm.Chart(hex_df)
            .mark_hex(cmap="viridis")
            .encode(x="x", y="y")
            ._resolve_pending()
        )
        assert resolved._layers is not None

    def test_stroke_width_is_a_transform_param(self):
        """stroke_width= IS a declared desugar_hex param, routes to transform (not style)."""
        from ferrum.chart import _split_style_kwargs
        from ferrum.marks.heavy_stat import desugar_hex

        transform_kwargs, style_kwargs = _split_style_kwargs(
            desugar_hex, {"stroke_width": 1.5, "opacity": 0.5}
        )
        assert "stroke_width" in transform_kwargs
        assert "stroke_width" not in style_kwargs
        assert "opacity" in style_kwargs

    def test_typo_raises_type_error(self, hex_df):
        chart = fm.Chart(hex_df).mark_hex(notakwarg=1).encode(x="x", y="y")
        with pytest.raises(TypeError, match="notakwarg"):
            chart._resolve_pending()

    def test_byte_identity_no_style(self, hex_df):
        """No user style kwargs => polygon layer mark_kwargs contains only desugar defaults."""
        resolved_no_style = (
            fm.Chart(hex_df).mark_hex().encode(x="x", y="y")._resolve_pending()
        )
        poly = next(l for l in resolved_no_style._layers if l.mark == "polygon")
        # desugar_hex always sets 'detail' and 'opacity' defaults; no user keys added
        assert poly.mark_kwargs == {"detail": "hex_id", "opacity": 1.0}


# ---------------------------------------------------------------------------
# mark_smooth (layered, CI path) — style on every emitted layer
# ---------------------------------------------------------------------------


class TestMarkSmoothLayeredStyle:
    def test_opacity_on_all_layers(self, smooth_df):
        """opacity=0.4 must appear in every layer emitted by the CI smooth path."""
        chart = (
            fm.Chart(smooth_df)
            .mark_smooth(method="lm", ci=0.95, opacity=0.4)
            .encode(x="x", y="y")
        )
        resolved = chart._resolve_pending()
        assert resolved._layers is not None
        for layer in resolved._layers:
            assert layer.mark_kwargs is not None, f"Layer {layer.name!r} has no mark_kwargs"
            assert layer.mark_kwargs.get("opacity") == pytest.approx(0.4), (
                f"Layer {layer.name!r}: expected opacity=0.4, got {layer.mark_kwargs!r}"
            )

    def test_prior_point_layer_untouched(self, smooth_df):
        """mark_point().mark_smooth(stroke_width=4): prior scatter layer must NOT have stroke_width."""
        chart = (
            fm.Chart(smooth_df)
            .mark_point()
            .mark_smooth(method="lm", stroke_width=4)
            .encode(x="x", y="y")
        )
        resolved = chart._resolve_pending()
        assert resolved._layers is not None

        # First layer is the prior scatter (point); it must NOT carry stroke_width.
        prior = resolved._layers[0]
        assert prior.mark == "point"
        prior_mk = prior.mark_kwargs or {}
        assert "stroke_width" not in prior_mk, (
            f"Prior point layer must not carry stroke_width; got {prior_mk!r}"
        )

        # The smooth layer (second) must carry stroke_width=4.
        smooth_layers = [l for l in resolved._layers if l.mark == "line"]
        assert smooth_layers, "Expected at least one line layer from mark_smooth"
        line_mk = smooth_layers[-1].mark_kwargs or {}
        assert line_mk.get("stroke_width") == 4, (
            f"Smooth line layer must carry stroke_width=4; got {line_mk!r}"
        )
