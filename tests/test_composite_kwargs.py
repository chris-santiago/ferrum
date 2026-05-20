"""Tests for composite mark desugar functions accepting and forwarding **mark_kwargs.

Covers: desugar_errorbar, desugar_boxplot, desugar_errorband, desugar_ribbon,
desugar_boxen — all five composite desugars in composite.py.

Each test verifies:
1. Known style kwargs (stroke, opacity, etc.) are accepted without error.
2. Unknown kwargs raise TypeError from validate_user_mark_kwargs.
3. Default behavior (no user kwargs) is preserved as a regression guard.
"""

from __future__ import annotations

import pytest
import polars as pl


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def errorbar_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "group": ["a"] * 10 + ["b"] * 10,
            "val": list(range(10)) + list(range(5, 15)),
        }
    )


@pytest.fixture()
def boxplot_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "species": ["a"] * 5 + ["b"] * 5,
            "val": [1, 2, 3, 2, 1, 4, 5, 4, 5, 6],
        }
    )


@pytest.fixture()
def continuous_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "x": list(range(20)),
            "y": [float(i) for i in range(20)],
        }
    )


@pytest.fixture()
def ribbon_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "x": [1, 2, 3, 4, 5],
            "y_lo": [0.5, 1.5, 2.5, 3.5, 4.5],
            "y_hi": [1.5, 2.5, 3.5, 4.5, 5.5],
        }
    )


# ---------------------------------------------------------------------------
# desugar_errorbar
# ---------------------------------------------------------------------------


class TestDesugarErrorbar:
    def test_accepts_stroke_color(self, errorbar_df):
        from ferrum.marks.composite import desugar_errorbar

        result = desugar_errorbar("group", "val", stroke="red")
        assert result is not None
        assert len(result.layers) > 0

    def test_stroke_applied_to_all_layers(self, errorbar_df):
        from ferrum.marks.composite import desugar_errorbar

        result = desugar_errorbar("group", "val", stroke="red")
        for layer in result.layers:
            assert layer.mark_kwargs.get("stroke") == "red", (
                f"Layer {layer.name!r}: expected stroke='red', "
                f"got mark_kwargs={layer.mark_kwargs!r}"
            )

    def test_accepts_opacity(self, errorbar_df):
        from ferrum.marks.composite import desugar_errorbar

        result = desugar_errorbar("group", "val", opacity=0.5)
        assert result is not None

    def test_rejects_unknown_kwarg(self, errorbar_df):
        from ferrum.marks.composite import desugar_errorbar

        with pytest.raises(TypeError, match="banana"):
            desugar_errorbar("group", "val", banana=True)

    def test_default_stroke_without_override(self, errorbar_df):
        """Regression guard: the default theme:label stroke is present when no user override."""
        from ferrum.marks.composite import desugar_errorbar

        result = desugar_errorbar("group", "val")
        rule_layer = next(l for l in result.layers if l.mark == "rule")
        assert rule_layer.mark_kwargs.get("stroke") == "theme:label"

    def test_via_chart_api_accepts_stroke(self, errorbar_df):
        """End-to-end: mark_errorbar(stroke='red') must not raise at resolve time."""
        import ferrum as fm

        chart = fm.Chart(errorbar_df).mark_errorbar(stroke="red").encode(x="group", y="val")
        # _resolve_pending is called implicitly; force it explicitly here too.
        resolved = chart._resolve_pending()
        assert resolved is not None

    def test_via_chart_api_rejects_unknown_kwarg(self, errorbar_df):
        """End-to-end: mark_errorbar(banana=True) must raise TypeError at resolve time."""
        import ferrum as fm

        chart = fm.Chart(errorbar_df).mark_errorbar(banana=True).encode(x="group", y="val")
        with pytest.raises(TypeError, match="banana"):
            chart._resolve_pending()


# ---------------------------------------------------------------------------
# desugar_boxplot
# ---------------------------------------------------------------------------


class TestDesugarBoxplot:
    def test_accepts_stroke_color(self, boxplot_df):
        from ferrum.marks.composite import desugar_boxplot

        result = desugar_boxplot("species", "val", stroke="blue")
        assert result is not None
        assert len(result.layers) > 0

    def test_stroke_applied_to_layers(self, boxplot_df):
        from ferrum.marks.composite import desugar_boxplot

        result = desugar_boxplot("species", "val", stroke="blue")
        for layer in result.layers:
            assert layer.mark_kwargs.get("stroke") == "blue", (
                f"Layer {layer.name!r}: expected stroke='blue', "
                f"got mark_kwargs={layer.mark_kwargs!r}"
            )

    def test_accepts_opacity(self, boxplot_df):
        from ferrum.marks.composite import desugar_boxplot

        result = desugar_boxplot("species", "val", opacity=0.7)
        assert result is not None

    def test_rejects_unknown_kwarg(self, boxplot_df):
        from ferrum.marks.composite import desugar_boxplot

        with pytest.raises(TypeError, match="banana"):
            desugar_boxplot("species", "val", banana=True)

    def test_default_whisker_stroke_without_override(self, boxplot_df):
        """Regression guard: whisker rule defaults to theme:label stroke."""
        from ferrum.marks.composite import desugar_boxplot

        result = desugar_boxplot("species", "val")
        whisker = next(l for l in result.layers if l.name == "whisker")
        assert whisker.mark_kwargs.get("stroke") == "theme:label"

    def test_via_chart_api_accepts_stroke(self, boxplot_df):
        import ferrum as fm

        chart = fm.Chart(boxplot_df).mark_boxplot(stroke="blue").encode(x="species", y="val")
        resolved = chart._resolve_pending()
        assert resolved is not None

    def test_via_chart_api_rejects_unknown_kwarg(self, boxplot_df):
        import ferrum as fm

        chart = fm.Chart(boxplot_df).mark_boxplot(banana=True).encode(x="species", y="val")
        with pytest.raises(TypeError, match="banana"):
            chart._resolve_pending()


# ---------------------------------------------------------------------------
# desugar_errorband
# ---------------------------------------------------------------------------


class TestDesugarErrorband:
    def test_accepts_opacity(self, continuous_df):
        from ferrum.marks.composite import desugar_errorband

        result = desugar_errorband("x", "y", opacity=0.5)
        assert result is not None

    def test_opacity_applied_to_ribbon_layer(self, continuous_df):
        from ferrum.marks.composite import desugar_errorband

        result = desugar_errorband("x", "y", opacity=0.5)
        ribbon = next(l for l in result.layers if l.mark == "ribbon")
        assert ribbon.mark_kwargs.get("opacity") == 0.5

    def test_accepts_stroke_color(self, continuous_df):
        from ferrum.marks.composite import desugar_errorband

        result = desugar_errorband("x", "y", stroke="green")
        assert result is not None

    def test_rejects_unknown_kwarg(self, continuous_df):
        from ferrum.marks.composite import desugar_errorband

        with pytest.raises(TypeError, match="banana"):
            desugar_errorband("x", "y", banana=True)

    def test_default_opacity_without_override(self, continuous_df):
        """Regression guard: ribbon defaults to opacity=0.2."""
        from ferrum.marks.composite import desugar_errorband

        result = desugar_errorband("x", "y")
        ribbon = next(l for l in result.layers if l.mark == "ribbon")
        assert ribbon.mark_kwargs.get("opacity") == 0.2

    def test_via_chart_api_accepts_opacity(self, continuous_df):
        import ferrum as fm

        chart = fm.Chart(continuous_df).mark_errorband(opacity=0.5).encode(x="x", y="y")
        resolved = chart._resolve_pending()
        assert resolved is not None

    def test_via_chart_api_rejects_unknown_kwarg(self, continuous_df):
        import ferrum as fm

        chart = fm.Chart(continuous_df).mark_errorband(banana=True).encode(x="x", y="y")
        with pytest.raises(TypeError, match="banana"):
            chart._resolve_pending()


# ---------------------------------------------------------------------------
# desugar_ribbon
# ---------------------------------------------------------------------------


class TestDesugarRibbon:
    def test_accepts_fill(self, ribbon_df):
        from ferrum.marks.composite import desugar_ribbon

        result = desugar_ribbon("x", "y_lo", y2_field="y_hi", fill="#aabbcc")
        assert result is not None

    def test_fill_applied_to_ribbon_layer(self, ribbon_df):
        from ferrum.marks.composite import desugar_ribbon

        result = desugar_ribbon("x", "y_lo", y2_field="y_hi", fill="#aabbcc")
        ribbon = result.layers[0]
        assert ribbon.mark_kwargs.get("fill") == "#aabbcc"

    def test_accepts_opacity_override(self, ribbon_df):
        """opacity is a named param on desugar_ribbon, but user can override via mark_kwargs too."""
        from ferrum.marks.composite import desugar_ribbon

        result = desugar_ribbon("x", "y_lo", y2_field="y_hi", opacity=0.8)
        ribbon = result.layers[0]
        assert ribbon.mark_kwargs.get("opacity") == 0.8

    def test_rejects_unknown_kwarg(self, ribbon_df):
        from ferrum.marks.composite import desugar_ribbon

        with pytest.raises(TypeError, match="banana"):
            desugar_ribbon("x", "y_lo", y2_field="y_hi", banana=True)

    def test_via_chart_api_accepts_fill(self, ribbon_df):
        import ferrum as fm

        chart = fm.Chart(ribbon_df).mark_ribbon(fill="#aabbcc").encode(x="x", y="y_lo", y2="y_hi")
        resolved = chart._resolve_pending()
        assert resolved is not None

    def test_via_chart_api_rejects_unknown_kwarg(self, ribbon_df):
        import ferrum as fm

        chart = fm.Chart(ribbon_df).mark_ribbon(banana=True).encode(x="x", y="y_lo", y2="y_hi")
        with pytest.raises(TypeError, match="banana"):
            chart._resolve_pending()


# ---------------------------------------------------------------------------
# desugar_boxen
# ---------------------------------------------------------------------------


class TestDesugarBoxen:
    def test_accepts_opacity(self, boxplot_df):
        from ferrum.marks.composite import desugar_boxen

        result = desugar_boxen("species", "val", opacity=0.9)
        assert result is not None

    def test_opacity_applied_to_depth_layers(self, boxplot_df):
        from ferrum.marks.composite import desugar_boxen

        result = desugar_boxen("species", "val", opacity=0.9)
        depth_layers = [l for l in result.layers if l.name.startswith("depth_")]
        assert len(depth_layers) > 0
        for layer in depth_layers:
            assert layer.mark_kwargs.get("opacity") == 0.9, (
                f"Layer {layer.name!r}: expected opacity=0.9, got mark_kwargs={layer.mark_kwargs!r}"
            )

    def test_accepts_stroke(self, boxplot_df):
        from ferrum.marks.composite import desugar_boxen

        result = desugar_boxen("species", "val", stroke="purple")
        assert result is not None

    def test_rejects_unknown_kwarg(self, boxplot_df):
        from ferrum.marks.composite import desugar_boxen

        with pytest.raises(TypeError, match="banana"):
            desugar_boxen("species", "val", banana=True)

    def test_via_chart_api_accepts_opacity(self, boxplot_df):
        import ferrum as fm

        chart = fm.Chart(boxplot_df).mark_boxen(opacity=0.9).encode(x="species", y="val")
        resolved = chart._resolve_pending()
        assert resolved is not None

    def test_via_chart_api_rejects_unknown_kwarg(self, boxplot_df):
        import ferrum as fm

        chart = fm.Chart(boxplot_df).mark_boxen(banana=True).encode(x="species", y="val")
        with pytest.raises(TypeError, match="banana"):
            chart._resolve_pending()
