"""Cohesion-campaign Step-4 regression guards (S4FIX, two sibling-drift defects).

ITEM 1 — ``min_band``/``max_band`` deprecated-alias resolution is single-sourced
on ``_resolve_band_alias`` across all three surfaces (``fm.Axis``,
``fm.configure.AxisConfig``, and ``Chart.configure_axis``).  Before the fix,
``Axis``/``AxisConfig`` hand-inlined an ``if alias is not None:`` block that
diverged from the canonical ``_MISSING``-sentinel helper on the explicit-``None``
edge: ``Axis(min_band=5, min_extent=None)`` silently accepted while
``configure_axis(min_band=5, min_extent=None)`` raised.  These tests assert the
three surfaces now behave IDENTICALLY for (a) the single-alias path, (b) the
double-supply path, and (c) the previously-divergent explicit-``None`` edge.

ITEM 2 — ``_PRIMITIVE_MARKS`` was triplicated (chart.py / _desugar.py /
marks/_chart_methods_statistical.py).  It is now defined once in ``_layer`` (the
desugar leaf) and imported everywhere; these tests assert every site sees the
SAME frozenset object so the dedup cannot silently regress.
"""

from __future__ import annotations

import warnings

import polars as pl
import pytest

import ferrum as fm
from ferrum.axis import Axis
from ferrum.configure import AxisConfig


@pytest.fixture()
def scatter_df() -> pl.DataFrame:
    return pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})


# ---------------------------------------------------------------------------
# Cross-sibling band-alias parameterisation
#
# Each entry is (label, builder) where builder(**kwargs) constructs the surface
# under test.  configure_axis is invoked on a freshly-built chart so the three
# call shapes are uniform.
# ---------------------------------------------------------------------------


def _build_configure_axis(**kwargs):
    chart = fm.Chart(pl.DataFrame({"x": [1.0], "y": [2.0]})).mark_point().encode(x="x", y="y")
    return chart.configure_axis(**kwargs)


_SURFACES = [
    ("Axis", lambda **kw: Axis(**kw)),
    ("AxisConfig", lambda **kw: AxisConfig(**kw)),
    ("configure_axis", _build_configure_axis),
]


class TestBandAliasCrossSibling:
    """All three surfaces resolve the deprecated band alias identically."""

    @pytest.mark.parametrize("label,build", _SURFACES, ids=[s[0] for s in _SURFACES])
    @pytest.mark.parametrize("bound", ["min", "max"])
    def test_single_alias_warns(self, label, build, bound) -> None:
        """``<bound>_extent=5`` alone warns DeprecationWarning on every surface."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            build(**{f"{bound}_extent": 5.0})
        dep = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert dep, f"{label}: expected DeprecationWarning for {bound}_extent= alias"

    @pytest.mark.parametrize("label,build", _SURFACES, ids=[s[0] for s in _SURFACES])
    @pytest.mark.parametrize("bound", ["min", "max"])
    def test_single_alias_maps_to_canonical_field(self, label, build, bound) -> None:
        """The alias value lands on the canonical ``<bound>_band`` field.

        ``configure_axis`` returns a chart (no public field), so the field-mapping
        assertion is exercised on the two value-class surfaces only.
        """
        if label == "configure_axis":
            pytest.skip("configure_axis stores the resolved value on a nested config")
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            obj = build(**{f"{bound}_extent": 5.0})
        assert getattr(obj, f"{bound}_band") == 5.0

    @pytest.mark.parametrize("label,build", _SURFACES, ids=[s[0] for s in _SURFACES])
    @pytest.mark.parametrize("bound", ["min", "max"])
    def test_double_supply_raises(self, label, build, bound) -> None:
        """Supplying both canonical and alias raises on every surface."""
        with pytest.raises(TypeError, match=f"{bound}_band"):
            build(**{f"{bound}_band": 5.0, f"{bound}_extent": 5.0})

    @pytest.mark.parametrize("label,build", _SURFACES, ids=[s[0] for s in _SURFACES])
    @pytest.mark.parametrize("bound", ["min", "max"])
    def test_explicit_none_alias_plus_canonical_raises(self, label, build, bound) -> None:
        """The previously-divergent edge is now CONSISTENT across all three.

        An explicit ``<bound>_extent=None`` counts as "alias supplied" (via the
        ``_MISSING`` sentinel), so combining it with ``<bound>_band=5`` is a
        double-supply on every surface.  Before the fix, ``Axis``/``AxisConfig``
        silently accepted this while ``configure_axis`` raised.
        """
        with pytest.raises(TypeError, match=f"{bound}_band"):
            build(**{f"{bound}_band": 5.0, f"{bound}_extent": None})

    @pytest.mark.parametrize("bound", ["min", "max"])
    def test_explicit_none_alias_alone_warns_on_value_classes(self, bound) -> None:
        """``<bound>_extent=None`` with no canonical resolves to ``None`` + warns.

        Explicit ``None`` is "supplied" under the sentinel contract, so it emits
        the deprecation warning and resolves the canonical field to ``None`` (the
        alias value).  Asserted on the value classes where the field is readable.
        """
        for build in (Axis, AxisConfig):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                obj = build(**{f"{bound}_extent": None})
            dep = [w for w in caught if issubclass(w.category, DeprecationWarning)]
            assert dep, f"{build.__name__}: explicit {bound}_extent=None must warn"
            assert getattr(obj, f"{bound}_band") is None


# ---------------------------------------------------------------------------
# ITEM 2 — single-sourced _PRIMITIVE_MARKS
# ---------------------------------------------------------------------------


class TestPrimitiveMarksSingleSourced:
    """``_PRIMITIVE_MARKS`` is defined once in ``_layer`` and shared everywhere."""

    def test_all_sites_are_the_same_object(self) -> None:
        """Every consuming module must reference the identical frozenset object."""
        from ferrum import _layer, chart, _desugar
        from ferrum.marks import _chart_methods_statistical

        canonical = _layer._PRIMITIVE_MARKS
        assert chart._PRIMITIVE_MARKS is canonical
        assert _desugar._PRIMITIVE_MARKS is canonical
        assert _chart_methods_statistical._PRIMITIVE_MARKS is canonical

    def test_members_unchanged(self) -> None:
        """The relocated constant keeps its exact 8 primitive-mark members."""
        from ferrum._layer import _PRIMITIVE_MARKS

        assert _PRIMITIVE_MARKS == frozenset(
            {"point", "line", "bar", "area", "rule", "text", "tick", "rect"}
        )
