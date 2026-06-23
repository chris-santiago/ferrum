"""Regression tests for T3.3b — axis layout-band kwarg rename (D-EXTENT-1, axis half).

Covers:
- ``fm.Axis``: canonical ``min_band``/``max_band`` accepted; old ``min_extent``/
  ``max_extent`` accepted as deprecated aliases.
- ``configure_axis`` mixin method: same contract.
- ``AxisConfig`` dataclass: same contract.

Each surface must:
1. Accept the new canonical kwargs and produce the correct wire dict / SVG.
2. Accept the old deprecated aliases (with a DeprecationWarning) and produce a
   spec/SVG identical to the canonical call.
3. Raise TypeError when both the canonical kwarg and its alias are supplied.
4. Emit wire keys ``min_band``/``max_band`` (NOT ``min_extent``/``max_extent``),
   confirming Rust can read them.
5. Prove an end-to-end render with ``min_band=80`` actually reserves the band
   (SVG differs from the no-constraint baseline), and that the deprecated alias
   produces the same render.

The canonical-kwarg tests are written so they FAIL against the pre-rename
signature (``min_extent=``/``max_extent=`` only), confirming the rename is in
place.
"""

from __future__ import annotations

import warnings

import polars as pl
import pytest

import ferrum as fm
from ferrum.axis import Axis
from ferrum.configure import AxisConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def scatter_df() -> pl.DataFrame:
    return pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})


# ---------------------------------------------------------------------------
# fm.Axis  (per-channel axis value class)
# ---------------------------------------------------------------------------


class TestAxisBandKwarg:
    """fm.Axis canonical and alias coverage for min_band/max_band."""

    def test_canonical_min_band_accepted(self) -> None:
        """Axis(min_band=...) must be accepted without error."""
        ax = Axis(min_band=80.0)
        assert ax.min_band == 80.0

    def test_canonical_max_band_accepted(self) -> None:
        """Axis(max_band=...) must be accepted without error."""
        ax = Axis(max_band=200.0)
        assert ax.max_band == 200.0

    def test_wire_dict_uses_min_band_key(self) -> None:
        """to_dict() must emit 'min_band', NOT 'min_extent'."""
        d = Axis(min_band=80.0).to_dict()
        assert "min_band" in d, "wire key must be 'min_band'"
        assert "min_extent" not in d, "old wire key 'min_extent' must NOT appear"
        assert d["min_band"] == 80.0

    def test_wire_dict_uses_max_band_key(self) -> None:
        """to_dict() must emit 'max_band', NOT 'max_extent'."""
        d = Axis(max_band=200.0).to_dict()
        assert "max_band" in d, "wire key must be 'max_band'"
        assert "max_extent" not in d, "old wire key 'max_extent' must NOT appear"
        assert d["max_band"] == 200.0

    def test_alias_min_extent_accepted_with_warning(self) -> None:
        """Axis(min_extent=...) is accepted but emits DeprecationWarning."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ax = Axis(min_extent=80.0)
        dep = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert dep, "Expected DeprecationWarning for min_extent= alias"
        assert ax.min_band == 80.0

    def test_alias_max_extent_accepted_with_warning(self) -> None:
        """Axis(max_extent=...) is accepted but emits DeprecationWarning."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ax = Axis(max_extent=200.0)
        dep = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert dep, "Expected DeprecationWarning for max_extent= alias"
        assert ax.max_band == 200.0

    def test_alias_min_extent_wire_dict_identical(self) -> None:
        """Axis(min_extent=80) to_dict() must equal Axis(min_band=80).to_dict()."""
        canonical = Axis(min_band=80.0).to_dict()
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            alias = Axis(min_extent=80.0).to_dict()
        assert alias == canonical

    def test_alias_max_extent_wire_dict_identical(self) -> None:
        """Axis(max_extent=200) to_dict() must equal Axis(max_band=200).to_dict()."""
        canonical = Axis(max_band=200.0).to_dict()
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            alias = Axis(max_extent=200.0).to_dict()
        assert alias == canonical

    def test_both_min_band_and_min_extent_raises(self) -> None:
        """Axis(min_band=80, min_extent=80) must raise TypeError."""
        with pytest.raises(TypeError, match="min_band"):
            Axis(min_band=80.0, min_extent=80.0)

    def test_both_max_band_and_max_extent_raises(self) -> None:
        """Axis(max_band=200, max_extent=200) must raise TypeError."""
        with pytest.raises(TypeError, match="max_band"):
            Axis(max_band=200.0, max_extent=200.0)

    def test_neither_kwarg_gives_none(self) -> None:
        """When neither canonical nor alias is supplied, fields default to None."""
        ax = Axis()
        assert ax.min_band is None
        assert ax.max_band is None

    def test_frozen_immutable(self) -> None:
        """Axis must still be frozen after the init=False change."""
        ax = Axis(min_band=80.0)
        with pytest.raises((AttributeError, TypeError)):
            ax.min_band = 99.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# AxisConfig  (chart-level configure dataclass)
# ---------------------------------------------------------------------------


class TestAxisConfigBandKwarg:
    """AxisConfig canonical and alias coverage for min_band/max_band."""

    def test_canonical_min_band_accepted(self) -> None:
        """AxisConfig(min_band=...) must be accepted without error."""
        cfg = AxisConfig(min_band=80.0)
        assert cfg.min_band == 80.0

    def test_canonical_max_band_accepted(self) -> None:
        """AxisConfig(max_band=...) must be accepted without error."""
        cfg = AxisConfig(max_band=200.0)
        assert cfg.max_band == 200.0

    def test_wire_dict_uses_min_band_key(self) -> None:
        """to_dict() must emit 'min_band', NOT 'min_extent'."""
        d = AxisConfig(min_band=80.0).to_dict()
        assert "min_band" in d
        assert "min_extent" not in d
        assert d["min_band"] == 80.0

    def test_wire_dict_uses_max_band_key(self) -> None:
        """to_dict() must emit 'max_band', NOT 'max_extent'."""
        d = AxisConfig(max_band=200.0).to_dict()
        assert "max_band" in d
        assert "max_extent" not in d
        assert d["max_band"] == 200.0

    def test_alias_min_extent_accepted_with_warning(self) -> None:
        """AxisConfig(min_extent=...) is accepted but emits DeprecationWarning."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cfg = AxisConfig(min_extent=80.0)
        dep = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert dep, "Expected DeprecationWarning for min_extent= alias on AxisConfig"
        assert cfg.min_band == 80.0

    def test_alias_max_extent_accepted_with_warning(self) -> None:
        """AxisConfig(max_extent=...) is accepted but emits DeprecationWarning."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cfg = AxisConfig(max_extent=200.0)
        dep = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert dep, "Expected DeprecationWarning for max_extent= alias on AxisConfig"
        assert cfg.max_band == 200.0

    def test_alias_min_extent_wire_dict_identical(self) -> None:
        """AxisConfig(min_extent=80).to_dict() == AxisConfig(min_band=80).to_dict()."""
        canonical = AxisConfig(min_band=80.0).to_dict()
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            alias = AxisConfig(min_extent=80.0).to_dict()
        assert alias == canonical

    def test_alias_max_extent_wire_dict_identical(self) -> None:
        """AxisConfig(max_extent=200).to_dict() == AxisConfig(max_band=200).to_dict()."""
        canonical = AxisConfig(max_band=200.0).to_dict()
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            alias = AxisConfig(max_extent=200.0).to_dict()
        assert alias == canonical

    def test_both_min_band_and_min_extent_raises(self) -> None:
        """AxisConfig(min_band=80, min_extent=80) must raise TypeError."""
        with pytest.raises(TypeError, match="min_band"):
            AxisConfig(min_band=80.0, min_extent=80.0)

    def test_both_max_band_and_max_extent_raises(self) -> None:
        """AxisConfig(max_band=200, max_extent=200) must raise TypeError."""
        with pytest.raises(TypeError, match="max_band"):
            AxisConfig(max_band=200.0, max_extent=200.0)

    def test_fields_default_to_none(self) -> None:
        """When neither canonical nor alias is supplied, fields default to None."""
        cfg = AxisConfig()
        assert cfg.min_band is None
        assert cfg.max_band is None

    def test_existing_validation_preserved(self) -> None:
        """Existing __post_init__ validation (label_format exclusivity) must still work."""
        with pytest.raises(ValueError, match="mutually exclusive"):
            AxisConfig(label_format="currency", label_format_raw=".2f")


# ---------------------------------------------------------------------------
# configure_axis mixin method
# ---------------------------------------------------------------------------


class TestConfigureAxisBandKwarg:
    """configure_axis() canonical and alias coverage for min_band/max_band."""

    def test_canonical_min_band_accepted(self, scatter_df: pl.DataFrame) -> None:
        """configure_axis(min_band=...) must be accepted without error."""
        chart = fm.Chart(scatter_df).mark_point().encode(x="x", y="y")
        # Must not raise.
        result = chart.configure_axis(min_band=80.0)
        assert result is not chart  # fluent returns new instance

    def test_canonical_max_band_accepted(self, scatter_df: pl.DataFrame) -> None:
        """configure_axis(max_band=...) must be accepted without error."""
        chart = fm.Chart(scatter_df).mark_point().encode(x="x", y="y")
        result = chart.configure_axis(max_band=200.0)
        assert result is not chart

    def test_alias_min_extent_accepted_with_warning(self, scatter_df: pl.DataFrame) -> None:
        """configure_axis(min_extent=...) is accepted but emits DeprecationWarning."""
        chart = fm.Chart(scatter_df).mark_point().encode(x="x", y="y")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            chart.configure_axis(min_extent=80.0)
        dep = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert dep, "Expected DeprecationWarning for configure_axis(min_extent=...)"

    def test_alias_max_extent_accepted_with_warning(self, scatter_df: pl.DataFrame) -> None:
        """configure_axis(max_extent=...) is accepted but emits DeprecationWarning."""
        chart = fm.Chart(scatter_df).mark_point().encode(x="x", y="y")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            chart.configure_axis(max_extent=200.0)
        dep = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert dep, "Expected DeprecationWarning for configure_axis(max_extent=...)"

    def test_both_min_band_and_min_extent_raises(self, scatter_df: pl.DataFrame) -> None:
        """configure_axis(min_band=80, min_extent=80) must raise TypeError."""
        chart = fm.Chart(scatter_df).mark_point().encode(x="x", y="y")
        with pytest.raises(TypeError, match="min_band"):
            chart.configure_axis(min_band=80.0, min_extent=80.0)

    def test_both_max_band_and_max_extent_raises(self, scatter_df: pl.DataFrame) -> None:
        """configure_axis(max_band=200, max_extent=200) must raise TypeError."""
        chart = fm.Chart(scatter_df).mark_point().encode(x="x", y="y")
        with pytest.raises(TypeError, match="max_band"):
            chart.configure_axis(max_band=200.0, max_extent=200.0)


# ---------------------------------------------------------------------------
# End-to-end render: wire key reaches Rust; alias produces identical render
# ---------------------------------------------------------------------------


class TestBandKwargEndToEnd:
    """Prove min_band/max_band wire reaches Rust and alias renders identically."""

    def test_min_band_changes_svg(self, scatter_df: pl.DataFrame) -> None:
        """Axis(min_band=80) must produce a different SVG from the default.

        This proves the wire key 'min_band' is read by Rust (the layout band
        is reserved) rather than silently ignored.
        """
        base = fm.Chart(scatter_df).mark_point().encode(x="x", y="y").to_svg()
        with_band = (
            fm.Chart(scatter_df)
            .mark_point()
            .encode(x="x", y=fm.Y("y", axis=fm.Axis(min_band=80.0)))
            .to_svg()
        )
        assert base != with_band, (
            "min_band=80 must change the SVG layout; if they are equal the wire "
            "key 'min_band' is not being read by Rust"
        )

    def test_min_extent_alias_same_svg_as_canonical(self, scatter_df: pl.DataFrame) -> None:
        """Axis(min_extent=80) must produce SVG identical to Axis(min_band=80).

        This confirms the alias maps to the same wire value and the same Rust
        layout computation.
        """
        canonical_svg = (
            fm.Chart(scatter_df)
            .mark_point()
            .encode(x="x", y=fm.Y("y", axis=fm.Axis(min_band=80.0)))
            .to_svg()
        )
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            alias_svg = (
                fm.Chart(scatter_df)
                .mark_point()
                .encode(x="x", y=fm.Y("y", axis=fm.Axis(min_extent=80.0)))
                .to_svg()
            )
        assert alias_svg == canonical_svg, (
            "Axis(min_extent=80) must produce the same SVG as Axis(min_band=80)"
        )

    def test_configure_axis_min_band_changes_svg(self, scatter_df: pl.DataFrame) -> None:
        """configure_axis(min_band=120) must produce a different SVG from the default."""
        base = fm.Chart(scatter_df).mark_point().encode(x="x", y="y").to_svg()
        with_band = (
            fm.Chart(scatter_df)
            .mark_point()
            .encode(x="x", y="y")
            .configure_axis(min_band=120)
            .to_svg()
        )
        assert base != with_band, "configure_axis(min_band=120) must change SVG layout"

    def test_configure_axis_min_extent_alias_same_svg(self, scatter_df: pl.DataFrame) -> None:
        """configure_axis(min_extent=120) must produce SVG identical to min_band=120."""
        canonical_svg = (
            fm.Chart(scatter_df)
            .mark_point()
            .encode(x="x", y="y")
            .configure_axis(min_band=120)
            .to_svg()
        )
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            alias_svg = (
                fm.Chart(scatter_df)
                .mark_point()
                .encode(x="x", y="y")
                .configure_axis(min_extent=120)
                .to_svg()
            )
        assert alias_svg == canonical_svg, (
            "configure_axis(min_extent=120) must produce the same SVG as min_band=120"
        )


# ---------------------------------------------------------------------------
# Guard: data-domain extent must NOT have been touched
# ---------------------------------------------------------------------------


def test_data_domain_extent_untouched() -> None:
    """The data-domain 'extent' parameter on relevant functions must be unchanged.

    D-EXTENT-1 only renames the *layout-band* sense.  Data-domain extent params
    (e.g. shared_extent on violin/boxen) must be untouched.
    """
    import inspect

    from ferrum.marks.heavy_stat import desugar_violin

    sig = inspect.signature(desugar_violin)
    assert "shared_extent" in sig.parameters, (
        "desugar_violin.shared_extent must NOT have been renamed — "
        "it is the data-domain extent sense preserved by D-EXTENT-1"
    )
