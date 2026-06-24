"""T2.2 — one palette source of truth + one color-set vocabulary.

Covers ENC-06 (registry dedup), XNAME-02/XSIB-07 (scheme=/cmap= vocabulary),
and ENC-11 (to_hex scale=).

The palette characterization tests pin ``color.palette()`` outputs so the
dedup (consuming the Rust registry instead of hand-mirrored hex tables) stays
byte-stable for the 19 byte-identical palettes and stays render-truth-aligned
for the 7 colorous-backed palettes whose lookup output was deliberately changed
to match what renders (see ``.git/sdd/T2.2-report.md``).
"""

from __future__ import annotations

import pytest

import ferrum as fm
from ferrum._core import list_palettes, palette_kind, palette_sample


# ---------------------------------------------------------------------------
# ENC-06: palette characterization (byte-identical 19 + render-truth 7)
# ---------------------------------------------------------------------------

# All 10 categorical palettes — byte-identical to the pre-dedup hex tables.
_CATEGORICAL_EXPECTED: dict[str, list[str]] = {
    "okabe_ito": [
        "#e69f00",
        "#56b4e9",
        "#009e73",
        "#f0e442",
        "#0072b2",
        "#d55e00",
        "#cc79a7",
        "#000000",
    ],
    "tableau10": [
        "#4c78a8",
        "#f58e18",
        "#e45756",
        "#72b7b2",
        "#54a24b",
        "#eeca3b",
        "#b279a2",
        "#ff9da6",
        "#9d755d",
        "#bab0ac",
    ],
    "set1": [
        "#e41a1c",
        "#377eb8",
        "#4daf4a",
        "#984ea3",
        "#ff7f00",
        "#ffff33",
        "#a65628",
        "#f781bf",
        "#999999",
    ],
    "set2": [
        "#66c2a5",
        "#fc8d62",
        "#8da0cb",
        "#e78ac3",
        "#a6d854",
        "#ffd92f",
        "#e5c494",
        "#b3b3b3",
    ],
    "paired": [
        "#a6cee3",
        "#1f78b4",
        "#b2df8a",
        "#33a02c",
        "#fb9a99",
        "#e31a1c",
        "#fdbf6f",
        "#ff7f00",
        "#cab2d6",
        "#6a3d9a",
        "#ffff99",
        "#b15928",
    ],
    "pastel": [
        "#fbb4ae",
        "#b3cde3",
        "#ccebc5",
        "#decbe4",
        "#fed9a6",
        "#ffffcc",
        "#e5d8bd",
        "#fddaec",
        "#f2f2f2",
    ],
    "dark2": [
        "#1b9e77",
        "#d95f02",
        "#7570b3",
        "#e7298a",
        "#66a61e",
        "#e6ab02",
        "#a6761d",
        "#666666",
    ],
    "paper_ink": [
        "#2563eb",
        "#dc2626",
        "#d4a017",
        "#0f766e",
        "#7c3aed",
        "#ea580c",
        "#4b5563",
        "#db2777",
    ],
    "slate_citrus": [
        "#60a5fa",
        "#a78bfa",
        "#a3e635",
        "#f59e0b",
        "#34d399",
        "#f472b6",
        "#f87171",
        "#22d3ee",
    ],
    "arctic_signal": [
        "#0284c7",
        "#7c3aed",
        "#ea580c",
        "#16a34a",
        "#dc2626",
        "#0891b2",
        "#ca8a04",
        "#db2777",
    ],
}

# 9 custom (non-colorous) continuous palettes at n=7 — byte-identical to the
# pre-dedup hand-mirrored tables (their control points land exactly on i/6).
_CUSTOM_CONTINUOUS_EXPECTED: dict[str, list[str]] = {
    "cool_blue": ["#eff6ff", "#dbeafe", "#93c5fd", "#60a5fa", "#2563eb", "#1d4ed8", "#1e3a8a"],
    "warm_ochre": ["#fff7e6", "#fdecc8", "#f8d88a", "#d4a017", "#b45309", "#92400e", "#78350f"],
    "night_blue": ["#1e293b", "#1d4ed8", "#2563eb", "#60a5fa", "#93c5fd", "#bfdbfe", "#e0f2fe"],
    "electric_lime": ["#365314", "#4d7c0f", "#65a30d", "#84cc16", "#a3e635", "#bef264", "#d9f99d"],
    "signal_blue": ["#f0f9ff", "#e0f2fe", "#7dd3fc", "#38bdf8", "#0284c7", "#0369a1", "#0c4a6e"],
    "ember_orange": ["#fff7ed", "#fed7aa", "#fdba74", "#ea580c", "#c2410c", "#9a3412", "#7c2d12"],
    "blue_to_red": ["#1e3a8a", "#60a5fa", "#dbeafe", "#faf7f2", "#fde68a", "#dc2626", "#7f1d1d"],
    "cyan_to_amber": ["#155e75", "#0891b2", "#67e8f9", "#111827", "#fde68a", "#f59e0b", "#b45309"],
    "blue_to_violet": ["#0c4a6e", "#38bdf8", "#bae6fd", "#f8fafc", "#e9d5ff", "#a78bfa", "#6d28d9"],
}

# The 7 colorous-backed palettes at n=7 — these now return RENDER-TRUTH (the
# intended ENC-06 change; the old hand-picked approximations are gone).
_RENDER_TRUTH_EXPECTED: dict[str, list[str]] = {
    "viridis": ["#440154", "#443983", "#30688e", "#20908c", "#35b778", "#8ed643", "#fde725"],
    "plasma": ["#0d0887", "#5d01a5", "#9b179e", "#cb4678", "#ec7853", "#fdb32f", "#f0f921"],
    "magma": ["#000004", "#2c115f", "#711f81", "#b53679", "#f1605d", "#feaf77", "#fcfdbf"],
    "inferno": ["#000004", "#33095e", "#781c6d", "#bb3654", "#ed6825", "#fbb419", "#fcffa4"],
    "cividis": ["#002051", "#1f3d6d", "#565c6d", "#7f7b74", "#a49c77", "#d5c163", "#fde945"],
    "blues": ["#f7fbff", "#d5e5f4", "#aacfe5", "#6caed5", "#3786c0", "#115ba3", "#08306b"],
    "rdbu": ["#67001f", "#c94842", "#f5b699", "#f1efed", "#a6cee3", "#3984bb", "#042f60"],
}


@pytest.mark.parametrize("name", sorted(_CATEGORICAL_EXPECTED))
def test_categorical_palette_byte_identical(name):
    assert fm.color.palette(name) == _CATEGORICAL_EXPECTED[name]


@pytest.mark.parametrize("name", sorted(_CUSTOM_CONTINUOUS_EXPECTED))
def test_custom_continuous_palette_byte_identical(name):
    assert fm.color.palette(name, n=7) == _CUSTOM_CONTINUOUS_EXPECTED[name]


@pytest.mark.parametrize("name", sorted(_RENDER_TRUTH_EXPECTED))
def test_colorous_palette_is_render_truth(name):
    assert fm.color.palette(name, n=7) == _RENDER_TRUTH_EXPECTED[name]


@pytest.mark.parametrize("name", ["viridis", "rdbu", "magma", "blues"])
def test_palette_midpoint_matches_render_path(name):
    """The lookup now agrees with the render path (the point of ENC-06): the
    n=7 midpoint equals the renderer's sample(0.5)."""
    p7 = fm.color.palette(name, n=7)
    assert p7[3] == palette_sample(name, 0.5)


def test_palette_covers_full_registry():
    """Every name the Rust registry knows is resolvable via palette()."""
    for name in list_palettes():
        colors = fm.color.palette(name)
        assert colors and all(c.startswith("#") for c in colors)


def test_sequential_accepts_new_registry_names():
    """reds/greens/oranges/purples are sequential schemes Python now accepts
    (previously a strict subset; now the single source of truth)."""
    for name in ("reds", "greens", "oranges", "purples"):
        assert palette_kind(name) == "sequential"
        assert len(fm.color.sequential(name, n=5)) == 5


def test_unknown_palette_raises_canonical_message():
    with pytest.raises(ValueError, match="Unknown palette"):
        fm.color.palette("not_a_palette")


# ---------------------------------------------------------------------------
# ENC-11: to_hex scale=
# ---------------------------------------------------------------------------


class TestToHexScale:
    def test_explicit_byte(self):
        assert fm.color.to_hex((128, 128, 128), scale="byte") == "#808080"

    def test_explicit_unit(self):
        assert fm.color.to_hex((0.5, 0.5, 0.5), scale="unit") == "#808080"

    def test_explicit_byte_overrides_low_values(self):
        # Without scale= this would be ambiguous; scale="byte" forces byte.
        assert fm.color.to_hex((1, 0, 0), scale="byte") == "#010000"

    def test_explicit_unit_on_ints(self):
        assert fm.color.to_hex((1, 1, 1), scale="unit") == "#ffffff"

    def test_default_heuristic_unit_floats(self):
        assert fm.color.to_hex((1.0, 0.0, 0.0)) == "#ff0000"

    def test_default_heuristic_byte_ints(self):
        assert fm.color.to_hex((255, 0, 0)) == "#ff0000"

    def test_integer_valued_float_treated_as_byte(self):
        # 255.0 is a float but clearly a byte value; range-based heuristic
        # treats it as byte, agreeing with the int 255.
        assert fm.color.to_hex((255.0, 0.0, 0.0)) == fm.color.to_hex((255, 0, 0))

    def test_int_and_float_one_agree(self):
        # The hidden-mode-switch fix: (1,0,0) and (1.0,0.0,0.0) now agree.
        assert fm.color.to_hex((1, 0, 0)) == fm.color.to_hex((1.0, 0.0, 0.0))

    def test_invalid_scale_raises(self):
        with pytest.raises(ValueError, match="scale must be"):
            fm.color.to_hex((1, 2, 3), scale="bogus")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# XNAME-02/XSIB-07: scheme= declaration-time validation
# ---------------------------------------------------------------------------


class TestSchemeValidation:
    def test_valid_scheme_accepted(self):
        c = fm.Color("v", scheme="viridis")
        assert c.option("scheme") == "viridis"

    def test_valid_categorical_scheme_accepted(self):
        assert fm.Color("v", scheme="set1").option("scheme") == "set1"

    @pytest.mark.parametrize("channel", [fm.Color, fm.Fill, fm.Stroke])
    def test_bogus_scheme_raises_at_construction(self, channel):
        with pytest.raises(ValueError, match="Unknown palette"):
            channel("v", scheme="not-a-real-scheme")

    def test_bogus_scheme_error_lists_available(self):
        with pytest.raises(ValueError, match="Available:.*viridis"):
            fm.Color("v", scheme="bogus")

    def test_scheme_none_is_not_validated(self):
        # Passing scheme=None (e.g. via clustermap default) is a no-op.
        assert fm.Color("v", scheme=None).option("scheme") is None


# ---------------------------------------------------------------------------
# XNAME-02/XSIB-07: scheme=/cmap= equivalence on marks
# ---------------------------------------------------------------------------


@pytest.fixture
def xy_df():
    import polars as pl

    return pl.DataFrame({"x": [1.0, 2.0, 3.0, 2.0, 1.5], "y": [4.0, 5.0, 4.5, 3.0, 4.0]})


class TestSchemeCmapEquivalence:
    def test_raster_scheme_equals_cmap(self, xy_df):
        a = fm.Chart(xy_df).mark_raster(scheme="plasma").encode(x="x", y="y")
        b = fm.Chart(xy_df).mark_raster(cmap="plasma").encode(x="x", y="y")
        assert a.to_spec().to_json() == b.to_spec().to_json()

    def test_contour_scheme_equals_cmap(self, xy_df):
        a = fm.Chart(xy_df).mark_contour(fill=True, scheme="viridis").encode(x="x", y="y")
        b = fm.Chart(xy_df).mark_contour(fill=True, cmap="viridis").encode(x="x", y="y")
        assert a.to_spec().to_json() == b.to_spec().to_json()

    def test_hex_scheme_equals_cmap(self, xy_df):
        a = fm.Chart(xy_df).mark_hex(scheme="magma").encode(x="x", y="y")
        b = fm.Chart(xy_df).mark_hex(cmap="magma").encode(x="x", y="y")
        assert a.to_spec().to_json() == b.to_spec().to_json()

    def test_conflicting_scheme_and_cmap_raises(self, xy_df):
        with pytest.raises(ValueError, match="not both"):
            fm.Chart(xy_df).mark_raster(scheme="plasma", cmap="viridis").encode(x="x", y="y")

    def test_same_scheme_and_cmap_ok(self, xy_df):
        # Identical values are allowed (no conflict).
        chart = fm.Chart(xy_df).mark_raster(scheme="plasma", cmap="plasma").encode(x="x", y="y")
        assert chart.to_spec() is not None

    def test_heatmap_scheme_equals_cmap(self):
        import polars as pl

        df = pl.DataFrame({"row": ["a", "b"], "c1": [1.0, 2.0], "c2": [3.0, 4.0]})
        a = fm.heatmap(df, scheme="rdbu")
        b = fm.heatmap(df, cmap="rdbu")
        assert a.to_spec().to_json() == b.to_spec().to_json()

    def test_heatmap_conflicting_scheme_and_cmap_raises(self):
        import polars as pl

        df = pl.DataFrame({"row": ["a", "b"], "c1": [1.0, 2.0], "c2": [3.0, 4.0]})
        with pytest.raises(ValueError, match="not both"):
            fm.heatmap(df, scheme="rdbu", cmap="viridis")

    def test_heatmap_same_scheme_and_cmap_ok(self):
        import polars as pl

        df = pl.DataFrame({"row": ["a", "b"], "c1": [1.0, 2.0], "c2": [3.0, 4.0]})
        # Identical values are allowed (no conflict).
        chart = fm.heatmap(df, scheme="rdbu", cmap="rdbu")
        assert chart.to_spec() is not None


# ---------------------------------------------------------------------------
# Issue #32: RenderConfig raster_scheme/raster_cmap vocabulary
# ---------------------------------------------------------------------------


class TestRenderConfigRasterScheme:
    """RenderConfig stores ``raster_scheme`` canonically; ``raster_cmap`` is a
    construct-time back-compat alias resolved via the shared helper."""

    def test_render_config_raster_scheme_equals_cmap(self):
        assert fm.RenderConfig(raster_scheme="plasma").raster_scheme == "plasma"
        assert fm.RenderConfig(raster_cmap="plasma").raster_scheme == "plasma"

    def test_render_config_default_raster_scheme_is_viridis(self):
        assert fm.RenderConfig().raster_scheme == "viridis"

    def test_render_config_conflicting_raster_scheme_and_cmap_raises(self):
        with pytest.raises(ValueError, match="not both"):
            fm.RenderConfig(raster_scheme="a", raster_cmap="b")

    def test_render_config_raster_cmap_reads_back_resolved(self):
        # The back-compat alias reads back as the resolved canonical scheme (no silent None).
        assert fm.RenderConfig(raster_cmap="plasma").raster_cmap == "plasma"
        assert fm.RenderConfig(raster_scheme="magma").raster_cmap == "magma"
        assert fm.RenderConfig().raster_cmap == "viridis"
