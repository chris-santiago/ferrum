"""Regression tests for defect D1 — continuous color scales are inert.

The bug: binding a continuous color channel through ``SequentialScale`` or
``DivergingScale`` with a named scheme (viridis, magma, blues, rdbu, …)
renders every mark in the default sequential/diverging fallback colors instead
of the requested scheme.

Root cause (Rust): ``scale_common_scheme()`` in
``crates/ferrum-core/src/render/scale_resolve/color.rs`` matches only
``Linear | Log | Time | Symlog | Pow | Sqrt | Utc`` variants.  The
``Sequential`` and ``Diverging`` variants are omitted, so their ``scheme``
field is silently discarded and the theme's default scheme is used instead.

These tests assert on OBSERVABLE rendered output: hex fill colors extracted
from the SVG, compared against known palette endpoint values.  They will FAIL
until the Rust fix lands.

Palette reference values are sourced from the ``colorous`` crate and from the
``ferrum-core/src/render/color/continuous.rs`` unit tests in the Rust source.
"""

from __future__ import annotations

import re

import polars as pl
import pytest

import ferrum as fm
from ferrum.encoding import Color


# ---------------------------------------------------------------------------
# SVG color helpers
# ---------------------------------------------------------------------------

# Structural fill colors that appear in every chart regardless of the color
# encoding: background (#faf7f2), label gray (#6b7280), dark text (#1f2937).
# These must be excluded when looking for mark-specific palette colors.
_STRUCTURAL_FILLS: frozenset[str] = frozenset({"faf7f2", "6b7280", "1f2937", "ffffff", "000000"})


def _mark_fills(svg: str) -> set[str]:
    """Extract lowercase hex fill colors from the SVG, excluding structural chrome.

    Filters out the background, axis, and label colors that appear in every
    chart so that the returned set contains only mark-level fill colors.
    """
    all_fills = {c.lower() for c in re.findall(r'fill="#([0-9a-fA-F]{6})"', svg)}
    return all_fills - _STRUCTURAL_FILLS


def _rgb(hex6: str) -> tuple[int, int, int]:
    """Parse a 6-char hex string (no leading #) into an (R, G, B) tuple."""
    h = hex6.lower()
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def linear_df() -> pl.DataFrame:
    """Five rows spanning 0–4 so the renderer samples across the full ramp."""
    return pl.DataFrame(
        {
            "x": [0.0, 1.0, 2.0, 3.0, 4.0],
            "y": [0.0, 1.0, 2.0, 3.0, 4.0],
            "val": [0.0, 1.0, 2.0, 3.0, 4.0],
        }
    )


@pytest.fixture
def diverging_df() -> pl.DataFrame:
    """Seven rows spanning -2 to 2 so domain clamping can be exercised.

    The declared domain for DivergingScale tests is [-1, 0, 1], meaning
    val=-2 and val=2 exceed the domain endpoints and should clamp to the same
    color as val=-1 and val=1 respectively once the fix is applied.
    """
    return pl.DataFrame(
        {
            "x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "y": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "val": [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0],
        }
    )


# ---------------------------------------------------------------------------
# D1-A — SequentialScale: scheme must change rendered fill colors
# ---------------------------------------------------------------------------


def test_sequential_viridis_differs_from_default(linear_df: pl.DataFrame) -> None:
    """SequentialScale(scheme='viridis') must produce viridis fills, not the
    default cool_blue fallback.

    When the bug is active, all SequentialScale schemes produce identical
    cool_blue colors regardless of the requested scheme.  This test asserts
    that viridis and magma produce distinguishable fill sets (they share no
    mark-level palette colors with each other under correct behavior).

    Bug indicator: the two sets are equal (both cool_blue).
    """
    svg_viridis = (
        fm.Chart(linear_df)
        .mark_point()
        .encode(x="x:Q", y="y:Q", color=Color("val:Q", scale=fm.SequentialScale(scheme="viridis")))
        .to_svg()
    )
    svg_magma = (
        fm.Chart(linear_df)
        .mark_point()
        .encode(x="x:Q", y="y:Q", color=Color("val:Q", scale=fm.SequentialScale(scheme="magma")))
        .to_svg()
    )

    fills_viridis = _mark_fills(svg_viridis)
    fills_magma = _mark_fills(svg_magma)

    assert fills_viridis != fills_magma, (
        "SequentialScale(scheme='viridis') and SequentialScale(scheme='magma') "
        "produced identical mark fill colors — both are falling through to the "
        "default scheme instead of honoring the requested scheme. "
        f"viridis fills: {fills_viridis}, magma fills: {fills_magma}"
    )


def test_sequential_viridis_has_characteristic_purple_at_min(
    linear_df: pl.DataFrame,
) -> None:
    """SequentialScale(scheme='viridis') must render the minimum-value mark in
    a dark purple hue (#440154 at t=0), not in a blue hue.

    Viridis starts at dark purple (R≈68, G≈1, B≈84) and ends at yellow-green
    (R≈253, G≈231, B≈37).  The cool_blue fallback starts at nearly-white light
    blue (#eff6ff) and ends at dark navy (#1e3a8a).

    The check is tolerant: we require the minimum-value mark to be a purple hue
    (R and B both > G, i.e. the magenta/purple channel dominates) rather than
    the blue-dominant cool_blue start color.

    Hex reference: colorous VIRIDIS evaluated at t=0.0 → approximately #440154
    (R=68, G=1, B=84).
    """
    svg = (
        fm.Chart(linear_df)
        .mark_point()
        .encode(x="x:Q", y="y:Q", color=Color("val:Q", scale=fm.SequentialScale(scheme="viridis")))
        .to_svg()
    )
    fills = _mark_fills(svg)

    # After removing structural fills, the mark-fill set should contain at
    # least one color that is distinctly purple (R > G, B > G, B and R
    # comparable) rather than blue-dominant (B >> R).
    # viridis(0) ~ #440154: R=68, G=1, B=84  → R < 100, G < 20, B < 100
    # cool_blue(0) ~ #eff6ff: R=239, G=246, B=255 → very light, B > R
    has_purple_endpoint = any(
        r < 100 and g < 20 and b < 100 for hex6 in fills for r, g, b in [_rgb(hex6)]
    )
    assert has_purple_endpoint, (
        "Expected a dark purple fill near #440154 (viridis t=0) in the SVG "
        "mark fills, but found none.  The scheme is being ignored and the "
        f"default cool_blue ramp is rendering instead.  Mark fills: {fills}"
    )


def test_sequential_viridis_has_characteristic_yellow_at_max(
    linear_df: pl.DataFrame,
) -> None:
    """SequentialScale(scheme='viridis') must render the maximum-value mark in
    a yellow-green hue (#fde725 at t=1), not in dark navy.

    Viridis ends at bright yellow-green (R≈253, G≈231, B≈37).  The tolerant
    check requires at least one fill to be high-R, high-G, low-B (yellowish),
    which no cool_blue color satisfies.

    Hex reference: colorous VIRIDIS evaluated at t=1.0 → approximately #fde725
    (R=253, G=231, B=37).
    """
    svg = (
        fm.Chart(linear_df)
        .mark_point()
        .encode(x="x:Q", y="y:Q", color=Color("val:Q", scale=fm.SequentialScale(scheme="viridis")))
        .to_svg()
    )
    fills = _mark_fills(svg)

    # viridis(1) ~ #fde725: R=253, G=231, B=37 → high R, high G, low B
    # cool_blue(1) ~ #1e3a8a: R=30, G=58, B=138 → low R, low G, mid-high B (navy)
    has_yellow_endpoint = any(
        r > 200 and g > 180 and b < 80 for hex6 in fills for r, g, b in [_rgb(hex6)]
    )
    assert has_yellow_endpoint, (
        "Expected a bright yellow-green fill near #fde725 (viridis t=1) in "
        "the SVG mark fills, but found none.  The scheme is being ignored and "
        f"the default cool_blue ramp is rendering instead.  Mark fills: {fills}"
    )


# ---------------------------------------------------------------------------
# D1-B — SequentialScale: scheme parametrization across multiple schemes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "scheme, expected_description, min_check, max_check",
    [
        # viridis: dark purple → yellow-green
        # viridis(0) #440154 R=68, G=1, B=84; viridis(1) #fde725 R=253, G=231, B=37
        (
            "viridis",
            "dark purple (min) → yellow-green (max)",
            lambda r, g, b: r < 100 and g < 20 and b < 100,
            lambda r, g, b: r > 200 and g > 180 and b < 80,
        ),
        # magma: near-black #000004 → pale yellow #fcfdbf
        # magma(0) R≈0, G≈0, B≈4; magma(1) R≈252, G≈253, B≈191
        (
            "magma",
            "near-black (min) → pale yellow (max)",
            lambda r, g, b: r < 20 and g < 20 and b < 20,
            lambda r, g, b: r > 220 and g > 220 and b > 150,
        ),
        # blues: near-white #f7fbff → deep blue #08306b
        # blues(0) R≈247, G≈251, B≈255; blues(1) R≈8, G≈48, B≈107
        (
            "blues",
            "near-white blue-tinted (min) → deep navy (max)",
            lambda r, g, b: r > 200 and b > r,
            lambda r, g, b: r < 30 and g < 70 and b > 80,
        ),
        # warm_ochre (custom scheme): near-white #fff7e6 → dark brown #78350f
        # warm_ochre(0) R=255, G=247, B=230; warm_ochre(1) R=120, G=53, B=15
        (
            "warm_ochre",
            "near-white warm (min) → dark brown (max)",
            lambda r, g, b: r > 220 and g > 200 and b > 180,
            lambda r, g, b: r > 80 and g < 70 and b < 30,
        ),
    ],
    ids=["viridis", "magma", "blues", "warm_ochre"],
)
def test_sequential_scheme_fills_match_palette_endpoints(
    linear_df: pl.DataFrame,
    scheme: str,
    expected_description: str,
    min_check,
    max_check,
) -> None:
    """SequentialScale(scheme=X) renders mark fills that match the X palette's
    characteristic endpoint colors, not the default cool_blue fallback.

    Each parametrized case checks both the low-end and high-end color
    characteristics of the named scheme.  Under the bug, all four produce
    identical cool_blue fills.
    """
    svg = (
        fm.Chart(linear_df)
        .mark_point()
        .encode(
            x="x:Q",
            y="y:Q",
            color=Color("val:Q", scale=fm.SequentialScale(scheme=scheme)),
        )
        .to_svg()
    )
    fills = _mark_fills(svg)

    has_min_color = any(min_check(*_rgb(h)) for h in fills)
    has_max_color = any(max_check(*_rgb(h)) for h in fills)

    assert has_min_color, (
        f"SequentialScale(scheme='{scheme}') — expected a fill matching the "
        f"low-end of the '{scheme}' ramp ({expected_description}), but found "
        f"none.  The scheme is being ignored.  Mark fills: {fills}"
    )
    assert has_max_color, (
        f"SequentialScale(scheme='{scheme}') — expected a fill matching the "
        f"high-end of the '{scheme}' ramp ({expected_description}), but found "
        f"none.  The scheme is being ignored.  Mark fills: {fills}"
    )


def test_sequential_different_schemes_produce_different_svgs(
    linear_df: pl.DataFrame,
) -> None:
    """Two SequentialScale charts with different schemes must produce distinct SVG.

    This is the simplest observable symptom of the bug: viridis, magma, blues,
    and warm_ochre are dramatically different palettes.  Under the bug, all
    four render identically (cool_blue default).
    """
    svgs = {}
    for scheme in ("viridis", "magma", "blues", "warm_ochre"):
        svgs[scheme] = (
            fm.Chart(linear_df)
            .mark_point()
            .encode(
                x="x:Q",
                y="y:Q",
                color=Color("val:Q", scale=fm.SequentialScale(scheme=scheme)),
            )
            .to_svg()
        )

    scheme_list = list(svgs.keys())
    for i, s1 in enumerate(scheme_list):
        for s2 in scheme_list[i + 1 :]:
            assert svgs[s1] != svgs[s2], (
                f"SequentialScale(scheme='{s1}') and SequentialScale(scheme='{s2}') "
                f"produced identical SVG — both are using the default fallback scheme "
                f"instead of the requested scheme.  Mark fills for '{s1}': "
                f"{_mark_fills(svgs[s1])}"
            )


# ---------------------------------------------------------------------------
# D1-C — DivergingScale: scheme must change rendered fill colors
# ---------------------------------------------------------------------------


def test_diverging_rdbu_differs_from_default(diverging_df: pl.DataFrame) -> None:
    """DivergingScale(scheme='rdbu') must produce rdbu fills, not the default
    blue_to_red fallback.

    The blue_to_red custom scheme uses navy blue (#1e3a8a) and dark red
    (#7f1d1d) as endpoints.  The canonical rdbu scheme uses deep red (#67001f)
    and deep blue (#042f60) — different hues at the same end of the diverging
    ramp.

    Bug indicator: DivergingScale(scheme='rdbu') renders with blue_to_red
    colors because the scale_common_scheme() function does not handle the
    Diverging variant.
    """
    svg_rdbu = (
        fm.Chart(diverging_df)
        .mark_point()
        .encode(
            x="x:Q",
            y="y:Q",
            color=Color("val:Q", scale=fm.DivergingScale(scheme="rdbu", domain=[-2.0, 0.0, 2.0])),
        )
        .to_svg()
    )
    svg_default_diverging = (
        fm.Chart(diverging_df)
        .mark_point()
        .encode(
            x="x:Q",
            y="y:Q",
            # No scheme: uses theme default (blue_to_red)
            color=Color("val:Q", scale=fm.DivergingScale(domain=[-2.0, 0.0, 2.0])),
        )
        .to_svg()
    )

    fills_rdbu = _mark_fills(svg_rdbu)
    fills_default = _mark_fills(svg_default_diverging)

    assert fills_rdbu != fills_default, (
        "DivergingScale(scheme='rdbu') and DivergingScale(no scheme) produced "
        "identical mark fill colors — 'rdbu' is being ignored and the default "
        f"blue_to_red scheme is used for both.  rdbu fills: {fills_rdbu}, "
        f"default fills: {fills_default}"
    )


def test_diverging_rdbu_has_red_endpoint(diverging_df: pl.DataFrame) -> None:
    """DivergingScale(scheme='rdbu') with a negative-spanning domain must render
    at least one mark in the distinctive dark-crimson hue of the rdbu low-end.

    rdbu(0) is approximately #67001f (R=103, G=0, B=31): a very dark crimson
    with near-zero green.  The default blue_to_red fallback uses #7f1d1d
    (R=127, G=29, B=29) which also has low green but noticeably more (G=29).

    The discriminating check requires G < 10 — present in rdbu's #67001f
    (G=0) but absent in blue_to_red's #7f1d1d (G=29).  This ensures the test
    fails when blue_to_red is used as a fallback and passes only when rdbu is
    actually applied.

    Hex reference: colorous RED_BLUE evaluated at t=0.0 → #67001f.
    """
    svg = (
        fm.Chart(diverging_df)
        .mark_point()
        .encode(
            x="x:Q",
            y="y:Q",
            color=Color("val:Q", scale=fm.DivergingScale(scheme="rdbu", domain=[-2.0, 0.0, 2.0])),
        )
        .to_svg()
    )
    fills = _mark_fills(svg)

    # rdbu(0) #67001f: R=103, G=0, B=31 — near-zero green distinguishes it
    # from blue_to_red's #7f1d1d (R=127, G=29, B=29).
    has_rdbu_crimson = any(r > 80 and g < 10 and b < 50 for h in fills for r, g, b in [_rgb(h)])
    assert has_rdbu_crimson, (
        "Expected a near-zero-green dark-red fill near #67001f (rdbu t=0, G≈0) "
        "in SVG mark fills, but found none.  The rdbu scheme is being ignored; "
        "the default blue_to_red scheme uses #7f1d1d (G=29) instead.  "
        f"Mark fills: {fills}"
    )


def test_diverging_rdbu_has_blue_endpoint(diverging_df: pl.DataFrame) -> None:
    """DivergingScale(scheme='rdbu') with a positive-spanning domain must render
    at least one mark in a deep-blue hue matching the rdbu high-end.

    rdbu(1) is approximately #042f60 (R=4, G=47, B=96): deep navy.
    The tolerant check requires R < 30, G < 70, B > 70.

    Hex reference: colorous RED_BLUE evaluated at t=1.0 → #042f60.
    """
    svg = (
        fm.Chart(diverging_df)
        .mark_point()
        .encode(
            x="x:Q",
            y="y:Q",
            color=Color("val:Q", scale=fm.DivergingScale(scheme="rdbu", domain=[-2.0, 0.0, 2.0])),
        )
        .to_svg()
    )
    fills = _mark_fills(svg)

    # rdbu(1) #042f60: R=4, G=47, B=96 — deep navy
    has_deep_blue = any(r < 30 and g < 70 and b > 70 for h in fills for r, g, b in [_rgb(h)])
    assert has_deep_blue, (
        "Expected a deep-blue fill near #042f60 (rdbu t=1) in SVG mark fills, "
        "but found none.  The rdbu scheme is being ignored and the default "
        f"blue_to_red scheme is rendering instead.  Mark fills: {fills}"
    )


def test_diverging_domain_clamping_collapses_out_of_range_to_endpoint(
    diverging_df: pl.DataFrame,
) -> None:
    """DivergingScale with domain=[-1, 0, 1] clamps out-of-range values to
    the endpoint colors.

    The diverging_df fixture has val=-2 and val=2 which exceed the declared
    domain [-1, 0, 1].  When domain clamping is correctly applied, the marks
    at val=-2 and val=-1 render identically (clamped to the same color as -1),
    and likewise val=2 and val=1 render identically.

    This test verifies that the number of distinct mark colors is LESS with an
    explicit narrow domain (fewer distinct colors because out-of-range values
    clamp to endpoints) than when the domain is wider (all 7 values map to
    distinct interpolated positions).

    NOTE: this test exercises domain application, which is a secondary concern
    of D1; scheme resolution is the primary bug.  This test only fails once the
    scheme IS honored (because domain application is only meaningful if the
    right scheme is used in the first place).
    """
    # Narrow domain [-1, 0, 1]: values ±2 clamp to ±1's color
    svg_narrow = (
        fm.Chart(diverging_df)
        .mark_point()
        .encode(
            x="x:Q",
            y="y:Q",
            color=Color("val:Q", scale=fm.DivergingScale(scheme="rdbu", domain=[-1.0, 0.0, 1.0])),
        )
        .to_svg()
    )
    # Wide domain [-2, 0, 2]: all 7 values map to distinct positions
    svg_wide = (
        fm.Chart(diverging_df)
        .mark_point()
        .encode(
            x="x:Q",
            y="y:Q",
            color=Color("val:Q", scale=fm.DivergingScale(scheme="rdbu", domain=[-2.0, 0.0, 2.0])),
        )
        .to_svg()
    )

    # With the narrow domain, ±2 values clamp to the same color as ±1,
    # so there should be strictly fewer distinct mark colors.
    fills_narrow = _mark_fills(svg_narrow)
    fills_wide = _mark_fills(svg_wide)

    # Primary gate: the rdbu scheme is being honored.
    # rdbu(0) #67001f: R=103, G=0, B=31 — G < 10 distinguishes it from the
    # blue_to_red fallback's dark red #7f1d1d (G=29).
    # rdbu(1) #042f60: R=4, G=47, B=96 — R < 10 distinguishes it from
    # the blue_to_red fallback's navy #1e3a8a (R=30).
    has_rdbu_color_narrow = any(
        (r > 80 and g < 10 and b < 50) or (r < 10 and g < 70 and b > 70)
        for h in fills_narrow
        for r, g, b in [_rgb(h)]
    )
    assert has_rdbu_color_narrow, (
        "DivergingScale(scheme='rdbu', domain=[-1,0,1]) — the rdbu scheme is "
        "not being honored (no rdbu-characteristic color found with G<10 for "
        "the red end or R<10 for the blue end).  The default blue_to_red "
        f"scheme is being used instead.  Mark fills: {fills_narrow}"
    )

    # Clamping gate: narrow domain must produce strictly fewer distinct colors
    # than wide. The fixture has values beyond ±1, so under the narrow domain
    # they clamp to the ±1 endpoint colors, collapsing distinct mark-color count.
    assert len(fills_narrow) < len(fills_wide), (
        f"Narrow domain [-1,0,1] produced {len(fills_narrow)} distinct mark colors "
        f"but wide domain [-2,0,2] produced {len(fills_wide)}.  Expected the "
        f"narrow domain to produce strictly fewer distinct colors because "
        f"out-of-range values should clamp to endpoint colors.  "
        f"narrow fills: {fills_narrow}, wide fills: {fills_wide}"
    )


# ---------------------------------------------------------------------------
# D1-D — SequentialScale vs. encoding-level scheme: must produce same colors
# ---------------------------------------------------------------------------


def test_sequential_scale_scheme_matches_encoding_level_scheme(
    linear_df: pl.DataFrame,
) -> None:
    """SequentialScale(scheme='viridis') must produce the same fills as
    Color('val:Q', scheme='viridis') (the encoding-level scheme shorthand).

    The encoding-level ``scheme=`` on ``Color(...)`` already works (it takes
    the D4 path via ``c_enc.scheme``).  ``SequentialScale(scheme=...)`` should
    produce identical output once the D1 fix lands.

    This test directly compares the two forms and fails when they differ,
    which is the current buggy state.
    """
    svg_scale = (
        fm.Chart(linear_df)
        .mark_point()
        .encode(
            x="x:Q",
            y="y:Q",
            color=Color("val:Q", scale=fm.SequentialScale(scheme="viridis")),
        )
        .to_svg()
    )
    svg_shorthand = (
        fm.Chart(linear_df)
        .mark_point()
        .encode(
            x="x:Q",
            y="y:Q",
            color=Color("val:Q", scheme="viridis"),
        )
        .to_svg()
    )

    fills_scale = _mark_fills(svg_scale)
    fills_shorthand = _mark_fills(svg_shorthand)

    assert fills_scale == fills_shorthand, (
        "SequentialScale(scheme='viridis') and Color(..., scheme='viridis') "
        "produced different mark fills — the SequentialScale path is not "
        "honoring the scheme and is using the default fallback instead.  "
        f"SequentialScale fills: {fills_scale}, "
        f"shorthand fills: {fills_shorthand}"
    )


def test_diverging_scale_scheme_matches_encoding_level_scheme(
    diverging_df: pl.DataFrame,
) -> None:
    """DivergingScale(scheme='rdbu') must produce the same fills as
    Color('val:Q', scheme='rdbu') (the encoding-level scheme shorthand).

    The encoding-level ``scheme=`` already works.  ``DivergingScale(scheme=...)``
    must produce identical output once the D1 fix lands.
    """
    svg_scale = (
        fm.Chart(diverging_df)
        .mark_point()
        .encode(
            x="x:Q",
            y="y:Q",
            color=Color("val:Q", scale=fm.DivergingScale(scheme="rdbu", domain=[-2.0, 0.0, 2.0])),
        )
        .to_svg()
    )
    svg_shorthand = (
        fm.Chart(diverging_df)
        .mark_point()
        .encode(
            x="x:Q",
            y="y:Q",
            color=Color("val:Q", scheme="rdbu"),
        )
        .to_svg()
    )

    fills_scale = _mark_fills(svg_scale)
    fills_shorthand = _mark_fills(svg_shorthand)

    assert fills_scale == fills_shorthand, (
        "DivergingScale(scheme='rdbu') and Color(..., scheme='rdbu') produced "
        "different mark fills — the DivergingScale path is not honoring the "
        "scheme and is using the default fallback instead.  "
        f"DivergingScale fills: {fills_scale}, "
        f"shorthand fills: {fills_shorthand}"
    )


# ---------------------------------------------------------------------------
# Gap 1 — Four spec-named sequential schemes must render their real palettes
# ---------------------------------------------------------------------------


@pytest.fixture
def cool_blue_fills(linear_df: pl.DataFrame) -> set[str]:
    """Render the default cool_blue ramp and return its mark fills.

    Used as a control to detect fall-through: if reds/greens/oranges/purples
    produce the same fill set as cool_blue, the scheme is being ignored.
    """
    svg = (
        fm.Chart(linear_df)
        .mark_point()
        .encode(
            x="x:Q",
            y="y:Q",
            color=Color("val:Q", scale=fm.SequentialScale(scheme="cool_blue")),
        )
        .to_svg()
    )
    return _mark_fills(svg)


@pytest.mark.parametrize(
    "scheme, expected_description, min_check, max_check",
    [
        # reds: near-white #fff5f0 → dark crimson #67000d
        # reds(0) R=255, G=245, B=240; reds(1) R=103, G=0, B=13
        (
            "reds",
            "near-white pink-tinted (min) → dark crimson (max)",
            lambda r, g, b: r > 220 and g > 200 and b > 200,
            lambda r, g, b: r > 80 and g < 30 and b < 40,
        ),
        # greens: near-white #f7fcf5 → dark green #00441b
        # greens(0) R=247, G=252, B=245; greens(1) R=0, G=68, B=27
        (
            "greens",
            "near-white green-tinted (min) → dark forest green (max)",
            lambda r, g, b: r > 220 and g > 220 and b > 220,
            lambda r, g, b: r < 30 and g > 50 and b < 40,
        ),
        # oranges: near-white #fff5eb → dark brown-orange #7f2704
        # oranges(0) R=255, G=245, B=235; oranges(1) R=127, G=39, B=4
        (
            "oranges",
            "near-white warm-tinted (min) → dark brown-orange (max)",
            lambda r, g, b: r > 220 and g > 200 and b > 180,
            lambda r, g, b: r > 100 and g < 60 and b < 20,
        ),
        # purples: near-white #fcfbfd → dark purple #3f007d
        # purples(0) R=252, G=251, B=253; purples(1) R=63, G=0, B=125
        (
            "purples",
            "near-white lavender-tinted (min) → dark purple (max)",
            lambda r, g, b: r > 220 and g > 220 and b > 220,
            lambda r, g, b: r < 80 and g < 20 and b > 80,
        ),
    ],
    ids=["reds", "greens", "oranges", "purples"],
)
def test_spec_sequential_schemes_render_characteristic_hues(
    linear_df: pl.DataFrame,
    cool_blue_fills: set[str],
    scheme: str,
    expected_description: str,
    min_check,
    max_check,
) -> None:
    """The four spec-named sequential schemes reds/greens/oranges/purples must
    render their characteristic ColorBrewer palette hues, not the cool_blue
    fallback.

    Each parametrized case asserts:
    1. The scheme's endpoint colors appear in the rendered fills (both min and
       max end of the ramp).
    2. The rendered fills differ from the cool_blue default — a fall-through
       to the default scheme is caught by the distinctness guard.

    Hex references from the colorous crate ColorBrewer tables:
      reds:    #fff5f0 (t=0, R=255, G=245, B=240) → #67000d (t=1, R=103, G=0,  B=13)
      greens:  #f7fcf5 (t=0, R=247, G=252, B=245) → #00441b (t=1, R=0,   G=68, B=27)
      oranges: #fff5eb (t=0, R=255, G=245, B=235) → #7f2704 (t=1, R=127, G=39, B=4)
      purples: #fcfbfd (t=0, R=252, G=251, B=253) → #3f007d (t=1, R=63,  G=0,  B=125)
    """
    svg = (
        fm.Chart(linear_df)
        .mark_point()
        .encode(
            x="x:Q",
            y="y:Q",
            color=Color("val:Q", scale=fm.SequentialScale(scheme=scheme)),
        )
        .to_svg()
    )
    fills = _mark_fills(svg)

    # Distinctness guard: if the scheme falls through to cool_blue, this fails.
    assert fills != cool_blue_fills, (
        f"SequentialScale(scheme='{scheme}') produced the same mark fills as "
        f"the cool_blue default — the scheme is being silently ignored and the "
        f"cool_blue fallback is rendering instead.  "
        f"fills: {fills}"
    )

    has_min_color = any(min_check(*_rgb(h)) for h in fills)
    has_max_color = any(max_check(*_rgb(h)) for h in fills)

    assert has_min_color, (
        f"SequentialScale(scheme='{scheme}') — expected a fill matching the "
        f"low-end of the '{scheme}' ramp ({expected_description}), but found "
        f"none.  The scheme is being ignored.  Mark fills: {fills}"
    )
    assert has_max_color, (
        f"SequentialScale(scheme='{scheme}') — expected a fill matching the "
        f"high-end of the '{scheme}' ramp ({expected_description}), but found "
        f"none.  The scheme is being ignored.  Mark fills: {fills}"
    )


# ---------------------------------------------------------------------------
# Gap 2 — DivergingScale: non-central midpoint must be honored
# ---------------------------------------------------------------------------


@pytest.fixture
def noncentral_midpoint_df() -> pl.DataFrame:
    """Four rows at exactly -1, 0.0, 0.5, 1.0.

    Used to test DivergingScale(domain=[-1, 0.5, 1]) where the supplied
    midpoint 0.5 is NOT the geometric center (0.0+1.0)/2 = 0.0 of [-1, 1].

    Having exact rows at all four values lets the test assert which fill
    corresponds to which data value by rank-ordering SVG cx positions.
    """
    return pl.DataFrame(
        {
            "x": [-1.0, 0.0, 0.5, 1.0],
            "y": [0.0, 1.0, 2.0, 3.0],
            "val": [-1.0, 0.0, 0.5, 1.0],
        }
    )


def _fills_by_cx(svg: str) -> list[str]:
    """Return mark fill hex colors sorted by SVG cx position (ascending x).

    Extracts ``cx="..." fill="#RRGGBB"`` pairs from circle elements and sorts
    them by ascending cx value, yielding fills in left-to-right (data value
    ascending) order.

    Returns lowercase 6-char hex strings, without the leading ``#``.
    """
    cx_fills = re.findall(r'cx="([^"]+)"[^>]*fill="#([0-9a-fA-F]{6})"', svg)
    sorted_pairs = sorted(cx_fills, key=lambda pair: float(pair[0]))
    return [h.lower() for _, h in sorted_pairs]


def test_diverging_noncentral_midpoint_is_neutral(
    noncentral_midpoint_df: pl.DataFrame,
) -> None:
    """DivergingScale(scheme='rdbu', domain=[-1, 0.5, 1]) must center the
    neutral color at the SUPPLIED midpoint 0.5, not at the geometric center 0.0.

    The dataset has exactly four rows: val=-1, val=0.0, val=0.5, val=1.0.
    With the non-central midpoint 0.5 declared:

    - val=-1.0 maps to t=0 → rdbu red end (dark crimson, R high, G near 0)
    - val= 0.0 maps to left of mid → should be REDDISH (R channel dominant,
      since 0.0 < 0.5 midpoint), not neutral
    - val= 0.5 maps to t=midpoint → NEAR NEUTRAL (RdBu center is pale, all
      channels high and close together: each > 200, spread < 40)
    - val= 1.0 maps to t=1 → rdbu blue end (deep navy, B channel dominant)

    Under the bug the midpoint is discarded and pure-linear interpolation is
    used: neutrality lands at the geometric center 0.0 instead of 0.5.
    So val=0.0 gets near-neutral and val=0.5 gets blue-ish.

    Assertions:
    1. val=0.5 (third in cx order) is near-neutral: R, G, B each > 200 and
       max(R,G,B) - min(R,G,B) < 40.
    2. val=0.0 (second in cx order) is NOT the most-neutral mark: it should be
       reddish (R > B, i.e. leans red) because it falls left of the 0.5
       midpoint.

    RdBu center reference: colorous RED_BLUE evaluated at t=0.5 is near white/
    pale grey (R≈247, G≈247, B≈247 or similar — all channels high, low spread).
    """
    svg = (
        fm.Chart(noncentral_midpoint_df)
        .mark_point()
        .encode(
            x="x:Q",
            y="y:Q",
            color=Color(
                "val:Q",
                scale=fm.DivergingScale(scheme="rdbu", domain=[-1.0, 0.5, 1.0]),
            ),
        )
        .to_svg()
    )

    fills_ordered = _fills_by_cx(svg)

    assert len(fills_ordered) == 4, (
        f"Expected 4 mark fills (one per data row) sorted by cx, "
        f"got {len(fills_ordered)}: {fills_ordered}"
    )

    # fills_ordered[0] = val=-1 (leftmost), [1] = val=0, [2] = val=0.5, [3] = val=1
    fill_at_midpoint = fills_ordered[2]  # val=0.5 — the supplied midpoint
    fill_at_geom_center = fills_ordered[1]  # val=0.0 — the geometric center

    r_mid, g_mid, b_mid = _rgb(fill_at_midpoint)
    r_geo, g_geo, b_geo = _rgb(fill_at_geom_center)

    # Assertion 1: val=0.5 must be near-neutral (all channels high and balanced)
    spread_mid = max(r_mid, g_mid, b_mid) - min(r_mid, g_mid, b_mid)
    assert r_mid > 200 and g_mid > 200 and b_mid > 200 and spread_mid < 40, (
        f"val=0.5 (the supplied midpoint) should render near-neutral "
        f"(R,G,B all > 200 and spread < 40) but got #{fill_at_midpoint} "
        f"(R={r_mid}, G={g_mid}, B={b_mid}, spread={spread_mid}).  "
        f"The non-central midpoint 0.5 is being ignored — the geometric "
        f"center 0.0 is being treated as neutral instead.  "
        f"All fills in cx order: {fills_ordered}"
    )

    # Assertion 2: val=0.0 must lean red (R > B), since 0.0 < 0.5 midpoint
    # means it is on the red side of the diverging scale.  Under the bug,
    # val=0.0 is near-neutral (treated as the geometric center).
    assert r_geo > b_geo, (
        f"val=0.0 should lean red (R > B) because it lies left of the "
        f"declared midpoint 0.5, but got #{fill_at_geom_center} "
        f"(R={r_geo}, G={g_geo}, B={b_geo}).  "
        f"The geometric center 0.0 is incorrectly being treated as neutral.  "
        f"All fills in cx order: {fills_ordered}"
    )
