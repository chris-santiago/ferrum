"""Feature tests for ``Chart.mark_boxen(palette=)`` (2026-08-27 residuals
batch).

Split out of ``tests/test_finding_p9.py`` (design-review remediation,
2026-08-27): P9 is a findings-regression file scoped to "accept-and-``del``
mark parameters" -- boxen's ``palette=`` was never a dropped parameter, it
is a from-scratch feature, so its ~310 lines of coverage did not belong
there. The split rule this repo uses going forward: a findings-ID-named
test file (``test_finding_p*.py``) stays scoped to pinning the specific
regression/disposition the finding describes; net-new feature coverage for
a mark or figure function gets its own ``test_<feature>.py`` module, even
when the feature landed in the same batch as a findings fix. Boxen's other,
pre-existing (non-palette) coverage is unaffected and stays wherever it
already lived (see ``tests/test_finding_p9.py`` and the seven other files
grep turns up for ``mark_boxen``).
"""

from __future__ import annotations

import re
import warnings

import polars as pl
import pytest

import ferrum
from ferrum._warn import reset_warnings


def _boxen_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "group": ["a"] * 10 + ["b"] * 10,
            "val": list(range(10)) + list(range(5, 15)),
        }
    )


def _boxen_df_multi_band() -> pl.DataFrame:
    """The standard palette-testing fixture (quality-review cycle-3
    census used n=200): large enough per group that the mark's default
    ``k_depth="tukey"`` selects several real letter-value depths, so
    palette-mapping assertions exercise the mark's actual default
    configuration, not just ``k_depth="full"``."""
    return pl.DataFrame(
        {
            "group": ["a"] * 200 + ["b"] * 200,
            "val": list(range(200)) + list(range(100, 300)),
        }
    )


def _boxen_df_small() -> pl.DataFrame:
    """Small enough to be a meaningfully different regime from
    ``_boxen_df_multi_band``, but still >=32 rows/group -- ``k_depth=
    "tukey"``'s ``floor(log2(n)) - 3`` needs ``n >= 32`` to reach a real
    depth beyond the median (``k=2``); below that, every row is the
    degenerate ``k=1`` band and no color could possibly be visible
    regardless of mapping. Reaches exactly one real band (``k=2``)."""
    return pl.DataFrame(
        {
            "group": ["a"] * 40 + ["b"] * 40,
            "val": list(range(40)) + list(range(20, 60)),
        }
    )


def _rect_fills(svg: str) -> list[str]:
    """Every ``fill="..."`` value on a ``<rect>`` element, in document order."""
    return re.findall(r'<rect\b[^>]*\bfill="([^"]+)"', svg)


def _rect_geoms(svg: str) -> list[tuple[float, float, float, float, str]]:
    """``(x, y, width, height, fill)`` for every ``<rect x=... y=...
    width=... height=... fill=...>`` element, in document order.

    Ties markup to *geometry* rather than fill alone, so a paint-order
    regression (a band emitted with the right color but occluded by a
    later, wider, fully-opaque band) is visible to a test even when the
    fill attributes alone would look correct."""
    pattern = re.compile(
        r'<rect\b[^>]*\bx="([-\d.]+)"[^>]*\by="([-\d.]+)"[^>]*'
        r'\bwidth="([-\d.]+)"[^>]*\bheight="([-\d.]+)"[^>]*\bfill="([^"]+)"'
    )
    return [
        (float(x), float(y), float(w), float(h), fill) for x, y, w, h, fill in pattern.findall(svg)
    ]


def _dedup_consecutive(values: list[str]) -> list[str]:
    """Collapse adjacent repeats (one rect per group, per depth band) while
    preserving depth order -- unlike ``dict.fromkeys``, a color that
    reappears in a later, non-adjacent band (palette cycling) is kept."""
    out: list[str] = []
    for value in values:
        if not out or out[-1] != value:
            out.append(value)
    return out


def test_mark_boxen_palette_named_yields_distinct_band_fills():
    """A named palette colors the depth bands directly, and never warns
    (spec §4.4/§9.4: ≥2 distinct band fills; no more warn-bridge)."""
    reset_warnings()
    df = _boxen_df_multi_band()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        svg = ferrum.Chart(df).mark_boxen(palette="tableau10").encode(x="group", y="val").to_svg()
    assert [w for w in caught if issubclass(w.category, UserWarning)] == []
    expected = ferrum.color.palette("tableau10", n=5)
    band_fills = [f for f in _rect_fills(svg) if f in expected]
    assert len(set(band_fills)) >= 2
    # k=1 (the always-degenerate median band) borrows k=2's color slot
    # instead of consuming one of its own. k=2 is the **base band** --
    # spec §4.4's re-amended anchor (quality-review cycle-3) -- so it (and
    # the k=1 band that shares its color) is colors[0] directly, and is
    # painted last/on top, i.e. the last fill in document order.
    assert band_fills[-1] == expected[0]


def test_mark_boxen_palette_list_applies_in_order_and_cycles():
    """An explicit color sequence is applied to bands in order, and a
    shorter-than-the-colorable-band-count list cycles (spec §4.4/§9.4).

    ``k_depth="full"`` on the standard fixture reaches all 6 configured
    depths; the mark's default ``k_depth="tukey"`` on the same data
    reaches only ~3 real depths, and since k=1 always shares k=2's color
    slot (previous test), a 3-depth render only ever shows 2
    dedup-distinct colors -- not enough to prove cycling through a
    2-color list."""
    df = _boxen_df_multi_band()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        svg = (
            ferrum.Chart(df)
            .mark_boxen(palette=["#111111", "#222222"], k_depth="full")
            .encode(x="group", y="val")
            .to_svg()
        )
    # Two rows per depth (one rect per group), so dedup consecutive
    # duplicates while preserving depth order.
    band_colors = _dedup_consecutive(_rect_fills(svg))
    band_colors = [c for c in band_colors if c in {"#111111", "#222222"}]
    assert len(band_colors) >= 3, "need >=3 bands to prove cycling"
    # Cycled to 5 slots (colors[i % 2] for i in 0..4): [c0,c1,c0,c1,c0],
    # indexed k=2->0, k=3->1, k=4->2, k=5->3, k=6->4 (base-band anchor).
    # Document order is widest-first (k=6..2, then k=1 merging into k=2):
    # fills = [c0,c1,c0,c1,c0,c0] -> dedup = [c0,c1,c0,c1,c0].
    assert band_colors[0] == "#111111"  # k=6, index 4 -> cycled color c0
    assert band_colors[1] == "#222222"  # k=5, index 3 -> c1
    assert band_colors[-1] == "#111111"  # k=2 (and k=1, merged): index 0 -> c0


def _assert_boxen_palette_visually_correct(svg: str, expected_colors: list[str]) -> None:
    """Shared assertion body for the base-band color-mapping contract
    (spec §4.4, re-amended after quality-review cycle-3): per group,
    (a) band heights strictly decrease in document order (widest-first
    paint order, so nesting stays visible -- S1), (b) the innermost
    *real* band's fill is ``expected_colors[0]`` (the base-band anchor --
    ``k=2``, painted last among real bands, right before the always-
    degenerate ``k=1`` that shares its color -- so it is the
    second-to-last entry in document order), and (c) no color is visible
    *only* on the degenerate (minimum-height) band.

    Verified on rect *geometry*, not fill attributes alone -- a band that
    is emitted with the right color but occluded, or a color assigned
    only to a band that never gets real extent, both look correct in
    markup while being invisible on screen."""
    expected_fills = set(expected_colors)
    band_rects = [g for g in _rect_geoms(svg) if g[4] in expected_fills]
    by_group: dict[float, list[tuple[float, str]]] = {}
    for x, _y, _w, h, fill in band_rects:
        by_group.setdefault(round(x, 1), []).append((h, fill))
    assert len(by_group) >= 2, "need >=2 groups (categorical positions)"
    for group_x, height_fills in by_group.items():
        heights = [h for h, _fill in height_fills]
        assert len(heights) >= 2, (
            f"group at x={group_x}: need at least the degenerate k=1 band plus one real band"
        )
        assert heights == sorted(heights, reverse=True) and len(set(heights)) == len(heights), (
            f"group at x={group_x}: band heights not strictly decreasing "
            f"in document order (widest-first, i.e. nested and visible): "
            f"{heights}"
        )

        # Base-band anchor: colors[0] lands on the innermost *real* band
        # (k=2), which is the second-to-last entry -- the last entry is
        # always the degenerate k=1 band (borrows k=2's color, smaller
        # height).
        innermost_real_height, innermost_real_fill = height_fills[-2]
        assert innermost_real_fill == expected_colors[0], (
            f"group at x={group_x}: innermost real band (height="
            f"{innermost_real_height}) has fill {innermost_real_fill!r}, "
            f"expected colors[0] = {expected_colors[0]!r}"
        )

        # No requested color may be visible *only* on the degenerate
        # (minimum-height) band -- every fill that appears must also
        # appear on a taller (real) band.
        min_height = min(heights)
        degenerate_only_fills = {f for h, f in height_fills if h == min_height} - {
            f for h, f in height_fills if h > min_height
        }
        assert not degenerate_only_fills, (
            f"group at x={group_x}: color(s) {degenerate_only_fills} only "
            f"appear on the degenerate (height={min_height}) band"
        )


def test_mark_boxen_palette_bands_paint_widest_first_per_group():
    """Structural nesting pin (quality-review S1, spec §4.4 "Paint order"
    amendment): under ``palette=``, depth bands must paint widest-first
    (outermost under, innermost on top) within each group, so every band
    stays visibly nested instead of the widest band occluding the rest.
    Also pins the **base-band color mapping** (spec §4.4, re-amended after
    quality-review cycle-3): ``colors[0]`` must be visible -- landing on
    the innermost real band, ``k=2``, which is guaranteed to render
    whenever *any* real depth exists -- at ``k_depth="full"``, the mark's
    default ``k_depth`` on the standard fixture, and at a small ``n`` that
    barely reaches one real band. An earlier anchor (widest *configured*
    band, ``k=_BOXEN_K_MAX``) only rendered ``colors[0]`` when a dataset
    happened to reach full depth -- dead on every typical dataset under
    the default ``k_depth`` (measured by quality review: only the last
    two of six colors ever appeared at n=200) -- so this test deliberately
    covers all three regimes, not just the one configuration
    (``k_depth="full"``) that cannot fail."""
    expected = ferrum.color.palette("tableau10", n=5)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        svg_full_depth = (
            ferrum.Chart(_boxen_df_multi_band())
            .mark_boxen(palette="tableau10", k_depth="full")
            .encode(x="group", y="val")
            .to_svg()
        )
        svg_default_kdepth = (
            ferrum.Chart(_boxen_df_multi_band())
            .mark_boxen(palette="tableau10")
            .encode(x="group", y="val")
            .to_svg()
        )
        svg_small_n = (
            ferrum.Chart(_boxen_df_small())
            .mark_boxen(palette="tableau10")
            .encode(x="group", y="val")
            .to_svg()
        )

    _assert_boxen_palette_visually_correct(svg_full_depth, expected)
    _assert_boxen_palette_visually_correct(svg_default_kdepth, expected)
    _assert_boxen_palette_visually_correct(svg_small_n, expected)


def test_mark_boxen_palette_continuous_scheme_full_depth_reaches_ramp_endpoint():
    """Regression guard for the palette-expansion-count defect (quality-
    review cycle-3 S3, fixed by sizing ``_resolve_boxen_palette``'s
    request to ``_BOXEN_VISIBLE_BANDS`` instead of ``_BOXEN_K_MAX``) --
    proven necessary by quality review's cycle-4 mutation probe: setting
    ``_BOXEN_VISIBLE_BANDS = _BOXEN_K_MAX`` left every existing palette
    test green, because ``_boxen_band_color_index`` always yields exactly
    5 distinct indices regardless of list length, and the categorical
    (``tableau10``) tests used elsewhere only check that *known* colors
    appear, not that the request count matches the consumable slot count.

    A *continuous* palette is the shape that actually discriminates: it
    is resampled evenly across ``[0, 1]`` at however many colors are
    requested, so requesting 6 points instead of 5 shifts *every* sample
    position, not just the extra one -- under the mutation, the last
    *consumed* color (index 4 of 6, at ``t=0.8``) is no longer the
    palette's endpoint (``t=1.0``), so ``viridis``'s final color never
    renders at all. ``k_depth="full"`` on the standard fixture guarantees
    every one of the 5 consumable bands (``k=2..6``) has real data, so
    every one of the 5 expected samples -- including the endpoint --
    must appear on a non-degenerate (height > 1) rect."""
    df = _boxen_df_multi_band()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        svg = (
            ferrum.Chart(df)
            .mark_boxen(palette="viridis", k_depth="full")
            .encode(x="group", y="val")
            .to_svg()
        )
    expected = ferrum.color.palette("viridis", n=5)
    rendered_non_degenerate = {fill for _x, _y, _w, h, fill in _rect_geoms(svg) if h > 1}
    missing = set(expected) - rendered_non_degenerate
    assert not missing, (
        f"expected viridis samples {expected} not all rendered at "
        f"non-degenerate height; missing {sorted(missing)} (a missing "
        f"endpoint, {expected[-1]!r}, means the palette expansion count "
        f"no longer matches the consumable slot count)"
    )


def test_mark_boxen_palette_conflicts_with_chart_level_color_encoding():
    """``palette=`` combined with a chart-level ``.encode(color=...)``
    channel raises ``ValueError`` instead of silently rendering a flat
    block (quality-review S2): the color encoding always overrides a
    layer's ``fill=``, so the palette would have no visible effect while
    still forcing opacity to 1.0. Checked both call orders -- desugaring
    is deferred until ``.encode()`` is fully known, regardless of whether
    ``mark_boxen()`` or ``.encode(color=...)`` came first in the chain."""
    df = _boxen_df_multi_band()
    with pytest.raises(ValueError, match="color encoding"):
        ferrum.Chart(df).mark_boxen(palette=["#111111", "#222222"]).encode(
            x="group", y="val", color="group"
        ).to_svg()
    with pytest.raises(ValueError, match="color encoding"):
        ferrum.Chart(df).encode(x="group", y="val", color="group").mark_boxen(
            palette=["#111111", "#222222"]
        ).to_svg()


def test_mark_boxen_palette_and_color_field_still_compose():
    """``color_field=`` (boxen's own per-group grouping kwarg) is a
    different mechanism from a chart-level ``color`` encoding and stays
    unaffected by the new conflict guard."""
    df = _boxen_df_multi_band()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        svg = (
            ferrum.Chart(df)
            .mark_boxen(palette=["#111111", "#222222"], color_field="group")
            .encode(x="group", y="val")
            .to_svg()
        )
    fills = set(_rect_fills(svg))
    assert {"#111111", "#222222"} <= fills


def test_mark_boxen_palette_non_iterable_raises_value_error():
    """A non-``str``, non-iterable ``palette=`` value (e.g. an ``int``)
    raises the same named ``ValueError`` shape as the empty-sequence
    guard, not a bare ``TypeError`` leaking from ``list(palette)``
    (quality-review S4)."""
    df = _boxen_df()
    with pytest.raises(ValueError, match=r"mark_boxen\(palette=\.\.\.\)"):
        ferrum.Chart(df).mark_boxen(palette=5).encode(x="group", y="val").to_svg()


def test_mark_boxen_palette_none_is_byte_identical_to_default():
    """``palette=None`` (explicit or omitted) keeps the opacity-ramp
    shading byte-identical -- one of the batch's pinned invariants
    (spec §7, §9.4)."""
    df = _boxen_df_multi_band()
    reset_warnings()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        svg_default = ferrum.Chart(df).mark_boxen().encode(x="group", y="val").to_svg()
        svg_none = ferrum.Chart(df).mark_boxen(palette=None).encode(x="group", y="val").to_svg()
    assert svg_default == svg_none


def test_mark_boxen_palette_invalid_name_raises_value_error():
    """An unrecognized palette name raises through the same validation
    path ``scheme=`` uses (spec §4.4/§9.4)."""
    df = _boxen_df()
    with pytest.raises(ValueError, match="Unknown palette"):
        ferrum.Chart(df).mark_boxen(palette="not-a-real-palette").encode(
            x="group", y="val"
        ).to_svg()


@pytest.mark.parametrize("empty_palette", [[], ()])
def test_mark_boxen_palette_empty_sequence_raises_value_error(empty_palette):
    """An empty color sequence can't color any band (and would otherwise
    ``ZeroDivisionError`` in the cycling ``i % len(colors)`` arithmetic) --
    hardening around the sequence path, named as a ``mark_boxen(palette=)``
    error so the caller knows which argument is at fault."""
    df = _boxen_df()
    with pytest.raises(ValueError, match=r"mark_boxen\(palette=\.\.\.\)"):
        ferrum.Chart(df).mark_boxen(palette=empty_palette).encode(x="group", y="val").to_svg()
