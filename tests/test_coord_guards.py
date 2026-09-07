"""Coord-requirement guards for `mark_geoshape` and `mark_arc` (batch E, Task 6).

`mark_geoshape` only knows how to project GeoJSON geometry under
`CoordGeo(...)`; `mark_arc` only knows how to place a wedge under
`CoordPolar(...)`. Before this fix, `mark_geoshape` under any other coord
silently rendered nothing (no error, no warning), and `mark_arc` under any
other coord surfaced the generic, misleading `"encoding 'y' expected
EncodingSpec, got None"` from scale resolution instead of naming the real
problem. Both now refuse with a typed `ValueError` (Rust
`RenderError::MarkRequiresCoord`, surfaced through PyO3) naming the mark,
the required coord, and the `.coord(...)` fix.

Covers every non-matching coord kind named in the design spec's edge-case
list (§9): default/absent, explicit Cartesian, CoordFlip, and the
crosswise case (geoshape under CoordPolar; arc under CoordGeo). Matching
coords must still render without error (`mark_image`'s own, unrelated
Polar/Geo empty-scene behavior is untouched — see #118).

`mark_arc`'s refusal is path-independent (quality review round 1, Required
Fix 1): both the both-`x`-and-`y`-bound shape (the branch that pre-fix
resolved scales cleanly and only THEN silently rendered nothing) and the
merged-flat `LayerChart` layer shape (`.interactive()`'s single-panel
layout, and `resolve={"y": "independent"}`, both of which resolve scales
against a DIFFERENT layer's mark and so never reach the scale-resolution
guard for the arc layer itself) are pinned below.
"""

from __future__ import annotations

import polars as pl
import pytest

import ferrum as fm

# A single unit-square GeoJSON polygon — enough for `mark_geoshape` to see
# real geometry (`_coerce.py` lifts it into the `__geometry__` column).
_BARE_POLYGON = {
    "type": "Polygon",
    "coordinates": [[[-10, -10], [10, -10], [10, 10], [-10, 10], [-10, -10]]],
}


def _pie_df() -> pl.DataFrame:
    return pl.DataFrame({"cat": ["A", "B", "C"], "val": [10.0, 20.0, 30.0]})


# ── mark_geoshape requires CoordGeo ───────────────────────────────────────


def test_geoshape_default_coord_refuses_with_typed_message():
    chart = fm.Chart(_BARE_POLYGON).mark_geoshape()
    with pytest.raises(ValueError, match=r"mark_geoshape requires a Geo coord"):
        chart.to_svg()


def test_geoshape_explicit_cartesian_coord_refuses():
    chart = fm.Chart(_BARE_POLYGON).mark_geoshape().coord(fm.CoordCartesian())
    with pytest.raises(ValueError, match=r"mark_geoshape requires a Geo coord"):
        chart.to_svg()


def test_geoshape_flip_coord_refuses():
    chart = fm.Chart(_BARE_POLYGON).mark_geoshape().coord(fm.CoordFlip())
    with pytest.raises(ValueError, match=r"mark_geoshape requires a Geo coord"):
        chart.to_svg()


def test_geoshape_polar_coord_refuses_crosswise():
    """The crosswise case: CoordPolar (arc's own requirement) is still just
    "not Geo" from geoshape's point of view."""
    chart = fm.Chart(_BARE_POLYGON).mark_geoshape().coord(fm.CoordPolar(theta="x"))
    with pytest.raises(ValueError, match=r"mark_geoshape requires a Geo coord"):
        chart.to_svg()


def test_geoshape_message_names_the_fix():
    chart = fm.Chart(_BARE_POLYGON).mark_geoshape()
    with pytest.raises(ValueError) as exc_info:
        chart.to_svg()
    assert str(exc_info.value) == (
        "mark_geoshape requires a Geo coord; add .coord(fm.CoordGeo(...))"
    )


def test_geoshape_matching_coord_renders_without_error():
    chart = fm.Chart(_BARE_POLYGON).mark_geoshape().coord(fm.CoordGeo(projection="equirectangular"))
    svg = chart.to_svg()
    assert "<svg" in svg


def test_geoshape_empty_no_geometry_batch_under_non_geo_coord_stays_empty():
    """A chart with no geometry data at all is a legitimately empty result
    under any coord — only geoshape WITH geometry present refuses."""
    empty_df = pl.DataFrame({"x": [], "y": []})
    chart = fm.Chart(empty_df).mark_geoshape()
    svg = chart.to_svg()
    assert "<svg" in svg


# ── mark_arc requires CoordPolar ──────────────────────────────────────────


def test_arc_default_coord_refuses_with_typed_message():
    chart = fm.Chart(_pie_df()).mark_arc().encode(x="val:Q", color="cat:N")
    with pytest.raises(ValueError, match=r"mark_arc requires a Polar coord"):
        chart.to_svg()


def test_arc_explicit_cartesian_coord_refuses():
    chart = (
        fm.Chart(_pie_df()).mark_arc().encode(x="val:Q", color="cat:N").coord(fm.CoordCartesian())
    )
    with pytest.raises(ValueError, match=r"mark_arc requires a Polar coord"):
        chart.to_svg()


def test_arc_flip_coord_refuses():
    chart = fm.Chart(_pie_df()).mark_arc().encode(x="val:Q", color="cat:N").coord(fm.CoordFlip())
    with pytest.raises(ValueError, match=r"mark_arc requires a Polar coord"):
        chart.to_svg()


def test_arc_geo_coord_refuses_crosswise():
    """The crosswise case: CoordGeo (geoshape's own requirement) is still
    just "not Polar" from arc's point of view."""
    chart = (
        fm.Chart(_pie_df())
        .mark_arc()
        .encode(x="val:Q", color="cat:N")
        .coord(fm.CoordGeo(projection="equirectangular"))
    )
    with pytest.raises(ValueError, match=r"mark_arc requires a Polar coord"):
        chart.to_svg()


def test_arc_message_names_the_fix():
    chart = fm.Chart(_pie_df()).mark_arc().encode(x="val:Q", color="cat:N")
    with pytest.raises(ValueError) as exc_info:
        chart.to_svg()
    assert str(exc_info.value) == (
        "mark_arc requires a Polar coord; add .coord(fm.CoordPolar(...))"
    )


def test_arc_matching_coord_renders_without_error():
    chart = (
        fm.Chart(_pie_df())
        .mark_arc()
        .encode(x="val:Q", color="cat:N")
        .coord(fm.CoordPolar(theta="x"))
    )
    svg = chart.to_svg()
    assert "<svg" in svg


def test_arc_both_x_and_y_bound_under_non_polar_coord_refuses():
    """Every refusal test above binds only `x` — exactly the shape that
    pre-fix tripped the misleading `EncodingTypeMismatch` on the missing
    `y`, so those tests pin only the error-ordering half of the finding.
    With `y` also bound, scale resolution pre-fix had nothing missing to
    complain about and resolved cleanly, so this branch used to reach
    `mark_arc`'s own mark builder and silently render nothing (quality
    review round 1, Required Fix 2 — mirrors
    `scale_resolve::tests::arc_both_x_and_y_bound_under_non_polar_coord_refuses`
    on the Rust side)."""
    df = pl.DataFrame({"cat": ["A", "B", "C"], "val": [10.0, 20.0, 30.0], "val2": [1.0, 2.0, 3.0]})
    chart = fm.Chart(df).mark_arc().encode(x="val:Q", y="val2:Q", color="cat:N")
    with pytest.raises(ValueError, match=r"mark_arc requires a Polar coord"):
        chart.to_svg()


# ── mark_arc's merged-flat LayerChart hole (quality review round 1, ───────
# ── Required Fix 1) ────────────────────────────────────────────────────────
#
# `mark_arc`'s scale-resolution guard (`resolve_scales_with_leaf_context`)
# only fires when the RESOLVED spec's own mark is Arc. A merged-flat
# `LayerChart` layer (an independent-y layer, or the single-panel layout
# `.interactive()` builds) resolves scales against a DIFFERENT layer's mark,
# so the arc layer's own scale-resolution guard never runs — before
# `marks::arc::build` grew its own non-Polar arm, this silently rendered an
# arc layer that contributed no nodes instead of refusing.


def test_arc_merged_flat_layer_interactive_refuses():
    df = pl.DataFrame({"cat": ["A", "B", "C"], "val": [10.0, 20.0, 30.0]})
    bar = fm.Chart(df).mark_bar().encode(x="cat:N", y="val:Q")
    arc = fm.Chart(df).mark_arc().encode(x="val:Q", color="cat:N")
    with pytest.raises(ValueError, match=r"mark_arc requires a Polar coord"):
        fm.layer(bar, arc).interactive()


def test_arc_merged_flat_layer_resolve_independent_y_static_refuses():
    df = pl.DataFrame({"cat": ["A", "B", "C"], "val": [10.0, 20.0, 30.0]})
    bar = fm.Chart(df).mark_bar().encode(x="cat:N", y="val:Q")
    arc = fm.Chart(df).mark_arc().encode(x="val:Q", color="cat:N")
    with pytest.raises(ValueError, match=r"mark_arc requires a Polar coord"):
        fm.layer(bar, arc, resolve={"y": "independent"}).to_svg()


# ── mark_image's own Polar/Geo empty-scene behavior is untouched (#118) ──


def test_image_under_polar_coord_still_renders_its_documented_empty_scene():
    df = pl.DataFrame({"x": [1, 2, 3], "y": [1, 2, 3], "url": ["a", "b", "c"]})
    chart = (
        fm.Chart(df).mark_image().encode(x="x", y="y", url="url").coord(fm.CoordPolar(theta="x"))
    )
    svg = chart.to_svg()
    assert "<svg" in svg
