"""Smoke tests for heavy-stat marks (Phase 8b Sub-batch F).

Each mark gets:
- A spec-build smoke test (assert spec.layers is not None or appropriate shape)
- Key kwarg propagation test (verify kwarg flows into spec JSON)
- A render smoke test (to_svg returns valid SVG) where supported
- An error-case test where applicable
"""

import polars as pl
import pyarrow as pa  # noqa: F401
import numpy as np
import pytest
import ferrum as fe


@pytest.fixture
def df_xy():
    return pl.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 1.5, 2.5, 3.5, 4.5, 1.2],
            "y": [1.0, 2.0, 3.0, 2.5, 1.5, 2.0, 3.0, 2.5, 1.5, 2.5],
        }
    )


@pytest.fixture
def df_xyc():
    return pl.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "y": [1.0, 2.0, 3.0, 2.5, 1.5],
            "color_value": [10.0, 20.0, 15.0, 25.0, 5.0],
        }
    )


# ---- mark_contour ----


def test_contour_smoke(df_xy):
    spec = fe.Chart(df_xy).mark_contour().encode(x="x", y="y")._build_spec()
    assert spec.layers is not None
    assert len(spec.layers) == 1


def test_contour_thresholds_propagates(df_xy):
    spec = fe.Chart(df_xy).mark_contour(thresholds=10).encode(x="x", y="y")._build_spec()
    json_str = spec.to_json()
    assert '"thresholds":10' in json_str


def test_contour_fill_mode(df_xy):
    spec = fe.Chart(df_xy).mark_contour(fill=True).encode(x="x", y="y")._build_spec()
    json_str = spec.to_json()
    assert '"fill":true' in json_str


# ---- mark_violin ----


def test_violin_smoke(df_xy):
    spec = fe.Chart(df_xy).mark_violin(inner=None).encode(x="x", y="y")._build_spec()
    assert spec.layers is not None
    assert len(spec.layers) == 1


def test_violin_inner_box_adds_layers(df_xy):
    spec = fe.Chart(df_xy).mark_violin(inner="box").encode(x="x", y="y")._build_spec()
    assert len(spec.layers) > 1  # violin polygon + box layers


def test_violin_inner_quartile_3_rules(df_xy):
    spec = fe.Chart(df_xy).mark_violin(inner="quartile").encode(x="x", y="y")._build_spec()
    assert len(spec.layers) == 4  # violin + 3 quartile rules


def test_violin_invalid_inner_raises(df_xy):
    with pytest.raises(ValueError, match="inner"):
        fe.Chart(df_xy).mark_violin(inner="bogus").encode(x="x", y="y")._build_spec()


def test_violin_shared_extent_kwarg_propagates_to_transform():
    """R6: mark_violin(shared_extent=True) serializes shared_extent into the Violin transform."""
    df = pl.DataFrame(
        {
            "grp": ["a"] * 20 + ["b"] * 20,
            "val": list(range(20)) + list(range(100, 120)),
        }
    )
    spec = fe.Chart(df).mark_violin(inner=None, shared_extent=True).encode(
        x="grp", y="val"
    )._build_spec()
    json_str = spec.to_json()
    # shared_extent=true must appear in the serialized spec.
    assert '"shared_extent":true' in json_str, (
        "mark_violin(shared_extent=True) must emit shared_extent in the transform JSON"
    )


def test_violin_shared_extent_false_is_default():
    """R6: mark_violin() with no shared_extent kwarg defaults to shared_extent=false."""
    df = pl.DataFrame({"grp": ["a"] * 10 + ["b"] * 10, "val": list(range(20))})
    spec = fe.Chart(df).mark_violin(inner=None).encode(x="grp", y="val")._build_spec()
    json_str = spec.to_json()
    # Default must NOT emit shared_extent (it is skipped when false per serde default).
    assert '"shared_extent":true' not in json_str, (
        "mark_violin() without shared_extent must not emit shared_extent:true in transform JSON"
    )


def test_violin_shared_extent_produces_shared_value_range():
    """R6: mark_violin(shared_extent=True) renders groups on a shared value range.

    Groups 'a' ([0,9]) and 'b' ([100,109]) have disjoint per-group ranges.
    With shared_extent=True, the SVG must contain y-coordinate values
    spanning the cross-group range in both groups' polygon paths.
    """
    df = pl.DataFrame(
        {
            "grp": ["a"] * 10 + ["b"] * 10,
            "val": list(range(10)) + list(range(100, 110)),
        }
    )
    # shared_extent=True: both groups share the full [0, 109] y range.
    svg_shared = (
        fe.Chart(df)
        .mark_violin(inner=None, shared_extent=True)
        .encode(x="grp", y="val")
        .to_svg()
    )
    # shared_extent=False (default): each group uses its own y range.
    svg_per = (
        fe.Chart(df)
        .mark_violin(inner=None, shared_extent=False)
        .encode(x="grp", y="val")
        .to_svg()
    )
    # Both must produce non-empty SVG.
    assert "<svg" in svg_shared
    assert "<svg" in svg_per
    # The shared-extent render must produce different output than per-group
    # (shared forces the same y-grid, per-group gives narrower ranges per group).
    assert svg_shared != svg_per, (
        "shared_extent=True and shared_extent=False must produce different SVG "
        "for groups with disjoint value ranges"
    )


# ---- mark_qq ----


def test_qq_smoke():
    df = pl.DataFrame({"v": [1.0, 2.0, 3.0, 4.0, 5.0]})
    spec = fe.Chart(df).mark_qq().encode(x="v")._build_spec()
    assert len(spec.layers) == 2  # point + line


def test_qq_no_line():
    df = pl.DataFrame({"v": [1.0, 2.0, 3.0, 4.0, 5.0]})
    spec = fe.Chart(df).mark_qq(line=False).encode(x="v")._build_spec()
    assert len(spec.layers) == 1


def test_qq_invalid_distribution():
    df = pl.DataFrame({"v": [1.0, 2.0]})
    with pytest.raises(ValueError, match="distribution"):
        fe.Chart(df).mark_qq(distribution="bogus").encode(x="v")._build_spec()


# ---- mark_raster ----


def test_raster_smoke(df_xy):
    spec = fe.Chart(df_xy).mark_raster().encode(x="x", y="y")._build_spec()
    assert spec.layers is not None
    assert len(spec.layers) == 1


def test_raster_aggregate_mean_requires_field(df_xyc):
    with pytest.raises(ValueError, match="field"):
        fe.Chart(df_xyc).mark_raster(aggregate="mean").encode(x="x", y="y")._build_spec()


def test_raster_resolution_int(df_xy):
    spec = fe.Chart(df_xy).mark_raster(resolution=64).encode(x="x", y="y")._build_spec()
    json_str = spec.to_json()
    assert "64" in json_str


# ---- mark_hex ----


def test_hex_smoke(df_xy):
    spec = fe.Chart(df_xy).mark_hex().encode(x="x", y="y")._build_spec()
    assert spec.layers is not None
    assert len(spec.layers) == 1


def test_hex_aggregate_mean_requires_field(df_xyc):
    with pytest.raises(ValueError, match="field"):
        fe.Chart(df_xyc).mark_hex(aggregate="mean").encode(x="x", y="y")._build_spec()


def test_hex_bin_size_propagates(df_xy):
    spec = fe.Chart(df_xy).mark_hex(bin_size=0.5).encode(x="x", y="y")._build_spec()
    json_str = spec.to_json()
    assert '"bin_size":0.5' in json_str


# ---- mark_hex stroke tests (item 17) ----


def test_hex_stroke_and_width_no_longer_raises(df_xy):
    """Passing stroke= and stroke_width= no longer raises ValueError."""
    # Both together — use full 6-digit hex (from_hex_str does not support 3-digit shorthand)
    spec = (
        fe.Chart(df_xy)
        .mark_hex(stroke="#ffffff", stroke_width=1)
        .encode(x="x", y="y")
        ._build_spec()
    )
    assert spec.layers is not None
    # stroke alone
    spec2 = fe.Chart(df_xy).mark_hex(stroke="#ffffff").encode(x="x", y="y")._build_spec()
    assert spec2.layers is not None
    # stroke_width alone (non-zero)
    spec3 = fe.Chart(df_xy).mark_hex(stroke_width=2).encode(x="x", y="y")._build_spec()
    assert spec3.layers is not None


def test_hex_stroke_with_width_renders_border_attributes(df_xy):
    """mark_hex(stroke="#ffffff", stroke_width=1) emits stroke/stroke-width on hex polygons.

    Note: stroke color must be a full 6-digit hex. The Rust color parser (from_hex_str)
    handles only 6- and 8-digit hex codes; 3-digit shorthand (e.g. #fff) is silently
    dropped. This is a pre-existing constraint in the renderer, not a limitation of the
    hex-stroke wiring.
    """
    svg = fe.Chart(df_xy).mark_hex(stroke="#ffffff", stroke_width=1).encode(x="x", y="y").to_svg()
    assert "<svg" in svg
    # The polygon <path> elements should carry stroke="..." (white = #ffffff)
    assert 'stroke="#ffffff"' in svg
    # stroke-width attribute is emitted for nonzero widths
    assert 'stroke-width="1"' in svg


def test_hex_stroke_alone_width_zero_renders_no_visible_border(df_xy):
    """mark_hex(stroke="#ffffff") alone (stroke_width=0 default) renders with no stroke-width.

    Per the locked literal semantics (spec §8): a stroke color with stroke_width left at
    its 0 default produces no visible border. The Rust SVG writer only emits stroke-width
    when the value is nonzero, so no stroke-width attribute appears on the hex polygons.
    """
    import re

    svg = fe.Chart(df_xy).mark_hex(stroke="#ffffff").encode(x="x", y="y").to_svg()
    assert "<svg" in svg
    # The polygon paths carry stroke="#ffffff" but no stroke-width (0 = invisible).
    assert 'stroke="#ffffff"' in svg
    # Confirm no nonzero stroke-width on polygon paths. The SVG may have stroke-width
    # from axis/colorbar chrome, but not from the hex polygon paths themselves.
    # Find path elements with stroke="#ffffff" and assert they have no stroke-width.
    poly_blocks = re.findall(r'<path\b[^>]*stroke="#ffffff"[^>]*/>', svg)
    for block in poly_blocks:
        assert "stroke-width" not in block, (
            f"Expected no stroke-width on zero-width hex polygon but found: {block!r}"
        )


# ---- mark_swarm ----


def test_swarm_smoke(df_xy):
    spec = fe.Chart(df_xy).mark_swarm().encode(x="x", y="y")._build_spec()
    assert spec.layers is not None
    assert len(spec.layers) == 1


def test_swarm_orient_horizontal(df_xy):
    spec = fe.Chart(df_xy).mark_swarm(orient="horizontal").encode(x="x", y="y")._build_spec()
    json_str = spec.to_json()
    assert "swarm" in json_str


def test_swarm_side_left(df_xy):
    spec = fe.Chart(df_xy).mark_swarm(side="left").encode(x="x", y="y")._build_spec()
    json_str = spec.to_json()
    assert '"side":"left"' in json_str


# ---- mark_function ----


def test_function_explicit_domain():
    df = pl.DataFrame({"x": pl.Series([], dtype=pl.Float64), "y": pl.Series([], dtype=pl.Float64)})
    chart = fe.Chart(df).mark_function(lambda x: x**2, domain=(0, 5), n=50)
    spec = chart._build_spec()
    json_str = spec.to_json()
    assert '"line"' in json_str


def test_function_inferred_domain(df_xy):
    chart = fe.Chart(df_xy).encode(x="x", y="y").mark_function(lambda x: x * 2)
    spec = chart._build_spec()
    json_str = spec.to_json()
    assert '"line"' in json_str


def test_function_missing_domain_raises():
    df = pl.DataFrame({"x": pl.Series([], dtype=pl.Float64), "y": pl.Series([], dtype=pl.Float64)})
    with pytest.raises(ValueError, match="domain"):
        fe.Chart(df).mark_function(lambda x: x**2)


def test_function_wrong_shape_raises():
    df = pl.DataFrame({"x": pl.Series([], dtype=pl.Float64), "y": pl.Series([], dtype=pl.Float64)})
    with pytest.raises(ValueError, match="shape"):
        fe.Chart(df).mark_function(lambda x: 42, domain=(0, 5), n=50)


# ---- Smoke renders (skip if engine doesn't support yet) ----


def test_violin_no_inner_renders(df_xy):
    """Smoke render -- violin polygon only (no inner box/quartile)."""
    svg = fe.Chart(df_xy).mark_violin(inner=None).encode(x="x", y="y").to_svg()
    assert "<svg" in svg


def test_qq_renders():
    df = pl.DataFrame({"v": [1.0, 2.0, 3.0, 4.0, 5.0, 1.5, 2.5, 3.5, 4.5, 1.2]})
    svg = fe.Chart(df).mark_qq(line=False).encode(x="v").to_svg()
    assert "<svg" in svg


def test_swarm_renders():
    """Smoke render -- swarm (point mark on transformed coords)."""
    df_cat = pl.DataFrame(
        {
            "g": ["a"] * 5 + ["b"] * 5,
            "v": [1.0, 2.0, 3.0, 4.0, 5.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    svg = fe.Chart(df_cat).mark_swarm().encode(x="g", y="v").to_svg()
    assert "<svg" in svg


def test_function_renders(df_xy):
    chart = fe.Chart(df_xy).encode(x="x", y="y").mark_function(lambda x: x**2)
    svg = chart.to_svg()
    assert "<svg" in svg


# --- Phase 8b Task 35: bivariate density routes through mark_contour ---


def test_bivariate_density_routes_through_contour():
    """When .encode() binds both x and y, mark_density() emits a 2D KDE +
    contour-fill layered spec (Phase 8b), not the 1D area+Kde path."""
    rng = np.random.default_rng(0)
    df = pl.DataFrame({"x": rng.standard_normal(50), "y": rng.standard_normal(50)})
    spec = fe.Chart(df).mark_density().encode(x="x", y="y")._build_spec()
    json_str = spec.to_json()
    # 2D KDE transform (Kde2D) and contour transform should both appear in the
    # serialized spec, regardless of exact casing/format of their tags. The
    # serde snake_case mangling on `Bin2D`/`Kde2D` yields `bin_2_d`/`kde2_d`
    # — see ferrum-phase-9 commit 041c528 for the explicit serde rename
    # applied to `Bin2D`; `Kde2D` is unaffected here because callers don't
    # need to match against a fixed string.
    assert any(t in json_str.lower() for t in ("kde_2d", "kde2d", "kde2_d"))
    assert "contour" in json_str.lower()
