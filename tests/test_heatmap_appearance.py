"""Feature tests for ``ferrum.heatmap(vmin=, vmax=, center=, robust=, linewidths=,
linecolor=)`` (Batch A appearance-resolution, 2026-08-28/2026-09-01,
F-L09-06/F-L09-07, spec §4.2 amended 2026-09-01, T11 review).

``matrix.py``'s ``_heatmap_build`` previously emitted a color-scale ``domain``
only when *both* ``vmin`` and ``vmax`` were set (a carried finding), and
``center=`` never reached the wire as anything but an inert ``domainMid`` key
on a plain ``"linear"`` scale spec that Rust's Linear resolution path ignores
for midpoint purposes. This module pins:

- ``vmin=``/``vmax=`` working one-sided (the missing endpoint is filled from
  the PRE-mask data extent in Python, per the spec's "fill from data extent
  Python-side" contract -- matching ``robust=``'s existing pre-mask
  convention, pinned explicitly with a ``mask=`` case) as well as two-sided.
- ``center=`` emitting a real ``ScaleSpec::Diverging`` (``type: "diverging"``,
  ``domainMid: center``, ``scheme: cmap``) -- no new wire field, reusing the
  existing Diverging ``domainMid``.
- A ``center=`` outside the effective ``[vmin, vmax]`` domain rendering
  deterministically (a one-sided compressed ramp, never an error) while
  emitting a ``UserWarning`` naming the collapse; a ``center=`` inside the
  domain warns nothing.
- ``robust=`` continuing to compute percentile ``vmin``/``vmax`` in Python and
  now taking effect through the honored domain, including when combined with
  an explicit one-sided bound.
- ``linewidths=``/``linecolor=`` discriminating rendered SVG bytes now that
  ``mark_rect`` strokes render (F-L09-07), including named CSS colors now
  that Task 8's Rust color-parser swap has landed in this tree.
- An all-NaN value column: explicit both-sided ``vmin``/``vmax`` still emit
  as given; a one-sided bound that cannot be completed (no finite data to
  fill the other endpoint from) emits a ``UserWarning`` naming the reason and
  drops the color-domain override rather than silently discarding the bound.
- A +/-inf value never leaks into a one-sided fill or a ``robust=``
  percentile (both share ``_heatmap_finite_values``, which filters
  ``np.isfinite`` -- an S3 regression fix: it previously filtered only NaN,
  so an inf cell wrote JSON ``Infinity`` into the wire domain and crashed
  the Rust spec parser).
- A one-sided fill whose given bound falls on the far side of the data
  extent produces an inverted domain (e.g. ``vmin=50`` with a data max of
  10); it still renders deterministically, but the descending domain
  collapses every cell to a single uniform color (not a reversed ramp,
  per spec §4.2's flat-collapse note), and emits a ``UserWarning`` naming
  both endpoints. Pinned by exercising the actual rendered cell fills, not
  just the warning text.
- A non-finite USER-SUPPLIED ``vmin=``/``vmax=``/``center=`` (``inf``,
  ``-inf``, ``NaN``) is a typed ``ValueError`` naming the kwarg and value,
  raised in Python before any table work or wire emission -- the sibling of
  the inf-in-data fix above, closing a second S3 the cycle-3 quality review
  found unguarded: the endpoint a caller passes directly still reached
  ``scale_kwargs["domain"]``/``"domainMid"`` unchecked and crashed the Rust
  spec parser with an opaque ``ValueError: scale: expected value ...``.

Byte-identity for the absent case (no vmin/vmax/center/robust/linewidths/
linecolor overrides) is pinned separately.
"""

from __future__ import annotations

import json
import math
import re
import warnings

import polars as pl
import pytest

import ferrum as fr


def _wide_df() -> pl.DataFrame:
    """Asymmetric-extent numeric data so a center= shift is visually distinct
    from the data-derived natural midpoint (a symmetric extent would make
    center=0 collide with the unset-scale midpoint by coincidence)."""
    return pl.DataFrame({"a": [-2.0, 0.0, 10.0], "b": [1.0, -1.0, 5.0]})


def _color_scale_dict(chart: fr.Chart) -> dict:
    """Extract the resolved color channel's ``scale`` dict from ``to_json()``."""
    spec = json.loads(chart.to_json())
    return spec["encoding"]["color"]["scale"]


def _cell_fills(svg: str) -> list[str]:
    return re.findall(r'fill="(#[0-9a-fA-F]+)"', svg)


def _heatmap_cell_fills(svg: str) -> list[str]:
    """Extract only the heatmap cells' own fills, not every hex ``fill=`` in
    the SVG (background rect, colorbar stroke, axis/text colors, etc.).
    Heatmap cell ``<rect>`` elements are the only ones stroked with the
    default ``linecolor="white"``, so anchoring on that stroke isolates them."""
    return re.findall(r'<rect[^>]*fill="(#[0-9a-fA-F]+)"[^>]*stroke="#ffffff"', svg)


def _cell_strokes(svg: str) -> list[str]:
    return re.findall(r'stroke="(#[0-9a-fA-F]+)"', svg)


# ---------------------------------------------------------------------------
# Byte-identity for the absent case
# ---------------------------------------------------------------------------


def test_heatmap_no_appearance_kwargs_is_byte_stable():
    """Two heatmaps built with no vmin/vmax/center/robust/linewidths/linecolor
    overrides render identical SVG bytes -- the fix must not perturb the
    unchanged path."""
    df = _wide_df()
    a = fr.heatmap(df, annot=False).to_svg()
    b = fr.heatmap(df, annot=False).to_svg()
    assert a == b


def test_heatmap_default_scale_has_no_explicit_domain_or_diverging_type():
    """With none of vmin/vmax/center/cmap set, no color scale is emitted at all
    (the color channel stays scheme-default), matching pre-fix behavior."""
    df = _wide_df()
    chart = fr.heatmap(df, annot=False)
    spec = json.loads(chart.to_json())
    assert "scale" not in spec["encoding"]["color"]


# ---------------------------------------------------------------------------
# vmin / vmax -- two-sided (already Rust-honored) and one-sided (this fix)
# ---------------------------------------------------------------------------


def test_heatmap_vmin_vmax_two_sided_discriminates():
    df = _wide_df()
    a = fr.heatmap(df, annot=False, vmin=-10, vmax=10).to_svg()
    b = fr.heatmap(df, annot=False, vmin=-0.1, vmax=0.1).to_svg()
    assert a != b
    assert _cell_fills(a) != _cell_fills(b)


def test_heatmap_vmin_only_fills_vmax_from_data_extent():
    df = _wide_df()
    chart = fr.heatmap(df, annot=False, vmin=-1.0)
    scale = _color_scale_dict(chart)
    assert scale["type"] == "linear"
    # vmax filled from the data extent: max(-2, 0, 10, 1, -1, 5) == 10.0
    assert scale["domain"] == [-1.0, 10.0]


def test_heatmap_vmax_only_fills_vmin_from_data_extent():
    df = _wide_df()
    chart = fr.heatmap(df, annot=False, vmax=3.0)
    scale = _color_scale_dict(chart)
    assert scale["type"] == "linear"
    # vmin filled from the data extent: min(-2, 0, 10, 1, -1, 5) == -2.0
    assert scale["domain"] == [-2.0, 3.0]


def test_heatmap_vmin_only_discriminates_from_default():
    df = _wide_df()
    default = fr.heatmap(df, annot=False).to_svg()
    one_sided = fr.heatmap(df, annot=False, vmin=-1.0).to_svg()
    assert default != one_sided


def test_heatmap_vmax_only_discriminates_from_default():
    df = _wide_df()
    default = fr.heatmap(df, annot=False).to_svg()
    one_sided = fr.heatmap(df, annot=False, vmax=3.0).to_svg()
    assert default != one_sided


def test_heatmap_vmin_only_discriminates_by_value():
    """Two different one-sided vmin values must render distinct cell colors --
    a non-discriminating assertion would only prove *some* domain was set."""
    df = _wide_df()
    a = fr.heatmap(df, annot=False, vmin=-1.0).to_svg()
    b = fr.heatmap(df, annot=False, vmin=-50.0).to_svg()
    assert _cell_fills(a) != _cell_fills(b)


def test_heatmap_vmax_only_discriminates_by_value():
    """Two different one-sided vmax values must render distinct cell colors --
    the vmax-side counterpart of the vmin value-vs-value probe above."""
    df = _wide_df()
    a = fr.heatmap(df, annot=False, vmax=3.0).to_svg()
    b = fr.heatmap(df, annot=False, vmax=50.0).to_svg()
    assert _cell_fills(a) != _cell_fills(b)


def test_heatmap_vmin_only_with_inf_value_excludes_inf_from_fill():
    """A +/-inf cell must not leak into the filled vmax -- it would serialize
    as JSON Infinity and crash the Rust spec parser (S3 regression: the
    one-sided fill previously filtered only NaN, not inf, via
    ~np.isnan)."""
    df = pl.DataFrame({"a": [1.0, float("inf")], "b": [2.0, 3.0]})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        chart = fr.heatmap(df, annot=False, vmin=0.0)
    scale = _color_scale_dict(chart)
    # vmax filled from the finite extent only: max(1.0, 2.0, 3.0) == 3.0,
    # never +inf.
    assert scale["domain"] == [0.0, 3.0]
    assert all(math.isfinite(v) for v in scale["domain"])


def test_heatmap_vmax_only_with_negative_inf_excludes_inf_from_fill():
    df = pl.DataFrame({"a": [1.0, float("-inf")], "b": [2.0, 3.0]})
    chart = fr.heatmap(df, annot=False, vmax=5.0)
    scale = _color_scale_dict(chart)
    # vmin filled from the finite extent only: min(1.0, 2.0, 3.0) == 1.0,
    # never -inf.
    assert scale["domain"] == [1.0, 5.0]


def test_heatmap_robust_with_inf_value_excludes_inf_from_percentile():
    """robust='s percentile computation shares _heatmap_finite_values with the
    one-sided fill, so the same inf-exclusion fix applies there too (the S3
    fix's stated side effect)."""
    df = pl.DataFrame(
        {"a": [-2.0, 0.0, 2.0, float("inf")], "b": [1.0, -1.0, 0.5, float("-inf")]}
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        chart = fr.heatmap(df, annot=False, robust=True)
    scale = _color_scale_dict(chart)
    assert all(math.isfinite(v) for v in scale["domain"])


# ---------------------------------------------------------------------------
# One-sided fill producing an inverted domain -- warns, renders unchanged
# (spec §4.2, 2026-09-01 amendment, adjudicated in the quality-review cycle)
# ---------------------------------------------------------------------------


def test_heatmap_vmin_above_data_max_warns_inverted_domain():
    """vmin= given above the data max fills vmax from that (lower) max,
    producing an inverted [vmin, vmax] domain. Must warn naming both
    endpoints and the flat-collapse degradation (not a reversed ramp); the
    render is unchanged (not corrected) -- and the render actually
    collapses every cell to one uniform color, verified by exercising the
    rendered SVG rather than trusting the warning text alone (the earlier
    "renders reversed" claim was never checked against a live render)."""
    df = _wide_df()  # data extent is roughly [-2.0, 10.0]
    with pytest.warns(UserWarning, match=r"inverted domain \[50\.0, 10\.0\]"):
        chart = fr.heatmap(df, annot=False, vmin=50.0)
    scale = _color_scale_dict(chart)
    assert scale["domain"] == [50.0, 10.0]
    fills = _heatmap_cell_fills(chart.to_svg())
    assert len(fills) == 6
    assert len(set(fills)) == 1, f"expected all cells the same flat color, got {set(fills)}"


def test_heatmap_vmax_below_data_min_warns_inverted_domain():
    """Same flat-collapse degradation, mirrored on the vmax side -- render
    actually exercised, not just the warning substring."""
    df = _wide_df()
    with pytest.warns(UserWarning, match=r"inverted domain \[-2\.0, -50\.0\]"):
        chart = fr.heatmap(df, annot=False, vmax=-50.0)
    scale = _color_scale_dict(chart)
    assert scale["domain"] == [-2.0, -50.0]
    fills = _heatmap_cell_fills(chart.to_svg())
    assert len(fills) == 6
    assert len(set(fills)) == 1, f"expected all cells the same flat color, got {set(fills)}"


def test_heatmap_one_sided_fill_within_range_warns_nothing():
    """A one-sided bound that lands inside the data extent (the common case)
    must not trigger the inverted-domain warning."""
    df = _wide_df()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fr.heatmap(df, annot=False, vmin=-1.0)
        fr.heatmap(df, annot=False, vmax=3.0)


def test_heatmap_one_sided_vmin_fill_uses_pre_mask_extent():
    """The one-sided fill reads the PRE-mask data extent, matching robust='s
    existing convention (spec §4.2, 2026-09-01 amendment) -- a cell hidden by
    mask= still contributes to the filled endpoint. This is deliberate, not
    an oversight: pin it explicitly rather than leaving it as an accident of
    implementation order.

    The fixture must put its extreme value somewhere mask="lower" actually
    HIDES, or pre-mask and post-mask extents coincide and the test cannot
    discriminate the two semantics. mask="lower" keeps row_idx >= col_idx;
    the 999.0 sits at (row 0, col "c" -> col index 2), and 0 >= 2 is False,
    so that cell is hidden. Every kept cell is 0.0 (post-mask extent
    collapses to (0.0, 0.0)), while the pre-mask extent is (0.0, 999.0) --
    the emitted domain must stay [0.0, 999.0], proving the fill reads the
    full pre-mask table rather than the post-mask rows actually rendered.
    """
    df = pl.DataFrame(
        {
            "id": ["r0", "r1", "r2"],
            "a": [0.0, 0.0, 0.0],
            "b": [0.0, 0.0, 0.0],
            "c": [999.0, 0.0, 0.0],
        }
    )
    chart = fr.heatmap(df, annot=False, mask="lower", vmin=0.0)
    scale = _color_scale_dict(chart)
    assert scale["domain"] == [0.0, 999.0]


# ---------------------------------------------------------------------------
# center -- Diverging scale emission
# ---------------------------------------------------------------------------


def test_heatmap_center_emits_diverging_scale_with_domain_mid():
    df = _wide_df()
    chart = fr.heatmap(df, annot=False, center=2.5)
    scale = _color_scale_dict(chart)
    assert scale["type"] == "diverging"
    assert scale["domainMid"] == 2.5
    # No vmin/vmax given -> no explicit domain, only the midpoint.
    assert "domain" not in scale


def test_heatmap_center_with_cmap_sets_diverging_scheme():
    df = _wide_df()
    chart = fr.heatmap(df, annot=False, cmap="rdbu", center=0.0)
    scale = _color_scale_dict(chart)
    assert scale["type"] == "diverging"
    assert scale["scheme"] == "rdbu"
    assert scale["domainMid"] == 0.0


def test_heatmap_center_with_one_sided_vmin_fills_domain_and_keeps_mid():
    df = _wide_df()
    chart = fr.heatmap(df, annot=False, center=0.0, vmin=-1.0)
    scale = _color_scale_dict(chart)
    assert scale["type"] == "diverging"
    assert scale["domainMid"] == 0.0
    assert scale["domain"] == [-1.0, 10.0]


def test_heatmap_no_new_wire_field_for_center():
    """center= must reuse the existing Diverging domainMid field -- no novel
    'center' key reaches the wire spec."""
    df = _wide_df()
    chart = fr.heatmap(df, annot=False, center=1.0)
    scale = _color_scale_dict(chart)
    assert "center" not in scale


def test_heatmap_center_discriminates_by_value():
    df = _wide_df()
    a = fr.heatmap(df, annot=False, cmap="rdbu", center=0.0).to_svg()
    b = fr.heatmap(df, annot=False, cmap="rdbu", center=5.0).to_svg()
    assert a != b
    assert _cell_fills(a) != _cell_fills(b)


# ---------------------------------------------------------------------------
# center outside the effective [vmin, vmax] domain -- warns, still renders
# deterministically (spec §4.2, 2026-09-01 amendment)
# ---------------------------------------------------------------------------


def test_heatmap_center_outside_explicit_domain_warns():
    df = _wide_df()
    with pytest.warns(UserWarning, match="center=50.*outside.*effective color domain"):
        chart = fr.heatmap(df, annot=False, vmin=-1.0, vmax=3.0, center=50)
    # Still renders deterministically -- no error, and the scale is emitted.
    scale = _color_scale_dict(chart)
    assert scale["type"] == "diverging"
    assert scale["domainMid"] == 50


def test_heatmap_center_inside_explicit_domain_warns_nothing():
    df = _wide_df()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fr.heatmap(df, annot=False, vmin=-1.0, vmax=3.0, center=1.0)


def test_heatmap_center_outside_data_derived_domain_warns():
    """No explicit vmin/vmax at all -- the warning check still compares
    center= against the data-derived extent, not just an explicit domain."""
    df = _wide_df()  # data extent spans roughly [-2.0, 10.0]
    with pytest.warns(UserWarning, match="center=100"):
        fr.heatmap(df, annot=False, center=100.0)


def test_heatmap_center_outside_domain_still_renders_one_sided_ramp():
    """The out-of-domain case is a defined, deterministic degradation (a
    one-sided compressed ramp), never an error and never silently absent."""
    df = _wide_df()
    with pytest.warns(UserWarning):
        svg = fr.heatmap(df, annot=False, cmap="rdbu", vmin=-1.0, vmax=3.0, center=50).to_svg()
    assert svg  # renders successfully, no exception


# ---------------------------------------------------------------------------
# robust -- percentile-derived domain, alone and combined with an explicit
# one-sided bound
# ---------------------------------------------------------------------------


def test_heatmap_robust_fills_both_bounds_from_percentiles():
    df = pl.DataFrame({"a": [-2.0, 0.0, 2.0, 100.0], "b": [1.0, -1.0, 0.5, -100.0]})
    chart = fr.heatmap(df, annot=False, robust=True)
    scale = _color_scale_dict(chart)
    assert scale["type"] == "linear"
    lo, hi = scale["domain"]
    # 2nd/98th percentile of the 8-value sample must clip well inside the
    # raw [-100, 100] extent.
    assert -100.0 < lo < -2.0
    assert 2.0 < hi < 100.0


def test_heatmap_robust_discriminates_from_default():
    df = pl.DataFrame({"a": [-2.0, 0.0, 2.0, 100.0], "b": [1.0, -1.0, 0.5, -100.0]})
    default = fr.heatmap(df, annot=False).to_svg()
    robust = fr.heatmap(df, annot=False, robust=True).to_svg()
    assert default != robust


def test_heatmap_robust_with_explicit_vmin_only_fills_vmax_by_percentile():
    """robust= combined with an explicit vmin leaves vmin untouched and fills
    only vmax (via the existing percentile path, not the data-extent fill)."""
    df = pl.DataFrame({"a": [-2.0, 0.0, 2.0, 100.0], "b": [1.0, -1.0, 0.5, -100.0]})
    chart = fr.heatmap(df, annot=False, robust=True, vmin=-5.0)
    scale = _color_scale_dict(chart)
    lo, hi = scale["domain"]
    assert lo == -5.0
    # Percentile-derived vmax must clip well inside the raw 100.0 max.
    assert 2.0 < hi < 100.0


# ---------------------------------------------------------------------------
# linewidths / linecolor -- cell-stroke rendering (F-L09-07)
# ---------------------------------------------------------------------------


def test_heatmap_linewidths_discriminates():
    df = pl.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    thin = fr.heatmap(df, annot=False, linewidths=0.5, linecolor="#000000").to_svg()
    thick = fr.heatmap(df, annot=False, linewidths=5.0, linecolor="#000000").to_svg()
    assert thin != thick
    assert 'stroke-width="0.5"' in thin
    assert 'stroke-width="5"' in thick


def test_heatmap_linecolor_discriminates():
    df = pl.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    black = fr.heatmap(df, annot=False, linewidths=2.0, linecolor="#000000").to_svg()
    red = fr.heatmap(df, annot=False, linewidths=2.0, linecolor="#ff0000").to_svg()
    assert black != red
    assert "#000000" in _cell_strokes(black)
    assert "#ff0000" in _cell_strokes(red)


def test_heatmap_linewidths_zero_disables_border():
    """linewidths=0 must omit the stroke entirely, not merely set width 0."""
    df = pl.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    chart = fr.heatmap(df, annot=False, linewidths=0, linecolor="#000000")
    spec = json.loads(chart.to_json())
    assert "stroke" not in spec.get("mark_style", {})


@pytest.mark.parametrize(
    ("linecolor", "expected_hex"),
    [("black", "#000000"), ("white", "#ffffff"), ("red", "#ff0000")],
)
def test_heatmap_named_linecolor_renders_expected_stroke(linecolor, expected_hex):
    """Named CSS colors now render the expected cell stroke (Task 8's Rust
    color-parser swap has landed in this tree)."""
    df = pl.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    svg = fr.heatmap(df, annot=False, linewidths=1.0, linecolor=linecolor).to_svg()
    assert expected_hex in _cell_strokes(svg)


# ---------------------------------------------------------------------------
# Edge cases -- all-NaN value data
# ---------------------------------------------------------------------------


def test_heatmap_vmin_only_with_one_all_nan_column_still_fills_from_other_column():
    """One value column is entirely NaN, but another column ("b") still has
    finite values -- the table-wide extent is NOT empty, so the one-sided
    fill succeeds normally and no warning fires. (Renamed from the previous
    "...all_nan_column_does_not_crash" name/docstring, which claimed "the
    data extent is entirely NaN" while asserting the fill DID go through --
    a self-contradiction; the true all-NaN cases are covered separately
    below.)"""
    df = pl.DataFrame({"a": [float("nan"), float("nan")], "b": [1.0, 2.0]})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        chart = fr.heatmap(df, annot=False, vmin=-1.0)
    scale = _color_scale_dict(chart)
    assert scale["domain"] == [-1.0, 2.0]


def test_heatmap_vmin_only_with_true_all_nan_data_warns_and_drops_scale():
    """Every value column is entirely NaN: the one-sided vmin= cannot be
    completed (no finite value anywhere to fill vmax from). Must warn naming
    the reason and drop the color-domain override -- never silently discard
    the user's explicit vmin=."""
    df = pl.DataFrame({"a": [float("nan"), float("nan")], "b": [float("nan"), float("nan")]})
    with pytest.warns(UserWarning, match="vmin=-1.*vmax.*could not be filled"):
        chart = fr.heatmap(df, annot=False, vmin=-1.0)
    spec = json.loads(chart.to_json())
    assert "scale" not in spec["encoding"]["color"]


def test_heatmap_vmax_only_with_true_all_nan_data_warns_and_drops_scale():
    df = pl.DataFrame({"a": [float("nan"), float("nan")], "b": [float("nan"), float("nan")]})
    with pytest.warns(UserWarning, match="vmax=3.*vmin.*could not be filled"):
        chart = fr.heatmap(df, annot=False, vmax=3.0)
    spec = json.loads(chart.to_json())
    assert "scale" not in spec["encoding"]["color"]


def test_heatmap_two_sided_vmin_vmax_with_all_nan_data_emits_domain_as_given():
    """Explicit BOTH-sided vmin/vmax bounds always emit as given, even when
    the underlying data is entirely NaN -- there is nothing to fill, so
    there is nothing to warn about."""
    df = pl.DataFrame({"a": [float("nan"), float("nan")], "b": [float("nan"), float("nan")]})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        chart = fr.heatmap(df, annot=False, vmin=-1.0, vmax=1.0)
    scale = _color_scale_dict(chart)
    assert scale["domain"] == [-1.0, 1.0]


def test_heatmap_vmin_only_all_nan_does_not_crash_and_still_renders():
    """The dropped-scale degradation must not raise -- the heatmap still
    renders (with the theme-default color scheme, since the override was
    dropped)."""
    df = pl.DataFrame({"a": [float("nan"), float("nan")], "b": [float("nan"), float("nan")]})
    with pytest.warns(UserWarning):
        svg = fr.heatmap(df, annot=False, vmin=-1.0).to_svg()
    assert svg


# ---------------------------------------------------------------------------
# Non-finite USER-SUPPLIED vmin/vmax/center -- typed refusal, not a crash
# (spec §4.2, 2026-09-01 amendment, T11 quality-review cycle 3 S3)
#
# Distinct from the inf-in-DATA tests above: those pin that a non-finite
# value *derived from the data* (via the one-sided fill or robust=) is
# filtered out before it can reach the wire. These pin the sibling case the
# cycle-3 review found unguarded -- a non-finite value the caller passes
# *directly* as vmin=/vmax=/center= reached scale_kwargs["domain"]/
# "domainMid"] unchecked and serialized as JSON Infinity/-Infinity/NaN,
# which the Rust spec parser rejected with an opaque
# `ValueError: scale: expected value at line 1 column N` naming neither the
# kwarg nor the reason. Pre-task (before the one-sided fill existed) this
# input was silently inert instead; post-fix it is a typed refusal raised in
# Python before any table work or wire emission.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwarg", "value"),
    [
        ("vmin", float("inf")),
        ("vmin", float("-inf")),
        ("vmin", float("nan")),
        ("vmax", float("inf")),
        ("vmax", float("-inf")),
        ("vmax", float("nan")),
        ("center", float("nan")),
        ("center", float("inf")),
        ("center", float("-inf")),
    ],
)
def test_heatmap_non_finite_kwarg_raises_value_error_naming_kwarg_and_value(kwarg, value):
    """Every guarded kwarg, every non-finite flavor (inf/-inf/nan): raises a
    ValueError whose message names both the offending kwarg and its value,
    not the opaque Rust serde error."""
    df = _wide_df()
    with pytest.raises(ValueError, match=rf"heatmap: {kwarg}={value!r} must be finite"):
        fr.heatmap(df, annot=False, **{kwarg: value})


def test_heatmap_two_sided_inf_bounds_raises_value_error():
    """Both vmin and vmax non-finite at once (the reviewer's named sibling
    crash) also raises a typed refusal rather than reaching the wire."""
    df = _wide_df()
    with pytest.raises(ValueError, match=r"heatmap: vmin=inf must be finite"):
        fr.heatmap(df, annot=False, vmin=float("inf"), vmax=float("-inf"))


def test_heatmap_robust_with_non_finite_vmin_still_raises():
    """robust=True never overrides an explicitly-given bound, so a
    non-finite vmin= combined with robust= must still be refused -- it must
    not be silently overwritten by the percentile fill."""
    df = _wide_df()
    with pytest.raises(ValueError, match=r"heatmap: vmin=inf must be finite"):
        fr.heatmap(df, annot=False, vmin=float("inf"), robust=True)


def test_heatmap_non_finite_vmin_raises_before_touching_data(monkeypatch):
    """The finiteness guard is the very first thing ``heatmap`` does after
    its local imports -- before ``to_arrow_table`` or any other table/wire
    work. Patch ``to_arrow_table`` to explode if reached; if the guard were
    ever moved past it, this test would fail with the patched
    ``AssertionError`` instead of the expected ``ValueError``."""
    import ferrum._coerce as _coerce

    def _boom(*_args, **_kwargs):
        raise AssertionError("to_arrow_table should not be reached")

    monkeypatch.setattr(_coerce, "to_arrow_table", _boom)
    df = _wide_df()
    with pytest.raises(ValueError, match=r"heatmap: vmin=inf must be finite"):
        fr.heatmap(df, annot=False, vmin=float("inf"))


def test_heatmap_finite_vmin_vmax_center_unaffected_by_guard():
    """Sanity control: finite vmin/vmax/center are unaffected by the new
    guard and still render normally."""
    df = _wide_df()
    svg = fr.heatmap(df, annot=False, vmin=-5.0, vmax=5.0, center=0.0).to_svg()
    assert svg
