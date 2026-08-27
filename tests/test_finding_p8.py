"""Regression tests — Finding P8 (2026-08-27 design-review remediation batch).

P8: ``_spec_build.py``'s ``_resolve_polar_remapping`` used to synthesize a
phantom positional channel for theta-only ``mark_arc`` charts under
``CoordPolar`` (``enc["y"] = enc["x"]`` when ``theta="x"``, mirrored for
``theta="y"``) so Rust's ``scale_resolve`` — which unconditionally required
both x and y — wouldn't error. That dummy channel leaked into
``Chart.to_dict()`` (a phantom ``y``/``x`` bound to the theta field the user
never encoded) and, because ``build_nominal_theta`` in ``arc.rs`` reads the
dummy as the radius column, a numeric-but-ordinal theta silently drove wedge
radius off the theta values (latent aliasing bug).

The companion Rust task (Task 11, committed) added a ``Mark::Arc`` arm to
``render::scale_resolve::resolve_scales_with_leaf_context``'s single-axis
exemption block (mirroring the existing ``Tick``/``Rule`` single-axis
exemption): a theta-only arc under ``CoordPolar`` now resolves with a real
scale for the theta axis and a ``dummy_unit_scale`` (``[0, 1]`` domain) for
the absent radius axis, so ``ResolvedScales`` stays fully populated without
Python needing to fake a channel. This task deletes the now-unnecessary
Python-side synthesis.

Verified regressions:
  (a) ``to_dict()`` publishes only the user's own channels — no phantom
      ``y``/``x`` — for both theta directions.
  (b) Pie/donut wedge geometry (path count, shared outer radius, absence of
      a rendered y-axis) is unaffected structurally by the fix.
  (c) A numeric-but-ordinal theta (``:O`` on a numeric column) no longer
      aliases the theta values into radius; every wedge reaches the coord's
      full outer radius, independently verified against a nominal
      string-typed theta (which can never alias, since a string column
      cannot cast through ``col_as_f64``).

Root cause of the one genuinely surprising side effect (kept for context,
not as an open question — see the settled outcome below): removing the
dummy channel makes theta-only ``mark_arc`` charts single-axis-eligible in
Rust's scale-resolve for the first time, so the absent radius axis now
resolves through ``dummy_unit_scale`` (domain ``[0, 1]``) instead of a real
quantitative scale copied from theta's own domain. Axis-margin reservation
(``render::prepare::build_axes``, consumed by ``layout::compute_layout``)
sizes the panel's margin from whatever ``ScaleKind`` the absent axis
resolves to, even under ``CoordPolar`` where that axis is never actually
drawn (suppressed later, in ``scene_build.rs::route_panel_axes_and_grid``).
Pre-fix, the dummy's stolen domain happened to reserve a realistic
2-digit-label margin; post-fix, the panel now reserves the SAME margin an
existing single-axis ``mark_tick(x=...)``/``mark_tick(y=...)`` chart already
gets (verified directly below) — the architecturally correct outcome, since
Arc now genuinely shares the Tick/Rule single-axis convention. This moves
every arc's absolute pixel coordinates (center, radii, circular theta-tick
positions) while leaving the wedges' *proportions* (sweep angles, relative
radii) and the full tag/text content unaffected — verified structurally
below rather than via an unblessed byte pin.
"""

from __future__ import annotations

import math
import re

import polars as pl
import pytest

import ferrum as fm


# ---------------------------------------------------------------------------
# SVG geometry helpers (same idiom as tests/test_flexibility_caps/test_fa1_arc_coxcomb.py)
# ---------------------------------------------------------------------------


def _filled_arc_paths(svg: str) -> list[str]:
    """Return ``d=`` values for ``<path>`` elements with arc commands and non-none fill."""
    result = []
    for attrs in re.findall(r"<path([^>]+)>", svg):
        d_m = re.search(r'd="([^"]+)"', attrs)
        if d_m is None:
            continue
        d = d_m.group(1)
        if "A" not in d:
            continue
        fill_m = re.search(r'fill="([^"]+)"', attrs)
        if fill_m is None or fill_m.group(1) == "none":
            continue
        result.append(d)
    return result


def _outer_radius_from_path(path_d: str) -> float:
    """Return the outer arc radius from the first ``A`` command in a path."""
    m = re.search(r"A\s*([\d.]+)", path_d)
    return float(m.group(1)) if m else math.nan


def _inner_outer_radii(path_d: str) -> tuple[float, float]:
    """Return ``(inner_radius, outer_radius)`` from a wedge path.

    A solid wedge (no donut hole) has one ``A`` command (outer arc only);
    inner radius is 0. An annular/donut wedge has two: outer arc first, then
    inner arc.
    """
    arc_radii = [float(m.group(1)) for m in re.finditer(r"A\s*([\d.]+)", path_d)]
    if len(arc_radii) == 0:
        raise ValueError(f"no Arc command in path: {path_d[:80]!r}")
    if len(arc_radii) == 1:
        return 0.0, arc_radii[0]
    return arc_radii[1], arc_radii[0]


def _arc_sweep_radians(path_d: str) -> float:
    """Compute a wedge path's outer-arc sweep angle from its chord length.

    ``chord = 2r sin(theta/2)`` is symmetric under ``theta -> tau - theta``
    (``sin(pi - x) == sin(x)``), so the chord alone cannot distinguish a
    minor sweep (``<= pi``) from its major/reflex complement (``> pi``). The
    SVG large-arc-flag (the 4th numeric field in the ``A`` command) resolves
    that ambiguity directly: ``0`` means the drawn sweep is the minor
    (``<= pi``) angle the chord-inversion formula returns as-is; ``1`` means
    it is the major (``> pi``) complement, ``tau`` minus the naive value.
    Callers relying on this for angles > pi (as this module's >180-degree
    test below does) must NOT strip the large-arc-flag digit out of the
    regex, or this disambiguation silently breaks.
    """
    m_match = re.match(r"M\s*([\d.]+)\s+([\d.]+)", path_d.strip())
    a_match = re.search(
        r"A\s*([\d.]+)\s+[\d.]+\s+[\d.]+\s+([01])\s+[01]\s+([\d.]+)\s+([\d.]+)", path_d
    )
    if m_match is None or a_match is None:
        return math.nan
    ox0, oy0 = float(m_match.group(1)), float(m_match.group(2))
    r = float(a_match.group(1))
    large_arc = a_match.group(2) == "1"
    ox1, oy1 = float(a_match.group(3)), float(a_match.group(4))
    chord = math.sqrt((ox1 - ox0) ** 2 + (oy1 - oy0) ** 2)
    ratio = chord / (2.0 * r)
    if ratio > 1.01:
        return math.nan
    naive = 2.0 * math.asin(min(ratio, 1.0))
    return math.tau - naive if large_arc else naive


def _clip_rect(svg: str) -> tuple[float, float, float, float]:
    """Return the panel's ``(x, y, width, height)`` clip rect."""
    m = re.search(
        r'<clipPath id="ferrum-clip-0"><rect x="([\d.]+)" y="([\d.]+)" '
        r'width="([\d.]+)" height="([\d.]+)"',
        svg,
    )
    assert m is not None, "no clip-path rect found in SVG"
    return tuple(float(g) for g in m.groups())  # type: ignore[return-value]


def _max_line_length(svg: str) -> float:
    """Return the longest ``<line>`` element's length in the SVG.

    A real rendered cartesian axis emits one domain line spanning the full
    plot-area extent (well over 100px in these fixtures); the circular polar
    grid's radial tick marks are a few pixels long. This distinguishes "no
    y-axis actually drew" from "a y-axis drew, coincidentally at 0 opacity."
    """
    longest = 0.0
    for attrs in re.findall(r"<line([^>]+)/>", svg):
        coords = {}
        for axis in ("x1", "y1", "x2", "y2"):
            m = re.search(rf'{axis}="([\-\d.]+)"', attrs)
            if m is None:
                break
            coords[axis] = float(m.group(1))
        else:
            length = math.hypot(coords["x2"] - coords["x1"], coords["y2"] - coords["y1"])
            longest = max(longest, length)
    return longest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def pie_df() -> pl.DataFrame:
    return pl.DataFrame({"category": ["A", "B", "C", "D"], "value": [10.0, 20.0, 30.0, 15.0]})


# ---------------------------------------------------------------------------
# (a) to_dict() publishes only the user's own channels, both theta directions
# ---------------------------------------------------------------------------


def test_to_dict_theta_x_has_no_phantom_y_channel(pie_df: pl.DataFrame) -> None:
    d = (
        fm.Chart(pie_df)
        .mark_arc()
        .encode(theta="value:Q")
        .coord(fm.CoordPolar(theta="x"))
        .to_dict()
    )
    assert "y" not in d["encoding"], (
        f"phantom 'y' channel leaked into to_dict(): {sorted(d['encoding'])}"
    )
    assert sorted(d["encoding"]) == ["x"]


def test_to_dict_theta_y_has_no_phantom_x_channel(pie_df: pl.DataFrame) -> None:
    d = (
        fm.Chart(pie_df)
        .mark_arc()
        .encode(theta="value:Q")
        .coord(fm.CoordPolar(theta="y"))
        .to_dict()
    )
    assert "x" not in d["encoding"], (
        f"phantom 'x' channel leaked into to_dict(): {sorted(d['encoding'])}"
    )
    assert sorted(d["encoding"]) == ["y"]


def test_to_dict_with_color_still_carries_only_user_channels(pie_df: pl.DataFrame) -> None:
    """A more realistic pie chart (theta + color) must not gain a phantom
    positional channel either -- the dummy synthesis ran unconditionally on
    every arc-under-CoordPolar encoding, not just the theta-only case."""
    d = (
        fm.Chart(pie_df)
        .mark_arc()
        .encode(theta="value:Q", color="category:N")
        .coord(fm.CoordPolar(theta="x"))
        .to_dict()
    )
    assert sorted(d["encoding"]) == ["color", "x"]


# ---------------------------------------------------------------------------
# (c) Numeric-ordinal theta no longer aliases into radius -- full-radius
#     wedges, pinned against an independently-derived ground truth (a
#     nominal STRING theta, which can never alias since col_as_f64 fails on
#     a string column regardless of pre/post-fix state) rather than mere
#     mutual agreement between wedges.
# ---------------------------------------------------------------------------


def _nominal_theta_full_radius(n_categories: int) -> float:
    """Outer radius of a theta-only arc chart whose theta is nominal-string
    typed -- immune to the aliasing bug by construction, so this is a safe
    ground truth for "the coord's true full outer radius" independent of
    whatever the ordinal-theta code path does."""
    cats = [chr(ord("a") + i) for i in range(n_categories)]
    df = pl.DataFrame({"cat": cats, "val": [10.0 * (i + 1) for i in range(n_categories)]})
    svg = fm.Chart(df).mark_arc().encode(theta="cat:N").coord(fm.CoordPolar(theta="x")).to_svg()
    paths = _filled_arc_paths(svg)
    radii = {round(_outer_radius_from_path(p), 2) for p in paths}
    assert len(radii) == 1, f"nominal-theta ground truth itself has non-uniform radii: {radii}"
    return next(iter(radii))


def test_numeric_ordinal_theta_renders_full_radius_wedges() -> None:
    """A theta column typed ``:O`` whose underlying values are numeric must
    not have those values silently read back as a radius column.

    Pre-fix (verified via the git-stash regression protocol, not re-run
    here): the dummy ``enc["y"] = enc["x"]`` copied the ordinal theta field
    into radius; ``arc.rs``'s ``build_nominal_theta`` read it back via
    ``col_as_f64`` and mapped it through the (ordinal, so domain-less)
    y-scale's ``[0, 1]`` fallback domain, producing distinct, aliased
    per-wedge radii for fractional theta values (e.g. ``[0.2, 0.5, 0.9]``
    aliased to outer radii ``[41.10, 102.74, 184.93]`` rather than a single
    full radius).

    Post-fix: ``radius_field`` is unconditionally ``None`` for a theta-only
    arc (no dummy to alias from), so every wedge falls back to the coord's
    full outer radius regardless of the theta column's numeric values --
    pinned here against the independently-derived nominal-theta ground
    truth, not just mutual agreement (a bug that collapsed every wedge to
    the SAME wrong radius would otherwise pass silently).
    """
    df = pl.DataFrame({"cat": [0.2, 0.5, 0.9], "val": [10.0, 20.0, 30.0]})
    svg = fm.Chart(df).mark_arc().encode(theta="cat:O").coord(fm.CoordPolar(theta="x")).to_svg()

    paths = _filled_arc_paths(svg)
    assert len(paths) == 3, f"expected 3 wedges for 3 categories, got {len(paths)}"

    radii = [_outer_radius_from_path(d) for d in paths]
    assert all(math.isfinite(r) for r in radii), f"could not extract outer radii: {radii}"

    expected = _nominal_theta_full_radius(3)
    assert all(abs(r - expected) < 0.5 for r in radii), (
        f"wedge radii {radii} != independently-derived full radius {expected} -- the theta "
        "column's numeric values are still leaking into the radius channel."
    )


def test_numeric_ordinal_theta_int_column_discriminates_zero_indexed() -> None:
    """Integer-valued ordinal theta must also reach full radius for every
    wedge, including for the zero-indexed case that actually discriminates
    the bug: ``[1, 2, 3]`` does NOT discriminate here (pre-fix, the ordinal
    y-scale's ``[0, 1]`` fallback domain clamps every value ``>= 1`` to the
    SAME full radius by coincidence, so that dataset renders identically
    whether or not the aliasing bug is present). ``[0, 1, 2]`` does: pre-fix,
    category 0 maps to ``t=0`` -> inner radius (0px, collapsed wedge) while
    categories 1 and 2 clamp to full radius, giving `[0.0, full, full]` --
    genuinely distinguishable from this test's post-fix expectation of three
    equal full-radius wedges.
    """
    df = pl.DataFrame({"cat": [0, 1, 2], "val": [10.0, 20.0, 30.0]})
    svg = fm.Chart(df).mark_arc().encode(theta="cat:O").coord(fm.CoordPolar(theta="x")).to_svg()

    paths = _filled_arc_paths(svg)
    assert len(paths) == 3
    radii = [_outer_radius_from_path(d) for d in paths]

    expected = _nominal_theta_full_radius(3)
    assert all(abs(r - expected) < 0.5 for r in radii), (
        f"wedge radii {radii} != independently-derived full radius {expected}"
    )


# ---------------------------------------------------------------------------
# (b) Pie/donut wedge geometry and margin-convention structural checks.
#     Deliberately NOT a byte/hash pin: see module docstring for why full-SVG
#     byte-identity to pre-fix does not hold, and why that's the sanctioned,
#     architecturally-correct outcome rather than a regression.
# ---------------------------------------------------------------------------


def test_pie_theta_x_wedge_geometry(pie_df: pl.DataFrame) -> None:
    svg = (
        fm.Chart(pie_df)
        .mark_arc()
        .encode(theta="value:Q", color="category:N")
        .coord(fm.CoordPolar(theta="x"))
        .to_svg()
    )
    paths = _filled_arc_paths(svg)
    assert len(paths) == 4, f"expected 4 wedges for 4 categories, got {len(paths)}"

    radii = [_inner_outer_radii(p) for p in paths]
    inners = {round(i, 1) for i, _ in radii}
    outers = {round(o, 1) for _, o in radii}
    assert inners == {0.0}, f"a pie (no inner_radius) must have every wedge inner_r == 0: {inners}"
    assert len(outers) == 1, f"a pie's wedges must share one outer radius: {outers}"


def test_pie_theta_y_wedge_geometry() -> None:
    df = pl.DataFrame({"category": ["A", "B", "C"], "value": [10.0, 20.0, 30.0]})
    svg = fm.Chart(df).mark_arc().encode(theta="value:Q").coord(fm.CoordPolar(theta="y")).to_svg()
    paths = _filled_arc_paths(svg)
    assert len(paths) == 3
    outers = {round(_outer_radius_from_path(p), 1) for p in paths}
    assert len(outers) == 1, f"a pie's wedges must share one outer radius: {outers}"


def test_donut_wedge_geometry(pie_df: pl.DataFrame) -> None:
    svg = (
        fm.Chart(pie_df)
        .mark_arc()
        .encode(theta="value:Q", color="category:N")
        .coord(fm.CoordPolar(theta="x", inner_radius=60))
        .to_svg()
    )
    paths = _filled_arc_paths(svg)
    assert len(paths) == 4

    radii = [_inner_outer_radii(p) for p in paths]
    inners = {round(i, 1) for i, _ in radii}
    outers = {round(o, 1) for _, o in radii}
    assert len(inners) == 1 and next(iter(inners)) > 1.0, (
        f"a donut's hole (inner_radius=60) must render as one shared, non-zero inner radius: {inners}"
    )
    assert len(outers) == 1, f"a donut's wedges must share one outer radius: {outers}"
    assert next(iter(outers)) > next(iter(inners)), "outer radius must exceed inner radius"


def test_wedge_proportions_unaffected_by_margin_shift(pie_df: pl.DataFrame) -> None:
    """Even though the panel's plot area shifts (see module docstring), the
    wedges' angular sweeps must still be proportional to their data values --
    the margin/layout side effect must not corrupt the arc geometry itself."""
    svg = (
        fm.Chart(pie_df)
        .mark_arc()
        .encode(theta="value:Q", color="category:N")
        .coord(fm.CoordPolar(theta="x"))
        .to_svg()
    )
    paths = _filled_arc_paths(svg)
    assert len(paths) == 4

    sweeps = [_arc_sweep_radians(d) for d in paths]
    assert all(math.isfinite(s) for s in sweeps), f"NaN sweep detected: {sweeps}"
    total = sum(sweeps)
    assert abs(total - math.tau) < 0.05, f"total sweep {math.degrees(total):.1f} deg != 360 deg"

    # value=[10, 20, 30, 15] -> sweep ratios [10, 20, 30, 15] / 75.
    values = [10.0, 20.0, 30.0, 15.0]
    expected_ratios = [v / sum(values) for v in values]
    actual_ratios = [s / total for s in sweeps]
    for expected, actual in zip(expected_ratios, actual_ratios):
        assert abs(expected - actual) < 0.01, f"sweep ratio {actual:.4f} != expected {expected:.4f}"


def test_arc_sweep_radians_handles_sweeps_over_180_degrees() -> None:
    """Regression guard for the chord-inversion ambiguity documented on
    ``_arc_sweep_radians``: a 2-category pie with a dominant slice (300deg /
    60deg) exercises the large-arc-flag disambiguation directly -- without
    it, the 300deg wedge would compute as its 60deg supplement and the total
    sweep assertion below would fail by half the circle, misattributing a
    test-helper bug to the renderer."""
    df = pl.DataFrame({"category": ["dominant", "minor"], "value": [300.0, 60.0]})
    svg = fm.Chart(df).mark_arc().encode(theta="value:Q").coord(fm.CoordPolar(theta="x")).to_svg()
    paths = _filled_arc_paths(svg)
    assert len(paths) == 2

    sweeps = sorted(_arc_sweep_radians(d) for d in paths)
    assert all(math.isfinite(s) for s in sweeps), f"NaN sweep detected: {sweeps}"
    assert abs(math.degrees(sweeps[0]) - 60.0) < 1.0, f"minor sweep {math.degrees(sweeps[0]):.1f}"
    assert abs(math.degrees(sweeps[1]) - 300.0) < 1.0, f"major sweep {math.degrees(sweeps[1]):.1f}"
    assert abs(sum(sweeps) - math.tau) < 0.05


def test_theta_x_arc_left_margin_matches_established_single_axis_convention() -> None:
    """Root-cause confirmation for the module docstring's finding: a
    theta="x" arc's plot-area left edge must now match the SAME left edge an
    existing single-axis ``mark_tick(x=...)`` chart gets (both resolve their
    absent axis through the identical ``dummy_unit_scale`` exemption arm),
    not the wider margin a real two-axis chart reserves for realistic tick
    labels."""
    df = pl.DataFrame({"category": ["A", "B", "C"], "value": [10.0, 20.0, 30.0]})
    pie_svg = (
        fm.Chart(df).mark_arc().encode(theta="value:Q").coord(fm.CoordPolar(theta="x")).to_svg()
    )
    tick_svg = (
        fm.Chart(pl.DataFrame({"x": [1, 2, 3, 20, 25, 30]})).mark_tick().encode(x="x:Q").to_svg()
    )
    assert _clip_rect(pie_svg)[0] == _clip_rect(tick_svg)[0]


def test_theta_y_arc_bottom_margin_matches_established_single_axis_convention() -> None:
    """Mirror of the theta="x" check above: a theta="y" arc's absent axis is
    x (the radius side), which governs the panel's BOTTOM margin, not the
    left. The panel height must match a single-axis ``mark_tick(y=...)``
    chart's height for the same reason."""
    df = pl.DataFrame({"category": ["A", "B", "C"], "value": [10.0, 20.0, 30.0]})
    pie_svg = (
        fm.Chart(df).mark_arc().encode(theta="value:Q").coord(fm.CoordPolar(theta="y")).to_svg()
    )
    tick_svg = (
        fm.Chart(pl.DataFrame({"y": [1, 2, 3, 20, 25, 30]})).mark_tick().encode(y="y:Q").to_svg()
    )
    assert _clip_rect(pie_svg)[3] == _clip_rect(tick_svg)[3]


def test_pie_never_renders_a_real_y_axis_line(pie_df: pl.DataFrame) -> None:
    """The margin reservation discovered above is pure overhead: CoordPolar
    suppresses the actual y-axis draw both before and after this fix, so no
    long domain/tick line should ever appear -- only the short circular
    radial tick marks (a few px) and legend swatches (4px radius circles,
    not lines)."""
    svg = (
        fm.Chart(pie_df)
        .mark_arc()
        .encode(theta="value:Q", color="category:N")
        .coord(fm.CoordPolar(theta="x"))
        .to_svg()
    )
    longest = _max_line_length(svg)
    assert longest < 20.0, (
        f"longest <line> element is {longest:.1f}px -- a real cartesian axis domain/tick "
        "line appears to have rendered under CoordPolar"
    )
