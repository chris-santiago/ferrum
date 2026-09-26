"""Negative-control self-tests for the conditional pixel harness's pure oracle logic.

Plain tests: no ``wasm_pixels`` marker, no browser, no fixture — these run
in every default suite. Several of the harness's pixel-classification
checks (dash alternation, dash solidity, opacity blend, size radius) can be
made non-discriminating by a threshold loose enough to also pass a broken
renderer. Each helper's positive case must pass here, and each helper's
negative (synthetic, hand-built) case — the exact shape a broken renderer
would produce — must fail, proving the check actually distinguishes real
behavior from that broken shape before this harness is trusted to gate
anything.
"""

from __future__ import annotations

import math

import pytest

from _conditional_oracle import (
    assert_radius_close,
    assert_rgb_close,
    predicted_fill_edge,
    ring_is_alternating,
    ring_is_solid,
    synthetic_dash_classes,
)

# Mirrors tests/test_conditional_pixels.py's BASE_MARK_SIZE / DASH_STROKE_W —
# duplicated as plain numbers (not imported) so this module stays free of any
# dependency on the wasm_pixels-marked module or a browser.
_POINT_MARK_SIZE = 452.0
_DASH_STROKE_W = 6.0


def test_ring_alternating_rejects_a_ring_edge_artifact() -> None:
    """A single contiguous arc (a band boundary crossing the sample radius) is not a dash.

    Sampling a circular dash whose phase depends on radial distance (not
    perimeter arc length) produces one contiguous arc of "stroke" samples
    at the sample radius, not an alternating pattern — a `min_frac`-only
    check (counting how many samples are "stroke" with no transition
    count) would accept this as "mostly stroke" and call it solid or
    alternating without ever checking the samples' spatial arrangement.
    """
    ring_edge = ["."] * 8 + ["S"] * 8
    assert not ring_is_alternating(ring_edge)


def test_ring_alternating_rejects_a_ring_edge_with_an_antialiasing_blip() -> None:
    """A single contiguous arc with one AA-noise pixel flipped at its own edge.

    The exact shape a 32-sample capture of this bug produced on the
    committed bundle: one contiguous run of stroke samples, with an
    isolated single-sample gap right at the run's own edge. A transition
    counter with no denoising sees 4 transitions here and would wrongly
    call it alternating.
    """
    ring_edge_with_blip = list("..SS.SSSSS......")
    assert not ring_is_alternating(ring_edge_with_blip)


def test_ring_alternating_rejects_a_wide_noisy_fringe_around_one_big_arc() -> None:
    """A real 128-sample capture of the same F19/F19b artifact, denser than 16.

    Captured directly on the committed bundle (`stroke_dash` `palette` on a
    point mark, click driver): the harness's fixed sampling radius happens
    to sit close to `circle.wgsl`'s own phase-wrap boundary (confirmed by
    reading the shader), so each of the arc's two true edges produces a
    *multi*-sample noisy fringe, not `denoise_ring`'s single-sample blip.
    6 transitions survive single-sample denoising alone — past
    `min_transitions=4` — while the ring is still fundamentally one big
    arc (its longest run is ~48% of the ring). Pins the exact captured
    shape so a future change to `ring_is_alternating` cannot silently
    regress into accepting this again.
    """
    captured = list(
        ".......S..S...SSSS.S.S.SS.SSSSSS.SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS."
        "SSSSSS.......S........................................"
    )
    assert not ring_is_alternating(captured)


def test_ring_alternating_accepts_a_genuine_multi_dash_pattern() -> None:
    # Each on/off run is 2 samples wide, not 1 — denoising a single-sample
    # anti-aliasing blip should never collapse a real (>=2-sample) dash run.
    dashed = ["S", "S", ".", "."] * 4
    assert ring_is_alternating(dashed)


def test_ring_alternating_accepts_the_real_dash_geometry_at_128_samples() -> None:
    """Positive control built from the real `[6, 3]` pattern and real ring geometry.

    Not a model of the renderer under test (which is F19/F19b-broken —
    radial, not perimeter, phase) but a proof that `ring_is_alternating`
    CAN see a genuine short-gap dash at the harness's actual sampling
    density (128) and ring radius, once something (a future shader fix)
    produces a perimeter-phased dash to sample. If this ever regresses to
    False, the 128-sample density (or the ring radius it is computed
    against) is too coarse for the shortest real segment, and a future fix
    to the shader's dash phase would read as a solid ring rather than
    flipping the `stroke_dash` xfail (see F19/F19b in
    `test_conditional_pixels.py`).
    """
    r = math.sqrt(_POINT_MARK_SIZE / math.pi)
    ring_r = r - _DASH_STROKE_W / 2
    circumference = 2 * math.pi * ring_r
    classes = synthetic_dash_classes(circumference, [6, 3], n=128)
    assert ring_is_alternating(classes)


def test_ring_solid_rejects_an_unpainted_ring() -> None:
    """The stroke never painted at all is neither alternating nor solid.

    `not ring_is_alternating(...)` alone would accept this (an
    all-background ring is trivially "not alternating"), which is not the
    same claim as "solidly stroked" — an unpainted ring must be positively
    rejected by `ring_is_solid` too, not just fail to match "alternating".
    """
    unpainted = ["."] * 16
    assert not ring_is_solid(unpainted)
    assert not ring_is_alternating(unpainted)


def test_ring_solid_accepts_a_fully_painted_ring() -> None:
    solid = ["S"] * 16
    assert ring_is_solid(solid)


def test_opacity_tolerance_rejects_a_blend_off_by_three_units() -> None:
    with pytest.raises(AssertionError):
        assert_rgb_close((200, 200, 200), (197, 197, 197), tol=2, label="test")


def test_opacity_tolerance_accepts_a_blend_within_tolerance() -> None:
    assert_rgb_close((198, 198, 198), (197, 197, 197), tol=2, label="test")


def test_size_tolerance_rejects_a_radius_off_by_more_than_tolerance() -> None:
    with pytest.raises(AssertionError):
        assert_radius_close(10.0, predicted_fill_edge("point", 900.0), tol=3.0, label="test")


def test_size_tolerance_accepts_a_radius_within_tolerance() -> None:
    predicted = predicted_fill_edge("point", 900.0)
    assert_radius_close(predicted - 1.0, predicted, tol=3.0, label="test")


def test_predicted_fill_edge_formula_per_mark_kind() -> None:
    # Circle: an area-to-radius conversion, then the stroke inset inward.
    assert predicted_fill_edge("point", 900.0, stroke_width=3.0) == pytest.approx(
        math.sqrt(900.0 / math.pi) - 3.0
    )
    # Rect: the raw pixel value itself, inset by the stroke — never value/2
    # (the rect grows right/down from its own fixed top-left corner).
    assert predicted_fill_edge("rect", 900.0, stroke_width=3.0) == pytest.approx(897.0)
