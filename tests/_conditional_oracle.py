"""Pure pixel-oracle logic for the conditional-encoding harness.

No page, no fixture, no browser — every function here takes plain numbers
or RGBA tuples and returns a bool or raises ``AssertionError``. Kept apart
from ``tests/test_conditional_pixels.py`` (which is ``wasm_pixels``-marked
end to end) so ``tests/test_conditional_oracle.py``'s negative-control
self-tests can exercise this logic directly, with no marker and no browser.
"""

from __future__ import annotations

import math


def rgb_distance3(a: tuple[int, ...], b: tuple[int, ...]) -> float:
    """Euclidean distance between two RGB(A) tuples' first three channels."""
    return sum((x - y) ** 2 for x, y in zip(a[:3], b[:3])) ** 0.5


def assert_rgb_close(
    actual: tuple[int, ...], expected: tuple[int, ...], *, tol: float = 2.0, label: str = ""
) -> None:
    """Assert every one of the first three channels of *actual* is within *tol* of *expected*."""
    diffs = [abs(a - e) for a, e in zip(actual[:3], expected[:3])]
    assert max(diffs) <= tol, (
        f"{label}: expected rgb {tuple(expected[:3])} within {tol} per channel, got "
        f"{tuple(actual[:3])} (diffs {diffs})"
    )


def predicted_fill_edge(mark: str, form_value: float, stroke_width: float = 0.0) -> float:
    """The SVG-space distance from the mark's fixed anchor to its *fill* edge.

    Circle: ``EncodingValue::Size`` sets ``inst.radius = sqrt(value / pi)``
    (`crates/ferrum-wasm/src/conditional.rs::apply_value_to_circle`) — the
    same area-to-radius convention the static style ``size=`` kwarg uses.
    The stroke insets *inward* from that outer radius (confirmed directly:
    a widened stroke reaches further inward, not outward), so a fill-edge
    scan outward from the center finds `outer_radius - stroke_width`, not
    the outer radius itself.

    Rect: the same variant sets ``inst.size = [value, value]`` directly (a
    raw pixel width/height, not an area) — ``RectInstance.position`` is the
    rect's fixed top-left corner (`crates/ferrum-core/src/render/pack_instances.rs`),
    unaffected by a size change, so the fill edge is `value - stroke_width`
    measured outward from that fixed corner (never `value / 2`: the rect
    grows right/down from its own corner, not symmetrically from a center).
    """
    if mark == "point":
        return math.sqrt(form_value / math.pi) - stroke_width
    return form_value - stroke_width


def assert_radius_close(
    measured: float, predicted: float, *, tol: float = 3.0, label: str = ""
) -> None:
    assert abs(measured - predicted) <= tol, (
        f"{label}: measured extent {measured} is more than {tol}px from the predicted "
        f"{predicted:.2f}"
    )


def classify_ring(
    samples: list[tuple[int, ...]], stroke_rgb: tuple[int, ...], *, tol: float = 30.0
) -> list[str]:
    """Classify each sampled pixel as stroke-colored ("S") or not (".")."""
    return ["S" if rgb_distance3(s, stroke_rgb) < tol else "." for s in samples]


def denoise_ring(classes: list[str]) -> list[str]:
    """Collapse isolated single-sample flips (both circular neighbors agree with each other).

    A true radial-band edge crossing the sample ring produces one
    contiguous arc, but anti-aliasing at the two points where the ring
    crosses the band boundary can flip a single sample to the other class
    (confirmed directly: a 32-sample scan of this exact case shows one
    contiguous run of 15+ "stroke" samples plus two isolated single-sample
    blips right at its edge). Each such blip looks like a spurious pair of
    transitions to a naive counter, letting one contiguous arc masquerade
    as several short dashes. A blip is only collapsed when both its
    neighbors already agree — a real 1-sample-wide dash between two
    genuinely different neighbors is never touched.
    """
    n = len(classes)
    if n < 3:
        return list(classes)
    out = list(classes)
    for i in range(n):
        prev_c = classes[(i - 1) % n]
        next_c = classes[(i + 1) % n]
        if classes[i] != prev_c and prev_c == next_c:
            out[i] = prev_c
    return out


def ring_transitions(classes: list[str]) -> int:
    """Count classification changes around the ring, treating it as circular."""
    n = len(classes)
    return sum(1 for i in range(n) if classes[i] != classes[(i + 1) % n])


def longest_run_frac(classes: list[str]) -> float:
    """The longest circular run of one class, as a fraction of the ring."""
    n = len(classes)
    if n == 0:
        return 0.0
    if all(c == classes[0] for c in classes):
        return 1.0
    start = next(i for i in range(n) if classes[i] != classes[(i - 1) % n])
    best = 0
    cur = 0
    for k in range(n):
        i = (start + k) % n
        if k == 0 or classes[i] == classes[(i - 1) % n]:
            cur += 1
        else:
            cur = 1
        best = max(best, cur)
    return best / n


def ring_is_alternating(
    classes: list[str], *, min_transitions: int = 4, max_run_frac: float = 0.4
) -> bool:
    """A genuine dash pattern crosses the on/off boundary many times around the ring.

    A single contiguous arc — the "ring edge" artifact a band boundary
    produces when it happens to cross the sample radius — has exactly 2
    transitions (one entering the arc, one leaving it) regardless of how
    large the arc is, once single-sample anti-aliasing blips at its own
    edges are denoised (see `denoise_ring`). Requiring several times that
    many transitions on the denoised sequence rules out a band-edge false
    positive while still passing any real multi-dash period (period 9 over
    even a small ring's circumference crosses many times).

    At dense sampling (n=128), a fixed-radius scan that happens to sit
    close to the shader's own phase-wrap boundary (`fract(dist / period)`
    crossing from just under 1.0 back to just over 0.0 — confirmed by
    reading `circle.wgsl`) produces a WIDE multi-sample noisy fringe at
    each of the arc's two true edges, not a single isolated blip:
    `denoise_ring`'s single-sample collapse leaves several residual
    transitions in that fringe, which alone can reach `min_transitions`
    (measured directly: 6, on a real captured ring) while the ring is
    still fundamentally one big arc, not a dash. A genuine multi-dash
    pattern's longest single run is always a small fraction of the full
    ring (one dash/gap segment out of many repeats); a band-boundary arc's
    dominant run is close to half the ring. Requiring the longest run stay
    under `max_run_frac` rejects that arc while still passing a real
    dash — measured against the real geometry (see
    `test_conditional_oracle.py`'s positive control): longest run ~15% of
    the ring, versus ~48% for the captured single-arc-plus-fringe case.
    """
    denoised = denoise_ring(classes)
    return (
        ring_transitions(denoised) >= min_transitions and longest_run_frac(denoised) <= max_run_frac
    )


def ring_is_solid(classes: list[str], *, min_frac: float = 0.9) -> bool:
    """A solid ring is positively almost-all "stroke", not merely "not alternating".

    (An all-background ring — the stroke never painted at all — is neither
    alternating nor solid; this must be able to say so.)
    """
    n = len(classes)
    return n > 0 and classes.count("S") / n >= min_frac


def synthetic_dash_classes(
    circumference: float, pattern: list[float], n: int, *, phase: float = 0.0
) -> list[str]:
    """Classify `n` evenly-angle-spaced ring samples against a *perimeter* dash pattern.

    A positive control, not a model of the renderer under test: it walks
    the pattern by arc length around the ring's own circumference,
    independent of `classify_ring`'s pixel-color logic. Used to prove
    `ring_is_alternating` can register a genuine short-gap dash at a given
    sampling density and ring geometry.
    """
    period = sum(pattern)
    out: list[str] = []
    for i in range(n):
        arc = (i / n) * circumference + phase
        pos = arc % period
        on = False
        acc = 0.0
        for j, seg in enumerate(pattern):
            if pos < acc + seg:
                on = j % 2 == 0
                break
            acc += seg
        out.append("S" if on else ".")
    return out


# A fixed sample count that resolves a small mark's dash pattern can
# under-resolve a much larger mark's copy of the same pattern: the sample
# stride (perimeter / n) must stay well under the pattern's own SHORTEST
# on/off segment, or that segment falls entirely between two consecutive
# probes (a beat/aliasing false negative, not a rendering defect). Measured
# directly: a `[6, 3]` pattern, shortest segment 3, on a 959.6px-perimeter
# rect reads as one solid arc at a fixed 128-sample stride of 7.5px, then
# correctly alternating at any density with at least 4 samples per
# segment. Keying on the pattern's own *period* instead of its shortest
# segment under-resolves a pattern whose "off" gap is much shorter than its
# "on" dash (e.g. `[6, 3]`'s 3px gap needs the same density as `[3, 3]`'s,
# not the coarser density its 9px period alone would suggest).
_MIN_RING_SAMPLES = 128
_MIN_SAMPLES_PER_SEGMENT = 4
# The largest sample count any real (mark, pattern) combination in this
# harness's own matrix computes is 1920 (an unpacked rect's ~960px
# perimeter against the named `[4, 2]` pattern's 2px shortest segment,
# `ceil(4 * 960 / 2)`); this cap sits comfortably above that so no row in
# the current matrix is ever refused, while `ring_sample_count` below fails
# loudly rather than silently under-sampling anything that does cross it.
_MAX_RING_SAMPLES = 4096


def ring_sample_count(perimeter: float, min_segment: float | None) -> int:
    """How many ring samples resolve a dash pattern whose shortest segment is *min_segment*.

    `perimeter / min_segment` scales with the mark's own size, so this
    derives the count from both rather than using one fixed density for
    every mark. Floored at `_MIN_RING_SAMPLES` (enough for every
    packed-grid mark's small perimeter). A solid pattern (`min_segment` is
    `None` or non-positive) never aliases: there is no on/off boundary to
    under-resolve, so it always uses the floor.

    Raises `ValueError` naming *perimeter* and the density it would take,
    rather than silently clamping to `_MAX_RING_SAMPLES`: a silent clamp
    below the required density could under-sample a genuinely alternating
    pattern into misreading as solid.
    """
    if min_segment is None or min_segment <= 0:
        return _MIN_RING_SAMPLES
    needed = math.ceil(_MIN_SAMPLES_PER_SEGMENT * perimeter / min_segment)
    if needed > _MAX_RING_SAMPLES:
        raise ValueError(
            f"ring_sample_count: a ring of perimeter {perimeter:.1f} against a "
            f"{min_segment:.1f}px minimum dash segment needs {needed} samples, "
            f"over the {_MAX_RING_SAMPLES} cap -- raise the cap rather than "
            "silently under-sampling a pattern this harness could then misread."
        )
    return max(_MIN_RING_SAMPLES, needed)
