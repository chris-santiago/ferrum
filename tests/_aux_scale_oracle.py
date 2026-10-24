"""Python oracle for the numeric auxiliary channels' Log/Pow/Sqrt transform.

Mirrors ``crates/ferrum-core/src/scale/{log,pow}.rs``'s ``LogScaleData::scale``
/``PowScaleData::scale``, always under ``clamp=True`` (the auxiliary
channels' standing clamp contract: ``numeric_aux_scale_kind`` in
``crates/ferrum-core/src/render/scale_resolve/auxiliary.rs`` always passes
``clamp=true`` regardless of scale kind). Shared by
``test_layer_encoding_model.py``'s door cells and
``test_appearance_kwarg_matrix.py``'s channel-population matrix, so a fix to
this math fixes both call sites.
"""

from __future__ import annotations

import math
import sys


def degenerate_ratio(num: float, den: float) -> float:
    """A zero-width domain maps to the range midpoint, never NaN."""
    return 0.5 if den == 0.0 else num / den


def range_clamp(mapped: float, r0: float, r1: float) -> float:
    lo, hi = (r0, r1) if r0 <= r1 else (r1, r0)
    return min(max(mapped, lo), hi)


def log_map(
    x: float, domain: tuple[float, float], range_: tuple[float, float], base: float
) -> float:
    """Mirrors ``LogScaleData::scale``'s in-domain mapping, with ``clamp=True``.

    A value off the admitted side (e.g. non-positive under a positive
    domain) is never mapped in a chart render -- the render drops it or
    refuses outright, before this formula would run -- so callers only
    feed this an on-side ``x``.
    """
    d0, d1 = domain
    r0, r1 = range_
    sign = -1.0 if d0 < 0.0 else 1.0
    log_base = math.log(base)
    lx = math.log(max(x * sign, sys.float_info.min)) / log_base
    ld0 = math.log(d0 * sign) / log_base
    ld1 = math.log(d1 * sign) / log_base
    t = degenerate_ratio(lx - ld0, ld1 - ld0)
    return range_clamp(r0 + t * (r1 - r0), r0, r1)


def pow_map(
    x: float, domain: tuple[float, float], range_: tuple[float, float], exponent: float
) -> float:
    """Mirrors ``PowScaleData::scale`` with ``clamp=True``; Sqrt is ``exponent=0.5``."""
    d0, d1 = domain
    r0, r1 = range_

    def fwd(v: float) -> float:
        return (1.0 if v >= 0.0 else -1.0) * (abs(v) ** exponent)

    t = degenerate_ratio(fwd(x) - fwd(d0), fwd(d1) - fwd(d0))
    return range_clamp(r0 + t * (r1 - r0), r0, r1)


def snap_exp(raw: float) -> float:
    """Snaps a near-integer exponent exactly, matching ``LogScaleData::snap_exp``."""
    rounded = round(raw)
    return rounded if abs(raw - rounded) < 1e-10 else raw


def niced_log_domain(lo: float, hi: float, base: float) -> tuple[float, float]:
    """Mirrors ``LogScaleData::nice``: rounds each endpoint outward to the nearest power of base."""
    log_base = math.log(base)
    lo_exp = math.floor(snap_exp(math.log(lo) / log_base))
    hi_exp = math.ceil(snap_exp(math.log(hi) / log_base))
    return (base**lo_exp, base**hi_exp)
