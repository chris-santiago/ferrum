"""Smoke + boundary tests for Phase 4 scales.

Math correctness is covered by `cargo test`; these tests verify the
Python boundary conveys f64s, errors propagate as ValueError, and the
public API surface matches the spec.
"""

import math
import pytest
import polars as pl

import ferrum as fm
from ferrum import (
    DivergingScale,
    LinearScale,
    LogScale,
    PowScale,
    SequentialScale,
    TimeScale,
    SymlogScale,
    OrdinalScale,
    QuantileScale,
    ThresholdScale,
)


# ---------- construction smoke ----------


def test_linear_construct_and_basic_scale():
    s = LinearScale(domain=[0.0, 10.0], range=[0.0, 1.0])
    assert math.isclose(s.scale(5.0), 0.5)
    assert math.isclose(s.invert(0.5), 5.0)
    assert s.domain == [0.0, 10.0]
    assert s.range == [0.0, 1.0]
    assert s.clamp is False


def test_log_construct_and_basic_scale():
    s = LogScale(domain=[1.0, 1000.0], range=[0.0, 3.0])
    assert math.isclose(s.scale(10.0), 1.0)
    assert math.isclose(s.invert(2.0), 100.0)
    assert s.base == 10.0


def test_time_construct_and_round_trip():
    s = TimeScale(domain=[0.0, 86400000.0], range=[0.0, 1.0])
    mid = 43200000.0
    assert math.isclose(s.scale(mid), 0.5)
    assert math.isclose(s.invert(0.5), mid)


def test_symlog_handles_zero():
    s = SymlogScale(domain=[-100.0, 100.0], range=[0.0, 1.0])
    assert math.isclose(s.scale(0.0), 0.5)
    assert s.constant == 1.0


def test_ordinal_basic():
    s = OrdinalScale(domain=["a", "b", "c"], range=[0.0, 30.0])
    assert math.isclose(s.scale("a"), 5.0)
    assert math.isclose(s.scale("b"), 15.0)
    assert math.isclose(s.scale("c"), 25.0)
    assert s.invert(5.0) == "a"
    assert s.invert(100.0) is None
    assert s.ticks() == ["a", "b", "c"]


def test_threshold_basic():
    s = ThresholdScale(domain=[0.0, 10.0], range=[1.0, 2.0, 3.0])
    assert s.scale(-1.0) == 1.0
    assert s.scale(5.0) == 2.0
    assert s.scale(11.0) == 3.0
    assert s.invert_extent(2.0) == (0.0, 10.0)


def test_quantile_basic():
    s = QuantileScale(domain=[1.0, 2.0, 3.0, 4.0, 5.0], range=[10.0, 20.0, 30.0])
    assert s.scale(0.0) == 10.0
    assert s.scale(2.5) == 20.0
    assert s.scale(100.0) == 30.0
    lo, hi = s.invert_extent(20.0)
    assert lo < hi
    assert len(s.quantiles) == 2


# ---------- inversion round trips ----------


def test_continuous_inversion_round_trip():
    for cls, kwargs in [
        (LinearScale, dict(domain=[0.0, 100.0], range=[-1.0, 1.0])),
        (LogScale, dict(domain=[1.0, 1e6], range=[0.0, 6.0])),
        (TimeScale, dict(domain=[0.0, 1e10], range=[0.0, 1.0])),
        (SymlogScale, dict(domain=[-100.0, 100.0], range=[0.0, 1.0])),
    ]:
        s = cls(**kwargs)
        for x in [
            kwargs["domain"][0],
            (kwargs["domain"][0] + kwargs["domain"][1]) / 2,
            kwargs["domain"][1],
        ]:
            y = s.scale(x)
            back = s.invert(y)
            assert math.isclose(back, x, rel_tol=1e-6, abs_tol=1e-6), (
                f"{cls.__name__}: x={x} -> y={y} -> back={back}"
            )


# ---------- nan propagation ----------


def test_scale_nan_propagates():
    s = LinearScale(domain=[0.0, 10.0], range=[0.0, 1.0])
    assert math.isnan(s.scale(float("nan")))
    assert math.isnan(s.invert(float("nan")))


def test_scale_out_of_domain_returns_nan_when_unclamped():
    s = LinearScale(domain=[0.0, 10.0], range=[0.0, 1.0])
    assert math.isnan(s.scale(-1.0))
    assert math.isnan(s.scale(11.0))


def test_scale_clamp_clamps_output():
    s = LinearScale(domain=[0.0, 10.0], range=[0.0, 1.0], clamp=True)
    assert s.scale(-1.0) == 0.0
    assert s.scale(11.0) == 1.0


# ---------- constructor errors ----------


def test_linear_rejects_wrong_domain_length():
    with pytest.raises(ValueError, match="domain must have length 2"):
        LinearScale(domain=[0.0, 1.0, 2.0], range=[0.0, 1.0])


def test_linear_rejects_degenerate_domain():
    with pytest.raises(ValueError, match="domain endpoints must differ"):
        LinearScale(domain=[5.0, 5.0], range=[0.0, 1.0])


def test_linear_rejects_short_range_with_no_domain():
    """A short `range` with `domain` omitted must raise, not panic.

    Regression test (GH #69 sibling): `resolve_continuous` (scale/core.rs)
    used to gate ALL range validation behind `domain_user_set`, so a 1-element
    `range` with no `domain` indexed past the end of the Vec at
    `range: [r[0], r[1]]` and crashed the process with an out-of-bounds
    panic instead of raising a typed `ValueError`. Every affine-continuous
    scale (Linear/Log/Pow/Symlog) shares this constructor prelude.
    """
    with pytest.raises(ValueError, match="range must have length 2"):
        LinearScale(range=[5.0])


def test_pow_rejects_short_range_with_no_domain():
    """Sibling of the LinearScale case above — same shared `resolve_continuous`."""
    with pytest.raises(ValueError, match="range must have length 2"):
        PowScale(range=[5.0])


def test_symlog_rejects_non_finite_range_with_no_domain():
    """A non-finite `range` with `domain` omitted used to be silently accepted
    (same validation gate as the short-range panic above); must now raise.
    """
    with pytest.raises(ValueError):
        SymlogScale(range=[0.0, float("inf")])


def test_sequential_scale_rejects_short_domain():
    """SequentialScale(domain=[5.0]) must raise, not silently become [0, 1].

    Regression test (GH #69 sibling): scale/sequential.rs had the identical
    silent-fabrication pattern as the pre-fix Band/Point/Diverging
    constructors (`if v.len() >= 2 {...} else { [0.0, 1.0] }`), which shipped
    the placeholder to the wire as if it were the user's explicit intent.
    """
    with pytest.raises(ValueError, match="domain must have length"):
        SequentialScale(domain=[5.0])


def test_sequential_scale_rejects_non_finite_domain():
    with pytest.raises(ValueError):
        SequentialScale(domain=[0.0, float("nan")])


def test_diverging_scale_rejects_degenerate_domain():
    """DivergingScale(domain=[5.0, 5.0]) must raise (lo == hi).

    Cohesion fix (#69 review): `resolve_continuous` and `QuantizeScale` both
    reject a degenerate (lo == hi) domain, but `DivergingScale::new` did not
    — adjudicated as an in-scope unification gap. A 3-element domain whose
    first and last entries match (`[5.0, 7.0, 5.0]`) is rejected the same
    way.
    """
    with pytest.raises(ValueError, match="domain endpoints must differ"):
        DivergingScale(domain=[5.0, 5.0])
    with pytest.raises(ValueError, match="domain endpoints must differ"):
        DivergingScale(domain=[5.0, 7.0, 5.0])


def test_log_rejects_zero_in_domain():
    with pytest.raises(ValueError, match="must not contain 0"):
        LogScale(domain=[0.0, 100.0], range=[0.0, 2.0])


def test_log_rejects_mixed_signs():
    with pytest.raises(ValueError, match="same sign"):
        LogScale(domain=[-1.0, 100.0], range=[0.0, 2.0])


def test_log_rejects_invalid_base():
    with pytest.raises(ValueError, match="base must be"):
        LogScale(domain=[1.0, 100.0], range=[0.0, 2.0], base=1.0)


def test_symlog_rejects_invalid_constant():
    with pytest.raises(ValueError, match="constant must be"):
        SymlogScale(domain=[-1.0, 1.0], range=[0.0, 1.0], constant=0.0)


def test_ordinal_rejects_empty_domain():
    with pytest.raises(ValueError, match="domain must be non-empty"):
        OrdinalScale(domain=[], range=[0.0, 10.0])


def test_ordinal_rejects_duplicates():
    with pytest.raises(ValueError, match="duplicate"):
        OrdinalScale(domain=["a", "a"], range=[0.0, 10.0])


def test_ordinal_rejects_bad_padding():
    with pytest.raises(ValueError, match="padding"):
        OrdinalScale(domain=["a"], range=[0.0, 10.0], padding=1.5)


def test_threshold_rejects_arity_mismatch():
    with pytest.raises(ValueError, match="range length must equal domain length"):
        ThresholdScale(domain=[0.0, 10.0], range=[1.0, 2.0])


def test_threshold_rejects_unsorted_domain():
    with pytest.raises(ValueError, match="strictly sorted"):
        ThresholdScale(domain=[10.0, 0.0], range=[1.0, 2.0, 3.0])


def test_quantile_rejects_short_domain():
    with pytest.raises(ValueError, match="length >= 2"):
        QuantileScale(domain=[1.0], range=[0.0, 1.0])


# ---------- ticks ----------


def test_linear_ticks_default_count():
    s = LinearScale(domain=[0.0, 10.0], range=[0.0, 1.0])
    t = s.ticks()
    assert len(t) >= 5


def test_quantile_ticks_default_uses_sturges():
    # domain length 100, sturges = ceil(log2(100)+1) = 8
    sample = [float(i) for i in range(100)]
    s = QuantileScale(domain=sample, range=[float(i) for i in range(20)])
    t = s.ticks()
    # default count should be sturges_floor(100) = 8
    assert len(t) == 8, f"expected 8 (sturges of 100), got {len(t)}: {t}"


def test_threshold_ticks_returns_thresholds():
    s = ThresholdScale(domain=[0.0, 10.0, 20.0], range=[1.0, 2.0, 3.0, 4.0])
    assert s.ticks() == [0.0, 10.0, 20.0]


# ---------- nice ----------


def test_linear_nice_extends_domain():
    s = LinearScale(domain=[0.13, 9.7], range=[0.0, 1.0]).nice()
    assert s.domain[0] <= 0.13
    assert s.domain[1] >= 9.7


def test_ordinal_nice_is_identity():
    s = OrdinalScale(domain=["a", "b"], range=[0.0, 10.0])
    n = s.nice()
    assert n.domain == s.domain
    assert n.range == s.range


# ---------- render-side raw-dict domain/range shape refusal (#152) ----------
#
# The tests above exercise the CLASS spelling (`LinearScale(domain=[...])`),
# which already refuses a bad shape at construction, before rendering is ever
# reached. `scale={'type': 'linear', 'domain': [5.0]}` is a distinct spelling
# that skips every pyclass constructor entirely: it is a legal wire KEY the
# raw-dict gate (`tests/test_scale_dict_gate.py`) accepts, with a malformed
# VALUE the gate never inspects. Pre-fix, that shape reached
# `LinearScale::new_internal`'s (or a sibling continuous scale's) unchecked
# `domain[1]` indexing at RENDER time and panicked `index out of bounds`
# across the PyO3 boundary as a raw `PanicException`, instead of raising.
# These tests render (not just build) a chart with a raw-dict scale, on all
# 7 continuous wire types, to prove the render-time funnel
# (`resolve_continuous_domain_and_range` /
# `validate_explicit_continuous_shape` in
# `crates/ferrum-core/src/render/scale_resolve/positional.rs`) now refuses
# typed instead of panicking — the funnel-level acceptance matrix itself
# (7 kinds x domain/range x length/degenerate) is proven exhaustively in Rust
# (`positional.rs::tests::explicit_continuous_domain_range_shape_matrix`);
# this is the PyO3-crossing proof for the exact raw-dict spelling.
#
# Scoped to LENGTH only, deliberately NOT degeneracy, unlike the class
# spelling's own `validate_continuous_domain` (which does refuse a degenerate
# `[c, c]` domain, per `test_linear_rejects_degenerate_domain` above): a
# raw-dict domain reaching this funnel is indistinguishable from a reactive
# `domainParam`'s current substituted value (`fm.param(..., value=[5, 5])`),
# and `tests/test_bug_hunt_reactive_params.py
# ::TestDomainParamDegenerate::test_collapsed_domain_no_nan` already pins
# that a degenerate domainParam value must keep rendering gracefully, not
# refuse — see the Rust doc comment on `validate_explicit_continuous_shape`
# for the full account of why widening the render-side refusal to match the
# class spelling's degenerate check is NOT safe here.

_CONTINUOUS_WIRE_TYPES = ["linear", "log", "time", "symlog", "pow", "sqrt", "utc"]


def _positional_domain_for(scale_type: str) -> list[float]:
    """A domain valid for `scale_type` — `log`/`symlog` need one that
    respects their own sign constraints; every other continuous type
    tolerates the plain `[0.0, 100.0]` default.
    """
    if scale_type == "symlog":
        return [-10.0, 10.0]
    if scale_type == "log":
        return [1.0, 100.0]
    return [0.0, 100.0]


def _render_with_raw_dict_scale(scale_type: str, **overrides: object) -> str:
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [0.0, 100.0, 50.0]})
    scale = {"type": scale_type, **overrides}
    chart = fm.Chart(df).mark_point().encode(x=fm.X("x", scale=scale), y="y")
    return chart.to_svg()


@pytest.mark.parametrize("scale_type", _CONTINUOUS_WIRE_TYPES)
def test_raw_dict_single_element_domain_refuses_not_panics(scale_type: str) -> None:
    """#152 RED proof: a raw-dict single-element `domain` used to panic at
    render time on every continuous wire type; now refuses typed, matching
    `LinearScale(domain=[5.0])`'s own construction-time message (pinned
    above in `test_linear_rejects_wrong_domain_length`).
    """
    with pytest.raises(ValueError, match="domain must have length 2"):
        _render_with_raw_dict_scale(scale_type, domain=[5.0])


@pytest.mark.parametrize("scale_type", _CONTINUOUS_WIRE_TYPES)
def test_raw_dict_degenerate_domain_still_renders(scale_type: str) -> None:
    """A raw-dict degenerate `domain=[c, c]` is NOT refused, unlike
    `LinearScale(domain=[5.0, 5.0])`'s own construction-time refusal
    (`test_linear_rejects_degenerate_domain` above): this exact shape is
    also what a reactive `domainParam`'s current value looks like once
    substituted (see the module-doc note above), and
    `test_bug_hunt_reactive_params.py::test_collapsed_domain_no_nan` already
    requires that a degenerate domainParam value keep rendering gracefully
    — a raw-dict literal domain converges to the identical wire shape, so it
    must render the same way, not refuse.
    """
    svg = _render_with_raw_dict_scale(scale_type, domain=[5.0, 5.0])
    assert "<svg" in svg


@pytest.mark.parametrize("scale_type", _CONTINUOUS_WIRE_TYPES)
def test_raw_dict_single_element_range_refuses(scale_type: str) -> None:
    """The RANGE-side sibling of the length-2 domain refusal above."""
    with pytest.raises(ValueError, match="range must have length 2"):
        _render_with_raw_dict_scale(
            scale_type, domain=_positional_domain_for(scale_type), range=[5.0]
        )


@pytest.mark.parametrize("scale_type", _CONTINUOUS_WIRE_TYPES)
def test_raw_dict_degenerate_range_still_renders(scale_type: str) -> None:
    """A degenerate `range=[c, c]` is NOT refused: `LinearScale(range=[5.0,
    5.0])` was never refused at construction either (`validate_continuous_range`
    checks length and finiteness only, never degeneracy), so #152 — which
    only closes the DOMAIN gap — must not start refusing a range shape the
    class spelling has always accepted.
    """
    svg = _render_with_raw_dict_scale(
        scale_type, domain=_positional_domain_for(scale_type), range=[5.0, 5.0]
    )
    assert "<svg" in svg
