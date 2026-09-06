"""`mark_smooth(k=)` and the conditional `bandwidth=` forward.

R1 exposed gam's basis dimension end to end in Rust (`PySmooth(k=)`, the wire
field, the stub) and made `bandwidth` an `Option<f64>` that refuses an
explicit non-loess value. Nothing in `src/ferrum/` forwarded either, so `k`
was reachable only through a direct `_core.Smooth(...)` construction, and
`desugar_smooth` forwarded `bandwidth` on *every* method -- which worked only
because `PySmooth` tolerates an explicit value equal to its own default.

Two contracts are pinned here:

* **`k` reaches the fit.** `mark_smooth(method="gam", k=20)` renders a
  materially different curve on a signal the default basis cannot resolve,
  and is refused on every other method.
* **A knob is forwarded only when the caller set it.** The recorded
  `Smooth(...)` kwargs carry `bandwidth`/`k`/`degree` only for a call that
  passed them, so `PySmooth`'s agrees-with-the-default tolerance stops being
  load-bearing for the whole mark layer.
"""

from __future__ import annotations

import math

import polars as pl
import pytest

import ferrum as fm
from ferrum.marks import statistical as _statistical


def _wiggly(n: int = 120) -> pl.DataFrame:
    """A six-period sine over the x span -- more turning points than a
    10-function cubic basis can follow."""
    xs = [i * (12.0 * math.pi) / n for i in range(n)]
    return pl.DataFrame({"x": xs, "y": [math.sin(x) for x in xs]})


def _record_smooth_kwargs(monkeypatch) -> list[dict]:
    """Capture the kwargs `desugar_smooth` hands the real `Smooth`.

    The production path is unchanged -- the recorder builds the same
    `Smooth` and returns it -- so what lands in the list is exactly what the
    transform was constructed with.
    """
    calls: list[dict] = []
    real = _statistical.Smooth

    def recording(*args, **kwargs):
        calls.append(dict(kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(_statistical, "Smooth", recording)
    return calls


# ---------------------------------------------------------------------------
# k= reaches the gam fit
# ---------------------------------------------------------------------------


def test_k_changes_the_gam_fit():
    frame = _wiggly()
    default = fm.Chart(frame).mark_smooth(method="gam").encode(x="x", y="y").to_svg()
    raised = fm.Chart(frame).mark_smooth(method="gam", k=20).encode(x="x", y="y").to_svg()
    assert default != raised, "k= did not reach the fit -- the two renders are identical"


def test_k_is_forwarded_through_the_layered_ci_path_too(monkeypatch):
    calls = _record_smooth_kwargs(monkeypatch)
    chart = fm.Chart(_wiggly()).mark_smooth(method="gam", k=15, ci=0.95).encode(x="x", y="y")
    chart._resolve_pending()
    assert [call.get("k") for call in calls] == [15]


@pytest.mark.parametrize("method", ["lm", "linear", "loess", "quadratic", "cubic", "log", "sqrt"])
def test_k_is_refused_on_every_non_gam_method(method):
    with pytest.raises(ValueError, match="k=12 is not valid"):
        fm.Chart(_wiggly()).mark_smooth(method=method, k=12).encode(x="x", y="y").to_svg()


def test_k_below_the_smallest_cubic_basis_is_refused():
    with pytest.raises(ValueError, match="below the smallest cubic spline basis"):
        fm.Chart(_wiggly()).mark_smooth(method="gam", k=3).encode(x="x", y="y").to_svg()


def test_omitting_k_is_the_documented_default_of_ten():
    frame = _wiggly()
    omitted = fm.Chart(frame).mark_smooth(method="gam").encode(x="x", y="y").to_svg()
    explicit = fm.Chart(frame).mark_smooth(method="gam", k=10).encode(x="x", y="y").to_svg()
    assert omitted == explicit


# ---------------------------------------------------------------------------
# Only explicitly-set knobs are forwarded
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["lm", "loess", "gam"])
def test_a_bare_call_forwards_no_bandwidth_degree_or_k(monkeypatch, method):
    """The RED half: before R2, `bandwidth=0.75` rode along on every call,
    so a non-loess method was handed a LOESS span it does not read."""
    calls = _record_smooth_kwargs(monkeypatch)
    chart = fm.Chart(_wiggly()).mark_smooth(method=method).encode(x="x", y="y")
    chart._resolve_pending()
    assert len(calls) == 1
    assert "bandwidth" not in calls[0]
    assert "degree" not in calls[0]
    assert "k" not in calls[0]


def test_an_explicit_bandwidth_is_still_forwarded(monkeypatch):
    calls = _record_smooth_kwargs(monkeypatch)
    chart = fm.Chart(_wiggly()).mark_smooth(method="loess", bandwidth=0.3).encode(x="x", y="y")
    chart._resolve_pending()
    assert calls[0]["bandwidth"] == 0.3


def test_an_explicit_bandwidth_still_changes_the_loess_fit():
    frame = _wiggly()
    default = fm.Chart(frame).mark_smooth(method="loess").encode(x="x", y="y").to_svg()
    narrow = (
        fm.Chart(frame).mark_smooth(method="loess", bandwidth=0.3).encode(x="x", y="y").to_svg()
    )
    assert default != narrow


def test_an_explicit_bandwidth_on_a_non_loess_method_is_refused():
    """The asymmetry R1 closed in Rust, reachable from the mark layer: a
    knob that does nothing raises instead of no-opping, and the message
    names the knob that method actually has."""
    with pytest.raises(ValueError, match="bandwidth=0.4 is not valid for method='gam'"):
        fm.Chart(_wiggly()).mark_smooth(method="gam", bandwidth=0.4).encode(x="x", y="y").to_svg()
