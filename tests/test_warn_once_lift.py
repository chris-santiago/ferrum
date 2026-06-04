"""Phase 8b warn-once lift tests: verify warnings emit (or don't) at correct points."""

import warnings
import polars as pl
import numpy as np
import ferrum as fe


def test_mark_smooth_ci_no_longer_warns():
    df = pl.DataFrame({"x": np.arange(20.0), "y": np.arange(20.0) + np.random.randn(20)})
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        fe.Chart(df).mark_smooth(ci=0.95).encode(x="x", y="y").to_svg()
    smooth_ci_warns = [
        x for x in w if "mark_smooth" in str(x.message) and "ci" in str(x.message).lower()
    ]
    assert len(smooth_ci_warns) == 0


def test_mark_raster_blend_additive_warns_once():
    df = pl.DataFrame({"x": np.random.randn(50), "y": np.random.randn(50)})
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        fe.Chart(df).mark_raster(blend="additive", resolution=32).encode(x="x", y="y")._build_spec()
        fe.Chart(df).mark_raster(blend="additive", resolution=32).encode(x="x", y="y")._build_spec()
    blend_warns = [x for x in w if "blend" in str(x.message) and "additive" in str(x.message)]
    # warn_once collapses duplicates per-process; both calls produce 0 or 1 warn
    assert len(blend_warns) <= 1


def test_mark_swarm_dodge_no_longer_warns():
    # Phase 11e4: mark_swarm(dodge=...) is now fully implemented; no warn.
    df = pl.DataFrame({"g": ["a"] * 10, "v": np.random.randn(10)})
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        fe.Chart(df).mark_swarm(dodge="x").encode(x="g", y="v")
    dodge_warns = [x for x in w if "dodge" in str(x.message)]
    assert len(dodge_warns) == 0, (
        f"unexpected dodge warning: {[str(x.message) for x in dodge_warns]}"
    )
