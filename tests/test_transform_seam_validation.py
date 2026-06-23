"""Regression tests for the transform Python/Rust seam (T4.11a).

SEAM-02 — one transform-construction API
  The Phase-12 data transforms are constructed only via the dict-emitting
  ``ferrum.transform_*`` functions. The parallel typed ``Data*`` PyO3 classes
  (DataAggregate / DataBin / DataCalculate / DataFilter / DataFold / DataPivot /
  DataStack / DataWindow) were dead (never imported, exported, or constructed)
  and have been removed from ``ferrum._core`` and the type stub. These tests pin
  that the dead classes are gone, the live transform classes remain, and the
  public ``transform_*`` surface is intact.

SEAM-04 — dict transform path enforces ``deny_unknown_fields``
  A mistyped/unknown key on a transform dict now raises a friendly ``ValueError``
  naming the bad field (the same strictness the theme path enforces), instead of
  silently dropping the key and running with defaults. These tests pin that
  unknown keys raise and name the field, that valid transforms still round-trip
  and render, and that the inner aggregate/window op dicts are validated too.
"""

from __future__ import annotations

import pytest
import polars as pl

import ferrum as fm
from ferrum import _core


_DF = pl.DataFrame(
    {
        "x": [1.0, 2.0, 3.0, 4.0, 5.0],
        "y": [5.0, 4.0, 3.0, 2.0, 1.0],
        "g": ["a", "a", "b", "b", "b"],
    }
)


# ---------------------------------------------------------------------------
# SEAM-02: the dead typed Data* classes are gone; live classes remain
# ---------------------------------------------------------------------------


# The eight Phase-12 data transforms whose typed pyclasses were removed.
_REMOVED_DATA_CLASSES = [
    "DataAggregate",
    "DataBin",
    "DataCalculate",
    "DataFilter",
    "DataFold",
    "DataPivot",
    "DataStack",
    "DataWindow",
]

# Live transform classes that DO still expose a Python wrapper (stat transforms
# constructed by per-layer resolution / `+` composition / figure desugars).
_LIVE_TRANSFORM_CLASSES = [
    "Bin",
    "Kde",
    "Aggregate",
    "AggregateOp",
    "Smooth",
    "Violin",
    "Summary",
    "QQ",
    "Linkage",
    "ReferenceLine",
    "PyIdentity",
]


@pytest.mark.parametrize("name", _REMOVED_DATA_CLASSES)
def test_dead_data_class_removed_from_core(name):
    """The dead typed ``Data*`` transform classes are no longer in ``_core``."""
    assert not hasattr(_core, name), f"{name} should have been removed from ferrum._core (SEAM-02)"


@pytest.mark.parametrize("name", _REMOVED_DATA_CLASSES)
def test_dead_data_class_not_in_public_namespace(name):
    """The dead classes were never public and must not have leaked back."""
    assert not hasattr(fm, name), f"{name} must not be in the ferrum namespace"
    assert name not in fm.__all__, f"{name} must not be in ferrum.__all__"


@pytest.mark.parametrize("name", _LIVE_TRANSFORM_CLASSES)
def test_live_transform_class_still_present(name):
    """Removing the dead classes must not disturb the live transform classes."""
    assert hasattr(_core, name), f"{name} should still be exported from ferrum._core"


def test_transform_public_surface_intact():
    """All 17 ``transform_*`` functions remain in the public surface."""
    transform_funcs = [s for s in fm.__all__ if s.startswith("transform_")]
    assert len(transform_funcs) == 17, transform_funcs
    for fn_name in transform_funcs:
        assert callable(getattr(fm, fn_name)), f"{fn_name} must be callable"


# ---------------------------------------------------------------------------
# SEAM-04: unknown transform keys raise (no more silent drop)
# ---------------------------------------------------------------------------


def test_unknown_top_level_key_raises_and_names_field():
    """A mistyped key on a transform dict raises, naming the bad field."""
    bad = {"type": "data_bin", "field": "x", "maxbin": 3}  # typo: maxbin -> maxbins
    with pytest.raises(ValueError) as exc_info:
        fm.Chart(_DF).mark_bar().encode(x="x:Q", y="g:N").transform(bad).to_svg()
    msg = str(exc_info.value)
    assert "maxbin" in msg, f"error must name the bad field; got: {msg!r}"
    # The serde error also lists the valid field set.
    assert "maxbins" in msg, f"error should list valid fields; got: {msg!r}"


def test_unknown_calculate_key_raises():
    """An unknown key on a calculate dict raises naming the field."""
    bad = {"type": "calculate", "as_field": "r", "expr": "datum.x", "extra": 1}
    with pytest.raises(ValueError) as exc_info:
        fm.Chart(_DF).mark_point().encode(x="x:Q", y="y:Q").transform(bad).to_svg()
    assert "extra" in str(exc_info.value)


def test_unknown_aggregate_op_inner_key_raises():
    """A typo in an aggregate op inner dict (AggSpec) raises naming the field."""
    # "fnc" is a typo for "fn" inside the aggregate op dict.
    t = fm.transform_aggregate({"field": "x", "fnc": "mean", "as": "m"}, groupby=["g"])
    with pytest.raises(ValueError) as exc_info:
        fm.Chart(_DF).mark_bar().encode(x="g:N", y="m:Q").transform(t).to_svg()
    assert "fnc" in str(exc_info.value)


def test_unknown_window_op_inner_key_raises():
    """A typo in a window op inner dict (WindowOp) raises naming the field."""
    bad_op = {"op": "rank", "as": "r", "bogus": 1}
    t = fm.transform_window(bad_op, sort=["x"])
    with pytest.raises(ValueError) as exc_info:
        fm.Chart(_DF).mark_point().encode(x="x:Q", y="r:Q").transform(t).to_svg()
    assert "bogus" in str(exc_info.value)


# ---------------------------------------------------------------------------
# SEAM-04 guard: valid transforms still round-trip and render
# ---------------------------------------------------------------------------


def test_valid_transform_bin_still_renders():
    svg = (
        fm.Chart(_DF)
        .mark_bar()
        .encode(x="x:Q", y="g:N")
        .transform(fm.transform_bin("x", maxbins=3))
        .to_svg()
    )
    assert svg, "valid transform_bin must render"


def test_valid_transform_aggregate_still_renders():
    svg = (
        fm.Chart(_DF)
        .mark_bar()
        .encode(x="g:N", y="m:Q")
        .transform(fm.transform_aggregate({"field": "x", "fn": "mean", "as": "m"}, groupby=["g"]))
        .to_svg()
    )
    assert svg, "valid transform_aggregate must render"


def test_valid_transform_calculate_still_renders():
    svg = (
        fm.Chart(_DF)
        .mark_point()
        .encode(x="x:Q", y="ratio:Q")
        .transform(fm.transform_calculate("ratio", "datum.x / datum.y"))
        .to_svg()
    )
    assert svg, "valid transform_calculate must render"


def test_valid_transform_stack_offsets_still_work():
    """All valid stack offsets still render (offset is validated Python-side)."""
    for offset in ("zero", "normalize", "center"):
        svg = (
            fm.Chart(_DF)
            .mark_bar()
            .encode(x="g:N", y="x:Q", color="g:N")
            .transform(fm.transform_stack("x", groupby=["g"], offset=offset))
            .to_svg()
        )
        assert svg, f"transform_stack(offset={offset!r}) must render"
