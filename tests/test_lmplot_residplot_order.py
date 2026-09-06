"""Batch-D T9 round 2: `lmplot`/`residplot`'s `order=` parameter now has a
real effect.

Before T9, `Smooth`'s Rust `"lm"` branch never read its `degree`
parameter, so `lmplot(order=2)` / `residplot(order=2)` silently forwarded
`degree=order` to a fit that ignored it — `order=1` and `order=2` produced
byte-identical output. T9's `"lm"`/`"linear"` + `degree` polynomial
promotion (spec §4.6) makes that forwarding load-bearing: `order=2` now
resolves to a real quadratic fit. This is net-new feature coverage for the
two figure functions (`regplot` delegates to `lmplot` and is not
separately covered), scoped to its own module per the Python
test-file convention (`tests/test_boxen_palette.py`'s module docstring) —
distinct from `tests/test_smooth_methods.py`, which scopes to `Smooth`/
`mark_smooth` routing itself. Both modules share the `curved_xy_df`
fixture in `tests/conftest.py`.

Batch-D T13 round-4 micro-round: `residplot(order=-1)` now raises a typed
`ValueError` (via `_smooth_degree_kwargs`) instead of surfacing an untyped
PyO3 `OverflowError` at the Rust boundary. `regplot` delegates to `lmplot`
and is not separately covered.
"""

from __future__ import annotations

import pytest

import ferrum as fm


def test_lmplot_order_1_differs_from_order_2(curved_xy_df):
    df = curved_xy_df
    order_1 = fm.lmplot(
        df, x="x", y="y", order=1, ci=None, scatter=False, show_metrics=False
    ).to_svg()
    order_2 = fm.lmplot(
        df, x="x", y="y", order=2, ci=None, scatter=False, show_metrics=False
    ).to_svg()
    assert order_1 != order_2


def test_residplot_order_1_differs_from_order_2(curved_xy_df):
    df = curved_xy_df
    order_1 = fm.residplot(df, x="x", y="y", order=1, show_metrics=False, zero_line=False).to_svg()
    order_2 = fm.residplot(df, x="x", y="y", order=2, show_metrics=False, zero_line=False).to_svg()
    assert order_1 != order_2


def test_residplot_negative_order_raises_typed_value_error(curved_xy_df):
    """Batch-D T13 round-4 micro-round: `residplot(order=-1)` used to surface
    an untyped PyO3 `OverflowError` when the Rust `degree` parameter (unsigned)
    tried to coerce the negative Python int. Now `_smooth_degree_kwargs`
    validates negative `order` directly and raises a typed `ValueError` with a
    clear message, matching `lmplot`'s behavior for the same case."""
    with pytest.raises(ValueError, match="non-negative integer"):
        fm.residplot(curved_xy_df, x="x", y="y", order=-1).to_svg()


def test_residplot_negative_order_message_names_residplot(curved_xy_df):
    """The error message for `residplot(order=-1)` must name `residplot(...)`,
    not `lmplot(...)`  -- the function name is now threaded through
    `_smooth_degree_kwargs` so the message matches the actual caller."""
    with pytest.raises(ValueError, match="residplot\\("):
        fm.residplot(curved_xy_df, x="x", y="y", order=-1).to_svg()
