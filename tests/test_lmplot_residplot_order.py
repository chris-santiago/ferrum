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
"""

from __future__ import annotations

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
