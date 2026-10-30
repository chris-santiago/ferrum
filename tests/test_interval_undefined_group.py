"""A grouped ``lmplot(method="logistic", hue=)`` fit whose Wald variance
cancels at some grid points for every group warns once per group, each
warning naming its own group label rather than reporting the band as
ungrouped.

Net-new coverage for a figure-function behavior; scoped to its own module
per the test-homing convention in CLAUDE.md.
"""

from __future__ import annotations

import warnings

import polars as pl

import ferrum as fm


def test_lmplot_logistic_grouped_interval_undefined_names_each_group():
    """Two hue groups, each shaped like the ungrouped cancellation fixture
    (x centered on a large constant with a small spread relative to its
    magnitude): the fit line still draws for both groups, and each group's
    confidence-band warning names that group, not the other one and not
    neither."""
    # lmplot's dedicated logistic fit evaluates a 200-point grid
    # (`_build_logistic_fit`'s n_grid=200 in src/ferrum/plots/regression.py), not the 100-point
    # grid the ungrouped Rust fixture and its Python mirror use directly
    # against `Logistic`. This offset and spread are what reach the
    # negative-variance branch on a 200-point grid; the smaller ungrouped
    # fixture's magnitudes do not.
    c = 50_000_000.0
    spread = 1.0
    group_x = [c - spread, c - spread, c + spread, c + spread] * 3
    group_y = [0.0, 1.0, 0.0, 1.0] * 3
    df = pl.DataFrame(
        {
            "x": group_x + group_x,
            "y": group_y + group_y,
            "g": ["a"] * len(group_x) + ["b"] * len(group_x),
        }
    )
    chart = fm.lmplot(df, x="x", y="y", method="logistic", hue="g", ci=95)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        svg = chart.to_svg()
        msgs = [str(w.message) for w in caught]

    assert svg.count("<polyline") == 2, "both groups' fit lines must still draw"

    reason = (
        "the Wald variance is negative (x's spread is tiny next to its magnitude; "
        "centering x avoids it) at 2 of 200 grid points"
    )
    expected = [
        f"Logistic: the ci=0.95 band of group g=a is undefined ({reason}); "
        "the fit line is drawn without it",
        f"Logistic: the ci=0.95 band of group g=b is undefined ({reason}); "
        "the fit line is drawn without it",
    ]
    assert sorted(msgs) == sorted(expected), msgs
