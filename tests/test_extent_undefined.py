"""A `Summary` or `ErrorExtent` group with exactly one observation has an
undefined interval -- there is no spread to estimate -- and now names that
cause once, through `TransformWarning::ExtentUndefined`, instead of letting
the null-position census report the resulting NaN bounds in the caller's
own voice (and three times over for `mark_errorbar`, once per internal
layer). The mark still draws every group's point or bar; only the interval
itself is missing for the one-observation group.
"""

from __future__ import annotations

import warnings
from typing import Callable

import polars as pl
import pytest

import ferrum as fm


def _render_capturing(render: Callable[[], str]) -> tuple[str, list[str]]:
    """Render inside a warning capture: the output and every warning raised."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = render()
    return out, [str(w.message) for w in caught]


@pytest.mark.parametrize("kind", ["bar", "point"])
def test_a_pre_aggregated_catplot_names_its_single_observation_groups_once(kind: str):
    """Every category already holds one row, so `Summary` cannot estimate a
    confidence interval for any of them; one warning lists all three, and
    the bars (or points) still draw."""
    df = pl.DataFrame({"cat": ["a", "b", "c"], "val": [1.0, 2.0, 3.0]})
    svg, messages = _render_capturing(lambda: fm.catplot(df, x="cat", y="val", kind=kind).to_svg())
    assert messages == [
        "Summary: groups cat=a, cat=b, and cat=c each hold one observation, so their "
        "ci=0.95 intervals are undefined and are not drawn"
    ]
    assert ">a<" in svg and ">b<" in svg and ">c<" in svg


def test_a_hued_catplot_lists_each_category_and_hue_group():
    """`Summary` groups by category and hue together, so every (cat, hue)
    pair with one row names itself; the names hold a comma, so the list
    joins with semicolons instead of commas."""
    df = pl.DataFrame(
        {"cat": ["a", "a", "b", "b"], "h": ["m", "f", "m", "f"], "val": [1.0, 2.0, 3.0, 4.0]}
    )
    svg, messages = _render_capturing(
        lambda: fm.catplot(df, x="cat", y="val", hue="h", kind="bar").to_svg()
    )
    assert messages == [
        "Summary: groups cat=a, h=m; cat=a, h=f; cat=b, h=m; and cat=b, h=f each hold "
        "one observation, so their ci=0.95 intervals are undefined and are not drawn"
    ]


def test_only_the_single_observation_group_is_named():
    """Category `b` and `c` hold two observations each and draw their
    interval normally; only `a` is named."""
    df = pl.DataFrame({"cat": ["a", "b", "b", "c", "c"], "val": [1.0, 2.0, 3.0, 4.0, 5.0]})
    svg, messages = _render_capturing(lambda: fm.catplot(df, x="cat", y="val", kind="bar").to_svg())
    assert messages == [
        "Summary: group cat=a holds one observation, so its ci=0.95 interval is "
        "undefined and is not drawn"
    ]


def test_mark_errorbar_with_one_observation_per_x_warns_once():
    """`mark_errorbar` reads `ErrorExtent`, which gives the same report in
    its own name, once for the whole call rather than once per internal
    layer (rule, tick caps)."""
    df = pl.DataFrame({"cat": ["a", "b", "c"], "val": [1.0, 2.0, 3.0]})
    chart = fm.Chart(df).mark_errorbar().encode(x="cat", y="val")
    svg, messages = _render_capturing(chart.to_svg)
    assert messages == [
        "ErrorExtent: groups cat=a, cat=b, and cat=c each hold one observation, so "
        "their ci=0.95 intervals are undefined and are not drawn"
    ]


def test_mark_errorband_names_its_groups_once():
    """`mark_errorband`'s `ErrorExtent` groups by category and color
    together, exactly as `Summary` does for a hued catplot."""
    df = pl.DataFrame(
        {"cat": ["a", "a", "b", "b"], "g": ["m", "f", "m", "f"], "val": [1.0, 2.0, 3.0, 4.0]}
    )
    chart = fm.Chart(df).mark_errorband().encode(x="cat", y="val", color="g")
    svg, messages = _render_capturing(chart.to_svg)
    assert messages == [
        "ErrorExtent: groups cat=a, g=m; cat=a, g=f; cat=b, g=m; and cat=b, g=f each "
        "hold one observation, so their ci=0.95 intervals are undefined and are not drawn"
    ]


def test_each_facet_panel_names_its_own_groups():
    """Panel `p`'s single-observation category is `a`, panel `q`'s is `b`;
    each panel gets its own warning, with its own facet clause."""
    df = pl.DataFrame(
        {
            "f": ["p", "p", "p", "q", "q", "q"],
            "cat": ["a", "b", "b", "a", "a", "b"],
            "val": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    chart = fm.catplot(df, x="cat", y="val", kind="bar", col="f")
    svg, messages = _render_capturing(chart.to_svg)
    assert messages == [
        "Summary: group cat=a in facet panel f=p holds one observation, so its "
        "ci=0.95 interval is undefined and is not drawn",
        "Summary: group cat=b in facet panel f=q holds one observation, so its "
        "ci=0.95 interval is undefined and is not drawn",
    ]
