"""`#161`: ``mark_swarm(size=s)`` draws dots at radius ``s / 2`` -- the same
diameter the ``Swarm`` transform places them at -- on every route into
``desugar_swarm``: ``mark_swarm`` directly, ``catplot(kind="swarm")``, and the
boxplot + swarm composite, on both the vertical and ``orient="horizontal"``
branches. An explicit ``layer_kwargs={"point": {"size": a}}`` or an encoded
``size`` channel still outranks the desugar's own default -- the encoded case
is checked against the exact radii a plain ``mark_point`` with the same
``size=`` encoding produces, not merely that the radii vary.
"""

from __future__ import annotations

import math
import re

import polars as pl
import pytest

import ferrum as fm

_CIRCLE_R_RE = re.compile(r'<circle[^>]*\sr="([0-9.]+)"')


def _radii(svg: str) -> list[float]:
    radii = [float(v) for v in _CIRCLE_R_RE.findall(svg)]
    assert radii, "expected at least one rendered <circle>"
    return radii


def _df(n: int = 12) -> pl.DataFrame:
    # Two tight-range groups: no IQR-fenced outliers, so a boxplot built
    # over this data draws no outlier circles of its own.
    return pl.DataFrame(
        {
            "group": (["a"] * n) + (["b"] * n),
            "value": ([float(i % 5) for i in range(n)] * 2),
        }
    )


@pytest.mark.parametrize("size", [2, 4, 20])
def test_mark_swarm_radius_is_half_size(size):
    svg = fm.Chart(_df()).mark_swarm(size=size).encode(x="group", y="value").to_svg()
    for r in _radii(svg):
        assert r == pytest.approx(size / 2), f"size={size} must draw r={size / 2}, got {r}"


@pytest.mark.parametrize("size", [2, 4, 20])
def test_mark_swarm_horizontal_radius_is_half_size(size):
    # orient="horizontal" is the second `_Layer` construction site in
    # desugar_swarm; it must apply the same `point_mark_kwargs` as the
    # vertical branch. Horizontal swarm encodes the value column on x and
    # the category column on y (per mark_swarm's horizontal convention).
    svg = (
        fm.Chart(_df())
        .mark_swarm(size=size, orient="horizontal")
        .encode(x="value", y="group")
        .to_svg()
    )
    for r in _radii(svg):
        assert r == pytest.approx(size / 2), f"size={size} must draw r={size / 2}, got {r}"


def test_catplot_swarm_radius_is_half_default_size():
    # catplot(kind="swarm") has no top-level size= passthrough; it always
    # builds mark_swarm() at the default size=4, which must draw r=2.
    svg = fm.catplot(_df(), x="group", y="value", kind="swarm").to_svg()
    for r in _radii(svg):
        assert r == pytest.approx(2.0)


@pytest.mark.parametrize("size", [2, 4, 20])
def test_boxplot_plus_swarm_composite_radius_is_half_size(size):
    df = _df()
    box = fm.Chart(df).mark_boxplot(outliers=False).encode(x="group", y="value")
    swarm = fm.Chart(df).mark_swarm(size=size).encode(x="group", y="value")
    svg = (box + swarm).to_svg()
    for r in _radii(svg):
        assert r == pytest.approx(size / 2)


def test_layer_kwargs_point_size_still_wins_over_desugar_default():
    # layer_kwargs aims at the "point" sub-layer by name and is merged OVER
    # the desugar's own mark_kwargs (`_aim_layer_kwargs`), so an explicit
    # area wins regardless of size=.
    area = 100.0
    svg = (
        fm.Chart(_df())
        .mark_swarm(size=4, layer_kwargs={"point": {"size": area}})
        .encode(x="group", y="value")
        .to_svg()
    )
    expected_r = math.sqrt(area / math.pi)
    for r in _radii(svg):
        assert r == pytest.approx(expected_r, abs=1e-3)
        assert r != pytest.approx(2.0)


def test_encoded_size_matches_plain_point_scale():
    df = _df()
    df = df.with_columns(pl.Series("weight", [float(1 + (i % 5)) for i in range(len(df))]))
    swarm_svg = fm.Chart(df).mark_swarm(size=4).encode(x="group", y="value", size="weight").to_svg()
    # The same size= encoding on a plain mark_point(), over the same data,
    # drives the same size scale. If desugar_swarm's constant mark_kwargs
    # default silently overrode the encoding, the swarm radii would collapse
    # to a single value instead of matching the point chart's radii exactly.
    point_svg = fm.Chart(df).mark_point().encode(x="group", y="value", size="weight").to_svg()
    swarm_radii = sorted(_radii(swarm_svg))
    point_radii = sorted(_radii(point_svg))
    assert len(set(round(r, 3) for r in swarm_radii)) > 1, (
        "an encoded size= must vary the rendered radius, not collapse to the "
        "desugar's constant default"
    )
    assert swarm_radii == pytest.approx(point_radii, abs=1e-3)


@pytest.mark.parametrize("size", [None, 10.0, 64.0])
def test_plain_mark_point_radius_matches_area_formula(size):
    # desugar_swarm is only reachable through mark_swarm(); a plain
    # mark_point() chart never touches it. Its constant size= is a marker
    # *area*, so the rendered radius is r = sqrt(area / pi) -- size=None
    # pins the theme default area of 36, matching the swarm tests' own
    # r = size / 2 formula being specific to the swarm desugar.
    chart = fm.Chart(_df()).mark_point(**({} if size is None else {"size": size}))
    area = 36.0 if size is None else size
    svg = chart.encode(x="group", y="value").to_svg()
    expected_r = math.sqrt(area / math.pi)
    for r in _radii(svg):
        assert r == pytest.approx(expected_r, abs=1e-3)
