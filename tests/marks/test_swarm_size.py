"""`#161`: ``mark_swarm(size=s)`` draws dots at radius ``s / 2`` -- the same
diameter they are placed at render time -- on every route into
``desugar_swarm``: ``mark_swarm`` directly, ``catplot(kind="swarm")``, and the
boxplot + swarm composite, on both the vertical and ``orient="horizontal"``
branches. An explicit ``layer_kwargs={"point": {"size": a}}`` or an encoded
``size`` channel still outranks the desugar's own default -- the encoded case
is checked against the exact radii a plain ``mark_point`` with the same
``size=`` encoding produces, not merely that the radii vary.

``test_dense_column_minimum_centre_gap_is_size_plus_spacing`` pins the basic
centre-to-centre contract (``size + spacing``) on one dense, same-valued
category, where every placed dot shares a ``cy`` and the cross-axis spacing
is the only thing moving.

``test_rendered_centres_sit_size_plus_spacing_apart`` extends that contract to
a spread column across ten layouts (default, a title, shared-y facets, an
explicit y domain, a log y scale, horizontal orientation, a two-group dodge,
and a ``Swarm`` transform followed by ``transform_filter``,
``transform_calculate`` or ``transform_sample``). Packing happens at render
time, in rendered pixels through the panel's resolved scale, not in the
transform and not from a pre-layout estimate, so the same ``size + spacing``
minimum centre distance must hold exactly regardless of what shrinks or
reshapes the plot area after the estimate was taken, or what later transform
the layer's data passes through on the way to the renderer.

``test_swarm_then_aggregate_warns_exactly_once_even_when_faceted`` pins the
companion degradation: a later transform that collapses rows (``Aggregate``)
drops the placeholder column the renderer packs by, so the renderer warns
instead of silently leaving dots stacked at the placeholder offset, exactly
once per chart even when the chart is faceted.
"""

from __future__ import annotations

import warnings

import itertools
import math
import re

import numpy as np
import polars as pl
import pytest

import ferrum as fm

_CIRCLE_R_RE = re.compile(r'<circle[^>]*\sr="([0-9.]+)"')
_CIRCLE_TAG_RE = re.compile(r"<circle[^>]*/>")
_ATTR_RE = {
    "cx": re.compile(r'\scx="([0-9.]+)"'),
    "cy": re.compile(r'\scy="([0-9.]+)"'),
    "r": re.compile(r'\sr="([0-9.]+)"'),
}


def _radii(svg: str) -> list[float]:
    radii = [float(v) for v in _CIRCLE_R_RE.findall(svg)]
    assert radii, "expected at least one rendered <circle>"
    return radii


def _circles(svg: str) -> list[tuple[float, float, float]]:
    """Every rendered dot as ``(cx, cy, r)``, attribute-order independent."""
    tags = _CIRCLE_TAG_RE.findall(svg)
    assert tags, "expected at least one rendered <circle>"
    points = []
    for tag in tags:
        cx = float(_ATTR_RE["cx"].search(tag).group(1))
        cy = float(_ATTR_RE["cy"].search(tag).group(1))
        r = float(_ATTR_RE["r"].search(tag).group(1))
        points.append((cx, cy, r))
    return points


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


def test_dense_column_minimum_centre_gap_is_size_plus_spacing():
    """``#161``: the swarm's documented render-time placement contract is a
    minimum centre-to-centre distance of ``size + spacing`` px between
    placed neighbours (edge gap = ``spacing``).

    One dense category ("A", 20 equal-valued rows) is placed purely by its
    x-offset -- every dot shares the same ``cy``, so sorting those dots by
    ``cx`` isolates the cross-axis spacing with no y-axis interference.
    Placement packs in rendered pixel space through the panel's resolved
    value-axis scale, so the ``size + spacing`` centre distance holds exactly
    here too even though a single flat category gives that scale a
    zero-span domain.

    Tolerance: the SVG writer formats ``cx``/``cy`` to 3 decimal places
    (e.g. ``cx="204.062"``), so a centre-distance computed from two such
    values carries at most 2 * 0.0005 px of rounding; ``abs=0.01`` is a
    10x margin on that.
    """
    size, spacing = 5.0, 1.0
    df = pl.DataFrame(
        {
            "group": ["A"] * 20,
            "value": [10.0] * 20,
        }
    )
    svg = fm.Chart(df).mark_swarm(size=size, spacing=spacing).encode(x="group", y="value").to_svg()
    points = _circles(svg)

    # Isolate category "A": the 20 dots sharing the most common cy.
    cy_counts: dict[float, int] = {}
    for _, cy, _ in points:
        key = round(cy, 3)
        cy_counts[key] = cy_counts.get(key, 0) + 1
    dense_cy = max(cy_counts, key=lambda k: cy_counts[k])
    dense = sorted((cx, r) for cx, cy, r in points if round(cy, 3) == dense_cy)
    assert len(dense) == 20

    gaps = [dense[i + 1][0] - dense[i][0] for i in range(len(dense) - 1)]
    min_centre_dist = min(gaps)
    # Edge gap from the drawn radii, not from size, so it is an independent check.
    min_edge_gap = min(
        (dense[i + 1][0] - dense[i + 1][1]) - (dense[i][0] + dense[i][1])
        for i in range(len(dense) - 1)
    )

    expected_centre_dist = size + spacing

    assert min_centre_dist == pytest.approx(expected_centre_dist, abs=0.01)
    assert min_edge_gap == pytest.approx(spacing, abs=0.01)


# ---------------------------------------------------------------------------
# Render-time packing: the same centre distance holds under every layout
# that reshapes the plot area after a pre-layout estimate would have been
# taken (`#161`).
# ---------------------------------------------------------------------------


def _spread_values(n: int = 300, seed: int = 0) -> list[float]:
    """300 seeded normal draws: enough points, spread widely enough, that a
    render-time packing defect (converting pixels to data units from a
    pre-layout estimate instead of the panel's resolved scale) produces a
    measurably different minimum rendered centre distance than the exact
    ``size + spacing`` the renderer packs to at render time."""
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1, n).tolist()


def _layout_default(values: list[float], size: float, spacing: float) -> str:
    df = pl.DataFrame({"group": ["A"] * len(values), "value": values})
    return fm.Chart(df).mark_swarm(size=size, spacing=spacing).encode(x="group", y="value").to_svg()


def _layout_title(values: list[float], size: float, spacing: float) -> str:
    df = pl.DataFrame({"group": ["A"] * len(values), "value": values})
    return (
        fm.Chart(df)
        .mark_swarm(size=size, spacing=spacing)
        .encode(x="group", y="value")
        .properties(title="Swarm")
        .to_svg()
    )


def _layout_shared_y_facet(values: list[float], size: float, spacing: float) -> str:
    n = len(values)
    third = n // 3
    panels = (["p1"] * third) + (["p2"] * third) + (["p3"] * (n - 2 * third))
    df = pl.DataFrame({"group": ["A"] * n, "value": values, "panel": panels})
    return (
        fm.Chart(df)
        .mark_swarm(size=size, spacing=spacing)
        .encode(x="group", y="value")
        .facet(col="panel")
        .to_svg()
    )


def _layout_explicit_y_domain(values: list[float], size: float, spacing: float) -> str:
    df = pl.DataFrame({"group": ["A"] * len(values), "value": values})
    domain = [min(values) - 10.0, max(values) + 10.0]
    return (
        fm.Chart(df)
        .mark_swarm(size=size, spacing=spacing)
        .encode(x="group", y=fm.Y("value", scale={"domain": domain}))
        .to_svg()
    )


def _layout_log_y(values: list[float], size: float, spacing: float) -> str:
    # A log scale needs a strictly positive domain; shift the spread
    # positive without changing the distances between its values.
    shifted = [v - min(values) + 1.0 for v in values]
    df = pl.DataFrame({"group": ["A"] * len(shifted), "value": shifted})
    return (
        fm.Chart(df)
        .mark_swarm(size=size, spacing=spacing)
        .encode(x="group", y=fm.Y("value", scale={"type": "log"}))
        .to_svg()
    )


def _layout_horizontal(values: list[float], size: float, spacing: float) -> str:
    df = pl.DataFrame({"group": ["A"] * len(values), "value": values})
    return (
        fm.Chart(df)
        .mark_swarm(size=size, spacing=spacing, orient="horizontal")
        .encode(x="value", y="group")
        .to_svg()
    )


def _layout_dodge(values: list[float], size: float, spacing: float) -> str:
    n = len(values)
    hues = (["h1", "h2"] * ((n + 1) // 2))[:n]
    df = pl.DataFrame({"group": ["A"] * n, "value": values, "hue": hues})
    return (
        fm.Chart(df)
        .mark_swarm(size=size, spacing=spacing, dodge="hue")
        .encode(x="group", y="value")
        .to_svg()
    )


def _layout_then_filter(values: list[float], size: float, spacing: float) -> str:
    """A later ``transform_filter`` drops rows after the ``Swarm``, which
    must not stop the renderer from finding the Swarm in the layer's
    lineage and packing the surviving dots (`#161`)."""
    df = pl.DataFrame({"group": ["A"] * len(values), "value": values})
    return (
        fm.Chart(df)
        .transform(
            fm.Swarm(category="group", value="value", point_size=size, spacing=spacing),
            fm.transform_filter("datum.value > 0"),
        )
        .mark_point()
        .encode(x="group:N", y="value")
        .to_svg()
    )


def _layout_then_calculate(values: list[float], size: float, spacing: float) -> str:
    """A later ``transform_calculate`` adds a column without dropping any
    rows; packing must still resolve through the Swarm in the lineage."""
    df = pl.DataFrame({"group": ["A"] * len(values), "value": values})
    return (
        fm.Chart(df)
        .transform(
            fm.Swarm(category="group", value="value", point_size=size, spacing=spacing),
            fm.transform_calculate("doubled", "datum.value * 2"),
        )
        .mark_point()
        .encode(x="group:N", y="value")
        .to_svg()
    )


def _layout_then_sample(values: list[float], size: float, spacing: float) -> str:
    """A later ``transform_sample`` drops rows by subsampling rather than
    filtering; same lineage requirement as ``_layout_then_filter``."""
    df = pl.DataFrame({"group": ["A"] * len(values), "value": values})
    return (
        fm.Chart(df)
        .transform(
            fm.Swarm(category="group", value="value", point_size=size, spacing=spacing),
            fm.transform_sample(n=len(values) // 2),
        )
        .mark_point()
        .encode(x="group:N", y="value")
        .to_svg()
    )


_LAYOUTS = {
    "default": _layout_default,
    "title": _layout_title,
    "shared_y_facet": _layout_shared_y_facet,
    "explicit_y_domain": _layout_explicit_y_domain,
    "log_y": _layout_log_y,
    "horizontal": _layout_horizontal,
    "dodge": _layout_dodge,
    "then_filter": _layout_then_filter,
    "then_calculate": _layout_then_calculate,
    "then_sample": _layout_then_sample,
}


@pytest.mark.parametrize("size,spacing", [(8.0, 0.0), (5.0, 2.0)], ids=["8,0", "5,2"])
@pytest.mark.parametrize("layout", sorted(_LAYOUTS))
def test_rendered_centres_sit_size_plus_spacing_apart(layout, size, spacing):
    """``#161``: the renderer packs swarm dots in RENDERED pixels, through
    the panel's resolved scale, not in the transform and not from a
    pre-layout estimate -- so the minimum
    rendered centre distance is ``size + spacing`` in every layout that
    changes the pixels-per-data-unit a pre-layout estimate could not see: a
    title or facet strip shrinking the plot area, a facet's shared y domain,
    an explicit y domain wider than the data, a log y scale's uneven pixel
    density, a horizontal cross-axis, and a two-group dodge band. It also
    covers a ``Swarm`` transform followed by a later row-dropping or
    row-preserving transform (``transform_filter``, ``transform_calculate``,
    ``transform_sample``): the renderer must still find the ``Swarm`` in the
    layer's data lineage and pack the rows it actually holds, rather than
    leaving them at the transform's ``0.0`` placeholder offset (`#161`).

    Both bounds matter: the lower bound alone would pass an over-spread
    render that merely avoids overlap without honoring the exact packing
    distance, so the upper bound (tightness) rules that out too -- a render
    whose minimum distance sat well above ``size + spacing`` would still
    fail here even though nothing overlaps.
    """
    values = _spread_values()
    svg = _LAYOUTS[layout](values, size, spacing)
    points = _circles(svg)
    min_dist = min(
        math.hypot(ax - bx, ay - by)
        for (ax, ay, _), (bx, by, _) in itertools.combinations(points, 2)
    )
    target = size + spacing

    assert min_dist >= target - 0.01, (
        f"{layout} (size={size}, spacing={spacing}): rendered dots overlap past "
        f"size + spacing -- min centre distance {min_dist:.4f} < {target}"
    )
    assert min_dist <= target + 0.1, (
        f"{layout} (size={size}, spacing={spacing}): rendered dots sit further apart "
        f"than size + spacing -- min centre distance {min_dist:.4f} > {target}, so "
        "the packing over-spread instead of packing tightly to the target"
    )


def _swarm_then_mean(as_name: str) -> list[str]:
    """Warnings from a 3-panel facet of ``Swarm`` followed by a mean aggregate
    writing its result to *as_name*."""
    values = _spread_values()
    n = len(values)
    third = n // 3
    panels = (["p1"] * third) + (["p2"] * third) + (["p3"] * (n - 2 * third))
    df = pl.DataFrame({"group": ["A"] * n, "value": values, "panel": panels})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fm.Chart(df).transform(
            fm.Swarm(category="group", value="value", point_size=8.0, spacing=0.0),
            fm.transform_aggregate(
                {"field": "value", "fn": "mean", "as": as_name}, groupby=["group"]
            ),
        ).mark_point().encode(x="group:N", y=as_name).facet(col="panel").to_svg()
    return [str(w.message) for w in caught]


def test_swarm_then_an_aggregate_keeping_its_columns_is_placed_without_a_warning():
    """`#161`: placement packs whichever rows the layer draws, so an aggregate
    after ``Swarm`` that keeps the category and value columns is placed, and
    nothing was dropped to warn about."""
    assert _swarm_then_mean("value") == []


def test_swarm_then_an_aggregate_dropping_its_value_warns_exactly_once_when_faceted():
    """An aggregate that drops the value column ``Swarm`` places by is a
    degradation (the layout is lost, not an error): it warns, naming the
    column, once per chart rather than once per facet panel."""
    assert _swarm_then_mean("mean_v") == [
        "swarm could not be applied (column 'value' that "
        "Swarm(category='group', value='value') places its dots by is not "
        "in the layer's data; a later transform dropped it); marks were "
        "not offset"
    ]


def test_hand_built_swarm_defaults_draw_exactly_like_mark_swarm():
    """``fm.Swarm``'s default ``point_size`` is ``mark_swarm``'s default
    ``size=4`` (user ruling 2026-10-04), so a hand-built
    ``transform(fm.Swarm(...)).mark_point()`` places and draws every dot
    exactly where ``mark_swarm()`` does. Control: an explicit
    ``point_size=5`` (the former default) moves the dots, so the comparison
    can tell the two defaults apart."""
    values = _spread_values()
    df = pl.DataFrame({"group": ["A"] * len(values), "value": values})

    def hand_built(**swarm_kwargs) -> list[tuple[float, float, float]]:
        chart = (
            fm.Chart(df)
            .transform(fm.Swarm(category="group", value="value", **swarm_kwargs))
            .mark_point()
            .encode(x="group:N", y="value")
        )
        return sorted(_circles(chart.to_svg()))

    composite = sorted(_circles(fm.Chart(df).mark_swarm().encode(x="group:N", y="value").to_svg()))
    assert {r for _, _, r in composite} == {2.0}
    assert hand_built() == composite
    assert hand_built(point_size=5.0) != composite


# ---------------------------------------------------------------------------
# A layer's own size: placement packs at the diameter the dot is drawn at, on
# every door that reaches a swarm-fed point layer.
# ---------------------------------------------------------------------------

_OWN_AREA = 40.0


def _spread_df() -> pl.DataFrame:
    values = _spread_values()
    return pl.DataFrame({"group": ["A"] * len(values), "value": values})


def _door_catplot(df: pl.DataFrame, area: float | None):
    mark = None if area is None else {"point": {"size": area}}
    return fm.catplot(df, x="group", y="value", kind="swarm", mark=mark)


def _door_layer_kwargs(df: pl.DataFrame, area: float | None):
    layer_kwargs = None if area is None else {"point": {"size": area}}
    return fm.Chart(df).mark_swarm(layer_kwargs=layer_kwargs).encode(x="group", y="value")


def _door_hand_built(df: pl.DataFrame, area: float | None):
    point_kwargs = {} if area is None else {"size": area}
    return (
        fm.Chart(df)
        .transform(fm.Swarm(category="group", value="value"))
        .mark_point(**point_kwargs)
        .encode(x="group:N", y="value")
    )


def _door_layer(df: pl.DataFrame, area: float | None):
    """An ``fm.Layer`` attached beside the chart's own swarm-fed layer, which
    draws squares so the measured circles are the ``fm.Layer``'s alone. Both
    layers are fed by the one swarm, so the squares sit on the circles'
    centres (``test_every_layer_fed_by_one_swarm_shares_its_centres``)."""
    mark_kwargs = None if area is None else {"size": area}
    return (
        fm.Chart(df)
        .transform(fm.Swarm(category="group", value="value"))
        .mark_point(shape="square")
        .encode(x="group:N", y="value")
        .layer(fm.Layer(mark="point", mark_kwargs=mark_kwargs))
    )


_DOORS = {
    "catplot_mark": _door_catplot,
    "mark_swarm_layer_kwargs": _door_layer_kwargs,
    "hand_built_swarm": _door_hand_built,
    "fm_layer": _door_layer,
}


def _min_centre_distance(points: list[tuple[float, float, float]]) -> float:
    return min(
        math.hypot(ax - bx, ay - by)
        for (ax, ay, _), (bx, by, _) in itertools.combinations(points, 2)
    )


@pytest.mark.parametrize("door", sorted(_DOORS))
def test_a_layer_size_packs_at_the_diameter_it_draws(door):
    """A point ``size`` on the swarm-fed layer is an area ``a``; the dot is
    drawn ``2 * sqrt(a / pi)`` across and placement packs at that diameter,
    so the minimum centre distance is the drawn diameter plus the default
    ``spacing`` of 1, on every door. Control: the same door with no size of
    its own draws and packs at the swarm's default 4 px."""
    df = _spread_df()
    points = _circles(_DOORS[door](df, _OWN_AREA).to_svg())
    drawn = 2.0 * max(r for _, _, r in points)
    assert drawn == pytest.approx(2.0 * math.sqrt(_OWN_AREA / math.pi), abs=2e-3)
    assert _min_centre_distance(points) == pytest.approx(drawn + 1.0, abs=0.01)

    control = _circles(_DOORS[door](df, None).to_svg())
    assert {r for _, _, r in control} == {2.0}
    assert _min_centre_distance(control) == pytest.approx(5.0, abs=0.01)


_RANGE_MAX = 2.0 * math.sqrt(36.0 / math.pi)


def _short_weights(n: int) -> list[float]:
    """Size values whose niced scale domain ends past the data, so no row maps
    to the range's largest area (36) and the widest dot drawn stops short."""
    return [0.9 * (i % 7) for i in range(n)]


def test_a_size_encoding_packs_at_the_size_range_maximum():
    """Under a size encoding the swarm packs at the size range's largest area
    (the theme's ``point_size_max`` of 36), read from the scale, not the data:
    here no row reaches it, yet centres sit one range-max diameter plus
    ``spacing`` apart, as they would in any facet panel whose data did."""
    df = _spread_df().with_columns(pl.Series("weight", _short_weights(300)))
    points = _circles(
        fm.Chart(df)
        .mark_swarm()
        .encode(x="group", y="value", size=fm.Size("weight", legend=None))
        .to_svg()
    )
    radii = {round(r, 3) for _, _, r in points}
    assert len(points) == 300
    assert len(radii) > 1
    assert 2.0 * max(radii) < _RANGE_MAX - 0.05, "the data stops short of the range max"
    assert _min_centre_distance(points) == pytest.approx(_RANGE_MAX + 1.0, abs=0.01)


_OVERFLOW_RE = re.compile(r"packs its dots up to ([0-9.]+) px")


def _overflow_reaches(chart) -> list[float]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        chart.to_svg()
    return [float(m.group(1)) for w in caught if (m := _OVERFLOW_RE.search(str(w.message)))]


@pytest.mark.parametrize("door", sorted(_DOORS))
def test_a_layer_size_that_overflows_the_slot_warns(door):
    """Eight equal values a category pack four to a side. At the swarm's own
    4 px they fit their slot; a layer area of ``400 * pi`` draws 40 px dots
    that reach past it, so exactly one overflow warning fires, measuring the
    reach at the drawn diameter: four steps of 41 px plus a 20 px radius."""
    df = pl.DataFrame({"group": ["a"] * 8 + ["b"] * 8, "value": [5.0] * 16})
    assert _overflow_reaches(_DOORS[door](df, None)) == []
    assert _overflow_reaches(_DOORS[door](df, 400.0 * math.pi)) == [pytest.approx(184.0, abs=0.05)]


# ---------------------------------------------------------------------------
# One packing per swarm per panel: every layer one swarm feeds sits on the
# same centres, packed at the largest bound of the point layers it feeds.
# ---------------------------------------------------------------------------

_SQUARE_RE = re.compile(
    r'<rect x="([0-9.]+)" y="([0-9.]+)" width="([0-9.]+)" height="\3"[^>]*\sfill='
)
_LABEL_RE = re.compile(r'<text x="([0-9.]+)" y="([0-9.]+)"[^>]*>L\d+</text>')


def _square_centres(svg: str) -> list[tuple[float, float]]:
    squares = [tuple(float(v) for v in m) for m in _SQUARE_RE.findall(svg)]
    return sorted((x + w / 2.0, y + w / 2.0) for x, y, w in squares if w < 100.0)


def _labelled_df() -> pl.DataFrame:
    df = _spread_df().head(60)
    return df.with_columns(pl.Series("lab", [f"L{i}" for i in range(len(df))]))


def test_every_layer_fed_by_one_swarm_shares_its_centres():
    """A hand-built ``Swarm`` feeding a ``mark_point(size=200)`` and a text
    layer packs once, at the sized dots' diameter, so every label sits on its
    dot: the label x equals the dot cx and the label y is the dot cy plus one
    constant baseline shift. The ``fm.Layer`` door's own-sized circles and the
    chart's unsized squares likewise share every centre."""
    df = _labelled_df()
    svg = (
        fm.Chart(df)
        .transform(fm.Swarm(category="group", value="value"))
        .mark_point(size=200.0)
        .encode(x="group:N", y="value")
        .layer(fm.Layer(mark="text", encoding={"text": "lab"}))
        .to_svg()
    )
    dots = sorted((cx, cy) for cx, cy, _ in _circles(svg))
    labels = sorted((float(x), float(y)) for x, y in _LABEL_RE.findall(svg))
    assert len(labels) == len(dots) == len(df)
    assert [x for x, _ in labels] == pytest.approx([x for x, _ in dots], abs=1e-3)
    shifts = [ly - dy for (_, ly), (_, dy) in zip(labels, dots)]
    assert max(shifts) - min(shifts) == pytest.approx(0.0, abs=1e-3)
    sized = 2.0 * math.sqrt(200.0 / math.pi)
    assert _min_centre_distance(_circles(svg)) == pytest.approx(sized + 1.0, abs=0.01)

    layered = _door_layer(_spread_df(), _OWN_AREA).to_svg()
    circles = sorted((cx, cy) for cx, cy, _ in _circles(layered))
    squares = _square_centres(layered)
    assert len(squares) == len(circles) == 300
    # The rect corner and width are each rounded to 3 decimals, so a square's
    # centre is within 1e-3 px of the circle it sits on.
    for sx, sy in squares:
        assert min(math.hypot(sx - cx, sy - cy) for cx, cy in circles) < 2e-3


def _null_size_df() -> pl.DataFrame:
    df = _spread_df().head(60)
    weights = [None, *_short_weights(len(df))[1:]]
    return df.with_columns(pl.Series("weight", weights, dtype=pl.Float64))


@pytest.mark.parametrize("fallback", ["layer_size", "theme_point_size"])
def test_a_null_size_fallback_wider_than_the_range_max_packs_at_the_fallback(fallback):
    """Under a size encoding a row whose size is null draws at the layer's
    fallback area, its own constant ``size`` or else the theme's
    ``point_size``. At 400 that is wider than the range's largest area (36),
    so the swarm packs at the fallback diameter and even that dot overlaps
    nothing."""
    area = 400.0
    chart = (
        fm.Chart(_null_size_df())
        .transform(fm.Swarm(category="group", value="value"))
        .mark_point(**({"size": area} if fallback == "layer_size" else {}))
        .encode(x="group:N", y="value", size=fm.Size("weight", legend=None))
    )
    if fallback == "theme_point_size":
        chart = chart.theme(fm.Theme(point_size=area))
    points = _circles(chart.to_svg())
    widest = 2.0 * math.sqrt(area / math.pi)
    assert widest > 2.0 * math.sqrt(36.0 / math.pi)
    assert len(points) == 60
    assert 2.0 * max(r for _, _, r in points) == pytest.approx(widest, abs=2e-3)
    assert len({round(r, 3) for _, _, r in points}) > 1
    assert _min_centre_distance(points) == pytest.approx(widest + 1.0, abs=0.01)


def test_an_unsized_sibling_draws_at_the_swarm_size_while_sharing_centres():
    """One swarm feeds sized squares (``size=100``) and an unsized ``fm.Layer``
    of circles. The swarm packs once, at the squares' diameter, so the circles
    sit on the squares' centres, but each layer draws at its own size: the
    unsized circles keep the swarm's own 4 px, not the shared packing
    diameter."""
    svg = (
        fm.Chart(_spread_df())
        .transform(fm.Swarm(category="group", value="value"))
        .mark_point(shape="square", size=100.0)
        .encode(x="group:N", y="value")
        .layer(fm.Layer(mark="point"))
        .to_svg()
    )
    circles = _circles(svg)
    squares = _square_centres(svg)
    assert len(squares) == len(circles) == 300
    assert {r for _, _, r in circles} == {2.0}
    for sx, sy in squares:
        assert min(math.hypot(sx - cx, sy - cy) for cx, cy, _ in circles) < 2e-3
    sized = 2.0 * math.sqrt(100.0 / math.pi)
    assert _min_centre_distance(circles) == pytest.approx(sized + 1.0, abs=0.01)


def _unsized_beside_smaller(df: pl.DataFrame):
    """The quality reviewer's probe: sized squares smaller than the swarm's
    own 4 px (``size=4``, about 2.26 px across) beside an unsized
    ``fm.Layer`` of circles, which draws at the swarm's 4 px."""
    return (
        fm.Chart(df)
        .transform(fm.Swarm(category="group", value="value"))
        .mark_point(shape="square", size=4.0)
        .encode(x="group:N", y="value")
        .layer(fm.Layer(mark="point"))
    )


def test_an_unsized_layer_beside_a_smaller_sized_sibling_overlaps_nothing():
    """An unsized layer bounds the packing at the swarm's own 4 px, so a
    sized sibling smaller than that cannot pull centres closer than the
    unsized dots are drawn: no two centres sit nearer than 4 + 1 px. The
    spread column only bounds that from below; packing at the squares'
    2.26 px would put neighbours about 3.27 px apart. The overflow reach
    is measured at that diameter too: 120 equal values pack 60 steps of
    5 px to one side, plus the 2 px radius."""
    values = [i * 0.01 for i in range(120)]
    df = pl.DataFrame({"group": ["A"] * 120, "value": values})
    circles = _circles(_unsized_beside_smaller(df).to_svg())
    assert len(circles) == 120
    assert {r for _, _, r in circles} == {2.0}
    assert _min_centre_distance(circles) >= 5.0 - 0.002

    stacked = pl.DataFrame({"group": ["a"] * 120 + ["b"] * 120, "value": [5.0] * 240})
    assert _overflow_reaches(_unsized_beside_smaller(stacked)) == [pytest.approx(302.0, abs=0.05)]


@pytest.mark.parametrize("fallback", ["layer_size", "theme_point_size"])
def test_a_size_encoding_with_no_null_rows_packs_at_the_range_max(fallback):
    """The fallback area (the layer's own ``size``, else the theme's
    ``point_size``) is drawn only by a row whose size is null or maps
    nowhere. With no such row, a fallback of 400 wider than the range's
    largest area (36) never widens the packing: centres sit one range-max
    diameter plus ``spacing`` apart, though no row reaches the range max."""
    df = _null_size_df().with_columns(pl.col("weight").fill_null(0.9))
    chart = (
        fm.Chart(df)
        .transform(fm.Swarm(category="group", value="value"))
        .mark_point(**({"size": 400.0} if fallback == "layer_size" else {}))
        .encode(x="group:N", y="value", size=fm.Size("weight", legend=None))
    )
    if fallback == "theme_point_size":
        chart = chart.theme(fm.Theme(point_size=400.0))
    points = _circles(chart.to_svg())
    assert len(points) == 60
    assert 2.0 * max(r for _, _, r in points) < _RANGE_MAX - 0.05, (
        "the data stops short of the range max"
    )
    assert _min_centre_distance(points) == pytest.approx(_RANGE_MAX + 1.0, abs=0.01)


# ---------------------------------------------------------------------------
# One packing diameter per swarm per chart: every facet panel packs at the
# bound taken across all panels, so facets share their spacing.
# ---------------------------------------------------------------------------


def _panels(points: list[tuple[float, float, float]]) -> list[list[tuple[float, float, float]]]:
    """Split a two-panel ``col=`` facet's dots at the widest horizontal gap."""
    xs = sorted(points)
    gap = max(range(len(xs) - 1), key=lambda i: xs[i + 1][0] - xs[i][0])
    mid = (xs[gap][0] + xs[gap + 1][0]) / 2.0
    return [[p for p in points if p[0] < mid], [p for p in points if p[0] >= mid]]


def test_a_null_size_row_in_one_facet_widens_every_facet():
    """The quality reviewer's facet probe: a size-encoded swarm with a
    constant ``size=200`` fallback, faceted into panels P and Q, with one null
    size row in P only. The null row draws at the 200 fallback, wider than the
    range max, so the swarm packs at that diameter in both panels: Q, whose
    rows all map, spaces its dots exactly as P does, not at the range max."""
    n = 6
    sizes = [float(i % 5 + 1) for i in range(n)]
    df = pl.DataFrame(
        {
            "f": ["P"] * n + ["Q"] * n,
            "group": ["A"] * (2 * n),
            "value": [5.0] * (2 * n),
            "s": [None, *sizes[1:], *sizes],
        }
    )
    svg = (
        fm.Chart(df)
        .transform(fm.Swarm(category="group", value="value"))
        .mark_point(size=200.0)
        .encode(x="group:N", y="value", size=fm.Size("s", legend=None))
        .facet(col="f")
        .to_svg()
    )
    panels = _panels(_circles(svg))
    assert [len(p) for p in panels] == [n, n]
    fallback = 2.0 * math.sqrt(200.0 / math.pi)
    assert fallback > _RANGE_MAX
    assert [_min_centre_distance(p) for p in panels] == [
        pytest.approx(fallback + 1.0, abs=0.01),
        pytest.approx(fallback + 1.0, abs=0.01),
    ]
