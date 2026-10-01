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


def test_swarm_then_aggregate_warns_exactly_once_even_when_faceted():
    """`#161`: an ``Aggregate``-style transform after ``Swarm``
    collapses rows, dropping the placeholder offset column the renderer
    needs to place dots. That is a degradation (the swarm layout is lost,
    not an error), so it warns instead of silently leaving dots stacked at
    the placeholder ``0.0`` offset. The warning is deduped per chart, not
    per facet panel, so a 3-panel facet must still emit exactly one.
    """
    values = _spread_values()
    n = len(values)
    third = n // 3
    panels = (["p1"] * third) + (["p2"] * third) + (["p3"] * (n - 2 * third))
    df = pl.DataFrame({"group": ["A"] * n, "value": values, "panel": panels})

    with pytest.warns(UserWarning) as caught:
        fm.Chart(df).transform(
            fm.Swarm(category="group", value="value", point_size=8.0, spacing=0.0),
            fm.transform_aggregate(
                {"field": "value", "fn": "mean", "as": "value"}, groupby=["group"]
            ),
        ).mark_point().encode(x="group:N", y="value").facet(col="panel").to_svg()

    assert len(caught) == 1, [str(w.message) for w in caught]
    assert str(caught[0].message) == (
        "swarm could not be applied (column '__pos_x_offset__' that "
        "Swarm(category='group', value='value') places its dots by is not "
        "in the layer's data; a later transform dropped it); marks were "
        "not offset"
    )
