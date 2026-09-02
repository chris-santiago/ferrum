"""``hue=`` on the public figure functions is a categorical discriminator.

**What this pins.** Every figure function that accepts ``hue=`` treats it as
a group discriminator and types it Nominal, so the rendered chart does not
depend on whether the group column happens to be ``Int64`` or ``Utf8``. The
assertions are dtype-parity assertions: render the same data twice, once
with an ``Int64`` group column and once with the identical values as
``Utf8``, and require the two renders to agree on the palette and the
legend. That form is deliberate — it is discriminating in a way that a
single-render assertion is not. An untyped numeric discriminator does not
render *nothing*; it renders a plausible-looking chart with a continuous
blues ramp and a colorbar whose ticks (0 / 0.25 / 0.5 / 0.75 / 1) are pure
invention for a two-group column. Only the comparison against the Utf8 twin
makes that visible.

**Why these five.** The Batch-A design review's Cycle 3 executed these
against the built extension and reported them as confirmed-wrong charts:

- ``regplot``/``lmplot`` with ``hue=`` collapsed two regression fits and
  their two CI bands into **one fit and one band**, silently. A materially
  wrong statistical chart on a headline seaborn-parity entry point.
- ``mark_contour(groupby=)`` drew two distributions' isolines in one color,
  silently — inside ``src/ferrum/marks``, the directory an earlier audit had
  claimed complete, because that audit was scoped to the literal token
  ``color_field`` rather than to the defect class.
- ``relplot(kind="line")`` drew one polyline instead of two (this one at
  least warned).
- ``catplot(kind="bar"|"boxen")`` and ``displot(kind="hist")`` rendered a
  fabricated colorbar in place of a categorical swatch legend, silently.

**Why a new module rather than an extension of an existing one.** The
project's test-file convention (see ``tests/test_boxen_palette.py``) keeps
findings-ID-named modules scoped to their finding and gives net-new feature
coverage its own ``test_<feature>.py``. ``tests/test_stroke_dash_rendering.py``
has ``relplot`` fixtures, but for the ``style=``/``stroke_dash`` channel, not
``hue=``; the two are different channels with different failure modes. The
shared SVG probes live in ``tests/_hue_probe.py`` so this module, the
composite-mark sweep pins, and the diagnostic-desugar pins share one
definition of "what a legend swatch looks like".

The structural guard that keeps this class from recurring at a *new* call
site is ``tests/test_color_binding_completeness.py``; this module pins the
rendered behavior that guard exists to protect.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import ferrum as fm
from tests._hue_probe import has_colorbar, legend_labels, paint_colors, render

# Kinds whose mark groups, aggregates, or fills rows by the hue column. These
# must all answer identically for one Int64 hue -- the uniformity that
# ``_catplot_build`` now enforces with no per-kind branch.
_CATPLOT_GROUPING_KINDS = ("violin", "bar", "boxen", "count")

# Point-mark kinds. catplot types these Nominal too (unlike relplot's
# scatter): its hue is categorical by construction -- ``hue_order=`` orders
# its levels and ``dodge=True`` offsets by them.
_CATPLOT_POINT_KINDS = ("strip", "swarm", "point")


def _twin(dtype: pl.DataType) -> pl.DataFrame:
    """Two groups of six rows, group column rendered as *dtype*.

    Built from Int64 labels 0/1 and cast, so the Utf8 twin's legend reads
    "0"/"1" as well and the two renders are comparable label-for-label
    rather than only count-for-count.
    """
    group = [0] * 6 + [1] * 6
    return pl.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] * 2,
            "y": [1.0, 2.2, 2.9, 4.3, 5.1, 5.8, 3.0, 3.9, 5.2, 5.8, 7.1, 8.2],
            "cat": ["a", "b"] * 6,
            "grp": pl.Series(group, dtype=pl.Int64),
        }
    ).with_columns(pl.col("grp").cast(dtype))


def _contour_twin(dtype: pl.DataType) -> pl.DataFrame:
    """Two well-separated 2-D Gaussian blobs, group column rendered as *dtype*.

    A contour needs enough mass per group for the KDE to produce isolines at
    all; a handful of points yields an empty render that would make any
    comparison trivially pass.
    """
    rng = np.random.default_rng(0)
    half = 60
    return pl.DataFrame(
        {
            "x": np.concatenate([rng.normal(0.0, 1.0, half), rng.normal(5.0, 1.0, half)]),
            "y": np.concatenate([rng.normal(0.0, 1.0, half), rng.normal(5.0, 1.0, half)]),
            "grp": pl.Series([0] * half + [1] * half, dtype=pl.Int64),
        }
    ).with_columns(pl.col("grp").cast(dtype))


def _assert_dtype_parity(build, *, frame=_twin) -> str:
    """Render *build* over the Int64 and Utf8 twins and require agreement.

    Returns the Int64 render so a caller can add case-specific structural
    assertions on top. Asserts three independent things, so a partial
    regression stays distinguishable: no colorbar, the same legend labels,
    and the same set of painted colors.
    """
    int_svg, int_warnings = render(build(frame(pl.Int64)))
    utf8_svg, _ = render(build(frame(pl.Utf8)))

    assert not has_colorbar(int_svg), (
        "an Int64 hue rendered a continuous colorbar; the group column was "
        "read as a quantity rather than as a discriminator"
    )
    assert legend_labels(int_svg) == legend_labels(utf8_svg)
    assert paint_colors(int_svg) == paint_colors(utf8_svg)
    assert int_warnings == []
    return int_svg


# ---------------------------------------------------------------------------
# catplot: one kwarg, one answer, every kind
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kind",
    [
        *_CATPLOT_GROUPING_KINDS,
        *_CATPLOT_POINT_KINDS,
        "box",
    ],
)
def test_catplot_hue_answers_identically_for_every_kind(kind: str) -> None:
    """``catplot(hue=<Int64>)`` renders the categorical palette for all kinds.

    This is the uniformity pin. Before the sweep, one Int64 hue answered
    three different ways across ``catplot``'s eight kinds -- ``violin``
    rendered the categorical palette, ``box`` raised, and
    ``bar``/``boxen``/``strip``/``swarm``/``point`` rendered a silent
    fabricated colorbar -- and the divergence was invisible in the
    signature. Parametrizing one assertion over every kind is the point:
    a future change that fixes one kind and forgets another fails here.
    """

    def build(df: pl.DataFrame):
        if kind == "count":
            return fm.catplot(df, x="cat", hue="grp", kind=kind)
        return fm.catplot(df, x="cat", y="y", hue="grp", kind=kind)

    _assert_dtype_parity(build)


def test_catplot_hue_order_types_the_color_channel_too() -> None:
    """The ``hue_order=`` path builds its own ``Color(...)`` and types it.

    ``_catplot_build`` rebuilds the color channel when ``hue_order=`` is
    given, so it is a second binding that has to carry the same typing --
    exactly the kind of parallel path a sweep misses.
    """
    svg = _assert_dtype_parity(
        lambda df: fm.catplot(df, x="cat", y="y", hue="grp", kind="bar", hue_order=["1", "0"])
    )
    assert len(legend_labels(svg)) == 2


# ---------------------------------------------------------------------------
# The five confirmed-wrong charts
# ---------------------------------------------------------------------------


def test_contour_groupby_renders_one_color_per_group() -> None:
    """Two groups' isolines get two colors, not one.

    ``desugar_contour`` binds its ``groupby`` column onto a ``segment`` mark,
    which is in the silent set -- the Int64 case drew both distributions'
    contours in a single color with no warning at all.
    """
    int_svg = _assert_dtype_parity(
        lambda df: fm.Chart(df).mark_contour(groupby="grp").encode(x="x", y="y"),
        frame=_contour_twin,
    )
    palette = paint_colors(int_svg)
    assert {"#2563eb", "#dc2626"} <= palette, (
        f"expected both categorical palette colors on the contour segments, got {sorted(palette)}"
    )


@pytest.mark.parametrize("figure", ["regplot", "lmplot"])
@pytest.mark.parametrize("scatter", [True, False], ids=["with-scatter", "fit-only"])
def test_regression_hue_renders_one_fit_and_band_per_group(figure: str, scatter: bool) -> None:
    """Two groups produce two fits and two CI bands, not one merged pair.

    The silent collapse this pins was the most serious of the reviewed
    defects: ``Smooth`` still fit each group, but a Continuous color scale
    merged the per-group output into a single rendered curve and band, so the
    chart asserted one relationship where the data holds two.

    ``scatter=False`` is not redundant coverage. With a scatter layer present,
    the layered chart resolves one shared color scale and the scatter layer's
    Nominal declaration alone is enough to type it -- so the fit layers'
    own bindings can be reverted without the ``scatter=True`` case noticing.
    Dropping the scatter layer makes the fit the sole color consumer and puts
    those bindings under test.
    """
    int_svg = _assert_dtype_parity(
        lambda df: getattr(fm, figure)(df, x="x", y="y", hue="grp", scatter=scatter)
    )
    # Two CI bands: <path> elements are the ribbon fills. The count is the
    # discriminating half -- the merged render produced exactly one.
    assert int_svg.count("<path") == 2, (
        f"expected one CI band path per group, got {int_svg.count('<path')}"
    )


def test_relplot_line_hue_renders_one_polyline_per_group() -> None:
    """``relplot(kind="line")`` draws a polyline per hue level.

    Unlike the other four this one at least warned
    (``continuous color scale is not supported on line``), which is why the
    parity helper's "no warnings" assertion is load-bearing here.
    """
    _assert_dtype_parity(lambda df: fm.relplot(df, x="x", y="y", hue="grp", kind="line"))


@pytest.mark.parametrize(
    ("figure", "kwargs"),
    [
        ("catplot", {"x": "cat", "y": "y", "kind": "bar"}),
        ("catplot", {"x": "cat", "y": "y", "kind": "boxen"}),
        ("displot", {"x": "x", "kind": "hist"}),
    ],
    ids=["catplot-bar", "catplot-boxen", "displot-hist"],
)
def test_categorical_hue_gets_a_swatch_legend_not_a_colorbar(figure: str, kwargs: dict) -> None:
    """A two-value group column gets two swatches, never a 0..1 colorbar.

    The fabricated tick values were the tell: a colorbar labelled
    0 / 0.25 / 0.5 / 0.75 / 1 for a column holding exactly two distinct
    values is a truth claim about the mapping that no mark expresses.
    """
    svg = _assert_dtype_parity(lambda df: getattr(fm, figure)(df, hue="grp", **kwargs))
    assert len(legend_labels(svg)) == 2, (
        f"expected a two-entry swatch legend, got {legend_labels(svg)}"
    )


# ---------------------------------------------------------------------------
# The deliberate carve-out
# ---------------------------------------------------------------------------


def test_relplot_scatter_hue_keeps_dtype_inference() -> None:
    """``relplot(kind="scatter")`` is the one surviving point-mark carve-out.

    Pinned so the exception stays deliberate rather than becoming an
    oversight someone later "fixes" by accident. A scatter is a single
    unpaired ``point`` mark: it shares no color scale with a grouping layer,
    and one mark per row genuinely can encode a continuous quantity, so a
    numeric ``hue=`` keeps rendering a gradient for seaborn parity. Contrast
    ``kind="line"`` directly above, which is typed.
    """
    svg, _ = render(fm.relplot(_twin(pl.Int64), x="x", y="y", hue="grp", kind="scatter"))
    assert has_colorbar(svg), (
        "relplot(kind='scatter') should still read a numeric hue as a "
        "continuous gradient; if this now renders a swatch legend the "
        "carve-out was removed and nominal_color_channel's docstring, "
        "test_color_binding_completeness.ALLOWED, and this test disagree"
    )


def test_pairplot_and_jointplot_share_one_scale_type_across_panels() -> None:
    """A shared color scale cannot be Continuous for one panel and Nominal for another.

    Both figures set ``resolve={"color": "shared"}`` to unify the hue domain
    across panels and to drive one figure-level legend. Their marginal /
    diagonal panels are always grouping marks, so their scatter panels take
    the Nominal typing too -- the "sole consumer" qualifier on the point-mark
    carve-out. If a future change reinstates the carve-out here, the shared
    scale is asked to be two types at once and this fails.
    """
    _assert_dtype_parity(lambda df: fm.pairplot(df, vars=["x", "y"], hue="grp"))
    _assert_dtype_parity(lambda df: fm.jointplot(df, x="x", y="y", hue="grp"))
