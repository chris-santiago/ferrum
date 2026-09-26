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
project's test-file convention (see ``tests/marks/test_boxen_palette.py``) keeps
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

import sys
import warnings

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


def _assert_dtype_parity(build, *, frame=_twin, expected_warnings=()) -> str:
    """Render *build* over the Int64 and Utf8 twins and require agreement.

    Returns the Int64 render so a caller can add case-specific structural
    assertions on top. Asserts three independent things, so a partial
    regression stays distinguishable: no colorbar, the same legend labels,
    and the same set of painted colors. ``expected_warnings`` names any
    warning the render is known to emit independent of hue typing (e.g.
    ``mark_violin``'s hue-inner warning); anything else still fails.
    """
    int_svg, int_warnings = render(build(frame(pl.Int64)))
    utf8_svg, _ = render(build(frame(pl.Utf8)))

    assert not has_colorbar(int_svg), (
        "an Int64 hue rendered a continuous colorbar; the group column was "
        "read as a quantity rather than as a discriminator"
    )
    assert legend_labels(int_svg) == legend_labels(utf8_svg)
    assert paint_colors(int_svg) == paint_colors(utf8_svg)
    assert int_warnings == list(expected_warnings)
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

    # violin's default inner="box" under this hue split (color distinct
    # from the categorical x axis) draws its inner mark undodged per hue,
    # which now warns once per render (see heavy_stat.py's
    # _VIOLIN_QUARTILE_BAND_SIZE) -- an intentional exception to the
    # otherwise-uniform zero-warning answer this test pins. Voiced as
    # catplot's own entry, not mark_violin's: catplot fixes inner="box"
    # itself and exposes no inner= of its own for a mark_violin-voiced
    # message to point at.
    expected_warnings = (
        [
            "catplot(kind='violin'): inner marks are drawn at each "
            "category's center and are not dodged per hue group; use "
            'Chart.mark_violin(inner="none") to drop the inner mark'
        ]
        if kind == "violin"
        else []
    )
    _assert_dtype_parity(build, expected_warnings=expected_warnings)


_CATPLOT_VIOLIN_HUE_WARNING = (
    "catplot(kind='violin'): inner marks are drawn at each category's "
    "center and are not dodged per hue group; use "
    'Chart.mark_violin(inner="none") to drop the inner mark'
)


def test_catplot_violin_hue_warning_names_catplot_not_mark_violin() -> None:
    """``catplot(kind="violin", hue=...)`` voices the hue-inner warning as
    its own entry, not ``mark_violin``'s -- catplot fixes ``inner="box"``
    itself (see the ``mark_violin`` door-threading in
    ``src/ferrum/marks/heavy_stat.py`` and ``_chart_methods_statistical.py``)
    and exposes no ``inner=`` of its own, so a message naming
    ``mark_violin(inner=...)`` would send a catplot caller looking for a
    kwarg catplot does not have.

    catplot has no per-kind option-forwarding channel for any kind's
    desugar-time options (`_catplot_build` only ever forwards `position=`
    to `mark_boxplot()`/`mark_violin()`/`mark_boxen()`; there is no
    existing mechanism to mirror for an `inner=` door, so none is added
    here -- a caller who needs a non-default `inner` uses `fm.Chart(...).
    mark_violin(inner=...)` directly instead of `catplot`).
    """
    df = pl.DataFrame(
        {
            "cat": ["a", "b"] * 6,
            "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] * 2,
            "grp": ["x", "y"] * 6,
        }
    )
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        fm.catplot(df, x="cat", y="y", hue="grp", kind="violin").to_svg()
    matching = [w for w in record if str(w.message) == _CATPLOT_VIOLIN_HUE_WARNING]
    assert len(matching) == 1, f"expected exactly 1 catplot-voiced warning, got {matching}"


def test_catplot_violin_voice_stays_off_the_wire_and_the_repr() -> None:
    """catplot's voice rides the pending mark, not its kwargs: it never
    reaches the spec JSON or the repr, and the warning still names the
    caller's own line."""
    df = pl.DataFrame(
        {
            "cat": ["a", "b"] * 6,
            "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] * 2,
            "grp": ["x", "y"] * 6,
        }
    )
    chart = fm.catplot(df, x="cat", y="y", hue="grp", kind="violin")
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        line = sys._getframe().f_lineno + 1  # the next line resolves the chart
        spec = chart.to_json()
    matching = [w for w in record if str(w.message) == _CATPLOT_VIOLIN_HUE_WARNING]
    assert [(w.filename, w.lineno) for w in matching] == [(__file__, line)]
    for surface in (spec, repr(chart)):
        assert "catplot(kind=" not in surface
        assert "mark_violin(inner=" not in surface


def test_catplot_violin_no_hue_never_warns() -> None:
    """With no ``hue=`` at all, the inner mark IS drawn at the (single)
    body's own position, so nothing is dropped and catplot's violin kind
    does not warn -- the negative control for the warning above."""
    df = pl.DataFrame({"cat": ["a", "b"] * 6, "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] * 2})
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        fm.catplot(df, x="cat", y="y", kind="violin").to_svg()
    matching = [w for w in record if str(w.message) == _CATPLOT_VIOLIN_HUE_WARNING]
    assert matching == [], f"expected no hue-inner warning with no hue=, got {matching}"


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
# The untyped channel-object door (#130)
# ---------------------------------------------------------------------------


def test_untyped_color_object_hue_reads_identically_to_bare_string_catplot() -> None:
    """``hue=fm.Color("grp")`` (untyped) on the silent catplot kind renders
    the same swatch legend and per-group paint as ``hue="grp"`` (bare
    string) -- not a fabricated gradient colorbar.

    Two doors into the same helper (``nominal_color_channel``) used to
    disagree: the bare-string door typed the column Nominal; an
    already-constructed but untyped ``Color(...)`` object passed straight
    through on object-ness alone and inferred a Continuous scale from the
    column's runtime dtype instead. On ``catplot(kind="bar")`` that produced
    a fabricated 1/1.5/2/2.5/3-tick colorbar over a 3-level discriminator
    with no warning at all -- the silent class the completeness guard
    cannot see, since its AST scan reads any call to ``Color`` as "typed"
    regardless of what the caller passes it at runtime.
    """
    df = _twin(pl.Int64)
    string_svg, string_warnings = render(fm.catplot(df, x="cat", y="y", hue="grp", kind="bar"))
    object_svg, object_warnings = render(
        fm.catplot(df, x="cat", y="y", hue=fm.Color("grp"), kind="bar")
    )
    assert has_colorbar(object_svg) is False, (
        "an untyped Color(...) object on hue= rendered a continuous "
        "colorbar over a categorical discriminator"
    )
    assert legend_labels(object_svg) == legend_labels(string_svg)
    assert paint_colors(object_svg) == paint_colors(string_svg)
    assert object_warnings == string_warnings == []


def test_untyped_color_object_hue_reads_identically_to_bare_string_relplot_line() -> None:
    """Same equivalence, on the one door that at least warned.

    ``relplot(kind="line")`` collapsed an untyped numeric hue into one
    polyline with a ``continuous color scale is not supported on line``
    warning; an untyped ``Color(...)`` object hit the identical bug because
    ``nominal_color_channel`` tested object-ness, not typedness.
    """
    df = _twin(pl.Int64)
    string_svg, string_warnings = render(fm.relplot(df, x="x", y="y", hue="grp", kind="line"))
    object_svg, object_warnings = render(
        fm.relplot(df, x="x", y="y", hue=fm.Color("grp"), kind="line")
    )
    assert has_colorbar(object_svg) is False
    assert legend_labels(object_svg) == legend_labels(string_svg)
    assert paint_colors(object_svg) == paint_colors(string_svg)
    assert object_warnings == string_warnings


def test_typed_color_object_hue_still_passes_through_untouched() -> None:
    """An already-typed ``Color(...)`` object is unaffected by the fix.

    Pins the other half of the branch: ``nominal_color_channel`` must not
    overrule a caller who stated the type explicitly -- the documented
    ``Color(field, scheme=...)`` spelling from ``Color``'s own docstring
    must keep working exactly as before.
    """
    df = _twin(pl.Int64)
    svg, warnings_ = render(
        fm.catplot(df, x="cat", y="y", hue=fm.Color("grp", type="nominal"), kind="bar")
    )
    assert has_colorbar(svg) is False
    assert len(legend_labels(svg)) == 2
    assert warnings_ == []


@pytest.mark.parametrize("spelling", ["type_", "type"], ids=["type_-kwarg", "type-kwarg"])
def test_an_explicitly_quantitative_hue_still_renders_its_colorbar(spelling: str) -> None:
    """ "Does not overrule an explicit type" is a claim about the RENDER.

    The test above states the same half of the branch with
    ``type="nominal"``, which is the type ``nominal_color_channel`` would
    have imposed anyway -- so it holds even for an implementation that
    ignores the caller's declaration entirely, and a mutation that retyped
    every channel object Nominal survived the whole suite: the only
    assertion that noticed was an object-identity check in
    ``tests/test_catplot_hue_order.py``. Identity is not the contract; the
    chart is.

    So: state the type the helper would NOT have chosen, and require the
    rendered consequence of the caller's choice -- a continuous colorbar for
    the numeric column, and no categorical swatch legend standing in for it.
    ``catplot(kind="bar")`` is the vehicle because a bar mark renders a
    continuous color scale silently (no ``UnsupportedColorScaleOnMark``), so
    the two readings differ in the picture rather than in a warning.

    Both spellings, because ``ChannelBase.__init__`` normalizes ``type_=``
    and ``type=`` into the one ``"type"`` key the helper's untypedness
    predicate reads.
    """
    df = _twin(pl.Int64)
    hue = fm.Color("grp", **{spelling: "quantitative"})
    svg, warnings_ = render(fm.catplot(df, x="cat", y="y", hue=hue, kind="bar"))

    assert has_colorbar(svg), (
        "an explicitly quantitative hue must still render its colorbar; "
        "nominal_color_channel overruled the caller's stated type"
    )
    assert legend_labels(svg) == [], (
        "a quantitative color scale has no categorical swatch legend; "
        f"got {legend_labels(svg)}, i.e. the channel was retyped Nominal"
    )
    assert warnings_ == []


def test_relplot_scatter_hue_with_untyped_color_object_keeps_dtype_inference() -> None:
    """The scatter carve-out is unaffected: it never calls
    ``nominal_color_channel`` at all, so an untyped ``Color(...)`` object on
    ``hue=`` keeps inferring a continuous gradient, identically to the
    bare-string spelling pinned by
    ``test_relplot_scatter_hue_keeps_dtype_inference`` above.
    """
    svg, _ = render(fm.relplot(_twin(pl.Int64), x="x", y="y", hue=fm.Color("grp"), kind="scatter"))
    assert has_colorbar(svg), (
        "relplot(kind='scatter') should still read an untyped Color(...) "
        "hue as a continuous gradient; #130's fix lives in "
        "nominal_color_channel, which the scatter carve-out never calls"
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
