"""Distribution domain figure functions.

Public API
----------
displot, catplot, relplot.

``displot`` dispatches to histogram, KDE, ECDF, or rug marks.
``catplot`` dispatches to strip, swarm, box, violin, boxen, point, bar,
or count marks.
``relplot`` dispatches to scatter or line marks with optional faceting.
"""

from __future__ import annotations
from typing import Any, NamedTuple

from ferrum import (
    Aggregate,
    AggregateOp,
    Chart,
    CoordFlip,
    Dodge,
    Identity,
    Jitter,
    LinearScale,
    Stack,
)
from ferrum._coerce import to_polars
from ferrum._core import validate_bin_rule
from ferrum._desugar import (
    _explicit_positional_scale,
    _propagate_layer_axis,
    _scale_domain,
)
from ferrum._encode_lowering import (
    _ECDF_CATEGORICAL_READS,
    _KDE_CATEGORICAL_READS,
    _KERNEL_DENSITY_READS_RAW_ROWS,
    _histogram_categorical_reads,
    _refuse_bin_aggregate_on_channels,
    _refuse_categorical_input,
    _refuse_ecdf_bin_aggregate,
    _refuse_ecdf_field_y,
    _refuse_histogram_bin_aggregate,
)
from ferrum._layer import _Layer
from ferrum._registry import registry_output_columns
from ferrum._validate import as_finite_real, coerce_positive_int, validate_choice
from ferrum._warn import warn_once
from ferrum.marks._desugar_helpers import (
    nominal_color_channel,
    resolve_color_groupby,
    titled_positional_channel,
)
from ferrum.marks._multiple import MULTIPLE_MODES, MultipleSemantics, multiple_semantics
from ferrum.marks.base import resolve_method_bool
from ferrum.marks.statistical import _resolve_ecdf
from ferrum.encoding import Y
from ferrum.encoding.base import declared_title, drawn_title
from ferrum.plots._helpers import (
    _checked_figure_size,
    _drawn_bindings,
    _drawn_positional_channels,
    _field_name,
    _figure_entry,
    _finalize_chart,
    _grouping_hue,
    _log_scaled_x,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DISPLOT_VALID_KINDS = {"hist", "kde", "ecdf", "rug"}
# The accepted `multiple=` words come from the module that owns what each
# word MEANS (`marks/_multiple.py`), so displot cannot accept a mode the
# owner has no mapping for, or refuse one it has.
_DISPLOT_VALID_MULTIPLE = MULTIPLE_MODES
_DISPLOT_VALID_STAT = {"count", "density"}

# The kinds whose distribution variable is the one `x=` names: each bins or
# evaluates that variable and COMPUTES the other axis (a count, a density, a
# cumulative frequency), so there is no slot for a `y=` variable to fill.
# `"rug"` is the one exception -- a tick mark draws its ticks against either
# axis -- which is why this is a named set and not "every kind".
_DISPLOT_KINDS_REQUIRING_X = frozenset({"hist", "ecdf", "kde"})

# The kinds that actually bin, i.e. the only ones `bins=` can act on.
# `"ecdf"` counts each observation itself (a rolling `transform_window`, not
# a bin edge), so it is not in this set.  The `kde=True` overlay reads
# `bins=` too, but only to count-scale itself onto a HISTOGRAM's count axis
# (`_overlay_count_scale`), so it adds no kind here either.
_DISPLOT_KINDS_THAT_BIN = frozenset({"hist"})

# displot's own `bins=` default, named so the "did the caller state a rule of
# their own?" test below and the signature cannot drift apart.
_DISPLOT_DEFAULT_BINS = "sturges"

_CATPLOT_VALID_KINDS = {"strip", "swarm", "box", "violin", "boxen", "point", "bar", "count"}
_RELPLOT_VALID_KINDS = {"scatter", "line"}

# The two catplot kinds that draw an ESTIMATE (one mark per category group)
# rather than one mark per row, and therefore consume ``ci=``/``n_boot=``.
_CATPLOT_AGGREGATING_KINDS = frozenset({"point", "bar"})

# ``catplot``'s ``ci=``/``n_boot=`` defaults: on a kind outside
# ``_CATPLOT_AGGREGATING_KINDS`` only these values pass silently.
_CATPLOT_DEFAULT_CI = 95
_CATPLOT_DEFAULT_N_BOOT = 1000

# Encoding channels that assert a PER-OBSERVATION visual mapping, refused by
# ``_CATPLOT_AGGREGATING_KINDS`` (see ``_reject_per_row_appearance_channels``
# for why, and for why ``hue=`` is not one of them).  Every non-positional
# channel ``Chart.encode`` accepts was checked against the aggregated output;
# the ones here either drew a fabricated legend over uniform marks or were
# byte-identical no-ops.  Hand-curated against
# ``encoding._channel_policy._RENDERER_HONORED_CHANNELS`` and TEST-ENFORCED
# against it -- a channel added there with no disposition fails
# ``test_every_renderer_honored_channel_has_a_disposition`` rather than
# silently escaping this refusal.
_CATPLOT_PER_ROW_APPEARANCE_CHANNELS = frozenset(
    {
        "size",
        "shape",
        "opacity",
        "fill_opacity",
        "stroke_opacity",
        "stroke_width",
        "stroke_dash",
        "angle",
        "color",
        "fill",
        "stroke",
    }
)

# The one ``ci=`` refusal text, shared by the three ways a value can fail it
# (a bool, a non-number, and an out-of-range number) so they cannot drift into
# three descriptions of one rule.
_CI_REFUSAL = (
    "catplot(ci={ci!r}) is not supported -- ci is a PERCENT and must be a number in "
    "the open interval (0, 100), exclusive at both ends (a 0 % interval has no width "
    "and a 100 % bootstrap interval is not estimable from a finite resample). Pass "
    "ci=95 (the default) or any percent strictly between 0 and 100."
)

# Name of catplot's ``Summary`` transform.  Named (not chained) so it reads the
# original rows while the chart's main batch stays the raw data; the estimator
# and whisker layers bind to it through ``data_source=``.
_CATPLOT_SUMMARY_NAME = "summary"

# Minimum per-panel height (px) used by the auto-sizing path in displot/catplot
# when `height` is not set and the chart is faceted across rows.  Derived from
# the Rust layout minimum: EmptyPanel triggers at ~55 px; 150 px gives comfortable
# axes + mark area and matches seaborn's default facet height.
_FACET_DEFAULT_PANEL_HEIGHT_PX: float = 150.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _count_facet_levels(data: Any, field: str) -> int:
    """Return the number of distinct non-null values in *field* within *data*.

    Used to auto-scale chart height when faceting across rows.  Returns 1 on
    any error (field absent, data not a supported type, etc.) so the caller
    degrades gracefully to the default size rather than raising.
    """
    try:
        df = to_polars(data)
        if field not in df.columns:
            return 1
        return df[field].n_unique()
    except Exception:
        return 1


def _multiple_to_position(multiple: str, hue: Any, *, value_axis: str | None) -> MultipleSemantics:
    """Return displot's ``(position, shared_extent)`` pair for *multiple*.

    A thin binding of the package-wide owner (``marks/_multiple.py``) to
    displot's two local conventions:

    - the position groups by ``hue`` explicitly, so the rendered ``Stack`` /
      ``Dodge`` names the column the caller asked to split on rather than
      resolving the color channel implicitly at render time;
    - ``multiple="layer"`` becomes an explicit ``Identity()`` rather than
      the owner's ``None``.  displot's overlay path copies the chart-level
      position onto the primary layer, and "no adjustment declared" and
      "explicitly no adjustment" are different values there.

    Both halves come from the owner, so the binning extent the histogram's
    ``Bin`` receives and the adjustment that positions its bars can never
    disagree about the mode again (GH #194).

    ``value_axis`` is required, with no default, for the same reason: it is
    the owner's third input (which axis a ``Stack`` cumulates, GH #77) and
    the mark desugar passes it too.  Leaving it out here made displot's
    adjustment and the desugar's equal only because displot happens never to
    build a horizontal histogram -- a silent drop that would have started
    disagreeing the moment it did.  The caller states the axis instead, so
    the equality is checkable where it is claimed.
    """
    position, shared_extent = multiple_semantics(multiple, by=hue, value_axis=value_axis)
    return MultipleSemantics(position if position is not None else Identity(), shared_extent)


def _overlay_count_scale(bins: Any) -> Any:
    """Return the `Kde(count_scale=...)` value matching what the bars were binned by.

    `Kde(count_scale=)` and `Bin(bins=)` read the identical vocabulary from
    the identical Rust parser, so the caller's own `bins` is forwarded
    unchanged and the overlay tracks the bars by construction -- there is no
    rule the bars can honor that the overlay must downgrade. The one
    translation: `bins=None` means "let `Bin` pick" (Sturges), while
    `count_scale=None` means "do not count-scale at all", so the `Bin`
    default is spelled out here rather than passed through as `None`.
    """
    return bins if bins is not None else "sturges"


def _displayed_name(binding: Any) -> str | None:
    """Return the name a positional *binding* shows: its ``title=``, else its field.

    ``None`` when nothing is bound. Read off the primary layer's own binding,
    so the overlay names whatever that layer's axis already shows.
    """
    title = declared_title(binding)
    return _field_name(binding) if title is None else title


def _kde_overlay(
    x_field: str,
    *,
    hue: Any,
    kind: str,
    stat: str,
    multiple: str,
    shared_extent: bool,
    bins: Any,
    bandwidth: Any,
    bw_adjust: float,
    cumulative: bool,
    x_title: str | None,
    y_title: str | None,
):
    """Build the named `Kde` transform and `_Layer` for displot's `kde=True` overlay.

    Returns `(transform, layer)`: the caller prepends the transform (so it
    reads the original, pre-`Bin` data batch) and appends the layer.

    `x_title`/`y_title` are the names the chart's primary layer shows on its
    x/y axes, read off its own bindings by the caller. The overlay reads its
    own grid/density columns, so its bindings carry those names instead: its
    warnings and tooltips then name the axes it shares with the primary
    mark. A title is `None` when the primary binds nothing on that axis
    (`y` under `kind="rug"`); the overlay then owns that axis and names its
    own column.

    `groupby=hue` gives one curve per hue level instead of one curve pooled
    across all rows; `shared_extent` is passed in -- it is the value the real
    histogram's `Bin` transform receives on this same call, taken from the
    one owner of what `multiple=` means (`_multiple_to_position` ->
    `marks/_multiple.py`). It is never re-derived here: a second reading of
    `multiple in ("stack", "fill")` would be a second answer that can drift
    from the bars', and a grouped count-scaled bin width computed off the
    wrong extent no longer matches the bars it is drawn against
    (`test_wire_shared_extent_matches_the_real_bin_transform` guards this).
    RATIONALE CORRECTED (#194): this used to read `shared_extent=False`
    unconditionally, mirroring a `Bin` that was itself wrong, because
    `_displot_build` did not forward the mode at all. It forwards it now, so
    stack/fill bin on a global extent and the overlay follows them there.
    `count_scale` is requested only when the histogram's own value axis is
    actually a count axis: `kind == "hist"` (not ecdf's cumulative-count
    line, not rug, which has no bar axis at all), `stat == "count"` (density
    is already normalized), and `multiple != "fill"` (fill normalizes bars
    to a `[0, 1]` proportion via `Stack(offset="normalize")` regardless of
    `stat`, and a count-scaled overlay sharing that axis would stretch it).

    `position=Identity()` is set explicitly: a density curve is never
    meaningfully stacked, dodged, or normalized, and leaving `position`
    unset lets the layer inherit the chart-level position adjustment --
    under `multiple="fill"` specifically, once `groupby=hue` put the hue
    column into this layer's own batch, `Stack(offset="normalize")` found a
    match and flattened the curve to a constant against the panel edge.
    """
    from ferrum import Kde as _KdeTransform
    from ferrum.encoding import X, Y

    kde_kwargs: dict = dict(
        bandwidth=bandwidth,
        bw_adjust=bw_adjust,
        cumulative=cumulative,
        shared_extent=shared_extent,
        name="kde_overlay",
    )
    if hue is not None:
        # `_field_name`, not the raw `hue`: a channel object (`fm.Color("g")`)
        # is a documented `hue=` spelling and `groupby=` takes only a column
        # name. This helper takes the unresolved `hue` because its color
        # ENCODING below wants the channel itself; `_displot_build` resolves
        # the same name once for its own consumers (see `hue_field` there).
        kde_kwargs["groupby"] = _field_name(hue)
    # count_scale may only be requested when the histogram's value axis is
    # measured IN COUNTS. That is the whole of this predicate's question --
    # its UNIT, not its baseline -- and the three clauses are the complete
    # reading of it: `kind == "hist"` (ecdf's axis is a cumulative count of a
    # different quantity, rug has no value axis at all), `stat == "count"`
    # (density is already normalized), `multiple != "fill"` (fill's
    # `Stack(offset="normalize")` rescales the bars to a [0, 1] proportion,
    # so the axis stops being counts at all).
    #
    # NOT answered here, and deliberately: where each group's bars START.
    # `multiple="stack"` keeps the axis in counts -- so a count-scaled curve
    # is in the right units and tracks each group's own bar SEGMENT height
    # (`test_overlay_curve_tracks_its_own_bars_in_every_count_mode` measures
    # that ratio) -- but offsets every group above the first onto the stack
    # below it, while the overlay is drawn from zero by the explicit
    # `position=Identity()` set below. Measured on two 200-row groups,
    # `stat="count"`: the bar stack tops out at y=24 while the curves peak at
    # y=165. Excluding `"stack"` from this predicate would NOT fix that -- it
    # would replace a curve in the right units with an unscaled density
    # (peak ~0.2 against an axis reaching ~200), i.e. an invisible sliver at
    # the baseline, which is strictly worse. The fix that would is stacking
    # the overlay with the bars (what seaborn draws), which is a change to
    # the overlay's POSITION, not to this reading; it is filed separately.
    histogram_value_axis_is_count = kind == "hist" and stat == "count" and multiple != "fill"
    if histogram_value_axis_is_count:
        kde_kwargs["count_scale"] = _overlay_count_scale(bins)
    transform = _KdeTransform(x_field, **kde_kwargs)

    # Derived from the registry (spec §4.6 / M6a) rather than hardcoded, so a
    # Rust-side rename of either Kde output column raises here instead of
    # silently drawing an empty overlay line.
    _kde_cols = registry_output_columns("kde", ["value", "density"])
    encoding: dict = {
        "x": titled_positional_channel(_kde_cols["value"], channel=X, title=x_title),
        "y": titled_positional_channel(_kde_cols["density"], channel=Y, title=y_title),
    }
    if hue is not None:
        encoding["color"] = nominal_color_channel(hue)

    layer = _Layer(
        name="kde",
        mark="line",
        encoding=encoding,
        data_source="kde_overlay",
        position=Identity(),
    )
    return transform, layer


def _apply_facet_sizing(
    chart: Any,
    *,
    data: Any,
    row: Any,
    height: float | None,
    aspect: float | None,
) -> Any:
    """Apply per-panel height/aspect sizing to a (possibly row-faceted) chart.

    When ``row`` is set, ``height`` is treated as per-panel height (seaborn
    semantics).  The total canvas height is ``per_panel_h * n_panels``.  Width
    is ``per_panel_h * aspect`` regardless of panel count, so the canvas width
    does not grow with the number of panels.

    When ``row`` is not set, height and aspect are applied directly: the canvas
    is ``height × (height * aspect)``.

    *height* and *aspect* are the caller's, already checked at the figure's
    entry (:func:`ferrum.plots._helpers._checked_figure_size`).

    Returns the chart with ``properties()`` applied, or the chart unchanged when
    neither ``height`` nor ``aspect`` is given and there is no row faceting.
    """
    if row is not None:
        n_row_panels = _count_facet_levels(data, str(row))
        per_panel_h = height if height is not None else _FACET_DEFAULT_PANEL_HEIGHT_PX
        total_h = per_panel_h * n_row_panels
        # Width is per-panel height × aspect (independent of panel count).
        # Using total_h here would make width grow linearly with n_panels, which
        # violates seaborn semantics where aspect governs per-panel width.
        w = per_panel_h * aspect if aspect is not None else None
        props: dict = {}
        # An empty frame has no row levels, so no panel height to total:
        # leave the height unset and let the chart render as empty.
        if total_h > 0:
            props["height"] = total_h
        if w is not None:
            props["width"] = w
        return chart.properties(**props) if props else chart
    elif height is not None or aspect is not None:
        h = height if height is not None else 300.0
        w = h * aspect if aspect is not None else h
        return chart.properties(width=w, height=h)
    return chart


# ---------------------------------------------------------------------------
# displot
# ---------------------------------------------------------------------------


def _reject_displot_without_x(*, kind: str, x: Any, y: Any, kde: bool, rug: bool) -> None:
    """Refuse, in displot's own voice, a call whose distribution variable is missing.

    Two questions, one answer, one site.  Three kinds
    (:data:`_DISPLOT_KINDS_REQUIRING_X`) draw the variable ``x=`` names and
    compute the other axis themselves; the ``kde=``/``rug=`` overlays are
    drawn against that same variable.  Neither can be built from a ``y=``
    binding, and both used to fail badly:

    - the kinds raised in the MARK's voice -- ``mark_histogram() requires
      .encode(x=...)`` / the ``mark_density()`` twin -- naming a method the
      caller never called, from a figure function that validates everything
      else as ``displot(...)`` (``kind="ecdf"`` was the one that already
      answered in displot's voice; its message is subsumed here so the three
      cannot drift into three descriptions of one rule);
    - the overlays said nothing at all.  ``displot(y=..., kind="rug",
      kde=True)`` promoted the rug into layered mode and then appended no
      overlay, because the overlay block guards on ``x_field is not None``:
      measured 0 ``<polyline>`` elements against 1 for the identical call
      with ``x=``.  A requested mark that vanishes is the REFUSE-MALFORMED /
      WARN-DEGRADED policy's central case, and it is a refusal rather than a
      warning: the overlay is not a knob that got ignored but a whole mark
      with nowhere to stand, and the identical "this needs x=" question one
      line up is already answered by refusing.

    Called from ``displot`` itself, beside the closed-vocabulary checks, so
    the refusal is raised before any chart is built.
    """
    if x is not None:
        return

    # The variable the caller DID bind, and the fix spelled in their own
    # column name.  `_field_name` (not `repr(y)`) so a channel-object `y=`
    # reports the column rather than the channel's repr.
    bound = _field_name(y)
    observed = f"y={bound!r}" if y is not None else "x=None, y=None"
    pass_x = f"Pass x={bound!r}" if y is not None else "Pass x='<column>'"
    binds = "binds y= instead" if y is not None else "binds neither"

    if kind in _DISPLOT_KINDS_REQUIRING_X:
        nowhere = "so a y= variable has nowhere to go" if y is not None else "and none was given"
        raise ValueError(
            f"displot(kind={kind!r}, {observed}) is not supported -- kind={kind!r} "
            "draws the distribution of the variable x= names and computes the other "
            f"axis itself (a count, a density, a cumulative frequency), {nowhere}. "
            f"{pass_x}, or kind='rug', which draws its ticks against either axis."
        )

    overlay = "kde" if (kde and kind != "kde") else "rug" if (rug and kind != "rug") else None
    if overlay is not None:
        raise ValueError(
            f"displot(kind={kind!r}, {observed}, {overlay}=True) is not supported -- "
            f"the {overlay}= overlay is drawn against the variable x= names, and this "
            f"call {binds}, so the overlay has no variable to read. {pass_x} "
            f"(kind='rug' draws the same ticks from x= or y=), or drop {overlay}=."
        )


def _refuse_displot_bin_aggregate(*, kind: str, drawn: dict, stat: str, multiple: str) -> None:
    """Refuse a bin=/aggregate= the statistic displot's ``kind=`` builds cannot honor,
    and a declared categorical type on an axis that statistic reads as numbers.

    *drawn* is the x/y channel objects the chart actually draws once
    ``encode=`` applies (:func:`~ferrum.plots._helpers._drawn_positional_channels`),
    so a keyword, an ``encode=`` entry and a shorthand string are all
    checked alike. As the matching mark's own refusal reads it
    (:func:`~ferrum._encode_lowering._refuse_histogram_bin_aggregate` for
    ``"hist"``, :func:`~ferrum._encode_lowering._refuse_bin_aggregate_on_channels`
    for ``"kde"``, :func:`~ferrum._encode_lowering._refuse_ecdf_bin_aggregate`
    for ``"ecdf"``), which
    resolves displot's own remedies itself (``_DOOR_VOCABULARY``, keyed by
    the voice the refusal speaks as) -- nothing door-specific is passed in
    here. The categorical-input check beside each one
    (:func:`~ferrum._encode_lowering._refuse_categorical_input`) is this
    call-time door's half of the same rule the resolve-time preps apply as
    a backstop for a later ``.encode()``. ``kind='rug'`` draws every row
    through the plain lowering, which consumes a channel's
    ``bin=``/``aggregate=`` as any primitive mark does, so it is not
    checked here.
    """
    if kind == "hist":
        _refuse_histogram_bin_aggregate(
            "displot",
            drawn,
            horizontal=False,
            density=(stat == "density"),
            share="fill" if multiple == "fill" else None,
        )
        _refuse_categorical_input("displot", drawn, ("x",), reads=_histogram_categorical_reads("y"))
    elif kind == "kde":
        _refuse_bin_aggregate_on_channels(
            "displot",
            drawn,
            ("x", "y"),
            reason=_KERNEL_DENSITY_READS_RAW_ROWS,
        )
        _refuse_categorical_input("displot", drawn, ("x", "y"), reads=_KDE_CATEGORICAL_READS)
    elif kind == "ecdf":
        _refuse_ecdf_bin_aggregate(drawn)
        _refuse_categorical_input("displot", drawn, ("x",), reads=_ECDF_CATEGORICAL_READS)


def _warn_inert_bins(*, kind: str, bins: Any) -> None:
    """Warn when the caller named a binning rule the ``kind=`` cannot act on.

    ``bins=`` is validated for every kind (so a malformed rule is refused as
    ``displot(...)`` rather than by whichever transform happens to be built),
    but only ``kind="hist"`` bins anything: measured,
    ``displot(x="v", kind="kde", bins=3)`` and ``bins=97`` render byte-
    identically, as do ``kind="ecdf"`` (a per-observation count, not a bin
    edge) and ``kind="rug", kde=True`` with either.  Accepted-and-inert is
    the WARN-DEGRADED half of the standing policy, so it warns rather than
    refusing -- the chart the caller asked for is still drawn, it just is
    not binned by their rule.

    Only a rule the caller actually STATED warns.  ``bins=None`` and
    ``bins="sturges"`` both mean "the default rule", which is what a
    non-binning kind would have used anyway, so nothing the caller said is
    lost and there is nothing to report.
    """
    if kind in _DISPLOT_KINDS_THAT_BIN or bins is None or bins == _DISPLOT_DEFAULT_BINS:
        return
    warn_once(
        "displot",
        "bins",
        f"displot(kind={kind!r}, bins={bins!r}): bins= is inert for kind={kind!r} -- "
        "it evaluates no binned statistic, so the rule you named cannot act on "
        "anything and the render is identical for every value. Drop bins=, or use "
        "kind='hist', which bins.",
    )


@_figure_entry
def displot(
    data: Any,
    *,
    x: Any = None,
    y: Any = None,
    hue: Any = None,
    col: Any = None,
    row: Any = None,
    kind: str = "hist",
    fill: bool = True,
    cumulative: bool = False,
    log_scale: bool = False,
    stat: str = "count",
    bins: Any = _DISPLOT_DEFAULT_BINS,
    bandwidth: Any = "scott",
    bw_adjust: float = 1.0,
    multiple: str = "layer",
    kde: bool = False,
    rug: bool = False,
    height: float | None = None,
    aspect: float | None = None,
    mark: dict | None = None,
    encode: dict | None = None,
    properties: dict | None = None,
    layers: list | None = None,
    theme: Any = None,
    **encode_kwargs: Any,
) -> Chart:
    """Univariate distribution plot.

    Convenience wrapper that dispatches to ``mark_histogram``, ``mark_density``,
    or ``mark_tick`` based on ``kind``.  The ``multiple`` parameter controls how
    overlapping groups (from ``hue``) are positioned, and the ``kde`` / ``rug``
    flags optionally layer additional marks on top of the primary kind.

    Parameters
    ----------
    data : DataFrame-like
        Input data accepted by ``Chart(data)``.
    x : str or encoding, optional
        Column name for the distribution variable (horizontal axis).
        Required by ``kind="hist"``, ``"kde"`` and ``"ecdf"``, which compute
        the other axis themselves, and by the ``kde=``/``rug=`` overlays,
        which are drawn against this same variable.
    y : str or encoding, optional
        Column name for the distribution variable (vertical axis), or a
        second variable.  ``kind="rug"`` honors it as a second distribution
        variable -- a tick mark draws against either axis.  ``kind="kde"``
        honors a ``y=`` bound alongside ``x=`` too, but draws it
        differently: together they produce a bivariate KDE (a filled
        contour) instead of two separate curves.  ``kind="hist"`` and
        ``kind="kde"`` both refuse a ``y=``-only call (no ``x=``) by name
        rather than rendering something the caller did not ask for.
        ``kind="ecdf"`` draws the cumulative count of ``x=`` on ``y``, so it
        refuses a ``y=`` that names a field (with or without ``x=``); a
        fieldless ``fm.Y(title=..., scale=..., axis=..., format=...)``
        styles that count axis instead.
    hue : str or encoding, optional
        Column name to map to color (one distribution per level).
    col : str, optional
        Column name for faceting across columns.
    row : str, optional
        Column name for faceting across rows.
    kind : {"hist", "kde", "ecdf", "rug"}, default "hist"
        Which distribution mark to draw.  ``"hist"`` calls
        ``mark_histogram``; ``"kde"`` calls ``mark_density`` (filled by
        default); ``"ecdf"`` draws a per-observation rolling count as a step
        line (``mark_line(interpolate="step-after")``); ``"rug"`` calls
        ``mark_tick``.
    fill : bool, default True
        Fill the area under the KDE curve (``kind="kde"`` only).
    cumulative : bool, default False
        Produce a cumulative histogram or density (``kind="hist"`` and
        ``kind="kde"``).  Also drives the ``kde=True`` overlay curve (below),
        so a cumulative histogram's overlay is a matching CDF, not a PDF.
    log_scale : bool, default False
        Apply a ``log`` scale to the ``x`` axis.  Resolved on the caller's
        own ``x=`` binding before the ``kind=`` dispatch, so every kind and
        every ``kde=``/``rug=`` overlay inherits it from that one binding,
        and a channel ``x=`` is rebuilt with the log scale added, keeping
        its other options.  A channel ``x=`` that already carries its own
        ``scale=`` is refused -- two instructions for one scale.
    stat : {"count", "density"}, default "count"
        Statistic to plot on the value axis for ``kind="hist"``.
        ``"density"`` normalises so the total area integrates to 1.
        Validated for every ``kind`` (not just ``"hist"``): an unsupported
        value raises regardless of ``kind``.
    bins : str, int or float, default "sturges"
        Binning rule for ``kind="hist"`` -- the only kind that bins.
        Stating a non-default rule on ``kind="kde"``, ``kind="ecdf"`` or
        ``kind="rug"`` emits a ``UserWarning``: the value is accepted and
        validated, but nothing evaluates it there.  ``"sturges"``
        (the default) and ``"fd"``/``"freedman_diaconis"`` select a rule; an
        integer fixes the bin count; a float fixes the bin *width* in data
        units (numpy / ``Bin2D`` / ``Kde`` parity).  ``None`` is Sturges.
        Anything else is refused by name: an unknown rule string, a ``bool``
        (``True`` is not a one-bin request), and a zero, negative or
        non-finite number all raise ``ValueError`` naming ``displot`` and the
        accepted set.  When ``kde=True`` scales its curve onto the count
        axis, it does so from this same rule, so the overlay always tracks
        the bars actually drawn.
    bandwidth : str, default "scott"
        Bandwidth selector for ``kind="kde"`` (``"scott"`` or ``"silverman"``).
    bw_adjust : float, default 1.0
        Multiplicative bandwidth adjustment for ``kind="kde"``.
    multiple : {"layer", "stack", "fill", "dodge"}, default "layer"
        How to render multiple distributions (one per ``hue`` level).
        ``"layer"`` overlays them (``Identity``); ``"dodge"`` places them
        side by side; ``"stack"`` and ``"fill"`` use ``Stack`` with
        ``offset="zero"`` or ``offset="normalize"`` respectively.
    kde : bool, default False
        When ``True`` and ``kind != "kde"``, layer a KDE curve (a
        ``mark="line"`` layer) on top of the primary mark, grouped and
        colored by ``hue`` the same way the primary mark is.  Under
        ``kind="hist", stat="count"``, the curve is rescaled onto the
        count axis (``density x n_group x bin_width`` using the histogram's
        own bin width) so it is visible against the bars instead of a flat
        sliver -- except when ``kind != "hist"`` (an ``"ecdf"``/``"rug"``
        primary mark has no matching count axis) or ``multiple="fill"``
        (the bars are normalised to a ``[0, 1]`` proportion, which a
        count-scaled curve would not match).  Under any other ``stat``/
        ``kind``/``multiple`` combination the curve is an unscaled density.
        Under ``kind="rug"`` given an ``x=`` variable -- the one primary
        mark that binds no value axis of its own -- the curve's own density
        axis becomes the chart's ``y`` axis, titled with the density column,
        and the rug ticks sit at its base.  ``kind="kde", rug=True,
        fill=False`` reaches the same two marks from the other side and
        places them identically.
    rug : bool, default False
        When ``True`` and ``kind != "rug"``, layer a ``mark_tick`` rug
        on top of the primary mark.
    height : float or None, optional
        Height of the chart in pixels.  Width is derived from ``aspect``.
    aspect : float or None, optional
        Aspect ratio (width = height * aspect).  Requires ``height``.
    theme : Theme, optional
        Visual theme applied via ``Chart.theme()``.
    mark : dict, optional
        Per-layer mark overrides.  For composite-mark charts, keys are
        layer names (e.g. ``{"scatter": {"opacity": 0.5}}``); for
        single-mark charts, a flat dict of mark properties.
    encode : dict, optional
        Additional encoding kwargs merged via ``Chart.encode(**encode)``.
    properties : dict, optional
        Chart properties merged via ``Chart.properties(**properties)``
        (e.g. ``{"width": 400, "title": "My chart"}``).
    layers : list, optional
        Extra layers appended via ``Chart.layer(*layers)``.
    **encode_kwargs
        Additional keyword arguments forwarded to ``Chart.encode()``.

    Returns
    -------
    Chart
        Configured chart (possibly layered, faceted, or sized).

    Raises
    ------
    ValueError
        If ``kind``, ``multiple``, or ``stat`` is not one of the supported values.
    ValueError
        If ``bins`` is not one of the supported rules, counts, or widths.
    ValueError
        If ``kind="hist"``, ``"kde"`` or ``"ecdf"`` is used without ``x=``,
        or if ``kde=``/``rug=`` requests an overlay on a call that binds no
        ``x=``.  Both name ``displot`` and the variable actually bound.
    ValueError
        If ``kind="ecdf"`` is used with a ``y=`` that names a field.
    ValueError
        If ``log_scale=True`` is used with a channel ``x=`` that already
        carries its own ``scale=``.
    ValueError
        If ``x=`` or ``y=`` -- through a keyword, ``encode=`` or a
        shorthand string -- carries a ``bin=``/``aggregate=`` the ``kind=``
        statistic cannot honor: a ``bin=`` rule on ``kind="hist"`` (move it
        to ``bins=``), any ``bin=``/``aggregate=`` on ``kind="kde"``, or
        either on ``kind="ecdf"``'s x and y alike. ``kind="hist"``'s value
        axis never binds a column of its own, so any ``bin=`` there, and
        any ``aggregate=`` other than a fieldless ``"count"``, is refused
        the same way. A fieldless one's remedy is to drop the kwarg. A
        field-bearing one's remedy draws the per-bin aggregate of that
        second field with ``mark_bar()`` (a ``bin=True`` x channel beside
        the aggregated y), counts both fields with ``mark_hex()`` for a
        ``bin=``, or drops ``y=`` and styles the axis with a fieldless
        ``fm.Y(title=...)``.
        ``kind="hist"``'s own ``bin=True`` and a fieldless
        ``aggregate="count"`` (outside ``stat="density"`` and a normalized
        stack -- ``multiple="fill"``) are honored, not refused.

    Warns
    -----
    UserWarning
        If ``bins=`` states a rule on a ``kind=`` that does not bin.

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.displot(df, x="sepal_length")

    KDE with per-species coloring:

    >>> fm.displot(df, x="sepal_length", hue="species", kind="kde")

    Stacked histogram with an overlaid rug:

    >>> fm.displot(df, x="tip", hue="sex", multiple="stack", rug=True)
    """
    validate_choice("displot", "kind", kind, _DISPLOT_VALID_KINDS)
    validate_choice("displot", "multiple", multiple, _DISPLOT_VALID_MULTIPLE)
    validate_choice("displot", "stat", stat, _DISPLOT_VALID_STAT)
    # `fill=` and `cumulative=` take only a truth value, on every `kind=`,
    # not only `kind="kde"` (whose own gate, `mark_density`'s in
    # `_chart_methods_statistical.py`, reads them for itself). A column
    # reaches the fill channel only through `encode={"fill": ...}`; `fill=`
    # here is this switch, not a channel.
    fill = resolve_method_bool("displot", "fill", fill, True)
    cumulative = resolve_method_bool("displot", "cumulative", cumulative, False)
    # `bins` is validated here, alongside the three closed vocabularies, so a
    # malformed value is refused as `displot: ...` rather than by whichever
    # transform this call happens to build (`Bin` for hist, `Kde` for the
    # overlay). The vocabulary itself is not restated: `validate_bin_rule` IS
    # the parser those transforms construct through.
    if bins is not None:
        validate_bin_rule(bins, "displot")
    # ...and a well-formed rule that this `kind=` cannot act on is reported
    # too, rather than accepted and silently dropped.
    _warn_inert_bins(kind=kind, bins=bins)
    # Every "this needs x=" refusal, in displot's voice, before any chart is
    # built -- including the overlay case, which used to render a chart
    # missing the mark it was asked for.
    _reject_displot_without_x(kind=kind, x=x, y=y, kde=kde, rug=rug)
    # The chart this call draws once `encode=` applies, not just the x=/y=
    # keywords: a bin=/aggregate= reaching through `encode=`, or as a
    # shorthand string, is checked the same way as one written as a keyword.
    drawn = _drawn_bindings({"x": x, "y": y}, encode)
    if kind == "ecdf":
        _refuse_ecdf_field_y(drawn.get("y"))
    _refuse_displot_bin_aggregate(
        kind=kind,
        drawn=_drawn_positional_channels({"x": x, "y": y}, encode),
        stat=stat,
        multiple=multiple,
    )
    height, aspect = _checked_figure_size("displot", height=height, aspect=aspect)
    # Read once, here, at displot's own door: a shorthand hue parses as
    # encode() reads it, a column name the grammar cannot read binds as
    # written, and an aggregate is refused -- every consumer below
    # (the color binding, the groupby=/Dodge(by=) readers) reads the same
    # parsed channel.
    hue = _grouping_hue("displot", hue)

    return _displot_build(
        data,
        x=x,
        y=y,
        hue=hue,
        col=col,
        row=row,
        kind=kind,
        fill=fill,
        cumulative=cumulative,
        log_scale=log_scale,
        stat=stat,
        bins=bins,
        bandwidth=bandwidth,
        bw_adjust=bw_adjust,
        multiple=multiple,
        kde=kde,
        rug=rug,
        height=height,
        aspect=aspect,
        mark=mark,
        encode=encode,
        properties=properties,
        layers=layers,
        theme=theme,
        encode_kwargs=encode_kwargs,
    )


def _displot_build(
    data: Any,
    *,
    x: Any,
    y: Any,
    hue: Any,
    col: Any,
    row: Any,
    kind: str,
    fill: bool,
    cumulative: bool,
    log_scale: bool,
    stat: str,
    bins: Any,
    bandwidth: Any,
    bw_adjust: float,
    multiple: str,
    kde: bool,
    rug: bool,
    height: float | None,
    aspect: float | None,
    mark: dict | None,
    encode: dict | None,
    properties: dict | None,
    layers: list | None,
    theme: Any,
    encode_kwargs: dict[str, Any],
) -> Chart:
    """Construct the displot chart from validated inputs.

    The public ``displot`` validates ``kind``/``multiple``/``stat``/``bins``
    AND refuses every call whose distribution variable is missing
    (``_reject_displot_without_x``); this builder owns the mark/transform
    dispatch, optional kde/rug overlays, faceting, and sizing.  It therefore
    assumes ``x is not None`` wherever the kind or an overlay needs it -- the
    guard is not repeated here, so there is one answer to "this needs x=" and
    it is spelled in displot's voice.
    """
    # `hue` may be a bare column name or a channel object (`fm.Color("g")`) --
    # both are documented spellings, and #130 made an untyped `fm.Color` read
    # as nominal, so the channel spelling is now the one the guidance points
    # at. Only the color ENCODING wants the channel; every other consumer
    # below (the position adjustment's `by=`, and each transform's `groupby=`)
    # takes a bare column name and rejects anything else. Resolved once here,
    # via the package's shared extraction point, so no consumer re-derives it
    # and none can be left holding the raw object.
    hue_field = _field_name(hue)

    # What `multiple=` means, decided once: the position adjustment AND the
    # binning extent mode. Every consumer below reads this pair; none
    # re-derives either half (GH #194).
    #
    # `value_axis=None` is displot's own statement, not an omission: displot
    # exposes no `orientation=`, so every histogram and density it builds is
    # vertical and cumulates on y -- the same value the desugar derives from
    # its own (unset, therefore "vertical") `orientation=`. Giving displot a
    # horizontal orientation means changing this argument, not discovering
    # later that it was never passed.
    semantics = _multiple_to_position(multiple, hue_field, value_axis=None)
    position = semantics.position

    # Build the base chart.
    chart = Chart(data)

    # Encoding: x (required for most kinds), color from hue.
    #
    # `hue` is typed Nominal: every displot kind renders a mark that bins,
    # aggregates, or groups rows by it (hist -> bar, kde -> area, ecdf ->
    # line, rug -> tick), so a numeric hue column inferring Continuous
    # produces a fabricated colorbar over a group discriminator rather than
    # a swatch legend -- silently on the bar/tick kinds. See
    # `nominal_color_channel` for the rule.
    enc: dict = {}
    if x is not None:
        enc["x"] = x
    if y is not None:
        enc["y"] = y
    if hue is not None:
        enc["color"] = nominal_color_channel(hue)
    enc.update(encode_kwargs)

    # `log_scale` is resolved on the x channel here, before the kind
    # dispatch, so the dispatcher's own re-point for hist/kde/rug carries the
    # scale onto whatever column ends up bound, and a layered `kde=`/`rug=`
    # overlay inherits it from that same chart-level binding instead of
    # reading a linear axis. `ecdf`'s own `x=` is never re-pointed, so the
    # scale rides the caller's own binding directly.
    if log_scale and "x" in enc:
        enc["x"] = _log_scaled_x("displot", enc["x"], _field_name(enc["x"]), param="log_scale")

    # Mark + transforms by kind.
    if kind == "hist":
        # When `multiple` requires per-group binning (stack/fill/dodge) and a
        # hue is bound, thread `groupby=hue` so the Bin transform emits per-
        # (bin, group) rows preserving the hue column for color encoding +
        # position adjustment.
        #
        # `multiple=` IS forwarded (GH #194): it is what tells the `Bin`
        # transform to bin every group against one global extent under
        # "stack"/"fill"/"dodge", without which the groups land on different
        # edges at different widths -- `apply_stack` has no pairs to stack,
        # and a dodged slot has no shared band to take its place in.
        # `position=` stays explicit alongside it and is not redundant: for
        # "layer" the desugar declares no adjustment and displot wants an
        # explicit `Identity()` (see `_multiple_to_position`), and for the
        # other three modes the desugar's own adjustment -- which overwrites
        # this one -- is built by the same owner from the same three inputs
        # (`multiple`, `position_by`/`by`, and a `value_axis` both sides now
        # state rather than drop), so the two are equal by construction.
        hist_kwargs: dict = dict(
            bins=bins,
            cumulative=cumulative,
            density=(stat == "density"),
            multiple=multiple,
            position_by=hue_field,
            position=position,
        )
        if hue is not None and multiple in ("stack", "fill", "dodge"):
            hist_kwargs["groupby"] = hue_field
        chart = chart.mark_histogram(**hist_kwargs)
    elif kind == "kde":
        # When hue is bound, thread `groupby=hue` so the Kde transform emits
        # per-(grid, group) rows preserving the hue column for color encoding.
        # Position adjustment is informational for KDE (continuous curves
        # overlay regardless of `multiple`), but groupby is required whenever
        # hue is set so each level gets its own curve.
        kde_kwargs: dict = dict(
            bandwidth=bandwidth,
            bw_adjust=bw_adjust,
            fill=fill,
            cumulative=cumulative,
            position=position,
        )
        if hue is not None:
            kde_kwargs["groupby"] = hue_field
        chart = chart.mark_density(**kde_kwargs)
    elif kind == "ecdf":
        # ECDF: a per-observation rolling count -> step line.  `x is not
        # None` here by `_reject_displot_without_x`. The pending mark
        # resolves once `.encode()` applies below -- and again after any
        # later `.encode()` -- reading the chart's own x binding at that
        # point (`_resolve_ecdf`, `_PENDING_PREP["ecdf"]`): the one column
        # every door (this keyword, `encode=`, a post-hoc `.encode()`, or
        # shorthand) ends up drawing, so the window counts exactly what
        # the axis shows.
        chart = chart._set_composite_mark("ecdf", _resolve_ecdf, {}, placeholder="line")
        # Neither `x=` nor a fieldless `y=` is re-pointed here: both stay on
        # the caller's own binding, with every option it carries. The
        # window's own count column reaches `y` through `desugar_ecdf`'s
        # `remap` once the pending mark resolves, titled "count" by its
        # `remap_titles` unless the caller's own fieldless `y=` already
        # declares a title (`ferrum._desugar._apply_remap`).
    elif kind == "rug":
        chart = chart.mark_tick()

    chart = chart.encode(**enc)
    # The chart's own encoded x, read right here -- before any pending mark
    # resolves (a resolved hist/ecdf remaps x/y onto its own statistic's
    # output columns) and before the figure-level `encode=` override lands
    # (`_finalize_chart`, applied once this builder returns) -- so a
    # shorthand `x="v:Q"` is parsed the way `Chart.encode()` parses it, on
    # every kind's overlay below. Same shared extraction point as
    # `hue_field` above. Not `None` here: an overlay on an x-less call is
    # refused in displot's voice (`_reject_displot_without_x`) rather than
    # promoting the primary mark to layered mode and appending nothing.
    x_field = _field_name(chart._encoding.get("x"))

    # Optional kde/rug overlay layers (only when not already that kind).
    #
    # These overlays must NOT use Chart.__add__ because the histogram's Bin
    # transform replaces the original data columns (e.g. "sepal_length" →
    # "bin_start"/"bin_end"/"count").  __add__ merges all transforms into a
    # single sequential pipeline, so the Kde overlay would run on the Bin
    # output and fail to find the original column.
    #
    # Instead we resolve the chart, then manually prepend named transforms
    # for the overlays.  Named transforms read from the current chain head
    # (the original data, since they run before the primary mark's own
    # unnamed statistic -- a `Bin` for hist, a `DataWindow` for ecdf)
    # without advancing it.  The overlay _Layer objects set data_source to
    # the named transform key so they read the correct output.
    if (kde and kind != "kde") or (rug and kind != "rug"):
        chart = chart._resolve_pending()
        chart = chart._clone()

        # Convert single-mark chart to layered mode: promote the existing
        # mark + encoding into a primary layer so overlay layers can be
        # appended alongside it.
        if chart._layers is None and chart._mark is not None:
            primary_layer = _Layer(
                mark=chart._mark,
                encoding=dict(chart._encoding),
                mark_kwargs=dict(chart._mark_kwargs) if chart._mark_kwargs else None,
                position=chart._position,
            )
            chart._layers = [primary_layer]
            chart._mark = None
        drawn_layers = chart._layers
        if not drawn_layers:
            # RuntimeError is the project's internal-invariant type (the
            # `_compose` rebuild guards): no public input reaches a resolved
            # displot without its primary mark, so no remedy can be named.
            raise RuntimeError(
                "displot: the primary mark resolved to no layer, so its overlays have "
                "nothing to join"
            )

        if kde and kind != "kde":
            primary_encoding = drawn_layers[0].encoding
            kde_transform, kde_layer = _kde_overlay(
                x_field,
                hue=hue,
                kind=kind,
                stat=stat,
                multiple=multiple,
                shared_extent=semantics.shared_extent,
                bins=bins,
                bandwidth=bandwidth,
                bw_adjust=bw_adjust,
                cumulative=cumulative,
                x_title=_displayed_name(primary_encoding.get("x")),
                y_title=_displayed_name(primary_encoding.get("y")),
            )
            # The named Kde sits right after the statistics marker, ahead of
            # the primary mark's own unnamed statistic (`Bin` for hist,
            # `DataWindow` for ecdf), so it reads the caller's rows, after
            # any `.transform(...)` step, never the bins.
            chart = chart._with_statistics(kde_transform, reading="rows")
            chart._layers.append(kde_layer)
            # The chart's own value axis, for the one kind whose primary mark
            # has none (GH #201). `kind="rug"` binds x only: `hist` remaps
            # y -> the Bin count column, `ecdf` binds the window's count
            # column above, and `kind="kde"` never reaches this block. So
            # the rug is the only primary mark that leaves the root encoding
            # without a y, and a density curve drawn over it has no
            # chart-level axis to name; left to the overlay layer's own `y`
            # alone, the axis would come out untitled.
            #
            # Taken from the overlay layer's own `y` rather than re-derived
            # from `registry_output_columns("kde", ...)`: the axis and the
            # curve then name one column by construction, so a Rust-side
            # rename of the Kde output cannot retitle the axis off the data it
            # describes. The title that follows is the density column's own
            # name, which is what `kind="kde"` (and `kind="kde", rug=True`,
            # the same two marks by the documented other route) already
            # titles this axis -- the two spellings of one chart read
            # identically. The promoted primary layer keeps its own x-only
            # encoding, so the ticks still sit at the value axis's base.
            #
            # Wrapped in `Y` because this runs after `_resolve_pending()`:
            # `chart._encoding` holds channel objects by now, and the bare
            # string spelling the `enc` dict accepts is no longer resolved.
            # The overlay's own `y` is that bare density column here: the rug
            # binds no y, so the overlay names its own column.
            if "y" not in chart._encoding:
                from ferrum.encoding import Y as _Y_overlay

                chart._encoding = {
                    **chart._encoding,
                    "y": _Y_overlay(kde_layer.encoding["y"]),
                }

        if rug and kind != "rug":
            from ferrum.marks._desugar_helpers import identity_transform

            rug_xform = identity_transform("rug_data")
            # The named Identity sits right after the statistics marker,
            # ahead of the primary mark's own unnamed statistic (`Bin` for
            # hist, `DataWindow` for ecdf), so it captures the caller's rows
            # for the rug tick marks, after any `.transform(...)` step.
            chart = chart._with_statistics(rug_xform, reading="rows")
            chart._layers.append(
                _Layer(
                    name="rug",
                    mark="tick",
                    encoding={"x": x_field},
                    data_source="rug_data",
                    # Explicit Identity(), not inherited chart-level position:
                    # a rug tick mark is never meaningfully stacked, dodged,
                    # or normalized either (same class of bug as the kde
                    # overlay above -- see `_kde_overlay`'s docstring).
                    position=Identity(),
                )
            )

    # Name the layers so override passthrough can target them.
    if chart._layers is not None:
        from dataclasses import replace as _dc_replace

        _kind_names = {
            "bar": "histogram",
            "area": "kde",
            "line": "kde",
            "tick": "rug",
            "point": "scatter",
        }
        seen: dict[str, int] = {}
        named: list = []
        for ly in chart._layers:
            if ly.name is not None:
                named.append(ly)
            else:
                base = _kind_names.get(ly.mark or "", ly.mark or "layer")
                count = seen.get(base, 0)
                seen[base] = count + 1
                lname = base if count == 0 else f"{base}_{count + 1}"
                named.append(_dc_replace(ly, name=lname))
        chart._layers = named

    # Faceting.
    if col is not None or row is not None:
        if col is not None and row is not None:
            chart = chart.facet(row=row, col=col)
        elif col is not None:
            chart = chart.facet(col=col)
        else:
            chart = chart.facet(row=row)

    # Properties: delegate to shared helper so displot and catplot cannot drift.
    chart = _apply_facet_sizing(chart, data=data, row=row, height=height, aspect=aspect)

    return _finalize_chart(
        chart,
        entry="displot",
        mark=mark,
        encode=encode,
        properties=properties,
        layers=layers,
        theme=theme,
    )


# ---------------------------------------------------------------------------
# catplot
# ---------------------------------------------------------------------------


def _is_default_value(value: Any, default: Any) -> bool:
    """Return whether *value* equals *default*, treating an incomparable value as different."""
    try:
        return bool(value == default)
    except (TypeError, ValueError, ArithmeticError):
        return False


def _refuse_interval_args_on_non_aggregating_kind(kind: str, ci: Any, n_boot: Any) -> None:
    """Refuse an explicit ``ci=``/``n_boot=`` on a kind that draws no interval.

    Only the kinds in :data:`_CATPLOT_AGGREGATING_KINDS` compute a bootstrap
    interval; on any other kind a non-default ``ci=`` or ``n_boot=`` would be
    accepted and silently ignored. The default values pass, since a caller
    who never wrote the argument cannot be told apart from one who wrote the
    default.
    """
    estimating = ", ".join(repr(k) for k in sorted(_CATPLOT_AGGREGATING_KINDS))
    for name, value, default in (
        ("ci", ci, _CATPLOT_DEFAULT_CI),
        ("n_boot", n_boot, _CATPLOT_DEFAULT_N_BOOT),
    ):
        if not _is_default_value(value, default):
            raise ValueError(
                f"catplot({name}={value!r}) is not supported with kind={kind!r} -- "
                f"that kind draws no confidence interval, so {name}= would be ignored. "
                f"Drop {name}=, or use one of the estimating kinds ({estimating})."
            )


def _catplot_interval_args(ci: Any, n_boot: Any) -> tuple[float, int]:
    """Validate ``ci``/``n_boot`` and return the CI as a level in ``(0, 1)`` and ``n_boot`` as an ``int``.

    ``ci`` is a PERCENT on catplot's public signature (``ci=95``) but a LEVEL
    on ``Summary`` (``ci=0.95``); this is the one site that converts between
    them, so the two spellings cannot drift. Both bounds are exclusive: a
    0 % interval has no width and a 100 % bootstrap interval is not
    estimable from a finite resample.

    Called only for the kinds that actually consume these arguments
    (:data:`_CATPLOT_AGGREGATING_KINDS`); the other six kinds refuse any
    non-default value through
    :func:`_refuse_interval_args_on_non_aggregating_kind`.

    Whether each is a number is decided by the shared predicates
    (:func:`ferrum._validate.as_finite_real`,
    :func:`ferrum._validate.coerce_positive_int`): ``bool`` is refused for
    both (it subclasses ``int``, so ``n_boot=True`` would otherwise mean
    "1 resample" and ``ci=True`` "a 1 % interval"), and so is a numeric
    string (``ci="95"``). ``ci=1`` and ``n_boot=1`` stay accepted, as do
    numpy scalars; ``n_boot`` is returned as the builtin ``int`` the shared
    predicate coerces it to.
    """
    # Message shape follows this family's house form -- `func(param=value) is
    # not supported -- <why>. <alternative>` -- the same one
    # `_reject_per_row_appearance_channels` and `lmplot`'s refusals use.
    # The shared predicates decide "is this a number"; a refusal is raised
    # in the house form, which also states the accepted range.
    ci_percent = as_finite_real(ci)
    if ci_percent is None or not 0.0 < ci_percent < 100.0:
        raise ValueError(_CI_REFUSAL.format(ci=ci))
    try:
        n_boot_count = coerce_positive_int("catplot(n_boot=...)", n_boot, required=True)
    except ValueError as exc:
        raise ValueError(
            f"catplot(n_boot={n_boot!r}) is not supported -- n_boot is the bootstrap "
            "resample count and must be an integer >= 1 and at most 2**53. Pass n_boot=1000 (the default) "
            "or any positive count."
        ) from exc
    return ci_percent / 100.0, n_boot_count


# The short type codes (`Channel.option("type")`) whose scale is continuous
# -- "Q" (quantitative) and "T" (temporal) -- and therefore has no discrete
# legend order for `hue_order=` to apply. "N" (nominal) and "O" (ordinal)
# both resolve to discrete scales and are excluded here on purpose: an
# explicit `type_="ordinal"`/`"nominal"` hue channel + `hue_order=` is the
# honored, byte-identical path (`test_catplot_channel_object_hue_explicit_type_honored_with_hue_order`).
_CATPLOT_CONTINUOUS_HUE_TYPES = frozenset({"Q", "T"})
_CATPLOT_TYPE_CODE_NAMES = {"Q": "quantitative", "T": "temporal"}


def _catplot_cat_field_parts(cat_field: Any) -> tuple[str | None, str | None, dict]:
    """Split catplot's category-channel argument into name / explicit type / kwargs.

    Accepts the documented ``x : str or encoding`` shape uniformly (cycle-2
    fix, quality finding 3 / spec issue 1): a bare string has no type_ and
    no other kwargs to preserve; an already-built channel object (e.g.
    ``x=fm.X("year")``) yields its ``.field``, its own ``.option("type")``
    (the already-normalized short code ``"Q"``/``"N"``/``"O"``/``"T"``, or
    ``None`` when the caller stated no ``type_``), and a COPY of its
    ``_kwargs`` (``sort=``, ``scale=``, ``axis=``, ``title=``, ...) so a
    caller can rebuild a fresh channel -- typed and/or re-sorted -- without
    mutating or discarding the original. ``None`` returns ``(None, None,
    {})``.
    """
    if cat_field is None:
        return None, None, {}
    name = _field_name(cat_field)
    if isinstance(cat_field, str):
        return name, None, {}
    return name, cat_field.option("type"), dict(cat_field._kwargs)


def _catplot_cat_type(
    data: Any, cat_field: str | None, *, native_scale: bool, explicit_type: str | None = None
) -> str | None:
    """Resolve catplot's category-channel ``type_`` from ``native_scale``.

    This is the ONE site that decides how catplot's category channel is
    typed, for every one of the 8 kinds (F-L09-03) AND both input shapes
    (bare field name or an already-built channel object, cycle-2 fix):
    ``native_scale=False`` (the default) resolves to ``"N"`` -- even a
    numeric category column renders as categorical bands. That is a
    disclosed behavior change: before this, the category channel was never
    explicitly typed and a numeric column silently inferred
    ``"quantitative"``, which contradicted the documented default (seaborn
    parity is the new behavior). ``native_scale=True`` resolves to ``"Q"``
    and requires the column to actually be numeric -- a non-numeric column
    has no native scale to place values on, so this raises naming the
    column and its dtype rather than silently falling back to nominal.

    ``explicit_type`` -- the caller's own ``type_=`` statement on a channel
    object they built themselves (``x=fm.X("cat", type_="Q")``), extracted
    by :func:`_catplot_cat_field_parts` -- is honored VERBATIM and returned
    immediately, bypassing ``native_scale``'s own typing decision (including
    its numeric-dtype check) entirely: this is the same pass-through
    precedent ``nominal_color_channel`` already sets for an explicit
    ``hue=`` channel -- the caller stated the type, so ``native_scale``
    does not overrule it.

    Returns ``None`` when ``cat_field`` is ``None`` (no category channel
    bound yet): callers that require one raise their own, more specific,
    "requires x=/y=" message downstream, and there is nothing to type-check
    here.
    """
    if cat_field is None:
        return None
    if explicit_type is not None:
        return explicit_type
    if not native_scale:
        return "N"
    frame = to_polars(data)
    if cat_field not in frame.columns:
        # Same wording as the Rust render-time check (`render/mod.rs`'s
        # `RenderError::UnknownColumn`) so a missing column reads the same
        # way regardless of which side of the CDI boundary catches it first
        # -- this Python-side dtype check runs eagerly, at `catplot()` call
        # time rather than `.to_svg()` render time, so it would otherwise
        # surface a raw `KeyError` for exactly the case that error exists to
        # explain.
        raise ValueError(f"unknown column {cat_field!r} referenced by an encoding")
    dtype = frame.schema[cat_field]
    if not dtype.is_numeric():
        raise ValueError(
            f"catplot(native_scale=True): the category column {cat_field!r} is "
            f"not numeric (dtype {dtype}), so it has no native scale to place "
            "values on. Pass native_scale=False (the default) to render it as "
            "categorical bands instead."
        )
    return "Q"


class _NativeScaleStats(NamedTuple):
    """Finite, deduplicated data-unit stats for a native-scale category column.

    Shared by the jitter-width and domain-padding computations below so both
    read the exact same filtered/sorted values instead of two independent
    (and potentially divergent) passes over the data.
    """

    data_min: float
    data_max: float
    #: Upper-median spacing between consecutive unique values (Rust's own
    #: `diffs[diffs.len() / 2]` convention, `render/position.rs`'s
    #: continuous-band `Dodge`), or `None` when fewer than 2 unique finite
    #: values exist to derive a spacing from.
    median_spacing: float | None


def _native_scale_stats(data: Any, cat_field: str) -> "_NativeScaleStats | None":
    """Finite, deduplicated, sorted stats for a native-scale category column.

    Filters non-finite values (NaN/±inf) in addition to nulls (cycle-2 fix,
    quality finding 2): ``drop_nulls()`` alone does not drop NaN, and
    ``sorted()`` over a diff sequence containing NaN is not a total order,
    so a NaN/inf category value previously poisoned the jitter-width
    computation into returning NaN/inf, which then failed at render time
    with an opaque JSON-serialization error naming nothing. A non-finite
    category value cannot contribute a meaningful spacing or extent, and
    excluding it from this computation does not stop ITS OWN row from
    rendering (a NaN x position is not touched by jitter/dodge either way;
    without them, the same frame already renders that row fine).

    Returns ``None`` when there are no finite values at all (a fully
    null/NaN/inf column) -- callers skip width-scaling/padding in that case
    since there is nothing to scale around.
    """
    series = to_polars(data).get_column(cat_field).drop_nulls()
    series = series.filter(series.is_finite())
    values = series.unique().sort().to_list()
    if not values:
        return None
    if len(values) == 1:
        return _NativeScaleStats(values[0], values[0], None)
    diffs = sorted(b - a for a, b in zip(values, values[1:]))
    median_spacing = diffs[len(diffs) // 2]
    return _NativeScaleStats(values[0], values[-1], median_spacing)


def _native_scale_jitter_width(
    stats: "_NativeScaleStats | None", *, base_width: float = 0.4
) -> float:
    """Jitter width for a native-scale (quantitative) category axis, in data units.

    ``native_scale=True`` types the category channel Quantitative, so it
    renders on a linear axis rather than a band -- Rust's non-ordinal
    ``Jitter`` branch (``render/position.rs``'s ``apply_jitter``) then
    applies ``width`` as a literal DATA-unit offset instead of a fraction of
    a band's pixel width. The ordinal default, ``base_width=0.4``, is a
    BAND-FRACTION: applied verbatim as a data-unit offset it is the wrong
    magnitude on any real numeric domain (e.g. ``±0.2`` total spread on a
    column spanning ``[0, 1000]`` is imperceptible; the same width on a
    ``[0, 1]`` column would swallow the whole axis).

    This mirrors the bandwidth proxy continuous-band ``Dodge`` already
    computes for the identical situation (``apply_dodge``'s "median spacing
    of unique band values"): scaling ``base_width`` by the category column's
    own median spacing between unique values keeps the jittered spread
    proportional to how far apart the categories actually sit, computed
    Python-side from the real data rather than guessed at a fixed magnitude.

    Falls back to ``base_width`` itself (0.4 data units) when there are
    fewer than two unique finite values to derive a spacing from -- a
    single-category column, an all-null/non-finite one, or ``stats is
    None`` -- since there is no spacing to scale and 0.4 is a small,
    bounded, deterministic default for that degenerate case.
    """
    if stats is None or stats.median_spacing is None:
        return base_width
    return base_width * stats.median_spacing


def _native_scale_padded_domain(
    stats: "_NativeScaleStats | None",
    *,
    jitter_active: bool,
    dodge_active: bool,
    jitter_base_width: float = 0.4,
) -> tuple[float, float] | None:
    """Domain wide enough that a Jitter/Dodge offset never lands outside it.

    Cycle-2 fix (quality finding 1 / S3, blocks): typing the category
    channel Quantitative makes its axis domain the raw data extent
    (``numeric_extent`` / ``column_min_max_f64``, resolved with no
    awareness that a position adjustment is about to move points). Both
    ``Jitter`` (a literal data-unit offset, ``render/position.rs``'s
    ``apply_jitter``) and continuous-band ``Dodge`` (a bandwidth-based
    offset, ``apply_dodge``) can then place a mark up to half their own
    extent away from its raw position -- Jitter's ``width/2``, Dodge's
    ``bandwidth/2`` (the outermost sub-band's offset is always strictly
    less than half the bandwidth by construction; see
    ``apply_dodge_ordinal``'s formula) -- and the renderer silently DROPS
    any mark whose adjusted position lands outside the domain it inferred:
    no error, no warning. Measured pre-fix: 17 of 30 marks under ``Jitter``,
    6 of 12 under ``Dodge``, on ordinary (non-adversarial) fixtures.

    The fix is Python-side, at the SAME single typing site this module
    already resolves the category channel's type at: attach an EXPLICIT
    domain wide enough to absorb the largest possible offset, plus a 5%
    safety margin against float-rounding pushing a mark that lands exactly
    on the padded edge back outside it. The renderer does not clip inside
    an explicit domain the way it re-derives one from the data, so widening
    it here is sufficient with no Rust change (`NS-1` in the code
    archaeology followups doc covers the underlying Rust divergence this
    module also works around).

    Returns ``None`` -- meaning "leave the axis domain exactly as it would
    already resolve, do not touch it" -- in two cases: ``stats is None``
    (nothing to pad around), or neither ``jitter_active`` nor
    ``dodge_active`` (no position adjustment will run for this kind/call,
    so the data's own extent is already the correct domain). Callers must
    ALSO skip attaching this domain when the caller already supplied their
    own explicit scale/domain for the category channel -- an explicit
    caller domain must still govern (checked at each call site, not here,
    since it depends on which of catplot's several encoding doors the
    caller used).
    """
    if stats is None:
        return None
    half_extent = 0.0
    if dodge_active and stats.median_spacing is not None:
        half_extent = stats.median_spacing / 2
    if jitter_active:
        jitter_width = _native_scale_jitter_width(stats, base_width=jitter_base_width)
        half_extent = max(half_extent, jitter_width / 2)
    if half_extent <= 0.0:
        return None
    pad = half_extent * 1.05
    return (stats.data_min - pad, stats.data_max + pad)


def _catplot_summary_layers(
    *,
    kind: str,
    cat_channel: Any,
    val_field: str,
    color_channel: Any,
    position: Any,
    chart_encoding: dict,
    ci_level: float,
    n_boot: int,
    seed: int | None,
) -> tuple[Any, list]:
    """Build the ``Summary`` transform and the two layers ``kind`` renders.

    ``kind="point"`` draws the CI whisker first and the mean marker over it;
    ``kind="bar"`` draws the bar (zero to the mean) first and the whisker over
    it. Both read one ``Summary`` output through ``data_source=``, so the
    bootstrap runs once per chart regardless of which kind asked for it.

    Layer geometry follows ``desugar_errorbar`` (``marks/composite.py``): a
    ranged ``rule`` from ``lower`` to ``upper`` in the neutral label color.
    The caps that desugar adds are deliberately absent -- spec §4 describes
    point/bar's interval as a plain whisker, matching seaborn's capless
    ``pointplot``/``barplot`` default.

    ``cat_channel`` and ``color_channel`` are the ALREADY-RESOLVED category
    and color channels ``_catplot_build`` binds at the chart level -- the
    same objects, not a rebuild. That is the whole of this function's channel
    handling, and it is deliberate (design-review F1): these layers used to
    reconstruct both channels from bare field names and re-thread three
    options by hand (``type_``, ``order=``, the ``cat_domain`` padding),
    which silently dropped every OTHER option the caller had stated on their
    own channel object -- ``x=fm.X("cat", sort=[...])`` and
    ``hue=fm.Color("g", sort=[...])`` were honored by the six per-row kinds
    and discarded here. A layered chart resolves its axis and color scales
    from the LAYER encodings, so a channel left only at the chart level is
    inert for these two kinds; taking the resolved channel is what makes the
    encoding spelling and the ``order=``/``hue_order=`` kwarg spelling land
    in one place instead of two, and makes the next channel option a
    zero-site change here.

    ``position`` is set EXPLICITLY on both layers rather than inherited from
    the chart: whisker and marker must carry the SAME adjustment or they
    separate (a dodged marker beside an undodged whisker), and an inherited
    chart-level adjustment is how the displot overlays previously acquired a
    ``Stack`` they had no business honoring (see ``_kde_overlay``).

    The value axis is titled on BOTH layers with the caller's ``title=`` on
    their value channel, else the original value column, so the axis reads
    ``"tip"`` rather than the transform's output column name -- whichever
    layer is first (it differs by kind) owns the axis title.

    No value-axis scale is invented here, and that is what keeps
    ``kind="bar"`` zero-anchored: the layered-bar rule in
    ``render/scale_resolve/domain.rs::numeric_domain_union`` widens the
    AUTO-INFERRED value-axis extent to include zero whenever a bar layer binds
    that channel (suppressed when the bar binds an explicit span channel,
    whose extent is taken literally), so the bar's baseline is on the axis
    while the union still spans every layer's ``mean``/``lower``/``upper``.
    That rule fires on the auto path ONLY, so fabricating a scale here --
    e.g. ``scale={"type": "linear", "zero": True}``, the mechanism a
    single-mark ``mark_bar`` gets from ``_spec_build``'s injection -- would
    both destroy the anchor it was meant to add and replace the cross-layer
    union with a single-column domain, clipping every upper CI above
    ``max(mean)`` off the panel (measured: 3 of 4 whiskers vanished).

    A scale the CALLER set is a different matter and is honored. On the
    category channel it simply rides ``cat_channel`` in, like every other
    option the caller stated; on the VALUE channel -- whose layer bindings are
    fresh ``Y(<summary column>)`` objects -- ``chart_encoding`` is propagated
    onto both layers through the shared ``_propagate_layer_axis`` (the same
    mechanism the composite-mark desugars and ``lmplot``'s fit layers use), so
    ``y=fm.Y("tip", scale={"domain": [0, 120]})`` governs the axis exactly as
    it does for ``kind="box"``. The one combination refused rather than
    honored is a caller scale on the VALUE channel that carries no ``domain``
    (e.g. a bare ``{"type": "log"}``) -- see ``_reject_domainless_value_scale``.
    """
    from ferrum import Summary
    from ferrum.encoding import Y as _Y

    _reject_domainless_value_scale(kind, chart_encoding)

    # The data-side names behind the two resolved channels. Read off the
    # channels themselves rather than passed alongside them, so the columns
    # the transform groups by cannot drift from the columns the layers bind.
    cat_field = _field_name(cat_channel)
    hue_field = _field_name(color_channel)

    # The shared "hue joins the groupby unless it IS the category" rule
    # (`desugar_errorbar` uses the same helper): a duplicate groupby column is
    # rejected by the Rust transform. Grouping by (category, hue) is also what
    # keeps the hue column in the summary batch for the layers' color channel
    # to read -- the same coupling `kind="count"` needs.
    #
    # `split_hue` is deliberately NOT consulted for the color binding, which
    # is where this diverges from `desugar_errorbar`: when `hue` names the
    # category column itself, errorbar drops the color channel, while catplot
    # keeps it, so `catplot(x="cat", hue="cat")` still colors its estimates by
    # category (verified rendering). Only the groupby needs the de-duplication.
    groupby, _split_hue = resolve_color_groupby(cat_field, hue_field, [cat_field])

    transform = Summary(
        val_field,
        groupby=groupby,
        error_fn="ci",
        ci=ci_level,
        n_boot=n_boot,
        # `seed=None` means "unspecified", not "unseeded": the bootstrap is
        # ChaCha8-seeded in Rust and ferrum's byte-determinism rule leaves no
        # room for an entropy-seeded default, so an unset seed maps to 0.
        seed=seed if seed is not None else 0,
        name=_CATPLOT_SUMMARY_NAME,
    )

    # Derived from the registry (spec §4.6 / M6a) rather than hardcoded, so a
    # Rust-side rename of a Summary output column raises here -- naming the
    # stale column -- instead of silently rendering an empty panel.
    cols = registry_output_columns("summary", ["mean", "lower", "upper"])

    layer_position = position if position is not None else Identity()
    # The value axis's name on both layers: the caller's title= on their value
    # channel, else the value column. A suppressed title keeps the column: the
    # caller's chart-level channel already draws no axis title.
    value_title = drawn_title(chart_encoding.get("y")) or val_field

    def _encoding(y_col: str, y2_col: str | None = None) -> dict:
        enc: dict = {"x": cat_channel, "y": _Y(y_col, title=value_title)}
        if y2_col is not None:
            enc["y2"] = y2_col
        if color_channel is not None:
            # `color_channel` is `_catplot_build`'s own resolved hue channel,
            # which is built through `nominal_color_channel`/`Color` there --
            # so this binding is typed at its single construction site rather
            # than re-typed here. `tests/test_color_binding_completeness.py`
            # reads these bindings syntactically and accepts a name that a
            # typed constructor assigned, which is the shape this is.
            enc["color"] = color_channel
        return enc

    whisker = _Layer(
        name="whisker",
        mark="rule",
        encoding=_encoding(cols["lower"], cols["upper"]),
        mark_kwargs={"stroke": "theme:label", "stroke_dash": []},
        data_source=_CATPLOT_SUMMARY_NAME,
        position=layer_position,
    )
    estimator = _Layer(
        # Both estimating kinds are named for a primitive mark, so `kind`
        # doubles as the layer's mark and its override key (`mark={"bar":
        # {...}}`); a future kind whose name is not a mark name must split
        # these two rather than reuse this coincidence.
        name=kind,
        mark=kind,
        encoding=_encoding(cols["mean"]),
        data_source=_CATPLOT_SUMMARY_NAME,
        position=layer_position,
    )
    layers = [whisker, estimator] if kind == "point" else [estimator, whisker]
    # The value channel is bound to a bare Summary output column, which is
    # exactly the shape `_propagate_layer_axis` exists to repair: without it
    # a caller's explicit positional `scale=` (and its `format=`/
    # `format_type=`/`axis=` presentation) is silently dropped, as it was
    # for every composite desugar before GH #45. (The category channel needs
    # no repair -- it IS the caller's channel -- and the propagation is a
    # no-op for a layer channel that already carries its own.) Returns each
    # layer unchanged when no positional channel carries any of these,
    # so the ordinary call is a no-op rather than a branch.
    return transform, [_propagate_layer_axis(chart_encoding, layer) for layer in layers]


def _reject_domainless_value_scale(kind: str, chart_encoding: dict) -> None:
    """Refuse a caller's value-channel ``scale=`` that carries no ``domain``.

    An explicit ``ScaleSpec`` routes the value axis onto Rust's
    ``build_from_scale_spec``, which resolves a domain-less scale from the
    ONE column the axis-defining layer encodes. For these two kinds the layers
    encode DIFFERENT columns of the same ``Summary`` output (``mean`` on the
    estimator, ``lower``/``upper`` on the whisker), so that single-column
    domain is narrower than the data and silently clips: measured on
    ``kind="point"`` with ``scale={"type": "log"}``, 3 of 4 whiskers rendered.
    The same limitation is visible today on ``kind="box"`` (1 of 4 whiskers)
    -- it is a renderer gap, not something these kinds introduce.

    Refused rather than propagated-and-clipped, and refused rather than
    silently dropped: a scale the caller set must either govern the axis or
    say why it cannot. A ``domain`` makes it governable -- Rust then uses the
    caller's bounds verbatim for every layer, and any clipping is the domain
    the caller asked for -- so the message names that route.

    **Delete condition, not a permanent rule.** The real close is Rust:
    ``build_from_scale_spec`` (``render/scale_resolve/positional.rs``) resolves
    a domain-less explicit scale through ``resolve_continuous_domain_and_range``,
    which reads a single column, while the cross-layer union lives only in
    ``numeric_domain_union`` on the auto-inferred path. WHEN THE EXPLICIT-SCALE
    PATH GAINS THAT UNION, DELETE THIS FUNCTION AND ITS CALL -- do not relax it
    to a warning: the propagation in ``_catplot_summary_layers`` already handles
    the honored case, so the refusal simply stops being reachable. Tracked in
    ``design-docs/superpowers/followups/2026-05-15-code-archaeology.md``
    (batch-F row, 2026-09-10); the same gap silently clips ``kind="box"`` and
    ``kind="boxen"`` today, so closing it fixes more than these two kinds.

    Only the value channel is affected. A scale on the CATEGORY channel is
    safe and propagates normally: every layer binds the same category column,
    so its single-column domain IS the union.
    """
    value_scale = _explicit_positional_scale(chart_encoding.get("y"))
    if value_scale is None or _scale_domain(value_scale) is not None:
        return
    raise ValueError(
        f"catplot(kind={kind!r}): a value-axis scale= without an explicit domain "
        f"cannot be honored, because {kind!r} draws an estimate and its CI whisker "
        "from different columns of one Summary output, and an explicit scale "
        "resolves its domain from a single column -- the whiskers would be "
        "silently clipped. Add a domain (e.g. "
        "scale={'type': 'log', 'domain': [lo, hi]}), or use a kind that draws "
        "one mark per row."
    )


def _reject_per_row_appearance_channels(kind: str, caller_channels: dict) -> None:
    """Refuse per-observation appearance channels on the estimating kinds.

    ``catplot`` forwards any encoding channel a caller names -- through
    ``**encode_kwargs`` (``catplot(..., size="sz")``) or the ``encode=`` dict
    -- straight onto the chart. On the six kinds that draw one mark per row
    that is exactly right. On ``kind="point"``/``"bar"`` the marks are
    ESTIMATES over a ``Summary`` aggregate: there is no observation for a
    per-row visual property to read, so the channel cannot reach the mark --
    and the chart-level encoding it lands on still builds the channel's
    LEGEND. Measured on the pre-refusal tree: a ``size=`` column drew every
    estimator mark at the default ``r=3.385`` beneath a six-swatch size
    legend, and ``opacity=`` was byte-identical to no argument at all.
    A legend asserting a mapping the marks do not carry is the same defect
    class the value-scale refusal above exists for -- "a materially wrong
    statistical chart, drawn silently" -- so it is refused, not dropped.

    :data:`_CATPLOT_PER_ROW_APPEARANCE_CHANNELS` is the refused set, checked
    against the aggregated output channel by channel rather than assumed.
    The color family (``color``/``fill``/``stroke``) is in it and ``hue=`` is
    NOT, which is not an inconsistency: ``hue=`` joins the ``Summary``
    groupby, so the estimate really is split by it and the legend it draws is
    true. A raw ``color=``/``fill=``/``stroke=`` reaches the chart-level
    encoding only -- it splits nothing, colors nothing, and draws a legend --
    so its route is ``hue=``, which the message names.

    The information channels are deliberately NOT in the set: they assert no
    visual mapping and fabricate no legend, so refusing them would reject
    calls that are merely inert. Their inertness has two different causes,
    with two different futures, and only one of them is about aggregation:

    - ``text`` and ``detail`` are inert for a MARK-LEVEL reason -- ``point``
      and ``bar`` do not read them, and ``text=`` is equally inert on
      ``kind="strip"``. ``detail`` already says so, through
      ``alias_detail_channel``'s own warning. A catplot-specific refusal here
      would be a second mechanism answering a question the alias layer
      already answers, in a different voice.
    - ``tooltip``, ``href``, ``description``, ``url`` and ``key`` are inert
      because the summary batch does not carry the source column. These are
      the ones with a real future: a tooltip reading "mean 3.2, CI [2.1,
      4.3]" is the natural affordance of an estimate mark, and ``Summary``'s
      output already holds the columns to build it. The tracked follow-up is
      therefore to THREAD them onto the summary layers -- refusing them would
      foreclose that feature.

    ``tests/test_catplot_aggregation.py`` pins both the refused set and the
    inert one, and asserts that every channel the renderer honors falls into
    exactly one of them (or is resolved by the builder), so a channel added
    to ``_RENDERER_HONORED_CHANNELS`` forces a decision here instead of
    defaulting to silence.
    """
    refused = [name for name in caller_channels if name in _CATPLOT_PER_ROW_APPEARANCE_CHANNELS]
    if not refused:
        return
    named = ", ".join(f"{name}=..." for name in sorted(refused))
    raise ValueError(
        f"catplot(kind={kind!r}, {named}) is not supported -- those channels map a "
        f"visual property per OBSERVATION, and kind={kind!r} draws one estimate per "
        "(category, hue) group from a Summary aggregate, so no row is left for them "
        "to read: the estimator marks would render uniform under a legend asserting "
        "a mapping they do not carry. Split the estimate with hue= (the one "
        "non-positional channel these kinds honor -- it joins the Summary groupby), "
        "or use a kind that draws one mark per row (strip, swarm, box, violin, "
        "boxen) for a per-observation mapping."
    )


@_figure_entry
def catplot(
    data: Any,
    *,
    x: Any = None,
    y: Any = None,
    hue: Any = None,
    col: Any = None,
    row: Any = None,
    kind: str = "strip",
    order: Any = None,
    hue_order: Any = None,
    orient: Any = None,
    dodge: bool = False,
    jitter: bool = True,
    native_scale: bool = False,
    ci: Any = _CATPLOT_DEFAULT_CI,
    n_boot: int = _CATPLOT_DEFAULT_N_BOOT,
    seed: int | None = None,
    height: float | None = None,
    aspect: float | None = None,
    mark: dict | None = None,
    encode: dict | None = None,
    properties: dict | None = None,
    layers: list | None = None,
    theme: Any = None,
    **encode_kwargs: Any,
) -> Chart:
    """Categorical figure-level function.

    Dispatches to the appropriate mark based on ``kind``:

    * ``"strip"``  -- ``mark_point`` with ``Jitter`` (when ``jitter=True``, default) or ``Dodge`` (when ``dodge=True`` and ``hue`` is set).
    * ``"swarm"``  -- ``mark_swarm``.
    * ``"box"``    -- ``mark_boxplot`` (box + whiskers + outliers).
    * ``"violin"`` -- ``mark_violin`` (kernel-density outline).
    * ``"boxen"``  -- ``mark_boxen`` (letter-value / extended box).
    * ``"point"``  -- ``Summary`` + one mean marker per category group over a bootstrap-CI whisker.
    * ``"bar"``    -- ``Summary`` + one mean bar per category group under a bootstrap-CI whisker.
    * ``"count"``  -- ``mark_bar`` at ``y=count()``, titled ``n``; a ``layers=`` label
      ``fm.Layer(mark="text", encoding={"text": "count()"})`` draws each bar's count.

    Parameters
    ----------
    data : DataFrame-like
        Input data accepted by ``Chart(data)``.
    x : str or encoding, optional
        Column name for the horizontal axis (categorical by default).
    y : str or encoding, optional
        Column name for the vertical axis (value by default).
    hue : str or encoding, optional
        Column name to map to color (one visual group per level).
    col : str, optional
        Column name for faceting across columns.
    row : str, optional
        Column name for faceting across rows.
    kind : {"strip", "swarm", "box", "violin", "boxen", "point", "bar", "count"}, default "strip"
        Which categorical mark to draw.
    order : list of str, optional
        Explicit ordering for the categorical axis levels.  Passed as
        ``sort=order`` on the categorical-axis encoding so the domain
        renders in the given order.
    hue_order : list of str, optional
        Explicit ordering for hue levels.  Passed as ``sort=hue_order``
        on the color encoding.
    orient : {"h", "v", None}, optional
        ``"h"`` flips the axes (``x`` becomes the value axis, ``y`` the category
        axis) and applies ``CoordFlip``.  ``"v"`` and ``None`` are both treated
        as vertical (the default); no error is raised for other values.
    dodge : bool, default False
        When ``True`` and ``hue`` is set, apply ``Dodge`` so each hue level
        is drawn side-by-side rather than overlaid.
    jitter : bool, default True
        For ``kind="strip"``, add ``Jitter`` on the categorical axis.
        Ignored when ``dodge=True``.
    native_scale : bool, default False
        Explicitly types the categorical axis for every ``kind`` (F-L09-03).
        ``False`` (the default) types it ``"nominal"``: values render as
        equal-spaced categorical bands, even when the category column is
        numeric.  **Behavior change:** before this parameter was implemented,
        the category channel was never explicitly typed and a numeric
        column silently inferred a quantitative (linear) axis, contradicting
        this documented default; ``native_scale=False`` now forces the
        nominal reading seaborn's ``catplot`` gives by default.  ``True``
        types it ``"quantitative"``: the category column's numeric values
        place marks at their real numeric positions (preserving spacing)
        instead of equal-spaced bands, and ``Jitter``'s width for
        ``kind="strip"`` is rescaled from a band-fraction to a data-unit
        offset proportional to the column's own median unique-value
        spacing.  Requires the category column to actually be numeric --
        raises on a non-numeric column (see Raises below) -- **unless an
        explicit** ``type_=`` **is set on the channel** (``x=fm.X("cat",
        type_=...)``), **which takes precedence**: a caller's own stated
        type bypasses ``native_scale``'s resolution entirely, including
        this numeric-dtype check, so it never raises here regardless of
        the column's dtype or ``native_scale``'s value.

        **Axis padding when a position adjustment is active.** A
        quantitative axis's domain is otherwise just the data's own extent,
        with no room for ``Jitter``/``Dodge`` to move a mark into -- so
        when ``kind="strip"`` jitters (the default, ``jitter=True``) or any
        kind dodges (``dodge=True`` with ``hue=`` set), the category
        channel's domain is padded by half that position adjustment's own
        maximum offset (plus a 5% safety margin) so no mark renders outside
        it and gets silently dropped.  This padding is skipped -- your own
        domain always governs -- when you set an explicit ``scale=`` on the
        category channel yourself, e.g. ``x=fm.X("year", scale=fm.LinearScale(domain=[...]))``
        or ``encode={"x": fm.X("year", scale=...)}``: in that case a
        mark whose adjusted position falls outside YOUR domain clips
        exactly as it would on any other explicit domain, and that is your
        choice to make, not this parameter's to override.

        **The** ``encode={"x": ...}`` **door bypasses this padding
        entirely, even with no** ``scale=`` **of your own.** ``encode=``
        is a raw ``Chart.encode(**encode)`` override applied after this
        builder's own normalization, so ``encode={"x": fm.X("year")}``
        (no ``scale=``) replaces the padded, typed channel this parameter
        built wholesale -- silently losing BOTH the padding (marks can
        drop again, with no domain of your own to blame it on) and, under
        ``native_scale=False``, the force-nominal typing itself (a numeric
        column reverts to native positions). This is not specific to
        ``native_scale``: the identical ``encode=`` override already
        erases this builder's other normalizations the same way (e.g. the
        ``hue=`` nominal-color typing), since ``encode=`` is documented as
        exactly that -- an override door, not a channel this builder
        reads back from. Prefer ``x=fm.X(...)`` (or a bare ``x="year"``)
        over ``encode={"x": ...}`` when you want ``native_scale`` and its
        padding to keep working.
    ci : int or float, default 95
        Bootstrap confidence-interval width, as a percent in the open
        interval ``(0, 100)``, for the estimating kinds ``"point"`` and
        ``"bar"``.  The other six kinds draw no interval and refuse any
        value but the default with ``ValueError``.
    n_boot : int, default 1000
        Bootstrap resample count used to compute ``ci``; must be ``>= 1``.
        Refused, like ``ci``, when set to anything but the default on a
        non-estimating kind.
    seed : int or None, optional
        Random seed for reproducible output: ``Jitter``'s strip positions
        (``kind="strip"``) and the ``"point"``/``"bar"`` bootstrap.  ``None``
        means the fixed default seed, so output is byte-deterministic either
        way; pass different seeds to see the bootstrap's sampling variation.
    height : float or None, optional
        Height of the chart in pixels.  When ``row`` is set, this is treated
        as per-panel height (seaborn semantics): the total canvas height is
        ``height * n_row_panels``.  Width is derived from ``aspect``.
    aspect : float or None, optional
        Aspect ratio (width = per_panel_height * aspect).  Requires ``height``.
    theme : Theme, optional
        Visual theme applied via ``Chart.theme()``.
    mark : dict, optional
        Per-layer mark overrides.  For layered charts, keys are layer names
        (e.g. ``{"scatter": {"opacity": 0.5}}``, or ``{"whisker": ...}`` /
        ``{"point": ...}`` / ``{"bar": ...}`` for the estimating kinds); for
        single-mark charts, a flat dict of mark properties.
    encode : dict, optional
        Additional encoding kwargs merged via ``Chart.encode(**encode)``.
    properties : dict, optional
        Chart properties merged via ``Chart.properties(**properties)``
        (e.g. ``{"width": 400, "title": "My chart"}``).
    layers : list, optional
        Extra layers appended via ``Chart.layer(*layers)``.
    **encode_kwargs
        Additional keyword arguments forwarded to ``Chart.encode()``.

    Returns
    -------
    Chart
        Configured chart (possibly faceted or coord-flipped).

    Raises
    ------
    ValueError
        If ``kind`` is not one of the supported values.
    ValueError
        If ``kind="count"`` is used without specifying ``x`` (or ``y`` when
        ``orient="h"``).
    ValueError
        If ``native_scale=True`` is passed and the category column (``x``,
        or ``y`` when ``orient="h"``) is not numeric -- unless ``x``/``y``
        is an already-built channel object stating its own explicit
        ``type_=`` (``x=fm.X("cat", type_=...)``); that statement takes
        precedence over ``native_scale`` and is honored verbatim, so this
        numeric-dtype check never runs for it and this never raises for
        that call, regardless of the column's actual dtype.
    ValueError
        If ``kind`` is ``"point"`` or ``"bar"`` and ``ci`` is outside the open
        interval ``(0, 100)``, ``n_boot`` is below 1, or either the category
        or the value channel is unbound.
    ValueError
        If ``kind`` is any other kind and ``ci`` or ``n_boot`` is set to
        anything but its default: those kinds draw no interval.
    ValueError
        If ``kind`` is ``"point"`` or ``"bar"`` and a per-observation
        appearance channel is bound -- ``size``, ``shape``, ``opacity``,
        ``fill_opacity``, ``stroke_opacity``, ``stroke_width``,
        ``stroke_dash``, ``angle``, ``color``, ``fill`` or ``stroke``,
        through either a bare keyword or the ``encode=`` dict. These kinds
        draw ONE mark per (category, ``hue``) group from an aggregate, so no
        row is left for a per-observation mapping to read: the marks would
        render uniform beneath a legend asserting a mapping they do not
        carry. Use ``hue=`` (which splits the aggregate itself) or a kind
        that draws one mark per row.

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.catplot(df, x="species", y="sepal_length", kind="box")

    Group by a hue variable with dodged bars:

    >>> fm.catplot(df, x="day", y="tip", hue="sex", kind="bar", dodge=True)

    Mean with a 90 % bootstrap CI whisker per category:

    >>> fm.catplot(df, x="day", y="tip", kind="point", ci=90, n_boot=500)

    Numeric category column at its true numeric positions instead of
    equal-spaced bands:

    >>> fm.catplot(df, x="year", y="value", kind="strip", native_scale=True)

    Horizontal violin plot:

    >>> fm.catplot(df, x="total_bill", y="day", kind="violin", orient="h")
    """
    validate_choice("catplot", "kind", kind, _CATPLOT_VALID_KINDS)

    if kind in _CATPLOT_AGGREGATING_KINDS:
        ci_level, n_boot = _catplot_interval_args(ci, n_boot)
    else:
        _refuse_interval_args_on_non_aggregating_kind(kind, ci, n_boot)
        ci_level = None
    height, aspect = _checked_figure_size("catplot", height=height, aspect=aspect)
    # Read once, here, at catplot's own door (see displot's twin comment).
    hue = _grouping_hue("catplot", hue)

    return _catplot_build(
        data,
        x=x,
        y=y,
        hue=hue,
        col=col,
        row=row,
        kind=kind,
        order=order,
        hue_order=hue_order,
        orient=orient,
        dodge=dodge,
        jitter=jitter,
        native_scale=native_scale,
        ci_level=ci_level,
        n_boot=n_boot,
        seed=seed,
        height=height,
        aspect=aspect,
        mark=mark,
        encode=encode,
        properties=properties,
        layers=layers,
        theme=theme,
        encode_kwargs=encode_kwargs,
    )


def _catplot_build(
    data: Any,
    *,
    x: Any,
    y: Any,
    hue: Any,
    col: Any,
    row: Any,
    kind: str,
    order: Any,
    hue_order: Any,
    orient: Any,
    dodge: bool,
    jitter: bool,
    native_scale: bool,
    ci_level: float | None,
    n_boot: int,
    seed: int | None,
    height: float | None,
    aspect: float | None,
    mark: dict | None,
    encode: dict | None,
    properties: dict | None,
    layers: list | None,
    theme: Any,
    encode_kwargs: dict[str, Any],
) -> Chart:
    """Construct the catplot chart from validated inputs.

    The public ``catplot`` validates ``kind`` and converts ``ci`` from a
    percent to the ``ci_level`` in ``(0, 1)`` this builder passes to
    ``Summary``; ``ci_level`` is ``None`` for the six kinds that draw no
    interval. This builder owns the axis-orientation handling, mark
    dispatch, coord-flip, faceting, and sizing -- and, for ``native_scale``,
    resolving the category channel's explicit type (see
    ``_catplot_cat_type``) once for every kind.
    """
    # Determine the categorical and value axes. By default x is categorical,
    # y is value; orient="h" flips to y categorical / x value (and we add
    # CoordFlip to the chart).
    horizontal = orient == "h"
    cat_field = x if not horizontal else y
    val_field = y if not horizontal else x

    # Resolved ONCE, shared by the chart-level enc normalisation below and by
    # `_catplot_summary_layers`'s own x_channel for point/bar -- so a
    # non-numeric column under native_scale=True refuses exactly once
    # regardless of which of the 8 kinds asked, and every kind types the
    # SAME field the SAME way. `cat_explicit_type`/`cat_kwargs` cover the
    # documented `x : str or encoding` shape uniformly (cycle-2 fix, quality
    # finding 3 / spec issue 1): a bare-string `cat_field` carries neither,
    # an already-built channel object (`x=fm.X("year")`) carries whatever
    # `type_=`/`sort=`/`scale=`/... the caller stated.
    cat_name, cat_explicit_type, cat_kwargs = _catplot_cat_field_parts(cat_field)
    cat_type = _catplot_cat_type(
        data, cat_name, native_scale=native_scale, explicit_type=cat_explicit_type
    )
    # True exactly when `cat_type` came FROM `native_scale`, not from the
    # caller's own explicit `type_=` statement -- gates every native-scale
    # safety mitigation below (the Float64 cast, the jitter-width rescale,
    # the domain padding) so an explicitly-typed channel's own statement is
    # honored verbatim, untouched by any of them, the same as its `type_`
    # already is.
    native_typed = native_scale and cat_explicit_type is None and cat_type == "Q"
    # The category channel arriving through the OTHER encoding door. `encode=`
    # is a raw `Chart.encode(**encode)` override applied last by
    # `_finalize_chart`, so an `x` in it replaces whatever this builder
    # normalizes -- for every kind -- and it is read here for both questions
    # that depend on it: whether the caller stated a scale of their own
    # (`explicit_cat_scale`, immediately below) and which channel the
    # estimating kinds bind on their layers (further down). Reading it for one
    # and not the other is what made the door govern six kinds and go inert on
    # two (design re-review, cycle 2).
    #
    # Layer space, not user space: `encode`'s `x` is the CATEGORY channel under
    # both orientations, matching the `chart_encoding={**enc, **encode}` merge
    # the value-scale refusal and the scale propagation already read it in.
    encode_cat_channel = (encode or {}).get("x")

    # "The caller stated a scale on the category channel themselves", through
    # EITHER door: their own channel object (`x=fm.X("year", scale=...)`, whose
    # kwargs `cat_kwargs` copies) or the `encode=` override. One predicate for
    # both, resolved once here, so the padding gate below reads the same answer
    # the layer build does -- the two used to spell this question differently
    # (`"scale" not in x_kwargs` at the chart level,
    # `_explicit_positional_scale(...)` in the layer build) and could disagree
    # about the same call.
    explicit_cat_scale = (
        "scale" in cat_kwargs or _explicit_positional_scale(encode_cat_channel) is not None
    )

    stats: "_NativeScaleStats | None" = None
    if native_typed:
        # `native_scale=True` types the category channel "quantitative", but
        # Rust's position-adjustment pipeline uses TWO different signals for
        # "is this axis a categorical band or a continuous one", and they
        # disagree for a non-Float64 numeric column: `apply_jitter`
        # (`render/position.rs`) reads the RESOLVED SCALE KIND, so it
        # correctly takes the continuous branch for an Int64 column typed
        # quantitative here -- then crashes writing a Float64 noise array
        # back into a still-Int64-schemaed column. `apply_dodge` instead
        # reads the RAW COLUMN DTYPE (`is_ordinal_band = dtype != Float64`),
        # so the SAME Int64 column takes the ORDINAL branch and silently
        # no-ops instead of dodging. Casting the category column to Float64
        # whenever it is typed quantitative makes both signals agree --
        # which is also simply the correct representation for a column the
        # chart is declaring continuous. DELETE CONDITION: when
        # `apply_dodge`'s `is_ordinal_band` reads the resolved scale kind
        # instead of the raw dtype (matching Jitter's own check), this cast
        # becomes redundant and can be dropped.
        import polars as pl

        data = to_polars(data).with_columns(pl.col(cat_name).cast(pl.Float64))
        stats = _native_scale_stats(data, cat_name)

    # Position adjustment.
    #
    # `_field_name(hue)`, not the raw `hue`: `Dodge(by=)` takes a column name,
    # and `hue=fm.Color("g")` is a documented catplot spelling, so the raw
    # channel object reached serialization and raised an untyped `TypeError:
    # Object of type Color is not JSON serializable` -- naming neither
    # `catplot` nor `dodge=`. Same resolution the three sibling sites in this
    # file already use (`_catplot_summary_layers`'s `hue_field`, the
    # `kind="count"` groupby, `_displot_build`'s `hue_field`); this was the
    # fourth of the class.
    position = None
    if dodge and hue is not None:
        position = Dodge(by=_field_name(hue))

    # Domain padding against the silent Jitter/Dodge mark-drop (cycle-2 fix,
    # quality finding 1): `jitter_active` mirrors the exact condition the
    # `kind == "strip"` dispatch below uses for a plain (non-dodged)
    # `Jitter`; `dodge_active` covers every kind whose mark call below passes
    # `position=position` (strip-with-dodge, swarm, box, violin, boxen,
    # point, bar, count). `None` when no position adjustment will run for
    # this call, or when `stats` has nothing to pad around -- see
    # `_native_scale_padded_domain`'s own docstring for the full mechanism
    # and the Rust-side root cause (`NS-1`, code archaeology followups doc).
    jitter_active = native_typed and kind == "strip" and jitter and position is None
    dodge_active = native_typed and position is not None
    cat_domain = _native_scale_padded_domain(
        stats, jitter_active=jitter_active, dodge_active=dodge_active
    )

    # Encoding shared across all kinds.
    from ferrum.encoding import Color as _Color
    from ferrum.encoding import X as _X

    # Build the encoding with x=categorical, y=value regardless of orientation.
    # CoordFlip (added below when horizontal=True) handles the visual axis swap.
    # Stat-mark desugars (boxplot, violin, swarm, etc.) always expect x=cat, y=val,
    # so we normalise the encoding here rather than propagating the orientation
    # through every desugar function.
    enc: dict = {}
    if not horizontal:
        if x is not None:
            enc["x"] = x
        if y is not None:
            enc["y"] = y
    else:
        # orient="h": user passed x=val, y=cat — swap so x=cat, y=val for desugars.
        if y is not None:
            enc["x"] = y  # cat_field goes on x
        if x is not None:
            enc["y"] = x  # val_field goes on y
    # `hue` is typed Nominal for a bare string on EVERY kind, with no
    # per-kind branch: catplot is the categorical-plot family, its hue is a
    # group discriminator by construction (`hue_order=` orders its levels,
    # `dodge=True` offsets by them), so a Continuous reading of a bare
    # column name is never the right one -- not even on the point-mark
    # kinds. Uniformity here is the point: before this, one Int64 hue
    # answered three different ways across the eight kinds (violin
    # categorical, box a typed raise, bar/boxen/strip/swarm/point a silent
    # fabricated colorbar), and the divergence was invisible in the
    # signature.
    #
    # An already-built channel object (`hue=fm.Color(...)`) is a DIFFERENT
    # case, and `hue_order=` is what decides which rule applies (#190,
    # spec-review cycle 2 correction): with no `hue_order=`, the channel goes
    # through `nominal_color_channel`, which honors an explicit `type_=`
    # verbatim and types the channel nominal when it declares none.
    # CORRECTED (#130, commit bb5149ef, this same wave): that pass-through
    # used to be total -- an untyped `fm.Color("g")` kept whatever its
    # column's dtype inferred, including a numeric column's continuous
    # colorbar -- so the branch below was once the only site that retyped
    # anything. It no longer is, and a numeric-column channel object without
    # `hue_order=` no longer renders byte-identically to the pre-#130 output.
    # `hue_order=` still changes what happens here: ordering a fixed list of
    # named levels is a statement `sort=` cannot honor against a Continuous
    # scale at all, so passing `hue_order=` is itself a declaration that
    # this channel is categorical -- forcing nominal typing the same way a
    # bare string always does, UNLESS the caller's own channel already
    # stated an explicit `type_=`, which is honored verbatim per
    # `nominal_color_channel`'s pass-through precedent. The rebuild routes
    # through `_catplot_cat_field_parts` -- the same channel-agnostic
    # splitter `cat_channel`/`order` use below -- so a channel object's own
    # other kwargs (`scheme=`, `legend=`, ...) survive, and `sort=` is
    # overridden rather than handed whole to `Color(hue, ...)`, which raises
    # `TypeError` on a non-string `field`: the SAME kwarg-wins precedent
    # `order=` sets for `cat_channel` below (checkpoint ruling: two halves
    # of one builder share one precedence rule).
    #
    # Bound to a name because the estimating kinds need this SAME resolved
    # channel on their layers (see `_catplot_summary_layers`): one resolution,
    # both consumers.
    hue_name, hue_explicit_type, hue_kwargs = _catplot_cat_field_parts(hue)
    color_channel = None
    if hue_name is not None:
        if hue_order is not None:
            # An explicit continuous type ("Q"/"T") is honored verbatim by
            # the pass-through rule above, but `sort=` on a continuous
            # scale is not a statement that scale can act on -- there is
            # no discrete legend order for it to reorder. The two knobs
            # contradict each other, so refuse rather than silently drop
            # `hue_order=` (#190 residual: this combination used to raise
            # `TypeError` before the #190 hardening; converting it to a
            # silent accept-and-drop violated the standing refuse-or-warn
            # policy).
            if hue_explicit_type in _CATPLOT_CONTINUOUS_HUE_TYPES:
                raise ValueError(
                    f"catplot(hue=fm.Color({hue_name!r}, type_=...), hue_order=...): "
                    f"hue_order= cannot order a {_CATPLOT_TYPE_CODE_NAMES[hue_explicit_type]} "
                    "hue channel -- there is no discrete legend order for a continuous "
                    "scale to apply. Drop hue_order=, or type the channel nominal/ordinal "
                    "(e.g. type_='nominal') so hue_order= has levels to order."
                )
            color_kwargs = dict(hue_kwargs)
            if hue_explicit_type is None:
                # `"type"`, not `"type_"`: `hue_kwargs` is a copy of a channel's
                # already-normalized `_kwargs`, and `ChannelBase.__init__`
                # DISCARDS `type_` whenever a `type` key is present. A channel
                # spelled `fm.Color("g", type_=None)` carries `{"type": None}`
                # -- untyped to `option("type")`, which is what selects this
                # branch -- so writing `type_` here silently lost the nominal
                # typing on exactly the channels that most needed it. One key,
                # one spelling, written the way the reader will read it.
                color_kwargs["type"] = "nominal"
            color_kwargs["sort"] = list(hue_order)
            color_channel = _Color(hue_name, **color_kwargs)
        else:
            color_channel = nominal_color_channel(hue)
        enc["color"] = color_channel

    # The category channel is ALWAYS explicitly typed here -- the one
    # normalisation site shared by every kind (F-L09-03) AND both input
    # shapes (cycle-2 fix, quality finding 3 / spec issue 1): native_scale
    # =False (default) types "N" even for a numeric category column, a
    # disclosed behavior change (see catplot's docstring); native_scale=True
    # types "Q" (refused earlier, in `_catplot_cat_type`, when the column
    # isn't numeric). Rebuilt from `cat_kwargs` -- a COPY of whatever the
    # caller already put on their own channel object (`scale=`, `axis=`,
    # `title=`, an existing `sort=`, ...), or `{}` for a bare-string
    # `cat_field` -- so nothing the caller stated is dropped. `type_` is
    # added only when the caller stated none of their own
    # (`cat_explicit_type is None`): an explicit `type_=` on the caller's
    # own channel is honored verbatim, mirroring `nominal_color_channel`'s
    # pass-through rule for an explicit `hue=`. `order=` always sets `sort=`
    # on the rebuilt channel -- overriding any `sort=` the channel already
    # carried, the same as it already does for a bare-string `cat_field` --
    # rather than the pre-fix `TypeError` a channel-object `cat_field` used
    # to raise when combined with `order=` (a loud failure that must not
    # silently become "order dropped"; rebuilding here restores it working,
    # a strictly better outcome than restoring the raise).
    #
    # Bound to a name for the same reason `color_channel` is: the estimating
    # kinds bind this exact channel on their layers.
    cat_channel = None
    if cat_name is not None:
        x_kwargs = dict(cat_kwargs)
        if cat_explicit_type is None:
            # `"type"` for the same reason as `color_kwargs` above: this dict
            # is a copy of an already-normalized `_kwargs`, where a `type_`
            # key loses to the `type` key already in it.
            x_kwargs["type"] = cat_type
        if order is not None:
            x_kwargs["sort"] = list(order)
        # `cat_domain` (Jitter/Dodge clipping padding, quality finding 1):
        # skipped when the caller already stated a scale of their own on
        # either encoding door (`explicit_cat_scale`, resolved above) -- an
        # explicit caller scale/domain must still govern, never silently
        # widened out from under it.
        if cat_domain is not None and not explicit_cat_scale:
            x_kwargs["scale"] = LinearScale(domain=list(cat_domain))
        cat_channel = _X(cat_name, **x_kwargs)
        enc["x"] = cat_channel

    # ...and the `encode=` door wins over it, for the LAYERS as it already does
    # for the chart level. `_finalize_chart` applies `encode` through
    # `Chart.encode`, which writes the chart-level encoding only -- invisible
    # to a layered chart, whose axis resolves from layer 0. So the estimating
    # kinds have to take the override here, from the same read
    # `explicit_cat_scale` performs, or `encode={"x": fm.X("cat", sort=[...])}`
    # would order the axis on the six per-row kinds and do nothing on these two
    # (measured before this: `c, b, a` on box, `a, b, c` on point).
    #
    # It is a REPLACEMENT, not a merge, because that is what the door means
    # everywhere else: `encode=` is documented as a raw override that discards
    # this builder's normalizations (the typing, the padding), and honoring it
    # here the same way is what makes the two kind families agree. Prefer
    # `x=fm.X(...)`, which this builder normalizes on top of -- see catplot's
    # `native_scale` docs.
    if encode_cat_channel is not None:
        cat_channel = encode_cat_channel

    enc.update(encode_kwargs)

    chart = Chart(data)

    if kind == "strip":
        if position is not None:
            # dodge=True with hue overrides jitter (per spec — single-position
            # adjustments aren't composable in Phase 9c).
            chart = chart.mark_point(position=position)
        elif jitter:
            # After the enc normalisation above, cat is always on x, so jitter
            # always acts on the x axis. On a native (quantitative) scale
            # that axis is linear, not a band, so the ordinal 0.4
            # band-fraction default is the wrong magnitude -- rescale it by
            # the category column's own median unique-value spacing (see
            # `_native_scale_jitter_width`). `jitter_active` (resolved above,
            # alongside `cat_domain`) is exactly this branch's own condition
            # (`kind == "strip"`, `jitter`, `position is None`) ANDed with
            # `native_typed`.
            jitter_width = _native_scale_jitter_width(stats) if jitter_active else 0.4
            chart = chart.mark_point(
                position=Jitter(axis="x", width=jitter_width, seed=seed),
            )
        else:
            chart = chart.mark_point(position=Identity())
    elif kind == "swarm":
        if position is not None:
            chart = chart.mark_swarm(position=position)
        else:
            chart = chart.mark_swarm()
    elif kind == "box":
        if position is not None:
            chart = chart.mark_boxplot(position=position)
        else:
            chart = chart.mark_boxplot()
    elif kind == "violin":
        if position is not None:
            chart = chart.mark_violin(position=position)
        else:
            chart = chart.mark_violin()
    elif kind == "boxen":
        if position is not None:
            chart = chart.mark_boxen(position=position)
        else:
            chart = chart.mark_boxen()
    elif kind in _CATPLOT_AGGREGATING_KINDS:
        # One estimate per (category, hue) group, not one mark per row: a
        # named `Summary` computes mean + bootstrap CI in Rust and the two
        # emitted layers read it through `data_source=` (F-L09-02/03).
        # `cat_channel`/`color_channel` were already resolved above, shared
        # verbatim with the chart-level enc normalisation.
        val_name = _field_name(val_field)
        if cat_name is None or val_name is None:
            raise ValueError(
                f"catplot(kind={kind!r}) summarises a value column per category, so it "
                "requires both a category and a value channel: x= (category) and y= "
                "(value), or y= (category) and x= (value) when orient='h'."
            )
        if ci_level is None:
            # A real raise, not an `assert`: `python -O` strips asserts, and
            # this invariant (the public `catplot` validates ci/n_boot and
            # converts the percent for exactly these kinds) must not degrade
            # into `Summary(ci=None)` surfacing as a transform-level error
            # under optimized runs.
            raise ValueError(
                f"catplot(kind={kind!r}) requires a validated CI level; it is computed "
                "by `catplot` via `_catplot_interval_args` for exactly the aggregating "
                "kinds, so reaching here means the builder was called directly with "
                "ci_level=None."
            )
        # Refused before anything is built: a per-observation appearance
        # channel cannot reach an aggregated mark, and the chart-level
        # encoding it would land on still draws its legend.  Both doors are
        # checked -- the bare keyword (`**encode_kwargs`) and the `encode=`
        # dict -- because both reach `chart.encode` for these kinds, read
        # through `_drawn_bindings` the way `chart.encode` will apply them.
        _reject_per_row_appearance_channels(kind, _drawn_bindings(encode_kwargs, encode))
        summary, summary_layers = _catplot_summary_layers(
            kind=kind,
            cat_channel=cat_channel,
            val_field=val_name,
            color_channel=color_channel,
            position=position,
            # `enc` is already normalized to cat-on-x / value-on-y (and carries
            # the caller's `encode_kwargs`), so it is the layer-space encoding
            # the propagation helper expects for BOTH orientations.
            #
            # `encode=` is the second door onto the same channels and has to be
            # merged in HERE, not left to `_finalize_chart`: that call applies
            # it through `Chart.encode`, which writes only the CHART-level
            # encoding -- and on a layered chart the renderer resolves its axis
            # from layer 0's encoding, which overlays the chart level away. A
            # scale arriving through `encode=` would therefore be silently
            # dropped and would slip past the refusal below. `_drawn_bindings`
            # applies `encode` over `enc` the same way `_finalize_chart`
            # applies the two: a value wins, and a `None` leaves `enc`'s own
            # binding, so a scale on `y=`/`x=` through `encode=` reaches the
            # chart-level encoding the refusal below reads.
            chart_encoding=_drawn_bindings(enc, encode),
            ci_level=ci_level,
            n_boot=n_boot,
            seed=seed,
        )
        # The summary is the figure's statistic, after the caller's steps. A
        # fresh Chart comes back, so assigning the layered state onto it
        # mutates nothing the caller holds (the idiom
        # `_fit_chart_from_named_transform` uses in plots/regression.py).
        chart = chart._with_statistics(summary)
        chart._mark = None
        chart._layers = summary_layers
    elif kind == "count":
        # One bar per category (and hue level) at its row count, as an
        # encoding statistic: `y=count()`, as Vega-Lite's count plot is. The
        # statistic groups by the drawn category, the hue and the facet like
        # every encoding statistic, runs after the caller's `.transform(...)`
        # steps, and a layer attached through `layers=` inherits it and
        # counts its own rows (`Layer(encoding={"text": "count()"})` labels
        # each bar with its height). The value axis keeps the title "n".
        if cat_name is None:
            raise ValueError("catplot(kind='count') requires x= (or y= when orient='h')")
        chart = chart.mark_bar(position=position) if position is not None else chart.mark_bar()
        # After enc normalisation the value is always on y (cat on x).
        from ferrum.encoding import Y as _Y_count

        enc["y"] = _Y_count(aggregate="count", title="n")
    chart = chart.encode(**enc)

    # orient="h" → CoordFlip.
    if horizontal:
        chart = chart.coord(CoordFlip())

    # Faceting.
    if col is not None or row is not None:
        if col is not None and row is not None:
            chart = chart.facet(row=row, col=col)
        elif col is not None:
            chart = chart.facet(col=col)
        else:
            chart = chart.facet(row=row)

    # Properties: delegate to shared helper so catplot and displot cannot drift.
    chart = _apply_facet_sizing(chart, data=data, row=row, height=height, aspect=aspect)

    chart = _finalize_chart(
        chart,
        entry="catplot",
        mark=mark,
        encode=encode,
        properties=properties,
        layers=layers,
        theme=theme,
    )
    if kind == "count" and layers:
        _refuse_count_column_reads(chart, len(layers))
    return chart


def _refuse_count_column_reads(chart: Chart, attached: int) -> None:
    """Refuse a layer of ``catplot(kind="count")`` binding ``n``, a column its rows lack.

    The count plot draws each bar's height as ``y=count()``, an encoding
    statistic, so no column named ``n`` exists; a layer attached through
    ``layers=`` (the last *attached* layers of the finished *chart*, one per
    argument) labels each bar with its count through its own statistic,
    ``text="count()"``. A binding of ``n`` is refused with that remedy only
    on a channel that takes ``aggregate=``, and only where Python knows the
    rows the layer reads (:func:`ferrum._compose.known_columns`) and they
    lack ``n``. A column ``n`` those rows hold is drawn as written; any
    other binding is left to the render's own missing-column refusal.
    """
    from ferrum._compose import known_columns
    from ferrum._layer_transforms import _own_channels
    from ferrum._spec_build import _encoding_field_and_type
    from ferrum.encoding._resolve import channel_class
    from ferrum.encoding.base import ChannelBase

    for layer in (chart._layers or [])[-attached:]:
        for channel in _own_channels(layer):
            value = layer.encoding[channel]
            if not isinstance(value, (str, ChannelBase)):
                continue
            if _encoding_field_and_type(value)[0] != "n":
                continue
            cls = channel_class(channel)
            if "aggregate" not in cls._honored_kwargs:
                continue
            columns = known_columns(layer, chart)
            if columns is None or "n" in columns:
                continue
            raise ValueError(
                f"catplot(kind='count'): layers= binds {channel}='n', but the count plot "
                "draws each bar's count as y=count(), so there is no column 'n'; label "
                f"each bar with {channel}='count()' (or fm.{cls.__name__}(aggregate='count')), "
                "which counts the rows the layer reads as each bar does"
            )


# ---------------------------------------------------------------------------
# relplot
# ---------------------------------------------------------------------------


@_figure_entry
def relplot(
    data: Any,
    *,
    x: Any,
    y: Any,
    hue: Any = None,
    size: Any = None,
    style: Any = None,
    col: Any = None,
    row: Any = None,
    kind: str = "scatter",
    height: float | None = None,
    aspect: float | None = None,
    mark: dict | None = None,
    encode: dict | None = None,
    properties: dict | None = None,
    layers: list | None = None,
    theme: Any = None,
    **encode_kwargs: Any,
) -> Chart:
    """Relational figure-level function for scatter and line plots.

    Dispatches to the appropriate mark based on ``kind``:

    * ``"scatter"`` -- ``mark_point()``.  ``style`` maps to ``Shape``.
    * ``"line"``    -- ``mark_line()``.  ``style`` maps to ``StrokeDash``.

    Parameters
    ----------
    data : DataFrame-like
        Input data accepted by ``Chart(data)``.
    x : str or encoding
        Column name for the horizontal axis (required).
    y : str or encoding
        Column name for the vertical axis (required).
    hue : str or encoding, optional
        Column name to map to color (one group per level).
    size : str or encoding, optional
        Column name to map to the ``Size`` channel.  With
        ``kind="scatter"`` it sets each point's area; with ``kind="line"``
        each line tapers along its length, its width at every vertex the
        diameter of a point of that row's area, ``2 * sqrt(area / pi)``
        pixels.
    style : str or encoding, optional
        Column name to map to ``Shape`` (scatter) or ``StrokeDash`` (line).
    col : str, optional
        Column name for faceting across columns.
    row : str, optional
        Column name for faceting across rows.
    kind : {"scatter", "line"}, default "scatter"
        Which relational mark to draw.
    height : float or None, optional
        Height of the chart in pixels.  Width is derived from ``aspect``.
    aspect : float or None, optional
        Aspect ratio (width = height * aspect).  Requires ``height``.
    theme : Theme, optional
        Visual theme applied via ``Chart.theme()``.
    mark : dict, optional
        Per-layer mark overrides.  For composite-mark charts, keys are
        layer names (e.g. ``{"scatter": {"opacity": 0.5}}``); for
        single-mark charts, a flat dict of mark properties.
    encode : dict, optional
        Additional encoding kwargs merged via ``Chart.encode(**encode)``.
    properties : dict, optional
        Chart properties merged via ``Chart.properties(**properties)``
        (e.g. ``{"width": 400, "title": "My chart"}``).
    layers : list, optional
        Extra layers appended via ``Chart.layer(*layers)``.
    **encode_kwargs
        Additional keyword arguments forwarded to ``Chart.encode()``.

    Returns
    -------
    Chart
        Configured chart (possibly faceted or sized).

    Raises
    ------
    ValueError
        If ``kind`` is not one of ``"scatter"`` or ``"line"``.

    Examples
    --------
    Scatter with hue grouping and column faceting:

    >>> import ferrum as fm
    >>> fm.relplot(df, x="total_bill", y="tip", hue="sex", col="time")

    Line plot with per-level dash styling:

    >>> fm.relplot(df, x="timepoint", y="signal", hue="region",
    ...            style="region", kind="line")
    """
    validate_choice("relplot", "kind", kind, _RELPLOT_VALID_KINDS)
    height, aspect = _checked_figure_size("relplot", height=height, aspect=aspect)

    return _relplot_build(
        data,
        x=x,
        y=y,
        hue=hue,
        size=size,
        style=style,
        col=col,
        row=row,
        kind=kind,
        height=height,
        aspect=aspect,
        mark=mark,
        encode=encode,
        properties=properties,
        layers=layers,
        theme=theme,
        encode_kwargs=encode_kwargs,
    )


def _relplot_build(
    data: Any,
    *,
    x: Any,
    y: Any,
    hue: Any,
    size: Any,
    style: Any,
    col: Any,
    row: Any,
    kind: str,
    height: float | None,
    aspect: float | None,
    mark: dict | None,
    encode: dict | None,
    properties: dict | None,
    layers: list | None,
    theme: Any,
    encode_kwargs: dict[str, Any],
) -> Chart:
    """Construct the relplot chart from validated inputs.

    The public ``relplot`` validates ``kind``; this builder owns the encoding
    assembly, mark dispatch, faceting, and sizing.
    """
    chart = Chart(data)

    enc: dict = {}
    if x is not None:
        enc["x"] = x
    if y is not None:
        enc["y"] = y
    if hue is not None:
        # The one surviving point-mark carve-out in plots/: kind="scatter" is
        # a single unpaired `point` mark (no sibling layer, no shared color
        # resolve), so a numeric hue may legitimately read as a gradient --
        # seaborn does the same, and this keeps hue the plain `encode()`
        # binding it already is, refusals included. kind="line" groups rows
        # into one polyline per level, so an inferred Continuous scale
        # collapses every group into a single line; that one is read at
        # relplot's own door (`_grouping_hue`, a shorthand type kept, an
        # aggregate refused) and typed Nominal, same as every other figure's
        # grouping hue. `_grouping_hue`'s own result is already typed, so
        # the `nominal_color_channel` wrap below is a no-op here -- kept
        # for the same reason every other hue-binding site in this module
        # routes through it: one declared-typed call the AST completeness
        # guard (`tests/test_color_binding_completeness.py`) can see.
        enc["color"] = (
            hue if kind == "scatter" else nominal_color_channel(_grouping_hue("relplot", hue))
        )
    if size is not None:
        enc["size"] = size
    if style is not None:
        if kind == "scatter":
            enc["shape"] = style
        else:
            enc["stroke_dash"] = style
    enc.update(encode_kwargs)

    if kind == "scatter":
        chart = chart.mark_point()
    else:
        chart = chart.mark_line()

    chart = chart.encode(**enc)

    if col is not None or row is not None:
        if col is not None and row is not None:
            chart = chart.facet(row=row, col=col)
        elif col is not None:
            chart = chart.facet(col=col)
        else:
            chart = chart.facet(row=row)

    if height is not None or aspect is not None:
        h = height if height is not None else 300.0
        w = h * aspect if aspect is not None else h
        chart = chart.properties(width=w, height=h)

    return _finalize_chart(
        chart,
        entry="relplot",
        mark=mark,
        encode=encode,
        properties=properties,
        layers=layers,
        theme=theme,
    )
