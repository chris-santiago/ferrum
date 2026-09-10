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
from typing import Any

from ferrum import (
    Aggregate,
    AggregateOp,
    Bin,
    Chart,
    CoordFlip,
    Dodge,
    Identity,
    Jitter,
    Stack,
)
from ferrum._coerce import to_polars
from ferrum._desugar import (
    _explicit_positional_scale,
    _propagate_layer_scale,
    _scale_domain,
)
from ferrum._layer import _Layer
from ferrum._registry import registry_output_columns
from ferrum._validate import validate_choice
from ferrum.marks._desugar_helpers import nominal_color_channel, resolve_color_groupby
from ferrum.plots._helpers import _field_name, _finalize_chart


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DISPLOT_VALID_KINDS = {"hist", "kde", "ecdf", "rug"}
_DISPLOT_VALID_MULTIPLE = {"layer", "stack", "fill", "dodge"}
_DISPLOT_VALID_STAT = {"count", "density"}

_CATPLOT_VALID_KINDS = {"strip", "swarm", "box", "violin", "boxen", "point", "bar", "count"}
_RELPLOT_VALID_KINDS = {"scatter", "line"}

# The two catplot kinds that draw an ESTIMATE (one mark per category group)
# rather than one mark per row, and therefore consume ``ci=``/``n_boot=``.
_CATPLOT_AGGREGATING_KINDS = frozenset({"point", "bar"})

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


def _multiple_to_position(multiple: str, hue: Any):
    if multiple == "layer":
        return Identity()
    if multiple == "dodge":
        return Dodge(by=hue)
    if multiple == "stack":
        return Stack(by=hue, offset="zero")
    if multiple == "fill":
        return Stack(by=hue, offset="normalize")
    raise ValueError(f"unknown multiple {multiple!r}")


def _histogram_bin_count(bins: Any) -> int | None:
    """Return the value `mark_histogram`'s `Bin` transform actually receives for `bin_count`.

    An int fixes the count; any other `bins` value (the `"sturges"` default,
    or any other rule string such as `"fd"`) becomes `None`, because 1-D
    `Bin`'s `bin_count` parameter is `Optional[int]` only -- it has no
    string-rule vocabulary, and `None` always falls back to `Bin`'s own
    Sturges default.
    """
    return bins if isinstance(bins, int) else None


def _histogram_count_scale(bins: Any) -> int | str:
    """Return the `Kde(count_scale=...)` value that bit-matches what `Bin` actually used.

    `Bin` cannot honor a non-int `bins` rule (see `_histogram_bin_count`), so
    matching the *real* bars means requesting `"sturges"` here too rather
    than the raw `bins` string -- passing e.g. `"fd"` through would compute
    the overlay's bin width from a rule the bars themselves never applied.
    """
    bin_count = _histogram_bin_count(bins)
    return bin_count if bin_count is not None else "sturges"


def _kde_overlay(
    x_field: str,
    *,
    hue: Any,
    kind: str,
    stat: str,
    multiple: str,
    bins: Any,
    bandwidth: Any,
    bw_adjust: float,
    cumulative: bool,
):
    """Build the named `Kde` transform and `_Layer` for displot's `kde=True` overlay.

    Returns `(transform, layer)`: the caller prepends the transform (so it
    reads the original, pre-`Bin` data batch) and appends the layer.

    `groupby=hue` gives one curve per hue level instead of one curve pooled
    across all rows; `shared_extent=False` mirrors what the real histogram's
    `Bin` transform actually receives on this call path (`_displot_build`
    never forwards `multiple=` into `mark_histogram`, so `Bin` always runs
    ungrouped-extent regardless of displot's `multiple=`) -- this must track
    that value, not re-derive `multiple in ("stack", "fill")`, or a grouped
    count-scaled bin width diverges from the bars' own
    (`test_wire_shared_extent_matches_the_real_bin_transform` guards this).
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

    kde_kwargs: dict = dict(
        bandwidth=bandwidth,
        bw_adjust=bw_adjust,
        cumulative=cumulative,
        shared_extent=False,
        name="kde_overlay",
    )
    if hue is not None:
        kde_kwargs["groupby"] = hue
    # count_scale may only be requested when the histogram's own value axis
    # is actually a count axis -- a future axis-changing mode must extend
    # this predicate's reading, not add a fourth ad hoc clause here.
    histogram_value_axis_is_count = kind == "hist" and stat == "count" and multiple != "fill"
    if histogram_value_axis_is_count:
        kde_kwargs["count_scale"] = _histogram_count_scale(bins)
    transform = _KdeTransform(x_field, **kde_kwargs)

    # Derived from the registry (spec §4.6 / M6a) rather than hardcoded, so a
    # Rust-side rename of either Kde output column raises here instead of
    # silently drawing an empty overlay line.
    _kde_cols = registry_output_columns("kde", ["value", "density"])
    encoding: dict = {"x": _kde_cols["value"], "y": _kde_cols["density"]}
    if hue is not None:
        encoding["color"] = nominal_color_channel(hue)

    from ferrum._layer import _Layer as _OverlayLayer

    layer = _OverlayLayer(
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
        props: dict = {"height": total_h}
        if w is not None:
            props["width"] = w
        return chart.properties(**props)
    elif height is not None or aspect is not None:
        h = height if height is not None else 300.0
        w = h * aspect if aspect is not None else h
        return chart.properties(width=w, height=h)
    return chart


# ---------------------------------------------------------------------------
# displot
# ---------------------------------------------------------------------------


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
    bins: Any = "sturges",
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
    y : str or encoding, optional
        Column name for the distribution variable (vertical axis).
    hue : str or encoding, optional
        Column name to map to color (one distribution per level).
    col : str, optional
        Column name for faceting across columns.
    row : str, optional
        Column name for faceting across rows.
    kind : {"hist", "kde", "ecdf", "rug"}, default "hist"
        Which distribution mark to draw.  ``"hist"`` calls
        ``mark_histogram``; ``"kde"`` calls ``mark_density`` (filled by
        default); ``"ecdf"`` builds a cumulative frequency line via ``Bin``
        + ``mark_line``; ``"rug"`` calls ``mark_tick``.
    fill : bool, default True
        Fill the area under the KDE curve (``kind="kde"`` only).
    cumulative : bool, default False
        Produce a cumulative histogram or density (``kind="hist"`` and
        ``kind="kde"``).  Also drives the ``kde=True`` overlay curve (below),
        so a cumulative histogram's overlay is a matching CDF, not a PDF.
    log_scale : bool, default False
        Apply a ``log`` scale to the ``x`` axis.
    stat : {"count", "density"}, default "count"
        Statistic to plot on the value axis for ``kind="hist"``.
        ``"density"`` normalises so the total area integrates to 1.
        Validated for every ``kind`` (not just ``"hist"``): an unsupported
        value raises regardless of ``kind``.
    bins : int or str, default "sturges"
        Binning rule for ``kind="hist"``.  An integer is forwarded as
        ``bin_count``.  A string is accepted but, today, only changes
        behavior for the exact value ``"sturges"`` (the default): any other
        string (e.g. ``"fd"``) is silently treated the same as ``"sturges"``,
        since the underlying ``Bin`` transform has no string-rule vocabulary
        and always falls back to Sturges' rule when no integer count is
        given.
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
        If ``kind="ecdf"`` is used without specifying ``x=``.

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

    The public ``displot`` validates ``kind``/``multiple``; this builder owns the
    mark/transform dispatch, optional kde/rug overlays, faceting, and sizing.
    """
    # Position adjustment from `multiple`.
    position = _multiple_to_position(multiple, hue)

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

    # Mark + transforms by kind.
    if kind == "hist":
        # When `multiple` requires per-group binning (stack/fill/dodge) and a
        # hue is bound, thread `groupby=hue` so the Bin transform emits per-
        # (bin, group) rows preserving the hue column for color encoding +
        # position adjustment.
        hist_kwargs: dict = dict(
            bin_count=_histogram_bin_count(bins),
            cumulative=cumulative,
            density=(stat == "density"),
            position=position,
        )
        if hue is not None and multiple in ("stack", "fill", "dodge"):
            hist_kwargs["groupby"] = hue
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
            kde_kwargs["groupby"] = hue
        chart = chart.mark_density(**kde_kwargs)
    elif kind == "ecdf":
        # ECDF: cumulative bin → step line.
        if x is None:
            raise ValueError("displot(kind='ecdf') requires x=")
        from ferrum.encoding import X as _X_ecdf
        from ferrum.encoding.base import ChannelBase as _CB_ecdf

        # Extract the original field name for the axis title.
        _ecdf_x_name = x.field if isinstance(x, _CB_ecdf) else str(x)
        chart = chart.transform(
            Bin(field=_ecdf_x_name, bin_count=_histogram_bin_count(bins), cumulative=True)
        )
        chart = chart.mark_line()
        # Re-route encoding to bin output columns, preserving the original
        # variable name as the x-axis title.
        # Derived from the registry (spec §4.6 / M6a) rather than hardcoded:
        # these read the cumulative Bin transform chained two lines above, so
        # a Rust-side rename of either bin output column raises here instead
        # of silently rendering an empty step line.
        _bin_cols = registry_output_columns("bin", ["bin_start", "count"])
        enc["x"] = _X_ecdf(_bin_cols["bin_start"], title=_ecdf_x_name)
        enc["y"] = _bin_cols["count"]
    elif kind == "rug":
        chart = chart.mark_tick()

    chart = chart.encode(**enc)

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
    # (the original data, since they run before the unnamed Bin) without
    # advancing it.  The overlay _Layer objects set data_source to the named
    # transform key so they read the correct output.
    if (kde and kind != "kde") or (rug and kind != "rug"):
        from ferrum._layer import _Layer as _OverlayLayer
        from ferrum.encoding.base import ChannelBase as _CB_overlay

        chart = chart._resolve_pending()
        chart = chart._clone()
        x_field = x.field if isinstance(x, _CB_overlay) else str(x) if x is not None else None

        # Convert single-mark chart to layered mode: promote the existing
        # mark + encoding into a primary layer so overlay layers can be
        # appended alongside it.
        if chart._layers is None and chart._mark is not None:
            primary_layer = _OverlayLayer(
                mark=chart._mark,
                encoding=dict(chart._encoding),
                mark_kwargs=dict(chart._mark_kwargs) if chart._mark_kwargs else None,
                position=chart._position,
            )
            chart._layers = [primary_layer]
            chart._mark = None

        if kde and kind != "kde" and x_field is not None:
            kde_transform, kde_layer = _kde_overlay(
                x_field,
                hue=hue,
                kind=kind,
                stat=stat,
                multiple=multiple,
                bins=bins,
                bandwidth=bandwidth,
                bw_adjust=bw_adjust,
                cumulative=cumulative,
            )
            # Prepend the named Kde before the unnamed Bin so it reads from
            # the original (pre-Bin) data batch.
            chart._transforms = [kde_transform] + list(chart._transforms or [])
            chart._layers.append(kde_layer)

        if rug and kind != "rug" and x_field is not None:
            from ferrum.marks._desugar_helpers import identity_transform

            rug_xform = identity_transform("rug_data")
            # Prepend the named Identity before the unnamed Bin so it
            # captures the original data for the rug tick marks.
            chart._transforms = [rug_xform] + list(chart._transforms or [])
            chart._layers.append(
                _OverlayLayer(
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

    # log_scale on x.
    if log_scale and x is not None:
        from ferrum.encoding import X

        chart = chart.encode(x=X(x, scale={"type": "log"}))

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
        chart, mark=mark, encode=encode, properties=properties, layers=layers, theme=theme
    )


# ---------------------------------------------------------------------------
# catplot
# ---------------------------------------------------------------------------


def _catplot_ci_level(ci: Any, n_boot: Any) -> float:
    """Validate ``ci``/``n_boot`` and return the CI as a level in ``(0, 1)``.

    ``ci`` is a PERCENT on catplot's public signature (``ci=95``) but a LEVEL
    on ``Summary`` (``ci=0.95``); this is the one site that converts between
    them, so the two spellings cannot drift. Both bounds are exclusive: a
    0 % interval has no width and a 100 % bootstrap interval is not
    estimable from a finite resample.

    Called only for the kinds that actually consume these arguments
    (:data:`_CATPLOT_AGGREGATING_KINDS`) -- on the other six kinds ``ci`` and
    ``n_boot`` are inert, and refusing a value there would reject calls that
    have always been accepted.

    ``bool`` is refused for both, explicitly: it subclasses ``int`` and
    converts to a float, so ``n_boot=True`` would otherwise mean "1 resample"
    and ``ci=True`` "a 1 % interval" -- silently answering a call no one
    intended rather than naming the accepted range. ``ci=1`` and ``n_boot=1``
    stay accepted; only the boolean spelling is rejected.
    """
    if isinstance(ci, bool):
        raise ValueError(
            f"catplot: ci= must be a number in the open interval (0, 100); got {ci!r}."
        )
    try:
        ci_percent = float(ci)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"catplot: ci= must be a number in the open interval (0, 100); got {ci!r}."
        ) from exc
    if not 0.0 < ci_percent < 100.0:
        raise ValueError(f"catplot: ci= must be in the open interval (0, 100); got {ci!r}.")
    if not isinstance(n_boot, int) or isinstance(n_boot, bool) or n_boot < 1:
        raise ValueError(f"catplot: n_boot= must be an integer >= 1; got {n_boot!r}.")
    return ci_percent / 100.0


def _catplot_summary_layers(
    *,
    kind: str,
    cat_field: str,
    val_field: str,
    hue_field: str | None,
    order: Any,
    hue_order: Any,
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

    ``position`` is set EXPLICITLY on both layers rather than inherited from
    the chart: whisker and marker must carry the SAME adjustment or they
    separate (a dodged marker beside an undodged whisker), and an inherited
    chart-level adjustment is how the displot overlays previously acquired a
    ``Stack`` they had no business honoring (see ``_kde_overlay``).

    The value axis is titled with the original value column on BOTH layers so
    the axis reads ``"tip"`` rather than the transform's output column name --
    whichever layer is first (it differs by kind) owns the axis title.

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

    A scale the CALLER set is a different matter and is honored:
    ``chart_encoding``'s positional channels are propagated onto both layers
    through the shared ``_propagate_layer_scale`` (the same mechanism the
    composite-mark desugars and ``lmplot``'s fit layers use), so
    ``y=fm.Y("tip", scale={"domain": [0, 120]})`` governs the axis exactly as
    it does for ``kind="box"``. The one combination refused rather than
    honored is a caller scale on the VALUE channel that carries no ``domain``
    (e.g. a bare ``{"type": "log"}``) -- see ``_reject_domainless_value_scale``.
    """
    from ferrum import Summary
    from ferrum.encoding import X as _X
    from ferrum.encoding import Y as _Y

    _reject_domainless_value_scale(kind, chart_encoding)

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

    # `order=`/`hue_order=` act through the layers' own channels, not the
    # chart-level encoding: a layered chart resolves its axis and color scales
    # from the layer encodings, so a sort left only at the chart level is
    # inert here (the same reason `desugar_errorbar` threads `x_sort`).
    x_channel = _X(cat_field, sort=list(order)) if order is not None else cat_field
    color_sort = list(hue_order) if hue_order is not None else None
    layer_position = position if position is not None else Identity()

    def _encoding(y_col: str, y2_col: str | None = None) -> dict:
        enc: dict = {"x": x_channel, "y": _Y(y_col, title=val_field)}
        if y2_col is not None:
            enc["y2"] = y2_col
        if hue_field is not None:
            # Built at the binding site (as `desugar_errorbar._enc` does) so
            # every color channel in the tree is visibly typed Nominal --
            # `tests/test_color_binding_completeness.py` reads these bindings
            # syntactically and a hoisted variable reads as undeclared.
            enc["color"] = nominal_color_channel(hue_field, sort=color_sort)
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
    # These layers are rebuilt from bare output-column names, which is exactly
    # the shape `_propagate_layer_scale` exists to repair: without it a
    # caller's explicit positional `scale=` is silently dropped, as it was for
    # every composite desugar before GH #45. Returns each layer unchanged when
    # no positional channel carries an explicit scale, so the ordinary call is
    # a no-op rather than a branch.
    return transform, [_propagate_layer_scale(chart_encoding, layer) for layer in layers]


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
    ci: Any = 95,
    n_boot: int = 1000,
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
    * ``"count"``  -- ``Aggregate(count)`` + ``mark_bar``.

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
        When ``True``, treat the categorical axis as quantitative instead
        of ordinal (preserves numeric spacing rather than equal-spacing
        categories).  Currently raises ``ValueError`` because the renderer
        does not support quantitative categorical axes.
    ci : int or float, default 95
        Bootstrap confidence-interval width, as a percent in the open
        interval ``(0, 100)``, for the estimating kinds ``"point"`` and
        ``"bar"``.  Inert (and unvalidated) for the other six kinds, which
        draw no interval.
    n_boot : int, default 1000
        Bootstrap resample count used to compute ``ci``; must be ``>= 1``.
        Inert for the non-estimating kinds, as ``ci`` is.
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
        If ``native_scale=True`` is passed (not yet supported by the renderer).
    ValueError
        If ``kind`` is ``"point"`` or ``"bar"`` and ``ci`` is outside the open
        interval ``(0, 100)``, ``n_boot`` is below 1, or either the category
        or the value channel is unbound.

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.catplot(df, x="species", y="sepal_length", kind="box")

    Group by a hue variable with dodged bars:

    >>> fm.catplot(df, x="day", y="tip", hue="sex", kind="bar", dodge=True)

    Mean with a 90 % bootstrap CI whisker per category:

    >>> fm.catplot(df, x="day", y="tip", kind="point", ci=90, n_boot=500)

    Horizontal violin plot:

    >>> fm.catplot(df, x="total_bill", y="day", kind="violin", orient="h")
    """
    validate_choice("catplot", "kind", kind, _CATPLOT_VALID_KINDS)

    if native_scale:
        raise ValueError(
            "catplot: native_scale=True is not supported; the renderer does not "
            "support quantitative categorical axes"
        )

    ci_level = _catplot_ci_level(ci, n_boot) if kind in _CATPLOT_AGGREGATING_KINDS else None

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

    The public ``catplot`` validates ``kind``, rejects unsupported
    ``native_scale``, and converts ``ci`` from a percent to the ``ci_level``
    in ``(0, 1)`` this builder passes to ``Summary``; ``ci_level`` is ``None``
    for the six kinds that draw no interval. This builder owns the
    axis-orientation handling, mark dispatch, coord-flip, faceting, and sizing.
    """
    # Determine the categorical and value axes. By default x is categorical,
    # y is value; orient="h" flips to y categorical / x value (and we add
    # CoordFlip to the chart).
    horizontal = orient == "h"
    cat_field = x if not horizontal else y
    val_field = y if not horizontal else x

    # Position adjustment.
    position = None
    if dodge and hue is not None:
        position = Dodge(by=hue)

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
    # `hue` is typed Nominal for EVERY kind, with no per-kind branch. catplot
    # is the categorical-plot family: its hue is a group discriminator by
    # construction (`hue_order=` orders its levels, `dodge=True` offsets by
    # them), so a Continuous reading is never the right one -- not even on the
    # point-mark kinds. Uniformity here is the point: before this, one Int64
    # hue answered three different ways across the eight kinds (violin
    # categorical, box a typed raise, bar/boxen/strip/swarm/point a silent
    # fabricated colorbar), and the divergence was invisible in the signature.
    if hue is not None:
        enc["color"] = nominal_color_channel(hue)

    # Wire order → sort on the categorical axis encoding.
    # After the swap above, cat is always on x, so cat_channel is always "x".
    if order is not None and cat_field is not None:
        enc["x"] = _X(cat_field, sort=list(order))

    # Wire hue_order → sort on the color encoding. Typed Nominal for the same
    # reason as above, and a fortiori: an explicit hue_order= is a statement
    # that the levels are categories.
    if hue_order is not None and hue is not None:
        enc["color"] = _Color(hue, sort=list(hue_order), type_="nominal")

    enc.update(encode_kwargs)

    chart = Chart(data)

    if kind == "strip":
        if position is not None:
            # dodge=True with hue overrides jitter (per spec — single-position
            # adjustments aren't composable in Phase 9c).
            chart = chart.mark_point(position=position)
        elif jitter:
            # After the enc normalisation above, cat is always on x, so jitter
            # always acts on the x axis (the categorical band axis).
            chart = chart.mark_point(
                position=Jitter(axis="x", width=0.4, seed=seed),
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
        cat_name = _field_name(cat_field)
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
                "by `catplot` via `_catplot_ci_level` for exactly the aggregating "
                "kinds, so reaching here means the builder was called directly with "
                "ci_level=None."
            )
        summary, summary_layers = _catplot_summary_layers(
            kind=kind,
            cat_field=cat_name,
            val_field=val_name,
            hue_field=_field_name(hue),
            order=order,
            hue_order=hue_order,
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
            # dropped AND would slip past the refusal below, the exact defect
            # the `y=`/`x=` door was fixed for. `encode` wins over `enc` because
            # that is the order `_finalize_chart` applies the two in.
            chart_encoding={**enc, **(encode or {})},
            ci_level=ci_level,
            n_boot=n_boot,
            seed=seed,
        )
        # `.transform()` returns a fresh Chart, so assigning the layered state
        # onto it mutates nothing the caller holds (the idiom
        # `_fit_chart_from_named_transform` uses in plots/regression.py).
        chart = chart.transform(summary)
        chart._mark = None
        chart._layers = summary_layers
    elif kind == "count":
        # Aggregate(count of cat_field) → bar.
        if cat_field is None:
            raise ValueError("catplot(kind='count') requires x= (or y= when orient='h')")
        op = AggregateOp(cat_field, "count", "n")
        # Group by the hue column too when one is bound. Aggregate emits only
        # its groupby columns, so grouping by the category alone dropped the
        # hue column from the transform's output and the color encoding then
        # referenced a column that no longer existed -- `catplot(kind="count",
        # hue=...)` raised "unknown column referenced by an encoding" for every
        # hue dtype. Counting per (category, hue) is also the reading the rest
        # of this branch already assumes, since `dodge=True` offsets the bars
        # by hue level, and it matches seaborn's countplot(hue=).
        count_groupby = [cat_field]
        if hue is not None:
            hue_name = _field_name(hue)
            if hue_name != cat_field:
                count_groupby.append(hue_name)
        chart = chart.transform(Aggregate([op], groupby=count_groupby))
        chart = chart.mark_bar(position=position) if position is not None else chart.mark_bar()
        # Remap value axis to the count column.
        # After enc normalisation, val is always on y (cat on x), so "n" always
        # goes on y regardless of orientation.
        enc["y"] = "n"

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

    return _finalize_chart(
        chart, mark=mark, encode=encode, properties=properties, layers=layers, theme=theme
    )


# ---------------------------------------------------------------------------
# relplot
# ---------------------------------------------------------------------------


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
        Column name to map to the ``Size`` channel (point area or line width).
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
        # seaborn does the same. kind="line" groups rows into one polyline per
        # level, so an inferred Continuous scale collapses every group into a
        # single line; that one is typed. See `nominal_color_channel`.
        enc["color"] = hue if kind == "scatter" else nominal_color_channel(hue)
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
        chart, mark=mark, encode=encode, properties=properties, layers=layers, theme=theme
    )
