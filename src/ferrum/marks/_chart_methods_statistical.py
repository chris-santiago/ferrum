"""Statistical mark methods extracted from Chart as a mixin.

These methods are duck-typed on ``self`` — they call ``self._set_composite_mark``,
``self._clone``, ``self._mark``, ``self._encoding``, ``self._data``, ``self._layers``,
``self._transforms``, ``self._position``, etc.  The mixin does not import ``Chart``
and introduces no ``__init__`` or ``__slots__``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

from ferrum._core import blend_accepted_values
from ferrum._encode_lowering import _refuse_bin_aggregate_on_channels, _unbound_input_refusal
from ferrum._layer import _Layer, _PendingMark, layer_mark, merge_style
from ferrum._validate import coerce_bool, validate_choice, validate_optional_vocabulary
from ferrum.encoding.base import ChannelBase
from ferrum.marks.base import _FIGURE_ONLY_KINDS, ORIENT_CHOICES
from ferrum.marks.composite import _MISSING
from ferrum.marks.statistical import (
    _resolve_density,
    _resolve_histogram,
    _resolve_smooth,
)

if TYPE_CHECKING:
    import polars as pl


def _mark_raster_field_schema(data: Any) -> "pl.Schema | None":
    """Best-effort, non-materializing schema lookup for ``mark_raster``'s
    eager ``field=`` check (close-remediation cycle 2, design-review S3).

    Returns ``None`` when the schema cannot be resolved cheaply and safely
    -- the caller then skips the eager check entirely, and the render-time
    ``stat_raster`` transform surfaces whatever error applies, exactly as
    it did before ``mark_raster``'s ``field=`` validation existed. A
    chart-*declaration* method must never execute a lazy pipeline or let
    an untyped, non-ferrum exception (e.g. a bare ``FileNotFoundError``
    from a ``pl.scan_parquet`` over a missing path) escape from it.

    - ``pl.LazyFrame``: ``collect_schema()`` resolves the schema WITHOUT
      executing the query plan or reading any of the underlying data.
      This is the fix's whole point -- the eager check previously called
      ``to_polars(self._data)``, which for a ``LazyFrame`` calls
      ``.collect()`` internally and materializes the entire result (27 ms
      on a 4M-row parquet scan; scales with the source, not with anything
      ``mark_raster()`` needs). ``mark_point()`` and every other
      declaration-time method on a lazy-data ``Chart`` stay free; this
      check must too.
    - Anything else (``pl.DataFrame``, dict, list, numpy, pandas, ...) is
      already fully realized in memory by construction, so
      :func:`ferrum._coerce.to_polars` is cheap and this path is unchanged
      from before this fix.

    Any exception raised while resolving the schema (an unreachable scan
    path, an input shape ``to_polars`` cannot coerce, ...) is caught and
    turned into ``None`` here rather than escaping from this public
    builder -- the broad catch is deliberate: whatever the underlying
    source's own exception type is, it means "the eager check cannot run
    safely right now", not "field= is invalid", so it must not be
    reported as a field-validation refusal.
    """
    import polars as pl
    from ferrum._coerce import to_polars

    try:
        if isinstance(data, pl.LazyFrame):
            return data.collect_schema()
        return to_polars(data).schema
    except Exception:
        return None


def _normalize_orient(func_name: str, orient: str | None, horizontal: bool) -> bool:
    """Return the effective ``horizontal`` flag from the canonical ``orient`` kwarg.

    ``orient`` is the canonical spelling across the mark family.  ``horizontal``
    is a legacy bool alias accepted by boxplot/boxen/violin.  When ``orient`` is
    not ``None``, it wins; otherwise the explicit ``horizontal`` value is used.

    Valid values for ``orient``: ``"vertical"`` (default) or ``"horizontal"``.
    Any other string raises ``ValueError`` immediately so the error surfaces at
    the call site rather than silently being absorbed as a style kwarg.
    ``func_name`` is the public ``mark_*`` method the caller invoked, used as
    the validator's error-message prefix.
    """
    if orient is not None:
        validate_choice(func_name, "orient", orient, ORIENT_CHOICES)
        return orient == "horizontal"
    return bool(horizontal)


def _normalize_orient_to_orientation(func_name: str, kwargs: dict) -> dict:
    """Normalize ``orient=`` to ``orientation=`` for density/histogram desugars.

    ``desugar_density`` and ``desugar_histogram`` use ``orientation`` as their
    internal parameter name (historical spelling).  This helper pops ``orient``
    from a copy of *kwargs* and injects ``orientation`` so the desugar receives
    its expected param name.  ``orient`` wins over ``orientation`` when both are
    present. ``func_name`` is the public ``mark_*`` method the caller invoked,
    used as the validator's error-message prefix.

    Returns a new dict; the caller's original dict is not mutated.
    """
    if "orient" not in kwargs:
        return kwargs
    orient_val = kwargs["orient"]
    validate_choice(func_name, "orient", orient_val, ORIENT_CHOICES)
    new_kwargs = {k: v for k, v in kwargs.items() if k != "orient"}
    new_kwargs["orientation"] = orient_val
    return new_kwargs


def _refuse_fitted_mark_after_pending_composite(entry: str, pending: _PendingMark | None) -> None:
    """Refuse *entry* (``mark_function``/``mark_smooth``) immediately after a
    pending composite mark on the same chart.

    A chart holds one ``_pending_stat_mark`` slot (``_layer._PendingMark``),
    so a second composite mark cannot be kept there alongside it. A fitted
    mark keeps every layer already drawn (unlike a composite mark, which
    replaces the chart's mark), so this refusal names the ``+`` spelling that
    gives each composite its own chart, layer, data and style instead --
    always the ``mark_<kind>()`` method every ``_set_composite_mark`` caller
    is named for (``pending.kind``), a real ``Chart`` method, never
    ``pending.entry``: a figure function (``displot()``, ``lmplot()``, ...)
    that leaves its own pending mark unresolved names no such method. When a
    figure or visualizer built the pending mark, its name still appears, as
    the source of the mark being replaced, not as the remedy to type.

    A figure-only kind (:data:`~ferrum.marks.base._FIGURE_ONLY_KINDS`, built
    only by a figure call and no ``mark_<kind>()`` method) is the one
    exception: the pending method would name a call that does not exist, so
    the refusal and its ``+`` remedy name the figure call itself instead,
    with no separate "(built by ...)" clause -- the name already is that
    call.
    """
    if pending is None:
        return
    figure_call = _FIGURE_ONLY_KINDS.get(pending.kind)
    if figure_call is not None:
        raise ValueError(
            f"{entry}() cannot follow {figure_call.name} on one chart: a chart holds "
            f"one composite mark, and {entry}() keeps every layer already drawn. Layer "
            f"them with +: {figure_call.operand} + chart.{entry}(...)"
        )
    pending_method = f"mark_{pending.kind}"
    source = f" (built by {pending.entry}())" if pending.entry else ""
    raise ValueError(
        f"{entry}() cannot follow {pending_method}(){source} on one chart: a chart holds "
        f"one composite mark, and {entry}() keeps every layer already drawn. Layer them "
        f"with +: chart.{pending_method}() + chart.{entry}(...)"
    )


def _refuse_primitive_mark_on_layered_chart(entry: str, layers: Sequence[_Layer] | None) -> None:
    """Refuse *entry* (``mark_point``/``mark_bar``/...) on a chart already drawn from layers.

    A chart holds either one mark of its own or a list of layers, never
    both meaningfully at once: a layered chart's ``_mark`` only mirrors
    layer 0, so a primitive call that "replaces" it would leave every
    layer's own mark still drawing the render unchanged, with no visible
    effect -- the silent drop the library's refuse-malformed rule forbids.
    There is no earlier single mark for this call to replace, so it refuses
    instead, before any of its own kwargs or ``position=`` are read.

    Each layer is named by its own ``name`` where it has one -- the same
    key a figure's ``mark={'<layer>': {...}}`` targets, so the refusal's
    list and its restyle remedy agree on what to call each layer -- or by
    the mark it draws otherwise (:func:`ferrum._layer.layer_mark`: a layer
    with no mark of its own draws as a point, the same wire default every
    other reader of an unmarked layer already uses).
    """
    if not layers:
        return
    names = ", ".join(layer.name or layer_mark(layer.mark) for layer in layers)
    count = len(layers)
    noun = "layer" if count == 1 else "layers"
    raise ValueError(
        f"{entry}() cannot replace the marks of a chart drawn from {count} {noun} "
        f"({names}): a mark call replaces a chart's one mark, and a layered chart "
        "has none of its own. Set the mark on a chart before layering it "
        f"(fm.Chart(...).{entry}() + ...), or restyle a layer by name "
        "(a figure's mark={'<layer>': {...}})."
    )


def _mark_function_alone_remedy(kwarg: str) -> str:
    """The remedy for *kwarg* (``"bin"``/``"aggregate"``) mark_function reads over no data layer.

    A data layer before ``mark_function`` in the chain (``chart.mark_point()
    .encode(x=fm.X('a', bin=True), ...).mark_function(fn)``) is the one
    layer that honors the option, since ``mark_function`` draws fn's own
    grid and keeps every prior layer beside it; there is no field for
    ``.transform(...)`` to bin or aggregate ahead of a layer that does not
    exist yet.
    """
    return (
        "Draw the data with a mark before mark_function "
        f"(chart.mark_point().encode(...).mark_function(fn)), which honors {kwarg}=, "
        f"or drop {kwarg}=."
    )


def _refuse_function_alone_statistics(encoding: dict) -> None:
    """Refuse an x/y bin=/aggregate= mark_function draws over no data layer.

    Its own layer is the synthetic ``(x, y)`` grid ``fn`` is evaluated over,
    so no layer reads the caller's field; a bin= or aggregate= bound on x/y
    before this call -- with no prior layer for ``mark_function`` to keep --
    would otherwise be dropped silently once the channel is rebound to that
    grid.
    """
    _refuse_bin_aggregate_on_channels(
        "mark_function",
        encoding,
        ("x", "y"),
        reason="fn is drawn over its own grid and no data layer reads the field",
        bin_remedy=_mark_function_alone_remedy("bin"),
        aggregate_remedy=_mark_function_alone_remedy("aggregate"),
    )


class StatisticalMarksMixin:
    """Mixin providing statistical mark methods for Chart."""

    # ---- Marks (statistical) ----

    def mark_density(
        self, *, fill=True, cumulative=False, smooth=True, position=None, **kwargs
    ) -> "Chart":
        """Render a kernel-density estimate (KDE) curve or filled contour.

        When only ``x`` is encoded, draws a 1-D KDE area curve.  When both
        ``x`` and ``y`` are encoded, renders a bivariate filled-contour density
        (routes through ``desugar_contour(fill=True)``).  Can be called before
        or after ``.encode()``.

        Parameters
        ----------
        bandwidth : float or "scott" or "silverman", optional
            KDE bandwidth.  ``"scott"`` and ``"silverman"`` use the
            corresponding rule-of-thumb estimates.  Default is ``"scott"``.
        kernel : str, optional
            Kernel type: ``"gaussian"``, ``"epanechnikov"`` (alias
            ``"epan"``), ``"tophat"``, ``"uniform"``, ``"cosine"``,
            ``"exponential"``, ``"linear"`` — validated against the live
            ``transform_registry("kde")["methods"]`` vocabulary.  Default
            is ``"gaussian"``.
        extent : list of float or None, optional
            ``[min, max]`` evaluation range.  Defaults to data extent.
        cumulative : bool, optional
            Produce a CDF instead of a PDF.  Default is ``False``.
        fill : bool, optional
            Draw the 1-D density as a filled area (``True``, the default) or
            as a line (``False``).  A truth value only: a fill color is
            ``color=``.
        smooth : bool, optional
            Smooth the bivariate density grid before contouring (``x`` and
            ``y`` both encoded).  Default is ``True``.
        n : int, optional
            Number of evaluation points.  Default is ``200``.
        multiple : str, optional
            How to handle multiple densities when a ``color`` encoding is
            present: ``"layer"`` (default), ``"stack"``, ``"fill"``.
        orient : {"vertical", "horizontal"}, optional
            Canonical orientation alias for ``orientation``.  ``"horizontal"``
            swaps the value and density axes.  Takes precedence over
            ``orientation`` when both are given.
        orientation : {"vertical", "horizontal"}, optional
            Legacy spelling.  ``orient`` is preferred.
        position : Position, optional
            Position adjustment.
        **mark_kwargs
            Constant mark-style overrides (``opacity``, ``stroke``,
            ``stroke_width``, ``stroke_dash``, ``size``; ``color`` sets the
            fill color, ``alpha``→``opacity``) forwarded to the rendered
            area/line layer.  ``fill=`` is the truth value above, not a color.

        Returns
        -------
        Chart
            New ``Chart`` configured for density rendering.

        Raises
        ------
        ValueError
            If the x or y channel carries a ``bin=`` or ``aggregate=``: the
            density is estimated over the raw rows, never a bin or
            aggregate of them. Compute it ahead of time with
            ``.transform(...)`` and bind its output column, or drop the
            kwarg.

        Examples
        --------
        >>> import ferrum as fm
        >>> import polars as pl
        >>> df = pl.DataFrame({"val": [1.0, 2.1, 1.8, 3.0, 2.5]})
        >>> fm.Chart(df).mark_density().encode(x="val")
        Chart(mark='area', encoding=['x'])
        """
        from ferrum.marks.base import resolve_method_bool

        # Normalize orient= → orientation= so the desugar's existing param name
        # is used.  orient= wins over orientation= when both are present.
        kwargs = _normalize_orient_to_orientation("mark_density", kwargs)
        # The truth-valued arguments the desugar reads to decide what it draws
        # (an area or a line, a CDF or a PDF, a smoothed bivariate grid) are
        # held to a truth value at the call, so `fill="red"` is refused rather
        # than read as True.
        fill = resolve_method_bool("mark_density", "fill", fill, True)
        cumulative = resolve_method_bool("mark_density", "cumulative", cumulative, False)
        smooth = resolve_method_bool("mark_density", "smooth", smooth, True)
        return self._set_composite_mark(
            "density",
            _resolve_density,
            {"fill": fill, "cumulative": cumulative, "smooth": smooth, **kwargs},
            placeholder="area",
            position=position,
        )

    def mark_histogram(self, *, position=None, **kwargs) -> "Chart":
        """Render data as a histogram.

        Bins the ``x``-encoded column and encodes bin extents as ``x``/``x2``
        with counts (or densities) on ``y``.  Can be called before or after
        ``.encode(x=...)``.

        Parameters
        ----------
        bins : str, int or float, optional
            The whole binning rule in one value: ``"sturges"``,
            ``"fd"``/``"freedman_diaconis"``, an ``int`` bin count, or a
            ``float`` bin width.  Mutually exclusive with ``bin_count`` /
            ``bin_width``.
        bin_count : int, optional
            Target number of bins.  Ignored when ``bin_width`` is set.
            Default is chosen automatically from Sturges' rule.
        bin_width : float, optional
            Exact bin width in data units.  Overrides ``bin_count``.
        density : bool, optional
            Normalise counts to a probability density.  Default is ``False``.
        right : bool, optional
            Whether bins are closed on the right.  Default is ``False``.
            ``right=True`` is not currently supported and raises
            ``ValueError``; bins are always left-closed, right-open
            ``[lo, hi)``.
        cumulative : bool, optional
            Render a cumulative histogram.  Default is ``False``.
        multiple : str, optional
            How to handle grouped histograms when a ``color`` encoding is
            present: ``"layer"`` (default), ``"stack"``, ``"fill"``,
            ``"dodge"``.  ``"stack"`` and ``"fill"`` also bin every group
            against one global extent, so the groups share bin edges and
            the bars are genuinely stackable.
        position_by : str, optional
            Column the ``multiple=`` position adjustment groups by.  Default
            (``None``) resolves the ``color`` channel at render time.  A
            column name only; an encoding object is refused, not unwrapped.
        orient : {"vertical", "horizontal"}, optional
            Canonical orientation alias for ``orientation``.  ``"horizontal"``
            swaps the bin-edge and count axes.  Takes precedence over
            ``orientation`` when both are given.
        orientation : {"vertical", "horizontal"}, optional
            Legacy spelling.  ``orient`` is preferred.
        position : Position, optional
            Position adjustment.
        **mark_kwargs
            Constant mark-style overrides (``opacity``, ``fill``, ``stroke``,
            ``stroke_width``, ``stroke_dash``, ``size``; aliases ``color``→``fill``,
            ``alpha``→``opacity``) forwarded to the rendered bar layer.

        Returns
        -------
        Chart
            New ``Chart`` configured for histogram rendering.

        Raises
        ------
        ValueError
            If the binned axis carries a ``bin=`` that states a rule of its
            own (``bin=True`` is a harmless redundancy and is honored; move
            a real rule to ``bins=``/``bin_count=``/``bin_width=``), or any
            ``aggregate=`` (the histogram bins that axis rather than
            aggregating it; aggregate ahead of it with
            ``.transform(fm.transform_aggregate(...))`` and bind the output
            column).
        ValueError
            If the value axis carries a ``bin=``, or an ``aggregate=`` other
            than a fieldless ``"count"``. The value axis draws the bin's own
            count and never binds a column, so a fieldless one's remedy is
            to drop the kwarg. A field-bearing one (``aggregate="count"``
            included) is refused with a remedy that draws what it asked for
            over both fields: the per-bin aggregate with ``mark_bar()`` (a
            ``bin=True`` channel on the histogram's field beside the
            aggregated one), a two-field count with ``mark_hex()`` for a
            ``bin=``, or dropping the binding and styling the axis with a
            fieldless ``fm.Y(title=...)`` (``fm.X`` when horizontal). With
            no field bound on the binned axis, the remedy binds one there
            too. A fieldless ``aggregate="count"`` is honored, except under
            ``density=True`` (the value axis draws density there, not a
            count) or a normalized stack -- ``multiple="fill"``,
            ``position=fm.Stack(offset="normalize")`` or a channel
            ``stack="normalize"`` -- which draws each bin's share of its
            stack there, not a count.
        ValueError
            If ``multiple=`` or ``orientation=`` is not one of its values,
            ``right=True`` is passed, ``position_by=`` is not a column name,
            or no field is bound on the binned axis.

        Examples
        --------
        >>> import ferrum as fm
        >>> import polars as pl
        >>> df = pl.DataFrame({"val": [1.0, 1.2, 2.3, 3.1, 2.8, 1.5]})
        >>> fm.Chart(df).mark_histogram(bin_count=5).encode(x="val")
        Chart(mark='bar', encoding=['x'])
        """
        # Normalize orient= → orientation= so the desugar's existing param name
        # is used.  orient= wins over orientation= when both are present.
        kwargs = _normalize_orient_to_orientation("mark_histogram", kwargs)
        return self._set_composite_mark(
            "histogram",
            _resolve_histogram,
            kwargs,
            placeholder="bar",
            position=position,
        )

    def mark_smooth(self, *, position=None, layer_kwargs: dict | None = None, **kwargs) -> "Chart":
        """Render a smoothed regression line with optional confidence interval band.

        Fits a smooth curve through ``(x, y)`` data using the method specified by
        ``method``.  When ``ci`` is set, emits a layered ribbon (CI band) + line
        chart.  Can be called before or after ``.encode(x=..., y=...)``.

        Parameters
        ----------
        method : str, optional
            Smoothing method.  Default is ``"loess"``.

            - ``"lm"`` or ``"linear"`` -- linear regression (OLS, degree-1 polynomial);
              promoted to a polynomial fit when ``degree >= 2`` is also given
            - ``"loess"`` -- local regression (LOWESS)
            - ``"quadratic"`` -- polynomial regression, degree 2
            - ``"cubic"`` -- polynomial regression, degree 3
            - ``"log"`` -- linear fit on log-transformed x
            - ``"sqrt"`` -- linear fit on sqrt-transformed x
            - ``"glm"`` -- Generalized Linear Model (Gaussian family, identity
              link) fit via IRLS
            - ``"gam"`` -- Generalized Additive Model: a penalized cubic
              B-spline (P-spline) whose smoothing parameter is chosen by
              generalized cross-validation. Bends to follow the data like
              ``"loess"`` does, but deterministically -- its confidence band
              is the spline's Bayesian posterior band, so ``seed`` does not
              affect it. Outside the observed x range it holds its boundary
              value rather than extrapolating. Needs at least 4 distinct x
              values to build its basis; fewer degrade to that slice's NaN
              range (no fit line drawn), like any degenerate fit in this
              family.
        degree : int, optional
            Acceptance depends on ``method``:

            - ``"lm"``/``"linear"`` -- any ``degree >= 1``; ``>= 2`` promotes
              to ``Poly{degree}`` (``degree=2`` is byte-equal to
              ``method="quadratic"``). ``degree=0`` raises. A degree that
              leaves too few data points to fit (rank-deficient) raises for
              a whole-frame fit; one group of a ``groupby`` fit, or one
              panel of a faceted chart, degrades that slice to NaN instead.
            - ``"quadratic"``/``"cubic"`` -- accepted only as a no-op when it
              matches the method's fixed degree (``2``/``3``); any other
              value raises.
            - ``"loess"`` -- the local-fit degree, ``1`` or ``2``.
            - ``"log"``/``"sqrt"``/``"glm"``/``"gam"`` -- no degree concept;
              passing any value raises. (``"gam"``'s spline degree is fixed
              at 3; its smoothness comes from the GCV-chosen penalty --
              ``k`` is its knob, see below.)
        k : int, optional
            ``method="gam"``'s spline basis dimension: the number of
            B-spline basis functions the penalized fit may spend, i.e. the
            ceiling on how wiggly the curve can get before the GCV-chosen
            penalty pulls it back. Default is ``10``; the effective value is
            ``clamp(k, 4, distinct_x)``, so a ``k`` above the number of
            distinct x values is capped there. ``k < 4`` raises (no smaller
            cubic basis exists), and so does ``k`` on any method other than
            ``"gam"`` -- raise it when the default under-smooths a signal
            with many turning points.
        bandwidth : float, optional
            LOESS span (the fraction of points in each local neighbourhood),
            in ``(0, 1]``.  Default is ``0.75``.  Applies to
            ``method="loess"`` only: it is the LOESS knob and no other
            method reads it, so an explicit value elsewhere raises rather
            than silently doing nothing.
        n : int, optional
            Number of evaluation points along the x range.  Default is ``200``.
        ci : float or None, optional
            Confidence level for the interval band, e.g. ``0.95``.  When set,
            produces a layered ribbon + line chart.  Default is ``None``.
        groupby : str, optional
            Group-key column (any groupby-supported dtype — string,
            integer, float, or boolean).  When set, the smooth is
            computed independently per group and the group column is
            preserved at its original dtype in the output
            so downstream ``color=`` encoding maps to it.
        layer_kwargs : dict of str to dict, optional
            Mark-style overrides aimed at ONE emitted sub-layer, keyed by
            layer name: ``{"ribbon": ..., "line": ..., "metrics": ...}`` for
            this mark. ``mark_smooth(layer_kwargs={"line": {"stroke":
            "red"}})`` is what ``lmplot``'s ``line_kws`` forwards here, and
            is distinct from ``**mark_kwargs`` below, which reach every
            emitted layer: a bare ``stroke=`` on a ``ci``-bearing call
            repaints the confidence band too, so styling just the line needs
            its own door.

            The door is not this mark's alone -- every composite ``mark_*``
            accepts ``layer_kwargs=`` through its ``**mark_kwargs``, keyed on
            the same per-mark layer vocabulary the figure-level
            ``mark={"line": {...}}`` override uses
            (``_overrides.LAYER_NAME_CATALOG``); e.g.
            ``mark_boxplot(layer_kwargs={"median": {"stroke": "black"}})``.
            A name that mark cannot draw raises ``ValueError`` naming the
            accepted set rather than silently styling nothing; a name it can
            draw but this call does not emit (``"ribbon"`` under ``ci=None``)
            is a no-op, exactly as it is for the figure-level override.

            **Precedence:** a per-layer value WINS over a broad
            ``**mark_kwargs`` naming the same key, because it is the more
            specific instruction -- ``mark_smooth(stroke="#0000ff",
            layer_kwargs={"line": {"stroke": "#ff0000"}})`` draws the fit
            line red and the CI band blue, on the layered shape and the no-CI
            single-line shape alike. The broad value still applies everywhere
            nothing narrower was said.
        position : Position, optional
            Position adjustment.
        **mark_kwargs
            Constant mark-style overrides (``opacity``, ``fill``, ``stroke``,
            ``stroke_width``, ``stroke_dash``, ``size``; aliases ``color``→``fill``,
            ``alpha``→``opacity``) forwarded to all emitted layers (the regression
            line, plus the CI ribbon when ``ci`` is set). See ``layer_kwargs``
            above to style the line without the band.

        Returns
        -------
        Chart
            New ``Chart`` configured for smooth rendering. A primitive mark
            already set (``mark_point()`` etc.) is kept as its own layer
            beside the fit, with its own data and style.

        Raises
        ------
        ValueError
            When this chart has a pending composite mark (``mark_histogram()``,
            another ``mark_smooth()``, ...); use ``+`` instead.
        ValueError
            If the x channel carries a ``bin=`` (the fit bins before it
            dispatches on method, so ``x_bins=`` is the remedy) or an
            ``aggregate=`` (compute it ahead of time with ``.transform(...)``
            and bind its output column -- never ``x_estimator=``, which
            aggregates the response within each ``x_bins`` bin, not the
            predictor), or if the y channel carries either (the fit reads
            its response raw; the same ``.transform(...)`` remedy applies).

        Examples
        --------
        >>> import ferrum as fm
        >>> import polars as pl
        >>> df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [1.1, 1.9, 3.1, 4.0]})
        >>> fm.Chart(df).mark_smooth(method="linear", ci=0.95).encode(x="x", y="y")
        Chart(mark='line', encoding=['x', 'y'])
        """
        _refuse_fitted_mark_after_pending_composite("mark_smooth", self._pending_stat_mark)
        from ferrum._desugar import validate_layer_kwargs
        from ferrum._core import x_estimator_accepted_values
        from ferrum._validate import validate_optional_choice

        # Declared in this signature rather than left to `**kwargs` because
        # it is the mark whose layered shape most needs it, and a named
        # parameter is what makes it discoverable. Checked HERE as well as in
        # the dispatcher -- the same function, so one rule -- so a typo'd
        # layer name refuses at the `mark_smooth(...)` call that wrote it
        # rather than at the eventual `.to_svg()`.
        validate_layer_kwargs("smooth", layer_kwargs, where="mark_smooth")
        # x_estimator= is refused at the call, like lmplot's.
        validate_optional_choice(
            "mark_smooth", "x_estimator", kwargs.get("x_estimator"), x_estimator_accepted_values()
        )
        if layer_kwargs is not None:
            kwargs["layer_kwargs"] = layer_kwargs
        # Any primitive mark already set is kept as its own prior layer (the
        # refusal above guarantees _mark cannot be a composite's placeholder
        # here, since that would mean a pending composite was still set).
        _pm = self._mark
        return self._set_composite_mark(
            "smooth",
            _resolve_smooth,
            kwargs,
            placeholder="line",
            position=position,
            prior_mark=_pm,
        )

    # ---- Marks (deferred) ----

    def mark_boxplot(
        self,
        *,
        whisker_mult: float | str = _MISSING,  # type: ignore[assignment]
        size=None,
        width=None,
        outliers=True,
        color_field=None,
        horizontal=False,
        orient=None,
        position=None,
        **mark_kwargs,
    ) -> "Chart":
        """Render a Tukey boxplot via composite mark desugaring.

        Desugars to a multi-layer chart: box (IQR rect), upper and lower
        whisker rules, median rule, and (optionally) outlier points.  Requires
        a categorical ``x`` encoding and a continuous ``y`` (or the reverse
        when ``horizontal=True`` or ``orient="horizontal"``).

        Parameters
        ----------
        whisker_mult : float or "min-max", optional
            Whisker reach as a multiple of IQR, or ``"min-max"`` to extend
            whiskers to the data minimum/maximum.  Default is ``1.5``.
            ``"min-max"`` also suppresses the outlier point layer regardless
            of ``outliers=``: whiskers already reach every observation, so
            nothing can lie beyond them.

            .. deprecated::
                The old spelling ``extent=`` is accepted as an alias for one
                release.  Use ``whisker_mult=`` instead.  ``extent`` is
                reserved for data-domain ``[min, max]`` bounds.
        size : float or None, optional
            Box band-width as a fraction of the category band (default ``0.6``).
            Alias: ``width``.
        width : float or None, optional
            Alias for ``size``.  Controls box band-width as a fraction of the
            category band.  When both ``size`` and ``width`` are provided,
            ``size`` takes precedence.
        outliers : bool, optional
            Whether to draw outlier points beyond the whiskers.  Default is
            ``True``.  Forced off when ``whisker_mult="min-max"`` (see above),
            regardless of this flag.
        color_field : str or None, optional
            Column name to drive per-group fill colour on the box layer.
        horizontal : bool, optional
            Swap x and y axes so boxes run horizontally.  Default is ``False``.
            Legacy alias; prefer ``orient="horizontal"``.
        orient : {"vertical", "horizontal"} or None, optional
            Canonical orientation control.  ``"horizontal"`` is equivalent to
            ``horizontal=True``.  When both are given, ``orient`` wins if it is
            not ``None``.
        position : Position, optional
            Position adjustment -- e.g. ``fm.Dodge()`` for side-by-side boxes.
        **mark_kwargs
            Constant mark-style overrides (``opacity``, ``fill``, ``stroke``,
            ``stroke_width``, ``stroke_dash``, ``size``; aliases ``color``→``fill``,
            ``alpha``→``opacity``) forwarded to each constituent layer.

        Returns
        -------
        Chart
            New layered ``Chart`` representing the boxplot.

        Examples
        --------
        >>> import ferrum as fm
        >>> import polars as pl
        >>> df = pl.DataFrame({"species": ["a"]*5 + ["b"]*5, "val": [1,2,3,2,1,4,5,4,5,6]})
        >>> fm.Chart(df).mark_boxplot().encode(x="species", y="val")
        Chart(mark='point', encoding=['x', 'y'])
        """
        from ferrum.marks.composite import desugar_boxplot, _resolve_deprecated_extent

        # width is an alias for size; size takes precedence when both are given.
        effective_size = size if size is not None else width
        # Normalize orient/horizontal: orient= is canonical; horizontal= is a legacy
        # alias. When orient is given explicitly, it overrides horizontal.
        effective_horizontal = _normalize_orient("mark_boxplot", orient, horizontal)
        # Resolve the deprecated extent= alias at the public API level using the
        # shared resolver from composite.py.  Sentinel-based detection ensures that
        # whisker_mult=1.5 (the default value, passed explicitly) is correctly
        # treated as "canonically supplied" and triggers TypeError when extent= is
        # also given.  mark_kwargs is mutated in-place (extent popped) by the helper.
        whisker_mult = _resolve_deprecated_extent(
            "whisker_mult", whisker_mult, mark_kwargs, real_default=1.5
        )

        return self._set_composite_mark(
            "boxplot",
            desugar_boxplot,
            {
                "whisker_mult": whisker_mult,
                "size": effective_size,
                "outliers": outliers,
                "color_field": color_field,
                "horizontal": effective_horizontal,
                **mark_kwargs,
            },
            placeholder="point",
            position=position,
        )

    def mark_boxen(
        self,
        *,
        k_depth: str = "tukey",
        k_proportion: float = 0.007,
        outlier_threshold: float = 0.0,
        palette: str | Sequence[str] | None = None,
        horizontal: bool = False,
        orient=None,
        color_field=None,
        position=None,
        **mark_kwargs,
    ) -> "Chart":
        """Render a letter-value (boxen) plot via composite mark desugaring.

        Produces nested rectangular bands for each letter-value depth, a
        band-confined median tick, and an outlier-point layer via the
        ``LetterValue`` transform. Requires a categorical ``x`` encoding and
        a continuous ``y`` (or the reverse when ``horizontal=True`` or
        ``orient="horizontal"``).

        Parameters
        ----------
        k_depth : {"tukey", "proportion", "trustworthy", "full"}, default "tukey"
            Rule for choosing the number of letter-value levels. Every
            strategy clamps at depth 12 (the registry-derived band cap,
            ``letter_value``'s ``K_MAX_NAMED``): two ``k_depth`` rules that
            differ only in the depth they'd compute beyond 12 (group sizes
            from roughly 4,000 rows for ``"full"`` up into the tens of
            thousands for ``"tukey"``/``"trustworthy"``, depending on the
            strategy's own formula) render identically, since neither ever
            builds a layer past the cap.
        k_proportion : float, optional
            Proportion parameter used when ``k_depth="proportion"``.  Default
            is ``0.007``.
        outlier_threshold : float, optional
            Fence multiple applied to the OUTERMOST computed letter-value
            band's own span (``upper - lower`` of the deepest depth
            ``LetterValue`` reaches for a group): a value is flagged as an
            outlier when it falls outside ``[lower - outlier_threshold *
            span, upper + outlier_threshold * span]``.  Default is ``0.0``,
            so the fence sits at the outermost letter values themselves and
            every observation strictly beyond them is flagged as an
            outlier; a value exactly at the fence (a tie) is not flagged.
            A positive value widens the fence by that multiple of the
            deepest band's own span on each side.
        palette : str, sequence of str, or None, optional
            Colors the depth bands directly at opacity 1.0, replacing the
            opacity ramp.  The mapping anchors on the **base band**:
            ``colors[0]`` lands on ``k=2``, the innermost real interval
            beyond the median -- guaranteed to render whenever any
            non-degenerate depth exists, for any dataset -- and later
            colors consume outward as richer data materializes more
            bands.  Band ``k=1`` (the median tick's own row, always
            zero-pixel by construction) never consumes a color of its
            own; it borrows ``k=2``'s.  Bands paint widest-first
            (outermost under, innermost on top) so the nesting stays
            visible once the ramp's alpha blending is gone.  A ``str``
            names a registered palette (expanded and cycled to the number
            of colorable bands, like ``scheme=`` on a ``Color`` channel);
            a sequence of color strings is applied in order and cycled if
            shorter.  For a dataset whose real letter-value depth falls
            short of the mark's configured band count, the *later*
            (outer) colors may go unused on that render -- the same as
            any palette longer than the actual number of categories
            elsewhere; ``colors[0]`` itself is visible for any dataset
            with at least one real depth level.  This is depth-band
            coloring, not a seaborn-style hue mapping, and ``palette``
            does not interact with ``color_field`` (which only changes the
            category, not color).  The chart's color, under any
            spelling (``color=``, ``fill=`` or ``stroke=``) and whether
            bound on the chart or on a composition it is part of, is the
            hue: the letter values are computed per category and hue, and
            each hue's bands and outliers draw in its color, placed within
            the category as ``mark_boxplot`` places its hue boxes (a
            ``Dodge`` position dodges both alike).  It conflicts with
            ``palette`` (the color encoding always overrides a layer's
            fill), so combining the two raises ``ValueError`` -- drop the
            color binding, or drop ``palette=``.  ``None`` (the default) keeps the opacity-ramp
            shading applied to the ordinary mark-color resolution (an
            explicit ``fill=`` override, else the chart's ``color``
            encoding through the theme's categorical palette, else the
            theme's default ``mark_color``).  An unrecognized palette
            name, or a non-str non-iterable value, raises ``ValueError``.
        horizontal : bool, optional
            Swap axes so bands run horizontally.  Default is ``False``.
            Legacy alias; prefer ``orient="horizontal"``.
        orient : {"vertical", "horizontal"} or None, optional
            Canonical orientation control.  ``"horizontal"`` is equivalent to
            ``horizontal=True``.  When both are given, ``orient`` wins if it is
            not ``None``.
        color_field : str or None, optional
            Column name to drive per-group colour.
        position : Position, optional
            Position adjustment.
        **mark_kwargs
            Constant mark-style overrides (``opacity``, ``fill``, ``stroke``,
            ``stroke_width``, ``stroke_dash``, ``size``; aliases ``color``→``fill``,
            ``alpha``→``opacity``) forwarded to constituent layers.

        Returns
        -------
        Chart
            New layered ``Chart`` representing the boxen plot.

        Examples
        --------
        >>> import ferrum as fm
        >>> import polars as pl
        >>> df = pl.DataFrame({"group": ["a"]*20 + ["b"]*20,
        ...                    "val": list(range(20)) + list(range(10, 30))})
        >>> fm.Chart(df).mark_boxen().encode(x="group", y="val")
        Chart(mark='point', encoding=['x', 'y'])
        """
        from ferrum.marks.composite import desugar_boxen

        effective_horizontal = _normalize_orient("mark_boxen", orient, horizontal)

        return self._set_composite_mark(
            "boxen",
            desugar_boxen,
            {
                "k_depth": k_depth,
                "k_proportion": k_proportion,
                "outlier_threshold": outlier_threshold,
                "palette": palette,
                "horizontal": effective_horizontal,
                "color_field": color_field,
                **mark_kwargs,
            },
            placeholder="point",
            position=position,
        )

    def mark_errorbar(
        self, *, method=_MISSING, ticks=True, position=None, **mark_kwargs
    ) -> "Chart":
        """Render error bars via the ``ErrorExtent`` transform.

        Computes extent values (CI, SD, SEM, or IQR) per group defined by the
        ``x`` encoding, then draws a vertical rule spanning the extent with
        optional tick caps.

        Parameters
        ----------
        method : {"ci", "stderr", "stdev", "iqr"}, default "ci"
            Aggregation method: confidence interval (``"ci"``), standard error
            (``"stderr"``), standard deviation (``"stdev"``), or
            interquartile range (``"iqr"``).  Forwarded verbatim as
            ``ErrorExtent.method`` (wire field unchanged).

            .. deprecated::
                The old spelling ``extent=`` is accepted as an alias for one
                release.  Use ``method=`` instead.  ``extent`` is reserved for
                data-domain ``[min, max]`` bounds.
        ticks : bool, optional
            Whether to draw horizontal tick caps at the extent endpoints.
            Default is ``True``.
        position : Position, optional
            Position adjustment.
        **mark_kwargs
            Constant mark-style overrides (``opacity``, ``fill``, ``stroke``,
            ``stroke_width``, ``stroke_dash``, ``size``; aliases ``color``→``fill``,
            ``alpha``→``opacity``) forwarded to the constituent rule/tick layers.

        Returns
        -------
        Chart
            New layered ``Chart`` representing the error bars.

        Examples
        --------
        >>> import ferrum as fm
        >>> import polars as pl
        >>> df = pl.DataFrame({"group": ["a"]*10 + ["b"]*10,
        ...                    "val": list(range(10)) + list(range(5, 15))})
        >>> fm.Chart(df).mark_errorbar(method="stdev").encode(x="group", y="val")
        Chart(mark='point', encoding=['x', 'y'])
        """
        from ferrum.marks.composite import desugar_errorbar, _resolve_deprecated_extent

        # Resolve the deprecated extent= alias at the public API level using the
        # shared resolver from composite.py.  Using the _MISSING sentinel for the
        # method default means method="ci" (the canonical default) passed explicitly
        # is correctly detected as "canonically supplied" and triggers TypeError
        # when extent= is also given.  mark_kwargs is mutated in-place by the helper.
        method = _resolve_deprecated_extent("method", method, mark_kwargs, real_default="ci")

        return self._set_composite_mark(
            "errorbar",
            desugar_errorbar,
            {"method": method, "ticks": ticks, **mark_kwargs},
            placeholder="point",
            position=position,
        )

    def mark_errorband(
        self, *, method=_MISSING, borders=False, position=None, **mark_kwargs
    ) -> "Chart":
        """Render an error band (ribbon) via the ``ErrorExtent`` transform.

        Similar to ``mark_errorbar`` but renders the extent as a filled ribbon
        rather than whisker rules.  Optionally draws border lines at the upper
        and lower extent edges.

        Parameters
        ----------
        method : {"ci", "stderr", "stdev", "iqr"}, default "ci"
            Aggregation method: confidence interval (``"ci"``), standard error
            (``"stderr"``), standard deviation (``"stdev"``), or
            interquartile range (``"iqr"``).  Forwarded verbatim as
            ``ErrorExtent.method`` (wire field unchanged).

            .. deprecated::
                The old spelling ``extent=`` is accepted as an alias for one
                release.  Use ``method=`` instead.  ``extent`` is reserved for
                data-domain ``[min, max]`` bounds.
        borders : bool, optional
            Whether to draw line borders at the band edges.  Default is ``False``.
        position : Position, optional
            Position adjustment.
        **mark_kwargs
            Constant mark-style overrides (``opacity``, ``fill``, ``stroke``,
            ``stroke_width``, ``stroke_dash``, ``size``; aliases ``color``→``fill``,
            ``alpha``→``opacity``) forwarded to the ribbon layer.

        Returns
        -------
        Chart
            New layered ``Chart`` representing the error band.

        Examples
        --------
        >>> import ferrum as fm
        >>> import polars as pl
        >>> df = pl.DataFrame({"x": list(range(10)), "y": [float(i) for i in range(10)]})
        >>> fm.Chart(df).mark_errorband(method="ci", borders=True).encode(x="x", y="y")
        Chart(mark='point', encoding=['x', 'y'])
        """
        from ferrum.marks.base import resolve_method_bool
        from ferrum.marks.composite import desugar_errorband, _resolve_deprecated_extent

        borders = resolve_method_bool("mark_errorband", "borders", borders, False)
        # Resolve the deprecated extent= alias at the public API level using the
        # shared resolver from composite.py.  Using the _MISSING sentinel for the
        # method default means method="ci" (the canonical default) passed explicitly
        # is correctly detected as "canonically supplied" and triggers TypeError
        # when extent= is also given.  mark_kwargs is mutated in-place by the helper.
        method = _resolve_deprecated_extent("method", method, mark_kwargs, real_default="ci")

        return self._set_composite_mark(
            "errorband",
            desugar_errorband,
            {"method": method, "borders": borders, **mark_kwargs},
            placeholder="point",
            position=position,
        )

    def mark_ribbon(
        self, *, opacity=0.3, interpolate="linear", position=None, **mark_kwargs
    ) -> "Chart":
        """Render a ribbon (filled area between ``y`` and ``y2`` along ``x``).

        Requires both ``y`` and ``y2`` encoding channels.  Typically used to
        visualise confidence intervals alongside a ``mark_line`` in the same
        layered chart.

        Parameters
        ----------
        opacity : float, optional
            Fill opacity of the ribbon.  Default is ``0.3``.
        interpolate : str, optional
            Curve drawn through both the lower (``y``) and upper (``y2``)
            boundaries: one of ``"linear"``, ``"step"``, ``"step-before"``,
            ``"step-after"``, ``"monotone"``, ``"basis"`` or ``"cardinal"``
            (the live ``ferrum._core.interpolate_accepted_values()``
            vocabulary). Any other value raises ``ValueError`` at this call.
            Default is ``"linear"``.
        position : Position, optional
            Position adjustment.
        **mark_kwargs
            Additional mark-style overrides (e.g. ``fill``, ``stroke``).

        Returns
        -------
        Chart
            New ``Chart`` configured for ribbon rendering.

        Examples
        --------
        >>> import ferrum as fm
        >>> import polars as pl
        >>> df = pl.DataFrame({"x": [1, 2, 3], "y_lo": [0.5, 1.5, 2.5], "y_hi": [1.5, 2.5, 3.5]})
        >>> fm.Chart(df).mark_ribbon().encode(x="x", y="y_lo", y2="y_hi")
        Chart(mark='ribbon', encoding=['x', 'y', 'y2'])
        """
        from ferrum._core import interpolate_accepted_values
        from ferrum._validate import validate_optional_vocabulary
        from ferrum.marks.composite import desugar_ribbon

        # Refused at the call, like every other interpolate= door.
        validate_optional_vocabulary(
            "mark_ribbon", "interpolate", interpolate, interpolate_accepted_values()
        )

        return self._set_composite_mark(
            "ribbon",
            desugar_ribbon,
            {"opacity": opacity, "interpolate": interpolate, **mark_kwargs},
            placeholder="ribbon",
            position=position,
        )

    def mark_contour(
        self,
        *,
        bandwidth="scott",
        thresholds=6,
        smooth=True,
        fill=False,
        scheme=None,
        cmap=None,
        groupby=None,
        position=None,
        **mark_kwargs,
    ) -> "Chart":
        """Render a bivariate contour plot via ``Kde2D`` + ``Contour`` transforms.

        Estimates a 2-D kernel density and draws iso-contour lines (or filled
        contour regions when ``fill=True``).  Requires both ``x`` and ``y``
        encodings.

        Parameters
        ----------
        bandwidth : float or "scott" or "silverman", optional
            Bandwidth for the 2-D KDE.  Default is ``"scott"``.
        thresholds : int or list of float, optional
            Number of evenly-spaced contour levels, or an explicit list of
            density thresholds.  Default is ``6``.
        smooth : bool, optional
            Whether to smooth contour paths via Gaussian filtering before
            contouring.  Default is ``True``.
        fill : bool, optional
            Render filled contour regions instead of contour lines.  Default is
            ``False``.  When ``groupby`` is set, this parameter is ignored and
            isoline (segment) mode is always used.
        scheme : str or None, optional
            Colour map name applied to contour levels.  ``None`` (default) defers
            to the theme's sequential scheme.  Ignored when ``groupby`` is set.
        cmap : str or None, optional
            Back-compat alias for ``scheme``.  Pass at most one of the two.
        groupby : str or None, optional
            Group-key column (any groupby-supported dtype — string,
            integer, float, or boolean). When set,
            one 2-D KDE surface is computed per group and contour lines for
            each group are colored by the group field. Use
            ``.encode(color=<groupby>)`` to match legend colors to contours.
            Grouped contours use isoline (segment) mode to avoid cross-group
            polygon merging issues.
        position : Position, optional
            Position adjustment.
        **mark_kwargs
            Constant mark-style overrides (``opacity``, ``fill``, ``stroke``,
            ``stroke_width``, ``stroke_dash``, ``size``; aliases ``color``→``fill``,
            ``alpha``→``opacity``) forwarded to the polygon/path layers.

        Returns
        -------
        Chart
            New layered ``Chart`` configured for contour rendering.

        Examples
        --------
        >>> import ferrum as fm
        >>> import polars as pl
        >>> df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 2.0, 1.5], "y": [4.0, 5.0, 4.5, 3.0, 4.0]})
        >>> fm.Chart(df).mark_contour(thresholds=4, fill=True).encode(x="x", y="y")
        Chart(mark='polygon', encoding=['x', 'y'])
        """
        from ferrum.marks.base import resolve_method_bool
        from ferrum.marks.heavy_stat import desugar_contour
        from ferrum.marks._desugar_helpers import resolve_cmap_alias

        smooth = resolve_method_bool("mark_contour", "smooth", smooth, True)
        fill = resolve_method_bool("mark_contour", "fill", fill, False)
        return self._set_composite_mark(
            "contour",
            desugar_contour,
            {
                "bandwidth": bandwidth,
                "thresholds": thresholds,
                "smooth": smooth,
                "fill": fill,
                "cmap": resolve_cmap_alias(scheme=scheme, cmap=cmap, where="mark_contour"),
                "groupby": groupby,
                **mark_kwargs,
            },
            placeholder="polygon",
            position=position,
        )

    def mark_violin(
        self,
        *,
        bandwidth="scott",
        inner="box",
        shared_extent=False,
        orient=None,
        horizontal=False,
        position=None,
        **mark_kwargs,
    ) -> "Chart":
        """Render violin plots via the ``Violin`` transform.

        Estimates a mirrored KDE per group and overlays an optional inner
        summary (box, quartile lines, or individual points).  Requires a
        categorical ``x`` and continuous ``y`` encoding (or swapped when
        ``orient="horizontal"`` or ``horizontal=True``).

        Parameters
        ----------
        bandwidth : float or "scott" or "silverman", optional
            KDE bandwidth.  Default is ``"scott"``.
        inner : {"box", "quartile", "point", "none"} or None, default "box"
            Inner mark drawn on top of each violin:
            ``"box"`` -- IQR box + median rule,
            ``"quartile"`` -- three confined crossbars (q1/median/q3,
            q1/q3 dashed) sized to the violin's own body width,
            ``"point"`` -- individual data points (strip),
            ``"none"`` (or ``None``, kept for backward compatibility) --
            no inner mark. With a hue split (``color=`` distinct from the
            categorical axis), no ``inner`` variant dodges per hue -- see
            ``heavy_stat.py``'s ``_VIOLIN_QUARTILE_BAND_SIZE`` comment for
            the limitation this shares across variants. With ``inner``
            other than ``"none"``/``None`` under a hue split, the chart
            warns once each time it is resolved -- every ``to_json()`` and
            every render (``to_svg()``, ``to_png()``, ``show()``, ...) --
            naming what is dropped. ``catplot(kind="violin", hue=...)``
            voices this same warning under its own name, since it drives
            this method with its own fixed ``inner`` default and exposes no
            ``inner=`` of its own for the message to point at.
        shared_extent : bool, default False
            When ``True``, all groups share the same KDE evaluation range
            (cross-group global min/max), making the value axis directly
            comparable across groups.  When ``False`` (default), each group
            evaluates its KDE on its own per-group data range.
        orient : {"vertical", "horizontal"}, default "vertical"
            Canonical orientation control.  ``"horizontal"`` swaps the axes so
            the categorical grouping is on ``y`` and the value distribution is
            on ``x``.  Equivalent to ``horizontal=True``.  When both are given,
            ``orient`` wins.
        horizontal : bool, default False
            Legacy alias for ``orient="horizontal"``.  Kept for backward
            compatibility.
        position : Position, optional
            Position adjustment.
        **mark_kwargs
            Constant mark-style overrides (``opacity``, ``fill``, ``stroke``,
            ``stroke_width``, ``stroke_dash``, ``size``; aliases ``color``→``fill``,
            ``alpha``→``opacity``) forwarded to the violin polygon layer.

        Returns
        -------
        Chart
            New layered ``Chart`` configured for violin rendering.

        Examples
        --------
        >>> import ferrum as fm
        >>> import polars as pl
        >>> df = pl.DataFrame({"group": ["a"]*10 + ["b"]*10,
        ...                    "val": list(range(10)) + list(range(5, 15))})
        >>> fm.Chart(df).mark_violin(inner="quartile").encode(x="group", y="val")
        Chart(mark='polygon', encoding=['x', 'y'])
        """
        from ferrum.marks.heavy_stat import desugar_violin

        effective_horizontal = _normalize_orient("mark_violin", orient, horizontal)

        return self._set_composite_mark(
            "violin",
            desugar_violin,
            {
                "bandwidth": bandwidth,
                "inner": inner,
                "shared_extent": shared_extent,
                "horizontal": effective_horizontal,
                **mark_kwargs,
            },
            placeholder="polygon",
            position=position,
        )

    def mark_qq(
        self, *, distribution="normal", dequantize=False, line=True, position=None, **mark_kwargs
    ) -> "Chart":
        """Render a quantile-quantile plot.

        Computes theoretical vs. sample quantiles via the ``QQ`` transform.
        Reads the sample column from the ``x`` encoding; ``y`` is ignored.

        Parameters
        ----------
        distribution : {"normal", "uniform", "exponential", "lognormal"}, default "normal"
            Theoretical distribution to compare the sample against.
        dequantize : bool, optional
            Whether to apply rank-based dequantization before comparison.
            Default is ``False``.
        line : bool, optional
            Whether to overlay a 45-degree reference line.  Default is ``True``.
        position : Position, optional
            Position adjustment.
        **mark_kwargs
            Constant mark-style overrides (``opacity``, ``fill``, ``stroke``,
            ``stroke_width``, ``stroke_dash``, ``size``; aliases ``color``→``fill``,
            ``alpha``→``opacity``) forwarded to the scatter layer.

        Returns
        -------
        Chart
            New ``Chart`` configured for QQ rendering.

        Examples
        --------
        >>> import ferrum as fm
        >>> import polars as pl
        >>> df = pl.DataFrame({"val": [1.2, 0.8, 1.5, -0.5, 0.3, 2.1, -1.0, 0.6]})
        >>> fm.Chart(df).mark_qq(distribution="normal").encode(x="val")
        Chart(mark='point', encoding=['x'])
        """
        from ferrum.marks.base import resolve_method_bool
        from ferrum.marks.heavy_stat import desugar_qq

        line = resolve_method_bool("mark_qq", "line", line, True)
        dequantize = resolve_method_bool("mark_qq", "dequantize", dequantize, False)

        def _resolve_qq(x_field, y_field, **kw):
            # QQ is single-column: use x_field as the sample field. y_field ignored.
            if x_field is None:
                raise _unbound_input_refusal(
                    "mark_qq",
                    ("x",),
                    "the Q-Q plot reads",
                    own="mark_qq() requires .encode(x=...) to specify the sample field",
                )
            return desugar_qq(x_field, **kw)

        # Expose desugar_qq's signature for the transform/style kwarg split in
        # Chart._resolve_pending (inspect.signature follows __wrapped__).
        _resolve_qq.__wrapped__ = desugar_qq

        return self._set_composite_mark(
            "qq",
            _resolve_qq,
            {"distribution": distribution, "dequantize": dequantize, "line": line, **mark_kwargs},
            placeholder="point",
            position=position,
        )

    def mark_raster(
        self,
        *,
        aggregate="count",
        field=None,
        scheme=None,
        cmap=None,
        resolution="screen",
        blend=None,
        min_count=None,
        log_scale=None,
        position=None,
        **mark_kwargs,
    ) -> "Chart":
        """Render a 2-D raster (pixel heatmap) via the ``Raster`` transform.

        Bins data into a pixel grid and colours each pixel by an aggregate
        statistic.  Most useful for very large datasets where a scatter plot
        would overplot.  Requires ``x`` and ``y`` encodings.

        Parameters
        ----------
        aggregate : str, optional
            Aggregate function applied to each pixel bin, validated against
            the live ``transform_registry("raster")["methods"]`` vocabulary
            (spec §4.6); default ``"count"``. Aggregates in
            ``transform_registry("raster")["methods_needing_field"]``
            require ``field`` (the column to aggregate); the rest need none.
            Both the choice itself and the field requirement are validated
            eagerly, at this call, rather than deferred to render time
            (#136).
        field : str or None, optional
            Column name to aggregate.  Required when ``aggregate`` is in
            ``transform_registry("raster")["methods_needing_field"]``;
            ignored otherwise. Must be numeric -- integer columns are
            coerced to ``Float64`` automatically; a non-numeric column
            raises at render time.
        scheme : str or None, optional
            Colour map name.  ``None`` (default) defers to the theme's sequential
            scheme.
        cmap : str or None, optional
            Back-compat alias for ``scheme``.  Pass at most one of the two.
        resolution : "screen" or int, optional
            Pixel grid resolution.  ``"screen"`` (default) genuinely responds
            to the chart's actual dimensions — the grid scales up as the
            panel grows — using a pre-layout size estimate (it excludes
            axis/legend/facet-strip gutters, which aren't known until
            layout runs, so it's a close approximation, not the exact
            rendered plot area).  Pass an integer, or a ``(nx, ny)`` tuple,
            for an explicit grid size unaffected by chart dimensions.
        blend : str, optional
            Blend mode of the image layer, one of
            ``ferrum._core.blend_accepted_values()``; ``None`` (default)
            leaves it unset, which draws as ``"normal"`` (``"alpha"`` is the
            older name of ``"normal"``).  ``"additive"`` adds the image's
            colors to what is drawn beneath it (static SVG approximates it
            with ``mix-blend-mode:screen``).  Any other value raises
            ``ValueError`` at this call.
        min_count : int or None, optional
            Minimum count threshold; pixels below this are rendered transparent.
            ``None`` (default) shows all pixels.
        log_scale : bool or None, optional
            Whether the cells' color scale is symlog (``log1p``-compressed)
            rather than linear. The cells always keep their raw aggregate
            values, so a ``color=`` colorbar on the value field labels raw
            counts, densities or means either way. ``None`` (default) is
            symlog for ``"count"`` and ``"density"`` and linear for every
            other aggregate; ``True`` or ``False`` is honored on any
            aggregate. An explicit ``Color(scale=...)`` on the value field
            wins over it. Any other value raises ``ValueError`` at this call.
        position : Position, optional
            Position adjustment.
        **mark_kwargs
            Constant mark-style overrides (``opacity``, ``fill``, ``stroke``,
            ``stroke_width``, ``stroke_dash``, ``size``; aliases ``color``→``fill``,
            ``alpha``→``opacity``) forwarded to the image layer.

        Returns
        -------
        Chart
            New ``Chart`` configured for raster rendering.

        Raises
        ------
        ValueError
            If ``aggregate`` is not one of
            ``transform_registry("raster")["methods"]``, or if ``aggregate``
            is in ``transform_registry("raster")["methods_needing_field"]``
            and ``field`` is not given -- both checked eagerly at this call
            (#136) -- or if ``blend`` is outside its vocabulary.

        Examples
        --------
        >>> import ferrum as fm
        >>> import polars as pl
        >>> df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 2.5], "y": [4.0, 5.0, 4.5, 4.0]})
        >>> fm.Chart(df).mark_raster(cmap="plasma").encode(x="x", y="y")
        Chart(mark='image', encoding=['x', 'y'])
        """
        from ferrum._registry import (
            aggregate_needs_field,
            require_aggregate_field,
            validate_aggregate,
        )
        from ferrum.marks.heavy_stat import desugar_raster
        from ferrum.marks._desugar_helpers import resolve_cmap_alias

        # The aggregate/field pair and the blend mode are refused here, at the
        # call, not at render.
        validate_aggregate("raster", "mark_raster", aggregate)
        require_aggregate_field("raster", "mark_raster", aggregate, field)
        validate_optional_vocabulary("mark_raster", "blend", blend, blend_accepted_values())
        if log_scale is not None:
            log_scale = coerce_bool("mark_raster", "log_scale", log_scale)

        # #136 (close-remediation): when the Chart already carries data
        # (Chart(df).mark_raster(...), not a deferred-data chart), field=
        # is a real column reference for `aggregate` values that consume
        # it -- its existence and numeric-ness are answerable right now,
        # not just at .to_svg() render time when the Rust `stat_raster`
        # transform runs. An aggregate outside
        # transform_registry("raster")["methods_needing_field"] leaves
        # field= inert (unchanged, pre-existing behavior; see the
        # docstring), so this only fires for the same "needs field" set the
        # check above already gates on. Deferred-data charts (self._data is None,
        # e.g. Chart().mark_raster(...) before any data is attached) skip
        # gracefully -- there is nothing to check yet. `_mark_raster_field_schema`
        # answers the schema question only -- it must never materialize a
        # lazy source or let an untyped exception escape a declaration-time
        # method (close-remediation cycle 2, design-review S3): `None`
        # means "could not resolve a schema cheaply/safely", in which case
        # this eager check is skipped and render time surfaces whatever
        # error applies, exactly as it did before this check existed.
        if (
            field is not None
            and self._data is not None
            and aggregate_needs_field("raster", aggregate)
        ):
            schema = _mark_raster_field_schema(self._data)
            if schema is not None:
                if field not in schema:
                    raise ValueError(
                        f"mark_raster: field={field!r} is not a column in the data "
                        f"(columns: {list(schema)})."
                    )
                dtype = schema[field]
                if not dtype.is_numeric():
                    raise ValueError(
                        f"mark_raster: field={field!r} must be numeric to aggregate "
                        f"with aggregate={aggregate!r}; got dtype {dtype}."
                    )

        return self._set_composite_mark(
            "raster",
            desugar_raster,
            {
                "aggregate": aggregate,
                "field": field,
                "cmap": resolve_cmap_alias(scheme=scheme, cmap=cmap, where="mark_raster"),
                "resolution": resolution,
                "blend": blend,
                "min_count": min_count,
                "log_scale": log_scale,
                **mark_kwargs,
            },
            placeholder="image",
            position=position,
        )

    def mark_hex(
        self,
        *,
        bin_size=None,
        aggregate=None,
        field=None,
        scheme=None,
        cmap=None,
        stroke=None,
        stroke_width=0,
        position=None,
        **mark_kwargs,
    ) -> "Chart":
        """Render a hexagonal bin plot via the ``Hex`` transform.

        Bins data into a regular hexagonal grid and colours each cell by an
        aggregate statistic.  Requires ``x`` and ``y`` encodings.

        Parameters
        ----------
        bin_size : float or None, optional
            Hexagon radius in data units.  ``None`` (default) chooses a size
            automatically from the data range and chart dimensions.
        aggregate : str or None, optional
            Aggregate function, validated against the live
            ``transform_registry("hex")["methods"]`` vocabulary (spec §4.6).
            ``None`` (default, unset) counts the cells' rows, or takes the
            op a chart-level ``color=``'s own ``aggregate=`` names -- an
            equal explicit op is consumed the same way, and a differing one
            is refused (see ``color=`` below).
        field : str or None, optional
            Column to aggregate; required for every aggregate in
            ``transform_registry("hex")["methods_needing_field"]``. An
            ``.encode(color=...)`` naming a column supplies it when omitted.
            The cells are always coloured by the per-cell aggregate, titled
            with the aggregated column, or ``"Count"``. A chart-level
            ``color=`` must name the aggregated column (or Hex's ``value``
            output column); its title, scheme and legend then style the cell
            colours, its scheme acting as ``cmap=``. Any other ``color=``, or
            a scheme alongside a different ``cmap=``, is refused with
            ``ValueError``.
        scheme : str or None, optional
            Colour map name.  ``None`` (default) defers to the theme's sequential
            scheme.
        cmap : str or None, optional
            Back-compat alias for ``scheme``.  Pass at most one of the two.
        stroke : str or None, optional
            Hex border colour.  ``None`` (default) means no border.
        stroke_width : float, optional
            Hex border width in pixels.  Default is ``0``.
        position : Position, optional
            Position adjustment.
        **mark_kwargs
            Constant mark-style overrides forwarded to the polygon layer.
            Note: ``stroke`` and ``stroke_width`` are method parameters (cell borders);
            ``scheme`` (alias ``cmap``) is a transform parameter (fill colormap). Other
            style kwargs like ``opacity``, ``fill``, ``size`` override the polygon defaults.

        Returns
        -------
        Chart
            New ``Chart`` configured for hexagonal binning.

        Examples
        --------
        >>> import ferrum as fm
        >>> import polars as pl
        >>> df = pl.DataFrame({"x": [1.0, 2.0, 1.5, 3.0, 2.5], "y": [4.0, 5.0, 4.5, 3.0, 4.0]})
        >>> fm.Chart(df).mark_hex(bin_size=0.5).encode(x="x", y="y")
        Chart(mark='polygon', encoding=['x', 'y'])
        """
        from ferrum._registry import validate_aggregate
        from ferrum.marks.heavy_stat import desugar_hex
        from ferrum.marks._desugar_helpers import resolve_cmap_alias

        # The vocabulary is refused at the call, for an explicit aggregate
        # only -- None is unset, resolved later from a chart-level color's
        # own aggregate= (`_prep_hex_field_from_color`), or left as the
        # cells' row count; the needs-field rule also waits for resolution,
        # since `.encode(color=...)` may still supply `field`.
        if aggregate is not None:
            validate_aggregate("hex", "mark_hex", aggregate)
        return self._set_composite_mark(
            "hex",
            desugar_hex,
            {
                "bin_size": bin_size,
                "aggregate": aggregate,
                "field": field,
                "cmap": resolve_cmap_alias(scheme=scheme, cmap=cmap, where="mark_hex"),
                "stroke": stroke,
                "stroke_width": stroke_width,
                **mark_kwargs,
            },
            placeholder="polygon",
            position=position,
        )

    def mark_swarm(
        self,
        *,
        size=4,
        orient=None,
        horizontal=None,
        spacing=1.0,
        side="both",
        dodge=None,
        position=None,
        **mark_kwargs,
    ) -> "Chart":
        """Render a beeswarm (strip swarm) plot via the ``Swarm`` transform.

        Computes non-overlapping point positions along the categorical axis
        using a deterministic placement algorithm seeded for reproducibility.
        Requires a continuous ``y`` (or ``x``) and an optional categorical
        grouping axis.

        Parameters
        ----------
        size : float or None, optional
            Point diameter in pixels: both the drawn dot's diameter and
            the placement diameter.  Default is ``4``; ``None`` is unset
            and draws at that default.  This is a diameter,
            unlike ``mark_point(size=)``, which is an area in square
            pixels; ``fm.Swarm(point_size=)`` is the same diameter, with
            the same default of ``4`` for a hand-built swarm.  Placement is
            computed at render time against the panel's resolved
            value-axis scale, so the rendered centre-to-centre distance
            between adjacent placed dots is exactly ``size + spacing`` px
            — a title, a facet strip, an explicit domain, a log scale, or
            a dodge band all reshape the panel without loosening this. Any
            other value must be a positive finite real number; ``0``, a
            negative number, ``NaN``/``inf``, a ``bool``, a ``str`` or any
            other non-number refuses as ``mark_swarm(size=...)``.
        orient : {"vertical", "horizontal"} or None, optional
            Canonical orientation control.  ``"vertical"`` (default) spreads
            points along x; ``"horizontal"`` spreads along y.  When both
            ``orient`` and ``horizontal`` are given, ``orient`` wins.
        horizontal : bool or None, optional
            Legacy alias for ``orient``.  ``True`` is equivalent to
            ``orient="horizontal"``.  Ignored when ``orient`` is also given;
            prefer ``orient`` in new code.
        spacing : float, optional
            Additive pixel gap between point edges.  Default is ``1.0`` (1px):
            the rendered centre-to-centre distance between adjacent placed
            dots is ``size + spacing``.
        side : {"both", "left", "right"}, default "both"
            Side of the axis on which points can be placed.
        dodge : str or None, optional
            Column name to use as a secondary grouping variable for dodging.
            ``None`` (default) means no dodging.
        position : Position, optional
            Position adjustment.
        **mark_kwargs
            Constant mark-style overrides (``opacity``, ``fill``, ``stroke``,
            ``stroke_width``, ``stroke_dash``; aliases ``color``→``fill``,
            ``alpha``→``opacity``) forwarded to the point layer.  ``size``
            is a named parameter above, not one of these.

        Returns
        -------
        Chart
            New ``Chart`` configured for beeswarm rendering.

        Examples
        --------
        >>> import ferrum as fm
        >>> import polars as pl
        >>> df = pl.DataFrame({"group": ["a"]*8 + ["b"]*8, "val": list(range(16))})
        >>> fm.Chart(df).mark_swarm(size=6).encode(x="group", y="val")
        Chart(mark='point', encoding=['x', 'y'])
        """
        from ferrum.marks.heavy_stat import desugar_swarm

        # Normalize: orient= is canonical; horizontal= is a legacy alias.
        # When orient is not None it wins unconditionally (consistent with
        # boxplot/boxen/violin which call _normalize_orient).  When orient is
        # None, fall back to horizontal (or default to "vertical").
        effective_horizontal = _normalize_orient("mark_swarm", orient, horizontal or False)
        effective_orient = "horizontal" if effective_horizontal else "vertical"

        return self._set_composite_mark(
            "swarm",
            desugar_swarm,
            {
                "size": size,
                "orient": effective_orient,
                "spacing": spacing,
                "side": side,
                "dodge": dodge,
                **mark_kwargs,
            },
            placeholder="point",
            position=position,
        )

    def mark_function(
        self, fn, *, domain=None, n=200, clip=True, position=None, **mark_kwargs
    ) -> "Chart":
        """Render a mathematical function as a line.

        Evaluates ``fn(xs)`` on ``n`` evenly-spaced ``x`` values in the
        specified domain and renders the result as a line. A chart that
        already draws something over its own data -- a primitive mark
        (``mark_point()`` etc.) with data, or an existing layered chart
        (``a + b``) -- keeps every one of its existing layers, each with its
        own data and style, and the function line joins them as one more
        layer. A bare chart (no mark set yet), and a primitive mark with no
        data of its own (``Chart(None).mark_line(stroke_width=7)``, a
        style-only placeholder with nothing to keep as a layer), both keep
        the single-layer shape: the chart's data becomes the synthetic
        ``(x, y)`` table, and any style the placeholder mark carried merges
        onto the function line's. A chart with a pending composite mark
        (``mark_histogram()`` etc.) refuses; compose the two with ``+``
        instead.

        Parameters
        ----------
        fn : callable
            A function accepting a 1-D NumPy array of x values and returning a
            1-D array of y values.
        domain : list of float or None, optional
            ``[x_min, x_max]`` evaluation range.  ``None`` (default) infers
            the range from the parent chart's ``x`` column if available,
            otherwise raises ``ValueError``.
        n : int, optional
            Number of evaluation points.  Default is ``200``.
        clip : bool, optional
            Whether the rendered line is clipped to the panel's plot area.
            Default is ``True`` (clipped, matching every other mark).
            ``False`` lets the curve render past the panel bounds wherever
            its evaluated geometry extends there (e.g. a steep tail near
            the domain edge) -- ``fn`` is still evaluated over the same
            resolved x-domain either way; ``clip`` only changes whether the
            resulting geometry is confined at render time.
        position : Position, optional
            Position adjustment.
        **mark_kwargs
            Mark-style overrides for the function line only (e.g. ``stroke``,
            ``stroke_width``, ``opacity``, or ``size`` for a dot-diameter
            width), validated like any composite mark's style kwargs. A prior
            layer this call keeps draws with its own style, never this one.

        Returns
        -------
        Chart
            A chart with the function line as one more layer when something
            was already drawn; otherwise a new ``Chart`` whose data is the
            synthetic ``(x, y)`` table.

        Raises
        ------
        ValueError
            When ``domain`` is not provided and no parent ``x`` data can be
            inferred, when ``clip`` is not ``True`` or ``False``, when a style
            kwarg is not one the line honors, when ``size`` is not a finite
            number above zero, or when this chart has a pending composite
            mark (use ``+`` instead).
        ValueError
            If x or y carries a ``bin=``/``aggregate=`` and no data layer
            is drawn before this call: ``fn`` is drawn over its own grid,
            so no layer reads the field. Draw the data with a mark first
            (``chart.mark_point().encode(...).mark_function(fn)``), which
            honors it.

        Examples
        --------
        >>> import ferrum as fm
        >>> import numpy as np
        >>> import polars as pl
        >>> fm.Chart(None).mark_function(np.sin, domain=[0, 6.28], n=100)
        Chart(mark='line', encoding=['x', 'y'])
        """
        _refuse_fitted_mark_after_pending_composite("mark_function", self._pending_stat_mark)
        if position is not None:
            from ferrum.position import validate_position_eligibility

            validate_position_eligibility("function", position)
        from ferrum.marks.base import (
            MarkBase,
            refuse_style_no_layer_draws,
            resolve_method_bool,
        )
        from ferrum.marks.heavy_stat import desugar_function

        clip = resolve_method_bool("mark_function", "clip", clip, True)
        style = MarkBase("function", **mark_kwargs).to_mark_kwargs_dict()
        refuse_style_no_layer_draws("mark_function", style, ["line"])

        # Whether this call keeps an existing layer (an existing layered
        # chart, or a primitive mark over real data) beside the function
        # line, or draws alone over fn's own synthetic grid. Only the
        # alone case reads no caller field from x/y, so only it is
        # checked for a bin=/aggregate= that would otherwise be dropped
        # silently once the channel is rebound to that grid.
        has_prior_layer = (self._layers is not None and self._layers) or (
            self._mark is not None and self._data is not None
        )
        if not has_prior_layer:
            _refuse_function_alone_statistics(self._encoding)

        # Infer parent x data for domain resolution.
        # For multi-layer charts, try the first layer's data; for single-chart,
        # use self._data.
        parent_x_data = None
        x_enc = self._encoding.get("x")
        data_source = self._data
        if data_source is None and self._layers:
            # Extract data from the first non-function layer.
            for existing_layer in self._layers:
                if hasattr(existing_layer, "_data") and existing_layer._data is not None:  # type: ignore[attr-defined]
                    data_source = existing_layer._data  # type: ignore[attr-defined]
                    break
        if x_enc is not None and data_source is not None:
            try:
                from ferrum._coerce import to_arrow_table

                x_field_name = x_enc.field if isinstance(x_enc, ChannelBase) else x_enc
                tbl = to_arrow_table(data_source)
                if x_field_name in tbl.column_names:
                    parent_x_data = tbl[x_field_name]
            except (ImportError, TypeError, ValueError, KeyError):
                pass
        elif x_enc is None and data_source is not None and domain is None:
            # Encoding not set yet (mark_function called before .encode()).
            # Infer domain from the first numeric column in the chart's data.
            try:
                import pyarrow as pa
                from ferrum._coerce import to_arrow_table

                tbl = to_arrow_table(data_source)
                for _col_name in tbl.column_names:
                    _col = tbl[_col_name]
                    if pa.types.is_floating(_col.type) or pa.types.is_integer(_col.type):
                        parent_x_data = _col
                        break
            except (ImportError, TypeError, ValueError, KeyError):
                pass

        result = desugar_function(
            fn,
            parent_chart_x_data=parent_x_data,
            domain=domain,
            n=n,
            clip=clip,
        )
        mark = result.mark
        transforms = result.transforms
        remap = result.remap
        synthetic = result.data
        # desugar_function computes the clip escape itself (mark_kwargs is
        # {"clip": False} when clip=False, else None -- see its docstring);
        # this mixin just applies whatever the desugar decided, the same way
        # every other composite mark's style dict is applied to its layers.
        line_mark_kwargs = {**style, **(result.mark_kwargs or {})}
        if has_prior_layer:
            # Something is already drawn -- an existing layered chart, or a
            # primitive mark over real data (the refusal above guarantees it
            # cannot be a composite's placeholder here). A primitive mark
            # with no data of its own (e.g. a bare `mark_line()` on
            # `Chart(None)`, used only to carry inherited style) has nothing
            # to keep as its own layer, so it still takes the single-chart
            # shortcut below. Build a fresh single-chart with the function
            # data and compose via + so the function line joins what is
            # already drawn as a proper layer, instead of overwriting it.
            fn_chart = self.__class__(synthetic)
            fn_chart._mark = mark
            fn_chart._entry = "mark_function"
            if line_mark_kwargs:
                fn_chart._mark_kwargs = merge_style(fn_chart._mark_kwargs, line_mark_kwargs)
            if remap:
                from ferrum._desugar import _apply_remap

                _apply_remap(
                    fn_chart._encoding,
                    remap,
                    orig_encoding=self._encoding,
                    titles=result.remap_titles,
                )
            fn_chart._position = position
            return self + fn_chart

        new = self._clone()
        new._mark = mark
        new._entry = "mark_function"
        new._data = synthetic
        new._transforms = list(self._transforms) + list(transforms)
        if line_mark_kwargs:
            new._mark_kwargs = merge_style(new._mark_kwargs, line_mark_kwargs)
        if remap:
            from ferrum._desugar import _apply_remap

            _apply_remap(
                new._encoding, remap, orig_encoding=self._encoding, titles=result.remap_titles
            )
        new._position = position
        return new
