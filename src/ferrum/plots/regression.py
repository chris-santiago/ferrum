"""Regression domain figure functions and their private builders.

Public API
----------
lmplot, regplot, residplot, residuals_chart, prediction_error_chart,
cooks_distance_chart.

Each public function wraps ``_resolve_source`` (shared across all figure
functions) and dispatches to a co-located ``_*_from_source`` builder that
produces a fully-formed ``Chart``.

Internal-only builders
----------------------
_residuals_chart_from_source, _residuals_panel,
_prediction_error_chart_from_source.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any

import polars as pl

from ferrum import (
    Chart,
    Glm,
    Jitter,
    Logistic,
    OrdinalScale,
    Robust,
    Smooth,
)
from ferrum._coerce import to_polars
from ferrum._compose import _PostBuildChannel, _PostBuildPositional
from ferrum._core import transform_registry, x_estimator_accepted_values
from ferrum._desugar import _propagate_layer_axis
from ferrum._encode_lowering import (
    COMPOSITE_GROUPING_CHANNELS,
    _FIT_CATEGORICAL_READS,
    _rebind_positional,
    _refuse_bin_aggregate_fit,
    _refuse_categorical_input,
    _refuse_residplot_another_variable,
    _refuse_residplot_bin_aggregate,
    _refuse_unreadable_positional_options,
    _residual_axis_categorical_reads,
)
from ferrum._layer import _Layer
from ferrum._registry import registry_output_columns
from ferrum._validate import validate_choice, validate_optional_choice
from ferrum.diagnostics.source import ComparedModelSource
from ferrum.encoding import Color, X, Y
from ferrum.encoding._channel_policy import _FACET_CHANNELS
from ferrum.encoding._resolve import channel_for_value
from ferrum.encoding.base import ChannelBase, declared_title, drawn_title
from ferrum.marks._desugar_helpers import _unique_col_name, with_grouping
from ferrum.marks._fit_layers import build_fit_layers
from ferrum.marks.diagnostic._regression import COOK_OUTLIER_STYLE
from ferrum.marks.statistical import desugar_smooth
from ferrum.plots._helpers import (
    _color_field_for,
    _column_or_literal_color,
    _compose_compare,
    _cook_outlier_rows,
    _drawn_bindings,
    _drawn_positional_channels,
    _encode_grouping,
    _figure_entry,
    _finalize_chart,
    _grid_panels,
    _grouping_hue,
    _inject_metrics_corner,
    _log_scaled_x,
    _merge_layers,
    _overlay_metrics_corner,
    _resolve_source,
    _style_kws,
)

if TYPE_CHECKING:
    from ferrum import ConcatChart, HConcatChart, VConcatChart


# `_SMOOTH_ROUTED_METHODS` is DERIVED from
# `transform_registry("smooth")["methods"]`, not hand-listed: every registry
# smooth spelling routes through `mark_smooth` below except two.
# "linear" is the same method as "lm" -- a second registry alias `lmplot`
# does not separately expose. "glm" is a name both sets could plausibly
# claim, and `lmplot` keeps it mapped to the standalone `Glm` transform
# (with its own `family=`/`link=` surface, matching the top-level
# `glm()`/`Glm` API) rather than `Smooth(method="glm")`'s grid-only one;
# both share the hoisted `wald_band` helper in Rust, so this is a choice of
# which surface owns the name, not a divergent implementation.
# `_DEDICATED_FIT_METHODS` are the three methods with their own fit-family
# transform (`Logistic`/`Glm`/`Robust`), all sharing the
# `x, y, ci_lower, ci_upper` registry output schema (spec §4.7/M6c).
# `test_lmplot_valid_methods_registry_pin.py` pins both halves against the
# registry, so a rename or a new registry smooth method is caught rather
# than silently under/over-widening `_VALID_METHODS`.
_SMOOTH_ROUTED_METHODS = frozenset(transform_registry("smooth")["methods"]) - {"linear", "glm"}
_DEDICATED_FIT_METHODS = frozenset({"logistic", "glm", "robust"})
_VALID_METHODS = _SMOOTH_ROUTED_METHODS | _DEDICATED_FIT_METHODS

# `truncate=False`'s margin, as a fraction of the predictor's own data span
# (seaborn's typical auto-scale margin). The only signal this function
# passes: Smooth/Robust derive the range themselves, from the fit's own
# whole-batch extent, and pad it in whatever space the fit uses -- data
# units on a linear x, log units on a demanded log x -- so the pad never
# pushes a log x's evaluation grid into territory the axis cannot place.
_TRUNCATE_FALSE_X_RANGE_PAD = 0.05


def _smooth_degree_kwargs(method: str, order: int | None, func_name: str = "lmplot") -> dict:
    """``degree=`` kwargs to merge into a ``mark_smooth(method=...)`` call.

    ONE forwarding rule for every ``_SMOOTH_ROUTED_METHODS`` member: forward
    ``order`` as ``degree=`` whenever the caller passed an explicit value
    (``order is not None``), regardless of which method it lands on, so a
    bare ``lmplot(method=<any>)`` call (``order`` left at its ``None``
    default) keeps using that method's own default degree concept
    untouched. ``Smooth``'s own degree-acceptance matrix (``smooth.rs``)
    decides what a forwarded value means for that specific method -- there
    is no per-method spelling here:

    - ``"lm"``: any ``degree >= 1`` is accepted (``>= 2`` promotes to a
      polynomial fit); ``degree=0`` raises. ``order=1`` explicit is a no-op
      against the default (``Smooth``'s own ``degree=None`` and explicit
      ``degree=1`` resolve identically for ``method="lm"``).
    - ``"quadratic"``/``"cubic"``: a degree matching the method's own fixed
      degree (``2``/``3``) is accepted as a no-op; any other value --
      including ``1`` -- raises, naming ``method="lm", degree=<order>`` as
      the way to get that fit.
    - ``"loess"``: ``degree`` is the LOCAL-FIT degree (``1`` or ``2``, not a
      polynomial order). ``order=1`` explicit is therefore a REAL,
      distinguishable request for the linear local fit -- not a no-op --
      and ``order=2`` explicit lands on loess's own default local-fit degree
      (a harmless no-op); ``order`` outside ``{1, 2}`` raises the same
      "LOESS degree must be 1 or 2" error a direct ``mark_smooth(
      method="loess", degree=<n>)`` call raises.
    - ``"log"``/``"sqrt"``/``"gam"``: no degree concept at all -- any
      forwarded value, including ``1``, raises "degree is not valid for
      method=...".

    Raises ``ValueError`` directly (rather than letting a negative value
    reach the Rust boundary) when ``order`` is a negative int: the Rust
    ``degree`` parameter is unsigned, so a negative Python int fails the
    PyO3 conversion with an untyped ``OverflowError`` instead of a
    diagnosable message.

    The "no explicit degree" sentinel is ``order is None``, never a
    particular value: ``order=1`` has to stay distinguishable from an
    omitted ``order``, because for ``"loess"`` the linear local fit is a
    real, different fit from its degree-2 default rather than a no-op.
    """
    if order is None:
        return {}
    if order < 0:
        raise ValueError(
            f"{func_name}(method={method!r}, order={order!r}) is not valid -- "
            "order must be a non-negative integer; a negative polynomial "
            "degree has no meaning for a smooth fit."
        )
    return {"degree": order}


# ---------------------------------------------------------------------------
# Private builders
# ---------------------------------------------------------------------------


def _resolve_residuals_panels(panels: Any) -> list[str] | None:
    """Expand ``panels=`` into what ``_residuals_chart_from_source`` expects.

    Shared by ``residuals_chart`` and ``ResidualsVisualizer`` so
    ``"auto"`` always expands to the canonical 4-panel list ---
    ``residuals_vs_fitted``, ``qq``, ``scale_location``,
    ``residuals_vs_leverage``. The builder's own fallback for an
    unrecognized value (anything that is not ``None``/``"single"`` and
    not already a list) is single-panel, so forwarding ``"auto"`` to it
    raw would silently render one panel instead of four -- the trap this
    closes (batch G T9).
    """
    if panels in (None, "single"):
        return None
    if panels == "auto":
        return [
            "residuals_vs_fitted",
            "qq",
            "scale_location",
            "residuals_vs_leverage",
        ]
    return list(panels)


def _residuals_chart_from_source(
    source: Any,
    *,
    kind: str = "studentized",
    cook_threshold: float | str | None = None,
    panels: Any = None,  # None / "single" / list of panel names
    annotate_metrics: bool = True,
    subtitle: str | None = None,
    mark: dict | None = None,
    encode: dict | None = None,
    properties: dict | None = None,
    layers: list | None = None,
    theme: Any = None,
):
    """Build a residuals diagnostic chart from a ModelSource.

    Schwabish SB3 (2026-05-11): the corner R^2/RMSE/MAE annotation rides
    on the residuals-vs-fitted view in both single-panel and 4-panel
    layouts. ``_inject_metrics_corner`` augments the shared DataFrame
    once at the top of this function; the single-panel chart and the
    ``residuals_vs_fitted`` cell of the 4-panel grid then call
    ``_overlay_metrics_corner`` which detects the injected columns and
    adds a same-data ``mark_text`` layer. The other 4-panel cells (QQ,
    scale-location, residuals-vs-leverage) inherit the extra columns but
    do not reference them, so their renderers are unaffected.

    ``cook_threshold`` attaches the Cook's-distance outlier overlay
    (:func:`ferrum.plots._helpers._cook_outlier_rows`) as its own layer,
    reading only the outlier rows -- never a null-padded share of the
    base scatter's own rows, or vice versa.
    """
    import ferrum

    df = source.predictions()
    # The corner R²/RMSE/MAE metrics conflate models (a single R²/RMSE over all
    # rows, anchored at one ``y_pred.arg_max()`` row), so skip the injection when
    # comparing. This mirrors how classification gates its active title to the
    # single-model case.
    is_single_model = "model" not in df.columns
    if annotate_metrics and is_single_model:
        df = _inject_metrics_corner(df, kind=kind)

    if panels in (None, "single"):
        y_col = "studentized_residual" if kind in ("studentized", "scaled") else "residual"
        # When a ``model`` column is present (compare-source path), drive
        # per-model colour on the residual scatter; absent it, ``None`` is
        # byte-identical to the single-model default.
        chart = ferrum.Chart(df).mark_residuals(
            kind=kind,
            color_field=_color_field_for(df, None),
        )
        if cook_threshold is not None:
            rows = _cook_outlier_rows(df, kind=kind, threshold=cook_threshold)
            chart = chart.layer(
                ferrum.Layer(
                    data=rows,
                    mark="point",
                    name="outlier",
                    encoding={"x": "y_pred", "y": y_col},
                    mark_kwargs=dict(COOK_OUTLIER_STYLE),
                )
            )
        chart = chart.properties(
            title=ferrum.Title("Residuals", subtitle=subtitle),
        )
        # ``_overlay_metrics_corner`` is a no-op when the injected
        # columns are absent, so the call is unconditional.
        chart = _overlay_metrics_corner(chart)
        return _finalize_chart(
            chart,
            entry="residuals_chart",
            mark=mark,
            encode=encode,
            properties=properties,
            layers=layers,
            theme=theme,
        )

    # Multi-panel path: each panel injects its own outlier columns with
    # the right x-axis encoding for that panel's coordinate system. The
    # leverage panel depends on hat-matrix quantities; for non-linear
    # estimators ModelSource.predictions emits NaN for every leverage row,
    # in which case we drop the panel so the remaining panels still render.
    # If the drop would leave nothing to render (leverage was the only
    # panel requested), we raise instead (GH #44).
    panel_list = panels if isinstance(panels, list) else ["residuals_vs_fitted"]
    if "residuals_vs_leverage" in panel_list and df["leverage"].is_nan().all():
        panel_list = [p for p in panel_list if p != "residuals_vs_leverage"]
        if not panel_list:
            model = getattr(source, "model", None)
            estimator_clause = f" Got {type(model).__name__!r}." if model is not None else ""
            raise ValueError(
                "Cook's distance and residuals-vs-leverage are hat-matrix "
                "quantities that require a linear estimator exposing "
                f"`coef_`.{estimator_clause} Pass a linear estimator (e.g. "
                "LinearRegression, Ridge, Lasso), or drop "
                "residuals_vs_leverage from panels= and use the other "
                "diagnostic panels instead."
            )
    charts = [
        _residuals_panel(df, name, kind=kind, cook_threshold=cook_threshold) for name in panel_list
    ]
    # _grid_panels applies theme internally; apply overrides before theme.
    chart = _grid_panels(charts)
    return _finalize_chart(
        chart,
        entry="residuals_chart",
        mark=mark,
        encode=encode,
        properties=properties,
        layers=layers,
        theme=theme,
    )


def _residuals_panel(
    df: pl.DataFrame,
    name: str,
    *,
    kind: str = "studentized",
    cook_threshold: float | str | None = None,
):
    """One sub-panel of a multi-panel residuals chart.

    When ``cook_threshold`` is set, the residuals_vs_fitted and
    residuals_vs_leverage panels overlay an outlier-highlight layer,
    reading only the outlier rows (:func:`ferrum.plots._helpers._cook_outlier_rows`)
    as its own layer, keyed on each panel's own x axis (y_pred for
    residuals_vs_fitted, leverage for residuals_vs_leverage).
    """
    import ferrum

    y_col = "studentized_residual" if kind in ("studentized", "scaled") else "residual"

    if name == "residuals_vs_fitted":
        chart = ferrum.Chart(df).mark_residuals(kind=kind)
        chart = chart.encode(
            x=X("y_pred", title="Fitted values"),
            y=Y(y_col, title="Studentized residual"),
        )
        if cook_threshold is not None:
            rows = _cook_outlier_rows(df, kind=kind, threshold=cook_threshold)
            chart = chart.layer(
                ferrum.Layer(
                    data=rows,
                    mark="point",
                    name="outlier",
                    encoding={"x": "y_pred", "y": y_col},
                    mark_kwargs=dict(COOK_OUTLIER_STYLE),
                )
            )
        chart = chart.properties(title=ferrum.Title("Residuals vs Fitted"))
        # Schwabish SB3: detect the ``_metrics_text``/``_metrics_y``
        # columns injected upstream by ``_inject_metrics_corner`` and
        # overlay the corner annotation. No-op when the columns are
        # absent, so an externally-built panel (e.g. a user invoking
        # ``_residuals_panel`` directly without metrics injection)
        # renders identically to the pre-SB3 behavior.
        return _overlay_metrics_corner(chart)

    if name == "qq":
        # mark_qq draws its own sample quantiles on y and refuses a
        # field-bearing y= outright (a column name is not a provenance), so
        # this panel styles the axis with a fieldless Y(title=...) rather
        # than binding the QQ transform's own output column by name.
        return (
            ferrum.Chart(df)
            .mark_qq()
            .encode(
                x=X("studentized_residual", title="Theoretical quantiles"),
                y=Y(title="Sample quantiles"),
            )
            .properties(title=ferrum.Title("Normal Q–Q"))
        )

    if name == "scale_location":
        d2 = df.with_columns(pl.col("studentized_residual").abs().sqrt().alias("sqrt_abs_resid"))
        return (
            ferrum.Chart(d2)
            .mark_point()
            .encode(
                x=X("y_pred", title="Fitted values"),
                y=Y("sqrt_abs_resid", title="√|Studentized residual|"),
            )
            .properties(title=ferrum.Title("Scale–Location"))
        )

    if name == "residuals_vs_leverage":
        base = (
            ferrum.Chart(df)
            .mark_point()
            .encode(
                x=X("leverage", title="Leverage"),
                y=Y(y_col, title="Studentized residual"),
            )
            .properties(title=ferrum.Title("Residuals vs Leverage"))
        )
        if cook_threshold is not None:
            rows = _cook_outlier_rows(df, kind=kind, threshold=cook_threshold, x_col="leverage")
            overlay = (
                ferrum.Chart(rows).mark_point(**COOK_OUTLIER_STYLE).encode(x="leverage", y=y_col)
            )
            return base + overlay
        return base

    raise ValueError(f"unknown residuals panel: {name!r}")


def _prediction_error_chart_from_source(
    source: Any,
    *,
    reference_line: bool = True,
    ci: float | None = None,
    reference_band: bool = False,
    mark: dict | None = None,
    encode: dict | None = None,
    properties: dict | None = None,
    layers: list | None = None,
    theme: Any = None,
):
    """Build an actual-vs-predicted error chart from a ModelSource.

    When ``ci`` or ``reference_band`` is set, computes a residual band
    and injects ``_pe_band_lo`` / ``_pe_band_hi`` columns:

    - ``ci`` (float in (0, 1)): band spans the central ``ci`` fraction
      of residuals -- ``y_true + quantile_low`` to ``y_true +
      quantile_high`` where the quantiles bracket ``(1 - ci) / 2``.
    - ``reference_band=True`` (without ``ci``): band spans +/-1 RMSE
      around the identity line.
    """
    import ferrum
    from ferrum.diagnostics._internal.precomputed import _PrecomputedSource

    # #208: predictions() is a shared DiagnosticSource protocol method
    # (ModelSource and _PrecomputedSource both implement it), but only
    # _PrecomputedSource's y_pred shape can be malformed -- pass this
    # builder's own public name so a bad shape refuses as
    # "prediction_error_chart requires 1-D fitted values...", not
    # predictions()'s other caller's name (residuals_chart). Mirrors the
    # isinstance(source, _PrecomputedSource) idiom already used in
    # classification.py for the same "adapter method, two public callers"
    # shape.
    if isinstance(source, _PrecomputedSource):
        df = source.predictions(diagnostic="prediction_error_chart").sort("y_true")
    else:
        df = source.predictions().sort("y_true")
    # Internal invariant: the residual band (``ci`` / ``reference_band``) is a
    # single-model aggregate, so this builder must be handed a single-model
    # source when a band is requested. ``prediction_error_chart`` enforces this
    # by routing the multi-model band case through ``_compose_compare`` (one
    # single-model panel per model). A pooled ``model`` frame reaching here with
    # a band set is therefore a routing bug, not user error — fail loudly rather
    # than conflate residuals across models.
    if "model" in df.columns and (ci is not None or reference_band):
        raise ValueError(
            "internal: _prediction_error_chart_from_source received a pooled "
            "multi-model frame with ci=/reference_band= set; the multi-model "
            "band path must be routed through _compose_compare (one single-model "
            "panel per model) by prediction_error_chart."
        )
    if ci is not None or reference_band:
        residuals = df["y_pred"] - df["y_true"]
        if ci is not None:
            if not 0.0 < ci < 1.0:
                raise ValueError(f"mark_prediction_error(ci={ci!r}) — expected a value in (0, 1).")
            alpha = (1.0 - float(ci)) / 2.0
            q_lo = float(residuals.quantile(alpha, interpolation="linear"))
            q_hi = float(residuals.quantile(1.0 - alpha, interpolation="linear"))
        else:
            sigma = float((residuals**2).mean() ** 0.5)
            q_lo = -sigma
            q_hi = sigma
        df = df.with_columns(
            (pl.col("y_true") + q_lo).alias("_pe_band_lo"),
            (pl.col("y_true") + q_hi).alias("_pe_band_hi"),
        )
    # When a ``model`` column is present (compare-source path), drive per-model
    # colour on the scatter; absent it, ``None`` is byte-identical to the
    # single-model default.
    chart = ferrum.Chart(df).mark_prediction_error(
        reference_line=reference_line,
        ci=ci,
        reference_band=reference_band,
        color_field=_color_field_for(df, None),
    )
    chart = chart.encode(
        x=X("y_pred", title="Predicted value"),
        y=Y("y_true", title="True value"),
    )
    chart = chart.properties(title=ferrum.Title("Prediction Error"))
    return _finalize_chart(
        chart,
        entry="prediction_error_chart",
        mark=mark,
        encode=encode,
        properties=properties,
        layers=layers,
        theme=theme,
    )


def _required_channel(value: Any, *, entry: str, param: str) -> ChannelBase:
    """Parse ``x=``/``y=`` for a figure parameter that has no meaning without a column.

    *value* is read exactly as ``encode()`` reads a channel value
    (:func:`~ferrum.encoding._resolve.channel_for_value`): a shorthand
    string (``"bill:Q"``) is parsed into the channel it spells, an
    already-built encoding object passes through untouched, and a value
    that is none of str/channel/``Repeat`` placeholder raises the parse's
    own ``TypeError`` naming *entry* and *param*. ``None`` refuses BEFORE
    the parse (``channel_for_value`` has no "unset" reading of its own):
    ``x``/``y`` are not optional, and their column names are what the fit
    layers title the axes with -- those layers are bound to the
    TRANSFORM's own output columns, so a missing name would retitle the
    value axis with one of those instead.

    Parsing into a CHANNEL, not just a field name, is what lets a
    caller's ``title=``/``scale=``/declared type survive onto the fit
    layers and the scatter alike, and what makes ``x="bill:Q"`` mean the
    same thing here that it means on ``Chart.encode``. Both regression
    entries (``_lmplot_impl`` and ``residplot``) call this once, at the
    top, and every consumer of ``x``/``y`` after that -- the refusals, the
    jitter column, the scatter's encoding, the fit request, the axis
    titles -- reads the one parsed channel it returns, never the raw
    keyword.

    *entry* is the public function the caller invoked and *param* the
    parameter being narrowed, so the refusal names what they wrote.
    ``regplot`` passes ``"regplot"``: it runs ``lmplot``'s body directly
    (:func:`_lmplot_impl`) and speaks as itself, which is how its other
    refusals already read (``truncate=False``, ``x_bins=`` on a dedicated
    method).
    """
    if value is None:
        raise ValueError(
            f"{entry}({param}=None): {param} names the column to fit and has no "
            f"default. Pass a column name, or an encoding object wrapping one."
        )
    return channel_for_value(param, value, declare=f"{entry}({param}=...)")


def _fit_chart_from_named_transform(
    data: Any,
    transform: Any,
    *,
    kind: str,
    ci: float | None,
    line_kwargs: dict,
    chart_x: Any,
    chart_y: Any,
    x_title: str,
    y_title: str,
    metrics: bool = False,
) -> Chart:
    """Build a layered ``Chart`` around a named fit-family transform.

    The layers are ``marks/_fit_layers.build_fit_layers``' -- the one
    ribbon+line(+metrics) shape of the fit family, shared with
    ``desugar_smooth`` so the CI band's opacity and the metrics offsets
    cannot drift between the two paths. This function's own job is the chart:
    the ``logistic``/``glm``/``robust`` transforms are constructed directly by
    ``lmplot`` rather than through ``mark_smooth``'s ``_PendingMark`` desugar,
    so the chart has to be assembled here.

    ``kind`` doubles as the transform's own ``name=`` (every call site below
    passes the same string for both), hence one parameter for the registry
    kind and the layers' ``data_source``.

    ``lmplot``'s ``mark_smooth``-routed methods use this too, whenever a
    scatter layer will be composed beside the fit: the layer shape it builds
    is the one that binds every layer to the NAMED transform output, which is
    what keeps the scatter reading the raw rows. See that call site.

    ``metrics`` appends the corner-statistics text layer, only meaningful
    when ``transform`` was itself built with ``inject_metrics=True`` (only
    ``Robust`` of the three dedicated fit transforms supports it; see
    ``lmplot``'s ``metrics_applied`` computation).

    ``chart_x``/``chart_y`` are the chart-level x/y channels the caller would
    have encoded. The layers below read the transform's own output columns, so
    an explicit ``scale=`` on either (``lmplot(logx=True)``, or a caller's
    ``y=fm.Y(..., scale=...)``), and its ``format=``/``format_type=``/``axis=``
    presentation, has to be propagated onto them, so the band, line and
    metrics carry the same scale and presentation the scatter does -- the
    step ``_resolve_pending_impl`` performs automatically for a chart built
    through the desugar path. Doing it here, at construction,
    is what keeps ``logx`` ONE mechanism: the alternative was a second
    implementation in ``lmplot``, re-running the propagation by hand for
    charts this function had left in a state the desugar pipeline never
    blessed, and a third for whatever chart-level feature came next.
    ``_propagate_layer_axis`` returns each layer unchanged when the channel
    carries none of these, so the ordinary case is a no-op rather than a
    branch.

    ``x_title``/``y_title`` are the names the caller's axes show (their
    ``title=``, else their own column names), which the layers need because
    they are bound to the transform's output columns and the renderer titles
    each axis from layer 0 (see ``build_fit_layers``). Required, not
    defaulted, and genuinely ``str``: ``lmplot`` resolves both once, over
    :func:`_required_channel`, which refuses a ``None`` in ``lmplot``'s own
    voice, so no call site can hand this parameter a missing title that
    would retitle the value axis with one of the transform's own output
    columns instead. When ``scatter=False`` leaves the fit alone there is
    no sibling layer to fall back on.
    """
    chart = Chart(data)._with_statistics(transform)
    chart._encoding = {"x": chart_x, "y": chart_y}
    chart._layers = [
        _propagate_layer_axis({"x": chart_x, "y": chart_y}, layer)
        for layer in build_fit_layers(
            kind,
            data_source=kind,
            ci=ci,
            x_title=x_title,
            y_title=y_title,
            line_kwargs=line_kwargs,
            metrics=metrics,
        )
    ]
    chart._mark = None
    return chart


@dataclass(frozen=True)
class _FitRequest:
    """Everything ``_lmplot_impl`` resolved before building its fit, in one value.

    Each ``_build_*_fit`` below reads the fields its method uses and ignores
    the rest, so the dispatch hands every builder the same argument instead
    of each call site picking a per-method subset. All fields are resolved
    once in ``_lmplot_impl``; none is re-derived here.

    ``entry`` is the public call the caller made (``lmplot`` or
    ``regplot``), so a builder's own refusal names it rather than
    ``lmplot`` unconditionally. ``x``/``y`` are the caller's channels
    (``x`` already carries ``logx``'s log scale); ``x_field``/``y_field``
    are their resolved column names, and ``x_title``/``y_title`` the names
    the fit layers title their axes with: the caller's ``title=`` on that
    channel, else the column name. ``grouping`` is the fields the fit
    groups by (the composite grouping of the hue and every grouping
    channel the caller binds), and ``grouping_bindings`` the bindings of
    those channels, which the fit chart binds itself when it is composed
    beside a scatter, so each per-group curve draws its group's color.
    ``keyword_bindings`` is the call's whole keyword encoding without
    ``x``/``y`` (the hue's resolved color, every grouping channel, and
    every other ``**encode_kwargs`` binding) -- what the fit chart binds
    instead, through ``Chart.encode``, when it draws with no scatter
    beside it, the same door ``encode=`` reaches it through, so an unknown
    channel refuses and a binding the fit cannot draw warns rather than
    vanishing. ``has_scatter`` says whether a scatter layer will be
    composed beside the fit, which decides the smooth path's layer shape
    and which of the two binding dicts the fit itself binds
    (:func:`_with_fit_bindings`). ``x_range_pad``, when set
    (``truncate=False``), asks the transform to extend the fit past its
    own on-side data extent by this fraction, in whatever space the fit
    uses (data units on a linear x, log units on a demanded log x): the
    transform derives the range itself, from its own whole-batch extent,
    so this function passes only the pad fraction, never a range it
    computed.
    """

    entry: str
    data: Any
    x: Any
    y: Any
    x_field: str
    y_field: str
    x_title: str
    y_title: str
    grouping: tuple[str, ...]
    grouping_bindings: dict
    keyword_bindings: dict
    ci: float | None
    line_kwargs: dict
    x_bins: Any
    x_estimator: Any
    order: int | None
    x_range_pad: float | None
    metrics: bool
    has_scatter: bool


def _build_smooth_fit(method: str, req: _FitRequest) -> Chart:
    """Every ``_SMOOTH_ROUTED_METHODS`` member: ONE ``mark_smooth`` call.

    The parameters that vary do so with the *arguments*, not with the method.

    - ``x_bins``/``x_estimator`` forward for every method. The engine bins
      method-agnostically -- ``smooth.rs``'s ``apply_one_group`` aggregates
      the x bins BEFORE it dispatches on method -- so the pre-R2 "only lm
      forwards binning" carve-out dropped both parameters silently for nine
      of lmplot's ten methods, which is the shape ``lmplot`` refuses
      everywhere else (``truncate=False`` and an inapplicable ``order`` both
      raise). Same rule as ``order``: forward what the caller passed and let
      the engine decide.
    - ``order`` forwards as ``degree=`` via ``_smooth_degree_kwargs``, whose
      docstring holds the per-method acceptance matrix.
    - ``groupby=`` the request's grouping fits one curve per group, and the
      fit binds the grouping channels, so each curve draws its group's color.
    - ``show_metrics`` needs no gate here: ``req.metrics`` is already False
      under any grouping (and for logistic/glm), computed in ``lmplot``.
    """
    lkw = req.line_kwargs
    smooth_kwargs: dict = dict(
        method=method,
        ci=req.ci,
        x_bins=req.x_bins,
        x_estimator=req.x_estimator,
        show_metrics=req.metrics,
        **_grouped(req),
        **_smooth_degree_kwargs(method, req.order, req.entry),
        **({"x_range_pad": req.x_range_pad} if req.x_range_pad is not None else {}),
    )
    if not req.has_scatter:
        # No sibling layer: the fit IS the chart, so `mark_smooth`'s own
        # shape is the right one -- including the single-line shortcut it
        # takes for `ci=None, show_metrics=False`, whose chain-advancing
        # (unnamed) `Smooth` correctly becomes the chart's data.
        #
        # `line_kws` is handed to `mark_smooth` through its per-layer
        # door, aimed at the `line` layer, rather than spread as mark
        # kwargs (which would repaint the CI band too) or rewritten onto
        # `out._layers` after the fact. The post-hoc rewrite is what
        # dropped it here: on this branch the chart still carries a
        # pending stat mark, so `out._layers` is empty until the desugar
        # runs at render time (#199). Forwarding at construction reaches
        # both shapes `desugar_smooth` can return without forcing that
        # desugar early, which would cost the single-line shortcut.
        fit = Chart(req.data).mark_smooth(
            **smooth_kwargs, layer_kwargs={"line": lkw} if lkw else None
        )
        return fit.encode(x=req.x, y=req.y, **req.keyword_bindings)
    # A scatter layer will be composed beside this fit, so the fit's
    # transform must be a NAMED fan-out its own layers read through
    # `data_source=` -- never a chain-advancing transform, which would
    # become the rows the scatter layer reads. That is
    # exactly what `mark_smooth`'s single-line shortcut produces for
    # `ci=None, show_metrics=False`, and it corrupted the scatter:
    # measured on the pre-fix tree, `lmplot(df, x=, y=, ci=None,
    # show_metrics=False)` drew all 200 "observations" on the fit
    # grid's own vertices (0.000 px spread about a perfect line) for
    # lm/loess/quadratic/gam/log/sqrt in both lmplot and regplot --
    # or raised "unknown column" outright when the raw columns were
    # not themselves named x/y. Batch D's fix reached the CI path
    # only, because only that path was already layered.
    #
    # The transform still comes from `desugar_smooth`, not a second
    # construction here: it owns the per-method kwarg matrix (which
    # knobs forward, which are omitted so Rust's own defaults apply)
    # and the fit grid's `n`/`seed`. Only its LAYER shape is replaced,
    # by the same `build_fit_layers` triple the CI path and the
    # dedicated fit methods below already emit.
    # One transform, unpacked rather than indexed so a desugar that
    # ever emits a second one fails here instead of silently losing it.
    (smooth_transform,) = desugar_smooth(
        req.x_field, req.y_field, name="smooth", **smooth_kwargs
    ).transforms
    # `kind` doubles as the transform's `name=` here exactly as it does for
    # the three dedicated methods below.
    fit = _fit_chart_from_named_transform(
        req.data,
        smooth_transform,
        kind="smooth",
        ci=req.ci,
        line_kwargs=lkw,
        chart_x=req.x,
        chart_y=req.y,
        x_title=req.x_title,
        y_title=req.y_title,
        metrics=req.metrics,
    )
    return _with_fit_bindings(fit, req)


def _with_fit_bindings(fit: Chart, req: _FitRequest) -> Chart:
    """*fit* binding the channels it draws alone with, or groups by beside a scatter.

    Composed beside a scatter, the fit draws only the hue and every
    grouping channel its statistic groups by (:attr:`_FitRequest.grouping_bindings`),
    so sibling isolation holds: the caller's other keyword bindings (size,
    opacity, tooltip, ...) stay the scatter's own. With no scatter, the fit
    chart IS the whole chart, so it binds every keyword binding the call
    carried (:attr:`_FitRequest.keyword_bindings`) through ``Chart.encode``,
    exactly as ``encode=`` would reach it: an unknown channel refuses, and
    a binding the fit's statistic does not draw warns rather than
    vanishing without a word.
    """
    bindings = req.grouping_bindings if req.has_scatter else req.keyword_bindings
    return fit.encode(**bindings) if bindings else fit


def _grouped(req: _FitRequest) -> dict:
    """Return the ``groupby=`` keyword every fit transform takes for *req*'s grouping."""
    return {"groupby": list(req.grouping)} if req.grouping else {}


def _dedicated_fit_chart(kind: str, transform: Any, req: _FitRequest) -> Chart:
    """Build the chart around one of the three dedicated fit-family transforms.

    Each transform was built with the request's grouping as its ``groupby``
    (:func:`_grouped`), so it fits one curve per group, and the chart binds
    the channels :func:`_with_fit_bindings` decides, so each curve and band
    draws its group's color (or, with no scatter beside it, every keyword
    binding the call carried).
    """
    fit = _fit_chart_from_named_transform(
        req.data,
        transform,
        kind=kind,
        ci=req.ci,
        line_kwargs=req.line_kwargs,
        chart_x=req.x,
        chart_y=req.y,
        x_title=req.x_title,
        y_title=req.y_title,
        # False for logistic/glm: `lmplot` never sets `metrics` for them.
        metrics=req.metrics,
    )
    return _with_fit_bindings(fit, req)


def _build_logistic_fit(method: str, req: _FitRequest) -> Chart:
    transform = Logistic(
        x=req.x_field, y=req.y_field, n_grid=200, ci=req.ci, name="logistic", **_grouped(req)
    )
    return _dedicated_fit_chart("logistic", transform, req)


def _build_glm_fit(method: str, req: _FitRequest) -> Chart:
    transform = Glm(
        x=req.x_field,
        y=req.y_field,
        family="gaussian",
        n_grid=200,
        ci=req.ci,
        name="glm",
        **_grouped(req),
    )
    return _dedicated_fit_chart("glm", transform, req)


def _build_robust_fit(method: str, req: _FitRequest) -> Chart:
    # The one dedicated transform that takes `inject_metrics=` and `x_range_pad=`.
    transform = Robust(
        x=req.x_field,
        y=req.y_field,
        n_grid=200,
        ci=req.ci,
        name="robust",
        inject_metrics=req.metrics,
        **({"x_range_pad": req.x_range_pad} if req.x_range_pad is not None else {}),
        **_grouped(req),
    )
    return _dedicated_fit_chart("robust", transform, req)


# `lmplot`'s method -> fit builder table. Keyed over `_VALID_METHODS` exactly:
# the smooth-routed half is derived from the same registry read, so a new
# registry smooth method routes through `mark_smooth` with no edit here.
_FIT_BUILDERS: dict[str, Callable[[str, _FitRequest], Chart]] = {
    **dict.fromkeys(_SMOOTH_ROUTED_METHODS, _build_smooth_fit),
    "logistic": _build_logistic_fit,
    "glm": _build_glm_fit,
    "robust": _build_robust_fit,
}


def _build_fit(method: str, req: _FitRequest) -> Chart:
    """``lmplot``'s fit chart for ``method``: ONE dispatch over ``_FIT_BUILDERS``.

    An unknown ``method`` raises the same ``validate_choice`` ``ValueError``
    ``lmplot`` raises at entry. An ACCEPTED method with no builder (a
    ``_VALID_METHODS`` member someone added without a ``_FIT_BUILDERS``
    entry) raises ``LookupError`` naming it, rather than falling through to
    an unreachable branch.
    """
    builder = _FIT_BUILDERS.get(method)
    if builder is None:
        validate_choice(req.entry, "method", method, _VALID_METHODS)
        raise LookupError(
            f"{req.entry}: method={method!r} is accepted but has no fit builder in "
            "ferrum.plots.regression._FIT_BUILDERS"
        )
    return builder(method, req)


# ---------------------------------------------------------------------------
# Post-build positional record: an x/y binding written after the figure's
# own call reads or refuses it, in the figure's own voice -- see
# ferrum._compose._PostBuildPositional, applied by Chart._resolve_pending.
# ---------------------------------------------------------------------------


def _eager_post_build_channel(
    out: Chart,
    channel: str,
    *,
    called_field: str,
    drawn_column: str,
    default_title: str | None = None,
    honors_fieldless: bool = False,
    categorical_reads: Callable[[str, str, str], Any] = _FIT_CATEGORICAL_READS,
    categorical_disallowed: tuple[str, ...] = ("N", "O"),
    unreadable: tuple[str, ...] = (),
    check_unreadable: Callable[[str, str, ChannelBase], None] | None = None,
    scale_remedy: Callable[[str], str] | None = None,
    type_remedy: Callable[[str], str] | None = None,
    always_honors_drawn_column: bool = False,
    restore_declared_type: str | None = None,
    refuse_another_variable: Callable[[str, str, ChannelBase], None] | None = None,
) -> _PostBuildChannel:
    """Build a regression figure's ``_PostBuildChannel`` for *channel*.

    The default shape -- a declared ``N``/``O`` refused by the
    categorical-input rule, a declared type that differs from the call's
    own refused naming the call door, and a field-less binding refused
    rather than honored -- fits regplot/lmplot's x and y and residplot's
    x alike: every one of the three reads *channel* as an input, never a
    computed axis, and *unreadable* defaults empty: a post-build
    ``scale=``/``format=`` pushes down onto the already-built layers the
    same way ``format_type=``/``axis=`` already do, through
    :func:`~ferrum._compose._push_down_positional_options`, rather than
    being refused. residplot's y passes
    ``honors_fieldless=True`` (it computes, rather than reads, its
    column), ``unreadable=("scale", "format")`` (the response column it
    would read is not the residual the figure draws) and overrides
    *categorical_reads*/*categorical_disallowed*/*check_unreadable* for
    its own log-scale remedy -- *scale_remedy* is unused with a custom
    *check_unreadable* of the caller's own, which builds its own ending
    if it wants one. *type_remedy* overrides the default declared-type
    ending for a form whose call door drops a declared ``T`` too
    (``scatter=False``, no sibling layer drawn).
    *always_honors_drawn_column*/*restore_declared_type*/*refuse_another_variable*
    pass straight through to :class:`~ferrum._compose._PostBuildChannel`
    (residplot's own use, documented there).
    """
    own = out._encoding.get(channel)
    call_type = own.option("type") if isinstance(own, ChannelBase) else None
    if check_unreadable is None:
        check_unreadable = partial(
            _refuse_unreadable_positional_options, unreadable=unreadable, scale_remedy=scale_remedy
        )
    return _PostBuildChannel(
        own=own,
        called_field=called_field,
        drawn_column=drawn_column,
        default_title=default_title,
        honors_fieldless=honors_fieldless,
        categorical_reads=categorical_reads,
        categorical_disallowed=categorical_disallowed,
        check_unreadable=check_unreadable,
        unreadable=unreadable,
        call_type=call_type,
        type_remedy=type_remedy,
        always_honors_drawn_column=always_honors_drawn_column,
        restore_declared_type=restore_declared_type,
        refuse_another_variable=refuse_another_variable,
    )


def _lmplot_type_remedy_without_scatter(voice: str) -> str:
    """Build the declared-``T`` ending for a ``scatter=False`` form, which has no call-time alternative either.

    ``scatter=False`` redraws the axis from the fit's own curve rather
    than a drawn scatter point, so the call door drops a declared ``T``
    the same way (``_lmplot_impl``'s own comment, next to
    :func:`_refuse_categorical_input`'s call): naming "pass it in the
    call" as this refusal's remedy would send the caller to a door that
    also fails.
    """
    return "Drop type=: with no scatter drawn, a declared type is never applied."


def _lmplot_post_build_positional(
    out: Chart, *, entry: str, method: str, x_field: str, y_field: str, has_scatter: bool
) -> _PostBuildPositional:
    """regplot/lmplot's ``_PostBuildPositional`` record, attached to *out* before it is returned.

    Built the same way on every form, pending smooth-routed included: a
    pending form's field-bearing x/y is left alone by
    :func:`~ferrum._compose.apply_post_build_positional`'s own live check
    of ``out._pending_stat_mark``, not by a differently-shaped record here
    -- the chart that check reads is the one ``Chart._resolve_pending``
    applies the record against, after ``_finalize_chart``'s own ``mark=``
    may already have resolved a pending smooth into an eager one, which a
    record shaped at this call could not see coming.
    ``statistic_refusal`` refuses a bin=/aggregate= the fit reads raw or
    folds in nowhere (:func:`~ferrum._encode_lowering._refuse_bin_aggregate_fit`),
    naming ``x_bins=`` as x's bin remedy on a ``mark_smooth``-routed
    method, the same knob the call-time check names, on every form too --
    the pending prep's own resolve-time refusal would say the same thing
    moments later, so running this one first keeps one wording. With
    ``has_scatter=False`` both channels pass
    :func:`_lmplot_type_remedy_without_scatter` as their declared-``T``
    ending.
    """
    type_remedy = None if has_scatter else _lmplot_type_remedy_without_scatter
    return _PostBuildPositional(
        entry=entry,
        channels={
            "x": _eager_post_build_channel(
                out, "x", called_field=x_field, drawn_column=x_field, type_remedy=type_remedy
            ),
            "y": _eager_post_build_channel(
                out, "y", called_field=y_field, drawn_column=y_field, type_remedy=type_remedy
            ),
        },
        statistic_refusal=partial(
            _refuse_bin_aggregate_fit, entry, x_bins_knob=method in _SMOOTH_ROUTED_METHODS
        ),
    )


def _residplot_y_scale_remedy(ch: ChannelBase) -> Callable[[str], str] | None:
    """Residplot's y scale remedy: ``None`` (the default ending), unless *ch* declares a log scale.

    A domain or ``format=`` names the call door as the working remedy
    (``residplot(y=fm.Y('w', ...))``); a log scale never does, because the
    call door refuses it too (residplot's own Rust refusal, wherever the
    resolved y scale is log), so naming it as a remedy would send the
    caller to a door that also fails.
    """
    scale = ch.option("scale")
    if scale is None:
        return None
    from ferrum.encoding._scale import _scale_to_dict

    if _scale_to_dict(scale).get("type") == "log":
        return lambda voice: "Use fm.SymlogScale(), or drop scale=."
    return None


def _refuse_residplot_y_unreadable(entry: str, channel: str, ch: ChannelBase) -> None:
    """Residplot's y unreadable-option check: ``scale=``/``format=`` always, with its own scale remedy.

    Unlike every other channel in this family, residplot's y keeps
    refusing a post-build ``scale=``/``format=``: the response column
    the call named is not the residual the figure draws, so there is
    nothing on the chart for either option to describe any more.
    """
    _refuse_unreadable_positional_options(
        entry,
        channel,
        ch,
        unreadable=("scale", "format"),
        scale_remedy=_residplot_y_scale_remedy(ch),
    )


def _residplot_post_build_positional(
    out: Chart,
    *,
    x_field: str,
    y_field: str,
    residual_title: str,
    drawn_x: str,
    drawn_residual: str,
    x_declared_type: str | None,
    flat: bool,
) -> _PostBuildPositional:
    """Residplot's ``_PostBuildPositional`` record, attached to *out* before it is returned.

    x is an input the fit reads, built like regplot/lmplot's own
    (:func:`_eager_post_build_channel`, re-pointed onto its drawn column
    *drawn_x* -- the name :func:`_residual_column_names` chose, read at
    the call site rather than repeated here -- with no *default_title*:
    the re-point's own fallback -- the post-build binding's own field,
    which equals *x_field* by the time this channel is re-pointed at all
    -- already matches residplot's call-time rebind, which passes none
    either). *x_declared_type* restores a call-time declared ``T`` after
    the re-point forces it back to ``"Q"`` (:attr:`~ferrum._compose._PostBuildChannel.restore_declared_type`),
    the same restoration residplot's own call-time rebind runs
    (``regression.py``, near the x re-point below): needed because flat
    residplot draws x straight from the chart's own encoding, with no
    layer of its own to keep the declared type safe from a later
    ``.encode()`` the way a layered figure's own layer does.

    y is the residual axis residplot computes, re-pointed onto
    *drawn_residual*: a field-less binding styles it
    (``honors_fieldless=True``) rather than being refused, a declared type
    also disallows ``T`` (the residual axis is never a time axis either,
    refused on a field-less binding the same way as a field-bearing one --
    :func:`~ferrum._encode_lowering._refuse_fieldless_categorical_type`),
    and its scale=/format= unreadable check uses residplot's own remedy
    (:func:`_refuse_residplot_y_unreadable`).

    Both channels pass ``always_honors_drawn_column=True``: a caller
    whose predictor happens to share residplot's own drawn x column's
    name, or whose response shares the drawn residual column's, is naming
    the column the figure already draws -- read as the figure's own
    drawn-column binding is, never refused as a rewrite of the called
    field. *flat* is ``True`` only when no overlay is drawn: flat
    residplot's one mark reads x/y straight from the chart-level encoding,
    with no layer to shadow a later binding naming neither the called
    field nor the drawn column, so only there does each channel also pass
    ``refuse_another_variable`` (:func:`~ferrum._encode_lowering._refuse_residplot_another_variable`),
    naming this call's own field. The layered form passes none: its
    layers keep drawing their own columns regardless of what the
    chart-level encoding says, as `regplot`/`lmplot` already do.
    """
    return _PostBuildPositional(
        entry="residplot",
        channels={
            "x": _eager_post_build_channel(
                out,
                "x",
                called_field=x_field,
                drawn_column=drawn_x,
                always_honors_drawn_column=True,
                restore_declared_type=x_declared_type,
                refuse_another_variable=(
                    partial(_refuse_residplot_another_variable, called_field=x_field)
                    if flat
                    else None
                ),
            ),
            "y": _eager_post_build_channel(
                out,
                "y",
                called_field=y_field,
                drawn_column=drawn_residual,
                default_title=residual_title,
                honors_fieldless=True,
                categorical_reads=_residual_axis_categorical_reads,
                categorical_disallowed=("N", "O", "T"),
                unreadable=("scale", "format"),
                check_unreadable=_refuse_residplot_y_unreadable,
                always_honors_drawn_column=True,
                refuse_another_variable=(
                    partial(_refuse_residplot_another_variable, called_field=y_field)
                    if flat
                    else None
                ),
            ),
        },
        statistic_refusal=_refuse_residplot_bin_aggregate,
    )


# ---------------------------------------------------------------------------
# Public figure functions
# ---------------------------------------------------------------------------


def _lmplot_impl(
    entry: str,
    data: Any,
    *,
    x: str,
    y: str,
    hue: Any = None,
    col: Any = None,
    row: Any = None,
    method: str = "lm",
    ci: Any = 95,
    order: int | None = None,
    scatter: bool = True,
    scatter_kws: Any = None,
    line_kws: Any = None,
    truncate: bool = True,
    x_bins: Any = None,
    x_estimator: Any = None,
    x_jitter: Any = None,
    logx: bool = False,
    show_metrics: bool | None = None,
    mark: dict | None = None,
    encode: dict | None = None,
    properties: dict | None = None,
    layers: list | None = None,
    theme: Any = None,
    **encode_kwargs: Any,
) -> Chart:
    """``lmplot``'s body, voiced as *entry* (``lmplot`` itself, or ``regplot``).

    Every self-naming refusal and warning below reads *entry* rather than
    the literal ``"lmplot"``, so a call that reached this through
    ``regplot`` names ``regplot``. See ``lmplot`` for the parameter
    documentation.
    """
    validate_choice(entry, "method", method, _VALID_METHODS)
    # x_estimator= is refused here naming entry, not by Smooth's own Rust
    # text or a raw PyO3 TypeError.
    validate_optional_choice(entry, "x_estimator", x_estimator, x_estimator_accepted_values())

    # x_bins/x_estimator: `Logistic`/`Glm`/`Robust` expose no binning
    # parameters at all (`transform_registry` lists none for any of the
    # three), so there is nothing to forward -- refuse rather than silently
    # drop, the same "no silent parameter drop" principle `truncate=False`
    # already enforces two methods of this same trio above. Either kwarg
    # alone is refused: an `x_estimator=` with no `x_bins=` would otherwise
    # be a second, quieter no-op with nothing to act on.
    if method in _DEDICATED_FIT_METHODS and (x_bins is not None or x_estimator is not None):
        bad = "x_bins" if x_bins is not None else "x_estimator"
        raise ValueError(
            f"{entry}(method={method!r}, {bad}=...) is not "
            f"supported -- the {method!r} transform has no x_bins/x_estimator "
            "binning parameters to pre-aggregate the fit input. Use a "
            "mark_smooth-routed method (lm, loess, quadratic, cubic, log, "
            "sqrt, gam) or leave x_bins=None (the default)."
        )

    # x/y and hue are parsed once, here, before any consumer: every one
    # below (the refusals, the jitter column, the scatter's encoding, the
    # fit request, the axis titles) reads the one parsed channel, never
    # the raw keyword, so a shorthand string, a declared type and a
    # channel's own options mean the same thing in all of them.
    x_ch = _required_channel(x, entry=entry, param="x")
    y_ch = _required_channel(y, entry=entry, param="y")
    hue = _grouping_hue(entry, hue)

    # Every method fits x as the raw predictor and y as the raw response;
    # neither folds in a bin= or aggregate= the caller bound on either
    # channel (through a keyword, `encode=`, or a shorthand string), so a
    # caller who wrote one would otherwise see the scatter layer honor it
    # and the fit silently ignore it. x_bins= is a working remedy only on
    # the methods that build Smooth (it bins before any method dispatch);
    # the dedicated methods have no such knob (refused above), so their x
    # bin= falls back to the generic, field-aware remedy.
    drawn = _drawn_positional_channels({"x": x_ch, "y": y_ch}, encode)
    _refuse_bin_aggregate_fit(
        entry,
        drawn,
        x_bins_knob=method in _SMOOTH_ROUTED_METHODS,
    )
    # A declared categorical type on either channel is refused the same
    # way: the fit reads both as numbers, so a `N`/`O` would place the
    # fitted curve's values as categories instead of positions on a
    # continuous axis. `Q` and `T` are honored -- a declared `T` draws a
    # time axis only with a scatter drawn; with `scatter=False` the fit
    # redraws the axis from its own curve, so the declared `T` is not
    # applied there.
    _refuse_categorical_input(entry, drawn, ("x", "y"), reads=_FIT_CATEGORICAL_READS)

    data = to_polars(data)
    x_field, y_field = x_ch.field, y_ch.field

    # Normalize CI: spec accepts ci=95 (percent) or ci=None.
    ci_frac = (ci / 100.0) if ci is not None else None

    # The corner-metrics overlay is supplied by the Smooth/Robust Rust
    # transform via ``inject_metrics=True`` (threaded through ``mark_smooth
    # (show_metrics=True)`` for smooth-routed methods, and directly on
    # ``Robust(...)`` below for ``method="robust"``), so R²/RMSE/MAE is
    # computed once in Rust and surfaced as ``_metrics_text`` /
    # ``_metrics_y`` columns on the fit-grid output — no Python-side
    # duplication. ``inject_metrics`` is generic (verified against
    # ``smooth.rs``: every ``SmoothMethod`` arm accepts it, computed
    # against the un-binned input rows, in whatever space the fit uses),
    # so this applies uniformly to every smooth-routed method and to
    # ``"robust"``.
    # ``"logistic"``/``"glm"`` expose no ``inject_metrics`` param at all
    # (``transform_registry`` lists none), so those two stay excluded
    # regardless of ``show_metrics=``. Computed below, once the grouping is
    # known: a per-group fit has no single corner to report.

    # truncate: True = clip to data range (default: x_range_pad stays
    # unset, so the transform uses its own per-group data extent). False =
    # extend the fit past its own on-side data extent by a 5% margin, in
    # the transform's own fit space (data units on a linear x, log units
    # on a demanded log x). `x_range_pad` alone is the request: Smooth and
    # Robust derive the range themselves, from the fit's own whole-batch
    # extent (the same facet pin Bin/Kde/Kde2D already use), and pad it in
    # whatever space they fit in -- this function passes no range, since
    # it cannot see which side of zero a demanded log axis places (that is
    # resolved in Rust, where the axis is). "logistic"/"glm" accept no
    # `x_range_pad` at all (`transform_registry` confirms neither lists
    # it), so extending their fit line past the data range isn't possible
    # without a Rust-side transform change -- truncate=False refuses for
    # those two rather than silently clipping anyway -- no silent
    # parameter drop.
    x_range_pad = None
    if not truncate:
        if method in ("logistic", "glm"):
            raise ValueError(
                f"{entry}(method={method!r}, truncate=False) is not "
                f"supported -- the {method!r} transform has no x_range "
                "parameter to extend the fit line past the observed data "
                "range. Use method='robust' (accepts x_range), a "
                "mark_smooth-routed method (lm, loess, quadratic, cubic, "
                "log, sqrt, gam), or leave truncate=True (the default)."
            )
        x_range_pad = _TRUNCATE_FALSE_X_RANGE_PAD

    # `x_jitter` offsets the scatter's x column by writing a Float64 noise
    # array back into it (`apply_jitter`, `render/position.rs`), which fails
    # Arrow's schema check on any other numeric dtype: `Jitter: ... expected
    # Int64 but found Float64`. Widen the column once, here, so BOTH charts
    # built below share it -- the merged chart carries the scatter chart's own
    # data, so casting one and not the other would put two dtypes on one
    # column. Gated on `scatter` because `x_jitter` adjusts nothing without a
    # scatter layer to adjust, and on the dtype so a Float64 column is
    # untouched (a non-numeric x is left alone too: the renderer refuses it in
    # its own terms rather than failing here on a cast).
    #
    # This widening step can be deleted once `apply_jitter` coerces the
    # column it writes itself, the way the transform engine already
    # coerces its other inputs (`numeric_util::coerce_to_float64`).
    # `catplot(native_scale=True)` carries the same workaround for the
    # same renderer behavior.
    if scatter and x_jitter is not None:
        jitter_frame = to_polars(data)
        jitter_column = x_field
        dtype = jitter_frame.schema.get(jitter_column)
        if dtype is not None and dtype.is_numeric() and dtype != pl.Float64:
            data = jitter_frame.with_columns(pl.col(jitter_column).cast(pl.Float64))

    # Shared encoding.
    #
    # `hue` is already the parsed, Nominal-unless-typed channel
    # `_grouping_hue` returned above, reused on every layer below. It is a
    # group discriminator: it drives Smooth's `groupby=`, so one fit and one
    # CI band are produced per level. Left to infer, a numeric hue column read
    # as Continuous collapsed all of those back into a single rendered fit and
    # a single band -- a materially wrong statistical chart, drawn silently.
    # The scatter layer takes the same typing rather than the `point` carve-out
    # because it shares one color scale with the fit layers in the same layered
    # chart; typing only the fit half would ask that one scale to be Continuous
    # and Nominal at once. See `nominal_color_channel` for the rule.

    # `logx` is part of the x channel every layer is built with, not a
    # post-hoc fixup applied to finished charts: resolving it once here means
    # the scatter, the `mark_smooth`-routed fit and the dedicated-transform
    # fit all take the same channel object.
    x_enc = _log_scaled_x(entry, x_ch, x_field, param="logx") if logx else x_ch

    enc: dict = {"x": x_enc, "y": y_ch}
    if hue is not None:
        enc["color"] = hue
    enc.update(encode_kwargs)
    # Every keyword binding but x/y: the hue's color, every grouping
    # channel, and every other `**encode_kwargs` entry. With no scatter
    # drawn, the fit chart binds this whole dict itself, through
    # `Chart.encode`, the same door `encode=` reaches it through
    # (`_with_fit_bindings`): a misspelled flag refuses there instead of
    # rendering a plain fit, and a binding the fit's statistic does not
    # draw warns rather than vanishing without a word.
    keyword_bindings = {
        channel: value for channel, value in enc.items() if channel not in ("x", "y")
    }

    # Every fit method groups by the hue and by every grouping channel the
    # caller binds (color, fill, stroke, detail, shape, stroke_dash), whether
    # as a bare keyword (merged into `enc` above) or through the `encode=`
    # dict `_finalize_chart` applies later: the composite grouping,
    # computed once here, so each method fits one curve per group as a
    # composite statistic does. The fit binds the same channels, so each
    # curve draws its group's color. `_drawn_bindings` reads `encode=` the
    # way `_finalize_chart` will apply it (a value wins, a `None` leaves the
    # keyword's own binding in place), so the fit groups by what is drawn.
    bound = _drawn_bindings(enc, encode)
    grouping = tuple(_encode_grouping(bound))
    grouping_bindings = {
        channel: value
        for channel, value in bound.items()
        if channel in COMPOSITE_GROUPING_CHANNELS and value is not None
    }
    # One R²/RMSE/MAE corner describes one fit: none under any grouping.
    # "logistic"/"glm" expose no single-fit corner at all (neither transform
    # has `inject_metrics`), excluded before `_metrics_applied` so neither
    # warns under `show_metrics=True`, matching their docs above.
    metrics_applied = method not in ("logistic", "glm") and _metrics_applied(
        entry, show_metrics, list(grouping)
    )

    # ---- Scatter layer (optional) --------------------------------------
    scatter_layer = None
    if scatter:
        s = Chart(data)
        skw = _style_kws(entry, "scatter_kws", "point", scatter_kws)
        if x_jitter is not None:
            s = s.mark_point(position=Jitter(axis="x", width=float(x_jitter)), **skw)
        else:
            s = s.mark_point(**skw)
        s = s.encode(**enc)
        scatter_layer = s

    # ---- Fit layer (per method) ----------------------------------------
    # Every `_FIT_BUILDERS` entry forwards `lkw` at CONSTRUCTION, to the fit
    # line alone, rather than rewriting it onto the finished chart's
    # `_layers`: with `scatter=False`, the chart still carries a pending
    # stat mark until render, so a post-hoc rewrite onto `_layers` would
    # silently drop it on that branch.
    lkw = _style_kws(entry, "line_kws", "line", line_kws)

    fit = _build_fit(
        method,
        _FitRequest(
            entry=entry,
            data=data,
            x=x_enc,
            y=y_ch,
            x_field=x_field,
            y_field=y_field,
            # A suppressed title keeps the column name: the caller's channel,
            # bound at the chart level, already draws no axis title.
            x_title=drawn_title(x_ch) or x_field,
            y_title=drawn_title(y_ch) or y_field,
            grouping=grouping,
            grouping_bindings=grouping_bindings,
            keyword_bindings=keyword_bindings,
            ci=ci_frac,
            line_kwargs=lkw,
            x_bins=x_bins,
            x_estimator=x_estimator,
            order=order,
            x_range_pad=x_range_pad,
            metrics=metrics_applied,
            has_scatter=scatter_layer is not None,
        ),
    )

    # Compose scatter + fit. (When ``metrics_applied``, the fit chart
    # already carries the metrics-text layer reading the Smooth output.)
    if scatter_layer is None:
        out = fit
    else:
        out = _merge_layers(scatter_layer, fit, scatter_name="scatter", fit_name="fit")

    # Faceting.
    if col is not None or row is not None:
        if col is not None and row is not None:
            out = out.facet(row=row, col=col)
        elif col is not None:
            out = out.facet(col=col)
        else:
            out = out.facet(row=row)

    # An x/y binding the caller writes after this call -- `encode=` below
    # included, which lands on `out._encoding` the same way a later
    # `.encode()` does -- reads or refuses this call's own x=/y=, in
    # `entry`'s voice, applied by `Chart._resolve_pending`.
    out = out._with_post_build_positional(
        _lmplot_post_build_positional(
            out,
            entry=entry,
            method=method,
            x_field=x_field,
            y_field=y_field,
            has_scatter=scatter_layer is not None,
        )
    )

    return _finalize_chart(
        out,
        entry=entry,
        mark=mark,
        encode=encode,
        properties=properties,
        layers=layers,
        theme=theme,
    )


@_figure_entry
def lmplot(
    data: Any,
    *,
    x: Any,
    y: Any,
    hue: Any = None,
    col: Any = None,
    row: Any = None,
    method: str = "lm",
    ci: Any = 95,
    order: int | None = None,
    scatter: bool = True,
    scatter_kws: Any = None,
    line_kws: Any = None,
    truncate: bool = True,
    x_bins: Any = None,
    x_estimator: Any = None,
    x_jitter: Any = None,
    logx: bool = False,
    show_metrics: bool | None = None,
    mark: dict | None = None,
    encode: dict | None = None,
    properties: dict | None = None,
    layers: list | None = None,
    theme: Any = None,
    **encode_kwargs: Any,
) -> Chart:
    """Linear (and non-linear) regression scatter overlay.

    Builds a layered chart with an optional scatter (``mark_point``) and a
    regression fit line, dispatching to the appropriate transform:

    * Every ``mark_smooth``-routed method (see ``method`` below for the
      list) -- ``mark_smooth(method=<method>)``. All of them accept an
      explicit ``order=`` (forwarded uniformly as ``degree=``, see ``order``
      below); a bare call (``order`` left at its ``None`` default) uses each
      method's own default degree concept.
    * ``"logistic"`` -- ``Logistic`` transform + fit line + CI band.
    * ``"glm"``      -- ``Glm`` transform + fit line + CI band.
    * ``"robust"``   -- ``Robust`` transform + fit line + CI band.

    Every method fits one curve per group of the chart's grouping: the
    ``hue`` and every grouping channel the caller binds (``color``,
    ``fill``, ``stroke``, ``detail``, ``shape``, ``stroke_dash``, as a
    keyword or through ``encode=``), each curve and band drawn in its
    group's color, as a composite statistic groups by its chart's grouping
    channels. Without a grouping the fit is one curve over every row.

    Parameters
    ----------
    data : DataFrame-like
        Input data accepted by ``Chart(data)``.
    x : str or encoding
        Column name for the horizontal (predictor) axis (required), read
        exactly as ``encode()`` reads a channel value: shorthand
        (``"bill:Q"``) and an encoding object both work. A declared
        ``N``/``O`` is refused -- the fit reads x as a number and draws
        its curve along a continuous axis. A declared ``T`` draws a time
        axis only with ``scatter=True`` (the default): the scatter's own
        channel establishes the axis, with the fit still computed on its
        raw numbers. With ``scatter=False`` the fit redraws the axis from
        its own curve, so the declared ``T`` is not applied and the chart
        renders as the bare field does.
    y : str or encoding
        Column name for the vertical (response) axis (required), read the
        same way as ``x=`` (shorthand or an encoding object), with the
        same ``N``/``O`` refusal and ``T`` handling.
    hue : str or encoding, optional
        Column name to map to color, read the way a column-naming keyword
        is read: shorthand (``"k:Q"``) parses as ``encode()`` reads it, and
        a column name the shorthand grammar cannot read (``"Weight (kg)"``)
        binds that exact column. A declared type is kept (``hue="k:Q"``
        draws the same as ``hue=fm.Color("k", type="Q")``), and an
        undeclared one is typed Nominal, a grouping discriminator rather
        than an inferred Continuous scale. An aggregate (``hue="mean(k)"``)
        is refused: ``hue=`` names the column the fit groups by, not a
        value to compute. For every ``method``, the fit is refit per hue
        level via its transform's ``groupby``, so each level gets its own
        fit line and CI band in that level's scatter color. Any other
        grouping channel the caller binds (``color=``/``fill=``/``stroke=``,
        ``detail``, ``shape`` or ``stroke_dash``, as a keyword or through
        ``encode=``) groups the fit the same way.
    col : str or encoding, optional
        Column name for faceting across columns, read as
        ``Chart.facet(col=...)`` reads it (shorthand or a
        ``fm.FacetCol(...)`` object).
    row : str or encoding, optional
        Column name for faceting across rows, read the same way as
        ``col=``.
    method : str, default "lm"
        Fitting method. THE list of accepted values -- every other entry
        below refers here rather than repeating it, and
        ``tests/test_lmplot_valid_methods_registry_pin.py`` fails if this
        enumeration and the derived ``_VALID_METHODS`` disagree, so a new
        registry smooth method cannot land undocumented.

        ``"lm"``, ``"loess"``, ``"quadratic"``, ``"cubic"``, ``"log"``,
        ``"sqrt"``, ``"gam"`` route through ``mark_smooth``; ``"logistic"``,
        ``"glm"``, ``"robust"`` route through their own dedicated fit-family
        transform. The accepted set is derived, not hand-held:
        ``transform_registry("smooth")["methods"] - {"linear", "glm"}``
        plus ``{"logistic", "glm", "robust"}``.
    ci : int or None, default 95
        Confidence interval level (0--100) shown as a band around the fit
        line.  Pass ``None`` to suppress.
    order : int or None, default None
        Polynomial degree. ``None`` (the default) means "use this method's
        own default degree concept" -- forwards nothing, so a bare call is
        unaffected by which method it lands on. Any explicit int, including
        ``1``, is forwarded to ``mark_smooth`` as ``degree=order``
        uniformly for every ``mark_smooth``-routed ``method`` -- one
        forwarding rule, not a per-method carve-out; see
        ``_smooth_degree_kwargs``. What a forwarded value
        means then depends on ``mark_smooth``'s own degree-acceptance matrix
        for that method:

        - ``"lm"``: any ``degree >= 1`` is accepted (``>= 2`` promotes to a
          polynomial fit; ``1`` is a no-op against the default). ``degree=0``
          raises.
        - ``"quadratic"``/``"cubic"``: a matching value (``order=2``/``3``
          respectively) is a no-op; any other value -- including ``1`` --
          raises ``ValueError`` naming ``method="lm", degree=<n>`` as the way
          to get that fit.
        - ``"loess"``: ``order`` becomes the LOCAL-FIT degree (``1`` or
          ``2``, not a polynomial order). ``order=1`` is therefore a REAL,
          distinguishable request for the linear local fit; ``order=2``
          lands on loess's own default local-fit degree (a harmless no-op).
          Any other value raises.
        - ``"log"``/``"sqrt"``/``"gam"``: no degree concept at all -- any
          explicit ``order=``, including ``1``, raises ``ValueError``.

        A negative ``order`` raises ``ValueError`` directly (rather than an
        untyped conversion error at the Rust boundary) for every method
        above. Ignored entirely for ``"logistic"``/``"glm"``/``"robust"``
        (a different fit family, with no ``order``/``degree`` parameter at
        all) -- no validation, typed or otherwise, applies to ``order`` for
        those three.
    scatter : bool, default True
        Include a scatter layer (``mark_point``).  Set to ``False`` to
        show only the fit line.
    scatter_kws : dict, optional
        Extra keyword arguments forwarded to the scatter ``mark_point``
        call (e.g. ``{"opacity": 0.3, "size": 20}``).
    line_kws : dict, optional
        Extra keyword arguments forwarded to the regression-line mark
        call (e.g. ``{"stroke_width": 3}``).
    truncate : bool, default True
        When ``True`` (default), the fit line spans only the observed data
        range (min to max of ``x``). When ``False``, extends the fit line
        past that range by a 5% margin the transform computes and applies
        itself, in its own fit space -- data units on a linear x, log
        units on a demanded log x (``logx=`` or a channel
        ``scale=fm.LogScale()``) -- via the transform's ``x_range_pad``
        support, supported for every ``mark_smooth``-routed method and for
        ``method="robust"``. Not supported for ``method="logistic"`` or
        ``"glm"`` (neither transform accepts ``x_range_pad``); passing
        ``truncate=False`` with either raises ``ValueError``.
    x_bins : int or None, optional
        Aggregate the data into this many equal-width x bins before fitting,
        so the fit runs on one point per bin. Forwarded to ``mark_smooth``
        for every ``mark_smooth``-routed ``method``: the engine bins before
        it dispatches on method, so binning is not method-specific. Not
        supported for ``"logistic"``/``"glm"``/``"robust"``, whose transforms
        take no binning parameters at all; passing ``x_bins=`` (or
        ``x_estimator=``) with one of those three raises ``ValueError``.
    x_estimator : str or None, optional
        The aggregate applied within each ``x_bins`` bin, validated against
        the same declared vocabulary ``fm.Smooth(x_estimator=...)`` accepts
        (a subset of the shared aggregate vocabulary -- see ``fm.Smooth``'s
        own ``x_estimator`` parameter for the accepted spellings); defaults
        to the transform's own ``"mean"``. Forwarded on the same terms as
        ``x_bins``, and only meaningful alongside it. Same
        ``"logistic"``/``"glm"``/``"robust"`` refusal as ``x_bins`` above
        applies on its own, even without ``x_bins=`` set.
    x_jitter : float or None, optional
        When set, applies ``Jitter(axis="x", width=x_jitter)`` to the
        scatter layer.
    logx : bool, default False
        Apply a ``log`` scale to the x-axis on both scatter and fit
        layers. Every method then fits in log x, not data x: an
        ``"lm"``-family fit draws a straight line on the log axis, its CI
        band maps back asymmetric, and ``method="sqrt"`` -- which would
        compose a second predictor transform on top of the axis's own log
        -- raises instead of silently fitting in data space.
    show_metrics : bool or None, default None
        Overlay a top-right corner annotation with ``R^2`` / ``RMSE`` /
        ``MAE`` computed from the fit, in whatever space the fit itself
        uses -- log x under ``logx=`` or a channel ``scale=fm.LogScale()``.
        ``inject_metrics`` is a generic ``Smooth``/``Robust`` transform
        kwarg (computed against the un-binned input rows, in whatever
        space the fit uses), so this applies to every
        ``mark_smooth``-routed method and to ``method="robust"``. Not
        supported for ``method="logistic"``/``"glm"``
        (neither transform exposes ``inject_metrics``) -- skipped for
        those two regardless of this flag, and never warned about. Left
        unset (``None``), the corner is drawn for one ungrouped fit and
        dropped silently under any grouping (``hue`` or a grouping
        channel, as a keyword or through ``encode=``), since one corner
        describes one fit, not one per group. Passed explicitly as
        ``True``, a grouping that leaves no single fit to describe warns
        instead of silently drawing nothing. ``False`` never draws it and
        never warns.
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
        Additional keyword arguments bound on the scatter when one is
        drawn (``scatter=True``, the default); with ``scatter=False``
        there is no scatter to bind them on, so they bind on the fit chart
        itself instead, through ``Chart.encode()`` -- an unknown channel
        (a misspelled flag) refuses the same way it would on the scatter,
        and a binding the fit's statistic does not draw warns rather than
        silently vanishing.

    Returns
    -------
    Chart
        Layered chart (scatter + fit) or fit-only when ``scatter=False``.
        May be faceted.

    Raises
    ------
    ValueError
        If ``method`` is not one of the supported values.
    ValueError
        If ``x`` declares type ``N``/``O``, or ``y`` declares ``N``/``O``:
        see ``x``/``y`` above.
    ValueError
        If ``method="sqrt"`` on a demanded log x (``logx=True``, or a
        channel ``scale=fm.LogScale()``) -- a sqrt predictor transform and
        the axis's own log cannot both apply to one x.
    ValueError
        If ``truncate=False`` with ``method="logistic"`` or ``"glm"``
        (neither transform accepts ``x_range_pad``).
    ValueError
        If ``x_bins`` or ``x_estimator`` is set with
        ``method="logistic"``, ``"glm"``, or ``"robust"`` (none of the
        three transforms accepts a binning parameter).
    ValueError
        If an explicit ``order`` is not valid for ``method`` (e.g.
        ``method="quadratic", order=3``; ``method="log", order=2``) --
        ``mark_smooth``'s own degree-acceptance matrix, see ``order`` above.
    ValueError
        If ``order`` is a negative int, for any ``mark_smooth``-routed
        ``method``.
    ValueError
        If ``x`` or ``y`` -- through a keyword, ``encode=`` or a shorthand
        string -- carries a ``bin=``/``aggregate=``: the fit reads them as
        the raw predictor/response, never a bin or aggregate of them. An
        ``x bin=`` names ``x_bins=`` as its remedy on a ``mark_smooth``-
        routed ``method`` (the fit bins before it dispatches on method);
        on ``"logistic"``/``"glm"``/``"robust"``, which have no such knob,
        and for an ``x aggregate=`` on every method (``x_estimator=``
        aggregates the response within each ``x_bins`` bin, never the
        predictor, so it is never the remedy), the remedy is to compute it
        ahead of the fit with ``.transform(...)`` and bind its output
        column. ``y``'s ``bin=``/``aggregate=`` take that same remedy on
        every method.

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.lmplot(df, x="total_bill", y="tip")

    Logistic regression, one fit per sex, each in its scatter's color:

    >>> fm.lmplot(df, x="total_bill", y="smoker_int", method="logistic", hue="sex")

    Polynomial fit (degree 2) with no confidence band:

    >>> fm.lmplot(df, x="size", y="tip", order=2, ci=None)
    """
    return _lmplot_impl(
        "lmplot",
        data,
        x=x,
        y=y,
        hue=hue,
        col=col,
        row=row,
        method=method,
        ci=ci,
        order=order,
        scatter=scatter,
        scatter_kws=scatter_kws,
        line_kws=line_kws,
        truncate=truncate,
        x_bins=x_bins,
        x_estimator=x_estimator,
        x_jitter=x_jitter,
        logx=logx,
        show_metrics=show_metrics,
        mark=mark,
        encode=encode,
        properties=properties,
        layers=layers,
        theme=theme,
        **encode_kwargs,
    )


def _metrics_applied(entry: str, show_metrics: bool | None, grouping: list[str]) -> bool:
    """Whether *entry*'s single-fit metrics corner is drawn, given its *grouping* fields.

    ``show_metrics`` left unset (``None``) draws the corner for one
    ungrouped fit and drops it silently under any grouping, the same as
    every other unset style default. ``True`` draws it ungrouped and, when
    *grouping* leaves no single fit to describe, warns once rather than
    drawing nothing silently. ``False`` never draws it and never warns.
    Callers exclude a method with no single fit to begin with (lmplot's
    ``"logistic"``/``"glm"``) before calling this, so that exclusion stays
    silent regardless of ``show_metrics`` or *grouping*.
    """
    if not grouping:
        return show_metrics is not False
    if show_metrics is True:
        import warnings

        from ferrum._warn import user_stacklevel
        from ferrum.encoding._aliases import _prose_list

        grouping_prose = _prose_list([repr(g) for g in grouping])
        warnings.warn(
            f"{entry}: show_metrics=True has no single fit to describe: the fit is "
            f"computed per group of {grouping_prose}, so the R²/RMSE/MAE corner "
            "is not drawn",
            stacklevel=user_stacklevel(),
        )
    return False


#: The channels residplot's layered form binds at the chart level: x/y/x2/y2,
#: a grouping channel (refits the residuals per group, so every layer that
#: reads the chart's rows must regroup the same way) or a facet channel
#: (partitions the rows before any layer draws them). Every other channel a
#: keyword or ``encode=`` binds on the layered form reaches the point layer
#: alone, through its own explicit encoding, never the chart's -- otherwise
#: the zero-reference rule and the corner metrics, which draw one mark for
#: the whole fit, and the lowess overlay, which reads its own fit grid,
#: would each inherit a per-row binding meant for the points (a residplot
#: with no overlay drawn -- the flat form -- has no sibling layer to shadow
#: it from, so every channel reaches the chart there instead).
_RESIDPLOT_CHART_LEVEL_CHANNELS = (
    frozenset(("x", "y", "x2", "y2")) | COMPOSITE_GROUPING_CHANNELS | _FACET_CHANNELS
)


#: The names the residual transform emits by default
#: (``crates/ferrum-core/src/transform/smooth/fits.rs::build_residuals_batch``,
#: mirrored byte-for-byte in ``transform/robust.rs::build_residuals_batch``),
#: keyed the way :func:`_residual_column_names` and ``Smooth``/``Robust``'s
#: own ``_with_residual_columns`` both key them.
_RESIDUAL_COLUMN_DEFAULTS = {
    "x": "x",
    "residual": "residual",
    "ref_zero": "_ref_zero",
    "metrics_text": "_metrics_text",
    "metrics_y": "_metrics_y",
}


def _residual_column_names(
    columns: Iterable[str], *, x_field: str, zero_line: bool, metrics_applied: bool
) -> dict[str, str]:
    """Return the names residplot's residual transform emits this call, read by every layer that draws them.

    The caller's data is never renamed: every door that reads the
    caller's rows -- a later binding, a ``.transform()`` step, an
    attached layer, a facet -- reads the caller's own column under its
    own name. Instead, when *columns* (the data's own, after the
    ``_label`` injection) already uses one of the transform's default names,
    the COMPUTED column takes a fresh one instead, through
    :func:`~ferrum.marks._desugar_helpers._unique_col_name`, via an internal
    ``residual_columns`` setting on the ``Smooth``/``Robust`` transform
    (never a public kwarg).

    ``x`` keeps its default when the predictor itself is named ``x`` --
    the batch's own ``x`` already holds those values, so there is nothing
    to carry a second time either way -- or when the data has no ``x`` at
    all. ``residual`` always takes a fresh name when the data already has
    one: the computed residual is never the same values as a raw column of
    that name, the caller's own response included. ``_ref_zero``,
    ``_metrics_text`` and ``_metrics_y`` are reserved only when THIS call's
    transform actually emits them (*zero_line*, *metrics_applied*): a
    column the transform never computes is left entirely the caller's, and
    no name is chosen for it at all.
    """
    taken = set(columns)
    names: dict[str, str] = {}
    names["x"] = (
        _RESIDUAL_COLUMN_DEFAULTS["x"]
        if x_field == "x" or "x" not in taken
        else _unique_col_name(taken, "x")
    )
    taken.add(names["x"])
    names["residual"] = _unique_col_name(taken, "residual")
    taken.add(names["residual"])
    if zero_line:
        names["ref_zero"] = _unique_col_name(taken, "_ref_zero")
        taken.add(names["ref_zero"])
    if metrics_applied:
        names["metrics_text"] = _unique_col_name(taken, "_metrics_text")
        taken.add(names["metrics_text"])
        names["metrics_y"] = _unique_col_name(taken, "_metrics_y")
        taken.add(names["metrics_y"])
    return names


@_figure_entry
def residplot(
    data: Any,
    *,
    x: Any,
    y: Any,
    lowess: bool = False,
    order: int = 1,
    robust: bool = False,
    dropna: bool = True,
    show_metrics: bool | None = None,
    zero_line: bool = True,
    label: Any = None,
    color: Any = None,
    mark: dict | None = None,
    encode: dict | None = None,
    properties: dict | None = None,
    layers: list | None = None,
    theme: Any = None,
    **encode_kwargs: Any,
) -> Chart:
    """Residual-diagnostic scatter plot.

    Computes regression residuals via ``Smooth(output="residuals")`` (or
    ``Robust(output="residuals")`` when ``robust=True``) and plots
    ``(x, residual)`` with ``mark_point``. When ``lowess=True``, a
    ``mark_line`` lowess smoother is layered over the residuals to help
    diagnose non-linearity.

    Parameters
    ----------
    data : DataFrame-like
        Input data accepted by ``Chart(data)``.
    x : str or encoding
        Column name for the horizontal axis (predictor; required), read
        exactly as ``encode()`` reads a channel value: shorthand
        (``"bill:Q"``) and an encoding object (``fm.X("col", scale=...,
        axis=..., format=..., title=...)``) both work, and every option a
        channel carries reaches the point layer's own x binding, the same
        re-pointing rule ``displot``'s histogram and ECDF kinds use. A
        declared ``N``/``O`` is refused -- the residual fit reads x as its
        numeric predictor -- and a declared ``T`` draws a time axis, with
        the fit computed on its raw numbers. A demanded log x
        (``scale=fm.LogScale()``) fits the underlying regression in log x,
        not data x.
    y : str or encoding
        Column name the residuals are computed from (response; required),
        read the same way as ``x=``, and its options reach the residual axis;
        its default title is ``"<y> residual"`` rather than the field
        itself. A declared ``N``, ``O`` or ``T`` is refused: the residual
        axis draws signed differences around zero, never a category or a
        time. A log scale on the RESIDUAL axis (through any door -- a
        channel ``scale=``, ``encode=`` or ``.override(y_scale_type="log")``)
        is refused rather than warning every row off the side: residuals
        are signed differences around 0, which a ``LogScale`` cannot place.
    lowess : bool, default False
        Overlay a lowess smoother on the residuals using
        ``Smooth(method="loess")``.
    order : int, default 1
        Polynomial degree of the regression used to compute residuals.
    robust : bool, default False
        Use ``Robust`` regression (MM-estimator) instead of OLS when
        computing residuals. Compatible with ``show_metrics=True`` and
        ``zero_line=True`` (annotations flow through the Robust
        transform's same opt-in kwargs).
    dropna : bool, default True
        Drop rows where ``x`` or ``y`` is null before fitting.
    show_metrics : bool or None, default None
        Schwabish SB-followup (2026-05-12): overlay a top-right corner
        annotation with ``R^2`` / ``RMSE`` / ``MAE`` computed inside the
        Rust Smooth/Robust transform via the ``inject_metrics=True``
        kwarg -- same single execution model as fitted residuals, no
        Python-side regression duplication. Left unset (``None``), the
        corner is drawn and dropped silently when ``color=`` names a
        column or a grouping channel is bound through ``encode=``,
        mirroring ``lmplot``'s ``hue`` exclusion: those residuals are
        refit per group, so there is no single corner value to report.
        Passed explicitly as ``True``, such a grouping warns instead of
        silently drawing nothing; ``False`` never draws it and never
        warns. ``label=`` does not suppress it either way -- that
        grouping has one level by construction, so the corner stays
        correct.
    zero_line : bool, default True
        Schwabish SB-followup: draw a dashed horizontal reference at
        ``y=0`` via the Rust transform's ``inject_zero_ref=True`` opt-in.
    label : str, optional
        Legend label for the residual series.  When set, a constant
        ``_label`` column is injected and mapped to color, producing a
        single-entry legend.  Composes with ``robust=True``, with a
        grouping ``color=`` (both reach the transform's ``groupby``), and
        with a constant ``color=`` (the legend entry takes that color).
    color : str or encoding, optional
        Either a **column name** (or an encoding object naming one) or a
        **constant color**, resolved in that order: a name that matches a
        column of ``data`` groups the residuals -- they are refit per
        group, coloured by that column, and a ``lowess=True`` overlay
        draws one curve per group -- so a column literally named ``"red"``
        groups rather than paints. A string that parses as shorthand
        naming a column (``"g:N"`` when the data has a column ``"g"``,
        not a column literally named ``"g:N"``) groups by that parsed
        column too, with its declared type kept. An aggregate shorthand
        (``"mean(g)"``) is neither a column nor a parseable color, so it
        is refused the same way any other unmatched string is. Otherwise
        the value is parsed as a color (CSS name, hex, ``rgb()``/
        ``rgba()``, or an RGB tuple) and painted onto the residual points
        as a constant ``fill``; combined with ``label=``, that same color
        is also the legend swatch. A value that is neither raises
        ``ValueError`` naming both readings.
    theme : Theme, optional
        Visual theme applied via ``Chart.theme()``.
    mark : dict, optional
        Per-layer mark overrides.  For composite-mark charts, keys are
        layer names (e.g. ``{"scatter": {"opacity": 0.5}}``); for
        single-mark charts, a flat dict of mark properties.
    encode : dict, optional
        Additional encoding kwargs, validated via ``Chart.encode()``. On
        the layered form (any overlay requested: the default), a channel
        other than x/y/x2/y2, a grouping channel, or a facet channel
        binds the point layer alone, not the chart -- the zero-reference
        rule, the corner metrics, and the lowess overlay never inherit a
        per-row binding meant for the points.
    properties : dict, optional
        Chart properties merged via ``Chart.properties(**properties)``
        (e.g. ``{"width": 400, "title": "My chart"}``).
    layers : list, optional
        Extra layers appended via ``Chart.layer(*layers)``.
    **encode_kwargs
        Additional keyword arguments, validated and bound the same way as
        ``encode``'s entries above.

    Returns
    -------
    Chart
        Scatter of residuals (possibly with a lowess layer, zero
        reference line, and corner R^2/RMSE/MAE annotation).

    Raises
    ------
    ValueError
        If ``color`` is neither a column of ``data`` nor a value ferrum's
        color parser accepts -- see its parameter above.
    ValueError
        If ``label`` is set and ``data`` already has a ``_label`` column,
        which this function needs for the label's own group key.
    ValueError
        If the residual axis resolves a ``LogScale``, through any door
        (``y=``'s own ``scale=``, ``encode=``, or
        ``.override(y_scale_type="log")``): residuals are signed
        differences around 0, which a ``LogScale`` cannot place. Use
        ``fm.SymlogScale()``, or drop the log scale.
    ValueError
        If ``x`` or ``y`` -- through a keyword, ``encode=`` or a shorthand
        string -- carries a ``bin=``/``aggregate=``: residuals are
        computed from the raw predictor/response columns, never a bin or
        aggregate of them. Compute it ahead of time with
        ``.transform(...)`` and bind its output column (the remedy call
        needs ``dropna=False``, since the default drops rows before any
        ``.transform()`` step runs), or drop the kwarg.
    ValueError
        If ``x`` declares type ``N``/``O``, or ``y`` declares ``N``,
        ``O`` or ``T``: see ``x``/``y`` above.
    ValueError
        On the flat form (no overlay drawn), if an ``x``/``y`` binding
        written after the call -- ``encode=`` or a later ``.encode()`` --
        names another column of the data: the figure's one mark would
        draw that column without refitting against it. Naming the call as
        the remedy. The layered form is unaffected -- its layers bind
        ``x``/``y`` themselves, as ``regplot``/``lmplot`` already do.

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.residplot(df, x="total_bill", y="tip")

    Robust residuals with annotations:

    >>> fm.residplot(df, x="size", y="tip", robust=True)

    Residuals refit and coloured per group:

    >>> fm.residplot(df, x="total_bill", y="tip", color="day")

    Every residual point painted one constant color:

    >>> fm.residplot(df, x="total_bill", y="tip", color="steelblue")
    """
    data = to_polars(data)
    # Narrowed once, before the first consumer: `drop_nulls` takes column
    # NAMES, and so do the residual transform and the two axis titles below.
    # Without this, `residplot(x=fm.X("bill"))` -- the spelling `lmplot`
    # accepts -- leaked polars' own `cannot turn 'X' into selector` from the
    # line below, and `x=None` leaked the same message for `NoneType`; behind
    # that, the titles would have announced the repr of a channel object
    # (`title=x`, below), which is #199's defect wearing a different value.
    # Parsed once into the channel each is, the way `encode()` reads a
    # channel value: a shorthand string ("bill:Q") is read as the channel
    # it spells, so `x_field`/`y_field` and the re-point below never see
    # the raw, un-split shorthand.
    x_ch = _required_channel(x, entry="residplot", param="x")
    y_ch = _required_channel(y, entry="residplot", param="y")
    x_field = x_ch.field
    y_field = y_ch.field
    # The residual transform reads x/y as the raw predictor/response; a
    # bin= or aggregate= on either (through a keyword, `encode=`, or a
    # shorthand string) would be dropped silently once the channel is
    # rebound to the residual output. dropna=True drops nulls over the
    # input frame before any transform runs, so the generic remedy's own
    # .transform() step needs dropna=False to see the column it binds.
    drawn = _drawn_positional_channels({"x": x_ch, "y": y_ch}, encode)
    _refuse_residplot_bin_aggregate(drawn)
    # A declared categorical type is refused on both axes: the residual
    # fit reads x as its predictor, the same as regplot's fit, and the
    # residual axis itself draws signed differences around zero, which is
    # never a category OR a time -- so y refuses a declared T too, unlike
    # x, which keeps a declared T as a time axis below.
    _refuse_categorical_input("residplot", drawn, ("x",), reads=_FIT_CATEGORICAL_READS)
    _refuse_categorical_input(
        "residplot",
        drawn,
        ("y",),
        reads=_residual_axis_categorical_reads,
        disallowed=("N", "O", "T"),
    )
    if dropna:
        data = data.drop_nulls(subset=[x_field, y_field])

    # `color=` is documented two ways -- a grouping column or a constant
    # color -- and the dispatch runs against the INPUT frame, before the
    # `_label` column below is injected, so membership is tested against
    # exactly the columns the caller supplied (spec §6: membership is
    # primary, so a column literally named "red" groups and never paints).
    color_arg = _column_or_literal_color(color, data.columns, func_name="residplot")
    # The one channel object every layer that reads the color scale binds
    # (the points and, when grouped, the lowess overlay), so that scale is
    # never asked to be two scale types at once. `None` unless `color=`
    # named a column; a literal color paints marks instead (see
    # `point_kwargs` below) and never reaches an encoding.
    color = color_arg.channel

    # A literal `color=` and `label=` are both honored, not one at the other's
    # expense: `label=` carries its legend entry on a one-level color scale,
    # and an encoding outranks a mark's own `fill`, so left alone the scale
    # would repaint the points in a palette color and silently discard the
    # literal. Pinning that scale's single range slot to the literal makes the
    # points AND the legend swatch the requested color -- one legend entry, in
    # the color the caller asked for. The domain is the single level the
    # `_label` column below is built from.
    label_paint_channel = None
    if label is not None and color_arg.paint is not None:
        label_paint_channel = Color(
            "_label",
            type_="nominal",
            scale=OrdinalScale(domain=[str(label)], range=[color_arg.paint]),
        )

    # The Rust Smooth/Robust transforms inject ``_ref_zero``,
    # ``_metrics_text``, and ``_metrics_y`` columns when the corresponding
    # ``inject_*`` kwargs are set — single source of truth for residual
    # computation lives in Rust; Python only declares the spec.
    # The residual transforms carry every input column of the row a
    # residual came from, a `groupby` column riding once, under its own
    # name, as the group key. So `label=` injects a constant `_label`
    # column and groups by it, and a grouping `color=` joins the same
    # list — both Smooth and Robust take a list of group columns, so the
    # two compose and residuals are refit per group.
    groupby: list[str] = []
    if label is not None:
        # `_label` is this function's own synthetic column, so a caller who
        # already has one loses it: the literal would overwrite their values
        # and `groupby` would carry the name twice. Rust tolerates the
        # duplicate key and renders, which is the silent half -- refuse and
        # name the collision instead.
        if "_label" in data.columns:
            raise ValueError(
                "residplot(label=...) needs a private '_label' column, but the "
                "data already has one -- the label would overwrite it silently. "
                "Rename that column, or drop label= and encode the existing "
                "column yourself with color='_label'."
            )
        data = data.with_columns(pl.lit(str(label)).alias("_label"))
        groupby.append("_label")
    if color_arg.group_field is not None:
        groupby.append(color_arg.group_field)
    # A grouping channel the caller binds through `encode=` or a keyword
    # (`encode={"color": "c"}`, `shape="c"`) groups the residuals too, as
    # every composite statistic groups by its chart's grouping fields: each
    # group is refit on its own rows, which carry the field its marks draw.
    # `_drawn_bindings` applies `encode=` over the keyword bindings, the
    # order `_finalize_chart` applies them in, so a keyword and `encode=`
    # naming the same channel differently refits by `encode=`'s value, the
    # one the chart draws.
    encode_grouping = _encode_grouping(_drawn_bindings(encode_kwargs, encode))
    groupby = with_grouping(groupby, encode_grouping)

    # `residplot` passes `degree=order` straight to `Smooth` below, so it
    # needs the same typed refusal `lmplot` gets from `_smooth_degree_kwargs`
    # -- called for the raise, since the kwargs it builds are not the ones
    # used here. The `Robust` path takes no `order`, so nothing to validate.
    if not robust:
        _smooth_degree_kwargs("lm", order, "residplot")

    # Mirrors `lmplot`'s `metrics_applied`: the corner annotation is one
    # R²/RMSE/MAE for one fit, so a residplot refit per data group has no
    # single corner to render, and labelling one group's metrics as the
    # chart's would be wrong rather than crowded.
    #
    # Gated on the data groupings (`color=` and the encoded grouping
    # channels) ONLY, not on `groupby` as a whole. A
    # `label=` grouping is one level by construction (`pl.lit`), so the
    # transform emits exactly one corner for it and the reason above simply
    # does not apply -- suppressing there would take the R²/RMSE/MAE away
    # from a plain `residplot(..., label="…")` for no benefit. That is the
    # same distinction `lmplot` draws: it gates on a real multi-level `hue`.
    #
    # It gates the transform's `inject_metrics` too, so the columns the
    # metrics layer reads exist exactly when that layer is built.
    metrics_grouping = with_grouping(
        [color_arg.group_field] if color_arg.group_field is not None else [], encode_grouping
    )
    metrics_applied = _metrics_applied("residplot", show_metrics, metrics_grouping)

    # The names this call's transform emits, read by every layer below: the
    # caller's own data is never renamed (see `_residual_column_names`), so
    # when it already uses one of these names the COMPUTED column takes a
    # fresh one instead, through an internal `residual_columns` setting on
    # the transform.
    names = _residual_column_names(
        data.columns, x_field=x_field, zero_line=zero_line, metrics_applied=metrics_applied
    )
    residual_column_overrides = {
        role: name for role, name in names.items() if name != _RESIDUAL_COLUMN_DEFAULTS[role]
    }

    # Both residual transforms take the same residual-mode arguments; only
    # the fit family differs (and `degree`, which Robust has no analogue for).
    resid_kwargs: dict = dict(
        x=x_field,
        y=y_field,
        output="residuals",
        inject_zero_ref=zero_line,
        inject_metrics=metrics_applied,
    )
    if groupby:
        resid_kwargs["groupby"] = groupby
    if robust:
        resid_transform = Robust(**resid_kwargs)
    else:
        resid_transform = Smooth(method="lm", degree=order, ci=None, **resid_kwargs)
    if residual_column_overrides:
        resid_transform = resid_transform._with_residual_columns(**residual_column_overrides)

    # A literal `color=` paints the residual points through the point mark's
    # own kwargs -- never through `encode(color=)`, which would fabricate a
    # one-level color scale and a legend for what the caller asked to be a
    # constant.
    point_kwargs: dict = {} if color_arg.paint is None else {"fill": color_arg.paint}

    chart = Chart(data)._with_statistics(resid_transform).mark_point(**point_kwargs)
    # Both channels are bound to the residual batch's own column names, which
    # are the axis titles the renderer reads off layer 0 -- so without a
    # `title=` this chart announced its x axis as "x" and its value axis as
    # "residual" on its DEFAULT path, losing the caller's columns entirely
    # (#199, the same layer-0 mechanism `build_fit_layers` carries). The value
    # axis names the response it residualizes, since "residual" alone leaves
    # the reader no way back to the fitted column. A title= the caller wrote
    # on x= or y= names its axis instead, and an empty one (title=None or "")
    # suppresses it, as on a plain chart's axis. The overlay layers below share
    # these two names.
    declared_x, declared_y = declared_title(x_ch), declared_title(y_ch)
    x_title = x_field if declared_x is None else declared_x
    residual_title = f"{y_field} residual" if declared_y is None else declared_y
    # The caller's own x/y channel re-pointed at the residual batch's
    # columns through the same re-pointing rule `kind="hist"` runs at
    # resolve time, so a `scale=`, `axis=` or `format=` the caller wrote on
    # either channel survives. The y side's default title is this
    # function's own "<y> residual", not the field itself, which is the
    # one difference from the bare re-point: an explicit title (including
    # a suppressing `None`) still wins either way.
    rebound_x = _rebind_positional(names["x"], x_ch, X)
    if x_ch.option("type") == "T":
        # `_rebind_positional` forces `type="Q"` on every re-point (its
        # other callers lower a `bin=`/`aggregate=`, never a declared axis
        # type); a predictor declared temporal still fits on its raw
        # numbers, so its declared `T` is restored here, the way
        # `regplot`'s own `X('v', type='T')` already draws it.
        rebound_x = rebound_x._with_options(type="T")
    enc: dict = {
        "x": rebound_x,
        "y": _rebind_positional(names["residual"], y_ch, Y, title=residual_title),
    }
    if color is not None:
        enc["color"] = color
    elif label_paint_channel is not None:
        enc["color"] = label_paint_channel
    elif label is not None:
        enc["color"] = "_label"
    enc.update(encode_kwargs)

    # Build the layered chart when any overlay is requested (zero rule,
    # metrics text, or lowess smoother). The augmented-DataFrame pattern
    # works because all overlays reference columns the Rust transform
    # already emitted into the same residuals batch. Requested, not drawn:
    # a grouping suppresses the metrics layer, and the chart keeps its
    # named layers either way, so a grouping bound after the call rebuilds
    # it layer for layer (`ferrum._compose.regrouped`). `show_metrics` is
    # requested whenever it is not explicitly `False` (unset still asks
    # for the corner when there is one fit to describe).
    layered = lowess or zero_line or show_metrics is not False
    # `Chart.encode` is every channel's one validator (a constant like
    # `size=30` raises its own `TypeError` here, the same as on any
    # chart): run it over the whole of `enc` regardless of form, then, on
    # the layered form only, drop the point-only channels back off the
    # chart's own encoding below -- stripping them before this call would
    # validate none of them.
    chart = chart.encode(**enc)
    if layered:
        # On the layered form, a value channel (keyword or `encode=`) binds
        # the point layer alone, never the chart level: the chart's own
        # encoding only ever carries x/y/x2/y2, a grouping channel
        # (`COMPOSITE_GROUPING_CHANNELS`) or a facet channel
        # (`_FACET_CHANNELS`), so the zero-reference rule, the corner
        # metrics and the lowess overlay -- which all read the chart-level
        # encoding or its own residual batch, never a per-row binding of
        # their own -- never inherit a row-level value meant for the
        # points. The point layer itself still binds every one of them,
        # below.
        chart = chart._clone()
        chart._encoding = {
            k: v for k, v in chart._encoding.items() if k in _RESIDPLOT_CHART_LEVEL_CHANNELS
        }
        internal_layers: list = [
            _Layer(
                name="point",
                mark="point",
                # Binds every `encode=` binding but x/y as its own, the
                # same overlay the chart's own encoding gets
                # (`_finalize_chart` applies `encode=` to the chart, not
                # to this layer): the point layer is this figure's
                # primary mark, so it draws what `encode=` binds and its
                # tooltip lists it, instead of losing it to the per-layer
                # own-fields-shadow rule. `_drawn_bindings` keeps a
                # `None` entry from unbinding a keyword's own channel,
                # the same rule the chart draws by. x/y are excluded: this
                # layer's own `enc["x"]`/`enc["y"]` already point at the
                # residual batch's columns, while `encode=`'s x/y reach
                # the chart level, where the post-build record re-points
                # them the same way a later `.encode()` does -- merging
                # `encode=`'s raw field here would re-bind this layer to
                # a column the residual batch never emits.
                encoding=_drawn_bindings(
                    enc, {k: v for k, v in (encode or {}).items() if k not in ("x", "y")}
                ),
                mark_kwargs=point_kwargs or None,
            )
        ]
        if zero_line:
            internal_layers.append(
                _Layer(
                    name="reference",
                    mark="rule",
                    encoding={"y": Y(names["ref_zero"], title=residual_title)},
                    mark_kwargs={"stroke_dash": [3, 3], "stroke": "#8a8a8a"},
                    fit_summary=True,
                )
            )
        if metrics_applied:
            internal_layers.append(
                _Layer(
                    name="metrics",
                    mark="text",
                    encoding={
                        "x": X(names["x"], title=x_title),
                        "y": Y(names["metrics_y"], title=residual_title),
                        "text": names["metrics_text"],
                    },
                    mark_kwargs={"align": "right", "dx": -4, "dy": 4},
                    fit_summary=True,
                )
            )
        if lowess:
            chart = chart._with_statistics(
                Smooth(
                    x=names["x"],
                    y=names["residual"],
                    method="loess",
                    ci=None,
                    name="lowess",
                    # The overlay reads the residual batch, which carries the
                    # group keys, so a grouped residplot smooths each group's
                    # own residuals -- one curve per group instead of one
                    # curve through the pooled cloud.
                    **({"groupby": groupby} if groupby else {}),
                ),
            )
            # Derived from the registry (spec §4.6 / M6a) rather than
            # hardcoded: this layer reads the named `lowess` Smooth
            # transform's fit-grid output, so a Rust-side rename raises here
            # instead of silently drawing nothing.
            _lowess_cols = registry_output_columns("smooth", ["x", "y"])
            lowess_enc: dict = {
                "x": X(_lowess_cols["x"], title=x_title),
                "y": Y(_lowess_cols["y"], title=residual_title),
            }
            # A line mark draws ONE polyline through the rows it is handed:
            # what splits the per-group curves is the color channel, so this
            # layer names it rather than leaning on the chart-level channel
            # being inherited by a layer that declares none (#106 -- real
            # today, deliberately left in flux, and written against a
            # different batch than the one this layer reads). It binds the
            # same channel object the points bind: one scale, one scale type.
            if color is not None:
                lowess_enc["color"] = color
            elif label_paint_channel is not None:
                lowess_enc["color"] = label_paint_channel
            elif label is not None:
                lowess_enc["color"] = "_label"
            internal_layers.append(
                _Layer(
                    name="lowess",
                    mark="line",
                    encoding=lowess_enc,
                    data_source="lowess",
                )
            )
        chart._layers = internal_layers
        chart._mark = None

    # An x/y binding the caller writes after this call -- `encode=` below
    # included -- reads or refuses this call's own x=/y=, in residplot's
    # voice, applied by `Chart._resolve_pending`. Only the flat form (no
    # layer of its own to shadow a chart-level binding) also refuses a
    # binding naming another column of the data.
    chart = chart._with_post_build_positional(
        _residplot_post_build_positional(
            chart,
            x_field=x_field,
            y_field=y_field,
            residual_title=residual_title,
            drawn_x=names["x"],
            drawn_residual=names["residual"],
            x_declared_type=x_ch.option("type"),
            flat=not layered,
        )
    )

    # On the layered form, `encode=` lands on the chart the same way the
    # keyword door does (above): only its x/y/grouping/facet channels
    # reach the chart level, since the point layer already has every
    # other channel through its own encoding, and the summary/lowess
    # layers must not inherit one meant for the points.
    finalize_encode = encode
    if layered and encode:
        point_only = {k: v for k, v in encode.items() if k not in _RESIDPLOT_CHART_LEVEL_CHANNELS}
        if point_only:
            # Validate the point-only entries `_finalize_chart` below never
            # sees once `finalize_encode` strips them out -- `_drawn_bindings`
            # merges them straight into the point layer's encoding dict
            # with no type check of its own, so a malformed value here
            # (`encode={"opacity": 0.5}`) would otherwise render with the
            # binding silently dropped instead of `Chart.encode`'s TypeError.
            chart.encode(**point_only)
        finalize_encode = {k: v for k, v in encode.items() if k in _RESIDPLOT_CHART_LEVEL_CHANNELS}

    return _finalize_chart(
        chart,
        entry="residplot",
        mark=mark,
        encode=finalize_encode,
        properties=properties,
        layers=layers,
        theme=theme,
    )


@_figure_entry
def regplot(
    data: Any,
    *,
    x: Any,
    y: Any,
    hue: Any = None,
    method: str = "lm",
    ci: Any = 95,
    order: int | None = None,
    scatter: bool = True,
    scatter_kws: Any = None,
    line_kws: Any = None,
    truncate: bool = True,
    x_bins: Any = None,
    x_estimator: Any = None,
    x_jitter: Any = None,
    logx: bool = False,
    show_metrics: bool | None = None,
    mark: dict | None = None,
    encode: dict | None = None,
    properties: dict | None = None,
    layers: list | None = None,
    theme: Any = None,
    **encode_kwargs: Any,
) -> Chart:
    """Axes-level regression scatter plot.

    The axes-level equivalent of ``lmplot`` — identical API except
    ``lmplot``'s figure-level ``col=``/``row=`` faceting parameters are
    excluded (passing either raises ``TypeError`` naming ``lmplot`` as the
    faceting route -- see Raises below). This exclusion is specific to
    ``col=``/``row=``; it does not extend to the grammar-level ``facet=``
    encoding channel, which remains available on ``regplot`` the same as
    on any other ``Chart``. Every other parameter is forwarded to
    ``lmplot`` unchanged.

    Parameters
    ----------
    data : DataFrame-like
        Input data accepted by ``Chart(data)``.
    x : str or encoding
        Column name for the horizontal (predictor) axis (required),
        forwarded to ``lmplot`` -- see ``lmplot``'s ``x`` parameter: read
        exactly as ``encode()`` reads a channel value, with the same
        categorical refusal and temporal handling.
    y : str or encoding
        Column name for the vertical (response) axis (required),
        forwarded the same way as ``x=`` -- see ``lmplot``'s ``y``
        parameter.
    hue : str or encoding, optional
        Column name to map to color, forwarded to ``lmplot`` -- see
        ``lmplot``'s ``hue`` parameter: every method fits one line and band
        per hue level, as it does per level of any grouping channel
        (``color=``/``fill=``/``stroke=``, ``detail``, ``shape``,
        ``stroke_dash``, or their ``encode=`` dict equivalents). A declared
        type is kept, exactly as ``lmplot``'s ``hue`` parameter describes.
    method : str, default "lm"
        Fitting method forwarded to ``lmplot`` -- see ``lmplot``'s ``method``
        parameter for the full accepted set.
    ci : int or None, default 95
        Confidence interval level (0--100) shown as a band around the fit
        line.  Pass ``None`` to suppress.
    order : int or None, default None
        Polynomial degree, forwarded to ``lmplot`` -- see ``lmplot``'s
        ``order`` parameter for the full forwarding rules. ``None`` means
        "use the method's own default degree"; any explicit int, including
        ``1``, is forwarded.
    scatter : bool, default True
        Include a scatter layer (``mark_point``).
    scatter_kws : dict, optional
        Extra keyword arguments forwarded to the scatter ``mark_point`` call.
    line_kws : dict, optional
        Extra keyword arguments forwarded to the regression-line mark call.
    truncate : bool, default True
        When ``True``, the fit line spans only the observed data range.
        When ``False``, forwarded to ``lmplot`` -- see ``lmplot``'s
        ``truncate`` parameter for the full accepted set (not supported for
        ``method="logistic"``/``"glm"``; raises ``ValueError``).
    x_bins : int or None, optional
        Aggregate the data into this many equal-width x bins before fitting,
        forwarded to ``lmplot`` -- see ``lmplot``'s ``x_bins`` parameter
        (not supported for ``"logistic"``/``"glm"``/``"robust"``; raises
        ``ValueError``).
    x_estimator : str or None, optional
        The aggregate applied within each ``x_bins`` bin, forwarded to
        ``lmplot`` -- see ``lmplot``'s ``x_estimator`` parameter for the
        accepted values and the same ``"logistic"``/``"glm"``/``"robust"``
        refusal as ``x_bins`` above.
    x_jitter : float or None, optional
        When set, applies ``Jitter(axis="x", width=x_jitter)`` to the
        scatter layer.
    logx : bool, default False
        Apply a ``log`` scale to the x-axis on both scatter and fit layers,
        forwarded to ``lmplot`` -- see ``lmplot``'s ``logx`` parameter:
        every method then fits in log x, and ``method="sqrt"`` raises
        instead of composing a second predictor transform on top of it.
    show_metrics : bool or None, default None
        Overlay a top-right corner annotation with ``R^2`` / ``RMSE`` /
        ``MAE`` computed from the fit, forwarded to ``lmplot`` -- see
        ``lmplot``'s ``show_metrics`` parameter for the method/``hue``
        exclusions and the warning an explicit ``True`` raises under a
        grouping.
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
        Additional keyword arguments, forwarded to ``lmplot`` -- see
        ``lmplot``'s ``**encode_kwargs``: bound on the scatter when
        ``scatter=True``, or on the fit chart itself through
        ``Chart.encode()`` when ``scatter=False``.

    Returns
    -------
    Chart
        Layered chart (scatter + fit) or fit-only when ``scatter=False``.

    Raises
    ------
    TypeError
        If ``col=`` or ``row=`` is passed -- ``regplot`` does not facet;
        use ``lmplot`` instead.
    ValueError
        If ``method`` is not one of the supported values.
    ValueError
        If ``x`` declares type ``N``/``O``, or ``y`` declares ``N``/``O``
        -- forwarded from ``lmplot``, see its ``x``/``y`` parameters.
    ValueError
        If ``method="sqrt"`` on a demanded log x -- forwarded from
        ``lmplot``, see its ``method`` parameter.
    ValueError
        If ``truncate=False`` with ``method="logistic"`` or ``"glm"``
        (neither transform accepts ``x_range_pad``).
    ValueError
        If ``x_bins`` or ``x_estimator`` is set with
        ``method="logistic"``, ``"glm"``, or ``"robust"`` -- forwarded from
        ``lmplot``, see its ``x_bins`` parameter.
    ValueError
        If an explicit ``order`` is not valid for ``method``, or is a
        negative int -- forwarded from ``lmplot``, see its ``order``
        parameter.
    ValueError
        If ``x`` or ``y`` -- through a keyword, ``encode=`` or a shorthand
        string -- carries a ``bin=``/``aggregate=`` the fit reads raw --
        forwarded from ``lmplot``, whose ``Raises`` entry names each
        method's remedy.

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.regplot(df, x="total_bill", y="tip")

    Robust regression:

    >>> fm.regplot(df, x="size", y="tip", method="robust")
    """
    # regplot is the axes-level (non-faceting) counterpart to lmplot.
    # col=/row= would otherwise reach lmplot silently through the
    # **encode_kwargs splice below (they bind to lmplot's own named
    # col/row params) and facet -- refuse explicitly rather than let an
    # unsupported feature work by accident. Checked before the splice, on
    # the same "name the unsupported thing and the supported alternative"
    # precedent as lmplot's own x_bins/x_estimator refusal above.
    if "col" in encode_kwargs or "row" in encode_kwargs:
        bad = "col" if "col" in encode_kwargs else "row"
        raise TypeError(
            f"regplot({bad}=...) is not supported -- regplot is the "
            "axes-level regression plot and does not facet. Use "
            f"lmplot(..., {bad}=...) instead."
        )
    return _lmplot_impl(
        "regplot",
        data,
        x=x,
        y=y,
        hue=hue,
        method=method,
        ci=ci,
        order=order,
        scatter=scatter,
        scatter_kws=scatter_kws,
        line_kws=line_kws,
        truncate=truncate,
        x_bins=x_bins,
        x_estimator=x_estimator,
        x_jitter=x_jitter,
        logx=logx,
        show_metrics=show_metrics,
        mark=mark,
        encode=encode,
        properties=properties,
        layers=layers,
        theme=theme,
        **encode_kwargs,
    )


@_figure_entry
def residuals_chart(
    model: Any = None,
    X: Any = None,
    y: Any = None,
    *,
    y_true: Any = None,
    y_pred: Any = None,
    kind: str = "studentized",
    cook_threshold: float | str | None = None,
    panels: Any = "auto",
    annotate_metrics: bool = True,
    subtitle: str | None = None,
    compare: dict[str, Any] | None = None,
    random_state: int | None = None,
    mark: dict | None = None,
    encode: dict | None = None,
    properties: dict | None = None,
    layers: list | None = None,
    theme: Any = None,
) -> "Chart | HConcatChart | VConcatChart | ConcatChart":
    """Residuals diagnostic chart for a regression estimator.

    Plots residuals vs. fitted values. Optional Cook's distance
    highlighting marks observations whose leverage-adjusted influence
    exceeds a user-supplied threshold. ``panels="auto"`` returns the
    canonical 4-panel diagnostic layout (residuals-vs-fitted, QQ,
    scale-location, residuals-vs-leverage) as a 2x2 grid.

    Parameters
    ----------
    model : estimator or ModelSource
        A fitted sklearn-compatible regression estimator, an explicit
        ``ferrum.ModelSource``, or a dict of named estimators.
    X : array-like, optional
        Feature matrix. Required when ``model`` is a raw
        estimator; ignored when it is already a ``ModelSource``.
    y : array-like, optional
        Target vector. Required when ``model`` is a raw
        estimator; ignored when it is already a ``ModelSource``.
    y_true : array-like, optional
        Ground-truth target values for the precomputed path.  Must be
        paired with ``y_pred``; mutually exclusive with ``model``.
    y_pred : array-like, optional
        Predicted target values for the precomputed path.
    kind : {"studentized", "raw"}, default "studentized"
        Residual type to plot on the y axis. ``"studentized"`` uses
        internally studentized residuals; ``"raw"`` uses raw residuals.
    cook_threshold : float, "auto", or None, default None
        Threshold for Cook's distance outlier highlighting. A float is
        used as an absolute cutoff; ``"auto"`` applies the ``4 / n``
        rule (Hair et al.); ``None`` disables highlighting. Cook's
        distance is only defined for estimators that expose ``coef_``;
        non-linear models surface ``NaN`` and produce no outliers.
    panels : "auto", None, "single", or list of str, default "auto"
        Panel selection. ``"auto"`` ships the canonical 4-panel layout
        -- ``residuals_vs_fitted``, ``qq``, ``scale_location``, and
        ``residuals_vs_leverage`` -- laid out as a 2x2 grid. ``None`` or
        ``"single"`` returns just the residuals-vs-fitted panel. Pass an
        explicit list such as ``["residuals_vs_fitted", "qq"]`` to
        customize the panel set.
    annotate_metrics : bool, default True
        Overlay a top-right corner annotation showing R^2/RMSE/MAE
        computed from the fit. Pass ``False`` to render the residual
        scatter alone.
    subtitle : str or None, default None
        Optional subtitle rendered beneath the active chart title.
    compare : dict of str -> estimator or None, default None
        Mapping of label -> fitted estimator to compare against ``model``.
        Behavior depends on ``panels``:

        - ``panels="single"`` (or ``None``): the residuals-vs-fitted scatter
          carries a ``model`` colour group with one overlaid series per model
          (unchanged single-panel overlay).
        - any multi-panel value (``"auto"`` or an explicit list): returns a
          [ConcatChart][ferrum.ConcatChart] of small multiples, one panel per model.
          Each model renders its full multi-panel diagnostic grid as one
          small-multiple panel, labeled with the model name. Scale sharing
          across models is not applied to the multi-panel grid (each panel
          carries its own faceted scales). The per-model hat-matrix leverage
          and Cook's-distance computations are correct by construction because
          each panel sees one model's data only.
    random_state : int or None, default None
        Seed forwarded to ``ModelSource``; does not affect deterministic
        residuals computation.
    theme : Theme or None, default None
        Ferrum theme to apply to the returned chart.
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

    Returns
    -------
    Chart or composite
        Residuals-vs-fitted scatter chart with optional Cook's-distance
        outlier overlay, the multi-panel diagnostic grid, or a
        small-multiples ``ConcatChart`` (one panel per model) when
        ``compare=`` is supplied with a multi-panel ``panels`` value.

    Examples
    --------
    >>> import ferrum as fm
    >>> from sklearn.linear_model import Ridge
    >>> fm.residuals_chart(Ridge().fit(X_train, y_train), X_test, y_test)

    Precomputed path — residuals are computed as ``y_true − y_pred`` internally.
    Leverage and Cook's distance are unavailable, so the leverage panel is
    omitted when ``panels="auto"``:

    >>> fm.residuals_chart(y_true=y_test, y_pred=reg.predict(X_test))
    """
    source = _resolve_source(
        model, X, y, y_true=y_true, y_pred=y_pred, random_state=random_state, compare=compare
    )
    panel_list = _resolve_residuals_panels(panels)
    # Both the compose path and the single-model fall-through pass identical
    # kwargs to the same builder (``panels=panel_list`` in both); only the
    # routing condition differs. Spell the kwargs once and route both paths
    # through it.
    builder_kwargs = dict(
        kind=kind,
        cook_threshold=cook_threshold,
        panels=panel_list,
        annotate_metrics=annotate_metrics,
        subtitle=subtitle,
        mark=mark,
        encode=encode,
        properties=properties,
        layers=layers,
        theme=theme,
    )
    # Multi-panel comparison renders small multiples (one full diagnostic grid
    # per model) so each model's leverage / Cook's-distance computations stay
    # single-model. The single-panel residuals-vs-fitted view keeps its
    # ``model`` colour-group overlay (``panel_list is None``), so only the
    # multi-panel + compare case routes through ``_compose_compare``.
    if isinstance(source, ComparedModelSource) and panel_list is not None:
        return _compose_compare(
            source,
            _residuals_chart_from_source,
            builder_kwargs=builder_kwargs,
            resolve={"x": "shared", "y": "shared"},
        )
    return _residuals_chart_from_source(source, **builder_kwargs)


@_figure_entry
def prediction_error_chart(
    model: Any = None,
    X: Any = None,
    y: Any = None,
    *,
    y_true: Any = None,
    y_pred: Any = None,
    reference_line: bool = True,
    ci: float | None = None,
    reference_band: bool = False,
    compare: dict[str, Any] | None = None,
    random_state: int | None = None,
    mark: dict | None = None,
    encode: dict | None = None,
    properties: dict | None = None,
    layers: list | None = None,
    theme: Any = None,
) -> "Chart | ConcatChart":
    """Actual-vs-predicted scatter for a regression estimator.

    Plots ``y_true`` on the y axis against ``y_pred`` on the x axis with
    an optional reference line (``y = x`` diagonal) and an optional
    residual-based confidence ribbon.

    Parameters
    ----------
    model : estimator or ModelSource
        A fitted sklearn-compatible regression estimator, an explicit
        ``ferrum.ModelSource``, or ``None`` when pre-computed arrays are
        supplied via ``y_true`` / ``y_pred``.
    X : array-like, optional
        Feature matrix. Required when ``model`` is a raw
        estimator; ignored when it is already a ``ModelSource``.
    y : array-like, optional
        Target vector. Required when ``model`` is a raw
        estimator; ignored when it is already a ``ModelSource``.
    y_true : array-like, optional
        Pre-computed true labels. Use with ``y_pred`` to bypass model
        inference entirely.
    y_pred : array-like, optional
        Pre-computed predictions. Use with ``y_true`` to bypass model
        inference entirely.
    reference_line : bool, default True
        Overlay the dashed ``y = x`` diagonal.
    ci : float or None, default None
        Confidence level in ``(0, 1)``. When set, overlays a ribbon
        spanning the central ``ci`` fraction of residuals around the
        reference line. Raises ``ValueError`` if not in ``(0, 1)``.
    reference_band : bool, default False
        When ``True`` (and ``ci`` is ``None``), overlays a ±1 RMSE
        ribbon around the reference line.
    compare : dict of str -> estimator or None, default None
        Mapping of label -> fitted estimator to compare against ``model``.
        Behavior depends on ``ci`` / ``reference_band``:

        - default scatter path (no band): the chart carries a ``model`` colour
          group with one actual-vs-predicted series per model and a shared y=x
          reference line (unchanged single-panel overlay).
        - with ``ci`` or ``reference_band``: returns a
          [ConcatChart][ferrum.ConcatChart] of small multiples, one panel per model,
          with shared x/y scales. Each panel's residual band is computed from
          that model's residuals only — never pooled across models.
    random_state : int or None, default None
        Seed forwarded to ``ModelSource``.
    theme : Theme or None, default None
        Ferrum theme to apply to the returned chart.
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

    Returns
    -------
    Chart or ConcatChart
        Actual-vs-predicted scatter with optional reference line and
        confidence ribbon, or a small-multiples ``ConcatChart`` (one panel per
        model) when ``compare=`` is combined with ``ci`` / ``reference_band``.

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.prediction_error_chart(model, X_test, y_test, reference_line=True)
    """
    source = _resolve_source(
        model,
        X,
        y,
        y_true=y_true,
        y_pred=y_pred,
        random_state=random_state,
        compare=compare,
    )
    # Both the compose path and the single-model fall-through pass identical
    # kwargs to the same builder; only the routing condition differs.
    builder_kwargs = dict(
        reference_line=reference_line,
        ci=ci,
        reference_band=reference_band,
        mark=mark,
        encode=encode,
        properties=properties,
        layers=layers,
        theme=theme,
    )
    # The residual confidence band (``ci`` / ``reference_band``) is a
    # single-model aggregate. Composing one panel per model gives the builder a
    # single-model source for each panel, so every band is computed from that
    # model's residuals only — never pooled across models. The default scatter
    # path (no band) keeps its single-panel ``model`` colour-group overlay.
    if isinstance(source, ComparedModelSource) and (ci is not None or reference_band):
        return _compose_compare(
            source,
            _prediction_error_chart_from_source,
            builder_kwargs=builder_kwargs,
            resolve={"x": "shared", "y": "shared"},
        )
    return _prediction_error_chart_from_source(source, **builder_kwargs)


@_figure_entry
def cooks_distance_chart(
    model: Any = None,
    X: Any = None,
    y: Any = None,
    *,
    threshold: float | str | None = None,
    compare: dict[str, Any] | None = None,
    random_state: int | None = None,
    mark: dict | None = None,
    encode: dict | None = None,
    properties: dict | None = None,
    layers: list | None = None,
    theme: Any = None,
) -> "Chart | ConcatChart":
    """Cook's distance / residuals-vs-leverage diagnostic for a linear estimator.

    Renders the residuals-vs-leverage panel with optional Cook's-distance
    outlier highlighting. Useful for identifying influential observations
    in a linear regression fit.

    Parameters
    ----------
    model : estimator or ModelSource
        A fitted sklearn-compatible regression estimator that exposes
        ``coef_`` (required for leverage-aware Cook's distance), or an
        explicit ``ferrum.ModelSource``.
    X : array-like, optional
        Feature matrix. Required when ``model`` is a raw
        estimator; ignored when it is already a ``ModelSource``.
    y : array-like, optional
        Target vector. Required when ``model`` is a raw
        estimator; ignored when it is already a ``ModelSource``.
    threshold : float, "auto", or None, default None
        Cook's-distance threshold for outlier highlighting. A float is
        used as an absolute cutoff; ``"auto"`` applies the ``4 / n``
        rule (Hair et al.); ``None`` disables highlighting.
    compare : dict of str -> estimator or None, default None
        Mapping of label -> fitted estimator to compare against ``model``.
        When supplied, returns a [ConcatChart][ferrum.ConcatChart] of small
        multiples, one residuals-vs-leverage panel per model, with shared x/y
        scales. Each panel's Cook's distance and leverage are derived from that
        model's own hat matrix only. The single-model path (no ``compare=``) is
        unchanged.
    random_state : int or None, default None
        Seed forwarded to ``ModelSource``.
    theme : Theme or None, default None
        Ferrum theme to apply to the returned chart.
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

    Returns
    -------
    Chart or ConcatChart
        Residuals-vs-leverage panel with optional Cook's-distance outlier
        overlay, or a small-multiples ``ConcatChart`` (one panel per model)
        when ``compare=`` is supplied.

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.cooks_distance_chart(linear_model, X_test, y_test, threshold="auto")
    """
    source = _resolve_source(
        model,
        X,
        y,
        random_state=random_state,
        compare=compare,
    )
    builder_kwargs = dict(
        kind="studentized",
        cook_threshold=threshold,
        panels=["residuals_vs_leverage"],
        mark=mark,
        encode=encode,
        properties=properties,
        layers=layers,
        theme=theme,
    )
    if isinstance(source, ComparedModelSource):
        return _compose_compare(
            source,
            _residuals_chart_from_source,
            builder_kwargs=builder_kwargs,
            resolve={"x": "shared", "y": "shared"},
        )
    return _residuals_chart_from_source(source, **builder_kwargs)
