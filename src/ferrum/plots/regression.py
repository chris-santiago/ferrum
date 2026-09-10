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

from dataclasses import replace
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
from ferrum._core import transform_registry
from ferrum._desugar import _propagate_layer_scale
from ferrum._layer import _Layer
from ferrum._registry import registry_output_columns
from ferrum._validate import validate_choice
from ferrum.diagnostics.source import ComparedModelSource
from ferrum.encoding import Color, X, Y
from ferrum.encoding._aliases import resolve_color_alias
from ferrum.marks._desugar_helpers import nominal_color_channel
from ferrum.marks._fit_layers import build_fit_layers
from ferrum.marks.statistical import desugar_smooth
from ferrum.plots._helpers import (
    _color_field_for,
    _column_or_literal_color,
    _compose_compare,
    _field_name,
    _finalize_chart,
    _grid_panels,
    _inject_cook_outliers,
    _inject_metrics_corner,
    _merge_layers,
    _overlay_metrics_corner,
    _resolve_source,
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
        if cook_threshold is not None:
            df = _inject_cook_outliers(df, kind=kind, threshold=cook_threshold)
        # When a ``model`` column is present (compare-source path), drive
        # per-model colour on the residual scatter; absent it, ``None`` is
        # byte-identical to the single-model default.
        chart = ferrum.Chart(df).mark_residuals(
            kind=kind,
            cook_threshold=cook_threshold,
            color_field=_color_field_for(df, None),
        )
        chart = chart.properties(
            title=ferrum.Title("Residuals", subtitle=subtitle),
        )
        # ``_overlay_metrics_corner`` is a no-op when the injected
        # columns are absent, so the call is unconditional.
        chart = _overlay_metrics_corner(chart)
        return _finalize_chart(
            chart, mark=mark, encode=encode, properties=properties, layers=layers, theme=theme
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
        chart, mark=mark, encode=encode, properties=properties, layers=layers, theme=theme
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
    residuals_vs_leverage panels overlay an outlier-highlight layer
    keyed on each panel's own x axis (y_pred for residuals_vs_fitted,
    leverage for residuals_vs_leverage).
    """
    import ferrum

    y_col = "studentized_residual" if kind in ("studentized", "scaled") else "residual"

    if name == "residuals_vs_fitted":
        if cook_threshold is not None:
            df = _inject_cook_outliers(df, kind=kind, threshold=cook_threshold)
            chart = ferrum.Chart(df).mark_residuals(
                kind=kind,
                cook_threshold=cook_threshold,
            )
        else:
            chart = ferrum.Chart(df).mark_residuals(kind=kind)
        chart = chart.encode(
            x=X("y_pred", title="Fitted values"),
            y=Y(y_col, title="Studentized residual"),
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
        # `sample` is the QQ transform's declared output column this panel
        # reads back on the y channel; derived from the registry (spec §4.6 /
        # M6a) so a Rust-side rename raises here.  The x channel names the
        # panel's own input column, not a transform output, so it stays a
        # plain literal.
        _qq_cols = registry_output_columns("qq", ["sample"])
        return (
            ferrum.Chart(df)
            .mark_qq()
            .encode(
                x=X("studentized_residual", title="Theoretical quantiles"),
                y=Y(_qq_cols["sample"], title="Sample quantiles"),
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
        # Inject outlier columns BEFORE constructing the base chart so
        # base and overlay layers share the same DataFrame identity
        # (avoids unnecessary null-pad merge in Chart.__add__).
        # x_col="leverage" so the overlay sits at the right
        # x coordinate for this panel.
        if cook_threshold is not None:
            df = _inject_cook_outliers(
                df,
                kind=kind,
                threshold=cook_threshold,
                x_col="leverage",
            )
        base = (
            ferrum.Chart(df)
            .mark_point()
            .encode(
                x=X("leverage", title="Leverage"),
                y=Y(y_col, title="Studentized residual"),
            )
            .properties(title=ferrum.Title("Residuals vs Leverage"))
        )
        if cook_threshold is not None and "_cook_outlier_x" in df.columns:
            overlay = (
                ferrum.Chart(df)
                .mark_point(
                    fill="#e15759",  # tableau red, matches mark_residuals overlay
                    stroke="#000000",
                    stroke_width=1.0,
                    size=80.0,
                )
                .encode(x="_cook_outlier_x", y="_cook_outlier_y")
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
        chart, mark=mark, encode=encode, properties=properties, layers=layers, theme=theme
    )


def _fit_chart_from_named_transform(
    data: Any,
    transform: Any,
    *,
    kind: str,
    ci: float | None,
    line_kwargs: dict,
    chart_x: Any,
    neutral_stroke: bool = False,
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

    ``neutral_stroke``: a POOLED fit -- ``Logistic``/``Glm``, which take no
    ``groupby`` at all, and ``Robust``, which has one that ``lmplot`` does
    not yet wire ``hue=`` into. When any channel that resolves to ``color``
    is active on the sibling scatter layer -- ``hue=``, or any of
    ``color=``/``fill=``/``stroke=`` in keyword or ``encode=`` dict form,
    resolved once as ``lmplot``'s ``color_active`` through
    ``ferrum.encoding._aliases.resolve_color_alias`` so a future alias needs
    no change here -- that pooled fit must not visually impersonate one color
    level, so its band and line take the neutral theme token instead. See
    ``build_fit_layers``.

    ``metrics`` appends the corner-statistics text layer, only meaningful
    when ``transform`` was itself built with ``inject_metrics=True`` (only
    ``Robust`` of the three dedicated fit transforms supports it; see
    ``lmplot``'s ``metrics_applied`` computation).

    ``chart_x`` is the chart-level x channel the caller would have encoded.
    The layers below read the transform's own output columns, so an explicit
    ``scale=`` on that channel (``lmplot(logx=True)``) has to be propagated
    onto them -- the step ``_resolve_pending_impl`` performs automatically for
    a chart built through the desugar path. Doing it here, at construction,
    is what keeps ``logx`` ONE mechanism: the alternative was a second
    implementation in ``lmplot``, re-running the propagation by hand for
    charts this function had left in a state the desugar pipeline never
    blessed, and a third for whatever chart-level feature came next.
    ``_propagate_layer_scale`` returns each layer unchanged when the channel
    carries no explicit scale, so the ordinary case is a no-op rather than a
    branch.
    """
    chart = Chart(data)._clone()
    chart._transforms = [transform]
    chart._layers = [
        _propagate_layer_scale({"x": chart_x}, layer)
        for layer in build_fit_layers(
            kind,
            data_source=kind,
            ci=ci,
            line_kwargs=line_kwargs,
            neutral_stroke=neutral_stroke,
            metrics=metrics,
        )
    ]
    chart._mark = None
    return chart


# ---------------------------------------------------------------------------
# Public figure functions
# ---------------------------------------------------------------------------


def lmplot(
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
    show_metrics: bool = True,
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
    * ``"logistic"`` -- ``Logistic`` transform + pooled fit line + CI band.
    * ``"glm"``      -- ``Glm`` transform + pooled fit line + CI band.
    * ``"robust"``   -- ``Robust`` transform + pooled fit line + CI band.

    ``"logistic"``/``"glm"``/``"robust"`` fit **one pooled curve across the
    whole dataset**, unlike ``mark_smooth``'s ``"lm"``/``"loess"``/etc.
    ``Logistic`` and ``Glm`` take no ``groupby`` at all; ``Robust`` gained one
    (``residplot(color=)`` uses it), so its pooling is a wiring gap here
    rather than a missing capability -- tracked as a follow-up. When any
    channel that resolves to ``color`` is active -- ``hue=``, a bare ``color=``/
    ``fill=``/``stroke=`` keyword, or the ``encode=`` dict equivalent -- the
    scatter is still colored per level, but the fit line and CI band render
    in a neutral stroke/fill (see ``hue`` below), not a level's color.

    Parameters
    ----------
    data : DataFrame-like
        Input data accepted by ``Chart(data)``.
    x : str
        Column name for the horizontal (predictor) axis (required).
    y : str
        Column name for the vertical (response) axis (required).
    hue : str or encoding, optional
        Column name to map to color. For every ``mark_smooth``-routed
        ``method`` (see ``method`` below), the fit is refit per hue level via
        ``groupby``, so each level gets its own fit line and CI band in that
        level's scatter color. For ``"logistic"``, ``"glm"``, ``"robust"``,
        the scatter is colored per hue level but the fit is a single curve
        pooled across every group,
        rendered in a neutral stroke/fill distinct from the palette -- it
        does not represent any one group's fit. This pooled-neutral behavior
        for the three dedicated methods is not specific to ``hue=`` -- it
        applies identically whenever any channel that resolves to ``color``
        reaches the chart (``hue=``, a bare ``color=``/``fill=``/``stroke=``
        keyword, or ``encode={"color": ...}``/``encode={"fill": ...}``/
        ``encode={"stroke": ...}``), since all are the same "one pooled fit,
        colored scatter" situation. Per-group fits are not built for any of
        the three: ``"logistic"``/``"glm"`` need ``groupby`` support on the
        underlying Rust transform first, while ``"robust"`` has it already
        and needs only the wiring here. Both tracked as follow-ups.
    col : str, optional
        Column name for faceting across columns.
    row : str, optional
        Column name for faceting across rows.
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
        across the axis-domain extent (the data range padded 5% on each
        side) via the transform's ``x_range`` support (spec §4 WI-7) --
        supported for every ``mark_smooth``-routed method and for
        ``method="robust"``. Not supported for ``method="logistic"`` or
        ``"glm"`` (neither transform accepts ``x_range``); passing
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
        The aggregate applied within each ``x_bins`` bin (``"mean"``,
        ``"median"``, ``"sum"``, ``"min"``, ``"max"``; defaults to the
        transform's own ``"mean"``). Forwarded on the same terms as
        ``x_bins``, and only meaningful alongside it. Same
        ``"logistic"``/``"glm"``/``"robust"`` refusal as ``x_bins`` above
        applies on its own, even without ``x_bins=`` set.
    x_jitter : float or None, optional
        When set, applies ``Jitter(axis="x", width=x_jitter)`` to the
        scatter layer.
    logx : bool, default False
        Apply a ``log`` scale to the x-axis on both scatter and fit layers.
    show_metrics : bool, default True
        Overlay a top-right corner annotation with ``R^2`` / ``RMSE`` /
        ``MAE`` computed from the fit. ``inject_metrics`` is a generic
        ``Smooth``/``Robust`` transform kwarg (computed against the raw
        input regardless of fit shape), so this applies to every
        ``mark_smooth``-routed method and to ``method="robust"``. Not supported for
        ``method="logistic"``/``"glm"`` (neither transform exposes
        ``inject_metrics``) -- skipped for those two regardless of this
        flag. Also skipped whenever ``hue`` is set, for every method
        (per-group corners would crowd).
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
        Layered chart (scatter + fit) or fit-only when ``scatter=False``.
        May be faceted.

    Raises
    ------
    ValueError
        If ``method`` is not one of the supported values.
    ValueError
        If ``truncate=False`` with ``method="logistic"`` or ``"glm"``
        (neither transform accepts ``x_range``).
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

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.lmplot(df, x="total_bill", y="tip")

    Logistic regression, scatter colored per sex (the fit is pooled across
    both sexes -- ``Logistic`` has no per-group ``groupby``):

    >>> fm.lmplot(df, x="total_bill", y="smoker_int", method="logistic", hue="sex")

    Polynomial fit (degree 2) with no confidence band:

    >>> fm.lmplot(df, x="size", y="tip", order=2, ci=None)
    """
    validate_choice("lmplot", "method", method, _VALID_METHODS)

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
            f"lmplot(method={method!r}, {bad}=...) is not "
            f"supported -- the {method!r} transform has no x_bins/x_estimator "
            "binning parameters to pre-aggregate the fit input. Use a "
            "mark_smooth-routed method (lm, loess, quadratic, cubic, log, "
            "sqrt, gam) or leave x_bins=None (the default)."
        )

    data = to_polars(data)

    # Normalize CI: spec accepts ci=95 (percent) or ci=None.
    ci_frac = (ci / 100.0) if ci is not None else None

    # Schwabish SB-followup (2026-05-12): the corner-metrics overlay is
    # supplied by the Smooth/Robust Rust transform via
    # ``inject_metrics=True`` (threaded through ``mark_smooth
    # (show_metrics=True)`` for smooth-routed methods, and directly on
    # ``Robust(...)`` below for ``method="robust"``), so R²/RMSE/MAE is
    # computed once in Rust and surfaced as ``_metrics_text`` /
    # ``_metrics_y`` columns on the fit-grid output — no Python-side
    # duplication. ``inject_metrics`` is generic (verified against
    # ``smooth.rs``: every ``SmoothMethod`` arm accepts it, computed
    # against the raw input regardless of fit shape), so this applies
    # uniformly to every smooth-routed method and to ``"robust"``.
    # ``"logistic"``/``"glm"`` expose no ``inject_metrics`` param at all
    # (``transform_registry`` lists none), so those two stay excluded
    # regardless of ``show_metrics=``. Restricted to hue=None across every
    # method: per-group corners would crowd.
    metrics_applied = show_metrics and hue is None and method not in ("logistic", "glm")

    # truncate: True = clip to data range (current Smooth/Robust default).
    # False = extend fit line to the full x-axis domain boundary via
    # SmoothSpec/RobustSpec.x_range (spec §4 WI-7 — wired end-to-end for
    # every mark_smooth-routed method and for "robust"). "logistic"/"glm"
    # accept no x_range at all (transform_registry confirms neither lists
    # it), so extending their fit line past the data range isn't possible
    # without a Rust-side transform change — truncate=False refuses for
    # those two rather than silently clipping anyway -- no silent parameter
    # drop.
    # The x_range is resolved from the chart's x column min/max at Python
    # desugar time; the Rust transform uses it as the evaluation domain.
    # When truncate=True (default) x_range stays None → clips to observed data.
    x_range = None
    if not truncate:
        if method in ("logistic", "glm"):
            raise ValueError(
                f"lmplot(method={method!r}, truncate=False) is not "
                f"supported -- the {method!r} transform has no x_range "
                "parameter to extend the fit line past the observed data "
                "range. Use method='robust' (accepts x_range), a "
                "mark_smooth-routed method (lm, loess, quadratic, cubic, "
                "log, sqrt, gam), or leave truncate=True (the default)."
            )
        # Compute the axis-domain extent here so lmplot does not need a
        # provisional scale pass. The Rust Smooth/Robust transform will
        # evaluate the fit line over [x_min, x_max] of the full dataset
        # regardless of which rows each facet panel sees.
        # seaborn behaviour: extend to the axis limits, which are the data
        # range padded by 5% on each side (matching the typical auto-scale margin).
        try:
            _df = data.collect() if hasattr(data, "collect") else data
            x_col_name = _field_name(x)
            if hasattr(_df, "to_arrow"):
                _x_series = _df[x_col_name]
                _x_min = float(_x_series.min())
                _x_max = float(_x_series.max())
                _margin = (_x_max - _x_min) * 0.05
                x_range = [_x_min - _margin, _x_max + _margin]
        except (AttributeError, KeyError):
            x_range = None

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
    # Until this batch the defect was visible on 3 of the 4 ci/show_metrics
    # combinations and hidden on the 4th only because the scatter layer was
    # reading the fit's own (Float64) grid there -- the corruption F-L10-02
    # closes. DELETE CONDITION: when `apply_jitter` coerces the column it
    # writes, the way the transform engine already coerces its inputs
    # (`numeric_util::coerce_to_float64`, batch D M1). `catplot(native_scale=
    # True)` carries the same workaround for the same renderer behavior.
    if scatter and x_jitter is not None:
        jitter_frame = to_polars(data)
        jitter_column = _field_name(x)
        dtype = jitter_frame.schema.get(jitter_column)
        if dtype is not None and dtype.is_numeric() and dtype != pl.Float64:
            data = jitter_frame.with_columns(pl.col(jitter_column).cast(pl.Float64))

    # Shared encoding.
    #
    # `hue` is typed Nominal once here and reused on every layer below. It is
    # a group discriminator: it drives Smooth's `groupby=`, so one fit and one
    # CI band are produced per level. Left to infer, a numeric hue column read
    # as Continuous collapsed all of those back into a single rendered fit and
    # a single band -- a materially wrong statistical chart, drawn silently.
    # The scatter layer takes the same typing rather than the `point` carve-out
    # because it shares one color scale with the fit layers in the same layered
    # chart; typing only the fit half would ask that one scale to be Continuous
    # and Nominal at once. See `nominal_color_channel` for the rule.
    hue_ch = nominal_color_channel(hue)

    # `logx` is part of the x channel every layer is built with, not a
    # post-hoc fixup applied to finished charts: resolving it once here means
    # the scatter, the `mark_smooth`-routed fit and the dedicated-transform
    # fit all take the same channel object. (It used to be two
    # implementations at the bottom of this function, chosen on whether the
    # fit chart happened to have `_layers` yet.)
    x_enc = X(x, scale={"type": "log"}) if logx else x

    enc: dict = {"x": x_enc, "y": y}
    if hue is not None:
        enc["color"] = hue_ch
    enc.update(encode_kwargs)

    # A pooled dedicated-fit method (logistic/glm/robust) can impersonate a
    # color level through every channel the repo's own alias policy resolves
    # onto `color`, not just the literal `color=` spelling:
    # `ferrum.encoding._aliases._ENCODING_ALIASES` is the single table of
    # which channels drive `color` (`fill` -- keep_target, `stroke` --
    # warn_drop), and `hue=`/`color=`/`fill=`/`stroke=` each reach this chart
    # two ways (a bare keyword via `**encode_kwargs`, merged into `enc`
    # above, or the separate `encode=` dict `_finalize_chart` applies later).
    # Resolving through `resolve_color_alias` over the merged effective
    # encoding -- the same alias table, already read for exactly this "what
    # would color resolve to" question by `composition.py`'s layer-color
    # promotion -- covers a future alias automatically, with no enumeration
    # here to fall behind.
    color_active = resolve_color_alias({**enc, **(encode or {})}) is not None

    # ---- Scatter layer (optional) --------------------------------------
    scatter_layer = None
    if scatter:
        s = Chart(data)
        skw = dict(scatter_kws) if scatter_kws else {}
        if x_jitter is not None:
            s = s.mark_point(position=Jitter(axis="x", width=float(x_jitter)), **skw)
        else:
            s = s.mark_point(**skw)
        s = s.encode(**enc)
        scatter_layer = s

    # ---- Fit layer (per method) ----------------------------------------
    lkw = dict(line_kws) if line_kws else {}

    # Every `_SMOOTH_ROUTED_METHODS` member takes ONE call: the parameters
    # that vary do so with the *arguments*, not with the method.
    #
    # - `x_bins`/`x_estimator` forward for every method. The engine bins
    #   method-agnostically -- `smooth.rs`'s `apply_one_group` aggregates the
    #   x bins BEFORE it dispatches on method -- so the pre-R2 "only lm
    #   forwards binning" carve-out dropped both parameters silently for nine
    #   of lmplot's ten methods, which is the shape this function refuses
    #   everywhere else (`truncate=False` and an inapplicable `order` both
    #   raise). Same rule as `order`: forward what the caller passed and let
    #   the engine decide.
    # - `order` forwards as `degree=` via `_smooth_degree_kwargs`, whose
    #   docstring holds the per-method acceptance matrix.
    # - `groupby=hue` fits per hue level and colors each fit by it.
    # - `show_metrics` needs no gate here: `metrics_applied` is already False
    #   whenever `hue` is set (and for logistic/glm), computed above.
    if method in _SMOOTH_ROUTED_METHODS:
        smooth_kwargs: dict = dict(
            method=method,
            ci=ci_frac,
            x_bins=x_bins,
            x_estimator=x_estimator,
            show_metrics=metrics_applied,
            **({"groupby": _field_name(hue)} if hue is not None else {}),
            **_smooth_degree_kwargs(method, order, "lmplot"),
            **({"x_range": x_range} if x_range is not None else {}),
        )
        if scatter_layer is None:
            # No sibling layer: the fit IS the chart, so `mark_smooth`'s own
            # shape is the right one -- including the single-line shortcut it
            # takes for `ci=None, show_metrics=False`, whose chain-advancing
            # (unnamed) `Smooth` correctly becomes the chart's data, and whose
            # chart-level encoding is what titles the axes with the caller's
            # own column names.
            fit = Chart(data).mark_smooth(**smooth_kwargs)
            fit = (
                fit.encode(x=x_enc, y=y, color=hue_ch)
                if hue is not None
                else fit.encode(x=x_enc, y=y)
            )
        else:
            # A scatter layer will be composed beside this fit, so the fit's
            # transform must be a NAMED fan-out its own layers read through
            # `data_source=` -- never a chain-advancing transform, which
            # `_merge_layers` would apply to the scatter layer too. That is
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
                _field_name(x), _field_name(y), name="smooth", **smooth_kwargs
            ).transforms
            fit = _fit_chart_from_named_transform(
                data,
                smooth_transform,
                # `kind` doubles as the transform's `name=` here exactly as it
                # does for the three dedicated methods below.
                kind="smooth",
                ci=ci_frac,
                # `line_kws` is applied to the emitted line layer below, in the
                # one block that already owns it for these methods.
                line_kwargs={},
                chart_x=x_enc,
                metrics=metrics_applied,
            )
    elif method == "logistic":
        fit = _fit_chart_from_named_transform(
            data,
            Logistic(x=x, y=y, n_grid=200, ci=ci_frac, name="logistic"),
            kind="logistic",
            ci=ci_frac,
            line_kwargs=lkw,
            chart_x=x_enc,
            neutral_stroke=color_active,
        )
    elif method == "glm":
        fit = _fit_chart_from_named_transform(
            data,
            Glm(x=x, y=y, family="gaussian", n_grid=200, ci=ci_frac, name="glm"),
            kind="glm",
            ci=ci_frac,
            line_kwargs=lkw,
            chart_x=x_enc,
            neutral_stroke=color_active,
        )
    elif method == "robust":
        fit = _fit_chart_from_named_transform(
            data,
            Robust(
                x=x,
                y=y,
                n_grid=200,
                ci=ci_frac,
                name="robust",
                inject_metrics=metrics_applied,
                **({"x_range": x_range} if x_range is not None else {}),
            ),
            kind="robust",
            ci=ci_frac,
            line_kwargs=lkw,
            chart_x=x_enc,
            neutral_stroke=color_active,
            metrics=metrics_applied,
        )
    else:  # pragma: no cover — guarded above
        raise ValueError(f"unreachable: method={method!r}")

    # `_DEDICATED_FIT_METHODS` (logistic/glm/robust) are called here WITHOUT a
    # `groupby`, so exactly one pooled fit is produced regardless of whether a
    # color encoding is active. `neutral_stroke=color_active` above is what
    # stops that pooled fit from rendering in a color level's palette color
    # beside a full legend -- a materially wrong statistical chart, drawn
    # silently, and the same failure class `nominal_color_channel` names for a
    # mistyped hue column. `Robust` did gain a Rust-side `groupby` in batch F
    # (`residplot(color=)` consumes it), so wiring `hue=` through to per-group
    # robust fits is now a Python-side change here; `Logistic`/`Glm` still need
    # the Rust capability first. Both tracked as batch-close follow-ups.

    # Compose scatter + fit. (When ``metrics_applied``, the fit chart
    # already carries the metrics-text layer reading the Smooth output.)
    if scatter_layer is None:
        out = fit
    else:
        out = _merge_layers(scatter_layer, fit, scatter_name="scatter", fit_name="fit")

    # Apply line_kws to the `mark_smooth`-routed methods, where it cannot be
    # passed through `mark_smooth` directly (its mark kwargs reach every
    # emitted layer, band included). The dedicated fit-family methods already
    # applied it as the line layer's own mark_kwargs, in `build_fit_layers`.
    if lkw and method in _SMOOTH_ROUTED_METHODS and out._layers:
        out._layers = [
            replace(layer, mark_kwargs={**(layer.mark_kwargs or {}), **lkw})
            if getattr(layer, "mark", None) == "line"
            else layer
            for layer in out._layers
        ]

    # Faceting.
    if col is not None or row is not None:
        if col is not None and row is not None:
            out = out.facet(row=row, col=col)
        elif col is not None:
            out = out.facet(col=col)
        else:
            out = out.facet(row=row)

    return _finalize_chart(
        out, mark=mark, encode=encode, properties=properties, layers=layers, theme=theme
    )


def residplot(
    data: Any,
    *,
    x: str,
    y: str,
    lowess: bool = False,
    order: int = 1,
    robust: bool = False,
    dropna: bool = True,
    show_metrics: bool = True,
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
    x : str
        Column name for the horizontal axis (predictor; required).
    y : str
        Column name used to compute residuals (response; required).
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
    show_metrics : bool, default True
        Schwabish SB-followup (2026-05-12): overlay a top-right corner
        annotation with ``R^2`` / ``RMSE`` / ``MAE`` computed inside the
        Rust Smooth/Robust transform via the ``inject_metrics=True``
        kwarg -- same single execution model as fitted residuals, no
        Python-side regression duplication. Suppressed when ``color=``
        names a column, mirroring ``lmplot``'s ``hue`` exclusion: those
        residuals are refit per group, so there is no single corner value
        to report. ``label=`` does not suppress it -- that grouping has
        one level by construction, so the corner stays correct.
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
        groups rather than paints. Otherwise the value is parsed as a
        color (CSS name, hex, ``rgb()``/``rgba()``, or an RGB tuple) and
        painted onto the residual points as a constant ``fill``; combined
        with ``label=``, that same color is also the legend swatch. A
        value that is neither raises ``ValueError`` naming both readings.
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
    if dropna:
        data = data.drop_nulls(subset=[x, y])

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
    # A column reaches the residual output only by being in ``groupby``: the
    # transforms emit their declared columns plus the group keys, nothing
    # else. So ``label=`` injects a constant ``_label`` column and groups by
    # it, and a grouping ``color=`` joins the same list — both Smooth and
    # Robust take a list of group columns (Robust gained ``groupby`` in
    # batch F), so the two compose and residuals are refit per group.
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

    # `residplot` passes `degree=order` straight to `Smooth` below, so it
    # needs the same typed refusal `lmplot` gets from `_smooth_degree_kwargs`
    # -- called for the raise, since the kwargs it builds are not the ones
    # used here. The `Robust` path takes no `order`, so nothing to validate.
    if not robust:
        _smooth_degree_kwargs("lm", order, "residplot")

    # `_resid_x` is a bare literal, deliberately NOT a registry lookup
    # (spec §4.6 / M6a). `Smooth`/`Robust` in `output="residuals"` mode build
    # their batch via `build_residuals_batch` (`smooth.rs`, `robust.rs`),
    # which declares its own `x` column uncoupled from `FIT_OUTPUT_COLUMNS`
    # -- the `output="fitted"` schema `transform_registry("smooth"|"robust")`
    # actually declares. Checking `_resid_x` against that schema would pass
    # today (both happen to list `x`) and stay silently green if the
    # residuals batch's own `x` were renamed independently, while raising
    # here on an unrelated `FIT_OUTPUT_COLUMNS` rename that never touched
    # this path: a lookup against the wrong schema is worse than no lookup.
    # Recorded as a dispositioned column reference in
    # `tests/test_registry_output_column_consumers.py`'s `_DISPOSITIONED`
    # (site: `("src/ferrum/plots/regression.py", "residplot", "x")`), which
    # fails if this code ever stops needing the disposition. The end state --
    # Rust declaring a residuals-mode output schema, the way it now declares
    # the secondary batches -- is a Rust change, tracked as a batch-close
    # follow-up.
    _resid_x = "x"

    # Mirrors `lmplot`'s `metrics_applied`: the corner annotation is one
    # R²/RMSE/MAE for one fit, so a residplot refit per data group has no
    # single corner to render, and labelling one group's metrics as the
    # chart's would be wrong rather than crowded.
    #
    # Gated on the `color=` grouping ONLY, not on `groupby` as a whole. A
    # `label=` grouping is one level by construction (`pl.lit`), so the
    # transform emits exactly one corner for it and the reason above simply
    # does not apply -- suppressing there would take the R²/RMSE/MAE away
    # from a plain `residplot(..., label="…")` for no benefit. That is the
    # same distinction `lmplot` draws: it gates on a real multi-level `hue`.
    #
    # It gates the transform's `inject_metrics` too, so the columns the
    # metrics layer reads exist exactly when that layer is built.
    metrics_applied = show_metrics and color_arg.group_field is None

    # Both residual transforms take the same residual-mode arguments; only
    # the fit family differs (and `degree`, which Robust has no analogue for).
    resid_kwargs: dict = dict(
        x=x,
        y=y,
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

    # A literal `color=` paints the residual points through the point mark's
    # own kwargs -- never through `encode(color=)`, which would fabricate a
    # one-level color scale and a legend for what the caller asked to be a
    # constant. Same neutral-paint idiom `build_fit_layers` uses.
    point_kwargs: dict = {} if color_arg.paint is None else {"fill": color_arg.paint}

    chart = Chart(data).transform(resid_transform).mark_point(**point_kwargs)
    enc: dict = {"x": _resid_x, "y": "residual"}
    if color is not None:
        enc["color"] = color
    elif label_paint_channel is not None:
        enc["color"] = label_paint_channel
    elif label is not None:
        enc["color"] = "_label"
    enc.update(encode_kwargs)
    chart = chart.encode(**enc)

    # Build the layered chart when any overlay is requested (zero rule,
    # metrics text, or lowess smoother). The augmented-DataFrame pattern
    # works because all overlays reference columns the Rust transform
    # already emitted into the same residuals batch.
    layered = lowess or zero_line or metrics_applied
    if layered:
        chart = chart._clone()
        internal_layers: list = [
            _Layer(
                name="point",
                mark="point",
                encoding=dict(enc),
                mark_kwargs=point_kwargs or None,
            )
        ]
        if zero_line:
            internal_layers.append(
                _Layer(
                    name="reference",
                    mark="rule",
                    encoding={"y": "_ref_zero"},
                    mark_kwargs={"stroke_dash": [3, 3], "stroke": "#8a8a8a"},
                )
            )
        if metrics_applied:
            internal_layers.append(
                _Layer(
                    name="metrics",
                    mark="text",
                    encoding={"x": _resid_x, "y": "_metrics_y", "text": "_metrics_text"},
                    mark_kwargs={"align": "right", "dx": -4, "dy": 4},
                )
            )
        if lowess:
            chart._transforms = [
                resid_transform,
                Smooth(
                    x=_resid_x,
                    y="residual",
                    method="loess",
                    ci=None,
                    name="lowess",
                    # The overlay reads the residual batch, which carries the
                    # group keys, so a grouped residplot smooths each group's
                    # own residuals -- one curve per group instead of one
                    # curve through the pooled cloud.
                    **({"groupby": groupby} if groupby else {}),
                ),
            ]
            # Derived from the registry (spec §4.6 / M6a) rather than
            # hardcoded: this layer reads the named `lowess` Smooth
            # transform's fit-grid output, so a Rust-side rename raises here
            # instead of silently drawing nothing.
            _lowess_cols = registry_output_columns("smooth", ["x", "y"])
            lowess_enc: dict = {"x": _lowess_cols["x"], "y": _lowess_cols["y"]}
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

    return _finalize_chart(
        chart, mark=mark, encode=encode, properties=properties, layers=layers, theme=theme
    )


def regplot(
    data: Any,
    *,
    x: str,
    y: str,
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
    show_metrics: bool = True,
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
    x : str
        Column name for the horizontal (predictor) axis (required).
    y : str
        Column name for the vertical (response) axis (required).
    hue : str or encoding, optional
        Column name to map to color, forwarded to ``lmplot`` -- see
        ``lmplot``'s ``hue`` parameter: fit lines are drawn per hue level
        for ``mark_smooth``-routed methods, but pooled (neutral stroke, not
        a color) for ``method="logistic"``/``"glm"``/``"robust"`` -- pooled
        whenever any channel that resolves to ``color`` reaches the chart
        (``hue=``, ``color=``/``fill=``/``stroke=``, or their ``encode=``
        dict equivalents).
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
        forwarded to ``lmplot`` -- see ``lmplot``'s ``logx`` parameter.
    show_metrics : bool, default True
        Overlay a top-right corner annotation with ``R^2`` / ``RMSE`` /
        ``MAE`` computed from the fit, forwarded to ``lmplot`` -- see
        ``lmplot``'s ``show_metrics`` parameter for the method/``hue``
        exclusions.
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
        Layered chart (scatter + fit) or fit-only when ``scatter=False``.

    Raises
    ------
    TypeError
        If ``col=`` or ``row=`` is passed -- ``regplot`` does not facet;
        use ``lmplot`` instead.
    ValueError
        If ``method`` is not one of the supported values.
    ValueError
        If ``truncate=False`` with ``method="logistic"`` or ``"glm"``
        (neither transform accepts ``x_range``).
    ValueError
        If ``x_bins`` or ``x_estimator`` is set with
        ``method="logistic"``, ``"glm"``, or ``"robust"`` -- forwarded from
        ``lmplot``, see its ``x_bins`` parameter.
    ValueError
        If an explicit ``order`` is not valid for ``method``, or is a
        negative int -- forwarded from ``lmplot``, see its ``order``
        parameter.

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
    return lmplot(
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
    if panels in (None, "single"):
        panel_list: Any = None
    elif panels == "auto":
        panel_list = [
            "residuals_vs_fitted",
            "qq",
            "scale_location",
            "residuals_vs_leverage",
        ]
    else:
        panel_list = list(panels)
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
