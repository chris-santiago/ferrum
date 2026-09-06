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

from typing import TYPE_CHECKING, Any

import polars as pl

from ferrum import (
    Chart,
    Glm,
    Jitter,
    Logistic,
    Robust,
    Smooth,
)
from ferrum._coerce import to_polars
from ferrum._core import transform_registry
from ferrum._layer import _Layer
from ferrum._validate import validate_choice
from ferrum.diagnostics.source import ComparedModelSource
from ferrum.encoding import X, Y
from ferrum.marks._desugar_helpers import nominal_color_channel, registry_output_columns
from ferrum.plots._helpers import (
    _color_field_for,
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


# Batch-D T13: closes the T12 spec-review obligation that `_VALID_METHODS`'
# relationship to the registry stay honest (`.sdd/task-12-spec-reviewer.md`
# cycle 5's carried ⚠️). `_SMOOTH_ROUTED_METHODS` is DERIVED from
# `transform_registry("smooth")["methods"]` -- every registry smooth
# spelling routes through `mark_smooth` below except "linear" (the same
# method as "lm", just a second registry alias lmplot doesn't separately
# expose) and "glm" (kept out of the derived set on purpose: see
# `_DEDICATED_FIT_METHODS` below). `_DEDICATED_FIT_METHODS` are the three
# methods with their own fit-family transform (`Logistic`/`Glm`/`Robust`),
# all sharing the `x, y, ci_lower, ci_upper` registry output schema (spec
# §4.7/M6c). "glm" is a name both sets could plausibly claim -- lmplot keeps
# it mapped to the standalone `Glm` transform (its own `family=`/`link=`
# surface, matching the top-level `glm()`/`Glm` API) rather than
# `Smooth(method="glm")`'s grid-only surface; T9 hoisted the shared
# `wald_band` helper so the two implementations' CI math cannot drift apart,
# so this is a choice of which surface owns the name, not a divergent
# implementation. `test_lmplot_valid_methods_registry_pin.py` pins both
# halves against the registry, so a rename or a new registry smooth method
# is caught rather than silently under/over-widening `_VALID_METHODS`.
_SMOOTH_ROUTED_METHODS = frozenset(transform_registry("smooth")["methods"]) - {"linear", "glm"}
_DEDICATED_FIT_METHODS = frozenset({"logistic", "glm", "robust"})
_VALID_METHODS = _SMOOTH_ROUTED_METHODS | _DEDICATED_FIT_METHODS

# batch-D T13 round-3 (`.sdd/task-13-python-quality-reviewer.md` cycle-2
# finding 2): `order=` forwarding is now ONE rule applied uniformly to every
# `_SMOOTH_ROUTED_METHODS` member, not a per-method carve-out -- see
# `_smooth_degree_kwargs`.


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
    diagnosable message (batch-D T13 round-4 spec-review ⚠️).

    History: prior rounds used ``order == 1`` (``lmplot``'s then-default) as
    the "no explicit degree" sentinel, which made ``order=1`` structurally
    unable to mean "forward degree=1" for *any* method -- silently wrong for
    ``"loess"``, whose own local-fit degree=1 is a real, different fit from
    its degree=2 default (round-4 spec-review finding: the docstring claimed
    ``order=1`` was a harmless loess no-op, which was false). Round 4 moves
    the sentinel to ``lmplot``'s ``order=None`` default instead, so
    "explicit value present" and "value equals 1" are no longer conflated
    -- every explicit value, including 1, now reaches ``Smooth``'s own
    degree-acceptance matrix uniformly, and a bare call (``order`` omitted)
    is the only shape that forwards nothing.
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
    neutral_stroke: bool = False,
    metrics: bool = False,
) -> Chart:
    """Build a ribbon(optional)+line(+metrics, optional) layered ``Chart``
    reading a named fit-family transform's output via the registry
    (spec §4.7/M6c).

    Mirrors the ribbon+line shape ``desugar_smooth`` (``marks/statistical.py``)
    builds for ``mark_smooth``, applied here for the ``logistic``/``glm``/
    ``robust`` transforms that ``lmplot`` constructs directly rather than
    through ``mark_smooth``'s own ``_PendingMark`` desugar. ``kind`` doubles
    as the transform's own ``name=`` (both call sites below pass the same
    string for both), so the layers' ``data_source`` can reference it without
    a separate name parameter.

    Fixes the T9-found bug: the transform's real fitted-curve schema is
    ``x, y, ci_lower, ci_upper`` (``FIT_OUTPUT_COLUMNS``, shared with
    ``Smooth``), not ``fitted`` -- encoding against the stale column name
    made every ``logistic``/``glm``/``robust`` ``lmplot`` call raise
    ``unknown column 'fitted' referenced by an encoding``.

    ``neutral_stroke`` (batch-D T13 round 2, gate widened rounds 3-4):
    ``Logistic``/``Glm``/``Robust`` have no ``groupby``, so they always
    produce one POOLED fit. When the caller has ANY channel that resolves to
    ``color`` active on the sibling scatter layer -- ``hue=``, or any of
    ``color=``/``fill=``/``stroke=`` (keyword or ``encode=`` dict form) --
    (``lmplot``'s ``color_active``, resolved once via
    ``ferrum.encoding._aliases.resolve_color_alias`` over the merged
    effective encoding, so a future alias needs no change here) -- that
    pooled fit must not visually impersonate one color level -- so the
    ribbon's fill and the line's stroke are pinned to the ``"theme:label"``
    sentinel,
    the same internal token ``marks/composite.py``'s boxplot whisker/median
    layers use to stay a neutral color under an unrelated ``color=``
    encoding (`resolve_paint_color` in
    ``crates/ferrum-core/src/render/draw.rs`` resolves it to the theme's
    label color -- never a scale/palette lookup, so it can't collide with a
    color level by coincidence the way the theme's default accent can). A
    caller-supplied ``line_kwargs["stroke"]`` still wins (``setdefault``).

    ``metrics`` appends the same ``"metrics"`` text layer ``desugar_smooth``
    builds when ``show_metrics=True`` -- only meaningful when ``transform``
    was itself built with its own ``inject_metrics=True`` (only ``Robust``
    of the three dedicated fit transforms supports it; see ``lmplot``'s
    ``metrics_applied`` computation).
    """
    cols = registry_output_columns(kind, ["x", "y", "ci_lower", "ci_upper"])
    fit_layers: list = []
    if ci is not None:
        ribbon_kwargs: dict = {"opacity": 0.3}
        if neutral_stroke:
            ribbon_kwargs["fill"] = "theme:label"
        fit_layers.append(
            _Layer(
                name="ribbon",
                mark="ribbon",
                encoding={"x": cols["x"], "y": cols["ci_lower"], "y2": cols["ci_upper"]},
                mark_kwargs=ribbon_kwargs,
                data_source=kind,
            )
        )
    line_kwargs_resolved = dict(line_kwargs) if line_kwargs else {}
    if neutral_stroke:
        line_kwargs_resolved.setdefault("stroke", "theme:label")
    fit_layers.append(
        _Layer(
            name="line",
            mark="line",
            encoding={"x": cols["x"], "y": cols["y"]},
            mark_kwargs=line_kwargs_resolved or None,
            data_source=kind,
        )
    )
    if metrics:
        fit_layers.append(
            _Layer(
                name="metrics",
                mark="text",
                encoding={"x": cols["x"], "y": "_metrics_y", "text": "_metrics_text"},
                mark_kwargs={"align": "right", "dx": -4, "dy": 4},
                data_source=kind,
            )
        )
    chart = Chart(data)._clone()
    chart._transforms = [transform]
    chart._layers = fit_layers
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

    * ``"lm"`` -- ``mark_smooth(method="lm")`` (polynomial degree
      controlled by ``order``).
    * ``"loess"``, ``"quadratic"``, ``"cubic"``, ``"log"``, ``"sqrt"``,
      ``"gam"`` -- ``mark_smooth(method=<method>)`` (batch-D T13: every
      ``transform_registry("smooth")["methods"]`` spelling except "linear"
      -- ``"lm"`` already covers it -- and "glm", reserved below). Every one
      of these accepts an explicit ``order=`` (forwarded uniformly as
      ``degree=``, see ``order`` below); a bare call (``order`` left at its
      ``None`` default) uses each method's own default degree concept.
    * ``"logistic"`` -- ``Logistic`` transform + pooled fit line + CI band.
    * ``"glm"``      -- ``Glm`` transform + pooled fit line + CI band.
    * ``"robust"``   -- ``Robust`` transform + pooled fit line + CI band.

    ``"logistic"``/``"glm"``/``"robust"`` fit **one pooled curve across the
    whole dataset** -- their Rust transforms take no ``groupby``, unlike
    ``mark_smooth``'s ``"lm"``/``"loess"``/etc. When any channel that
    resolves to ``color`` is active -- ``hue=``, a bare ``color=``/
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
        Column name to map to color. For ``method`` values routed through
        ``mark_smooth`` (``"lm"``, ``"loess"``, and the rest of
        ``_SMOOTH_ROUTED_METHODS``), the fit is refit per hue level via
        ``groupby``, so each level gets its own fit line and CI band in that
        level's scatter color. For ``"logistic"``, ``"glm"``, ``"robust"``
        (no ``groupby`` on those transforms), the scatter is colored per hue
        level but the fit is a single curve pooled across every group,
        rendered in a neutral stroke/fill distinct from the palette -- it
        does not represent any one group's fit. This pooled-neutral behavior
        for the three dedicated methods is not specific to ``hue=`` -- it
        applies identically whenever any channel that resolves to ``color``
        reaches the chart (``hue=``, a bare ``color=``/``fill=``/``stroke=``
        keyword, or ``encode={"color": ...}``/``encode={"fill": ...}``/
        ``encode={"stroke": ...}``), since all are the same "one pooled fit,
        colored scatter" situation. Per-group fits for these three methods
        require ``groupby`` support on the underlying Rust transform
        (tracked as a follow-up, not yet built).
    col : str, optional
        Column name for faceting across columns.
    row : str, optional
        Column name for faceting across rows.
    method : str, default "lm"
        Fitting method. ``"lm"``, ``"loess"``, ``"quadratic"``, ``"cubic"``,
        ``"log"``, ``"sqrt"``, ``"gam"`` route through ``mark_smooth``;
        ``"logistic"``, ``"glm"``, ``"robust"`` route through their own
        dedicated fit-family transform. The accepted set is
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
        uniformly for every ``method`` routed through ``mark_smooth``
        (``"lm"``, ``"loess"``, ``"quadratic"``, ``"cubic"``, ``"log"``,
        ``"sqrt"``, ``"gam"``) -- one forwarding rule, not a per-method
        carve-out; see ``_smooth_degree_kwargs``. What a forwarded value
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
        supported for every ``mark_smooth``-routed method (``"lm"``,
        ``"loess"``, and the rest of ``_SMOOTH_ROUTED_METHODS``) and for
        ``method="robust"``. Not supported for ``method="logistic"`` or
        ``"glm"`` (neither transform accepts ``x_range``); passing
        ``truncate=False`` with either raises ``ValueError``.
    x_bins : any, optional
        Forwarded as ``x_bins`` to ``mark_smooth`` for binning the
        x-axis before fitting (``method="lm"`` only).
    x_estimator : any, optional
        Forwarded as ``x_estimator`` to ``mark_smooth`` (``method="lm"``
        only).
    x_jitter : float or None, optional
        When set, applies ``Jitter(axis="x", width=x_jitter)`` to the
        scatter layer.
    logx : bool, default False
        Apply a ``log`` scale to the x-axis on both scatter and fit layers.
    show_metrics : bool, default True
        Schwabish SB-followup (2026-05-12): overlay a top-right corner
        annotation with ``R^2`` / ``RMSE`` / ``MAE`` computed from the fit.
        ``inject_metrics`` is a generic ``Smooth``/``Robust`` transform
        kwarg (computed against the raw input regardless of fit shape), so
        this applies to every ``mark_smooth``-routed method (``"lm"``,
        ``"loess"``, ``"quadratic"``, ``"cubic"``, ``"log"``, ``"sqrt"``,
        ``"gam"``) and to ``method="robust"``. Not supported for
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

    # Rust Smooth/Robust transforms require Float64 columns. Auto-cast
    # integer columns to Float64, matching catplot's pattern (chart.py).
    _INT_DTYPES = (
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
    )
    data = to_polars(data)
    casts = []
    for fld in (x, y):
        col_name = _field_name(fld)
        if col_name in data.columns and data[col_name].dtype in _INT_DTYPES:
            casts.append(pl.col(col_name).cast(pl.Float64))
    if casts:
        data = data.with_columns(casts)

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
    # those two rather than silently clipping anyway (no silent parameter
    # drop; batch-D T13 round-2 spec-review finding 3).
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
    enc: dict = {"x": x, "y": y}
    if hue is not None:
        enc["color"] = hue_ch
    enc.update(encode_kwargs)

    # batch-D T13 round-4 (`.sdd/task-13-python-quality-reviewer.md`
    # cycle-3 finding 1): a pooled dedicated-fit-method (logistic/glm/
    # robust) impersonating a color-level color reaches this chart through
    # every channel the repo's own alias policy resolves onto `color` --
    # not just the three spellings round 3 enumerated (`hue=`, `color=`,
    # `encode={"color": ...}`). `ferrum.encoding._aliases._ENCODING_ALIASES`
    # is the codebase's single table of which channels drive `color`
    # (`fill` -- keep_target, `stroke` -- warn_drop); `fill=`/`stroke=`
    # reach this chart the same two ways `color=` does (a bare keyword via
    # `**encode_kwargs`, merged into `enc` above, or `encode={"fill": ...}`/
    # `encode={"stroke": ...}`, a separate `encode=` dict `_finalize_chart`
    # applies later). Round 3's `"color" in enc or "color" in (encode or
    # {})` tested two literal keys and stayed blind to both alias doors --
    # the same enumeration failure shape recurring a third time. Resolving
    # through `resolve_color_alias` (the same alias table, already used
    # read-only for exactly this "what would color resolve to" question by
    # `composition.py`'s layer-color promotion) over the merged effective
    # encoding -- `enc` (which already carries `hue`'s `enc["color"]`
    # assignment and every `**encode_kwargs` channel) plus the separate
    # `encode=` dict -- means a future alias added to that table is covered
    # here automatically, with no enumeration to maintain or fall behind.
    from ferrum.encoding._aliases import resolve_color_alias

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

    # When hue is set and the method is Smooth-routed (lm, loess, and the
    # rest of `_SMOOTH_ROUTED_METHODS`), use Smooth's groupby support to fit
    # per-group smoothing in one pass. Only "lm" forwards `x_bins`/
    # `x_estimator` -- "loess" and the widened batch-D T13 set (quadratic/
    # cubic/log/sqrt/gam) don't accept binning the way "lm" does (see the
    # non-hue branch below for why), so this generalizes "loess"'s existing
    # shape rather than "lm"'s. `order` is forwarded as `degree=` via
    # `_smooth_degree_kwargs` uniformly for every `_SMOOTH_ROUTED_METHODS`
    # member -- see that helper's docstring for the one rule.
    if hue is not None and method in _SMOOTH_ROUTED_METHODS:
        hue_col = _field_name(hue)
        if method == "lm":
            fit = (
                Chart(data)
                .mark_smooth(
                    method="lm",
                    ci=ci_frac,
                    x_bins=x_bins,
                    x_estimator=x_estimator,
                    groupby=hue_col,
                    **_smooth_degree_kwargs(method, order, "lmplot"),
                    **({"x_range": x_range} if x_range is not None else {}),
                )
                .encode(x=x, y=y, color=hue_ch)
            )
        else:
            fit = (
                Chart(data)
                .mark_smooth(
                    method=method,
                    ci=ci_frac,
                    groupby=hue_col,
                    **_smooth_degree_kwargs(method, order, "lmplot"),
                    **({"x_range": x_range} if x_range is not None else {}),
                )
                .encode(x=x, y=y, color=hue_ch)
            )
    elif method == "lm":
        fit = (
            Chart(data)
            .mark_smooth(
                method="lm",
                ci=ci_frac,
                x_bins=x_bins,
                x_estimator=x_estimator,
                show_metrics=metrics_applied,
                **_smooth_degree_kwargs(method, order, "lmplot"),
                **({"x_range": x_range} if x_range is not None else {}),
            )
            .encode(x=x, y=y)
        )
    elif method in _SMOOTH_ROUTED_METHODS:
        # "loess" and the batch-D T13 widened registry set (quadratic/cubic/
        # log/sqrt/gam): `order=` forwards as `degree=` uniformly via
        # `_smooth_degree_kwargs` (batch-D T13 round 3) -- for "quadratic"/
        # "cubic" a matching value is a no-op and a conflicting one raises;
        # for "loess" a forwarded value changes the local-fit degree (or
        # raises outside {1, 2}); for "log"/"sqrt"/"gam" (no degree concept
        # at all) any forwarded value raises. `show_metrics` forwards
        # unconditionally (`metrics_applied` already gates on hue=None and
        # excludes logistic/glm upstream).
        fit = (
            Chart(data)
            .mark_smooth(
                method=method,
                ci=ci_frac,
                show_metrics=metrics_applied,
                **_smooth_degree_kwargs(method, order, "lmplot"),
                **({"x_range": x_range} if x_range is not None else {}),
            )
            .encode(x=x, y=y)
        )
    elif method == "logistic":
        fit = _fit_chart_from_named_transform(
            data,
            Logistic(x=x, y=y, n_grid=200, ci=ci_frac, name="logistic"),
            kind="logistic",
            ci=ci_frac,
            line_kwargs=lkw,
            neutral_stroke=color_active,
        )
    elif method == "glm":
        fit = _fit_chart_from_named_transform(
            data,
            Glm(x=x, y=y, family="gaussian", n_grid=200, ci=ci_frac, name="glm"),
            kind="glm",
            ci=ci_frac,
            line_kwargs=lkw,
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
            neutral_stroke=color_active,
            metrics=metrics_applied,
        )
    else:  # pragma: no cover — guarded above
        raise ValueError(f"unreachable: method={method!r}")

    # `_DEDICATED_FIT_METHODS` (logistic/glm/robust) have no `groupby` on
    # their Rust transform, so exactly one pooled fit is produced regardless
    # of whether a color encoding is active. This used to re-apply
    # `color=hue_ch` to that pooled fit, painting it in the first color
    # level's palette color beside a full legend -- a materially wrong
    # statistical chart, drawn silently (the same failure class
    # `nominal_color_channel` above names for a mistyped hue column;
    # batch-D T13 round-2 spec-review finding). Round 2's fix gated the
    # replacement on `hue is not None`, which closed only the `hue=` door --
    # a bare `color=` keyword or `encode={"color": ...}` still reached this
    # branch with an active color encoding and an unfixed impersonation
    # (round-3 finding). The `neutral_stroke=color_active` argument passed
    # to `_fit_chart_from_named_transform` above closes all six doors with
    # one gate (see `color_active`'s definition above, which resolves through
    # `resolve_color_alias` over {color, fill, stroke} channels x {keyword,
    # encode={...}} doors): the pooled line and its CI ribbon render with
    # the `"theme:label"` sentinel instead, the same internal token
    # `marks/composite.py`'s boxplot whisker/median layers use to stay a
    # neutral color under an unrelated `color=` encoding, so the pooled fit
    # can never visually collide with a color-level palette color.
    # Per-group fits for these three methods need `groupby` added to the
    # `Logistic`/`Glm`/`Robust` Rust transforms -- tracked as a batch-close
    # follow-up, not this task.

    # logx → log scale on x.
    if logx:
        if fit._layers is not None:
            # The dedicated-transform fit-family branches (logistic/glm/
            # robust, batch-D T13) build `_layers` directly rather than
            # through the `_PendingMark`/desugar path, so they never run
            # `_resolve_pending_impl`'s scale-through-desugar propagation
            # (`ferrum._desugar._propagate_layer_scale`, design spec §6 /
            # GH #45 prerequisite) that lets a chart-level `.encode(x=...)`
            # reach each layer's own x encoding. Apply the same propagation
            # directly since these layers already exist.
            from ferrum._desugar import _propagate_layer_scale

            _log_chart_enc = {"x": X(x, scale={"type": "log"})}
            new_fit = fit._clone()
            new_fit._layers = [_propagate_layer_scale(_log_chart_enc, lyr) for lyr in fit._layers]
            fit = new_fit
        else:
            fit = fit.encode(x=X(x, scale={"type": "log"}))
        if scatter_layer is not None:
            scatter_layer = scatter_layer.encode(x=X(x, scale={"type": "log"}))

    # Compose scatter + fit. (When ``metrics_applied``, the fit chart
    # already carries the metrics-text layer reading the Smooth output.)
    if scatter_layer is None:
        out = fit
    else:
        out = _merge_layers(scatter_layer, fit, scatter_name="scatter", fit_name="fit")

    # Apply line_kws to Smooth-routed methods (lm, loess, and the rest of
    # `_SMOOTH_ROUTED_METHODS`) where line_kws can't be passed through
    # mark_smooth directly. The dedicated fit-family methods (logistic, glm,
    # robust) already applied lkw as the line layer's mark_kwargs inside
    # `_fit_chart_from_named_transform`.
    if lkw and method in _SMOOTH_ROUTED_METHODS and out._layers:
        from dataclasses import replace as _dc_replace

        out._layers = [
            _dc_replace(layer, mark_kwargs={**(layer.mark_kwargs or {}), **lkw})
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
        Python-side regression duplication.
    zero_line : bool, default True
        Schwabish SB-followup: draw a dashed horizontal reference at
        ``y=0`` via the Rust transform's ``inject_zero_ref=True`` opt-in.
    label : str, optional
        Legend label for the residual series.  When set, a constant
        ``_label`` column is injected and mapped to color, producing a
        single-entry legend.
    color : str or encoding, optional
        Column name or constant color forwarded to ``Chart.encode(color=)``.
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

    Examples
    --------
    >>> import ferrum as fm
    >>> fm.residplot(df, x="total_bill", y="tip")

    Robust residuals with annotations:

    >>> fm.residplot(df, x="size", y="tip", robust=True)
    """
    data = to_polars(data)
    if dropna:
        data = data.drop_nulls(subset=[x, y])

    # Rust Smooth/Robust transforms require Float64 columns.  Auto-cast
    # integer columns, matching lmplot and catplot.
    _INT_DTYPES = (
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
    )
    casts = []
    for fld in (x, y):
        if fld in data.columns and data[fld].dtype in _INT_DTYPES:
            casts.append(pl.col(fld).cast(pl.Float64))
    if casts:
        data = data.with_columns(casts)

    # The Rust Smooth/Robust transforms inject ``_ref_zero``,
    # ``_metrics_text``, and ``_metrics_y`` columns when the corresponding
    # ``inject_*`` kwargs are set — single source of truth for residual
    # computation lives in Rust; Python only declares the spec.
    # When ``label`` is set, inject a constant ``_label`` column into the data
    # and use ``groupby="_label"`` on the Smooth transform so the column
    # survives the Rust transform's output schema (which otherwise only emits
    # its declared columns).  Robust does not support groupby, so the label
    # column cannot be preserved through that path — a Rust-side ``groupby``
    # addition to the Robust transform would be needed.
    if label is not None:
        data = data.with_columns(pl.lit(str(label)).alias("_label"))

    # Validate `order` via the same typed refusal `lmplot` uses for
    # `_SMOOTH_ROUTED_METHODS` members (batch-D T13 round-4 micro-round:
    # residplot's direct `degree=order` bypass needs the same negative-int
    # guard that `lmplot` applies via `_smooth_degree_kwargs`). The Robust
    # path doesn't use `order` so no validation is needed there.
    if not robust:
        _smooth_degree_kwargs("lm", order, "residplot")

    # `_resid_x` is a bare literal, not a registry lookup -- reverted after a
    # quality-review mispin (spec §4.6 / M6a adjudication; same reasoning
    # `desugar_boxen` states at `marks/composite.py:837-850` for its own
    # dispositioned literals). `Smooth`/`Robust` in `output="residuals"` mode
    # build this batch via `build_residuals_batch` (`smooth.rs:667`,
    # `robust.rs:365`), which declares its own literal `x` column in Rust,
    # uncoupled from `FIT_OUTPUT_COLUMNS` -- the `output="fitted"` schema
    # `transform_registry("smooth"|"robust")` actually declares
    # (`registry/mod.rs:165-167` scopes that constant to fitted mode
    # explicitly, and both kinds route through the same shared closure, so
    # looking up "smooth" vs "robust" here would not even diverge). Checking
    # `_resid_x` against `FIT_OUTPUT_COLUMNS` would pass today -- both
    # schemas happen to list `x` -- and stay silently green if the residuals
    # batch's own `x` were ever renamed independently: a lookup against the
    # wrong schema is worse than no lookup, since it would suppress the exact
    # drift this pattern exists to catch, and would also raise here on an
    # unrelated `FIT_OUTPUT_COLUMNS` rename that never touched this code
    # path. Recorded as a dispositioned column reference in
    # `tests/test_registry_output_column_consumers.py`'s `_DISPOSITIONED`
    # (site: `("src/ferrum/plots/regression.py", "residplot", "x")`), which
    # fails if this code ever stops needing the disposition. The correct end
    # state -- Rust declaring a residuals-mode output schema so this stops
    # being invisible to the registry by construction -- is a Rust change,
    # tracked as a batch-close follow-up, not this task's scope.
    _resid_x = "x"

    if robust:
        resid_transform = Robust(
            x=x,
            y=y,
            output="residuals",
            inject_zero_ref=zero_line,
            inject_metrics=show_metrics,
        )
    else:
        smooth_kwargs: dict = dict(
            method="lm",
            degree=order,
            ci=None,
            output="residuals",
            inject_zero_ref=zero_line,
            inject_metrics=show_metrics,
        )
        if label is not None:
            smooth_kwargs["groupby"] = "_label"
        resid_transform = Smooth(x=x, y=y, **smooth_kwargs)

    chart = Chart(data).transform(resid_transform).mark_point()
    enc: dict = {"x": _resid_x, "y": "residual"}
    if label is not None and color is None and not robust:
        enc["color"] = "_label"
    if color is not None:
        enc["color"] = color
    enc.update(encode_kwargs)
    chart = chart.encode(**enc)

    # Build the layered chart when any overlay is requested (zero rule,
    # metrics text, or lowess smoother). The augmented-DataFrame pattern
    # works because all overlays reference columns the Rust transform
    # already emitted into the same residuals batch.
    layered = lowess or zero_line or show_metrics
    if layered:
        chart = chart._clone()
        internal_layers: list = [_Layer(name="point", mark="point", encoding=dict(enc))]
        if zero_line:
            internal_layers.append(
                _Layer(
                    name="reference",
                    mark="rule",
                    encoding={"y": "_ref_zero"},
                    mark_kwargs={"stroke_dash": [3, 3], "stroke": "#8a8a8a"},
                )
            )
        if show_metrics:
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
                ),
            ]
            # Derived from the registry (spec §4.6 / M6a) rather than
            # hardcoded: this layer reads the named `lowess` Smooth
            # transform's fit-grid output, so a Rust-side rename raises here
            # instead of silently drawing nothing.
            _lowess_cols = registry_output_columns("smooth", ["x", "y"])
            internal_layers.append(
                _Layer(
                    name="lowess",
                    mark="line",
                    encoding={"x": _lowess_cols["x"], "y": _lowess_cols["y"]},
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
    x_jitter: Any = None,
    mark: dict | None = None,
    encode: dict | None = None,
    properties: dict | None = None,
    layers: list | None = None,
    theme: Any = None,
    **encode_kwargs: Any,
) -> Chart:
    """Axes-level regression scatter plot.

    The axes-level equivalent of ``lmplot`` — identical API except
    ``col=`` and ``row=`` are excluded because ``regplot`` does not
    facet.  Every parameter is forwarded to ``lmplot`` unchanged.

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
    x_jitter : float or None, optional
        When set, applies ``Jitter(axis="x", width=x_jitter)`` to the
        scatter layer.
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
    ValueError
        If ``method`` is not one of the supported values.
    ValueError
        If ``truncate=False`` with ``method="logistic"`` or ``"glm"``
        (neither transform accepts ``x_range``).
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
        x_jitter=x_jitter,
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
