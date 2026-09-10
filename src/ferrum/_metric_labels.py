"""Metric-label subsystem: AUCLabel, APLabel, BrierLabel, OutlierLabel.

These dataclass descriptors implement ``__radd__`` so they can be composed
onto a ``Chart`` via ``chart + AUCLabel()``.  The private helpers
(``_apply_metric_label``, ``_apply_metric_label_explicit``, etc.) are
internal to this module; external callers should use the public classes or
``_apply_metric_label_explicit`` for figure-level builders.

Single-source contract (T4.7b, 2026-06-22)
-----------------------------------------
There are two surfaces that overlay a metric value on a diagnostic curve:

* the **direct-mark** composite (``Chart.mark_roc(annotate_auc=...)`` etc.),
  whose ``__radd__`` path runs through ``_apply_metric_label``; and
* the **figure-function** builders (``_roc_chart_from_source`` etc.), which
  call ``_apply_metric_label_explicit`` with field overrides.

Both paths share one definition of *what a metric kind is* — its value
function and its overlay-text prefix — via :data:`_METRIC_LABEL_SPECS`.
A placement or prefix fix therefore lands in one place and is observed by
both surfaces.  (The value functions ``_trapezoid_auc`` / ``_ap_step`` /
``_brier_score`` were already single-homed here; the metric→(fn, prefix)
association was the piece duplicated between ``_apply_metric_label_explicit``
and the per-class dataclass ``__radd__`` methods.)

Where the metric value comes from (F-L11-01, 2026-09-10)
--------------------------------------------------------
:func:`_curve_metric` is the one rule every AUC/AP display surface obeys:
when the frame carries the Rust kernel's broadcast metric column (``auc``
on a ROC frame, ``ap`` on a PR frame), that value is shown; only data
without such a column -- the ``+ AUCLabel()`` composable path over
arbitrary user data -- is integrated here in Python. The chart title, the
legend text, the label overlay and the ``mark_roc(annotate_auc=True)``
annotation therefore cannot disagree about what a curve's AUC is.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, ClassVar, Literal, Optional

import polars as pl

from ferrum._validate import validate_choice


def _resolve_field(enc_value: Any) -> Optional[str]:
    """Extract the field name from an encoding value.

    Encoding-dict values may be ``ChannelBase`` wrappers (with ``.field``)
    or plain strings, mirroring the pattern in ``chart.py``.
    """
    if enc_value is None:
        return None
    field = getattr(enc_value, "field", None)
    if field is not None:
        return field
    if isinstance(enc_value, str):
        return enc_value
    return None


def _trapezoid_auc(x, y) -> float:
    """Trapezoidal AUC for a curve. Sorts by x before integrating.

    The sort is **stable** (F-L11-01): ROC/PR curves routinely carry many
    rows sharing one x value (``fpr``/``recall`` ties at the curve
    extremes), and numpy's default quicksort scrambles the y order within
    each tied-x run. Which y ends up last in a run sets the next segment's
    starting height, so an unstable order corrupts the integral outright --
    it is not a tie-breaking preference but a wrong number. With
    ``kind="stable"`` ties keep their original (threshold) order and the
    result matches the Rust kernel's AUC exactly.
    """
    import numpy as np

    order = np.argsort(x, kind="stable")
    trap = getattr(np, "trapezoid", None) or np.trapz  # type: ignore[attr-defined]
    return float(trap(y[order], x[order]))


def _ap_step(x, y) -> float:
    """Step-integrated average precision: sum((R_i - R_{i-1}) * P_i).

    Stable-sorted for the same reason as :func:`_trapezoid_auc`.
    """
    import numpy as np

    order = np.argsort(x, kind="stable")
    xs, ys = x[order], y[order]
    return float(np.sum(np.diff(np.concatenate([[0.0], xs])) * ys))


def _brier_score(p, obs) -> float:
    """Brier score as mean squared error between predicted prob and observed rate."""
    import numpy as np

    return float(np.mean((p - obs) ** 2))


def _broadcast_metric(df: pl.DataFrame, metric_field: Optional[str]) -> Optional[float]:
    """The one value a broadcast metric column carries for ``df``, or ``None``.

    Diagnostic curve frames repeat a kernel-computed scalar on every row of
    a curve (``auc`` on ROC frames, ``ap`` on PR frames -- see
    ``diagnostics/_internal/_curve_frames.py``). ``None`` means "this frame
    does not carry the metric for these rows": the column is absent, holds
    no usable value, or holds *more than one* distinct value, which says the
    rows span several curves and no single broadcast value describes them.
    """
    if metric_field is None or metric_field not in df.columns:
        return None
    values = df[metric_field].drop_nulls().unique().to_list()
    if len(values) != 1:
        return None
    value = float(values[0])
    return None if math.isnan(value) else value


def _curve_metric(
    df: pl.DataFrame,
    *,
    metric_field: Optional[str],
    metric_fn: Callable[[Any, Any], float],
    x_col: str,
    y_col: str,
) -> float:
    """The metric for the single curve held in ``df``.

    One resolution rule, shared by every surface that displays an AUC/AP
    (chart title, legend text, label overlay, mark annotation): read the
    frame's broadcast metric column when it carries this curve's value,
    and integrate ``metric_fn`` over ``(x_col, y_col)`` only when it does
    not. The column is the Rust kernel's own number, so a diagnostic chart
    shows one metric from one source; the integration path is what makes
    the composable ``+ AUCLabel()`` marks work on arbitrary user data,
    which has no such column.
    """
    value = _broadcast_metric(df, metric_field)
    if value is not None:
        return value

    import numpy as np

    return float(
        metric_fn(
            np.asarray(df[x_col].to_list(), dtype=float),
            np.asarray(df[y_col].to_list(), dtype=float),
        )
    )


def _curve_metrics_by_group(
    df: pl.DataFrame,
    *,
    group_field: str,
    metric_field: Optional[str],
    metric_fn: Callable[[Any, Any], float],
    x_col: str,
    y_col: str,
) -> dict[str, float]:
    """:func:`_curve_metric` per ``group_field`` value, keyed by ``str(value)``.

    Groups are enumerated in first-appearance order (null group values are
    skipped) so callers that assign per-group display slots -- staggered
    label rows, for instance -- get a deterministic order; polars'
    ``unique()`` does not promise one.
    """
    groups = dict.fromkeys(v for v in df[group_field].to_list() if v is not None)
    return {
        str(value): _curve_metric(
            df.filter(pl.col(group_field) == value),
            metric_field=metric_field,
            metric_fn=metric_fn,
            x_col=x_col,
            y_col=y_col,
        )
        for value in groups
    }


# Canonical metric-kind table — the single source for the
# ``kind -> (value function, overlay-text prefix)`` association shared by the
# direct-mark ``__radd__`` path and the figure-function explicit-field path.
# Keys match the ``label_kind`` strings accepted by
# ``_apply_metric_label_explicit``; the prefixes match the per-class dataclass
# defaults so consolidating here is byte-identical on both surfaces.
_METRIC_LABEL_SPECS: dict[str, tuple[Callable[[Any, Any], float], str]] = {
    "auc": (_trapezoid_auc, "AUC = "),
    "ap": (_ap_step, "AP = "),
    "brier": (_brier_score, "Brier = "),
}


def _default_prefix(kind: str) -> str:
    """Canonical overlay-text prefix for a metric kind (drives dataclass defaults)."""
    return _METRIC_LABEL_SPECS[kind][1]


_VALID_METRIC_LABEL_POSITIONS = ("end", "corner")


def _validate_metric_label_fields(label: "AUCLabel | APLabel | BrierLabel") -> None:
    """Construction-time validation shared by ``AUCLabel``/``APLabel``/``BrierLabel``.

    Finding P6 (2026-08-27): ``position`` is ``Literal``-annotated but was
    unenforced at runtime, and a bad ``format`` spec previously only
    exploded deep inside an f-string with no indication of which label
    constructed it. Both fail fast here, with the constructing class named
    in the message.
    """
    cls_name = type(label).__name__
    validate_choice(
        f"{cls_name}.position", "position", label.position, _VALID_METRIC_LABEL_POSITIONS
    )
    try:
        format(1.0, label.format)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"{cls_name}.format: {label.format!r} is not a valid format spec for a float value ({exc})"
        ) from exc


def _resolve_precomputed_metric(
    precomputed_value: "float | dict[str, float] | None", group_key: Any = None
) -> Optional[float]:
    """The precomputed override for one curve/group, or ``None`` when unset.

    ``precomputed_value`` is either a single float (one curve, no color
    grouping -- e.g. a single-model calibration chart) or a
    ``dict[str, float]`` keyed by ``str(group_key)`` (one value per color
    group -- e.g. a compared calibration chart's per-model Brier), matching
    the keying convention :func:`_curve_metrics_by_group` already uses.
    ``None`` (either the whole override or the specific group's entry) means
    "fall back to ``metric_fn``/``metric_field`` for this curve."
    """
    if precomputed_value is None:
        return None
    if isinstance(precomputed_value, dict):
        if group_key is None:
            return None
        value = precomputed_value.get(str(group_key))
        return None if value is None else float(value)
    return float(precomputed_value)


def _apply_metric_label(
    base: "Chart",
    label: "AUCLabel | APLabel | BrierLabel",
    *,
    metric_fn,
    metric_field: Optional[str] = None,
    precomputed_value: "float | dict[str, float] | None" = None,
    x_col_override: Optional[str] = None,
    y_col_override: Optional[str] = None,
    color_col_override: Optional[str] = None,
) -> "Chart":
    """Compute a metric per series and emit a text-overlay layer on the
    base chart's augmented data.

    Mirrors the augmented-DataFrame idiom used by ``_inject_curve_annotation``
    in the legacy ROC/PR paths: extend ``base._data`` with ``_label_text``
    and ``_label_y`` columns (one non-null entry per series at the endpoint
    along the y axis, staggered so multi-curve labels do not collide).
    Adds a ``mark_text`` layer reading those columns via ``+``.

    ``metric_field`` (F-L11-01, 2026-09-10) names a broadcast metric column
    on the base chart's data -- ``"auc"`` on a ROC frame, ``"ap"`` on a PR
    frame. When the data carries it, the overlay shows that kernel-computed
    value, so the label, the chart title and the legend text are one number
    from one source. ``None`` (the ``+ AUCLabel()`` composable path, whose
    data is arbitrary user data) integrates ``metric_fn`` instead.

    ``precomputed_value`` (F-L11-04, 2026-09-10) overrides both of the
    above: when set, ``metric_fn`` is not consulted and no broadcast column
    is read for that curve/group -- the caller's own per-sample value is
    shown verbatim. Calibration's Brier overlay is the motivating case: a
    bin-level Brier computed from ``(mean_predicted, fraction_positive)``
    swings with ``n_bins=``/``strategy=`` and is not the Brier score, so the
    calibration builder always passes the true per-sample value it computes
    from the held predictions, invariant to binning.

    Schwabish SB3 (2026-05-11): accepts explicit field overrides so the
    figure-level builders (which hide encoding behind ``_pending_stat_mark``
    composites like ``mark_roc``) can compose metric labels without setting
    a chart-level color encoding that would leak into adjacent layers
    (e.g. the diagonal reference line of ``mark_roc``).

    Finding P6 (2026-08-27): ``label.position`` routes the label's x anchor.
    ``"end"`` (the historical, only-implemented behavior) anchors each
    series' label at that series' own rightmost data point, via ``_label_x``
    copied from the endpoint row's own ``x_col`` value — byte-identical to
    the pre-fix unconditional behavior. ``"corner"`` anchors every series'
    label at the same x — the dataset's overall max ``x_col`` value — so
    multi-series labels stack in a column near the top-right of the plot
    area instead of trailing each curve's own end, matching the documented
    "corner" placement. Reuses this function's own per-series staggering
    (the established idiom, shared with ``_inject_metrics_corner`` in
    ``plots/_helpers.py``) rather than importing that residuals-specific,
    hardcoded-column helper, which would invert the top-level/``plots``
    dependency direction and does not compute this module's metrics anyway.
    """
    from ferrum._coerce import to_arrow_table
    from ferrum.chart import Chart

    x_col = x_col_override or _resolve_field(base._encoding.get("x"))
    y_col = y_col_override or _resolve_field(base._encoding.get("y"))
    color_col = color_col_override or _resolve_field(base._encoding.get("color"))
    if x_col is None or y_col is None:
        raise ValueError(f"{type(label).__name__} requires x and y encodings on the base chart")
    tbl = to_arrow_table(base._data)
    if x_col not in tbl.column_names or y_col not in tbl.column_names:
        raise ValueError(f"{type(label).__name__}: column {x_col!r} or {y_col!r} missing from data")
    df = pl.from_arrow(tbl).with_row_index("_idx")
    n = df.height
    labels_col: list[Optional[str]] = [None] * n

    y_range = float(df[y_col].max() - df[y_col].min()) if n > 0 else 1.0
    y_top = float(df[y_col].max()) if n > 0 else 1.0
    x_col_max = df[x_col].max() if n > 0 else None
    x_top = float(x_col_max) if x_col_max is not None else 0.0
    stagger_step = max(y_range * 0.06, 1e-9)
    label_y_col: list[Optional[float]] = [None] * n
    label_x_col: list[Optional[float]] = [None] * n
    corner = label.position == "corner"

    if color_col is not None and color_col in df.columns:
        unique_colors = sorted(df[color_col].unique().to_list(), key=str)
        for stack_i, cls in enumerate(unique_colors):
            group = df.filter(pl.col(color_col) == cls)
            if group.is_empty():
                continue
            metric = _resolve_precomputed_metric(precomputed_value, cls)
            if metric is None:
                metric = _curve_metric(
                    group,
                    metric_field=metric_field,
                    metric_fn=metric_fn,
                    x_col=x_col,
                    y_col=y_col,
                )
            text = f"{label.prefix}{metric:{label.format}}"
            # nulls_last=True so a null x (which polars' default descending
            # sort places FIRST) never becomes the selected "endpoint" row;
            # the endpoint is the group's highest *non-null* x, matching what
            # the pre-fix code rendered for this data (the label text/metric
            # already reflects any null via metric_fn's NaN propagation
            # above -- only the label's x-position selection is guarded
            # here). Only an entirely-null-x group falls through with the
            # null row still selected; that case is caught explicitly below.
            endpoint_row = group.sort(x_col, descending=True, nulls_last=True).row(0, named=True)
            global_idx = int(endpoint_row["_idx"])
            endpoint_x = endpoint_row[x_col]
            labels_col[global_idx] = text
            label_y_col[global_idx] = y_top - stack_i * stagger_step
            if corner:
                label_x_col[global_idx] = x_top
            else:
                label_x_col[global_idx] = float(endpoint_x) if endpoint_x is not None else x_top
    else:
        metric = _resolve_precomputed_metric(precomputed_value)
        if metric is None:
            metric = _curve_metric(
                df,
                metric_field=metric_field,
                metric_fn=metric_fn,
                x_col=x_col,
                y_col=y_col,
            )
        text = f"{label.prefix}{metric:{label.format}}"
        global_idx = df[x_col].arg_max()
        # arg_max() returns None (not an int index) for an empty or
        # entirely-null x column -- pre-existing edge case, not introduced
        # by this change, but reachable via the same empty/null-input
        # scenarios this fix now covers. No non-null row exists to anchor a
        # label, so skip emission entirely (mirrors the grouped branch's
        # own `if group.is_empty(): continue`); the base chart still renders.
        if global_idx is not None:
            labels_col[global_idx] = text
            label_y_col[global_idx] = y_top
            label_x_col[global_idx] = x_top

    base_pl = base._data if isinstance(base._data, pl.DataFrame) else pl.from_arrow(tbl)
    augmented = base_pl.with_columns(
        pl.Series("_label_text", labels_col, dtype=pl.Utf8),
        pl.Series("_label_x", label_x_col, dtype=pl.Float64),
        pl.Series("_label_y", label_y_col, dtype=pl.Float64),
    )
    base_aug = base._clone()
    base_aug._data = augmented
    annot_layer = (
        Chart(augmented)
        .mark_text(align="right", baseline="hanging", dx=-4, dy=4)
        .encode(x="_label_x", y="_label_y", text="_label_text")
    )
    return base_aug + annot_layer


@dataclass(frozen=True)
class AUCLabel:
    """Auto-placed AUC annotation for ROC charts.

    ``chart + AUCLabel()`` reads the surrounding chart's line data
    (``x`` = FPR, ``y`` = TPR), computes trapezoidal AUC per series
    (grouped by ``color`` when present), and emits one text annotation
    per series at the line endpoint.

    Parameters
    ----------
    position : {"end", "corner"}, default "end"
        Where to place the label.  ``"end"`` puts it at the rightmost
        point of each series curve; ``"corner"`` anchors it near the
        top-right of the plot area.
    format : str, default ".3f"
        Python format spec applied to the AUC value (e.g. ``".2f"`` for
        two decimal places).
    prefix : str, default "AUC = "
        Text prepended to the formatted metric value.

    Examples
    --------
    >>> import ferrum as fm
    >>> chart = fm.roc_chart(model, X_test, y_test)
    >>> annotated = chart + fm.AUCLabel()
    """

    _kind: ClassVar[str] = "auc"
    position: Literal["end", "corner"] = "end"
    format: str = ".3f"
    prefix: str = _default_prefix("auc")

    def __post_init__(self) -> None:
        _validate_metric_label_fields(self)

    def __radd__(self, base: "Chart") -> "Chart":
        from ferrum.chart import Chart

        if not isinstance(base, Chart):
            return NotImplemented
        return _apply_metric_label(base, self, metric_fn=_METRIC_LABEL_SPECS[self._kind][0])


@dataclass(frozen=True)
class APLabel:
    """Auto-placed Average Precision annotation for PR charts.

    Sibling of [AUCLabel][ferrum.AUCLabel] for precision-recall curves.  ``x`` is
    treated as recall and ``y`` as precision.  Computes step-function
    area under the curve per series.

    Parameters
    ----------
    position : {"end", "corner"}, default "end"
        Where to place the label.  ``"end"`` puts it at the rightmost
        point of each series curve; ``"corner"`` anchors it near the
        top-right of the plot area.
    format : str, default ".3f"
        Python format spec applied to the AP value.
    prefix : str, default "AP = "
        Text prepended to the formatted metric value.

    Examples
    --------
    >>> import ferrum as fm
    >>> chart = fm.pr_chart(model, X_test, y_test)
    >>> annotated = chart + fm.APLabel()
    """

    _kind: ClassVar[str] = "ap"
    position: Literal["end", "corner"] = "end"
    format: str = ".3f"
    prefix: str = _default_prefix("ap")

    def __post_init__(self) -> None:
        _validate_metric_label_fields(self)

    def __radd__(self, base: "Chart") -> "Chart":
        from ferrum.chart import Chart

        if not isinstance(base, Chart):
            return NotImplemented
        return _apply_metric_label(base, self, metric_fn=_METRIC_LABEL_SPECS[self._kind][0])


@dataclass(frozen=True)
class BrierLabel:
    """Auto-placed Brier-score annotation for calibration charts.

    ``x`` is treated as predicted probability and ``y`` as observed rate
    per bin.  Multi-series charts emit one Brier score per series.  Lower
    scores indicate better calibration.

    Parameters
    ----------
    position : {"end", "corner"}, default "corner"
        Where to place the label.  ``"corner"`` anchors it near the
        top-right of the plot area; ``"end"`` puts it at the last bin.
    format : str, default ".3f"
        Python format spec applied to the Brier score value.
    prefix : str, default "Brier = "
        Text prepended to the formatted metric value.

    Examples
    --------
    >>> import ferrum as fm
    >>> chart = fm.calibration_chart(model, X_test, y_test)
    >>> annotated = chart + fm.BrierLabel()
    """

    _kind: ClassVar[str] = "brier"
    position: Literal["end", "corner"] = "corner"
    format: str = ".3f"
    prefix: str = _default_prefix("brier")

    def __post_init__(self) -> None:
        _validate_metric_label_fields(self)

    def __radd__(self, base: "Chart") -> "Chart":
        from ferrum.chart import Chart

        if not isinstance(base, Chart):
            return NotImplemented
        return _apply_metric_label(base, self, metric_fn=_METRIC_LABEL_SPECS[self._kind][0])


# Registry of metric-label classes keyed by each class's own ``_kind``, so the
# figure-function explicit-field path (:func:`_apply_metric_label_explicit`)
# constructs the right class per kind without an if/elif ladder that could
# desync from the classes. Derived from the classes themselves; the keys match
# ``_METRIC_LABEL_SPECS`` (both keyed by the same ``"auc"``/``"ap"``/``"brier"``
# strings; enforced by the single-source test).
_METRIC_LABEL_CLASSES: dict[str, type] = {cls._kind: cls for cls in (AUCLabel, APLabel, BrierLabel)}


@dataclass(frozen=True)
class OutlierLabel:
    """Auto-label high-leverage or high-residual points on a scatter chart.

    ``chart + OutlierLabel()`` scans the chart's ``y`` column for values
    whose z-score exceeds ``threshold``, then overlays text labels at those
    points.  The label text is taken from ``label_field`` when supplied,
    otherwise from the ``field`` column, otherwise from the ``y`` value.

    Parameters
    ----------
    threshold : float, default 3.0
        Z-score threshold above which a point is considered an outlier and
        labelled.
    field : str, optional
        Column to compute z-scores from.  Defaults to the chart's ``y``
        encoding field.
    label_field : str, optional
        Column whose value is used as the text label.  Defaults to ``field``
        when omitted.
    max_labels : int, default 10
        Maximum number of labels to emit.  Points are ranked by absolute
        z-score and only the top ``max_labels`` are labelled.

    Examples
    --------
    >>> import ferrum as fm
    >>> chart = fm.residuals_chart(model, X_test, y_test)
    >>> annotated = chart + fm.OutlierLabel(threshold=2.5, max_labels=5)
    """

    threshold: float = 3.0
    field: Optional[str] = None
    label_field: Optional[str] = None
    max_labels: int = 10

    def __radd__(self, base: "Chart") -> "Chart":
        from ferrum._coerce import to_arrow_table
        from ferrum.chart import Chart

        if not isinstance(base, Chart):
            return NotImplemented
        x_col = _resolve_field(base._encoding.get("x"))
        y_col = _resolve_field(base._encoding.get("y"))
        field = self.field or y_col
        if x_col is None or y_col is None:
            raise ValueError("OutlierLabel requires x and y encodings on the base chart")
        tbl = to_arrow_table(base._data)
        if field is None or field not in tbl.column_names:
            raise ValueError(f"OutlierLabel: cannot locate field {field!r}")
        base_pl = base._data if isinstance(base._data, pl.DataFrame) else pl.from_arrow(tbl)
        df = base_pl.with_row_index("_idx")
        mu = float(df[field].mean())
        sigma = float(df[field].std(ddof=1)) or 1.0
        df = df.with_columns(((pl.col(field) - mu).abs() / sigma).alias("_z"))
        outliers = (
            df.filter(pl.col("_z") > self.threshold)
            .sort("_z", descending=True)
            .head(self.max_labels)
        )
        if outliers.is_empty():
            return base
        label_col_name = "_outlier_text"
        labels_col: list[Optional[str]] = [None] * base_pl.height
        has_label_field = self.label_field and self.label_field in base_pl.columns
        for row in outliers.iter_rows(named=True):
            idx = int(row["_idx"])
            if has_label_field:
                labels_col[idx] = str(base_pl[self.label_field][idx])
            else:
                labels_col[idx] = str(base_pl[field][idx])
        augmented = base_pl.with_columns(pl.Series(label_col_name, labels_col))
        base_aug = base._clone()
        base_aug._data = augmented
        annot_layer = (
            Chart(augmented)
            .mark_text(align="left", dx=4, dy=-4)
            .encode(x=x_col, y=y_col, text=label_col_name)
        )
        return base_aug + annot_layer


def _apply_metric_label_explicit(
    base: "Chart",
    label_kind: str,
    *,
    x_col: str,
    y_col: str,
    color_col: Optional[str] = None,
    metric_field: Optional[str] = None,
    precomputed_value: "float | dict[str, float] | None" = None,
    position: str = "end",
    fmt: str = ".3f",
    prefix: Optional[str] = None,
) -> "Chart":
    """Apply a metric label to ``base`` with explicit field overrides.

    Schwabish SB3 helper used by figure-level diagnostic builders that
    compose ``AUCLabel`` / ``APLabel`` / ``BrierLabel`` onto charts whose
    ``_pending_stat_mark`` composite hides the encoding.

    Shares the metric-kind table (:data:`_METRIC_LABEL_SPECS`) with the
    direct-mark ``__radd__`` path, so a value-function or prefix change is
    observed identically by both surfaces. ``metric_field`` is forwarded to
    :func:`_apply_metric_label`: a diagnostic builder names the broadcast
    metric column its frame already carries (``"auc"`` / ``"ap"``) so the
    overlay reads the same number the title and legend do.

    ``precomputed_value`` (F-L11-04, 2026-09-10) is forwarded verbatim: when
    set, neither ``metric_fn`` nor ``metric_field`` is consulted for the
    matching curve/group -- see :func:`_apply_metric_label`'s docstring.
    Calibration is the only caller today (its Brier must be a per-sample
    value, never the bin-level number ``metric_fn``/``metric_field`` would
    otherwise produce from the binned reliability-diagram data).
    """
    validate_choice("_apply_metric_label_explicit", "label_kind", label_kind, _METRIC_LABEL_SPECS)
    metric_fn, default_prefix = _METRIC_LABEL_SPECS[label_kind]
    pos_lit: Any = position
    prefix_str = prefix if prefix is not None else default_prefix
    label_obj: Any = _METRIC_LABEL_CLASSES[label_kind](
        position=pos_lit, format=fmt, prefix=prefix_str
    )
    return _apply_metric_label(
        base,
        label_obj,
        metric_fn=metric_fn,
        metric_field=metric_field,
        precomputed_value=precomputed_value,
        x_col_override=x_col,
        y_col_override=y_col,
        color_col_override=color_col,
    )
