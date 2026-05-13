"""Reference-line, rectangle, and text annotation helpers."""

from __future__ import annotations

from typing import Optional

import polars as pl

from ferrum.chart import Chart


def annotate_hline(
    y: float, *, label: Optional[str] = None, stroke: Optional[str] = None, stroke_dash=None
) -> Chart:
    """Horizontal reference line at a fixed y position.

    Returns a single-mark ``Chart`` suitable for ``|`` / ``&`` concatenation
    composition; for true overlay/layer, use ``+`` with a chart that shares
    the same DataFrame.

    Parameters
    ----------
    y : float
        Y position of the line in data coordinates.
    label : str, optional
        Reserved for future use (no-op today).
    stroke : str, optional
        Line color as a CSS color string. Defaults to the mark default when
        omitted.
    stroke_dash : list of float, optional
        SVG dash array, e.g. ``[4, 4]`` for evenly dashed.

    Returns
    -------
    Chart
        Annotation chart suitable for ``|`` / ``&`` composition.

    Examples
    --------
    >>> import ferrum as fm
    >>> ref = fm.annotate_hline(y=0.0, stroke="red", stroke_dash=[4, 4])
    >>> chart = fm.Chart(df).encode(x="t", y="r").mark_line() & ref
    """
    df = pl.DataFrame({"_y": [y]})
    kwargs: dict = {}
    if stroke is not None:
        kwargs["stroke"] = stroke
    if stroke_dash is not None:
        kwargs["stroke_dash"] = stroke_dash
    return Chart(df).mark_rule(**kwargs).encode(y="_y")


def annotate_vline(
    x: float, *, label: Optional[str] = None, stroke: Optional[str] = None, stroke_dash=None
) -> Chart:
    """Vertical reference line at a fixed x position.

    Returns a single-mark ``Chart`` suitable for ``|`` / ``&`` concatenation
    composition; for true overlay/layer, use ``+`` with a chart that shares
    the same DataFrame.

    Parameters
    ----------
    x : float
        X position of the line in data coordinates.
    label : str, optional
        Reserved for future use (no-op today).
    stroke : str, optional
        Line color as a CSS color string.
    stroke_dash : list of float, optional
        SVG dash array, e.g. ``[4, 4]``.

    Returns
    -------
    Chart
        Annotation chart suitable for ``|`` / ``&`` composition.

    Examples
    --------
    >>> import ferrum as fm
    >>> ref = fm.annotate_vline(x=2020, stroke="#888")
    >>> chart = fm.Chart(df).encode(x="year", y="val").mark_line() & ref
    """
    df = pl.DataFrame({"_x": [x]})
    kwargs: dict = {}
    if stroke is not None:
        kwargs["stroke"] = stroke
    if stroke_dash is not None:
        kwargs["stroke_dash"] = stroke_dash
    return Chart(df).mark_rule(**kwargs).encode(x="_x")


def annotate_rect(
    x1: float,
    x2: float,
    y1: float,
    y2: float,
    *,
    fill: Optional[str] = None,
    opacity: float = 0.1,
    label: Optional[str] = None,
) -> Chart:
    """Shaded rectangle region spanning (x1, y1) to (x2, y2).

    Returns a ``mark_rect`` annotation chart for ``|`` / ``&`` concatenation
    composition; for true overlay/layer, use ``+`` with a chart that shares
    the same DataFrame.

    Parameters
    ----------
    x1 : float
        Left x boundary in data coordinates.
    x2 : float
        Right x boundary in data coordinates.
    y1 : float
        Bottom y boundary in data coordinates.
    y2 : float
        Top y boundary in data coordinates.
    fill : str, optional
        Fill color as a CSS color string.
    opacity : float, default 0.1
        Fill opacity in ``[0, 1]``.
    label : str, optional
        Reserved for future use (no-op today).

    Returns
    -------
    Chart
        Annotation chart suitable for ``|`` / ``&`` composition.

    Examples
    --------
    >>> import ferrum as fm
    >>> shade = fm.annotate_rect(x1=2018, x2=2020, y1=0, y2=100,
    ...                          fill="#ffcc00", opacity=0.2)
    >>> chart = fm.Chart(df).encode(x="year", y="val").mark_line() & shade
    """
    df = pl.DataFrame({"_x1": [x1], "_x2": [x2], "_y1": [y1], "_y2": [y2]})
    kwargs: dict = {"opacity": opacity}
    if fill is not None:
        kwargs["fill"] = fill
    return Chart(df).mark_rect(**kwargs).encode(x="_x1", y="_y1", x2="_x2", y2="_y2")


def annotate_text(
    x: float,
    y: float,
    text: str,
    *,
    dx: float = 0,
    dy: float = 0,
    align: str = "center",
    baseline: str = "middle",
    font_size: Optional[float] = None,
    color: Optional[str] = None,
    angle: Optional[float] = None,
) -> Chart:
    """Free-floating text annotation at a fixed (x, y) position.

    Returns a ``mark_text`` chart for ``|`` / ``&`` concatenation composition;
    for true overlay/layer, use ``+`` with a chart that shares the same
    DataFrame.

    Parameters
    ----------
    x : float
        X position in data coordinates.
    y : float
        Y position in data coordinates.
    text : str
        Text string to display.
    dx : float, default 0
        Horizontal pixel offset from ``(x, y)``.
    dy : float, default 0
        Vertical pixel offset from ``(x, y)``.
    align : str, default "center"
        Horizontal text alignment (SVG ``text-anchor``): ``"left"``,
        ``"center"``, or ``"right"``.
    baseline : str, default "middle"
        Vertical text baseline: ``"top"``, ``"middle"``, or ``"bottom"``.
    font_size : float, optional
        Font size in points.
    color : str, optional
        Text fill color as a CSS color string.
    angle : float, optional
        Rotation angle in degrees (clockwise).

    Returns
    -------
    Chart
        Annotation chart suitable for ``|`` / ``&`` composition.

    Examples
    --------
    >>> import ferrum as fm
    >>> label = fm.annotate_text(x=2020, y=95, text="peak", dy=-8,
    ...                          color="#333", font_size=11)
    >>> chart = fm.Chart(df).encode(x="year", y="val").mark_line() & label
    """
    df = pl.DataFrame({"_x": [x], "_y": [y], "_text": [text]})
    kwargs: dict = {"dx": dx, "dy": dy, "align": align, "baseline": baseline}
    if font_size is not None:
        kwargs["font_size"] = font_size
    if color is not None:
        kwargs["fill"] = color
    if angle is not None:
        kwargs["angle"] = angle
    return Chart(df).mark_text(**kwargs).encode(x="_x", y="_y", text="_text")


# -------------------------------------------------------------------
# Schwabish SB1 — metric labels + annotate_arrow
# -------------------------------------------------------------------

from dataclasses import dataclass  # noqa: E402
from typing import Literal  # noqa: E402


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
    """Trapezoidal AUC for a curve. Sorts by x before integrating."""
    import numpy as np
    order = np.argsort(x)
    trap = getattr(np, "trapezoid", None) or np.trapz  # type: ignore[attr-defined]
    return float(trap(y[order], x[order]))


def _ap_step(x, y) -> float:
    """Step-integrated average precision: sum((R_i - R_{i-1}) * P_i)."""
    import numpy as np
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    return float(np.sum(np.diff(np.concatenate([[0.0], xs])) * ys))


def _brier_score(p, obs) -> float:
    """Brier score as mean squared error between predicted prob and observed rate."""
    import numpy as np
    return float(np.mean((p - obs) ** 2))


def _apply_metric_label(
    base: Chart,
    label: "AUCLabel | APLabel | BrierLabel",
    *,
    metric_fn,
    x_col_override: Optional[str] = None,
    y_col_override: Optional[str] = None,
    color_col_override: Optional[str] = None,
) -> Chart:
    """Compute a metric per series and emit a text-overlay layer on the
    base chart's augmented data.

    Mirrors the augmented-DataFrame idiom used by ``_inject_curve_annotation``
    in the legacy ROC/PR paths: extend ``base._data`` with ``_label_text``
    and ``_label_y`` columns (one non-null entry per series at the endpoint
    along the y axis, staggered so multi-curve labels do not collide).
    Adds a ``mark_text`` layer reading those columns via ``+``.

    Schwabish SB3 (2026-05-11): accepts explicit field overrides so the
    figure-level builders (which hide encoding behind ``_pending_stat_mark``
    composites like ``mark_roc``) can compose metric labels without setting
    a chart-level color encoding that would leak into adjacent layers
    (e.g. the diagonal reference line of ``mark_roc``).
    """
    from ferrum._coerce import to_arrow_table

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
    stagger_step = max(y_range * 0.06, 1e-9)
    label_y_col: list[Optional[float]] = [None] * n

    if color_col is not None and color_col in df.columns:
        unique_colors = sorted(df[color_col].unique().to_list(), key=str)
        for stack_i, cls in enumerate(unique_colors):
            group = df.filter(pl.col(color_col) == cls)
            if group.is_empty():
                continue
            # metric_fn expects numpy arrays (AUC/AP/Brier integration)
            import numpy as np
            metric = metric_fn(
                np.asarray(group[x_col].to_list(), dtype=float),
                np.asarray(group[y_col].to_list(), dtype=float),
            )
            text = f"{label.prefix}{metric:{label.format}}"
            endpoint_row = group.sort(x_col, descending=True).row(0, named=True)
            global_idx = int(endpoint_row["_idx"])
            labels_col[global_idx] = text
            label_y_col[global_idx] = y_top - stack_i * stagger_step
    else:
        import numpy as np
        metric = metric_fn(
            np.asarray(df[x_col].to_list(), dtype=float),
            np.asarray(df[y_col].to_list(), dtype=float),
        )
        text = f"{label.prefix}{metric:{label.format}}"
        global_idx = df[x_col].arg_max()
        labels_col[global_idx] = text
        label_y_col[global_idx] = y_top

    base_pl = base._data if isinstance(base._data, pl.DataFrame) else pl.from_arrow(tbl)
    augmented = base_pl.with_columns(
        pl.Series("_label_text", labels_col, dtype=pl.Utf8),
        pl.Series("_label_y", label_y_col, dtype=pl.Float64),
    )
    base_aug = base._clone()
    base_aug._data = augmented
    annot_layer = (
        Chart(augmented)
        .mark_text(align="right", dx=-4, dy=-2)
        .encode(x=x_col, y="_label_y", text="_label_text")
    )
    return base_aug + annot_layer


@dataclass(frozen=True)
class AUCLabel:
    """Auto-placed AUC annotation for ROC charts — spec §3.11.

    ``chart + AUCLabel()`` reads the surrounding chart's line data
    (``x`` = FPR, ``y`` = TPR), computes trapezoidal AUC per series
    (grouped by ``color`` when present), and emits one text annotation
    per series at the line endpoint.
    """

    position: Literal["end", "corner"] = "end"
    format: str = ".3f"
    prefix: str = "AUC = "

    def __radd__(self, base: Chart) -> Chart:
        if not isinstance(base, Chart):
            return NotImplemented
        return _apply_metric_label(base, self, metric_fn=_trapezoid_auc)


@dataclass(frozen=True)
class APLabel:
    """Auto-placed Average Precision annotation for PR charts.

    Sibling of :class:`AUCLabel` for precision-recall curves. ``x`` is
    treated as recall and ``y`` as precision.
    """

    position: Literal["end", "corner"] = "end"
    format: str = ".3f"
    prefix: str = "AP = "

    def __radd__(self, base: Chart) -> Chart:
        if not isinstance(base, Chart):
            return NotImplemented
        return _apply_metric_label(base, self, metric_fn=_ap_step)


@dataclass(frozen=True)
class BrierLabel:
    """Auto-placed Brier-score annotation for calibration charts.

    ``x`` is treated as predicted probability and ``y`` as observed rate
    per bin. Multi-series charts emit one Brier per series.
    """

    position: Literal["end", "corner"] = "corner"
    format: str = ".3f"
    prefix: str = "Brier = "

    def __radd__(self, base: Chart) -> Chart:
        if not isinstance(base, Chart):
            return NotImplemented
        return _apply_metric_label(base, self, metric_fn=_brier_score)


@dataclass(frozen=True)
class OutlierLabel:
    """Auto-labeled high-leverage / high-residual points — spec §3.11."""

    threshold: float = 3.0
    field: Optional[str] = None
    label_field: Optional[str] = None
    max_labels: int = 10

    def __radd__(self, base: Chart) -> Chart:
        from ferrum._coerce import to_arrow_table

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
        df = df.with_columns(
            ((pl.col(field) - mu).abs() / sigma).alias("_z")
        )
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
    base: Chart,
    label_kind: str,
    *,
    x_col: str,
    y_col: str,
    color_col: Optional[str] = None,
    position: str = "end",
    fmt: str = ".3f",
    prefix: Optional[str] = None,
) -> Chart:
    """Apply a metric label to ``base`` with explicit field overrides.

    Schwabish SB3 helper used by figure-level diagnostic builders that
    compose ``AUCLabel`` / ``APLabel`` / ``BrierLabel`` onto charts whose
    ``_pending_stat_mark`` composite hides the encoding.
    """
    metric_map = {
        "auc": (_trapezoid_auc, "AUC = "),
        "ap": (_ap_step, "AP = "),
        "brier": (_brier_score, "Brier = "),
    }
    if label_kind not in metric_map:
        raise ValueError(
            f"_apply_metric_label_explicit(label_kind={label_kind!r}): "
            f"expected one of {sorted(metric_map)}"
        )
    metric_fn, default_prefix = metric_map[label_kind]
    pos_lit: Any = position
    prefix_str = prefix if prefix is not None else default_prefix
    if label_kind == "auc":
        label_obj: Any = AUCLabel(position=pos_lit, format=fmt, prefix=prefix_str)
    elif label_kind == "ap":
        label_obj = APLabel(position=pos_lit, format=fmt, prefix=prefix_str)
    else:
        label_obj = BrierLabel(position=pos_lit, format=fmt, prefix=prefix_str)
    return _apply_metric_label(
        base,
        label_obj,
        metric_fn=metric_fn,
        x_col_override=x_col,
        y_col_override=y_col,
        color_col_override=color_col,
    )


def annotate_arrow(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    label: Optional[str] = None,
    label_side: str = "start",
    stroke: Optional[str] = None,
) -> Chart:
    """Arrow from ``(x1, y1)`` to ``(x2, y2)`` with optional text label.

    Composes a ``mark_segment`` with optional ``annotate_text`` at the
    ``label_side`` endpoint.
    """
    # Spec §3.3 lists `arrow=True` for mark_segment but the validator does
    # not yet accept it; emit a plain segment for now.
    df = pl.DataFrame({"_x1": [x1], "_y1": [y1], "_x2": [x2], "_y2": [y2]})
    seg_kwargs: dict = {}
    if stroke is not None:
        seg_kwargs["stroke"] = stroke
    arrow_chart = (
        Chart(df)
        .mark_segment(**seg_kwargs)
        .encode(
            x="_x1",
            y="_y1",
            x2="_x2",
            y2="_y2",
        )
    )
    if label is None:
        return arrow_chart
    lx, ly = (x1, y1) if label_side == "start" else (x2, y2)
    dx = -6 if label_side == "start" else 6
    align = "right" if label_side == "start" else "left"
    return arrow_chart & annotate_text(lx, ly, label, dx=dx, align=align)
