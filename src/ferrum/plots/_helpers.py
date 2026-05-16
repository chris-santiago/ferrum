"""Shared helper functions used across diagnostic and plot builders.

Shared by the ``ferrum.plots.*`` domain modules.
"""

from __future__ import annotations

from typing import Any

import polars as pl

from ferrum._overrides import _apply_overrides


def _inject_cook_outliers(
    df: pl.DataFrame,
    *,
    kind: str = "studentized",
    threshold: float | str = "auto",
    x_col: str = "y_pred",
) -> pl.DataFrame:
    """Inject ``_cook_outlier_x`` / ``_cook_outlier_y`` columns.

    For each row whose ``cooks_distance`` exceeds ``threshold``, the
    outlier-coordinate columns hold the ``(x_col, y_col)`` pair; all
    other rows are null so the residual mark's outlier-overlay layer
    skips them. ``x_col`` defaults to ``"y_pred"`` (the residuals-vs-
    fitted view); passing ``"leverage"`` produces an outlier overlay
    keyed on the leverage panel's x axis.

    ``threshold`` accepts:
    - a float — use that value directly.
    - ``"auto"`` — use the conventional ``4 / n`` rule (Hair et al.).
    """
    if df.height == 0 or "cooks_distance" not in df.columns:
        return df
    if threshold == "auto":
        thr = 4.0 / df.height
    else:
        thr = float(threshold)
    y_col = "studentized_residual" if kind in ("studentized", "scaled") else "residual"
    # Polars treats NaN > anything as True (NaN sorts after Infinity), so
    # the comparison must be guarded with `is_not_nan()` to keep non-linear
    # estimators (whose cooks_distance is NaN for every row) from
    # silently flagging every observation as an outlier.
    is_outlier = (
        pl.col("cooks_distance").is_not_nan()
        & pl.col("cooks_distance").is_not_null()
        & (pl.col("cooks_distance") > thr)
    )
    return df.with_columns(
        pl.when(is_outlier).then(pl.col(x_col)).otherwise(None).alias("_cook_outlier_x"),
        pl.when(is_outlier).then(pl.col(y_col)).otherwise(None).alias("_cook_outlier_y"),
    )


def _r2_score(y_true: pl.Series, y_pred: pl.Series) -> float:
    """Coefficient of determination — Schwabish SB3 corner-metrics helper."""
    diff = y_true - y_pred
    ss_res = float((diff**2).sum())
    mean_y = float(y_true.mean())
    ss_tot = float(((y_true - mean_y) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def _inject_metrics_corner(
    df: pl.DataFrame,
    *,
    kind: str = "studentized",
) -> pl.DataFrame:
    """Augment a residuals DataFrame with ``_metrics_text`` and ``_metrics_y``.

    Reads ``y_true`` / ``y_pred`` to compute R² / RMSE / MAE, then writes
    the formatted corner string on the single row with the largest
    ``y_pred`` (other rows null, so ``mark_text`` skips them). ``_metrics_y``
    anchors the text to the top of the residual-axis range so the
    annotation lands in the top-right corner of the plot.

    Schwabish SB3 (2026-05-11). Used by both the single-panel residuals
    chart and the residuals-vs-fitted cell of the 4-panel layout; the
    column is harmless on the other panels (their renderers do not read
    it). ``kind`` selects ``studentized_residual`` vs ``residual`` for
    the y-anchor — must match what ``mark_residuals`` will render so the
    text lands at the correct corner.
    """
    if df.height == 0:
        return df
    y_col = "studentized_residual" if kind in ("studentized", "scaled") else "residual"
    from ferrum._metrics_fmt import format_corner_metrics

    diff = df["y_pred"] - df["y_true"]
    r2 = _r2_score(df["y_true"], df["y_pred"])
    rmse = float((diff**2).mean() ** 0.5)
    mae = float(diff.abs().mean())
    corner_text = format_corner_metrics(r2, rmse, mae)

    n = df.height
    anchor_idx = df["y_pred"].arg_max()
    text_col: list[str | None] = [None] * n
    text_col[anchor_idx] = corner_text
    y_col_vals: list[float | None] = [None] * n
    y_col_vals[anchor_idx] = float(df[y_col].max())
    return df.with_columns(
        pl.Series("_metrics_text", text_col, dtype=pl.Utf8),
        pl.Series("_metrics_y", y_col_vals, dtype=pl.Float64),
    )


def _overlay_metrics_corner(chart):
    """Add a same-data ``mark_text`` overlay reading the injected columns.

    Assumes ``chart._data`` already carries ``_metrics_text`` / ``_metrics_y``
    (see :func:`_inject_metrics_corner`). The overlay layer reuses
    ``chart._data`` so ``+`` produces a true layer rather than the
    HConcat fallback.

    Schwabish SB3 (2026-05-11). Returns the original chart unchanged
    when the columns are absent, so callers can invoke unconditionally.
    """
    from ferrum._layer import _Layer

    data = chart._data
    if data is None or "_metrics_text" not in data.columns:
        return chart
    return chart.layer(
        _Layer(
            mark="text",
            encoding={"x": "y_pred", "y": "_metrics_y", "text": "_metrics_text"},
            mark_kwargs={"align": "right", "dx": -4, "dy": 4},
            name="metrics",
        )
    )


def _sort_by(df: pl.DataFrame, col: str) -> pl.DataFrame:
    """Sort the frame ascending by `col` so a downstream ``mark_line`` over
    that column draws a monotonic polyline.
    """
    if col not in df.columns:
        return df
    return df.sort(col, nulls_last=True)


def _grid_panels(charts: list, theme: Any = None):
    """Compose up to 4 panels into a grid using Phase 8a hstack/vstack."""
    if len(charts) == 1:
        c = charts[0]
    elif len(charts) == 2:
        c = charts[0] | charts[1]
    elif len(charts) == 3:
        c = (charts[0] | charts[1]) & charts[2]
    else:
        c = (charts[0] | charts[1]) & (charts[2] | charts[3])
    if theme is not None:
        c = c.theme(theme)
    return c


def _color_field_for(df: pl.DataFrame, default: str) -> str:
    """Return ``'model'`` if a ``model`` column is present (compare-source
    path), otherwise the supplied default.
    """
    return "model" if "model" in df.columns else default


def _dedupe_aggregated(df: pl.DataFrame, *group_keys: str) -> pl.DataFrame:
    """Drop per-fold duplicate rows when only the aggregated (mean/lower/upper)
    columns are needed. Sorts ascending by the primary group key so a
    downstream line layer renders a monotonic polyline.
    """
    keep = df.unique(subset=list(group_keys), keep="first", maintain_order=True)
    return keep.sort(list(group_keys), nulls_last=True)


def _coerce_to_polars(data: Any) -> pl.DataFrame:
    """Coerce a polars / pandas / 2D-numpy input into a polars DataFrame."""
    import numpy as np

    if isinstance(data, pl.DataFrame):
        return data
    if hasattr(data, "to_numpy") and hasattr(data, "columns"):
        return pl.from_pandas(data)
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"data must be a 2D array; got shape {arr.shape}")
    return pl.DataFrame({f"f{j}": arr[:, j].tolist() for j in range(arr.shape[1])})


def _require(func_name: str, arg_name: str, value: Any, *, hint: str) -> Any:
    """Raise ``ValueError`` when a required figure-function argument is ``None``."""
    if value is None:
        raise ValueError(
            f"{func_name}({arg_name}=...) is required — {hint}.",
        )
    return value


def _finalize_chart(chart, *, mark=None, encode=None, properties=None, layers=None, theme=None):
    """Apply overrides and optional theme to a chart, then return it.

    Encapsulates the identical 3-4 line closing pattern shared by every
    ``_*_from_source`` builder across the ``ferrum.plots.*`` domain modules.
    """
    chart = _apply_overrides(chart, mark=mark, encode=encode, properties=properties, layers=layers)
    if theme is not None:
        chart = chart.theme(theme)
    return chart


def _resolve_source(
    model: Any,
    X: Any = None,
    y: Any = None,
    *,
    y_true: Any = None,
    y_pred: Any = None,
    random_state: int | None = None,
    compare: dict[str, Any] | None = None,
) -> Any:
    """Resolve a figure-function input into a ``ModelSource``, ``ComparedModelSource``,
    or ``_PrecomputedSource``.

    Exactly one input path must be active:
    - **Model path**: ``model`` is not ``None``; ``y_true``/``y_pred`` must both be ``None``.
    - **Precomputed path**: both ``y_true`` and ``y_pred`` are not ``None``; ``model`` must be ``None``.
    - **Neither**: ``ValueError``.

    The precomputed path is incompatible with ``compare=``.
    """
    import ferrum
    from ferrum._diagnostics.source import ComparedModelSource

    has_precomputed = y_true is not None or y_pred is not None
    has_model = model is not None

    if has_precomputed:
        if has_model:
            raise ValueError(
                "Supply either a model/source (model=) or precomputed arrays "
                "(y_true=, y_pred=), not both."
            )
        if compare is not None:
            raise ValueError(
                "compare= is not supported with precomputed y_true/y_pred inputs.  "
                "Multi-model comparison requires a fitted model on each path."
            )
        if y_true is None or y_pred is None:
            missing = "y_pred" if y_true is not None else "y_true"
            raise ValueError(
                f"Precomputed path requires both y_true= and y_pred=; {missing}= is missing."
            )
        from ferrum._diagnostics.precomputed import _PrecomputedSource

        return _PrecomputedSource(y_true, y_pred)

    if not has_model:
        raise ValueError(
            "Supply either a fitted model/source (model=) or precomputed arrays (y_true=, y_pred=)."
        )

    if isinstance(model, ComparedModelSource):
        return model
    if compare is not None:
        if not isinstance(compare, dict):
            raise TypeError(
                f"compare= must be dict[str, model] or None; got {type(compare).__name__}."
            )
        models = {"base": model, **compare}
        return ferrum.ModelSource.compare(
            models,
            X,
            y,
            random_state=random_state,
        )
    if isinstance(model, dict):
        return ferrum.ModelSource.compare(
            model,
            X,
            y,
            random_state=random_state,
        )
    if isinstance(model, ferrum.ModelSource):
        return model
    return ferrum.ModelSource(model, X, y, random_state=random_state)
