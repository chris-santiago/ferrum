"""Shared helpers for composite-mark desugar functions.

These utilities are internal to the marks package and not part of the public API.
"""

from __future__ import annotations

import re

import polars as pl


def _sort_by(df: pl.DataFrame, col: str) -> pl.DataFrame:
    """Sort the frame ascending by `col` so a downstream ``mark_line`` over
    that column draws a monotonic polyline.
    """
    if col not in df.columns:
        return df
    return df.sort(col, nulls_last=True)


def _utf8_col(name: str) -> pl.Expr:
    """Cast a string-typed discriminator column to ``Utf8`` before comparing
    it against a Python string literal.

    Shared by every P9 row filter (``average``, ``reference_line``,
    ``split``) so an out-of-contract numeric/categorical dtype produces a
    normal empty-match miss instead of polars' ``ComputeError: cannot
    compare string with numeric type`` — one shared cast, not a
    per-filter variant.
    """
    return pl.col(name).cast(pl.Utf8)


def _normalize_names(values: "set[str] | frozenset[str] | tuple[str, ...]") -> set[str]:
    """Lowercase and strip non-alphanumeric characters for loose comparison.

    Used to recognize a case/spacing relabeling (e.g. a figure builder that
    renames ``"queue_rate"`` to ``"Queue rate"`` for display) as the *same*
    name rather than a mismatch, without hardcoding the specific rename map.
    """
    return {re.sub(r"[^a-z0-9]", "", str(v).lower()) for v in values}


def _filter_class_average(df: pl.DataFrame, average: str | None, *, mark_name: str) -> pl.DataFrame:
    """Restrict ``df`` to the requested ``average`` row(s) on a ``class`` column.

    Shared by ``Chart.mark_roc`` / ``Chart.mark_pr``: both accept an
    ``average`` kwarg (``"macro"``/``"micro"``/``"weighted"``) selecting a
    single summary curve out of a ``class`` discriminator column emitted by
    ``ModelSource.roc_curve``/``pr_curve``.  A figure builder that annotates
    curves (``annotate_auc``/``annotate_ap``) rewrites ``class`` values to
    ``"{class} ({METRIC} = {value:.3f})"`` *before* calling the mark, so an
    exact match alone would miss the annotated row — match either the raw
    value or that renamed-prefix form.

    When ``average`` matches no row, the correct response depends on *why*:

    - **Binary curves never carry an average row at all** — the ``class``
      column holds exactly one value (the single positive-class curve), so
      a request like ``average="macro"`` can never match anything.  This is
      the load-bearing case: ``roc_chart``/``pr_chart`` unconditionally
      forward their own ``average`` default to the mark even for binary
      models (``average=None if per_class else average``), so silently
      leaving ``df`` unfiltered here is required, not a fallback of last
      resort.
    - **Multiclass data with more than one ``class`` value present** means
      an average row plausibly *could* have matched — a mismatch here is
      most likely a typo (e.g. ``average="marco"``), so this warns once
      instead of silently rendering every class.
    """
    if average is None or "class" not in df.columns:
        return df
    class_col = _utf8_col("class")
    mask = (class_col == average) | class_col.str.starts_with(f"{average} (")
    filtered = df.filter(mask)
    if filtered.height > 0:
        return filtered
    if df["class"].cast(pl.Utf8).n_unique() > 1:
        from ferrum._warn import warn_once

        warn_once(
            mark_name,
            "average",
            f"{mark_name}(average={average!r}) matched no rows in the class "
            "column; rendering every class unfiltered. Check for a typo in "
            "average= (expected one of 'macro', 'micro', 'weighted', or a "
            "value actually present in the data).",
        )
    return df


def resolve_cmap_alias(
    *,
    scheme: str | None,
    cmap: str | None,
    where: str,
) -> str | None:
    """Resolve the canonical ``scheme=`` / legacy ``cmap=`` colormap alias.

    ``scheme`` is the canonical kwarg name for a named color set (D-COLOR-1);
    ``cmap`` is a documented back-compat alias.  Exactly one of them (or
    neither) may be supplied.  The resolved name is returned for the mark's
    internal colormap kwarg; ``None`` means "defer to the theme's scheme".

    Parameters
    ----------
    scheme:
        The canonical colormap name, or None.
    cmap:
        The legacy-alias colormap name, or None.
    where:
        Construction-site label (e.g. ``"mark_raster"``) for the error message.

    Returns
    -------
    str or None
        The resolved colormap name, or None when both are None.

    Raises
    ------
    ValueError
        When both ``scheme`` and ``cmap`` are supplied with different values.
    """
    if scheme is not None and cmap is not None and scheme != cmap:
        raise ValueError(
            f"{where}: pass either scheme= or cmap= (its alias), not both with "
            f"different values; got scheme={scheme!r}, cmap={cmap!r}"
        )
    return scheme if scheme is not None else cmap


def resolve_color_groupby(
    cat_field: str | None,
    color_field: str | None,
    base_groupby: list[str | None],
) -> tuple[list[str], bool]:
    """Return (groupby, split_hue) for a composite mark that groups by a categorical field.

    Parameters
    ----------
    cat_field:
        The primary categorical grouping field (e.g. the x-axis category for a
        vertical boxplot/violin/errorbar).  May be None.
    color_field:
        The hue/color encoding field, or None when no color encoding is present.
    base_groupby:
        The base list of groupby columns (without any color field appended).
        Typically ``[cat_field]`` but callers may pass a pre-built list (e.g.
        for multi-field grouping).  ``None`` entries are silently dropped so
        callers can safely pass ``[cat_field]`` even when ``cat_field`` is None.

    Returns
    -------
    groupby : list[str]
        The groupby list with the color field appended when ``split_hue`` is True.
    split_hue : bool
        True when ``color_field`` is non-None AND distinct from ``cat_field``
        (i.e. it names a genuinely different column that should split the groups).
        When ``color_field == cat_field``, adding it to groupby would create a
        duplicate that Rust transforms reject, so ``split_hue`` is False.
    """
    split_hue = color_field is not None and color_field != cat_field
    raw = base_groupby + ([color_field] if split_hue else [])
    groupby: list[str] = [g for g in raw if g is not None]
    return groupby, split_hue
