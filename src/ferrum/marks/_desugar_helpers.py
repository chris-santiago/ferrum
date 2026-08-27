"""Shared helpers for composite-mark desugar functions.

These utilities are internal to the marks package and not part of the public API.
"""

from __future__ import annotations

import polars as pl


def _sort_by(df: pl.DataFrame, col: str) -> pl.DataFrame:
    """Sort the frame ascending by `col` so a downstream ``mark_line`` over
    that column draws a monotonic polyline.
    """
    if col not in df.columns:
        return df
    return df.sort(col, nulls_last=True)


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
