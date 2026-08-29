"""Shared helpers for composite-mark desugar functions.

These utilities are internal to the marks package and not part of the public API.
"""

from __future__ import annotations

import re

import polars as pl

#: Feature-ranking / display-ordering vocabulary shared by
#: ``Chart.mark_shap_beeswarm(order=)`` and the ``shap_chart`` figure-function
#: family (``ferrum.plots.explanation._shap_order_features`` and siblings),
#: so both sides of that pair validate one closed set with one meaning per
#: value instead of two independently-drifting vocabularies (2026-08-27
#: close-out -- previously ``{"abs_mean", "max"}`` on the figure side vs.
#: ``{"abs_mean", "mean", "none"}`` on the mark side, a sibling-drift finding
#: from the findings-remediation design review; unified to the **union** of
#: both, not a narrowing -- ``"max"`` was pre-existing, documented,
#: implemented figure-side behavior and dropping it would have broken it).
#: ``"abs_mean"`` ranks by descending ``mean(|shap_value|)``; ``"max"`` by
#: descending ``max(|shap_value|)`` (surfaces high-impact-outlier features);
#: ``"mean"`` by descending signed ``mean(shap_value)``; ``"none"`` performs
#: no reordering.
SHAP_ORDER_VALUES: tuple[str, ...] = ("abs_mean", "mean", "max", "none")

#: Presentable name for the SHAP beeswarm's per-point feature-value color
#: field. ``Chart.mark_shap_beeswarm``'s ``data_transform`` renames the raw
#: ``ModelSource.shap_values()`` column (``feature_value_normalized``) to
#: this label before the chart holds the data (2026-08-27 close-out). The
#: rename exists because the Rust colorbar-legend construction falls back to
#: the field name when ``Color(title=)`` is dropped -- a pre-existing,
#: package-wide gap (see ``design-docs/superpowers/followups/2026-05-15-code-archaeology.md``)
#: that this mark cannot fix from Python. Renaming the column is the
#: proportionate workaround: it makes the fallback render something a user
#: would want to see instead of an internal schema name.
SHAP_BEESWARM_COLOR_FIELD = "Feature value"


def nominal_color_channel(field_name: str | None):
    """Bind *field_name* to the color channel typed Nominal, or return
    ``None`` unchanged when *field_name* is ``None``.

    Shared by every diagnostic desugar that groups per-class/per-group
    polylines (or ribbon/rule CI bands) by a caller-named discriminator
    column on a line-or-ribbon mark: the classification-curve family
    (roc/pr/gain/lift/calibration, ``marks/diagnostic/_classification.py``),
    PDP/ICE (``marks/diagnostic/_explanation.py``), learning/validation
    curve CI bands + mean lines (``marks/diagnostic/_selection.py``), and
    parallel coordinates (``marks/diagnostic/_ranking.py``).

    Left untyped, a plain string field name infers its scale type from the
    column's runtime dtype: a numeric dtype (e.g. an integer class/model-id
    column, or any caller-supplied ``color_field`` override that happens to
    be numeric) infers Quantitative -> Continuous, which is inert on a line
    or ribbon mark and trips the ``UnsupportedColorScaleOnMark`` warning
    (the colorbar is suppressed instead of a per-group symbol legend, and
    on some marks the per-group polylines collapse into one). Every one of
    these desugars' discriminator fields is categorical in intent regardless
    of runtime dtype -- a Continuous color scale was never a legitimate
    reading of "which curve/segment does this row belong to" -- so bind it
    Nominal explicitly. This also keeps a Utf8-typed column byte-identical,
    since Nominal is what it would already infer to.

    The ``None`` pass-through lets call sites that build an encoding dict
    literal (e.g. ``desugar_learning_curve``'s CI-band/line layers, which do
    not gate the assignment behind ``if color_field is not None``) call this
    unconditionally: ``_build_layers_list`` already drops an encoding
    channel whose resolved field is falsy, so a ``None`` here reaches the
    same "channel absent" outcome a bare ``None`` value did before this
    helper existed.
    """
    if field_name is None:
        return None
    from ferrum.encoding import Color

    return Color(field_name, type_="nominal")


def shap_beeswarm_color_channel(*, color_bar: bool):
    """Return the ``Color(...)`` channel shared by both places that need it.

    ``desugar_shap_beeswarm`` sets this on the point layer to drive per-point
    fill; ``Chart.mark_shap_beeswarm`` mirrors the identical config onto the
    chart-level encoding because the Rust colorbar-legend construction for a
    layered chart reads ``legend=`` from ``encoding.color`` at the chart
    level, not from any per-layer color channel. One factory keeps both call
    sites from drifting out of sync (2026-08-27 close-out).
    """
    from ferrum.encoding import Color

    legend = {"tickLabels": ["Low", "", "", "", "High"]} if color_bar else None
    return Color(
        SHAP_BEESWARM_COLOR_FIELD, scheme="rdbu", title=SHAP_BEESWARM_COLOR_FIELD, legend=legend
    )


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


#: Shared by :func:`_normalize_names` (Python-side) and :func:`_normalized_col`
#: (polars-side) so the loose case/spacing-match rule has exactly one
#: definition instead of two copies that can silently drift apart.
_NORMALIZE_PATTERN = r"[^a-z0-9]"


def _normalize_names(values: "set[str] | frozenset[str] | tuple[str, ...]") -> set[str]:
    """Lowercase and strip non-alphanumeric characters for loose comparison.

    Used to recognize a case/spacing relabeling (e.g. a figure builder that
    renames ``"queue_rate"`` to ``"Queue rate"`` for display) as the *same*
    name rather than a mismatch, without hardcoding the specific rename map.
    """
    return {re.sub(_NORMALIZE_PATTERN, "", str(v).lower()) for v in values}


def _normalized_col(name: str) -> pl.Expr:
    """Polars-side counterpart to :func:`_normalize_names`: lowercase and
    strip non-alphanumeric characters from a column, for loose comparison
    against a normalized Python string literal.

    Built off the same :data:`_NORMALIZE_PATTERN` regex so the two loose-
    match rules cannot silently diverge (a divergence between "how one name
    is written in two places" is exactly the class of bug this pairs with —
    see the discrimination-threshold F1-row lookup that used to hand-roll
    this a second time). Casts via :func:`_utf8_col` first so a non-string
    discriminator column produces a normal empty-match miss rather than a
    polars ``ComputeError``.
    """
    return _utf8_col(name).str.to_lowercase().str.replace_all(_NORMALIZE_PATTERN, "")


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


def _roc_render_frame(
    df: pl.DataFrame, average: str | None, *, reference_line: bool, mark_name: str = "mark_roc"
) -> pl.DataFrame:
    """Return the exact frame ``mark_roc`` renders: class-filtered, then
    ``fpr``-sorted when ``reference_line`` is set.

    Shared by ``Chart.mark_roc``'s ``data_transform`` closure and
    ``roc_chart``'s post-filter AUC title (``plots/classification.py``), so
    the figure builder derives the rendered frame from this one function
    instead of reading ``Chart._data`` and depending on
    ``_set_composite_mark``'s internal ordering guarantee.
    """
    df = _filter_class_average(df, average, mark_name=mark_name)
    if reference_line:
        df = _sort_by(df, "fpr")
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


def identity_transform(name: str):
    """Construct a named ``PyIdentity`` pass-through transform.

    Three sites route a layer through its own named identity transform
    (same data, new name) specifically so the layer does NOT inherit a
    chart-level or top-level encoding it should not have: ``Chart.__add__``'s
    column-collision path (``chart.py``), the rug-overlay layer in
    ``plots/distribution.py``, and the threshold-line rule layer in
    ``desugar_discrimination_threshold`` (``marks/diagnostic/_classification.py``).
    Each site still builds its own name and wires the resulting transform
    into its own transform list / ``_NamedTransform`` wrapper / layer
    ``data_source=`` differently -- this hoists only the shared
    construction step, which previously had three different import
    aliases (``_RustIdentity``, ``_IdentityTransform``, bare ``PyIdentity``)
    for the same class.

    Named ``identity_transform`` rather than re-exporting the bare
    ``PyIdentity`` name some call sites used to import directly: the
    public ``ferrum.Identity`` is the unrelated *position adjustment*
    (``fm.Identity()`` for ``position=``), and a bare re-export invites
    exactly that mix-up.
    """
    from ferrum._core import PyIdentity

    return PyIdentity(name)


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
