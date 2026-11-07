"""Shared helper functions used across diagnostic and plot builders.

Shared by the ``ferrum.plots.*`` domain modules.
"""

from __future__ import annotations

import functools
import inspect
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Callable, Iterable, NamedTuple, TypeVar

import polars as pl

from ferrum._compose import figure_recorded
from ferrum._encode_lowering import COMPOSITE_GROUPING_CHANNELS, _composite_groupby
from ferrum._entry import EntryRefusal, _voiced, speak_as
from ferrum._overrides import _apply_overrides, refuse_malformed_mark
from ferrum._validate import coerce_positive_pixel_value, coerce_positive_real
from ferrum._layer import merge_style, set_style
from ferrum.color import to_hex
from ferrum.encoding import X
from ferrum.encoding._resolve import channel_for_value
from ferrum.encoding.base import ChannelBase
from ferrum.marks._desugar_helpers import nominal_color_channel
from ferrum.marks.base import resolve_layer_style

if TYPE_CHECKING:
    from ferrum.chart import Chart
    from ferrum.composition import ConcatChart


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


def _grid_panels(charts: list, theme: Any = None):
    """Compose up to 4 panels into a grid using Phase 8a hstack/vstack."""
    if not 1 <= len(charts) <= 4:
        raise ValueError(
            f"_grid_panels() requires 1-4 charts to compose into a grid; got {len(charts)}."
        )
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


def _color_field_for(df: pl.DataFrame, default: str | None) -> str | None:
    """Return ``'model'`` if a ``model`` column is present (compare-source
    path), otherwise the supplied default (which may be ``None`` for marks
    whose single-model colour field is unset).
    """
    return "model" if "model" in df.columns else default


def _field_name(value: Any) -> str | None:
    """Extract the plain column-name string from a ``str`` or encoding object.

    Several figure-level builders document a parameter (typically ``hue``)
    as accepting ``str or encoding`` (e.g. ``fm.Color("grp")``), then need
    the bare column name to thread into a transform's ``groupby=`` kwarg —
    which rejects anything but ``None``, a ``str``, or a ``list[str]``.
    Passing the raw encoding object through silently drops the group (or
    raises, depending on the transform) instead of honoring the documented
    contract. Mirrors the ad hoc ``hue.field if hasattr(hue, "field") else
    str(hue)`` pattern already used by ``lmplot`` (``regression.py``);
    pulled out here as the one shared extraction point for ``pairplot`` and
    ``jointplot`` (``matrix.py``).
    """
    if value is None:
        return None
    return value.field if hasattr(value, "field") else str(value)


def _drawn_bindings(bindings: dict, encode: dict | None) -> dict:
    """Return the bindings a figure's chart draws once ``_finalize_chart``
    applies ``encode=`` over its keyword *bindings*, read the way
    ``Chart.encode`` itself reads ``encode=``: a value replaces the
    channel's binding, and a ``None`` leaves that binding in place rather
    than unbinding it (:func:`ferrum._layer.merge_style`, the one
    None-is-unset merge every style door already shares). A figure that
    computes a statistic, builds a fit layer's channels, or reads a
    channel's scale before ``encode=`` is applied reads this instead of
    re-deriving the merge by hand, so what it groups or scales by is what
    the chart actually draws.
    """
    return merge_style(bindings, encode or {})


def _drawn_positional_channels(bindings: dict, encode: dict | None) -> dict[str, ChannelBase]:
    """Return the x/y channel objects the figure's chart draws once ``encode=`` applies.

    Reads :func:`_drawn_bindings` -- the merge every figure applies at
    ``_finalize_chart`` -- then, for ``"x"`` and ``"y"``, keeps a
    :class:`ChannelBase` as is and resolves a plain string the way
    ``encode()`` reads it (:func:`channel_for_value`, the same shorthand
    reading :func:`_encode_grouping` uses for its own channels), so
    ``"mean(b)"`` is the ``Y("b", aggregate="mean")`` the chart draws. A
    missing binding, ``None``, a ``Repeat`` placeholder or any other value
    is omitted: each is left to the door's own refusals and to
    ``Chart.encode``.
    """
    drawn = _drawn_bindings(bindings, encode)
    channels: dict[str, ChannelBase] = {}
    for name in ("x", "y"):
        value = drawn.get(name)
        if isinstance(value, ChannelBase):
            channels[name] = value
        elif isinstance(value, str):
            channels[name] = channel_for_value(name, value, declare=f"encode({name}=...)")
    return channels


def _encode_grouping(bindings: dict) -> list[str]:
    """Return the composite grouping fields the caller's own *bindings* bind.

    Each grouping channel's value is read as ``encode()`` reads it
    (shorthand included), so ``"c:N"`` groups by ``c``; ``None`` binds
    nothing. Shared by every figure that fits or counts one curve per group
    the same way a composite mark's statistic does (``lmplot``/``regplot``/
    ``residplot``, ``displot(kind="ecdf")``).
    """
    channels = {
        name: channel_for_value(name, value, declare=f"encode({name}=...)")
        for name, value in bindings.items()
        if name in COMPOSITE_GROUPING_CHANNELS and value is not None
    }
    return _composite_groupby(channels, None)


def _log_scaled_x(entry: str, value: Any, field: str, *, param: str) -> Any:
    """Return the caller's x channel, with a log scale added, voiced as *entry*/*param*.

    A bare column name becomes ``X(field, scale=log)``. A channel object the
    caller built themselves is REBUILT with the log scale added -- same
    class, same field, its own kwargs preserved (the idiom
    ``nominal_color_channel`` uses for the same situation) -- rather than
    replaced by ``X(field, ...)``, which would silently drop their
    ``title=``/``axis=``, or passed to ``X(...)`` as a field, which raises
    ``X: field must be str ... got X`` since a channel constructor refuses a
    non-``str`` field.

    A channel that already carries its own ``scale=`` is two instructions for
    one scale slot, so it is refused rather than silently overruled in either
    direction. *param* names the keyword the caller wrote (``lmplot``'s
    ``logx``, ``displot``'s ``log_scale``), so the refusal reads in their own
    voice.
    """
    if not hasattr(value, "field"):
        return X(field, scale={"type": "log"})
    if value.option("scale") is not None:
        raise ValueError(
            f"{entry}({param}=True, x=<channel with its own scale=>): two "
            f"instructions for one x scale. Keep your scale= and drop {param}=True, "
            f"or pass x as a column name and let {param} set the log scale."
        )
    return value._with_options(scale={"type": "log"})


def _title_legend(chart: "Chart", channel: str, title: str) -> "Chart":
    """Title *chart*'s *channel* legend *title*, on the bindings its marks chose.

    The one way a figure titles a legend. A composite mark binds *channel*
    on its own layers to a column it chose (``mark_confusion``'s cell
    value, ``mark_roc``'s class); every layer binding *channel* to a column
    takes *title*, so the figure names the legend without naming the column
    and without a chart-level binding its other layers (a reference line)
    would inherit. A layer that binds no *channel* is left alone. A flat
    chart titles its own binding. A pending composite mark carries the
    title to its resolution (``_PendingMark.legend_titles``), so the figure
    does not resolve it early.
    """
    from ferrum._layer import titled_encoding

    title_pair = ((channel, title),)
    new = chart._clone()
    pending = new._pending_stat_mark
    if pending is not None:
        new._pending_stat_mark = replace(pending, legend_titles=pending.legend_titles + title_pair)
    elif new._layers:
        new._layers = [
            replace(ly, encoding=titled_encoding(ly.encoding, title_pair)) for ly in new._layers
        ]
    else:
        new._encoding = titled_encoding(new._encoding, title_pair)
    return new


class _ColorArgument(NamedTuple):
    """How a two-way ``color=`` argument resolved.

    Exactly one interpretation is live: either the argument named a data
    column -- ``group_field`` is the column name and ``channel`` is the
    ready-to-encode color channel -- or it named a literal color, in which
    case ``paint`` is its normalized hex and the other two are ``None``.
    ``color=None`` leaves all three ``None``.
    """

    group_field: str | None
    channel: "ChannelBase | None"
    paint: str | None


def _column_or_literal_color(
    color: Any,
    columns: Iterable[str],
    *,
    func_name: str,
) -> _ColorArgument:
    """Resolve a ``color=`` argument documented as "column name *or* constant color".

    The dispatch order is normative: **column membership in the input frame
    wins**, so a column literally named ``"red"`` groups by that column and
    never paints. Only when no such column exists is the value offered to
    ferrum's single color parser (:func:`ferrum.color.to_hex`), and only when
    that also refuses does this raise -- naming both interpretations, because
    from the outside a rejected value is ambiguous between a column typo and a
    color typo.

    The column branch types the channel Nominal via ``nominal_color_channel``
    (an already-constructed channel object passes through untouched): a
    ``color=`` that drives a transform's ``groupby=`` is a discriminator by
    construction, so a numeric group column must not infer a Continuous scale
    -- and its consumer layers all share one color scale.

    The literal branch normalizes through ``to_hex`` so the value handed to a
    mark's ``fill=``/``stroke=`` is the same hex every other ferrum color
    boundary would produce, tuples included.
    """
    if color is None:
        return _ColorArgument(None, None, None)

    field = _field_name(color)
    if field is not None and field in set(columns):
        return _ColorArgument(field, nominal_color_channel(color), None)

    try:
        paint = to_hex(color)
    except (ValueError, TypeError) as exc:
        # The third leg of the dispatch is normatively "a typed ValueError
        # naming both readings", and this is the site that owes it. `to_hex`
        # now raises `ValueError` for every unparseable tuple, including a
        # non-numeric one (`("a", "b", "c")`) -- previously that shape
        # escaped `to_hex` as a raw, uncaught `TypeError` from the internal
        # component comparison, which this `except` clause had to name and
        # catch specially (#122 closed that escape in `ferrum.color.to_hex`
        # itself). The `TypeError` arm stays here as a defensive belt, not
        # because `to_hex` is known to still raise one.
        raise ValueError(
            f"{func_name}(color={color!r}): not a column of the data and not a "
            f"parseable color. As a column name it would group the data, but no "
            f"column {field!r} exists; as a literal color it would paint the "
            f"marks, but the color parser refused it ({exc})."
        ) from exc
    return _ColorArgument(None, None, paint)


def _reject_compare(compare: dict | None, *, chart: str, reason: str) -> None:
    """Raise a clear ``ValueError`` when ``compare=`` is passed to a chart whose
    builder cannot render a multi-model :class:`ComparedModelSource`.

    Mirrors the loud rejection ``_resolve_source`` already applies on the
    precomputed path (D-COMPARE-1): the parameter exists for signature
    uniformity across the model-diagnostic family, but a non-``None`` value is
    surfaced as an explicit error rather than silently dropped.
    """
    if compare is not None:
        raise ValueError(f"compare= is not supported for {chart}: {reason}")


def _dedupe_aggregated(df: pl.DataFrame, *group_keys: str) -> pl.DataFrame:
    """Drop per-fold duplicate rows when only the aggregated (mean/lower/upper)
    columns are needed. Sorts ascending by the primary group key so a
    downstream line layer renders a monotonic polyline.
    """
    keep = df.unique(subset=list(group_keys), keep="first", maintain_order=True)
    return keep.sort(list(group_keys), nulls_last=True)


def _charts_with_endpoint_labels(
    chart: "Chart",
    *,
    label_field: str,
    x_col: str,
    y_col: str,
) -> "Chart":
    """Append endpoint direct labels to ``chart`` for each ``label_field`` series.

    Thin wrapper over :func:`ferrum._direct_label._direct_label_endpoint`
    that the gain / lift / learning-curve / validation-curve builders
    share. Always binds the label layer's own color to ``label_field``
    with no legend of its own, so each label reads in its own series'
    color -- the same field and the same ``legend=None`` suppression the
    base chart's own color binding already uses, rather than falling
    through to the theme's label ink.
    """
    from ferrum._direct_label import _direct_label_endpoint

    return _direct_label_endpoint(
        chart,
        label_field=label_field,
        x_col=x_col,
        y_col=y_col,
        color_field=label_field,
    )


def _is_raw_tabular_data(value: Any) -> bool:
    """Return ``True`` when ``value`` is raw tabular data, not a fitted estimator.

    Distinguishes "raw data the caller wants ranked/decomposed directly"
    from "a fitted estimator whose attributes must be resolved through a
    ``ModelSource``". ``rank1d_chart``/``rank2d_chart`` (``plots/ranking.py``)
    and ``pca_scree_chart`` (``plots/clustering.py``) all need this probe
    before dispatching, so a fitted estimator that reaches the raw-data path
    doesn't get handed to ``np.asarray()``/``polars_or_array_to_record_batch()``
    and leak an internal ``TypeError`` instead of the typed ``ModelSource``
    refusal (#216). Previously hand-copied at three call sites.

    The recognized-container checks (``pl.DataFrame``, ``pd.DataFrame``,
    ``pa.Table``, ``pa.RecordBatch``, 2-D ``np.ndarray``) cover the fast
    paths every raw-data coercer (``polars_or_array_to_record_batch`` in
    ``diagnostics/_internal/_rank_helpers.py``, ``to_polars`` in
    ``_coerce.py``) recognizes explicitly. Anything else falls through to
    the same probe those coercers themselves use as their own last resort
    (``np.asarray``): if numpy can wrap the value as an array of at least
    one dimension, it is candidate tabular data (e.g. a list-of-lists, or a
    malformed 1-D array), and the raw path's own coercer decides whether
    the shape is actually usable -- producing its accurate shape-mismatch
    message instead of this probe pre-empting it. A fitted estimator (e.g.
    a scikit-learn model) has no array structure, so ``np.asarray`` wraps
    it as a 0-D object array rather than raising; ``ndim == 0`` is exactly
    the signal that keeps such an estimator routed to ``ModelSource``
    resolution instead of leaking into the raw path (closes the #216
    residual: ``rank1d_chart``/``rank2d_chart`` had narrowed past what
    their own coercer accepts -- ``pa.Table``, ``pa.RecordBatch``, and
    list-of-lists all worked pre-batch and must keep working).
    """
    import numpy as np
    import pyarrow as pa

    if isinstance(value, pl.DataFrame):
        return True
    if isinstance(value, (pa.Table, pa.RecordBatch)):
        return True
    if isinstance(value, np.ndarray) and value.ndim == 2:
        return True
    try:
        import pandas as pd

        if isinstance(value, pd.DataFrame):
            return True
    except ImportError:
        pass
    try:
        # Broad catch is deliberate: this is a boolean probe over an
        # arbitrary caller-supplied object, and np.asarray raises whatever
        # exception type the underlying object's conversion machinery
        # chooses (ValueError, TypeError, or a library-specific error for
        # exotic __array__ implementations) -- any of them means "not
        # coercible", i.e. not raw tabular data.
        return np.asarray(value).ndim >= 1
    except Exception:
        return False


def _should_facet_by_class(df: pl.DataFrame, *, per_class: bool) -> bool:
    """Decide whether a SHAP chart facets by ``class_label``.

    A chart facets only when ``per_class`` is requested *and* the data
    carries more than one class. Shared by the SHAP beeswarm and bar
    builders, which previously inlined ``per_class and
    df["class_label"].n_unique() > 1`` at separate sites.
    """
    return per_class and df["class_label"].n_unique() > 1


def _zero_anchored_domain(lower: pl.Series, upper: pl.Series) -> tuple[float, float]:
    """Compute a zero-anchored value-axis domain with 5% headroom above the max.

    Shared domain formula for bar-style diagnostic charts (feature importance,
    SHAP bar) whose value axis conventionally starts at zero. The low end is
    ``min(0.0, lower.min())``; the high end is ``upper.max() * 1.05`` when
    positive, else ``1.0`` (an all-zero or all-negative frame still needs a
    non-zero-width domain). Previously duplicated at three call sites
    (GH #76).

    Non-finite (``inf``/``-inf``/``nan``) entries in either series are excluded from
    both aggregates -- a single infinite importance/SHAP value must not push
    the domain to infinity, which breaks spec-JSON serialization (Rust's
    serde rejects the bare ``Infinity`` token). The offending row's own value
    is left untouched by this function; it renders at (or just past) the
    finite domain edge. When a series has no finite entries at all, its
    aggregate falls back to ``0.0``.
    """
    finite_lower = lower.filter(lower.is_finite())
    finite_upper = upper.filter(upper.is_finite())
    lower_min = float(finite_lower.min()) if finite_lower.len() > 0 else 0.0
    upper_max = float(finite_upper.max()) if finite_upper.len() > 0 else 0.0
    domain_lo = min(0.0, lower_min)
    domain_hi = upper_max * 1.05 if upper_max > 0 else 1.0
    return domain_lo, domain_hi


def _require_positive(func_name: str, param: str, value: int | None) -> None:
    """Raise ``ValueError`` when an optional positive-integer parameter is < 1.

    ``None`` means "no limit" and passes through unchanged. This only rejects
    an explicit non-positive value (e.g. ``top_k=0``), which would otherwise
    empty the frame and crash a downstream ``.max()``/``.min()`` aggregate on
    an empty series with an unhelpfully generic ``TypeError`` (GH #76).
    """
    if value is not None and value < 1:
        raise ValueError(f"{func_name}: {param} must be >= 1 or None; got {value!r}.")


def _warn_deprecated_dispatcher(old_name: str, param_name: str, replacements: str) -> None:
    """Emit the canonical ``DeprecationWarning`` for a split dispatcher shim.

    Both ``shap_chart(kind=...)`` and ``rank_chart(rank=...)`` are deprecated
    dispatchers that warn then forward to their split siblings. This factors
    out the duplicated warn idiom so each shim is a thin validate-then-delegate
    wrapper; the heterogeneous per-kind dispatch stays explicit in each shim::

        {old_name}({param_name}=...) is deprecated; use {replacements} instead.

    ``stacklevel=3`` (this helper -> the shim -> the user) points the warning at
    the user's call site, preserving the pre-refactor behavior where each shim
    called ``warnings.warn(..., stacklevel=2)`` inline.
    """
    import warnings

    warnings.warn(
        f"{old_name}({param_name}=...) is deprecated; use {replacements} instead.",
        DeprecationWarning,
        stacklevel=3,
    )


def _require(func_name: str, arg_name: str, value: Any, *, hint: str) -> Any:
    """Raise ``ValueError`` when a required figure-function argument is ``None``."""
    if value is None:
        raise ValueError(
            f"{func_name}({arg_name}=...) is required — {hint}.",
        )
    return value


# Sentinel distinguishing "argument omitted" from an explicit ``None`` so that a
# deprecated keyword alias can be ``None`` and still count as supplied.
_UNSET: Any = object()


def _resolve_first_param(
    canonical_value: Any,
    alias_value: Any,
    *,
    canonical_name: str,
    alias_name: str,
    func_name: str,
) -> Any:
    """Resolve a renamed first positional parameter against its deprecated alias.

    The single keyword-alias mechanism shared by every figure function whose
    first positional parameter was renamed to its family-canonical name
    (D-FIRSTPARAM-1). The canonical parameter keeps the positional slot, so
    positional callers are unaffected; the old name is accepted as a deprecated
    keyword whose default is :data:`_UNSET` (so an explicit ``alias=None`` is
    still detected as "supplied").

    Returns the value the function should use. Raises ``TypeError`` when both
    the canonical and alias names are supplied, mirroring Python's own
    "got multiple values for argument" error shape.
    """
    if alias_value is _UNSET:
        return canonical_value
    if canonical_value is not _UNSET:
        raise TypeError(
            f"{func_name}() got both {canonical_name}= and {alias_name}=; "
            f"{alias_name}= is a deprecated alias for {canonical_name}=, "
            "supply only one."
        )
    return alias_value


def _finalize_chart(
    chart, *, entry: str, mark=None, encode=None, properties=None, layers=None, theme=None
):
    """Apply overrides and optional theme to a chart, then return it.

    Encapsulates the identical 3-4 line closing pattern shared by every
    ``_*_from_source`` builder across the ``ferrum.plots.*`` domain modules.
    *entry* is the figure function the builder builds for; a refusal of the
    overrides speaks as it, and the public call that actually ran (a figure
    building through another figure's builder, a visualizer's ``fit``)
    re-voices it as itself (:func:`ferrum._entry.voiced_as`).
    """
    chart = _apply_overrides(
        chart, entry=entry, mark=mark, encode=encode, properties=properties, layers=layers
    )
    if theme is not None:
        chart = chart.theme(theme)
    return chart


_F = TypeVar("_F", bound=Callable[..., Any])


def _style_kws(entry: str, param: str, mark: str, kws: Any) -> dict:
    """Return a figure's ``<param>=`` style dict (``line_kws``) for the layer drawing *mark*.

    Held to *mark*'s honored kwargs with aliases resolved
    (:func:`ferrum.marks.base.resolve_layer_style`); ``None`` is no style,
    and a key set to ``None`` is unset, so it is left out and the layer's own
    default for it stands (:func:`ferrum._layer.set_style`). A refusal is an
    :class:`EntryRefusal` voiced as *entry*, the figure that received the
    dict.
    """
    if kws is None:
        return {}
    if not isinstance(kws, dict):
        kind = type(kws).__name__
        raise EntryRefusal(
            lambda voice: f"{voice}({param}=...) must be a dict of mark kwargs; got {kind}", entry
        )
    return set_style(resolve_layer_style(entry, mark, kws))


def _checked_figure_size(
    entry: str, *, height: Any = None, aspect: Any = None
) -> tuple[float | None, int | float | None]:
    """Return a figure's user-passed ``height`` and ``aspect``, refusing a malformed one as *entry*.

    ``height`` is a viewport length (finite, positive, ``bool`` refused) and
    ``aspect`` a finite ratio above zero; ``None`` leaves either unset. The
    refusal names the argument the caller passed (``displot(aspect=...)``),
    so it is checked here, at the figure's entry, before any width or height
    is derived from it.
    """
    checked_height = _voiced(
        entry, lambda voice: coerce_positive_pixel_value(f"{voice}(height=...)", height)
    )
    checked_aspect = _voiced(
        entry, lambda voice: coerce_positive_real(f"{voice}(aspect=...)", aspect)
    )
    return checked_height, checked_aspect


def _checked_figsize(entry: str, figsize: Any) -> tuple[float, float] | None:
    """Return a figure's user-passed ``figsize`` as ``(width, height)`` floats, refusing a malformed one as *entry*.

    ``None`` leaves the size unset. Anything else must be a ``(width,
    height)`` pair of finite, positive pixel lengths; the refusal names
    ``figsize``, the argument the caller passed.
    """
    if figsize is None:
        return None

    def check(voice: str) -> tuple[float, float]:
        label = f"{voice}(figsize=...)"
        try:
            width, height = figsize
        except (TypeError, ValueError):
            raise ValueError(
                f"{label} must be a (width, height) pair; received {figsize!r}"
            ) from None
        return (
            coerce_positive_pixel_value(label, width, required=True),
            coerce_positive_pixel_value(label, height, required=True),
        )

    return _voiced(entry, check)


def _figure_entry(fn: _F) -> _F:
    """Make the figure function *fn* the public entry its chart speaks as.

    Every warning and refusal about a layer the figure draws (a line's
    ``size`` and ``stroke_width`` both set, a size the mark cannot draw, a
    degraded composite) names the public call the user made, not the
    ``mark_*`` method or primitive the figure used to draw it, and so does
    every refusal a door inside the figure raises while it builds (a
    ``mark=`` override, a ``line_kws=`` dict). Applies to every leaf of a
    composite result. Every public function of a
    ``ferrum.plots`` module carries it; ``tests/test_public_entry_voice.py``
    derives that population from the modules themselves.

    A malformed ``mark=`` (keyword-only on every figure function) refuses
    before *fn* runs, so before anything is computed from the model
    (:func:`ferrum._overrides.refuse_malformed_mark`).

    The chart it returns carries the record that builds its statistics again
    (:func:`ferrum._compose.figure_recorded`): a grouping bound after the
    call (the chart's own ``.encode(color=...)``, or a composition's it
    becomes an operand of) calls the figure again with that binding in its
    ``encode=``, so its statistics group by it as for the same ``encode=``
    passed to the call.
    """
    regroups = "encode" in inspect.signature(fn).parameters

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        refuse_malformed_mark(fn.__name__, kwargs.get("mark"))
        chart = speak_as(fn.__name__, fn, *args, **kwargs)
        if not regroups:
            return chart

        def call(extra: dict) -> Any:
            encode = {**(kwargs.get("encode") or {}), **extra}
            return wrapper(*args, **{**kwargs, "encode": encode})

        return figure_recorded(chart, call)

    return wrapper  # type: ignore[return-value]


def _compose_compare(
    source,
    builder,
    *,
    builder_kwargs: dict,
    resolve: dict[str, str],
    columns: int | None = None,
) -> "ConcatChart":
    """Build one panel per model via ``builder(model_source, **builder_kwargs)``,
    label each panel with its model name, and compose as small multiples.

    The caller must have already confirmed ``isinstance(source,
    ComparedModelSource)`` before calling this helper — it does not re-resolve
    the source.

    Parameters
    ----------
    source : ComparedModelSource
        Multi-model wrapper whose ``.items()`` ``(name, source)`` pairs are iterated.
    builder : callable
        The chart's ``_<name>_chart_from_source`` builder; called once per model
        as ``builder(model_source, **builder_kwargs)``.  May return a ``Chart``
        or any ``_ChartLike`` composite (e.g. a nested ``ConcatChart`` from
        ``pdp`` or ``residuals``).
    builder_kwargs : dict
        Forwarded verbatim to *builder* for every model.
    resolve : dict[str, str]
        Per-channel scale-sharing policy passed through to the outer
        ``ConcatChart`` (e.g. ``{"x": "shared", "y": "shared"}`` for
        supervised aggregates; ``{"x": "independent", "y": "independent"}``
        for unsupervised diagnostics).
    columns : int, optional
        Grid columns for the outer ``ConcatChart``.  Defaults to the number of
        models so all panels appear in a single row.

    Returns
    -------
    ConcatChart
        Small-multiples composition with one labeled panel per model.
    """
    from ferrum.composition import ConcatChart

    children = []
    for name, model_source in source.items():
        # Every refusal names the model whose panel refused, the same way
        # whatever raised it; a style refusal keeps its voice, so the
        # figure re-voices it on the way out.
        prefix = f"compare={{{name!r}: ...}}: "
        try:
            child = builder(model_source, **builder_kwargs)
        except EntryRefusal as err:
            raise err.prefixed(prefix) from None
        except ValueError as exc:
            raise ValueError(f"{prefix}{exc}") from exc
        child = child.properties(title=name)
        children.append(child)

    n_cols = columns if columns is not None else len(children)
    return ConcatChart(*children, columns=n_cols, resolve=resolve)


def _order_compare_rows(
    df: pl.DataFrame,
    primary_col: str,
    primary_order: list,
    model_order: list,
) -> pl.DataFrame:
    """Sort a stacked ``compare=`` frame by ``(primary_col rank, model rank)``.

    Shared row-ordering idiom for the dodge-by-model ``compare=`` builders
    (importance, shap_bar, cv_scores — GH #42). Casts ``primary_col`` and
    ``"model"`` to ``pl.Enum`` domains fixed by ``primary_order`` /
    ``model_order``, then sorts by ``(primary rank, model rank)`` so Rust's
    encounter-order ordinal domains place ``primary_col``'s bands in
    ``primary_order`` (e.g. global feature rank, or split order) and each
    band's model sub-groups in ``model_order`` (compare registration order,
    ``"base"`` first). The two temporary rank columns are dropped before
    returning.

    Parameters
    ----------
    df : polars.DataFrame
        The stacked per-model frame to reorder. Any pre-filtering (e.g. to
        a global top-k feature set) must already be applied by the caller.
    primary_col : str
        The band-axis column whose domain order is fixed (e.g. ``"feature"``
        or ``"split"``).
    primary_order : list
        The desired domain order for ``primary_col``.
    model_order : list
        The desired domain order for ``"model"`` (compare registration
        order).

    Returns
    -------
    polars.DataFrame
        ``df`` sorted by ``(primary_col, model)`` rank, with no extra
        columns.
    """
    ranked = df.with_columns(
        pl.col(primary_col).cast(pl.Enum(primary_order)).alias("_primary_rank"),
        pl.col("model").cast(pl.Enum(model_order)).alias("_model_rank"),
    )
    return ranked.sort(["_primary_rank", "_model_rank"]).drop("_primary_rank", "_model_rank")


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
    from ferrum.diagnostics.source import ComparedModelSource

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
        from ferrum.diagnostics._internal.precomputed import _PrecomputedSource

        return _PrecomputedSource(y_true, y_pred)

    if not has_model:
        raise ValueError(
            "Supply either a fitted model/source (model=) or precomputed arrays (y_true=, y_pred=)."
        )

    if isinstance(model, ComparedModelSource):
        return model
    if compare is None:
        if isinstance(model, dict):
            return ferrum.ModelSource.compare(model, X, y, random_state=random_state)
        if isinstance(model, ferrum.ModelSource):
            return model
    return ferrum.ModelSource._from_base_and_compare(
        model, compare, X, y, random_state=random_state
    )


# ---------------------------------------------------------------------------
# _merge_layers — general-purpose layer composer (moved from regression.py)
# ---------------------------------------------------------------------------


def _merge_layers(
    scatter_chart: "Chart",
    fit_chart: "Chart",
    *,
    scatter_name: str | None = None,
    fit_name: str | None = None,
) -> "Chart":
    """Compose a scatter Chart and a fit Chart with ``+`` and name their layers.

    The composition is ``scatter_chart + fit_chart``, so a figure draws its
    layers by the same rule a user's ``+`` does: each chart's bindings stay
    on its own layers, and a fit that should share the scatter's hue binds
    it itself. Layer 0 (the scatter) takes *scatter_name* and the fit's first
    layer *fit_name* when it has none, the names a figure's ``mark=`` and
    ``layer_kwargs=`` doors address.

    Imported by both ``regression.py`` and ``matrix.py``.
    """
    composed = scatter_chart + fit_chart
    layers = list(composed._layers)
    if scatter_name is not None:
        layers[0] = replace(layers[0], name=scatter_name)
    if fit_name and len(layers) > 1 and layers[1].name is None:
        layers[1] = replace(layers[1], name=fit_name)
    composed._layers = layers
    return composed
