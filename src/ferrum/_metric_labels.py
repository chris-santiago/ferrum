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
function, its display name and format, and the broadcast frame column it
reads — via :data:`_METRIC_LABEL_SPECS`. A placement, prefix, or column fix
therefore lands in one place and is observed by both surfaces.  (The value
functions ``_trapezoid_auc`` / ``_ap_step`` were already single-homed here;
the metric→spec association was the piece duplicated between
``_apply_metric_label_explicit`` and the per-class dataclass ``__radd__``
methods.)

Where the metric value comes from (F-L11-01, 2026-09-10)
--------------------------------------------------------
:func:`_curve_metric` is the one rule every metric display surface obeys:
when the frame carries the broadcast metric column for that kind (``auc``
on a ROC frame, ``ap`` on a PR frame, ``brier`` on a calibration frame),
that value is shown; only data without such a column -- the
``+ AUCLabel()`` composable path over arbitrary user data -- is integrated
here in Python. The label overlay and the
``mark_roc(annotate_auc=True)`` annotation therefore cannot disagree
about what a curve's AUC is.

The column is a property of the *kind*, not of the calling surface (close
review, 2026-09-10). When it was a per-call argument, the composable path
passed none and silently integrated instead -- harmless for AUC/AP, whose
integral over the drawn curve *is* the metric, and wrong for Brier, whose
integral over a binned reliability diagram is a different statistic that
moves with ``n_bins=``/``strategy=``.

Which surface shows the metric (#219, D4, 2026-09-13)
-----------------------------------------------------
Exactly one: the per-curve overlay label this module emits. The chart
title's ``— AUC 0.987`` suffix and the legend's ``1 (AUC = 0.987)``
rename both retired, so a multi-curve ROC no longer prints each curve's
AUC two or three times, and ``annotate_*=False`` means no metric text
anywhere. Titles keep their plain names (``"ROC Curve"``).

With the duplicates gone, the surviving surfaces all derive their text
from :func:`_render_metric` -- the single producer of a user-visible
metric string. The display name, the ``.3f`` format and the ``"AUC = "``
prefix are all properties of the kind in :data:`_METRIC_LABEL_SPECS`, so
a surface can no longer spell or round a metric its own way. That claim
covers the discrimination-threshold optimum label too, since 2026-09-17:
it composed ``f"max F1 = {v:.3f} @ t={t:.2f}"`` inline with its own
spelling and rounding -- the sixth-site shape #219 set out to end, inside
#219's own subsystem -- and now renders ``f"max {_render_metric('f1', v)}
@ t={t:.2f}"``, with ``"max "`` and the threshold suffix as that surface's
own framing around the one metric string. Registering ``"f1"`` is also why
a *kind* and a composable ``+ Label()`` *class* are no longer the same
set: ``"f1"`` has no class, because nothing composes it onto a chart.

A kind whose ``value_fn`` is a :class:`_NoCurveDefinition` (Brier) has no
fallback at all: a frame without the column cannot answer the question,
and the surface refuses naming the column instead of quietly showing the
bin-level statistic the old fallback produced, which is a different
number under the same label.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, ClassVar, Literal, NamedTuple, Optional

import polars as pl

from ferrum._validate import validate_choice
from ferrum.encoding._aliases import resolve_channel_alias


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


def _broadcast_metric(df: pl.DataFrame, kind: str) -> Optional[float]:
    """The one value ``kind``'s broadcast column carries for ``df``, or ``None``.

    Diagnostic curve frames repeat a kernel-computed scalar on every row of
    a curve (``auc`` on ROC frames, ``ap`` on PR frames, ``brier`` on
    calibration frames -- see ``diagnostics/_internal/_curve_frames.py``).
    ``None`` means "this frame does not carry the metric for these rows":
    the column is absent, holds no value at all, or holds *more than one*
    distinct value, which says the rows span several curves and no single
    broadcast value describes them.

    A present-but-NaN value is **not** ``None`` (close review, 2026-09-10).
    "The frame does not carry this metric" and "the kernel computed this
    metric and it is undefined" are different facts, and collapsing them
    turned the one honest signal a degenerate curve emits -- a NaN AUC from
    a single-class target -- into a plausible-looking number integrated
    from the broken curve. The NaN propagates to the display surface
    instead, which renders it as ``nan``.
    """
    validate_choice("_broadcast_metric", "kind", kind, _METRIC_LABEL_SPECS)
    metric_field = _METRIC_LABEL_SPECS[kind].column
    if metric_field not in df.columns:
        return None
    values = df[metric_field].drop_nulls().unique().to_list()
    if len(values) != 1:
        return None
    return float(values[0])


#: The complete set of reasons :func:`_broadcast_metric` can return ``None``.
#: Spelled as a type rather than as three loose strings so a fourth cause
#: added to :func:`_broadcast_metric` cannot reach :func:`_curve_metric` as an
#: unhandled tag: the producer's return annotation and the consumer's
#: exhaustive branch both name this alias.
_MetricAbsence = Literal["missing_column", "no_value", "multiple_values"]


def _metric_absence_reason(df: pl.DataFrame, kind: str) -> _MetricAbsence:
    """Which of :func:`_broadcast_metric`'s three ``None``-causes applies.

    Mirrors that function's own column lookup exactly (same field, same
    ``drop_nulls().unique()``), so the two cannot disagree about why a
    frame's broadcast value is unusable -- this only runs after
    :func:`_broadcast_metric` has already returned ``None`` for ``kind``.
    """
    metric_field = _METRIC_LABEL_SPECS[kind].column
    if metric_field not in df.columns:
        return "missing_column"
    if len(df[metric_field].drop_nulls().unique()) == 0:
        return "no_value"
    return "multiple_values"


def _curve_metric(df: pl.DataFrame, *, kind: str, x_col: str, y_col: str, where: str) -> float:
    """The metric of ``kind`` for the single curve held in ``df``.

    One resolution rule, shared by every surface that displays a metric
    (label overlay, mark annotation): read the frame's broadcast metric
    column when it carries this curve's value, and integrate the kind's
    value function over ``(x_col, y_col)`` only when it does not. The
    column is the kernel's own number, so a diagnostic chart shows one
    metric from one source; the integration path is what makes the
    composable ``+ AUCLabel()`` marks work on arbitrary user data, which
    has no such column.

    Callers name the metric **kind**, never a column-and-function pair
    (close review, 2026-09-10). Both halves come from
    :data:`_METRIC_LABEL_SPECS` together, so a surface cannot pair the AUC
    integrator with the wrong column, and cannot omit the column and
    silently get the integrated curve instead of the kernel's value -- the
    exact F-L11-04 failure mode.

    A kind whose value function is a :class:`_NoCurveDefinition` has no
    integration path: without a usable broadcast value the frame simply
    cannot answer the question, so this refuses naming the surface and
    *which* of :func:`_broadcast_metric`'s three facts applies -- the column
    is missing, present but all-null, or present with more than one distinct
    value (close review, cycle 2: the message must not claim a column is
    absent when it is sitting right there in "Columns present") (#219(3),
    2026-09-13). ``where`` is that surface's user-facing name
    (``"BrierLabel"``, ``"mark_roc"``) -- required, so a new caller has to
    decide what a reader would recognise rather than inherit a generic one.
    """
    validate_choice("_curve_metric", "kind", kind, _METRIC_LABEL_SPECS)
    value = _broadcast_metric(df, kind)
    if value is not None:
        return value

    spec = _METRIC_LABEL_SPECS[kind]
    if isinstance(spec.value_fn, _NoCurveDefinition):
        reason = _metric_absence_reason(df, kind)
        # One branch per :data:`_MetricAbsence` member, none of them a
        # catch-all: a fourth absence cause must add its own sentence here
        # rather than inheriting "more than one distinct value", which would
        # be a measurably false diagnosis in a message whose whole job is to
        # say which of the three facts applies.
        if reason == "missing_column":
            cause = f"the data has no {spec.column!r} column"
        elif reason == "no_value":
            cause = f"the {spec.column!r} column holds no non-null value for these rows"
        elif reason == "multiple_values":
            cause = f"the {spec.column!r} column holds more than one distinct value for these rows"
        else:
            raise ValueError(
                f"_curve_metric: unreachable: unhandled metric-absence reason "
                f"{reason!r} for kind {kind!r}. Add a branch here when "
                f"_broadcast_metric gains a None-cause."
            )
        raise ValueError(
            f"{where}: {cause}, so no "
            f"{spec.display_name} value can be shown. {spec.value_fn.reason} "
            f"Columns present: {sorted(df.columns)}."
        )

    import numpy as np

    return float(
        spec.value_fn(
            np.asarray(df[x_col].to_list(), dtype=float),
            np.asarray(df[y_col].to_list(), dtype=float),
        )
    )


def _curve_metrics_by_group(
    df: pl.DataFrame,
    *,
    group_field: str,
    kind: str,
    x_col: str,
    y_col: str,
    where: str,
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
            kind=kind,
            x_col=x_col,
            y_col=y_col,
            where=where,
        )
        for value in groups
    }


@dataclass(frozen=True)
class _NoCurveDefinition:
    """Stand-in for the value function of a kind the drawn curve cannot answer.

    Brier is the case: it is a per-sample statistic, and integrating the
    binned reliability points a calibration chart draws yields a different,
    ``n_bins=``/``strategy=``-dependent number. Holding the explanation
    *in place of* the callable (rather than beside it as a second nullable
    field) means the "no fallback" fact and the sentence explaining it
    cannot drift apart, and a kind added later must state its own reason to
    opt out.
    """

    reason: str


class _MetricKind(NamedTuple):
    """Everything one metric kind needs to put a number on a chart.

    ``column`` is the broadcast column a ferrum diagnostic frame carries for
    this metric (see ``diagnostics/_internal/_curve_frames.py``'s schemas).
    It is part of the kind, not of the call site: whether a surface reads
    the kernel's number or integrates the drawn curve must not depend on
    which of the two surfaces is asking.

    ``value_fn`` integrates the drawn curve when the frame carries no such
    column, or is a :class:`_NoCurveDefinition` when the drawn curve does
    not define the metric at all -- see :func:`_curve_metric`.

    ``display_name`` is how the metric is spelled to a reader -- ``"AUC"``,
    ``"AP"``, ``"Brier"`` -- and ``format`` is how its value is rounded.
    Every display surface derives its text from the two together via
    :func:`_render_metric`; the overlay's ``prefix`` is a property rather
    than a stored field for the same reason (close review, 2026-09-11):
    the legend renames, the mark annotations and the label overlays each
    used to carry their own literal spelling and their own ``.3f`` beside
    the kind, so one metric had three names and three roundings in three
    places and a mismatch rendered wrong text. There is now one of each.
    """

    value_fn: Callable[[Any, Any], float] | _NoCurveDefinition
    display_name: str
    column: str
    format: str

    @property
    def prefix(self) -> str:
        """Overlay-text prefix, e.g. ``"AUC = "``. Derived, never stored."""
        return f"{self.display_name} = "


# Canonical metric-kind table — the single source for the
# ``kind -> (value function, display name, broadcast column, format)``
# association shared by every surface that shows a metric: the direct-mark
# ``__radd__`` path, the figure-function explicit-field path and the mark
# annotations. Keys are the display vocabulary — a superset of the
# ``label_kind`` strings ``_apply_metric_label_explicit`` accepts, since a
# kind can be displayable without being composable (``"f1"`` has no label
# class; see its entry). The derived prefixes and the format match the
# per-class dataclass defaults, which read them from here.
_METRIC_LABEL_SPECS: dict[str, _MetricKind] = {
    "auc": _MetricKind(_trapezoid_auc, "AUC", "auc", ".3f"),
    "ap": _MetricKind(_ap_step, "AP", "ap", ".3f"),
    "brier": _MetricKind(
        _NoCurveDefinition(
            "Brier is computed from per-sample predicted probabilities, not from the "
            "binned reliability points a calibration chart draws -- integrating those "
            "gives a bin-level statistic that moves with n_bins=/strategy=. Supply a frame from "
            "ModelSource.calibration_curve(), which carries the column (or call "
            "ferrum.calibration_chart(), which already places this label)."
        ),
        "Brier",
        "brier",
        ".3f",
    ),
    # The discrimination-threshold optimum label's metric (#219 residue,
    # cycle-2 review finding B). It has no composable `+ F1Label()` class --
    # the only surface that shows it is `mark_discrimination_threshold`'s
    # own optimum annotation, which reads the F1-best row of the sweep it
    # was handed -- so it is a kind without a class; see
    # `_apply_metric_label_explicit`, whose vocabulary is the classes.
    # Registering it is what makes `_render_metric`'s "single producer"
    # claim true: that annotation used to carry its own "max F1 = " literal
    # and its own `.3f`, the sixth-site shape #219 set out to end.
    "f1": _MetricKind(
        _NoCurveDefinition(
            "F1 is read from the threshold sweep's own 'f1' column at the optimum row. "
            "The curves a discrimination-threshold chart draws are precision/recall/F1/"
            "queue-rate against threshold, and the area under one of them is not an F1 "
            "score. Supply a frame from ModelSource.discrimination_threshold(), which "
            "carries the column."
        ),
        "F1",
        "f1",
        ".3f",
    ),
}


def _default_prefix(kind: str) -> str:
    """Canonical overlay-text prefix for a metric kind (drives dataclass defaults)."""
    return _METRIC_LABEL_SPECS[kind].prefix


def _default_format(kind: str) -> str:
    """Canonical float format spec for a metric kind (drives dataclass defaults)."""
    return _METRIC_LABEL_SPECS[kind].format


def _render_metric(
    kind: str,
    value: float,
    *,
    prefix: Optional[str] = None,
    fmt: Optional[str] = None,
) -> str:
    """The one producer of a user-visible metric string (#219, spec §6).

    Returns ``"{prefix}{value:{format}}"`` -- e.g. ``"AUC = 0.987"`` --
    with both halves taken from ``kind``'s row of
    :data:`_METRIC_LABEL_SPECS`. Every surface that shows a metric calls
    this; none formats a float itself. Before, six sites carried their own
    ``:.3f`` literal, so changing the rounding (or a metric's spelling)
    meant finding all six and a missed one rendered a differently-rounded
    number beside the others.

    ``prefix`` and ``fmt`` override the kind's defaults for the one shipped
    reason they exist: ``AUCLabel(format=".2f", prefix="auc: ")`` and its
    siblings are public, documented parameters. Passing neither (what every
    internal caller does) is what the byte-parity test pins.
    """
    validate_choice("_render_metric", "kind", kind, _METRIC_LABEL_SPECS)
    spec = _METRIC_LABEL_SPECS[kind]
    return (
        f"{spec.prefix if prefix is None else prefix}{value:{spec.format if fmt is None else fmt}}"
    )


_VALID_METRIC_LABEL_POSITIONS = ("end", "corner")

#: Columns ``_apply_metric_label`` writes onto the base chart's data to carry
#: its overlay, and the encoding the emitted ``mark_text`` layer reads. Named
#: so the emitting code and the idempotence check below cannot disagree about
#: which layer belongs to this module.
_LABEL_TEXT_COL = "_label_text"
_LABEL_X_COL = "_label_x"
_LABEL_Y_COL = "_label_y"


def _is_metric_label_layer(layer: Any) -> bool:
    """Whether ``layer`` is an overlay this module emitted.

    Recognised by structure -- a ``text`` mark reading
    :data:`_LABEL_TEXT_COL` -- rather than by its ``<kind>_label`` name,
    which a caller's own layer could share. The column is private to this
    module, so nothing else can produce that shape.
    """
    return layer.mark == "text" and _resolve_field(layer.encoding.get("text")) == _LABEL_TEXT_COL


def _curve_hue_field(base: "Chart") -> Optional[str]:
    """The field *base*'s curves are colored by, one label per value of it, or ``None``.

    The base chart's own ``color``; on a ``+`` composition, whose operands
    keep their bindings on their own layers, the ``color`` of the first
    layer that draws one (the binding the panel's color legend is titled
    by). A composite mark's layer-level colors are its own and group no
    label.
    """
    field = _resolve_field(base._encoding.get("color"))
    if field is not None or base._composite_kind is not None:
        return field
    for layer in base._layers or []:
        field = _resolve_field(resolve_channel_alias(layer.encoding, "color"))
        if field is not None:
            return field
    return None


def _without_metric_label_layer(base: "Chart") -> "Chart":
    """``base`` with any previously-applied metric-label overlay removed.

    Returns ``base`` itself when there is nothing to drop, so the common
    first-application path clones nothing.
    """
    if base._layers is None:
        return base
    kept = [layer for layer in base._layers if not _is_metric_label_layer(layer)]
    if len(kept) == len(base._layers):
        return base
    stripped = base._clone()
    stripped._layers = kept
    return stripped


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


def _apply_metric_label(
    base: "Chart",
    label: "AUCLabel | APLabel | BrierLabel",
    *,
    kind: str,
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

    ``kind`` (required) names the metric -- ``"auc"``, ``"ap"``, ``"brier"``
    -- and :func:`_curve_metric` resolves it: the frame's broadcast column
    when the data carries one, the kind's integrator over the drawn curve
    when it does not (arbitrary user data). So the label, the chart title
    and the legend text are one number from one source.

    ``kind`` carries no default (close review, 2026-09-10): every caller
    supplies it, and a default would let a fifth caller silently integrate
    the drawn curve. That matters most for Brier, whose curve integration is
    a *bin-level* statistic that swings with ``n_bins=``/``strategy=`` and
    is not the Brier score -- composing ``+ BrierLabel()`` onto a
    calibration chart used to print that number over the correct one the
    figure had already placed.

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

    #219(2) (2026-09-13): applying a label to a chart that already carries
    one **replaces** it. ``fm.roc_chart(...) + fm.AUCLabel()`` used to stack
    a second text layer holding the same text at the same coordinates -- an
    exact overstrike, visible only as slightly bolder glyphs -- because the
    figure function already places one by default. The existing layer is
    recognised by the encoding this function owns (see
    :func:`_is_metric_label_layer`), and dropping it is also what lets a
    re-applied label change ``position=``/``format=``: the replacement
    re-derives every ``_label_*`` value from scratch.
    """
    from ferrum._coerce import to_arrow_table
    from ferrum.chart import Chart

    base = _without_metric_label_layer(base)
    x_col = x_col_override or _resolve_field(base._encoding.get("x"))
    y_col = y_col_override or _resolve_field(base._encoding.get("y"))
    color_col = color_col_override or _curve_hue_field(base)
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
        # Null group values are skipped (#220), mirroring
        # _curve_metrics_by_group's guard above: a null color value would
        # otherwise form its own loop iteration, and `pl.col(color_col) ==
        # None` always evaluates to null in polars, matching no row and
        # warning once per such iteration for nothing.
        unique_colors = sorted(
            (v for v in df[color_col].unique().to_list() if v is not None), key=str
        )
        for stack_i, cls in enumerate(unique_colors):
            group = df.filter(pl.col(color_col) == cls)
            if group.is_empty():
                continue
            # `_idx` is this function's own row index, not part of the
            # caller's data -- drop it so a refusal lists the columns the
            # caller actually passed.
            metric = _curve_metric(
                group.drop("_idx"), kind=kind, x_col=x_col, y_col=y_col, where=type(label).__name__
            )
            text = _render_metric(kind, metric, prefix=label.prefix, fmt=label.format)
            # nulls_last=True so a null x (which polars' default descending
            # sort places FIRST) never becomes the selected "endpoint" row;
            # the endpoint is the group's highest *non-null* x, matching what
            # the pre-fix code rendered for this data (the label text/metric
            # already reflects any null via the kind's value-function NaN
            # propagation
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
        metric = _curve_metric(
            df.drop("_idx"), kind=kind, x_col=x_col, y_col=y_col, where=type(label).__name__
        )
        text = _render_metric(kind, metric, prefix=label.prefix, fmt=label.format)
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
        pl.Series(_LABEL_TEXT_COL, labels_col, dtype=pl.Utf8),
        pl.Series(_LABEL_X_COL, label_x_col, dtype=pl.Float64),
        pl.Series(_LABEL_Y_COL, label_y_col, dtype=pl.Float64),
    )
    base_aug = base._clone()
    base_aug._data = augmented
    annot_layer = (
        Chart(augmented)
        .mark_text(align="right", baseline="hanging", dx=-4, dy=4)
        .encode(x=_LABEL_X_COL, y=_LABEL_Y_COL, text=_LABEL_TEXT_COL)
    )
    labelled = base_aug + annot_layer
    # Named for the sub-layer a figure's `mark=` override aims at
    # (`roc_chart(mark={"auc_label": {...}})`).
    *kept, overlay = labelled._layers
    labelled._layers = [*kept, replace(overlay, name=f"{kind}_label")]
    return labelled


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
    format: str = _default_format("auc")
    prefix: str = _default_prefix("auc")

    def __post_init__(self) -> None:
        _validate_metric_label_fields(self)

    def __radd__(self, base: "Chart") -> "Chart":
        from ferrum.chart import Chart

        if not isinstance(base, Chart):
            return NotImplemented
        return _apply_metric_label(base, self, kind=self._kind)


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
    format: str = _default_format("ap")
    prefix: str = _default_prefix("ap")

    def __post_init__(self) -> None:
        _validate_metric_label_fields(self)

    def __radd__(self, base: "Chart") -> "Chart":
        from ferrum.chart import Chart

        if not isinstance(base, Chart):
            return NotImplemented
        return _apply_metric_label(base, self, kind=self._kind)


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
    format: str = _default_format("brier")
    prefix: str = _default_prefix("brier")

    def __post_init__(self) -> None:
        _validate_metric_label_fields(self)

    def __radd__(self, base: "Chart") -> "Chart":
        from ferrum.chart import Chart

        if not isinstance(base, Chart):
            return NotImplemented
        return _apply_metric_label(base, self, kind=self._kind)


# Registry of metric-label classes keyed by each class's own ``_kind``, so the
# figure-function explicit-field path (:func:`_apply_metric_label_explicit`)
# constructs the right class per kind without an if/elif ladder that could
# desync from the classes. Derived from the classes themselves; the keys are a
# strict subset of ``_METRIC_LABEL_SPECS``'s (``specs - classes == {"f1"}`` — a
# display kind with no composable class; the one-directional invariant is
# enforced by the single-source test).
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
    position: str = "end",
    fmt: Optional[str] = None,
    prefix: Optional[str] = None,
) -> "Chart":
    """Apply a metric label to ``base`` with explicit field overrides.

    Schwabish SB3 helper used by figure-level diagnostic builders that
    compose ``AUCLabel`` / ``APLabel`` / ``BrierLabel`` onto charts whose
    ``_pending_stat_mark`` composite hides the encoding.

    Shares the metric-kind table (:data:`_METRIC_LABEL_SPECS`) with the
    direct-mark ``__radd__`` path, so a value-function, prefix, format or
    broadcast-column change is observed identically by both surfaces. The
    kind's broadcast column is read from the table rather than passed by
    the caller (close review, 2026-09-10): a builder cannot ask for the
    wrong column, or forget to ask for one and silently get an integrated
    curve instead of the kernel's number. ``fmt``/``prefix`` default to the
    table's for the same reason (#219, 2026-09-13) -- a builder that wrote
    its own ``".3f"`` was a sixth place the rounding had to be changed.
    """
    # Validated against the *class* registry, not the kind table: this
    # function's whole job is to construct one of the composable label
    # classes, and a kind can exist without one (``"f1"``, whose only
    # surface is the discrimination-threshold optimum annotation). Checking
    # the kinds here would have let such a kind through to a ``KeyError``
    # one line later.
    validate_choice("_apply_metric_label_explicit", "label_kind", label_kind, _METRIC_LABEL_CLASSES)
    spec = _METRIC_LABEL_SPECS[label_kind]
    pos_lit: Any = position
    label_obj: Any = _METRIC_LABEL_CLASSES[label_kind](
        position=pos_lit,
        format=spec.format if fmt is None else fmt,
        prefix=spec.prefix if prefix is None else prefix,
    )
    return _apply_metric_label(
        base,
        label_obj,
        kind=label_kind,
        x_col_override=x_col,
        y_col_override=y_col,
        color_col_override=color_col,
    )
