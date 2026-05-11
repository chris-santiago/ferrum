"""§3.14 Group B figure-level functions.

Each function is a thin facade over a ``_*_chart_from_source`` builder in
``ferrum._diagnostics.charts``. The ``_resolve_source`` helper accepts a
fitted model, an explicit ``ModelSource``, or (10h) a dict of named models
for comparison.
"""
from __future__ import annotations

from typing import Any


def _resolve_source(
    model_or_source: Any,
    X: Any = None,
    y: Any = None,
    *,
    random_state: int | None = None,
    compare: dict[str, Any] | None = None,
) -> Any:
    """Resolve a figure-function input into a ModelSource (or comparable)."""
    import ferrum
    if compare is not None:
        raise NotImplementedError(
            "compare= support lands in Phase 10h; for now pass a single model "
            "or a pre-built ModelSource."
        )
    if isinstance(model_or_source, ferrum.ModelSource):
        return model_or_source
    if isinstance(model_or_source, dict):
        raise NotImplementedError(
            "Multi-model dict input lands in Phase 10h."
        )
    return ferrum.ModelSource(model_or_source, X, y, random_state=random_state)


def residuals_chart(
    model_or_source: Any,
    X: Any = None,
    y: Any = None,
    *,
    kind: str = "studentized",
    panels: Any = "auto",
    random_state: int | None = None,
    theme: Any = None,
):
    """Residuals diagnostic chart — see ferrum-spec.md §3.14.

    ``panels="auto"`` ships only the residuals-vs-fitted panel in 10a;
    the QQ / scale-location / leverage panels join in 10h. Pass an explicit
    list (e.g. ``panels=["residuals_vs_fitted", "qq"]``) to force the
    multi-panel path.
    """
    from ferrum._diagnostics.charts import _residuals_chart_from_source
    source = _resolve_source(model_or_source, X, y, random_state=random_state)
    if panels in (None, "single"):
        panel_list: Any = None
    elif panels == "auto":
        panel_list = None  # 10a auto == single; 10h expands this to 2x2
    else:
        panel_list = list(panels)
    return _residuals_chart_from_source(
        source, kind=kind, panels=panel_list, theme=theme,
    )


# --- 10b: classification curves --------------------------------------


def roc_chart(
    model_or_source: Any,
    X: Any = None,
    y: Any = None,
    *,
    per_class: bool = True,
    average: str | None = "macro",
    annotate_auc: bool = False,
    compare: dict[str, Any] | None = None,
    random_state: int | None = None,
    theme: Any = None,
):
    """ROC curve(s) — see ferrum-spec.md §3.14.

    ``per_class=True`` (default) plots one curve per class; pass ``False``
    along with ``average="macro"`` (or "micro"/"weighted") to plot a single
    averaged curve. ``annotate_auc=True`` is reserved for Phase 10h.
    """
    from ferrum._diagnostics.charts import _roc_chart_from_source
    source = _resolve_source(
        model_or_source, X, y, random_state=random_state, compare=compare,
    )
    return _roc_chart_from_source(
        source,
        per_class=per_class,
        average=average,
        annotate_auc=annotate_auc,
        theme=theme,
    )


def pr_chart(
    model_or_source: Any,
    X: Any = None,
    y: Any = None,
    *,
    per_class: bool = True,
    annotate_ap: bool = False,
    iso_lines: bool = False,
    compare: dict[str, Any] | None = None,
    random_state: int | None = None,
    theme: Any = None,
):
    """Precision-recall curve(s) — see ferrum-spec.md §3.14.

    ``annotate_ap=True`` and ``iso_lines=True`` are reserved for Phase 10h.
    """
    from ferrum._diagnostics.charts import _pr_chart_from_source
    source = _resolve_source(
        model_or_source, X, y, random_state=random_state, compare=compare,
    )
    return _pr_chart_from_source(
        source,
        per_class=per_class,
        annotate_ap=annotate_ap,
        iso_lines=iso_lines,
        theme=theme,
    )


def calibration_chart(
    *model_or_sources: Any,
    X: Any = None,
    y: Any = None,
    n_bins: int = 10,
    strategy: str = "uniform",
    random_state: int | None = None,
    theme: Any = None,
):
    """Calibration (reliability) curve — see ferrum-spec.md §3.14.

    Variadic in ``*model_or_sources`` for the eventual multi-model overlay
    (Phase 10h). Phase 10b accepts a single model or source only.
    """
    if len(model_or_sources) == 0:
        raise TypeError(
            "calibration_chart requires at least one model or ModelSource"
        )
    if len(model_or_sources) > 1:
        raise NotImplementedError(
            "Multi-model calibration_chart ships in Phase 10h."
        )
    from ferrum._diagnostics.charts import _calibration_chart_from_source
    source = _resolve_source(
        model_or_sources[0], X, y, random_state=random_state,
    )
    return _calibration_chart_from_source(
        source, n_bins=n_bins, strategy=strategy, theme=theme,
    )


def gain_chart(
    model_or_source: Any,
    X: Any = None,
    y: Any = None,
    *,
    random_state: int | None = None,
    theme: Any = None,
):
    """Cumulative-gain curve — see ferrum-spec.md §3.14."""
    from ferrum._diagnostics.charts import _gain_chart_from_source
    source = _resolve_source(model_or_source, X, y, random_state=random_state)
    return _gain_chart_from_source(source, theme=theme)


def lift_chart(
    model_or_source: Any,
    X: Any = None,
    y: Any = None,
    *,
    random_state: int | None = None,
    theme: Any = None,
):
    """Lift curve — see ferrum-spec.md §3.14."""
    from ferrum._diagnostics.charts import _lift_chart_from_source
    source = _resolve_source(model_or_source, X, y, random_state=random_state)
    return _lift_chart_from_source(source, theme=theme)


def discrimination_threshold_chart(
    model_or_source: Any,
    X: Any = None,
    y: Any = None,
    *,
    n_thresholds: int = 50,
    metrics: tuple[str, ...] = ("precision", "recall", "f1", "queue_rate"),
    cv: Any = None,
    threshold_line: bool = False,
    random_state: int | None = None,
    theme: Any = None,
):
    """Discrimination-threshold sweep — see ferrum-spec.md §3.14.

    ``threshold_line=True`` (rule at the F1-best threshold) is reserved
    for Phase 10h.
    """
    from ferrum._diagnostics.charts import (
        _discrimination_threshold_chart_from_source,
    )
    source = _resolve_source(model_or_source, X, y, random_state=random_state)
    return _discrimination_threshold_chart_from_source(
        source,
        n_thresholds=n_thresholds,
        metrics=metrics,
        cv=cv,
        threshold_line=threshold_line,
        theme=theme,
    )
