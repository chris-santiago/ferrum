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
