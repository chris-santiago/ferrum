"""Figure-level override passthrough utility.

``_apply_overrides`` is the single shared entry point called by every figure-
function builder (``_*_chart_from_source``) to merge user overrides into the
assembled chart just before ``.theme()`` is applied.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ferrum.chart import Chart

# Full catalog of possible sub-layer names for each composite mark type.
# Keyed by the ``kind`` string passed to ``_set_composite_mark``.
# Validation checks this registry so conditionally-absent layers
# (e.g. boxplot outlier when outliers=False) are accepted as no-ops.
LAYER_NAME_CATALOG: dict[str, frozenset[str]] = {}


def register_layer_names(kind: str, names: frozenset[str]) -> None:
    """Register possible sub-layer names for *kind* (additive)."""
    existing = LAYER_NAME_CATALOG.get(kind, frozenset())
    LAYER_NAME_CATALOG[kind] = existing | names


def _apply_overrides(
    chart: Any,
    *,
    mark: dict[str, Any] | None = None,
    encode: dict[str, Any] | None = None,
    properties: dict[str, Any] | None = None,
    layers: list | None = None,
    _skip_unknown_mark_keys: bool = False,
) -> Any:
    """Merge user overrides into *chart* and return the modified copy.

    Application order: mark → encode → properties → layers.
    For compound views (VConcatChart, HConcatChart, etc.), fans out to
    each child chart via ``_rebuild_with_charts``.
    """
    from ferrum.chart import Chart

    if not isinstance(chart, Chart):
        try:

            def _apply(c: Any) -> Any:
                return _apply_overrides(
                    c,
                    mark=mark,
                    encode=encode,
                    properties=properties,
                    layers=layers,
                    _skip_unknown_mark_keys=True,
                )

            return chart._rebuild_with_charts(_apply)
        except (NotImplementedError, AttributeError):
            if properties is not None and hasattr(chart, "properties"):
                chart = chart.properties(**properties)
            return chart

    if mark is not None:
        chart = _apply_mark_overrides(chart, mark, _skip_unknown=_skip_unknown_mark_keys)
    if encode is not None:
        chart = chart.encode(**encode)
    if properties is not None:
        chart = chart.properties(**properties)
    if layers is not None:
        chart = chart.layer(*layers)
    return chart


def _apply_mark_overrides(
    chart: "Chart",
    overrides: dict[str, Any],
    *,
    _skip_unknown: bool = False,
) -> "Chart":
    resolved = chart._resolve_pending()
    names = resolved.layer_names

    # Single-mark chart with no named layers: if _skip_unknown, leave it
    # untouched (this child panel doesn't have composite layers at all).
    # Otherwise treat overrides as flat mark kwargs.
    if not names:
        if _skip_unknown:
            return resolved
        new = resolved._clone()
        existing_kwargs = dict(new._mark_kwargs) if new._mark_kwargs else {}
        existing_kwargs.update(overrides)
        new._mark_kwargs = existing_kwargs
        return new

    # Composite chart: validate keys against full catalog.
    valid = _valid_layer_names(resolved)
    for key in overrides:
        if key not in valid:
            if _skip_unknown:
                continue
            raise ValueError(
                f"Unknown sub-layer {key!r}; valid names for this chart: {sorted(valid)}"
            )

    new = resolved._clone()
    new_layers: list = []
    for ly in new._layers or []:
        if ly.name is None or ly.name not in overrides:
            new_layers.append(ly)
            continue
        val = overrides[ly.name]
        if val is False or val is None:
            continue
        if isinstance(val, dict):
            merged = dict(ly.mark_kwargs or {})
            merged.update(val)
            new_layers.append(replace(ly, mark_kwargs=merged))
        else:
            raise TypeError(
                f"mark override for {ly.name!r} must be a dict of mark "
                f"kwargs or False to suppress; got {type(val).__name__}"
            )
    new._layers = new_layers
    return new


def _valid_layer_names(chart: "Chart") -> set[str]:
    """Return all *possible* sub-layer names for this chart's mark type.

    Checks the ``LAYER_NAME_CATALOG`` registry first (covers conditional
    layers). Falls back to the currently-active layer names.
    """
    kind = getattr(chart, "_composite_kind", None)
    if kind and kind in LAYER_NAME_CATALOG:
        return set(LAYER_NAME_CATALOG[kind])
    return {ly.name for ly in (chart._layers or []) if ly.name}
