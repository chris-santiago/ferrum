"""Render-side consumer for ``Chart.override`` payloads.

:mod:`ferrum._override_apply` is a pure transform that resolves, validates, and
folds a flat override dict into an :class:`~ferrum._override_apply.OverridePayload`.
This module is the matching *consumer*: it deep-merges each payload piece into the
assembled spec / chart-config / viewport at its injection point and emits the
deprecation warnings.  It runs **last** among presentation sources so override wins
the cascade (spec §7: override > per-channel ``axis=``/``legend=`` > ``configure_*``
> theme > ``set_default_theme`` > Rust defaults).

The functions here are pure (no ``Chart`` import, no I/O, no global state) apart from
:func:`emit_deprecations`, whose sole effect is the documented ``DeprecationWarning``.
The Chart render path (:meth:`Chart._render_inputs`, :meth:`Chart.to_spec`) builds the
payload once per render via ``build_payload`` and threads it through these helpers.
"""

from __future__ import annotations

import warnings
from typing import Any

from ferrum._override_apply import OverridePayload
from ferrum.exceptions import FerrumOverrideError


def merge_chart_config(chart_config: dict[str, Any], payload: OverridePayload) -> dict[str, Any]:
    """Return *chart_config* deep-merged with ``payload.chart_config`` (override wins).

    ``payload.chart_config`` is ``{target_key: {leaf: value}}`` (e.g.
    ``{"axis_x": {"label_angle": -45}}``).  Each target-key dict is merged into the
    matching key of *chart_config*, with override leaves replacing configure-layer
    leaves on conflict.  *chart_config* is not mutated; a new dict is returned.

    Section keys stay where the caller put them.  A ``_redistribute_general_axis``
    helper used to rewrite them here — moving a leaf off the general ``axis`` key
    and re-pinning it onto the *opposite* axis — on the premise that Rust applied
    ``axis`` before ``axis_x``/``axis_y`` under first-wins semantics, so a per-axis
    override would otherwise lose.  That premise was false, and the rewrite was
    the thing causing the loss: for the axis fields whose only carrier was the
    shared theme, the synthesized opposite-axis entry ran *last* into that one
    global slot and so the general value won.  Rust now applies the per-axis
    sections first (fill-only) and gives every such field its own per-axis slot,
    so precedence is settled there and nothing is left for this layer to fix.
    """
    if not payload.chart_config:
        return chart_config
    merged = {k: dict(v) if isinstance(v, dict) else v for k, v in chart_config.items()}
    for target_key, leaves in payload.chart_config.items():
        existing = merged.get(target_key)
        if isinstance(existing, dict):
            merged[target_key] = {**existing, **leaves}
        else:
            merged[target_key] = dict(leaves)
    return merged


def apply_mark_style(
    mark_style: dict[str, Any], payload: OverridePayload, *, is_multi_mark: bool
) -> dict[str, Any]:
    """Return *mark_style* merged with ``payload.mark_style`` (override wins).

    Mark-style overrides apply to the chart's single base/primary mark.  A chart
    with multiple marks (a layered chart) has no single primary mark, so a
    ``mark_*`` override is ambiguous and raises :class:`FerrumOverrideError`
    (spec §11 Q4).  *mark_style* is not mutated; a new dict is returned.

    Parameters
    ----------
    mark_style:
        The base mark's style dict assembled in ``Chart.to_spec``.
    payload:
        The resolved override payload.
    is_multi_mark:
        ``True`` when the chart has more than one mark (layered).  When this holds
        and the payload carries any mark-style override, raise.
    """
    if not payload.mark_style:
        return mark_style
    if is_multi_mark:
        leaves = ", ".join(sorted(payload.mark_style))
        raise FerrumOverrideError(
            f"mark_* override ({leaves}) is ambiguous on a multi-mark chart: "
            "a layered chart has no single primary mark to apply it to. "
            "Set the style on the specific layer's mark_*() call instead."
        )
    return {**mark_style, **payload.mark_style}


def apply_coord(coord: Any, payload: OverridePayload) -> Any:
    """Return the coord spec value with ``payload.coord`` leaves applied (override wins).

    ``payload.coord`` carries **dataclass-field** leaves (``clip``, ``expand``,
    ``xlim``, ``ratio``, …), not the serialized spec-dict keys.  They are applied by
    ``dataclasses.replace`` onto the chart's existing coord dataclass, defaulting to
    :class:`~ferrum.coord.CoordCartesian` when the chart has no coord set, then
    re-serialized via ``to_spec_dict()``.  This keeps override coord leaves in the
    same field namespace Unit A validated them against.

    Parameters
    ----------
    coord:
        The chart's existing coord object (a coord dataclass, ``CoordFlip``, or
        ``None``).  ``CoordFlip`` carries no fields, so a coord-leaf override on a
        flipped chart raises (no field to set).
    payload:
        The resolved override payload.

    Returns
    -------
    The serialized coord spec value (``dict`` or ``str``) to assign to
    ``kw["coord"]``, or the chart's original serialized coord when the payload
    carries no coord override.

    Raises
    ------
    FerrumOverrideError
        When an override leaf is not a field of the target coord dataclass.
    """
    import dataclasses

    from ferrum.coord import CoordCartesian

    if not payload.coord:
        return _serialize_coord(coord)
    base = coord if dataclasses.is_dataclass(coord) and not isinstance(coord, type) else None
    if base is None:
        base = CoordCartesian()
    valid_fields = {f.name for f in dataclasses.fields(base)}
    bad = [leaf for leaf in payload.coord if leaf not in valid_fields]
    if bad:
        leaves = ", ".join(sorted(bad))
        raise FerrumOverrideError(
            f"coord_* override ({leaves}) is not a field of "
            f"{type(base).__name__}. Set a matching coord via .coord(...) first, "
            "or override a field that coord exposes."
        )
    updated = dataclasses.replace(base, **payload.coord)
    return _serialize_coord(updated)


def _serialize_coord(coord: Any) -> Any:
    """Serialize a coord object to its spec value, mirroring ``Chart.to_spec``."""
    if coord is None:
        return None
    return coord.to_spec_dict() if hasattr(coord, "to_spec_dict") else coord


def apply_properties(
    viewport: tuple[float, float], payload: OverridePayload
) -> tuple[float, float]:
    """Return *viewport* with ``payload.properties`` ``width``/``height`` applied.

    Property overrides beat ``.properties()`` (spec §8 D5).  Missing keys leave the
    corresponding viewport dimension unchanged.
    """
    if not payload.properties:
        return viewport
    width, height = viewport
    if "width" in payload.properties:
        width = float(payload.properties["width"])
    if "height" in payload.properties:
        height = float(payload.properties["height"])
    return width, height


def emit_deprecations(payload: OverridePayload) -> None:
    """Emit one ``DeprecationWarning`` per ``payload.deprecations`` entry.

    Each entry is ``(path, typed_equivalent_descriptor)``; the warning names the
    override path and the typed ``configure_*`` / ``properties`` method that
    supersedes it.  Fired once per render, before the payload is applied.
    """
    for path, typed_equivalent in payload.deprecations:
        warnings.warn(
            f"override({path}=...) is deprecated; use {typed_equivalent} instead.",
            DeprecationWarning,
            stacklevel=2,
        )
