"""Render-side consumer for ``Chart.override`` payloads.

:mod:`ferrum._override_apply` is a pure transform that resolves, validates, and
folds a flat override dict into an :class:`~ferrum._override_apply.OverridePayload`.
This module is the matching *consumer*: it deep-merges each payload piece into the
assembled spec / chart-config / viewport at its injection point and emits the
deprecation warnings.  It runs **last** among presentation sources so override wins
the cascade (spec §7: override > per-channel ``axis=``/``legend=`` > ``configure_*``
> theme > ``set_default_theme`` > Rust defaults).

There is one function per payload piece: :func:`merge_encoding_scale` (encoding),
:func:`merge_chart_config`, :func:`apply_mark_style`, :func:`apply_coord`,
:func:`apply_properties`, plus :func:`emit_deprecations`.  The encoding piece's
merge used to live in ``_spec_build.py`` beside its one call site, which put the
override contract's two halves — the leaf registry and the merge gate — in
modules that could not see each other; a design review traced a live
override-leaf/wire-key divergence to exactly that separation.

The functions here are pure (no ``Chart`` import, no I/O, no global state) apart from
:func:`emit_deprecations`, whose sole effect is the documented ``DeprecationWarning``.
The Chart render path (:meth:`Chart._render_inputs`, :meth:`Chart.to_spec`) builds the
payload once per render via ``build_payload`` and threads it through these helpers.
"""

from __future__ import annotations

import warnings
from typing import Any

from ferrum._core import (  # type: ignore[attr-defined]
    scale_accepted_keys as _scale_accepted_keys,
    validate_scale_dict as _validate_scale_dict,
)
from ferrum._override_apply import OverridePayload
from ferrum.exceptions import FerrumOverrideError


def merge_encoding_scale(base_scale: dict, scale_overrides: dict) -> dict:
    """Return one channel's own scale dict merged with its override scale leaves.

    The encoding piece's injection point (called from
    ``SpecBuildMixin._build_encoding_specs``), and so the sibling of
    :func:`merge_chart_config` / :func:`apply_mark_style` / :func:`apply_coord`
    / :func:`apply_properties` for ``payload.encoding``.

    Contract:

    - Override leaves win over the channel's own ``scale=`` keys (spec §7).
    - When the effective ``type`` is unchanged — or the channel carries no
      ``scale=`` at all, which reaches here as ``{}`` — nothing is filtered and
      every base-scale key still applies.
    - When the type IS changing, each base-scale key is validated exactly once
      and only the keys the new type also accepts survive.  A key the OLD type
      never accepted, and a key the switch is about to drop, are both probed
      under the old type through the real wire gate
      (``ferrum._core.validate_scale_dict``), so they refuse with the identical
      message that base scale raises standalone.  The switch may neither
      launder a bad key into silence nor promote one into effect because the
      new type happens to declare a same-named field.
    - A key BOTH types accept is deliberately NOT value-validated here: its
      value must reach the new type's own downstream gate untouched (a string
      ``domain`` is illegal under ``linear`` and legal under ``band``).
    - A non-``str`` or unknown override ``type`` skips filtering entirely, so
      the gate's own "unknown variant" / type error surfaces instead of a PyO3
      argument-coercion error naming ``scale_accepted_keys``'s parameter.

    ``base_scale`` is never mutated.  The eight remediation rounds behind these
    rules are recorded in the commit history and ``.sdd/``, not here;
    ``tests/test_override_scale_merge.py`` pins each one.
    """
    old_type = base_scale.get("type")
    new_type = scale_overrides.get("type", old_type)
    # "type" not in base_scale, NOT "old_type is None": a channel with no
    # scale= at all reaches here as `{}` and must short-circuit, but an
    # EXPLICIT `{"type": None}` is a real claim the user made and must be
    # probed like any other invalid tag.
    if "type" not in base_scale or new_type == old_type:
        return {**base_scale, **scale_overrides}

    # Validate the base scale's TAG alone first, through the real gate: an
    # unknown tag, or a value serde cannot even reach, raises exactly what a
    # standalone `scale={"type": old_type}` raises.
    _validate_scale_dict({"type": old_type})

    # old_type is now known to be a real ScaleSpec variant tag.
    accepted_old = set(_scale_accepted_keys(old_type))

    # Bucket 1: keys old_type does not recognize. Probed with ONLY those keys,
    # so a key old_type DOES recognize is never value-validated under a type
    # the switch is replacing.
    unknown_under_old = {k for k in base_scale if k != "type" and k not in accepted_old}
    if unknown_under_old:
        _validate_scale_dict(
            {"type": old_type, **{k: base_scale[k] for k in unknown_under_old}},
        )

    accepted_new: set[str] | None
    if isinstance(new_type, str):
        try:
            accepted_new = set(_scale_accepted_keys(new_type))
        except ValueError:
            # Unknown new type: fall through unfiltered so the gate's own
            # "unknown variant" error surfaces.
            accepted_new = None
    else:
        # Non-string type: don't hand it to the &str-typed
        # scale_accepted_keys, whose argument error is not this gate's voice.
        accepted_new = None
    if accepted_new is not None:
        drop = accepted_old - accepted_new
        # Bucket 2: keys the filter below is about to drop. They never reach
        # new_type's downstream gate, so nobody validates their VALUE if this
        # probe doesn't. Bucket 3 (accepted by both) is the survivors, checked
        # downstream under new_type.
        dropped_present = {k for k in base_scale if k != "type" and k in drop}
        if dropped_present:
            _validate_scale_dict(
                {"type": old_type, **{k: base_scale[k] for k in dropped_present}},
            )
        base_scale = {k: v for k, v in base_scale.items() if k not in drop}
    return {**base_scale, **scale_overrides}


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
