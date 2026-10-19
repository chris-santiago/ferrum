"""Render-side consumer for ``Chart.override`` payloads.

:mod:`ferrum._override_apply` is a pure transform that resolves, validates, and
folds a flat override dict into an :class:`~ferrum._override_apply.OverridePayload`.
This module is the matching *consumer*: it deep-merges each payload piece into the
assembled spec / chart-config / viewport at its injection point and emits the
deprecation warnings.  It runs **last** among presentation sources so override wins
the cascade (spec §7: override > per-channel ``axis=``/``legend=`` > ``configure_*``
> theme > ``set_default_theme`` > Rust defaults).

Six payload pieces are consumed here, called in this order:
:func:`apply_mark_style`; the encoding piece, which is three functions
(:func:`refuse_binned_color_scale_overrides` on a binned color, then
:func:`merge_encoding_scale`, then :func:`refuse_malformed_scale_overrides`
— all called from ``_spec_build.py``'s ``_with_override_scale``, with a
staged validation order inside ``merge_encoding_scale`` itself, see its own
docstring); and :func:`apply_coord` — all three from inside
:meth:`Chart.to_spec` — then, once ``to_spec`` returns,
:func:`merge_chart_config`, :func:`apply_properties` and
:func:`emit_deprecations` from :meth:`Chart._render_inputs`.  The encoding
piece's leaf-path registry (:func:`refuse_malformed_scale_overrides`, which
knows each override's own flat path) and its merge gate
(:func:`merge_encoding_scale`, which knows the channel's wire keys) share
this one module, so the override leaves and the wire keys they merge into
are defined together.

The functions here are pure (no ``Chart`` import, no I/O, no global state) apart from
:func:`emit_deprecations`, whose sole effect is the documented ``DeprecationWarning``.
The Chart render path (:meth:`Chart._render_inputs`, :meth:`Chart.to_spec`) builds the
payload once per render via ``build_payload`` and threads it through these helpers.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import Any, NoReturn

from ferrum._core import (  # type: ignore[attr-defined]
    scale_accepted_keys as _scale_accepted_keys,
    scale_padding_keys as _scale_padding_keys,
    translate_scale_switch as _translate_scale_switch,
    validate_scale_dict as _validate_scale_dict,
    validate_scale_values as _validate_scale_values,
)
from ferrum._override_apply import _SCALE_TYPE_TAGS, OverridePayload
from ferrum._entry import EntryRefusal
from ferrum._layer import merge_style
from ferrum.exceptions import _OverrideRefusal
from ferrum.marks.base import resolve_layer_style


def _refuse_old_type_padding_value(old_type: str, key: str, value: Any) -> None:
    """Refuse *value* under *old_type*'s own bound for padding-family *key*.

    The one place :func:`merge_encoding_scale`'s padding-family check
    voices a refusal, naming both the scale and the key the user wrote:
    ``scale: <key> <clause>``, e.g. ``scale: padding must be in [0, 1];
    got 1.5``.

    *value* is probed under the real shape gate
    (``ferrum._core.validate_scale_dict``) FIRST. A wrong-type value (a
    string where a number belongs) is refused there, in that gate's own
    ``"scale: "``-prefixed clause -- unwrapped and un-keyed, since that
    clause names no key at all, unlike the bound clause this helper wraps
    below. This is what makes a shape-malformed value read identically
    whether this check reaches it (because the switch carries its key
    under a new name) or the existing dropped-key probe does (because the
    switch drops it), and identically with or without the override in play
    (the ``tests/test_override_scale_merge.py`` ``_DROPPED_BUCKET_CASES``
    rule).

    Only a value that passes shape but fails *old_type*'s own BOUND
    (``ferrum._core.validate_scale_values``, whose own clause is bare -- no
    call, channel or key, by that function's contract) reaches the
    ``except`` below, where the key name is added -- nothing else would
    ever name it, since a carried value's key is renamed before it
    reaches any downstream gate.
    """
    _validate_scale_dict({"type": old_type, key: value})
    try:
        _validate_scale_values({"type": old_type, key: value})
    except ValueError as err:
        raise ValueError(f"scale: {key} {err}") from None


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
    - When the type IS changing, each base-scale key is validated exactly once.
      A key the OLD type never accepted is probed under the old type through
      the real wire gate (``ferrum._core.validate_scale_dict``), so it refuses
      with the identical message that base scale raises standalone. The
      padding-family keys (``padding``, ``paddingInner``, ``paddingOuter``,
      ``align``) are translated by meaning onto the new type -- an ordinal
      inner gap becomes a band inner gap, a point's end padding becomes a
      band outer padding, and so on -- by ``ferrum._core.translate_scale_switch``,
      the one Rust owner of that meaning table. Every other key
      the new type doesn't accept is dropped, same as a padding-family key
      with no meaning under the new type. A dropped key's value is probed
      under the old type the same way an unknown-to-old-type key is, so a
      value the switch is about to discard still refuses if it was already
      malformed.
    - Every padding-family key present on the base scale (``ferrum._core.
      scale_padding_keys(old_type)``, so this restates no pair rule of its
      own) is also value-checked under the OLD type, through
      :func:`_refuse_old_type_padding_value`, before the merge returns --
      whether the switch carries that key's value under a new name or drops
      it. The translation preserves bounds one way by construction: the key
      a carry writes accepts every value the key it reads accepts under the
      old type (band ``padding``'s ``[0, 1]`` lies inside point
      ``padding``'s finite ``>= 0``), so this old-type check catches every
      out-of-bounds carried value, in the key the user wrote. It also
      refuses a value the new type alone would accept, because that value
      was out of bounds under the type the user wrote: a band
      ``padding=1.5`` switched to point is refused, though
      ``PointScale(padding=1.5)`` is valid. This
      is the one check for BOTH a carried value (nothing else ever checks
      it -- it reaches the new type's own field under a different key name,
      so the new type's downstream gate sees a key the user never wrote) and
      a dropped padding-family value's BOUNDS specifically (the dropped-key
      probe above only checks shape, via ``validate_scale_dict``). Two
      refusal voices exist by design: a BOUND failure names the scale and
      the key the user wrote (``scale: <key> <clause>``) -- caught only by
      this check, and only when the override switches the scale's type;
      without the override, the same value reaches ``to_spec()``
      unchecked. A wrong-type value is caught by the existing shape probe
      first instead and keeps THAT probe's own ``scale: invalid type ...``
      text, which names no key; this value reads identically whether this
      check or the dropped-key probe catches it, and identically with or
      without the override in play.
    - A key BOTH types accept, and survives the translation untouched, is
      deliberately NOT value-validated here: its value must reach the new
      type's own downstream gate untouched (a string ``domain`` is illegal
      under ``linear`` and legal under ``band``).
    - A non-``str`` type, or one no ``ferrum._core.scale_type_tags()`` entry
      names, skips filtering entirely, so the gate's own "unknown variant" /
      type error surfaces instead of a PyO3 argument-coercion error naming
      ``scale_accepted_keys``'s parameter -- checked against the published
      tag list rather than by catching ``translate_scale_switch``'s own
      ``ValueError``, so a genuine Rust refusal (e.g. the collision case in
      its docstring) can never be laundered into this fall-through.

    ``base_scale`` is never mutated.  ``tests/test_override_scale_merge.py``
    pins each of these rules.
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

    # Non-string type, or one no real ScaleSpec tag names: don't hand it to
    # the &str-typed translate_scale_switch, whose own "unknown variant"
    # ValueError is not this gate's voice to catch -- checking against the
    # published tag list up front means a ValueError translate_scale_switch
    # DOES raise (its collision refusal) always propagates, never falls
    # through here as if it meant "unknown type".
    if not isinstance(new_type, str) or new_type not in _SCALE_TYPE_TAGS:
        return {**base_scale, **scale_overrides}
    translated, dropped = _translate_scale_switch(old_type, new_type, base_scale)

    # Bucket 2: keys the translation above drops (the keys the new type does
    # not accept, and padding-family keys no carry reads). They never reach
    # new_type's downstream gate, so nobody validates their SHAPE if this
    # probe doesn't. Bucket 3 (survivors) is checked downstream, under
    # new_type.
    if dropped:
        _validate_scale_dict(
            {"type": old_type, **{k: base_scale[k] for k in dropped}},
        )

    # Padding-family BOUNDS (and, for a carried key, shape too -- nothing
    # else ever checks a carried value): see the docstring and
    # _refuse_old_type_padding_value. Runs after the dropped-key shape
    # probe above, so a key that probe already refuses (e.g. a
    # shape-malformed `align`) keeps that probe's voice; this adds only the
    # cases it doesn't cover -- a bound failure as "scale: <key> <clause>",
    # a wrong-type value through the same shape probe's own
    # "scale: invalid type ..." text (naming no key).
    for key in _scale_padding_keys(old_type):
        value = base_scale.get(key)
        if value is not None:
            _refuse_old_type_padding_value(old_type, key, value)

    return {**translated, **scale_overrides}


def refuse_binned_color_scale_overrides(
    binned_class: str, scale_overrides: Mapping[str, Any], paths: Mapping[str, str]
) -> None:
    """Refuse an override leaf that would replace what a binned color's ``bin=`` builds.

    A binned color (``Color/Fill/Stroke(bin=...)``, written as *binned_class*)
    draws through a ``bin-ordinal`` scale whose rule and edges its ``bin=``
    owns, so an override leaf is honored only when it is a key that scale
    accepts and ``bin=`` does not own (read from Rust's accepted-key table,
    today ``scheme``), or ``type`` set to ``"bin-ordinal"`` itself. Any other
    leaf (another type, ``bins``, ``bin``, ``domain``, ...) contradicts
    ``bin=`` as an explicit ``scale=`` on the channel does, and is refused in
    that refusal's voice, naming the class and the override path written.

    *scale_overrides* and *paths* are one channel's wire-keyed override leaves
    and their flat override paths, as for
    :func:`refuse_malformed_scale_overrides`.

    Raises
    ------
    FerrumOverrideError
        On the first refused leaf, as an ``_OverrideRefusal`` naming
        ``{entry}({path}=...)``.
    """
    from ferrum.encoding.base import _BIN_OWNED_SCALE_KEYS, _BIN_WITH_SCALE_MESSAGE

    honored = set(_scale_accepted_keys("bin-ordinal")) - set(_BIN_OWNED_SCALE_KEYS)
    for key, value in scale_overrides.items():
        if key in honored or (key == "type" and value == "bin-ordinal"):
            continue
        path = paths[key]
        raise _OverrideRefusal(
            lambda entry, path=path: _BIN_WITH_SCALE_MESSAGE.format(
                channel=binned_class, scale=f"{entry}({path}=...)"
            )
        )


def refuse_malformed_scale_overrides(
    scale_type: Any, scale_overrides: Mapping[str, Any], paths: Mapping[str, str]
) -> None:
    """Refuse a malformed OVERRIDE scale value, naming the override path.

    Rust owns every verdict: ``ferrum._core.validate_scale_values`` runs the
    same data-free checks the render path would otherwise run -- band
    padding bounds, point ``padding`` (finite >= 0) and ``align`` (``[0,
    1]``), and ordinal ``padding`` (``[0, 1]``) --
    and raises a bare clause with no call, channel or key in it. This
    function adds only provenance -- it routes each override wire key's
    value to that gate one at a time and, on refusal, names the flat
    override path the caller actually wrote (``x_scale_padding``, not
    ``padding``). It holds no bound, no vocabulary and no default of its own.

    *scale_overrides* is one channel's ``payload.encoding[channel]["scale"]``
    (wire-keyed) and *paths* its matching ``payload.encoding_paths[channel]``
    (wire key -> flat override path); both are derived from the same
    ``{wire_key: (value, path)}`` record in ``build_payload``, so every key
    in *scale_overrides* has a matching entry in *paths* by construction, not
    by convention. *scale_type* is the EFFECTIVE type after
    :func:`merge_encoding_scale` -- the override's own type when the
    override changes it, the chart's own when it doesn't, or ``None`` when
    neither side names one (the typeless case ``_scale_to_dict`` itself
    defaults to ``"linear"``).

    If the override sets ``"type"`` itself, that value is probed alone
    first, under its own raw value (not yet merged) -- a malformed type
    (non-``str``, or an unknown variant) is refused immediately, naming
    ``type``'s own override path, without probing any other key under a
    type that is not even real. Every other overridden wire key is then
    probed as a one-key fragment, built through
    :func:`ferrum.encoding._scale._scale_to_dict` so a ``Parameter`` domain
    or a temporal date/string converts exactly as it would in the merged
    dict, before the bare value ever reaches Rust. The fragment omits the
    ``type`` key entirely when *scale_type* is ``None``, so
    ``_scale_to_dict`` supplies its own typeless default rather than this
    function mirroring it as a second ``"linear"`` literal. That conversion
    call sits INSIDE the probe's ``try``, alongside the Rust check, because
    the conversion itself can raise on a malformed value (an unparseable
    date, a non-numeric non-date element) -- that failure is named here
    too, not left to surface unnamed from a later, unguarded conversion of
    the merged dict.

    Raises
    ------
    FerrumOverrideError
        On the first malformed value, as ``_OverrideRefusal(lambda entry:
        f"{entry}({path}=...) {clause}")`` -- the same voice the constructor
        refusals (e.g. ``BandScale(padding=...)``) and this module's own
        ``{entry}(mark={...})`` refusal use.
    """
    from ferrum.encoding._scale import _scale_to_dict

    def _refuse(path: str, clause: str) -> NoReturn:
        raise _OverrideRefusal(lambda entry: f"{entry}({path}=...) {clause}") from None

    if "type" in scale_overrides:
        try:
            _validate_scale_values({"type": scale_overrides["type"]})
        except ValueError as err:
            _refuse(paths["type"], str(err))

    for key, value in scale_overrides.items():
        if key == "type":
            continue
        try:
            type_key = {} if scale_type is None else {"type": scale_type}
            fragment = _scale_to_dict({**type_key, key: value})
            _validate_scale_values(fragment)
        except (TypeError, ValueError) as err:
            _refuse(paths[key], str(err))


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
    mark_style: dict[str, Any], payload: OverridePayload, *, mark: Any
) -> dict[str, Any]:
    """Return *mark_style* merged with ``payload.mark_style`` (override wins).

    Mark-style overrides apply to the chart's single base/primary mark.  A chart
    with multiple marks (a layered chart) has no single primary mark, so a
    ``mark_*`` override is ambiguous and raises :class:`FerrumOverrideError`
    (spec §11 Q4).  A leaf the chart's own mark does not honor (``mark_shape``
    on a line) raises it too, as ``mark_<mark>(...)`` would refuse the kwarg,
    and so does a value of the wrong type. What is merged is the style as
    :func:`ferrum.marks.base.resolve_layer_style` returns it, so an override
    reaches the wire in the form every other door stores. *mark_style* is
    not mutated; a new dict is returned.

    Parameters
    ----------
    mark_style:
        The base mark's style dict assembled in ``Chart.to_spec``.
    payload:
        The resolved override payload.
    mark:
        The chart's one mark, or ``None`` for a layered chart, which has
        none and so refuses any mark-style override.
    """
    if not payload.mark_style:
        return mark_style
    if mark is None:
        leaves = ", ".join(sorted(payload.mark_style))
        raise _OverrideRefusal(
            lambda entry: (
                f"{entry}: mark_* override ({leaves}) is ambiguous on a multi-mark chart: "
                "a layered chart has no single primary mark to apply it to. "
                "Set the style on the specific layer's mark_*() call instead."
            )
        )
    try:
        style = resolve_layer_style("Chart.override", mark, payload.mark_style)
    except EntryRefusal as err:
        raise _OverrideRefusal(err.message_for, err.entry) from None
    return merge_style(mark_style, style)


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
        coord_kind = type(base).__name__
        raise _OverrideRefusal(
            lambda entry: (
                f"{entry}: coord_* override ({leaves}) is not a field of "
                f"{coord_kind}. Set a matching coord via .coord(...) first, "
                "or override a field that coord exposes."
            )
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
    corresponding viewport dimension unchanged, and so does a ``None``
    override, which :func:`ferrum._override_apply.build_payload` leaves out.
    Each present value is the positive ``float`` (or ``"container"``) that
    ``build_payload`` validated and stored.
    """
    if not payload.properties:
        return viewport
    width, height = viewport
    if "width" in payload.properties:
        width = payload.properties["width"]
    if "height" in payload.properties:
        height = payload.properties["height"]
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
