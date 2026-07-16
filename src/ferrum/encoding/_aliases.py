"""Channel-alias rules — map convenience channels onto their render targets.

The alias graph is expressed as DATA (the ``_ENCODING_ALIASES`` table) driven by
one generic loop, so the policy is visible in one place rather than spread across
inline branches. `apply_channel_aliases` operates on shallow copies of the
encoding and mark-kwargs dicts from ``Chart.to_spec`` — it does not mutate the
chart's internal state.

Ordering is load-bearing: rules are applied top-to-bottom, and an earlier alias
that already populated the target wins over a later one (the later one then sees
``color`` present and follows its conflict policy).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class _EncodingAlias:
    """One ``source -> target`` aliasing rule within the encoding dict.

    Parameters
    ----------
    source :
        The convenience channel name read from the encoding dict.
    target :
        The canonical channel name the source maps onto.
    on_conflict :
        What to do when *target* is already present in the encoding:

        - ``"keep_target"`` — silently leave the existing target, dropping the
          source's mapping (no warning).
        - ``"warn_drop"`` — drop the source's mapping and emit a one-time
          warning (only when the source carries a concrete field).
    """

    source: str
    target: str
    on_conflict: Literal["keep_target", "warn_drop"]


# Order matters: fill is resolved before stroke, so when both are present and
# color is absent, fill wins the color target and stroke then sees color present
# and follows its warn_drop policy.
_ENCODING_ALIASES: tuple[_EncodingAlias, ...] = (
    _EncodingAlias(source="fill", target="color", on_conflict="keep_target"),
    _EncodingAlias(source="stroke", target="color", on_conflict="warn_drop"),
)


def apply_channel_aliases(enc: dict, mk: dict) -> tuple[dict, dict]:
    """Apply channel-alias rules, mapping convenience channels to their targets.

    Operates on shallow copies of the encoding and mark-kwargs dicts from
    ``to_spec()`` — does not mutate the chart's internal state.

    Encoding-dict aliases (in `_ENCODING_ALIASES`, applied in order):

    1. ``fill`` -> ``color`` when ``color`` is not already present.
    2. ``stroke`` -> ``color`` when ``color`` is not already present;
       when ``color`` IS present, the stroke encoding is dropped with a
       one-time warning.

    Separately, ``detail`` -> ``mk["detail"]`` via ``setdefault`` (always,
    regardless of other channels — it targets the mark-style kwargs, not the
    encoding dict).

    Note: ``fill_opacity`` is no longer aliased to ``opacity``. It is a
    first-class renderer-honored channel that emits a per-element SVG
    ``fill-opacity`` attribute, separate from ``opacity`` (which bakes into the
    fill RGBA alpha).

    Returns the (possibly-modified) ``(enc, mk)`` pair.
    """
    from ferrum.repeat import _RepeatPlaceholder

    for rule in _ENCODING_ALIASES:
        if rule.source not in enc:
            continue
        if rule.target not in enc:
            enc[rule.target] = enc[rule.source]
            continue
        # Target already present — apply the conflict policy.
        if rule.on_conflict == "warn_drop":
            src_ch = enc[rule.source]
            if src_ch.field is not None and not isinstance(src_ch.field, _RepeatPlaceholder):
                # Can't map to a scale -- the source encoding produces no visual
                # effect when the target is already mapped.  Warn once so callers
                # know the channel was dropped.
                from ferrum._warn import warn_once

                warn_once(
                    "encoding",
                    "stroke_dropped_by_color",
                    "encode(stroke=...) is ignored when color= is also encoded; "
                    "stroke is aliased to color only when color is absent.",
                )
        # "keep_target" drops the source silently; both policies leave the
        # existing target untouched.

    # Detail -> mark_style.detail (separate target: mark kwargs, not encoding).
    if "detail" in enc:
        detail_ch = enc["detail"]
        if detail_ch.field is not None and not isinstance(detail_ch.field, _RepeatPlaceholder):
            mk.setdefault("detail", detail_ch.field)

    return enc, mk
