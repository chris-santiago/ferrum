"""Axis value class for per-channel axis configuration."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

from ferrum._configure_mixin import _MISSING, _resolve_band_alias
from ferrum._title_sentinel import TitleParam, _UNSET, _UnsetType, is_unspecified, serialize_title

# Fields whose Python default (``True``/``False``/``"greedy"``) matches the
# renderer's own built-in default.  These are declared with ``_UNSET`` (not
# their concrete default) so an explicitly-passed value that happens to equal
# the renderer default is still distinguishable from "not specified" and
# always reaches the wire (NF-B3, F-L04-04's ``_AXIS_DEFAULTS`` twin): only an
# omitted field (OR an explicit ``None`` — see
# ``ferrum._title_sentinel.is_unspecified``, the shared two-way
# omit-vs-explicit gate this set's fields share with ``Legend.orient``/
# ``direction``) is dropped from ``to_dict()``, never any other explicit
# value — see ``to_dict()``'s docstring for why silently dropping an explicit
# equals-default value breaks the per-channel-wins cascade (D7).
_UNSET_DEFAULTED_FIELDS: frozenset[str] = frozenset(
    {"ticks", "tick_extra", "grid", "labels", "label_flush", "label_overlap", "domain"}
)

# Fields that exist only in the Python layer and must not be forwarded to
# the Rust renderer (they have no corresponding key in EncodingSpec.axis.extra).
_PYTHON_ONLY_FIELDS: frozenset[str] = frozenset({"label_map"})


@dataclass(frozen=True, slots=True, init=False)
class Axis:
    """Per-channel axis configuration.

    Parameters
    ----------
    title : str, optional
        Axis title text.  Pass ``title=None`` to suppress the axis title
        (the field name will not be shown).  Omitting ``title`` entirely keeps
        the field-name default.
    orient : str, optional
        Axis orientation ("top", "bottom", "left", "right").
    ticks : bool, optional
        Show tick marks.  Omitting ``ticks`` (or passing ``ticks=None``,
        treated identically — the same "unset" spelling every other optional
        field here accepts) keeps the renderer's own default (shown); passing
        any other value explicitly — even ``True`` — always reaches the wire,
        so an explicit value beats a conflicting chart-level
        ``configure_axis(ticks=...)`` (per-channel wins).
    tick_count : int, optional
        Suggested number of ticks.
    tick_extra : bool, optional
        Include extra tick at domain boundary.  Same omit-vs-explicit contract
        as ``ticks``.
    tick_min_step : float, optional
        Minimum step between ticks.
    grid : bool, optional
        Show grid lines.  Same omit-vs-explicit contract as ``ticks``.
    grid_dash : list[float], optional
        Grid line dash pattern.
    grid_width : float, optional
        Grid line width.
    grid_color : str, optional
        Grid line color.
    grid_opacity : float, optional
        Grid line opacity.
    labels : bool, optional
        Show tick labels.  Same omit-vs-explicit contract as ``ticks``.
    label_angle : float, optional
        Tick label rotation angle.
    label_flush : bool, optional
        Flush the first and last tick labels against the axis ends so they do
        not overhang the plot area.  Omitting ``label_flush`` keeps the
        renderer's own default (``False``, no flush); passing it explicitly —
        even as ``False`` — always reaches the wire (same omit-vs-explicit
        contract as ``ticks``).
    label_overlap : str, optional
        Label overlap strategy ("greedy", "parity", "rotate").  Omitting
        ``label_overlap`` keeps the renderer's own default (``"greedy"``, the
        graduated collision cascade); passing it explicitly — even as
        ``"greedy"`` — always reaches the wire (same omit-vs-explicit contract
        as ``ticks``).
    label_format : str, optional
        d3-format string for labels.
    label_format_type : str, optional
        Format type ("number", "time").
    label_font_size : float, optional
        Label font size.
    label_color : str, optional
        Label color.
    domain : bool, optional
        Show axis domain line.  Same omit-vs-explicit contract as ``ticks``.
    domain_width : float, optional
        Domain line width.
    domain_color : str, optional
        Domain line color.
    offset : float, optional
        Axis offset from plot area.
    translate : float, optional
        Pixel translation.
    min_band : float, optional
        Lower bound for the reserved axis margin band, in pixels.
    max_band : float, optional
        Upper bound for the reserved axis margin band, in pixels.

        .. deprecated:: 0.17.0
            ``min_extent`` and ``max_extent`` are accepted as aliases for
            ``min_band`` and ``max_band`` respectively, but are deprecated and
            will be removed in a future release.
    title_orient : str, optional
        Title orientation.
    title_font_size : float, optional
        Title font size.
    title_color : str, optional
        Title color.
    title_padding : float, optional
        Title padding from axis.
    values : list, optional
        Explicit tick values.
    zindex : int, optional
        Z-index for layering.
    label_map : dict[str, str], optional
        Mapping from original tick-label text to display text.  Applied in the
        Python layer at render time by renaming the corresponding column values
        in the DataFrame before Rust computes the scale domain.  Keys not
        present in the data are silently ignored.

        Example — rename categorical axis labels::

            fm.Axis(label_map={"a": "Group A", "b": "Group B"})

    Examples
    --------
    >>> import ferrum as fm
    >>> ax = fm.Axis(title="Speed (km/h)", label_angle=-45)
    >>> ax.title
    'Speed (km/h)'
    >>> ax.to_dict()
    {'title': 'Speed (km/h)', 'label_angle': -45}
    """

    title: TitleParam = _UNSET
    orient: str | None = None
    ticks: "bool | None | _UnsetType" = _UNSET
    tick_count: int | None = None
    tick_extra: "bool | None | _UnsetType" = _UNSET
    tick_min_step: float | None = None
    grid: "bool | None | _UnsetType" = _UNSET
    grid_dash: list[float] | None = None
    grid_width: float | None = None
    grid_color: str | None = None
    grid_opacity: float | None = None
    labels: "bool | None | _UnsetType" = _UNSET
    label_angle: float | None = None
    label_flush: "bool | None | _UnsetType" = _UNSET
    label_overlap: "str | None | _UnsetType" = _UNSET
    label_format: str | None = None
    label_format_type: str | None = None
    label_font_size: float | None = None
    label_color: str | None = None
    domain: "bool | None | _UnsetType" = _UNSET
    domain_width: float | None = None
    domain_color: str | None = None
    offset: float | None = None
    translate: float | None = None
    min_band: float | None = None
    max_band: float | None = None
    title_orient: str | None = None
    title_font_size: float | None = None
    title_color: str | None = None
    title_padding: float | None = None
    values: list | None = None
    zindex: int | None = None
    label_map: dict[str, str] | None = None

    # ------------------------------------------------------------------
    # Custom __init__ — accepts deprecated min_extent=/max_extent= aliases
    # and maps them to the canonical min_band=/max_band= fields.
    # The dataclass is declared with init=False so we own the full init.
    # We use object.__setattr__ because the dataclass is frozen=True.
    # ------------------------------------------------------------------

    def __init__(
        self,
        title: TitleParam = _UNSET,
        orient: str | None = None,
        ticks: "bool | None | _UnsetType" = _UNSET,
        tick_count: int | None = None,
        tick_extra: "bool | None | _UnsetType" = _UNSET,
        tick_min_step: float | None = None,
        grid: "bool | None | _UnsetType" = _UNSET,
        grid_dash: list[float] | None = None,
        grid_width: float | None = None,
        grid_color: str | None = None,
        grid_opacity: float | None = None,
        labels: "bool | None | _UnsetType" = _UNSET,
        label_angle: float | None = None,
        label_flush: "bool | None | _UnsetType" = _UNSET,
        label_overlap: "str | None | _UnsetType" = _UNSET,
        label_format: str | None = None,
        label_format_type: str | None = None,
        label_font_size: float | None = None,
        label_color: str | None = None,
        domain: "bool | None | _UnsetType" = _UNSET,
        domain_width: float | None = None,
        domain_color: str | None = None,
        offset: float | None = None,
        translate: float | None = None,
        min_band: float | None = None,
        max_band: float | None = None,
        title_orient: str | None = None,
        title_font_size: float | None = None,
        title_color: str | None = None,
        title_padding: float | None = None,
        values: list | None = None,
        zindex: int | None = None,
        label_map: dict[str, str] | None = None,
        # Deprecated aliases — accepted with a DeprecationWarning.
        min_extent: object = _MISSING,
        max_extent: object = _MISSING,
    ) -> None:
        # Resolve deprecated aliases (min_extent → min_band, max_extent → max_band)
        # through the shared canonical helper so all three surfaces stay in sync.
        min_band = _resolve_band_alias(
            min_band,
            min_extent,
            canonical_name="min_band",
            alias_name="min_extent",
            owner="Axis",
            stacklevel=2,
        )
        max_band = _resolve_band_alias(
            max_band,
            max_extent,
            canonical_name="max_band",
            alias_name="max_extent",
            owner="Axis",
            stacklevel=2,
        )

        object.__setattr__(self, "title", title)
        object.__setattr__(self, "orient", orient)
        object.__setattr__(self, "ticks", ticks)
        object.__setattr__(self, "tick_count", tick_count)
        object.__setattr__(self, "tick_extra", tick_extra)
        object.__setattr__(self, "tick_min_step", tick_min_step)
        object.__setattr__(self, "grid", grid)
        object.__setattr__(self, "grid_dash", grid_dash)
        object.__setattr__(self, "grid_width", grid_width)
        object.__setattr__(self, "grid_color", grid_color)
        object.__setattr__(self, "grid_opacity", grid_opacity)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "label_angle", label_angle)
        object.__setattr__(self, "label_flush", label_flush)
        object.__setattr__(self, "label_overlap", label_overlap)
        object.__setattr__(self, "label_format", label_format)
        object.__setattr__(self, "label_format_type", label_format_type)
        object.__setattr__(self, "label_font_size", label_font_size)
        object.__setattr__(self, "label_color", label_color)
        object.__setattr__(self, "domain", domain)
        object.__setattr__(self, "domain_width", domain_width)
        object.__setattr__(self, "domain_color", domain_color)
        object.__setattr__(self, "offset", offset)
        object.__setattr__(self, "translate", translate)
        object.__setattr__(self, "min_band", min_band)
        object.__setattr__(self, "max_band", max_band)
        object.__setattr__(self, "title_orient", title_orient)
        object.__setattr__(self, "title_font_size", title_font_size)
        object.__setattr__(self, "title_color", title_color)
        object.__setattr__(self, "title_padding", title_padding)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "zindex", zindex)
        object.__setattr__(self, "label_map", label_map)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for the renderer, omitting only unset/None/Python-only fields.

        ``ticks``/``tick_extra``/``grid``/``labels``/``label_flush``/
        ``label_overlap``/``domain`` follow the two-way omit-vs-explicit
        contract :func:`ferrum._title_sentinel.is_unspecified` implements
        (distinct from ``title``'s three-way contract below): "not specified"
        is either omitting the kwarg (``_UNSET``, the field default) or
        passing the field ``=None`` explicitly — the same "unset" spelling
        every other optional field on this class already accepts — and either
        spelling drops the key so the renderer's own default applies. Any
        OTHER explicitly passed value — including one that happens to equal
        what the renderer would have picked anyway — always reaches the wire
        (NF-B3). Skipping an explicit-equals-default value here previously
        made it indistinguishable from "not specified", which silently lost
        the per-channel-wins cascade for that field (e.g. an explicit
        ``Axis(label_overlap="greedy")`` could be overridden by a conflicting
        ``configure_axis(label_overlap=...)`` even though the per-channel
        value should always win).

        ``label_format`` preset names are resolved to their d3-format/strftime
        strings via :func:`ferrum.format_presets.resolve_format_field` before
        serialization (NF-B1); an unrecognized name passes through as an
        honest raw spec. The resolved ``format_type`` fills
        ``label_format_type`` only when the caller did not set it explicitly.
        """
        from ferrum.format_presets import resolve_format_field

        resolved_format, resolved_format_type = resolve_format_field(
            self.label_format, self.label_format_type
        )
        result: dict[str, Any] = {}
        for f in fields(self):
            # Python-only fields have no Rust counterpart — never forward them.
            if f.name in _PYTHON_ONLY_FIELDS:
                continue
            if f.name == "title":
                # Three-way title contract (mirrors base.py and prepare.rs):
                #   _UNSET  → omit key; Rust falls back to field name (default)
                #   None    → emit ""; Rust treats "" as suppress
                #   "Foo"   → emit "Foo" verbatim
                serialized = serialize_title(getattr(self, f.name))
                if serialized is not None:
                    result["title"] = serialized
                continue
            if f.name in _UNSET_DEFAULTED_FIELDS:
                val = getattr(self, f.name)
                if not is_unspecified(val):
                    result[f.name] = val
                continue
            if f.name == "label_format":
                if resolved_format is not None:
                    result["label_format"] = resolved_format
                continue
            if f.name == "label_format_type":
                if resolved_format_type is not None:
                    result["label_format_type"] = resolved_format_type
                continue
            val = getattr(self, f.name)
            # Skip None values for all other fields
            if val is None:
                continue
            result[f.name] = val
        return result


def _axis_suppressed_dict() -> dict[str, Any]:
    """Return the dict equivalent of axis=False (suppress all axis elements)."""
    return {
        "domain": False,
        "ticks": False,
        "labels": False,
        "title": None,
        "grid": False,
    }


_LABEL_FORMAT_KEYS = ("label_format", "labelFormat")
_LABEL_FORMAT_TYPE_KEYS = ("label_format_type", "labelFormatType")


def _resolve_axis_dict_format(value: dict[str, Any]) -> dict[str, Any]:
    """Resolve a raw axis dict's ``label_format`` preset name before forwarding.

    Mirrors :meth:`Axis.to_dict`'s resolution (NF-B1) so a preset name never
    reaches the renderer unresolved regardless of whether the caller built an
    :class:`Axis` or passed a raw dict directly (``fm.X("f", axis={...})``).

    Resolves under either serde-honored spelling the caller used
    (``label_format``/``labelFormat``, and its type sibling
    ``label_format_type``/``labelFormatType`` — both pairs are accepted wire
    spellings per ``AXIS_STYLE_ALIAS_KEYS`` in
    ``crates/ferrum-core/src/render/chart_config.rs``), and writes the
    resolved values back under whichever spelling the caller supplied. If
    the caller already supplied a type key (either spelling), that key is
    reused; a derived type is never written under a second spelling beside
    a caller-supplied one.

    Always returns a fresh dict (never the caller's own object), so
    ``_normalize_axis``'s dict path has one aliasing contract regardless of
    whether a format key was present.
    """
    result = dict(value)
    format_key = next((k for k in _LABEL_FORMAT_KEYS if k in value), None)
    if format_key is None:
        return result

    from ferrum.format_presets import resolve_format_field

    type_key = next((k for k in _LABEL_FORMAT_TYPE_KEYS if k in value), None)
    spec, format_type = resolve_format_field(
        value.get(format_key), value.get(type_key) if type_key is not None else None
    )
    if spec is not None:
        result[format_key] = spec
    if format_type is not None:
        if type_key is None:
            type_key = "labelFormatType" if format_key == "labelFormat" else "label_format_type"
        result[type_key] = format_type
    return result


def _normalize_axis(value: Any) -> dict[str, Any] | None:
    """Normalize an axis kwarg value to a dict or None.

    Accepts:
    - Axis instance -> .to_dict()
    - False -> suppression dict
    - dict -> pass through (with ``label_format`` preset resolution)
    - None -> None (meaning "not specified")
    """
    if value is None:
        return None
    if value is False:
        return _axis_suppressed_dict()
    if isinstance(value, Axis):
        return value.to_dict()
    if isinstance(value, dict):
        return _resolve_axis_dict_format(value)
    return None
