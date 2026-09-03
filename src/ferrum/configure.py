"""Chart-level configuration dataclasses for Ferrum's declarative config surface."""

from __future__ import annotations

import math
import numbers
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any

from ferrum._configure_mixin import _MISSING, _resolve_band_alias
from ferrum._title_sentinel import _UNSET, TitleParam, is_unspecified, serialize_title
from ferrum._validate import validate_choice, validate_fraction_value, validate_pixel_value
from ferrum.axis import validate_label_overlap
from ferrum.legend import validate_legend_direction, validate_legend_orient


_VALID_TITLE_ANCHORS = frozenset({"start", "middle", "end"})

_AXIS_XY_DEPRECATION_MSG = (
    "AxisConfig.x / configure_axis(x=...) has no effect and is deprecated; "
    "use Chart.axis(x=False) / Chart.axis(y=False) to show/hide an axis."
)


def _warn_axis_xy_deprecated(*, x: bool, y: bool, stacklevel: int) -> None:
    """Emit a DeprecationWarning when the vestigial ``x``/``y`` flags are set to ``False``.

    ``x=True``/``y=True`` are the do-nothing defaults and never warn; only the
    meaningful (but no-op) ``False`` intent is flagged. Centralised here so the
    direct-construction path (:meth:`AxisConfig.__init__`) and the
    ``configure_axis`` mixin method share one message.
    """
    if x is False or y is False:
        warnings.warn(_AXIS_XY_DEPRECATION_MSG, DeprecationWarning, stacklevel=stacklevel + 1)


def _to_dict_omit_none(obj: Any) -> dict[str, Any]:
    """Serialize a dataclass to dict, omitting fields whose value is None."""
    result: dict[str, Any] = {}
    for f in fields(obj):
        val = getattr(obj, f.name)
        if val is not None:
            result[f.name] = val
    return result


def _validate_domain_bounds(domain_min: float | None, domain_max: float | None) -> None:
    """Refuse a scale-domain pair the scale constructors would refuse.

    ``AxisConfig(domain_min=, domain_max=)`` asks for the same thing
    ``LinearScale(domain=[lo, hi])`` asks for, so it must refuse the same
    inputs with the same words — the construction-time contract, at the
    construction-time surface. Two shapes are rejected:

    * a **degenerate** pair (``lo == hi``): a zero-width domain clips every
      mark away and renders a plot indistinguishable from a bug. The sentence
      is Rust's ``DEGENERATE_DOMAIN_MESSAGE`` verbatim, which
      ``LinearScale(domain=[10, 10])`` also raises, so a user meeting the
      contract from either surface reads identical words.
    * a **non-finite** bound (``nan``/``inf``): these are not JSON-encodable,
      so without this they surfaced as
      ``ValueError: chart_config: expected value at line 1 column 27`` — a
      serializer artifact naming neither the field nor the contract.

    ``lo > hi`` is deliberately NOT refused: ``LinearScale(domain=[50, 0])``
    is an accepted reversed axis, and this surface matches it.
    """
    for name, value in (("domain_min", domain_min), ("domain_max", domain_max)):
        if value is None:
            continue
        # `isinstance` first: `math.isfinite` raises a bare
        # `TypeError: must be real number, not str` on a non-numeric bound,
        # which escapes the ferrum voice this same function raises two lines
        # below. All three bad shapes (non-numeric, non-finite, degenerate)
        # now read alike. `bool` is excluded deliberately — it is a `Real`
        # subclass, and `domain_min=True` is a mistake, not a bound of 1.
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise ValueError(f"AxisConfig: {name}={value!r} must be a finite number.")
        if not math.isfinite(value):
            raise ValueError(f"AxisConfig: {name}={value!r} must be a finite number.")
    if domain_min is not None and domain_max is not None and domain_min == domain_max:
        raise ValueError(f"AxisConfig: domain_min/domain_max: {_DEGENERATE_DOMAIN_MESSAGE}")


#: Rust's ``scale::core::DEGENERATE_DOMAIN_MESSAGE``, quoted verbatim so the
#: chart-level domain surface and every scale constructor refuse a zero-width
#: domain in the same words.
_DEGENERATE_DOMAIN_MESSAGE = "domain endpoints must differ (lo != hi)"


@dataclass(frozen=True, init=False)
class AxisConfig:
    """Chart-level axis configuration applied uniformly to all axes (or a specific one).

    Parameters
    ----------
    x, y : bool, default True
        Deprecated and has no effect. Use ``Chart.axis(x=False)`` /
        ``Chart.axis(y=False)`` to show or hide an axis.
    label_angle : float, optional
        Tick label rotation angle in degrees.
    label_font_size : float, optional
        Tick label font size.
    label_color : str, optional
        Tick label color.
    label_format : str, optional
        Named format preset (see ``ferrum.format_presets``). Must be a
        recognized preset key; raises ``ValueError`` at construction
        otherwise. For a raw d3-format/strftime string, use
        ``label_format_raw`` instead. Mutually exclusive with
        ``label_format_raw``.
    label_format_raw : str, optional
        Raw d3-format or strftime string passed directly to the renderer.
        Mutually exclusive with ``label_format``.
    label_overlap : str, optional
        Label overlap strategy: ``"greedy"`` (the graduated collision
        cascade, the default), ``"parity"`` (stride-2 decimation),
        ``"rotate"`` (force the rotate stage), or ``"true"``/``"false"``
        (show every label / the ``"greedy"`` alias). An unrecognized token
        raises ``ValueError`` at construction naming the accepted set.
    tick_count : int, optional
        Suggested number of ticks.
    tick_size : float, optional
        Tick mark length in pixels.
    tick_values : list, optional
        Explicit tick values.
    title_font_size : float, optional
        Axis title font size.
    title_color : str, optional
        Axis title color.
    title_padding : float, optional
        Padding between the axis title and tick labels.
    label_padding : float, optional
        Pixel gap between the tick mark endpoint and the label text baseline.
        Defaults to 2.0 when not set.
    domain : bool, optional
        Show the axis domain line.
    domain_color : str, optional
        Domain line color.
    domain_width : float, optional
        Domain line width.
    grid : bool, optional
        Show grid lines.
    grid_color : str, optional
        Grid line color.
    grid_dash : list[float], optional
        Grid line dash pattern.
    grid_width : float, optional
        Grid line width.
    domain_min : float, optional
        Minimum value of the scale domain.
    domain_max : float, optional
        Maximum value of the scale domain.
    nice : bool, optional
        Round the scale domain to nice round values.
    zero : bool, optional
        Include zero in the scale domain.
    grid_opacity : float, optional
        Grid line opacity (0--1).
    orient : str, optional
        Axis side: ``"top"``/``"bottom"`` (x) or ``"left"``/``"right"`` (y).
        Because chart-level ``axis`` applies to both axes, a single value is
        valid for only one of them; set ``orient`` via ``axis_x`` / ``axis_y``
        (or per-channel ``fm.Axis(orient=...)``) instead of the general ``axis``.
    translate : float, optional
        Pixel translation of the axis group perpendicular to its line.
    min_band : float, optional
        Lower bound for the reserved axis margin band, in pixels.
    max_band : float, optional
        Upper bound for the reserved axis margin band, in pixels.

        .. deprecated:: 0.17.0
            ``min_extent`` and ``max_extent`` are accepted as aliases for
            ``min_band`` and ``max_band`` respectively, but are deprecated and
            will be removed in a future release.
    tick_extra : bool, optional
        Append an extra tick at each domain boundary.
    tick_min_step : float, optional
        Minimum step between ticks in data space.
    title_orient : str, optional
        Side/orientation of the axis title.
    zindex : int, optional
        Coarse draw order of the axis relative to marks.
    labels : bool, optional
        Show or hide the tick labels. A per-channel ``fm.Axis(labels=...)``
        wins over this.
    ticks : bool, optional
        Show or hide the tick marks. Same precedence as ``labels``.
    title : str or None, optional
        Axis title text, following the **same** three-way contract as the
        per-channel :class:`~ferrum.axis.Axis` surface (one policy, one
        implementation — see :mod:`ferrum._title_sentinel`): omit it to keep
        the field-name default, pass ``None`` to suppress the title, or pass
        a string to use it verbatim. ``""`` also suppresses.

        A per-channel ``X(title=...)`` / ``fm.Axis(title=...)`` wins over
        this, *including* when the per-channel value is the suppression — a
        chart-level title never resurrects a title the channel deliberately
        removed.
    offset : float, optional
        Shift the axis away from the plot edge by N pixels.
    label_flush : bool, optional
        Align the first and last tick labels flush with the axis ends
        instead of letting them overhang the plot bounds.
    """

    x: bool = True
    y: bool = True
    label_angle: float | None = None
    label_font_size: float | None = None
    label_color: str | None = None
    label_format: str | None = None
    label_format_raw: str | None = None
    label_overlap: str | None = None
    tick_count: int | None = None
    tick_size: float | None = None
    tick_values: list | None = None
    title_font_size: float | None = None
    title_color: str | None = None
    title_padding: float | None = None
    domain: bool | None = None
    domain_color: str | None = None
    domain_width: float | None = None
    grid: bool | None = None
    grid_color: str | None = None
    grid_dash: list[float] | None = None
    grid_width: float | None = None
    domain_min: float | None = None
    domain_max: float | None = None
    nice: bool | None = None
    zero: bool | None = None
    label_padding: float | None = None
    grid_opacity: float | None = None
    orient: str | None = None
    translate: float | None = None
    min_band: float | None = None
    max_band: float | None = None
    tick_extra: bool | None = None
    tick_min_step: float | None = None
    title_orient: str | None = None
    zindex: int | None = None
    labels: bool | None = None
    ticks: bool | None = None
    #: Three-way (`_UNSET` / `None` / `str`) — see :mod:`ferrum._title_sentinel`.
    title: TitleParam = _UNSET
    offset: float | None = None
    label_flush: bool | None = None

    # ------------------------------------------------------------------
    # Custom __init__ — accepts deprecated min_extent=/max_extent= aliases
    # and maps them to the canonical min_band=/max_band= fields.
    # The dataclass is declared with init=False so we own the full init.
    # We use object.__setattr__ because the dataclass is frozen=True.
    # ------------------------------------------------------------------

    def __init__(
        self,
        x: bool = True,
        y: bool = True,
        label_angle: float | None = None,
        label_font_size: float | None = None,
        label_color: str | None = None,
        label_format: str | None = None,
        label_format_raw: str | None = None,
        label_overlap: str | None = None,
        tick_count: int | None = None,
        tick_size: float | None = None,
        tick_values: list | None = None,
        title_font_size: float | None = None,
        title_color: str | None = None,
        title_padding: float | None = None,
        domain: bool | None = None,
        domain_color: str | None = None,
        domain_width: float | None = None,
        grid: bool | None = None,
        grid_color: str | None = None,
        grid_dash: list[float] | None = None,
        grid_width: float | None = None,
        domain_min: float | None = None,
        domain_max: float | None = None,
        nice: bool | None = None,
        zero: bool | None = None,
        label_padding: float | None = None,
        grid_opacity: float | None = None,
        orient: str | None = None,
        translate: float | None = None,
        min_band: float | None = None,
        max_band: float | None = None,
        tick_extra: bool | None = None,
        tick_min_step: float | None = None,
        title_orient: str | None = None,
        zindex: int | None = None,
        labels: bool | None = None,
        ticks: bool | None = None,
        title: TitleParam = _UNSET,
        offset: float | None = None,
        label_flush: bool | None = None,
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
            owner="AxisConfig",
            stacklevel=2,
        )
        max_band = _resolve_band_alias(
            max_band,
            max_extent,
            canonical_name="max_band",
            alias_name="max_extent",
            owner="AxisConfig",
            stacklevel=2,
        )

        # Reuse the existing validation logic from __post_init__.
        _warn_axis_xy_deprecated(x=x, y=y, stacklevel=2)
        if label_format is not None and label_format_raw is not None:
            raise ValueError(
                "AxisConfig: 'label_format' and 'label_format_raw' are mutually exclusive; "
                "provide at most one."
            )
        # Eager preset-name validation (NF-B1, 2026-09-02): AxisConfig's
        # label_format is preset-names-only by contract — it is not one of
        # the four raw-spec-accepting surfaces. A raw d3-format/strftime
        # string belongs in the dedicated, mutually-exclusive
        # label_format_raw field. An unrecognized name is a typed
        # construction-time refusal, not a silently-passed-through raw spec
        # (the exact NF-B1 harm class this surface must stay immune to).
        if label_format is not None:
            from ferrum.format_presets import resolve_format

            resolve_format(label_format)

        _validate_domain_bounds(domain_min, domain_max)

        if label_overlap is not None:
            validate_label_overlap("AxisConfig.label_overlap", label_overlap)

        object.__setattr__(self, "x", x)
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "ticks", ticks)
        object.__setattr__(self, "title", title)
        object.__setattr__(self, "offset", offset)
        object.__setattr__(self, "label_flush", label_flush)
        object.__setattr__(self, "label_angle", label_angle)
        object.__setattr__(self, "label_font_size", label_font_size)
        object.__setattr__(self, "label_color", label_color)
        object.__setattr__(self, "label_format", label_format)
        object.__setattr__(self, "label_format_raw", label_format_raw)
        object.__setattr__(self, "label_overlap", label_overlap)
        object.__setattr__(self, "tick_count", tick_count)
        object.__setattr__(self, "tick_size", tick_size)
        object.__setattr__(self, "tick_values", tick_values)
        object.__setattr__(self, "title_font_size", title_font_size)
        object.__setattr__(self, "title_color", title_color)
        object.__setattr__(self, "title_padding", title_padding)
        object.__setattr__(self, "domain", domain)
        object.__setattr__(self, "domain_color", domain_color)
        object.__setattr__(self, "domain_width", domain_width)
        object.__setattr__(self, "grid", grid)
        object.__setattr__(self, "grid_color", grid_color)
        object.__setattr__(self, "grid_dash", grid_dash)
        object.__setattr__(self, "grid_width", grid_width)
        object.__setattr__(self, "domain_min", domain_min)
        object.__setattr__(self, "domain_max", domain_max)
        object.__setattr__(self, "nice", nice)
        object.__setattr__(self, "zero", zero)
        object.__setattr__(self, "label_padding", label_padding)
        object.__setattr__(self, "grid_opacity", grid_opacity)
        object.__setattr__(self, "orient", orient)
        object.__setattr__(self, "translate", translate)
        object.__setattr__(self, "min_band", min_band)
        object.__setattr__(self, "max_band", max_band)
        object.__setattr__(self, "tick_extra", tick_extra)
        object.__setattr__(self, "tick_min_step", tick_min_step)
        object.__setattr__(self, "title_orient", title_orient)
        object.__setattr__(self, "zindex", zindex)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict, omitting None values.

        ``label_format`` preset names are resolved to their d3-format/strftime
        strings via :func:`ferrum.format_presets.resolve_format_field` before
        serialization (NF-B1) so the Rust side receives a ready-to-use format
        spec; ``__init__`` already rejected any unrecognized name at
        construction, so resolution here always succeeds. The resolved
        ``format_type`` is threaded onto the wire as ``label_format_type``.
        The deprecated, no-op ``x``/``y`` flags are never emitted — the wire
        schema does not accept them.
        """
        d = _to_dict_omit_none(self)
        d.pop("x", None)
        d.pop("y", None)
        # `title` follows the SAME three-way contract as the per-channel
        # `fm.Axis(title=...)` surface, through the same single implementation
        # (`_title_sentinel.serialize_title`): omitted -> key absent (Rust
        # falls back to the field name); explicit `None` -> `""` (suppress);
        # a string -> verbatim. `_to_dict_omit_none` drops it above only when
        # it is `_UNSET`-or-`None`, so re-resolve it here rather than letting
        # the omit-None rule swallow the explicit suppression.
        d.pop("title", None)
        title = serialize_title(self.title)
        if title is not None:
            d["title"] = title
        if "label_format" in d:
            from ferrum.format_presets import resolve_format_field

            # AxisConfig has no user-facing label_format_type field; the
            # preset-derived type is always what gets threaded (label_format
            # is preset-names-only by contract, enforced in __init__).
            spec, format_type = resolve_format_field(d["label_format"], None)
            if spec is not None:
                d["label_format"] = spec
            if format_type is not None:
                d["label_format_type"] = format_type
        return d


@dataclass(frozen=True)
class LegendConfig:
    """Chart-level legend configuration.

    Parameters
    ----------
    orient : str, optional
        Legend position: ``"right"``, ``"left"``, ``"top"``, ``"bottom"``, or ``"none"``.
    direction : str, optional
        Layout direction: ``"vertical"`` or ``"horizontal"``.
    columns : int, optional
        Number of columns for multi-column layout.
    title_font_size : float, optional
        Legend title font size.
    label_font_size : float, optional
        Legend label font size.
    label_color : str, optional
        Legend label color.
    label_limit : float, optional
        Maximum legend-label width in pixels; wider labels are truncated with
        an ellipsis.
    symbol_size : float, optional
        Symbol size.
    symbol_stroke_width : float, optional
        Stroke width of legend symbol swatches in pixels.
    symbol_type : str, optional
        Symbol shape type.
    gradient_length : float, optional
        Gradient legend length.
    gradient_thickness : float, optional
        Gradient legend thickness in pixels.
    title_padding : float, optional
        Padding between the legend title and its entries.
    row_padding : float, optional
        Vertical entry spacing in pixels (vertical-direction legends).
    column_padding : float, optional
        Horizontal entry spacing in pixels (horizontal-direction legends).
    clip_height : float, optional
        Cap on the legend group height in pixels; overflow is hard-clipped.
    tick_min_step : float, optional
        Minimum step between colorbar ticks in data units.
    offset : float, optional
        Offset from the plot area.
    padding : float, optional
        Internal padding.
    zindex : int, optional
        Coarse draw order of the legend relative to marks.
    """

    orient: str | None = None
    direction: str | None = None
    columns: int | None = None
    title_font_size: float | None = None
    label_font_size: float | None = None
    label_color: str | None = None
    label_limit: float | None = None
    symbol_size: float | None = None
    symbol_stroke_width: float | None = None
    symbol_type: str | None = None
    gradient_length: float | None = None
    gradient_thickness: float | None = None
    title_padding: float | None = None
    row_padding: float | None = None
    column_padding: float | None = None
    clip_height: float | None = None
    tick_min_step: float | None = None
    offset: float | None = None
    padding: float | None = None
    zindex: int | None = None

    def __post_init__(self) -> None:
        # is_unspecified (not a bare `is not None` check) so this surface
        # shares its "not specified" gate with Legend's — a bare-None LegendConfig
        # field has no separate _UNSET state (its default already IS None), so
        # this reduces to the same check, but sharing the function is what
        # keeps the three orient/direction surfaces (Legend, LegendConfig, the
        # raw legend dict) from re-deriving the None policy independently.
        if not is_unspecified(self.orient):
            validate_legend_orient("LegendConfig.orient", self.orient)
        if not is_unspecified(self.direction):
            validate_legend_direction("LegendConfig.direction", self.direction)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict, omitting None values."""
        return _to_dict_omit_none(self)


@dataclass(frozen=True)
class TitleConfig:
    """Chart-level title configuration.

    Parameters
    ----------
    font_size : float, optional
        Title font size.
    font_weight : str, optional
        Title font weight (e.g. ``"bold"``, ``"600"``).
    anchor : str, optional
        Horizontal anchor: ``"start"``, ``"middle"``, or ``"end"``.
    color : str, optional
        Title color.
    offset : float, optional
        Pixel offset from the plot area.
    subtitle_font_size : float, optional
        Subtitle font size.
    subtitle_color : str, optional
        Subtitle color.
    """

    font_size: float | None = None
    font_weight: str | None = None
    anchor: str | None = None
    color: str | None = None
    offset: float | None = None
    subtitle_font_size: float | None = None
    subtitle_color: str | None = None

    def __post_init__(self) -> None:
        if self.anchor is not None:
            validate_choice("TitleConfig.anchor", "anchor", self.anchor, _VALID_TITLE_ANCHORS)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict, omitting None values."""
        return _to_dict_omit_none(self)


def _validate_grid_style(owner: str, width: Any, opacity: Any) -> None:
    """Validate the numeric grid-style values shared by both grid classes.

    :class:`GridConfig` (both-axes shorthand) and :class:`GridAxisConfig`
    (per-axis) declare the same ``width``/``opacity`` pair, so they validate it
    through one body — the family was uneven for exactly one cycle, after
    ``GridConfig`` gained shape validation and this pair kept passing silently.

    Both halves delegate to :mod:`ferrum._validate`, which owns these leaf
    predicates: ``width`` carries the spec §4.7 pixel contract via
    :func:`~ferrum._validate.validate_pixel_value` (the same single authority
    ``PaddingConfig`` uses), and ``opacity`` carries the bounded-fraction
    contract via its sibling :func:`~ferrum._validate.validate_fraction_value`.
    The fraction check was briefly hand-rolled here, which put the identical
    numeric-and-finite predicate in two places inside one function body — the
    shape ``validate_pixel_value`` was consolidated to remove.

    ``color`` is deliberately NOT validated here: config-surface color-VALUE
    refusal is a separate decision (#107) and an explicit non-goal of this
    batch's spec (§3), whose gate is about KEYS. An unparseable color keeps
    the theme value, unchanged.
    """
    validate_pixel_value(f"{owner}.width", width)
    validate_fraction_value(f"{owner}.opacity", opacity)


@dataclass(frozen=True)
class GridAxisConfig:
    """One axis's own grid settings, for ``configure_grid(x=...)`` / ``y=``.

    The richer of the two spellings a per-axis grid slot accepts. A bare
    ``bool`` (``configure_grid(x=False)``) is the enable-only shorthand and
    stays supported unchanged; pass this instead when the axis also needs its
    own gridline style::

        chart.configure_grid(
            x=fm.GridAxisConfig(enabled=True, color="#eee", dash=[2, 2]),
            y=False,
        )

    Anything left ``None`` falls back to ``GridConfig``'s flat both-axes
    shorthand, and then to the theme.

    Parameters
    ----------
    enabled : bool, optional
        Draw gridlines for this axis.
    color : str, optional
        Gridline color for this axis.
    width : float, optional
        Gridline width in pixels for this axis.
    dash : list[float], optional
        Gridline dash pattern (on/off pixel lengths) for this axis.
    opacity : float, optional
        Gridline opacity (0--1) for this axis.
    """

    enabled: bool | None = None
    color: str | None = None
    width: float | None = None
    dash: list[float] | None = None
    opacity: float | None = None

    def __post_init__(self) -> None:
        """Validate the numeric style values (see :func:`_validate_grid_style`)."""
        _validate_grid_style("GridAxisConfig", self.width, self.opacity)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict, omitting None values."""
        return _to_dict_omit_none(self)


def _grid_axis_from_mapping(axis: str, mapping: Mapping) -> "GridAxisConfig":
    """Normalize a raw per-axis grid mapping into a :class:`GridAxisConfig`.

    One authority for the slot: the dataclass validates the values, and its
    own signature refuses an unknown key — restated here as a ``ValueError``
    in the ferrum voice, naming the axis and the accepted keys, rather than
    letting a bare ``TypeError: __init__() got an unexpected keyword
    argument`` escape.
    """
    try:
        return GridAxisConfig(**dict(mapping))
    except TypeError as exc:
        accepted = ", ".join(f.name for f in fields(GridAxisConfig))
        raise ValueError(
            f"GridConfig: {axis}={dict(mapping)!r} has an unknown grid key "
            f"({exc}); accepted: {accepted}"
        ) from None


@dataclass(frozen=True)
class GridConfig:
    """Chart-level grid configuration.

    Parameters
    ----------
    x : bool or GridAxisConfig, optional
        Show x-axis grid lines. Pass a :class:`GridAxisConfig` to give the x
        axis its own gridline style as well as its own enable flag.
    y : bool or GridAxisConfig, optional
        Show y-axis grid lines; same two spellings as ``x``.
    color : str, optional
        Grid line color for both axes; an axis's own ``color`` wins.
    width : float, optional
        Grid line width.
    dash : list[float], optional
        Grid line dash pattern.
    opacity : float, optional
        Grid line opacity.
    band_colors : list[str], optional
        Alternating band fill colors between grid lines.
    """

    x: "bool | GridAxisConfig | None" = None
    y: "bool | GridAxisConfig | None" = None
    color: str | None = None
    width: float | None = None
    dash: list[float] | None = None
    opacity: float | None = None
    band_colors: list[str] | None = None

    def __post_init__(self) -> None:
        """Refuse a wrong shape on the per-axis slots.

        ``x``/``y`` accept a bool (enable-only), a :class:`GridAxisConfig`
        (enable plus that axis's own style), or the raw mapping spelling of
        that same object. Anything else — a string, a number, a sequence —
        reached Rust as an untagged-enum mismatch whose message
        (``data did not match any variant of untagged enum Wire``) names
        neither the key, the axis, nor the accepted spellings. This is the
        Python boundary that owns the widened slot, so it is the boundary that
        should say so — matching :class:`TitleConfig`, :class:`PaddingConfig`
        and :class:`LegendConfig`, which all validate here.

        A mapping is accepted rather than refused because it is the raw
        spelling of a value ferrum already accepts (and the one
        ``.override(grid_x={...})`` naturally uses) — but it is NORMALIZED
        into a :class:`GridAxisConfig` rather than passed through, so the
        validation is a property of the SLOT and not of one spelling of it.
        Without that, the dict spelling silently skipped the value refusals
        the object spelling has (``{"width": -5}`` rendered;
        ``{"width": nan}`` reached the json serializer artifact this batch set
        out to eliminate). Normalizing also gets the unknown-key refusal for
        free — from the dataclass's own signature — and means ``to_dict``
        handles exactly one nested type, so any ``Mapping`` (not just
        ``dict``) serializes correctly.
        """
        _validate_grid_style("GridConfig", self.width, self.opacity)
        for axis in ("x", "y"):
            value = getattr(self, axis)
            if value is None or isinstance(value, (bool, GridAxisConfig)):
                continue
            if isinstance(value, Mapping):
                object.__setattr__(self, axis, _grid_axis_from_mapping(axis, value))
                continue
            raise ValueError(
                f"GridConfig: {axis}={value!r} is not a valid grid setting; "
                f"pass a bool to enable/disable the {axis}-axis grid, or a "
                f"GridAxisConfig (or its dict form) to also style it."
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict, omitting None values.

        ``x``/``y`` serialize as a bare bool when given one (the historical
        wire spelling every existing caller emits) and as a nested object
        otherwise. ``__post_init__`` has already normalized any mapping into a
        :class:`GridAxisConfig`, so exactly one nested type reaches here.
        """
        d = _to_dict_omit_none(self)
        for axis in ("x", "y"):
            value = d.get(axis)
            if isinstance(value, GridAxisConfig):
                d[axis] = value.to_dict()
        return d


@dataclass(frozen=True)
class PaddingConfig:
    """Chart-level padding configuration.

    Parameters
    ----------
    top : float, optional
        Top padding in pixels.
    right : float, optional
        Right padding in pixels.
    bottom : float, optional
        Bottom padding in pixels.
    left : float, optional
        Left padding in pixels.
    auto : bool, default False
        Opt-in, per-side auto-expand (D10, spec §4.7): a side still at its
        own default here (`None`) is expanded, when `auto=True`, to keep a
        continuous axis's edge-tick label or axis title from clipping past
        the rendered viewport edge. A side you *do* set is never touched by
        `auto` — an explicit value always wins on its own side. Does not
        affect ordinal/nominal axis labels (which never overhang the plot
        edge) or annotations; a y-axis capped tight enough via `max_band`
        can still leave its own label or title outside the canvas on that
        axis.
    """

    top: float | None = None
    right: float | None = None
    bottom: float | None = None
    left: float | None = None
    auto: bool = False

    def __post_init__(self) -> None:
        # Validate at construction time (NF-B5/B6/B7): each side must be a
        # finite, non-negative pixel value (spec §4.7's pixel contract).
        validate_pixel_value("padding.top", self.top)
        validate_pixel_value("padding.right", self.right)
        validate_pixel_value("padding.bottom", self.bottom)
        validate_pixel_value("padding.left", self.left)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict, omitting None values."""
        return _to_dict_omit_none(self)


@dataclass(frozen=True)
class ColorConfig:
    """Chart-level color scale configuration.

    Parameters
    ----------
    scheme : str, optional
        Named color scheme for categorical encodings.
    sequential_scheme : str, optional
        Named scheme for sequential (single-hue) encodings.
    diverging_scheme : str, optional
        Named scheme for diverging encodings.
    domain : list, optional
        Explicit scale domain values.
    range : list[str], optional
        Explicit list of color strings for the range.
    """

    scheme: str | None = None
    sequential_scheme: str | None = None
    diverging_scheme: str | None = None
    domain: list | None = None
    range: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict, omitting None values."""
        return _to_dict_omit_none(self)


@dataclass(frozen=True)
class Configure:
    """Container for chart-level configuration overrides.

    Each field maps to a specific configuration domain.  Unset fields
    (``None``) mean "use the chart/theme default" — only fields that are
    explicitly set are forwarded to the renderer.

    Parameters
    ----------
    axis : AxisConfig, optional
        Applies to all axes.
    axis_x : AxisConfig, optional
        Applies only to the x axis (overrides ``axis`` for x).
    axis_y : AxisConfig, optional
        Applies only to the y axis (overrides ``axis`` for y).
    axis_y2 : AxisConfig, optional
        Applies only to the secondary y axis.
    legend : LegendConfig, optional
        Legend appearance.
    title : TitleConfig, optional
        Chart title appearance.
    grid : GridConfig, optional
        Grid line appearance.
    padding : PaddingConfig, optional
        Plot-area padding.
    color : ColorConfig, optional
        Default color scale settings.
    """

    axis: AxisConfig | None = None
    axis_x: AxisConfig | None = None
    axis_y: AxisConfig | None = None
    axis_y2: AxisConfig | None = None
    legend: LegendConfig | None = None
    title: TitleConfig | None = None
    grid: GridConfig | None = None
    padding: PaddingConfig | None = None
    color: ColorConfig | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict, omitting None fields and recursing into sub-configs."""
        result: dict[str, Any] = {}
        for f in fields(self):
            val = getattr(self, f.name)
            if val is not None:
                result[f.name] = val.to_dict()
        return result
