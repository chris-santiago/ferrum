"""Registry, resolution, validation, and payload construction for ``Chart.override``.

``Chart.override(**kwargs)`` is a flat snake_case escape hatch onto the chart's
*presentation* spec (config dataclasses, encoding scales, mark style, coord,
``width``/``height``).  For most of those targets the Rust spec deserializers
silently drop unknown fields, so a misspelled path cannot fail-loud on the Rust
side; validation is therefore Python-side against a registry built **at import
from the live schemas** so the valid-leaf sets stay synchronized with the typed
surface, except for deprecated keys (``x``/``y`` on ``AxisConfig``) which are
intentionally excluded and refused with a typed error.

Encoding scales are the one target that DOES fail loud on the Rust side (the
F-L04-07 raw-dict scale-key gate refuses an unknown key naming the accepted
list).  The registry is therefore not a substitute for that gate but a mirror
of it, and it is built from the gate's own published table
(``ferrum._core.scale_accepted_keys``) rather than from Python attribute names
— when the two disagreed, an advertised override leaf refused at the wire with
no working spelling anywhere.  See :func:`_scale_leaves`.

This module is a pure transform: it resolves paths, validates a whole override
dict, and builds a plain :class:`OverridePayload` data object.  It never imports
``Chart``, touches the filesystem, renders, or mutates global state.  The render
consumer (a separate unit) deep-merges each payload piece at its injection point
and emits the deprecations as warnings.

Targets
-------
chart-config
    ``x_axis_``/``y_axis_``/``axis_``/``legend_``/``title_``/``grid_``/
    ``padding_``/``color_`` prefixes.  Valid leaves are the ``dataclasses.fields``
    of the matching :mod:`ferrum.configure` class, minus deprecated keys
    (``x``/``y`` on ``AxisConfig``) which are refused with
    :class:`~ferrum.exceptions.FerrumOverrideError`.  Folds into
    ``{target_key: {leaf: value}}``.
encoding-scale
    ``<channel>_scale_<leaf>`` for each scale-bearing encoding channel.  Valid
    leaves are the wire keys the scale-key gate accepts for at least one scale
    type, plus a snake_case spelling of each camelCase one
    (:data:`_SCALE_LEAF_WIRE_ALIASES`).  Folds into
    ``{channel: {"scale": {wire_leaf: value}}}`` — both spellings are accepted,
    the wire one is emitted.
mark
    ``mark_<leaf>``; valid leaves are ``ferrum.marks.base._VALID_MARK_KWARGS``.
    Folds into ``mark_style[leaf]``.
coord
    ``coord_<leaf>``; valid leaves are the coord-dataclass field names.  Folds
    into ``coord[leaf]``.
properties
    Exact keys ``width`` and ``height``.

Per-channel ``<channel>_axis_*`` / ``<channel>_legend_*`` (the opaque
``AxisSpec.extra`` / ``LegendSpec.extra`` maps) are intentionally **excluded** in
v1: they resolve as UNKNOWN and raise :class:`~ferrum.exceptions.FerrumOverrideError`.
Route those through the typed chart-config target (``x_axis_*`` / ``legend_*``).
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, fields
from enum import Enum
from typing import Any, NamedTuple

from ferrum._validate import validate_pixel_value
from ferrum.configure import (
    AxisConfig,
    ColorConfig,
    GridConfig,
    LegendConfig,
    PaddingConfig,
    TitleConfig,
)
from ferrum.coord import (
    CoordCartesian,
    CoordFixed,
    CoordGeo,
    CoordPolar,
)
from ferrum.encoding import _channel_class_map
from ferrum.exceptions import FerrumOverrideError
from ferrum.marks.base import _VALID_MARK_KWARGS


class Target(Enum):
    """The presentation-spec subsystem an override path routes into."""

    CHART_CONFIG = "chart-config"
    ENCODING_SCALE = "encoding-scale"
    MARK = "mark"
    COORD = "coord"
    PROPERTIES = "properties"


class SpecLocation(NamedTuple):
    """Where a resolved override writes in the assembled spec.

    Parameters
    ----------
    target_key:
        The top-level key within the target.  For chart-config this is the
        config dict key (``"axis_x"``, ``"legend"``, …); for encoding-scale it is
        the channel name (``"x"``, ``"color"``, …); for properties it is the
        property name (``"width"``/``"height"``).  ``None`` for the mark target,
        which has no intermediate key (leaves fold directly into ``mark_style``).
    leaf:
        The terminal field name written under ``target_key`` (e.g.
        ``"label_angle"``, ``"domain"``, ``"corner_radius"``), as the user
        spelled it.  For the properties target this equals ``target_key``
        (``"width"``/``"height"``).  For encoding-scale leaves this is the
        registry spelling, which :func:`build_payload` translates to the wire
        spelling through :data:`_SCALE_LEAF_WIRE_ALIASES` before folding.
    """

    target_key: str | None
    leaf: str


@dataclass(frozen=True)
class ResolvedPath:
    """The registry entry a flat override path resolves to.

    Parameters
    ----------
    target:
        Which presentation subsystem the path routes into.
    location:
        The spec target-key + leaf the value is written to.
    typed_equivalent:
        A human-readable descriptor of the typed ``configure_*`` / ``properties``
        method that supersedes this path (e.g. ``".configure_axis(label_angle=...)"``),
        or ``None`` for genuine escape-hatch paths with no typed surface.
    """

    target: Target
    location: SpecLocation
    typed_equivalent: str | None
    #: See :attr:`_PrefixRule.config_cls`.
    config_cls: type | None = None


@dataclass(frozen=True)
class OverridePayload:
    """The pure result of resolving + validating an override dict.

    Each field is a ready-to-merge piece for the render consumer.  The consumer
    deep-merges ``chart_config`` into the resolved chart-config dict, ``encoding``
    into the assembled encoding specs, ``mark_style`` into the mark-style dict,
    applies ``properties`` to the render-time dimensions, and emits one
    ``DeprecationWarning`` per ``deprecations`` entry.

    Parameters
    ----------
    chart_config:
        ``{target_key: {leaf: value}}`` for chart-config targets, e.g.
        ``{"axis_x": {"label_angle": -45}, "color": {"scheme": "viridis"}}``.
    encoding:
        ``{channel: {"scale": {wire_key: value}}}`` for encoding-scale targets,
        e.g. ``{"x": {"scale": {"domain": [0, 10]}}}``.  Keys are always the
        gate's wire spelling, so ``x_scale_padding_inner`` and
        ``x_scale_paddingInner`` both fold to ``{"paddingInner": ...}``.
    mark_style:
        ``{leaf: value}`` for mark-style targets, e.g. ``{"corner_radius": 4}``.
    coord:
        ``{leaf: value}`` for coord targets, e.g. ``{"clip": False}``.  The
        consumer merges these into the assembled ``coord`` spec.
    properties:
        A subset of ``{"width": value, "height": value}``.
    deprecations:
        ``(path, typed_equivalent_descriptor)`` pairs for every override path that
        has a typed equivalent.  The consumer emits one ``DeprecationWarning`` per
        pair before applying.
    """

    chart_config: dict[str, dict[str, Any]]
    encoding: dict[str, dict[str, dict[str, Any]]]
    mark_style: dict[str, Any]
    coord: dict[str, Any]
    properties: dict[str, Any]
    deprecations: list[tuple[str, str]]


# ---------------------------------------------------------------------------
# Leaf-set introspection (no hand-maintained lists)
# ---------------------------------------------------------------------------

# Deprecated AxisConfig leaf names (x/y) that are excluded from the override registry.
# Shared by both the registry exclusion and the deprecated-path-map builder to prevent drift.
_DEPRECATED_AXIS_LEAVES = frozenset({"x", "y"})

# Non-pixel PaddingConfig leaves: excluded from the numeric pixel-contract
# guard below. ``auto`` is a bool, not a pixel value, and must pass through
# ``.override(padding_auto=...)`` untouched.
_PADDING_NON_NUMERIC_LEAVES = frozenset({"auto"})


def _config_leaves(config_cls: type) -> frozenset[str]:
    """Return the valid override leaves for a :mod:`ferrum.configure` dataclass."""
    return frozenset(f.name for f in fields(config_cls))


# The numeric-pixel PaddingConfig leaves that carry the spec §4.7 pixel
# contract, derived from PaddingConfig's own fields (minus the non-pixel
# leaves above) so a future pixel-valued field is validated automatically
# instead of silently passing through unchecked.
_PADDING_NUMERIC_LEAVES = _config_leaves(PaddingConfig) - _PADDING_NON_NUMERIC_LEAVES


def _coord_leaves() -> frozenset[str]:
    """Return the union of field names across the coord dataclasses.

    ``CoordFlip`` carries no fields (it serializes to a bare token); the
    remaining coord types expose their tunable options as dataclass fields.
    """
    leaves: set[str] = set()
    for coord_cls in (CoordCartesian, CoordFixed, CoordPolar, CoordGeo):
        leaves.update(f.name for f in fields(coord_cls))
    return frozenset(leaves)


# Every ``ScaleSpec`` ``"type"`` tag, i.e. every argument
# ``ferrum._core.scale_accepted_keys`` answers for. Only the TAGS are listed
# here; each tag's accepted-key set is asked of the gate, never mirrored. A tag
# that stopped being a real variant fails ``_scale_leaves()`` loudly at import
# (``scale_accepted_keys`` raises for an unknown tag), and a tag missing from
# this tuple is caught by ``tests/test_override.py``'s drift test, which checks
# it against the scale classes' own emitted tags.
_SCALE_TYPE_TAGS: tuple[str, ...] = (
    "linear",
    "log",
    "time",
    "symlog",
    "pow",
    "sqrt",
    "utc",
    "ordinal",
    "band",
    "point",
    "sequential",
    "diverging",
    "quantize",
    "quantile",
    "threshold",
    "bin-ordinal",
)

#: snake_case override spellings for the camelCase wire keys, so
#: ``.override(x_scale_padding_inner=0.3)`` reads like every other flat
#: snake_case override path while still emitting the spelling the wire gate
#: accepts. Both spellings resolve; :func:`build_payload` emits the wire one.
#: An entry whose wire key no scale type accepts is dropped from the registry
#: by :func:`_scale_leaves` — an override path that cannot reach the wire is
#: worse than no path at all, because it advertises a spelling the gate then
#: refuses.
_SCALE_LEAF_WIRE_ALIASES: dict[str, str] = {
    "padding_inner": "paddingInner",
    "padding_outer": "paddingOuter",
    "domain_mid": "domainMid",
    "domain_param": "domainParam",
}


def _scale_leaves() -> frozenset[str]:
    """Return the valid ``<channel>_scale_<leaf>`` leaf names.

    Derived from the wire gate's own per-type accepted-key table
    (``ferrum._core.scale_accepted_keys``, published from
    ``crates/ferrum-core/src/spec/encoding.rs``), unioned across every scale
    type, plus ``type`` (the scale-kind discriminator the user sets via
    override) and the snake_case aliases in :data:`_SCALE_LEAF_WIRE_ALIASES`.

    This asks the gate rather than introspecting the ``*Scale`` pyclasses'
    attributes, which is what the registry used to do.  Those attributes are
    Python spellings of a union across all types, and the gate enforces wire
    spellings per type, so the two sets diverged: five advertised leaves
    (``padding_inner``, ``padding_outer``, ``domain_mid``, ``quantiles``,
    ``utc``) resolved here and were then refused at the wire with no working
    spelling, while seven wire keys (``nice``, ``zero``, ``stops``,
    ``domainParam``, ``paddingInner``, ``paddingOuter``, ``domainMid``) had no
    override path at all.  ``quantiles`` and ``utc`` are gone entirely: neither
    names a wire key of any scale type (``quantiles`` is
    ``QuantileScale``'s computed thresholds, a read-only getter; ``utc`` is a
    ``TimeScale`` constructor flag that selects the ``"utc"`` *type tag*, so
    ``.override(<channel>_scale_type="utc")`` is its real spelling), and they
    now refuse with this registry's own "Unknown override path" error instead
    of an opaque wire refusal.
    """
    from ferrum._core import scale_accepted_keys  # type: ignore[attr-defined]

    wire: set[str] = {"type"}
    for scale_type in _SCALE_TYPE_TAGS:
        wire.update(scale_accepted_keys(scale_type))
    aliases = {snake for snake, key in _SCALE_LEAF_WIRE_ALIASES.items() if key in wire}
    return frozenset(wire | aliases)


def _scale_bearing_channels() -> tuple[str, ...]:
    """Return the encoding channel names whose ``scale=`` option is honored."""
    return tuple(
        name for name, cls in _channel_class_map().items() if "scale" in cls._honored_kwargs
    )


# ---------------------------------------------------------------------------
# Registry (built once at import from the live schemas)
# ---------------------------------------------------------------------------


class _PrefixRule(NamedTuple):
    """A registry prefix → target binding.

    Parameters
    ----------
    prefix:
        The flat-path prefix (e.g. ``"x_axis_"``, ``"mark_"``).  Longer prefixes
        are matched first so ``x_axis_`` binds before ``axis_``.
    target:
        The presentation subsystem the matched paths route into.
    target_key:
        The spec target-key shared by every leaf under this prefix (the config
        dict key for chart-config, the channel name for encoding-scale).  ``None``
        for the mark target, whose leaves fold directly into ``mark_style``.
    valid_leaves:
        The set of leaf names accepted after the prefix.
    typed_equivalent:
        A format string with a single ``{leaf}`` placeholder describing the typed
        method that supersedes paths under this prefix, or ``None`` when the prefix
        has no typed equivalent.
    """

    prefix: str
    target: Target
    target_key: str | None
    valid_leaves: frozenset[str]
    typed_equivalent: str | None
    #: The ``ferrum.configure`` dataclass these leaves were derived from, and
    #: therefore the single authority on what each leaf ACCEPTS and how it
    #: SERIALIZES. ``None`` for prefixes whose leaves come from elsewhere
    #: (encoding scales, mark kwargs, coord dataclasses) — those keep the
    #: raw-value path.
    config_cls: type | None = None


def _build_chart_config_rules() -> list[_PrefixRule]:
    """Build the chart-config prefix rules, longest-prefix-first within axis."""
    # ``x``/``y`` are still ``AxisConfig`` dataclass fields (the deprecated,
    # no-op show/hide flags — see ``AxisConfig.x``/``.y``), but
    # ``AxisConfig.to_dict()`` never emits them and the Rust wire schema does
    # not accept them. Excluded here so a deprecated ``axis_x=...`` /
    # ``axis_y=...`` override spelling refuses at this Python boundary with
    # the standard "Unknown override path" error (naming the kwarg), instead
    # of resolving and then hard-failing as an opaque wire-gate ValueError.
    axis_leaves = _config_leaves(AxisConfig) - _DEPRECATED_AXIS_LEAVES
    # ``x_axis_`` / ``y_axis_`` MUST be ordered before ``axis_`` so the longest
    # prefix wins (``x_axis_grid_color`` is not mis-split as ``axis_…``).
    return [
        _PrefixRule(
            "x_axis_",
            Target.CHART_CONFIG,
            "axis_x",
            axis_leaves,
            ".configure_axis({leaf}=...)",
            config_cls=AxisConfig,
        ),
        _PrefixRule(
            "y_axis_",
            Target.CHART_CONFIG,
            "axis_y",
            axis_leaves,
            ".configure_axis({leaf}=...)",
            config_cls=AxisConfig,
        ),
        _PrefixRule(
            "axis_",
            Target.CHART_CONFIG,
            "axis",
            axis_leaves,
            ".configure_axis({leaf}=...)",
            config_cls=AxisConfig,
        ),
        _PrefixRule(
            "legend_",
            Target.CHART_CONFIG,
            "legend",
            _config_leaves(LegendConfig),
            ".configure_legend({leaf}=...)",
            config_cls=LegendConfig,
        ),
        _PrefixRule(
            "title_",
            Target.CHART_CONFIG,
            "title",
            _config_leaves(TitleConfig),
            ".configure_title({leaf}=...)",
            config_cls=TitleConfig,
        ),
        _PrefixRule(
            "grid_",
            Target.CHART_CONFIG,
            "grid",
            _config_leaves(GridConfig),
            ".configure_grid({leaf}=...)",
            config_cls=GridConfig,
        ),
        _PrefixRule(
            "padding_",
            Target.CHART_CONFIG,
            "padding",
            _config_leaves(PaddingConfig),
            ".configure_padding({leaf}=...)",
            config_cls=PaddingConfig,
        ),
        _PrefixRule(
            "color_",
            Target.CHART_CONFIG,
            "color",
            _config_leaves(ColorConfig),
            ".configure_color({leaf}=...)",
            config_cls=ColorConfig,
        ),
    ]


def _build_prefix_rules() -> list[_PrefixRule]:
    """Build the full prefix-rule list, ordered longest-prefix-first.

    Encoding-scale rules (``<channel>_scale_``) precede chart-config so that a
    channel-scoped scale path (``x_scale_domain``) binds to the scale target,
    never to the ``x_axis_`` / ``axis_`` chart-config prefixes.  ``mark_`` and
    ``coord_`` bind their own targets.
    """
    scale_leaves = _scale_leaves()
    rules: list[_PrefixRule] = []
    for channel in _scale_bearing_channels():
        rules.append(
            _PrefixRule(
                f"{channel}_scale_",
                Target.ENCODING_SCALE,
                channel,
                scale_leaves,
                None,
            )
        )
    rules.extend(_build_chart_config_rules())
    rules.append(_PrefixRule("mark_", Target.MARK, None, _VALID_MARK_KWARGS, None))
    rules.append(_PrefixRule("coord_", Target.COORD, "coord", _coord_leaves(), None))
    # Longest prefix first guarantees ``x_axis_`` beats ``axis_`` and
    # ``x_scale_`` beats ``x_axis_`` regardless of insertion order.
    rules.sort(key=lambda r: len(r.prefix), reverse=True)
    return rules


# Exact-path properties (no prefix; the whole key is the leaf).
_PROPERTY_PATHS: dict[str, str] = {
    "width": ".properties(width=...)",
    "height": ".properties(height=...)",
}

_PREFIX_RULES: tuple[_PrefixRule, ...] = tuple(_build_prefix_rules())


def _all_known_paths() -> frozenset[str]:
    """Return every fully-enumerated valid override path (for did-you-mean)."""
    paths: set[str] = set(_PROPERTY_PATHS)
    for rule in _PREFIX_RULES:
        for leaf in rule.valid_leaves:
            paths.add(f"{rule.prefix}{leaf}")
    return frozenset(paths)


_KNOWN_PATHS: frozenset[str] = _all_known_paths()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def resolve(path: str) -> ResolvedPath | None:
    """Resolve a flat override *path* to its registry entry, or ``None``.

    Resolution is longest-prefix-first against the enumerated registry, so prefix
    ambiguity is structurally impossible (``x_axis_`` binds before ``axis_``;
    ``x_scale_`` binds before ``x_axis_``).  A path resolves only when its prefix
    is known **and** its leaf is a valid field of that prefix's target; an unknown
    prefix or a known prefix with an invalid leaf returns ``None`` (the caller
    raises :class:`~ferrum.exceptions.FerrumOverrideError`).

    Parameters
    ----------
    path:
        A flat snake_case override key (e.g. ``"x_axis_label_angle"``, ``"width"``).

    Returns
    -------
    ResolvedPath or None
        The resolved registry entry, or ``None`` when the path is unknown.
    """
    if path in _PROPERTY_PATHS:
        return ResolvedPath(
            target=Target.PROPERTIES,
            location=SpecLocation(target_key=path, leaf=path),
            typed_equivalent=_PROPERTY_PATHS[path],
        )
    for rule in _PREFIX_RULES:
        if not path.startswith(rule.prefix):
            continue
        leaf = path[len(rule.prefix) :]
        if leaf not in rule.valid_leaves:
            return None
        typed = rule.typed_equivalent.format(leaf=leaf) if rule.typed_equivalent else None
        return ResolvedPath(
            target=rule.target,
            location=SpecLocation(target_key=rule.target_key, leaf=leaf),
            typed_equivalent=typed,
            config_cls=rule.config_cls,
        )
    return None


def _build_deprecated_paths() -> dict[str, str]:
    """Build deprecated override paths from the axis-leaves exclusion set.

    The ``AxisConfig`` dataclass has ``x`` and ``y`` fields, but they are
    deprecated no-ops. The three axis-config prefixes (``x_axis_``, ``y_axis_``,
    ``axis_``) all use the same ``axis_leaves`` set, which excludes these keys.
    This function derives all six possible deprecated spellings (3 prefixes × 2
    excluded leaves) and maps each to its documented replacement, so users get
    a specific error message instead of a misdirecting difflib suggestion.
    """
    # The three axis-config prefixes that use the excluded leaves.
    axis_prefixes = ["x_axis_", "y_axis_", "axis_"]
    # The documented replacement for each deprecated leaf.
    replacements = {
        "x": "Chart.axis(x=False)",
        "y": "Chart.axis(y=False)",
    }

    paths = {}
    for prefix in axis_prefixes:
        for leaf in _DEPRECATED_AXIS_LEAVES:
            path = f"{prefix}{leaf}"
            paths[path] = replacements[leaf]
    return paths


# Deprecated override path spellings that have documented replacements.
# Derived from the axis-leaves exclusion set and axis-config prefixes to ensure
# all six spellings (3 prefixes × 2 excluded leaves) get accurate hints.
_DEPRECATED_PATHS: dict[str, str] = _build_deprecated_paths()


def _suggestion(path: str) -> str:
    """Return a '` Did you mean: 'x'?`' suffix when a close known path exists."""
    matches = difflib.get_close_matches(path, _KNOWN_PATHS, n=1)
    if matches:
        return f" Did you mean: {matches[0]!r}?"
    return ""


def validate(overrides: dict[str, Any]) -> None:
    """Validate a whole override dict, raising on the first unresolvable path.

    Each key is resolved through the registry.  An unknown prefix, or a known
    prefix with a leaf that is not a valid field of its target, raises
    :class:`~ferrum.exceptions.FerrumOverrideError` naming the offending path.
    Deprecated paths with documented replacements raise with a specific message
    naming the replacement (e.g., ``x_axis_x`` → ``Chart.axis(x=False)``).
    For other unknown paths, a suggestion is offered when a close match exists
    among the known paths via :func:`difflib.get_close_matches`.
    Keys are checked in iteration order and the error is raised on the first
    failure (fail-fast).

    Also validates the numeric PaddingConfig leaves (``top``/``right``/
    ``bottom``/``left``): they must be finite, non-negative pixel values per
    spec §4.7's pixel contract (:func:`ferrum._validate.validate_pixel_value`,
    the same predicate ``PaddingConfig.__post_init__`` runs). ``padding_auto``
    (a bool leaf) is not part of the pixel contract and passes through
    unvalidated here.

    Parameters
    ----------
    overrides:
        The flat override dict stored by ``Chart.override``.

    Raises
    ------
    FerrumOverrideError
        When any path does not resolve.
    ValueError
        When a padding side value violates the pixel contract.
    """
    for path, value in overrides.items():
        resolved = resolve(path)
        if resolved is None:
            # Check if this is a known deprecated path with a documented replacement.
            if path in _DEPRECATED_PATHS:
                replacement = _DEPRECATED_PATHS[path]
                raise FerrumOverrideError(
                    f"Unknown override path {path!r} (deprecated). Use {replacement} instead."
                )
            raise FerrumOverrideError(f"Unknown override path {path!r}.{_suggestion(path)}")

        # Validate padding-side values (NF-B5/B6/B7): numeric, finite,
        # non-negative. Narrowed to exactly the numeric sides; "padding_auto"
        # (bool) is a valid, registered leaf and must not be routed through
        # the pixel-value validator.
        if (
            resolved.location.target_key == "padding"
            and resolved.location.leaf in _PADDING_NUMERIC_LEAVES
        ):
            validate_pixel_value(f"override {path!r}", value)


def _chart_config_wire_fragment(config_cls: type | None, leaves: dict[str, Any]) -> dict[str, Any]:
    """Validate and serialize a chart-config section's override leaves, via its owner.

    An override leaf is the *deprecated spelling of the same request* its
    advertised typed equivalent expresses — the registry says so itself
    (``.configure_grid({leaf}=...)``). So the leaves named for one section
    (e.g. every ``x_axis_*`` key in one ``.override(...)`` call) must accept
    exactly what the equivalent ``configure_axis(...)`` call accepts, refuse
    exactly what it refuses, and reach the wire in exactly the same shape.
    The way to guarantee all three is to let the owning dataclass answer,
    rather than to re-implement its answers here: construct it with every
    leaf named for this section **at once** and take what ``to_dict()``
    emits.

    Constructing with the combined set — not one leaf at a time — is what
    makes the dataclass the single validation authority per *section*, not
    merely per leaf. A single-leaf construction can only run validators that
    look at one field; ``AxisConfig``'s cross-field validators
    (``label_format``/``label_format_raw`` mutual exclusion,
    ``_validate_domain_bounds``'s ``domain_min``/``domain_max`` pair) need
    both fields present in the same call to fire at all, so
    ``override(axis_label_format="percent", axis_label_format_raw=",.2f")``
    previously built each leaf in isolation, silently combined two
    individually-valid fragments into a combination the typed surface
    refuses, and let Rust's raw-first precedence pick a winner no
    construction-time check ever saw. Building one instance from the whole
    section closes that gap the same way the single-leaf construction closed
    the original three: ``override(grid_x="nonsense")`` (was serde's
    untagged-enum message), ``override(x_axis_domain_min=nan)`` (was a json
    serializer artifact), and ``override(axis_title=None)`` (was a no-op).

    Returning a fragment rather than a value dict as-is is load-bearing: a
    leaf can serialize to more than one wire key — ``label_format`` resolves
    a preset and emits ``label_format_type`` alongside it — and taking only
    the named leaves would drop the companion and mis-classify the format.

    The empty-construction baseline is subtracted for the mirror-image
    reason. ``to_dict()`` emits every non-``None`` field, including the
    class's own DEFAULTS, so an instance carries keys the caller never named:
    ``PaddingConfig`` declares ``auto: bool = False``, so
    ``PaddingConfig(left=70).to_dict()`` is ``{"auto": False, "left": 70.0}``.
    Merging that fragment would inject an unsolicited ``auto`` onto the wire.
    Building the section from one combined construction call also retires
    the kwarg-order hazard the single-leaf version had to reason about
    separately: ``override(padding_auto=True, padding_left=70)`` and the
    reversed spelling both become one ``PaddingConfig(auto=True, left=70)``
    call regardless of which key the caller wrote first, so there is no
    order for the two spellings to disagree on. Dropping any key the section
    did not name that a default-constructed instance also emits keeps
    derived companions (absent from the baseline) and drops injected
    defaults (present in it), so a fragment can only contain keys the named
    leaves themselves produced.

    ``config_cls is None`` (encoding-scale, mark and coord prefixes, whose
    leaves are not owned by a ``ferrum.configure`` dataclass) keeps the raw
    multi-key behavior; those surfaces have their own validation stories and
    are outside the class this closes.
    """
    if config_cls is None:
        return dict(leaves)
    fragment = config_cls(**leaves).to_dict()
    baseline = config_cls().to_dict()
    return {k: v for k, v in fragment.items() if k in leaves or k not in baseline}


def build_payload(overrides: dict[str, Any]) -> OverridePayload:
    """Validate *overrides* and build the ready-to-merge :class:`OverridePayload`.

    Calls :func:`validate` first (a bad dict raises before any payload is built),
    then routes each path by target into the corresponding payload piece.  This is
    a pure transform: it constructs and returns a plain data object and touches no
    ``Chart``.  Paths with a typed equivalent are recorded in ``deprecations`` so
    the consumer can warn before applying.

    Chart-config leaves are grouped by section (``target_key``) before the
    owning dataclass is constructed, so every leaf named for one section in
    this call — e.g. both ``axis_label_format`` and ``axis_label_format_raw``
    — reaches the owning ``configure.py`` dataclass in a **single** combined
    construction. That is what lets cross-field validators (mutual
    exclusion, paired-bounds checks) fire on the override spelling exactly
    as they do on the typed ``configure_*(...)`` surface; see
    :func:`_chart_config_wire_fragment`.

    Parameters
    ----------
    overrides:
        The flat override dict stored by ``Chart.override``.

    Returns
    -------
    OverridePayload
        The merged-by-target payload pieces plus the deprecation list.

    Raises
    ------
    FerrumOverrideError
        When any path does not resolve (propagated from :func:`validate`).
    """
    validate(overrides)

    chart_config_leaves: dict[str, dict[str, Any]] = {}
    chart_config_classes: dict[str, type | None] = {}
    encoding: dict[str, dict[str, dict[str, Any]]] = {}
    mark_style: dict[str, Any] = {}
    coord: dict[str, Any] = {}
    properties: dict[str, Any] = {}
    deprecations: list[tuple[str, str]] = []

    for path, value in overrides.items():
        resolved = resolve(path)
        # validate() guarantees resolve() is non-None here.
        assert resolved is not None
        loc = resolved.location

        if resolved.target is Target.CHART_CONFIG:
            assert loc.target_key is not None
            # Collect every leaf for this section first; the owning
            # dataclass is constructed once, from the whole section, after
            # the loop (see `_chart_config_wire_fragment`).
            chart_config_leaves.setdefault(loc.target_key, {})[loc.leaf] = value
            chart_config_classes[loc.target_key] = resolved.config_cls
        elif resolved.target is Target.ENCODING_SCALE:
            assert loc.target_key is not None
            channel = encoding.setdefault(loc.target_key, {})
            # Emit the WIRE spelling: the scale-key gate accepts
            # ``paddingInner``, the flat override surface reads
            # ``padding_inner``, and both must land on the same wire key.
            wire_leaf = _SCALE_LEAF_WIRE_ALIASES.get(loc.leaf, loc.leaf)
            channel.setdefault("scale", {})[wire_leaf] = value
        elif resolved.target is Target.MARK:
            mark_style[loc.leaf] = value
        elif resolved.target is Target.COORD:
            coord[loc.leaf] = value
        elif resolved.target is Target.PROPERTIES:
            properties[loc.leaf] = value

        if resolved.typed_equivalent is not None:
            deprecations.append((path, resolved.typed_equivalent))

    chart_config = {
        target_key: _chart_config_wire_fragment(chart_config_classes[target_key], leaves)
        for target_key, leaves in chart_config_leaves.items()
    }

    return OverridePayload(
        chart_config=chart_config,
        encoding=encoding,
        mark_style=mark_style,
        coord=coord,
        properties=properties,
        deprecations=deprecations,
    )
