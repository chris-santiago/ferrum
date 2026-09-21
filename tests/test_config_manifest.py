"""Python twin of the chart-config disposition manifest (spec §6, NF-B13).

The Rust half — the checked-in ``{field: {honored, reason}}`` disposition
table at ``crates/ferrum-core/src/render/chart_config_manifest.json``,
generated and consumed by a test in
``crates/ferrum-core/src/render/chart_config.rs`` (Task 1, closed
2026-09-02) — enumerates every serde field of the wire config schema. This
module carries three legs:

1. **Python-vs-Python parity** (the original leg): each chart-level config
   dataclass in ``configure.py`` is paired with the
   ``ConfigureMixin.configure_*`` method (``_configure_mixin.py``) that
   builds it; every dataclass field must have a same-named method kwarg and
   vice versa, so a field added to one Python surface without the other
   never silently loses data on whichever surface a caller happens to use.
2. **Rust-schema-vs-Python leg** (added in the 2026-09-02 fix round, now
   that Task 1's manifest file exists and is readable): every manifest
   field must be reachable from a Python surface (a ``configure.py``
   dataclass field, or its ``_configure_mixin.py`` kwarg — the two are
   already proven identical by leg 1) or carry an explicit, justified
   allowlist entry. Before this leg, a schema field absent from BOTH Python
   surfaces passed leg 1 silently (nothing to compare against nothing) and
   passed Task 1's Rust-side manifest test too (that test only checks the
   manifest is complete relative to the Rust struct fields, not relative to
   Python) — see the spec-reviewer's ``cannot_verify[0]`` and the
   quality-reviewer's ``lossiness[1]`` findings on task 2, cycle 2.
3. **``object.__setattr__`` mirror completeness** (2026-09-03, design-review
   remediation, finding F3): ``AxisConfig``/``Axis`` are
   ``dataclass(init=False)`` with hand-written ``__init__``s that mirror
   every field three times (field declaration, ``__init__`` parameter,
   ``object.__setattr__`` line); a dropped ``object.__setattr__`` line
   silently leaves that field at its class default. This leg mechanically
   derives the field list, the constructor's parameters, and the constructor
   source's ``object.__setattr__`` calls and asserts all three agree, so a
   missing mirror line fails loudly instead of silently. This is a cheap
   structural guard, not the full ``init=False`` refactor (a separate filed
   follow-up).
"""

from __future__ import annotations

import inspect
import json
import re
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import polars as pl
import pytest

import ferrum as fm
from ferrum._configure_mixin import ConfigureMixin
from ferrum.axis import Axis
from ferrum.configure import (
    AxisConfig,
    ColorConfig,
    Configure,
    GridAxisConfig,
    GridConfig,
    LegendConfig,
    PaddingConfig,
    TitleConfig,
)

# Each entry: (dataclass, mixin method, method-only kwargs, dataclass-only fields).
#
# - method-only kwargs: accepted by the mixin method but resolved into a
#   *different* canonical field before construction (deprecated aliases), so
#   they never appear as a dataclass field of their own.
# - dataclass-only fields: real dataclass fields the mixin method does not
#   expose, by documented design (not an omission) — e.g. AxisConfig.orient
#   applies to only one of x/y, so `configure_axis` (which applies to both)
#   intentionally omits it; callers use `configure(axis_x=...)` /
#   `configure(axis_y=...)` instead (see configure_axis's own docstring).
_MANIFEST = (
    (
        AxisConfig,
        ConfigureMixin.configure_axis,
        frozenset({"min_extent", "max_extent"}),
        frozenset({"orient"}),
    ),
    (LegendConfig, ConfigureMixin.configure_legend, frozenset(), frozenset()),
    (TitleConfig, ConfigureMixin.configure_title, frozenset(), frozenset()),
    (GridConfig, ConfigureMixin.configure_grid, frozenset(), frozenset()),
    (PaddingConfig, ConfigureMixin.configure_padding, frozenset(), frozenset()),
    (ColorConfig, ConfigureMixin.configure_color, frozenset(), frozenset()),
)


def _mirror_gaps(
    config_cls: type,
    method,
    method_only: frozenset[str],
    dataclass_only: frozenset[str],
) -> tuple[set[str], set[str]]:
    """Return the (missing_from_method, missing_from_dataclass) field-name gaps.

    Both sets are empty when the two surfaces mirror each other, modulo the
    documented exceptions passed in *method_only* / *dataclass_only*.
    """
    dataclass_fields = {f.name for f in fields(config_cls)}
    method_kwargs = {name for name in inspect.signature(method).parameters if name != "self"}
    missing_from_method = dataclass_fields - method_kwargs - dataclass_only
    missing_from_dataclass = method_kwargs - dataclass_fields - method_only
    return missing_from_method, missing_from_dataclass


@pytest.mark.parametrize(
    "config_cls,method,method_only,dataclass_only",
    _MANIFEST,
    ids=[cls.__name__ for cls, _, _, _ in _MANIFEST],
)
def test_configure_dataclass_mirrors_mixin_method(config_cls, method, method_only, dataclass_only):
    """Every config dataclass field has a same-named ``configure_*`` kwarg (or vice versa).

    A field present on one Python config surface but not the other is
    exactly the "accepted on one surface, silently lost on the other" shape
    NF-B13 closes.
    """
    missing_from_method, missing_from_dataclass = _mirror_gaps(
        config_cls, method, method_only, dataclass_only
    )
    assert not missing_from_method, (
        f"{config_cls.__name__} field(s) {sorted(missing_from_method)} have no "
        f"matching kwarg on ConfigureMixin.{method.__name__}"
    )
    assert not missing_from_dataclass, (
        f"ConfigureMixin.{method.__name__} kwarg(s) {sorted(missing_from_dataclass)} "
        f"have no matching {config_cls.__name__} field"
    )


def test_mirror_check_is_red_provable():
    """``_mirror_gaps`` discriminates rather than vacuously passing.

    A synthetic dataclass/method pair with a deliberately dropped field must
    surface as a gap — the RED proof the manifest twin's completeness claim
    rests on (acceptance criterion: "a schema field missing from the Python
    surface fails").
    """
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class _SyntheticConfig:
        known: int | None = None
        orphaned: int | None = None  # never reaches _synthetic_method below

    def _synthetic_method(self, *, known: int | None = None) -> None: ...

    missing_from_method, missing_from_dataclass = _mirror_gaps(
        _SyntheticConfig, _synthetic_method, frozenset(), frozenset()
    )
    assert missing_from_method == {"orphaned"}
    assert missing_from_dataclass == set()


# ---------------------------------------------------------------------------
# Leg 2 — Rust schema vs. Python (2026-09-02 fix round).
#
# Reads Task 1's disposition manifest and asserts the cross-language
# direction leg 1 above cannot: every field in the WIRE schema is reachable
# from a Python configure.py/_configure_mixin.py surface, or is named in
# `_EXPECTED_PYTHON_ABSENT` with a reason. `honored`/`reason` in the
# manifest are not consulted here — a field can be `honored: false` (no
# Rust consumer yet) and still need a Python-reachable name once its
# consumer lands; this leg only asks "does a caller have a way to name this
# field on a configure.py surface today", independent of whether Rust
# currently does anything with it.
# ---------------------------------------------------------------------------

_MANIFEST_PATH = (
    Path(__file__).resolve().parents[1]
    / "crates"
    / "ferrum-core"
    / "src"
    / "render"
    / "chart_config_manifest.json"
)


def _load_rust_manifest() -> dict[str, dict[str, Any]]:
    """Load Task 1's ``{field_key: {"honored": bool, "reason": str}}`` table.

    ``field_key`` is either ``"<Struct>.<field>"`` (e.g.
    ``"AxisStyleSpec.grid"``) or, for the axis_y2-scoped duplicate
    disposition of an ``AxisStyleSpec`` field,
    ``"AxisConfigSpec.axis_y2.<field>"``.
    """
    return json.loads(_MANIFEST_PATH.read_text())


# Rust struct name -> (Python config dataclass, {rust_field: python_field})
# alias map for the handful of fields whose wire name differs from the
# chart-level Python attribute name.
_STRUCT_TO_PYTHON_CONFIG: dict[str, tuple[type, dict[str, str]]] = {
    "ChartConfig": (Configure, {}),
    "AxisConfigSpec": (AxisConfig, {}),
    # AxisStyleSpec's fields are shared (via serde flatten) by the axis /
    # axis_x / axis_y / axis_y2 wire positions; chart-level Python reaches
    # all four through the same AxisConfig dataclass, so they resolve to
    # the same field set. The wire field is `values`; AxisConfig's Python
    # attribute is `tick_values` (aliased on the Rust side — see
    # chart_config.rs's `values` doc comment: "the chart-level `AxisConfig`
    # spelling is `tick_values`").
    "AxisStyleSpec": (AxisConfig, {"values": "tick_values"}),
    "ColorConfigSpec": (ColorConfig, {}),
    "GridConfigSpec": (GridConfig, {}),
    # The per-axis grid sub-struct (`grid.x` / `grid.y`'s object spelling).
    # Python's `GridAxisConfig` is its mirror; a bare bool on the same slot is
    # the enable-only shorthand both sides still accept.
    "GridAxisSpec": (GridAxisConfig, {}),
    "LegendStyleSpec": (LegendConfig, {}),
    "PaddingConfigSpec": (PaddingConfig, {}),
    "TitleConfigSpec": (TitleConfig, {}),
}


def _resolve_python_target(manifest_key: str) -> tuple[type, str] | None:
    """Map one manifest key to ``(python_config_class, expected_field_name)``.

    ``AxisConfigSpec.axis_y2.<field>`` resolves through
    ``AxisStyleSpec``'s mapping: on the Python side, ``Configure.axis_y2``
    is just another ``AxisConfig`` instance, so the axis_y2-scoped
    disposition of an ``AxisStyleSpec`` field shares that field's Python
    reachability with the axis/axis_x/axis_y positions.

    Returns ``None`` when *manifest_key* names a struct with no
    ``Configure``/``configure()``-family surface at all — the
    ``AnnotationSpec.<Variant>.<field>`` / ``StructuralSpec.<Variant>.<field>``
    shape (#92/#148): a THREE-part key that isn't the ``axis_y2`` special
    case. Those structs are reached via ``Chart.annotate()`` and the
    break_axis/inset structural methods, not `_STRUCT_TO_PYTHON_CONFIG`, so
    there is no dataclass to resolve a field name against — distinct from
    "resolved, but the field is absent from that dataclass" (`False` from a
    real lookup), which is what every other 3-part/2-part key still gets.
    """
    parts = manifest_key.split(".")
    if len(parts) == 3:
        struct, nested, field = parts
        if struct != "AxisConfigSpec" or nested != "axis_y2":
            return None
        struct = "AxisStyleSpec"
    else:
        struct, field = parts
    if struct not in _STRUCT_TO_PYTHON_CONFIG:
        return None
    cls, alias = _STRUCT_TO_PYTHON_CONFIG[struct]
    return cls, alias.get(field, field)


def _is_reachable_from_python(manifest_key: str) -> bool:
    target = _resolve_python_target(manifest_key)
    if target is None:
        return False
    cls, python_field = target
    return python_field in {f.name for f in fields(cls)}


# Manifest keys with no matching Python configure.py/_configure_mixin.py
# field, each with a recorded justification. An entry here asserts "this is
# a deliberate, understood gap" — not a silent pass. Adding an entry
# without investigating the field's own `chart_config_manifest.json` reason
# re-opens exactly the "schema field absent from both Python surfaces
# passes silently" hole this leg closes.
_EXPECTED_PYTHON_ABSENT: dict[str, str] = {
    # --- Internal/synthetic wire keys: never user-facing on ANY Python
    # --- surface (not even the per-channel Axis/Legend classes).
    "LegendStyleSpec.disabled": (
        "synthesized by _normalize_legend for legend=None/False; not a "
        "user-facing fm.Legend field on any surface"
    ),
    "LegendStyleSpec.tickLabels": (
        "internal-only explicit colorbar tick labels (e.g. SHAP beeswarm); "
        "LegendStyleSpec's own doc comment: 'Not a user-facing fm.Legend field'"
    ),
    # --- Wire-only derived key.
    "AxisStyleSpec.label_format_type": (
        "wire-only derived key emitted by AxisConfig.to_dict()'s label_format "
        "resolution; not a constructor parameter (label_format is "
        "preset-names-only at this surface, nothing for a caller to set "
        "explicitly)"
    ),
    "AxisConfigSpec.axis_y2.label_format_type": (
        "same wire-only-derived key as AxisStyleSpec.label_format_type above "
        "(the axis_y2 position resolves through the same AxisConfig "
        "dataclass, per _STRUCT_TO_PYTHON_CONFIG's 3-part-key mapping) — not "
        "a constructor parameter on axis_y2's AxisConfig instance either"
    ),
    "LegendStyleSpec.format_type": (
        "chart-level configure_legend was never one of NF-B1's five target "
        "format-resolution surfaces; reachable today via the per-channel "
        "fm.Legend(format_type=) surface (ferrum/legend.py), not chart-level "
        "— matches format's chart-level scope boundary below"
    ),
    # --- Separate top-level chart-config sections reached through their
    # --- own dedicated Python APIs, not the Configure/configure() family.
    "ChartConfig.annotations": (
        "a separate top-level chart-config section reached via "
        "Chart.annotate(), not the Configure/configure() family this leg "
        "covers"
    ),
    "ChartConfig.structural": (
        "a separate top-level chart-config section reached via the "
        "break_axis/inset structural methods, not the Configure/configure() "
        "family this leg covers"
    ),
    # --- AnnotationSpec/StructuralSpec array-item fields (#92/#148): the
    # --- wire gate's array branch closed the per-item key carve-out, which
    # --- is what makes each variant's fields a distinct, gate-checked schema
    # --- surface for the first time; every one is reached via a dedicated
    # --- annotation/structural constructor, not the Configure/configure()
    # --- family this leg covers (same shape as ChartConfig.annotations /
    # --- ChartConfig.structural above, one level deeper).
    "AnnotationSpec.Text.type": (
        "AnnotationSpec.Text's own item-level field, reached via "
        "ferrum.annotation.primitives.AnnotationText.to_dict(), not the "
        "Configure/configure() family this leg covers"
    ),
    "AnnotationSpec.Text.x": (
        "AnnotationSpec.Text's own item-level field, reached via "
        "ferrum.annotation.primitives.AnnotationText.to_dict(), not the "
        "Configure/configure() family this leg covers"
    ),
    "AnnotationSpec.Text.y": (
        "AnnotationSpec.Text's own item-level field, reached via "
        "ferrum.annotation.primitives.AnnotationText.to_dict(), not the "
        "Configure/configure() family this leg covers"
    ),
    "AnnotationSpec.Text.text": (
        "AnnotationSpec.Text's own item-level field, reached via "
        "ferrum.annotation.primitives.AnnotationText.to_dict(), not the "
        "Configure/configure() family this leg covers"
    ),
    "AnnotationSpec.Text.font_size": (
        "AnnotationSpec.Text's own item-level field, reached via "
        "ferrum.annotation.primitives.AnnotationText.to_dict(), not the "
        "Configure/configure() family this leg covers"
    ),
    "AnnotationSpec.Text.color": (
        "AnnotationSpec.Text's own item-level field, reached via "
        "ferrum.annotation.primitives.AnnotationText.to_dict(), not the "
        "Configure/configure() family this leg covers"
    ),
    "AnnotationSpec.Text.anchor": (
        "AnnotationSpec.Text's own item-level field, reached via "
        "ferrum.annotation.primitives.AnnotationText.to_dict(), not the "
        "Configure/configure() family this leg covers"
    ),
    "AnnotationSpec.Text.baseline": (
        "AnnotationSpec.Text's own item-level field, reached via "
        "ferrum.annotation.primitives.AnnotationText.to_dict(), not the "
        "Configure/configure() family this leg covers"
    ),
    "AnnotationSpec.Text.angle": (
        "AnnotationSpec.Text's own item-level field, reached via "
        "ferrum.annotation.primitives.AnnotationText.to_dict(), not the "
        "Configure/configure() family this leg covers"
    ),
    "AnnotationSpec.Text.dx": (
        "AnnotationSpec.Text's own item-level field, reached via "
        "ferrum.annotation.primitives.AnnotationText.to_dict(), not the "
        "Configure/configure() family this leg covers"
    ),
    "AnnotationSpec.Text.dy": (
        "AnnotationSpec.Text's own item-level field, reached via "
        "ferrum.annotation.primitives.AnnotationText.to_dict(), not the "
        "Configure/configure() family this leg covers"
    ),
    "AnnotationSpec.Text.z": (
        "AnnotationSpec.Text's own item-level field, reached via "
        "ferrum.annotation.primitives.AnnotationText.to_dict(), not the "
        "Configure/configure() family this leg covers"
    ),
    "AnnotationSpec.Arrow.type": (
        "AnnotationSpec.Arrow's own item-level field, reached via "
        "ferrum.annotation.primitives.AnnotationArrow.to_dict(), not the "
        "Configure/configure() family this leg covers"
    ),
    "AnnotationSpec.Arrow.x": (
        "AnnotationSpec.Arrow's own item-level field, reached via "
        "ferrum.annotation.primitives.AnnotationArrow.to_dict(), not the "
        "Configure/configure() family this leg covers"
    ),
    "AnnotationSpec.Arrow.y": (
        "AnnotationSpec.Arrow's own item-level field, reached via "
        "ferrum.annotation.primitives.AnnotationArrow.to_dict(), not the "
        "Configure/configure() family this leg covers"
    ),
    "AnnotationSpec.Arrow.x2": (
        "AnnotationSpec.Arrow's own item-level field, reached via "
        "ferrum.annotation.primitives.AnnotationArrow.to_dict(), not the "
        "Configure/configure() family this leg covers"
    ),
    "AnnotationSpec.Arrow.y2": (
        "AnnotationSpec.Arrow's own item-level field, reached via "
        "ferrum.annotation.primitives.AnnotationArrow.to_dict(), not the "
        "Configure/configure() family this leg covers"
    ),
    "AnnotationSpec.Arrow.stroke": (
        "AnnotationSpec.Arrow's own item-level field, reached via "
        "ferrum.annotation.primitives.AnnotationArrow.to_dict(), not the "
        "Configure/configure() family this leg covers"
    ),
    "AnnotationSpec.Arrow.stroke_width": (
        "AnnotationSpec.Arrow's own item-level field, reached via "
        "ferrum.annotation.primitives.AnnotationArrow.to_dict(), not the "
        "Configure/configure() family this leg covers"
    ),
    "AnnotationSpec.Arrow.head_size": (
        "AnnotationSpec.Arrow's own item-level field, reached via "
        "ferrum.annotation.primitives.AnnotationArrow.to_dict(), not the "
        "Configure/configure() family this leg covers"
    ),
    "AnnotationSpec.Rect.type": (
        "AnnotationSpec.Rect's own item-level field, reached via "
        "ferrum.annotation.primitives.AnnotationRect.to_dict(), not the "
        "Configure/configure() family this leg covers"
    ),
    "AnnotationSpec.Rect.x1": (
        "AnnotationSpec.Rect's own item-level field, reached via "
        "ferrum.annotation.primitives.AnnotationRect.to_dict(), not the "
        "Configure/configure() family this leg covers"
    ),
    "AnnotationSpec.Rect.y1": (
        "AnnotationSpec.Rect's own item-level field, reached via "
        "ferrum.annotation.primitives.AnnotationRect.to_dict(), not the "
        "Configure/configure() family this leg covers"
    ),
    "AnnotationSpec.Rect.x2": (
        "AnnotationSpec.Rect's own item-level field, reached via "
        "ferrum.annotation.primitives.AnnotationRect.to_dict(), not the "
        "Configure/configure() family this leg covers"
    ),
    "AnnotationSpec.Rect.y2": (
        "AnnotationSpec.Rect's own item-level field, reached via "
        "ferrum.annotation.primitives.AnnotationRect.to_dict(), not the "
        "Configure/configure() family this leg covers"
    ),
    "AnnotationSpec.Rect.fill": (
        "AnnotationSpec.Rect's own item-level field, reached via "
        "ferrum.annotation.primitives.AnnotationRect.to_dict(), not the "
        "Configure/configure() family this leg covers"
    ),
    "AnnotationSpec.Rect.opacity": (
        "AnnotationSpec.Rect's own item-level field, reached via "
        "ferrum.annotation.primitives.AnnotationRect.to_dict(), not the "
        "Configure/configure() family this leg covers"
    ),
    "AnnotationSpec.Rect.corner_radius": (
        "AnnotationSpec.Rect's own item-level field, reached via "
        "ferrum.annotation.primitives.AnnotationRect.to_dict(), not the "
        "Configure/configure() family this leg covers"
    ),
    "AnnotationSpec.Rect.stroke": (
        "AnnotationSpec.Rect's own item-level field, reached via "
        "ferrum.annotation.primitives.AnnotationRect.to_dict(), not the "
        "Configure/configure() family this leg covers"
    ),
    "AnnotationSpec.Line.type": (
        "AnnotationSpec.Line's own item-level field, reached via "
        "ferrum.annotation.primitives.AnnotationLine.to_dict(), not the "
        "Configure/configure() family this leg covers"
    ),
    "AnnotationSpec.Line.x1": (
        "AnnotationSpec.Line's own item-level field, reached via "
        "ferrum.annotation.primitives.AnnotationLine.to_dict(), not the "
        "Configure/configure() family this leg covers"
    ),
    "AnnotationSpec.Line.y1": (
        "AnnotationSpec.Line's own item-level field, reached via "
        "ferrum.annotation.primitives.AnnotationLine.to_dict(), not the "
        "Configure/configure() family this leg covers"
    ),
    "AnnotationSpec.Line.x2": (
        "AnnotationSpec.Line's own item-level field, reached via "
        "ferrum.annotation.primitives.AnnotationLine.to_dict(), not the "
        "Configure/configure() family this leg covers"
    ),
    "AnnotationSpec.Line.y2": (
        "AnnotationSpec.Line's own item-level field, reached via "
        "ferrum.annotation.primitives.AnnotationLine.to_dict(), not the "
        "Configure/configure() family this leg covers"
    ),
    "AnnotationSpec.Line.stroke": (
        "AnnotationSpec.Line's own item-level field, reached via "
        "ferrum.annotation.primitives.AnnotationLine.to_dict(), not the "
        "Configure/configure() family this leg covers"
    ),
    "AnnotationSpec.Line.stroke_width": (
        "AnnotationSpec.Line's own item-level field, reached via "
        "ferrum.annotation.primitives.AnnotationLine.to_dict(), not the "
        "Configure/configure() family this leg covers"
    ),
    "AnnotationSpec.Line.dash": (
        "AnnotationSpec.Line's own item-level field, reached via "
        "ferrum.annotation.primitives.AnnotationLine.to_dict(), not the "
        "Configure/configure() family this leg covers"
    ),
    "AnnotationSpec.Span.type": (
        "AnnotationSpec.Span's own item-level field, reached via "
        "ferrum.annotation.primitives.AnnotationSpan.to_dict(), not the "
        "Configure/configure() family this leg covers"
    ),
    "AnnotationSpec.Span.axis": (
        "AnnotationSpec.Span's own item-level field, reached via "
        "ferrum.annotation.primitives.AnnotationSpan.to_dict(), not the "
        "Configure/configure() family this leg covers"
    ),
    "AnnotationSpec.Span.start": (
        "AnnotationSpec.Span's own item-level field, reached via "
        "ferrum.annotation.primitives.AnnotationSpan.to_dict(), not the "
        "Configure/configure() family this leg covers"
    ),
    "AnnotationSpec.Span.end": (
        "AnnotationSpec.Span's own item-level field, reached via "
        "ferrum.annotation.primitives.AnnotationSpan.to_dict(), not the "
        "Configure/configure() family this leg covers"
    ),
    "AnnotationSpec.Span.fill": (
        "AnnotationSpec.Span's own item-level field, reached via "
        "ferrum.annotation.primitives.AnnotationSpan.to_dict(), not the "
        "Configure/configure() family this leg covers"
    ),
    "AnnotationSpec.Span.opacity": (
        "AnnotationSpec.Span's own item-level field, reached via "
        "ferrum.annotation.primitives.AnnotationSpan.to_dict(), not the "
        "Configure/configure() family this leg covers"
    ),
    "AnnotationSpec.Span.label_position": (
        "AnnotationSpec.Span's own item-level field, reached via "
        "ferrum.annotation.primitives.AnnotationSpan.to_dict(), not the "
        "Configure/configure() family this leg covers"
    ),
    "AnnotationSpec.Span.label": (
        "AnnotationSpec.Span's own item-level field, reached via "
        "ferrum.annotation.primitives.AnnotationSpan.to_dict(), not the "
        "Configure/configure() family this leg covers"
    ),
    "AnnotationSpec.Bracket.type": (
        "AnnotationSpec.Bracket's own item-level field, reached via "
        "ferrum.annotation.primitives.AnnotationBracket.to_dict(), not the "
        "Configure/configure() family this leg covers"
    ),
    "AnnotationSpec.Bracket.x1": (
        "AnnotationSpec.Bracket's own item-level field, reached via "
        "ferrum.annotation.primitives.AnnotationBracket.to_dict(), not the "
        "Configure/configure() family this leg covers"
    ),
    "AnnotationSpec.Bracket.y1": (
        "AnnotationSpec.Bracket's own item-level field, reached via "
        "ferrum.annotation.primitives.AnnotationBracket.to_dict(), not the "
        "Configure/configure() family this leg covers"
    ),
    "AnnotationSpec.Bracket.x2": (
        "AnnotationSpec.Bracket's own item-level field, reached via "
        "ferrum.annotation.primitives.AnnotationBracket.to_dict(), not the "
        "Configure/configure() family this leg covers"
    ),
    "AnnotationSpec.Bracket.y2": (
        "AnnotationSpec.Bracket's own item-level field, reached via "
        "ferrum.annotation.primitives.AnnotationBracket.to_dict(), not the "
        "Configure/configure() family this leg covers"
    ),
    "AnnotationSpec.Bracket.label": (
        "AnnotationSpec.Bracket's own item-level field, reached via "
        "ferrum.annotation.primitives.AnnotationBracket.to_dict(), not the "
        "Configure/configure() family this leg covers"
    ),
    "AnnotationSpec.Bracket.direction": (
        "AnnotationSpec.Bracket's own item-level field, reached via "
        "ferrum.annotation.primitives.AnnotationBracket.to_dict(), not the "
        "Configure/configure() family this leg covers"
    ),
    "AnnotationSpec.Bracket.stroke": (
        "AnnotationSpec.Bracket's own item-level field, reached via "
        "ferrum.annotation.primitives.AnnotationBracket.to_dict(), not the "
        "Configure/configure() family this leg covers"
    ),
    "AnnotationSpec.Bracket.tip_length": (
        "AnnotationSpec.Bracket's own item-level field, reached via "
        "ferrum.annotation.primitives.AnnotationBracket.to_dict(), not the "
        "Configure/configure() family this leg covers"
    ),
    "AnnotationSpec.Callout.type": (
        "AnnotationSpec.Callout's own item-level field, reached via "
        "ferrum.annotation.primitives.AnnotationCallout.to_dict(), not the "
        "Configure/configure() family this leg covers"
    ),
    "AnnotationSpec.Callout.x": (
        "AnnotationSpec.Callout's own item-level field, reached via "
        "ferrum.annotation.primitives.AnnotationCallout.to_dict(), not the "
        "Configure/configure() family this leg covers"
    ),
    "AnnotationSpec.Callout.y": (
        "AnnotationSpec.Callout's own item-level field, reached via "
        "ferrum.annotation.primitives.AnnotationCallout.to_dict(), not the "
        "Configure/configure() family this leg covers"
    ),
    "AnnotationSpec.Callout.text": (
        "AnnotationSpec.Callout's own item-level field, reached via "
        "ferrum.annotation.primitives.AnnotationCallout.to_dict(), not the "
        "Configure/configure() family this leg covers"
    ),
    "AnnotationSpec.Callout.arrow": (
        "AnnotationSpec.Callout's own item-level field, reached via "
        "ferrum.annotation.primitives.AnnotationCallout.to_dict(), not the "
        "Configure/configure() family this leg covers"
    ),
    "AnnotationSpec.Callout.padding": (
        "AnnotationSpec.Callout's own item-level field, reached via "
        "ferrum.annotation.primitives.AnnotationCallout.to_dict(), not the "
        "Configure/configure() family this leg covers"
    ),
    "AnnotationSpec.Callout.background": (
        "AnnotationSpec.Callout's own item-level field, reached via "
        "ferrum.annotation.primitives.AnnotationCallout.to_dict(), not the "
        "Configure/configure() family this leg covers"
    ),
    "AnnotationSpec.Callout.border_color": (
        "AnnotationSpec.Callout's own item-level field, reached via "
        "ferrum.annotation.primitives.AnnotationCallout.to_dict(), not the "
        "Configure/configure() family this leg covers"
    ),
    "AnnotationSpec.Callout.border_radius": (
        "AnnotationSpec.Callout's own item-level field, reached via "
        "ferrum.annotation.primitives.AnnotationCallout.to_dict(), not the "
        "Configure/configure() family this leg covers"
    ),
    "AnnotationSpec.Callout.text_x": (
        "AnnotationSpec.Callout's own item-level field, reached via "
        "ferrum.annotation.primitives.AnnotationCallout.to_dict(), not the "
        "Configure/configure() family this leg covers"
    ),
    "AnnotationSpec.Callout.text_y": (
        "AnnotationSpec.Callout's own item-level field, reached via "
        "ferrum.annotation.primitives.AnnotationCallout.to_dict(), not the "
        "Configure/configure() family this leg covers"
    ),
    "AnnotationSpec.Image.type": (
        "AnnotationSpec.Image's own item-level field, reached via "
        "ferrum.annotation.primitives.AnnotationImage.to_dict(), not the "
        "Configure/configure() family this leg covers"
    ),
    "AnnotationSpec.Image.x": (
        "AnnotationSpec.Image's own item-level field, reached via "
        "ferrum.annotation.primitives.AnnotationImage.to_dict(), not the "
        "Configure/configure() family this leg covers"
    ),
    "AnnotationSpec.Image.y": (
        "AnnotationSpec.Image's own item-level field, reached via "
        "ferrum.annotation.primitives.AnnotationImage.to_dict(), not the "
        "Configure/configure() family this leg covers"
    ),
    "AnnotationSpec.Image.src": (
        "AnnotationSpec.Image's own item-level field, reached via "
        "ferrum.annotation.primitives.AnnotationImage.to_dict(), not the "
        "Configure/configure() family this leg covers"
    ),
    "AnnotationSpec.Image.width": (
        "AnnotationSpec.Image's own item-level field, reached via "
        "ferrum.annotation.primitives.AnnotationImage.to_dict(), not the "
        "Configure/configure() family this leg covers"
    ),
    "AnnotationSpec.Image.height": (
        "AnnotationSpec.Image's own item-level field, reached via "
        "ferrum.annotation.primitives.AnnotationImage.to_dict(), not the "
        "Configure/configure() family this leg covers"
    ),
    "AnnotationSpec.Image.anchor": (
        "AnnotationSpec.Image's own item-level field, reached via "
        "ferrum.annotation.primitives.AnnotationImage.to_dict(), not the "
        "Configure/configure() family this leg covers"
    ),
    "StructuralSpec.BreakAxis.type": (
        "StructuralSpec.BreakAxis's own item-level field, reached via "
        "ferrum.structural.BreakAxis, not the Configure/configure() family "
        "this leg covers"
    ),
    "StructuralSpec.BreakAxis.axis": (
        "StructuralSpec.BreakAxis's own item-level field, reached via "
        "ferrum.structural.BreakAxis, not the Configure/configure() family "
        "this leg covers"
    ),
    "StructuralSpec.BreakAxis.gaps": (
        "StructuralSpec.BreakAxis's own item-level field, reached via "
        "ferrum.structural.BreakAxis, not the Configure/configure() family "
        "this leg covers"
    ),
    "StructuralSpec.BreakAxis.break_size": (
        "StructuralSpec.BreakAxis's own item-level field, reached via "
        "ferrum.structural.BreakAxis, not the Configure/configure() family "
        "this leg covers"
    ),
    "StructuralSpec.BreakAxis.break_style": (
        "StructuralSpec.BreakAxis's own item-level field, reached via "
        "ferrum.structural.BreakAxis, not the Configure/configure() family "
        "this leg covers"
    ),
    "StructuralSpec.Inset.type": (
        "StructuralSpec.Inset's own item-level field, reached via "
        "ferrum.structural.Inset, not the Configure/configure() family "
        "this leg covers"
    ),
    "StructuralSpec.Inset.svg": (
        "StructuralSpec.Inset's own item-level field, reached via "
        "ferrum.structural.Inset, not the Configure/configure() family "
        "this leg covers"
    ),
    "StructuralSpec.Inset.bounds": (
        "StructuralSpec.Inset's own item-level field, reached via "
        "ferrum.structural.Inset, not the Configure/configure() family "
        "this leg covers"
    ),
    "StructuralSpec.Inset.border": (
        "StructuralSpec.Inset's own item-level field, reached via "
        "ferrum.structural.Inset, not the Configure/configure() family "
        "this leg covers"
    ),
    "StructuralSpec.Inset.border_color": (
        "StructuralSpec.Inset's own item-level field, reached via "
        "ferrum.structural.Inset, not the Configure/configure() family "
        "this leg covers"
    ),
    "StructuralSpec.Inset.border_dash": (
        "StructuralSpec.Inset's own item-level field, reached via "
        "ferrum.structural.Inset, not the Configure/configure() family "
        "this leg covers"
    ),
    "StructuralSpec.Inset.background": (
        "StructuralSpec.Inset's own item-level field, reached via "
        "ferrum.structural.Inset, not the Configure/configure() family "
        "this leg covers"
    ),
    "StructuralSpec.Inset.shadow": (
        "StructuralSpec.Inset's own item-level field, reached via "
        "ferrum.structural.Inset, not the Configure/configure() family "
        "this leg covers"
    ),
    "StructuralSpec.Inset.connect_to": (
        "StructuralSpec.Inset's own item-level field, reached via "
        "ferrum.structural.Inset, not the Configure/configure() family "
        "this leg covers"
    ),
    "StructuralSpec.Inset.connect_style": (
        "StructuralSpec.Inset's own item-level field, reached via "
        "ferrum.structural.Inset, not the Configure/configure() family "
        "this leg covers"
    ),
    # --- Deliberate per-channel-only design choices: Rust honors these,
    # --- Python exposes them on the per-channel fm.Legend class, but never
    # --- promoted them to chart-level configure_legend.
    "LegendStyleSpec.format": (
        "chart-level configure_legend was never one of NF-B1's five target "
        "format-resolution surfaces (per-channel fm.Legend(format=), "
        "encoding format=, and the raw-dict normalize paths were); reachable "
        "today via those per-channel surfaces, not chart-level"
    ),
    "LegendStyleSpec.tick_count": (
        "per-channel-only by design (fm.Legend(tick_count=) exists); not "
        "yet promoted to chart-level configure_legend"
    ),
    "LegendStyleSpec.title": (
        "per-channel-only by design: a single chart-level title would apply "
        "identically to every color/size-encoded legend, which is not the "
        "intended per-encoding UX; fm.Legend(title=) exists"
    ),
    "LegendStyleSpec.type": (
        "per-channel-only by design (fm.Legend(type=) exists); not yet "
        "promoted to chart-level configure_legend"
    ),
    "LegendStyleSpec.values": (
        "per-channel-only by design: explicit legend values are inherently "
        "per-encoding, not a sensible chart-wide default; fm.Legend(values=) "
        "exists"
    ),
}


def _unexplained_gaps(manifest: dict[str, dict[str, Any]]) -> list[str]:
    """Manifest keys neither Python-reachable nor allowlisted."""
    return sorted(
        key
        for key in manifest
        if key not in _EXPECTED_PYTHON_ABSENT and not _is_reachable_from_python(key)
    )


def test_rust_manifest_field_reachable_from_python_or_allowlisted():
    """Every Rust wire-schema field is nameable from a Python configure.py
    surface, or its absence is recorded with a reason.

    This is the direction leg 1 (Python-vs-Python) cannot check: a schema
    field missing from BOTH configure.py and _configure_mixin.py mirrors
    itself perfectly (nothing to compare against nothing) and so passed
    silently before this leg existed.
    """
    manifest = _load_rust_manifest()
    unexplained = _unexplained_gaps(manifest)
    assert not unexplained, (
        f"Rust manifest field(s) {unexplained} have no matching Python "
        "configure.py field/kwarg and no _EXPECTED_PYTHON_ABSENT "
        "justification — this is exactly the 'schema field absent from "
        "both Python surfaces passes silently' hole this test closes."
    )


def test_expected_python_absent_allowlist_has_no_stale_entries():
    """Every allowlisted key must still exist in the manifest and still
    actually be absent from Python.

    An allowlist entry that no longer matches reality — the field was
    added to Python, or dropped from the Rust manifest — is itself a
    silent-loss risk in the other direction (a stale "known gap" masking a
    field that either doesn't exist anymore or was already fixed).
    """
    manifest = _load_rust_manifest()
    for key, reason in _EXPECTED_PYTHON_ABSENT.items():
        assert reason, f"{key!r} has an empty allowlist reason"
        assert key in manifest, f"{key!r} no longer in the Rust manifest; remove from allowlist"
        assert not _is_reachable_from_python(key), (
            f"{key!r} is now reachable from Python; remove from allowlist (it's no longer a gap)"
        )


def test_manifest_twin_is_red_provable():
    """``_unexplained_gaps`` discriminates rather than vacuously passing.

    A synthetic manifest naming a field absent from Python and absent from
    the allowlist must surface as a gap.
    """
    synthetic_manifest = {
        "ColorConfigSpec.totally_made_up_field": {"honored": True, "reason": "synthetic"}
    }
    assert _unexplained_gaps(synthetic_manifest) == ["ColorConfigSpec.totally_made_up_field"]
    assert "ColorConfigSpec.totally_made_up_field" not in _EXPECTED_PYTHON_ABSENT


# ---------------------------------------------------------------------------
# Leg 3 — object.__setattr__ mirror completeness (2026-09-03, design-review
# remediation, finding F3).
#
# `AxisConfig` and `Axis` are `dataclass(init=False)` with hand-written
# `__init__`s that mirror every field three times: field declaration,
# `__init__` parameter, and `object.__setattr__(self, "<name>", <name>)`
# line. A dropped `object.__setattr__` line silently leaves that field at its
# class default and the caller's value vanishes — the precise "accepted on
# one surface, silently lost on another" failure class legs 1/2 above exist
# to catch, reproduced structurally inside these two classes' own
# hand-written constructors. This leg does not attempt the full `init=False`
# refactor (a separate filed follow-up); it is a cheap structural guard that
# mechanically derives the field list, the constructor's parameter list, and
# the constructor source's `object.__setattr__` calls, and asserts all three
# agree.
# ---------------------------------------------------------------------------

_SETATTR_NAME_RE = re.compile(r'object\.__setattr__\(\s*self\s*,\s*"([^"]+)"')


def _setattr_names(cls: type) -> set[str]:
    """Return the field names ``object.__setattr__`` writes inside ``cls.__init__``."""
    source = inspect.getsource(cls.__init__)
    return set(_SETATTR_NAME_RE.findall(source))


# Each entry: (dataclass, alias-only __init__ params). Alias-only params
# (the deprecated `min_extent=`/`max_extent=` spellings) resolve into a
# different canonical field (`min_band`/`max_band`) before construction and
# so never get their own `object.__setattr__` line or dataclass field of
# their own — the same documented-exception shape as `_MANIFEST`'s
# `method_only` above.
_SETATTR_MIRROR_TARGETS = (
    (AxisConfig, frozenset({"min_extent", "max_extent"})),
    (Axis, frozenset({"min_extent", "max_extent"})),
)


@pytest.mark.parametrize(
    "config_cls,alias_only_params",
    _SETATTR_MIRROR_TARGETS,
    ids=[cls.__name__ for cls, _ in _SETATTR_MIRROR_TARGETS],
)
def test_setattr_mirrors_every_field(config_cls, alias_only_params):
    """Every dataclass field has exactly one ``object.__setattr__`` line in
    ``__init__``, and every constructor parameter maps to a real field
    (modulo the documented deprecated-alias params)."""
    dataclass_fields = {f.name for f in fields(config_cls)}
    setattr_names = _setattr_names(config_cls)
    params = {n for n in inspect.signature(config_cls.__init__).parameters if n != "self"}

    missing_setattr = dataclass_fields - setattr_names
    assert not missing_setattr, (
        f"{config_cls.__name__}.__init__ has no object.__setattr__ line for "
        f"field(s) {sorted(missing_setattr)} — a caller's value for this "
        "field would silently vanish, left at the class default"
    )

    orphaned_setattr = setattr_names - dataclass_fields
    assert not orphaned_setattr, (
        f"{config_cls.__name__}.__init__ has object.__setattr__ line(s) for "
        f"non-field name(s) {sorted(orphaned_setattr)}"
    )

    missing_param = dataclass_fields - params - alias_only_params
    assert not missing_param, (
        f"{config_cls.__name__} field(s) {sorted(missing_param)} have no "
        "matching __init__ parameter"
    )


def test_setattr_mirror_check_is_red_provable():
    """``_setattr_names`` discriminates rather than vacuously passing.

    A synthetic frozen, ``init=False`` dataclass whose hand-written
    ``__init__`` drops one ``object.__setattr__`` line must surface as a
    gap — the RED proof this leg's completeness claim rests on.
    """

    @dataclass(frozen=True, init=False)
    class _SyntheticAxisLike:
        kept: int | None = None
        dropped: int | None = None  # __init__ below never assigns this

        def __init__(self, kept: int | None = None, dropped: int | None = None) -> None:
            object.__setattr__(self, "kept", kept)
            # BUG (deliberate, for this test): no object.__setattr__ for "dropped".

    dataclass_fields = {f.name for f in fields(_SyntheticAxisLike)}
    setattr_names = _setattr_names(_SyntheticAxisLike)
    assert dataclass_fields - setattr_names == {"dropped"}


# ---------------------------------------------------------------------------
# Wire-refusal pin — sending a raw config dict with an unknown key through
# the real render path must raise the pinned verbatim-substring refusal
# (D1, spec §4.1/§6, `binding.rs::chart_config_unknown_key_err`).
#
# There is no public chart-level Python surface that accepts a raw config
# dict today (unlike per-channel `fm.X(axis={...})`) — every configure_*
# path builds a typed dataclass. `Chart._append_configure` is the one seam
# every configure_* method already funnels through
# (`self._append_configure(Configure(...))`); appending an object whose
# `to_dict()` returns an arbitrary dict is the minimal way to reach the
# real `to_svg()` -> `render_svg` -> `chart_config_from_dict` ->
# `validate_chart_config_keys` path with a caller-controlled key, without
# inventing a new Python entry point this fix round wasn't asked to build.
# ---------------------------------------------------------------------------


class _RawConfigLayer:
    """Minimal stand-in for a ``Configure`` layer, carrying an arbitrary dict.

    Duck-types the one method ``Chart._resolve_chart_config`` calls
    (``to_dict()``) so it can be appended to ``Chart._configure`` directly,
    bypassing every typed dataclass field check on the way to the wire.
    """

    def __init__(self, raw: dict[str, Any]) -> None:
        self._raw = raw

    def to_dict(self) -> dict[str, Any]:
        return self._raw


def _chart_with_raw_config_layer(raw: dict[str, Any]):
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0]})
    chart = fm.Chart(df).mark_point().encode(x="x", y="y")
    return chart._append_configure(_RawConfigLayer(raw))


def test_wire_gate_refuses_unknown_key_verbatim_substring():
    """Real render path: an unknown key inside a gated chart_config section
    raises ``chart config: unknown key '<k>' in <section>; accepted:
    <sorted list>`` verbatim (pinned substring, both languages per
    binding.rs's own doc comment: 'Do not reword independently — Python's
    mirror of this text must match')."""
    chart = _chart_with_raw_config_layer({"axis": {"totally_bogus_key": 1}})
    with pytest.raises(ValueError) as exc_info:
        chart.to_svg()
    msg = str(exc_info.value)
    assert "chart config: unknown key 'totally_bogus_key' in axis; accepted:" in msg


def test_wire_gate_refuses_unknown_top_level_section_verbatim_substring():
    """Same pin for an unknown top-level chart_config section name (not a
    key nested inside a known section)."""
    chart = _chart_with_raw_config_layer({"totally_bogus_section": {"x": 1}})
    with pytest.raises(ValueError) as exc_info:
        chart.to_svg()
    msg = str(exc_info.value)
    assert "chart config: unknown key 'totally_bogus_section' in chart_config; accepted:" in msg


def test_wire_gate_refuses_unknown_nested_axis_key_verbatim_substring():
    """Same pin one level deeper: the nested per-axis ``grid.x`` / ``grid.y``
    leaf gate (``validate_chart_config_keys``'s third, deepest level).

    The two pins above cover the top level (``chart_config``) and one level
    in (a section like ``axis``); neither reaches the nested-under-a-section
    key gate that ``grid.x``/``grid.y`` go through. Without this pin, a
    disabled third gate level falls back to the untyped serde diagnostic
    (``chart_config: data did not match any variant of untagged enum
    Wire...``) instead of this gate's own ``chart config: unknown key
    'totally_bogus_key' in grid.x; accepted: ...`` — a real regression this
    module's own existing two pins cannot see."""
    chart = _chart_with_raw_config_layer({"grid": {"x": {"totally_bogus_key": 1}}})
    with pytest.raises(ValueError) as exc_info:
        chart.to_svg()
    msg = str(exc_info.value)
    assert "chart config: unknown key 'totally_bogus_key' in grid.x; accepted:" in msg


# ---------------------------------------------------------------------------
# #92/#148 — the array-item wire gate. `annotations`/`structural` are JSON
# arrays of internally-tagged variant structs; before this fix
# `accepted_keys_for_section` returned `None` for both and
# `validate_chart_config_keys` walked no further, so an unknown key inside an
# annotation/structural ITEM reached serde ungated. RED proof (run manually,
# not committed): comment out `validate_chart_config_array_section`'s call
# site in `binding.rs::validate_chart_config_keys` and re-run
# `test_wire_gate_refuses_unknown_annotation_item_key_verbatim_substring` — it
# passes pre-fix (the bogus key sails through to a real render) and fails
# post-fix is reverted, proving this pin is live.
# ---------------------------------------------------------------------------


def test_wire_gate_refuses_unknown_annotation_item_key_verbatim_substring():
    """An unknown key inside an `annotations[]` item raises the same pinned
    substring, naming the item's index and `type` tag as its "section"."""
    chart = _chart_with_raw_config_layer(
        {
            "annotations": [
                {"type": "text", "x": 1.0, "y": 2.0, "text": "hi", "totally_bogus_key": 1}
            ]
        }
    )
    with pytest.raises(ValueError) as exc_info:
        chart.to_svg()
    msg = str(exc_info.value)
    assert (
        "chart config: unknown key 'totally_bogus_key' in annotations[0] (text); accepted:" in msg
    )


def test_wire_gate_refuses_unknown_structural_item_key_verbatim_substring():
    """Same pin for a `structural[]` item (the `break_axis` variant)."""
    chart = _chart_with_raw_config_layer(
        {
            "structural": [
                {
                    "type": "break_axis",
                    "axis": "y",
                    "gaps": [[10.0, 20.0]],
                    "totally_bogus_key": 1,
                }
            ]
        }
    )
    with pytest.raises(ValueError) as exc_info:
        chart.to_svg()
    msg = str(exc_info.value)
    assert (
        "chart config: unknown key 'totally_bogus_key' in structural[0] (break_axis); accepted:"
        in msg
    )


def test_wire_gate_refuses_nan_break_axis_gap():
    """#92's wire half: a hand-built `chart_config` dict carrying a
    ``float('nan')`` gap bound — bypassing ``ferrum.structural.BreakAxis.
    __post_init__``'s construction-time refusal entirely — is refused at the
    wire gate too, not silently resolved into a domain-covering gap that
    erases every datum (BRK-1)."""
    chart = _chart_with_raw_config_layer(
        {
            "structural": [
                {"type": "break_axis", "axis": "y", "gaps": [[float("nan"), float("nan")]]}
            ]
        }
    )
    with pytest.raises(ValueError) as exc_info:
        chart.to_svg()
    msg = str(exc_info.value)
    assert "chart config: structural[0] (break_axis).gaps[0]: start must be finite" in msg


def test_wire_gate_refuses_reversed_break_axis_gap():
    """Same wire gate, the ordering half: `start >= end` refuses even though
    both bounds are perfectly finite numbers — this half has no JSON-encoding
    accident to fall back on, so without this check it would render silently
    (a reversed gap clips to an empty range, per `normalize_gaps`'s doc)."""
    chart = _chart_with_raw_config_layer(
        {"structural": [{"type": "break_axis", "axis": "y", "gaps": [[200.0, 100.0]]}]}
    )
    with pytest.raises(ValueError) as exc_info:
        chart.to_svg()
    msg = str(exc_info.value)
    assert (
        "chart config: structural[0] (break_axis).gaps[0]: start (200.0) must be less than end (100.0)"
        in msg
    )


def test_wire_gate_refuses_out_of_vocabulary_break_axis_token():
    """`#229`(2)'s wire half, sibling to the two `gaps` pins above.

    `BreakAxis.axis` is typed in Rust (`spec::axis::AlignAxis`), so a third
    token is malformed input. It is refused at the same array-item VALUE gate
    the `gaps` checks above come from, in ferrum's own voice and naming the
    item and its index — not left to serde's downstream ``chart_config:
    unknown variant `z` `` , which names neither and carries a different
    prefix. Pre-`#229`(2) this deserialized as a plain string, failed every
    downstream ``axis == "y"`` test and rendered as an **x** break with no
    diagnostic.

    This asserts the message a caller actually receives, through
    ``chart.to_svg()`` — i.e. through ``pyo3_serde``/``chart_config_from_dict``
    — because it is the string ``ferrum-spec.md`` and the changelog quote
    verbatim. The Rust-side unit test
    (``chart_config.rs::structural_break_axis_refuses_an_out_of_vocabulary_axis_token``)
    covers the serde path, which remains for a consumer reaching
    ``Deserialize`` without this gate in front of it (a saved spec read
    straight through serde, say). The gate owns the VOCABULARY for every key
    spelling ``json.dumps`` can produce — see
    ``test_wire_gate_refuses_every_json_dumps_key_spelling_out_of_vocabulary``,
    which derives its cases from ``json.dumps`` rather than listing them — and
    serde keeps the SHAPE errors, pinned by
    ``test_wire_gate_leaves_break_axis_axis_shape_errors_to_serde``.
    """
    chart = _chart_with_raw_config_layer(
        {"structural": [{"type": "break_axis", "axis": "z", "gaps": [[10.0, 20.0]]}]}
    )
    with pytest.raises(ValueError) as exc_info:
        chart.to_svg()
    assert (
        str(exc_info.value) == "chart config: structural[0] (break_axis).axis: axis must be one of "
        "['x', 'y']; got 'z'"
    )


def test_wire_gate_break_axis_token_refusal_is_case_sensitive():
    """The vocabulary is the lowercase wire token, not a case-insensitive
    spelling: `"Y"` is as malformed as `"z"`, and the refusal echoes what was
    received rather than silently lowercasing it. Discriminating against a
    fix that "helpfully" normalizes case — the Python constructor does not,
    so the wire gate must not either."""
    chart = _chart_with_raw_config_layer(
        {"structural": [{"type": "break_axis", "axis": "Y", "gaps": [[10.0, 20.0]]}]}
    )
    with pytest.raises(ValueError) as exc_info:
        chart.to_svg()
    assert "got 'Y'" in str(exc_info.value)


def test_wire_gate_accepts_both_break_axis_tokens():
    """The negative control the two refusals above need: the gate accepts
    exactly the vocabulary it names, so neither pin is passing because the
    gate refuses everything."""
    for axis in ("x", "y"):
        chart = _chart_with_raw_config_layer(
            {"structural": [{"type": "break_axis", "axis": axis, "gaps": [[1.5, 2.5]]}]}
        )
        assert chart.to_svg().startswith("<svg"), axis


def test_wire_gate_refuses_out_of_vocabulary_break_axis_token_in_map_spelling():
    """A unit variant has TWO wire spellings and the gate must cover both.

    Besides ``"axis": "y"``, serde's derived impl accepts the externally-tagged
    single-key map ``"axis": {"y": null}`` — which is why its own type error
    says "expected string **or** map". The first cut of this gate checked only
    the string and let the map form through on its non-string carve-out, so
    ``{"z": null}`` still reached serde and got the opaque
    ``chart_config: unknown variant `z` ... at line 1 column N``: one user
    error, two differently-worded refusals. Both spellings now get the same
    sentence. This pin covers the ``str``-keyed map; the *completeness* claim
    over every key type ``json.dumps`` coerces is
    ``test_wire_gate_refuses_every_json_dumps_key_spelling_out_of_vocabulary``
    below, which derives its cases from the transform instead of enumerating
    shapes.
    """
    chart = _chart_with_raw_config_layer(
        {"structural": [{"type": "break_axis", "axis": {"z": None}, "gaps": [[10.0, 20.0]]}]}
    )
    with pytest.raises(ValueError) as exc_info:
        chart.to_svg()
    assert (
        str(exc_info.value) == "chart config: structural[0] (break_axis).axis: axis must be one of "
        "['x', 'y']; got 'z'"
    )


def test_wire_gate_map_spelling_of_a_valid_break_axis_token_still_renders():
    """The control that makes the pin above a vocabulary check rather than a
    ban on the map spelling: ``{"y": null}`` is a *valid* input serde accepts,
    and it renders byte-identically to the string spelling. A gate that
    refused the map form outright would pass the pin above and break this."""

    def svg_for(axis_value):
        chart = _chart_with_raw_config_layer(
            {"structural": [{"type": "break_axis", "axis": axis_value, "gaps": [[1.5, 2.5]]}]}
        )
        return chart.to_svg()

    assert svg_for({"y": None}) == svg_for("y")
    assert svg_for({"x": None}) == svg_for("x")


def test_wire_gate_refuses_every_json_dumps_key_spelling_out_of_vocabulary():
    """The completeness pin, derived from the transform rather than from shapes.

    The gate reads the caller's **pre-JSON Python object**, while the vocabulary
    it guards is defined over what ``json.dumps`` *emits*. So the question "does
    the gate see every token a caller can present" is a question about
    ``json.dumps``, and CPython answers it: its encoder coerces exactly five key
    types — ``str``, ``float``, ``bool``, ``None``, ``int`` — and raises
    ``TypeError: keys must be str, int, float, bool or None, not X`` for
    everything else (pinned by the boundary test below). Covering those five is
    therefore complete, with nothing behind it.

    Two earlier cuts of this gate claimed completeness from probed shapes and
    were false both times: the first covered only ``"axis": "z"``; the second
    covered ``{"z": None}`` but not the non-``str`` keys ``json.dumps``
    *coerces* into keys, so ``{1: None}``, ``{True: None}``, ``{None: None}``,
    ``{1.5: None}``, ``{nan: None}`` and ``{inf: None}`` all still reached
    serde's opaque ``unknown variant``.

    Each expected token here is computed **by asking ``json.dumps``**, not
    written out, so this test cannot drift from the spelling the bridge actually
    produces — which is the point: ``1e16`` must echo ``1e+16`` (Python's
    ``float.__repr__``, where a Rust-side format would write
    ``10000000000000000``), ``True`` must echo ``true`` rather than ``1``
    (``bool`` is an ``int`` subclass and ``json`` spells it ``true``), and a
    30-digit ``int`` must survive at full precision.
    """
    out_of_vocabulary_keys = [
        "z",
        1,
        -3,
        10**30,
        True,
        False,
        None,
        1.5,
        -0.0,
        1e16,
        1e-7,
        float("nan"),
        float("inf"),
        float("-inf"),
    ]
    for key in out_of_vocabulary_keys:
        # The token the bridge would hand serde, asked of json itself.
        expected_token = json.loads(json.dumps({key: None})).popitem()[0]
        assert expected_token not in ("x", "y"), (
            f"fixture error: {key!r} spells an IN-vocabulary token"
        )
        chart = _chart_with_raw_config_layer(
            {"structural": [{"type": "break_axis", "axis": {key: None}, "gaps": [[10.0, 20.0]]}]}
        )
        with pytest.raises(ValueError) as exc_info:
            chart.to_svg()
        msg = str(exc_info.value)
        assert msg == (
            "chart config: structural[0] (break_axis).axis: axis must be one of "
            f"['x', 'y']; got '{expected_token}'"
        ), (key, msg)
        # The whole point: serde's opaque message is not what the caller gets.
        assert "unknown variant" not in msg, (key, msg)


def test_json_dumps_key_coercion_boundary_is_the_five_types_the_gate_covers():
    """The other edge of the derivation: a key type ``json.dumps`` does NOT
    coerce never reaches serde at all, so the five types above are the whole
    space the gate has to cover.

    This pins the premise of the completeness argument rather than ferrum
    behavior, so that a future CPython change widening the accepted key set
    fails here — beside the gate that would then be incomplete — instead of
    silently reopening the hole.

    (The refusal for such a key is a raw ``TypeError`` from ``json.dumps``
    inside the PyO3 bridge, which is pre-existing for *any* malformed
    ``chart_config`` dict and logged as a follow-up. Asserted here as the
    current fact, not endorsed as the right voice.)
    """
    for key in [(1, 2), frozenset({1}), b"y", 1 + 2j]:
        with pytest.raises(TypeError, match="keys must be str, int, float, bool or None"):
            json.dumps({key: None})

    chart = _chart_with_raw_config_layer(
        {"structural": [{"type": "break_axis", "axis": {(1, 2): None}, "gaps": [[10.0, 20.0]]}]}
    )
    with pytest.raises(TypeError, match="keys must be str, int, float, bool or None"):
        chart.to_svg()


def test_wire_gate_leaves_break_axis_axis_shape_errors_to_serde():
    """The documented carve-outs: this gate owns the vocabulary, serde owns the
    SHAPE. A value that names no token — a non-string/non-dict, or a dict whose
    key count is not exactly one — walks past the gate and is reported by
    serde, which describes those shapes better than the gate could. Pinned so
    the carve-outs are a stated contract rather than an accident of the
    ``let ... else`` ladder."""
    cases = [
        (7, "expected string or map"),
        (None, "expected string or map"),
        ({}, "expected map with a single key"),
        ({"x": None, "y": None}, "expected map with a single key"),
    ]
    for axis_value, expected in cases:
        chart = _chart_with_raw_config_layer(
            {"structural": [{"type": "break_axis", "axis": axis_value, "gaps": [[10.0, 20.0]]}]}
        )
        with pytest.raises(ValueError) as exc_info:
            chart.to_svg()
        msg = str(exc_info.value)
        assert expected in msg, (axis_value, msg)
        # Not the gate's message: these shapes name no token to report.
        assert "axis must be one of" not in msg, (axis_value, msg)
