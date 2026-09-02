"""Python twin of the chart-config disposition manifest (spec §6, NF-B13).

The Rust half — the checked-in ``{field: {honored, reason}}`` disposition
table at ``crates/ferrum-core/src/render/chart_config_manifest.json``,
generated and consumed by a test in
``crates/ferrum-core/src/render/chart_config.rs`` (Task 1, closed
2026-09-02) — enumerates every serde field of the wire config schema. This
module carries two legs:

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
"""

from __future__ import annotations

import inspect
import json
from dataclasses import fields
from pathlib import Path
from typing import Any

import polars as pl
import pytest

import ferrum as fm
from ferrum._configure_mixin import ConfigureMixin
from ferrum.configure import (
    AxisConfig,
    ColorConfig,
    Configure,
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
    "LegendStyleSpec": (LegendConfig, {}),
    "PaddingConfigSpec": (PaddingConfig, {}),
    "TitleConfigSpec": (TitleConfig, {}),
}


def _resolve_python_target(manifest_key: str) -> tuple[type, str]:
    """Map one manifest key to ``(python_config_class, expected_field_name)``.

    ``AxisConfigSpec.axis_y2.<field>`` resolves through
    ``AxisStyleSpec``'s mapping: on the Python side, ``Configure.axis_y2``
    is just another ``AxisConfig`` instance, so the axis_y2-scoped
    disposition of an ``AxisStyleSpec`` field shares that field's Python
    reachability with the axis/axis_x/axis_y positions.
    """
    parts = manifest_key.split(".")
    if len(parts) == 3:
        _struct, nested, field = parts
        assert nested == "axis_y2", f"unexpected 3-part manifest key: {manifest_key!r}"
        struct = "AxisStyleSpec"
    else:
        struct, field = parts
    cls, alias = _STRUCT_TO_PYTHON_CONFIG[struct]
    return cls, alias.get(field, field)


def _is_reachable_from_python(manifest_key: str) -> bool:
    cls, python_field = _resolve_python_target(manifest_key)
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
    # --- No chart-level Rust consumer yet (pending a later batch task);
    # --- Python correctly withholds the parameter rather than accepting a
    # --- value that would silently no-op today.
    "AxisStyleSpec.labels": (
        "no chart-level Rust consumer (AxisInput.show_labels is a plain "
        "bool, per-channel-only); adding a Python kwarg today would "
        "silently no-op. Pending Task 8 per chart_config_manifest.json"
    ),
    "AxisStyleSpec.ticks": (
        "same disposition as labels — no chart-level Rust consumer; "
        "pending Task 8 per chart_config_manifest.json"
    ),
    "AxisStyleSpec.title": (
        "no chart-level Rust consumer (AxisInput.title's unset/suppressed "
        "tri-state conflation makes a naive chart-level fill unsafe); "
        "chart_config_manifest.json's own reason confirms AxisConfig does "
        "not expose title either today. Pending Task 8's tri-state model"
    ),
    "LegendStyleSpec.format_type": (
        "zero Rust consumers yet (pending Task 4, the legend half of "
        "format_type threading); matches format's chart-level scope "
        "boundary below"
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
    # --- Rust already honors these at the chart-level axis position;
    # --- AxisConfig has simply never exposed a matching Python kwarg. Not a
    # --- silent-loss bug (there is no Python field to lose data from), but
    # --- called out distinctly from the "no Rust consumer" entries above —
    # --- this is an unbuilt feature, flagged for the orchestrator rather
    # --- than added here (public-API addition; out of this fix round's
    # --- two-item scope).
    "AxisStyleSpec.offset": (
        "Rust honors this at chart-level (axis_style_fill_from -> "
        "AxisStyleOverrides.offset) but AxisConfig has never exposed a "
        "matching kwarg — an unbuilt feature, not a silent-loss bug. "
        "Flagged for the orchestrator, not fixed in this round."
    ),
    "AxisStyleSpec.label_flush": (
        "same situation as offset above: Rust honors this at chart-level "
        "(axis_style_fill_from -> AxisStyleOverrides.label_flush) but "
        "AxisConfig has never exposed a matching kwarg."
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
            f"{key!r} is now reachable from Python; remove from allowlist "
            "(it's no longer a gap)"
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
