# `Chart.override()` Wiring — Design Spec (Full Escape-Hatch)

- **Date:** 2026-06-14
- **Status:** Draft for review (scoping only — no implementation)
- **Related:** RCA `design-docs/superpowers/followups/2026-06-14-override-silent-drop-rca.md` (bug B4); original design `design-docs/superpowers/specs/2026-05-24-declarative-configure-design.md`

## 1. Scope

Make `Chart.override(**kwargs)` a working **spec-path escape hatch**: a flat snake_case key sets any field on the chart's *presentation* spec, reaching beyond the typed `configure_*` surface (scales, coord, mark style, per-channel axis/legend, title, padding, width/height — not just the six config dataclasses). Override is applied at render with top-of-cascade precedence. Today the kwargs are stored in `self._overrides` and never read, so every override is silently dropped (the feature is a no-op for all keys).

Because the Rust spec deserializers **silently drop unknown fields** (verified: `ChartSpec`, `Encoding`, `EncodingSpec`, `MarkKwargsSpec`, `ChartConfig` carry no `deny_unknown_fields`; only `TitleSpec` / `ThemeOverridesSpec` reject unknowns), a typo cannot fail-loud on the Rust side. Validation is therefore **Python-side against a generated registry of valid paths**.

## 2. Goals

- Every documented path category takes effect in the rendered output: per-axis (`x_axis_*`, `y_axis_*`), legend (`legend_*`), title (`title_*`), grid (`grid_*`), padding (`padding_*`), color (`color_*`), per-channel scale (`<channel>_scale_*`) and per-channel axis/legend, mark style (`mark_*`), coordinate (`coord_*`), and chart properties (`width`, `height`).
- Override **wins the cascade**: not overridden by per-channel `axis=`/`legend=`, `configure_*`, themes, `set_default_theme`, or Rust defaults.
- Unknown / misspelled paths raise `FerrumOverrideError` **at render time**, naming the bad path and the closest valid match (Python-side validation; the Rust silent-drop cannot be relied on).
- Paths that have a typed `configure_*`/`properties` equivalent emit a `DeprecationWarning` at render pointing to that method, but still apply.
- `.override()` stays a pure immutable builder; multi-call and composition merge semantics unchanged (later-wins).
- Docs in `concepts/override.md` / `customizing-charts.md` match shipped behavior; the storage-only `tests/test_override.py` is replaced with render-level coverage.

## 3. Non-goals

- **Structural grammar via override.** Override does not set the mark type, encoding *field bindings* (`encoding.x.field`), data, transforms, faceting structure, layers, selections, params, or conditionals. Those are the typed grammar surface, not presentation. Override reaches presentation/config leaves only.
- **Reaching Rust-only spec fields that have no Python-side schema to enumerate.** The registry is generated from Python-introspectable schemas (see §5). Fields that exist only in Rust structs with no Python mirror are out of scope until exposed Python-side. (See Open Question Q2.)
- **Relaxing Rust deserialization.** No change to serde attributes is required by this design (validation is Python-side). Optionally hardening Rust with `deny_unknown_fields` is a separate, complementary effort (Q3), not in scope here.
- **A new capability that no spec field expresses.** Override exposes existing spec leaves; it does not invent renderer behavior.

## 4. System behavior

- **Apply at render.** Each stored override key is resolved through the registry to a `(target, spec-location, value)` and applied during render prep, **after** all configure layers and per-channel settings, so override wins on conflict.
- **Routing by target.** The registry assigns each path to one of:
  - *chart-config* (`axis_x`/`axis_y`/`axis`/`legend`/`title`/`grid`/`padding`/`color`) → folds into the chart-config dict.
  - *encoding sub-spec* (`<channel>_scale_*`, `<channel>_axis_*`, `<channel>_legend_*`, `<channel>_sort`, …) → merges into that channel's `EncodingSpec` in the assembled spec.
  - *mark style* (`mark_*`) → merges into `mark_style`.
  - *coord* (`coord_*`) → sets fields on the `coord` spec.
  - *properties* (`width`, `height`) → sets the chart's pixel dimensions.
- **Validation (Python-side, at render, once).** Each key is checked against the registry. Unknown prefix, or known prefix with a leaf not in that target's valid-field set, raises `FerrumOverrideError` with the offending path and the nearest valid path via `difflib.get_close_matches`. `.override()` itself never validates (stays typo-tolerant per docs).
- **Deprecation.** A path whose target+leaf has a typed equivalent (e.g. `x_axis_label_angle` ↔ `.configure_axis(label_angle=…)`, `width` ↔ `.properties(width=…)`) emits a `DeprecationWarning` naming the typed method, then applies. Paths with no typed equivalent (the genuine escape-hatch cases — e.g. a scale field not exposed by `configure_*`) apply silently.
- **Cascade & merge.** Override beats per-channel `axis=`/`legend=`, configure, theme, defaults. `.override(a=1).override(a=2)` → 2; composition merges later-wins; `.override()` never mutates the receiver.

## 5. Architecture

- **Consumer** runs in render prep. Two injection points already exist and are reused: chart-config-targeted overrides merge into the dict built by `Chart._resolve_chart_config` (`_render.py:478`); spec-targeted overrides (encoding sub-specs, mark style, coord) merge into the kw dict assembled by `Chart.to_spec` before the `ChartSpec(**kw)` construction; property overrides set width/height on the render-time chart. All merges run last in their respective assembly so override wins.
- **Registry** is a single declarative structure mapping a path (or path-prefix + channel) to `(target, spec-location, valid-leaf-set, typed-equivalent?)`. Valid-leaf sets are **generated from Python-introspectable schemas** so the registry cannot drift:
  - config leaves ← `configure` dataclasses (`AxisConfig`, `LegendConfig`, `TitleConfig`, `GridConfig`, `PaddingConfig`, `ColorConfig`).
  - encoding channels ← the encoding channel classes in `src/ferrum/encoding/`; per-channel sub-spec leaves (`scale`, `axis`, `legend`, `sort`, `stack`, …) ← `EncodingSpec` fields and the scale/axis/legend sub-schemas.
  - mark-style leaves ← the mark-style kwarg surface (the `MarkKwargsSpec` field set mirrored Python-side).
  - coord leaves ← the `CoordKind` variant fields.
  - properties ← the `properties()` signature (`width`, `height`).
- **Path grammar.** Flat snake_case (matches the docs). Resolution is **registry-driven, longest-prefix-first**, not a free parser: `x_axis_` / `y_axis_` bind before `axis_`; `<channel>_scale_` / `<channel>_axis_` / `<channel>_legend_` bind per encoding channel; `mark_`, `coord_`, `legend_`, `title_`, `grid_`, `padding_`, `color_` bind their targets; `width` / `height` are exact. A flat path resolves only if it is an enumerated registry entry — ambiguity is impossible because the registry is an explicit set, not a heuristic split.
- **Error type** `FerrumOverrideError` is a new public exception, raised only by the applicator.
- **Rust untouched.** Overrides reach Rust through the existing `chart_config` / `ChartSpec` / properties channels; no new binding fields and no serde changes.

## 6. Canonical interfaces / data contracts

Representative registry entries (path → target spec-location → valid-leaf source):

| Path pattern              | Target           | Spec location                          | Valid leaves from        |
|---------------------------|------------------|----------------------------------------|--------------------------|
| `x_axis_<leaf>`           | chart-config     | `chart_config["axis_x"][leaf]`         | `AxisConfig`             |
| `y_axis_<leaf>`           | chart-config     | `chart_config["axis_y"][leaf]`         | `AxisConfig`             |
| `legend_<leaf>`           | chart-config     | `chart_config["legend"][leaf]`         | `LegendConfig`           |
| `title_<leaf>`, `grid_*`, `padding_*`, `color_*` | chart-config | `chart_config[target][leaf]` | resp. config dataclass |
| `<channel>_scale_<leaf>`  | encoding sub-spec| `encoding[channel]["scale"][leaf]`     | scale schema             |
| `<channel>_axis_<leaf>` / `<channel>_legend_<leaf>` / `<channel>_sort` | encoding sub-spec | `encoding[channel][...]` | `EncodingSpec` sub-schemas |
| `mark_<leaf>`             | mark style       | `mark_style[leaf]`                     | mark-style field set     |
| `coord_<leaf>`            | coord            | `coord[leaf]`                          | `CoordKind` variant      |
| `width`, `height`         | properties       | chart dimensions                       | exact keys               |

Example resolutions: `x_axis_label_angle=-45` → `chart_config["axis_x"]["label_angle"]`; `color_scheme="viridis"` → color config; `x_scale_domain=[0,10]` → `encoding["x"]["scale"]["domain"]`; `mark_corner_radius=4` → `mark_style["corner_radius"]`.

`FerrumOverrideError` message contract (shape, not exact text):
```
FerrumOverrideError: Unknown override path 'x_axis_lable_angle'. Did you mean: 'x_axis_label_angle'?
```

## 7. Invariants and constraints

- **No matplotlib; no new global mutable state.** The registry is a module-level constant built at import from schema introspection (read-only), not a process-scoped mutable.
- **Cascade order fixed:** override > per-channel `axis=`/`legend=` > `configure_*` > theme > `set_default_theme` > Rust defaults. The applicator runs last among presentation sources.
- **Fail loud, never silent.** Any override path that does not resolve raises `FerrumOverrideError`. Closes the B4 class. Note Rust will *not* catch a bad path (silent-drop), so Python-side validation is the only guard — it must cover every accepted path.
- **Registry parity.** Valid-leaf sets are derived from the live Python schemas at import, so extending the typed surface automatically extends the override surface; no second edit, no drift.
- **Backward-compatible storage API.** `.override()` signature, immutability, and merge semantics unchanged; only the previously-absent render effect is added. Existing code that called `.override()` expecting an effect now gets one — changelog-worthy.

## 8. Key decisions and tradeoffs

- **D1 — Full escape-hatch over the presentation spec.** Override reaches the whole presentation/config surface (config + scales + per-channel axis/legend + mark style + coord + properties), not just the six config dataclasses. *Rationale:* matches the documented intent ("for the rare case the typed surface hasn't caught up"). *Bounded* to presentation, excluding structural grammar (mark/field/transforms), to keep the registry finite and meaningful.
- **D2 — Python-side validation is mandatory, not optional.** Because the Rust spec structs silently drop unknown fields, a bad path would otherwise be a silent no-op again — re-creating the exact bug. The registry must enumerate every valid path so unknowns fail loud Python-side. *Rejected:* relying on Rust to reject unknowns (it doesn't, except `TitleSpec`).
- **D3 — Deprecation routing is in scope.** Under the full surface, some paths have typed equivalents and some don't, so the documented `DeprecationWarning` behavior is meaningful (warn on the ones with a typed method; apply silently otherwise). This restores the docs §"Deprecation warnings" as implementable.
- **D4 — Flat snake_case, registry-resolved (not a parser).** Path resolution is lookup against an explicit enumerated registry, so prefix ambiguity (`x_axis_` vs `axis_`, `x_scale_` vs `x_axis_`) is structurally impossible. *Rejected:* dotted paths (`encoding.x.scale.type`) — unambiguous but breaks the documented flat style and the existing examples.
- **D5 — Property/mark/coord overrides beat their typed setters.** `width`/`height`/`mark_*`/`coord_*` via override win over `.properties()`/`mark_*()`/coord settings, honoring "override wins everything." Surprising-but-documented; acceptance tests pin it.
- **D6 — Validation at render, once.** Matches the documented "valid Python even with typos; error at render," keeps `.override()` pure, and validates the full set in one pass during render prep.

## 9. Acceptance criteria

- Each path category changes a specific observable in the SVG relative to the un-overridden chart: `x_axis_label_angle=-45` → rotated end-anchored x labels; `width=900` → 900-wide viewBox; `legend_orient="bottom"` → relocated legend; `color_scheme=...` → changed mark colors; `x_scale_domain=[a,b]` → changed axis extent; `mark_corner_radius=r` → rounded bars; `coord_*` → changed coordinate behavior.
- Conflict resolution: `.configure_axis(label_angle=0).override(x_axis_label_angle=-45)` renders `-45`, both call orders.
- `.override(x_axis_lable_angle=-45)` (typo) raises `FerrumOverrideError` naming `x_axis_label_angle`; `.override(bogus=1)` raises with no/À-closest suggestion; a known prefix with an invalid leaf (`mark_notathing=1`) raises.
- A path with a typed equivalent emits `DeprecationWarning` and still applies; a path without one applies with no warning.
- `.override()` remains immutable and merge-correct (purity tests retained).
- `tests/test_override.py` asserts rendered output + error/deprecation paths; storage-only assertions are replaced/augmented.
- Charts that never call `.override()` produce byte-identical output (no golden churn); spot-checked on a representative golden.

## 10. Validation strategy

- **Behavioral render tests** (Python): per target category, assert an override changes a specific observable SVG attribute vs. the baseline chart — assert output, never `_overrides` contents. This is the exact gap that let B4 ship; it is the load-bearing test layer.
- **Cascade tests:** override-vs-configure and override-vs-per-channel conflicts resolve to the override value, both call orders.
- **Validation tests:** unknown prefix, known-prefix-unknown-leaf, and typo each raise `FerrumOverrideError`; suggestion present when a close match exists.
- **Deprecation tests:** `pytest.warns(DeprecationWarning)` for typed-equivalent paths; `pytest` asserts no warning for genuine-gap paths; both still apply.
- **Registry-parity test:** every registry leaf set equals the live schema field set it is generated from (guards drift if a config field is added/removed).
- **Purity/merge tests:** retained from the current suite.

## 11. Open questions

- **Q1 (path-flattening rule):** the exact flat-name convention for nested encoding leaves (`<channel>_scale_<leaf>` vs `<channel>_<leaf>` where the sub-target is implied). Needs a single documented rule before the encoding-sub-spec slice is built. Default proposal: explicit sub-target segment (`x_scale_domain`, `color_legend_orient`).
- **Q2 (Rust-only fields):** some presentation fields may exist in Rust spec structs with no Python schema to enumerate. Decision: (a) leave them out of the registry (out of scope, raise `FerrumOverrideError`), or (b) mirror them Python-side so they become reachable. Default: (a) for v1; widen later.
- **Q3 (defense in depth):** optionally add `#[serde(deny_unknown_fields)]` to the presentation spec structs so Rust *also* rejects unknowns — a backstop against future Python/registry drift. Complementary, not required by this design; flag as a follow-up.
- **Q4 (`mark_*` on multi-layer/composite charts):** apply to the base/primary mark, all layers, or error as ambiguous? Default proposal: base/primary mark; error if no single primary. Confirm before the mark slice is built.
