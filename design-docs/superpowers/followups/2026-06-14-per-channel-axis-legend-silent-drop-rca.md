# RCA — B5: per-channel `axis={…}` / `legend={…}` keys silently dropped

- **Date:** 2026-06-14
- **Status:** Open — root cause identified, remediation not yet started
- **Severity:** S3 — documented per-channel styling parameters silently no-op at render. A working chart-level sibling (`configure_axis`/`configure_legend`) exists for most of them, which is why this is less severe than B4, but it is still a multi-parameter silent drop across ~20+ documented `Axis`/`Legend` fields.
- **Discovered:** surfaced during the `Chart.override()` investigation (B4); filed as its own follow-up (B5) in `2026-05-15-code-archaeology.md`.
- **Subsystem:** `src/ferrum/axis.py` + `src/ferrum/legend.py` (Python value classes), `crates/ferrum-core/src/spec/encoding.rs` (`AxisSpec`/`LegendSpec`), `crates/ferrum-core/src/render/prepare.rs` (the only consumer).

---

## 1. Symptom

```python
# Per-channel — silently dropped:
fm.Chart(df).mark_point().encode(
    x=fm.X("a", axis=fm.Axis(label_color="#ff00ff", grid_color="#ccc", domain_width=2.0))
).to_svg()
# → labels are not magenta, grid is not #ccc, domain line is default width.

# Chart-level — works, same properties:
fm.Chart(df).mark_point().encode(x="a", y="b") \
    .configure_axis(label_color="#ff00ff", grid_color="#ccc", domain_width=2.0).to_svg()
# → all three honored.
```

`fm.Axis(...)` / `fm.Legend(...)` advertise ~32 and ~26 parameters respectively (full NumPy docstrings, phase 12). The renderer honors only a hand-picked subset per channel. Every other parameter serializes cleanly into the spec and is then never read.

## 2. Root cause

Per-channel axis/legend config crosses the PyO3 boundary as an **opaque, zero-named-field blob**, and the Rust renderer hand-plucks individual keys out of it one at a time. There is no typed struct and no single deserialize+merge step, so coverage equals "whichever keys some past feature happened to need."

`crates/ferrum-core/src/spec/encoding.rs:285-296`:
```rust
/// Opaque-but-typed axis spec. Round-trips JSON; renderer ignores in 8a.
pub struct AxisSpec  { #[serde(flatten)] pub extra: serde_json::Map<String, serde_json::Value> }
pub struct LegendSpec { #[serde(flatten)] pub extra: serde_json::Map<String, serde_json::Value> }
```

The **only** consumer of `extra` in the entire crate is `render/prepare.rs` (verified: zero `extra.get` reads in `layout/`, `render/binding.rs`, or anywhere else). It reaches into the map for a fixed list of string keys:

- **Axis** (`prepare.rs:511-577`, plus the `tick_count`/`label_format` helpers at `prepare.rs:705-736`): `labels`, `ticks`, `domain`, `grid`, `label_angle`/`labelAngle`, `title`, `tick_count`/`tickCount`, `label_format`/`labelFormat`, `label_format_type`/`labelFormatType` — **9 distinct properties**.
- **Legend** (D13 block `prepare.rs:280-356` + SB3 `disabled`/`format`/`tickLabels`): `disabled`, `title`, `format`, `orient`, `direction`, `type`, `tick_count`, `values`, `columns`, `title_font_size`, `label_font_size`, `gradient_length`, `gradient_thickness`, `tickLabels` (internal) — **~14 properties**.

The Python side serializes the *full* field set with no gating. `Axis.to_dict()` (`axis.py:154-178`) and `Legend.to_dict()` (`legend.py:117-138`) emit every set field; `_normalize_axis`/`_normalize_legend` pass raw dicts straight through. So everything reaches `extra`; only the listed keys are read; the rest fall on the floor.

## 3. The smoking gun — a typed sibling already implements the dropped keys

This is the part that makes the fix tractable and the defect clear. The chart-level `.configure_axis(...)` / `.configure_legend(...)` path does **not** use an opaque map. It deserializes into a fully typed serde struct, `render/chart_config.rs:119-169`:

```rust
pub struct AxisConfigSpec {
    label_angle, label_font_size, label_color, label_format, label_overlap,
    tick_count, tick_size, domain, domain_color, domain_width,
    grid, grid_color, grid_dash, grid_width, domain_min, domain_max,
    nice, zero, tick_values, title_font_size, title_color, title_padding,
    label_format_raw, label_padding,   // 24 named fields
}
pub struct LegendConfigSpec { orient, direction, columns, title_font_size,
    label_font_size, symbol_size, offset, padding, symbol_type, gradient_length }
```

These fields are genuinely consumed downstream (e.g. `layout/axis.rs:571,1091` map `title_color`/`title_padding` into the rendered axis; grid color/dash/width flow through the grid pipeline).

So `label_color`, `grid_color`, `grid_dash`, `grid_width`, `domain_color`, `domain_width`, `title_color`, `title_padding`, `tick_values`, `label_overlap`, `label_font_size` (axis) and `symbol_size`, `symbol_type`, `offset`, `padding` (legend) are **already rendered** when set at chart level. The per-channel path simply never routes into that machinery. There is **no code anywhere that folds `AxisSpec.extra` into `AxisConfigSpec`** (verified — the only "merge" in `prepare.rs` is for data batches and same-field color/size legends). The two paths run in parallel and never meet.

The fix is therefore not "implement grid_color rendering" — it exists. It is "give the per-channel path the same typed deserialize, and merge it into the per-axis/per-legend config at the right cascade level (per-channel beats configure)."

## 4. Precise emitted-vs-consumed catalog

**Axis** — `fm.Axis` declares 32 params (31 cross to Rust; `label_map` is Python-only, applied in the Python layer). Rust honors 9. **Dropped per-channel (~22):** `orient`, `tick_extra`, `tick_min_step`, `grid_dash`, `grid_width`, `grid_color`, `grid_opacity`, `label_flush`, `label_overlap`, `label_font_size`, `label_color`, `domain_width`, `domain_color`, `offset`, `translate`, `min_extent`, `max_extent`, `title_orient`, `title_font_size`, `title_color`, `title_padding`, `values` (explicit ticks), `zindex`.

**Legend** — `fm.Legend` declares 26 params. Rust honors ~13. **Dropped per-channel (~13):** `tick_min_step`, `format_type`, `label_color`, `label_limit`, `symbol_size`, `symbol_stroke_width`, `symbol_type`, `column_padding`, `row_padding`, `clip_height`, `title_padding`, `offset`, `padding`, `zindex`. (Verified absent: no `extra.get` for any of `format_type`, `label_color`, `symbol_size`, `symbol_type`, `label_limit`, `padding`, `offset`, `zindex`, `tick_min_step`.)

## 5. Why it was never caught — tests assert serialization, not render (same pattern as B4)

`tests/test_phase_12_axis_legend.py` is the per-channel coverage, and every assertion stops at `.to_dict()`:

```python
def test_grid_styling(self):                              # line 50
    result = Axis(grid_dash=[4.0, 2.0], grid_width=0.5, grid_color="#ccc").to_dict()
    assert result == {"grid_dash": [4.0, 2.0], "grid_width": 0.5, "grid_color": "#ccc"}
def test_values_explicit(self): ...                       # line 62 — to_dict only
def test_symbol_options(self): ...                        # line 125 — to_dict only
```

None of these calls `.to_svg()`. They prove the Python dataclass serializes its fields, which is true and irrelevant — the field is dropped one layer down. The same properties are tested **and verified at render** when routed through `AxisConfig` (chart-level): `test_bug_hunt_composition_facet.py:2927-2954` (grid_color/domain_width/title_color), `test_configure_integration.py:87` (title_color → SVG). So the suite simultaneously certifies that these properties render (chart-level) and that their per-channel twin serializes (no render check), and the gap between the two is exactly the bug. CI stays green.

## 6. Provenance

1. **`31ceee4` "feat(spec): EncodingSpec gains scale/title (honored) + 6 deferred kwargs"** introduced `AxisSpec`/`LegendSpec` as opaque round-trip blobs. The commit body itself classifies `scale`/`title` as *honored* and `axis`/`legend`/`sort`/`stack`/`impute`/`scheme`/`format` as **"deferred kwargs"**; the struct doc says *"renderer ignores in 8a."* So per-channel styling was deferred-by-design at the spec layer.
2. **Phase 12** then built `Axis`/`Legend` as full value classes (~32/~26 fields, complete NumPy docstrings) — advertising the whole capability with no caveat that most fields are inert.
3. The deferred keys were never given a typed struct. They were hand-wired piecemeal as later features needed them: **`73b9964` "wire 63 disconnected pipeline fields end-to-end"** added the axis block (D7/D12), the flexibility campaign added D3 (`tick_count`) and D13 (the legend block), and Schwabish SB3 added `disabled`/`format`/`tickLabels`. Each pass wired exactly the keys it needed and left the rest.

The root-cause shape is identical to B4 (`override`): a feature designed in two halves, where the storage/serialization half shipped with passing tests and docs, and the render-consumption half was partially built or omitted, with no doc-vs-render liveness gate to expose the difference.

## 7. Correction to the B5 tracker note

The archaeology entry (written 2026-05-15) says Rust reads *"~3 legend keys (`disabled`, `title`, `format`)."* That is **stale** — the D13 flexibility-campaign legend wiring (`prepare.rs:280-356`) landed afterward and now reads ~14 legend keys. The axis count (~7) is roughly right (actually 9). The note also frames B5 as purely "unrelated to B4, surfaced during the override investigation," which is accurate, but the sharper framing is that B5 and B4 are **the same silent-drop archetype** (serialize-half shipped, consume-half partial, tests assert storage). The defect itself stands; only the legend coverage count needs updating.

## 8. Where a fix goes

1. **Replace the opaque maps with typed structs.** Give `AxisSpec`/`LegendSpec` named `Option<…>` fields mirroring the Python value classes (the field set already exists in `AxisConfigSpec`/`LegendConfigSpec` — reuse or share it), with serde aliases for the camelCase/snake_case pairs the current code juggles by hand.
2. **Route per-channel into existing consumption.** In `prepare.rs`, fold each channel's axis/legend spec into the same per-axis/per-legend config the chart-level `configure` path feeds, at cascade level **per-channel > configure > theme > default**. This deletes the ad-hoc `extra.get("…")` ladder and lights up all ~22 axis + ~13 legend dropped keys at once, since their rendering already exists.
3. **Render-level tests.** Extend `test_phase_12_axis_legend.py` from `.to_dict()` assertions to `.to_svg()` assertions for the previously-dropped keys (e.g. `Axis(label_color="#ff00ff")` → magenta label fill in SVG; `Axis(grid_color="#ccc")` → grid stroke; `Legend(symbol_size=…)` → symbol geometry), plus a precedence test that per-channel beats a conflicting `configure_axis`.
4. **Honesty caveat meanwhile:** the `EncodingSpec` Rust docstring (`encoding.rs:315-322`) already lists only the small honored set; the **Python** `Axis`/`Legend` docstrings do not warn that most fields are currently inert per-channel. That mismatch is what misleads users.

Per project rules ("do the work now, do it the right way; NotImplementedErrors not acceptable") the right remediation is (1)+(2) — full typed wiring — not narrowing the Python docstrings to match the truncated set. A do-nothing / partial-wire stub is not acceptable.

## 9. Key references

- Opaque specs: `crates/ferrum-core/src/spec/encoding.rs:285-296`
- Sole consumer (axis): `crates/ferrum-core/src/render/prepare.rs:511-577`, `705-736`
- Sole consumer (legend D13): `crates/ferrum-core/src/render/prepare.rs:280-356`
- Typed sibling that already works: `crates/ferrum-core/src/render/chart_config.rs:119-169` → `crates/ferrum-core/src/layout/axis.rs:571,1091`
- Python value classes (full field set): `src/ferrum/axis.py:28-178`, `src/ferrum/legend.py:19-138`
- Serialization path: `src/ferrum/encoding/base.py:154-174`
- Storage-only tests: `tests/test_phase_12_axis_legend.py`
- Provenance: `31ceee4` (deferred), `73b9964` + D3/D13 + SB3 (piecemeal wiring)
- Sibling RCA (same archetype): `design-docs/superpowers/followups/2026-06-14-override-silent-drop-rca.md`
- Tracker entry: `design-docs/superpowers/followups/2026-05-15-code-archaeology.md` (item B5)
