# Refactor Roadmap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use chris-code:subagent-driven-development to implement this plan task-by-task. Rust → `rust-coder`; Python → `python-coder`; never general-purpose.

## 1. Objective

Apply all 11 open cohesion findings from this session's two heavyweight reviews (rust-review F2–F5, F8; python-review R1, R2, R3b, R5, R6, R7), preserving behavior except where a gated item makes an approved public-API/wire change.

## 2. Spec references

The roadmap **is** the spec — the two heavyweight reviews. Per-item detail (problem / proposed fix / risk) lives in the consolidated table below; each row is self-contained.

| ID | Sev | Files | Problem → Fix | Gate |
|----|-----|-------|---------------|------|
| F3 | S3 | `render/marks/bar.rs`, `render/draw.rs` | Continuous color re-parses f64→str→f64 (point uses `col_as_f64`→`lookup_f64`); 4 byte-identical dead match arms. → Shared `resolve_fill_color` in draw.rs; load `col_as_f64` for continuous. | internal |
| F4 | S2–S3 | `render/marks/{rule,segment,line,tick}.rs`, `render/draw.rs` | Per-row stroke precedence copy-pasted rule≡segment; per-row channel loading hand-rolled. → Extract `resolve_stroke_color(ms,row_color)` in draw.rs; adopt bar's `StrokeChannels` across the marks. | internal |
| F5 | S3 | `render/scale_resolve/color.rs` | `build_color_scale` builds default categorical palette twice (byte-identical); silent `Ok(_)=>{}` dead branch. → Extract `build_default_categorical_scale`; delete dead branch. | internal |
| F8 | S2 | `render/prepare.rs`, `render/format.rs` | `apply_format_spec` is a second weaker number formatter (only `.Nf`/`.N%`) for colorbar ticks; size-legend uses full `format_with_spec`. → Delegate `apply_format_spec`→`format_with_spec`; delete mini-parser. | internal; verify colorbar goldens |
| R6 | S1 | `src/ferrum/chart.py` | `_infer_type_from_data` apply-block duplicated at 2 call sites. → Extract `_apply_inferred_type(d, field, data)`. | internal |
| R5 | S2 | `src/ferrum/chart.py`, `src/ferrum/encoding/base.py` | Private `_kwargs["sort"]/["axis"]` reach-ins. → Add `ChannelBase.option(name)` accessor; swap reach-ins. | internal (additive) |
| R7 | S1 | `src/ferrum/composition.py` | `_promote_layer_color` aliases a shared `ChannelBase` into chart encoding (mutation hazard). → Reconstruct the channel instead of aliasing. | internal |
| R1 | S3 | `src/ferrum/chart.py`, `src/ferrum/_marks_statistical.py` | "Which axis is categorical" decided 3× with drifting `horizontal` vs `orient`; `desugar_errorbar` ignores `y_sort` (dead param, subsumes R4). → One helper → `(cat_field, cat_sort, val_field)`; all desugars + boxen call it. | internal |
| F2 | S3 | `spec/scale*` (`ScaleSpec::Ordinal.range`), `render/scale_resolve/{positional,color}.rs`, `scale/ordinal.rs`, PyO3 binding | `range: serde_json::Value` JSON-sniffed at 3 sites. → Typed `Option<Vec<OrdinalRangeValue>>`; collapse sniffers to one match. | **GATED — wire format** |
| R2 | S3 | `crates/ferrum-core/src/transform/letter_value.rs`, `src/ferrum/chart.py` | Boxen aggregation is a Python layer-violation existing only because `LetterValue` drops the value column. → `LetterValue` retains/re-emits the column; delete `_resolve_boxen_cat_sort`. | **GATED — Rust transform schema** |
| R3b | S2 | `src/ferrum/plots/distribution.py` (`catplot`), shared `_apply_facet_sizing` | `catplot` lacks `height`/`aspect`/per-panel facet sizing that `displot` has. → Add params; extract shared sizing helper. | **GATED — new public params** |

## 3. Files

| Action | Path | Reason |
|--------|------|--------|
| Modify | `crates/ferrum-core/src/render/draw.rs` | F3+F4 shared `resolve_fill_color`/`resolve_stroke_color` |
| Modify | `crates/ferrum-core/src/render/marks/{bar,rule,segment,line,tick}.rs` | F3/F4 adopt helpers |
| Modify | `crates/ferrum-core/src/render/scale_resolve/color.rs` | F5 extract default-palette builder; F2 typed range |
| Modify | `crates/ferrum-core/src/render/{prepare,format}.rs` | F8 fold formatter |
| Modify | `crates/ferrum-core/src/spec/encoding.rs`, `scale/ordinal.rs`, `render/scale_resolve/positional.rs` | F2 typed ordinal range + sniffer collapse |
| Modify | `crates/ferrum-core/src/transform/letter_value.rs` | R2 retain value column |
| Modify | `src/ferrum/chart.py` | R6/R5/R1 helpers; R2 delete `_resolve_boxen_cat_sort` |
| Modify | `src/ferrum/encoding/base.py` | R5 `option()` accessor |
| Modify | `src/ferrum/composition.py` | R7 reconstruct channel |
| Modify | `src/ferrum/_marks_statistical.py` | R1 desugars call shared sort helper |
| Modify | `src/ferrum/plots/distribution.py` | R3b catplot params + facet sizing |
| Modify | `ferrum-spec.md` | F2 wire-format dated note; R3b catplot signature |
| Test | `tests/test_flexibility_campaign.py` | add a regression section per applied item |
| Test | `tests/goldens/**`, regen + visual PNG inspect | F8 colorbar, F2 ordinal-range, F5 palette goldens if bytes shift |

## 4. Constraints

- **Behavior-preserving except the 3 gated items.** Internal items must be byte-stable on existing goldens (categorical paths) or correct-by-construction (F3 continuous). If any golden's bytes shift, regenerate AND visually inspect the PNG per CLAUDE.md before blessing — never bless a byte-diff alone.
- **F2 is a serialized wire-format change.** `ScaleSpec` is `Serialize`/`Deserialize` + round-trip tested. Add a dated note to `ferrum-spec.md`, keep `from_json(to_json())` round-trip green, regenerate any ordinal-range golden.
- **R2 changes a Rust transform's output schema.** `LetterValue` must re-emit the original category column so the standard Rust field-sort applies; only then delete the Python `_resolve_boxen_cat_sort` shim. Boxen output must stay identical for all backends (polars/pandas/pyarrow).
- **R3b adds public `catplot` params.** Mirror `displot`'s `height`/`aspect` names/semantics exactly (sibling parity); update the spec signature.
- `cargo test -p ferrum-core` on this machine needs the full env recipe (PYO3_PYTHON absolute path + PYTHONHOME + RUSTFLAGS `-L base/lib` + DYLD_LIBRARY_PATH; `unset CONDA_PREFIX PYTHONPATH`). See `project_flexibility_campaign` memory.
- Each task is TDD-first and passes `/regression-test` before its commit. Use `commit-commands:commit`.

## 5. Tasks

Staging respects file overlap: draw.rs serializes F3→F4; chart.py serializes R6→R5→R1→R2.

### Stage 1 — Rust internal (parallel; disjoint files)
### Task F5: extract default categorical palette builder
- [ ] Failing/characterization test: default palette identical before/after.
- [ ] Extract `build_default_categorical_scale`; call from both arms; remove `Ok(_)=>{}` dead branch in `color.rs`.
- [ ] Verify: `cargo test -p ferrum-core` (env recipe).

### Task F8: fold colorbar formatter into format_with_spec
- [ ] Test: colorbar tick labels match `format_with_spec` output for `.Nf`/`.N%` and a previously-unsupported spec (e.g. `,`/`~s`).
- [ ] Delegate `prepare.rs::apply_format_spec`→`format::format_with_spec`; delete mini-parser.
- [ ] Regen + visually inspect any colorbar golden whose bytes shift.
- [ ] Verify: `cargo test -p ferrum-core`.

### Stage 2 — Rust draw.rs (serialized: F3 then F4, one coder, one coherent change)
### Task F3+F4: shared fill/stroke color resolution
- [ ] Tests: bar continuous color byte-stable; catplot whisker/rule/segment stroke precedence unchanged (regression guard for the `stroke_is_user_set` path).
- [ ] Extract `resolve_fill_color` (load `col_as_f64` for continuous; drop dead match arms) and `resolve_stroke_color(ms,row_color)` into `draw.rs`; adopt in bar/rule/segment/line/tick; adopt `StrokeChannels` where hand-rolled.
- [ ] Verify: `cargo test -p ferrum-core`; full pytest golden suite byte-stable.

### Stage 3 — Python internal (serialized on chart.py: R6 → R5 → R1; R7 parallel on composition.py)
### Task R7: reconstruct promoted layer color channel
- [ ] Test: `_promote_layer_color` no longer aliases (mutating the promoted channel does not affect the layer's channel).
- [ ] Reconstruct the `ChannelBase` instead of aliasing.
- [ ] Verify: `uv run pytest tests/test_flexibility_campaign.py -k d2`.

### Task R6: extract `_apply_inferred_type`
- [ ] Test: inferred-type behavior unchanged at both call sites.
- [ ] Extract helper; call from both sites.

### Task R5: `ChannelBase.option()` accessor
- [ ] Test: `option("sort")`/`option("axis")` return the same values the `_kwargs` reach-ins did.
- [ ] Add accessor in `encoding/base.py`; swap reach-ins in `chart.py`.

### Task R1: consolidate sort/axis helper (subsumes R4)
- [ ] Test: each of the 5 desugars + boxen produces identical sort/axis resolution; remove dead `y_sort` from `desugar_errorbar`.
- [ ] One helper returning `(cat_field, cat_sort, val_field)` honoring `horizontal`/`orient`; all callers use it.
- [ ] Verify: `uv run pytest -n auto`.

### Stage 4 — GATED items (checkpoint before each; sequence after Stages 1–3)
### Task F2: typed ordinal range (wire format)
- [ ] **Checkpoint:** confirm wire-format change approved; add dated note to `ferrum-spec.md`.
- [ ] Replace `ScaleSpec::Ordinal.range: serde_json::Value` with `Option<Vec<OrdinalRangeValue>>`; collapse the 3 sniffers; update PyO3 binding.
- [ ] Verify: `from_json(to_json())` round-trip; `cargo test -p ferrum-core`; regen + inspect ordinal-range golden.

### Task R2: LetterValue retains value column (Rust transform schema)
- [ ] **Checkpoint:** confirm transform-schema change approved.
- [ ] `LetterValue` re-emits the original category column; verify standard Rust field-sort applies; delete Python `_resolve_boxen_cat_sort`.
- [ ] Verify: boxen output identical across polars/pandas/pyarrow; `cargo test -p ferrum-core`; `uv run pytest -k boxen`.

### Task R3b: catplot facet sizing (public params)
- [ ] **Checkpoint:** confirm `height`/`aspect` param names mirror `displot`.
- [ ] Add params; extract shared `_apply_facet_sizing` used by catplot + displot; update spec signature.
- [ ] Verify: `uv run pytest -k "catplot or displot"`.

## 6. Acceptance checks

- `cargo test -p ferrum-core` (env recipe) — all pass.
- `uv run pytest -n auto` — all pass; goldens byte-stable except deliberately-regenerated F8/F2/F5 goldens (each visually inspected via `scripts/snapshot-goldens.py`).
- `from_json(to_json())` ChartSpec round-trip green after F2.
- Boxen renders identically across all three DataFrame backends after R2.
- `ferrum-spec.md` carries dated notes for F2 and the R3b catplot signature.
- New regression section per applied item in `tests/test_flexibility_campaign.py`.
