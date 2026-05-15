# Packed Tooltips + Data-Indices Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

## 1. Objective

Extend the binary sidecar stream to include tooltip string tables and data_indices arrays, eliminating the remaining 24MB of JSON overhead for 200k-point interactive charts.

## 2. Spec references

- `docs/superpowers/specs/2026-05-15-packed-tooltips-design.md` — full spec
- `docs/superpowers/specs/2026-05-15-packed-tooltips-design.md §5` — binary layout with flags, data_indices, tooltip string table

## 3. Files

| Action | Path | Reason |
|--------|------|--------|
| Modify | `crates/ferrum-core/src/render/pack_instances.rs` | Extend header to 20 bytes (add flags), pack tooltips + data_indices |
| Modify | `crates/ferrum-wasm/src/scene_load.rs` | Parse extended header, store tooltip bytes per batch |
| Modify | `crates/ferrum-wasm/src/lib.rs` | Add `getTooltip(panel, batch, idx)` WASM method |
| Modify | `src/ferrum/_wasm/ferrum-anywidget.js` | Call `renderer.getTooltip()` for packed batches |
| Test | `tests/test_interactive_regression.py` | Tooltip correctness for packed batches |

## 4. Constraints

- **Header format change is breaking** — both Rust packer and WASM unpacker must be updated atomically. The header goes from 16 bytes (4 × u32) to 20 bytes (5 × u32, adding flags).
- **Lazy tooltip decoding.** WASM stores raw tooltip bytes; decodes one row on demand per `getTooltip` call. Never bulk-decode 200k tooltips.
- **Backward compatible for small charts.** Batches below 1000 marks are unaffected — tooltips stay in JSON.
- **Clear tooltips and data_indices from JSON** for packed batches (same as nodes are already cleared).

## 5. Tasks

### Task 1: Extend binary header + pack data_indices and tooltips (Rust)
- [ ] Change header from 16 to 20 bytes: add `flags: u32` after `count` (spec §5)
- [ ] Flag bit 0x1 = has tooltips, bit 0x2 = has data_indices
- [ ] After instance bytes: if batch has `data_indices`, write `count × u32` and set flag 0x2, then clear `batch.data_indices`
- [ ] After data_indices: if batch has `tooltips`, write string table (spec §5 layout), set flag 0x1, then clear `batch.tooltips`
- [ ] Update existing tests for 20-byte header
- [ ] Verify: `source ~/.cargo/env && cargo check -p ferrum-core`

### Task 2: Parse extended header + store tooltip bytes (WASM)
- [ ] Update `unpack_binary_instances` to read 20-byte header with flags
- [ ] If flags & 0x2: read `count × u32` data_indices after instances (store per-batch for selection)
- [ ] If flags & 0x1: read remaining bytes as tooltip data, store per packed batch (keyed by `(panel_idx, batch_idx)`)
- [ ] Add `PackedBatchData` struct to hold tooltip bytes + data_indices per batch
- [ ] Store in `LoadedScene` (or `SceneData`) so `getTooltip` can access it
- [ ] Verify: `source ~/.cargo/env && cargo test -p ferrum-wasm --target aarch64-apple-darwin`

### Task 3: Implement `getTooltip` WASM method
- [ ] Add `get_tooltip(panel_id, batch_idx, node_idx) -> String` to `WasmRenderer` in lib.rs (spec §6)
- [ ] Seek into tooltip string table: skip field names header, then skip `idx × num_fields` value entries to reach the target row
- [ ] Read `num_fields` values, construct `{"fields":[{"name":"x","value":"0.30"},...]}`
- [ ] Return `"{}"` when no tooltip data exists
- [ ] Verify: `source ~/.cargo/env && cargo test -p ferrum-wasm --target aarch64-apple-darwin -- get_tooltip`

### Task 4: Wire JS tooltip lookup
- [ ] In `ferrum-anywidget.js`, when hit-test finds a hit in a batch with empty `nodes`:
  - Call `renderer.getTooltip(panelId, batchIdx, nodeIdx)` instead of reading `batch.tooltips[idx]`
  - Parse the returned JSON string and display as before
- [ ] Keep existing path for non-packed batches (nodes array present)
- [ ] Verify: rebuild WASM + manual notebook test

### Task 5: Tests + profile
- [ ] Add test: 200k scatter with tooltips → scene JSON <2MB
- [ ] Add test: `_render_scene` completes in <0.5s for 200k with tooltips
- [ ] Add test: small chart (100 rows) tooltip behavior unchanged
- [ ] Verify: `uv run --no-sync pytest tests/test_interactive_regression.py -v`
- [ ] Verify: `uv run --no-sync pytest -x -q` (all tests pass)

## 6. Acceptance checks

- `source ~/.cargo/env && cargo test -p ferrum-wasm --target aarch64-apple-darwin` — all pass
- `uv run --no-sync pytest -x -q` — 2044+ pass, 0 fail
- 200k scatter with 3 tooltip fields: JSON <2MB, total render <0.5s
- Existing tooltip regression tests pass unchanged
