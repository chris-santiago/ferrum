# Packed Tooltip + Data-Index Design Spec

## 1. Scope

Pack tooltip and data_indices arrays for high-cardinality interactive mark batches so they travel as compact binary alongside the packed instance bytes, not as per-element JSON. After the binary instance bridge landed, these two arrays are the remaining bottleneck: tooltips (23MB) + data_indices (1.4MB) = 24.4MB of JSON for a 200k scatter. This spec eliminates that overhead.

## 2. Goals

- Tooltip data for packed batches (>1000 marks) travels as binary bytes, not JSON.
- The scene JSON for a 200k-point scatter with 3 tooltip fields drops from ~23MB to <2MB.
- `data_indices` (1.4MB JSON for 200k points) is also packed as binary (800KB raw u32).
- Total Python-side render time for 200k interactive scatter drops from ~1.7s to <0.5s.
- Tooltip hover behavior is unchanged — user sees the same field names and values.
- Charts below the packing threshold are unaffected.

## 3. Non-goals

- Changing the tooltip API or encoding interface.
- Packing tooltips for non-packed batches (small charts keep per-element JSON).
- WASM-side tooltip rendering changes (JS tooltip display logic is unchanged).

## 4. System behavior

### When packing fires

Tooltip packing fires on the same batches that instance packing fires (>1000 homogeneous circle/rect nodes). When a batch's `packed_instances` is extracted, its `tooltips` array is also extracted into the packed binary stream and cleared from the JSON.

### Binary tooltip format

Tooltip values are pre-formatted strings. The packed format uses a string table approach:

1. **Field names header**: `[num_fields: u32][len0: u32][name0 bytes][len1: u32][name1 bytes]...`
2. **Per-row values**: for each row, for each field: `[len: u32][value bytes]`

All strings are UTF-8. Lengths are byte counts, little-endian u32.

### JS tooltip lookup

When the JS hit-test identifies a node index `idx` in a packed batch, it looks up tooltip data from the packed binary instead of `batch.tooltips[idx]`. The WASM module exposes a `getTooltip(panel_id, batch_idx, node_idx) -> String` method that returns a JSON string for the single requested tooltip (lazy per-hit, not upfront deserialization).

### Fallback

If tooltip unpacking fails (corrupt data, version mismatch), the batch has no tooltips — hover shows nothing. This is acceptable for >1000-mark batches where per-element tooltips are already marginal.

## 5. Architecture

The tooltip bytes are appended to the same `packed_data` byte stream that carries instance data. The batch header gains a tooltip-data-present flag.

**Extended batch header** (20 bytes):
```
[panel_idx: u32][batch_idx: u32][kind: u32][count: u32][flags: u32]
```
Flags: bit 0 = has tooltip data, bit 1 = has data_indices.

**Data layout per packed batch**:
```
[header: 20 bytes]
[instance_data: count × instance_size bytes]
[if flags & 0x2: data_indices: count × u32 bytes]
[if flags & 0x1: tooltip_data]
```

**Tooltip data layout**:
```
[num_fields: u32]
[field_name_0_len: u32][field_name_0_bytes]
...
[field_name_N_len: u32][field_name_N_bytes]
[row_0_field_0_len: u32][row_0_field_0_bytes]
[row_0_field_1_len: u32][row_0_field_1_bytes]
...
[row_count-1_field_N_len: u32][row_count-1_field_N_bytes]
```

**WASM side**: Stores the raw tooltip bytes per packed batch. On `getTooltip(panel, batch, idx)`, seeks to the correct row offset, reads field values, and returns a JSON string matching the existing `TooltipContent` format.

**JS side**: When hit-testing finds a hit in a packed batch (empty nodes array), calls `renderer.getTooltip(panel, batch, idx)` instead of reading `batch.tooltips[idx]`.

## 6. Canonical interfaces

Extended batch header flag:
```
const HAS_TOOLTIPS: u32 = 0x1;
```

WASM method:
```rust
#[wasm_bindgen(js_name = "getTooltip")]
pub fn get_tooltip(&self, panel_id: u32, batch_idx: u32, node_idx: u32) -> Result<String, JsValue>
```

Returns `"{}"` when no tooltip data exists for the index.

## 7. Invariants and constraints

- **Backward compatible.** Batches without `flags` bit 0 work exactly as before.
- **Lazy decoding.** Tooltip strings are only deserialized on hover, not upfront. 200k tooltip entries are never all decoded — only the one under the cursor.
- **Same visual behavior.** The tooltip HTML table shows identical field names and values.
- **The packed_data byte stream is self-describing.** Each batch header includes its own size information so the WASM decoder can skip batches it doesn't need.

## 8. Key decisions and tradeoffs

**String table, not columnar arrays.** Tooltip values are heterogeneous pre-formatted strings (some numeric "0.3047", some text "hello"). A columnar approach (separate arrays per field) would save ~10% via shared field-name dedup but adds complexity. The string table is simpler and the savings from eliminating JSON structure overhead (key names, quotes, braces, commas) are already >80%.

**Lazy WASM lookup, not bulk JS decode.** Decoding all 200k tooltips in JS at load time would just move the bottleneck. Instead, the WASM module stores the raw bytes and decodes one tooltip on demand per hover event (<0.01ms per lookup).

**Extend packed_data, not a third trait.** Adding tooltip bytes to the existing `packed_data` stream avoids a third anywidget trait and keeps the transfer atomic.

## 9. Acceptance criteria

1. 200k scatter with 3 tooltip fields: scene JSON <2MB (was 22MB).
2. Python-side `_render_scene` completes in <0.5s for 200k points with tooltips.
3. Hovering a point in the notebook shows correct tooltip fields.
4. Charts with <1000 points: tooltip behavior unchanged.
5. Charts with no tooltip encoding: no regression.
6. All existing interactive regression tests pass.

## 10. Validation strategy

- Profile `_render_scene` on 200k scatter with tooltips — assert JSON <2MB.
- Existing `test_interactive_regression.py` tooltip tests verify small-chart behavior.
- Manual notebook test: hover 200k scatter, confirm tooltip displays.
