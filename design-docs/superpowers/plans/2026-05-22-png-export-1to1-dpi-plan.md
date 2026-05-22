# PNG Export: 1:1 Capture with DPI Metadata — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task.

## 1. Objective

Replace the DPR-upscale PNG export with a 1:1 pixel capture that injects a PNG `pHYs` chunk for DPI metadata, fixing gridline artifacts in composed charts and producing WYSIWYG-sized files.

## 2. Spec references

- No design spec — this plan is the spec. Motivated by: grid lines pixel-snapped once at scene_load (scene_load.rs:405-427) become misaligned when the canvas is resized for DPR capture, causing uneven gridline spacing in HConcat charts.

## 3. Files

| Action | Path | Reason |
|--------|------|--------|
| Modify | `src/ferrum/_wasm/ferrum-anywidget.js` | Rewrite `onSave()` — remove DPR upscale, add pHYs injection |
| Modify | `crates/ferrum-wasm/src/lib.rs` | Remove `maxTextureSize()` export (dead code after this change) |
| Test | Manual: export PNG from linked-views HConcat, verify gridlines even and dimensions 1:1 | |

## 4. Constraints

- **No canvas resize during export.** The grid mesh is tessellated once at scene_load with pixel-center snapping for the original canvas size. Any resize invalidates those snap positions.
- **pHYs chunk must precede IDAT.** PNG spec requires pHYs before the first IDAT chunk. Insert after IHDR (always 33 bytes from file start: 8-byte signature + 25-byte IHDR chunk).
- **SVG text composite stays at 1:1.** No viewBox scaling needed — svgClone width/height match canvas exactly.
- **ResizeObserver disconnect is still needed** during the async save (toBlob + compositing) to prevent layout-triggered canvas resizes mid-capture.
- `maxTextureSize()` in lib.rs becomes dead code — remove it to keep the API clean.

## 5. Tasks

### Task 1: Rewrite `onSave()` in ferrum-anywidget.js

- [ ] Remove all DPR-upscale logic: `dpr`, `maxTex`, `maxScale`, `captureScale`, `captureW`, `captureH` variables and all `if (captureScale > 1)` branches (lines 291-299, 301-305, 319-321, 364-370)
- [ ] Remove `origW`/`origH` — no longer needed since canvas size doesn't change
- [ ] Keep: ResizeObserver disconnect/reconnect, `renderFrame()` call, offscreen canvas composite, SVG text overlay (simplified — no viewBox override needed), font-face inlining
- [ ] After SVG composite, convert offscreen canvas to PNG blob via `off.toDataURL('image/png')` (existing)
- [ ] Add `injectPHYs(dataUrl)` helper: decode base64 → Uint8Array, compute pixels-per-meter from `(window.devicePixelRatio || 1) * 72` DPI (formula: `Math.round(dpi * 39.3701)`), build 21-byte pHYs chunk (4-byte length + 4-byte type `pHYs` + 4-byte ppuX + 4-byte ppuY + 1-byte unit=1 + 4-byte CRC), splice after IHDR (byte offset 33), re-encode to data URL
- [ ] Use the pHYs-injected data URL for the download `<a>` tag
- [ ] Verify: open linked-views HConcat in browser, click Save PNG, confirm exported image dimensions match canvas `width×height` exactly and gridlines are evenly spaced

### Task 2: Remove `maxTextureSize()` from WASM renderer

- [ ] Remove `max_texture_size()` method from `crates/ferrum-wasm/src/lib.rs` (lines ~347-353)
- [ ] Verify: `source ~/.cargo/env && cargo clippy -p ferrum-wasm --target wasm32-unknown-unknown -- -D warnings`

### Task 3: Rebuild WASM and regenerate docs demos

- [ ] `source ~/.cargo/env && wasm-pack build crates/ferrum-wasm --target web --out-dir ../../src/ferrum/_wasm/`
- [ ] `uv run python scripts/export-interactive-examples.py`
- [ ] Copy demos 01-06 to `docs/site/assets/demos/`
- [ ] Visually inspect at least one composed-chart demo PNG export

## 6. Acceptance checks

- `source ~/.cargo/env && cargo clippy -p ferrum-wasm --target wasm32-unknown-unknown -- -D warnings` — clean
- Exported PNG from any single chart: pixel dimensions == canvas `width × height`
- Exported PNG from linked-views HConcat: gridlines evenly spaced (no aliasing mismatch between panels)
- PNG file contains a `pHYs` chunk (verify: `python3 -c "import struct; d=open('ferrum-chart.png','rb').read(); i=d.find(b'pHYs'); print(struct.unpack('>II',d[i+4:i+12]) if i>0 else 'MISSING')"`)
