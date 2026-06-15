# Concat Chrome Positioning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: chris-code:subagent-driven-development.

## 1. Objective

Thread a default left inset (`16`) plus `configure_padding(left/right)` and
`configure_title(anchor)` into the figure-chrome emitter so concat AND single-chart
captions/titles stop rendering flush at `x=0`.

## 2. Spec references

- `design-docs/superpowers/specs/2026-06-15-concat-chrome-positioning-design.md` §4–§8
- GitHub issue #1

## 3. Files

| Action | Path | Reason |
|--------|------|--------|
| Modify | `crates/ferrum-core/src/render/figure_chrome.rs` | inset+anchor fields on `FigureChrome`; emit x/text-anchor; pass `panel_w` to `emit_footer`; unit tests |
| Modify | `crates/ferrum-core/src/render/binding.rs` | 3 `compose_svg_*_py` signatures gain `left_inset`/`right_inset`/`anchor`; `DEFAULT_CHROME_INSET` const |
| Create | `src/ferrum/_chrome.py` | shared resolver: merge `_configure_layers` → dict; extract `{left_inset,right_inset,anchor}` chrome kwargs |
| Modify | `src/ferrum/composition.py` | 6 `compose_svg_*` call sites pass resolved chrome kwargs (HConcat, VConcat, 4 grid) |
| Modify | `src/ferrum/_render.py` | 2 single-chart caption `compose_svg_vertical` calls pass chrome kwargs from `chart_config_dict` |
| Modify | `src/ferrum/_core.pyi` | update 3 compose stub signatures |
| Test | `tests/test_flexibility_caps/test_d10_figure_title.py` | render-level x/text-anchor assertions across surfaces |

## 4. Constraints

- **No-chrome byte-stability:** title=subtitle=caption all `None` ⇒ output unchanged. The new params must not perturb `wrap_with_chrome`'s early-return path.
- **Default inset lives in Rust** as `DEFAULT_CHROME_INSET = 16.0` (the `ThemePadding::default().padding` value). Python passes `None` when unset → Rust applies the default. Do not duplicate `16.0` in Python.
- **One figure anchor** governs title + subtitle + caption uniformly. Validate nothing new in Rust; `TitleConfig` already validates `anchor ∈ {start,middle,end}`.
- Geometry: `start → x=left_inset, "start"` · `middle → x=panel_w/2, "middle"` · `end → x=panel_w−right_inset, "end"`. `panel_w` = composed figure width.
- **No new silent drops:** every resolved chrome value reaches the emitter; a value that can't be honored must surface, not vanish.
- Goldens that shift `x:0→16` (concat chrome + single-chart captions) must be regenerated AND visually inspected (rasterize PNG, Read) per CLAUDE.md before commit.
- All Python coding → python-coder; Rust → rust-coder.

## 5. Tasks

### Task 1: Rust emitter + bindings (rust-coder)
- [ ] Add `DEFAULT_CHROME_INSET: f64 = 16.0` (doc-comment its tie to `ThemePadding::default().padding`).
- [ ] Extend `FigureChrome` with `left_inset: f64`, `right_inset: f64`, and an anchor (enum `start|middle|end` or `&str`); keep `Default` reproducing `(16,16,start)`.
- [ ] `emit_header`/`emit_footer`: compute `x` + `text-anchor` per the geometry rule (spec §6); pass `panel_w` into `emit_footer`; stop hardcoding `x="0"`/`text-anchor="start"`.
- [ ] `compose_svg_horizontal_py`/`_vertical_py`/`_grid_py`: append `left_inset: Option<f64>`, `right_inset: Option<f64>`, `anchor: Option<&str>` kwargs (defaults `None`); build `FigureChrome` with `unwrap_or(DEFAULT_CHROME_INSET)` / `unwrap_or("start")`. Invalid anchor → `PyValueError`.
- [ ] Unit tests: x/text-anchor for start/middle/end; non-default inset; no-chrome round-trip byte-identical; caption honors anchor+inset.
- [ ] Verify: `nox -s cargo_test` (or the DYLD `cargo test -p ferrum-core`) + `cargo clippy -p ferrum-core -- -D warnings`.

### Task 2: Python resolver + wiring + tests (python-coder)
> Prereq: Task 1 built via `maturin develop` (orchestrator rebuilds between tasks).
- [ ] Create `src/ferrum/_chrome.py`: `merge_configure_layers(layers) -> dict` (mirror `_render.py` `_resolve_chart_config` merge) and `chrome_kwargs(merged: dict) -> dict` returning only the set keys among `left_inset`/`right_inset`/`anchor` (from `merged["padding"]["left"/"right"]`, `merged["title"]["anchor"]`).
- [ ] `composition.py`: each of the 6 `compose_svg_*` calls passes `**chrome_kwargs(merge_configure_layers(self._configure_layers))`.
- [ ] `_render.py`: both single-chart caption `compose_svg_vertical` calls pass `**chrome_kwargs(chart_config_dict)`.
- [ ] Update `_core.pyi` stubs for the 3 compose functions.
- [ ] Render-level tests in `test_d10_figure_title.py`: parse SVG, assert title+caption `x`/`text-anchor` for (default→16/start), (`configure_padding(left=60,auto=False)`→60), (`configure_title(anchor="middle")`→panel_w/2/middle) on HConcat, VConcat, a grid composite, and a single chart with caption. Assert overrides are not dropped.
- [ ] Regenerate + visually inspect affected goldens (`python scripts/snapshot-goldens.py`, Read PNGs).
- [ ] Verify: `uv run pytest tests/test_flexibility_caps/test_d10_figure_title.py -v` + full `uv run pytest -n auto`.

## 6. Acceptance checks

- `uv run pytest -n auto` — all pass.
- `cargo test -p ferrum-core` — all pass; `cargo clippy -D warnings` clean.
- Default concat + single-chart caption render at `x≈16`/start; `configure_padding(left=60)`→60; `configure_title(anchor=middle)`→centered, across HConcat/VConcat/grid/single.
- No-chrome composites byte-identical to pre-fix.
- `/regression-test` invoked before done.

## 7. Open questions

None.
