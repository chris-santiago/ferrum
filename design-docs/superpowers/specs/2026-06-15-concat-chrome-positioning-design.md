# Concat Chrome Positioning Design Spec

> GitHub issue #1. Fixes the concat-level figure title/subtitle/caption that render
> flush to the left edge (`x=0`) and ignore `configure_padding(left=…)` /
> `configure_title(anchor=…)`. Two halves: a **default-spacing** bug and a
> **silent-override-drop** bug, same archetype as the B4/B5 campaign.

## 1. Scope

Make the figure-level chrome (title, subtitle, caption) emitted via
`render/figure_chrome.rs` honor a sensible default left inset and the two
documented positioning knobs: `configure_padding(left=…, right=…)` and
`configure_title(anchor=…)`. The chrome is emitted at the SVG-string level,
post-composition; this spec threads inset + anchor into that emitter and into the
three `compose_svg_*` PyO3 bindings, then resolves the values on the Python side.

This chrome emitter is shared by two surfaces, both of which currently emit at
`x=0`:
- **Composite charts** (`HConcatChart`, `VConcatChart`, `ConcatChart`, and the
  grid-based composites `JointChart`, `FacetGrid`/repeat, `ClusterMap`) — resolved
  from each composite's own `_configure_layers`.
- **Single-chart captions** — a single `Chart.properties(caption=…)` is rendered by
  wrapping its SVG through `compose_svg_vertical([svg], caption=…)`
  (`_render.py:662,679`), so its caption goes through the same emitter and is also
  at `x=0` today (while the single-chart title is at `x=16` via the layout engine).
  Resolved from the chart's already-computed `chart_config_dict`.

Including the single-chart caption path is required for coherence: fixing only the
composite path would leave single-chart captions at `x=0`, creating a new
single-vs-concat inconsistency. Both surfaces share the same resolver.

## 2. Goals

- A concat figure title/subtitle/caption carries a **default left inset of `16.0`**
  (the `ThemePadding::default().padding` value the single-chart title uses), instead
  of `x=0`.
- `configure_padding(left=N)` set on a composite repositions the chrome's start
  inset to `N`; `configure_padding(right=N)` sets the end inset (for `end` anchor).
- `configure_title(anchor="start"|"middle"|"end")` set on a composite anchors
  **all three** chrome lines (title, subtitle, caption) uniformly:
  - `start`  → `x = left_inset`, `text-anchor="start"`
  - `middle` → `x = panel_w / 2`, `text-anchor="middle"`
  - `end`    → `x = panel_w − right_inset`, `text-anchor="end"`
  where `panel_w` is the composed figure width.
- The values win over nothing else (chrome has no competing source); they simply
  stop being dropped. No new silent drops introduced.
- Byte-stability preserved for the no-chrome case (all of title/subtitle/caption
  `None` → output unchanged).

## 3. Non-goals

- Interactive/WASM composite chrome positioning (the issue is static SVG/PNG; the
  interactive composite path does not emit this chrome band today and is out of
  scope here).
- Per-line independent anchors (separate caption anchor knob). One figure anchor
  governs all three lines; a separate caption-anchor config is not added.
- `configure_padding` top/bottom affecting chrome vertical bands (vertical paddings
  for the chrome bands stay the existing constants).
- Reworking single-chart title placement (already correct via the layout engine).

## 4. System behavior

Given a composite built as `(L | R)`:

| Config | Title/subtitle/caption x | text-anchor |
|---|---|---|
| none (default) | `16` | `start` |
| `configure_padding(left=60, auto=False)` | `60` | `start` |
| `configure_title(anchor="middle")` | `panel_w/2` | `middle` |
| `configure_title(anchor="end")` | `panel_w − 16` | `end` |
| `anchor="end"` + `configure_padding(right=40)` | `panel_w − 40` | `end` |

The default case changes current output (`x=0` → `x=16`): existing concat goldens
with figure chrome must be regenerated and **visually inspected** per the goldens
rule. A composite with no figure title/subtitle/caption is byte-identical to today.

## 5. Architecture

Two layers change; the architectural seam (Python composition vs Rust per-chart
layout) is preserved — chrome stays a post-composition SVG-string band, it just
gains inset + anchor inputs.

- **Rust `render/figure_chrome.rs`** — `FigureChrome` gains the resolved geometry
  inputs; `emit_header`/`emit_footer` compute `x` and `text-anchor` from anchor +
  insets + the composed `panel_w` (already passed into `emit_header`; must also be
  passed into `emit_footer`).
- **Rust `render/binding.rs`** — `compose_svg_horizontal_py`, `compose_svg_vertical_py`,
  `compose_svg_grid_py` gain optional `left_inset`, `right_inset`, `anchor` params
  with defaults that reproduce current behavior only when chrome is absent
  (when chrome is present, default inset = `16.0`, anchor = `start`).
- **Python `composition.py`** — a shared helper resolves a composite's own
  `_configure_layers` (the list of `Configure` objects, same merge logic as
  `Chart._resolve_chart_config`) into `(left_inset, right_inset, anchor)`, and each
  composite `to_svg()` passes them to its `compose_svg_*` call. Child-panel config
  injection (`_inject_parent_config`) is unchanged.

## 6. Canonical interfaces / data contracts

Resolved-config extraction (Python side), reading the already-merged dict:

```
padding = merged.get("padding", {})
left_inset  = padding.get("left",  16.0)
right_inset = padding.get("right", 16.0)
anchor      = merged.get("title", {}).get("anchor", "start")  # start|middle|end
```

Chrome geometry (Rust emitter), per chrome line:

```
x, text_anchor = match anchor {
    Start  => (left_inset,            "start"),
    Middle => (panel_w / 2.0,         "middle"),
    End    => (panel_w - right_inset, "end"),
}
```

`FigureChrome` carries `title/subtitle/caption: Option<&str>` (unchanged) plus the
resolved `left_inset: f64`, `right_inset: f64`, and an `anchor` (start/middle/end).
The PyO3 `compose_svg_*` signatures append `left_inset`, `right_inset`, `anchor`
keyword params after `caption`.

## 7. Invariants and constraints

- **No-chrome byte-stability:** title=subtitle=caption all `None` ⇒ `wrap_with_chrome`
  returns the composed SVG unchanged. The new params must not perturb this path.
- **No new silent drops:** an unknown/misspelled `configure_*` is still rejected by
  the existing `configure_*` validation; the anchor value is validated by the
  existing `TitleConfig.__post_init__` (start/middle/end).
- **Single figure anchor** governs title + subtitle + caption together.
- **Default inset = `16.0`** sourced as the same value the single-chart title uses
  (`ThemePadding::default().padding`); not a free-floating magic number.
- All three compositors (`horizontal`/`vertical`/`grid`) behave identically, so
  grid-based composites (`JointChart`, `ClusterMap`, `FacetGrid`) inherit the fix.

## 8. Key decisions and tradeoffs

- **Default inset matches single-chart margin (`16`), not panel-content alignment.**
  Chosen for simplicity and robustness: no introspection of a child panel's y-axis
  gutter. A default concat title lines up like a single-chart title.
- **Anchor governs all three chrome lines.** There is no separate caption-anchor
  config today; one knob gives the caption anchoring the issue asked for. Rejected
  the narrower title+subtitle-only scope because it leaves the caption-anchor gap.
- **Inset is resolved Python-side, geometry computed Rust-side.** Keeps the
  Python/Rust boundary a thin data pass (floats + a string), no chart-config dict
  crossing into the compositor binding.
- **`end` anchor insets from the right by `padding.right`** (default 16), symmetric
  with `start`/`left`. `middle` ignores both insets.

## 9. Acceptance criteria

- Default `(L | R).properties(title=…, caption=…)` renders title and caption at
  `x≈16` (was `x=0`), `text-anchor="start"`.
- `.configure_padding(left=60, auto=False)` moves both to `x=60`.
- `.configure_title(anchor="middle")` centers all chrome lines
  (`x=panel_w/2`, `text-anchor="middle"`); `anchor="end"` right-aligns them.
- Works for `HConcat`, `VConcat`, and grid composites (`JointChart`/`ClusterMap`/facet).
- Composite with no figure chrome is byte-identical to pre-fix output.
- `cargo test` and `uv run pytest -n auto` green; concat chrome goldens regenerated
  and visually inspected.

## 10. Validation strategy

- Rust unit tests in `figure_chrome.rs`: assert emitted `x`/`text-anchor` for each
  anchor and for a non-default inset; assert the no-chrome round-trip stays
  byte-identical; assert `emit_footer` caption honors anchor + inset.
- Python render-level tests (extend `tests/test_flexibility_caps/test_d10_figure_title.py`):
  parse the composed SVG and assert the figure title/caption `x` + `text-anchor`
  for default, `configure_padding(left=…)`, and `configure_title(anchor=…)` across
  HConcat / VConcat / grid composites. Assert overrides are not silently dropped.
- Regenerate + visually inspect any concat goldens whose chrome x shifted 0→16.

## 11. Open questions

None — the two design decisions (default inset = single-chart margin; anchor
governs all three lines) are resolved.
