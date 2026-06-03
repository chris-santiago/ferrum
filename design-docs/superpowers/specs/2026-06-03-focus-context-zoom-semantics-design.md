# Focus+Context Zoom Semantics — Design Spec

> Status: **APPROVED 2026-06-03** (FA-20). Surfaced while browser-validating FA-18:
> box-zooming the overview of a focus+context chart made the detail panel vanish.
> Root cause is pre-existing (the D3-zoom path is global → panel 0); FA-18's
> per-panel transforms turned it from masked into visible. This spec defines how
> multi-panel zoom/pan/box-zoom behave under focus+context semantics.

## 1. Scope

The interactive (WASM/JS) zoom/pan/box-zoom path in `ferrum-anywidget.js` +
`ferrum-wasm`. Today the global D3 `zoomBehavior` on the chart wrapper maps every
wheel, drag-pan, and box-zoom gesture to a single transform applied to panel 0
(`set_transform` → `set_absolute(0, …)`). On a multi-panel focus+context chart
this misroutes: box-zooming the overview applies an overview-derived transform to
the detail (panel 0), pushing it off its scissor. This spec makes the zoom path
panel-aware under focus+context semantics. Static SVG is untouched.

## 2. Goals

- Box-zooming the overview never corrupts/hides the detail; it rescales the detail.
- Wheel, drag-pan, and box-zoom of the focus panel operate on the focus panel.
- The behavior generalizes when the focus panel is not panel 0.
- Single-panel charts and at-rest renders are unchanged.

## 3. Non-goals

- No fully-independent per-panel zoom for generic multi-panel charts (panels with
  no domain-rescale relationship). Those keep today's focus-panel-targeted behavior
  and are explicitly out of scope (tracked separately if needed).
- No change to the brush→rescale path itself (FA-18; already correct), to static
  SVG, or to the per-panel mark-transform mechanism.

## 4. System behavior

For a chart that declares a `domain`-role param binding (the focus+context pattern):

- **Focus panel** = the panel named in the `domain`-role param binding. Wheel,
  drag-pan (pan mode), and box-zoom-on-the-focus all transform the focus panel.
- **Context (overview) panel** = a panel carrying the interval selection that drives
  the focus domain. It is brush-only:
  - A **brush** (select mode) drives the focus domain — unchanged, working.
  - A **box-zoom** drawn on the context panel is routed through the same
    rescale path as the brush (it sets the focus domain to the boxed x-range),
    rather than zooming the context panel itself.
  - Wheel and drag-pan do not transform the context panel (they target the focus).
- **Wheel anywhere** → focus panel (no pointer hit-testing; the focus is what zooms).
- **Box-zoom on the focus panel** → a 2-D box-zoom of the focus, computed relative
  to the focus panel's `plot_area` (not global chart dimensions).

For a chart with **no** `domain`-role binding (single-panel, or generic multi-panel):
`focusPanel = 0`; behavior is identical to today.

## 5. Architecture

- **Focus-panel resolution (JS):** `focusPanel = (cfg.param_bindings || [])
  .find(b => b.role === 'domain')?.panel ?? 0`. Computed once at render init.
- **Rust API:** `set_transform(panel_id, k, tx, ty)` replaces `set_transform(k, tx, ty)`;
  it calls `set_absolute(panel_id, …)` then `upload_transform_and_render(panel_id)`
  (which already uploads all panels' affines after FA-18). The hardcoded `0` is gone.
- **Wheel / drag-pan (JS):** the global `zoomBehavior` 'zoom' handler calls
  `renderer.setTransform(focusPanel, k, x, y)`.
- **Box-zoom (JS):** the per-panel brush `end` handler, in `boxzoom` mode, branches
  on `panelIdx`:
  - `panelIdx === focusPanel` → compute `k`/`tx`/`ty` against that panel's
    `plot_area` and call `renderer.setTransform(focusPanel, k, tx, ty)`.
  - `panelIdx !== focusPanel` (a context/overview panel) → route through the
    rescale path (`renderer.handleDrag(panelIdx, x0, y0, x1, y1)` and handle the
    `rescaled` envelope exactly as select-mode does), then clear the brush.
- **Overlay/zoom-state:** the raw SVG overlay transform and the `onZoomChange`
  callback key continue to track the focus panel (the panel being transformed).

## 6. Canonical interfaces / data contracts

- `set_transform(panel_id: u32, k: f32, tx: f32, ty: f32) -> Result<String, JsValue>`
  — applies an absolute zoom transform to `panel_id` and returns that panel's
  zoomed text-label JSON. (Was hardcoded to panel 0.)
- Focus signal: `param_bindings[*] = {param, role, panel, channel}` with
  `role == "domain"` identifying the focus `panel` (already emitted; see FA-17 audit).

## 7. Invariants and constraints

- **Single-panel unchanged:** `focusPanel == 0`; `set_transform(0, …)` is the old path.
- **No NotImplementedError / silent drop:** box-zoom on a context panel does
  something coherent (rescale), never nothing and never the disappear bug.
- **FA-18 mark-transform path untouched;** static SVG untouched.
- **WASM gates:** `cargo test -p ferrum-wasm`, `clippy --target wasm32 -- -D warnings`,
  wasm32 build all green; bundle rebuilt.

## 8. Key decisions and tradeoffs

- **Focus+context semantics over full independent per-panel zoom.** Matches the
  canonical idiom (you brush the context, you zoom the focus) and avoids per-panel
  D3-zoom state + pointer hit-testing. Generic-multi-panel independent zoom is the
  larger build and is deferred.
- **Wheel always targets focus (no hit-testing).** Simpler and matches the model;
  the focus is the only zoomable view in a focus+context chart.
- **Box-zoom on context reuses the rescale path** rather than a new code path —
  a boxed x-range on the overview is semantically the same as a brush there.
- **Box-zoom computed per-panel.** The current global-coordinate box-zoom math is a
  latent bug; the focus box-zoom is computed against the focus `plot_area`.

## 9. Acceptance criteria

- Focus+context demo: box-zoom on the overview rescales the detail (detail stays
  visible, no shear); box-zoom on the detail zooms the detail; wheel/pan act on the
  detail; the overview never independently zooms.
- Single-panel charts: zoom/pan/box-zoom unchanged.
- `cargo test -p ferrum-wasm` + `clippy`/`build` for wasm32 green; bundle rebuilt;
  static SVG goldens unchanged.

## 10. Validation strategy

- **Rust unit:** `set_transform(panel_id, …)` writes the given panel's slot (not
  always 0); a 2-panel state with `set_transform(1, …)` leaves panel 0 identity.
- **WASM build + clippy gates.**
- **Browser (human):** the focus+context demo — box-zoom overview → detail rescales;
  box-zoom detail → detail zooms; wheel/pan → detail; overview never zooms; at-rest
  parity.

## 11. Open questions

- None blocking. Generic multi-panel (no domain binding) independent zoom is
  explicitly out of scope.
