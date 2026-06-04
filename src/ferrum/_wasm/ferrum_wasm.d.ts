/* tslint:disable */
/* eslint-disable */

export class WasmRenderer {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    clearSelections(): string;
    static create(canvas: HTMLCanvasElement): Promise<WasmRenderer>;
    /**
     * Return the href string for a specific mark node, or an empty string if
     * none is present.
     *
     * `panel_id`, `batch_idx`, and `node_idx` correspond to the triple returned
     * by `hitTestAt`.  The href is sourced from `batch.hrefs[node_idx]` in the
     * scene graph.
     */
    getHref(panel_id: number, batch_idx: number, node_idx: number): string;
    /**
     * `{"fields":[{"name":"x","value":"1.23"},…]}`, or `"{}"` if no
     * tooltip data is available for this batch/instance.
     */
    getTooltip(panel_id: number, batch_idx: number, node_idx: number): string;
    /**
     * Hit-test a click at canvas pixel (x, y), update selection state, apply
     * conditional encodings (dim non-selected marks), re-render frame, and
     * return the new selection state as a JSON string.
     *
     * The returned JSON is a map of `selection_name → {field_name: field_value}`.
     * The JS caller should forward this to `model.set('selection_state', ...)`.
     */
    handleClick(x: number, y: number, shift_held: boolean): string;
    /**
     * Handle a brush-drag on a panel: update interval selection state, apply
     * conditional encodings, rebuild GPU buffers, re-render, and return
     * the new selection state as JSON.
     */
    handleDrag(panel_id: number, x0: number, y0: number, x1: number, y1: number): string;
    /**
     * Return tooltip JSON for a specific mark instance.
     *
     * `panel_id` and `batch_idx` identify the packed batch; `node_idx` is
     * the index of the mark within that batch.  Returns a JSON object
     */
    hitTestAt(x: number, y: number): string;
    loadScene(scene_json: string, packed_data: Uint8Array): string;
    maxTextureSize(): number;
    /**
     * Apply a pan delta on the given panel and re-render via GPU affine transform.
     *
     * Returns updated text-element JSON.
     */
    onPan(panel_id: number, dx: number, dy: number): string;
    /**
     * Apply a wheel-zoom event on the given panel and re-render via GPU affine transform.
     *
     * Returns updated text-element JSON (tick labels at new positions) so the JS
     * overlay can reposition axis labels without a Python round-trip.
     */
    onWheel(panel_id: number, delta_y: number, cx: number, cy: number): string;
    renderFrame(): void;
    /**
     * Reset zoom/pan to identity for the given panel and re-render.
     *
     * Returns text-element JSON with tick labels at their original positions.
     */
    resetZoom(panel_id: number): string;
    resize(width: number, height: number): void;
    /**
     * Select all indexed marks (circles and rects) within the given scene-space
     * rectangle `(x0, y0) – (x1, y1)` using the R-tree spatial index.
     *
     * Updates the first `Interval` selection spec found in `self.selections`,
     * then applies conditional encodings and re-renders.  Returns the new
     * selection state JSON.
     *
     * If no spatial index has been built yet (scene not loaded), returns `"{}"`.
     */
    selectInRect(_panel_id: number, x0: number, y0: number, x1: number, y1: number): string;
    /**
     * Set an absolute zoom+pan transform from D3-zoom for the given panel.
     *
     * `panel_id` identifies the panel to zoom (0-indexed); `k` is the uniform
     * scale factor; `tx`/`ty` are the translation offsets.
     * This replaces the accumulated state from `onWheel`/`onPan` and is the
     * entry point for HTML-export zoom driven by D3's `d3.zoom()`.
     *
     * Returns updated text-element JSON so the JS overlay can reposition labels.
     */
    setTransform(panel_id: number, k: number, tx: number, ty: number): string;
    /**
     * Begin a GPU-interpolated transition from an old scene to the currently
     * loaded scene.
     *
     * `old_scene_json` is the **previous** scene JSON string. The transition
     * target is `self.loaded.data` (the scene already loaded via `loadScene`).
     *
     * B4 fix: the old API accepted the *new* scene JSON and cloned `loaded.data`
     * as old. But `loadScene(new_json)` was already called before
     * `startTransition`, so `loaded.data` was already the new scene — making
     * old == new and the transition a no-op (self-to-self interpolation).
     * Now the caller passes the *old* scene JSON and we use `loaded.data` as
     * the transition target.
     *
     * Call ``tick_transition(t)`` (t in [0, 1]) from a requestAnimationFrame loop
     * to drive the animation.  ``start_transition`` does not start the loop —
     * the JavaScript caller owns the timing.
     *
     * Returns `Ok(())` immediately (no-op) if no scene is currently loaded.
     */
    startTransition(old_scene_json: string): void;
    /**
     * Advance the transition to fractional progress ``t`` ∈ [0, 1].
     *
     * Applies eased interpolation and re-renders the GPU frame.
     * When ``t >= 1.0`` the transition state is cleared and the new scene
     * is committed as the loaded scene.
     */
    tickTransition(t: number): void;
    /**
     * Toggle a legend-bound point selection's membership for one category
     * (D6 `BindingRole::Legend`).
     *
     * `selection_name` is the legend-bound point selection (from the `Legend`
     * param binding); `category` is the legend entry's label. Toggling mirrors
     * `handle_click`'s field-value point-selection update: the category is
     * stored as a `FieldValue::String` so the existing conditional path dims
     * or highlights every mark whose tooltip carries that value. Calling again
     * with the same category removes it. After updating selection state this
     * re-runs `apply_conditionals_and_render` (the same machinery legend-less
     * point selections use).
     */
    toggleLegend(selection_name: string, category: string): string;
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_wasmrenderer_free: (a: number, b: number) => void;
    readonly wasmrenderer_clearSelections: (a: number) => [number, number, number, number];
    readonly wasmrenderer_create: (a: any) => any;
    readonly wasmrenderer_getHref: (a: number, b: number, c: number, d: number) => [number, number];
    readonly wasmrenderer_getTooltip: (a: number, b: number, c: number, d: number) => [number, number];
    readonly wasmrenderer_handleClick: (a: number, b: number, c: number, d: number) => [number, number, number, number];
    readonly wasmrenderer_handleDrag: (a: number, b: number, c: number, d: number, e: number, f: number) => [number, number, number, number];
    readonly wasmrenderer_hitTestAt: (a: number, b: number, c: number) => [number, number];
    readonly wasmrenderer_loadScene: (a: number, b: number, c: number, d: number, e: number) => [number, number, number, number];
    readonly wasmrenderer_maxTextureSize: (a: number) => number;
    readonly wasmrenderer_onPan: (a: number, b: number, c: number, d: number) => [number, number, number, number];
    readonly wasmrenderer_onWheel: (a: number, b: number, c: number, d: number, e: number) => [number, number, number, number];
    readonly wasmrenderer_renderFrame: (a: number) => [number, number];
    readonly wasmrenderer_resetZoom: (a: number, b: number) => [number, number, number, number];
    readonly wasmrenderer_resize: (a: number, b: number, c: number) => void;
    readonly wasmrenderer_selectInRect: (a: number, b: number, c: number, d: number, e: number, f: number) => [number, number, number, number];
    readonly wasmrenderer_setTransform: (a: number, b: number, c: number, d: number, e: number) => [number, number, number, number];
    readonly wasmrenderer_startTransition: (a: number, b: number, c: number) => [number, number];
    readonly wasmrenderer_tickTransition: (a: number, b: number) => [number, number];
    readonly wasmrenderer_toggleLegend: (a: number, b: number, c: number, d: number, e: number) => [number, number, number, number];
    readonly wasm_bindgen__convert__closures_____invoke__h4eb714a55877aa02: (a: number, b: number, c: any) => [number, number];
    readonly wasm_bindgen__convert__closures_____invoke__h5ec2816eae335d2e: (a: number, b: number, c: any, d: any) => void;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_exn_store: (a: number) => void;
    readonly __externref_table_alloc: () => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_destroy_closure: (a: number, b: number) => void;
    readonly __externref_table_dealloc: (a: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
