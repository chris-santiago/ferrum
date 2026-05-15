/* tslint:disable */
/* eslint-disable */

export class WasmRenderer {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    static create(canvas: HTMLCanvasElement): Promise<WasmRenderer>;
    /**
     * Hit-test a click at canvas pixel (x, y), update selection state, apply
     * conditional encodings (dim non-selected marks), re-render frame, and
     * return the new selection state as a JSON string.
     *
     * The returned JSON is a map of `selection_name → {field_name: field_value}`.
     * The JS caller should forward this to `model.set('selection_state', ...)`.
     */
    handleClick(x: number, y: number): string;
    loadScene(scene_json: string, packed_data: Uint8Array): string;
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
     * Begin a GPU-interpolated transition from the currently loaded scene to a
     * new scene JSON string.
     *
     * Call ``tick_transition(t)`` (t ∈ [0, 1]) from a requestAnimationFrame loop
     * to drive the animation.  ``start_transition`` does not start the loop —
     * the JavaScript caller owns the timing.
     *
     * Returns `Ok(())` immediately (no-op) if no scene is currently loaded.
     */
    startTransition(new_scene_json: string): void;
    /**
     * Advance the transition to fractional progress ``t`` ∈ [0, 1].
     *
     * Applies eased interpolation and re-renders the GPU frame.
     * When ``t >= 1.0`` the transition state is cleared and the new scene
     * is committed as the loaded scene.
     */
    tickTransition(t: number): void;
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_wasmrenderer_free: (a: number, b: number) => void;
    readonly wasmrenderer_create: (a: any) => any;
    readonly wasmrenderer_handleClick: (a: number, b: number, c: number) => [number, number, number, number];
    readonly wasmrenderer_loadScene: (a: number, b: number, c: number, d: number, e: number) => [number, number, number, number];
    readonly wasmrenderer_onPan: (a: number, b: number, c: number, d: number) => [number, number, number, number];
    readonly wasmrenderer_onWheel: (a: number, b: number, c: number, d: number, e: number) => [number, number, number, number];
    readonly wasmrenderer_renderFrame: (a: number) => [number, number];
    readonly wasmrenderer_resetZoom: (a: number, b: number) => [number, number, number, number];
    readonly wasmrenderer_resize: (a: number, b: number, c: number) => void;
    readonly wasmrenderer_startTransition: (a: number, b: number, c: number) => [number, number];
    readonly wasmrenderer_tickTransition: (a: number, b: number) => [number, number];
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
