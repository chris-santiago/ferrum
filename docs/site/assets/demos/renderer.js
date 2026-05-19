/* @ts-self-types="./ferrum_wasm.d.ts" */

export class WasmRenderer {
    static __wrap(ptr) {
        const obj = Object.create(WasmRenderer.prototype);
        obj.__wbg_ptr = ptr;
        WasmRendererFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmRendererFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmrenderer_free(ptr, 0);
    }
    /**
     * @param {HTMLCanvasElement} canvas
     * @returns {Promise<WasmRenderer>}
     */
    static create(canvas) {
        const ret = wasm.wasmrenderer_create(canvas);
        return ret;
    }
    /**
     * `{"fields":[{"name":"x","value":"1.23"},…]}`, or `"{}"` if no
     * tooltip data is available for this batch/instance.
     * @param {number} panel_id
     * @param {number} batch_idx
     * @param {number} node_idx
     * @returns {string}
     */
    getTooltip(panel_id, batch_idx, node_idx) {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.wasmrenderer_getTooltip(this.__wbg_ptr, panel_id, batch_idx, node_idx);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * Hit-test a click at canvas pixel (x, y), update selection state, apply
     * conditional encodings (dim non-selected marks), re-render frame, and
     * return the new selection state as a JSON string.
     *
     * The returned JSON is a map of `selection_name → {field_name: field_value}`.
     * The JS caller should forward this to `model.set('selection_state', ...)`.
     * @param {number} x
     * @param {number} y
     * @param {boolean} shift_held
     * @returns {string}
     */
    handleClick(x, y, shift_held) {
        let deferred2_0;
        let deferred2_1;
        try {
            const ret = wasm.wasmrenderer_handleClick(this.__wbg_ptr, x, y, shift_held);
            var ptr1 = ret[0];
            var len1 = ret[1];
            if (ret[3]) {
                ptr1 = 0; len1 = 0;
                throw takeFromExternrefTable0(ret[2]);
            }
            deferred2_0 = ptr1;
            deferred2_1 = len1;
            return getStringFromWasm0(ptr1, len1);
        } finally {
            wasm.__wbindgen_free(deferred2_0, deferred2_1, 1);
        }
    }
    /**
     * Handle a brush-drag on a panel: update interval selection state, apply
     * conditional encodings, rebuild GPU buffers, re-render, and return
     * the new selection state as JSON.
     * @param {number} panel_id
     * @param {number} x0
     * @param {number} y0
     * @param {number} x1
     * @param {number} y1
     * @returns {string}
     */
    handleDrag(panel_id, x0, y0, x1, y1) {
        let deferred2_0;
        let deferred2_1;
        try {
            const ret = wasm.wasmrenderer_handleDrag(this.__wbg_ptr, panel_id, x0, y0, x1, y1);
            var ptr1 = ret[0];
            var len1 = ret[1];
            if (ret[3]) {
                ptr1 = 0; len1 = 0;
                throw takeFromExternrefTable0(ret[2]);
            }
            deferred2_0 = ptr1;
            deferred2_1 = len1;
            return getStringFromWasm0(ptr1, len1);
        } finally {
            wasm.__wbindgen_free(deferred2_0, deferred2_1, 1);
        }
    }
    /**
     * Return tooltip JSON for a specific mark instance.
     *
     * `panel_id` and `batch_idx` identify the packed batch; `node_idx` is
     * the index of the mark within that batch.  Returns a JSON object
     * @param {number} x
     * @param {number} y
     * @returns {string}
     */
    hitTestAt(x, y) {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.wasmrenderer_hitTestAt(this.__wbg_ptr, x, y);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * @param {string} scene_json
     * @param {Uint8Array} packed_data
     * @returns {string}
     */
    loadScene(scene_json, packed_data) {
        let deferred4_0;
        let deferred4_1;
        try {
            const ptr0 = passStringToWasm0(scene_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            const len0 = WASM_VECTOR_LEN;
            const ptr1 = passArray8ToWasm0(packed_data, wasm.__wbindgen_malloc);
            const len1 = WASM_VECTOR_LEN;
            const ret = wasm.wasmrenderer_loadScene(this.__wbg_ptr, ptr0, len0, ptr1, len1);
            var ptr3 = ret[0];
            var len3 = ret[1];
            if (ret[3]) {
                ptr3 = 0; len3 = 0;
                throw takeFromExternrefTable0(ret[2]);
            }
            deferred4_0 = ptr3;
            deferred4_1 = len3;
            return getStringFromWasm0(ptr3, len3);
        } finally {
            wasm.__wbindgen_free(deferred4_0, deferred4_1, 1);
        }
    }
    /**
     * Apply a pan delta on the given panel and re-render via GPU affine transform.
     *
     * Returns updated text-element JSON.
     * @param {number} panel_id
     * @param {number} dx
     * @param {number} dy
     * @returns {string}
     */
    onPan(panel_id, dx, dy) {
        let deferred2_0;
        let deferred2_1;
        try {
            const ret = wasm.wasmrenderer_onPan(this.__wbg_ptr, panel_id, dx, dy);
            var ptr1 = ret[0];
            var len1 = ret[1];
            if (ret[3]) {
                ptr1 = 0; len1 = 0;
                throw takeFromExternrefTable0(ret[2]);
            }
            deferred2_0 = ptr1;
            deferred2_1 = len1;
            return getStringFromWasm0(ptr1, len1);
        } finally {
            wasm.__wbindgen_free(deferred2_0, deferred2_1, 1);
        }
    }
    /**
     * Apply a wheel-zoom event on the given panel and re-render via GPU affine transform.
     *
     * Returns updated text-element JSON (tick labels at new positions) so the JS
     * overlay can reposition axis labels without a Python round-trip.
     * @param {number} panel_id
     * @param {number} delta_y
     * @param {number} cx
     * @param {number} cy
     * @returns {string}
     */
    onWheel(panel_id, delta_y, cx, cy) {
        let deferred2_0;
        let deferred2_1;
        try {
            const ret = wasm.wasmrenderer_onWheel(this.__wbg_ptr, panel_id, delta_y, cx, cy);
            var ptr1 = ret[0];
            var len1 = ret[1];
            if (ret[3]) {
                ptr1 = 0; len1 = 0;
                throw takeFromExternrefTable0(ret[2]);
            }
            deferred2_0 = ptr1;
            deferred2_1 = len1;
            return getStringFromWasm0(ptr1, len1);
        } finally {
            wasm.__wbindgen_free(deferred2_0, deferred2_1, 1);
        }
    }
    renderFrame() {
        const ret = wasm.wasmrenderer_renderFrame(this.__wbg_ptr);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Reset zoom/pan to identity for the given panel and re-render.
     *
     * Returns text-element JSON with tick labels at their original positions.
     * @param {number} panel_id
     * @returns {string}
     */
    resetZoom(panel_id) {
        let deferred2_0;
        let deferred2_1;
        try {
            const ret = wasm.wasmrenderer_resetZoom(this.__wbg_ptr, panel_id);
            var ptr1 = ret[0];
            var len1 = ret[1];
            if (ret[3]) {
                ptr1 = 0; len1 = 0;
                throw takeFromExternrefTable0(ret[2]);
            }
            deferred2_0 = ptr1;
            deferred2_1 = len1;
            return getStringFromWasm0(ptr1, len1);
        } finally {
            wasm.__wbindgen_free(deferred2_0, deferred2_1, 1);
        }
    }
    /**
     * @param {number} width
     * @param {number} height
     */
    resize(width, height) {
        wasm.wasmrenderer_resize(this.__wbg_ptr, width, height);
    }
    /**
     * Set an absolute zoom+pan transform from D3-zoom.
     *
     * `k` is the uniform scale factor; `tx`/`ty` are the translation offsets.
     * This replaces the accumulated state from `onWheel`/`onPan` and is the
     * entry point for HTML-export zoom driven by D3's `d3.zoom()`.
     *
     * Operates on panel 0 (single-panel charts; multi-panel support later).
     * Returns updated text-element JSON so the JS overlay can reposition labels.
     * @param {number} k
     * @param {number} tx
     * @param {number} ty
     * @returns {string}
     */
    setTransform(k, tx, ty) {
        let deferred2_0;
        let deferred2_1;
        try {
            const ret = wasm.wasmrenderer_setTransform(this.__wbg_ptr, k, tx, ty);
            var ptr1 = ret[0];
            var len1 = ret[1];
            if (ret[3]) {
                ptr1 = 0; len1 = 0;
                throw takeFromExternrefTable0(ret[2]);
            }
            deferred2_0 = ptr1;
            deferred2_1 = len1;
            return getStringFromWasm0(ptr1, len1);
        } finally {
            wasm.__wbindgen_free(deferred2_0, deferred2_1, 1);
        }
    }
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
     * @param {string} old_scene_json
     */
    startTransition(old_scene_json) {
        const ptr0 = passStringToWasm0(old_scene_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmrenderer_startTransition(this.__wbg_ptr, ptr0, len0);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Advance the transition to fractional progress ``t`` ∈ [0, 1].
     *
     * Applies eased interpolation and re-renders the GPU frame.
     * When ``t >= 1.0`` the transition state is cleared and the new scene
     * is committed as the loaded scene.
     * @param {number} t
     */
    tickTransition(t) {
        const ret = wasm.wasmrenderer_tickTransition(this.__wbg_ptr, t);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
}
if (Symbol.dispose) WasmRenderer.prototype[Symbol.dispose] = WasmRenderer.prototype.free;
function __wbg_get_imports() {
    const import0 = {
        __proto__: null,
        __wbg___wbindgen_boolean_get_2304fb8c853028c8: function(arg0) {
            const v = arg0;
            const ret = typeof(v) === 'boolean' ? v : undefined;
            return isLikeNone(ret) ? 0xFFFFFF : ret ? 1 : 0;
        },
        __wbg___wbindgen_debug_string_edece8177ad01481: function(arg0, arg1) {
            const ret = debugString(arg1);
            const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            const len1 = WASM_VECTOR_LEN;
            getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
        },
        __wbg___wbindgen_is_function_5cd60d5cf78b4eef: function(arg0) {
            const ret = typeof(arg0) === 'function';
            return ret;
        },
        __wbg___wbindgen_is_undefined_35bb9f4c7fd651d5: function(arg0) {
            const ret = arg0 === undefined;
            return ret;
        },
        __wbg___wbindgen_number_get_f73a1244370fcc2c: function(arg0, arg1) {
            const obj = arg1;
            const ret = typeof(obj) === 'number' ? obj : undefined;
            getDataViewMemory0().setFloat64(arg0 + 8 * 1, isLikeNone(ret) ? 0 : ret, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, !isLikeNone(ret), true);
        },
        __wbg___wbindgen_string_get_d109740c0d18f4d7: function(arg0, arg1) {
            const obj = arg1;
            const ret = typeof(obj) === 'string' ? obj : undefined;
            var ptr1 = isLikeNone(ret) ? 0 : passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            var len1 = WASM_VECTOR_LEN;
            getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
        },
        __wbg___wbindgen_throw_9c31b086c2b26051: function(arg0, arg1) {
            throw new Error(getStringFromWasm0(arg0, arg1));
        },
        __wbg__wbg_cb_unref_3fa391f3fcdb55f8: function(arg0) {
            arg0._wbg_cb_unref();
        },
        __wbg_activeTexture_37cff0753870753b: function(arg0, arg1) {
            arg0.activeTexture(arg1 >>> 0);
        },
        __wbg_activeTexture_4d2afad7cfda1396: function(arg0, arg1) {
            arg0.activeTexture(arg1 >>> 0);
        },
        __wbg_attachShader_0a37c762590e5e1c: function(arg0, arg1, arg2) {
            arg0.attachShader(arg1, arg2);
        },
        __wbg_attachShader_515800f4051247dc: function(arg0, arg1, arg2) {
            arg0.attachShader(arg1, arg2);
        },
        __wbg_beginQuery_6c6c5b6d0d8a2c72: function(arg0, arg1, arg2) {
            arg0.beginQuery(arg1 >>> 0, arg2);
        },
        __wbg_bindAttribLocation_07b2841d89fca977: function(arg0, arg1, arg2, arg3, arg4) {
            arg0.bindAttribLocation(arg1, arg2 >>> 0, getStringFromWasm0(arg3, arg4));
        },
        __wbg_bindAttribLocation_1bbbcdee8d08ba2a: function(arg0, arg1, arg2, arg3, arg4) {
            arg0.bindAttribLocation(arg1, arg2 >>> 0, getStringFromWasm0(arg3, arg4));
        },
        __wbg_bindBufferRange_b3fd6bf5761eb1af: function(arg0, arg1, arg2, arg3, arg4, arg5) {
            arg0.bindBufferRange(arg1 >>> 0, arg2 >>> 0, arg3, arg4, arg5);
        },
        __wbg_bindBuffer_1a31fd3809dc22c8: function(arg0, arg1, arg2) {
            arg0.bindBuffer(arg1 >>> 0, arg2);
        },
        __wbg_bindBuffer_4bf3ab31e8e200ed: function(arg0, arg1, arg2) {
            arg0.bindBuffer(arg1 >>> 0, arg2);
        },
        __wbg_bindFramebuffer_751e5064f23ee1c4: function(arg0, arg1, arg2) {
            arg0.bindFramebuffer(arg1 >>> 0, arg2);
        },
        __wbg_bindFramebuffer_92449a44405b6557: function(arg0, arg1, arg2) {
            arg0.bindFramebuffer(arg1 >>> 0, arg2);
        },
        __wbg_bindRenderbuffer_1742855b643a7566: function(arg0, arg1, arg2) {
            arg0.bindRenderbuffer(arg1 >>> 0, arg2);
        },
        __wbg_bindRenderbuffer_c46a8b6f3f8ba246: function(arg0, arg1, arg2) {
            arg0.bindRenderbuffer(arg1 >>> 0, arg2);
        },
        __wbg_bindSampler_708d9901a5e548b8: function(arg0, arg1, arg2) {
            arg0.bindSampler(arg1 >>> 0, arg2);
        },
        __wbg_bindTexture_7fd7f85d6f942f6f: function(arg0, arg1, arg2) {
            arg0.bindTexture(arg1 >>> 0, arg2);
        },
        __wbg_bindTexture_85abbde679bce760: function(arg0, arg1, arg2) {
            arg0.bindTexture(arg1 >>> 0, arg2);
        },
        __wbg_bindVertexArrayOES_fb7e8c5e8e106919: function(arg0, arg1) {
            arg0.bindVertexArrayOES(arg1);
        },
        __wbg_bindVertexArray_f8587a616356d307: function(arg0, arg1) {
            arg0.bindVertexArray(arg1);
        },
        __wbg_blendColor_82716e22a8f522ff: function(arg0, arg1, arg2, arg3, arg4) {
            arg0.blendColor(arg1, arg2, arg3, arg4);
        },
        __wbg_blendColor_f877221c780bdbaf: function(arg0, arg1, arg2, arg3, arg4) {
            arg0.blendColor(arg1, arg2, arg3, arg4);
        },
        __wbg_blendEquationSeparate_946c10181ab6c6cf: function(arg0, arg1, arg2) {
            arg0.blendEquationSeparate(arg1 >>> 0, arg2 >>> 0);
        },
        __wbg_blendEquationSeparate_985f782fb54b29fe: function(arg0, arg1, arg2) {
            arg0.blendEquationSeparate(arg1 >>> 0, arg2 >>> 0);
        },
        __wbg_blendEquation_519c57992eed79c1: function(arg0, arg1) {
            arg0.blendEquation(arg1 >>> 0);
        },
        __wbg_blendEquation_f496fde4a67ecc1e: function(arg0, arg1) {
            arg0.blendEquation(arg1 >>> 0);
        },
        __wbg_blendFuncSeparate_6f525092629a20ae: function(arg0, arg1, arg2, arg3, arg4) {
            arg0.blendFuncSeparate(arg1 >>> 0, arg2 >>> 0, arg3 >>> 0, arg4 >>> 0);
        },
        __wbg_blendFuncSeparate_ea29c928bc1c4984: function(arg0, arg1, arg2, arg3, arg4) {
            arg0.blendFuncSeparate(arg1 >>> 0, arg2 >>> 0, arg3 >>> 0, arg4 >>> 0);
        },
        __wbg_blendFunc_2e7b7adf253717a0: function(arg0, arg1, arg2) {
            arg0.blendFunc(arg1 >>> 0, arg2 >>> 0);
        },
        __wbg_blendFunc_d29c837f8be35d6e: function(arg0, arg1, arg2) {
            arg0.blendFunc(arg1 >>> 0, arg2 >>> 0);
        },
        __wbg_blitFramebuffer_8fd7726fe3c57e1a: function(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10) {
            arg0.blitFramebuffer(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9 >>> 0, arg10 >>> 0);
        },
        __wbg_bufferData_74a0b79b4c9d8f96: function(arg0, arg1, arg2, arg3) {
            arg0.bufferData(arg1 >>> 0, arg2, arg3 >>> 0);
        },
        __wbg_bufferData_886f34df840b0814: function(arg0, arg1, arg2, arg3) {
            arg0.bufferData(arg1 >>> 0, arg2, arg3 >>> 0);
        },
        __wbg_bufferData_aebf4ed69e98d559: function(arg0, arg1, arg2, arg3) {
            arg0.bufferData(arg1 >>> 0, arg2, arg3 >>> 0);
        },
        __wbg_bufferData_e8afecf0042a3eb9: function(arg0, arg1, arg2, arg3) {
            arg0.bufferData(arg1 >>> 0, arg2, arg3 >>> 0);
        },
        __wbg_bufferSubData_0e5936ef36f518d2: function(arg0, arg1, arg2, arg3) {
            arg0.bufferSubData(arg1 >>> 0, arg2, arg3);
        },
        __wbg_bufferSubData_ca02a13879fa62e8: function(arg0, arg1, arg2, arg3) {
            arg0.bufferSubData(arg1 >>> 0, arg2, arg3);
        },
        __wbg_call_dfde26266607c996: function() { return handleError(function (arg0, arg1, arg2) {
            const ret = arg0.call(arg1, arg2);
            return ret;
        }, arguments); },
        __wbg_clearBufferfv_a0bddf84cc04ef84: function(arg0, arg1, arg2, arg3, arg4) {
            arg0.clearBufferfv(arg1 >>> 0, arg2, getArrayF32FromWasm0(arg3, arg4));
        },
        __wbg_clearBufferiv_9a3f2d1ec3f2296f: function(arg0, arg1, arg2, arg3, arg4) {
            arg0.clearBufferiv(arg1 >>> 0, arg2, getArrayI32FromWasm0(arg3, arg4));
        },
        __wbg_clearBufferuiv_d52433002e7330f8: function(arg0, arg1, arg2, arg3, arg4) {
            arg0.clearBufferuiv(arg1 >>> 0, arg2, getArrayU32FromWasm0(arg3, arg4));
        },
        __wbg_clearDepth_1eae37358a24b9db: function(arg0, arg1) {
            arg0.clearDepth(arg1);
        },
        __wbg_clearDepth_f42ada4795e5a943: function(arg0, arg1) {
            arg0.clearDepth(arg1);
        },
        __wbg_clearStencil_999f2e1ef49323e6: function(arg0, arg1) {
            arg0.clearStencil(arg1);
        },
        __wbg_clearStencil_a58c15a1dcbf1fbe: function(arg0, arg1) {
            arg0.clearStencil(arg1);
        },
        __wbg_clear_252bb7b11d5bea06: function(arg0, arg1) {
            arg0.clear(arg1 >>> 0);
        },
        __wbg_clear_7d0a8d124c2a4b66: function(arg0, arg1) {
            arg0.clear(arg1 >>> 0);
        },
        __wbg_clientWaitSync_fb0623a14def0f1e: function(arg0, arg1, arg2, arg3) {
            const ret = arg0.clientWaitSync(arg1, arg2 >>> 0, arg3 >>> 0);
            return ret;
        },
        __wbg_colorMask_0f86a23bfc7696a7: function(arg0, arg1, arg2, arg3, arg4) {
            arg0.colorMask(arg1 !== 0, arg2 !== 0, arg3 !== 0, arg4 !== 0);
        },
        __wbg_colorMask_2d4b38c34bf55a02: function(arg0, arg1, arg2, arg3, arg4) {
            arg0.colorMask(arg1 !== 0, arg2 !== 0, arg3 !== 0, arg4 !== 0);
        },
        __wbg_compileShader_a20e7b68d3edcd8a: function(arg0, arg1) {
            arg0.compileShader(arg1);
        },
        __wbg_compileShader_b77bd79d00a03b02: function(arg0, arg1) {
            arg0.compileShader(arg1);
        },
        __wbg_compressedTexSubImage2D_12adc86b34c12d28: function(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8) {
            arg0.compressedTexSubImage2D(arg1 >>> 0, arg2, arg3, arg4, arg5, arg6, arg7 >>> 0, arg8);
        },
        __wbg_compressedTexSubImage2D_5336c9efcad92150: function(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8) {
            arg0.compressedTexSubImage2D(arg1 >>> 0, arg2, arg3, arg4, arg5, arg6, arg7 >>> 0, arg8);
        },
        __wbg_compressedTexSubImage2D_7eb545d3f1d37773: function(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9) {
            arg0.compressedTexSubImage2D(arg1 >>> 0, arg2, arg3, arg4, arg5, arg6, arg7 >>> 0, arg8, arg9);
        },
        __wbg_compressedTexSubImage3D_1bca0af82425d03d: function(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11) {
            arg0.compressedTexSubImage3D(arg1 >>> 0, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9 >>> 0, arg10, arg11);
        },
        __wbg_compressedTexSubImage3D_7f820492cb5a6d5e: function(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10) {
            arg0.compressedTexSubImage3D(arg1 >>> 0, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9 >>> 0, arg10);
        },
        __wbg_copyBufferSubData_8855e4c7f24415d6: function(arg0, arg1, arg2, arg3, arg4, arg5) {
            arg0.copyBufferSubData(arg1 >>> 0, arg2 >>> 0, arg3, arg4, arg5);
        },
        __wbg_copyTexSubImage2D_68eb6addf3f910bb: function(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8) {
            arg0.copyTexSubImage2D(arg1 >>> 0, arg2, arg3, arg4, arg5, arg6, arg7, arg8);
        },
        __wbg_copyTexSubImage2D_c56507367f94e004: function(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8) {
            arg0.copyTexSubImage2D(arg1 >>> 0, arg2, arg3, arg4, arg5, arg6, arg7, arg8);
        },
        __wbg_copyTexSubImage3D_7f30d563975b3710: function(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9) {
            arg0.copyTexSubImage3D(arg1 >>> 0, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9);
        },
        __wbg_createBuffer_1c3448547584bc5a: function(arg0) {
            const ret = arg0.createBuffer();
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_createBuffer_77da03de0620a199: function(arg0) {
            const ret = arg0.createBuffer();
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_createFramebuffer_22f50a7a9f8afdf0: function(arg0) {
            const ret = arg0.createFramebuffer();
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_createFramebuffer_73699dac20f72ffb: function(arg0) {
            const ret = arg0.createFramebuffer();
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_createProgram_a175fc4c32429a24: function(arg0) {
            const ret = arg0.createProgram();
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_createProgram_c9d6396ea0bc7522: function(arg0) {
            const ret = arg0.createProgram();
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_createQuery_5d92b56f0ca718af: function(arg0) {
            const ret = arg0.createQuery();
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_createRenderbuffer_483c206d1b62e6bd: function(arg0) {
            const ret = arg0.createRenderbuffer();
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_createRenderbuffer_f26e2b467988cc7e: function(arg0) {
            const ret = arg0.createRenderbuffer();
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_createSampler_80eb58b226692482: function(arg0) {
            const ret = arg0.createSampler();
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_createShader_25e11081fd48d141: function(arg0, arg1) {
            const ret = arg0.createShader(arg1 >>> 0);
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_createShader_9c5e52918428bd27: function(arg0, arg1) {
            const ret = arg0.createShader(arg1 >>> 0);
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_createTexture_5e721dc1ddd865e3: function(arg0) {
            const ret = arg0.createTexture();
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_createTexture_f1cc0c64fa9e22cf: function(arg0) {
            const ret = arg0.createTexture();
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_createVertexArrayOES_03fccccc43c10f77: function(arg0) {
            const ret = arg0.createVertexArrayOES();
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_createVertexArray_050d27763dfd72fa: function(arg0) {
            const ret = arg0.createVertexArray();
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_cullFace_632c5f88d252b4d7: function(arg0, arg1) {
            arg0.cullFace(arg1 >>> 0);
        },
        __wbg_cullFace_962911677f1c30c6: function(arg0, arg1) {
            arg0.cullFace(arg1 >>> 0);
        },
        __wbg_deleteBuffer_5c5c23d034945b7c: function(arg0, arg1) {
            arg0.deleteBuffer(arg1);
        },
        __wbg_deleteBuffer_dd1d6f71883058cb: function(arg0, arg1) {
            arg0.deleteBuffer(arg1);
        },
        __wbg_deleteFramebuffer_4d8be9eb882b0525: function(arg0, arg1) {
            arg0.deleteFramebuffer(arg1);
        },
        __wbg_deleteFramebuffer_712016837ba2592e: function(arg0, arg1) {
            arg0.deleteFramebuffer(arg1);
        },
        __wbg_deleteProgram_35e4ff7b82f1c4d5: function(arg0, arg1) {
            arg0.deleteProgram(arg1);
        },
        __wbg_deleteProgram_771559436a63e7c1: function(arg0, arg1) {
            arg0.deleteProgram(arg1);
        },
        __wbg_deleteQuery_1c30cae3b68f3fd7: function(arg0, arg1) {
            arg0.deleteQuery(arg1);
        },
        __wbg_deleteRenderbuffer_16d1501ab6903d8e: function(arg0, arg1) {
            arg0.deleteRenderbuffer(arg1);
        },
        __wbg_deleteRenderbuffer_aee8ffc30e0e35cb: function(arg0, arg1) {
            arg0.deleteRenderbuffer(arg1);
        },
        __wbg_deleteSampler_ec0248a7607fb5e6: function(arg0, arg1) {
            arg0.deleteSampler(arg1);
        },
        __wbg_deleteShader_5f66fd162cd9b6b4: function(arg0, arg1) {
            arg0.deleteShader(arg1);
        },
        __wbg_deleteShader_718c5020e3d4f188: function(arg0, arg1) {
            arg0.deleteShader(arg1);
        },
        __wbg_deleteSync_b589decdc7180f91: function(arg0, arg1) {
            arg0.deleteSync(arg1);
        },
        __wbg_deleteTexture_3472fc261bb7ff34: function(arg0, arg1) {
            arg0.deleteTexture(arg1);
        },
        __wbg_deleteTexture_6990124dfb5053bd: function(arg0, arg1) {
            arg0.deleteTexture(arg1);
        },
        __wbg_deleteVertexArrayOES_b1b88aa74410f620: function(arg0, arg1) {
            arg0.deleteVertexArrayOES(arg1);
        },
        __wbg_deleteVertexArray_85b79d70fae1d1da: function(arg0, arg1) {
            arg0.deleteVertexArray(arg1);
        },
        __wbg_depthFunc_11c361d188403f52: function(arg0, arg1) {
            arg0.depthFunc(arg1 >>> 0);
        },
        __wbg_depthFunc_cd5ad66da02ddb7c: function(arg0, arg1) {
            arg0.depthFunc(arg1 >>> 0);
        },
        __wbg_depthMask_a00e4725581ef05d: function(arg0, arg1) {
            arg0.depthMask(arg1 !== 0);
        },
        __wbg_depthMask_e15ec83686756c88: function(arg0, arg1) {
            arg0.depthMask(arg1 !== 0);
        },
        __wbg_depthRange_2ed081b96c5c19be: function(arg0, arg1, arg2) {
            arg0.depthRange(arg1, arg2);
        },
        __wbg_depthRange_7f3fef7f421c00d4: function(arg0, arg1, arg2) {
            arg0.depthRange(arg1, arg2);
        },
        __wbg_disableVertexAttribArray_18b9a9fe235412a1: function(arg0, arg1) {
            arg0.disableVertexAttribArray(arg1 >>> 0);
        },
        __wbg_disableVertexAttribArray_40a8f7d4d882728e: function(arg0, arg1) {
            arg0.disableVertexAttribArray(arg1 >>> 0);
        },
        __wbg_disable_79f65722e686303b: function(arg0, arg1) {
            arg0.disable(arg1 >>> 0);
        },
        __wbg_disable_df908054ffee7971: function(arg0, arg1) {
            arg0.disable(arg1 >>> 0);
        },
        __wbg_document_3540635616a18455: function(arg0) {
            const ret = arg0.document;
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_drawArraysInstancedANGLE_a7a04432fa5e1577: function(arg0, arg1, arg2, arg3, arg4) {
            arg0.drawArraysInstancedANGLE(arg1 >>> 0, arg2, arg3, arg4);
        },
        __wbg_drawArraysInstanced_0e6f9f2102461c2a: function(arg0, arg1, arg2, arg3, arg4) {
            arg0.drawArraysInstanced(arg1 >>> 0, arg2, arg3, arg4);
        },
        __wbg_drawArrays_7f9a3dcec5315ce5: function(arg0, arg1, arg2, arg3) {
            arg0.drawArrays(arg1 >>> 0, arg2, arg3);
        },
        __wbg_drawArrays_bceea06128f9d778: function(arg0, arg1, arg2, arg3) {
            arg0.drawArrays(arg1 >>> 0, arg2, arg3);
        },
        __wbg_drawBuffersWEBGL_5fbba2b83de4c122: function(arg0, arg1) {
            arg0.drawBuffersWEBGL(arg1);
        },
        __wbg_drawBuffers_217bd25bf75ccebd: function(arg0, arg1) {
            arg0.drawBuffers(arg1);
        },
        __wbg_drawElementsInstancedANGLE_6794fe36875c5120: function(arg0, arg1, arg2, arg3, arg4, arg5) {
            arg0.drawElementsInstancedANGLE(arg1 >>> 0, arg2, arg3 >>> 0, arg4, arg5);
        },
        __wbg_drawElementsInstanced_767ab401cd072fd4: function(arg0, arg1, arg2, arg3, arg4, arg5) {
            arg0.drawElementsInstanced(arg1 >>> 0, arg2, arg3 >>> 0, arg4, arg5);
        },
        __wbg_enableVertexAttribArray_9963bb377f60317c: function(arg0, arg1) {
            arg0.enableVertexAttribArray(arg1 >>> 0);
        },
        __wbg_enableVertexAttribArray_9e6e81b8b603d999: function(arg0, arg1) {
            arg0.enableVertexAttribArray(arg1 >>> 0);
        },
        __wbg_enable_5c8f846164bc8138: function(arg0, arg1) {
            arg0.enable(arg1 >>> 0);
        },
        __wbg_enable_ee1b63abdc3fdeb5: function(arg0, arg1) {
            arg0.enable(arg1 >>> 0);
        },
        __wbg_endQuery_42d36ba1d568a37a: function(arg0, arg1) {
            arg0.endQuery(arg1 >>> 0);
        },
        __wbg_error_a6fa202b58aa1cd3: function(arg0, arg1) {
            let deferred0_0;
            let deferred0_1;
            try {
                deferred0_0 = arg0;
                deferred0_1 = arg1;
                console.error(getStringFromWasm0(arg0, arg1));
            } finally {
                wasm.__wbindgen_free(deferred0_0, deferred0_1, 1);
            }
        },
        __wbg_fenceSync_59d6455faf4ba50a: function(arg0, arg1, arg2) {
            const ret = arg0.fenceSync(arg1 >>> 0, arg2 >>> 0);
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_flush_1e5245bab2bbc54b: function(arg0) {
            arg0.flush();
        },
        __wbg_flush_279c03f2320388de: function(arg0) {
            arg0.flush();
        },
        __wbg_framebufferRenderbuffer_49b9288b6a7b5629: function(arg0, arg1, arg2, arg3, arg4) {
            arg0.framebufferRenderbuffer(arg1 >>> 0, arg2 >>> 0, arg3 >>> 0, arg4);
        },
        __wbg_framebufferRenderbuffer_9417c925d5389962: function(arg0, arg1, arg2, arg3, arg4) {
            arg0.framebufferRenderbuffer(arg1 >>> 0, arg2 >>> 0, arg3 >>> 0, arg4);
        },
        __wbg_framebufferTexture2D_8882fef6f47df627: function(arg0, arg1, arg2, arg3, arg4, arg5) {
            arg0.framebufferTexture2D(arg1 >>> 0, arg2 >>> 0, arg3 >>> 0, arg4, arg5);
        },
        __wbg_framebufferTexture2D_91e307404924ae24: function(arg0, arg1, arg2, arg3, arg4, arg5) {
            arg0.framebufferTexture2D(arg1 >>> 0, arg2 >>> 0, arg3 >>> 0, arg4, arg5);
        },
        __wbg_framebufferTextureLayer_8256c57e84c45762: function(arg0, arg1, arg2, arg3, arg4, arg5) {
            arg0.framebufferTextureLayer(arg1 >>> 0, arg2 >>> 0, arg3, arg4, arg5);
        },
        __wbg_framebufferTextureMultiviewOVR_fd3136c9d479feb2: function(arg0, arg1, arg2, arg3, arg4, arg5, arg6) {
            arg0.framebufferTextureMultiviewOVR(arg1 >>> 0, arg2 >>> 0, arg3, arg4, arg5, arg6);
        },
        __wbg_frontFace_1ab53137f5dcd7a2: function(arg0, arg1) {
            arg0.frontFace(arg1 >>> 0);
        },
        __wbg_frontFace_53fc2aad7ead45c9: function(arg0, arg1) {
            arg0.frontFace(arg1 >>> 0);
        },
        __wbg_getBufferSubData_f3d6368ec0319180: function(arg0, arg1, arg2, arg3) {
            arg0.getBufferSubData(arg1 >>> 0, arg2, arg3);
        },
        __wbg_getContext_32d5f94659d12566: function() { return handleError(function (arg0, arg1, arg2, arg3) {
            const ret = arg0.getContext(getStringFromWasm0(arg1, arg2), arg3);
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        }, arguments); },
        __wbg_getContext_50a6668bd78d1120: function() { return handleError(function (arg0, arg1, arg2, arg3) {
            const ret = arg0.getContext(getStringFromWasm0(arg1, arg2), arg3);
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        }, arguments); },
        __wbg_getExtension_c76ccfc25e343ce6: function() { return handleError(function (arg0, arg1, arg2) {
            const ret = arg0.getExtension(getStringFromWasm0(arg1, arg2));
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        }, arguments); },
        __wbg_getIndexedParameter_b83fcd0ac4c3a462: function() { return handleError(function (arg0, arg1, arg2) {
            const ret = arg0.getIndexedParameter(arg1 >>> 0, arg2 >>> 0);
            return ret;
        }, arguments); },
        __wbg_getParameter_5f25c05c9a0f445a: function() { return handleError(function (arg0, arg1) {
            const ret = arg0.getParameter(arg1 >>> 0);
            return ret;
        }, arguments); },
        __wbg_getParameter_827c3142b1ce3364: function() { return handleError(function (arg0, arg1) {
            const ret = arg0.getParameter(arg1 >>> 0);
            return ret;
        }, arguments); },
        __wbg_getProgramInfoLog_6d6e22f0179f1acf: function(arg0, arg1, arg2) {
            const ret = arg1.getProgramInfoLog(arg2);
            var ptr1 = isLikeNone(ret) ? 0 : passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            var len1 = WASM_VECTOR_LEN;
            getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
        },
        __wbg_getProgramInfoLog_e2fe4bdd00a597bc: function(arg0, arg1, arg2) {
            const ret = arg1.getProgramInfoLog(arg2);
            var ptr1 = isLikeNone(ret) ? 0 : passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            var len1 = WASM_VECTOR_LEN;
            getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
        },
        __wbg_getProgramParameter_6927dedbc507dfc7: function(arg0, arg1, arg2) {
            const ret = arg0.getProgramParameter(arg1, arg2 >>> 0);
            return ret;
        },
        __wbg_getProgramParameter_c7abe52a31622ce2: function(arg0, arg1, arg2) {
            const ret = arg0.getProgramParameter(arg1, arg2 >>> 0);
            return ret;
        },
        __wbg_getQueryParameter_6817ddd38edd8e5c: function(arg0, arg1, arg2) {
            const ret = arg0.getQueryParameter(arg1, arg2 >>> 0);
            return ret;
        },
        __wbg_getShaderInfoLog_246aba1bd0b04ad2: function(arg0, arg1, arg2) {
            const ret = arg1.getShaderInfoLog(arg2);
            var ptr1 = isLikeNone(ret) ? 0 : passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            var len1 = WASM_VECTOR_LEN;
            getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
        },
        __wbg_getShaderInfoLog_edfc45fd76ba8c81: function(arg0, arg1, arg2) {
            const ret = arg1.getShaderInfoLog(arg2);
            var ptr1 = isLikeNone(ret) ? 0 : passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            var len1 = WASM_VECTOR_LEN;
            getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
        },
        __wbg_getShaderParameter_07fb35844118558b: function(arg0, arg1, arg2) {
            const ret = arg0.getShaderParameter(arg1, arg2 >>> 0);
            return ret;
        },
        __wbg_getShaderParameter_ac9e7f81d3268efe: function(arg0, arg1, arg2) {
            const ret = arg0.getShaderParameter(arg1, arg2 >>> 0);
            return ret;
        },
        __wbg_getSupportedExtensions_76f42c1e788da832: function(arg0) {
            const ret = arg0.getSupportedExtensions();
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_getSupportedProfiles_e4f6fd61b7c0362c: function(arg0) {
            const ret = arg0.getSupportedProfiles();
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_getSyncParameter_9f6e0bba77b398fa: function(arg0, arg1, arg2) {
            const ret = arg0.getSyncParameter(arg1, arg2 >>> 0);
            return ret;
        },
        __wbg_getUniformBlockIndex_3aa1c4c48062a404: function(arg0, arg1, arg2, arg3) {
            const ret = arg0.getUniformBlockIndex(arg1, getStringFromWasm0(arg2, arg3));
            return ret;
        },
        __wbg_getUniformLocation_1717b4ed42e2ccee: function(arg0, arg1, arg2, arg3) {
            const ret = arg0.getUniformLocation(arg1, getStringFromWasm0(arg2, arg3));
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_getUniformLocation_46373021b59d8832: function(arg0, arg1, arg2, arg3) {
            const ret = arg0.getUniformLocation(arg1, getStringFromWasm0(arg2, arg3));
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_get_unchecked_1dfe6d05ad91d9b7: function(arg0, arg1) {
            const ret = arg0[arg1 >>> 0];
            return ret;
        },
        __wbg_height_aef2a2eb10d0d530: function(arg0) {
            const ret = arg0.height;
            return ret;
        },
        __wbg_includes_0ec85e8f9acc8cac: function(arg0, arg1, arg2) {
            const ret = arg0.includes(arg1, arg2);
            return ret;
        },
        __wbg_instanceof_HtmlCanvasElement_a02da0a417f1bf3f: function(arg0) {
            let result;
            try {
                result = arg0 instanceof HTMLCanvasElement;
            } catch (_) {
                result = false;
            }
            const ret = result;
            return ret;
        },
        __wbg_instanceof_WebGl2RenderingContext_419098f7bf88e87e: function(arg0) {
            let result;
            try {
                result = arg0 instanceof WebGL2RenderingContext;
            } catch (_) {
                result = false;
            }
            const ret = result;
            return ret;
        },
        __wbg_instanceof_Window_faa5cf994f49cca7: function(arg0) {
            let result;
            try {
                result = arg0 instanceof Window;
            } catch (_) {
                result = false;
            }
            const ret = result;
            return ret;
        },
        __wbg_invalidateFramebuffer_02a63100f262d6cb: function() { return handleError(function (arg0, arg1, arg2) {
            arg0.invalidateFramebuffer(arg1 >>> 0, arg2);
        }, arguments); },
        __wbg_is_032c49d03f47f420: function(arg0, arg1) {
            const ret = Object.is(arg0, arg1);
            return ret;
        },
        __wbg_length_2591a0f4f659a55c: function(arg0) {
            const ret = arg0.length;
            return ret;
        },
        __wbg_linkProgram_7689cb555b14a359: function(arg0, arg1) {
            arg0.linkProgram(arg1);
        },
        __wbg_linkProgram_ec865896be2835c2: function(arg0, arg1) {
            arg0.linkProgram(arg1);
        },
        __wbg_new_02d162bc6cf02f60: function() {
            const ret = new Object();
            return ret;
        },
        __wbg_new_227d7c05414eb861: function() {
            const ret = new Error();
            return ret;
        },
        __wbg_new_310879b66b6e95e1: function() {
            const ret = new Array();
            return ret;
        },
        __wbg_new_typed_c072c4ce9a2a0cdf: function(arg0, arg1) {
            try {
                var state0 = {a: arg0, b: arg1};
                var cb0 = (arg0, arg1) => {
                    const a = state0.a;
                    state0.a = 0;
                    try {
                        return wasm_bindgen__convert__closures_____invoke__h5ec2816eae335d2e(a, state0.b, arg0, arg1);
                    } finally {
                        state0.a = a;
                    }
                };
                const ret = new Promise(cb0);
                return ret;
            } finally {
                state0.a = 0;
            }
        },
        __wbg_of_d694dacacb7afa7f: function(arg0) {
            const ret = Array.of(arg0);
            return ret;
        },
        __wbg_pixelStorei_06b86995306b01dc: function(arg0, arg1, arg2) {
            arg0.pixelStorei(arg1 >>> 0, arg2);
        },
        __wbg_pixelStorei_171e6a6629fd9e3c: function(arg0, arg1, arg2) {
            arg0.pixelStorei(arg1 >>> 0, arg2);
        },
        __wbg_polygonOffset_690c52c5bfca2a27: function(arg0, arg1, arg2) {
            arg0.polygonOffset(arg1, arg2);
        },
        __wbg_polygonOffset_cd648f07839ab009: function(arg0, arg1, arg2) {
            arg0.polygonOffset(arg1, arg2);
        },
        __wbg_push_b77c476b01548d0a: function(arg0, arg1) {
            const ret = arg0.push(arg1);
            return ret;
        },
        __wbg_queryCounterEXT_d92c246603070eed: function(arg0, arg1, arg2) {
            arg0.queryCounterEXT(arg1, arg2 >>> 0);
        },
        __wbg_querySelector_54149fe79b2a2091: function() { return handleError(function (arg0, arg1, arg2) {
            const ret = arg0.querySelector(getStringFromWasm0(arg1, arg2));
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        }, arguments); },
        __wbg_queueMicrotask_78d584b53af520f5: function(arg0) {
            const ret = arg0.queueMicrotask;
            return ret;
        },
        __wbg_queueMicrotask_b39ea83c7f01971a: function(arg0) {
            queueMicrotask(arg0);
        },
        __wbg_readBuffer_dc685ea6f3a7d5aa: function(arg0, arg1) {
            arg0.readBuffer(arg1 >>> 0);
        },
        __wbg_readPixels_0529efa834a6960a: function() { return handleError(function (arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7) {
            arg0.readPixels(arg1, arg2, arg3, arg4, arg5 >>> 0, arg6 >>> 0, arg7);
        }, arguments); },
        __wbg_readPixels_3509816172f67b8a: function() { return handleError(function (arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7) {
            arg0.readPixels(arg1, arg2, arg3, arg4, arg5 >>> 0, arg6 >>> 0, arg7);
        }, arguments); },
        __wbg_readPixels_76225de67eebec03: function() { return handleError(function (arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7) {
            arg0.readPixels(arg1, arg2, arg3, arg4, arg5 >>> 0, arg6 >>> 0, arg7);
        }, arguments); },
        __wbg_renderbufferStorageMultisample_25941e0e73e50cd2: function(arg0, arg1, arg2, arg3, arg4, arg5) {
            arg0.renderbufferStorageMultisample(arg1 >>> 0, arg2, arg3 >>> 0, arg4, arg5);
        },
        __wbg_renderbufferStorage_e46ef4833287e3bf: function(arg0, arg1, arg2, arg3, arg4) {
            arg0.renderbufferStorage(arg1 >>> 0, arg2 >>> 0, arg3, arg4);
        },
        __wbg_renderbufferStorage_fd35a40ea121e819: function(arg0, arg1, arg2, arg3, arg4) {
            arg0.renderbufferStorage(arg1 >>> 0, arg2 >>> 0, arg3, arg4);
        },
        __wbg_resolve_d17db9352f5a220e: function(arg0) {
            const ret = Promise.resolve(arg0);
            return ret;
        },
        __wbg_samplerParameterf_eb39264d0b3431ea: function(arg0, arg1, arg2, arg3) {
            arg0.samplerParameterf(arg1, arg2 >>> 0, arg3);
        },
        __wbg_samplerParameteri_7a90e6197a393b63: function(arg0, arg1, arg2, arg3) {
            arg0.samplerParameteri(arg1, arg2 >>> 0, arg3);
        },
        __wbg_scissor_eefeb709a030fe62: function(arg0, arg1, arg2, arg3, arg4) {
            arg0.scissor(arg1, arg2, arg3, arg4);
        },
        __wbg_scissor_ffbc9d8b3e5bb99b: function(arg0, arg1, arg2, arg3, arg4) {
            arg0.scissor(arg1, arg2, arg3, arg4);
        },
        __wbg_set_a0e911be3da02782: function() { return handleError(function (arg0, arg1, arg2) {
            const ret = Reflect.set(arg0, arg1, arg2);
            return ret;
        }, arguments); },
        __wbg_set_height_bb0dc35fd1d941f5: function(arg0, arg1) {
            arg0.height = arg1 >>> 0;
        },
        __wbg_set_height_bdd58e6b04e88cca: function(arg0, arg1) {
            arg0.height = arg1 >>> 0;
        },
        __wbg_set_width_25112eb6bf1148df: function(arg0, arg1) {
            arg0.width = arg1 >>> 0;
        },
        __wbg_set_width_9d385df435c1f79d: function(arg0, arg1) {
            arg0.width = arg1 >>> 0;
        },
        __wbg_shaderSource_a304cd4ebd95c11b: function(arg0, arg1, arg2, arg3) {
            arg0.shaderSource(arg1, getStringFromWasm0(arg2, arg3));
        },
        __wbg_shaderSource_eceb56c4b827824d: function(arg0, arg1, arg2, arg3) {
            arg0.shaderSource(arg1, getStringFromWasm0(arg2, arg3));
        },
        __wbg_stack_3b0d974bbf31e44f: function(arg0, arg1) {
            const ret = arg1.stack;
            const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            const len1 = WASM_VECTOR_LEN;
            getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
        },
        __wbg_static_accessor_GLOBAL_THIS_02344c9b09eb08a9: function() {
            const ret = typeof globalThis === 'undefined' ? null : globalThis;
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_static_accessor_GLOBAL_ac6d4ac874d5cd54: function() {
            const ret = typeof global === 'undefined' ? null : global;
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_static_accessor_SELF_9b2406c23aeb2023: function() {
            const ret = typeof self === 'undefined' ? null : self;
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_static_accessor_WINDOW_b34d2126934e16ba: function() {
            const ret = typeof window === 'undefined' ? null : window;
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_stencilFuncSeparate_00281c346ccf1e19: function(arg0, arg1, arg2, arg3, arg4) {
            arg0.stencilFuncSeparate(arg1 >>> 0, arg2 >>> 0, arg3, arg4 >>> 0);
        },
        __wbg_stencilFuncSeparate_5f7154fe74881dab: function(arg0, arg1, arg2, arg3, arg4) {
            arg0.stencilFuncSeparate(arg1 >>> 0, arg2 >>> 0, arg3, arg4 >>> 0);
        },
        __wbg_stencilMaskSeparate_bd7c034fdfc6620c: function(arg0, arg1, arg2) {
            arg0.stencilMaskSeparate(arg1 >>> 0, arg2 >>> 0);
        },
        __wbg_stencilMaskSeparate_d14d6ba494aeff5f: function(arg0, arg1, arg2) {
            arg0.stencilMaskSeparate(arg1 >>> 0, arg2 >>> 0);
        },
        __wbg_stencilMask_15dfb3e60c15e612: function(arg0, arg1) {
            arg0.stencilMask(arg1 >>> 0);
        },
        __wbg_stencilMask_2d63c2d3e068aca1: function(arg0, arg1) {
            arg0.stencilMask(arg1 >>> 0);
        },
        __wbg_stencilOpSeparate_1fea3ed309a817f9: function(arg0, arg1, arg2, arg3, arg4) {
            arg0.stencilOpSeparate(arg1 >>> 0, arg2 >>> 0, arg3 >>> 0, arg4 >>> 0);
        },
        __wbg_stencilOpSeparate_32876bf4c07b7065: function(arg0, arg1, arg2, arg3, arg4) {
            arg0.stencilOpSeparate(arg1 >>> 0, arg2 >>> 0, arg3 >>> 0, arg4 >>> 0);
        },
        __wbg_texImage2D_17593ae6c467ae79: function() { return handleError(function (arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9) {
            arg0.texImage2D(arg1 >>> 0, arg2, arg3, arg4, arg5, arg6, arg7 >>> 0, arg8 >>> 0, arg9);
        }, arguments); },
        __wbg_texImage2D_2495ff54823b531b: function() { return handleError(function (arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9) {
            arg0.texImage2D(arg1 >>> 0, arg2, arg3, arg4, arg5, arg6, arg7 >>> 0, arg8 >>> 0, arg9);
        }, arguments); },
        __wbg_texImage2D_364c83aae17ba6d2: function() { return handleError(function (arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9) {
            arg0.texImage2D(arg1 >>> 0, arg2, arg3, arg4, arg5, arg6, arg7 >>> 0, arg8 >>> 0, arg9);
        }, arguments); },
        __wbg_texImage3D_3bcfec50659cc5ae: function() { return handleError(function (arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10) {
            arg0.texImage3D(arg1 >>> 0, arg2, arg3, arg4, arg5, arg6, arg7, arg8 >>> 0, arg9 >>> 0, arg10);
        }, arguments); },
        __wbg_texImage3D_79d27507fa4470dd: function() { return handleError(function (arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10) {
            arg0.texImage3D(arg1 >>> 0, arg2, arg3, arg4, arg5, arg6, arg7, arg8 >>> 0, arg9 >>> 0, arg10);
        }, arguments); },
        __wbg_texParameteri_2ef5b781bcfbdd64: function(arg0, arg1, arg2, arg3) {
            arg0.texParameteri(arg1 >>> 0, arg2 >>> 0, arg3);
        },
        __wbg_texParameteri_c22838926a5dca2b: function(arg0, arg1, arg2, arg3) {
            arg0.texParameteri(arg1 >>> 0, arg2 >>> 0, arg3);
        },
        __wbg_texStorage2D_afb762382f8a4678: function(arg0, arg1, arg2, arg3, arg4, arg5) {
            arg0.texStorage2D(arg1 >>> 0, arg2, arg3 >>> 0, arg4, arg5);
        },
        __wbg_texStorage3D_66ff900ad02f2247: function(arg0, arg1, arg2, arg3, arg4, arg5, arg6) {
            arg0.texStorage3D(arg1 >>> 0, arg2, arg3 >>> 0, arg4, arg5, arg6);
        },
        __wbg_texSubImage2D_0f88243806532534: function() { return handleError(function (arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9) {
            arg0.texSubImage2D(arg1 >>> 0, arg2, arg3, arg4, arg5, arg6, arg7 >>> 0, arg8 >>> 0, arg9);
        }, arguments); },
        __wbg_texSubImage2D_203ff6bcf48e4d08: function() { return handleError(function (arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9) {
            arg0.texSubImage2D(arg1 >>> 0, arg2, arg3, arg4, arg5, arg6, arg7 >>> 0, arg8 >>> 0, arg9);
        }, arguments); },
        __wbg_texSubImage2D_57a710f2064ab4ef: function() { return handleError(function (arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9) {
            arg0.texSubImage2D(arg1 >>> 0, arg2, arg3, arg4, arg5, arg6, arg7 >>> 0, arg8 >>> 0, arg9);
        }, arguments); },
        __wbg_texSubImage2D_62d9e38e9378faff: function() { return handleError(function (arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9) {
            arg0.texSubImage2D(arg1 >>> 0, arg2, arg3, arg4, arg5, arg6, arg7 >>> 0, arg8 >>> 0, arg9);
        }, arguments); },
        __wbg_texSubImage2D_668c5714e23e0e83: function() { return handleError(function (arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9) {
            arg0.texSubImage2D(arg1 >>> 0, arg2, arg3, arg4, arg5, arg6, arg7 >>> 0, arg8 >>> 0, arg9);
        }, arguments); },
        __wbg_texSubImage2D_781892a0e05abd13: function() { return handleError(function (arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9) {
            arg0.texSubImage2D(arg1 >>> 0, arg2, arg3, arg4, arg5, arg6, arg7 >>> 0, arg8 >>> 0, arg9);
        }, arguments); },
        __wbg_texSubImage2D_ad417daf4e038863: function() { return handleError(function (arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9) {
            arg0.texSubImage2D(arg1 >>> 0, arg2, arg3, arg4, arg5, arg6, arg7 >>> 0, arg8 >>> 0, arg9);
        }, arguments); },
        __wbg_texSubImage2D_e1be0f65e9a35343: function() { return handleError(function (arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9) {
            arg0.texSubImage2D(arg1 >>> 0, arg2, arg3, arg4, arg5, arg6, arg7 >>> 0, arg8 >>> 0, arg9);
        }, arguments); },
        __wbg_texSubImage3D_11a4e6f278359fc4: function() { return handleError(function (arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11) {
            arg0.texSubImage3D(arg1 >>> 0, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9 >>> 0, arg10 >>> 0, arg11);
        }, arguments); },
        __wbg_texSubImage3D_36a195d4f535cfe6: function() { return handleError(function (arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11) {
            arg0.texSubImage3D(arg1 >>> 0, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9 >>> 0, arg10 >>> 0, arg11);
        }, arguments); },
        __wbg_texSubImage3D_54374f7f12d16e40: function() { return handleError(function (arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11) {
            arg0.texSubImage3D(arg1 >>> 0, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9 >>> 0, arg10 >>> 0, arg11);
        }, arguments); },
        __wbg_texSubImage3D_5cfc6bdc70a23b0d: function() { return handleError(function (arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11) {
            arg0.texSubImage3D(arg1 >>> 0, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9 >>> 0, arg10 >>> 0, arg11);
        }, arguments); },
        __wbg_texSubImage3D_72a9517857b52f44: function() { return handleError(function (arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11) {
            arg0.texSubImage3D(arg1 >>> 0, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9 >>> 0, arg10 >>> 0, arg11);
        }, arguments); },
        __wbg_texSubImage3D_a5b225452b0d7de3: function() { return handleError(function (arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11) {
            arg0.texSubImage3D(arg1 >>> 0, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9 >>> 0, arg10 >>> 0, arg11);
        }, arguments); },
        __wbg_texSubImage3D_ebb4d2dbc4680374: function() { return handleError(function (arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11) {
            arg0.texSubImage3D(arg1 >>> 0, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9 >>> 0, arg10 >>> 0, arg11);
        }, arguments); },
        __wbg_then_837494e384b37459: function(arg0, arg1) {
            const ret = arg0.then(arg1);
            return ret;
        },
        __wbg_uniform1f_429e664ea89191db: function(arg0, arg1, arg2) {
            arg0.uniform1f(arg1, arg2);
        },
        __wbg_uniform1f_709baed741125e5e: function(arg0, arg1, arg2) {
            arg0.uniform1f(arg1, arg2);
        },
        __wbg_uniform1i_2be01a75c6619c15: function(arg0, arg1, arg2) {
            arg0.uniform1i(arg1, arg2);
        },
        __wbg_uniform1i_717096cfb8ca6bc1: function(arg0, arg1, arg2) {
            arg0.uniform1i(arg1, arg2);
        },
        __wbg_uniform1ui_eafd8b7523d6d39e: function(arg0, arg1, arg2) {
            arg0.uniform1ui(arg1, arg2 >>> 0);
        },
        __wbg_uniform2fv_63f8c49c9f57e258: function(arg0, arg1, arg2, arg3) {
            arg0.uniform2fv(arg1, getArrayF32FromWasm0(arg2, arg3));
        },
        __wbg_uniform2fv_9f8ce1c86ee13440: function(arg0, arg1, arg2, arg3) {
            arg0.uniform2fv(arg1, getArrayF32FromWasm0(arg2, arg3));
        },
        __wbg_uniform2iv_c67b4ee9d082abdf: function(arg0, arg1, arg2, arg3) {
            arg0.uniform2iv(arg1, getArrayI32FromWasm0(arg2, arg3));
        },
        __wbg_uniform2iv_ec7e5887f2386d2c: function(arg0, arg1, arg2, arg3) {
            arg0.uniform2iv(arg1, getArrayI32FromWasm0(arg2, arg3));
        },
        __wbg_uniform2uiv_55a0e084de75c7b9: function(arg0, arg1, arg2, arg3) {
            arg0.uniform2uiv(arg1, getArrayU32FromWasm0(arg2, arg3));
        },
        __wbg_uniform3fv_2fb5418c1304ba72: function(arg0, arg1, arg2, arg3) {
            arg0.uniform3fv(arg1, getArrayF32FromWasm0(arg2, arg3));
        },
        __wbg_uniform3fv_7c2935b7f05414ef: function(arg0, arg1, arg2, arg3) {
            arg0.uniform3fv(arg1, getArrayF32FromWasm0(arg2, arg3));
        },
        __wbg_uniform3iv_ad46bb9ddf29111f: function(arg0, arg1, arg2, arg3) {
            arg0.uniform3iv(arg1, getArrayI32FromWasm0(arg2, arg3));
        },
        __wbg_uniform3iv_d82127ddeebb5154: function(arg0, arg1, arg2, arg3) {
            arg0.uniform3iv(arg1, getArrayI32FromWasm0(arg2, arg3));
        },
        __wbg_uniform3uiv_30e97efe980f53c9: function(arg0, arg1, arg2, arg3) {
            arg0.uniform3uiv(arg1, getArrayU32FromWasm0(arg2, arg3));
        },
        __wbg_uniform4f_7bc8db9ead983de4: function(arg0, arg1, arg2, arg3, arg4, arg5) {
            arg0.uniform4f(arg1, arg2, arg3, arg4, arg5);
        },
        __wbg_uniform4f_be0bd0ea203aedfe: function(arg0, arg1, arg2, arg3, arg4, arg5) {
            arg0.uniform4f(arg1, arg2, arg3, arg4, arg5);
        },
        __wbg_uniform4fv_622c64d35acf9214: function(arg0, arg1, arg2, arg3) {
            arg0.uniform4fv(arg1, getArrayF32FromWasm0(arg2, arg3));
        },
        __wbg_uniform4fv_b0c5721b35cd3f06: function(arg0, arg1, arg2, arg3) {
            arg0.uniform4fv(arg1, getArrayF32FromWasm0(arg2, arg3));
        },
        __wbg_uniform4iv_24df1fbc803c05db: function(arg0, arg1, arg2, arg3) {
            arg0.uniform4iv(arg1, getArrayI32FromWasm0(arg2, arg3));
        },
        __wbg_uniform4iv_2cccd5ae55d77224: function(arg0, arg1, arg2, arg3) {
            arg0.uniform4iv(arg1, getArrayI32FromWasm0(arg2, arg3));
        },
        __wbg_uniform4uiv_6f594d049d6d0038: function(arg0, arg1, arg2, arg3) {
            arg0.uniform4uiv(arg1, getArrayU32FromWasm0(arg2, arg3));
        },
        __wbg_uniformBlockBinding_25e6ae614200cf4d: function(arg0, arg1, arg2, arg3) {
            arg0.uniformBlockBinding(arg1, arg2 >>> 0, arg3 >>> 0);
        },
        __wbg_uniformMatrix2fv_6918fd0909b6a167: function(arg0, arg1, arg2, arg3, arg4) {
            arg0.uniformMatrix2fv(arg1, arg2 !== 0, getArrayF32FromWasm0(arg3, arg4));
        },
        __wbg_uniformMatrix2fv_840e6434707032cd: function(arg0, arg1, arg2, arg3, arg4) {
            arg0.uniformMatrix2fv(arg1, arg2 !== 0, getArrayF32FromWasm0(arg3, arg4));
        },
        __wbg_uniformMatrix2x3fv_4a2dd969ec740f7d: function(arg0, arg1, arg2, arg3, arg4) {
            arg0.uniformMatrix2x3fv(arg1, arg2 !== 0, getArrayF32FromWasm0(arg3, arg4));
        },
        __wbg_uniformMatrix2x4fv_e3cdd10c182a5354: function(arg0, arg1, arg2, arg3, arg4) {
            arg0.uniformMatrix2x4fv(arg1, arg2 !== 0, getArrayF32FromWasm0(arg3, arg4));
        },
        __wbg_uniformMatrix3fv_6abd62dbed68830a: function(arg0, arg1, arg2, arg3, arg4) {
            arg0.uniformMatrix3fv(arg1, arg2 !== 0, getArrayF32FromWasm0(arg3, arg4));
        },
        __wbg_uniformMatrix3fv_e380a7aa532c175a: function(arg0, arg1, arg2, arg3, arg4) {
            arg0.uniformMatrix3fv(arg1, arg2 !== 0, getArrayF32FromWasm0(arg3, arg4));
        },
        __wbg_uniformMatrix3x2fv_2b07ce888bfa37c8: function(arg0, arg1, arg2, arg3, arg4) {
            arg0.uniformMatrix3x2fv(arg1, arg2 !== 0, getArrayF32FromWasm0(arg3, arg4));
        },
        __wbg_uniformMatrix3x4fv_0439a4fdd88af9de: function(arg0, arg1, arg2, arg3, arg4) {
            arg0.uniformMatrix3x4fv(arg1, arg2 !== 0, getArrayF32FromWasm0(arg3, arg4));
        },
        __wbg_uniformMatrix4fv_b5f678dc15314524: function(arg0, arg1, arg2, arg3, arg4) {
            arg0.uniformMatrix4fv(arg1, arg2 !== 0, getArrayF32FromWasm0(arg3, arg4));
        },
        __wbg_uniformMatrix4fv_d2b5005a92d27115: function(arg0, arg1, arg2, arg3, arg4) {
            arg0.uniformMatrix4fv(arg1, arg2 !== 0, getArrayF32FromWasm0(arg3, arg4));
        },
        __wbg_uniformMatrix4x2fv_7d12ae09d4b61a26: function(arg0, arg1, arg2, arg3, arg4) {
            arg0.uniformMatrix4x2fv(arg1, arg2 !== 0, getArrayF32FromWasm0(arg3, arg4));
        },
        __wbg_uniformMatrix4x3fv_f60d424ca4a02635: function(arg0, arg1, arg2, arg3, arg4) {
            arg0.uniformMatrix4x3fv(arg1, arg2 !== 0, getArrayF32FromWasm0(arg3, arg4));
        },
        __wbg_useProgram_3cc1a6d58dac88b4: function(arg0, arg1) {
            arg0.useProgram(arg1);
        },
        __wbg_useProgram_e45f506b921ab3f8: function(arg0, arg1) {
            arg0.useProgram(arg1);
        },
        __wbg_vertexAttribDivisorANGLE_47b6b82921bbf062: function(arg0, arg1, arg2) {
            arg0.vertexAttribDivisorANGLE(arg1 >>> 0, arg2 >>> 0);
        },
        __wbg_vertexAttribDivisor_74454522a4976fc2: function(arg0, arg1, arg2) {
            arg0.vertexAttribDivisor(arg1 >>> 0, arg2 >>> 0);
        },
        __wbg_vertexAttribIPointer_e65b21fd97a67466: function(arg0, arg1, arg2, arg3, arg4, arg5) {
            arg0.vertexAttribIPointer(arg1 >>> 0, arg2, arg3 >>> 0, arg4, arg5);
        },
        __wbg_vertexAttribPointer_7f7185558bcaf24b: function(arg0, arg1, arg2, arg3, arg4, arg5, arg6) {
            arg0.vertexAttribPointer(arg1 >>> 0, arg2, arg3 >>> 0, arg4 !== 0, arg5, arg6);
        },
        __wbg_vertexAttribPointer_85566c79cb366300: function(arg0, arg1, arg2, arg3, arg4, arg5, arg6) {
            arg0.vertexAttribPointer(arg1 >>> 0, arg2, arg3 >>> 0, arg4 !== 0, arg5, arg6);
        },
        __wbg_viewport_3c149d0c6435f0ed: function(arg0, arg1, arg2, arg3, arg4) {
            arg0.viewport(arg1, arg2, arg3, arg4);
        },
        __wbg_viewport_c25030cfbe3cddf4: function(arg0, arg1, arg2, arg3, arg4) {
            arg0.viewport(arg1, arg2, arg3, arg4);
        },
        __wbg_warn_c4e0780980765a86: function(arg0) {
            console.warn(arg0);
        },
        __wbg_wasmrenderer_new: function(arg0) {
            const ret = WasmRenderer.__wrap(arg0);
            return ret;
        },
        __wbg_width_e987166926c3367c: function(arg0) {
            const ret = arg0.width;
            return ret;
        },
        __wbindgen_cast_0000000000000001: function(arg0, arg1) {
            // Cast intrinsic for `Closure(Closure { owned: true, function: Function { arguments: [Externref], shim_idx: 175, ret: Result(Unit), inner_ret: Some(Result(Unit)) }, mutable: true }) -> Externref`.
            const ret = makeMutClosure(arg0, arg1, wasm_bindgen__convert__closures_____invoke__h4eb714a55877aa02);
            return ret;
        },
        __wbindgen_cast_0000000000000002: function(arg0) {
            // Cast intrinsic for `F64 -> Externref`.
            const ret = arg0;
            return ret;
        },
        __wbindgen_cast_0000000000000003: function(arg0, arg1) {
            // Cast intrinsic for `Ref(Slice(F32)) -> NamedExternref("Float32Array")`.
            const ret = getArrayF32FromWasm0(arg0, arg1);
            return ret;
        },
        __wbindgen_cast_0000000000000004: function(arg0, arg1) {
            // Cast intrinsic for `Ref(Slice(I16)) -> NamedExternref("Int16Array")`.
            const ret = getArrayI16FromWasm0(arg0, arg1);
            return ret;
        },
        __wbindgen_cast_0000000000000005: function(arg0, arg1) {
            // Cast intrinsic for `Ref(Slice(I32)) -> NamedExternref("Int32Array")`.
            const ret = getArrayI32FromWasm0(arg0, arg1);
            return ret;
        },
        __wbindgen_cast_0000000000000006: function(arg0, arg1) {
            // Cast intrinsic for `Ref(Slice(I8)) -> NamedExternref("Int8Array")`.
            const ret = getArrayI8FromWasm0(arg0, arg1);
            return ret;
        },
        __wbindgen_cast_0000000000000007: function(arg0, arg1) {
            // Cast intrinsic for `Ref(Slice(U16)) -> NamedExternref("Uint16Array")`.
            const ret = getArrayU16FromWasm0(arg0, arg1);
            return ret;
        },
        __wbindgen_cast_0000000000000008: function(arg0, arg1) {
            // Cast intrinsic for `Ref(Slice(U32)) -> NamedExternref("Uint32Array")`.
            const ret = getArrayU32FromWasm0(arg0, arg1);
            return ret;
        },
        __wbindgen_cast_0000000000000009: function(arg0, arg1) {
            // Cast intrinsic for `Ref(Slice(U8)) -> NamedExternref("Uint8Array")`.
            const ret = getArrayU8FromWasm0(arg0, arg1);
            return ret;
        },
        __wbindgen_cast_000000000000000a: function(arg0, arg1) {
            // Cast intrinsic for `Ref(String) -> Externref`.
            const ret = getStringFromWasm0(arg0, arg1);
            return ret;
        },
        __wbindgen_init_externref_table: function() {
            const table = wasm.__wbindgen_externrefs;
            const offset = table.grow(4);
            table.set(0, undefined);
            table.set(offset + 0, undefined);
            table.set(offset + 1, null);
            table.set(offset + 2, true);
            table.set(offset + 3, false);
        },
    };
    return {
        __proto__: null,
        "./ferrum_wasm_bg.js": import0,
    };
}

function wasm_bindgen__convert__closures_____invoke__h4eb714a55877aa02(arg0, arg1, arg2) {
    const ret = wasm.wasm_bindgen__convert__closures_____invoke__h4eb714a55877aa02(arg0, arg1, arg2);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
}

function wasm_bindgen__convert__closures_____invoke__h5ec2816eae335d2e(arg0, arg1, arg2, arg3) {
    wasm.wasm_bindgen__convert__closures_____invoke__h5ec2816eae335d2e(arg0, arg1, arg2, arg3);
}

const WasmRendererFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmrenderer_free(ptr, 1));

function addToExternrefTable0(obj) {
    const idx = wasm.__externref_table_alloc();
    wasm.__wbindgen_externrefs.set(idx, obj);
    return idx;
}

const CLOSURE_DTORS = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(state => wasm.__wbindgen_destroy_closure(state.a, state.b));

function debugString(val) {
    // primitive types
    const type = typeof val;
    if (type == 'number' || type == 'boolean' || val == null) {
        return  `${val}`;
    }
    if (type == 'string') {
        return `"${val}"`;
    }
    if (type == 'symbol') {
        const description = val.description;
        if (description == null) {
            return 'Symbol';
        } else {
            return `Symbol(${description})`;
        }
    }
    if (type == 'function') {
        const name = val.name;
        if (typeof name == 'string' && name.length > 0) {
            return `Function(${name})`;
        } else {
            return 'Function';
        }
    }
    // objects
    if (Array.isArray(val)) {
        const length = val.length;
        let debug = '[';
        if (length > 0) {
            debug += debugString(val[0]);
        }
        for(let i = 1; i < length; i++) {
            debug += ', ' + debugString(val[i]);
        }
        debug += ']';
        return debug;
    }
    // Test for built-in
    const builtInMatches = /\[object ([^\]]+)\]/.exec(toString.call(val));
    let className;
    if (builtInMatches && builtInMatches.length > 1) {
        className = builtInMatches[1];
    } else {
        // Failed to match the standard '[object ClassName]'
        return toString.call(val);
    }
    if (className == 'Object') {
        // we're a user defined class or Object
        // JSON.stringify avoids problems with cycles, and is generally much
        // easier than looping through ownProperties of `val`.
        try {
            return 'Object(' + JSON.stringify(val) + ')';
        } catch (_) {
            return 'Object';
        }
    }
    // errors
    if (val instanceof Error) {
        return `${val.name}: ${val.message}\n${val.stack}`;
    }
    // TODO we could test for more things here, like `Set`s and `Map`s.
    return className;
}

function getArrayF32FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getFloat32ArrayMemory0().subarray(ptr / 4, ptr / 4 + len);
}

function getArrayI16FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getInt16ArrayMemory0().subarray(ptr / 2, ptr / 2 + len);
}

function getArrayI32FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getInt32ArrayMemory0().subarray(ptr / 4, ptr / 4 + len);
}

function getArrayI8FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getInt8ArrayMemory0().subarray(ptr / 1, ptr / 1 + len);
}

function getArrayU16FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getUint16ArrayMemory0().subarray(ptr / 2, ptr / 2 + len);
}

function getArrayU32FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getUint32ArrayMemory0().subarray(ptr / 4, ptr / 4 + len);
}

function getArrayU8FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getUint8ArrayMemory0().subarray(ptr / 1, ptr / 1 + len);
}

let cachedDataViewMemory0 = null;
function getDataViewMemory0() {
    if (cachedDataViewMemory0 === null || cachedDataViewMemory0.buffer.detached === true || (cachedDataViewMemory0.buffer.detached === undefined && cachedDataViewMemory0.buffer !== wasm.memory.buffer)) {
        cachedDataViewMemory0 = new DataView(wasm.memory.buffer);
    }
    return cachedDataViewMemory0;
}

let cachedFloat32ArrayMemory0 = null;
function getFloat32ArrayMemory0() {
    if (cachedFloat32ArrayMemory0 === null || cachedFloat32ArrayMemory0.byteLength === 0) {
        cachedFloat32ArrayMemory0 = new Float32Array(wasm.memory.buffer);
    }
    return cachedFloat32ArrayMemory0;
}

let cachedInt16ArrayMemory0 = null;
function getInt16ArrayMemory0() {
    if (cachedInt16ArrayMemory0 === null || cachedInt16ArrayMemory0.byteLength === 0) {
        cachedInt16ArrayMemory0 = new Int16Array(wasm.memory.buffer);
    }
    return cachedInt16ArrayMemory0;
}

let cachedInt32ArrayMemory0 = null;
function getInt32ArrayMemory0() {
    if (cachedInt32ArrayMemory0 === null || cachedInt32ArrayMemory0.byteLength === 0) {
        cachedInt32ArrayMemory0 = new Int32Array(wasm.memory.buffer);
    }
    return cachedInt32ArrayMemory0;
}

let cachedInt8ArrayMemory0 = null;
function getInt8ArrayMemory0() {
    if (cachedInt8ArrayMemory0 === null || cachedInt8ArrayMemory0.byteLength === 0) {
        cachedInt8ArrayMemory0 = new Int8Array(wasm.memory.buffer);
    }
    return cachedInt8ArrayMemory0;
}

function getStringFromWasm0(ptr, len) {
    return decodeText(ptr >>> 0, len);
}

let cachedUint16ArrayMemory0 = null;
function getUint16ArrayMemory0() {
    if (cachedUint16ArrayMemory0 === null || cachedUint16ArrayMemory0.byteLength === 0) {
        cachedUint16ArrayMemory0 = new Uint16Array(wasm.memory.buffer);
    }
    return cachedUint16ArrayMemory0;
}

let cachedUint32ArrayMemory0 = null;
function getUint32ArrayMemory0() {
    if (cachedUint32ArrayMemory0 === null || cachedUint32ArrayMemory0.byteLength === 0) {
        cachedUint32ArrayMemory0 = new Uint32Array(wasm.memory.buffer);
    }
    return cachedUint32ArrayMemory0;
}

let cachedUint8ArrayMemory0 = null;
function getUint8ArrayMemory0() {
    if (cachedUint8ArrayMemory0 === null || cachedUint8ArrayMemory0.byteLength === 0) {
        cachedUint8ArrayMemory0 = new Uint8Array(wasm.memory.buffer);
    }
    return cachedUint8ArrayMemory0;
}

function handleError(f, args) {
    try {
        return f.apply(this, args);
    } catch (e) {
        const idx = addToExternrefTable0(e);
        wasm.__wbindgen_exn_store(idx);
    }
}

function isLikeNone(x) {
    return x === undefined || x === null;
}

function makeMutClosure(arg0, arg1, f) {
    const state = { a: arg0, b: arg1, cnt: 1 };
    const real = (...args) => {

        // First up with a closure we increment the internal reference
        // count. This ensures that the Rust closure environment won't
        // be deallocated while we're invoking it.
        state.cnt++;
        const a = state.a;
        state.a = 0;
        try {
            return f(a, state.b, ...args);
        } finally {
            state.a = a;
            real._wbg_cb_unref();
        }
    };
    real._wbg_cb_unref = () => {
        if (--state.cnt === 0) {
            wasm.__wbindgen_destroy_closure(state.a, state.b);
            state.a = 0;
            CLOSURE_DTORS.unregister(state);
        }
    };
    CLOSURE_DTORS.register(real, state, state);
    return real;
}

function passArray8ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 1, 1) >>> 0;
    getUint8ArrayMemory0().set(arg, ptr / 1);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function passStringToWasm0(arg, malloc, realloc) {
    if (realloc === undefined) {
        const buf = cachedTextEncoder.encode(arg);
        const ptr = malloc(buf.length, 1) >>> 0;
        getUint8ArrayMemory0().subarray(ptr, ptr + buf.length).set(buf);
        WASM_VECTOR_LEN = buf.length;
        return ptr;
    }

    let len = arg.length;
    let ptr = malloc(len, 1) >>> 0;

    const mem = getUint8ArrayMemory0();

    let offset = 0;

    for (; offset < len; offset++) {
        const code = arg.charCodeAt(offset);
        if (code > 0x7F) break;
        mem[ptr + offset] = code;
    }
    if (offset !== len) {
        if (offset !== 0) {
            arg = arg.slice(offset);
        }
        ptr = realloc(ptr, len, len = offset + arg.length * 3, 1) >>> 0;
        const view = getUint8ArrayMemory0().subarray(ptr + offset, ptr + len);
        const ret = cachedTextEncoder.encodeInto(arg, view);

        offset += ret.written;
        ptr = realloc(ptr, len, offset, 1) >>> 0;
    }

    WASM_VECTOR_LEN = offset;
    return ptr;
}

function takeFromExternrefTable0(idx) {
    const value = wasm.__wbindgen_externrefs.get(idx);
    wasm.__externref_table_dealloc(idx);
    return value;
}

let cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
cachedTextDecoder.decode();
const MAX_SAFARI_DECODE_BYTES = 2146435072;
let numBytesDecoded = 0;
function decodeText(ptr, len) {
    numBytesDecoded += len;
    if (numBytesDecoded >= MAX_SAFARI_DECODE_BYTES) {
        cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
        cachedTextDecoder.decode();
        numBytesDecoded = len;
    }
    return cachedTextDecoder.decode(getUint8ArrayMemory0().subarray(ptr, ptr + len));
}

const cachedTextEncoder = new TextEncoder();

if (!('encodeInto' in cachedTextEncoder)) {
    cachedTextEncoder.encodeInto = function (arg, view) {
        const buf = cachedTextEncoder.encode(arg);
        view.set(buf);
        return {
            read: arg.length,
            written: buf.length
        };
    };
}

let WASM_VECTOR_LEN = 0;

let wasmModule, wasmInstance, wasm;
function __wbg_finalize_init(instance, module) {
    wasmInstance = instance;
    wasm = instance.exports;
    wasmModule = module;
    cachedDataViewMemory0 = null;
    cachedFloat32ArrayMemory0 = null;
    cachedInt16ArrayMemory0 = null;
    cachedInt32ArrayMemory0 = null;
    cachedInt8ArrayMemory0 = null;
    cachedUint16ArrayMemory0 = null;
    cachedUint32ArrayMemory0 = null;
    cachedUint8ArrayMemory0 = null;
    wasm.__wbindgen_start();
    return wasm;
}

async function __wbg_load(module, imports) {
    if (typeof Response === 'function' && module instanceof Response) {
        if (typeof WebAssembly.instantiateStreaming === 'function') {
            try {
                return await WebAssembly.instantiateStreaming(module, imports);
            } catch (e) {
                const validResponse = module.ok && expectedResponseType(module.type);

                if (validResponse && module.headers.get('Content-Type') !== 'application/wasm') {
                    console.warn("`WebAssembly.instantiateStreaming` failed because your server does not serve Wasm with `application/wasm` MIME type. Falling back to `WebAssembly.instantiate` which is slower. Original error:\n", e);

                } else { throw e; }
            }
        }

        const bytes = await module.arrayBuffer();
        return await WebAssembly.instantiate(bytes, imports);
    } else {
        const instance = await WebAssembly.instantiate(module, imports);

        if (instance instanceof WebAssembly.Instance) {
            return { instance, module };
        } else {
            return instance;
        }
    }

    function expectedResponseType(type) {
        switch (type) {
            case 'basic': case 'cors': case 'default': return true;
        }
        return false;
    }
}

function initSync(module) {
    if (wasm !== undefined) return wasm;


    if (module !== undefined) {
        if (Object.getPrototypeOf(module) === Object.prototype) {
            ({module} = module)
        } else {
            console.warn('using deprecated parameters for `initSync()`; pass a single object instead')
        }
    }

    const imports = __wbg_get_imports();
    if (!(module instanceof WebAssembly.Module)) {
        module = new WebAssembly.Module(module);
    }
    const instance = new WebAssembly.Instance(module, imports);
    return __wbg_finalize_init(instance, module);
}

async function __wbg_init(module_or_path) {
    if (wasm !== undefined) return wasm;


    if (module_or_path !== undefined) {
        if (Object.getPrototypeOf(module_or_path) === Object.prototype) {
            ({module_or_path} = module_or_path)
        } else {
            console.warn('using deprecated parameters for the initialization function; pass a single object instead')
        }
    }

    if (module_or_path === undefined) {
        module_or_path = new URL('ferrum_wasm_bg.wasm', import.meta.url);
    }
    const imports = __wbg_get_imports();

    if (typeof module_or_path === 'string' || (typeof Request === 'function' && module_or_path instanceof Request) || (typeof URL === 'function' && module_or_path instanceof URL)) {
        module_or_path = fetch(module_or_path);
    }

    const { instance, module } = await __wbg_load(await module_or_path, imports);

    return __wbg_finalize_init(instance, module);
}

export { initSync, __wbg_init as default };


var fi={value:()=>{}};function nr(){for(var t=0,e=arguments.length,r={},n;t<e;++t){if(!(n=arguments[t]+"")||n in r||/[\s.]/.test(n))throw new Error("illegal type: "+n);r[n]=[]}return new re(r)}function re(t){this._=t}function li(t,e){return t.trim().split(/^|\s+/).map(function(r){var n="",i=r.indexOf(".");if(i>=0&&(n=r.slice(i+1),r=r.slice(0,i)),r&&!e.hasOwnProperty(r))throw new Error("unknown type: "+r);return{type:r,name:n}})}re.prototype=nr.prototype={constructor:re,on:function(t,e){var r=this._,n=li(t+"",r),i,a=-1,o=n.length;if(arguments.length<2){for(;++a<o;)if((i=(t=n[a]).type)&&(i=ci(r[i],t.name)))return i;return}if(e!=null&&typeof e!="function")throw new Error("invalid callback: "+e);for(;++a<o;)if(i=(t=n[a]).type)r[i]=rr(r[i],t.name,e);else if(e==null)for(i in r)r[i]=rr(r[i],t.name,null);return this},copy:function(){var t={},e=this._;for(var r in e)t[r]=e[r].slice();return new re(t)},call:function(t,e){if((i=arguments.length-2)>0)for(var r=new Array(i),n=0,i,a;n<i;++n)r[n]=arguments[n+2];if(!this._.hasOwnProperty(t))throw new Error("unknown type: "+t);for(a=this._[t],n=0,i=a.length;n<i;++n)a[n].value.apply(e,r)},apply:function(t,e,r){if(!this._.hasOwnProperty(t))throw new Error("unknown type: "+t);for(var n=this._[t],i=0,a=n.length;i<a;++i)n[i].value.apply(e,r)}};function ci(t,e){for(var r=0,n=t.length,i;r<n;++r)if((i=t[r]).name===e)return i.value}function rr(t,e,r){for(var n=0,i=t.length;n<i;++n)if(t[n].name===e){t[n]=fi,t=t.slice(0,n).concat(t.slice(n+1));break}return r!=null&&t.push({name:e,value:r}),t}var xt=nr;var ne="http://www.w3.org/1999/xhtml",De={svg:"http://www.w3.org/2000/svg",xhtml:ne,xlink:"http://www.w3.org/1999/xlink",xml:"http://www.w3.org/XML/1998/namespace",xmlns:"http://www.w3.org/2000/xmlns/"};function ft(t){var e=t+="",r=e.indexOf(":");return r>=0&&(e=t.slice(0,r))!=="xmlns"&&(t=t.slice(r+1)),De.hasOwnProperty(e)?{space:De[e],local:t}:t}function hi(t){return function(){var e=this.ownerDocument,r=this.namespaceURI;return r===ne&&e.documentElement.namespaceURI===ne?e.createElement(t):e.createElementNS(r,t)}}function pi(t){return function(){return this.ownerDocument.createElementNS(t.space,t.local)}}function ie(t){var e=ft(t);return(e.local?pi:hi)(e)}function mi(){}function gt(t){return t==null?mi:function(){return this.querySelector(t)}}function ir(t){typeof t!="function"&&(t=gt(t));for(var e=this._groups,r=e.length,n=new Array(r),i=0;i<r;++i)for(var a=e[i],o=a.length,u=n[i]=new Array(o),s,l,c=0;c<o;++c)(s=a[c])&&(l=t.call(s,s.__data__,c,a))&&("__data__"in s&&(l.__data__=s.__data__),u[c]=l);return new D(n,this._parents)}function Rt(t){return t==null?[]:Array.isArray(t)?t:Array.from(t)}function di(){return[]}function Dt(t){return t==null?di:function(){return this.querySelectorAll(t)}}function xi(t){return function(){return Rt(t.apply(this,arguments))}}function or(t){typeof t=="function"?t=xi(t):t=Dt(t);for(var e=this._groups,r=e.length,n=[],i=[],a=0;a<r;++a)for(var o=e[a],u=o.length,s,l=0;l<u;++l)(s=o[l])&&(n.push(t.call(s,s.__data__,l,o)),i.push(s));return new D(n,i)}function $t(t){return function(){return this.matches(t)}}function oe(t){return function(e){return e.matches(t)}}var gi=Array.prototype.find;function yi(t){return function(){return gi.call(this.children,t)}}function _i(){return this.firstElementChild}function ar(t){return this.select(t==null?_i:yi(typeof t=="function"?t:oe(t)))}var wi=Array.prototype.filter;function vi(){return Array.from(this.children)}function bi(t){return function(){return wi.call(this.children,t)}}function ur(t){return this.selectAll(t==null?vi:bi(typeof t=="function"?t:oe(t)))}function sr(t){typeof t!="function"&&(t=$t(t));for(var e=this._groups,r=e.length,n=new Array(r),i=0;i<r;++i)for(var a=e[i],o=a.length,u=n[i]=[],s,l=0;l<o;++l)(s=a[l])&&t.call(s,s.__data__,l,a)&&u.push(s);return new D(n,this._parents)}function ae(t){return new Array(t.length)}function fr(){return new D(this._enter||this._groups.map(ae),this._parents)}function Xt(t,e){this.ownerDocument=t.ownerDocument,this.namespaceURI=t.namespaceURI,this._next=null,this._parent=t,this.__data__=e}Xt.prototype={constructor:Xt,appendChild:function(t){return this._parent.insertBefore(t,this._next)},insertBefore:function(t,e){return this._parent.insertBefore(t,e)},querySelector:function(t){return this._parent.querySelector(t)},querySelectorAll:function(t){return this._parent.querySelectorAll(t)}};function lr(t){return function(){return t}}function Ai(t,e,r,n,i,a){for(var o=0,u,s=e.length,l=a.length;o<l;++o)(u=e[o])?(u.__data__=a[o],n[o]=u):r[o]=new Xt(t,a[o]);for(;o<s;++o)(u=e[o])&&(i[o]=u)}function Ei(t,e,r,n,i,a,o){var u,s,l=new Map,c=e.length,x=a.length,g=new Array(c),w;for(u=0;u<c;++u)(s=e[u])&&(g[u]=w=o.call(s,s.__data__,u,e)+"",l.has(w)?i[u]=s:l.set(w,s));for(u=0;u<x;++u)w=o.call(t,a[u],u,a)+"",(s=l.get(w))?(n[u]=s,s.__data__=a[u],l.delete(w)):r[u]=new Xt(t,a[u]);for(u=0;u<c;++u)(s=e[u])&&l.get(g[u])===s&&(i[u]=s)}function ki(t){return t.__data__}function cr(t,e){if(!arguments.length)return Array.from(this,ki);var r=e?Ei:Ai,n=this._parents,i=this._groups;typeof t!="function"&&(t=lr(t));for(var a=i.length,o=new Array(a),u=new Array(a),s=new Array(a),l=0;l<a;++l){var c=n[l],x=i[l],g=x.length,w=Ni(t.call(c,c&&c.__data__,l,n)),z=w.length,C=u[l]=new Array(z),m=o[l]=new Array(z),p=s[l]=new Array(g);r(c,x,C,m,p,w,e);for(var b=0,_=0,N,I;b<z;++b)if(N=C[b]){for(b>=_&&(_=b+1);!(I=m[_])&&++_<z;);N._next=I||null}}return o=new D(o,n),o._enter=u,o._exit=s,o}function Ni(t){return typeof t=="object"&&"length"in t?t:Array.from(t)}function hr(){return new D(this._exit||this._groups.map(ae),this._parents)}function pr(t,e,r){var n=this.enter(),i=this,a=this.exit();return typeof t=="function"?(n=t(n),n&&(n=n.selection())):n=n.append(t+""),e!=null&&(i=e(i),i&&(i=i.selection())),r==null?a.remove():r(a),n&&i?n.merge(i).order():i}function mr(t){for(var e=t.selection?t.selection():t,r=this._groups,n=e._groups,i=r.length,a=n.length,o=Math.min(i,a),u=new Array(i),s=0;s<o;++s)for(var l=r[s],c=n[s],x=l.length,g=u[s]=new Array(x),w,z=0;z<x;++z)(w=l[z]||c[z])&&(g[z]=w);for(;s<i;++s)u[s]=r[s];return new D(u,this._parents)}function dr(){for(var t=this._groups,e=-1,r=t.length;++e<r;)for(var n=t[e],i=n.length-1,a=n[i],o;--i>=0;)(o=n[i])&&(a&&o.compareDocumentPosition(a)^4&&a.parentNode.insertBefore(o,a),a=o);return this}function xr(t){t||(t=Si);function e(x,g){return x&&g?t(x.__data__,g.__data__):!x-!g}for(var r=this._groups,n=r.length,i=new Array(n),a=0;a<n;++a){for(var o=r[a],u=o.length,s=i[a]=new Array(u),l,c=0;c<u;++c)(l=o[c])&&(s[c]=l);s.sort(e)}return new D(i,this._parents).order()}function Si(t,e){return t<e?-1:t>e?1:t>=e?0:NaN}function gr(){var t=arguments[0];return arguments[0]=this,t.apply(null,arguments),this}function yr(){return Array.from(this)}function _r(){for(var t=this._groups,e=0,r=t.length;e<r;++e)for(var n=t[e],i=0,a=n.length;i<a;++i){var o=n[i];if(o)return o}return null}function wr(){let t=0;for(let e of this)++t;return t}function vr(){return!this.node()}function br(t){for(var e=this._groups,r=0,n=e.length;r<n;++r)for(var i=e[r],a=0,o=i.length,u;a<o;++a)(u=i[a])&&t.call(u,u.__data__,a,i);return this}function Ti(t){return function(){this.removeAttribute(t)}}function Ii(t){return function(){this.removeAttributeNS(t.space,t.local)}}function zi(t,e){return function(){this.setAttribute(t,e)}}function Mi(t,e){return function(){this.setAttributeNS(t.space,t.local,e)}}function Ci(t,e){return function(){var r=e.apply(this,arguments);r==null?this.removeAttribute(t):this.setAttribute(t,r)}}function Oi(t,e){return function(){var r=e.apply(this,arguments);r==null?this.removeAttributeNS(t.space,t.local):this.setAttributeNS(t.space,t.local,r)}}function Ar(t,e){var r=ft(t);if(arguments.length<2){var n=this.node();return r.local?n.getAttributeNS(r.space,r.local):n.getAttribute(r)}return this.each((e==null?r.local?Ii:Ti:typeof e=="function"?r.local?Oi:Ci:r.local?Mi:zi)(r,e))}function ue(t){return t.ownerDocument&&t.ownerDocument.defaultView||t.document&&t||t.defaultView}function Ri(t){return function(){this.style.removeProperty(t)}}function Di(t,e,r){return function(){this.style.setProperty(t,e,r)}}function $i(t,e,r){return function(){var n=e.apply(this,arguments);n==null?this.style.removeProperty(t):this.style.setProperty(t,n,r)}}function Er(t,e,r){return arguments.length>1?this.each((e==null?Ri:typeof e=="function"?$i:Di)(t,e,r??"")):pt(this.node(),t)}function pt(t,e){return t.style.getPropertyValue(e)||ue(t).getComputedStyle(t,null).getPropertyValue(e)}function Xi(t){return function(){delete this[t]}}function Bi(t,e){return function(){this[t]=e}}function Pi(t,e){return function(){var r=e.apply(this,arguments);r==null?delete this[t]:this[t]=r}}function kr(t,e){return arguments.length>1?this.each((e==null?Xi:typeof e=="function"?Pi:Bi)(t,e)):this.node()[t]}function Nr(t){return t.trim().split(/^|\s+/)}function $e(t){return t.classList||new Sr(t)}function Sr(t){this._node=t,this._names=Nr(t.getAttribute("class")||"")}Sr.prototype={add:function(t){var e=this._names.indexOf(t);e<0&&(this._names.push(t),this._node.setAttribute("class",this._names.join(" ")))},remove:function(t){var e=this._names.indexOf(t);e>=0&&(this._names.splice(e,1),this._node.setAttribute("class",this._names.join(" ")))},contains:function(t){return this._names.indexOf(t)>=0}};function Tr(t,e){for(var r=$e(t),n=-1,i=e.length;++n<i;)r.add(e[n])}function Ir(t,e){for(var r=$e(t),n=-1,i=e.length;++n<i;)r.remove(e[n])}function qi(t){return function(){Tr(this,t)}}function Vi(t){return function(){Ir(this,t)}}function Hi(t,e){return function(){(e.apply(this,arguments)?Tr:Ir)(this,t)}}function zr(t,e){var r=Nr(t+"");if(arguments.length<2){for(var n=$e(this.node()),i=-1,a=r.length;++i<a;)if(!n.contains(r[i]))return!1;return!0}return this.each((typeof e=="function"?Hi:e?qi:Vi)(r,e))}function Yi(){this.textContent=""}function Fi(t){return function(){this.textContent=t}}function Li(t){return function(){var e=t.apply(this,arguments);this.textContent=e??""}}function Mr(t){return arguments.length?this.each(t==null?Yi:(typeof t=="function"?Li:Fi)(t)):this.node().textContent}function Gi(){this.innerHTML=""}function Ki(t){return function(){this.innerHTML=t}}function Ui(t){return function(){var e=t.apply(this,arguments);this.innerHTML=e??""}}function Cr(t){return arguments.length?this.each(t==null?Gi:(typeof t=="function"?Ui:Ki)(t)):this.node().innerHTML}function Qi(){this.nextSibling&&this.parentNode.appendChild(this)}function Or(){return this.each(Qi)}function Zi(){this.previousSibling&&this.parentNode.insertBefore(this,this.parentNode.firstChild)}function Rr(){return this.each(Zi)}function Dr(t){var e=typeof t=="function"?t:ie(t);return this.select(function(){return this.appendChild(e.apply(this,arguments))})}function Wi(){return null}function $r(t,e){var r=typeof t=="function"?t:ie(t),n=e==null?Wi:typeof e=="function"?e:gt(e);return this.select(function(){return this.insertBefore(r.apply(this,arguments),n.apply(this,arguments)||null)})}function Ji(){var t=this.parentNode;t&&t.removeChild(this)}function Xr(){return this.each(Ji)}function ji(){var t=this.cloneNode(!1),e=this.parentNode;return e?e.insertBefore(t,this.nextSibling):t}function to(){var t=this.cloneNode(!0),e=this.parentNode;return e?e.insertBefore(t,this.nextSibling):t}function Br(t){return this.select(t?to:ji)}function Pr(t){return arguments.length?this.property("__data__",t):this.node().__data__}function eo(t){return function(e){t.call(this,e,this.__data__)}}function ro(t){return t.trim().split(/^|\s+/).map(function(e){var r="",n=e.indexOf(".");return n>=0&&(r=e.slice(n+1),e=e.slice(0,n)),{type:e,name:r}})}function no(t){return function(){var e=this.__on;if(e){for(var r=0,n=-1,i=e.length,a;r<i;++r)a=e[r],(!t.type||a.type===t.type)&&a.name===t.name?this.removeEventListener(a.type,a.listener,a.options):e[++n]=a;++n?e.length=n:delete this.__on}}}function io(t,e,r){return function(){var n=this.__on,i,a=eo(e);if(n){for(var o=0,u=n.length;o<u;++o)if((i=n[o]).type===t.type&&i.name===t.name){this.removeEventListener(i.type,i.listener,i.options),this.addEventListener(i.type,i.listener=a,i.options=r),i.value=e;return}}this.addEventListener(t.type,a,r),i={type:t.type,name:t.name,value:e,listener:a,options:r},n?n.push(i):this.__on=[i]}}function qr(t,e,r){var n=ro(t+""),i,a=n.length,o;if(arguments.length<2){var u=this.node().__on;if(u){for(var s=0,l=u.length,c;s<l;++s)for(i=0,c=u[s];i<a;++i)if((o=n[i]).type===c.type&&o.name===c.name)return c.value}return}for(u=e?io:no,i=0;i<a;++i)this.each(u(n[i],e,r));return this}function Vr(t,e,r){var n=ue(t),i=n.CustomEvent;typeof i=="function"?i=new i(e,r):(i=n.document.createEvent("Event"),r?(i.initEvent(e,r.bubbles,r.cancelable),i.detail=r.detail):i.initEvent(e,!1,!1)),t.dispatchEvent(i)}function oo(t,e){return function(){return Vr(this,t,e)}}function ao(t,e){return function(){return Vr(this,t,e.apply(this,arguments))}}function Hr(t,e){return this.each((typeof e=="function"?ao:oo)(t,e))}function*Yr(){for(var t=this._groups,e=0,r=t.length;e<r;++e)for(var n=t[e],i=0,a=n.length,o;i<a;++i)(o=n[i])&&(yield o)}var Bt=[null];function D(t,e){this._groups=t,this._parents=e}function Fr(){return new D([[document.documentElement]],Bt)}function uo(){return this}D.prototype=Fr.prototype={constructor:D,select:ir,selectAll:or,selectChild:ar,selectChildren:ur,filter:sr,data:cr,enter:fr,exit:hr,join:pr,merge:mr,selection:uo,order:dr,sort:xr,call:gr,nodes:yr,node:_r,size:wr,empty:vr,each:br,attr:Ar,style:Er,property:kr,classed:zr,text:Mr,html:Cr,raise:Or,lower:Rr,append:Dr,insert:$r,remove:Xr,clone:Br,datum:Pr,on:qr,dispatch:Hr,[Symbol.iterator]:Yr};var at=Fr;function Y(t){return typeof t=="string"?new D([[document.querySelector(t)]],[document.documentElement]):new D([[t]],Bt)}function Lr(t){let e;for(;e=t.sourceEvent;)t=e;return t}function et(t,e){if(t=Lr(t),e===void 0&&(e=t.currentTarget),e){var r=e.ownerSVGElement||e;if(r.createSVGPoint){var n=r.createSVGPoint();return n.x=t.clientX,n.y=t.clientY,n=n.matrixTransform(e.getScreenCTM().inverse()),[n.x,n.y]}if(e.getBoundingClientRect){var i=e.getBoundingClientRect();return[t.clientX-i.left-e.clientLeft,t.clientY-i.top-e.clientTop]}}return[t.pageX,t.pageY]}function Gr(t){return typeof t=="string"?new D([document.querySelectorAll(t)],[document.documentElement]):new D([Rt(t)],Bt)}var se={capture:!0,passive:!1};function fe(t){t.preventDefault(),t.stopImmediatePropagation()}function Pt(t){var e=t.document.documentElement,r=Y(t).on("dragstart.drag",fe,se);"onselectstart"in e?r.on("selectstart.drag",fe,se):(e.__noselect=e.style.MozUserSelect,e.style.MozUserSelect="none")}function qt(t,e){var r=t.document.documentElement,n=Y(t).on("dragstart.drag",null);e&&(n.on("click.drag",fe,se),setTimeout(function(){n.on("click.drag",null)},0)),"onselectstart"in r?n.on("selectstart.drag",null):(r.style.MozUserSelect=r.__noselect,delete r.__noselect)}function le(t,e,r){t.prototype=e.prototype=r,r.constructor=t}function Xe(t,e){var r=Object.create(t.prototype);for(var n in e)r[n]=e[n];return r}function Yt(){}var Vt=.7,pe=1/Vt,Et="\\s*([+-]?\\d+)\\s*",Ht="\\s*([+-]?(?:\\d*\\.)?\\d+(?:[eE][+-]?\\d+)?)\\s*",ut="\\s*([+-]?(?:\\d*\\.)?\\d+(?:[eE][+-]?\\d+)?)%\\s*",so=/^#([0-9a-f]{3,8})$/,fo=new RegExp(`^rgb\\(${Et},${Et},${Et}\\)$`),lo=new RegExp(`^rgb\\(${ut},${ut},${ut}\\)$`),co=new RegExp(`^rgba\\(${Et},${Et},${Et},${Ht}\\)$`),ho=new RegExp(`^rgba\\(${ut},${ut},${ut},${Ht}\\)$`),po=new RegExp(`^hsl\\(${Ht},${ut},${ut}\\)$`),mo=new RegExp(`^hsla\\(${Ht},${ut},${ut},${Ht}\\)$`),Kr={aliceblue:15792383,antiquewhite:16444375,aqua:65535,aquamarine:8388564,azure:15794175,beige:16119260,bisque:16770244,black:0,blanchedalmond:16772045,blue:255,blueviolet:9055202,brown:10824234,burlywood:14596231,cadetblue:6266528,chartreuse:8388352,chocolate:13789470,coral:16744272,cornflowerblue:6591981,cornsilk:16775388,crimson:14423100,cyan:65535,darkblue:139,darkcyan:35723,darkgoldenrod:12092939,darkgray:11119017,darkgreen:25600,darkgrey:11119017,darkkhaki:12433259,darkmagenta:9109643,darkolivegreen:5597999,darkorange:16747520,darkorchid:10040012,darkred:9109504,darksalmon:15308410,darkseagreen:9419919,darkslateblue:4734347,darkslategray:3100495,darkslategrey:3100495,darkturquoise:52945,darkviolet:9699539,deeppink:16716947,deepskyblue:49151,dimgray:6908265,dimgrey:6908265,dodgerblue:2003199,firebrick:11674146,floralwhite:16775920,forestgreen:2263842,fuchsia:16711935,gainsboro:14474460,ghostwhite:16316671,gold:16766720,goldenrod:14329120,gray:8421504,green:32768,greenyellow:11403055,grey:8421504,honeydew:15794160,hotpink:16738740,indianred:13458524,indigo:4915330,ivory:16777200,khaki:15787660,lavender:15132410,lavenderblush:16773365,lawngreen:8190976,lemonchiffon:16775885,lightblue:11393254,lightcoral:15761536,lightcyan:14745599,lightgoldenrodyellow:16448210,lightgray:13882323,lightgreen:9498256,lightgrey:13882323,lightpink:16758465,lightsalmon:16752762,lightseagreen:2142890,lightskyblue:8900346,lightslategray:7833753,lightslategrey:7833753,lightsteelblue:11584734,lightyellow:16777184,lime:65280,limegreen:3329330,linen:16445670,magenta:16711935,maroon:8388608,mediumaquamarine:6737322,mediumblue:205,mediumorchid:12211667,mediumpurple:9662683,mediumseagreen:3978097,mediumslateblue:8087790,mediumspringgreen:64154,mediumturquoise:4772300,mediumvioletred:13047173,midnightblue:1644912,mintcream:16121850,mistyrose:16770273,moccasin:16770229,navajowhite:16768685,navy:128,oldlace:16643558,olive:8421376,olivedrab:7048739,orange:16753920,orangered:16729344,orchid:14315734,palegoldenrod:15657130,palegreen:10025880,paleturquoise:11529966,palevioletred:14381203,papayawhip:16773077,peachpuff:16767673,peru:13468991,pink:16761035,plum:14524637,powderblue:11591910,purple:8388736,rebeccapurple:6697881,red:16711680,rosybrown:12357519,royalblue:4286945,saddlebrown:9127187,salmon:16416882,sandybrown:16032864,seagreen:3050327,seashell:16774638,sienna:10506797,silver:12632256,skyblue:8900331,slateblue:6970061,slategray:7372944,slategrey:7372944,snow:16775930,springgreen:65407,steelblue:4620980,tan:13808780,teal:32896,thistle:14204888,tomato:16737095,turquoise:4251856,violet:15631086,wheat:16113331,white:16777215,whitesmoke:16119285,yellow:16776960,yellowgreen:10145074};le(Yt,it,{copy(t){return Object.assign(new this.constructor,this,t)},displayable(){return this.rgb().displayable()},hex:Ur,formatHex:Ur,formatHex8:xo,formatHsl:go,formatRgb:Qr,toString:Qr});function Ur(){return this.rgb().formatHex()}function xo(){return this.rgb().formatHex8()}function go(){return en(this).formatHsl()}function Qr(){return this.rgb().formatRgb()}function it(t){var e,r;return t=(t+"").trim().toLowerCase(),(e=so.exec(t))?(r=e[1].length,e=parseInt(e[1],16),r===6?Zr(e):r===3?new tt(e>>8&15|e>>4&240,e>>4&15|e&240,(e&15)<<4|e&15,1):r===8?ce(e>>24&255,e>>16&255,e>>8&255,(e&255)/255):r===4?ce(e>>12&15|e>>8&240,e>>8&15|e>>4&240,e>>4&15|e&240,((e&15)<<4|e&15)/255):null):(e=fo.exec(t))?new tt(e[1],e[2],e[3],1):(e=lo.exec(t))?new tt(e[1]*255/100,e[2]*255/100,e[3]*255/100,1):(e=co.exec(t))?ce(e[1],e[2],e[3],e[4]):(e=ho.exec(t))?ce(e[1]*255/100,e[2]*255/100,e[3]*255/100,e[4]):(e=po.exec(t))?jr(e[1],e[2]/100,e[3]/100,1):(e=mo.exec(t))?jr(e[1],e[2]/100,e[3]/100,e[4]):Kr.hasOwnProperty(t)?Zr(Kr[t]):t==="transparent"?new tt(NaN,NaN,NaN,0):null}function Zr(t){return new tt(t>>16&255,t>>8&255,t&255,1)}function ce(t,e,r,n){return n<=0&&(t=e=r=NaN),new tt(t,e,r,n)}function yo(t){return t instanceof Yt||(t=it(t)),t?(t=t.rgb(),new tt(t.r,t.g,t.b,t.opacity)):new tt}function kt(t,e,r,n){return arguments.length===1?yo(t):new tt(t,e,r,n??1)}function tt(t,e,r,n){this.r=+t,this.g=+e,this.b=+r,this.opacity=+n}le(tt,kt,Xe(Yt,{brighter(t){return t=t==null?pe:Math.pow(pe,t),new tt(this.r*t,this.g*t,this.b*t,this.opacity)},darker(t){return t=t==null?Vt:Math.pow(Vt,t),new tt(this.r*t,this.g*t,this.b*t,this.opacity)},rgb(){return this},clamp(){return new tt(_t(this.r),_t(this.g),_t(this.b),me(this.opacity))},displayable(){return-.5<=this.r&&this.r<255.5&&-.5<=this.g&&this.g<255.5&&-.5<=this.b&&this.b<255.5&&0<=this.opacity&&this.opacity<=1},hex:Wr,formatHex:Wr,formatHex8:_o,formatRgb:Jr,toString:Jr}));function Wr(){return`#${yt(this.r)}${yt(this.g)}${yt(this.b)}`}function _o(){return`#${yt(this.r)}${yt(this.g)}${yt(this.b)}${yt((isNaN(this.opacity)?1:this.opacity)*255)}`}function Jr(){let t=me(this.opacity);return`${t===1?"rgb(":"rgba("}${_t(this.r)}, ${_t(this.g)}, ${_t(this.b)}${t===1?")":`, ${t})`}`}function me(t){return isNaN(t)?1:Math.max(0,Math.min(1,t))}function _t(t){return Math.max(0,Math.min(255,Math.round(t)||0))}function yt(t){return t=_t(t),(t<16?"0":"")+t.toString(16)}function jr(t,e,r,n){return n<=0?t=e=r=NaN:r<=0||r>=1?t=e=NaN:e<=0&&(t=NaN),new nt(t,e,r,n)}function en(t){if(t instanceof nt)return new nt(t.h,t.s,t.l,t.opacity);if(t instanceof Yt||(t=it(t)),!t)return new nt;if(t instanceof nt)return t;t=t.rgb();var e=t.r/255,r=t.g/255,n=t.b/255,i=Math.min(e,r,n),a=Math.max(e,r,n),o=NaN,u=a-i,s=(a+i)/2;return u?(e===a?o=(r-n)/u+(r<n)*6:r===a?o=(n-e)/u+2:o=(e-r)/u+4,u/=s<.5?a+i:2-a-i,o*=60):u=s>0&&s<1?0:o,new nt(o,u,s,t.opacity)}function rn(t,e,r,n){return arguments.length===1?en(t):new nt(t,e,r,n??1)}function nt(t,e,r,n){this.h=+t,this.s=+e,this.l=+r,this.opacity=+n}le(nt,rn,Xe(Yt,{brighter(t){return t=t==null?pe:Math.pow(pe,t),new nt(this.h,this.s,this.l*t,this.opacity)},darker(t){return t=t==null?Vt:Math.pow(Vt,t),new nt(this.h,this.s,this.l*t,this.opacity)},rgb(){var t=this.h%360+(this.h<0)*360,e=isNaN(t)||isNaN(this.s)?0:this.s,r=this.l,n=r+(r<.5?r:1-r)*e,i=2*r-n;return new tt(Be(t>=240?t-240:t+120,i,n),Be(t,i,n),Be(t<120?t+240:t-120,i,n),this.opacity)},clamp(){return new nt(tn(this.h),he(this.s),he(this.l),me(this.opacity))},displayable(){return(0<=this.s&&this.s<=1||isNaN(this.s))&&0<=this.l&&this.l<=1&&0<=this.opacity&&this.opacity<=1},formatHsl(){let t=me(this.opacity);return`${t===1?"hsl(":"hsla("}${tn(this.h)}, ${he(this.s)*100}%, ${he(this.l)*100}%${t===1?")":`, ${t})`}`}}));function tn(t){return t=(t||0)%360,t<0?t+360:t}function he(t){return Math.max(0,Math.min(1,t||0))}function Be(t,e,r){return(t<60?e+(r-e)*t/60:t<180?r:t<240?e+(r-e)*(240-t)/60:e)*255}function Pe(t,e,r,n,i){var a=t*t,o=a*t;return((1-3*t+3*a-o)*e+(4-6*a+3*o)*r+(1+3*t+3*a-3*o)*n+o*i)/6}function nn(t){var e=t.length-1;return function(r){var n=r<=0?r=0:r>=1?(r=1,e-1):Math.floor(r*e),i=t[n],a=t[n+1],o=n>0?t[n-1]:2*i-a,u=n<e-1?t[n+2]:2*a-i;return Pe((r-n/e)*e,o,i,a,u)}}function on(t){var e=t.length;return function(r){var n=Math.floor(((r%=1)<0?++r:r)*e),i=t[(n+e-1)%e],a=t[n%e],o=t[(n+1)%e],u=t[(n+2)%e];return Pe((r-n/e)*e,i,a,o,u)}}var Ft=t=>()=>t;function wo(t,e){return function(r){return t+r*e}}function vo(t,e,r){return t=Math.pow(t,r),e=Math.pow(e,r)-t,r=1/r,function(n){return Math.pow(t+n*e,r)}}function an(t){return(t=+t)==1?de:function(e,r){return r-e?vo(e,r,t):Ft(isNaN(e)?r:e)}}function de(t,e){var r=e-t;return r?wo(t,r):Ft(isNaN(t)?e:t)}var wt=(function t(e){var r=an(e);function n(i,a){var o=r((i=kt(i)).r,(a=kt(a)).r),u=r(i.g,a.g),s=r(i.b,a.b),l=de(i.opacity,a.opacity);return function(c){return i.r=o(c),i.g=u(c),i.b=s(c),i.opacity=l(c),i+""}}return n.gamma=t,n})(1);function un(t){return function(e){var r=e.length,n=new Array(r),i=new Array(r),a=new Array(r),o,u;for(o=0;o<r;++o)u=kt(e[o]),n[o]=u.r||0,i[o]=u.g||0,a[o]=u.b||0;return n=t(n),i=t(i),a=t(a),u.opacity=1,function(s){return u.r=n(s),u.g=i(s),u.b=a(s),u+""}}}var bo=un(nn),Ao=un(on);function sn(t,e){e||(e=[]);var r=t?Math.min(e.length,t.length):0,n=e.slice(),i;return function(a){for(i=0;i<r;++i)n[i]=t[i]*(1-a)+e[i]*a;return n}}function fn(t){return ArrayBuffer.isView(t)&&!(t instanceof DataView)}function ln(t,e){var r=e?e.length:0,n=t?Math.min(r,t.length):0,i=new Array(n),a=new Array(r),o;for(o=0;o<n;++o)i[o]=vt(t[o],e[o]);for(;o<r;++o)a[o]=e[o];return function(u){for(o=0;o<n;++o)a[o]=i[o](u);return a}}function cn(t,e){var r=new Date;return t=+t,e=+e,function(n){return r.setTime(t*(1-n)+e*n),r}}function Z(t,e){return t=+t,e=+e,function(r){return t*(1-r)+e*r}}function hn(t,e){var r={},n={},i;(t===null||typeof t!="object")&&(t={}),(e===null||typeof e!="object")&&(e={});for(i in e)i in t?r[i]=vt(t[i],e[i]):n[i]=e[i];return function(a){for(i in r)n[i]=r[i](a);return n}}var Ve=/[-+]?(?:\d+\.?\d*|\.?\d+)(?:[eE][-+]?\d+)?/g,qe=new RegExp(Ve.source,"g");function Eo(t){return function(){return t}}function ko(t){return function(e){return t(e)+""}}function Lt(t,e){var r=Ve.lastIndex=qe.lastIndex=0,n,i,a,o=-1,u=[],s=[];for(t=t+"",e=e+"";(n=Ve.exec(t))&&(i=qe.exec(e));)(a=i.index)>r&&(a=e.slice(r,a),u[o]?u[o]+=a:u[++o]=a),(n=n[0])===(i=i[0])?u[o]?u[o]+=i:u[++o]=i:(u[++o]=null,s.push({i:o,x:Z(n,i)})),r=qe.lastIndex;return r<e.length&&(a=e.slice(r),u[o]?u[o]+=a:u[++o]=a),u.length<2?s[0]?ko(s[0].x):Eo(e):(e=s.length,function(l){for(var c=0,x;c<e;++c)u[(x=s[c]).i]=x.x(l);return u.join("")})}function vt(t,e){var r=typeof e,n;return e==null||r==="boolean"?Ft(e):(r==="number"?Z:r==="string"?(n=it(e))?(e=n,wt):Lt:e instanceof it?wt:e instanceof Date?cn:fn(e)?sn:Array.isArray(e)?ln:typeof e.valueOf!="function"&&typeof e.toString!="function"||isNaN(e)?hn:Z)(t,e)}var pn=180/Math.PI,xe={translateX:0,translateY:0,rotate:0,skewX:0,scaleX:1,scaleY:1};function He(t,e,r,n,i,a){var o,u,s;return(o=Math.sqrt(t*t+e*e))&&(t/=o,e/=o),(s=t*r+e*n)&&(r-=t*s,n-=e*s),(u=Math.sqrt(r*r+n*n))&&(r/=u,n/=u,s/=u),t*n<e*r&&(t=-t,e=-e,s=-s,o=-o),{translateX:i,translateY:a,rotate:Math.atan2(e,t)*pn,skewX:Math.atan(s)*pn,scaleX:o,scaleY:u}}var ge;function mn(t){let e=new(typeof DOMMatrix=="function"?DOMMatrix:WebKitCSSMatrix)(t+"");return e.isIdentity?xe:He(e.a,e.b,e.c,e.d,e.e,e.f)}function dn(t){return t==null?xe:(ge||(ge=document.createElementNS("http://www.w3.org/2000/svg","g")),ge.setAttribute("transform",t),(t=ge.transform.baseVal.consolidate())?(t=t.matrix,He(t.a,t.b,t.c,t.d,t.e,t.f)):xe)}function xn(t,e,r,n){function i(l){return l.length?l.pop()+" ":""}function a(l,c,x,g,w,z){if(l!==x||c!==g){var C=w.push("translate(",null,e,null,r);z.push({i:C-4,x:Z(l,x)},{i:C-2,x:Z(c,g)})}else(x||g)&&w.push("translate("+x+e+g+r)}function o(l,c,x,g){l!==c?(l-c>180?c+=360:c-l>180&&(l+=360),g.push({i:x.push(i(x)+"rotate(",null,n)-2,x:Z(l,c)})):c&&x.push(i(x)+"rotate("+c+n)}function u(l,c,x,g){l!==c?g.push({i:x.push(i(x)+"skewX(",null,n)-2,x:Z(l,c)}):c&&x.push(i(x)+"skewX("+c+n)}function s(l,c,x,g,w,z){if(l!==x||c!==g){var C=w.push(i(w)+"scale(",null,",",null,")");z.push({i:C-4,x:Z(l,x)},{i:C-2,x:Z(c,g)})}else(x!==1||g!==1)&&w.push(i(w)+"scale("+x+","+g+")")}return function(l,c){var x=[],g=[];return l=t(l),c=t(c),a(l.translateX,l.translateY,c.translateX,c.translateY,x,g),o(l.rotate,c.rotate,x,g),u(l.skewX,c.skewX,x,g),s(l.scaleX,l.scaleY,c.scaleX,c.scaleY,x,g),l=c=null,function(w){for(var z=-1,C=g.length,m;++z<C;)x[(m=g[z]).i]=m.x(w);return x.join("")}}}var Ye=xn(mn,"px, ","px)","deg)"),Fe=xn(dn,", ",")",")");var No=1e-12;function gn(t){return((t=Math.exp(t))+1/t)/2}function So(t){return((t=Math.exp(t))-1/t)/2}function To(t){return((t=Math.exp(2*t))-1)/(t+1)}var Le=(function t(e,r,n){function i(a,o){var u=a[0],s=a[1],l=a[2],c=o[0],x=o[1],g=o[2],w=c-u,z=x-s,C=w*w+z*z,m,p;if(C<No)p=Math.log(g/l)/e,m=function(B){return[u+B*w,s+B*z,l*Math.exp(e*B*p)]};else{var b=Math.sqrt(C),_=(g*g-l*l+n*C)/(2*l*r*b),N=(g*g-l*l-n*C)/(2*g*r*b),I=Math.log(Math.sqrt(_*_+1)-_),M=Math.log(Math.sqrt(N*N+1)-N);p=(M-I)/e,m=function(B){var $=B*p,L=gn(I),R=l/(r*b)*(L*To(e*$+I)-So(I));return[u+R*w,s+R*z,l*L/gn(e*$+I)]}}return m.duration=p*1e3*e/Math.SQRT2,m}return i.rho=function(a){var o=Math.max(.001,+a),u=o*o,s=u*u;return t(o,u,s)},i})(Math.SQRT2,2,4);var Nt=0,Kt=0,Gt=0,_n=1e3,ye,Ut,_e=0,bt=0,we=0,Qt=typeof performance=="object"&&performance.now?performance:Date,wn=typeof window=="object"&&window.requestAnimationFrame?window.requestAnimationFrame.bind(window):function(t){setTimeout(t,17)};function Wt(){return bt||(wn(Io),bt=Qt.now()+we)}function Io(){bt=0}function Zt(){this._call=this._time=this._next=null}Zt.prototype=ve.prototype={constructor:Zt,restart:function(t,e,r){if(typeof t!="function")throw new TypeError("callback is not a function");r=(r==null?Wt():+r)+(e==null?0:+e),!this._next&&Ut!==this&&(Ut?Ut._next=this:ye=this,Ut=this),this._call=t,this._time=r,Ge()},stop:function(){this._call&&(this._call=null,this._time=1/0,Ge())}};function ve(t,e,r){var n=new Zt;return n.restart(t,e,r),n}function vn(){Wt(),++Nt;for(var t=ye,e;t;)(e=bt-t._time)>=0&&t._call.call(void 0,e),t=t._next;--Nt}function yn(){bt=(_e=Qt.now())+we,Nt=Kt=0;try{vn()}finally{Nt=0,Mo(),bt=0}}function zo(){var t=Qt.now(),e=t-_e;e>_n&&(we-=e,_e=t)}function Mo(){for(var t,e=ye,r,n=1/0;e;)e._call?(n>e._time&&(n=e._time),t=e,e=e._next):(r=e._next,e._next=null,e=t?t._next=r:ye=r);Ut=t,Ge(n)}function Ge(t){if(!Nt){Kt&&(Kt=clearTimeout(Kt));var e=t-bt;e>24?(t<1/0&&(Kt=setTimeout(yn,t-Qt.now()-we)),Gt&&(Gt=clearInterval(Gt))):(Gt||(_e=Qt.now(),Gt=setInterval(zo,_n)),Nt=1,wn(yn))}}function be(t,e,r){var n=new Zt;return e=e==null?0:+e,n.restart(i=>{n.stop(),t(i+e)},e,r),n}var Co=xt("start","end","cancel","interrupt"),Oo=[],En=0,bn=1,Ee=2,Ae=3,An=4,ke=5,Jt=6;function mt(t,e,r,n,i,a){var o=t.__transition;if(!o)t.__transition={};else if(r in o)return;Ro(t,r,{name:e,index:n,group:i,on:Co,tween:Oo,time:a.time,delay:a.delay,duration:a.duration,ease:a.ease,timer:null,state:En})}function jt(t,e){var r=H(t,e);if(r.state>En)throw new Error("too late; already scheduled");return r}function F(t,e){var r=H(t,e);if(r.state>Ae)throw new Error("too late; already running");return r}function H(t,e){var r=t.__transition;if(!r||!(r=r[e]))throw new Error("transition not found");return r}function Ro(t,e,r){var n=t.__transition,i;n[e]=r,r.timer=ve(a,0,r.time);function a(l){r.state=bn,r.timer.restart(o,r.delay,r.time),r.delay<=l&&o(l-r.delay)}function o(l){var c,x,g,w;if(r.state!==bn)return s();for(c in n)if(w=n[c],w.name===r.name){if(w.state===Ae)return be(o);w.state===An?(w.state=Jt,w.timer.stop(),w.on.call("interrupt",t,t.__data__,w.index,w.group),delete n[c]):+c<e&&(w.state=Jt,w.timer.stop(),w.on.call("cancel",t,t.__data__,w.index,w.group),delete n[c])}if(be(function(){r.state===Ae&&(r.state=An,r.timer.restart(u,r.delay,r.time),u(l))}),r.state=Ee,r.on.call("start",t,t.__data__,r.index,r.group),r.state===Ee){for(r.state=Ae,i=new Array(g=r.tween.length),c=0,x=-1;c<g;++c)(w=r.tween[c].value.call(t,t.__data__,r.index,r.group))&&(i[++x]=w);i.length=x+1}}function u(l){for(var c=l<r.duration?r.ease.call(null,l/r.duration):(r.timer.restart(s),r.state=ke,1),x=-1,g=i.length;++x<g;)i[x].call(t,c);r.state===ke&&(r.on.call("end",t,t.__data__,r.index,r.group),s())}function s(){r.state=Jt,r.timer.stop(),delete n[e];for(var l in n)return;delete t.__transition}}function st(t,e){var r=t.__transition,n,i,a=!0,o;if(r){e=e==null?null:e+"";for(o in r){if((n=r[o]).name!==e){a=!1;continue}i=n.state>Ee&&n.state<ke,n.state=Jt,n.timer.stop(),n.on.call(i?"interrupt":"cancel",t,t.__data__,n.index,n.group),delete r[o]}a&&delete t.__transition}}function kn(t){return this.each(function(){st(this,t)})}function Do(t,e){var r,n;return function(){var i=F(this,t),a=i.tween;if(a!==r){n=r=a;for(var o=0,u=n.length;o<u;++o)if(n[o].name===e){n=n.slice(),n.splice(o,1);break}}i.tween=n}}function $o(t,e,r){var n,i;if(typeof r!="function")throw new Error;return function(){var a=F(this,t),o=a.tween;if(o!==n){i=(n=o).slice();for(var u={name:e,value:r},s=0,l=i.length;s<l;++s)if(i[s].name===e){i[s]=u;break}s===l&&i.push(u)}a.tween=i}}function Nn(t,e){var r=this._id;if(t+="",arguments.length<2){for(var n=H(this.node(),r).tween,i=0,a=n.length,o;i<a;++i)if((o=n[i]).name===t)return o.value;return null}return this.each((e==null?Do:$o)(r,t,e))}function St(t,e,r){var n=t._id;return t.each(function(){var i=F(this,n);(i.value||(i.value={}))[e]=r.apply(this,arguments)}),function(i){return H(i,n).value[e]}}function Ne(t,e){var r;return(typeof e=="number"?Z:e instanceof it?wt:(r=it(e))?(e=r,wt):Lt)(t,e)}function Xo(t){return function(){this.removeAttribute(t)}}function Bo(t){return function(){this.removeAttributeNS(t.space,t.local)}}function Po(t,e,r){var n,i=r+"",a;return function(){var o=this.getAttribute(t);return o===i?null:o===n?a:a=e(n=o,r)}}function qo(t,e,r){var n,i=r+"",a;return function(){var o=this.getAttributeNS(t.space,t.local);return o===i?null:o===n?a:a=e(n=o,r)}}function Vo(t,e,r){var n,i,a;return function(){var o,u=r(this),s;return u==null?void this.removeAttribute(t):(o=this.getAttribute(t),s=u+"",o===s?null:o===n&&s===i?a:(i=s,a=e(n=o,u)))}}function Ho(t,e,r){var n,i,a;return function(){var o,u=r(this),s;return u==null?void this.removeAttributeNS(t.space,t.local):(o=this.getAttributeNS(t.space,t.local),s=u+"",o===s?null:o===n&&s===i?a:(i=s,a=e(n=o,u)))}}function Sn(t,e){var r=ft(t),n=r==="transform"?Fe:Ne;return this.attrTween(t,typeof e=="function"?(r.local?Ho:Vo)(r,n,St(this,"attr."+t,e)):e==null?(r.local?Bo:Xo)(r):(r.local?qo:Po)(r,n,e))}function Yo(t,e){return function(r){this.setAttribute(t,e.call(this,r))}}function Fo(t,e){return function(r){this.setAttributeNS(t.space,t.local,e.call(this,r))}}function Lo(t,e){var r,n;function i(){var a=e.apply(this,arguments);return a!==n&&(r=(n=a)&&Fo(t,a)),r}return i._value=e,i}function Go(t,e){var r,n;function i(){var a=e.apply(this,arguments);return a!==n&&(r=(n=a)&&Yo(t,a)),r}return i._value=e,i}function Tn(t,e){var r="attr."+t;if(arguments.length<2)return(r=this.tween(r))&&r._value;if(e==null)return this.tween(r,null);if(typeof e!="function")throw new Error;var n=ft(t);return this.tween(r,(n.local?Lo:Go)(n,e))}function Ko(t,e){return function(){jt(this,t).delay=+e.apply(this,arguments)}}function Uo(t,e){return e=+e,function(){jt(this,t).delay=e}}function In(t){var e=this._id;return arguments.length?this.each((typeof t=="function"?Ko:Uo)(e,t)):H(this.node(),e).delay}function Qo(t,e){return function(){F(this,t).duration=+e.apply(this,arguments)}}function Zo(t,e){return e=+e,function(){F(this,t).duration=e}}function zn(t){var e=this._id;return arguments.length?this.each((typeof t=="function"?Qo:Zo)(e,t)):H(this.node(),e).duration}function Wo(t,e){if(typeof e!="function")throw new Error;return function(){F(this,t).ease=e}}function Mn(t){var e=this._id;return arguments.length?this.each(Wo(e,t)):H(this.node(),e).ease}function Jo(t,e){return function(){var r=e.apply(this,arguments);if(typeof r!="function")throw new Error;F(this,t).ease=r}}function Cn(t){if(typeof t!="function")throw new Error;return this.each(Jo(this._id,t))}function On(t){typeof t!="function"&&(t=$t(t));for(var e=this._groups,r=e.length,n=new Array(r),i=0;i<r;++i)for(var a=e[i],o=a.length,u=n[i]=[],s,l=0;l<o;++l)(s=a[l])&&t.call(s,s.__data__,l,a)&&u.push(s);return new W(n,this._parents,this._name,this._id)}function Rn(t){if(t._id!==this._id)throw new Error;for(var e=this._groups,r=t._groups,n=e.length,i=r.length,a=Math.min(n,i),o=new Array(n),u=0;u<a;++u)for(var s=e[u],l=r[u],c=s.length,x=o[u]=new Array(c),g,w=0;w<c;++w)(g=s[w]||l[w])&&(x[w]=g);for(;u<n;++u)o[u]=e[u];return new W(o,this._parents,this._name,this._id)}function jo(t){return(t+"").trim().split(/^|\s+/).every(function(e){var r=e.indexOf(".");return r>=0&&(e=e.slice(0,r)),!e||e==="start"})}function ta(t,e,r){var n,i,a=jo(e)?jt:F;return function(){var o=a(this,t),u=o.on;u!==n&&(i=(n=u).copy()).on(e,r),o.on=i}}function Dn(t,e){var r=this._id;return arguments.length<2?H(this.node(),r).on.on(t):this.each(ta(r,t,e))}function ea(t){return function(){var e=this.parentNode;for(var r in this.__transition)if(+r!==t)return;e&&e.removeChild(this)}}function $n(){return this.on("end.remove",ea(this._id))}function Xn(t){var e=this._name,r=this._id;typeof t!="function"&&(t=gt(t));for(var n=this._groups,i=n.length,a=new Array(i),o=0;o<i;++o)for(var u=n[o],s=u.length,l=a[o]=new Array(s),c,x,g=0;g<s;++g)(c=u[g])&&(x=t.call(c,c.__data__,g,u))&&("__data__"in c&&(x.__data__=c.__data__),l[g]=x,mt(l[g],e,r,g,l,H(c,r)));return new W(a,this._parents,e,r)}function Bn(t){var e=this._name,r=this._id;typeof t!="function"&&(t=Dt(t));for(var n=this._groups,i=n.length,a=[],o=[],u=0;u<i;++u)for(var s=n[u],l=s.length,c,x=0;x<l;++x)if(c=s[x]){for(var g=t.call(c,c.__data__,x,s),w,z=H(c,r),C=0,m=g.length;C<m;++C)(w=g[C])&&mt(w,e,r,C,g,z);a.push(g),o.push(c)}return new W(a,o,e,r)}var ra=at.prototype.constructor;function Pn(){return new ra(this._groups,this._parents)}function na(t,e){var r,n,i;return function(){var a=pt(this,t),o=(this.style.removeProperty(t),pt(this,t));return a===o?null:a===r&&o===n?i:i=e(r=a,n=o)}}function qn(t){return function(){this.style.removeProperty(t)}}function ia(t,e,r){var n,i=r+"",a;return function(){var o=pt(this,t);return o===i?null:o===n?a:a=e(n=o,r)}}function oa(t,e,r){var n,i,a;return function(){var o=pt(this,t),u=r(this),s=u+"";return u==null&&(s=u=(this.style.removeProperty(t),pt(this,t))),o===s?null:o===n&&s===i?a:(i=s,a=e(n=o,u))}}function aa(t,e){var r,n,i,a="style."+e,o="end."+a,u;return function(){var s=F(this,t),l=s.on,c=s.value[a]==null?u||(u=qn(e)):void 0;(l!==r||i!==c)&&(n=(r=l).copy()).on(o,i=c),s.on=n}}function Vn(t,e,r){var n=(t+="")=="transform"?Ye:Ne;return e==null?this.styleTween(t,na(t,n)).on("end.style."+t,qn(t)):typeof e=="function"?this.styleTween(t,oa(t,n,St(this,"style."+t,e))).each(aa(this._id,t)):this.styleTween(t,ia(t,n,e),r).on("end.style."+t,null)}function ua(t,e,r){return function(n){this.style.setProperty(t,e.call(this,n),r)}}function sa(t,e,r){var n,i;function a(){var o=e.apply(this,arguments);return o!==i&&(n=(i=o)&&ua(t,o,r)),n}return a._value=e,a}function Hn(t,e,r){var n="style."+(t+="");if(arguments.length<2)return(n=this.tween(n))&&n._value;if(e==null)return this.tween(n,null);if(typeof e!="function")throw new Error;return this.tween(n,sa(t,e,r??""))}function fa(t){return function(){this.textContent=t}}function la(t){return function(){var e=t(this);this.textContent=e??""}}function Yn(t){return this.tween("text",typeof t=="function"?la(St(this,"text",t)):fa(t==null?"":t+""))}function ca(t){return function(e){this.textContent=t.call(this,e)}}function ha(t){var e,r;function n(){var i=t.apply(this,arguments);return i!==r&&(e=(r=i)&&ca(i)),e}return n._value=t,n}function Fn(t){var e="text";if(arguments.length<1)return(e=this.tween(e))&&e._value;if(t==null)return this.tween(e,null);if(typeof t!="function")throw new Error;return this.tween(e,ha(t))}function Ln(){for(var t=this._name,e=this._id,r=Se(),n=this._groups,i=n.length,a=0;a<i;++a)for(var o=n[a],u=o.length,s,l=0;l<u;++l)if(s=o[l]){var c=H(s,e);mt(s,t,r,l,o,{time:c.time+c.delay+c.duration,delay:0,duration:c.duration,ease:c.ease})}return new W(n,this._parents,t,r)}function Gn(){var t,e,r=this,n=r._id,i=r.size();return new Promise(function(a,o){var u={value:o},s={value:function(){--i===0&&a()}};r.each(function(){var l=F(this,n),c=l.on;c!==t&&(e=(t=c).copy(),e._.cancel.push(u),e._.interrupt.push(u),e._.end.push(s)),l.on=e}),i===0&&a()})}var pa=0;function W(t,e,r,n){this._groups=t,this._parents=e,this._name=r,this._id=n}function Kn(t){return at().transition(t)}function Se(){return++pa}var lt=at.prototype;W.prototype=Kn.prototype={constructor:W,select:Xn,selectAll:Bn,selectChild:lt.selectChild,selectChildren:lt.selectChildren,filter:On,merge:Rn,selection:Pn,transition:Ln,call:lt.call,nodes:lt.nodes,node:lt.node,size:lt.size,empty:lt.empty,each:lt.each,on:Dn,attr:Sn,attrTween:Tn,style:Vn,styleTween:Hn,text:Yn,textTween:Fn,remove:$n,tween:Nn,delay:In,duration:zn,ease:Mn,easeVarying:Cn,end:Gn,[Symbol.iterator]:lt[Symbol.iterator]};function Te(t){return((t*=2)<=1?t*t*t:(t-=2)*t*t+2)/2}var ma={time:null,delay:0,duration:250,ease:Te};function da(t,e){for(var r;!(r=t.__transition)||!(r=r[e]);)if(!(t=t.parentNode))throw new Error(`transition ${e} not found`);return r}function Un(t){var e,r;t instanceof W?(e=t._id,t=t._name):(e=Se(),(r=ma).time=Wt(),t=t==null?null:t+"");for(var n=this._groups,i=n.length,a=0;a<i;++a)for(var o=n[a],u=o.length,s,l=0;l<u;++l)(s=o[l])&&mt(s,t,e,l,o,r||da(s,e));return new W(n,this._parents,t,e)}at.prototype.interrupt=kn;at.prototype.transition=Un;var Ie=t=>()=>t;function Ke(t,{sourceEvent:e,target:r,selection:n,mode:i,dispatch:a}){Object.defineProperties(this,{type:{value:t,enumerable:!0,configurable:!0},sourceEvent:{value:e,enumerable:!0,configurable:!0},target:{value:r,enumerable:!0,configurable:!0},selection:{value:n,enumerable:!0,configurable:!0},mode:{value:i,enumerable:!0,configurable:!0},_:{value:a}})}function Qn(t){t.stopImmediatePropagation()}function ze(t){t.preventDefault(),t.stopImmediatePropagation()}var Zn={name:"drag"},Ue={name:"space"},Tt={name:"handle"},It={name:"center"},{abs:Wn,max:K,min:U}=Math;function Jn(t){return[+t[0],+t[1]]}function Ze(t){return[Jn(t[0]),Jn(t[1])]}var Me={name:"x",handles:["w","e"].map(te),input:function(t,e){return t==null?null:[[+t[0],e[0][1]],[+t[1],e[1][1]]]},output:function(t){return t&&[t[0][0],t[1][0]]}},Ce={name:"y",handles:["n","s"].map(te),input:function(t,e){return t==null?null:[[e[0][0],+t[0]],[e[1][0],+t[1]]]},output:function(t){return t&&[t[0][1],t[1][1]]}},xa={name:"xy",handles:["n","w","e","s","nw","ne","sw","se"].map(te),input:function(t){return t==null?null:Ze(t)},output:function(t){return t}},ct={overlay:"crosshair",selection:"move",n:"ns-resize",e:"ew-resize",s:"ns-resize",w:"ew-resize",nw:"nwse-resize",ne:"nesw-resize",se:"nwse-resize",sw:"nesw-resize"},jn={e:"w",w:"e",nw:"ne",ne:"nw",se:"sw",sw:"se"},ti={n:"s",s:"n",nw:"sw",ne:"se",se:"ne",sw:"nw"},ga={overlay:1,selection:1,n:null,e:1,s:null,w:-1,nw:-1,ne:1,se:1,sw:-1},ya={overlay:1,selection:1,n:-1,e:null,s:1,w:null,nw:-1,ne:-1,se:1,sw:1};function te(t){return{type:t}}function _a(t){return!t.ctrlKey&&!t.button}function wa(){var t=this.ownerSVGElement||this;return t.hasAttribute("viewBox")?(t=t.viewBox.baseVal,[[t.x,t.y],[t.x+t.width,t.y+t.height]]):[[0,0],[t.width.baseVal.value,t.height.baseVal.value]]}function va(){return navigator.maxTouchPoints||"ontouchstart"in this}function Qe(t){for(;!t.__brush;)if(!(t=t.parentNode))return;return t.__brush}function ba(t){return t[0][0]===t[1][0]||t[0][1]===t[1][1]}function ei(t){var e=t.__brush;return e?e.dim.output(e.selection):null}function ri(){return We(Me)}function ni(){return We(Ce)}function ii(){return We(xa)}function We(t){var e=wa,r=_a,n=va,i=!0,a=xt("start","brush","end"),o=6,u;function s(m){var p=m.property("__brush",C).selectAll(".overlay").data([te("overlay")]);p.enter().append("rect").attr("class","overlay").attr("pointer-events","all").attr("cursor",ct.overlay).merge(p).each(function(){var _=Qe(this).extent;Y(this).attr("x",_[0][0]).attr("y",_[0][1]).attr("width",_[1][0]-_[0][0]).attr("height",_[1][1]-_[0][1])}),m.selectAll(".selection").data([te("selection")]).enter().append("rect").attr("class","selection").attr("cursor",ct.selection).attr("fill","#777").attr("fill-opacity",.3).attr("stroke","#fff").attr("shape-rendering","crispEdges");var b=m.selectAll(".handle").data(t.handles,function(_){return _.type});b.exit().remove(),b.enter().append("rect").attr("class",function(_){return"handle handle--"+_.type}).attr("cursor",function(_){return ct[_.type]}),m.each(l).attr("fill","none").attr("pointer-events","all").on("mousedown.brush",g).filter(n).on("touchstart.brush",g).on("touchmove.brush",w).on("touchend.brush touchcancel.brush",z).style("touch-action","none").style("-webkit-tap-highlight-color","rgba(0,0,0,0)")}s.move=function(m,p,b){m.tween?m.on("start.brush",function(_){c(this,arguments).beforestart().start(_)}).on("interrupt.brush end.brush",function(_){c(this,arguments).end(_)}).tween("brush",function(){var _=this,N=_.__brush,I=c(_,arguments),M=N.selection,B=t.input(typeof p=="function"?p.apply(this,arguments):p,N.extent),$=vt(M,B);function L(R){N.selection=R===1&&B===null?null:$(R),l.call(_),I.brush()}return M!==null&&B!==null?L:L(1)}):m.each(function(){var _=this,N=arguments,I=_.__brush,M=t.input(typeof p=="function"?p.apply(_,N):p,I.extent),B=c(_,N).beforestart();st(_),I.selection=M===null?null:M,l.call(_),B.start(b).brush(b).end(b)})},s.clear=function(m,p){s.move(m,null,p)};function l(){var m=Y(this),p=Qe(this).selection;p?(m.selectAll(".selection").style("display",null).attr("x",p[0][0]).attr("y",p[0][1]).attr("width",p[1][0]-p[0][0]).attr("height",p[1][1]-p[0][1]),m.selectAll(".handle").style("display",null).attr("x",function(b){return b.type[b.type.length-1]==="e"?p[1][0]-o/2:p[0][0]-o/2}).attr("y",function(b){return b.type[0]==="s"?p[1][1]-o/2:p[0][1]-o/2}).attr("width",function(b){return b.type==="n"||b.type==="s"?p[1][0]-p[0][0]+o:o}).attr("height",function(b){return b.type==="e"||b.type==="w"?p[1][1]-p[0][1]+o:o})):m.selectAll(".selection,.handle").style("display","none").attr("x",null).attr("y",null).attr("width",null).attr("height",null)}function c(m,p,b){var _=m.__brush.emitter;return _&&(!b||!_.clean)?_:new x(m,p,b)}function x(m,p,b){this.that=m,this.args=p,this.state=m.__brush,this.active=0,this.clean=b}x.prototype={beforestart:function(){return++this.active===1&&(this.state.emitter=this,this.starting=!0),this},start:function(m,p){return this.starting?(this.starting=!1,this.emit("start",m,p)):this.emit("brush",m),this},brush:function(m,p){return this.emit("brush",m,p),this},end:function(m,p){return--this.active===0&&(delete this.state.emitter,this.emit("end",m,p)),this},emit:function(m,p,b){var _=Y(this.that).datum();a.call(m,this.that,new Ke(m,{sourceEvent:p,target:s,selection:t.output(this.state.selection),mode:b,dispatch:a}),_)}};function g(m){if(u&&!m.touches||!r.apply(this,arguments))return;var p=this,b=m.target.__data__.type,_=(i&&m.metaKey?b="overlay":b)==="selection"?Zn:i&&m.altKey?It:Tt,N=t===Ce?null:ga[b],I=t===Me?null:ya[b],M=Qe(p),B=M.extent,$=M.selection,L=B[0][0],R,P,ot=B[0][1],q,f,y=B[1][0],h,d,E=B[1][1],v,A,k=0,S=0,J,X=N&&I&&i&&m.shiftKey,G,j,O=Array.from(m.touches||[m],T=>{let V=T.identifier;return T=et(T,p),T.point0=T.slice(),T.identifier=V,T});st(p);var Q=c(p,arguments,!0).beforestart();if(b==="overlay"){$&&(J=!0);let T=[O[0],O[1]||O[0]];M.selection=$=[[R=t===Ce?L:U(T[0][0],T[1][0]),q=t===Me?ot:U(T[0][1],T[1][1])],[h=t===Ce?y:K(T[0][0],T[1][0]),v=t===Me?E:K(T[0][1],T[1][1])]],O.length>1&&dt(m)}else R=$[0][0],q=$[0][1],h=$[1][0],v=$[1][1];P=R,f=q,d=h,A=v;var At=Y(p).attr("pointer-events","none"),Ct=At.selectAll(".overlay").attr("cursor",ct[b]);if(m.touches)Q.moved=tr,Q.ended=er;else{var je=Y(m.view).on("mousemove.brush",tr,!0).on("mouseup.brush",er,!0);i&&je.on("keydown.brush",ui,!0).on("keyup.brush",si,!0),Pt(m.view)}l.call(p),Q.start(m,_.name);function tr(T){for(let V of T.changedTouches||[T])for(let Ot of O)Ot.identifier===V.identifier&&(Ot.cur=et(V,p));if(X&&!G&&!j&&O.length===1){let V=O[0];Wn(V.cur[0]-V[0])>Wn(V.cur[1]-V[1])?j=!0:G=!0}for(let V of O)V.cur&&(V[0]=V.cur[0],V[1]=V.cur[1]);J=!0,ze(T),dt(T)}function dt(T){let V=O[0],Ot=V.point0;var ht;switch(k=V[0]-Ot[0],S=V[1]-Ot[1],_){case Ue:case Zn:{N&&(k=K(L-R,U(y-h,k)),P=R+k,d=h+k),I&&(S=K(ot-q,U(E-v,S)),f=q+S,A=v+S);break}case Tt:{O[1]?(N&&(P=K(L,U(y,O[0][0])),d=K(L,U(y,O[1][0])),N=1),I&&(f=K(ot,U(E,O[0][1])),A=K(ot,U(E,O[1][1])),I=1)):(N<0?(k=K(L-R,U(y-R,k)),P=R+k,d=h):N>0&&(k=K(L-h,U(y-h,k)),P=R,d=h+k),I<0?(S=K(ot-q,U(E-q,S)),f=q+S,A=v):I>0&&(S=K(ot-v,U(E-v,S)),f=q,A=v+S));break}case It:{N&&(P=K(L,U(y,R-k*N)),d=K(L,U(y,h+k*N))),I&&(f=K(ot,U(E,q-S*I)),A=K(ot,U(E,v+S*I)));break}}d<P&&(N*=-1,ht=R,R=h,h=ht,ht=P,P=d,d=ht,b in jn&&Ct.attr("cursor",ct[b=jn[b]])),A<f&&(I*=-1,ht=q,q=v,v=ht,ht=f,f=A,A=ht,b in ti&&Ct.attr("cursor",ct[b=ti[b]])),M.selection&&($=M.selection),G&&(P=$[0][0],d=$[1][0]),j&&(f=$[0][1],A=$[1][1]),($[0][0]!==P||$[0][1]!==f||$[1][0]!==d||$[1][1]!==A)&&(M.selection=[[P,f],[d,A]],l.call(p),Q.brush(T,_.name))}function er(T){if(Qn(T),T.touches){if(T.touches.length)return;u&&clearTimeout(u),u=setTimeout(function(){u=null},500)}else qt(T.view,J),je.on("keydown.brush keyup.brush mousemove.brush mouseup.brush",null);At.attr("pointer-events","all"),Ct.attr("cursor",ct.overlay),M.selection&&($=M.selection),ba($)&&(M.selection=null,l.call(p)),Q.end(T,_.name)}function ui(T){switch(T.keyCode){case 16:{X=N&&I;break}case 18:{_===Tt&&(N&&(h=d-k*N,R=P+k*N),I&&(v=A-S*I,q=f+S*I),_=It,dt(T));break}case 32:{(_===Tt||_===It)&&(N<0?h=d-k:N>0&&(R=P-k),I<0?v=A-S:I>0&&(q=f-S),_=Ue,Ct.attr("cursor",ct.selection),dt(T));break}default:return}ze(T)}function si(T){switch(T.keyCode){case 16:{X&&(G=j=X=!1,dt(T));break}case 18:{_===It&&(N<0?h=d:N>0&&(R=P),I<0?v=A:I>0&&(q=f),_=Tt,dt(T));break}case 32:{_===Ue&&(T.altKey?(N&&(h=d-k*N,R=P+k*N),I&&(v=A-S*I,q=f+S*I),_=It):(N<0?h=d:N>0&&(R=P),I<0?v=A:I>0&&(q=f),_=Tt),Ct.attr("cursor",ct[b]),dt(T));break}default:return}ze(T)}}function w(m){c(this,arguments).moved(m)}function z(m){c(this,arguments).ended(m)}function C(){var m=this.__brush||{selection:null};return m.extent=Ze(e.apply(this,arguments)),m.dim=t,m}return s.extent=function(m){return arguments.length?(e=typeof m=="function"?m:Ie(Ze(m)),s):e},s.filter=function(m){return arguments.length?(r=typeof m=="function"?m:Ie(!!m),s):r},s.touchable=function(m){return arguments.length?(n=typeof m=="function"?m:Ie(!!m),s):n},s.handleSize=function(m){return arguments.length?(o=+m,s):o},s.keyModifiers=function(m){return arguments.length?(i=!!m,s):i},s.on=function(){var m=a.on.apply(a,arguments);return m===a?s:m},s}var ee=t=>()=>t;function Je(t,{sourceEvent:e,target:r,transform:n,dispatch:i}){Object.defineProperties(this,{type:{value:t,enumerable:!0,configurable:!0},sourceEvent:{value:e,enumerable:!0,configurable:!0},target:{value:r,enumerable:!0,configurable:!0},transform:{value:n,enumerable:!0,configurable:!0},_:{value:i}})}function rt(t,e,r){this.k=t,this.x=e,this.y=r}rt.prototype={constructor:rt,scale:function(t){return t===1?this:new rt(this.k*t,this.x,this.y)},translate:function(t,e){return t===0&e===0?this:new rt(this.k,this.x+this.k*t,this.y+this.k*e)},apply:function(t){return[t[0]*this.k+this.x,t[1]*this.k+this.y]},applyX:function(t){return t*this.k+this.x},applyY:function(t){return t*this.k+this.y},invert:function(t){return[(t[0]-this.x)/this.k,(t[1]-this.y)/this.k]},invertX:function(t){return(t-this.x)/this.k},invertY:function(t){return(t-this.y)/this.k},rescaleX:function(t){return t.copy().domain(t.range().map(this.invertX,this).map(t.invert,t))},rescaleY:function(t){return t.copy().domain(t.range().map(this.invertY,this).map(t.invert,t))},toString:function(){return"translate("+this.x+","+this.y+") scale("+this.k+")"}};var zt=new rt(1,0,0);Oe.prototype=rt.prototype;function Oe(t){for(;!t.__zoom;)if(!(t=t.parentNode))return zt;return t.__zoom}function Re(t){t.stopImmediatePropagation()}function Mt(t){t.preventDefault(),t.stopImmediatePropagation()}function Aa(t){return(!t.ctrlKey||t.type==="wheel")&&!t.button}function Ea(){var t=this;return t instanceof SVGElement?(t=t.ownerSVGElement||t,t.hasAttribute("viewBox")?(t=t.viewBox.baseVal,[[t.x,t.y],[t.x+t.width,t.y+t.height]]):[[0,0],[t.width.baseVal.value,t.height.baseVal.value]]):[[0,0],[t.clientWidth,t.clientHeight]]}function oi(){return this.__zoom||zt}function ka(t){return-t.deltaY*(t.deltaMode===1?.05:t.deltaMode?1:.002)*(t.ctrlKey?10:1)}function Na(){return navigator.maxTouchPoints||"ontouchstart"in this}function Sa(t,e,r){var n=t.invertX(e[0][0])-r[0][0],i=t.invertX(e[1][0])-r[1][0],a=t.invertY(e[0][1])-r[0][1],o=t.invertY(e[1][1])-r[1][1];return t.translate(i>n?(n+i)/2:Math.min(0,n)||Math.max(0,i),o>a?(a+o)/2:Math.min(0,a)||Math.max(0,o))}function ai(){var t=Aa,e=Ea,r=Sa,n=ka,i=Na,a=[0,1/0],o=[[-1/0,-1/0],[1/0,1/0]],u=250,s=Le,l=xt("start","zoom","end"),c,x,g,w=500,z=150,C=0,m=10;function p(f){f.property("__zoom",oi).on("wheel.zoom",$,{passive:!1}).on("mousedown.zoom",L).on("dblclick.zoom",R).filter(i).on("touchstart.zoom",P).on("touchmove.zoom",ot).on("touchend.zoom touchcancel.zoom",q).style("-webkit-tap-highlight-color","rgba(0,0,0,0)")}p.transform=function(f,y,h,d){var E=f.selection?f.selection():f;E.property("__zoom",oi),f!==E?I(f,y,h,d):E.interrupt().each(function(){M(this,arguments).event(d).start().zoom(null,typeof y=="function"?y.apply(this,arguments):y).end()})},p.scaleBy=function(f,y,h,d){p.scaleTo(f,function(){var E=this.__zoom.k,v=typeof y=="function"?y.apply(this,arguments):y;return E*v},h,d)},p.scaleTo=function(f,y,h,d){p.transform(f,function(){var E=e.apply(this,arguments),v=this.__zoom,A=h==null?N(E):typeof h=="function"?h.apply(this,arguments):h,k=v.invert(A),S=typeof y=="function"?y.apply(this,arguments):y;return r(_(b(v,S),A,k),E,o)},h,d)},p.translateBy=function(f,y,h,d){p.transform(f,function(){return r(this.__zoom.translate(typeof y=="function"?y.apply(this,arguments):y,typeof h=="function"?h.apply(this,arguments):h),e.apply(this,arguments),o)},null,d)},p.translateTo=function(f,y,h,d,E){p.transform(f,function(){var v=e.apply(this,arguments),A=this.__zoom,k=d==null?N(v):typeof d=="function"?d.apply(this,arguments):d;return r(zt.translate(k[0],k[1]).scale(A.k).translate(typeof y=="function"?-y.apply(this,arguments):-y,typeof h=="function"?-h.apply(this,arguments):-h),v,o)},d,E)};function b(f,y){return y=Math.max(a[0],Math.min(a[1],y)),y===f.k?f:new rt(y,f.x,f.y)}function _(f,y,h){var d=y[0]-h[0]*f.k,E=y[1]-h[1]*f.k;return d===f.x&&E===f.y?f:new rt(f.k,d,E)}function N(f){return[(+f[0][0]+ +f[1][0])/2,(+f[0][1]+ +f[1][1])/2]}function I(f,y,h,d){f.on("start.zoom",function(){M(this,arguments).event(d).start()}).on("interrupt.zoom end.zoom",function(){M(this,arguments).event(d).end()}).tween("zoom",function(){var E=this,v=arguments,A=M(E,v).event(d),k=e.apply(E,v),S=h==null?N(k):typeof h=="function"?h.apply(E,v):h,J=Math.max(k[1][0]-k[0][0],k[1][1]-k[0][1]),X=E.__zoom,G=typeof y=="function"?y.apply(E,v):y,j=s(X.invert(S).concat(J/X.k),G.invert(S).concat(J/G.k));return function(O){if(O===1)O=G;else{var Q=j(O),At=J/Q[2];O=new rt(At,S[0]-Q[0]*At,S[1]-Q[1]*At)}A.zoom(null,O)}})}function M(f,y,h){return!h&&f.__zooming||new B(f,y)}function B(f,y){this.that=f,this.args=y,this.active=0,this.sourceEvent=null,this.extent=e.apply(f,y),this.taps=0}B.prototype={event:function(f){return f&&(this.sourceEvent=f),this},start:function(){return++this.active===1&&(this.that.__zooming=this,this.emit("start")),this},zoom:function(f,y){return this.mouse&&f!=="mouse"&&(this.mouse[1]=y.invert(this.mouse[0])),this.touch0&&f!=="touch"&&(this.touch0[1]=y.invert(this.touch0[0])),this.touch1&&f!=="touch"&&(this.touch1[1]=y.invert(this.touch1[0])),this.that.__zoom=y,this.emit("zoom"),this},end:function(){return--this.active===0&&(delete this.that.__zooming,this.emit("end")),this},emit:function(f){var y=Y(this.that).datum();l.call(f,this.that,new Je(f,{sourceEvent:this.sourceEvent,target:p,type:f,transform:this.that.__zoom,dispatch:l}),y)}};function $(f,...y){if(!t.apply(this,arguments))return;var h=M(this,y).event(f),d=this.__zoom,E=Math.max(a[0],Math.min(a[1],d.k*Math.pow(2,n.apply(this,arguments)))),v=et(f);if(h.wheel)(h.mouse[0][0]!==v[0]||h.mouse[0][1]!==v[1])&&(h.mouse[1]=d.invert(h.mouse[0]=v)),clearTimeout(h.wheel);else{if(d.k===E)return;h.mouse=[v,d.invert(v)],st(this),h.start()}Mt(f),h.wheel=setTimeout(A,z),h.zoom("mouse",r(_(b(d,E),h.mouse[0],h.mouse[1]),h.extent,o));function A(){h.wheel=null,h.end()}}function L(f,...y){if(g||!t.apply(this,arguments))return;var h=f.currentTarget,d=M(this,y,!0).event(f),E=Y(f.view).on("mousemove.zoom",S,!0).on("mouseup.zoom",J,!0),v=et(f,h),A=f.clientX,k=f.clientY;Pt(f.view),Re(f),d.mouse=[v,this.__zoom.invert(v)],st(this),d.start();function S(X){if(Mt(X),!d.moved){var G=X.clientX-A,j=X.clientY-k;d.moved=G*G+j*j>C}d.event(X).zoom("mouse",r(_(d.that.__zoom,d.mouse[0]=et(X,h),d.mouse[1]),d.extent,o))}function J(X){E.on("mousemove.zoom mouseup.zoom",null),qt(X.view,d.moved),Mt(X),d.event(X).end()}}function R(f,...y){if(t.apply(this,arguments)){var h=this.__zoom,d=et(f.changedTouches?f.changedTouches[0]:f,this),E=h.invert(d),v=h.k*(f.shiftKey?.5:2),A=r(_(b(h,v),d,E),e.apply(this,y),o);Mt(f),u>0?Y(this).transition().duration(u).call(I,A,d,f):Y(this).call(p.transform,A,d,f)}}function P(f,...y){if(t.apply(this,arguments)){var h=f.touches,d=h.length,E=M(this,y,f.changedTouches.length===d).event(f),v,A,k,S;for(Re(f),A=0;A<d;++A)k=h[A],S=et(k,this),S=[S,this.__zoom.invert(S),k.identifier],E.touch0?!E.touch1&&E.touch0[2]!==S[2]&&(E.touch1=S,E.taps=0):(E.touch0=S,v=!0,E.taps=1+!!c);c&&(c=clearTimeout(c)),v&&(E.taps<2&&(x=S[0],c=setTimeout(function(){c=null},w)),st(this),E.start())}}function ot(f,...y){if(this.__zooming){var h=M(this,y).event(f),d=f.changedTouches,E=d.length,v,A,k,S;for(Mt(f),v=0;v<E;++v)A=d[v],k=et(A,this),h.touch0&&h.touch0[2]===A.identifier?h.touch0[0]=k:h.touch1&&h.touch1[2]===A.identifier&&(h.touch1[0]=k);if(A=h.that.__zoom,h.touch1){var J=h.touch0[0],X=h.touch0[1],G=h.touch1[0],j=h.touch1[1],O=(O=G[0]-J[0])*O+(O=G[1]-J[1])*O,Q=(Q=j[0]-X[0])*Q+(Q=j[1]-X[1])*Q;A=b(A,Math.sqrt(O/Q)),k=[(J[0]+G[0])/2,(J[1]+G[1])/2],S=[(X[0]+j[0])/2,(X[1]+j[1])/2]}else if(h.touch0)k=h.touch0[0],S=h.touch0[1];else return;h.zoom("touch",r(_(A,k,S),h.extent,o))}}function q(f,...y){if(this.__zooming){var h=M(this,y).event(f),d=f.changedTouches,E=d.length,v,A;for(Re(f),g&&clearTimeout(g),g=setTimeout(function(){g=null},w),v=0;v<E;++v)A=d[v],h.touch0&&h.touch0[2]===A.identifier?delete h.touch0:h.touch1&&h.touch1[2]===A.identifier&&delete h.touch1;if(h.touch1&&!h.touch0&&(h.touch0=h.touch1,delete h.touch1),h.touch0)h.touch0[1]=this.__zoom.invert(h.touch0[0]);else if(h.end(),h.taps===2&&(A=et(A,this),Math.hypot(x[0]-A[0],x[1]-A[1])<m)){var k=Y(this).on("dblclick.zoom");k&&k.apply(this,arguments)}}}return p.wheelDelta=function(f){return arguments.length?(n=typeof f=="function"?f:ee(+f),p):n},p.filter=function(f){return arguments.length?(t=typeof f=="function"?f:ee(!!f),p):t},p.touchable=function(f){return arguments.length?(i=typeof f=="function"?f:ee(!!f),p):i},p.extent=function(f){return arguments.length?(e=typeof f=="function"?f:ee([[+f[0][0],+f[0][1]],[+f[1][0],+f[1][1]]]),p):e},p.scaleExtent=function(f){return arguments.length?(a[0]=+f[0],a[1]=+f[1],p):[a[0],a[1]]},p.translateExtent=function(f){return arguments.length?(o[0][0]=+f[0][0],o[1][0]=+f[1][0],o[0][1]=+f[0][1],o[1][1]=+f[1][1],p):[[o[0][0],o[0][1]],[o[1][0],o[1][1]]]},p.constrain=function(f){return arguments.length?(r=f,p):r},p.duration=function(f){return arguments.length?(u=+f,p):u},p.interpolate=function(f){return arguments.length?(s=f,p):s},p.on=function(){var f=l.on.apply(l,arguments);return f===l?p:f},p.clickDistance=function(f){return arguments.length?(C=(f=+f)*f,p):Math.sqrt(C)},p.tapDistance=function(f){return arguments.length?(m=+f,p):m},p}var ZoomTransform=rt,brush=ii,brushSelection=ei,brushX=ri,brushY=ni,pointer=et,select=Y,selectAll=Gr,selection=at,zoom=ai,zoomIdentity=zt,zoomTransform=Oe;;


// ── anywidget render entry point ──────────────────────────────────────────
// This file is read by _interactive.py which replaces __B64__ with the
// base64-encoded ferrum_wasm_bg.wasm blob before sending to the browser.
// Keep all JS here — never embed JS strings in Python.
//
// Adapter pattern: _render() accepts an adapter object (not a raw model).
// Two adapters exist:
//   1. Jupyter adapter — constructed in render() for anywidget
//   2. Standalone adapter — exported as createStandaloneAdapter() for HTML exports



// D3 interactions (brush, zoom, select, zoomTransform, pointer) are provided
// by d3-interactions.js which is inlined before this file in both standalone
// HTML and Jupyter ESM builds.  The D3 bundle's `export { ... }` is stripped
// by the assembler, leaving the symbols in module scope.

// ── SVG text placement ───────────────────────────────────────────────────
function _placeTextSvg(svgEl, texts) {
  const svg = select(svgEl);
  svg.selectAll('text.ferrum-label').remove();
  for (const t of texts) {
    const anchor = t.anchor === 'center' ? 'middle' : t.anchor;
    let baseline;
    switch (t.baseline) {
      case 'top': baseline = 'hanging'; break;
      case 'middle': baseline = 'central'; break;
      case 'bottom': baseline = 'text-after-edge'; break;
      case 'alphabetic': default: baseline = 'auto'; break;
    }
    const el = svg.append('text')
      .attr('class', 'ferrum-label')
      .attr('x', t.x)
      .attr('y', t.y)
      .attr('text-anchor', anchor)
      .attr('dominant-baseline', baseline)
      .attr('font-size', t.fontSize + 'px')
      .attr('font-weight', t.fontWeight)
      .attr('font-family', t.fontFamily)
      .attr('fill', t.color)
      .attr('pointer-events', 'none')
      .text(t.content);
    if (t.angle) {
      el.attr('transform', `rotate(${t.angle}, ${t.x}, ${t.y})`);
    }
  }
}

// Hit-test pixel (x, y) against the mark batches.
// marks is an array of {batch, panel} pairs so arc paths can use panel.plot_area.
function _hitTest(marks, x, y) {
  for (let bi = marks.length - 1; bi >= 0; bi--) {
    const { batch: b, panel } = marks[bi];
    if (!b.nodes) continue;
    for (let ni = b.nodes.length - 1; ni >= 0; ni--) {
      const n = b.nodes[ni];
      let hit = false;
      if (n.type === 'circle') {
        const dx = x - n.cx, dy = y - n.cy;
        hit = dx * dx + dy * dy <= n.r * n.r;
      } else if (n.type === 'rect') {
        hit = x >= n.x && x <= n.x + n.w && y >= n.y && y <= n.y + n.h;
      } else if (n.type === 'path' && b.kind === 'arc') {
        // Pie / donut wedge hit test from plot_area center + path commands.
        const pa = panel.plot_area;
        const cx = pa.x + pa.w / 2, cy = pa.y + pa.h / 2;
        const dx = x - cx, dy = y - cy;
        const dist = Math.sqrt(dx * dx + dy * dy);
        const arcCmd = n.commands && n.commands.find(c => c.op === 'arc_to');
        const outerR = arcCmd ? arcCmd.rx : 0;
        if (dist <= outerR) {
          const lineTo = n.commands && n.commands.find(c => c.op === 'line_to');
          const innerR = lineTo
            ? Math.sqrt((lineTo.x - cx) ** 2 + (lineTo.y - cy) ** 2)
            : 0;
          if (dist >= innerR) {
            const moveTo = n.commands && n.commands.find(c => c.op === 'move_to');
            if (moveTo) {
              const norm = a => ((a % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI);
              const pointAngle = Math.atan2(dx, -dy);
              const startAngle = Math.atan2(moveTo.x - cx, -(moveTo.y - cy));
              const endAngle = arcCmd
                ? Math.atan2(arcCmd.x - cx, -(arcCmd.y - cy))
                : startAngle;
              const sa = norm(startAngle);
              let ea = norm(endAngle);
              if (ea <= sa) ea += 2 * Math.PI;
              const pa2 = norm(pointAngle);
              const pa3 = pa2 < sa ? pa2 + 2 * Math.PI : pa2;
              hit = pa3 >= sa && pa3 <= ea;
            } else {
              hit = true; // no move_to — treat as full circle
            }
          }
        }
      }
      if (hit) return { batch: b, idx: ni };
    }
  }
  return null;
}

// ── Adapter interface (duck-typed) ───────────────────────────────────────
// {
//   getPackedData()           → Uint8Array
//   getInteractionConfig()    → string (JSON)
//   onSelectionChange(state)  → void (called when selection changes)
//   onZoomChange(state)       → void (called when zoom changes)
// }

async function _render(container, sceneJson, adapter) {
  container.replaceChildren();
  container.style.position = 'relative';

  const scene = JSON.parse(sceneJson);
  const w = scene.width || 640, h = scene.height || 480;

  // ── Canvas ───────────────────────────────────────────────────────
  const canvas = document.createElement('canvas');
  canvas.width = w; canvas.height = h; canvas.style.display = 'block';
  container.appendChild(canvas);

  // ── SVG overlay for text labels ──────────────────────────────────
  const svgEl = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svgEl.setAttribute('width', w);
  svgEl.setAttribute('height', h);
  // SVG inherits CSS @font-face from the parent HTML document (Inter).
  svgEl.style.cssText = 'position:absolute;top:0;left:0;pointer-events:none;';
  container.appendChild(svgEl);

  // ── Tooltip ──────────────────────────────────────────────────────
  const tip = document.createElement('div');
  tip.className = 'ferrum-tooltip';
  Object.assign(tip.style, { position: 'absolute', pointerEvents: 'none',
    opacity: '0', transition: 'opacity 0.1s ease' });
  container.appendChild(tip);

  // marks carries {batch, panel} pairs so hit-testers have panel context.
  const marks = scene.panels
    ? scene.panels.flatMap(p => (p.marks || []).map(b => ({ batch: b, panel: p })))
    : [];

  // ── Brush / interval selection detection ──────────────────────────
  const cfg = JSON.parse(adapter.getInteractionConfig());
  const _hasPointSelections = (cfg.selections || []).some(s => s.type === 'point');
  const hasInterval = (cfg.selections || []).some(s => s.type === 'interval');

  // ── GPU init (may fail when WebGPU/WebGL context limit exceeded) ──
  // Event listeners below still work without GPU — tooltips + click state.
  let renderer = null;
  try {
    // WASM already initialized
    renderer = await WasmRenderer.create(canvas);
    const packedArr = adapter.getPackedData();
    const textJson = renderer.loadScene(sceneJson, packedArr);
    _placeTextSvg(svgEl, JSON.parse(textJson));
  } catch (e) {
    console.warn('[ferrum] GPU init failed — rendering disabled, tooltips still active.', e);
  }

  // ── D3-zoom on canvas ─────────────────────────────────────────────
  let _zoomDebounceId = null;
  const zoomBehavior = zoom()
    .scaleExtent([0.1, 50])
    .filter(event => {
      // Always allow wheel-zoom.
      if (event.type === 'wheel') return true;
      // When interval selections are active, require Alt/Option or Cmd/Meta
      // for pan (drag without modifier belongs to the brush).
      if (hasInterval && !event.altKey && !event.metaKey) return false;
      // Only left-button drags.
      return !event.button;
    })
    .on('zoom', event => {
      if (!renderer) return;
      const { k, x, y } = event.transform;
      try {
        const textJson = renderer.setTransform(k, x, y);
        _placeTextSvg(svgEl, JSON.parse(textJson));
      } catch (err) { /* GPU not ready */ }
      // Debounced adapter callback for Jupyter zoom rebuild.
      clearTimeout(_zoomDebounceId);
      _zoomDebounceId = setTimeout(() => {
        adapter.onZoomChange({ '0': { k, x, y } });
      }, 400);
    });

  // Attach zoom to the container (wraps both canvas and SVG) so wheel/pan
  // events work regardless of which layer captures them.
  select(container).call(zoomBehavior);

  // Double-click: reset zoom to identity.
  select(container).on('dblclick.zoom', () => {
    if (!renderer) return;
    select(container).call(zoomBehavior.transform, zoomIdentity);
  });

  // ── D3-brush on SVG (per-panel overlays for interval selections) ────
  if (hasInterval && scene.panels) {
    // Extract brush styling from the interval selection's SelectionMark.
    let brushFill = 'rgba(51, 136, 204, 0.2)';
    let brushStroke = 'rgba(51, 136, 204, 0.6)';
    const intervalSel = (cfg.selections || []).find(s => s.type === 'interval');
    if (intervalSel && intervalSel.mark) {
      if (intervalSel.mark.fill) brushFill = intervalSel.mark.fill;
      if (intervalSel.mark.stroke) brushStroke = intervalSel.mark.stroke;
    }

    // Enable pointer events on the SVG so brushes can capture gestures.
    svgEl.style.pointerEvents = 'all';

    for (let pi = 0; pi < scene.panels.length; pi++) {
      const pa = scene.panels[pi].plot_area;
      if (!pa) continue;

      const brushBehavior = brush()
        .extent([[pa.x, pa.y], [pa.x + pa.w, pa.y + pa.h]])
        .filter(event => !event.altKey && !event.metaKey && event.button === 0);

      // Capture panel index for the closure.
      const panelIdx = pi;
      brushBehavior.on('end', function(event) {
        if (!renderer) return;
        if (!event.selection) return;
        const [[x0, y0], [x1, y1]] = event.selection;
        try {
          const resultJson = renderer.handleDrag(panelIdx, x0, y0, x1, y1);
          adapter.onSelectionChange(JSON.parse(resultJson));
          // Re-render text with current zoom preserved.
          const t = zoomTransform(container);
          const textJson = renderer.setTransform(t.k, t.x, t.y);
          _placeTextSvg(svgEl, JSON.parse(textJson));
        } catch (err) {
          console.warn('[ferrum] handleDrag error:', err);
        }
      });

      const brushG = select(svgEl).append('g')
        .attr('class', 'ferrum-brush')
        .attr('data-panel', panelIdx)
        .call(brushBehavior);

      // Style the brush rectangle.
      brushG.selectAll('.selection')
        .style('fill', brushFill)
        .style('stroke', brushStroke);
    }
  }

  // ── Tooltip mousemove ─────────────────────────────────────────────
  canvas.addEventListener('mousemove', e => {
    const r = canvas.getBoundingClientRect();
    const mx = (e.clientX - r.left) * (canvas.width / r.width);
    const my = (e.clientY - r.top) * (canvas.height / r.height);

    // Inverse-zoom for hit-test in original mark space.
    const t = zoomTransform(container);
    const hx = t.k !== 0 ? (mx - t.x) / t.k : mx;
    const hy = t.k !== 0 ? (my - t.y) / t.k : my;

    let tooltipData = null;
    // Try JS hit-test first (non-packed batches with nodes).
    const hh = _hitTest(marks, hx, hy);
    if (hh && hh.batch.tooltips && hh.batch.tooltips[hh.idx]) {
      tooltipData = hh.batch.tooltips[hh.idx];
    }
    // Fallback: WASM hit-test + getTooltip for packed batches (empty nodes).
    if (!tooltipData && renderer) {
      try {
        const hitJson = renderer.hitTestAt(mx, my);
        const hit = JSON.parse(hitJson);
        if (hit.panel != null && hit.batch != null && hit.idx != null) {
          const tJson = renderer.getTooltip(hit.panel, hit.batch, hit.idx);
          const parsed = JSON.parse(tJson);
          if (parsed.fields && parsed.fields.length > 0) tooltipData = parsed;
        }
      } catch (err) { /* WASM not ready or no tooltip data */ }
    }
    if (tooltipData) {
      tip.replaceChildren();
      const tbl = document.createElement('table');
      for (const f of tooltipData.fields) {
        const tr = document.createElement('tr');
        const k = document.createElement('td');
        k.textContent = f.name; k.style.fontWeight = 'bold'; k.style.paddingRight = '6px';
        const v = document.createElement('td'); v.textContent = f.value;
        tr.appendChild(k); tr.appendChild(v); tbl.appendChild(tr);
      }
      tip.appendChild(tbl);
      // Position tooltip in CSS coords.
      const cssMx = mx / (canvas.width / r.width);
      const csMy = my / (canvas.height / r.height);
      tip.style.left = (cssMx + 12) + 'px';
      tip.style.top = (csMy - 12) + 'px';
      tip.style.opacity = '1';
    } else {
      tip.style.opacity = '0';
    }
  });

  canvas.addEventListener('mouseleave', () => {
    tip.style.opacity = '0';
  });

  // ── Click: href navigation + point selection ──────────────────────
  canvas.addEventListener('click', e => {
    const r = canvas.getBoundingClientRect();
    const cx = (e.clientX - r.left) * (canvas.width / r.width);
    const cy = (e.clientY - r.top) * (canvas.height / r.height);

    // Inverse-zoom for JS hit-test.
    const t = zoomTransform(container);
    const hx = t.k !== 0 ? (cx - t.x) / t.k : cx;
    const hy = t.k !== 0 ? (cy - t.y) / t.k : cy;

    // Href navigation.
    const h = _hitTest(marks, hx, hy);
    if (h && h.batch.hrefs && h.batch.hrefs[h.idx]) {
      window.open(h.batch.hrefs[h.idx], '_blank', 'noopener,noreferrer');
      return;
    }

    // Delegate clicks to WASM handleClick only when point selections exist.
    // Interval selections only respond to drags (handleDrag), not clicks.
    if (renderer && _hasPointSelections) {
      try {
        const stateJson = renderer.handleClick(cx, cy, e.shiftKey);
        const state = JSON.parse(stateJson);
        adapter.onSelectionChange(state);
      } catch (err) {
        console.warn('[ferrum] handleClick error:', err);
      }
      return;
    }

    // Fallback (no GPU): use JS hit test + tooltip field extraction.
    if (!h) return;
    const icfg = adapter.getInteractionConfig();
    let selConfig = {};
    try { selConfig = JSON.parse(icfg || '{}'); } catch (_e) { /* ignore */ }
    const selections = selConfig.selections || [];
    const tooltip = h.batch.tooltips && h.batch.tooltips[h.idx];
    const fieldMap = {};
    if (tooltip) { for (const f of tooltip.fields) fieldMap[f.name] = f.value; }
    const selState = {};
    for (const sel of selections) {
      if (!sel.fields) continue;
      const vals = {};
      for (const field of sel.fields) {
        if (fieldMap[field] !== undefined) vals[field] = fieldMap[field];
      }
      if (Object.keys(vals).length > 0) selState[sel.name] = vals;
    }
    if (Object.keys(selState).length > 0) {
      adapter.onSelectionChange(selState);
    }
  });

  // ── ResizeObserver ────────────────────────────────────────────────
  if (renderer) {
    const ro = new ResizeObserver(() => {
      try { renderer.resize(canvas.width, canvas.height); } catch (err) { /* ignore */ }
    });
    ro.observe(canvas);
  }

  return { canvas, renderer, scene, svgEl };
}

// ── Standalone adapter factory (for HTML exports) ────────────────────────
function createStandaloneAdapter(packedB64, interactionConfig) {
  let packedArr;
  if (packedB64) {
    const raw = atob(packedB64);
    packedArr = new Uint8Array(raw.length);
    for (let i = 0; i < raw.length; i++) packedArr[i] = raw.charCodeAt(i);
  } else {
    packedArr = new Uint8Array(0);
  }
  return {
    getPackedData() { return packedArr; },
    getInteractionConfig() { return interactionConfig || '{}'; },
    onSelectionChange(_state) { /* local-only, no Python round-trip */ },
    onZoomChange(_state) { /* local-only, no Python round-trip */ },
  };
}


export { _render, createStandaloneAdapter };
