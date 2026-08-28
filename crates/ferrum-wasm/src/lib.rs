#![deny(clippy::unwrap_used)]

// Pure-logic modules: compile on all targets (host tests use these).
pub mod conditional;
pub mod error;
pub mod hit_test;
// MSAA sample-count selection (FA-19). Pure logic consumed by the wasm32 GPU
// context; gated to wasm32+test so the host build does not flag it as dead.
#[cfg(any(target_arch = "wasm32", test))]
pub mod msaa;
// Reactive-parameter runtime (D6): consumed by the wasm32 `WasmRenderer`; its
// pure pixel↔data helpers are also unit-tested on the host. Gated to those two
// targets so the non-test host build does not flag the wasm-only helpers as
// dead code.
#[cfg(any(target_arch = "wasm32", test))]
pub mod param_runtime;
pub mod scene_load;
pub mod selection_state;
pub mod spatial_index;
pub mod tessellate;
pub mod text_json;
pub mod transition;
pub mod zoom_pan;

// GPU / wasm-bindgen modules: only compile when targeting wasm32.
#[cfg(target_arch = "wasm32")]
pub mod gpu;
#[cfg(target_arch = "wasm32")]
pub mod pipelines;
#[cfg(target_arch = "wasm32")]
pub mod render;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;
#[cfg(target_arch = "wasm32")]
use web_sys::HtmlCanvasElement;

#[cfg(target_arch = "wasm32")]
use crate::error::WasmRenderError;
#[cfg(target_arch = "wasm32")]
use crate::gpu::GpuContext;
#[cfg(target_arch = "wasm32")]
use crate::pipelines::RenderPipelines;
#[cfg(target_arch = "wasm32")]
use crate::render::GpuBuffers;
#[cfg(target_arch = "wasm32")]
use crate::scene_load::SceneData;
#[cfg(target_arch = "wasm32")]
use crate::selection_state::InteractionState;
#[cfg(target_arch = "wasm32")]
use crate::spatial_index::SpatialIndex;
#[cfg(target_arch = "wasm32")]
use crate::transition::{
    ease_in_out_cubic, plan_transition, Instances, SceneSnapshot, TransitionPlan, TransitionSide,
};
#[cfg(target_arch = "wasm32")]
use crate::zoom_pan::ZoomPanState;

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub struct WasmRenderer {
    gpu: GpuContext,
    pipelines: RenderPipelines,
    loaded: Option<LoadedScene>,
    transition: Option<ActiveTransition>,
    selections: Vec<ferrum_scene::SelectionSpec>,
    interaction_state: InteractionState,
    zoom: ZoomPanState,
    /// Per-(panel, y-slot) domain-rescale affines (secondary-y-axis, GH #52),
    /// flat-indexed by `scene_load::transform_slot_index`. A `domainParam`/brush
    /// bound to an independent-y layer's `y` writes a y-only rescale into that
    /// layer's slot here; the render upload composes it with the per-panel
    /// zoom/pan affine (`compose_panel_slot`). Identity for every slot at rest
    /// and for every single-y scene, so the composed transform is exactly the
    /// panel affine — byte-stable. Kept separate from `zoom.transforms` so a
    /// per-layer domain rescale never disturbs the shared zoom/pan gesture.
    slot_rescales: Vec<crate::zoom_pan::Affine2>,
    interaction: ferrum_scene::InteractionConfig,
    spatial_index: Option<SpatialIndex>,
    /// The frame this renderer was showing before the current `loadScene`
    /// (GH #93 old-side identity, spec §4.3).
    ///
    /// Captured in `load_scene` before the incoming scene replaces the loaded
    /// one, because a packed batch's instances and its `HAS_KEYS` sidecar
    /// exist ONLY in this in-memory form — the previous scene's JSON carries
    /// neither. `start_transition` consumes it; when there is none (first
    /// load) it falls back to parsing the JSON it was handed.
    ///
    /// SYNC (JS): this is only reachable if the caller keeps ONE renderer
    /// alive across data updates. `ferrum-anywidget.js`'s `_reload` does —
    /// its `update` path calls `loadScene` on the live renderer and only
    /// rebuilds when it cannot (size/config change, or failure). A caller that
    /// constructs a fresh renderer per update always takes the JSON fallback.
    previous: Option<SceneSnapshot>,
}

#[cfg(target_arch = "wasm32")]
struct ActiveTransition {
    /// The scene being transitioned away from, in the form the pairing was
    /// resolved against.
    old: SceneSnapshot,
    new_data: SceneData,
    new_scene: ferrum_scene::SceneGraph,
    /// Old-to-new instance pairing, resolved once when the transition starts
    /// (GH #93). Holding the plan rather than the old `SceneGraph` is what
    /// "retaining the old keys" amounts to: the keys are consumed into
    /// `key → instance index` maps here and never read again.
    plan: TransitionPlan,
}

#[cfg(target_arch = "wasm32")]
struct LoadedScene {
    data: SceneData,
    buffers: GpuBuffers,
    scene: ferrum_scene::SceneGraph,
    /// Interaction geometry: `scene.panels` with every panel's `layout_scale`
    /// baked into `plot_area`/`clip`/mark-node coordinates
    /// (`scene_load::bake_panels`) — the single source of truth every
    /// interaction consumer (hit-test, spatial index, brush/crossfilter
    /// reprojection, the per-panel scissor upload) reads instead of
    /// `scene.panels` directly (D4a amendment addendum). Identity for every
    /// panel today; only a ratio-fitted composite panel actually differs from
    /// `scene.panels`.
    baked_panels: Vec<ferrum_scene::Panel>,
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl WasmRenderer {
    #[wasm_bindgen(js_name = "create")]
    pub async fn create(canvas: HtmlCanvasElement) -> Result<WasmRenderer, JsValue> {
        console_error_panic_hook::set_once();
        let gpu = gpu::init_gpu(canvas).await.map_err(JsValue::from)?;
        let pipelines = RenderPipelines::new(&gpu.device, gpu.format, gpu.sample_count);
        Ok(WasmRenderer {
            gpu,
            pipelines,
            loaded: None,
            transition: None,
            selections: Vec::new(),
            interaction_state: InteractionState::new(&[]),
            zoom: ZoomPanState::new(0, &ferrum_scene::InteractionConfig::default()),
            slot_rescales: Vec::new(),
            interaction: ferrum_scene::InteractionConfig::default(),
            spatial_index: None,
            previous: None,
        })
    }

    #[wasm_bindgen(js_name = "loadScene")]
    pub fn load_scene(&mut self, scene_json: &str, packed_data: &[u8]) -> Result<String, JsValue> {
        let scene: ferrum_scene::SceneGraph = serde_json::from_str(scene_json)
            .map_err(|e| JsValue::from(WasmRenderError::SceneDeserialization(e.to_string())))?;

        let data = scene_load::load_scene_with_packed(&scene, packed_data);
        let text_json = text_json::build_overlay_json(&data);
        let buffers = GpuBuffers::from_scene(&self.gpu, &self.pipelines, &data);
        let clear_color = data.background;
        // Interaction-geometry single source of truth (D4a amendment
        // addendum): bake every panel's `layout_scale` into its geometry
        // once here, so hit-testing, the spatial index, and brush/crossfilter
        // reprojection all read the same final on-screen positions the GPU
        // frame renders — never `scene.panels` directly.
        let baked_panels = scene_load::bake_panels(&scene);

        self.selections = scene.selections.clone();
        self.interaction_state = InteractionState::new(&self.selections);
        self.interaction = scene.interaction.clone();
        self.zoom = ZoomPanState::new(scene.panels.len(), &self.interaction);
        // Per-(panel, y-slot) domain-rescale affines (secondary-y-axis, #52),
        // one per allocated transform slot, all identity until a per-layer
        // domainParam/brush rescales one. For a single-y scene this is one
        // identity per panel — the byte-stable default.
        self.slot_rescales = vec![
            crate::zoom_pan::Affine2::identity();
            scene_load::total_transform_slots(&data.panel_slot_counts)
        ];
        // Build spatial index over all panels for O(log n) hit-testing.
        // Pass scene data so packed instances (>= 1000 marks) are also indexed.
        self.spatial_index = Some(SpatialIndex::build_with_packed(&baked_panels, Some(&data)));
        // Snapshot the OUTGOING frame before it is replaced (GH #93 old-side
        // identity). This is the only moment a packed batch's instances and
        // sidecar keys are still addressable: the scene JSON a later
        // `startTransition` receives has neither. Its GPU buffers drop here,
        // as they did before the snapshot existed.
        self.previous = self
            .loaded
            .take()
            .map(|prev| SceneSnapshot::from_scene_data(prev.data, prev.scene.panels));
        self.loaded = Some(LoadedScene {
            data,
            buffers,
            scene,
            baked_panels,
        });

        if let Some(ref loaded) = self.loaded {
            render::render_frame(&self.gpu, &self.pipelines, &loaded.buffers, clear_color)
                .map_err(JsValue::from)?;
        }

        Ok(text_json)
    }

    #[wasm_bindgen(js_name = "renderFrame")]
    pub fn render_frame_js(&self) -> Result<(), JsValue> {
        if let Some(ref loaded) = self.loaded {
            render::render_frame(
                &self.gpu,
                &self.pipelines,
                &loaded.buffers,
                loaded.data.background,
            )
            .map_err(JsValue::from)?;
        }
        Ok(())
    }

    /// Begin a GPU-interpolated transition from an old scene to the currently
    /// loaded scene.
    ///
    /// The transition target is `self.loaded.data` (the scene already loaded
    /// via `loadScene`). The transition SOURCE is the snapshot `loadScene`
    /// took of the outgoing frame (GH #93 old-side identity, spec §4.3);
    /// `old_scene_json` is the fallback source, used only when this renderer
    /// has no predecessor to snapshot — a first load. The parameter is kept
    /// (rather than removed) so the JS caller's contract is unchanged and the
    /// fallback stays available.
    ///
    /// B4 fix: the old API accepted the *new* scene JSON and cloned `loaded.data`
    /// as old. But `loadScene(new_json)` was already called before
    /// `startTransition`, so `loaded.data` was already the new scene — making
    /// old == new and the transition a no-op (self-to-self interpolation).
    /// Now the caller passes the *old* scene JSON and we use `loaded.data` as
    /// the transition target.
    ///
    /// Call ``tick_transition(t)`` (t in [0, 1]) from a requestAnimationFrame loop
    /// to drive the animation.  ``start_transition`` does not start the loop —
    /// the JavaScript caller owns the timing.
    ///
    /// Returns `Ok(())` immediately (no-op) if no scene is currently loaded.
    #[wasm_bindgen(js_name = "startTransition")]
    pub fn start_transition(&mut self, old_scene_json: &str) -> Result<(), JsValue> {
        // Resolve the old side first: `previous` is consumed here, so the
        // snapshot is never reused across two transitions, and taking it now
        // leaves no borrow outstanding when `loaded` is read below.
        let previous = self.previous.take();
        let Some(loaded) = &self.loaded else {
            return Ok(());
        };
        // B4+B5 fix: the new scene (transition target) is already in loaded.data,
        // which was populated by loadScene with full packed data.
        let new_data = loaded.data.clone();
        let old = match previous {
            Some(snapshot) => snapshot,
            None => transition::SceneSnapshot::from_scene_json(old_scene_json)?,
        };
        // Resolve the old→new instance pairing once (GH #93). Keyed batches
        // pair by `encode(key=...)` so each mark transitions to its own new
        // position; scenes without keys resolve to `TransitionPlan::IndexZip`
        // and interpolate exactly as they did before keys existed.
        let plan = plan_transition(
            old.side(),
            TransitionSide {
                panels: &loaded.scene.panels,
                packed_batch_meta: &new_data.packed_batch_meta,
                draw_commands: &new_data.draw_commands,
            },
        );
        self.transition = Some(ActiveTransition {
            old,
            new_data,
            new_scene: loaded.scene.clone(),
            plan,
        });
        Ok(())
    }

    /// Advance the transition to fractional progress ``t`` ∈ [0, 1].
    ///
    /// Applies eased interpolation and re-renders the GPU frame.
    /// When ``t >= 1.0`` the transition state is cleared and the new scene
    /// is committed as the loaded scene.
    #[wasm_bindgen(js_name = "tickTransition")]
    pub fn tick_transition(&mut self, t: f32) -> Result<(), JsValue> {
        let t_eased = ease_in_out_cubic(t.clamp(0.0, 1.0));
        if let Some(ref tr) = self.transition {
            let frame = transition::interpolate(
                &tr.plan,
                tr.old.instances(),
                Instances {
                    circles: &tr.new_data.circle_instances,
                    rects: &tr.new_data.rect_instances,
                },
                t_eased,
            );
            // Exiting marks live past the new scene's own instance ranges, so
            // every draw command the new scene emitted still addresses its own
            // slots; their fade-out draws append after it. Empty for an
            // unkeyed transition, leaving the command list untouched.
            let mut draw_commands = tr.new_data.draw_commands.clone();
            draw_commands.extend(frame.exit_draws);
            let lerped_data = SceneData {
                circle_instances: frame.circles,
                rect_instances: frame.rects,
                mesh_buffers: tr.new_data.mesh_buffers.clone(),
                static_mesh_buffers: tr.new_data.static_mesh_buffers.clone(),
                annotation_mesh_buffers: tr.new_data.annotation_mesh_buffers.clone(),
                text_elements: tr.new_data.text_elements.clone(),
                image_quads: tr.new_data.image_quads.clone(),
                raw_fragments: tr.new_data.raw_fragments.clone(),
                background: tr.new_data.background,
                width: tr.new_data.width,
                height: tr.new_data.height,
                packed_batch_meta: tr.new_data.packed_batch_meta.clone(),
                draw_commands,
                mark_mesh_panels: tr.new_data.mark_mesh_panels.clone(),
                panel_count: tr.new_data.panel_count,
                panel_slot_counts: tr.new_data.panel_slot_counts.clone(),
            };
            let buffers = GpuBuffers::from_scene(&self.gpu, &self.pipelines, &lerped_data);
            render::render_frame(&self.gpu, &self.pipelines, &buffers, lerped_data.background)
                .map_err(JsValue::from)?;
            if t >= 1.0 {
                let final_data = tr.new_data.clone();
                let final_buffers = GpuBuffers::from_scene(&self.gpu, &self.pipelines, &final_data);
                let final_scene = tr.new_scene.clone();
                let final_baked_panels = scene_load::bake_panels(&final_scene);
                self.transition = None;
                self.loaded = Some(LoadedScene {
                    data: final_data,
                    buffers: final_buffers,
                    scene: final_scene,
                    baked_panels: final_baked_panels,
                });
            }
        }
        Ok(())
    }

    // ── Selection-mutating methods: TWO return-shape contracts ───────────────
    //
    // WASM-07 (tracked carry): the four selection-mutating wasm_bindgen methods
    // all update selection state, run `apply_conditionals_and_render`, and
    // return JSON for the same JS `onSelectionChange` consumer — but they return
    // TWO DIFFERENT JSON SHAPES, and the JS loader unwraps each accordingly:
    //
    //   * `handleClick`, `clearSelections`, `toggleLegend`
    //         → the BARE selection-state map: `{selection_name: {field: value}}`.
    //         JS does `onSelectionChange(JSON.parse(stateJson))` directly.
    //
    //   * `handleDrag`
    //         → an ENVELOPE: `{ "selection": <state-map>, "rescaled": <panel|null>,
    //           "rescaled_text": <label-json|null> }`. JS does
    //           `onSelectionChange(result.selection)` and separately consumes
    //           `rescaled`/`rescaled_text` for the D6 Domain-rescale path.
    //
    // The envelope exists ONLY on `handleDrag` because a Domain rescale can today
    // only be signaled from a drag. Unifying all four onto the single envelope
    // shape is desirable (it would let a future click-driven rescale be
    // expressed without another shape change) but is deferred: it is
    // behavior-preserving ONLY if the JS `ferrum-anywidget.js` consumer is
    // updated in lock-step AND validated in a headless browser, which this
    // campaign cannot perform. Tracked as the full-envelope-unification carry —
    // do NOT change any return shape here without the JS + browser lockstep.

    /// Hit-test a click at canvas pixel (x, y), update selection state, apply
    /// conditional encodings (dim non-selected marks), re-render frame, and
    /// return the new selection state as a JSON string.
    ///
    /// Returns the BARE selection-state map `{selection_name: {field: value}}`
    /// (see the two-shape contract note above the selection-mutating methods).
    /// The JS caller should forward this to `model.set('selection_state', ...)`.
    #[wasm_bindgen(js_name = "handleClick")]
    pub fn handle_click(&mut self, x: f32, y: f32, shift_held: bool) -> Result<String, JsValue> {
        let Some(loaded) = self.loaded.as_mut() else {
            return Ok("{}".to_string());
        };

        // Update selection state via Rust hit-test (authoritative — operates on actual scene data).
        // Pass spatial index for O(log n) circle/rect hit-testing. Baked
        // geometry (D4a amendment addendum) — see `LoadedScene::baked_panels`.
        self.interaction_state.handle_click_with_index(
            &loaded.baked_panels,
            &self.selections,
            x as f64,
            y as f64,
            &self.zoom,
            shift_held,
            self.spatial_index.as_ref(),
            &self.slot_rescales,
            &loaded.data.panel_slot_counts,
        );

        self.apply_conditionals_and_render()
    }

    /// Handle a brush-drag on a panel: update interval selection state, apply
    /// conditional encodings, rebuild GPU buffers, re-render, and return
    /// the new selection state as JSON.
    ///
    /// Returns the ENVELOPE shape `{selection, rescaled, rescaled_text}` (the
    /// only selection-mutating method that does — see the two-shape contract
    /// note above the selection-mutating methods). `rescaled`/`rescaled_text`
    /// are non-null only when a D6 Domain binding rescaled a target panel.
    #[wasm_bindgen(js_name = "handleDrag")]
    pub fn handle_drag(
        &mut self,
        panel_id: u32,
        x0: f32,
        y0: f32,
        x1: f32,
        y1: f32,
    ) -> Result<String, JsValue> {
        if self.loaded.is_none() {
            return Ok("{}".to_string());
        }

        // Convert canvas-space brush coordinates to scene-space so
        // contains_point comparisons in conditional resolution use
        // the same coordinate space as mark positions.
        let (sx0, sy0) = self
            .zoom
            .transforms
            .get(panel_id as usize)
            .map(|t| t.inverse_apply(x0 as f64, y0 as f64))
            .unwrap_or((x0 as f64, y0 as f64));
        let (sx1, sy1) = self
            .zoom
            .transforms
            .get(panel_id as usize)
            .map(|t| t.inverse_apply(x1 as f64, y1 as f64))
            .unwrap_or((x1 as f64, y1 as f64));

        // Update interval selection state in scene-space.
        self.interaction_state
            .handle_drag(&self.selections, panel_id as usize, sx0, sy0, sx1, sy1);

        // D6 crossfilter (BindingRole::Filter): dim marks on a bound target
        // panel that fall outside this brush. Re-projects the source brush
        // extent into the target panel's pixel space via the shared data
        // domain, then reuses the conditional-containment dimming path.
        // No-op (and no behavior change) when there are no Filter bindings.
        let selection = self.apply_crossfilter(panel_id as usize, (sx0, sx1), (sy0, sy1))?;

        // D6 reactive rescale (BindingRole::Domain): rescale a bound target
        // panel to the brushed sub-domain via the existing zoom transform.
        // No-op (returns None) when there are no Domain bindings for the
        // brushed selection.
        let rescaled = self.apply_reactive_rescale(
            panel_id as usize,
            (x0 as f64, x1 as f64),
            (y0 as f64, y1 as f64),
        );

        // Envelope the selection-state JSON with the rescale signal. When a
        // Domain binding rescaled a target panel, the JS brush handler must NOT
        // clobber that affine with a trailing identity `setTransform` (Fix 1).
        // `rescaled` is the affected panel index (or null for plain
        // crossfilter/selection drags — unchanged behavior); `rescaled_text` is
        // that panel's re-placed label JSON so the overlay can reposition text
        // without resetting the transform.
        let selection_value: serde_json::Value =
            serde_json::from_str(&selection).unwrap_or(serde_json::Value::Null);
        let (rescaled_panel, rescaled_text) = match rescaled {
            Some((panel, text)) => (
                serde_json::Value::from(panel),
                serde_json::from_str(&text).unwrap_or(serde_json::Value::Null),
            ),
            None => (serde_json::Value::Null, serde_json::Value::Null),
        };
        Ok(serde_json::json!({
            "selection": selection_value,
            "rescaled": rescaled_panel,
            "rescaled_text": rescaled_text,
        })
        .to_string())
    }

    /// Set an absolute zoom+pan transform from D3-zoom for the given panel.
    ///
    /// `panel_id` identifies the panel to zoom (0-indexed); `k` is the uniform
    /// scale factor; `tx`/`ty` are the translation offsets.
    /// This replaces any accumulated per-panel zoom/pan state and is the sole
    /// entry point for HTML-export zoom driven by D3's `d3.zoom()`, including
    /// the dblclick-to-identity reset gesture.
    ///
    /// Secondary-y-axis (#52): resetting a panel's zoom/pan transform also
    /// resets every per-slot rescale affine that panel owns
    /// (`self.slot_rescales`) back to identity. Without this, a
    /// domainParam/brush rescale on an independent-y layer
    /// (`apply_reactive_rescale`) survives a view reset and keeps distorting
    /// that layer even though the panel affine itself is back at identity. A
    /// single-y panel owns exactly one slot, so this is a no-op there
    /// (byte-stable).
    ///
    /// Returns updated text-element JSON so the JS overlay can reposition labels.
    #[wasm_bindgen(js_name = "setTransform")]
    pub fn set_transform(
        &mut self,
        panel_id: u32,
        k: f32,
        tx: f32,
        ty: f32,
    ) -> Result<String, JsValue> {
        let panel = panel_id as usize;
        let Some(loaded) = &self.loaded else {
            return Ok("[]".to_string());
        };
        let slot_range = scene_load::panel_slot_range(&loaded.data.panel_slot_counts, panel);
        self.zoom
            .set_absolute(panel, k as f64, tx as f64, ty as f64);
        // Out-of-range `panel_id` must be a no-op (mirrors `set_absolute`'s
        // `get_mut` above): `panel_slot_range` always returns a range past
        // the end of `slot_rescales` for an out-of-range panel, so indexing
        // it directly would panic and abort the WASM module (#52 Task 9e).
        if let Some(slots) = self.slot_rescales.get_mut(slot_range) {
            for slot in slots {
                *slot = crate::zoom_pan::Affine2::identity();
            }
        }
        self.upload_transform_and_render(panel)
    }

    #[wasm_bindgen(js_name = "maxTextureSize")]
    pub fn max_texture_size(&self) -> u32 {
        self.gpu.device.limits().max_texture_dimension_2d
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.gpu.config.width = width.max(1);
        self.gpu.config.height = height.max(1);
        self.gpu
            .surface
            .configure(&self.gpu.device, &self.gpu.config);
        // FA-19: keep the MSAA target sized to the surface (this is also the
        // PNG-capture path that resizes the canvas to 2× DPR).
        self.gpu.rebuild_msaa_view();
        let _ = self.render_frame_js();
    }

    /// Return tooltip JSON for a specific mark instance.
    ///
    /// `panel_id` and `batch_idx` identify the packed batch; `node_idx` is
    /// the index of the mark within that batch.  Returns a JSON object
    #[wasm_bindgen(js_name = "hitTestAt")]
    pub fn hit_test_at(&self, x: f32, y: f32) -> String {
        let Some(loaded) = &self.loaded else {
            return "{}".to_string();
        };

        // Spatial-index + scene-graph hit-test (covers both packed and
        // non-packed batches via the R-tree built in load_scene). Baked
        // geometry (D4a amendment addendum).
        if let Some(hr) = hit_test::hit_test_nearest_with_index(
            &loaded.baked_panels,
            x as f64,
            y as f64,
            &self.zoom,
            self.spatial_index.as_ref(),
            &hit_test::SlotRescalePlan {
                slot_rescales: &self.slot_rescales,
                panel_slot_counts: &loaded.data.panel_slot_counts,
            },
        ) {
            return serde_json::json!({
                "panel": hr.panel_id,
                "batch": hr.batch_idx,
                "idx": hr.node_idx,
            })
            .to_string();
        }

        "{}".to_string()
    }

    /// `{"fields":[{"name":"x","value":"1.23"},…]}`, or `"{}"` if no
    /// tooltip data is available for this batch/instance.
    #[wasm_bindgen(js_name = "getTooltip")]
    pub fn get_tooltip(&self, panel_id: u32, batch_idx: u32, node_idx: u32) -> String {
        let Some(loaded) = &self.loaded else {
            return "{}".to_string();
        };

        // Try packed batch tooltip bytes first (binary sidecar for >= 1000 marks).
        let key = (panel_id, batch_idx);
        if let Some(meta) = loaded.data.packed_batch_meta.get(&key) {
            if let Some(ref tooltip_bytes) = meta.tooltip_bytes {
                return scene_load::parse_tooltip_json(tooltip_bytes, node_idx as usize);
            }
        }

        // Fall back to scene-graph tooltips (non-packed batches with < 1000 marks).
        if let Some(tooltip) = loaded
            .scene
            .panels
            .get(panel_id as usize)
            .and_then(|p| p.marks.get(batch_idx as usize))
            .and_then(|b| b.tooltips.as_ref())
            .and_then(|tips| tips.get(node_idx as usize))
        {
            return text_json::format_tooltip_content(tooltip);
        }

        "{}".to_string()
    }

    /// Return the href string for a specific mark node, or an empty string if
    /// none is present.
    ///
    /// `panel_id`, `batch_idx`, and `node_idx` correspond to the triple returned
    /// by `hitTestAt`.  The href is sourced from `batch.hrefs[node_idx]` in the
    /// scene graph.
    #[wasm_bindgen(js_name = "getHref")]
    pub fn get_href(&self, panel_id: u32, batch_idx: u32, node_idx: u32) -> String {
        let Some(loaded) = &self.loaded else {
            return String::new();
        };
        loaded
            .scene
            .panels
            .get(panel_id as usize)
            .and_then(|p| p.marks.get(batch_idx as usize))
            .and_then(|b| b.hrefs.as_ref())
            .and_then(|hrefs| hrefs.get(node_idx as usize))
            .and_then(|opt| opt.as_deref())
            .unwrap_or_default()
            .to_owned()
    }

    #[wasm_bindgen(js_name = "clearSelections")]
    pub fn clear_selections(&mut self) -> Result<String, JsValue> {
        for state in self.interaction_state.selections.values_mut() {
            *state = selection_state::SelectionState::Empty;
        }
        self.apply_conditionals_and_render()
    }

    /// Toggle a legend-bound point selection's membership for one category
    /// (D6 `BindingRole::Legend`).
    ///
    /// `selection_name` is the legend-bound point selection (from the `Legend`
    /// param binding); `category` is the legend entry's label. Toggling mirrors
    /// `handle_click`'s field-value point-selection update: the category is
    /// stored as a `FieldValue::String` so the existing conditional path dims
    /// or highlights every mark whose tooltip carries that value. Calling again
    /// with the same category removes it. After updating selection state this
    /// re-runs `apply_conditionals_and_render` (the same machinery legend-less
    /// point selections use).
    #[wasm_bindgen(js_name = "toggleLegend")]
    pub fn toggle_legend(
        &mut self,
        selection_name: &str,
        category: &str,
    ) -> Result<String, JsValue> {
        use ferrum_scene::FieldValue;
        use selection_state::SelectionState;

        // Find the field this selection toggles on. A legend-bound point
        // selection declares its `fields` (typically the color field); the
        // category is a value of that field. Fall back to a single synthetic
        // field name when none is declared so the toggle is still expressible.
        let field_name = self
            .selections
            .iter()
            .find_map(|spec| match spec {
                ferrum_scene::SelectionSpec::Point { name, fields, .. }
                    if name == selection_name =>
                {
                    fields.as_ref().and_then(|f| f.first()).cloned()
                }
                _ => None,
            })
            .unwrap_or_else(|| "_legend".to_string());

        let entry = (
            field_name,
            FieldValue::String {
                value: category.to_string(),
            },
        );
        let state = self
            .interaction_state
            .selections
            .entry(selection_name.to_string())
            .or_insert(SelectionState::Empty);

        // Toggle this category in/out of the field-value set, mirroring the
        // shift-click toggle in `handle_click`.
        let mut field_values: Vec<(String, FieldValue)> = match state {
            SelectionState::Point { field_values, .. } => field_values.clone(),
            _ => Vec::new(),
        };
        if let Some(pos) = field_values.iter().position(|fv| *fv == entry) {
            field_values.remove(pos);
        } else {
            field_values.push(entry);
        }
        if field_values.is_empty() {
            *state = SelectionState::Empty;
        } else {
            *state = SelectionState::Point {
                indices: Vec::new(),
                field_values,
            };
        }

        self.apply_conditionals_and_render()
    }
}

/// Private helpers that share implementation across public `wasm_bindgen` methods.
#[cfg(target_arch = "wasm32")]
impl WasmRenderer {
    /// Resolve conditional encodings against the current selection state,
    /// rebuild GPU buffers with updated instance colors, render a frame, and
    /// return the serialized selection state JSON.
    ///
    /// Called by both `handle_click` and `handle_drag` after they update the
    /// selection state. Delegates to `conditional::apply_conditionals_and_render`.
    fn apply_conditionals_and_render(&mut self) -> Result<String, JsValue> {
        let Some(loaded) = self.loaded.as_mut() else {
            return Ok("{}".to_string());
        };
        conditional::apply_conditionals_and_render(
            &loaded.baked_panels,
            &loaded.scene.interaction.conditionals,
            &loaded.data,
            &mut loaded.buffers,
            &self.interaction_state,
            &self.gpu,
            &self.pipelines,
        )
    }

    /// D6 crossfilter (`BindingRole::Filter`): dim marks on each bound target
    /// panel that fall outside the brush just applied on `source_panel`.
    ///
    /// `brush_x`/`brush_y` are the *scene-space* brush extents on the source
    /// panel (the same coordinates `handle_drag` stored in the interval
    /// selection). For each `Filter` binding whose target panel differs from
    /// the source, the source extent is re-projected into the target panel's
    /// pixel space through the shared data domain, then the existing
    /// conditional-containment dimming runs over that panel only.
    ///
    /// Returns the selection-state JSON (from the conditional re-render). When
    /// there are no `Filter` bindings this delegates straight to
    /// `apply_conditionals_and_render`, so non-param charts are unchanged.
    fn apply_crossfilter(
        &mut self,
        source_panel: usize,
        brush_x: (f64, f64),
        brush_y: (f64, f64),
    ) -> Result<String, JsValue> {
        use crate::param_runtime::{reproject_extent, Axis};
        use crate::selection_state::SelectionState;

        // Gather the active Filter bindings up front so we can drop the
        // immutable borrow of `self.interaction` before mutating buffers.
        let filter_targets: Vec<usize> = self
            .interaction
            .param_bindings
            .iter()
            .filter(|b| matches!(b.role, ferrum_scene::BindingRole::Filter))
            .filter_map(|b| b.panel)
            .collect();

        if filter_targets.is_empty() {
            return self.apply_conditionals_and_render();
        }

        let Some(loaded) = self.loaded.as_mut() else {
            return Ok("{}".to_string());
        };

        // Start from the base instance buffers so crossfilter dimming composes
        // cleanly with any declared conditionals applied below. Baked
        // geometry throughout (D4a amendment addendum) — `src.plot_area`/
        // `src.coord` below feed the pixel-space reprojection, so they must
        // already be at final on-screen position for a ratio-fitted panel.
        let conditionals = &loaded.scene.interaction.conditionals;
        let updates = conditional::resolve_conditionals_with_packed(
            &loaded.baked_panels,
            conditionals,
            &self.interaction_state.selections,
            &loaded.data.circle_instances,
            &loaded.data.rect_instances,
            &loaded.data.packed_batch_meta,
        );
        let mut circles = updates.circle_instances;
        let mut rects = updates.rect_instances;

        let source = loaded.baked_panels.get(source_panel);
        for &target_panel in &filter_targets {
            if target_panel == source_panel {
                continue;
            }
            let (Some(src), Some(tgt)) = (source, loaded.baked_panels.get(target_panel)) else {
                continue;
            };

            // Re-project the brushed extent on whichever axis has a shared
            // domain. Most crossfilters share the x-domain; try x first, then y.
            let mut x_range = None;
            let mut y_range = None;
            // Cross-panel crossfilter shares the primary scale group; the
            // per-layer y-slot (secondary-y, #52) is a within-panel concept and
            // same-panel targets are skipped above, so this inverts through
            // slot 0 — byte-identical to the pre-#52 crossfilter path.
            if let Some(r) = reproject_extent(
                brush_x,
                &src.plot_area,
                &src.coord,
                &tgt.plot_area,
                &tgt.coord,
                Axis::X,
                0,
            ) {
                x_range = Some(r);
            }
            if let Some(r) = reproject_extent(
                brush_y,
                &src.plot_area,
                &src.coord,
                &tgt.plot_area,
                &tgt.coord,
                Axis::Y,
                0,
            ) {
                y_range = Some(r);
            }
            if x_range.is_none() && y_range.is_none() {
                continue;
            }

            let sel = SelectionState::Interval { x_range, y_range };
            conditional::apply_crossfilter_to_panel(
                &loaded.baked_panels,
                target_panel,
                &sel,
                conditional::CROSSFILTER_DIM_OPACITY,
                &mut circles,
                &mut rects,
                &loaded.data.packed_batch_meta,
            );
        }

        loaded.buffers.update_instances(&self.gpu, &circles, &rects);
        render::render_frame(
            &self.gpu,
            &self.pipelines,
            &loaded.buffers,
            loaded.data.background,
        )
        .map_err(JsValue::from)?;
        Ok(self.interaction_state.to_json())
    }

    /// D6 reactive rescale (`BindingRole::Domain`): rescale each bound target
    /// panel to the brushed sub-region via the existing zoom transform.
    ///
    /// `brush_x`/`brush_y` are *canvas-space* brush extents on the source panel
    /// (pre-`inverse_apply`), matching the pixel space the zoom transform
    /// operates in. For each `Domain` binding, the per-axis boxzoom affine that
    /// maps the brushed source sub-extent onto the target panel's plot area is
    /// computed and pushed through `ZoomPanState::set_absolute` — the same
    /// affine the wheel/pan/D3-zoom path applies — then the target panel is
    /// re-rendered with that transform.
    ///
    /// A no-op (no transform mutation, no extra render) when there are no
    /// `Domain` bindings, so non-param interactive charts are unchanged.
    ///
    /// Returns the target panel that was rescaled (the last bound panel touched)
    /// together with that panel's re-placed text JSON, so the JS caller can
    /// avoid clobbering the freshly-applied affine with a trailing
    /// `setTransform`/identity reset and can still reposition the rescaled
    /// panel's labels. `None` means no rescale ran.
    fn apply_reactive_rescale(
        &mut self,
        source_panel: usize,
        brush_x: (f64, f64),
        brush_y: (f64, f64),
    ) -> Option<(usize, String)> {
        use crate::param_runtime::{rescale_affine_cross_panel, Axis};

        let bindings: Vec<(usize, Axis, (f64, f64), usize)> = self
            .interaction
            .param_bindings
            .iter()
            .filter(|b| matches!(b.role, ferrum_scene::BindingRole::Domain))
            .filter_map(|b| {
                let panel = b.panel?;
                let axis = b.channel.as_deref().and_then(Axis::from_channel)?;
                let brush = match axis {
                    Axis::X => brush_x,
                    Axis::Y => brush_y,
                };
                // `y_slot` selects the independent-y layer a y-domainParam is
                // bound to (secondary-y, #52); 0 = the primary scale group.
                Some((panel, axis, brush, b.y_slot))
            })
            .collect();

        if bindings.is_empty() {
            return None;
        }

        let zoom_range = self.zoom.zoom_range;
        let loaded = self.loaded.as_ref()?;
        let mut rendered_panel = None;
        for (panel, axis, brush, y_slot) in bindings {
            // Baked geometry (D4a amendment addendum): `src.plot_area`/
            // `src.coord` feed `rescale_affine_cross_panel`'s pixel-space
            // reprojection below.
            let (Some(src), Some(tgt)) = (
                loaded.baked_panels.get(source_panel),
                loaded.baked_panels.get(panel),
            ) else {
                continue;
            };
            // Reproject the brush from source-panel pixel space through the
            // owning layer's y-slot domain into target-panel pixel space before
            // building the affine. This is the correct path for `hconcat(
            // overview, detail)` where source and target occupy different pixel
            // regions; when both panels share the same plot area (single-panel
            // self-rescale) the reprojection is a no-op, preserving existing
            // behavior. See `param_runtime::rescale_affine_cross_panel`.
            let Some((scale, offset)) = rescale_affine_cross_panel(
                brush,
                &src.plot_area,
                &src.coord,
                &tgt.plot_area,
                &tgt.coord,
                axis,
                y_slot,
            ) else {
                continue;
            };
            // Fix 2: writing sx/sy directly bypasses set_absolute's clamp, so we
            // re-apply the same per-axis zoom_range clamp here. A narrow brush
            // must not exceed the 50x cap the wheel/D3-zoom path enforces.
            let scale = scale.clamp(zoom_range.0, zoom_range.1);
            // Route the rescale (secondary-y, #52). A y-domainParam on a panel
            // with independent-y layers rescales ONLY that layer's slot: it
            // writes a y-only affine into `slot_rescales`, which the render
            // upload composes with the shared per-panel zoom/pan affine
            // (`compose_panel_slot`), so overlaid layers stay locked under
            // zoom/pan while this one layer's marks move. Every other case (x on
            // any panel; y on a single-y panel) keeps the per-panel affine path,
            // byte-identical to the pre-#52 behavior.
            let slotted = axis == Axis::Y
                && loaded
                    .data
                    .panel_slot_counts
                    .get(panel)
                    .copied()
                    .unwrap_or(1)
                    > 1;
            if slotted {
                let idx = crate::scene_load::transform_slot_index(
                    &loaded.data.panel_slot_counts,
                    panel,
                    y_slot,
                );
                if let Some(r) = self.slot_rescales.get_mut(idx) {
                    r.sy = scale;
                    r.ty = offset;
                }
            } else if let Some(t) = self.zoom.transforms.get_mut(panel) {
                match axis {
                    Axis::X => {
                        t.sx = scale;
                        t.tx = offset;
                    }
                    Axis::Y => {
                        t.sy = scale;
                        t.ty = offset;
                    }
                }
            }
            rendered_panel = Some(panel);
        }

        let panel = rendered_panel?;
        let text_json = self
            .upload_transform_and_render(panel)
            .unwrap_or_else(|_| "[]".to_string());
        Some((panel, text_json))
    }

    /// Upload the per-panel affine transform uniform and re-render, then return zoomed text JSON.
    /// Delegates to `render::upload_transform_and_render`.
    fn upload_transform_and_render(&mut self, panel_id: usize) -> Result<String, JsValue> {
        let Some(loaded) = &self.loaded else {
            return Ok("[]".to_string());
        };
        // Baked geometry (D4a amendment addendum): the scissor/plot_area this
        // reads must already be at final on-screen position for a
        // ratio-fitted panel.
        render::upload_transform_and_render(
            &render::GpuResources {
                gpu: &self.gpu,
                pipelines: &self.pipelines,
                buffers: &loaded.buffers,
            },
            &loaded.data,
            &loaded.baked_panels,
            &self.interaction,
            &render::PanelTransformState {
                zoom_transforms: &self.zoom.transforms,
                slot_rescales: &self.slot_rescales,
            },
            panel_id,
        )
        .map_err(JsValue::from)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── W1: tick_label_json preserves text element angle ─────────────────

    /// tick_label_json must forward the style's angle, not hardcode 0.0.
    #[test]
    fn w1_tick_label_json_preserves_angle() {
        use ferrum_scene::{Color, FontWeight, TextAnchor, TextBaseline, TextStyle};

        let style = TextStyle {
            font_size: 12.0,
            font_weight: FontWeight::Normal,
            font_family: "sans-serif".to_string(),
            color: Color {
                r: 51,
                g: 51,
                b: 51,
                a: 255,
            },
            opacity: 1.0,
            anchor: TextAnchor::Middle,
            baseline: TextBaseline::Alphabetic,
            angle: 45.0,
        };

        let json = text_json::tick_label_json(100.0, 200.0, "label", "center", Some(&style));
        let angle = json["angle"].as_f64().expect("angle must be present");
        assert!(
            (angle - 45.0).abs() < 0.01,
            "tick_label_json must preserve style.angle (45.0), got {angle}"
        );
    }

    /// When no style is provided, the default angle must be 0.0.
    #[test]
    fn w1_tick_label_json_defaults_angle_to_zero() {
        let json = text_json::tick_label_json(0.0, 0.0, "0", "center", None);
        let angle = json["angle"].as_f64().expect("angle must be present");
        assert!(
            angle.abs() < 0.01,
            "tick_label_json with no style must have angle=0.0, got {angle}"
        );
    }

    /// tick_label_json with zero angle must still include the angle field.
    #[test]
    fn w1_tick_label_json_zero_angle_still_present() {
        use ferrum_scene::{Color, FontWeight, TextAnchor, TextBaseline, TextStyle};

        let style = TextStyle {
            font_size: 11.0,
            font_weight: FontWeight::Normal,
            font_family: "sans-serif".to_string(),
            color: Color {
                r: 51,
                g: 51,
                b: 51,
                a: 255,
            },
            opacity: 1.0,
            anchor: TextAnchor::Middle,
            baseline: TextBaseline::Alphabetic,
            angle: 0.0,
        };

        let json = text_json::tick_label_json(50.0, 300.0, "tick", "end", Some(&style));
        let angle = json["angle"].as_f64().expect("angle field must be present");
        assert!(
            angle.abs() < 0.01,
            "zero angle must produce angle=0.0 in JSON, got {angle}"
        );
    }

    // ── Bug 1a: format_tooltip_content for scene-graph tooltips ─────────

    #[test]
    fn format_tooltip_content_single_field() {
        use ferrum_scene::{TooltipContent, TooltipField};
        let tooltip = TooltipContent {
            fields: vec![TooltipField {
                name: "x".to_string(),
                value: "1.23".to_string(),
            }],
        };
        let json = text_json::format_tooltip_content(&tooltip);
        let parsed: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
        assert_eq!(parsed["fields"][0]["name"], "x");
        assert_eq!(parsed["fields"][0]["value"], "1.23");
        assert_eq!(parsed["fields"].as_array().map(|a| a.len()), Some(1));
    }

    #[test]
    fn format_tooltip_content_multiple_fields() {
        use ferrum_scene::{TooltipContent, TooltipField};
        let tooltip = TooltipContent {
            fields: vec![
                TooltipField {
                    name: "x".to_string(),
                    value: "1.23".to_string(),
                },
                TooltipField {
                    name: "y".to_string(),
                    value: "4.56".to_string(),
                },
            ],
        };
        let json = text_json::format_tooltip_content(&tooltip);
        let parsed: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
        assert_eq!(parsed["fields"][0]["name"], "x");
        assert_eq!(parsed["fields"][0]["value"], "1.23");
        assert_eq!(parsed["fields"][1]["name"], "y");
        assert_eq!(parsed["fields"][1]["value"], "4.56");
        assert_eq!(parsed["fields"].as_array().map(|a| a.len()), Some(2));
    }

    #[test]
    fn format_tooltip_content_empty_fields() {
        use ferrum_scene::TooltipContent;
        let tooltip = TooltipContent { fields: vec![] };
        let json = text_json::format_tooltip_content(&tooltip);
        assert_eq!(json, "{}");
    }

    #[test]
    fn format_tooltip_content_escapes_quotes() {
        use ferrum_scene::{TooltipContent, TooltipField};
        let tooltip = TooltipContent {
            fields: vec![TooltipField {
                name: "label".to_string(),
                value: r#"say "hello""#.to_string(),
            }],
        };
        let json = text_json::format_tooltip_content(&tooltip);
        let parsed: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
        assert_eq!(parsed["fields"][0]["name"], "label");
        assert_eq!(parsed["fields"][0]["value"], "say \"hello\"");
    }

    // ── R2: hit-testing packed batches via spatial index ─────────────────

    #[test]
    fn test_hit_test_packed_uses_spatial_index() {
        // Verify that packed batches (empty nodes, instances in binary sidecar)
        // are findable through the spatial index path in hit_test_nearest_with_index.
        use crate::scene_load::{CircleInstance, DrawKind, PackedBatchMeta, SceneData};
        use crate::spatial_index::SpatialIndex;
        use ferrum_scene::{
            BlendMode, CoordKind, LayoutScale, MarkBatch, MarkBatchKind, Panel, Rect,
        };
        use lyon::tessellation::VertexBuffers;
        use std::collections::HashMap;

        // Build a panel with one packed batch (empty nodes).
        let panels = vec![Panel {
            id: 0,
            plot_area: Rect {
                x: 0.0,
                y: 0.0,
                w: 500.0,
                h: 500.0,
            },
            clip: Rect {
                x: 0.0,
                y: 0.0,
                w: 500.0,
                h: 500.0,
            },
            coord: CoordKind::Cartesian {
                x_domain: None,
                y_domain: None,
                expand: true,
                clip: true,
                y_domains: Vec::new(),
            },
            grid: vec![],
            marks: vec![MarkBatch {
                kind: MarkBatchKind::Point,
                nodes: vec![], // empty — packed batch
                data_indices: None,
                tooltips: None,
                hrefs: None,
                keys: None,
                blend: BlendMode::Normal,
                descriptions: None,
                stroke_cap: None,
                stroke_join: None,
                packed_instances: None,
                y_slot: 0,
            }],
            axes: vec![],
            annotations: vec![],
            strip_title: vec![],
            layout_scale: LayoutScale::identity(),
            below_marks: Vec::new(),
            chrome_above: Vec::new(),
        }];

        let mut packed_meta = HashMap::new();
        packed_meta.insert(
            (0u32, 0u32),
            PackedBatchMeta {
                keys: None,
                data_indices: Some(vec![7, 8, 9]),
                tooltip_bytes: None,
                kind: DrawKind::Circle,
                instance_start: 0,
                instance_count: 3,
            },
        );

        let data = SceneData {
            circle_instances: vec![
                CircleInstance {
                    center: [100.0, 100.0],
                    radius: 5.0,
                    fill_color: [0.0; 4],
                    stroke_color: [0.0; 4],
                    stroke_width: 0.0,
                    opacity: 1.0,
                    stroke_opacity: 0.0,
                    stroke_dash: 0.0,
                    angle: 0.0,
                },
                CircleInstance {
                    center: [200.0, 200.0],
                    radius: 5.0,
                    fill_color: [0.0; 4],
                    stroke_color: [0.0; 4],
                    stroke_width: 0.0,
                    opacity: 1.0,
                    stroke_opacity: 0.0,
                    stroke_dash: 0.0,
                    angle: 0.0,
                },
                CircleInstance {
                    center: [300.0, 300.0],
                    radius: 5.0,
                    fill_color: [0.0; 4],
                    stroke_color: [0.0; 4],
                    stroke_width: 0.0,
                    opacity: 1.0,
                    stroke_opacity: 0.0,
                    stroke_dash: 0.0,
                    angle: 0.0,
                },
            ],
            rect_instances: vec![],
            mesh_buffers: VertexBuffers::new(),
            static_mesh_buffers: VertexBuffers::new(),
            annotation_mesh_buffers: VertexBuffers::new(),
            text_elements: vec![],
            image_quads: vec![],
            raw_fragments: vec![],
            background: None,
            width: 500.0,
            height: 500.0,
            packed_batch_meta: packed_meta,
            draw_commands: vec![],
            mark_mesh_panels: vec![],
            panel_count: 1,
            panel_slot_counts: vec![1],
        };

        // Build spatial index with packed data.
        let idx = SpatialIndex::build_with_packed(&panels, Some(&data));
        let zoom =
            crate::zoom_pan::ZoomPanState::new(1, &ferrum_scene::InteractionConfig::default());

        // Use the spatial-index-aware hit test (the same path hit_test_at uses).
        let result = hit_test::hit_test_nearest_with_index(
            &panels,
            100.0,
            100.0,
            &zoom,
            Some(&idx),
            &hit_test::SlotRescalePlan::identity(),
        );
        let hr = result.expect("packed circle at (100,100) must be found via spatial index");
        assert_eq!(hr.panel_id, 0);
        assert_eq!(hr.batch_idx, 0);
        assert_eq!(hr.node_idx, 0);
        assert_eq!(hr.data_idx, Some(7));

        // Also check the second packed circle.
        let result2 = hit_test::hit_test_nearest_with_index(
            &panels,
            200.0,
            200.0,
            &zoom,
            Some(&idx),
            &hit_test::SlotRescalePlan::identity(),
        );
        let hr2 = result2.expect("packed circle at (200,200) must be found");
        assert_eq!(hr2.data_idx, Some(8));
    }
}
