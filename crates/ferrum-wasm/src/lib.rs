#![deny(clippy::unwrap_used)]

// Pure-logic modules: compile on all targets (host tests use these).
pub mod conditional;
pub mod error;
pub mod hit_test;
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
use crate::render::{GpuBuffers, Uniforms};
#[cfg(target_arch = "wasm32")]
use crate::scene_load::SceneData;
#[cfg(target_arch = "wasm32")]
use crate::selection_state::InteractionState;
#[cfg(target_arch = "wasm32")]
use crate::spatial_index::SpatialIndex;
#[cfg(target_arch = "wasm32")]
use crate::transition::{ease_in_out_cubic, lerp_circles, lerp_rects};
#[cfg(target_arch = "wasm32")]
use crate::zoom_pan::{ScaleMode, ZoomPanState};

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
    interaction: ferrum_scene::InteractionConfig,
    spatial_index: Option<SpatialIndex>,
}

#[cfg(target_arch = "wasm32")]
struct ActiveTransition {
    old_data: SceneData,
    new_data: SceneData,
    new_scene: ferrum_scene::SceneGraph,
}

#[cfg(target_arch = "wasm32")]
struct LoadedScene {
    data: SceneData,
    buffers: GpuBuffers,
    scene: ferrum_scene::SceneGraph,
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl WasmRenderer {
    #[wasm_bindgen(js_name = "create")]
    pub async fn create(canvas: HtmlCanvasElement) -> Result<WasmRenderer, JsValue> {
        console_error_panic_hook::set_once();
        let gpu = gpu::init_gpu(canvas).await.map_err(JsValue::from)?;
        let pipelines = RenderPipelines::new(&gpu.device, gpu.format);
        Ok(WasmRenderer {
            gpu,
            pipelines,
            loaded: None,
            transition: None,
            selections: Vec::new(),
            interaction_state: InteractionState::new(&[]),
            zoom: ZoomPanState::new(0, &ferrum_scene::InteractionConfig::default()),
            interaction: ferrum_scene::InteractionConfig::default(),
            spatial_index: None,
        })
    }

    #[wasm_bindgen(js_name = "loadScene")]
    pub fn load_scene(&mut self, scene_json: &str, packed_data: &[u8]) -> Result<String, JsValue> {
        let scene: ferrum_scene::SceneGraph = serde_json::from_str(scene_json)
            .map_err(|e| JsValue::from(WasmRenderError::SceneDeserialization(e.to_string())))?;

        let data = scene_load::load_scene_with_packed(&scene, packed_data);
        let text_json = text_json::build_text_json(&data);
        let buffers = GpuBuffers::from_scene(&self.gpu, &self.pipelines, &data);
        let clear_color = data.background;

        self.selections = scene.selections.clone();
        self.interaction_state = InteractionState::new(&self.selections);
        self.interaction = scene.interaction.clone();
        self.zoom = ZoomPanState::new(scene.panels.len(), &self.interaction);
        // Build spatial index over all panels for O(log n) hit-testing.
        // Pass scene data so packed instances (>= 1000 marks) are also indexed.
        self.spatial_index = Some(SpatialIndex::build_with_packed(&scene.panels, Some(&data)));
        self.loaded = Some(LoadedScene { data, buffers, scene });

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
    /// `old_scene_json` is the **previous** scene JSON string. The transition
    /// target is `self.loaded.data` (the scene already loaded via `loadScene`).
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
    pub fn start_transition(
        &mut self,
        old_scene_json: &str,
    ) -> Result<(), JsValue> {
        let Some(loaded) = &self.loaded else {
            return Ok(());
        };
        // B4+B5 fix: the new scene (transition target) is already in loaded.data,
        // which was populated by loadScene with full packed data. Parse the old
        // scene from JSON (no packed data needed — it is only the animation source
        // for a brief 300ms transition).
        let new_data = loaded.data.clone();
        let old_scene: ferrum_scene::SceneGraph = serde_json::from_str(old_scene_json)
            .map_err(|e| JsValue::from(WasmRenderError::SceneDeserialization(e.to_string())))?;
        let old_data = scene_load::load_scene(&old_scene);
        self.transition = Some(ActiveTransition { old_data, new_data, new_scene: loaded.scene.clone() });
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
            let lerped_circles = lerp_circles(&tr.old_data.circle_instances, &tr.new_data.circle_instances, t_eased);
            let lerped_rects = lerp_rects(&tr.old_data.rect_instances, &tr.new_data.rect_instances, t_eased);
            let lerped_data = SceneData {
                circle_instances: lerped_circles,
                rect_instances: lerped_rects,
                mesh_buffers: tr.new_data.mesh_buffers.clone(),
                static_mesh_buffers: tr.new_data.static_mesh_buffers.clone(),
                annotation_mesh_buffers: tr.new_data.annotation_mesh_buffers.clone(),
                text_elements: tr.new_data.text_elements.clone(),
                image_quads: tr.new_data.image_quads.clone(),
                background: tr.new_data.background,
                width: tr.new_data.width,
                height: tr.new_data.height,
                packed_batch_meta: tr.new_data.packed_batch_meta.clone(),
                draw_commands: tr.new_data.draw_commands.clone(),
            };
            let buffers = GpuBuffers::from_scene(&self.gpu, &self.pipelines, &lerped_data);
            render::render_frame(&self.gpu, &self.pipelines, &buffers, lerped_data.background)
                .map_err(JsValue::from)?;
            if t >= 1.0 {
                let final_data = tr.new_data.clone();
                let final_buffers = GpuBuffers::from_scene(&self.gpu, &self.pipelines, &final_data);
                let final_scene = tr.new_scene.clone();
                self.transition = None;
                self.loaded = Some(LoadedScene { data: final_data, buffers: final_buffers, scene: final_scene });
            }
        }
        Ok(())
    }

    /// Hit-test a click at canvas pixel (x, y), update selection state, apply
    /// conditional encodings (dim non-selected marks), re-render frame, and
    /// return the new selection state as a JSON string.
    ///
    /// The returned JSON is a map of `selection_name → {field_name: field_value}`.
    /// The JS caller should forward this to `model.set('selection_state', ...)`.
    #[wasm_bindgen(js_name = "handleClick")]
    pub fn handle_click(&mut self, x: f32, y: f32, shift_held: bool) -> Result<String, JsValue> {
        let Some(loaded) = self.loaded.as_mut() else {
            return Ok("{}".to_string());
        };

        // Update selection state via Rust hit-test (authoritative — operates on actual scene data).
        // Pass spatial index for O(log n) circle/rect hit-testing.
        self.interaction_state.handle_click_with_index(
            &loaded.scene.panels,
            &self.selections,
            x as f64,
            y as f64,
            &self.zoom,
            shift_held,
            self.spatial_index.as_ref(),
        );

        self.apply_conditionals_and_render()
    }

    /// Handle a brush-drag on a panel: update interval selection state, apply
    /// conditional encodings, rebuild GPU buffers, re-render, and return
    /// the new selection state as JSON.
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
        let (sx0, sy0) = self.zoom.transforms
            .get(panel_id as usize)
            .map(|t| t.inverse_apply(x0 as f64, y0 as f64))
            .unwrap_or((x0 as f64, y0 as f64));
        let (sx1, sy1) = self.zoom.transforms
            .get(panel_id as usize)
            .map(|t| t.inverse_apply(x1 as f64, y1 as f64))
            .unwrap_or((x1 as f64, y1 as f64));

        // Update interval selection state in scene-space.
        self.interaction_state.handle_drag(
            &self.selections,
            panel_id as usize,
            sx0, sy0, sx1, sy1,
        );

        self.apply_conditionals_and_render()
    }

    /// Apply a wheel-zoom event on the given panel and re-render via GPU affine transform.
    ///
    /// Returns updated text-element JSON (tick labels at new positions) so the JS
    /// overlay can reposition axis labels without a Python round-trip.
    #[wasm_bindgen(js_name = "onWheel")]
    pub fn on_wheel(&mut self, panel_id: u32, delta_y: f32, cx: f32, cy: f32) -> Result<String, JsValue> {
        let Some(loaded) = &self.loaded else { return Ok("[]".to_string()); };
        let coord = loaded.scene.panels.get(panel_id as usize).map(|p| &p.coord);
        if matches!(coord, Some(ferrum_scene::CoordKind::Polar { .. } | ferrum_scene::CoordKind::Geo { .. })) {
            return Ok(text_json::build_text_json_from(&loaded.data.text_elements));
        }
        let scale_mode = match coord {
            Some(ferrum_scene::CoordKind::Fixed { .. }) => ScaleMode::Uniform,
            _ => ScaleMode::Independent,
        };
        self.zoom.on_wheel(panel_id as usize, delta_y as f64, cx as f64, cy as f64, scale_mode);
        self.upload_transform_and_render(panel_id as usize)
    }

    /// Apply a pan delta on the given panel and re-render via GPU affine transform.
    ///
    /// Returns updated text-element JSON.
    #[wasm_bindgen(js_name = "onPan")]
    pub fn on_pan(&mut self, panel_id: u32, dx: f32, dy: f32) -> Result<String, JsValue> {
        let Some(loaded) = &self.loaded else { return Ok("[]".to_string()); };
        let coord = loaded.scene.panels.get(panel_id as usize).map(|p| &p.coord);
        if matches!(coord, Some(ferrum_scene::CoordKind::Polar { .. } | ferrum_scene::CoordKind::Geo { .. })) {
            return Ok(text_json::build_text_json_from(&loaded.data.text_elements));
        }
        self.zoom.on_pan(panel_id as usize, dx as f64, dy as f64);
        self.upload_transform_and_render(panel_id as usize)
    }

    /// Reset zoom/pan to identity for the given panel and re-render.
    ///
    /// Returns text-element JSON with tick labels at their original positions.
    #[wasm_bindgen(js_name = "resetZoom")]
    pub fn reset_zoom(&mut self, panel_id: u32) -> Result<String, JsValue> {
        let Some(loaded) = &self.loaded else { return Ok("[]".to_string()); };
        self.zoom.reset(panel_id as usize);
        let uniforms = Uniforms::identity(loaded.data.width, loaded.data.height);
        loaded.buffers.upload_uniforms(&self.gpu, &uniforms);
        render::render_frame(&self.gpu, &self.pipelines, &loaded.buffers, loaded.data.background)
            .map_err(JsValue::from)?;
        Ok(text_json::build_text_json(&loaded.data))
    }

    /// Set an absolute zoom+pan transform from D3-zoom.
    ///
    /// `k` is the uniform scale factor; `tx`/`ty` are the translation offsets.
    /// This replaces the accumulated state from `onWheel`/`onPan` and is the
    /// entry point for HTML-export zoom driven by D3's `d3.zoom()`.
    ///
    /// Operates on panel 0 (single-panel charts; multi-panel support later).
    /// Returns updated text-element JSON so the JS overlay can reposition labels.
    #[wasm_bindgen(js_name = "setTransform")]
    pub fn set_transform(&mut self, k: f32, tx: f32, ty: f32) -> Result<String, JsValue> {
        let Some(_loaded) = &self.loaded else { return Ok("[]".to_string()); };
        self.zoom.set_absolute(0, k as f64, tx as f64, ty as f64);
        self.upload_transform_and_render(0)
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
        let _ = self.render_frame_js();
    }

    /// Return tooltip JSON for a specific mark instance.
    ///
    /// `panel_id` and `batch_idx` identify the packed batch; `node_idx` is
    /// the index of the mark within that batch.  Returns a JSON object
    #[wasm_bindgen(js_name = "hitTestAt")]
    pub fn hit_test_at(&self, x: f32, y: f32) -> String {
        let Some(loaded) = &self.loaded else { return "{}".to_string(); };

        // Spatial-index + scene-graph hit-test (covers both packed and
        // non-packed batches via the R-tree built in load_scene).
        if let Some(hr) = hit_test::hit_test_nearest_with_index(
            &loaded.scene.panels, x as f64, y as f64, &self.zoom,
            self.spatial_index.as_ref(),
        ) {
            return serde_json::json!({
                "panel": hr.panel_id,
                "batch": hr.batch_idx,
                "idx": hr.node_idx,
            }).to_string();
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
        if let Some(tooltip) = loaded.scene.panels
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

    /// Select all indexed marks (circles and rects) within the given scene-space
    /// rectangle `(x0, y0) – (x1, y1)` using the R-tree spatial index.
    ///
    /// Updates the first `Interval` selection spec found in `self.selections`,
    /// then applies conditional encodings and re-renders.  Returns the new
    /// selection state JSON.
    ///
    /// If no spatial index has been built yet (scene not loaded), returns `"{}"`.
    #[wasm_bindgen(js_name = "selectInRect")]
    pub fn select_in_rect(
        &mut self,
        _panel_id: u32,
        x0: f32,
        y0: f32,
        x1: f32,
        y1: f32,
    ) -> Result<String, JsValue> {
        if self.loaded.is_none() {
            return Ok("{}".to_string());
        }

        // Normalise the selection rectangle to (lo, hi) on each axis.
        let lo_x = (x0 as f64).min(x1 as f64);
        let hi_x = (x0 as f64).max(x1 as f64);
        let lo_y = (y0 as f64).min(y1 as f64);
        let hi_y = (y0 as f64).max(y1 as f64);

        // Update the interval selection state with the bounding rectangle.
        // The Interval selection uses spatial containment (contains_point) during
        // conditional encoding resolution — it does not need explicit data indices.
        // The R-tree query above determines *which* marks are inside the box;
        // the conditional layer uses contains_point against mark positions at
        // render time, so storing x_range/y_range is sufficient.
        for spec in &self.selections {
            if let ferrum_scene::SelectionSpec::Interval { name, .. } = spec {
                let name = name.clone();
                self.interaction_state.selections.insert(
                    name,
                    crate::selection_state::SelectionState::Interval {
                        x_range: Some((lo_x, hi_x)),
                        y_range: Some((lo_y, hi_y)),
                    },
                );
                break;
            }
        }

        self.apply_conditionals_and_render()
    }

    #[wasm_bindgen(js_name = "clearSelections")]
    pub fn clear_selections(&mut self) -> Result<String, JsValue> {
        for state in self.interaction_state.selections.values_mut() {
            *state = selection_state::SelectionState::Empty;
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
            &loaded.scene,
            &loaded.data,
            &mut loaded.buffers,
            &self.interaction_state,
            &self.gpu,
            &self.pipelines,
        )
    }

    /// Upload the per-panel affine transform uniform and re-render, then return zoomed text JSON.
    /// Delegates to `render::upload_transform_and_render`.
    fn upload_transform_and_render(&mut self, panel_id: usize) -> Result<String, JsValue> {
        let Some(loaded) = &self.loaded else { return Ok("[]".to_string()); };
        render::upload_transform_and_render(
            &self.gpu,
            &self.pipelines,
            &loaded.buffers,
            &loaded.data,
            &loaded.scene.panels,
            &self.interaction,
            &self.zoom.transforms,
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
            color: Color { r: 51, g: 51, b: 51, a: 255 },
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
            color: Color { r: 51, g: 51, b: 51, a: 255 },
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
                TooltipField { name: "x".to_string(), value: "1.23".to_string() },
                TooltipField { name: "y".to_string(), value: "4.56".to_string() },
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
        use crate::scene_load::{CircleInstance, PackedBatchMeta, SceneData};
        use crate::spatial_index::SpatialIndex;
        use ferrum_scene::{
            BlendMode, CoordKind, MarkBatch, MarkBatchKind, Panel, Rect,
        };
        use lyon::tessellation::VertexBuffers;
        use std::collections::HashMap;

        // Build a panel with one packed batch (empty nodes).
        let panels = vec![Panel {
            id: 0,
            plot_area: Rect { x: 0.0, y: 0.0, w: 500.0, h: 500.0 },
            clip: Rect { x: 0.0, y: 0.0, w: 500.0, h: 500.0 },
            coord: CoordKind::Cartesian {
                x_domain: None, y_domain: None, expand: true, clip: true,
            },
            grid: vec![],
            marks: vec![MarkBatch {
                kind: MarkBatchKind::Point,
                nodes: vec![],  // empty — packed batch
                data_indices: None,
                tooltips: None,
                hrefs: None,
                keys: None,
                blend: BlendMode::Normal,
                descriptions: None,
                stroke_cap: None,
                stroke_join: None,
                packed_instances: None,
            }],
            axes: vec![],
            annotations: vec![],
            strip_title: vec![],
        }];

        let mut packed_meta = HashMap::new();
        packed_meta.insert(
            (0u32, 0u32),
            PackedBatchMeta {
                data_indices: Some(vec![7, 8, 9]),
                tooltip_bytes: None,
                kind: 0,
                instance_start: 0,
                instance_count: 3,
            },
        );

        let data = SceneData {
            circle_instances: vec![
                CircleInstance {
                    center: [100.0, 100.0], radius: 5.0,
                    fill_color: [0.0; 4], stroke_color: [0.0; 4],
                    stroke_width: 0.0, opacity: 1.0, stroke_opacity: 0.0,
                    stroke_dash: 0.0, angle: 0.0,
                },
                CircleInstance {
                    center: [200.0, 200.0], radius: 5.0,
                    fill_color: [0.0; 4], stroke_color: [0.0; 4],
                    stroke_width: 0.0, opacity: 1.0, stroke_opacity: 0.0,
                    stroke_dash: 0.0, angle: 0.0,
                },
                CircleInstance {
                    center: [300.0, 300.0], radius: 5.0,
                    fill_color: [0.0; 4], stroke_color: [0.0; 4],
                    stroke_width: 0.0, opacity: 1.0, stroke_opacity: 0.0,
                    stroke_dash: 0.0, angle: 0.0,
                },
            ],
            rect_instances: vec![],
            mesh_buffers: VertexBuffers::new(),
            static_mesh_buffers: VertexBuffers::new(),
            annotation_mesh_buffers: VertexBuffers::new(),
            text_elements: vec![],
            image_quads: vec![],
            background: None,
            width: 500.0,
            height: 500.0,
            packed_batch_meta: packed_meta,
            draw_commands: vec![],
        };

        // Build spatial index with packed data.
        let idx = SpatialIndex::build_with_packed(&panels, Some(&data));
        let zoom = crate::zoom_pan::ZoomPanState::new(1, &ferrum_scene::InteractionConfig::default());

        // Use the spatial-index-aware hit test (the same path hit_test_at uses).
        let result = hit_test::hit_test_nearest_with_index(
            &panels, 100.0, 100.0, &zoom, Some(&idx),
        );
        let hr = result.expect("packed circle at (100,100) must be found via spatial index");
        assert_eq!(hr.panel_id, 0);
        assert_eq!(hr.batch_idx, 0);
        assert_eq!(hr.node_idx, 0);
        assert_eq!(hr.data_idx, Some(7));

        // Also check the second packed circle.
        let result2 = hit_test::hit_test_nearest_with_index(
            &panels, 200.0, 200.0, &zoom, Some(&idx),
        );
        let hr2 = result2.expect("packed circle at (200,200) must be found");
        assert_eq!(hr2.data_idx, Some(8));
    }
}
