//! Text-element JSON serialization for the WASM renderer overlay.
//!
//! The JavaScript rendering layer positions DOM text elements (axis labels,
//! tick labels, titles) from a JSON array emitted by the Rust side. This
//! module centralizes that serialization.

use crate::scene_load::TextElementData;

// ── Shared enum-to-str helpers ──────────────────────────────────────────────

fn font_weight_string(w: &ferrum_scene::FontWeight) -> String {
    match w {
        ferrum_scene::FontWeight::Normal => "normal".to_string(),
        ferrum_scene::FontWeight::Bold => "bold".to_string(),
        ferrum_scene::FontWeight::Custom(s) => s.clone(),
    }
}

fn text_anchor_str(a: &ferrum_scene::TextAnchor) -> &'static str {
    match a {
        ferrum_scene::TextAnchor::Start => "start",
        ferrum_scene::TextAnchor::Middle => "center",
        ferrum_scene::TextAnchor::End => "end",
    }
}

/// Owned variant that handles `Custom` baselines correctly.
fn text_baseline_string(b: &ferrum_scene::TextBaseline) -> String {
    match b {
        ferrum_scene::TextBaseline::Top => "top".to_string(),
        ferrum_scene::TextBaseline::Middle => "middle".to_string(),
        ferrum_scene::TextBaseline::Bottom => "bottom".to_string(),
        ferrum_scene::TextBaseline::Alphabetic => "alphabetic".to_string(),
        ferrum_scene::TextBaseline::Custom(s) => s.clone(),
    }
}

fn color_string(style: &ferrum_scene::TextStyle) -> String {
    format!(
        "rgba({},{},{},{})",
        style.color.r, style.color.g, style.color.b, style.opacity
    )
}

// ── Public serialization functions ──────────────────────────────────────────

/// Build the combined overlay JSON object `{"text": [...], "raw": [...]}` for
/// `WasmRenderer::load_scene`.
///
/// The JS widget parses this once on scene load: `text` is forwarded to
/// `_placeTextSvg`; `raw` is cached and injected into the SVG overlay as
/// verbatim fragments with ID namespacing and anchor-based grouping.
#[cfg(target_arch = "wasm32")]
pub(crate) fn build_overlay_json(data: &crate::scene_load::SceneData) -> String {
    let text_elements: Vec<serde_json::Value> = data
        .text_elements
        .iter()
        .map(text_element_to_json)
        .collect();
    let raw_elements: Vec<serde_json::Value> = data
        .raw_fragments
        .iter()
        .map(|r| {
            serde_json::json!({
                "svg": r.svg,
                "anchor": r.anchor,
            })
        })
        .collect();
    serde_json::json!({
        "text": text_elements,
        "raw": raw_elements,
    })
    .to_string()
}

/// Build the text-element JSON array from a slice of text elements.
pub(crate) fn build_text_json_from(all_text: &[TextElementData]) -> String {
    let elements: Vec<serde_json::Value> = all_text.iter().map(text_element_to_json).collect();
    serde_json::to_string(&elements).unwrap_or_else(|_| "[]".to_string())
}

/// Look up the composed `panel ∘ slot-rescale` affine for a right-axis tick's
/// y-slot (secondary-y-axis, GH #52/#60/#63/#73).
///
/// `secondary_affines` is secondary-only and 1-based: `secondary_affines[r]` is
/// the right axis of rank `r` (0-based, stacking outward) — y-slot `r + 1` in
/// the all-slot numbering every mark/domain collection uses. `slot` here is
/// the all-slot number (`>= 1`, guaranteed by the caller's `is_secondary_y_tick`
/// gate below), so this is the ONE place that translates it into
/// `secondary_affines`' 0-based index — no consumer should compute
/// `slot - 1` itself. Falls back to `secondary_affines.last()` when `slot`
/// has no corresponding entry (a documented degradation, unchanged from the
/// pre-accessor inline chain), and to `panel_affine` when no secondary
/// affines were supplied at all (pure zoom/pan or single-y), byte-identical
/// to the pre-9d panel-affine-only path.
fn secondary_affine_for_slot(
    secondary_affines: &[crate::zoom_pan::Affine2],
    slot: usize,
    panel_affine: &crate::zoom_pan::Affine2,
) -> crate::zoom_pan::Affine2 {
    secondary_affines
        .get(slot - 1)
        .or_else(|| secondary_affines.last())
        .copied()
        .unwrap_or(*panel_affine)
}

/// Build text-element JSON for a zoomed panel.
///
/// Axis tick labels are identified by clustering text elements that share the
/// same y coordinate (x-axis row) or same x coordinate (y-axis column) and
/// whose content appears in the known tick-label set.  Each identified tick
/// label is repositioned by applying the affine zoom transform to its varying
/// coordinate (x for x-axis ticks, y for y-axis ticks).  All other text
/// (chart title, axis title, legend) is emitted at its original position.
///
/// Labels whose transformed position falls outside the panel's `plot_area`
/// are filtered out to prevent tick labels from extending past the plot
/// boundary during zoom.
///
/// Secondary right-axis labels (#52 / criterion 8) relabel through their own
/// composed affine, not the shared panel affine: `secondary_affines[r]` is the
/// `panel ∘ slot-rescale` for the right axis of rank `r` (0-based, stacking
/// outward from the plot edge → y-slot `r + 1`). A `domainParam`/brush bound to
/// one independent-y layer writes a y-only rescale into that layer's slot, so
/// its axis labels must move even when the shared panel affine is identity —
/// symmetric with the primary axis, which relabels through its own domainParam
/// via the panel affine. `secondary_affines` is empty for single-y panels and
/// under pure zoom/pan every entry equals the panel affine, so the output is
/// byte-identical to the primary-only path. Looked up per tick via
/// [`secondary_affine_for_slot`], which owns the all-slot-to-rank translation.
pub(crate) fn build_zoomed_text_json(
    all_text: &[TextElementData],
    interaction: &ferrum_scene::InteractionConfig,
    panel_id: usize,
    transform: &crate::zoom_pan::Affine2,
    secondary_affines: &[crate::zoom_pan::Affine2],
    plot_area: Option<(f64, f64, f64, f64)>,
) -> String {
    use std::collections::{HashMap, HashSet};

    let Some(ptl) = interaction
        .tick_levels
        .iter()
        .find(|p| p.panel_id == panel_id)
    else {
        return build_text_json_from(all_text);
    };

    // Union of tick label strings across all zoom levels.
    let x_tick_labels: HashSet<&str> = ptl
        .x_levels
        .iter()
        .flat_map(|lvl| lvl.ticks.iter().map(|t| t.label.as_str()))
        .collect();
    let y_tick_labels: HashSet<&str> = ptl
        .y_levels
        .iter()
        .flat_map(|lvl| lvl.ticks.iter().map(|t| t.label.as_str()))
        .collect();
    // --- Identify x-axis tick row ------------------------------------------
    // All x-axis tick labels share the same y coordinate.  Find the most
    // common rounded-y among elements whose content is a known x-tick label.
    let mut x_y_freq: HashMap<i64, usize> = HashMap::new();
    for te in all_text
        .iter()
        .filter(|te| x_tick_labels.contains(te.content.as_str()))
    {
        *x_y_freq.entry((te.y * 10.0) as i64).or_insert(0) += 1;
    }
    let x_axis_y: Option<f64> = x_y_freq
        .into_iter()
        .max_by_key(|(_, c)| *c)
        .map(|(k, _)| k as f64 / 10.0);

    // --- Identify y-axis tick column ----------------------------------------
    // All y-axis tick labels share the same x coordinate.
    let mut y_x_freq: HashMap<i64, usize> = HashMap::new();
    for te in all_text
        .iter()
        .filter(|te| y_tick_labels.contains(te.content.as_str()))
    {
        *y_x_freq.entry((te.x * 10.0) as i64).or_insert(0) += 1;
    }
    let y_axis_x: Option<f64> = y_x_freq
        .into_iter()
        .max_by_key(|(_, c)| *c)
        .map(|(k, _)| k as f64 / 10.0);

    // 1 px tolerance covers label_font_size / 3.0 baseline offset and rounding.
    const COORD_TOL: f64 = 1.5;

    let is_x_tick = |te: &TextElementData| {
        x_tick_labels.contains(te.content.as_str())
            && x_axis_y
                .map(|ay| (te.y - ay).abs() < COORD_TOL)
                .unwrap_or(false)
    };
    let is_y_tick = |te: &TextElementData| {
        y_tick_labels.contains(te.content.as_str())
            && y_axis_x
                .map(|ax| (te.x - ax).abs() < COORD_TOL)
                .unwrap_or(false)
    };
    // Secondary-y (#52/#60/#73): a right-axis tick is identified by its
    // explicit `slot` tag (`Some(k)`, k >= 1 — slot 0 is the primary axis,
    // per the `MarkBatch::y_slot` convention `route_y_axis_slotted` mirrors),
    // not by column-frequency inference. This retires the `c >= 2` column
    // heuristic: untagged text (`slot: None`) — titles, legends, or a stray
    // label that happens to match a tick string — is *never* treated as an
    // axis label, which is a STRONGER structural guarantee than the
    // frequency filter it replaces (spec §7). A single-tick right axis
    // (degenerate/constant secondary domain) now relabels exactly like a
    // multi-tick one, since recognition no longer depends on counting
    // repeated column occupants.
    let is_secondary_y_tick = |te: &TextElementData| matches!(te.slot, Some(slot) if slot >= 1);

    let mut elements: Vec<serde_json::Value> = Vec::new();
    for te in all_text {
        if is_x_tick(te) {
            // Apply sx + tx to the x coordinate; y stays at the axis level.
            let new_x = te.x * transform.sx + transform.tx;
            // Skip if outside plot area horizontally.
            if let Some((px, _py, pw, _ph)) = plot_area {
                if new_x < px || new_x > px + pw {
                    continue;
                }
            }
            // R2: forward the tick's OWN anchor instead of a hardcoded
            // `"center"` — a rotated x tick (label_angle override) is
            // `End`-anchored (see `render/marks/axis.rs`'s Bottom/Top rotation
            // fixup), and re-emitting `"center"` here after zoom/pan would
            // mis-anchor it (root-cause family with the y-branch fix below).
            elements.push(tick_label_json(
                new_x,
                te.y,
                &te.content,
                text_anchor_str(&te.style.anchor),
                Some(&te.style),
            ));
        } else if is_y_tick(te) {
            // Apply sy + ty to the y coordinate; x stays at the axis level.
            let new_y = te.y * transform.sy + transform.ty;
            // Skip if outside plot area vertically.
            if let Some((_px, py, _pw, ph)) = plot_area {
                if new_y < py || new_y > py + ph {
                    continue;
                }
            }
            // R2: forward the tick's OWN anchor instead of a hardcoded
            // `"end"`. Before R2, every y tick WAS `End`(Left)/`Start`(Right)
            // regardless of rotation, so the literal happened to match the
            // common (Left-axis) case; a rotated y tick still carries the
            // same `End`/`Start` anchor (see `render/marks/axis.rs`'s
            // Left/Right arms — rotation does not flip the anchor), but a
            // Right-oriented axis was always mis-anchored here (`"end"` where
            // the real anchor is `Start`) until now.
            elements.push(tick_label_json(
                te.x,
                new_y,
                &te.content,
                text_anchor_str(&te.style.anchor),
                Some(&te.style),
            ));
        } else if is_secondary_y_tick(te) {
            // Right-axis label: reposition its y through THIS axis's composed
            // affine (`panel ∘ its slot rescale`, #52 / criterion 8), so a
            // domainParam/brush bound to only this layer relabels it even when
            // the shared panel affine is identity. The stacked column (x) stays
            // fixed; the label keeps its own anchor (right-axis labels are
            // `start`-anchored, not `end`). The explicit slot tag resolves
            // through `secondary_affine_for_slot` (GH #63: the 1-based
            // `secondary_affines` index translation lives there, not here),
            // which also owns the `.last()` degradation fallback and the
            // panel-affine fallback for when no secondary affines were
            // supplied at all (pure zoom/pan or single-y), byte-identical to
            // the pre-9d panel-affine path.
            let slot = te.slot.unwrap_or(1); // guarded by is_secondary_y_tick (>= 1)
            let aff = secondary_affine_for_slot(secondary_affines, slot, transform);
            let new_y = te.y * aff.sy + aff.ty;
            if let Some((_px, py, _pw, ph)) = plot_area {
                if new_y < py || new_y > py + ph {
                    continue;
                }
            }
            elements.push(tick_label_json(
                te.x,
                new_y,
                &te.content,
                text_anchor_str(&te.style.anchor),
                Some(&te.style),
            ));
        } else {
            elements.push(text_element_to_json(te));
        }
    }

    serde_json::to_string(&elements).unwrap_or_else(|_| "[]".to_string())
}

/// Serialize a single text element to a JSON value.
pub(crate) fn text_element_to_json(t: &TextElementData) -> serde_json::Value {
    serde_json::json!({
        "x": t.x,
        "y": t.y,
        "content": t.content,
        "fontSize": t.style.font_size,
        "fontWeight": font_weight_string(&t.style.font_weight),
        "fontFamily": t.style.font_family,
        "anchor": text_anchor_str(&t.style.anchor),
        "baseline": text_baseline_string(&t.style.baseline),
        "angle": t.style.angle,
        "color": color_string(&t.style),
    })
}

/// Serialize a tick label to a JSON value, optionally inheriting style from a
/// `TextStyle`.  Falls back to sensible defaults when `style` is `None`.
pub(crate) fn tick_label_json(
    x: f64,
    y: f64,
    label: &str,
    anchor: &str,
    style: Option<&ferrum_scene::TextStyle>,
) -> serde_json::Value {
    let (font_size, font_weight, font_family, baseline, angle, color) = match style {
        Some(s) => (
            s.font_size,
            font_weight_string(&s.font_weight),
            s.font_family.clone(),
            text_baseline_string(&s.baseline),
            s.angle,
            color_string(s),
        ),
        None => (
            11.0,
            "normal".to_string(),
            "sans-serif".to_string(),
            "alphabetic".to_string(),
            0.0,
            "rgba(51,51,51,1)".to_string(),
        ),
    };
    serde_json::json!({
        "x": x, "y": y, "content": label,
        "fontSize": font_size, "fontWeight": font_weight,
        "fontFamily": font_family, "anchor": anchor,
        "baseline": baseline, "angle": angle, "color": color,
    })
}

/// Format a scene-graph `TooltipContent` to the same JSON structure as
/// `parse_tooltip_json`: `{"fields":[{"name":"x","value":"1.23"},…]}`.
///
/// Used by `get_tooltip` to serve tooltips from non-packed batches where
/// tooltip data lives in the scene graph rather than a binary sidecar.
#[cfg(any(target_arch = "wasm32", test))]
pub(crate) fn format_tooltip_content(tooltip: &ferrum_scene::TooltipContent) -> String {
    if tooltip.fields.is_empty() {
        return "{}".to_string();
    }
    let fields: Vec<serde_json::Value> = tooltip
        .fields
        .iter()
        .map(|f| {
            serde_json::json!({
                "name": f.name,
                "value": f.value,
            })
        })
        .collect();
    serde_json::json!({ "fields": fields }).to_string()
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
#[cfg(not(target_arch = "wasm32"))]
mod bug_hunt_interactive_slots {
    use super::*;
    use ferrum_scene::{Color, FontWeight, TextAnchor, TextBaseline, TextStyle};

    fn style() -> TextStyle {
        TextStyle {
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
            anchor: TextAnchor::Start,
            baseline: TextBaseline::Alphabetic,
            angle: 0.0,
        }
    }

    fn text(x: f64, y: f64, content: &str) -> TextElementData {
        TextElementData {
            x,
            y,
            content: content.to_string(),
            style: style(),
            slot: None,
        }
    }

    /// A tick-label text node explicitly tagged with its y-scale slot (GH
    /// #60/#73), as `route_y_axis_slotted`/`build_axis` emit for a dual-axis
    /// panel: slot 0 = primary, slot `k >= 1` = the k-th stacked right axis.
    fn text_slot(x: f64, y: f64, content: &str, slot: usize) -> TextElementData {
        TextElementData {
            x,
            y,
            content: content.to_string(),
            style: style(),
            slot: Some(slot),
        }
    }

    fn tick(label: &str, pixel: f64) -> ferrum_scene::Tick {
        ferrum_scene::Tick {
            value: pixel,
            label: label.to_string(),
            pixel,
        }
    }

    fn level(labels: &[(&str, f64)]) -> ferrum_scene::TickLevel {
        ferrum_scene::TickLevel {
            min_zoom: 1.0,
            max_zoom: 10.0,
            ticks: labels.iter().map(|(l, p)| tick(l, *p)).collect(),
        }
    }

    fn interaction(
        y_labels: &[(&str, f64)],
        slot_levels: Vec<Vec<ferrum_scene::TickLevel>>,
    ) -> ferrum_scene::InteractionConfig {
        ferrum_scene::InteractionConfig {
            zoom_enabled: false,
            pan_enabled: false,
            conditionals: vec![],
            linked_panels: vec![],
            toolbar: true,
            params: vec![],
            param_bindings: vec![],
            tick_levels: vec![ferrum_scene::PanelTickLevels {
                panel_id: 0,
                x_levels: vec![],
                y_levels: vec![level(y_labels)],
                y_slot_levels: slot_levels,
            }],
        }
    }

    fn parse(json: &str) -> Vec<serde_json::Value> {
        serde_json::from_str(json).expect("valid JSON")
    }

    fn y_of(parsed: &[serde_json::Value], content: &str) -> f64 {
        parsed
            .iter()
            .find(|v| v["content"] == content)
            .and_then(|v| v["y"].as_f64())
            .unwrap_or_else(|| panic!("{content} must be present"))
    }

    /// A secondary right axis with a SINGLE tick now relabels via its
    /// explicit slot tag (GH #60/#73) rather than the retired `c >= 2`
    /// column-frequency filter, which never recognized a 1-tick column as a
    /// secondary axis — asymmetric with the primary axis, which relabels
    /// fine with 1 tick. A degenerate secondary domain (all values identical)
    /// can legitimately produce one tick, and its axis must not freeze under
    /// a domainParam/brush rescale.
    #[test]
    fn bug_hunt_single_tick_secondary_axis_relabels_under_slot_rescale() {
        let cfg = interaction(&[("L", 100.0)], vec![vec![level(&[("R99", 120.0)])]]);
        let all_text = vec![text(40.0, 100.0, "L"), text_slot(460.0, 120.0, "R99", 1)];
        let panel = crate::zoom_pan::Affine2::identity();
        let secondary = [crate::zoom_pan::compose_panel_slot(
            panel,
            crate::zoom_pan::Affine2 {
                sx: 1.0,
                sy: 2.0,
                tx: 0.0,
                ty: 0.0,
            },
        )];
        let parsed = parse(&build_zoomed_text_json(
            &all_text, &cfg, 0, &panel, &secondary, None,
        ));
        assert_eq!(
            y_of(&parsed, "R99"),
            240.0,
            "single-tick right-axis label must reposition through its slot \
             rescale (120 * 2 = 240) exactly as a multi-tick axis does"
        );
    }

    /// Regression (#73, §7 stray-label guarantee): the retired `c >= 2`
    /// column-frequency heuristic existed to stop a NON-axis text node that
    /// happens to match a right-axis tick string from being repositioned by
    /// the secondary rescale. Slot tagging makes that guarantee structural,
    /// not statistical: an untagged node (`slot: None`) is never an axis
    /// label regardless of its content or column, so even a stray label whose
    /// text and column BOTH collide with a genuine slot-1 tick must stay put
    /// under an active slot rescale. Old code with a `c >= 1` relaxation (the
    /// tempting shortcut) would have relabeled it; this pins that it does not.
    #[test]
    fn untagged_node_matching_a_secondary_tick_string_is_not_rescaled() {
        let cfg = interaction(&[("L", 100.0)], vec![vec![level(&[("R99", 120.0)])]]);
        let all_text = vec![
            text(40.0, 100.0, "L"),
            text_slot(460.0, 120.0, "R99", 1),
            // A stray, UNtagged data label that collides with the slot-1 tick
            // string "R99" and sits in the same right-hand column x=460.
            text(460.0, 300.0, "R99"),
        ];
        let panel = crate::zoom_pan::Affine2::identity();
        let secondary = [crate::zoom_pan::compose_panel_slot(
            panel,
            crate::zoom_pan::Affine2 {
                sx: 1.0,
                sy: 2.0,
                tx: 0.0,
                ty: 0.0,
            },
        )];
        let parsed = parse(&build_zoomed_text_json(
            &all_text, &cfg, 0, &panel, &secondary, None,
        ));
        // The genuine tagged tick rescales (120 * 2 = 240); the untagged stray,
        // sharing content "R99", must remain at its original y=300, never 600.
        let ys: Vec<f64> = parsed
            .iter()
            .filter(|v| v["content"] == "R99")
            .filter_map(|v| v["y"].as_f64())
            .collect();
        assert!(
            ys.contains(&240.0) && ys.contains(&300.0),
            "expected the tagged tick at 240 and the untagged stray held at 300, got {ys:?}"
        );
        assert!(
            !ys.contains(&600.0),
            "untagged stray was wrongly rescaled (300 * 2 = 600): {ys:?}"
        );
    }

    /// Sanity control: the SAME setup with TWO ticks on the right axis
    /// reposition identically to the single-tick case above — proving slot
    /// identity, not tick count, drives recognition.
    #[test]
    fn bug_hunt_two_tick_secondary_axis_relabels_under_slot_rescale() {
        let cfg = interaction(
            &[("L", 100.0)],
            vec![vec![level(&[("R1", 120.0), ("R2", 200.0)])]],
        );
        let all_text = vec![
            text(40.0, 100.0, "L"),
            text_slot(460.0, 120.0, "R1", 1),
            text_slot(460.0, 200.0, "R2", 1),
        ];
        let panel = crate::zoom_pan::Affine2::identity();
        let secondary = [crate::zoom_pan::compose_panel_slot(
            panel,
            crate::zoom_pan::Affine2 {
                sx: 1.0,
                sy: 2.0,
                tx: 0.0,
                ty: 0.0,
            },
        )];
        let parsed = parse(&build_zoomed_text_json(
            &all_text, &cfg, 0, &panel, &secondary, None,
        ));
        assert_eq!(y_of(&parsed, "R1"), 240.0);
        assert_eq!(y_of(&parsed, "R2"), 400.0);
        assert_eq!(
            y_of(&parsed, "L"),
            100.0,
            "primary axis frozen under slot-only rescale"
        );
    }

    /// A slot rescale that pushes right-axis labels outside the plot area must
    /// clip them (same policy as primary ticks) — otherwise rescaled labels
    /// pile up in the margin below/above the panel.
    #[test]
    fn bug_hunt_secondary_labels_clipped_when_slot_rescale_pushes_them_out() {
        let cfg = interaction(
            &[("L", 100.0)],
            vec![vec![level(&[("R1", 100.0), ("R2", 200.0)])]],
        );
        let all_text = vec![
            text(40.0, 100.0, "L"),
            text_slot(460.0, 100.0, "R1", 1),
            text_slot(460.0, 200.0, "R2", 1),
        ];
        let panel = crate::zoom_pan::Affine2::identity();
        // Slot rescale moves everything down by 500 px — outside (0,0,500,400).
        let secondary = [crate::zoom_pan::compose_panel_slot(
            panel,
            crate::zoom_pan::Affine2 {
                sx: 1.0,
                sy: 1.0,
                tx: 0.0,
                ty: 500.0,
            },
        )];
        let parsed = parse(&build_zoomed_text_json(
            &all_text,
            &cfg,
            0,
            &panel,
            &secondary,
            Some((0.0, 0.0, 500.0, 400.0)),
        ));
        let contents: Vec<&str> = parsed
            .iter()
            .filter_map(|v| v["content"].as_str())
            .collect();
        assert!(
            !contents.contains(&"R1"),
            "R1 at y=600 must be clipped (plot y ≤ 400)"
        );
        assert!(!contents.contains(&"R2"), "R2 at y=700 must be clipped");
        assert!(
            contents.contains(&"L"),
            "unmoved primary label must survive"
        );
    }

    /// Two right axes (slot 1, slot 2) but only ONE secondary affine
    /// supplied: slot 2 must fall back to `secondary_affines.last()` — the
    /// documented degradation — not to the panel affine and not panic.
    #[test]
    fn bug_hunt_more_columns_than_affines_falls_back_to_last() {
        let cfg = interaction(
            &[("L", 100.0)],
            vec![
                vec![level(&[("A1", 100.0), ("A2", 200.0)])],
                vec![level(&[("B1", 100.0), ("B2", 200.0)])],
            ],
        );
        let all_text = vec![
            text(40.0, 100.0, "L"),
            text_slot(460.0, 100.0, "A1", 1),
            text_slot(460.0, 200.0, "A2", 1),
            text_slot(500.0, 100.0, "B1", 2),
            text_slot(500.0, 200.0, "B2", 2),
        ];
        let panel = crate::zoom_pan::Affine2::identity();
        let only = crate::zoom_pan::compose_panel_slot(
            panel,
            crate::zoom_pan::Affine2 {
                sx: 1.0,
                sy: 3.0,
                tx: 0.0,
                ty: 0.0,
            },
        );
        let parsed = parse(&build_zoomed_text_json(
            &all_text,
            &cfg,
            0,
            &panel,
            &[only],
            None,
        ));
        // Slot 1 (x=460) uses affines[0]; slot 2 (x=500) has no affines[1] and
        // must degrade to `last()` — the same ×3 rescale.
        assert_eq!(y_of(&parsed, "A1"), 300.0);
        assert_eq!(
            y_of(&parsed, "B1"),
            300.0,
            "slot 2 must fall back to last affine"
        );
        assert_eq!(y_of(&parsed, "B2"), 600.0);
    }

    /// Two right axes (slot 1, slot 2) sharing an identical tick STRING ("5")
    /// must each relabel through their own slot's affine — the explicit slot
    /// tag, not a string-set/column mapping, disambiguates them.
    #[test]
    fn bug_hunt_shared_tick_string_across_two_right_axes_uses_column_position() {
        let cfg = interaction(
            &[("L", 100.0)],
            vec![
                vec![level(&[("5", 100.0), ("7", 200.0)])],
                vec![level(&[("5", 100.0), ("9", 200.0)])],
            ],
        );
        let all_text = vec![
            text(40.0, 100.0, "L"),
            text_slot(460.0, 100.0, "5", 1), // inner axis, slot 1
            text_slot(460.0, 200.0, "7", 1),
            text_slot(500.0, 100.0, "5", 2), // outer axis, slot 2 — same label string
            text_slot(500.0, 200.0, "9", 2),
        ];
        let panel = crate::zoom_pan::Affine2::identity();
        let secondary = [
            crate::zoom_pan::compose_panel_slot(
                panel,
                crate::zoom_pan::Affine2 {
                    sx: 1.0,
                    sy: 2.0,
                    tx: 0.0,
                    ty: 0.0,
                },
            ),
            crate::zoom_pan::compose_panel_slot(
                panel,
                crate::zoom_pan::Affine2 {
                    sx: 1.0,
                    sy: 1.0,
                    tx: 0.0,
                    ty: 50.0,
                },
            ),
        ];
        let parsed = parse(&build_zoomed_text_json(
            &all_text, &cfg, 0, &panel, &secondary, None,
        ));
        let ys_of_5: Vec<f64> = parsed
            .iter()
            .filter(|v| v["content"] == "5")
            .filter_map(|v| v["y"].as_f64())
            .collect();
        assert_eq!(ys_of_5.len(), 2, "both '5' labels must be emitted");
        // Inner (x=460): 100 * 2 = 200. Outer (x=500): 100 + 50 = 150.
        assert!(
            ys_of_5.contains(&200.0) && ys_of_5.contains(&150.0),
            "each '5' must move through its OWN column's affine, got {ys_of_5:?}"
        );
    }

    /// Empty `y_slot_levels` inner vec (a slot with no tick levels at all —
    /// e.g. an ordinal secondary axis) must not panic and must leave
    /// right-side labels untouched.
    #[test]
    fn bug_hunt_empty_slot_level_vec_is_safe() {
        let cfg = interaction(&[("L", 100.0)], vec![vec![]]);
        let all_text = vec![text(40.0, 100.0, "L"), text(460.0, 100.0, "R1")];
        let panel = crate::zoom_pan::Affine2 {
            sx: 1.0,
            sy: 2.0,
            tx: 0.0,
            ty: 0.0,
        };
        let parsed = parse(&build_zoomed_text_json(
            &all_text,
            &cfg,
            0,
            &panel,
            &[],
            None,
        ));
        assert_eq!(y_of(&parsed, "L"), 200.0, "primary still relabels");
        assert_eq!(
            y_of(&parsed, "R1"),
            100.0,
            "unrecognized right label stays put"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_scene::{Color, FontWeight, TextAnchor, TextBaseline, TextStyle};

    fn make_style() -> TextStyle {
        TextStyle {
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
        }
    }

    fn make_text(x: f64, y: f64, content: &str) -> TextElementData {
        TextElementData {
            x,
            y,
            content: content.to_string(),
            style: make_style(),
            slot: None,
        }
    }

    /// A tick-label text node explicitly tagged with its y-scale slot (GH
    /// #60/#73): slot 0 = primary, slot `k >= 1` = the k-th stacked right axis.
    fn make_text_slot(x: f64, y: f64, content: &str, slot: usize) -> TextElementData {
        TextElementData {
            x,
            y,
            content: content.to_string(),
            style: make_style(),
            slot: Some(slot),
        }
    }

    // ── font_weight_string / text_anchor_str / text_baseline_string ────

    #[test]
    fn font_weight_string_normal_bold() {
        assert_eq!(font_weight_string(&FontWeight::Normal), "normal");
        assert_eq!(font_weight_string(&FontWeight::Bold), "bold");
    }

    #[test]
    fn font_weight_string_custom() {
        let s = font_weight_string(&FontWeight::Custom("600".into()));
        assert_eq!(s, "600");
    }

    #[test]
    fn text_anchor_str_variants() {
        assert_eq!(text_anchor_str(&TextAnchor::Start), "start");
        assert_eq!(text_anchor_str(&TextAnchor::Middle), "center");
        assert_eq!(text_anchor_str(&TextAnchor::End), "end");
    }

    #[test]
    fn text_baseline_string_variants() {
        assert_eq!(text_baseline_string(&TextBaseline::Top), "top");
        assert_eq!(text_baseline_string(&TextBaseline::Middle), "middle");
        assert_eq!(text_baseline_string(&TextBaseline::Bottom), "bottom");
        assert_eq!(
            text_baseline_string(&TextBaseline::Alphabetic),
            "alphabetic"
        );
        assert_eq!(
            text_baseline_string(&TextBaseline::Custom("hanging".into())),
            "hanging"
        );
    }

    // ── B7: zoomed tick labels clipped to plot area ─────────────────────

    #[test]
    fn test_zoomed_tick_labels_clipped_to_plot_area() {
        // Simulate a zoom that pushes some x-tick labels outside the plot area.
        //
        // Setup: plot_area is (50, 50, 400, 300) — x ranges from 50 to 450.
        // Two x-tick labels at x=100 and x=400, both at y=360 (below the plot area).
        // Zoom: sx=2.0, tx=-200.0
        //   - Label at x=100 transforms to: 100*2 - 200 = 0 (outside left edge at 50)
        //   - Label at x=400 transforms to: 400*2 - 200 = 600 (outside right edge at 450)
        //
        // The y-tick label at x=40, y=200 transforms y to: 200*1.0 + 0.0 = 200 (inside).

        let interaction = ferrum_scene::InteractionConfig {
            zoom_enabled: false,
            pan_enabled: false,
            conditionals: vec![],
            linked_panels: vec![],
            toolbar: true,
            params: vec![],
            param_bindings: vec![],
            tick_levels: vec![ferrum_scene::PanelTickLevels {
                panel_id: 0,
                x_levels: vec![ferrum_scene::TickLevel {
                    min_zoom: 1.0,
                    max_zoom: 10.0,
                    ticks: vec![
                        ferrum_scene::Tick {
                            value: 100.0,
                            label: "100".to_string(),
                            pixel: 100.0,
                        },
                        ferrum_scene::Tick {
                            value: 400.0,
                            label: "400".to_string(),
                            pixel: 400.0,
                        },
                    ],
                }],
                y_levels: vec![ferrum_scene::TickLevel {
                    min_zoom: 1.0,
                    max_zoom: 10.0,
                    ticks: vec![ferrum_scene::Tick {
                        value: 200.0,
                        label: "200".to_string(),
                        pixel: 200.0,
                    }],
                }],
                y_slot_levels: vec![],
            }],
        };

        let all_text = vec![
            make_text(100.0, 360.0, "100"),  // x-tick label
            make_text(400.0, 360.0, "400"),  // x-tick label
            make_text(40.0, 200.0, "200"),   // y-tick label
            make_text(250.0, 20.0, "Title"), // non-tick (title)
        ];

        let transform = crate::zoom_pan::Affine2 {
            sx: 2.0,
            sy: 1.0,
            tx: -200.0,
            ty: 0.0,
        };

        let plot_area = Some((50.0, 50.0, 400.0, 300.0));

        let json_str =
            build_zoomed_text_json(&all_text, &interaction, 0, &transform, &[], plot_area);
        let parsed: Vec<serde_json::Value> = serde_json::from_str(&json_str).expect("valid JSON");

        // The two x-tick labels should be clipped out (transformed to 0 and 600,
        // both outside plot_area x=[50, 450]).
        let contents: Vec<&str> = parsed
            .iter()
            .filter_map(|v| v["content"].as_str())
            .collect();

        assert!(
            !contents.contains(&"100"),
            "x-tick label '100' at transformed x=0 should be clipped (outside plot_area x=50..450)"
        );
        assert!(
            !contents.contains(&"400"),
            "x-tick label '400' at transformed x=600 should be clipped (outside plot_area x=50..450)"
        );

        // The y-tick label and title should still be present.
        assert!(
            contents.contains(&"200"),
            "y-tick label '200' at transformed y=200 should be kept (inside plot_area y=50..350)"
        );
        assert!(
            contents.contains(&"Title"),
            "non-tick text 'Title' should always be kept"
        );
    }

    #[test]
    fn test_zoomed_tick_labels_inside_plot_area_kept() {
        // All tick labels transform to positions inside the plot area.
        let interaction = ferrum_scene::InteractionConfig {
            zoom_enabled: false,
            pan_enabled: false,
            conditionals: vec![],
            linked_panels: vec![],
            toolbar: true,
            params: vec![],
            param_bindings: vec![],
            tick_levels: vec![ferrum_scene::PanelTickLevels {
                panel_id: 0,
                x_levels: vec![ferrum_scene::TickLevel {
                    min_zoom: 1.0,
                    max_zoom: 10.0,
                    ticks: vec![
                        ferrum_scene::Tick {
                            value: 100.0,
                            label: "A".to_string(),
                            pixel: 100.0,
                        },
                        ferrum_scene::Tick {
                            value: 200.0,
                            label: "B".to_string(),
                            pixel: 200.0,
                        },
                    ],
                }],
                y_levels: vec![],
                y_slot_levels: vec![],
            }],
        };

        let all_text = vec![
            make_text(100.0, 360.0, "A"), // x-tick
            make_text(200.0, 360.0, "B"), // x-tick
        ];

        // Identity-ish transform that keeps labels inside.
        let transform = crate::zoom_pan::Affine2 {
            sx: 1.0,
            sy: 1.0,
            tx: 0.0,
            ty: 0.0,
        };

        let plot_area = Some((0.0, 0.0, 500.0, 500.0));

        let json_str =
            build_zoomed_text_json(&all_text, &interaction, 0, &transform, &[], plot_area);
        let parsed: Vec<serde_json::Value> = serde_json::from_str(&json_str).expect("valid JSON");

        let contents: Vec<&str> = parsed
            .iter()
            .filter_map(|v| v["content"].as_str())
            .collect();

        assert!(contents.contains(&"A"), "label 'A' should be kept");
        assert!(contents.contains(&"B"), "label 'B' should be kept");
    }

    /// R2: `build_zoomed_text_json`'s x-tick branch must forward the tick's
    /// OWN anchor (via `text_anchor_str`), not the hardcoded `"center"` that
    /// pre-R2 code always emitted. A rotated x tick (`label_angle` override)
    /// is `End`-anchored — re-emitting `"center"` after zoom/pan would
    /// mis-anchor it even though the pre-zoom SVG got it right.
    #[test]
    fn zoomed_x_tick_forwards_its_own_anchor_not_hardcoded_center() {
        let interaction = ferrum_scene::InteractionConfig {
            zoom_enabled: false,
            pan_enabled: false,
            conditionals: vec![],
            linked_panels: vec![],
            toolbar: true,
            params: vec![],
            param_bindings: vec![],
            tick_levels: vec![ferrum_scene::PanelTickLevels {
                panel_id: 0,
                x_levels: vec![ferrum_scene::TickLevel {
                    min_zoom: 1.0,
                    max_zoom: 10.0,
                    ticks: vec![ferrum_scene::Tick {
                        value: 100.0,
                        label: "Rotated".to_string(),
                        pixel: 100.0,
                    }],
                }],
                y_levels: vec![],
                y_slot_levels: vec![],
            }],
        };
        // Simulate a rotated (label_angle override) x tick: End-anchored,
        // non-zero angle — exactly what `render/marks/axis.rs`'s Bottom
        // rotation fixup produces.
        let rotated_style = TextStyle {
            anchor: TextAnchor::End,
            angle: -45.0,
            ..make_style()
        };
        let all_text = vec![TextElementData {
            x: 100.0,
            y: 360.0,
            content: "Rotated".to_string(),
            style: rotated_style,
            slot: None,
        }];
        let transform = crate::zoom_pan::Affine2 {
            sx: 1.0,
            sy: 1.0,
            tx: 0.0,
            ty: 0.0,
        };
        let json_str = build_zoomed_text_json(&all_text, &interaction, 0, &transform, &[], None);
        let parsed: Vec<serde_json::Value> = serde_json::from_str(&json_str).expect("valid JSON");
        assert_eq!(
            parsed[0]["anchor"], "end",
            "rotated x tick must forward its own End anchor, not a hardcoded center"
        );
    }

    /// R2: `build_zoomed_text_json`'s y-tick branch must forward the tick's
    /// OWN anchor, not the hardcoded `"end"` that pre-R2 code always emitted
    /// (correct for the common Left-oriented y-axis, but wrong for a
    /// Right-oriented primary y-axis, whose ticks are Start-anchored).
    #[test]
    fn zoomed_y_tick_forwards_its_own_anchor_not_hardcoded_end() {
        let interaction = ferrum_scene::InteractionConfig {
            zoom_enabled: false,
            pan_enabled: false,
            conditionals: vec![],
            linked_panels: vec![],
            toolbar: true,
            params: vec![],
            param_bindings: vec![],
            tick_levels: vec![ferrum_scene::PanelTickLevels {
                panel_id: 0,
                x_levels: vec![],
                y_levels: vec![ferrum_scene::TickLevel {
                    min_zoom: 1.0,
                    max_zoom: 10.0,
                    ticks: vec![ferrum_scene::Tick {
                        value: 10.0,
                        label: "R".to_string(),
                        pixel: 100.0,
                    }],
                }],
                y_slot_levels: vec![],
            }],
        };
        // Simulate a Right-oriented primary y-axis tick: Start-anchored.
        let right_style = TextStyle {
            anchor: TextAnchor::Start,
            ..make_style()
        };
        let all_text = vec![TextElementData {
            x: 460.0,
            y: 100.0,
            content: "R".to_string(),
            style: right_style,
            slot: None,
        }];
        let transform = crate::zoom_pan::Affine2 {
            sx: 1.0,
            sy: 2.0,
            tx: 0.0,
            ty: 0.0,
        };
        let json_str = build_zoomed_text_json(&all_text, &interaction, 0, &transform, &[], None);
        let parsed: Vec<serde_json::Value> = serde_json::from_str(&json_str).expect("valid JSON");
        assert_eq!(
            parsed[0]["anchor"], "start",
            "Right-oriented y tick must forward its own Start anchor, not a hardcoded end"
        );
    }

    /// Regression guard: the default (flat Bottom x-tick / flat Left y-tick)
    /// anchors are byte-identical to the pre-R2 hardcoded literals — `"center"`
    /// for x, `"end"` for y — since `make_style()`'s default anchor (`Middle`
    /// for the shared style used by x ticks below) and a Left-axis y tick's
    /// `End` anchor are exactly what the old hardcoded strings encoded.
    #[test]
    fn zoomed_tick_anchor_forwarding_is_byte_identical_for_default_orients() {
        let interaction = ferrum_scene::InteractionConfig {
            zoom_enabled: false,
            pan_enabled: false,
            conditionals: vec![],
            linked_panels: vec![],
            toolbar: true,
            params: vec![],
            param_bindings: vec![],
            tick_levels: vec![ferrum_scene::PanelTickLevels {
                panel_id: 0,
                x_levels: vec![ferrum_scene::TickLevel {
                    min_zoom: 1.0,
                    max_zoom: 10.0,
                    ticks: vec![ferrum_scene::Tick {
                        value: 100.0,
                        label: "X".to_string(),
                        pixel: 100.0,
                    }],
                }],
                y_levels: vec![ferrum_scene::TickLevel {
                    min_zoom: 1.0,
                    max_zoom: 10.0,
                    ticks: vec![ferrum_scene::Tick {
                        value: 10.0,
                        label: "Y".to_string(),
                        pixel: 200.0,
                    }],
                }],
                y_slot_levels: vec![],
            }],
        };
        let x_style = TextStyle {
            anchor: TextAnchor::Middle,
            ..make_style()
        };
        let y_style = TextStyle {
            anchor: TextAnchor::End,
            ..make_style()
        };
        let all_text = vec![
            TextElementData {
                x: 100.0,
                y: 360.0,
                content: "X".to_string(),
                style: x_style,
                slot: None,
            },
            TextElementData {
                x: 40.0,
                y: 200.0,
                content: "Y".to_string(),
                style: y_style,
                slot: None,
            },
        ];
        let transform = crate::zoom_pan::Affine2 {
            sx: 1.0,
            sy: 1.0,
            tx: 0.0,
            ty: 0.0,
        };
        let json_str = build_zoomed_text_json(&all_text, &interaction, 0, &transform, &[], None);
        let parsed: Vec<serde_json::Value> = serde_json::from_str(&json_str).expect("valid JSON");
        let anchor_of = |content: &str| -> String {
            parsed
                .iter()
                .find(|v| v["content"] == content)
                .and_then(|v| v["anchor"].as_str())
                .unwrap()
                .to_string()
        };
        assert_eq!(
            anchor_of("X"),
            "center",
            "default x-tick anchor must stay byte-identical"
        );
        assert_eq!(
            anchor_of("Y"),
            "end",
            "default y-tick anchor must stay byte-identical"
        );
    }

    #[test]
    fn secondary_y_tick_labels_reposition_under_panel_affine() {
        // Secondary-y (#52): right-axis labels (in `y_slot_levels`, sitting at
        // their own right-side column) reposition their y under the panel affine
        // exactly as the left axis does — criterion 7.
        let interaction = ferrum_scene::InteractionConfig {
            zoom_enabled: false,
            pan_enabled: false,
            conditionals: vec![],
            linked_panels: vec![],
            toolbar: true,
            params: vec![],
            param_bindings: vec![],
            tick_levels: vec![ferrum_scene::PanelTickLevels {
                panel_id: 0,
                x_levels: vec![],
                // Primary (left) axis tick.
                y_levels: vec![ferrum_scene::TickLevel {
                    min_zoom: 1.0,
                    max_zoom: 10.0,
                    ticks: vec![ferrum_scene::Tick {
                        value: 10.0,
                        label: "L".to_string(),
                        pixel: 100.0,
                    }],
                }],
                // One right axis with two ticks.
                y_slot_levels: vec![vec![ferrum_scene::TickLevel {
                    min_zoom: 1.0,
                    max_zoom: 10.0,
                    ticks: vec![
                        ferrum_scene::Tick {
                            value: 1.0,
                            label: "R1".to_string(),
                            pixel: 100.0,
                        },
                        ferrum_scene::Tick {
                            value: 2.0,
                            label: "R2".to_string(),
                            pixel: 200.0,
                        },
                    ],
                }]],
            }],
        };

        let all_text = vec![
            make_text(40.0, 100.0, "L"),           // left-axis tick (column x=40)
            make_text_slot(460.0, 100.0, "R1", 1), // right-axis tick, slot 1 (x=460)
            make_text_slot(460.0, 200.0, "R2", 1), // right-axis tick, slot 1 (x=460)
        ];

        // Panel zoom: sy=2.0 moves every y-position; both axes track together.
        let transform = crate::zoom_pan::Affine2 {
            sx: 1.0,
            sy: 2.0,
            tx: 0.0,
            ty: 0.0,
        };
        let plot_area = Some((0.0, 0.0, 500.0, 500.0));

        // Regression (criterion 8, case 2): under a pure panel zoom every
        // right axis composes to the panel affine (identity slot rescale), so
        // all columns relabel exactly as the primary axis does — the pre-9d
        // behavior. Pass the composed affine to exercise the real path.
        let secondary = [crate::zoom_pan::compose_panel_slot(
            transform,
            crate::zoom_pan::Affine2::identity(),
        )];
        let json_str = build_zoomed_text_json(
            &all_text,
            &interaction,
            0,
            &transform,
            &secondary,
            plot_area,
        );
        let parsed: Vec<serde_json::Value> = serde_json::from_str(&json_str).expect("valid JSON");

        let y_of = |content: &str| -> f64 {
            parsed
                .iter()
                .find(|v| v["content"] == content)
                .and_then(|v| v["y"].as_f64())
                .unwrap_or_else(|| panic!("{content} present"))
        };
        // Left axis relabels through the panel affine: 100 * 2 = 200.
        assert_eq!(y_of("L"), 200.0);
        // Right axis relabels the SAME way (panel-level zoom): 100*2, 200*2.
        assert_eq!(y_of("R1"), 200.0);
        assert_eq!(y_of("R2"), 400.0);
        // The right column x stays fixed (only y moves under zoom/pan).
        let x_of = |content: &str| -> f64 {
            parsed
                .iter()
                .find(|v| v["content"] == content)
                .and_then(|v| v["x"].as_f64())
                .unwrap()
        };
        assert_eq!(x_of("R1"), 460.0);
    }

    #[test]
    fn single_axis_chart_ignores_secondary_slot_path() {
        // Byte-stability: with `y_slot_levels` empty, a right-column label that
        // happens to match no primary tick string is left untouched (emitted at
        // its original position), identical to pre-#52 behavior.
        let interaction = ferrum_scene::InteractionConfig {
            zoom_enabled: false,
            pan_enabled: false,
            conditionals: vec![],
            linked_panels: vec![],
            toolbar: true,
            params: vec![],
            param_bindings: vec![],
            tick_levels: vec![ferrum_scene::PanelTickLevels {
                panel_id: 0,
                x_levels: vec![],
                y_levels: vec![ferrum_scene::TickLevel {
                    min_zoom: 1.0,
                    max_zoom: 10.0,
                    ticks: vec![ferrum_scene::Tick {
                        value: 10.0,
                        label: "L".to_string(),
                        pixel: 100.0,
                    }],
                }],
                y_slot_levels: vec![],
            }],
        };
        let all_text = vec![make_text(40.0, 100.0, "L"), make_text(460.0, 100.0, "R1")];
        let transform = crate::zoom_pan::Affine2 {
            sx: 1.0,
            sy: 2.0,
            tx: 0.0,
            ty: 0.0,
        };
        let json_str = build_zoomed_text_json(
            &all_text,
            &interaction,
            0,
            &transform,
            &[],
            Some((0.0, 0.0, 500.0, 500.0)),
        );
        let parsed: Vec<serde_json::Value> = serde_json::from_str(&json_str).expect("valid JSON");
        let y_of = |content: &str| -> f64 {
            parsed
                .iter()
                .find(|v| v["content"] == content)
                .and_then(|v| v["y"].as_f64())
                .unwrap()
        };
        // Left axis still relabels; the unrelated right label stays put.
        assert_eq!(y_of("L"), 200.0);
        assert_eq!(y_of("R1"), 100.0);
    }

    /// Shared builder for the two-right-axis dual-axis fixture the criterion-8
    /// tests below exercise: left axis "L" at column x=40, an inner right axis
    /// (slot 1) at x=460 with ticks A1/A2, an outer right axis (slot 2) at x=500
    /// with ticks B1/B2, and one x-tick "X".
    fn dual_axis_fixture() -> (ferrum_scene::InteractionConfig, Vec<TextElementData>) {
        let interaction = ferrum_scene::InteractionConfig {
            zoom_enabled: false,
            pan_enabled: false,
            conditionals: vec![],
            linked_panels: vec![],
            toolbar: true,
            params: vec![],
            param_bindings: vec![],
            tick_levels: vec![ferrum_scene::PanelTickLevels {
                panel_id: 0,
                x_levels: vec![ferrum_scene::TickLevel {
                    min_zoom: 1.0,
                    max_zoom: 10.0,
                    ticks: vec![ferrum_scene::Tick {
                        value: 1.0,
                        label: "X".to_string(),
                        pixel: 100.0,
                    }],
                }],
                y_levels: vec![ferrum_scene::TickLevel {
                    min_zoom: 1.0,
                    max_zoom: 10.0,
                    ticks: vec![ferrum_scene::Tick {
                        value: 10.0,
                        label: "L".to_string(),
                        pixel: 100.0,
                    }],
                }],
                // Two right axes in slot order: inner (slot 1) then outer (slot 2).
                y_slot_levels: vec![
                    vec![ferrum_scene::TickLevel {
                        min_zoom: 1.0,
                        max_zoom: 10.0,
                        ticks: vec![
                            ferrum_scene::Tick {
                                value: 1.0,
                                label: "A1".to_string(),
                                pixel: 100.0,
                            },
                            ferrum_scene::Tick {
                                value: 2.0,
                                label: "A2".to_string(),
                                pixel: 200.0,
                            },
                        ],
                    }],
                    vec![ferrum_scene::TickLevel {
                        min_zoom: 1.0,
                        max_zoom: 10.0,
                        ticks: vec![
                            ferrum_scene::Tick {
                                value: 1.0,
                                label: "B1".to_string(),
                                pixel: 100.0,
                            },
                            ferrum_scene::Tick {
                                value: 2.0,
                                label: "B2".to_string(),
                                pixel: 200.0,
                            },
                        ],
                    }],
                ],
            }],
        };
        let all_text = vec![
            make_text(40.0, 100.0, "L"),           // left axis (column x=40)
            make_text(100.0, 360.0, "X"),          // x-axis tick (row y=360)
            make_text_slot(460.0, 100.0, "A1", 1), // inner right axis, slot 1 (x=460)
            make_text_slot(460.0, 200.0, "A2", 1),
            make_text_slot(500.0, 100.0, "B1", 2), // outer right axis, slot 2 (x=500)
            make_text_slot(500.0, 200.0, "B2", 2),
        ];
        (interaction, all_text)
    }

    #[test]
    fn secondary_slot_only_rescale_moves_secondary_not_primary() {
        // Criterion 8, case 1: a domainParam/brush bound to ONE right-axis layer
        // writes a y-only slot rescale while the shared panel affine stays
        // identity. Only that layer's axis labels move; the primary (left) axis
        // and the x axis are untouched. The two right axes carry DIFFERENT
        // rescales, discriminating the per-column slot mapping.
        let (interaction, all_text) = dual_axis_fixture();
        let panel = crate::zoom_pan::Affine2::identity();
        // Slot 1 (inner, x=460): sy=2, ty=0. Slot 2 (outer, x=500): sy=1, ty=50.
        let secondary = [
            crate::zoom_pan::compose_panel_slot(
                panel,
                crate::zoom_pan::Affine2 {
                    sx: 1.0,
                    sy: 2.0,
                    tx: 0.0,
                    ty: 0.0,
                },
            ),
            crate::zoom_pan::compose_panel_slot(
                panel,
                crate::zoom_pan::Affine2 {
                    sx: 1.0,
                    sy: 1.0,
                    tx: 0.0,
                    ty: 50.0,
                },
            ),
        ];
        let json = build_zoomed_text_json(&all_text, &interaction, 0, &panel, &secondary, None);
        let parsed: Vec<serde_json::Value> = serde_json::from_str(&json).expect("valid JSON");
        let pos = |content: &str| -> (f64, f64) {
            let v = parsed
                .iter()
                .find(|v| v["content"] == content)
                .unwrap_or_else(|| panic!("{content} present"));
            (v["x"].as_f64().unwrap(), v["y"].as_f64().unwrap())
        };
        // Primary axis and x axis do NOT move (panel affine is identity).
        assert_eq!(
            pos("L"),
            (40.0, 100.0),
            "left axis frozen under slot-only rescale"
        );
        assert_eq!(
            pos("X"),
            (100.0, 360.0),
            "x axis frozen under slot-only rescale"
        );
        // Inner right axis (slot 1) relabels: 100*2=200, 200*2=400.
        assert_eq!(pos("A1"), (460.0, 200.0));
        assert_eq!(pos("A2"), (460.0, 400.0));
        // Outer right axis (slot 2) relabels with its OWN rescale: +50.
        assert_eq!(pos("B1"), (500.0, 150.0));
        assert_eq!(pos("B2"), (500.0, 250.0));
    }

    #[test]
    fn secondary_slot_rescale_and_panel_zoom_compose() {
        // Criterion 8, case 3: a panel zoom AND a per-slot rescale compose. The
        // inner right axis relabels through `panel ∘ slot`, the outer through
        // the panel affine alone (identity slot), and the primary axes through
        // the panel affine — proving the composition is per column.
        let (interaction, all_text) = dual_axis_fixture();
        let panel = crate::zoom_pan::Affine2 {
            sx: 1.0,
            sy: 2.0,
            tx: 0.0,
            ty: 10.0,
        };
        let slot1 = crate::zoom_pan::Affine2 {
            sx: 1.0,
            sy: 3.0,
            tx: 0.0,
            ty: 5.0,
        };
        let secondary = [
            crate::zoom_pan::compose_panel_slot(panel, slot1),
            crate::zoom_pan::compose_panel_slot(panel, crate::zoom_pan::Affine2::identity()),
        ];
        let json = build_zoomed_text_json(&all_text, &interaction, 0, &panel, &secondary, None);
        let parsed: Vec<serde_json::Value> = serde_json::from_str(&json).expect("valid JSON");
        let y_of = |content: &str| -> f64 {
            parsed
                .iter()
                .find(|v| v["content"] == content)
                .and_then(|v| v["y"].as_f64())
                .unwrap_or_else(|| panic!("{content} present"))
        };
        // Primary axis relabels through the panel affine: 100*2+10 = 210.
        assert_eq!(y_of("L"), 210.0);
        // Inner right axis (slot 1) composes: sy=2*3=6, ty=2*5+10=20 → 100*6+20.
        assert_eq!(y_of("A1"), 620.0);
        assert_eq!(y_of("A2"), 1220.0);
        // Outer right axis (slot 2, identity slot) tracks the panel affine only.
        assert_eq!(y_of("B1"), 210.0);
        assert_eq!(y_of("B2"), 410.0);
    }

    #[test]
    fn test_zoomed_text_no_plot_area_keeps_all() {
        // When plot_area is None, no clipping should occur.
        let interaction = ferrum_scene::InteractionConfig {
            zoom_enabled: false,
            pan_enabled: false,
            conditionals: vec![],
            linked_panels: vec![],
            toolbar: true,
            params: vec![],
            param_bindings: vec![],
            tick_levels: vec![ferrum_scene::PanelTickLevels {
                panel_id: 0,
                x_levels: vec![ferrum_scene::TickLevel {
                    min_zoom: 1.0,
                    max_zoom: 10.0,
                    ticks: vec![ferrum_scene::Tick {
                        value: 100.0,
                        label: "X".to_string(),
                        pixel: 100.0,
                    }],
                }],
                y_levels: vec![],
                y_slot_levels: vec![],
            }],
        };

        let all_text = vec![make_text(100.0, 360.0, "X")];

        // Transform that would push label far off-screen.
        let transform = crate::zoom_pan::Affine2 {
            sx: 10.0,
            sy: 1.0,
            tx: -5000.0,
            ty: 0.0,
        };

        let json_str = build_zoomed_text_json(&all_text, &interaction, 0, &transform, &[], None);
        let parsed: Vec<serde_json::Value> = serde_json::from_str(&json_str).expect("valid JSON");

        let contents: Vec<&str> = parsed
            .iter()
            .filter_map(|v| v["content"].as_str())
            .collect();

        assert!(
            contents.contains(&"X"),
            "without plot_area, no clipping should occur"
        );
    }

    // ── bug_hunt: text_json edge cases ──────────────────────────────────

    #[test]
    fn bug_hunt_build_text_json_from_empty_slice() {
        // Empty text elements must produce an empty JSON array "[]".
        let result = build_text_json_from(&[]);
        assert_eq!(result, "[]", "empty text elements must produce '[]'");
    }

    #[test]
    fn bug_hunt_text_element_with_special_chars_in_content() {
        // Quotes, newlines, and backslashes in text content must be valid JSON.
        let te = TextElementData {
            x: 100.0,
            y: 200.0,
            content: r#"say "hello" \ world"#.to_string(),
            style: make_style(),
            slot: None,
        };
        let json_str = build_text_json_from(&[te]);
        let parsed: Vec<serde_json::Value> = serde_json::from_str(&json_str)
            .expect("special chars in content must produce valid JSON");
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0]["content"], r#"say "hello" \ world"#);
    }

    #[test]
    fn bug_hunt_text_element_with_nan_coordinates() {
        // NaN coordinates must not crash serialization.
        let te = TextElementData {
            x: f64::NAN,
            y: f64::NAN,
            content: "NaN test".to_string(),
            style: make_style(),
            slot: None,
        };
        let json_str = build_text_json_from(&[te]);
        // serde_json serializes NaN as null.
        let parsed: Vec<serde_json::Value> =
            serde_json::from_str(&json_str).expect("NaN coordinates must produce valid JSON");
        assert_eq!(parsed.len(), 1);
        assert!(parsed[0]["x"].is_null(), "NaN x must serialize as null");
    }

    #[test]
    fn bug_hunt_text_element_with_inf_coordinates() {
        // Infinity coordinates must not crash serialization.
        let te = TextElementData {
            x: f64::INFINITY,
            y: f64::NEG_INFINITY,
            content: "inf test".to_string(),
            style: make_style(),
            slot: None,
        };
        let json_str = build_text_json_from(&[te]);
        let parsed: Vec<serde_json::Value> =
            serde_json::from_str(&json_str).expect("Infinity coordinates must produce valid JSON");
        assert_eq!(parsed.len(), 1);
    }

    #[test]
    fn bug_hunt_text_element_empty_content() {
        // Empty content string must produce valid JSON.
        let te = make_text(50.0, 50.0, "");
        let json_str = build_text_json_from(&[te]);
        let parsed: Vec<serde_json::Value> =
            serde_json::from_str(&json_str).expect("empty content must produce valid JSON");
        assert_eq!(parsed[0]["content"], "");
    }

    #[test]
    fn bug_hunt_text_element_unicode_content() {
        // Unicode (emoji, CJK) in content must serialize correctly.
        let te = make_text(10.0, 20.0, "\u{1F600}\u{4E16}\u{754C}");
        let json_str = build_text_json_from(&[te]);
        let parsed: Vec<serde_json::Value> =
            serde_json::from_str(&json_str).expect("unicode content must produce valid JSON");
        assert_eq!(parsed[0]["content"], "\u{1F600}\u{4E16}\u{754C}");
    }

    #[test]
    fn bug_hunt_format_tooltip_content_with_unicode() {
        // Unicode in tooltip name/value must produce valid JSON.
        use ferrum_scene::{TooltipContent, TooltipField};
        let tooltip = TooltipContent {
            fields: vec![TooltipField {
                name: "\u{1F4CA}chart".to_string(),
                value: "\u{2714}pass".to_string(),
            }],
        };
        let json = format_tooltip_content(&tooltip);
        let parsed: serde_json::Value =
            serde_json::from_str(&json).expect("unicode in tooltip must produce valid JSON");
        assert_eq!(parsed["fields"][0]["name"], "\u{1F4CA}chart");
    }

    #[test]
    fn bug_hunt_tick_label_json_with_empty_label() {
        // Empty label string must produce valid JSON.
        let json = tick_label_json(0.0, 0.0, "", "center", None);
        assert_eq!(json["content"], "");
    }

    #[test]
    fn bug_hunt_color_string_with_fractional_opacity() {
        // color_string must produce valid rgba() with fractional opacity.
        let style = TextStyle {
            font_size: 12.0,
            font_weight: FontWeight::Normal,
            font_family: "sans-serif".to_string(),
            color: Color {
                r: 100,
                g: 200,
                b: 50,
                a: 255,
            },
            opacity: 0.5,
            anchor: TextAnchor::Middle,
            baseline: TextBaseline::Alphabetic,
            angle: 0.0,
        };
        let result = color_string(&style);
        assert!(result.starts_with("rgba("), "must start with rgba(");
        assert!(result.contains("0.5"), "must contain opacity 0.5");
    }

    #[test]
    fn bug_hunt_zoomed_text_panel_id_mismatch_falls_back() {
        // When panel_id doesn't match any tick_level entry, must fall back
        // to the non-zoomed text JSON (all elements at original positions).
        let interaction = ferrum_scene::InteractionConfig::default();
        let texts = vec![make_text(100.0, 200.0, "hello")];
        let transform = crate::zoom_pan::Affine2 {
            sx: 2.0,
            sy: 2.0,
            tx: 0.0,
            ty: 0.0,
        };
        let result = build_zoomed_text_json(&texts, &interaction, 99, &transform, &[], None);
        let parsed: Vec<serde_json::Value> = serde_json::from_str(&result).unwrap();
        // Must contain all text elements (no zoom-specific filtering).
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0]["content"], "hello");
    }
}
