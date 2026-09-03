//! Chart-config application test coverage.
//!
//! This module carries `chart_config_application_tests`, relocated out of
//! `render/mod.rs` alongside the `config_apply` extraction (#143), together
//! with its subject: every `configure_*()` surface, the precedence orderings
//! between them, and the tier fns that own those orderings.
//!
//! It could not stay inline: the relocated body is the larger half of the
//! module, so keeping it in `mod.rs` would make the pipeline itself the
//! minority of the file it lives in — the promotion trigger in CLAUDE.md's
//! Rust test-module convention.
//!
//! What the relocation changed, precisely: every test kept its name and its
//! assertions, and the mod kept its own inner `//!` docstring. Four bodies are
//! not byte-identical to what `render/mod.rs` carried, all deliberately:
//! `axis_x_wins_on_x_without_disturbing_the_shared_theme_fallback`,
//! `axis_x_tick_size_lands_on_x_only` and
//! `axis_x_styling_field_wins_over_axis_via_fill_none_ordering` were rewired to
//! assert THROUGH `fill_axis_slots_specific_before_shared` instead of
//! hand-sequencing the four `apply_axis_config_to_axis_input` calls (so a
//! production reorder now fails them), and
//! `chart_level_orient_none_suppresses_without_the_python_conversion` was
//! extended past the `chart_config_legend_disabled` predicate to the clear it
//! gates. Nothing was dropped or merged. The `#[cfg(test)]` attribute on the
//! mod itself is gone, redundant under this file's `#[cfg(test)] mod tests;`
//! gate.
//!
//! `axis_style_fill_from_tests` is NOT here: the #143 remediation moved its
//! subject to `prepare` (breaking the `prepare` → `config_apply` back-edge),
//! and the characterization tests followed it there.
use super::*;

mod chart_config_application_tests {
    //! Unit tests for `apply_chart_config_to_theme` — verify that ChartConfig overrides
    //! are correctly applied to ThemeInputs.

    use super::*;
    use chart_config::{
        AxisConfigSpec, AxisStyleSpec, ChartConfig, ColorConfigSpec, GridConfigSpec,
        LegendConfigSpec, LegendStyleSpec, PaddingConfigSpec, TitleConfigSpec,
    };
    // Sibling render stage: the mark-color resolution a `configure_color`
    // domain/range override is ultimately observed through.
    use crate::render::draw;

    /// A minimal `Linear` scale for tests that only need `apply_label_format_to_axis`'s
    /// `scale`/`tick_count` parameters to satisfy the signature (its numeric,
    /// non-temporal formatting paths never consult the scale directly).
    fn linear_scale(lo: f64, hi: f64) -> scale_resolve::ScaleKind {
        scale_resolve::ScaleKind::Linear(crate::scale::linear::LinearScale::new_internal(
            vec![lo, hi],
            vec![0.0, 1.0],
            false,
            false,
        ))
    }

    #[test]
    fn apply_chart_config_noop_on_empty_config() {
        let default_theme = ThemeInputs::default();
        let mut theme = default_theme.clone();
        apply_chart_config_to_theme(&mut theme, &ChartConfig::default());
        // Empty config must not change anything.
        assert_eq!(theme.grid.grid, default_theme.grid.grid);
        assert_eq!(theme.sizes.grid_width, default_theme.sizes.grid_width);
        assert_eq!(theme.colors.grid_color, default_theme.colors.grid_color);
        assert_eq!(theme.padding.padding, default_theme.padding.padding);
        assert_eq!(theme.legend.legend_orient, default_theme.legend.legend_orient);
        assert_eq!(theme.palette.color_scheme, default_theme.palette.color_scheme);
        assert_eq!(theme.typography.label_font_size, default_theme.typography.label_font_size);
        assert_eq!(theme.typography.title_font_size, default_theme.typography.title_font_size);
    }

    #[test]
    fn apply_chart_config_grid_color_and_width() {
        let mut theme = ThemeInputs::default();
        let config = ChartConfig {
            grid: Some(GridConfigSpec {
                color: Some("#ff0000".to_string()),
                width: Some(2.0),
                opacity: Some(0.5),
                ..Default::default()
            }),
            ..Default::default()
        };
        apply_chart_config_to_theme(&mut theme, &config);
        assert_eq!(theme.colors.grid_color, color::parse_color("#ff0000").unwrap());
        assert_eq!(theme.sizes.grid_width, 2.0);
        assert_eq!(theme.grid.grid_opacity, 0.5);
    }

    /// Batch A Task 8 sweep: `configure_grid(color=…)` was hex-only (it read
    /// through `from_hex_str`), so a CSS name or `rgb()` string silently kept
    /// the theme default. It now accepts the full vocabulary — and resolves to
    /// exactly the same color the equivalent hex does.
    #[test]
    fn apply_chart_config_grid_color_accepts_named_and_rgb_forms() {
        let grid_color = |spelling: &str| {
            let mut theme = ThemeInputs::default();
            let config = ChartConfig {
                grid: Some(GridConfigSpec {
                    color: Some(spelling.to_string()),
                    ..Default::default()
                }),
                ..Default::default()
            };
            apply_chart_config_to_theme(&mut theme, &config);
            theme.colors.grid_color
        };
        let expected = color::parse_color("#4682b4").unwrap();
        assert_eq!(grid_color("steelblue"), expected, "CSS name must take effect");
        assert_eq!(grid_color("rgb(70, 130, 180)"), expected, "rgb() must take effect");
        assert_eq!(grid_color("#4682b4"), expected, "hex is unchanged");
        // An unparseable value keeps the theme default, as before the sweep.
        assert_eq!(grid_color("not-a-color"), ThemeInputs::default().colors.grid_color);
    }

    /// A bare `AxesInput` for the per-axis grid/toggle unit tests below: two
    /// axes with no overrides at all, i.e. "no opinion" on every toggle.
    #[cfg(test)]
    pub(super) fn blank_axes() -> crate::layout::AxesInput {
        use crate::layout::{AxisInput, AxisOrient};
        crate::layout::AxesInput {
            x: AxisInput::new(AxisOrient::Bottom, None, vec!["a".into()], None),
            y: AxisInput::new(AxisOrient::Left, None, vec!["0".into()], None),
            show_x: true,
            show_y: true,
            secondary_y: Vec::new(),
        }
    }

    /// D4/F-L07-01 (spec §4.3): `configure_grid(x=False, y=False)` reaches
    /// each axis's OWN slot, not the single global theme flag. Was previously
    /// a `theme.grid.grid = false` assertion — the global write this task
    /// removed, because it is the thing that made a per-axis grid request
    /// inexpressible.
    #[test]
    fn grid_config_disables_grid_per_axis() {
        use crate::render::chart_config::GridAxisSpec;
        let mut axes = blank_axes();
        let config = ChartConfig {
            grid: Some(GridConfigSpec {
                x: Some(GridAxisSpec { enabled: Some(false), ..Default::default() }),
                y: Some(GridAxisSpec { enabled: Some(false), ..Default::default() }),
                ..Default::default()
            }),
            ..Default::default()
        };
        apply_grid_config_to_axis_inputs(&mut axes, &config);
        assert!(!axes.x.show_grid());
        assert!(!axes.y.show_grid());
    }

    /// The case the old `apply_chart_config_to_theme` block dropped ENTIRELY: x and y
    /// disagreeing. Its equality guard (`grid_cfg.y.unwrap_or(enabled) ==
    /// enabled`) failed, so neither branch wrote anything and the caller's
    /// whole request vanished. RED before this task on any assertion at all.
    #[test]
    fn grid_config_honors_disagreeing_x_and_y() {
        use crate::render::chart_config::GridAxisSpec;
        let mut axes = blank_axes();
        let config = ChartConfig {
            grid: Some(GridConfigSpec {
                x: Some(GridAxisSpec { enabled: Some(true), ..Default::default() }),
                y: Some(GridAxisSpec { enabled: Some(false), ..Default::default() }),
                ..Default::default()
            }),
            ..Default::default()
        };
        apply_grid_config_to_axis_inputs(&mut axes, &config);
        assert!(axes.x.show_grid(), "x grid explicitly enabled");
        assert!(!axes.y.show_grid(), "y grid explicitly disabled");
    }

    /// The per-axis grid STYLE half of §4.3: an axis's own `grid.x` object
    /// styles only that axis; the other axis keeps the theme fallback (a
    /// `None` override slot).
    #[test]
    fn grid_config_per_axis_style_does_not_leak_to_the_other_axis() {
        use crate::render::chart_config::GridAxisSpec;
        let mut axes = blank_axes();
        let config = ChartConfig {
            grid: Some(GridConfigSpec {
                x: Some(GridAxisSpec {
                    color: Some("#ff0000".into()),
                    width: Some(3.0),
                    ..Default::default()
                }),
                ..Default::default()
            }),
            ..Default::default()
        };
        apply_grid_config_to_axis_inputs(&mut axes, &config);
        assert_eq!(axes.x.overrides.grid_color, color::parse_color("#ff0000").ok());
        assert_eq!(axes.x.overrides.grid_width, Some(3.0));
        assert!(axes.y.overrides.grid_color.is_none(), "y keeps the theme fallback");
        assert!(axes.y.overrides.grid_width.is_none());
    }

    /// `configure_axis(grid=True)` now lands on both axes' own slots
    /// (previously: the global `theme.grid.grid`), and — because the toggle is
    /// a precedence chain rather than an AND — it lights the grid up on a
    /// theme that disables gridlines globally.
    #[test]
    fn axis_config_grid_enables_per_axis_over_a_grid_less_theme() {
        let mut axes = blank_axes();
        let cfg = AxisConfigSpec {
            style: AxisStyleSpec { grid: Some(true), ..Default::default() },
            ..Default::default()
        };
        apply_axis_config_to_axis_input(&mut axes.x, Some(&cfg)).unwrap();
        apply_axis_config_to_axis_input(&mut axes.y, Some(&cfg)).unwrap();
        let mut theme = ThemeInputs::default();
        theme.grid.grid = false;
        axes.apply_show_defaults(&theme);
        assert!(axes.x.show_grid());
        assert!(axes.y.show_grid());
    }

    /// The theme is the BOTTOM of the chain, not a veto: an axis with no
    /// opinion takes the theme's answer, an axis with one keeps its own.
    #[test]
    fn apply_show_defaults_fills_only_unopinionated_axes() {
        let mut axes = blank_axes();
        axes.x.overrides.show_grid = Some(true);
        let mut theme = ThemeInputs::default();
        theme.grid.grid = false;
        theme.axis.axis_line = false;
        axes.apply_show_defaults(&theme);
        assert!(axes.x.show_grid(), "explicit per-axis value survives");
        assert!(!axes.y.show_grid(), "unopinionated axis takes the theme's");
        assert!(!axes.x.show_domain(), "domain rides the same fold");
        assert!(!axes.y.show_domain());
    }

    #[test]
    fn apply_chart_config_padding_per_side() {
        let mut theme = ThemeInputs::default();
        let config = ChartConfig {
            padding: Some(PaddingConfigSpec {
                top: Some(10.0),
                right: Some(20.0),
                bottom: Some(30.0),
                left: Some(40.0),
                auto: None,
            }),
            ..Default::default()
        };
        apply_chart_config_to_theme(&mut theme, &config);
        // Each side is set independently.
        assert_eq!(theme.padding.padding_top, Some(10.0));
        assert_eq!(theme.padding.padding_right, Some(20.0));
        assert_eq!(theme.padding.padding_bottom, Some(30.0));
        assert_eq!(theme.padding.padding_left, Some(40.0));
        // Uniform fallback padding is unchanged.
        assert_eq!(theme.padding.padding, 16.0);
    }

    #[test]
    fn apply_chart_config_padding_auto_does_not_block_explicit_sides() {
        // auto=true (opt-in; `False` is the Python default) must NOT block
        // explicit side values.
        let mut theme = ThemeInputs::default();
        let config = ChartConfig {
            padding: Some(PaddingConfigSpec {
                top: Some(5.0),
                auto: Some(true),
                ..Default::default()
            }),
            ..Default::default()
        };
        apply_chart_config_to_theme(&mut theme, &config);
        // The explicit top value must be applied even when auto=true.
        assert_eq!(theme.padding.padding_top, Some(5.0));
        // Sides not specified remain None.
        assert!(theme.padding.padding_right.is_none());
        assert!(theme.padding.padding_bottom.is_none());
        assert!(theme.padding.padding_left.is_none());
        // The one line this function actually adds to this block
        // (render/mod.rs's `if let Some(auto) = pad.auto { theme.padding
        // .padding_auto = auto; }`) — the plumbing this test exists for was
        // previously covered only end to end from Python.
        assert!(theme.padding.padding_auto);
    }

    #[test]
    fn apply_chart_config_legend_orient_and_direction() {
        let mut theme = ThemeInputs::default();
        let config = legend_config(LegendStyleSpec {
            orient: Some("bottom".to_string()),
            direction: Some("horizontal".to_string()),
            columns: Some(4),
            title_font_size: Some(16.0),
            label_font_size: Some(9.0),
            ..Default::default()
        });
        // `direction` is the only legend field `apply_chart_config_to_theme` still
        // writes; the other three ThemeInputs-backed ones resolve in
        // `apply_legend_cascade_to_theme` (D7) so per-channel can win.
        apply_chart_config_to_theme(&mut theme, &config);
        assert_eq!(theme.legend.legend_direction, Some(LegendDirection::Horizontal));
        apply_legend_cascade_to_theme(
            &mut theme,
            &prepare::LegendPreparedOverrides::default(),
            &config,
        );
        assert_eq!(theme.legend.legend_orient, LegendOrient::Bottom);
        assert_eq!(theme.legend.legend_columns, Some(4));
        assert_eq!(theme.typography.legend_title_font_size, 16.0);
        // D7: `configure_legend(label_font_size=)` no longer writes the SHARED
        // typography slot (which axis tick labels also read) — it fills the
        // legend-own `LegendStyleOpts` slot instead.
        let theme_default = ThemeInputs::default();
        assert_eq!(
            theme.typography.label_font_size, theme_default.typography.label_font_size,
            "configure_legend(label_font_size=) must not resize axis labels",
        );
        let mut overrides = LegendOverrides::default();
        apply_chart_config_to_legend_overrides(&mut overrides, &config);
        assert_eq!(overrides.style.label_font_size, Some(9.0));
    }

    fn legend_config(style: LegendStyleSpec) -> ChartConfig {
        ChartConfig { legend: Some(LegendConfigSpec { style }), ..Default::default() }
    }

    /// D7 cascade repair: for each of the three fields whose effective value
    /// lives on `ThemeInputs`, a per-channel `Legend(...)` beats chart-level
    /// `configure_legend(...)`. RED before the fix — `apply_chart_config_to_theme` ran
    /// after the per-channel write and clobbered all three.
    #[test]
    fn legend_cascade_per_channel_beats_chart_level() {
        let mut theme = ThemeInputs::default();
        let config = legend_config(LegendStyleSpec {
            orient: Some("right".to_string()),
            columns: Some(4),
            title_font_size: Some(16.0),
            ..Default::default()
        });
        let per_channel = prepare::LegendPreparedOverrides {
            orient: Some(LegendOrient::Bottom),
            columns: Some(2),
            title_font_size: Some(21.0),
            ..Default::default()
        };
        apply_chart_config_to_theme(&mut theme, &config);
        apply_legend_cascade_to_theme(&mut theme, &per_channel, &config);
        assert_eq!(theme.legend.legend_orient, LegendOrient::Bottom);
        assert_eq!(theme.legend.legend_columns, Some(2));
        assert_eq!(theme.typography.legend_title_font_size, 21.0);
    }

    /// The other half of the same cascade: with no per-channel value, the
    /// chart-level one still lands (it is a fallback, not a no-op).
    #[test]
    fn legend_cascade_chart_level_fills_absent_per_channel() {
        let mut theme = ThemeInputs::default();
        let config = legend_config(LegendStyleSpec {
            orient: Some("left".to_string()),
            columns: Some(3),
            title_font_size: Some(17.0),
            ..Default::default()
        });
        apply_legend_cascade_to_theme(
            &mut theme,
            &prepare::LegendPreparedOverrides::default(),
            &config,
        );
        assert_eq!(theme.legend.legend_orient, LegendOrient::Left);
        assert_eq!(theme.legend.legend_columns, Some(3));
        assert_eq!(theme.typography.legend_title_font_size, 17.0);
    }

    // ── D6/F-L04-05: categorical `values` filter + order ──────────────────

    fn legend_entry(label: &str) -> crate::layout::LegendEntry {
        crate::layout::LegendEntry {
            label: label.to_string(),
            symbol: crate::layout::SymbolKind::Circle,
        }
    }

    fn labels_of(entries: &[crate::layout::LegendEntry]) -> Vec<&str> {
        entries.iter().map(|e| e.label.as_str()).collect()
    }

    #[test]
    fn legend_values_filters_and_orders_entries() {
        let mut entries = vec![legend_entry("a"), legend_entry("b"), legend_entry("c")];
        let mut warnings = Vec::new();
        apply_legend_values_to_entries(
            &mut entries,
            Some(&["c".to_string(), "a".to_string()]),
            &mut warnings,
        );
        assert_eq!(labels_of(&entries), ["c", "a"]);
        assert!(warnings.is_empty(), "{warnings:?}");
    }

    #[test]
    fn legend_values_absent_leaves_entries_untouched() {
        let mut entries = vec![legend_entry("a"), legend_entry("b")];
        let mut warnings = Vec::new();
        apply_legend_values_to_entries(&mut entries, None, &mut warnings);
        assert_eq!(labels_of(&entries), ["a", "b"]);
        assert!(warnings.is_empty());
    }

    /// A colorbar chart has no categorical entries; `values` is the colorbar
    /// arm's own tick-label override there and must not be consumed here.
    #[test]
    fn legend_values_on_empty_entries_is_a_no_op_and_never_warns() {
        let mut entries: Vec<crate::layout::LegendEntry> = Vec::new();
        let mut warnings = Vec::new();
        apply_legend_values_to_entries(
            &mut entries,
            Some(&["lo".to_string(), "hi".to_string()]),
            &mut warnings,
        );
        assert!(entries.is_empty());
        assert!(warnings.is_empty(), "a colorbar's values must not warn: {warnings:?}");
    }

    #[test]
    fn legend_values_unknown_names_warn_and_are_skipped() {
        let mut entries = vec![legend_entry("a"), legend_entry("b")];
        let mut warnings = Vec::new();
        apply_legend_values_to_entries(
            &mut entries,
            Some(&["a".to_string(), "zzz".to_string(), "qqq".to_string()]),
            &mut warnings,
        );
        assert_eq!(labels_of(&entries), ["a"]);
        assert_eq!(
            warnings,
            vec![RenderWarning::LegendValuesUnknown {
                values: vec!["zzz".to_string(), "qqq".to_string()],
            }],
        );
    }

    /// Cycle-2 S2: `chart_config_legend_disabled` reads the same
    /// `suppressed_by` predicate as the per-channel and aux askers, so a
    /// raw-dict chart-level `orient="none"` — one that never passed through
    /// Python's `_resolve_chart_config` → `disabled` conversion — suppresses
    /// too. RED before the fix: it read the bare `disabled` field, so
    /// `orient="none"` fell through `LegendOrient::parse` as no placement and
    /// the theme orient stood.
    #[test]
    fn chart_level_orient_none_suppresses_without_the_python_conversion() {
        let raw = legend_config(LegendStyleSpec {
            orient: Some("none".into()),
            ..Default::default()
        });
        assert!(chart_config_legend_disabled(&raw));
        let converted = legend_config(LegendStyleSpec {
            disabled: Some(true),
            ..Default::default()
        });
        assert!(chart_config_legend_disabled(&converted));
        let placed = legend_config(LegendStyleSpec {
            orient: Some("bottom".into()),
            ..Default::default()
        });
        assert!(!chart_config_legend_disabled(&placed));
        assert!(!chart_config_legend_disabled(&ChartConfig::default()));
    }

    /// Neither level set → the theme's own values stand untouched.
    #[test]
    fn legend_cascade_no_config_leaves_theme_untouched() {
        let mut theme = ThemeInputs::default();
        let before = theme.clone();
        apply_legend_cascade_to_theme(
            &mut theme,
            &prepare::LegendPreparedOverrides::default(),
            &ChartConfig::default(),
        );
        assert_eq!(theme.legend.legend_orient, before.legend.legend_orient);
        assert_eq!(theme.legend.legend_columns, before.legend.legend_columns);
        assert_eq!(
            theme.typography.legend_title_font_size,
            before.typography.legend_title_font_size
        );
    }

    #[test]
    fn apply_chart_config_color_scheme_override() {
        let mut theme = ThemeInputs::default();
        let config = ChartConfig {
            color: Some(ColorConfigSpec {
                scheme: Some("tableau10".to_string()),
                sequential_scheme: Some("viridis".to_string()),
                diverging_scheme: Some("rdbu".to_string()),
                ..Default::default()
            }),
            ..Default::default()
        };
        apply_chart_config_to_theme(&mut theme, &config);
        assert_eq!(theme.palette.color_scheme, "tableau10");
        assert_eq!(theme.palette.sequential_scheme, "viridis");
        assert_eq!(theme.palette.diverging_scheme, "rdbu");
    }

    #[test]
    fn apply_chart_config_axis_label_font_size() {
        let mut theme = ThemeInputs::default();
        let config = ChartConfig {
            axis: Some(AxisConfigSpec {
                style: AxisStyleSpec { label_font_size: Some(14.0), ..Default::default() },
                ..Default::default()
            }),
            ..Default::default()
        };
        apply_chart_config_to_theme(&mut theme, &config);
        assert_eq!(theme.typography.label_font_size, 14.0);
    }

    #[test]
    fn apply_chart_config_axis_tick_size_and_domain_visibility() {
        let mut theme = ThemeInputs::default();
        assert!(theme.axis.axis_line); // default on
        let config = ChartConfig {
            axis: Some(AxisConfigSpec {
                style: AxisStyleSpec {
                    tick_size: Some(6.0),
                    domain: Some(false),
                    domain_width: Some(2.0),
                    ..Default::default()
                },
                ..Default::default()
            }),
            ..Default::default()
        };
        apply_chart_config_to_theme(&mut theme, &config);
        assert_eq!(theme.sizes.tick_size, 6.0);
        assert_eq!(theme.sizes.axis_line_width, 2.0);
        // `domain` is no longer a theme write (D12, spec §4.9): it is a
        // per-axis toggle, so the shared `axis` key reaches each axis's own
        // slot instead of the one global flag that could not tell x from y.
        assert!(theme.axis.axis_line, "domain no longer mutates the global theme flag");
        let mut axes = blank_axes();
        apply_axis_config_to_axis_input(&mut axes.x, config.axis.as_ref()).unwrap();
        apply_axis_config_to_axis_input(&mut axes.y, config.axis.as_ref()).unwrap();
        axes.apply_show_defaults(&theme);
        assert!(!axes.x.show_domain());
        assert!(!axes.y.show_domain());
    }

    #[test]
    fn apply_chart_config_title_overrides() {
        let mut theme = ThemeInputs::default();
        let config = ChartConfig {
            title: Some(TitleConfigSpec {
                font_size: Some(22.0),
                font_weight: Some("700".to_string()),
                anchor: Some("end".to_string()),
                color: Some("#123456".to_string()),
                offset: Some(8.0),
                subtitle_font_size: Some(13.0),
                subtitle_color: Some("#ff0000".to_string()),
            }),
            ..Default::default()
        };
        apply_chart_config_to_theme(&mut theme, &config);
        assert_eq!(theme.typography.title_font_size, 22.0);
        assert_eq!(theme.typography.title_font_weight, "700");
        assert_eq!(theme.typography.title_anchor, TextAnchor::End);
        assert_eq!(theme.colors.title_color, color::parse_color("#123456").unwrap());
        assert_eq!(theme.typography.title_offset, 8.0);
        assert_eq!(theme.typography.subtitle_font_size, Some(13.0));
        assert_eq!(
            theme.colors.subtitle_color,
            Some(color::parse_color("#ff0000").unwrap())
        );
    }

    #[test]
    fn apply_chart_config_title_subtitle_defaults_unset() {
        // No `title` config → subtitle theme fields stay `None`, preserving the
        // pre-config default subtitle styling (font_color + title*0.85).
        let mut theme = ThemeInputs::default();
        apply_chart_config_to_theme(&mut theme, &ChartConfig::default());
        assert_eq!(theme.typography.subtitle_font_size, None);
        assert_eq!(theme.colors.subtitle_color, None);
    }

    #[test]
    fn apply_chart_config_invalid_hex_color_silently_ignored() {
        let mut theme = ThemeInputs::default();
        let original_grid_color = theme.colors.grid_color;
        let config = ChartConfig {
            grid: Some(GridConfigSpec {
                color: Some("not-a-hex-color".to_string()),
                ..Default::default()
            }),
            ..Default::default()
        };
        apply_chart_config_to_theme(&mut theme, &config);
        // Bad color must not change the existing value.
        assert_eq!(theme.colors.grid_color, original_grid_color);
    }

    /// The precedence this test used to assert on the THEME is now asserted
    /// where it belongs: on each axis. `axis_x` no longer writes the shared
    /// theme at all (D12, spec §4.9), so the shared `axis` value stays the
    /// fallback for every axis the per-axis sections don't address, while the
    /// x axis itself takes `axis_x`'s value — one answer per axis instead of
    /// one global slot two sections fight over.
    #[test]
    fn axis_x_wins_on_x_without_disturbing_the_shared_theme_fallback() {
        let mut theme = ThemeInputs::default();
        let config = ChartConfig {
            axis: Some(AxisConfigSpec {
                style: AxisStyleSpec { label_font_size: Some(10.0), ..Default::default() },
                ..Default::default()
            }),
            axis_x: Some(AxisConfigSpec {
                style: AxisStyleSpec { label_font_size: Some(14.0), ..Default::default() },
                ..Default::default()
            }),
            ..Default::default()
        };
        apply_chart_config_to_theme(&mut theme, &config);
        assert_eq!(
            theme.typography.label_font_size, 10.0,
            "only the SHARED `axis` key writes the theme; `axis_x` must not"
        );

        let mut axes = blank_axes();
        fill_axis_slots_specific_before_shared(&mut axes, &config, &mut Vec::new()).unwrap();
        assert_eq!(axes.x.overrides.label_font_size, Some(14.0), "axis_x wins on x");
        assert_eq!(axes.y.overrides.label_font_size, Some(10.0), "y takes the shared value");
    }

    /// The Rust half of retiring Python's `_redistribute_general_axis` (spec
    /// §7 cascade constraint). `tick_size` was the last `AxisStyleSpec` field
    /// whose ONLY carrier was the global theme, so a per-axis section could
    /// not express it and the Python helper tried to compensate by re-pinning
    /// the general value onto the opposite axis key — which, on a global
    /// last-writer-wins slot, made the general value win instead.
    ///
    /// RED before this task twice over: `AxisStyleOverrides` had no
    /// `tick_size` field at all, and `axis_x` wrote `theme.sizes.tick_size`.
    #[test]
    fn axis_x_tick_size_lands_on_x_only() {
        let mut theme = ThemeInputs::default();
        let config = ChartConfig {
            axis: Some(AxisConfigSpec {
                style: AxisStyleSpec { tick_size: Some(12.0), ..Default::default() },
                ..Default::default()
            }),
            axis_x: Some(AxisConfigSpec {
                style: AxisStyleSpec { tick_size: Some(2.0), ..Default::default() },
                ..Default::default()
            }),
            ..Default::default()
        };
        apply_chart_config_to_theme(&mut theme, &config);
        let mut axes = blank_axes();
        fill_axis_slots_specific_before_shared(&mut axes, &config, &mut Vec::new()).unwrap();
        assert_eq!(axes.x.tick_size(&theme), 2.0, "axis_x's tick_size reaches x");
        assert_eq!(axes.y.tick_size(&theme), 12.0, "y keeps the shared value");
    }

    #[test]
    fn axis_x_styling_field_wins_over_axis_via_fill_none_ordering() {
        // Per-axis STYLING fields (grid_color/width/dash, label_color, domain_*,
        // title_*, label_padding) flow through `apply_axis_config_to_axis_input`,
        // which fills `AxisInput.overrides` only when still `None` (first writer
        // wins), so `axis_x > axis` holds only if the MORE-SPECIFIC key is
        // applied FIRST. Asserted through `fill_axis_slots_specific_before_shared`
        // — the production tier fn that owns that order — rather than by
        // hand-sequencing the calls, so a reorder there fails here.
        let config = ChartConfig {
            axis: Some(AxisConfigSpec {
                style: AxisStyleSpec { grid_color: Some("#00ff00".into()), ..Default::default() },
                ..Default::default()
            }),
            axis_x: Some(AxisConfigSpec {
                style: AxisStyleSpec { grid_color: Some("#0000ff".into()), ..Default::default() },
                ..Default::default()
            }),
            ..Default::default()
        };
        let mut axes = blank_axes();
        fill_axis_slots_specific_before_shared(&mut axes, &config, &mut Vec::new()).unwrap();
        // axis_x blue (0,0,255) must win over axis green.
        assert_eq!(
            axes.x.overrides.grid_color.map(|c| [c.red, c.green, c.blue]),
            Some([0, 0, 255]),
            "axis_x grid_color must win over the shared axis grid_color",
        );
    }

    #[test]
    fn apply_chart_config_grid_dash() {
        let mut theme = ThemeInputs::default();
        assert!(theme.grid.grid_dash.is_none());
        let config = ChartConfig {
            grid: Some(GridConfigSpec {
                dash: Some(vec![4.0, 4.0]),
                ..Default::default()
            }),
            ..Default::default()
        };
        apply_chart_config_to_theme(&mut theme, &config);
        assert_eq!(theme.grid.grid_dash, Some(vec![4.0, 4.0]));
    }

    // ── New wired fields (configure-punchlist, 2026-05-24) ───────────────────

    #[test]
    fn axis_config_title_font_size_propagates_to_axis_input() {
        let mut axis = crate::layout::AxisInput::new(
            crate::layout::AxisOrient::Bottom,
            Some("X Axis".to_string()),
            vec!["0".to_string(), "50".to_string(), "100".to_string()],
            None,
        );
        let cfg = AxisConfigSpec {
            style: AxisStyleSpec { title_font_size: Some(16.0), ..Default::default() },
            ..Default::default()
        };
        apply_axis_config_to_axis_input(&mut axis, Some(&cfg)).unwrap();
        assert_eq!(axis.overrides.title_font_size, Some(16.0));
    }

    #[test]
    fn chart_config_orient_propagates_when_at_default() {
        // configure_axis(orient="top") on an x-axis with no per-channel orient
        // override fills `overrides.orient`; `resolve_orient` then moves the
        // concrete side to Top. The orphan positioning fields flow too.
        let mut axis = crate::layout::AxisInput::new(
            crate::layout::AxisOrient::Bottom,
            Some("X".to_string()),
            vec!["0".to_string()],
            None,
        );
        let cfg = AxisConfigSpec {
            style: AxisStyleSpec {
                orient: Some("top".to_string()),
                translate: Some(5.0),
                grid_opacity: Some(0.4),
                zindex: Some(1),
                ..Default::default()
            },
            ..Default::default()
        };
        apply_axis_config_to_axis_input(&mut axis, Some(&cfg)).unwrap();
        axis.resolve_orient();
        assert_eq!(axis.orient, crate::layout::AxisOrient::Top);
        assert_eq!(axis.overrides.translate, Some(5.0));
        assert_eq!(axis.overrides.grid_opacity, Some(0.4));
        assert_eq!(axis.overrides.zindex, Some(1));
    }

    #[test]
    fn chart_config_cross_dimension_orient_fails_loud() {
        // configure_axis(orient="left") on an x-axis is a cross-dimension error.
        let mut axis = crate::layout::AxisInput::new(
            crate::layout::AxisOrient::Bottom,
            None,
            vec![],
            None,
        );
        let cfg = AxisConfigSpec {
            style: AxisStyleSpec { orient: Some("left".to_string()), ..Default::default() },
            ..Default::default()
        };
        let err = apply_axis_config_to_axis_input(&mut axis, Some(&cfg)).unwrap_err();
        assert!(matches!(err, RenderError::InvalidAxisOrient { channel: "x", .. }));
    }

    #[test]
    fn chart_config_orient_does_not_override_per_channel() {
        // An x-axis already moved to Top by the per-channel path (which sets the
        // `overrides.orient` INPUT, mirroring prepare.rs) must not be moved again
        // by a conflicting chart-level orient — the chart-level fill is gated on
        // `is_none()`, so per-channel wins.
        let mut axis = crate::layout::AxisInput::new(
            crate::layout::AxisOrient::Top,
            None,
            vec![],
            None,
        );
        axis.overrides.orient = Some(crate::layout::AxisOrient::Top); // per-channel already set Top
        let cfg = AxisConfigSpec {
            style: AxisStyleSpec { orient: Some("bottom".to_string()), ..Default::default() },
            ..Default::default()
        };
        apply_axis_config_to_axis_input(&mut axis, Some(&cfg)).unwrap();
        axis.resolve_orient();
        assert_eq!(axis.orient, crate::layout::AxisOrient::Top, "per-channel orient must win");
        assert_eq!(axis.overrides.orient, Some(crate::layout::AxisOrient::Top));
    }

    #[test]
    fn per_channel_orient_at_default_side_still_beats_chart_level() {
        // Regression (B5 follow-up, Issue 1): an EXPLICIT per-channel
        // `fm.Axis(orient="bottom")` lands on the x default side (Bottom). The old
        // value-heuristic treated "at default side" as "unset" and let a
        // chart-level `configure(axis_x=AxisConfig(orient="top"))` win — a
        // precedence inversion. With the `is_none()` sentinel the explicit
        // per-channel Bottom wins and the axis stays at the bottom.
        let mut axis = crate::layout::AxisInput::new(
            crate::layout::AxisOrient::Bottom,
            None,
            vec![],
            None,
        );
        // Mirror prepare.rs: an explicit per-channel orient sets BOTH the concrete
        // side and the `overrides.orient` input (even when it equals the default).
        axis.overrides.orient = Some(crate::layout::AxisOrient::Bottom);
        let cfg = AxisConfigSpec {
            style: AxisStyleSpec { orient: Some("top".to_string()), ..Default::default() },
            ..Default::default()
        };
        apply_axis_config_to_axis_input(&mut axis, Some(&cfg)).unwrap();
        axis.resolve_orient();
        assert_eq!(
            axis.orient,
            crate::layout::AxisOrient::Bottom,
            "explicit per-channel orient='bottom' must beat chart-level orient='top'",
        );

        // y mirror: explicit per-channel Left must beat chart-level Right.
        let mut yaxis = crate::layout::AxisInput::new(
            crate::layout::AxisOrient::Left,
            None,
            vec![],
            None,
        );
        yaxis.overrides.orient = Some(crate::layout::AxisOrient::Left);
        let ycfg = AxisConfigSpec {
            style: AxisStyleSpec { orient: Some("right".to_string()), ..Default::default() },
            ..Default::default()
        };
        apply_axis_config_to_axis_input(&mut yaxis, Some(&ycfg)).unwrap();
        yaxis.resolve_orient();
        assert_eq!(
            yaxis.orient,
            crate::layout::AxisOrient::Left,
            "explicit per-channel orient='left' must beat chart-level orient='right'",
        );
    }

    #[test]
    fn chart_config_offset_flush_overlap_propagate() {
        // configure_axis(offset=, label_flush=, label_overlap=) flow to AxisInput.
        let mut axis = crate::layout::AxisInput::new(
            crate::layout::AxisOrient::Bottom,
            None,
            vec!["0".to_string()],
            None,
        );
        let cfg = AxisConfigSpec {
            style: AxisStyleSpec {
                offset: Some(30.0),
                label_flush: Some(true),
                label_overlap: Some("parity".to_string()),
                ..Default::default()
            },
            ..Default::default()
        };
        apply_axis_config_to_axis_input(&mut axis, Some(&cfg)).unwrap();
        assert_eq!(axis.overrides.offset, Some(30.0));
        assert_eq!(axis.overrides.label_flush, Some(true));
        assert_eq!(axis.overrides.label_overlap, Some(crate::layout::LabelOverlap::Parity));
    }

    #[test]
    fn chart_config_offset_flush_overlap_do_not_override_per_channel() {
        // Per-channel values already on the AxisInput must win over chart-level.
        let mut axis = crate::layout::AxisInput::new(
            crate::layout::AxisOrient::Bottom,
            None,
            vec!["0".to_string()],
            None,
        );
        axis.overrides.offset = Some(5.0);
        axis.overrides.label_flush = Some(false);
        axis.overrides.label_overlap = Some(crate::layout::LabelOverlap::ShowAll);
        let cfg = AxisConfigSpec {
            style: AxisStyleSpec {
                offset: Some(30.0),
                label_flush: Some(true),
                label_overlap: Some("rotate".to_string()),
                ..Default::default()
            },
            ..Default::default()
        };
        apply_axis_config_to_axis_input(&mut axis, Some(&cfg)).unwrap();
        assert_eq!(axis.overrides.offset, Some(5.0), "per-channel offset must win");
        assert_eq!(axis.overrides.label_flush, Some(false), "per-channel label_flush must win");
        assert_eq!(
            axis.overrides.label_overlap,
            Some(crate::layout::LabelOverlap::ShowAll),
            "per-channel label_overlap must win",
        );
    }

    #[test]
    fn axis_config_title_color_propagates_to_axis_input() {
        let mut axis = crate::layout::AxisInput::new(
            crate::layout::AxisOrient::Left,
            Some("Y".to_string()),
            vec!["0".to_string()],
            None,
        );
        let cfg = AxisConfigSpec {
            style: AxisStyleSpec { title_color: Some("#ff0000".to_string()), ..Default::default() },
            ..Default::default()
        };
        apply_axis_config_to_axis_input(&mut axis, Some(&cfg)).unwrap();
        let c = axis.overrides.title_color.expect("title_color should be Some");
        assert_eq!(c.red, 0xff);
        assert_eq!(c.green, 0x00);
        assert_eq!(c.blue, 0x00);
    }

    #[test]
    fn axis_config_title_padding_propagates_to_axis_input() {
        let mut axis = crate::layout::AxisInput::new(
            crate::layout::AxisOrient::Bottom,
            None,
            vec![],
            None,
        );
        let cfg = AxisConfigSpec {
            style: AxisStyleSpec { title_padding: Some(12.0), ..Default::default() },
            ..Default::default()
        };
        apply_axis_config_to_axis_input(&mut axis, Some(&cfg)).unwrap();
        assert_eq!(axis.overrides.title_padding, Some(12.0));
    }

    #[test]
    fn axis_config_label_format_raw_reformats_numeric_labels() {
        let mut axis = crate::layout::AxisInput::new(
            crate::layout::AxisOrient::Bottom,
            None,
            vec!["1000".to_string(), "2000".to_string(), "3000".to_string()],
            None,
        );
        let cfg = AxisConfigSpec { label_format_raw: Some(",.0f".to_string()), ..Default::default() };
        apply_axis_config_to_axis_input(&mut axis, Some(&cfg)).unwrap();
        let scale = linear_scale(0.0, 3000.0);
        apply_label_format_to_axis(&mut axis, &scale, 3, false);
        // ",.0f" formats with thousands separator and 0 decimal places.
        assert_eq!(axis.tick_labels, vec!["1,000", "2,000", "3,000"]);
    }

    #[test]
    fn axis_config_label_format_raw_with_tick_values_replaces_labels() {
        let mut axis = crate::layout::AxisInput::new(
            crate::layout::AxisOrient::Bottom,
            None,
            vec!["0.0".to_string(), "0.5".to_string(), "1.0".to_string()],
            None,
        );
        let cfg = AxisConfigSpec {
            label_format_raw: Some(".1%".to_string()),
            style: AxisStyleSpec { values: Some(vec![0.0, 0.5, 1.0]), ..Default::default() },
            ..Default::default()
        };
        apply_axis_config_to_axis_input(&mut axis, Some(&cfg)).unwrap();
        let scale = linear_scale(0.0, 1.0);
        apply_label_format_to_axis(&mut axis, &scale, 3, false);
        // Raw `label_format_raw` carries no sibling type field (D8:
        // `effective_label_format_type()` returns `None` for it) — NF-B2's
        // exact regression guard: a raw `%`-bearing NUMERIC spec must still
        // resolve as percent, not get swept into the time branch.
        assert_eq!(axis.tick_labels, vec!["0.0%", "50.0%", "100.0%"]);
    }

    /// D8/F-L07-05 (the batch's motivating repro): a chart-level TIME preset
    /// (`configure_axis(label_format="date_iso")`, resolved by Python into
    /// `label_format="%Y-%m-%d"` + `label_format_type="time"`) must render
    /// real dates, not misparse the strftime pattern as a d3 numeric spec
    /// (the pre-fix bug: `%` tokenized as the d3 percent TYPE char, yielding
    /// output like `"300.000000%"`).
    #[test]
    fn axis_config_time_preset_reformats_default_date_labels_as_real_dates() {
        use crate::scale::time::TimeScale;
        // 2020-01-01T00:00:00Z .. 2020-01-04T00:00:00Z, 4 daily ticks.
        let epoch_lo = 1_577_836_800_000.0;
        let epoch_hi = epoch_lo + 3.0 * 86_400_000.0;
        let scale = scale_resolve::ScaleKind::Time(TimeScale::new_internal(
            vec![epoch_lo, epoch_hi],
            vec![0.0, 1.0],
            false,
            false,
        ));
        // The default (no explicit format) labels the axis already carries —
        // spacing-keyed date strings from `format::format_time`, NOT
        // reparseable epoch-ms integers (this is exactly why the fix must
        // re-derive raw values from `scale`, not reparse these strings).
        let default_labels = scale.tick_labels(4);
        let mut axis = crate::layout::AxisInput::new(
            crate::layout::AxisOrient::Bottom,
            None,
            default_labels,
            None,
        );
        let cfg = AxisConfigSpec {
            style: AxisStyleSpec {
                label_format: Some("%Y-%m-%d".to_string()),
                label_format_type: Some("time".to_string()),
                ..Default::default()
            },
            ..Default::default()
        };
        apply_axis_config_to_axis_input(&mut axis, Some(&cfg)).unwrap();
        assert_eq!(axis.overrides.label_format_type.as_deref(), Some("time"));
        apply_label_format_to_axis(&mut axis, &scale, 4, false);
        for label in &axis.tick_labels {
            assert!(
                !label.contains('%'),
                "date-formatted label must not carry a literal '%': {label}"
            );
            // ISO date shape: YYYY-MM-DD.
            assert_eq!(label.len(), 10, "expected an ISO date, got {label:?}");
            assert!(label.starts_with("2020-01-0"), "expected a Jan 2020 date, got {label:?}");
        }
    }

    /// NF-B2, chart level: `label_format` + explicit `values=` compose in
    /// BOTH directions — a numeric percent spec applies correctly to
    /// explicit tick values even though `label_format_type` is now real
    /// (not hardcoded `None`).
    #[test]
    fn axis_config_label_format_percent_composes_with_explicit_values() {
        let mut axis = crate::layout::AxisInput::new(
            crate::layout::AxisOrient::Bottom,
            None,
            vec!["ignored".to_string()],
            None,
        );
        let cfg = AxisConfigSpec {
            style: AxisStyleSpec {
                label_format: Some(".1%".to_string()),
                label_format_type: Some("number".to_string()),
                values: Some(vec![0.0, 0.5, 1.0]),
                ..Default::default()
            },
            ..Default::default()
        };
        apply_axis_config_to_axis_input(&mut axis, Some(&cfg)).unwrap();
        assert_eq!(axis.overrides.label_format_type.as_deref(), Some("number"));
        let scale = linear_scale(0.0, 1.0);
        apply_label_format_to_axis(&mut axis, &scale, 3, false);
        assert_eq!(axis.tick_labels, vec!["0.0%", "50.0%", "100.0%"]);
    }

    /// D8: the `"ordinal"` preset's sentinel threads through the chart-level
    /// path end to end.
    #[test]
    fn axis_config_ordinal_preset_renders_suffixes() {
        let mut axis = crate::layout::AxisInput::new(
            crate::layout::AxisOrient::Bottom,
            None,
            vec!["1".to_string(), "2".to_string(), "3".to_string(), "11".to_string()],
            None,
        );
        let cfg = AxisConfigSpec {
            style: AxisStyleSpec {
                label_format: Some("__ordinal__".to_string()),
                label_format_type: Some("number".to_string()),
                ..Default::default()
            },
            ..Default::default()
        };
        apply_axis_config_to_axis_input(&mut axis, Some(&cfg)).unwrap();
        let scale = linear_scale(1.0, 11.0);
        apply_label_format_to_axis(&mut axis, &scale, 4, false);
        assert_eq!(axis.tick_labels, vec!["1st", "2nd", "3rd", "11th"]);
    }

    /// D8 cascade-inversion fix: `label_format_claimed`
    /// must block the chart-level fill EVEN THOUGH `label_format` itself
    /// reads `None` — the exact shape `prepare::build_axis_input` produces
    /// for a per-channel TEMPORAL format (applied eagerly, threads `None`
    /// back). Simulates that shape directly (bypassing the full
    /// prepare/scale-resolve pipeline, which the end-to-end
    /// `render_svg`-level test below also covers) to pin the mechanism in
    /// isolation.
    #[test]
    fn axis_config_chart_level_fill_skipped_when_per_channel_claimed_via_flag() {
        let mut axis = crate::layout::AxisInput::new(
            crate::layout::AxisOrient::Bottom,
            None,
            vec!["Feb 06".to_string(), "Mar 06".to_string()],
            None,
        );
        // Simulates prepare::build_axis_input's post-eager-temporal-apply
        // state: label_format threaded None, but the axis IS claimed.
        axis.overrides.label_format_claimed = true;
        let cfg = AxisConfigSpec {
            style: AxisStyleSpec {
                label_format: Some("%Y-%m-%d".to_string()),
                label_format_type: Some("time".to_string()),
                ..Default::default()
            },
            ..Default::default()
        };
        apply_axis_config_to_axis_input(&mut axis, Some(&cfg)).unwrap();
        // Chart-level must not have filled the slot at all.
        assert_eq!(axis.overrides.label_format, None);
        assert_eq!(axis.overrides.label_format_type, None);
        // And apply_label_format_to_axis (which no-ops on a None label_format)
        // must leave the already-applied per-channel labels untouched.
        let scale = linear_scale(0.0, 1.0);
        apply_label_format_to_axis(&mut axis, &scale, 2, false);
        assert_eq!(axis.tick_labels, vec!["Feb 06".to_string(), "Mar 06".to_string()]);
    }

    /// The unclaimed case (control): with `label_format_claimed = false` (the
    /// default), the chart-level time preset DOES fill and re-derive — this
    /// is the correct, desired "chart-level alone" behavior the claimed-flag
    /// fix must not regress.
    #[test]
    fn axis_config_chart_level_fill_proceeds_when_unclaimed() {
        use crate::scale::time::TimeScale;
        let epoch_lo = 1_577_836_800_000.0; // 2020-01-01
        let epoch_hi = epoch_lo + 86_400_000.0; // 2020-01-02
        let scale = scale_resolve::ScaleKind::Time(TimeScale::new_internal(
            vec![epoch_lo, epoch_hi],
            vec![0.0, 1.0],
            false,
            false,
        ));
        let default_labels = scale.tick_labels(2);
        let mut axis =
            crate::layout::AxisInput::new(crate::layout::AxisOrient::Bottom, None, default_labels, None);
        assert!(!axis.overrides.label_format_claimed, "default must be unclaimed");
        let cfg = AxisConfigSpec {
            style: AxisStyleSpec {
                label_format: Some("%Y-%m-%d".to_string()),
                label_format_type: Some("time".to_string()),
                ..Default::default()
            },
            ..Default::default()
        };
        apply_axis_config_to_axis_input(&mut axis, Some(&cfg)).unwrap();
        assert_eq!(axis.overrides.label_format.as_deref(), Some("%Y-%m-%d"));
        apply_label_format_to_axis(&mut axis, &scale, 2, false);
        for label in &axis.tick_labels {
            assert!(label.starts_with("2020-01-0"), "expected an ISO date, got {label:?}");
        }
    }

    #[test]
    fn axis_config_label_padding_propagates_to_axis_input() {
        let mut axis = crate::layout::AxisInput::new(
            crate::layout::AxisOrient::Bottom,
            None,
            vec!["0".to_string(), "50".to_string(), "100".to_string()],
            None,
        );
        let cfg = AxisConfigSpec {
            style: AxisStyleSpec { label_padding: Some(6.0), ..Default::default() },
            ..Default::default()
        };
        apply_axis_config_to_axis_input(&mut axis, Some(&cfg)).unwrap();
        assert_eq!(axis.overrides.label_padding, Some(6.0));
    }

    #[test]
    fn axis_config_per_channel_wins_over_configure() {
        // Per-channel label_angle (level 2) wins over configure (level 3).
        let mut axis = crate::layout::AxisInput::new(
            crate::layout::AxisOrient::Bottom,
            None,
            vec![],
            Some(-45.0), // per-channel override already set (seeds overrides.label_angle)
        );
        let cfg = AxisConfigSpec {
            style: AxisStyleSpec { label_angle: Some(-90.0), ..Default::default() },
            ..Default::default()
        };
        apply_axis_config_to_axis_input(&mut axis, Some(&cfg)).unwrap();
        // -45.0 should win because it was already set (Some).
        assert_eq!(axis.overrides.label_angle, Some(-45.0));
    }

    #[test]
    fn legend_config_gradient_length_fills_legend_overrides() {
        let prep_overrides = legend_overrides_from_prep_default();
        // When prep has no gradient_length, configure() should fill it.
        let mut overrides = prep_overrides;
        let config = ChartConfig {
            legend: Some(chart_config::LegendConfigSpec {
                style: LegendStyleSpec { gradient_length: Some(300.0), ..Default::default() },
            }),
            ..Default::default()
        };
        apply_chart_config_to_legend_overrides(&mut overrides, &config);
        assert_eq!(overrides.gradient_length, Some(300.0));
    }

    #[test]
    fn legend_config_symbol_type_fills_legend_overrides() {
        let mut overrides = legend_overrides_from_prep_default();
        let config = ChartConfig {
            legend: Some(chart_config::LegendConfigSpec {
                style: LegendStyleSpec { symbol_type: Some("square".to_string()), ..Default::default() },
            }),
            ..Default::default()
        };
        apply_chart_config_to_legend_overrides(&mut overrides, &config);
        assert_eq!(overrides.symbol_type.as_deref(), Some("square"));
    }

    #[test]
    fn legend_config_per_encoding_wins_over_configure() {
        // Per-encoding gradient_length (already set) must not be overwritten by configure.
        let mut overrides = LegendOverrides {
            gradient_length: Some(150.0), // already set at level 2
            ..Default::default()
        };
        let config = ChartConfig {
            legend: Some(chart_config::LegendConfigSpec {
                // configure (level 3) tries to override.
                style: LegendStyleSpec { gradient_length: Some(300.0), ..Default::default() },
            }),
            ..Default::default()
        };
        apply_chart_config_to_legend_overrides(&mut overrides, &config);
        assert_eq!(overrides.gradient_length, Some(150.0)); // level 2 wins
    }

    /// B5 unit 3: `configure_legend(symbol_stroke_width=...)` fills the override
    /// only when the per-channel value is absent.
    #[test]
    fn legend_config_symbol_stroke_width_fills_when_absent() {
        let mut overrides = legend_overrides_from_prep_default();
        let config = ChartConfig {
            legend: Some(chart_config::LegendConfigSpec {
                style: LegendStyleSpec { symbol_stroke_width: Some(2.0), ..Default::default() },
            }),
            ..Default::default()
        };
        apply_chart_config_to_legend_overrides(&mut overrides, &config);
        assert_eq!(overrides.style.symbol_stroke_width, Some(2.0));
    }

    /// B5 unit 3: a per-channel orphan (here `symbol_stroke_width`) beats the
    /// chart-level `configure_legend` value.
    #[test]
    fn legend_orphan_per_channel_wins_over_configure() {
        let mut overrides = LegendOverrides {
            // 380: per-channel (level 2) style fields nest on `style`.
            style: crate::layout::LegendStyleOpts {
                symbol_stroke_width: Some(5.0),
                row_padding: Some(18.0),
                clip_height: Some(40.0),
                // B5 unit 6a orphans, per-channel.
                symbol_size: Some(300.0),
                label_color: Some("#ff0000".into()),
                offset: Some(50.0),
                padding: Some(30.0),
                title_padding: Some(25.0),
                ..Default::default()
            },
            tick_min_step: Some(2.0),
            ..Default::default()
        };
        let config = ChartConfig {
            legend: Some(chart_config::LegendConfigSpec {
                style: LegendStyleSpec {
                    symbol_stroke_width: Some(1.0),
                    row_padding: Some(4.0),
                    clip_height: Some(999.0),
                    tick_min_step: Some(99.0),
                    symbol_size: Some(1.0),
                    label_color: Some("#0000ff".into()),
                    offset: Some(1.0),
                    padding: Some(1.0),
                    title_padding: Some(1.0),
                    ..Default::default()
                },
            }),
            ..Default::default()
        };
        apply_chart_config_to_legend_overrides(&mut overrides, &config);
        assert_eq!(overrides.style.symbol_stroke_width, Some(5.0), "per-channel wins");
        assert_eq!(overrides.style.row_padding, Some(18.0), "per-channel wins");
        assert_eq!(overrides.style.clip_height, Some(40.0), "per-channel wins");
        assert_eq!(overrides.tick_min_step, Some(2.0), "per-channel wins");
        assert_eq!(overrides.style.symbol_size, Some(300.0), "per-channel wins");
        assert_eq!(overrides.style.label_color.as_deref(), Some("#ff0000"), "per-channel wins");
        assert_eq!(overrides.style.offset, Some(50.0), "per-channel wins");
        assert_eq!(overrides.style.padding, Some(30.0), "per-channel wins");
        assert_eq!(overrides.style.title_padding, Some(25.0), "per-channel wins");
    }

    /// When per-channel leaves the 6a fields `None`, `configure_legend` fills
    /// them (level 3) — the chart-level fallback.
    #[test]
    fn legend_6a_orphan_chart_level_fills_when_absent() {
        let mut overrides = LegendOverrides::default();
        let config = ChartConfig {
            legend: Some(chart_config::LegendConfigSpec {
                style: LegendStyleSpec {
                    symbol_size: Some(300.0),
                    label_color: Some("#0000ff".into()),
                    offset: Some(50.0),
                    padding: Some(30.0),
                    title_padding: Some(25.0),
                    ..Default::default()
                },
            }),
            ..Default::default()
        };
        apply_chart_config_to_legend_overrides(&mut overrides, &config);
        assert_eq!(overrides.style.symbol_size, Some(300.0));
        assert_eq!(overrides.style.label_color.as_deref(), Some("#0000ff"));
        assert_eq!(overrides.style.offset, Some(50.0));
        assert_eq!(overrides.style.padding, Some(30.0));
        assert_eq!(overrides.style.title_padding, Some(25.0));
    }

    #[test]
    fn color_config_domain_override_updates_continuous_scale() {
        use scale_resolve::ColorScale;
        use crate::render::color::{ContinuousScheme, NamedContinuous};
        let mut color_scale: Option<ColorScale> = Some(ColorScale::Continuous {
            domain: (0.0, 1.0),
            scheme: ContinuousScheme::Named(NamedContinuous::Viridis),
            midpoint: None,
        });
        let cfg = ColorConfigSpec { domain: Some(vec![serde_json::json!(10.0), serde_json::json!(90.0)]), ..Default::default() };
        let warnings = apply_color_config_to_color_scale(&mut color_scale, &cfg);
        assert!(warnings.is_empty(), "a well-formed override must not warn: {warnings:?}");
        if let Some(ColorScale::Continuous { domain, .. }) = color_scale {
            assert_eq!(domain, (10.0, 90.0));
        } else {
            panic!("expected Continuous color scale");
        }
    }

    #[test]
    fn color_config_range_override_builds_gradient() {
        use scale_resolve::ColorScale;
        use crate::render::color::{ContinuousScheme, NamedContinuous};
        let mut color_scale: Option<ColorScale> = Some(ColorScale::Continuous {
            domain: (0.0, 1.0),
            scheme: ContinuousScheme::Named(NamedContinuous::Viridis),
            midpoint: None,
        });
        let cfg = ColorConfigSpec {
            range: Some(vec!["#ffffff".to_string(), "#000000".to_string()]),
            ..Default::default()
        };
        let warnings = apply_color_config_to_color_scale(&mut color_scale, &cfg);
        assert!(warnings.is_empty(), "a well-formed override must not warn: {warnings:?}");
        if let Some(ColorScale::Continuous { scheme: ContinuousScheme::Gradient(stops), .. }) = color_scale {
            assert_eq!(stops.len(), 2);
            // First stop at t=0, last at t=1.
            assert!((stops[0].0 - 0.0).abs() < 1e-9);
            assert!((stops[1].0 - 1.0).abs() < 1e-9);
        } else {
            panic!("expected Gradient color scheme after range override");
        }
    }

    /// Helper: build a zero-filled LegendOverrides (as if no per-encoding overrides existed).
    fn legend_overrides_from_prep_default() -> LegendOverrides {
        LegendOverrides::default()
    }

    /// A categorical color scale over `domain`, painted from the tableau10
    /// palette (the shape `build_color_scale` produces).
    fn categorical_scale(domain: &[&str]) -> scale_resolve::ColorScale {
        scale_resolve::ColorScale::Categorical {
            domain: domain.iter().map(|s| s.to_string()).collect(),
            palette: std::borrow::Cow::Borrowed(color::palette::categorical_palette("tableau10")),
        }
    }

    /// `configure_color(domain=[…])` on a categorical scale reorders the
    /// category order (and therefore which palette color each category takes).
    #[test]
    fn color_config_domain_reorders_categorical_scale() {
        use scale_resolve::ColorScale;
        let mut color_scale = Some(categorical_scale(&["a", "b", "c"]));
        let before = color_scale.as_ref().unwrap().lookup("c").unwrap();

        let cfg = ColorConfigSpec {
            domain: Some(vec![serde_json::json!("c"), serde_json::json!("a")]),
            ..Default::default()
        };
        let warnings = apply_color_config_to_color_scale(&mut color_scale, &cfg);
        // Dropping "b" is a reportable degradation — see
        // `color_config_domain_omitting_a_data_category_warns` for the contract.
        assert_eq!(
            warnings,
            vec![RenderWarning::ColorDomainOmitsCategories { categories: vec!["b".into()] }],
        );

        let scale = color_scale.expect("scale survives the override");
        match &scale {
            ColorScale::Categorical { domain, .. } => {
                assert_eq!(domain, &["c", "a"], "listed order wins; unlisted 'b' is dropped");
            }
            other => panic!("expected Categorical, got {other:?}"),
        }
        assert!(scale.lookup("b").is_none(), "'b' is no longer in the domain");
        assert_ne!(
            (scale.lookup("c").unwrap().red, scale.lookup("c").unwrap().green),
            (before.red, before.green),
            "'c' moved to palette slot 0, so its color must change"
        );
    }

    /// Spec §4.2 (amended 2026-08-28): a `configure_color(domain=…)` that omits
    /// a data category keeps that category's marks rendering — in the theme mark
    /// color, with no legend entry — and *names the omission* so the gap is
    /// diagnosable rather than looking like a rendering bug. A domain that
    /// covers every data category is silent.
    #[test]
    fn color_config_domain_omitting_a_data_category_warns() {
        use crate::layout::{LegendEntry, SymbolKind};

        let entry = |label: &str| LegendEntry { label: label.into(), symbol: SymbolKind::Circle };
        let theme_fallback = color::from_rgba(1, 2, 3, 255);
        let data_categories = ["a", "b", "c"];

        // ── Partial domain: two of three categories listed ───────────────────
        let mut color_scale = Some(categorical_scale(&data_categories));
        let mut legend_entries: Vec<LegendEntry> =
            data_categories.iter().map(|c| entry(c)).collect();
        let warnings = apply_color_config_to_color_scale(
            &mut color_scale,
            &ColorConfigSpec {
                domain: Some(vec![serde_json::json!("a"), serde_json::json!("c")]),
                ..Default::default()
            },
        );

        // (1) The warning names the omitted category — and only it.
        assert_eq!(
            warnings,
            vec![RenderWarning::ColorDomainOmitsCategories { categories: vec!["b".into()] }],
        );
        assert_eq!(
            warnings[0].to_string(),
            "color domain does not list b; those marks paint in the default mark color \
             with no legend entry",
        );

        // (2) "b"'s marks still render: the fill resolver falls back to the
        //     theme mark color rather than dropping the row or panicking.
        let scale = color_scale.as_ref().expect("scale survives");
        assert_eq!(
            draw::resolve_fill_color(Some(scale), Some("b"), None, theme_fallback, false).0,
            theme_fallback,
            "an omitted category must still paint, in the fallback color"
        );
        // The listed categories keep a real scale color, distinct from fallback.
        assert_ne!(
            draw::resolve_fill_color(Some(scale), Some("a"), None, theme_fallback, false).0,
            theme_fallback,
        );

        // (3) The legend carries exactly the two listed categories.
        resync_categorical_legend_entries(&mut legend_entries, color_scale.as_ref());
        assert_eq!(legend_entries, vec![entry("a"), entry("c")]);

        // ── Full domain: nothing omitted, nothing reported ───────────────────
        let mut full = Some(categorical_scale(&data_categories));
        let warnings = apply_color_config_to_color_scale(
            &mut full,
            &ColorConfigSpec {
                domain: Some(
                    data_categories.iter().map(|c| serde_json::json!(c)).collect(),
                ),
                ..Default::default()
            },
        );
        assert!(warnings.is_empty(), "a domain covering every category is silent: {warnings:?}");
    }

    /// A category listed in `configure_color(domain=…)` but absent from the data
    /// is kept, matching positional explicit-domain behavior.
    #[test]
    fn color_config_domain_keeps_categories_absent_from_data() {
        use scale_resolve::ColorScale;
        let mut color_scale = Some(categorical_scale(&["a"]));
        let cfg = ColorConfigSpec {
            domain: Some(vec![serde_json::json!("a"), serde_json::json!("ghost")]),
            ..Default::default()
        };
        let warnings = apply_color_config_to_color_scale(&mut color_scale, &cfg);
        assert!(warnings.is_empty(), "a well-formed override must not warn: {warnings:?}");
        match color_scale.as_ref().unwrap() {
            ColorScale::Categorical { domain, .. } => assert_eq!(domain, &["a", "ghost"]),
            other => panic!("expected Categorical, got {other:?}"),
        }
    }

    /// A numeric `configure_color(domain=…)` (the continuous form) leaves a
    /// categorical domain untouched rather than replacing it with junk.
    #[test]
    fn color_config_numeric_domain_leaves_categorical_domain_alone() {
        use scale_resolve::ColorScale;
        let mut color_scale = Some(categorical_scale(&["a", "b"]));
        let cfg = ColorConfigSpec {
            domain: Some(vec![serde_json::json!(0.0), serde_json::json!(1.0)]),
            ..Default::default()
        };
        let warnings = apply_color_config_to_color_scale(&mut color_scale, &cfg);
        assert!(warnings.is_empty(), "a well-formed override must not warn: {warnings:?}");
        match color_scale.as_ref().unwrap() {
            ColorScale::Categorical { domain, .. } => assert_eq!(domain, &["a", "b"]),
            other => panic!("expected Categorical, got {other:?}"),
        }
    }

    /// The legend entries follow the reordered domain, so labels and swatch
    /// colors stay in agreement after a `configure_color(domain=…)` override.
    /// The resync is inert when the entries already match the domain.
    #[test]
    fn legend_entries_resync_to_the_overridden_categorical_domain() {
        use crate::layout::{LegendEntry, SymbolKind};
        let entry = |label: &str, symbol| LegendEntry { label: label.into(), symbol };
        let mut entries = vec![
            entry("a", SymbolKind::Square),
            entry("b", SymbolKind::Circle),
            entry("c", SymbolKind::Circle),
        ];
        let unchanged = entries.clone();

        let scale = categorical_scale(&["a", "b", "c"]);
        resync_categorical_legend_entries(&mut entries, Some(&scale));
        assert_eq!(entries, unchanged, "matching domain must leave entries untouched");

        let reordered = categorical_scale(&["c", "a"]);
        resync_categorical_legend_entries(&mut entries, Some(&reordered));
        assert_eq!(
            entries,
            vec![entry("c", SymbolKind::Circle), entry("a", SymbolKind::Square)],
            "entries follow the new domain and each keeps its own symbol"
        );

        // No color scale (conditional-color legends) and no entries: both inert.
        let mut conditional_entries = vec![entry("x", SymbolKind::Circle)];
        resync_categorical_legend_entries(&mut conditional_entries, None);
        assert_eq!(conditional_entries, vec![entry("x", SymbolKind::Circle)]);
        let mut empty: Vec<LegendEntry> = Vec::new();
        resync_categorical_legend_entries(&mut empty, Some(&reordered));
        assert!(empty.is_empty(), "a suppressed legend stays suppressed");
    }

    /// `configure_color(range=…)` repaints a discretizing scale's swatches when
    /// the range describes the same partition. A count mismatch leaves the
    /// swatches alone and is **reported**, naming both counts — spec §4.2
    /// (amended 2026-08-28) forbids a silent drop here.
    #[test]
    fn color_config_range_repaints_discretizing_swatches() {
        use scale_resolve::{ColorScale, DiscretizedColors};
        let black = color::from_rgba(0, 0, 0, 255);
        let buckets = DiscretizedColors::new(vec![0.0, 1.0, 2.0], vec![black, black]).unwrap();
        let mut color_scale = Some(ColorScale::Discretizing(buckets));

        // One color for a 2-bucket partition: swatches stand, and the refusal
        // is warned with both counts.
        let warnings = apply_color_config_to_color_scale(
            &mut color_scale,
            &ColorConfigSpec { range: Some(vec!["#ffffff".into()]), ..Default::default() },
        );
        assert_eq!(color_scale.as_ref().unwrap().lookup_f64(0.5).unwrap().red, 0);
        assert_eq!(
            warnings,
            vec![RenderWarning::ColorRangeBucketCountMismatch { expected: 2, received: 1 }],
        );
        assert_eq!(
            warnings[0].to_string(),
            "color range names 1 color(s) but the binned color scale has 2 bucket(s); \
             the range was not applied",
        );

        let warnings = apply_color_config_to_color_scale(
            &mut color_scale,
            &ColorConfigSpec {
                range: Some(vec!["#ffffff".into(), "#ffffff".into()]),
                ..Default::default()
            },
        );
        assert_eq!(color_scale.as_ref().unwrap().lookup_f64(0.5).unwrap().red, 255);
        assert!(warnings.is_empty(), "a matching range must not warn: {warnings:?}");
    }

    /// Batch A Task 8 sweep: all three `configure_color(range=…)` arms —
    /// continuous gradient, categorical palette, discretizing swatches — were
    /// hex-only. A CSS name or `rgb()` string was dropped by the first two
    /// (shortening the gradient / palette, or making the override a silent
    /// no-op) and reported as a parse failure by the third. All three now
    /// resolve the full vocabulary, and every spelling of one color produces
    /// the identical resolved color.
    ///
    /// One test rather than three: the three arms share a single sweep and a
    /// single parser, so pinning them together is what keeps a future
    /// re-narrowing of *any* of them visible.
    #[test]
    fn color_config_range_accepts_named_and_rgb_forms_on_every_arm() {
        use scale_resolve::{ColorScale, DiscretizedColors};
        use crate::render::color::{ContinuousScheme, NamedContinuous};

        // steelblue / tomato, in the three spellings that must agree.
        let spellings: [[&str; 2]; 3] = [
            ["steelblue", "tomato"],
            ["rgb(70, 130, 180)", "rgb(255, 99, 71)"],
            ["#4682b4", "#ff6347"],
        ];
        let expected = [
            color::from_rgb(0x46, 0x82, 0xb4),
            color::from_rgb(0xff, 0x63, 0x47),
        ];
        let range = |pair: [&str; 2]| ColorConfigSpec {
            range: Some(pair.iter().map(|s| s.to_string()).collect()),
            ..Default::default()
        };

        // ── Continuous: the range becomes an evenly spaced gradient. ────────
        for pair in spellings {
            let mut scale = Some(ColorScale::Continuous {
                domain: (0.0, 1.0),
                scheme: ContinuousScheme::Named(NamedContinuous::Viridis),
                midpoint: None,
            });
            let warnings = apply_color_config_to_color_scale(&mut scale, &range(pair));
            assert!(warnings.is_empty(), "{pair:?} must not warn: {warnings:?}");
            match scale {
                Some(ColorScale::Continuous {
                    scheme: ContinuousScheme::Gradient(stops), ..
                }) => {
                    assert_eq!(
                        stops.iter().map(|&(_, c)| c).collect::<Vec<_>>(),
                        expected,
                        "{pair:?} must build the same gradient stops"
                    );
                }
                other => panic!("{pair:?}: expected a Gradient scheme, got {other:?}"),
            }
        }

        // ── Categorical: the range replaces the palette. ────────────────────
        for pair in spellings {
            let mut scale = Some(categorical_scale(&["a", "b"]));
            let warnings = apply_color_config_to_color_scale(&mut scale, &range(pair));
            assert!(warnings.is_empty(), "{pair:?} must not warn: {warnings:?}");
            match scale {
                Some(ColorScale::Categorical { palette, .. }) => {
                    assert_eq!(palette.as_ref(), &expected, "{pair:?} must set the same palette");
                }
                other => panic!("{pair:?}: expected Categorical, got {other:?}"),
            }
        }

        // ── Discretizing: the range repaints the bucket swatches. ───────────
        let black = color::from_rgba(0, 0, 0, 255);
        for pair in spellings {
            let mut scale = Some(ColorScale::Discretizing(
                DiscretizedColors::new(vec![0.0, 1.0, 2.0], vec![black; 2]).unwrap(),
            ));
            let warnings = apply_color_config_to_color_scale(&mut scale, &range(pair));
            assert!(warnings.is_empty(), "{pair:?} must not warn: {warnings:?}");
            let scale = scale.unwrap();
            assert_eq!(
                [scale.lookup_f64(0.5).unwrap(), scale.lookup_f64(1.5).unwrap()],
                expected,
                "{pair:?} must repaint both buckets alike"
            );
        }
    }

    /// The discretizing `range` parse is all-or-nothing. Filtering unparseable
    /// entries out *before* the count check would let a too-long range "fit"
    /// after the bad entry is dropped — silently repainting the buckets with a
    /// shifted mapping — and would misreport a too-short one as a count
    /// mismatch. Both shapes must report the parse failure and leave the
    /// swatches untouched.
    #[test]
    fn color_config_range_on_discretizing_scale_parses_all_or_nothing() {
        use scale_resolve::{ColorScale, DiscretizedColors};
        let black = color::from_rgba(0, 0, 0, 255);
        let three_buckets = || {
            Some(ColorScale::Discretizing(
                DiscretizedColors::new(vec![0.0, 1.0, 2.0, 3.0], vec![black; 3]).unwrap(),
            ))
        };
        let parse_failure = |entry: &str| {
            vec![RenderWarning::ColorRangeParseFailure { entry: entry.to_string() }]
        };

        // 4 entries, one unparseable: the survivors would number exactly 3 and
        // would have been accepted, repainting bucket 2 with the 4th color.
        let mut scale = three_buckets();
        let warnings = apply_color_config_to_color_scale(
            &mut scale,
            &ColorConfigSpec {
                range: Some(vec![
                    "#ff0000".into(),
                    "bogus".into(),
                    "#0000ff".into(),
                    "#00ff00".into(),
                ]),
                ..Default::default()
            },
        );
        assert_eq!(warnings, parse_failure("bogus"), "must report the parse, not repaint");
        for probe in [0.5, 1.5, 2.5] {
            assert_eq!(
                scale.as_ref().unwrap().lookup_f64(probe).unwrap().red,
                0,
                "bucket at {probe} must keep its resolved swatch"
            );
        }

        // 3 entries, one unparseable: a post-filter count check would call this
        // a 2-vs-3 mismatch, misattributing a parse failure.
        let mut scale = three_buckets();
        let warnings = apply_color_config_to_color_scale(
            &mut scale,
            &ColorConfigSpec {
                range: Some(vec!["#ff0000".into(), "nope".into(), "#0000ff".into()]),
                ..Default::default()
            },
        );
        assert_eq!(warnings, parse_failure("nope"));
        assert_eq!(scale.as_ref().unwrap().lookup_f64(0.5).unwrap().red, 0);
    }

    /// The categorical `range` parse is all-or-nothing too, for a reason the
    /// Discretizing arm's rationale does not cover: a categorical palette is
    /// indexed by DOMAIN POSITION (`lookup` → `palette[i % palette.len()]`), so
    /// dropping an unparseable entry does not shorten a cycling palette — it
    /// re-points every category after the dropped one.
    ///
    /// RED (verified in place): with the former
    /// `filter_map(|s| parse_color(s).ok())`,
    /// `range=["red", "notacolor", "blue"]` over domain `[a, b, c]` parsed to
    /// `[red, blue]` and rendered `a=red`, `b=blue`, `c=red` — silently wrong,
    /// with no warning, under a doc comment asserting this could not happen.
    /// The assertion is per-category (which category gets which color), not a
    /// palette-length check, so a fix that merely shortened the palette would
    /// still fail it.
    #[test]
    fn color_config_range_on_categorical_scale_parses_all_or_nothing() {
        let red = color::parse_color("red").unwrap();
        let blue = color::parse_color("blue").unwrap();

        // Control: an all-valid range applies, in listed order.
        let mut scale = Some(categorical_scale(&["a", "b", "c"]));
        let warnings = apply_color_config_to_color_scale(
            &mut scale,
            &ColorConfigSpec {
                range: Some(vec!["red".into(), "green".into(), "blue".into()]),
                ..Default::default()
            },
        );
        assert!(warnings.is_empty(), "a valid range must not warn: {warnings:?}");
        let applied = scale.as_ref().unwrap();
        assert_eq!(applied.lookup("a"), Some(red), "a must take the 1st entry");
        assert_eq!(applied.lookup("c"), Some(blue), "c must take the 3rd entry");

        // One unparseable entry: the whole range is discarded and reported.
        let resolved = categorical_scale(&["a", "b", "c"]);
        let (before_a, before_b, before_c) =
            (resolved.lookup("a"), resolved.lookup("b"), resolved.lookup("c"));
        let mut scale = Some(resolved);
        let warnings = apply_color_config_to_color_scale(
            &mut scale,
            &ColorConfigSpec {
                range: Some(vec!["red".into(), "notacolor".into(), "blue".into()]),
                ..Default::default()
            },
        );
        assert_eq!(
            warnings,
            vec![RenderWarning::ColorRangeParseFailure { entry: "notacolor".into() }],
            "one bad entry must be reported, not silently dropped"
        );
        let kept = scale.as_ref().unwrap();
        assert_eq!(
            [kept.lookup("a"), kept.lookup("b"), kept.lookup("c")],
            [before_a, before_b, before_c],
            "the resolved palette must be left untouched; the old filter_map gave \
             a=red, b=blue, c=red — a shifted mapping, not a shortened cycle"
        );
        assert_ne!(
            kept.lookup("b"),
            Some(blue),
            "b must NOT inherit the third entry's color — that is the shift this \
             test exists to catch"
        );
    }
}

/// Pipeline-level ordering tests (#143 remediation, mutation review M2–M6).
///
/// Every test here asserts THROUGH the production entry — `apply_chart_config_pipeline`
/// or the tier fn that owns the order under test — and never hand-sequences the
/// helpers in its own body. That distinction is the whole point: the pre-existing
/// unit tests baked the correct call order into the test, which made them
/// precedence tests of the HELPERS and left the PIPELINE's order unobserved. Each
/// test below was RED-proven against the specific reorder named in its doc.
mod pipeline_order_tests {
    use super::*;
    use crate::layout::ThemeInputs;
    use crate::render::chart_config::{
        AxisConfigSpec, AxisStyleSpec, ChartConfig, ColorConfigSpec, GridAxisSpec, GridConfigSpec,
        LegendConfigSpec, LegendStyleSpec,
    };
    use crate::render::prepare;
    // The bare two-axis `AxesInput` the sibling mod's per-axis tests use.
    use super::chart_config_application_tests::blank_axes;
    use crate::spec::chart::ChartSpec;
    use crate::spec::data_ref::DataRef;
    use crate::spec::encoding::{Encoding, EncodingSpec};
    use crate::spec::mark::Mark;
    use arrow::array::{Float64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use std::sync::Arc;

    /// How the `color` channel is typed, which decides whether `prepare` builds a
    /// categorical entry list or a continuous colorbar.
    enum ColorKind {
        /// A `Utf8` category column — yields `legend_entries`.
        Categorical,
        /// A `Float64` column — yields a `colorbar`, the arm M6 showed uncovered.
        Continuous,
        None,
    }

    /// x/y plus an optional color channel and an optional `size` channel (a
    /// distinct field, so `prepare` builds a real aux legend rather than folding
    /// size into the color legend). `color_legend` wires a per-channel
    /// `Legend(...)` onto the color encoding — the LEVEL-2 side of every cascade
    /// asserted here.
    fn chart(
        color: ColorKind,
        color_legend: Option<LegendStyleSpec>,
        with_size: bool,
    ) -> (ChartSpec, RecordBatch) {
        let mut fields = vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
        ];
        let mut columns: Vec<arrow::array::ArrayRef> = vec![
            Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])),
            Arc::new(Float64Array::from(vec![10.0, 20.0, 30.0])),
        ];
        match color {
            ColorKind::Categorical => {
                fields.push(Field::new("kind", DataType::Utf8, false));
                columns.push(Arc::new(StringArray::from(vec!["a", "b", "c"])));
            }
            ColorKind::Continuous => {
                fields.push(Field::new("kind", DataType::Float64, false));
                columns.push(Arc::new(Float64Array::from(vec![1.0, 5.0, 9.0])));
            }
            ColorKind::None => {}
        }
        if with_size {
            fields.push(Field::new("mass", DataType::Float64, false));
            columns.push(Arc::new(Float64Array::from(vec![2.0, 4.0, 6.0])));
        }
        let batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap();
        let color_enc = match color {
            ColorKind::None => None,
            _ => Some(EncodingSpec {
                field: "kind".into(),
                type_: None,
                legend: color_legend.map(Box::new),
                ..Default::default()
            }),
        };
        let spec = ChartSpec {
            data: DataRef::default(),
            mark: Mark::Point,
            encoding: Encoding {
                x: Some(EncodingSpec { field: "x".into(), type_: None, ..Default::default() }),
                y: Some(EncodingSpec { field: "y".into(), type_: None, ..Default::default() }),
                color: color_enc,
                size: with_size.then(|| EncodingSpec {
                    field: "mass".into(),
                    type_: None,
                    ..Default::default()
                }),
                ..Default::default()
            },
            transforms: Vec::new(),
            facet: None,
            layers: None,
            coord: None,
            mark_style: None,
            position: None,
            title: None,
            axis_x: None,
            axis_y: None,
            selections: Vec::new(),
            conditionals: Vec::new(),
            chart_description: None,
            params: Vec::new(),
        };
        (spec, batch)
    }

    /// Run the real pipeline over a real `PreparedInputs`, returning both halves
    /// so a test can assert on the mutated `prep` and on the returned bundle.
    fn run_pipeline(
        spec: &ChartSpec,
        batch: &RecordBatch,
        chart_config: &ChartConfig,
    ) -> (prepare::PreparedInputs, AppliedChartConfig, Vec<RenderWarning>) {
        let theme = ThemeInputs::default();
        let mut prep =
            prepare::prepare_render_inputs(spec, batch, &theme, chart_config, None).unwrap();
        let mut warnings = prep.warnings.clone();
        let applied =
            apply_chart_config_pipeline(&mut prep, &theme, chart_config, &mut warnings).unwrap();
        (prep, applied, warnings)
    }

    /// **M2.** The `configure_grid(x=…)` layer sits BETWEEN `axis_x`/`axis_y` and
    /// the shared `axis`, so the documented middle of the cascade is
    /// `axis_x > grid.x > axis`. The pre-existing ordering tests pinned only the
    /// OUTER boundary (`axis_x` vs `axis`) because none of them set
    /// `chart_config.grid`, and the grid tests that would discriminate call
    /// `apply_grid_config_to_axis_inputs` directly — bypassing the tier fn that
    /// owns the order. Asserted here through `fill_axis_slots_specific_before_shared`.
    ///
    /// RED against: hoisting `apply_grid_config_to_axis_inputs` above the
    /// `axis_x`/`axis_y` fills.
    #[test]
    fn grid_layer_sits_between_per_axis_and_shared_axis() {
        let config = ChartConfig {
            // Least specific: loses on both axes.
            axis: Some(AxisConfigSpec {
                style: AxisStyleSpec { grid_color: Some("#00ff00".into()), ..Default::default() },
                ..Default::default()
            }),
            // Most specific: wins on x.
            axis_x: Some(AxisConfigSpec {
                style: AxisStyleSpec { grid_color: Some("#0000ff".into()), ..Default::default() },
                ..Default::default()
            }),
            grid: Some(GridConfigSpec {
                // Middle layer: loses to `axis_x` on x …
                x: Some(GridAxisSpec { color: Some("#ff0000".into()), ..Default::default() }),
                // … and wins over the shared `axis` on y, where no `axis_y` exists.
                y: Some(GridAxisSpec { color: Some("#ff0000".into()), ..Default::default() }),
                ..Default::default()
            }),
            ..Default::default()
        };
        let mut axes = blank_axes();
        fill_axis_slots_specific_before_shared(&mut axes, &config, &mut Vec::new()).unwrap();
        let rgb = |a: &crate::layout::AxisInput| a.overrides.grid_color.map(|c| [c.red, c.green, c.blue]);
        assert_eq!(
            rgb(&axes.x),
            Some([0, 0, 255]),
            "axis_x (blue) must beat grid.x (red): the per-axis AXIS section is more \
             specific than the per-axis GRID section"
        );
        assert_eq!(
            rgb(&axes.y),
            Some([255, 0, 0]),
            "grid.y (red) must beat the shared axis (green): the grid layer is more \
             specific than the axis-unspecified shorthand"
        );
    }

    /// **M3.** The tick products (`label_format` re-formatting, `tick_extra`,
    /// `tick_min_step`, projected fractions) must be re-derived AFTER the axis
    /// merge, or a chart-level `configure_axis(label_format=…)` is silently
    /// dropped — it would not yet be on `AxisInput.overrides` when the re-sync
    /// reads it. The pre-existing label-format tests hand-sequence
    /// `apply_axis_config_to_axis_input` then `apply_label_format_to_axis`, so
    /// they bake the correct order into the test body and cannot observe a
    /// production reorder. Asserted here through `apply_chart_config_pipeline`.
    ///
    /// RED against: hoisting `resync_ticks_after_axis_merge` above
    /// `fill_axis_slots_specific_before_shared`.
    #[test]
    fn chart_level_label_format_survives_because_ticks_resync_after_the_axis_merge() {
        let (spec, batch) = chart(ColorKind::None, None, false);
        let config = ChartConfig {
            axis_y: Some(AxisConfigSpec {
                label_format_raw: Some(".2f".into()),
                ..Default::default()
            }),
            ..Default::default()
        };
        let (prep, _applied, _warnings) = run_pipeline(&spec, &batch, &config);
        assert!(
            prep.axes.y.tick_labels.iter().all(|l| l.contains('.')
                && l.split('.').nth(1).is_some_and(|frac| frac.len() == 2)),
            "every y tick label must carry the chart-level `.2f` format; got {:?} — \
             the re-sync ran before the axis merge, so `label_format` was not yet \
             on the axis when the labels were rebuilt",
            prep.axes.y.tick_labels
        );
    }

    /// **M4.** The `Legend(values=[…])` filter must run AFTER the categorical
    /// entry rebuild, because the rebuild reconstructs `legend_entries` wholesale
    /// from the overridden domain and would otherwise undo the filter — the A-B-A
    /// the tier fn's own doc calls load-bearing. The pre-existing tests exercise
    /// each pass in isolation; nothing exercised the PAIR. Asserted here through
    /// `apply_chart_config_pipeline` with both layers set at once.
    ///
    /// RED against: moving `apply_legend_values_to_entries` above
    /// `resync_categorical_legend_entries`.
    #[test]
    fn legend_values_filter_survives_the_color_domain_rebuild() {
        let (spec, batch) = chart(
            ColorKind::Categorical,
            Some(LegendStyleSpec {
                values: Some(vec!["c".into(), "a".into()]),
                ..Default::default()
            }),
            false,
        );
        let config = ChartConfig {
            // Reorders/refreshes the domain, which rebuilds every entry.
            color: Some(ColorConfigSpec {
                domain: Some(vec!["c".into(), "b".into(), "a".into()]),
                ..Default::default()
            }),
            ..Default::default()
        };
        let (prep, _applied, _warnings) = run_pipeline(&spec, &batch, &config);
        let labels: Vec<&str> = prep.legend_entries.iter().map(|e| e.label.as_str()).collect();
        assert_eq!(
            labels,
            vec!["c", "a"],
            "the `Legend(values=)` filter must be the LAST word on the entry set; \
             the full rebuilt domain here means the rebuild ran after the filter \
             and undid it"
        );
    }

    /// **M5.** `resolve_leaf_legend_overrides` projects the per-channel (level 2)
    /// `Legend(...)` bundle before `configure_legend` (level 3) fills what is
    /// still unset. The pre-existing cascade tests build their level-2 side as a
    /// hand-written `LegendOverrides` literal and call
    /// `apply_chart_config_to_legend_overrides` directly, so the projection itself
    /// — the sub-entry #143 extracted to de-duplicate `composite_render` — was
    /// never asserted. Asserted here through `apply_chart_config_pipeline` on a
    /// real `prep`, on both a field the chart level also sets (per-channel must
    /// win) and one only the per-channel level sets (must survive at all).
    ///
    /// RED against: `legend_overrides_from_prep(prep)` → `LegendOverrides::default()`.
    #[test]
    fn per_channel_legend_fields_reach_the_bundle_through_the_projection() {
        let (spec, batch) = chart(
            ColorKind::Categorical,
            Some(LegendStyleSpec {
                symbol_type: Some("square".into()),
                symbol_size: Some(77.0),
                ..Default::default()
            }),
            false,
        );
        let config = ChartConfig {
            legend: Some(LegendConfigSpec {
                style: LegendStyleSpec {
                    // Level 3 must LOSE to the per-channel value above …
                    symbol_type: Some("triangle".into()),
                    // … and fill this one, which level 2 leaves unset.
                    row_padding: Some(9.0),
                    ..Default::default()
                },
            }),
            ..Default::default()
        };
        let (_prep, applied, _warnings) = run_pipeline(&spec, &batch, &config);
        assert_eq!(
            applied.legend_overrides.symbol_type.as_deref(),
            Some("square"),
            "per-channel `Legend(symbol_type=)` must survive the projection and beat \
             `configure_legend`"
        );
        assert_eq!(
            applied.legend_overrides.style.symbol_size,
            Some(77.0),
            "a per-channel field the chart level never mentions must still reach the \
             bundle — a dropped projection is invisible to any test that only checks \
             contested fields"
        );
        assert_eq!(
            applied.legend_overrides.style.row_padding,
            Some(9.0),
            "`configure_legend` must still fill what level 2 left unset"
        );
    }

    /// **M6.** Chart-level `configure_legend(orient="none")` clears the whole
    /// legend bundle, not just the categorical entries. No Rust test reached that
    /// body at all — the existing test asserts on the
    /// `chart_config_legend_disabled` PREDICATE and stops there — and no Python
    /// test covers the continuous arm either, so a mutant deleting
    /// `colorbar = None` / `aux_legends.clear()` left a suppressed chart still
    /// drawing its colorbar. `prepare::legend`'s coupling comment ("if a new
    /// legend-content field is added, wire it into that clear too") demands a
    /// test on the far side; this is it.
    ///
    /// RED against: deleting `prep.colorbar = None` and `prep.aux_legends.clear()`
    /// from `suppress_legend_if_chart_level_disabled`.
    #[test]
    fn chart_level_suppression_clears_colorbar_and_aux_legends_not_just_entries() {
        // Continuous color → a colorbar; a distinct size field → an aux legend.
        let (spec, batch) = chart(ColorKind::Continuous, None, true);

        // Guard: without suppression this chart really does carry both, so the
        // assertions below cannot pass vacuously.
        let (baseline, _, _) = run_pipeline(&spec, &batch, &ChartConfig::default());
        assert!(baseline.colorbar.is_some(), "fixture must build a colorbar to suppress");
        assert!(!baseline.aux_legends.is_empty(), "fixture must build an aux legend to suppress");

        let config = ChartConfig {
            legend: Some(LegendConfigSpec {
                style: LegendStyleSpec { orient: Some("none".into()), ..Default::default() },
            }),
            ..Default::default()
        };
        let (prep, applied, _warnings) = run_pipeline(&spec, &batch, &config);
        assert!(prep.colorbar.is_none(), "the colorbar must be cleared, not just the entries");
        assert!(prep.aux_legends.is_empty(), "the size/shape aux legends must be cleared too");
        assert!(prep.legend_entries.is_empty(), "categorical entries must be cleared");
        assert!(prep.legend_title.is_none(), "the legend title must be cleared");
        assert!(
            applied.legend_title.is_none(),
            "and the cleared title must reach the layout bundle, not just `prep`"
        );
    }
}
