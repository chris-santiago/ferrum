# Pipeline Audit Findings — 2026-05-12

Three independent Opus agents audited the full Python→Rust→SVG pipeline. This is the union-pooled, deduplicated superset of all findings. Phase 10 is complete — nothing here should be deferred.

---

## Tier 1: CODE BUGS (broken serialization / logic errors)

| # | Finding | File(s) | Impact |
|---|---------|---------|--------|
| B1 | `_build_layers_list()` checks `d.get("type")` but `to_encoding_spec_dict()` emits `"type_"` — data type silently dropped for ALL layer encodings in every composite mark | `chart.py:4137`, `base.py:139` | **FIXED in 00bae5d** |
| B2 | `_build_layers_list()` checks `"formatType"` but dict has `"format_type"` — format_type dropped for layers | `chart.py:4144`, `base.py:163` | Low (format_type not consumed yet) |
| B3 | `_build_layers_list()` opt_keys missing `axis`, `legend`, `sort`, `stack`, `impute` — these encoding fields silently dropped for all layers | `chart.py:4139-4148` | Medium — `legend=False` suppression fails on composite charts |
| B4 | `ScaleSpec::Linear.zero` destructured with `..`, explicitly skipped — `zero=True` on a linear scale does nothing | `scale_resolve.rs:599` | **Rendering correctness bug** |

## Tier 2: DEAD STRUCT FIELDS (deserialized, never consumed by renderer)

### TitleSpec (6 dead fields — per-chart overrides silently ignored)

| # | Field | File | What happens |
|---|-------|------|-------------|
| D1 | `TitleSpec.anchor` | `title.rs:11` | Renderer uses `theme.title_anchor` — per-title `anchor="middle"` silently dropped |
| D2 | `TitleSpec.offset` | `title.rs:13` | Renderer uses `theme.title_offset` |
| D3 | `TitleSpec.font_size` | `title.rs:15` | Renderer uses `theme.title_font_size` |
| D4 | `TitleSpec.font_weight` | `title.rs:17` | Renderer uses `theme.title_font_weight` |
| D5 | `TitleSpec.color` | `title.rs:19` | Renderer uses `theme.title_color` |
| D6 | `TitleSpec.subtitle_color` | `title.rs:23` | Renderer uses `theme.font_color` for subtitle |

### EncodingSpec (5 dead/partial fields)

| # | Field | File | What happens |
|---|-------|------|-------------|
| D7 | `axis: Option<AxisSpec>` | `encoding.rs:222` | Deserialized into opaque bag, never read by layout/render |
| D8 | `sort: Option<Value>` | `encoding.rs:226` | Deserialized, never read |
| D9 | `stack: Option<String>` | `encoding.rs:228` | Deserialized, never read (stacking via `PositionAdjust` instead) |
| D10 | `impute: Option<Value>` | `encoding.rs:230` | Deserialized, never read |
| D11 | `format_type: Option<String>` | `encoding.rs:235` | Deserialized, never read |

### EncodingSpec partial wirings

| # | Field | File | What happens |
|---|-------|------|-------------|
| D12 | `format: Option<String>` | `encoding.rs:234` | Only read by `marks/text.rs:72` for text channel — NOT honored for axis tick labels |
| D13 | `legend: Option<LegendSpec>` | `encoding.rs:224` | Only `disabled` key read at `prepare.rs:337` — all other legend styling keys dropped |

### ThemeInputs (4 dead fields)

| # | Field | File | What happens |
|---|-------|------|-------------|
| D14 | `font_weight` | `mod.rs:151` | Set via binding, never read (distinct from `title_font_weight` which works) |
| D15 | `diverging_scheme` | `mod.rs:173` | Set via binding, never read by scale resolution |
| D16 | `reference_line_color` | `mod.rs:180` | Set via binding, never read by any mark renderer |
| D17 | `reference_line_dash` | `mod.rs:181` | Set via binding, never read by any mark renderer |

### MarkKwargsSpec / RenderConfig

| # | Field | File | What happens |
|---|-------|------|-------------|
| D18 | `MarkKwargsSpec.baseline` | `mark_style.rs:31` | Set on MarkStyle by `draw.rs:147`, never read by any mark renderer |
| D19 | `RenderConfig.embed_fonts` | `config.rs:15` | Settable via binding, `render/mod.rs:211` hardcodes `true` |

## Tier 3: SILENT_DROP MARK KWARGS (accepted by Python, never serialized to Rust)

These are in `_VALID_MARK_KWARGS` (accepted without error) but NOT in `to_mark_kwargs_dict()` and have no `MarkKwargsSpec` field.

| # | Kwarg | Used internally by | Impact |
|---|-------|-------------------|--------|
| S1 | `interpolate` | line/area marks | `mark_line(interpolate="step")` silently ignored |
| S2 | `stroke_cap` | line marks | `mark_line(stroke_cap="round")` silently ignored |
| S3 | `stroke_join` | line/area marks | silently ignored |
| S4 | `orient` | bar/tick marks | silently ignored (orientation via CoordFlip) |
| S5 | `filled` | point marks, boxplot outliers (`composite.py:136`) | `mark_point(filled=False)` ignored — boxplot outliers render filled despite intent |
| S6 | `shape` (constant) | point marks | `mark_point(shape="diamond")` ignored — only encoding-driven shape works |
| S7 | `limit` | text marks | silently ignored |
| S8 | `band_size` | tick marks, boxplot caps (`composite.py:123`), errorbar caps (`composite.py:213`) | Tick caps render full-width instead of narrow — **visible boxplot/errorbar defect** |
| S9 | `line` | area marks | `mark_area(line=True)` ignored |
| S10 | `borders` | area/errorband | silently ignored |
| S11 | `width` | boxplot IQR rect (`composite.py:125`) | Box width silently dropped — uses default rect width |

## Tier 4: WARN_UNIMPLEMENTED ENCODING CHANNELS (18 channels still warning)

These channels have `_renders_in_phase_8a = False` and trigger `warn_once()`. Phase 10 is complete — these should all work or be removed.

| # | Channel | File | Notes |
|---|---------|------|-------|
| W1 | `Fill` | `appearance.py:156` | |
| W2 | `Stroke` | `appearance.py:187` | |
| W3 | `FillOpacity` | `appearance.py:216` | |
| W4 | `StrokeOpacity` | `appearance.py:246` | |
| W5 | `StrokeWidth` | `appearance.py:276` | |
| W6 | `StrokeDash` | `appearance.py:305` | |
| W7 | `Angle` | `appearance.py:335` | |
| W8 | `Detail` | `text.py:44` | Used internally by polygon grouping |
| W9 | `Tooltip` | `text.py:74` | Static SVG — may not apply |
| W10 | `Href` | `text.py:152` | Static SVG — may not apply |
| W11 | `Description` | `text.py:182` | Accessibility — should be in SVG |
| W12 | `Key` | `text.py:211` | Animation key — may not apply to static |
| W13-16 | `XError`, `YError`, `XError2`, `YError2` | `positional.py` | Error bar encodings |
| W17 | `Theta` | `positional.py:270` | Polar — may not apply to static cartesian |
| W18 | `Radius` | `positional.py:307` | Polar — may not apply to static cartesian |

## Tier 5: CHART-LEVEL GAPS

| # | Finding | File | Impact |
|---|---------|------|--------|
| G1 | `Chart(description=...)` accepted, stored in `_description`, never serialized to spec or emitted in SVG | `chart.py` | Accessibility gap |

## Tier 6: DESUGAR SILENT DELETES (mark params accepted then `del`'d)

These are accepted by mark constructors but explicitly deleted in desugar functions. Users pass them expecting an effect. Lower priority — many are "informational" params that the chart builder already consumed.

| # | Param | Mark | File |
|---|-------|------|------|
| X1 | `kernel` | density | `statistical.py:135` — only Gaussian supported |
| X2 | `right` | histogram | `statistical.py:257` — always left-closed |
| X3 | `multiple` | histogram | `statistical.py:257` — silently deleted |
| X4 | `interpolate` | ribbon | `composite.py:348` — always linear |
| X5 | `stroke`/`stroke_width` | hex | `heavy_stat.py:476` — no-op |
| X6 | `clip` | function | `heavy_stat.py:656` — no-op |

## Tier 7: STALE DOCSTRINGS

| # | Finding | File |
|---|---------|------|
| L1 | X/Y docstrings say `axis` is "reserved for future use (no-op today)" — stale from Phase 8a | `positional.py:35` |
| L2 | Color docstring says `legend` is "reserved" — it IS partially honored (`disabled` key) | `appearance.py:36` |
