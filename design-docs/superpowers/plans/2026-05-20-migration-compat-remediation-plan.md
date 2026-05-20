# Migration Compatibility Remediation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

## 1. Objective

Fix 33 audit findings (B1–B7, K1–K3, K5–K7, F1–F4, F13–F17, M2–M3, M5, D1–D3) from the plotnine/altair migration gap audit so migrating users hit zero silent-failure bugs and can accomplish all common workflows.

## 2. Spec references

- `design-docs/superpowers/audits/2026-05-20-plotnine-altair-migration-audit.md` — full finding details
- `ferrum-spec.md §3.2` — channel shorthand grammar
- `ferrum-spec.md §3.13` — theme/scale keys

## 3. Files

| Action | Path | Reason |
|--------|------|--------|
| Modify | `src/ferrum/encoding/base.py` | B1 type\_ alias, B2 count() field, B3 auto-groupby |
| Modify | `src/ferrum/encoding/_scale.py` | B4 nice= serialization, F16 reverse= |
| Modify | `src/ferrum/chart.py` | B3 groupby inference in encode(), B5/D2 mark\_smooth docstring, D3 mark\_tick docstring, F1 labs(), F2 xlim/ylim, F4 mark\_circle/mark\_square, K3 point= on mark\_line, F15 to\_dict() |
| Modify | `src/ferrum/marks/base.py` | K1 color→fill alias, K2 alpha→opacity alias |
| Modify | `src/ferrum/_coerce.py` | K5 Series acceptance, K6 Duration cast, K7 Date32 cast (pyarrow path) |
| Modify | `src/ferrum/_render.py` | B7 config defaults, F13 scale= param on show\_png/save |
| Modify | `src/ferrum/composition.py` | F13 scale= param on show\_png |
| Modify | `src/ferrum/display.py` | F14 PDF export via resvg→PNG→PDF or SVG→PDF |
| Modify | `src/ferrum/marks/statistical.py` | M3 kernel support |
| Modify | `src/ferrum/config.py` | B7 wire defaults to render |
| Modify | `src/ferrum/axis.py` | F17 labels= mapping |
| Modify | `crates/ferrum-core/src/render/arrow_cast.rs` | B6 Int8–UInt64 in distinct\_values, K6 Duration in col\_as\_f64, K7 Date32/Date64 in col\_as\_f64 |
| Modify | `crates/ferrum-core/src/transform/aggregate.rs` | B2 allow empty field for count, M2 add variance/stdev/q1/q3/distinct |
| Modify | `crates/ferrum-core/src/transform/smooth.rs` | B5 accept "linear" as alias for "lm" |
| Modify | `crates/ferrum-core/src/transform/kde.rs` | M3 epanechnikov/tophat/cosine kernels |
| Create | `src/ferrum/annotations.py` (or modify existing) | M5 annotate\_abline |
| Create | `docs/site/comparison/altair.md` | D1 altair migration guide |
| Test | `tests/test_migration_compat.py` | umbrella test file for all fixes |

## 4. Constraints

- **No matplotlib.** PDF export must use resvg or cairosvg, not matplotlib.
- **No breaking changes.** `color=` and `alpha=` are _aliases_, not replacements — `fill=`/`stroke=`/`opacity=` remain canonical.
- **`mark_line(point=True)`** desugars to a two-layer chart (line + point) at resolve time, not a new mark. Do not add `point` to `_VALID_MARK_KWARGS` — handle it in `mark_line()` itself.
- **Auto-groupby (B3)** must be inferred at `_resolve_pending` / `to_spec()` time when all sibling channels are known, not at `encode()` time (channels may be added progressively).
- **Aggregate functions (M2):** add `variance`, `stdev`, `q1`, `q3`, `distinct` to the Rust `AggFn` enum. Do not add the full Vega-Lite set (argmin/argmax/values/missing/valid require non-numeric return types).
- **KDE kernels (M3):** epanechnikov, tophat, cosine are sufficient. The Rust `Kde` transform already has the bandwidth/grid logic — kernels are a weight function swap.
- **PDF export (F14):** use `resvg-py` (already a dev dep for golden snapshots) to rasterize SVG→PNG, then `fpdf2` or raw PDF wrapper to embed the PNG. If `resvg-py` or a PDF library is unavailable at runtime, raise `ImportError` with install instructions. Do not add a hard runtime dependency.
- **Coding agent dispatch:** Python changes → `python-coder` agent. Rust changes → `rust-coder` agent. Both in parallel when touching independent subsystems.

## 5. Tasks

### Task 1: Rust dtype & aggregate fixes (B2, B6, K6, K7, M2)
- [ ] `arrow_cast.rs`: add Int8/Int16/Int32/UInt8/UInt16/UInt32/UInt64 arms to `distinct_values_in_order` (cast to string via `as i64` then `.to_string()`)
- [ ] `arrow_cast.rs`: add Duration(ns/us/ms/s) arms to `col_as_f64` (cast to f64 nanoseconds)
- [ ] `arrow_cast.rs`: add Date32/Date64 arms to `col_as_f64` (cast to epoch-millis f64)
- [ ] `aggregate.rs`: allow `field=""` when `fn_==Count` (use row count, skip column lookup)
- [ ] `aggregate.rs`: add `Variance`, `Stdev`, `Q1`, `Q3`, `Distinct` to `AggFn` enum + `aggregate()` match + string parsing
- [ ] Verify: `cargo test -p ferrum-core`

### Task 2: Rust smooth & KDE fixes (B5, M3)
- [ ] `smooth.rs`: accept `"linear"`, `"quadratic"`, `"cubic"`, `"log"`, `"sqrt"` as aliases mapping to the corresponding polynomial/transform methods already available (or to `"lm"` for linear)
- [ ] `kde.rs`: add `kernel` field to `KdeSpec`, implement epanechnikov `(3/4)(1-u^2)`, tophat `0.5`, cosine `(pi/4)cos(pi*u/2)` weight functions alongside gaussian
- [ ] Verify: `cargo test -p ferrum-core`

### Task 3: Python encoding & channel fixes (B1, B3, B4)
- [ ] `encoding/base.py`: in `__init__`, normalize `type_` key to `type` in `_kwargs` so both forms work
- [ ] `encoding/base.py`: in `to_implicit_transforms()`, when aggregate is `count` and field is None, set field to `"*"` or skip field validation
- [ ] `encoding/_scale.py`: serialize `nice` key in `_scale_to_dict()` for all continuous scale branches
- [ ] `encoding/_scale.py`: serialize `reverse` key in `_scale_to_dict()` for all continuous scale branches (F16)
- [ ] `chart.py`: at `_resolve_pending` or `to_spec()`, infer `groupby` for implicit aggregate transforms from sibling non-aggregate encoding fields (B3)
- [ ] Verify: `uv run pytest tests/test_encoding.py tests/test_shorthand.py -v`

### Task 4: Mark kwargs aliases & features (K1, K2, K3, F3, F4)
- [ ] `marks/base.py`: add `color` and `alpha` to a `_MARK_KWARG_ALIASES` dict mapping `color→fill`, `alpha→opacity`. Resolve aliases in `__init__` before validation.
- [ ] `chart.py` `mark_line()`: intercept `point=True` kwarg, store it, and at resolve time desugar to `self + self.mark_point()` (two-layer chart). Remove `point` from kwargs before passing to `_set_mark`.
- [ ] `marks/base.py` or `chart.py`: add `linetype` to `_MARK_KWARG_ALIASES` mapping to `stroke_dash`, with a name lookup (`"dashed"→"4,2"`, `"dotted"→"1,3"`, `"dashdot"→"4,2,1,2"`, `"longdash"→"8,4"`, `"solid"→""`)
- [ ] `chart.py`: add `mark_circle(**kw)` and `mark_square(**kw)` as thin wrappers calling `mark_point(shape="circle", **kw)` / `mark_point(shape="square", **kw)` (F4)
- [ ] Verify: `uv run pytest tests/test_mark_kwargs_no_silent_drop.py -v`

### Task 5: Data coercion fixes (K5, K6, K7)
- [ ] `_coerce.py`: detect `pd.Series` / `pl.Series` before narwhals fallback, call `.to_frame()` (K5)
- [ ] `_coerce.py`: in polars fast path, cast `pl.Duration` to `pl.Int64` (nanoseconds) (K6)
- [ ] `_coerce.py`: in pyarrow path, cast `pa.date32()` / `pa.date64()` columns to `pa.timestamp("ms")` (K7)
- [ ] Verify: `uv run pytest tests/test_coerce.py -v`

### Task 6: Fluent API additions (F1, F2, F15, B7)
- [ ] `chart.py`: add `.labs(**kwargs)` method — for each `x=`, `y=`, `title=`, `subtitle=`, wrap in the appropriate channel title or chart title override (F1)
- [ ] `chart.py`: add `.xlim(lo, hi)` and `.ylim(lo, hi)` convenience methods that delegate to `.coord(CoordCartesian(xlim=..., ylim=...))` (F2)
- [ ] `chart.py`: add `.to_dict()` returning `json.loads(self.to_json())` (F15)
- [ ] `_render.py` + `config.py`: read `ferrum.config` defaults for width/height instead of hardcoded 600×400 (B7)
- [ ] Verify: `uv run pytest tests/test_migration_compat.py -v`

### Task 7: PNG scale & PDF export (F13, F14)
- [ ] `_render.py` `show_png()`: add `scale: float = 2.0` parameter, pass through to `render_png` or `rasterize_svg`
- [ ] `_render.py` `save()`: accept `scale=` for PNG format
- [ ] `composition.py` `show_png()`: add `scale: float = 2.0` param instead of hardcoded `2.0`
- [ ] `display.py` `save_chart()`: recognize `.pdf` extension — rasterize SVG via `resvg-py` then wrap in PDF (try `fpdf2`, fallback to minimal raw PDF). `ImportError` if neither available.
- [ ] Verify: `uv run pytest tests/test_migration_compat.py -k "png or pdf" -v`

### Task 8: Axis label remapping & annotate\_abline (F17, M5)
- [ ] `axis.py`: add `labels: dict[str, str] | None = None` param to `Axis`. Serialize as `"label_expr"` Vega expression or handle in Python pre-render by renaming column values (F17)
- [ ] `annotations.py` (or new): add `annotate_abline(slope, intercept, ...)` returning a Chart with `mark_line` over a computed two-point DataFrame spanning the x domain (M5)
- [ ] Export `annotate_abline` from `__init__.py`
- [ ] Verify: `uv run pytest tests/test_migration_compat.py -k "abline or label_remap" -v`

### Task 9: Docstring fixes & altair migration guide (D1, D2, D3)
- [ ] `chart.py` `mark_smooth()` docstring: replace `"linear"`, `"quadratic"`, `"cubic"`, `"log"`, `"sqrt"` with whatever method names are now valid after Task 2 lands (D2)
- [ ] `chart.py` `mark_tick()` docstring: fix `band_size` description from "Tick length in pixels" to "Tick length as fraction of band width (0–1)" (D3)
- [ ] Create `docs/site/comparison/altair.md` covering: Chart pattern, encode shorthand, mark mapping, transform API differences, composition operators, selection/condition, scale/axis, theme, save/export. Follow structure of existing `plotnine.md`. (D1)
- [ ] Verify: `nox -s docs` builds without warnings

## 6. Acceptance checks

- `unset CONDA_PREFIX && uv run --no-sync maturin develop` — builds clean
- `source ~/.cargo/env && DYLD_LIBRARY_PATH=... cargo test -p ferrum-core` — all pass
- `uv run pytest -n auto` — all pass, zero new warnings
- `nox -s docs` — builds without warnings (altair.md renders)
- Quick smoke tests:
  - `fm.Chart(df).mark_bar().encode(x="cat:N", y="count():Q")` renders a bar chart
  - `fm.Chart(df).mark_point(color="red", alpha=0.5).encode(x="x", y="y")` works
  - `fm.Chart(pd.Series([1,2,3], name="x")).mark_point().encode(x="x").show_svg()` works
  - `chart.save("out.pdf")` produces a valid PDF
  - `chart.labs(x="My X", title="My Title").show_svg()` shows custom labels/title

## 7. Open questions

- **F14 PDF library:** `fpdf2` is lightweight (pure Python, ~200KB) but adds a runtime dep. Alternative: write a minimal raw-PDF wrapper (PNG image in a single-page PDF is ~30 lines). Recommend `fpdf2` as optional dep (`pip install ferrum-viz[pdf]`).
- **B3 auto-groupby scope:** should auto-groupby include `facet_row`/`facet_col` fields? Altair does. Recommend yes — infer from all non-aggregate encoding fields + facet fields.
- **M3 kernel in Rust vs Python:** the Rust `Kde` transform has the hot loop. Adding kernel variants there is more performant than Python-side. Recommend Rust.
