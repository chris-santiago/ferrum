# Python → Rust Kwarg Piping Test Coverage

**Date:** 2026-05-13  
**Status:** Proposed  
**Motivation:** `parallel_coordinates_chart(alpha=0.3)` renders at full opacity — the alpha kwarg is lost somewhere between Python and Rust. No regression test caught this because existing tests only verify "renders without error", not that styling arguments actually arrive in the SVG output.

## Problem

Every mark and figure function accepts styling kwargs (opacity, stroke_width, stroke_dash, size, filled, font_size, etc.) that flow through:

```
Python mark_*() / figure function
  → MarkKwargsSpec (Python dataclass)
    → JSON serialization
      → Rust serde deserialization
        → resolve_mark_style()
          → SVG attributes
```

Any stage can silently drop a kwarg: typo in the field name, missing serde attribute, wrong type coercion, resolve_mark_style not reading the field. Today there is no systematic test that a kwarg set in Python produces the expected SVG attribute.

## Solution

A single parametrized test file (`tests/test_kwarg_piping.py`) that verifies round-trip fidelity for every mark-style kwarg on every mark type where it's meaningful.

### Test shape

```python
@pytest.mark.parametrize("mark_method, kwarg, value, svg_check", [
    ("mark_point",   "opacity",      0.3,     'opacity="0.3"'),
    ("mark_point",   "size",         120.0,   'r="'),           # radius changes
    ("mark_point",   "filled",       True,    'fill='),
    ("mark_line",    "stroke_width", 3.0,     'stroke-width="3"'),
    ("mark_line",    "stroke_dash",  [4, 2],  'stroke-dasharray="4 2"'),
    ("mark_line",    "opacity",      0.5,     'opacity="0.5"'),
    ("mark_bar",     "corner_radius",4.0,     'rx="4"'),
    ("mark_bar",     "opacity",      0.7,     'opacity="0.7"'),
    ("mark_text",    "font_size",    16.0,    'font-size="16"'),
    ("mark_text",    "font_weight",  "bold",  'font-weight="bold"'),
    ("mark_text",    "align",        "right", 'text-anchor="end"'),
    ("mark_rule",    "stroke_dash",  [6, 3],  'stroke-dasharray="6 3"'),
    ("mark_area",    "opacity",      0.4,     'opacity="0.4"'),
    ("mark_ribbon",  "opacity",      0.2,     'opacity="0.2"'),
    # ... every mark × every applicable kwarg
])
def test_kwarg_reaches_svg(mark_method, kwarg, value, svg_check, simple_df):
    chart = getattr(fm.Chart(simple_df), mark_method)(**{kwarg: value})
    chart = chart.encode(x=fm.X("x"), y=fm.Y("y"))
    svg = chart.show_svg()
    assert svg_check in svg, f"{mark_method}({kwarg}={value!r}) not found in SVG"
```

### Coverage matrix

Build the parametrize list from two sources:
1. `MarkKwargsSpec` fields in `src/ferrum/chart.py` — every field that can be set
2. `resolve_mark_style` in `crates/ferrum-core/src/render/draw.rs` — every field that's read per mark type

Cross-reference: if a field exists in `MarkKwargsSpec` but no mark reads it, that's dead code. If a mark reads a field but it's not in the parametrize list, that's a test gap.

### Composite marks and figure functions

Composite marks (boxplot, violin, smooth, etc.) desugar into layers. Add targeted tests for kwargs that should propagate to specific sub-layers:
- `mark_smooth(ci=0.95)` → ribbon layer has opacity < 1.0
- `mark_boxplot(box_fill="red")` → rect layer has fill="red" (or similar)
- `parallel_coordinates_chart(alpha=0.3)` → polylines have opacity="0.3"

### Figure-function kwargs

Figure functions forward kwargs to marks. Test the critical paths:
- `lmplot(line_kws={"stroke_width": 3})` → line has stroke-width="3"
- `catplot(kind="bar", scatter_kws={"opacity": 0.5})` → points have opacity
- `heatmap(annot=True, fmt=".1f")` → text elements present with formatted values

## Deliverable

One file: `tests/test_kwarg_piping.py`. Parametrized, fast (each test renders a minimal 5-row DataFrame), and acts as a living contract between the Python API and Rust renderer.
