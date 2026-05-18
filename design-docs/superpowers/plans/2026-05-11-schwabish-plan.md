# Schwabish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Operationalize Schwabish's "integrate text and graphics" principle across ferrum via missing-primitives implementation, defaults work on 8 figure-level functions, a `/schwabish-improve` skill (advisory + gallery-autonomous modes), and a one-page principles doc — five sub-phases SB1 → SB5.

**Architecture:** Add new primitives (`Title(subtitle=...)`, `AUCLabel`/`APLabel`/`BrierLabel`/`OutlierLabel`, `annotate_arrow`) by closing spec drift in §3.11 + §3.19. Bias figure-level functions to ship Schwabish-compliant defaults out of the box. Add a skill that judges charts against a four-category rubric (T1–T4) in advisory mode, and selectively applies *objective* findings in gallery-autonomous mode via an `Edit`-based fixer with idempotence + lite-review gate.

**Tech Stack:** Python 3.10+ (figure-level functions, annotations, skill), Rust + PyO3 (TitleSpec IR + subtitle rendering), Polars (chart data backbone), maturin (extension build), pytest (Python tests), cargo test (Rust tests), resvg-py + `scripts/snapshot-goldens.py` (golden PNG inspection).

**Spec:** `docs/superpowers/specs/2026-05-11-schwabish-design.md` (uncommitted; lands as the first commit of this plan).

---

## File Structure

### Files created

| Path | Responsibility |
|---|---|
| `docs/superpowers/specs/2026-05-11-schwabish-principles.md` | Canonical one-page reference — Schwabish's "integrate text and graphics" principle as four T-categories; objective/subjective split; cross-references. Cached prefix of the judge prompt. |
| `src/ferrum/title.py` | `Title` value class accepting `text`, `subtitle`, anchor / offset / font styling, with `subtitle` defaulting to `None`. |
| `src/ferrum/_direct_label.py` | Private `_direct_label_endpoint(chart, label_field, position="end")` helper used by `learning_curve_chart`, `validation_curve_chart`, and the gallery-autonomous fixer. |
| `crates/ferrum-core/src/spec/title.rs` | `TitleSpec` struct with `text: String`, `subtitle: Option<String>`, plus styling fields. Replaces `Option<String>` field on `ChartSpec.title`. |
| `.claude/skills/schwabish/SKILL.md` | Skill entry point — target detection (Python file vs SVG vs directory), advisory + `--from-audit` modes, judge/fixer dispatch. |
| `.claude/skills/schwabish/judge_prompt.md` | Cached prefix for `schwabish-judge` subagent (rubric + principles doc embedded). |
| `.claude/skills/schwabish/apply_eligibility.md` | Objective-only finding IDs the autonomous fixer is allowed to apply. |
| `.claude/agents/schwabish-judge.md` | Per-chart judge subagent — reads chart artifact + rubric, writes `schwabish_verdict.md`. |
| `.claude/agents/schwabish-fixer.md` | Gallery-autonomous fixer subagent — reads verdicts, filters objective findings, edits panel scripts via `Edit`, idempotent. |

### Files modified

| Path | Change |
|---|---|
| `src/ferrum/annotations.py` | Add `AUCLabel`, `APLabel`, `BrierLabel`, `OutlierLabel` classes and `annotate_arrow` helper. Existing functions untouched. |
| `src/ferrum/__init__.py` | Re-export `Title`, `AUCLabel`, `APLabel`, `BrierLabel`, `OutlierLabel`, `annotate_arrow`. |
| `src/ferrum/chart.py` | `Chart.__init__(title=...)` and `Chart.properties(title=...)` accept `Title | str`; internal `_title` always stored as `Title`; ChartSpec receives a dict serialization. |
| `src/ferrum/figures.py` | Default flips + new kwargs on the 8 figure-level functions; active-title assembly for single-curve roc/pr/calibration. |
| `src/ferrum/_diagnostics/charts.py` | `_roc_chart_from_source` (and siblings) accept the new defaults / produce active titles. |
| `crates/ferrum-core/src/spec/chart.rs` | `ChartSpec.title: Option<String>` → `Option<TitleSpec>`. |
| `crates/ferrum-core/src/spec/mod.rs` | Re-export `TitleSpec`. |
| `crates/ferrum-core/src/render/title.rs` *(or equivalent — locate during Task 4)* | Render subtitle as a second line below title baseline when `Some(s)`. |
| `crates/ferrum-core/src/render/binding.rs` | Deserialize the title dict into `TitleSpec`. |
| `ferrum-spec.md` | Four dated `2026-05-11 (Schwabish ...)` notes in §3.11, §3.13, §3.14, §3.19 (per design spec Section 8). |

### Files generated / regenerated

| Path | Phase |
|---|---|
| `tests/goldens/**/*.svg` (subset) and `tests/test_phase_9_e2e/goldens/*.svg` (subset) | SB3 — ~30–50 SVGs regenerate when figure-function defaults flip |
| `gallery/output/<row>/schwabish_verdict.md` | SB5 — one per audited gallery row |
| `gallery/output/<row>/schwabish_applied.diff` | SB5 — diff snapshot per touched row |
| `gallery/output/SCHWABISH_REPORT.md` | SB5 — aggregate report |

---

## Task 0: Worktree Setup + Land Design Spec

**Files:**
- Create worktree: `.claude/worktrees/schwabish/` on new branch `feat/schwabish`
- Existing (uncommitted): `docs/superpowers/specs/2026-05-11-schwabish-design.md`
- Existing: `/Users/chrissantiago/.claude/projects/-Users-chrissantiago-Dropbox-GitHub-ferrum/memory/project_themes_overhaul_design.md` (will be referenced)

- [ ] **Step 1: Create the worktree on a fresh branch based on latest main**

Run from main checkout (not the current working branch):
```bash
git fetch origin
git worktree add -b feat/schwabish .claude/worktrees/schwabish origin/main
```

Expected: new directory `.claude/worktrees/schwabish/` containing a checkout of `feat/schwabish` based on `origin/main`.

- [ ] **Step 2: Move the uncommitted design spec into the worktree**

The spec was written in the main checkout but never committed. Copy it across:
```bash
cp docs/superpowers/specs/2026-05-11-schwabish-design.md .claude/worktrees/schwabish/docs/superpowers/specs/
rm docs/superpowers/specs/2026-05-11-schwabish-design.md
```

(If the spec was already moved or only ever existed in the worktree, skip this step.)

- [ ] **Step 3: Set up the worktree's Python venv and build the extension**

From `.claude/worktrees/schwabish/`:
```bash
unset CONDA_PREFIX
uv sync --extra models --extra dev
uv run --no-sync maturin develop
```

Expected: `_core` extension built; `uv run pytest --co -q | head -5` lists collected tests.

- [ ] **Step 4: Verify `cargo test` works in the worktree**

The CLAUDE.md `DYLD_LIBRARY_PATH=$(uv run python -c …)` form is fragile in worktrees per memory note. Use the explicit `PYTHONHOME` form:
```bash
PYTHONHOME=~/.local/share/uv/python/cpython-3.10.14-macos-aarch64-none \
DYLD_LIBRARY_PATH=$PYTHONHOME/lib \
cargo test --quiet 2>&1 | tail -5
```

Expected: a line like `test result: ok. NNN passed`. Note the exact baseline pass count for later sub-phase ratchet checks.

- [ ] **Step 5: Pin golden footprint at T0**

From the worktree:
```bash
find tests/goldens -name '*.svg' | wc -l
find tests/test_phase_9_e2e -name '*.svg' | wc -l
find crates/ferrum-core/tests -name '*.svg' 2>/dev/null | wc -l
```

Record the three counts at the top of this plan (replace the `≈X` placeholders in the spec's §7) — these are the baseline before SB3 regens anything.

- [ ] **Step 6: Commit the design spec as the first commit on `feat/schwabish`**

```bash
git add docs/superpowers/specs/2026-05-11-schwabish-design.md
git commit -m "docs(specs): schwabish design spec (SB1-SB5)"
```

- [ ] **Step 7: Verify baseline tests are green**

```bash
uv run pytest -q 2>&1 | tail -3
PYTHONHOME=~/.local/share/uv/python/cpython-3.10.14-macos-aarch64-none \
DYLD_LIBRARY_PATH=$PYTHONHOME/lib \
cargo test --quiet 2>&1 | tail -3
```

Expected: pytest reports `NNN passed`; cargo test reports `test result: ok`. Record both baselines.

---

## SB1 — Missing Primitives

Pure-additive sub-phase. No golden regenerates (subtitle only renders when supplied; new annotations are additive). Closes spec drift in `ferrum-spec.md §3.11` and `§3.19`.

### Task 1: `Title` Python value class

**Files:**
- Create: `src/ferrum/title.py`
- Test: `tests/test_title_value_class.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_title_value_class.py
from ferrum.title import Title


def test_title_text_only():
    t = Title("Sales")
    assert t.text == "Sales"
    assert t.subtitle is None
    assert t.anchor == "start"


def test_title_with_subtitle():
    t = Title("Sales", subtitle="2024 Q3")
    assert t.text == "Sales"
    assert t.subtitle == "2024 Q3"


def test_title_is_immutable():
    t = Title("Sales")
    import pytest
    with pytest.raises((AttributeError, TypeError)):
        t.text = "Changed"  # type: ignore[misc]


def test_title_repr_round_trip():
    t = Title("Sales", subtitle="2024 Q3", font_weight="bold")
    rendered = repr(t)
    assert "Sales" in rendered
    assert "2024 Q3" in rendered
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_title_value_class.py -v
```

Expected: `ImportError: cannot import name 'Title' from 'ferrum.title'` (or `ModuleNotFoundError`).

- [ ] **Step 3: Write minimal implementation**

```python
# src/ferrum/title.py
"""Two-line title value class — spec §3.19."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Title:
    """Chart title with optional subtitle.

    Accepted everywhere a title string is accepted: ``Chart(title=...)``,
    ``Chart.properties(title=...)``, ``HConcat/VConcat/Layer(title=...)``.
    Passing a bare string is equivalent to ``Title(text=string)``.

    Parameters
    ----------
    text : str
        Primary title text.
    subtitle : str, optional
        Secondary line below the title; ``None`` suppresses the second line.
    anchor : {"start", "middle", "end"}, default "start"
        Horizontal anchor.
    offset : float, optional
        Pixel offset from plot area; defaults to the theme's title_offset.
    font_size : float, optional
        Title font size in points.
    font_weight : str, optional
        Title font weight (e.g. ``"600"``, ``"bold"``).
    color : str, optional
        Title color as a CSS color string.
    subtitle_font_size : float, optional
        Subtitle font size; defaults to ``font_size * 0.85``.
    subtitle_color : str, optional
        Subtitle color; defaults to the theme's label_color.
    """

    text: str
    subtitle: Optional[str] = None
    anchor: str = "start"
    offset: Optional[float] = None
    font_size: Optional[float] = None
    font_weight: Optional[str] = None
    color: Optional[str] = None
    subtitle_font_size: Optional[float] = None
    subtitle_color: Optional[str] = None

    def to_spec_dict(self) -> dict:
        """Serialize to the dict shape the Rust binding expects."""
        out: dict = {"text": self.text}
        if self.subtitle is not None:
            out["subtitle"] = self.subtitle
        if self.anchor != "start":
            out["anchor"] = self.anchor
        for field_name in (
            "offset", "font_size", "font_weight", "color",
            "subtitle_font_size", "subtitle_color",
        ):
            value = getattr(self, field_name)
            if value is not None:
                out[field_name] = value
        return out
```

- [ ] **Step 4: Re-export from package**

Modify `src/ferrum/__init__.py` — add `from ferrum.title import Title` and add `"Title"` to `__all__`.

- [ ] **Step 5: Run test to verify it passes**

```bash
uv run pytest tests/test_title_value_class.py -v
```

Expected: all 4 tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/ferrum/title.py src/ferrum/__init__.py tests/test_title_value_class.py
git commit -m "feat(title): Title value class accepting text + subtitle (SB1)"
```

---

### Task 2: `Chart.__init__` and `Chart.properties` accept `Title | str`

**Files:**
- Modify: `src/ferrum/chart.py` (`__init__` ~line 105; `properties` ~line 3912; spec assembly ~line 4010)
- Test: `tests/test_chart_title_accepts_title_class.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_chart_title_accepts_title_class.py
import polars as pl
import pytest

from ferrum import Chart, Title


@pytest.fixture
def df():
    return pl.DataFrame({"x": [1, 2, 3], "y": [1, 4, 9]})


def test_chart_accepts_title_string(df):
    c = Chart(df, title="My chart").encode(x="x", y="y").mark_point()
    # internal stays-as-Title invariant
    assert c._title is not None
    assert c._title.text == "My chart"
    assert c._title.subtitle is None


def test_chart_accepts_title_class(df):
    c = Chart(df, title=Title("My chart", subtitle="2024")).encode(x="x", y="y").mark_point()
    assert c._title.text == "My chart"
    assert c._title.subtitle == "2024"


def test_properties_accepts_title_class(df):
    c = Chart(df).encode(x="x", y="y").mark_point().properties(title=Title("foo", subtitle="bar"))
    assert c._title.subtitle == "bar"


def test_chartspec_title_serialized_as_dict(df):
    c = Chart(df, title=Title("foo", subtitle="bar")).encode(x="x", y="y").mark_point()
    spec = c._to_chart_spec()
    # ChartSpec.title is a dict-shaped payload, not a bare string
    assert isinstance(spec.title, dict)
    assert spec.title["text"] == "foo"
    assert spec.title["subtitle"] == "bar"


def test_string_title_serializes_with_text_field(df):
    c = Chart(df, title="just a string").encode(x="x", y="y").mark_point()
    spec = c._to_chart_spec()
    assert isinstance(spec.title, dict)
    assert spec.title == {"text": "just a string"}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_chart_title_accepts_title_class.py -v
```

Expected: failures — Title isn't normalized; `spec.title` is still a string.

- [ ] **Step 3: Modify `Chart.__init__` to normalize title**

In `src/ferrum/chart.py`, change the title-related lines (~105–119 region):

Find the signature line:
```python
        title: Optional[str] = None,
```
Replace with:
```python
        title: "Optional[str | Title]" = None,
```

Find the assignment:
```python
        self._title = title
```
Replace with:
```python
        from ferrum.title import Title as _TitleCls
        if title is None:
            self._title = None
        elif isinstance(title, _TitleCls):
            self._title = title
        else:
            self._title = _TitleCls(text=str(title))
```

- [ ] **Step 4: Modify `Chart.properties` analogously**

In the `properties()` method body (~line 3947):

Find:
```python
        if title is not None: new._title = title
```
Replace with:
```python
        if title is not None:
            from ferrum.title import Title as _TitleCls
            new._title = title if isinstance(title, _TitleCls) else _TitleCls(text=str(title))
```

- [ ] **Step 5: Modify ChartSpec assembly to serialize Title to dict**

In the `_to_chart_spec` method body (~line 4010):

Find:
```python
        if resolved._title is not None:
            kw["title"] = resolved._title
```
Replace with:
```python
        if resolved._title is not None:
            kw["title"] = resolved._title.to_spec_dict()
```

- [ ] **Step 6: Run test to verify it passes**

```bash
uv run pytest tests/test_chart_title_accepts_title_class.py -v
```

Expected: all 5 tests pass.

- [ ] **Step 7: Run the broader test suite to ensure no regressions**

```bash
uv run pytest -q 2>&1 | tail -3
```

Expected: same pass count as baseline + 5 new passes (or higher). Note: at this stage the Rust binding still expects `Option<String>` for title. The dict serialization will fail at Rust-level if any existing test actually renders a chart with a title. **Expect failures here** — they are addressed in Task 4.

If existing render tests fail with a title-related deserialization error, that's the signal to proceed to Task 3 + 4. Do **not** commit yet.

- [ ] **Step 8: Hold commit until Task 4**

The Python ↔ Rust contract is now broken (Python sends dict, Rust expects string). Tasks 3 + 4 restore it. Don't commit until then.

---

### Task 3: Rust `TitleSpec` struct

**Files:**
- Create: `crates/ferrum-core/src/spec/title.rs`
- Modify: `crates/ferrum-core/src/spec/mod.rs` (export)
- Modify: `crates/ferrum-core/src/spec/chart.rs:72` (replace `Option<String>` with `Option<TitleSpec>`)
- Test: inline `#[cfg(test)] mod tests` in `title.rs`

- [ ] **Step 1: Write the failing test (inline in new file)**

Create `crates/ferrum-core/src/spec/title.rs`:
```rust
//! Two-line title spec — see ferrum-spec.md §3.19.
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct TitleSpec {
    pub text: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subtitle: Option<String>,
    #[serde(default = "default_anchor", skip_serializing_if = "is_default_anchor")]
    pub anchor: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub offset: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub font_size: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub font_weight: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub color: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subtitle_font_size: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subtitle_color: Option<String>,
}

fn default_anchor() -> String { "start".into() }
fn is_default_anchor(s: &String) -> bool { s == "start" }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_only_roundtrip() {
        let t = TitleSpec {
            text: "foo".into(),
            subtitle: None,
            anchor: "start".into(),
            offset: None,
            font_size: None,
            font_weight: None,
            color: None,
            subtitle_font_size: None,
            subtitle_color: None,
        };
        let json = serde_json::to_string(&t).unwrap();
        assert_eq!(json, r#"{"text":"foo"}"#);
        let parsed: TitleSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, t);
    }

    #[test]
    fn with_subtitle_roundtrip() {
        let json = r#"{"text":"foo","subtitle":"bar"}"#;
        let t: TitleSpec = serde_json::from_str(json).unwrap();
        assert_eq!(t.text, "foo");
        assert_eq!(t.subtitle.as_deref(), Some("bar"));
        let reserialized = serde_json::to_string(&t).unwrap();
        assert_eq!(reserialized, json);
    }

    #[test]
    fn unknown_key_rejected() {
        let json = r#"{"text":"foo","typo":"x"}"#;
        let result: Result<TitleSpec, _> = serde_json::from_str(json);
        assert!(result.is_err());
    }
}
```

- [ ] **Step 2: Export from `spec/mod.rs`**

Add to `crates/ferrum-core/src/spec/mod.rs`:
```rust
pub mod title;
pub use title::TitleSpec;
```

- [ ] **Step 3: Replace `Option<String>` with `Option<TitleSpec>` in `spec/chart.rs`**

Find at `crates/ferrum-core/src/spec/chart.rs:72`:
```rust
    pub title: Option<String>,
```
Replace with:
```rust
    pub title: Option<TitleSpec>,
```

Also update any places in the same file that construct `ChartSpec` with `title: None` — those should still compile (None is None). Search for `title:` and verify no construction passes a `String` directly. The `title = None` defaults at lines 98, 184, 448, 536, 566, 596 stay unchanged.

If any place in the codebase still constructs `ChartSpec { title: Some("..."), ... }`, change to `Some(TitleSpec { text: "...".into(), ..Default::default() })` — but TitleSpec has no `Default`, so add one:

In `crates/ferrum-core/src/spec/title.rs` append:
```rust
impl Default for TitleSpec {
    fn default() -> Self {
        Self {
            text: String::new(),
            subtitle: None,
            anchor: "start".into(),
            offset: None,
            font_size: None,
            font_weight: None,
            color: None,
            subtitle_font_size: None,
            subtitle_color: None,
        }
    }
}
```

- [ ] **Step 4: Search-and-fix any other `title: Option<String>` consumers**

```bash
grep -rn -E "(title:|\.title)" crates/ferrum-core/src/ --include="*.rs" | grep -v "test"
```

Each result that pattern-matches `Option<String>` against the field needs an update to handle `Option<TitleSpec>` instead. Most likely consumer: the title-rendering site at `crates/ferrum-core/src/render/marks/strip_title.rs` (for panel/strip titles, *separate* from chart-level title) — verify, but expect strip title to be its own type. The chart-level title consumer (Themes-T2.5 wired in `b731931 feat(themes-T2.5a): chart-level title rendering`) is the main one to fix in Task 4.

- [ ] **Step 5: Run cargo test for the new module**

```bash
PYTHONHOME=~/.local/share/uv/python/cpython-3.10.14-macos-aarch64-none \
DYLD_LIBRARY_PATH=$PYTHONHOME/lib \
cargo test --package ferrum-core spec::title 2>&1 | tail -5
```

Expected: 3 tests passing in `spec::title::tests`.

- [ ] **Step 6: Full cargo test (compile check + regression check)**

```bash
PYTHONHOME=~/.local/share/uv/python/cpython-3.10.14-macos-aarch64-none \
DYLD_LIBRARY_PATH=$PYTHONHOME/lib \
cargo test --quiet 2>&1 | tail -5
```

Expected: same total pass count as baseline + 3 (new title tests). If anything fails, it's a place that depended on `title: Option<String>`; fix per Step 4 pattern.

- [ ] **Step 7: Hold commit until Task 4**

Title rendering hasn't been updated to read `TitleSpec.subtitle` yet — that's Task 4. The build compiles but subtitle is ignored. Don't commit until after Task 4 so the SB1 title commit ships subtitle rendering atomically.

---

### Task 4: Rust subtitle rendering

**Files:**
- Locate and modify: chart-level title rendering site (added in commit `b731931`; likely in `crates/ferrum-core/src/render/title.rs` or `crates/ferrum-core/src/render/draw.rs`)
- Modify: `crates/ferrum-core/src/render/binding.rs` (title dict deserialization path)
- Test: `tests/test_subtitle_renders.py` (Python integration)

- [ ] **Step 1: Locate the chart-level title renderer**

```bash
grep -rn "title" crates/ferrum-core/src/render/ --include="*.rs" | grep -i -E "draw|render|spec\.title|chart.*title" | head -10
```

Identify the function that emits the SVG `<text>` element for the chart-level title. This was added in `b731931`. Expected location: a `draw_chart_title` function in `crates/ferrum-core/src/render/title.rs` or inside the top-level render in `crates/ferrum-core/src/render/mod.rs`.

- [ ] **Step 2: Write the failing Python integration test**

```python
# tests/test_subtitle_renders.py
import polars as pl

from ferrum import Chart, Title


def test_subtitle_renders_as_second_text_element():
    df = pl.DataFrame({"x": [1, 2, 3], "y": [1, 4, 9]})
    chart = Chart(df, title=Title("Main", subtitle="Sub")).encode(x="x", y="y").mark_point()
    svg = chart.to_svg()
    # both lines present
    assert ">Main<" in svg
    assert ">Sub<" in svg
    # subtitle text node appears after main title in document order
    assert svg.index(">Main<") < svg.index(">Sub<")


def test_no_subtitle_byte_identical_to_string_title():
    df = pl.DataFrame({"x": [1, 2, 3], "y": [1, 4, 9]})
    a = Chart(df, title="Main").encode(x="x", y="y").mark_point().to_svg()
    b = Chart(df, title=Title("Main")).encode(x="x", y="y").mark_point().to_svg()
    assert a == b
```

- [ ] **Step 3: Run the failing test**

```bash
uv run --no-sync maturin develop && uv run pytest tests/test_subtitle_renders.py -v
```

Expected: failure — subtitle text is not emitted.

- [ ] **Step 4: Update the title renderer to emit a second `<text>` line when subtitle is `Some(...)`**

In the located title-rendering function, after the existing title `<text>` emission, add (sketch — adapt to the actual function signature):
```rust
if let Some(ref subtitle_text) = spec.subtitle {
    let subtitle_y = title_y + title_font_size + 2.0;
    let subtitle_font_size = spec.subtitle_font_size.unwrap_or(title_font_size * 0.85);
    let subtitle_color = spec.subtitle_color.as_deref()
        .unwrap_or(theme.label_color_as_css());
    writeln!(
        out,
        r#"<text x="{x}" y="{subtitle_y}" font-family="{family}" font-size="{subtitle_font_size}" fill="{subtitle_color}" text-anchor="{anchor}">{text}</text>"#,
        x = anchor_x,
        family = font_family,
        anchor = svg_text_anchor,
        text = subtitle_text,
    )?;
}
```

The existing title `<text>` emission already exists; just add the subtitle below it. Reserve `subtitle_font_size + 2.0` extra px in the title vertical strip when subtitle is `Some`.

- [ ] **Step 5: Update `render/binding.rs` to accept the dict shape**

Find the place that reads `chart_spec.title` (likely a `PyDict` extract). Change from extracting `Option<String>` to extracting `Option<TitleSpec>` via `serde_pyobject` or `pythonize::depythonize`. Sketch:
```rust
// before: let title: Option<String> = chart_dict.get_item("title")?.map(...).transpose()?;
let title: Option<TitleSpec> = match chart_dict.get_item("title")? {
    None => None,
    Some(v) if v.is_none() => None,
    Some(v) => Some(pythonize::depythonize(&v).map_err(...)?),
};
```

(Adapt to whichever serde-pyo3 bridge the codebase already uses — search for nearby `depythonize` or `serde_pyobject::from_pyobject` calls and mirror the style.)

- [ ] **Step 6: Rebuild and re-run the test**

```bash
unset CONDA_PREFIX && uv run --no-sync maturin develop
uv run pytest tests/test_subtitle_renders.py -v
```

Expected: both tests pass.

- [ ] **Step 7: Run the wider test suite — title roundtrip must be green**

```bash
uv run pytest -q 2>&1 | tail -3
PYTHONHOME=~/.local/share/uv/python/cpython-3.10.14-macos-aarch64-none \
DYLD_LIBRARY_PATH=$PYTHONHOME/lib \
cargo test --quiet 2>&1 | tail -3
```

Expected: pytest baseline + 7 new tests (Tasks 1–4), cargo baseline + 3.

- [ ] **Step 8: Commit Tasks 1–4 as a single atomic SB1 commit**

```bash
git add src/ferrum/title.py src/ferrum/chart.py src/ferrum/__init__.py \
  crates/ferrum-core/src/spec/title.rs crates/ferrum-core/src/spec/mod.rs \
  crates/ferrum-core/src/spec/chart.rs crates/ferrum-core/src/render/ \
  crates/ferrum-core/src/render/binding.rs \
  tests/test_title_value_class.py tests/test_chart_title_accepts_title_class.py \
  tests/test_subtitle_renders.py
git commit -m "feat(title): subtitle support end-to-end (Title class + TitleSpec IR + render) (SB1)"
```

---

### Task 5: `AUCLabel` composite annotation

**Files:**
- Modify: `src/ferrum/annotations.py` (extend)
- Modify: `src/ferrum/__init__.py` (re-export)
- Test: `tests/test_auc_label.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_auc_label.py
import numpy as np
import polars as pl

from ferrum import Chart, AUCLabel


def _roc_data():
    # synthetic 2-class ROC curve for one class
    fpr = np.linspace(0, 1, 50)
    tpr = np.sqrt(fpr)  # AUC ≈ 2/3
    return pl.DataFrame({"fpr": fpr, "tpr": tpr, "class": ["c0"] * len(fpr)})


def test_auc_label_default_position_end():
    df = _roc_data()
    base = Chart(df).encode(x="fpr", y="tpr", color="class").mark_line()
    chart = base + AUCLabel()
    svg = chart.to_svg()
    # AUC text present with .3f format
    assert "AUC = 0.66" in svg or "AUC = 0.67" in svg


def test_auc_label_custom_prefix_and_format():
    df = _roc_data()
    base = Chart(df).encode(x="fpr", y="tpr", color="class").mark_line()
    chart = base + AUCLabel(format=".2f", prefix="auc:")
    svg = chart.to_svg()
    assert "auc:0.67" in svg or "auc:0.66" in svg


def test_auc_label_corner_position():
    df = _roc_data()
    base = Chart(df).encode(x="fpr", y="tpr", color="class").mark_line()
    chart = base + AUCLabel(position="corner")
    svg = chart.to_svg()
    # corner-placed text should be near x=0.95, y=0.05 in data coords
    # we just verify both a text element and the value are present
    assert "AUC =" in svg


def test_auc_label_multi_class_emits_one_per_class():
    df = pl.concat([
        _roc_data(),
        _roc_data().with_columns(pl.lit("c1").alias("class")),
    ])
    base = Chart(df).encode(x="fpr", y="tpr", color="class").mark_line()
    chart = base + AUCLabel()
    svg = chart.to_svg()
    # two AUC text emissions
    assert svg.count("AUC =") == 2
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_auc_label.py -v
```

Expected: `ImportError: cannot import name 'AUCLabel'`.

- [ ] **Step 3: Implement `AUCLabel` in `src/ferrum/annotations.py`**

Append to `src/ferrum/annotations.py`:
```python
from dataclasses import dataclass
from typing import Literal

import numpy as np


def _trapezoid_auc(x: np.ndarray, y: np.ndarray) -> float:
    """Trapezoidal AUC for a curve sorted by x."""
    order = np.argsort(x)
    return float(np.trapz(y[order], x[order]))


@dataclass(frozen=True)
class AUCLabel:
    """Auto-placed AUC annotation for ROC charts — spec §3.11.

    Used as ``chart + AUCLabel()``. Reads the surrounding chart's
    line data (assumes ``x`` is FPR, ``y`` is TPR), computes the
    trapezoidal AUC per series (grouped by the ``color`` encoding
    if present), and emits one text annotation per series.

    Parameters
    ----------
    position : {"end", "corner"}, default "end"
        ``"end"`` places the label at the endpoint of each line.
        ``"corner"`` stacks labels in the lower-right corner.
    format : str, default ".3f"
        Numeric format spec passed to f-string.
    prefix : str, default "AUC = "
        Text prefix prepended to the formatted number.
    """

    position: Literal["end", "corner"] = "end"
    format: str = ".3f"
    prefix: str = "AUC = "

    def __radd__(self, base: Chart) -> Chart:
        from ferrum.chart import Chart as _ChartCls
        if not isinstance(base, _ChartCls):
            return NotImplemented
        return _apply_metric_label(
            base, self, metric_fn=_trapezoid_auc,
        )


def _resolve_field(enc_value) -> Optional[str]:
    """Extract the field name from an encoding value.

    Encoding dict values are either ChannelBase wrappers (with `.field`)
    or plain strings, per chart.py's existing pattern (chart.py:225-230).
    """
    if enc_value is None:
        return None
    field = getattr(enc_value, "field", None)
    if field is not None:
        return field
    if isinstance(enc_value, str):
        return enc_value
    return None


def _apply_metric_label(
    base: Chart,
    label: "AUCLabel | APLabel | BrierLabel",
    *,
    metric_fn,
) -> Chart:
    """Compute metric per series, emit an annotation overlay on the same data.

    Uses the augmented-DataFrame pattern referenced by Chart.__add__'s
    fallback warning (`decision_boundary_chart` is the canonical example):
    concatenate annotation rows into the base data with null padding on
    base columns and a new ``_label_text`` column, then add a ``mark_text``
    layer to the base chart. This is the only way to overlay without
    triggering the HConcat fallback when the annotation has its own data.
    """
    from ferrum._coerce import to_arrow_table

    x_col = _resolve_field(base._encoding.get("x"))
    y_col = _resolve_field(base._encoding.get("y"))
    color_col = _resolve_field(base._encoding.get("color"))
    if x_col is None or y_col is None:
        raise ValueError(
            f"{type(label).__name__} requires x and y encodings on the base chart"
        )
    tbl = to_arrow_table(base._data)
    x_arr = np.asarray(tbl.column(x_col).to_pylist(), dtype=float)
    y_arr = np.asarray(tbl.column(y_col).to_pylist(), dtype=float)
    rows: list[dict] = []
    if color_col is not None and color_col in tbl.column_names:
        color_vals = np.asarray(tbl.column(color_col).to_pylist())
        for cls in sorted(set(color_vals.tolist()), key=str):
            mask = color_vals == cls
            metric = metric_fn(x_arr[mask], y_arr[mask])
            text = f"{label.prefix}{metric:{label.format}}"
            if label.position == "end":
                idx = int(np.argmax(x_arr[mask]))
                rows.append({x_col: float(x_arr[mask][idx]),
                             y_col: float(y_arr[mask][idx]),
                             color_col: cls, "_label_text": text})
            else:
                rows.append({x_col: 0.95, y_col: 0.05 + 0.06 * len(rows),
                             color_col: cls, "_label_text": text})
    else:
        metric = metric_fn(x_arr, y_arr)
        text = f"{label.prefix}{metric:{label.format}}"
        if label.position == "end":
            idx = int(np.argmax(x_arr))
            rows.append({x_col: float(x_arr[idx]), y_col: float(y_arr[idx]),
                         "_label_text": text})
        else:
            rows.append({x_col: 0.95, y_col: 0.05, "_label_text": text})
    # Build a Chart whose data IS base._data extended with annotation rows.
    # The base's existing mark/encoding ignores rows lacking the encoded
    # fields; the new mark_text layer reads only the `_label_text` rows.
    base_pl = pl.from_arrow(tbl) if not isinstance(base._data, pl.DataFrame) else base._data
    # Pad base rows with a null `_label_text` column; pad annotation rows
    # with whatever columns base has that the annotation row doesn't set.
    base_padded = base_pl.with_columns(pl.lit(None).alias("_label_text"))
    annot_df = pl.DataFrame(rows)
    for col in base_padded.columns:
        if col not in annot_df.columns:
            annot_df = annot_df.with_columns(pl.lit(None).alias(col))
    annot_df = annot_df.select(base_padded.columns)  # column order match
    combined = pl.concat([base_padded, annot_df], how="vertical")
    # New chart: same encoding as base, but the data is combined and a
    # mark_text layer is added that filters to non-null _label_text.
    annot_layer = (
        Chart(combined)
        .mark_text(align="right", dx=-4, dy=-2)
        .encode(x=x_col, y=y_col, text="_label_text")
    )
    # Rebuild base over the combined data so layering shares data.
    base_over_combined = base._clone()
    base_over_combined._data = combined
    return base_over_combined + annot_layer
```

Note: this references `base._df` and `base._encoding_kwargs` — verify the exact attribute names on `Chart` before locking the implementation. If `_df` is named differently (e.g., `_data` or `_frame`), update.

- [ ] **Step 4: Re-export from `__init__.py`**

Add `from ferrum.annotations import AUCLabel` and `"AUCLabel"` to `__all__`.

- [ ] **Step 5: Run test to verify it passes**

```bash
uv run pytest tests/test_auc_label.py -v
```

Expected: all 4 tests pass. If `Chart + Chart` composition is unsupported but `Chart & Chart` or `Chart.layer(Chart)` is, update the `_apply_metric_label` last lines accordingly:
```python
# fallback: use & or .layer() per the existing Chart compose API
result = base
for lc in labels_charts:
    result = result & lc  # or result.layer(lc)
```

Check `src/ferrum/chart.py` for `__add__`, `__and__`, or `layer` methods and pick the one that adds the annotation as an overlay (same panel) rather than a side-by-side concat.

- [ ] **Step 6: Hold commit until Task 9** (annotations live in one file; commit all four metric labels + annotate_arrow together)

---

### Task 6: `APLabel` composite annotation

**Files:**
- Modify: `src/ferrum/annotations.py` (extend)
- Modify: `src/ferrum/__init__.py`
- Test: `tests/test_ap_label.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ap_label.py
import numpy as np
import polars as pl

from ferrum import Chart, APLabel


def _pr_data():
    # synthetic PR curve — recall axis ascending, precision strictly decreasing
    recall = np.linspace(0, 1, 50)
    precision = 1.0 - 0.5 * recall  # AP ≈ 0.75
    return pl.DataFrame({"recall": recall, "precision": precision, "class": ["c0"] * len(recall)})


def test_ap_label_default():
    df = _pr_data()
    chart = Chart(df).encode(x="recall", y="precision", color="class").mark_line() + APLabel()
    svg = chart.to_svg()
    assert "AP = 0.75" in svg or "AP = 0.74" in svg or "AP = 0.76" in svg
```

- [ ] **Step 2: Run, expect fail**

```bash
uv run pytest tests/test_ap_label.py -v
```

- [ ] **Step 3: Implement `APLabel` in `annotations.py`**

```python
def _ap_step(x: np.ndarray, y: np.ndarray) -> float:
    """Step-integrated average precision: sum((R_i - R_{i-1}) * P_i)."""
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    return float(np.sum(np.diff(np.concatenate([[0.0], xs])) * ys))


@dataclass(frozen=True)
class APLabel:
    """Auto-placed AP annotation for PR charts — spec §3.11 (sibling of AUCLabel)."""
    position: Literal["end", "corner"] = "end"
    format: str = ".3f"
    prefix: str = "AP = "

    def __radd__(self, base: Chart) -> Chart:
        from ferrum.chart import Chart as _ChartCls
        if not isinstance(base, _ChartCls):
            return NotImplemented
        return _apply_metric_label(base, self, metric_fn=_ap_step)
```

- [ ] **Step 4: Re-export, run, verify**

```bash
uv run pytest tests/test_ap_label.py -v
```

Expected: pass.

- [ ] **Step 5: Hold commit until Task 9.**

---

### Task 7: `BrierLabel` composite annotation

**Files:**
- Modify: `src/ferrum/annotations.py`, `src/ferrum/__init__.py`
- Test: `tests/test_brier_label.py`

- [ ] **Step 1: Test**

```python
# tests/test_brier_label.py
import numpy as np
import polars as pl

from ferrum import Chart, BrierLabel


def _calibration_data():
    # perfect calibration → Brier ≈ 0
    p = np.linspace(0.05, 0.95, 19)
    obs = p.copy()
    return pl.DataFrame({"predicted": p, "observed": obs, "model": ["m0"] * len(p)})


def test_brier_label_default_corner():
    df = _calibration_data()
    chart = Chart(df).encode(x="predicted", y="observed", color="model").mark_line() + BrierLabel()
    svg = chart.to_svg()
    assert "Brier =" in svg
```

- [ ] **Step 2: Implement**

In `annotations.py`:
```python
def _brier_score(p: np.ndarray, obs: np.ndarray) -> float:
    """Brier as mean squared error between predicted prob and observed rate per bin."""
    return float(np.mean((p - obs) ** 2))


@dataclass(frozen=True)
class BrierLabel:
    position: Literal["end", "corner"] = "corner"
    format: str = ".3f"
    prefix: str = "Brier = "

    def __radd__(self, base: Chart) -> Chart:
        from ferrum.chart import Chart as _ChartCls
        if not isinstance(base, _ChartCls):
            return NotImplemented
        return _apply_metric_label(base, self, metric_fn=_brier_score)
```

- [ ] **Step 3: Re-export, run.**

```bash
uv run pytest tests/test_brier_label.py -v
```

- [ ] **Step 4: Hold commit until Task 9.**

---

### Task 8: `OutlierLabel` composite annotation

**Files:**
- Modify: `src/ferrum/annotations.py`, `src/ferrum/__init__.py`
- Test: `tests/test_outlier_label.py`

- [ ] **Step 1: Test**

```python
# tests/test_outlier_label.py
import numpy as np
import polars as pl

from ferrum import Chart, OutlierLabel


def test_outlier_label_threshold_3_emits_only_high_z():
    rng = np.random.default_rng(0)
    n = 1000
    residuals = rng.normal(0, 1, n)
    residuals[[100, 200, 300]] = 5.0  # 3 extreme outliers
    residuals[[400, 500]] = -4.5
    df = pl.DataFrame({
        "fitted": np.linspace(0, 10, n),
        "residual": residuals,
        "obs_id": [f"id_{i}" for i in range(n)],
    })
    chart = (
        Chart(df).encode(x="fitted", y="residual").mark_point()
        + OutlierLabel(threshold=3.0, field="residual", label_field="obs_id")
    )
    svg = chart.to_svg()
    # 5 outliers; default max_labels=10, so all 5 emitted
    for idx in (100, 200, 300, 400, 500):
        assert f"id_{idx}" in svg


def test_outlier_label_max_labels_caps():
    rng = np.random.default_rng(0)
    n = 1000
    residuals = rng.normal(0, 1, n)
    residuals[:20] = 5.0  # 20 outliers
    df = pl.DataFrame({
        "fitted": np.linspace(0, 10, n),
        "residual": residuals,
        "obs_id": [f"id_{i}" for i in range(n)],
    })
    chart = (
        Chart(df).encode(x="fitted", y="residual").mark_point()
        + OutlierLabel(threshold=3.0, field="residual", label_field="obs_id", max_labels=3)
    )
    svg = chart.to_svg()
    # only top 3 by |z| in the SVG (all are 5.0 so first 3 by index)
    count = sum(1 for i in range(20) if f">id_{i}<" in svg)
    assert count == 3
```

- [ ] **Step 2: Implement**

```python
@dataclass(frozen=True)
class OutlierLabel:
    """Auto-labeled high-leverage / high-residual points — spec §3.11.

    Used as ``chart + OutlierLabel()``. Reads the chart's data, identifies
    points where |z(field)| > threshold, emits annotate_text for the top-N
    (default 10) using ``label_field`` as the label text.
    """
    threshold: float = 3.0
    field: Optional[str] = None
    label_field: Optional[str] = None
    max_labels: int = 10

    def __radd__(self, base):
        from ferrum.chart import Chart as _ChartCls
        from ferrum._coerce import to_arrow_table
        if not isinstance(base, _ChartCls):
            return NotImplemented
        x_col = _resolve_field(base._encoding.get("x"))
        y_col = _resolve_field(base._encoding.get("y"))
        field = self.field or y_col
        tbl = to_arrow_table(base._data)
        if field is None or field not in tbl.column_names:
            raise ValueError(f"OutlierLabel: cannot locate field {field!r}")
        values = np.asarray(tbl.column(field).to_pylist(), dtype=float)
        mu = float(np.mean(values))
        sigma = float(np.std(values, ddof=1)) or 1.0
        z = np.abs((values - mu) / sigma)
        mask = z > self.threshold
        if not mask.any():
            return base
        candidate_idx = np.where(mask)[0]
        ordered = candidate_idx[np.argsort(-z[candidate_idx])][: self.max_labels]
        # Use the augmented-DataFrame pattern (same as _apply_metric_label)
        # so the labels overlay properly on base._data.
        x_all = np.asarray(tbl.column(x_col).to_pylist(), dtype=float)
        y_all = np.asarray(tbl.column(y_col).to_pylist(), dtype=float)
        label_col_name = "_outlier_text"
        labels_col = [None] * len(values)
        for i in ordered:
            if self.label_field and self.label_field in tbl.column_names:
                labels_col[int(i)] = str(tbl.column(self.label_field).to_pylist()[int(i)])
            else:
                labels_col[int(i)] = str(values[int(i)])
        base_pl = pl.from_arrow(tbl) if not isinstance(base._data, pl.DataFrame) else base._data
        augmented = base_pl.with_columns(pl.Series(label_col_name, labels_col))
        base_aug = base._clone()
        base_aug._data = augmented
        annot_layer = (
            Chart(augmented)
            .mark_text(align="left", dx=4, dy=-4)
            .encode(x=x_col, y=y_col, text=label_col_name)
        )
        return base_aug + annot_layer
```

- [ ] **Step 3: Run, verify both tests pass.**

- [ ] **Step 4: Hold commit until Task 9.**

---

### Task 9: `annotate_arrow` + commit all SB1 annotation primitives

**Files:**
- Modify: `src/ferrum/annotations.py`, `src/ferrum/__init__.py`
- Test: `tests/test_annotate_arrow.py`

- [ ] **Step 1: Test**

```python
# tests/test_annotate_arrow.py
import polars as pl

from ferrum import Chart, annotate_arrow


def test_annotate_arrow_emits_segment_and_label():
    df = pl.DataFrame({"x": [0, 1, 2], "y": [0, 1, 4]})
    chart = (
        Chart(df).encode(x="x", y="y").mark_line()
        & annotate_arrow(0.5, 1.5, 1.5, 3.5, label="trend")
    )
    svg = chart.to_svg()
    # segment with arrow marker + label text
    assert ">trend<" in svg
```

- [ ] **Step 2: Implement**

```python
def annotate_arrow(
    x1: float, y1: float, x2: float, y2: float,
    *, label: Optional[str] = None, label_side: str = "start",
    stroke: Optional[str] = None,
) -> Chart:
    """Arrow from (x1, y1) to (x2, y2) with optional text label — spec §3.11.

    Composed from ``mark_segment(arrow=True)`` + optional ``annotate_text``
    at the ``label_side`` endpoint.
    """
    df = pl.DataFrame({"_x1": [x1], "_y1": [y1], "_x2": [x2], "_y2": [y2]})
    seg_kwargs: dict = {"arrow": True}
    if stroke is not None:
        seg_kwargs["stroke"] = stroke
    arrow_chart = (
        Chart(df).mark_segment(**seg_kwargs).encode(x="_x1", y="_y1", x2="_x2", y2="_y2")
    )
    if label is None:
        return arrow_chart
    lx, ly = (x1, y1) if label_side == "start" else (x2, y2)
    dx = -6 if label_side == "start" else 6
    align = "right" if label_side == "start" else "left"
    return arrow_chart & annotate_text(lx, ly, label, dx=dx, align=align)
```

- [ ] **Step 3: Run all SB1 annotation tests**

```bash
uv run pytest tests/test_auc_label.py tests/test_ap_label.py tests/test_brier_label.py tests/test_outlier_label.py tests/test_annotate_arrow.py -v
```

Expected: all pass.

- [ ] **Step 4: Run the full suite for regressions**

```bash
uv run pytest -q 2>&1 | tail -3
```

Expected: baseline + Task-1..9 new tests.

- [ ] **Step 5: Commit all five SB1 annotation primitives**

```bash
git add src/ferrum/annotations.py src/ferrum/__init__.py \
  tests/test_auc_label.py tests/test_ap_label.py tests/test_brier_label.py \
  tests/test_outlier_label.py tests/test_annotate_arrow.py
git commit -m "feat(annotations): AUCLabel/APLabel/BrierLabel/OutlierLabel + annotate_arrow (SB1)"
```

- [ ] **Step 6: Append SB1 dated note to `ferrum-spec.md §3.11`**

Modify `ferrum-spec.md` — append after the §3.11 table (per design spec §8):
```markdown
> **2026-05-11 (Schwabish SB1):** `AUCLabel`, `OutlierLabel`, and
> `annotate_arrow` are now implemented. Two sibling composites,
> `APLabel(*, position="end", format=".3f", prefix="AP = ")` and
> `BrierLabel(*, position="corner", format=".3f", prefix="Brier = ")`,
> are added — same pattern as `AUCLabel` but for PR (AP) and
> calibration (Brier score). When added to a multi-series chart, each
> composite emits one label per series. All four metric labels read
> the surrounding chart's mark_line data and compute the metric in
> Python (no new Rust IR).
```

Also append to §3.19 after the `Title(...)` signature:
```markdown
> **2026-05-11 (Schwabish SB1):** `Title(text, subtitle=...)` is now
> implemented. Accepted everywhere a title string is accepted today
> (`Chart(title=...)`, `Chart.properties(title=...)`, `HConcat/VConcat/
> Layer(title=...)`). Subtitle renders as a second line below the title
> baseline using `subtitle_font_size` (default `title_font_size * 0.85`)
> and `subtitle_color` (default `theme.label_color`). When subtitle is
> `None`, title rendering is byte-identical to passing a bare string.
```

And to §3.13 after the existing dated notes:
```markdown
> **2026-05-11 (Schwabish SB1):** `Title(..., subtitle_font_size,
> subtitle_color)` defaults fall back to `title_font_size * 0.85` and
> `theme.label_color` respectively. No new Theme keys introduced.
```

```bash
git add ferrum-spec.md
git commit -m "docs(spec): §3.11 + §3.13 + §3.19 Schwabish-SB1 dated notes"
```

- [ ] **Step 7: SB1 ratchet check**

```bash
uv run pytest -q 2>&1 | tail -3
PYTHONHOME=~/.local/share/uv/python/cpython-3.10.14-macos-aarch64-none \
DYLD_LIBRARY_PATH=$PYTHONHOME/lib \
cargo test --quiet 2>&1 | tail -3
```

Both must report all-green. SB1 complete.

---

## SB2 — Principles Doc + Skill Scaffolding

No code changes; pure docs and skill plumbing.

### Task 10: Write the principles doc

**Files:**
- Create: `docs/superpowers/specs/2026-05-11-schwabish-principles.md`

- [ ] **Step 1: Write the principles doc following design spec §5 structure**

Write ~600–800 words, no code, this exact section structure:

```markdown
# Schwabish Text-Integration Principles for Ferrum

**Date:** 2026-05-11
**Status:** canonical reference
**Scope:** Schwabish's "integrate text and graphics" principle, operationalized for ferrum's statistical gallery.

## Source

Jonathan Schwabish, *Better Data Visualizations: A Guide for Scholars, Researchers, and Wonks* (Columbia University Press, 2021), Part I "Visualizing Data Effectively," third principle: *integrate text and graphics*. The other two core principles — *show the data* and *reduce the clutter* — are addressed elsewhere in ferrum (the gallery audit's B-rubric and the themes overhaul T1–T4, respectively). This doc operationalizes only the third.

## Why this principle for ferrum

[Two paragraphs — comparative-parity blind spot in the gallery audit; what themes T1–T4 covered (visual polish); what remains (text integration).]

## The four T-categories

### T1 — Active title
[One paragraph. Active title communicates a finding ("ROC — AUC 0.94 (good separation)") vs. descriptive ("ROC curve"). Default is **subjective**; becomes **objective** when a single metric is computable and a clear template exists.]

### T2 — Direct labels
[One paragraph. When ≤4 series and labels are short, prefer text-at-line-endpoint over legend. Tradeoff: legend scales to many series; direct labels lead the eye.]

### T3 — Callouts
[One paragraph. Point at the punchline — max, threshold-crossing, anomaly. Implementation: `annotate_arrow` for leader-line callouts, `annotate_text` for floating annotations.]

### T4 — Inline metrics
[One paragraph. Domain-expected numbers belong on the plot, not the caption: AUC, AP, Brier, R², per-cell counts, importance values.]

## Objective vs subjective

[Half a page. Objective findings can apply autonomously in `/schwabish-improve --from-audit`. Subjective findings stay advisory. Rationale: auto-applied changes must produce a sensible default for every caller; subjective changes depend on dataset and intent.]

## Where these principles live in ferrum

- **Defaults** — `src/ferrum/figures.py`. Each figure-level function carries Schwabish-compliant kwargs (`annotate_auc=True`, `annotate_brier=True`, etc.).
- **Primitives** — `src/ferrum/annotations.py` (`AUCLabel`, `APLabel`, `BrierLabel`, `OutlierLabel`, `annotate_arrow`) and `src/ferrum/title.py` (`Title(subtitle=...)`).
- **Audit** — `.claude/skills/schwabish/`. `/schwabish-improve <target>` runs advisory mode; `/schwabish-improve --from-audit` runs the gallery-autonomous pass.
- **Override hierarchy** — a user passing `Title("custom string")`, `annotate_auc=False`, or `legend=...` explicitly always wins. Schwabish defaults set the floor, not the ceiling.

## Out of scope

[One paragraph — chart-type taxonomy from the book that doesn't apply to ferrum's statistical gallery; "show the data" and "reduce clutter" covered elsewhere; implementation details (those live in the design spec).]
```

Fill in the bracketed paragraphs with content matching the design spec §5; do not leave brackets in the final file.

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/specs/2026-05-11-schwabish-principles.md
git commit -m "docs(specs): schwabish principles reference (SB2)"
```

---

### Task 11: Skill scaffolding

**Files:**
- Create: `.claude/skills/schwabish/SKILL.md`
- Create: `.claude/skills/schwabish/judge_prompt.md`
- Create: `.claude/skills/schwabish/apply_eligibility.md`

- [ ] **Step 1: Write `SKILL.md`**

```markdown
---
name: schwabish
description: Apply Schwabish "integrate text and graphics" principles to a ferrum chart or the entire gallery. Use when the user says /schwabish-improve, "improve text integration", "add direct labels", "make titles active", or wants a Schwabish-style audit of plots beyond peer-parity.
---

# Schwabish — Text-Integration Audit

A two-mode skill that judges ferrum charts against the Schwabish "integrate text and graphics" rubric (four T-categories: active title, direct labels, callouts, inline metrics).

## Modes

### Advisory mode (default)

`/schwabish-improve <target>` — `<target>` is one of:
- a path to a Python file that builds a ferrum chart (e.g. `gallery/plots/01_roc/ferrum_panel.py`)
- a path to an SVG file
- a directory (recursive scan of all charts found)

Dispatches a `schwabish-judge` subagent per target, which reads the chart artifact and a stripped-down rubric (from `judge_prompt.md`). Writes a `schwabish_verdict.md` next to the target (or to `--out <path>`).

Advisory mode never edits chart code.

### Gallery-autonomous mode

`/schwabish-improve --from-audit` — no target argument. Reads `gallery/plots/<row>/ferrum_panel.py` and `gallery/output/<row>/`. For each row:
1. Dispatch `schwabish-judge` (parallel).
2. Filter verdicts to findings with `objective: true` (see `apply_eligibility.md`).
3. Dispatch `schwabish-fixer` to apply those findings to the panel script via `Edit`.
4. Regenerate the panel via `audit.py generate --row <id>`.
5. Run `python-review-lite` on the staged diff. Block status un-stages; clean commits.
6. Aggregate to `gallery/output/SCHWABISH_REPORT.md`.

Subjective findings (title rewrites, subtitle wording, callout placement) appear in `schwabish_verdict.md` for user review but are never auto-applied.

## Reference docs

- `docs/superpowers/specs/2026-05-11-schwabish-principles.md` — canonical reference, embedded as cached prefix in `judge_prompt.md`.
- `docs/superpowers/specs/2026-05-11-schwabish-design.md` — full design spec (the *how*).
```

- [ ] **Step 2: Write `judge_prompt.md`**

```markdown
You are auditing a single ferrum chart against the Schwabish "integrate text and graphics" rubric. You receive:

1. The chart artifact (Python file building the chart, or an SVG, or both).
2. Optional context: a `--context "<string>"` describing dataset / model / intent.

[Embed the contents of `docs/superpowers/specs/2026-05-11-schwabish-principles.md` here verbatim — this is the cached prefix.]

## Output format

Respond with **exactly** this structure, no preamble or trailing prose:

\`\`\`
---
target: <path>
status: <OK | NEEDS_TEXT_INTEGRATION>
findings:
  - id: T1_active_title
    severity: <HIGH | MEDIUM | LOW | NONE>
    objective: <true | false>
  - id: T2_direct_labels
    severity: <...>
    objective: <...>
  - id: T3_callout
    severity: <...>
    objective: <...>
  - id: T4_inline_metric
    severity: <...>
    objective: <...>
---

# Schwabish verdict: <chart description>

## T1 — Active title
<current title> → <suggested title>
**Why:** <one sentence>
**How to apply:** <code snippet>

## T2 — Direct labels
...

## T3 — Callouts
...

## T4 — Inline metrics
...

## Notes
<1–2 sentences qualitative observation>
\`\`\`

## Severity rules
- HIGH = missing objective metric (T4) where a default exists
- MEDIUM = T1 active title or T2 direct labels eligible
- LOW = T3 callout opportunity or cosmetic text issue
- NONE = chart already satisfies the rubric

## Objectivity rules
- T1, T3, and T1_subtitle findings are always `objective: false`.
- T2 is `objective: true` only when series count ≤ 4 AND labels are short strings.
- T4 is `objective: true` when a shipped composite (`AUCLabel`, `APLabel`, `BrierLabel`, etc.) covers the gap, or when flipping a default kwarg closes it.
```

- [ ] **Step 3: Write `apply_eligibility.md`**

```markdown
# Eligibility List — objective findings the autonomous fixer applies

| Finding ID | Rubric | Autonomous action |
|---|---|---|
| `T4_auc_label_missing` | T4 | Append `+ AUCLabel()` on ROC panels |
| `T4_ap_label_missing` | T4 | Append `+ APLabel()` on PR panels |
| `T4_brier_label_missing` | T4 | Append `+ BrierLabel()` on calibration panels |
| `T4_residual_metrics_missing` | T4 | Add `annotate_metrics=True` kwarg or append corner annotation on residuals panels |
| `T4_cell_counts_missing` | T4 | Flip `annotate=True` on `confusion_matrix_chart` |
| `T4_importance_values_missing` | T4 | Flip `show_values=True` on `importance_chart` |
| `T2_direct_labels_eligible` | T2 | Add direct-label overlay + remove legend, only when series count ≤ 4 |
| `T4_pr_baseline_missing` | T4 | Append `+ annotate_hline(prevalence, label="baseline")` on PR panels |
| `T4_residual_zero_line_missing` | T4 | Append `+ annotate_hline(0, stroke_dash=[3,3])` on residuals panels |
| `T4_calibration_diagonal_missing` | T4 | Append diagonal y=x on calibration panels (if missing) |

## Non-eligible (advisory only)

| Finding ID | Rubric | Why not autonomous |
|---|---|---|
| `T1_active_title_*` | T1 | Title rewriting is subjective |
| `T3_callout_*` | T3 | Where to callout depends on data + intent |
| `T1_subtitle_*` | T1 | Subtitle wording is user-supplied semantic context |
| Anything with `objective: false` | — | Per-finding judgment by `schwabish-judge` |
```

- [ ] **Step 4: Commit**

```bash
git add .claude/skills/schwabish/
git commit -m "feat(skill): schwabish skill scaffolding — SKILL.md + judge_prompt + eligibility (SB2)"
```

---

### Task 12: `schwabish-judge` agent

**Files:**
- Create: `.claude/agents/schwabish-judge.md`

- [ ] **Step 1: Write the agent file**

Mirror `.claude/agents/gallery-judge.md` structure. Frontmatter restricts tools to read-only:
```markdown
---
name: schwabish-judge
description: Judges one chart (Python file, SVG, or panel directory) against the Schwabish text-integration rubric (four T-categories). Dispatched in parallel by /schwabish-improve, one per target. Writes verdict.md with YAML frontmatter + prose; never edits code.
tools: Read, Grep, Glob, Bash
---

# Schwabish Judge

You judge **one** ferrum chart artifact against the four-category Schwabish text-integration rubric. Reads the cached `judge_prompt.md` prefix (rubric + principles doc embedded). Writes a single `schwabish_verdict.md` to a path the orchestrator specifies.

## Your input (from the orchestrator)

- `target` — path to a Python panel script and/or SVG
- `out_path` — where to write the verdict
- `context` — optional one-line description of dataset / model / intent

## What to do

1. Read `target`. If it's a directory, find the panel script (`ferrum_panel.py`) and/or rendered PNG.
2. Read the panel script and trace the chart construction — what title is being set? What encodings? What annotations are already present?
3. Apply the rubric from `judge_prompt.md`. For each of T1, T2, T3, T4, decide:
   - Severity (`HIGH | MEDIUM | LOW | NONE`)
   - `objective: true | false` per the rules in `judge_prompt.md`
4. Write `out_path` with the YAML + prose format from `judge_prompt.md`.

## What NOT to do

- Do not edit any chart code. You're read-only.
- Do not propose fabricated subtitles. If `context` is empty, subtitle suggestions stay generic ("consider supplying a subtitle via context").
- Do not score findings outside the four T-categories.
```

- [ ] **Step 2: Commit**

```bash
git add .claude/agents/schwabish-judge.md
git commit -m "feat(agent): schwabish-judge subagent (SB2)"
```

- [ ] **Step 3: SB2 ratchet check**

```bash
uv run pytest -q 2>&1 | tail -3
PYTHONHOME=~/.local/share/uv/python/cpython-3.10.14-macos-aarch64-none \
DYLD_LIBRARY_PATH=$PYTHONHOME/lib \
cargo test --quiet 2>&1 | tail -3
```

Both must report all-green. SB2 complete.

---

## SB3 — Defaults Work on the 8 Figure-Level Functions

This sub-phase regenerates ~30–50 SVG goldens. Standard protocol: `FERRUM_REGENERATE_GOLDENS=1 FERRUM_UPDATE_GOLDENS=1 uv run pytest`, then `python scripts/snapshot-goldens.py`, then `Read` every PNG.

### Task 13: `roc_chart` — flip `annotate_auc=True`, swap to `AUCLabel`, add active title

**Files:**
- Modify: `src/ferrum/figures.py` (`roc_chart` signature ~line 138; docstring; body)
- Modify: `src/ferrum/_diagnostics/charts.py` (`_roc_chart_from_source` body to use AUCLabel)
- Test: `tests/test_roc_default_annotates_auc.py`, `tests/test_roc_annotate_auc_false.py`, `tests/test_roc_chart_active_title.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_roc_default_annotates_auc.py
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

import ferrum as fm


def test_roc_default_emits_auc_text():
    X, y = load_iris(return_X_y=True)
    model = LogisticRegression(max_iter=200).fit(X, y)
    chart = fm.roc_chart(model, X, y)
    svg = chart.to_svg()
    assert "AUC = " in svg
```

```python
# tests/test_roc_annotate_auc_false.py
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

import ferrum as fm


def test_roc_annotate_auc_false_omits_label():
    X, y = load_iris(return_X_y=True)
    model = LogisticRegression(max_iter=200).fit(X, y)
    chart = fm.roc_chart(model, X, y, annotate_auc=False)
    svg = chart.to_svg()
    assert "AUC = " not in svg
```

```python
# tests/test_roc_chart_active_title.py
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

import ferrum as fm


def test_roc_single_curve_active_title():
    X, y = make_classification(n_samples=200, n_classes=2, random_state=0)
    model = LogisticRegression().fit(X, y)
    chart = fm.roc_chart(model, X, y, per_class=False)
    svg = chart.to_svg()
    # active title format: "ROC — AUC X.XXX"
    import re
    assert re.search(r">ROC — AUC \d\.\d{3}<", svg)


def test_roc_per_class_falls_back_to_descriptive_title():
    X, y = make_classification(n_samples=200, n_classes=3, n_informative=3, random_state=0)
    model = LogisticRegression(max_iter=300).fit(X, y)
    chart = fm.roc_chart(model, X, y, per_class=True)
    svg = chart.to_svg()
    assert ">ROC<" in svg
```

- [ ] **Step 2: Run, verify all fail**

```bash
uv run pytest tests/test_roc_default_annotates_auc.py tests/test_roc_annotate_auc_false.py tests/test_roc_chart_active_title.py -v
```

- [ ] **Step 3: Modify `roc_chart` signature**

In `src/ferrum/figures.py` at line 146:
```python
    annotate_auc: bool = False,
```
Replace with:
```python
    annotate_auc: bool = True,
```

Update the docstring accordingly — change `default False` to `default True`, add a sentence: "Uses :class:`AUCLabel` per spec §3.11."

Add new kwarg `subtitle: str | None = None,` to the signature. Document it.

- [ ] **Step 4: Update `_roc_chart_from_source` to emit active title + AUCLabel**

In `src/ferrum/_diagnostics/charts.py` (or wherever `_roc_chart_from_source` lives), at the point the chart is assembled:
```python
# after building the base ROC chart with mark_line + encode(x="fpr", y="tpr", color="class")
from ferrum import AUCLabel, Title
from ferrum.annotations import _trapezoid_auc, _resolve_field
from ferrum._coerce import to_arrow_table

chart = base_chart  # the assembled line chart

# Active title only when a single AUC value makes sense — one curve total.
# Multi-class (`per_class=True`) and multi-model (multiple values in the color
# encoding) both produce multiple AUCs; fall back to the descriptive title and
# let AUCLabel annotate each line.
tbl = to_arrow_table(chart._data)
color_field = _resolve_field(chart._encoding.get("color"))
n_curves = len(set(tbl.column(color_field).to_pylist())) if color_field else 1
if not per_class and n_curves == 1:
    fpr = np.asarray(tbl.column("fpr").to_pylist(), dtype=float)
    tpr = np.asarray(tbl.column("tpr").to_pylist(), dtype=float)
    auc_value = _trapezoid_auc(fpr, tpr)
    chart = chart.properties(title=Title(f"ROC — AUC {auc_value:.3f}", subtitle=subtitle))
else:
    chart = chart.properties(title=Title("ROC", subtitle=subtitle))

if annotate_auc:
    chart = chart + AUCLabel(position="end")

return chart
```

Remove any existing manual-AUC-text-injection branch (the old `annotate_auc=True` path).

- [ ] **Step 5: Run the 3 new tests**

```bash
uv run pytest tests/test_roc_default_annotates_auc.py tests/test_roc_annotate_auc_false.py tests/test_roc_chart_active_title.py -v
```

Expected: all pass.

- [ ] **Step 6: Hold golden regen + commit until Task 21**

The full SB3 set must regen and inspect together.

---

### Task 14: `pr_chart` — flip `annotate_ap=True`, add baseline hline, add active title

**Files:**
- Modify: `src/ferrum/figures.py` (`pr_chart`)
- Modify: `src/ferrum/_diagnostics/charts.py` (`_pr_chart_from_source`)
- Test: `tests/test_pr_default_annotates_ap.py`, `tests/test_pr_baseline_hline.py`

Follow the same pattern as Task 13:

- [ ] **Step 1: Tests**

```python
# tests/test_pr_default_annotates_ap.py
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

import ferrum as fm


def test_pr_default_emits_ap_text():
    X, y = load_iris(return_X_y=True)
    model = LogisticRegression(max_iter=200).fit(X, y)
    chart = fm.pr_chart(model, X, y)
    svg = chart.to_svg()
    assert "AP = " in svg
```

```python
# tests/test_pr_baseline_hline.py
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

import ferrum as fm


def test_pr_baseline_hline_drawn():
    X, y = make_classification(n_samples=200, n_classes=2, weights=[0.7, 0.3], random_state=0)
    model = LogisticRegression().fit(X, y)
    chart = fm.pr_chart(model, X, y, per_class=False)
    svg = chart.to_svg()
    # baseline at positive-class prevalence ≈ 0.3 — at minimum, a rule was drawn
    # (full coordinate check is brittle; presence of an extra <line> in the chart is a proxy)
    assert svg.count("<line") >= 1  # placeholder — refine after locating actual rule emission
```

- [ ] **Step 2: Modify `pr_chart` signature**

In `src/ferrum/figures.py`, change `annotate_ap: bool = False` → `annotate_ap: bool = True`. Add `subtitle: str | None = None` kwarg.

- [ ] **Step 3: Modify `_pr_chart_from_source` body**

Similar to roc:
```python
from ferrum import APLabel, Title, annotate_hline
from ferrum.annotations import _ap_step, _resolve_field
from ferrum._coerce import to_arrow_table

# after assembling the base PR chart
tbl = to_arrow_table(chart._data)
color_field = _resolve_field(chart._encoding.get("color"))
n_curves = len(set(tbl.column(color_field).to_pylist())) if color_field else 1
if not per_class and n_curves == 1:
    recall = np.asarray(tbl.column("recall").to_pylist(), dtype=float)
    precision = np.asarray(tbl.column("precision").to_pylist(), dtype=float)
    ap_value = _ap_step(recall, precision)
    chart = chart.properties(title=Title(f"Precision–Recall — AP {ap_value:.3f}", subtitle=subtitle))
else:
    chart = chart.properties(title=Title("Precision–Recall", subtitle=subtitle))

# Baseline at positive-class prevalence — only emit when binary (a single
# prevalence value is meaningful). Multi-class would need per-class baselines;
# defer that to a follow-up rather than fabricating.
y_arr = np.asarray(source._y) if hasattr(source, "_y") else None
if y_arr is not None and len(np.unique(y_arr)) == 2:
    # interpret class "1" as positive when binary
    prevalence = float((y_arr == 1).mean())
    if 0 < prevalence < 1:
        chart = chart + annotate_hline(prevalence, stroke_dash=[3, 3])

if annotate_ap:
    chart = chart + APLabel(position="end")

return chart
```

- [ ] **Step 4: Run new tests + hold commit until Task 21.**

---

### Task 15: `calibration_chart` — new `annotate_brier=True` kwarg + active title

**Files:**
- Modify: `src/ferrum/figures.py` (`calibration_chart`)
- Test: `tests/test_calibration_default_annotates_brier.py`

- [ ] **Step 1: Test**

```python
# tests/test_calibration_default_annotates_brier.py
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

import ferrum as fm


def test_calibration_default_emits_brier_text():
    X, y = make_classification(n_samples=500, n_classes=2, random_state=0)
    model = LogisticRegression().fit(X, y)
    chart = fm.calibration_chart(model, X=X, y=y)
    svg = chart.to_svg()
    assert "Brier = " in svg


def test_calibration_brier_false_omits_label():
    X, y = make_classification(n_samples=500, n_classes=2, random_state=0)
    model = LogisticRegression().fit(X, y)
    chart = fm.calibration_chart(model, X=X, y=y, annotate_brier=False)
    svg = chart.to_svg()
    assert "Brier = " not in svg
```

- [ ] **Step 2: Modify `calibration_chart` signature**

Add `annotate_brier: bool = True,` and `subtitle: str | None = None,` to the signature. Document both kwargs.

- [ ] **Step 3: Modify body to emit BrierLabel + active title**

```python
from ferrum import BrierLabel, Title
from ferrum.annotations import _brier_score
from ferrum._coerce import to_arrow_table

# after assembling the chart, before return.
# `n_models` is the count of positional model_or_sources the caller passed.
if n_models == 1:
    tbl = to_arrow_table(chart._data)
    p = np.asarray(tbl.column("predicted").to_pylist(), dtype=float)
    obs = np.asarray(tbl.column("observed").to_pylist(), dtype=float)
    brier = _brier_score(p, obs)
    chart = chart.properties(title=Title(f"Calibration — Brier {brier:.3f}", subtitle=subtitle))
else:
    chart = chart.properties(title=Title("Calibration", subtitle=subtitle))

if annotate_brier:
    chart = chart + BrierLabel(position="corner")
return chart
```

- [ ] **Step 4: Test + hold commit.**

---

### Task 16: `confusion_matrix_chart` verification

**Files:**
- Test: `tests/test_confusion_matrix_cell_counts_render.py`

The docstring claims `annotate=True` is default. Audit historically flagged missing cell counts. Verify which is true; fix if broken.

- [ ] **Step 1: Test**

```python
# tests/test_confusion_matrix_cell_counts_render.py
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

import ferrum as fm


def test_confusion_matrix_default_renders_cell_text():
    X, y = load_iris(return_X_y=True)
    model = LogisticRegression(max_iter=200).fit(X, y)
    chart = fm.confusion_matrix_chart(model, X, y)
    svg = chart.to_svg()
    # at minimum one numeric cell value emitted as <text>
    import re
    assert re.search(r"<text[^>]*>\s*\d", svg), "no numeric <text> in confusion matrix SVG"
```

- [ ] **Step 2: Run test**

```bash
uv run pytest tests/test_confusion_matrix_cell_counts_render.py -v
```

If it passes: docstring is accurate; no code change needed. Skip to Step 5.
If it fails: cell annotations are broken. Continue to Step 3.

- [ ] **Step 3 (only if Step 2 failed): Locate the confusion_matrix chart construction**

```bash
grep -rn "confusion_matrix" src/ferrum/_diagnostics/ src/ferrum/figures.py | grep -v test
```

Trace the `annotate=True` path. Likely missing: a `mark_text` overlay layer keyed to the cell values.

- [ ] **Step 4 (only if Step 2 failed): Fix the cell annotation emission**

In the confusion_matrix chart construction, ensure when `annotate=True` a `mark_text` layer is added with the cell value as the `text=` encoding. Re-run the test.

- [ ] **Step 5: Hold commit until Task 21.**

---

### Task 17: `residuals_chart` — new `annotate_metrics=True`, corner R²/RMSE/MAE

**Files:**
- Modify: `src/ferrum/figures.py` (`residuals_chart` signature + body)
- Test: `tests/test_residuals_default_annotates_metrics.py`

- [ ] **Step 1: Test**

```python
# tests/test_residuals_default_annotates_metrics.py
import numpy as np
from sklearn.linear_model import LinearRegression

import ferrum as fm


def test_residuals_default_emits_r2_rmse_mae():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (200, 3))
    y = X.sum(axis=1) + rng.normal(0, 0.5, 200)
    model = LinearRegression().fit(X, y)
    chart = fm.residuals_chart(model, X, y)
    svg = chart.to_svg()
    assert "R²" in svg
    assert "RMSE" in svg
    assert "MAE" in svg


def test_residuals_annotate_metrics_false_omits():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (200, 3))
    y = X.sum(axis=1) + rng.normal(0, 0.5, 200)
    model = LinearRegression().fit(X, y)
    chart = fm.residuals_chart(model, X, y, annotate_metrics=False)
    svg = chart.to_svg()
    assert "R²" not in svg
```

- [ ] **Step 2: Modify `residuals_chart` signature**

Add `annotate_metrics: bool = True,` and `subtitle: str | None = None,` kwargs. Document.

- [ ] **Step 3: Modify body**

```python
from ferrum.annotations import annotate_text
from ferrum import Title
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from ferrum._coerce import to_arrow_table

# after assembling the base residuals chart
if annotate_metrics:
    # ModelSource stores X/y under _X/_y; compute predictions via model.predict.
    y_true = np.asarray(source._y)
    y_pred = np.asarray(source._model.predict(source._X.to_pandas() if hasattr(source._X, "to_pandas") else source._X))
    r2 = r2_score(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    text = f"R² {r2:.3f}\nRMSE {rmse:.3f}\nMAE {mae:.3f}"
    # Place at the top-right corner of the existing residuals plot. Read the
    # already-assembled chart's data to find the corner of the plotted region.
    tbl = to_arrow_table(chart._data)
    fitted_col = "fitted" if "fitted" in tbl.column_names else "predicted"
    resid_col = "residual" if "residual" in tbl.column_names else "residuals"
    fitted_arr = np.asarray(tbl.column(fitted_col).to_pylist(), dtype=float)
    resid_arr = np.asarray(tbl.column(resid_col).to_pylist(), dtype=float)
    chart = chart + annotate_text(
        float(np.max(fitted_arr)), float(np.max(resid_arr)),
        text, dx=-4, dy=4, align="right", baseline="top",
    )

chart = chart.properties(title=Title("Residuals", subtitle=subtitle))
return chart
```

- [ ] **Step 4: Run, hold commit.**

---

### Task 18: `importance_chart` — new `show_values=True`

**Files:**
- Modify: `src/ferrum/figures.py` (`importance_chart`)
- Test: `tests/test_importance_default_shows_values.py`

- [ ] **Step 1: Test**

```python
# tests/test_importance_default_shows_values.py
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

import ferrum as fm


def test_importance_default_shows_numeric_labels_on_bars():
    X, y = load_iris(return_X_y=True)
    model = RandomForestClassifier(random_state=0).fit(X, y)
    chart = fm.importance_chart(model, X, y)
    svg = chart.to_svg()
    # at minimum one numeric importance value emitted as text near a bar end
    import re
    assert re.search(r"<text[^>]*>\s*0\.\d{2,}\s*<", svg)


def test_importance_show_values_false_omits():
    X, y = load_iris(return_X_y=True)
    model = RandomForestClassifier(random_state=0).fit(X, y)
    chart = fm.importance_chart(model, X, y, show_values=False)
    svg = chart.to_svg()
    import re
    # the only numeric <text> nodes should be tick labels — no bar-end labels
    # heuristic: ≤ ~12 numeric text elements (axis ticks + a few)
    matches = re.findall(r"<text[^>]*>\s*0\.\d", svg)
    assert len(matches) <= 12
```

- [ ] **Step 2: Modify `importance_chart` signature**

Add `show_values: bool = True,` and `subtitle: str | None = None,` kwargs.

- [ ] **Step 3: Modify body to emit text-at-bar-end when `show_values=True`**

```python
from ferrum.annotations import annotate_text
from ferrum import Title

# after the base bar chart is assembled
if show_values:
    importances = df["importance"].to_numpy()
    features = df["feature"].to_list()
    for feat, imp in zip(features, importances):
        if orient == "horizontal":
            chart = chart + annotate_text(imp, feat, f"{imp:.2f}", dx=4, align="left")
        else:
            chart = chart + annotate_text(feat, imp, f"{imp:.2f}", dy=-4, baseline="bottom")

chart = chart.properties(title=Title("Feature importance", subtitle=subtitle))
return chart
```

- [ ] **Step 4: Run, hold commit.**

---

### Task 19: Direct-label helper + `learning_curve_chart`

**Files:**
- Create: `src/ferrum/_direct_label.py`
- Modify: `src/ferrum/figures.py` (`learning_curve_chart`)
- Test: `tests/test_direct_label_helper.py`, `tests/test_learning_curve_default_direct_labels.py`

- [ ] **Step 1: Write helper test**

```python
# tests/test_direct_label_helper.py
import polars as pl

from ferrum import Chart
from ferrum._direct_label import _direct_label_endpoint


def test_direct_label_endpoint_emits_text_at_max_x_per_series():
    df = pl.DataFrame({
        "x": [1, 2, 3, 1, 2, 3],
        "y": [0.5, 0.6, 0.7, 0.4, 0.5, 0.5],
        "split": ["train", "train", "train", "val", "val", "val"],
    })
    base = Chart(df).encode(x="x", y="y", color="split").mark_line()
    chart = _direct_label_endpoint(base, label_field="split")
    svg = chart.to_svg()
    assert ">train<" in svg
    assert ">val<" in svg
```

- [ ] **Step 2: Implement helper**

```python
# src/ferrum/_direct_label.py
"""Private helper — emit text labels at the endpoint of each series."""
from __future__ import annotations

import numpy as np
import polars as pl

from ferrum.annotations import annotate_text
from ferrum.chart import Chart


def _direct_label_endpoint(chart: Chart, label_field: str, position: str = "end") -> Chart:
    """Append a text label at the endpoint of each series of ``chart``.

    Returns the chart augmented with one ``annotate_text`` overlay per
    unique value of ``label_field``. Used by ``learning_curve_chart``,
    ``validation_curve_chart``, and the gallery-autonomous fixer.
    """
    from ferrum.annotations import _resolve_field
    from ferrum._coerce import to_arrow_table

    x_col = _resolve_field(chart._encoding.get("x"))
    y_col = _resolve_field(chart._encoding.get("y"))
    tbl = to_arrow_table(chart._data)
    if x_col is None or y_col is None or label_field not in tbl.column_names:
        return chart  # bail rather than crash
    series_arr = np.asarray(tbl.column(label_field).to_pylist())
    x_all = np.asarray(tbl.column(x_col).to_pylist(), dtype=float)
    y_all = np.asarray(tbl.column(y_col).to_pylist(), dtype=float)
    # Augmented-DataFrame pattern (mirrors _apply_metric_label) so the labels
    # share `chart._data` and overlay without HConcat fallback.
    label_col_name = "_direct_label_text"
    labels_col: list = [None] * len(series_arr)
    for series in sorted(set(series_arr.tolist()), key=str):
        mask = series_arr == series
        idx_in_mask = int(np.argmax(x_all[mask]) if position == "end" else np.argmin(x_all[mask]))
        # find global index of the chosen row
        global_idx = int(np.where(mask)[0][idx_in_mask])
        labels_col[global_idx] = str(series)
    base_pl = pl.from_arrow(tbl) if not isinstance(chart._data, pl.DataFrame) else chart._data
    augmented = base_pl.with_columns(pl.Series(label_col_name, labels_col))
    chart_aug = chart._clone()
    chart_aug._data = augmented
    annot_layer = (
        Chart(augmented)
        .mark_text(align="left", dx=4)
        .encode(x=x_col, y=y_col, text=label_col_name)
    )
    return chart_aug + annot_layer
```

- [ ] **Step 3: Write learning_curve test**

```python
# tests/test_learning_curve_default_direct_labels.py
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

import ferrum as fm


def test_learning_curve_default_emits_direct_labels():
    X, y = load_iris(return_X_y=True)
    chart = fm.learning_curve_chart(LogisticRegression(max_iter=200), X, y, cv=3)
    svg = chart.to_svg()
    assert ">train<" in svg
    assert ">val<" in svg


def test_learning_curve_legend_suppressed_when_direct_labels():
    X, y = load_iris(return_X_y=True)
    chart = fm.learning_curve_chart(LogisticRegression(max_iter=200), X, y, cv=3)
    svg = chart.to_svg()
    # heuristic: legend rendering emits a <g class="legend"> or similar
    # If the existing chart structure marks legend distinctly, assert absent.
    # Otherwise, count text occurrences of "train"/"val" — should be exactly 2 (the direct labels), not 4 (2 direct + 2 in legend).
    assert svg.count(">train<") == 1
    assert svg.count(">val<") == 1
```

- [ ] **Step 4: Modify `learning_curve_chart` body**

In `src/ferrum/_diagnostics/charts.py` (or wherever `_learning_curve_chart_from_source` lives), after assembling the base chart:
```python
from ferrum._direct_label import _direct_label_endpoint
from ferrum import Title

# series_count is always 2 (train + val); apply direct labels
chart = _direct_label_endpoint(chart, label_field="split")
# suppress the legend — pass a flag to .encode(color=alt.Color(..., legend=None))
# or set chart-level legend kwarg if the API supports it
chart = chart.properties(title=Title("Learning curve", subtitle=subtitle))
```

Find the existing `color=` encoding for `split` and suppress its legend (consult `src/ferrum/chart.py` for the exact syntax — `Color(legend=None)` or `.encode(color={"field": "split", "legend": null})` are likely candidates).

- [ ] **Step 5: Test + hold commit.**

---

### Task 20: `validation_curve_chart` — same treatment

**Files:**
- Modify: `src/ferrum/figures.py` (`validation_curve_chart`)
- Test: `tests/test_validation_curve_default_direct_labels.py`

- [ ] **Step 1: Test (mirror of learning_curve)**

```python
# tests/test_validation_curve_default_direct_labels.py
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import numpy as np

import ferrum as fm


def test_validation_curve_default_emits_direct_labels():
    X, y = load_iris(return_X_y=True)
    chart = fm.validation_curve_chart(
        LogisticRegression(max_iter=200), X, y,
        param_name="C", param_range=np.logspace(-3, 1, 5), cv=3,
    )
    svg = chart.to_svg()
    assert ">train<" in svg
    assert ">val<" in svg
```

- [ ] **Step 2: Apply same `_direct_label_endpoint` + `Title` pattern** as Task 19.

- [ ] **Step 3: Run test + hold commit.**

---

### Task 21: Golden regen + inspection + commit SB3

**Files:**
- Regenerate: `tests/goldens/**/*.svg` (~30–50 files) + `tests/test_phase_9_e2e/goldens/*.svg`
- Snapshot: `tests/goldens/**/*.png` via `python scripts/snapshot-goldens.py`

- [ ] **Step 1: Run the full pytest suite to surface every golden test that depends on the 8 figure functions**

```bash
uv run pytest -q 2>&1 | tail -20
```

Note the failures — these are the goldens that drift on SB3 changes.

- [ ] **Step 2: Regenerate**

```bash
FERRUM_REGENERATE_GOLDENS=1 FERRUM_UPDATE_GOLDENS=1 uv run pytest -q 2>&1 | tail -5
```

Expected: full suite green; goldens updated.

- [ ] **Step 3: Rasterize PNG snapshots**

```bash
uv run python scripts/snapshot-goldens.py
```

Expected: a PNG next to every regenerated SVG.

- [ ] **Step 4: Read every regenerated PNG in batches of ~10**

Identify the regenerated files:
```bash
git status --short tests/goldens tests/test_phase_9_e2e/goldens | head -50
```

For each PNG, `Read` it. Check:
- AUC / AP / Brier text present at expected position; numeric format `.3f`.
- R² / RMSE / MAE corner annotation present on residuals goldens.
- Cell text visible on confusion-matrix goldens (verifies Task 16).
- Direct-label text at line endpoints on learning / validation curve goldens; legend absent.
- No resvg-py path truncation. For any panel that looks empty, cross-check:
  ```bash
  grep -oE 'd="M' tests/goldens/path/to/foo.svg | wc -l
  ```
  Many paths spanning the plot extent in the SVG but tiny in PNG = resvg-py truncation, not a real bug.

If any PNG shows a regression (text overlapping the data, label off-screen, etc.), revert that golden, fix the figure-function code, regenerate, re-inspect.

- [ ] **Step 5: Commit SB3 in two parts**

First, the code changes:
```bash
git add src/ferrum/figures.py src/ferrum/_diagnostics/charts.py src/ferrum/_direct_label.py \
  tests/test_roc_default_annotates_auc.py tests/test_roc_annotate_auc_false.py \
  tests/test_roc_chart_active_title.py tests/test_pr_default_annotates_ap.py \
  tests/test_pr_baseline_hline.py tests/test_calibration_default_annotates_brier.py \
  tests/test_confusion_matrix_cell_counts_render.py \
  tests/test_residuals_default_annotates_metrics.py \
  tests/test_importance_default_shows_values.py \
  tests/test_direct_label_helper.py tests/test_learning_curve_default_direct_labels.py \
  tests/test_validation_curve_default_direct_labels.py
git commit -m "feat(figures): schwabish defaults across 8 figure-level functions (SB3)"
```

Then, the regenerated goldens:
```bash
git add tests/goldens/ tests/test_phase_9_e2e/goldens/
git commit -m "test(goldens): regenerate after SB3 schwabish-defaults landing"
```

- [ ] **Step 6: Append SB3 dated note to `ferrum-spec.md §3.14`**

Append at the end of §3.14 (per design spec §8). Use the exact wording from design spec §8.

```bash
git add ferrum-spec.md
git commit -m "docs(spec): §3.14 Schwabish-SB3 dated note — figure-function defaults"
```

- [ ] **Step 7: SB3 ratchet check**

```bash
uv run pytest -q 2>&1 | tail -3
PYTHONHOME=~/.local/share/uv/python/cpython-3.10.14-macos-aarch64-none \
DYLD_LIBRARY_PATH=$PYTHONHOME/lib \
cargo test --quiet 2>&1 | tail -3
```

Both must report all-green. SB3 complete.

---

## SB4 — Skill Advisory Mode

### Task 22: SKILL.md target detection + dispatch logic

**Files:**
- Modify: `.claude/skills/schwabish/SKILL.md`

- [ ] **Step 1: Extend SKILL.md with target-detection prose**

Add this section to `.claude/skills/schwabish/SKILL.md` (after the "Modes" header):

```markdown
## Target detection (advisory mode)

When `/schwabish-improve <target>` is invoked, classify `<target>`:

1. **Single Python file** (ends with `.py`): treat as a panel script; dispatch one `schwabish-judge` with `target=<path>` and `out_path=<path>.schwabish_verdict.md`.

2. **Single SVG file** (ends with `.svg`): same as above but without panel-script source; judge reads only the SVG.

3. **Directory**: walk for `*.py` and `*.svg` files. Skip anything matching `__pycache__/`, `.venv/`, `node_modules/`. Dispatch one `schwabish-judge` per discovered chart artifact, in parallel.

4. **Otherwise**: error out with "target must be a .py file, .svg file, or directory".

## Dispatch (advisory mode)

For each classified target:
- Build a prompt for `schwabish-judge`:
  - "Read `<target>`. Apply the rubric in `judge_prompt.md` (cached prefix). Write the verdict to `<out_path>`. Context: `<--context value or empty>`."
- Use the `Agent` tool with `subagent_type=schwabish-judge`, in parallel where multiple targets exist.

## Aggregation (advisory mode)

After all judges return:
- If `--out` was a single file, the single verdict is already written.
- If targets were a directory, write `<directory>/SCHWABISH_VERDICTS_INDEX.md` listing all per-target verdict paths with severities at-a-glance.
```

- [ ] **Step 2: Commit**

```bash
git add .claude/skills/schwabish/SKILL.md
git commit -m "feat(skill): schwabish advisory mode target detection + dispatch (SB4)"
```

---

### Task 23: Skill test — judge dispatch

**Files:**
- Test: `tests/test_schwabish_judge_dispatch.py`

- [ ] **Step 1: Test (lightweight smoke — full skill behavior is tested via manual invocation)**

```python
# tests/test_schwabish_judge_dispatch.py
"""Smoke test that the schwabish skill and judge agent files exist and parse."""
from pathlib import Path

import yaml


def test_skill_md_has_frontmatter():
    skill_path = Path(".claude/skills/schwabish/SKILL.md")
    text = skill_path.read_text()
    assert text.startswith("---")
    body = text.split("---", 2)
    assert len(body) == 3
    fm = yaml.safe_load(body[1])
    assert fm["name"] == "schwabish"
    assert "description" in fm


def test_judge_agent_has_read_only_tools():
    agent_path = Path(".claude/agents/schwabish-judge.md")
    text = agent_path.read_text()
    body = text.split("---", 2)
    fm = yaml.safe_load(body[1])
    tools = set(t.strip() for t in fm["tools"].split(","))
    assert "Edit" not in tools
    assert "Write" not in tools
    assert "Read" in tools


def test_judge_prompt_embeds_principles_doc():
    prompt = Path(".claude/skills/schwabish/judge_prompt.md").read_text()
    assert "T1" in prompt and "T2" in prompt and "T3" in prompt and "T4" in prompt
    assert "objective" in prompt.lower()


def test_eligibility_list_lists_objective_findings():
    elig = Path(".claude/skills/schwabish/apply_eligibility.md").read_text()
    for finding_id in [
        "T4_auc_label_missing",
        "T4_ap_label_missing",
        "T4_brier_label_missing",
        "T4_residual_metrics_missing",
        "T4_cell_counts_missing",
        "T4_importance_values_missing",
        "T2_direct_labels_eligible",
        "T4_pr_baseline_missing",
        "T4_residual_zero_line_missing",
        "T4_calibration_diagonal_missing",
    ]:
        assert finding_id in elig, f"missing eligibility entry: {finding_id}"
```

- [ ] **Step 2: Run**

```bash
uv run pytest tests/test_schwabish_judge_dispatch.py -v
```

Expected: all pass.

- [ ] **Step 3: Commit**

```bash
git add tests/test_schwabish_judge_dispatch.py
git commit -m "test(skill): schwabish advisory-mode smoke test (SB4)"
```

- [ ] **Step 4: SB4 ratchet check**

```bash
uv run pytest -q 2>&1 | tail -3
```

Must be green. SB4 complete.

---

## SB5 — Skill Gallery-Autonomous Mode

### Task 24: `schwabish-fixer` agent

**Files:**
- Create: `.claude/agents/schwabish-fixer.md`

- [ ] **Step 1: Write the fixer agent file**

```markdown
---
name: schwabish-fixer
description: Applies objective Schwabish findings to gallery panel scripts. Reads schwabish_verdict.md, filters to findings with objective:true, applies eligibility-listed actions via Edit, idempotent. Restricted to gallery/plots/<row>/ferrum_panel.py — never edits src/ferrum/.
tools: Read, Grep, Glob, Edit, Bash
---

# Schwabish Fixer

You apply **objective** Schwabish findings to one gallery row's panel script. You are restricted to `gallery/plots/<row>/ferrum_panel.py` — you do not edit `src/ferrum/` source code.

## Your input (from the orchestrator)

- `row` — gallery row identifier (e.g., `01_roc`)
- `verdict_path` — path to the row's `schwabish_verdict.md`
- `eligibility_path` — path to `.claude/skills/schwabish/apply_eligibility.md`

## What to do

1. Read the verdict. Parse the YAML frontmatter `findings` list.
2. Read the eligibility list. Note the action per `finding_id`.
3. For each finding where `objective: true` AND the `finding_id` appears in the eligibility list:
   - Read `gallery/plots/<row>/ferrum_panel.py`.
   - Check **idempotence first**: if the action's target primitive is already present (e.g., `AUCLabel()` is already in the file), skip.
   - Otherwise, apply the action via `Edit`. Append composites at the end of the chart construction expression chain; flip kwargs in-place.
4. After all eligible findings are applied, regenerate:
   ```bash
   uv run python .claude/skills/audit-gallery/audit.py generate --row <row>
   ```
5. Write a diff snapshot:
   ```bash
   git diff -- gallery/plots/<row>/ > gallery/output/<row>/schwabish_applied.diff
   ```
6. Return a summary listing applied finding IDs and skipped (idempotent) finding IDs.

## What NOT to do

- Do not edit `src/ferrum/`, `crates/`, `tests/`, or any file outside `gallery/`.
- Do not apply subjective findings (`objective: false`). They stay in the verdict for the user.
- Do not commit. The orchestrator handles commits after the lite-review gate.
```

- [ ] **Step 2: Commit**

```bash
git add .claude/agents/schwabish-fixer.md
git commit -m "feat(agent): schwabish-fixer subagent for gallery-autonomous mode (SB5)"
```

---

### Task 25: SKILL.md gallery-autonomous flow

**Files:**
- Modify: `.claude/skills/schwabish/SKILL.md`

- [ ] **Step 1: Extend SKILL.md with the autonomous flow**

Append to `.claude/skills/schwabish/SKILL.md`:

```markdown
## Gallery-autonomous flow (`--from-audit`)

When `/schwabish-improve --from-audit` is invoked:

1. **Discover rows.** Walk `gallery/plots/`. For each row directory:
   - Read `config.toml`. Skip if `ferrum_status` is `BLOCKED` or `NOT_WIRED`.
   - Verify `gallery/plots/<row>/ferrum_panel.py` exists.

2. **Judge in parallel.** Dispatch one `schwabish-judge` per discovered row:
   - `target = gallery/plots/<row>/ferrum_panel.py`
   - `out_path = gallery/output/<row>/schwabish_verdict.md`
   - `context = ""` (row config files do not carry semantic context yet)

3. **Filter to objective findings.** For each verdict file, parse the YAML; collect rows where ≥1 finding has `objective: true`.

4. **Apply via fixer (parallel).** For each row from step 3, dispatch `schwabish-fixer`:
   - `row = <id>`
   - `verdict_path = gallery/output/<row>/schwabish_verdict.md`
   - `eligibility_path = .claude/skills/schwabish/apply_eligibility.md`

5. **Stage + lite-review.** After all fixers return:
   ```bash
   git add gallery/plots/
   ```
   Dispatch `python-review-lite` with the staged diff. Three outcomes:
   - **clean** → proceed to step 6.
   - **block** → un-stage (`git reset HEAD gallery/plots/`), report the review verdict back to the user, halt.
   - **escalate** → un-stage, report, halt.

6. **Commit per row.** For each row touched, one commit:
   ```bash
   git add gallery/plots/<row>/ gallery/output/<row>/
   git commit -m "feat(gallery): schwabish improvements on row <id>"
   ```

7. **Aggregate.** Write `gallery/output/SCHWABISH_REPORT.md`:
   ```markdown
   # Schwabish Report — <ISO timestamp>

   ## Rows updated: <N>

   ### Row 01_roc
   - T4_auc_label_missing: applied
   - T2_direct_labels_eligible: skipped (only 2 series, but legend already absent)

   ### Subjective findings for user review
   <list of objective:false findings across all rows, grouped by row>
   ```
   Commit:
   ```bash
   git add gallery/output/SCHWABISH_REPORT.md
   git commit -m "docs(gallery): schwabish report — <ISO>"
   ```

## Cycle tracking

If `python-review-lite` returns `block`, the orchestrator dispatches the same fixer for that row up to 3 times. On the 3rd consecutive block for the same row, escalate to the user and halt.
```

- [ ] **Step 2: Commit**

```bash
git add .claude/skills/schwabish/SKILL.md
git commit -m "feat(skill): schwabish gallery-autonomous flow (SB5)"
```

---

### Task 26: Idempotence + eligibility tests

**Files:**
- Test: `tests/test_schwabish_eligibility_list.py`, `tests/test_schwabish_from_audit_idempotent.py`

- [ ] **Step 1: Write eligibility-list integrity test**

```python
# tests/test_schwabish_eligibility_list.py
"""Subjective findings never appear in the eligibility list."""
import re
from pathlib import Path


def test_eligibility_list_excludes_subjective_finding_ids():
    text = Path(".claude/skills/schwabish/apply_eligibility.md").read_text()
    body = text.split("## Non-eligible")[0]  # only the eligible section
    # subjective IDs that must NOT appear in the eligible section
    forbidden = ["T1_active_title", "T3_callout", "T1_subtitle"]
    for fid in forbidden:
        assert fid not in body, f"subjective finding {fid!r} found in eligibility list"


def test_eligibility_list_only_lists_T2_or_T4_in_eligible_section():
    text = Path(".claude/skills/schwabish/apply_eligibility.md").read_text()
    eligible_section = text.split("## Non-eligible")[0]
    finding_ids = re.findall(r"T\d_\w+", eligible_section)
    for fid in finding_ids:
        assert fid.startswith("T2_") or fid.startswith("T4_"), \
            f"non-T2/T4 finding {fid!r} in eligible section"
```

- [ ] **Step 2: Write idempotence smoke test**

```python
# tests/test_schwabish_from_audit_idempotent.py
"""Idempotence: if the panel already has AUCLabel(), the fixer must skip.

This is a documentation/contract test — actually invoking the agent requires
the full Agent tool runtime. We assert the agent's frontmatter documents
the idempotence requirement.
"""
from pathlib import Path


def test_fixer_agent_documents_idempotence():
    text = Path(".claude/agents/schwabish-fixer.md").read_text()
    assert "idempotence" in text.lower() or "idempotent" in text.lower(), \
        "schwabish-fixer must document idempotence"


def test_fixer_restricted_to_gallery():
    text = Path(".claude/agents/schwabish-fixer.md").read_text()
    assert "do not edit `src/ferrum/`" in text.lower() or \
           "restricted to `gallery/" in text.lower(), \
        "schwabish-fixer must document scope restriction"
```

- [ ] **Step 3: Run**

```bash
uv run pytest tests/test_schwabish_eligibility_list.py tests/test_schwabish_from_audit_idempotent.py -v
```

Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add tests/test_schwabish_eligibility_list.py tests/test_schwabish_from_audit_idempotent.py
git commit -m "test(skill): schwabish eligibility + idempotence contract tests (SB5)"
```

---

### Task 27: Manual smoke run against the gallery

**Files:** no new files — manual verification step.

- [ ] **Step 1: Run the audit-gallery generate step to ensure panels exist**

```bash
uv run python .claude/skills/audit-gallery/audit.py generate 2>&1 | tail -10
```

Expected: panels regenerate in `gallery/output/`.

- [ ] **Step 2: Manually invoke `/schwabish-improve --from-audit`**

In the Claude Code session, run: `/schwabish-improve --from-audit`.

Expected: Claude dispatches `schwabish-judge` per row in parallel, writes verdicts to `gallery/output/<row>/schwabish_verdict.md`, dispatches `schwabish-fixer` for rows with objective findings, regenerates touched panels, runs `python-review-lite`, commits, writes `SCHWABISH_REPORT.md`.

- [ ] **Step 3: Inspect the output**

```bash
ls gallery/output/*/schwabish_verdict.md
cat gallery/output/SCHWABISH_REPORT.md
```

- [ ] **Step 4: Verify no `src/ferrum/` files were touched by the fixer**

```bash
git log --since="1 hour ago" --name-only | grep -E "^src/ferrum/|^crates/"
```

Expected: empty output. The fixer must not have touched library source.

- [ ] **Step 5: SB5 + full ratchet check**

```bash
uv run pytest -q 2>&1 | tail -3
PYTHONHOME=~/.local/share/uv/python/cpython-3.10.14-macos-aarch64-none \
DYLD_LIBRARY_PATH=$PYTHONHOME/lib \
cargo test --quiet 2>&1 | tail -3
```

Both must report all-green. SB5 complete.

---

## Final Step: Merge `feat/schwabish` to `main`

- [ ] **Step 1: Final rebase from main**

```bash
git fetch origin
git rebase origin/main
```

Resolve any conflicts; re-run `uv run pytest -q` and `cargo test --quiet` after each conflict resolution.

- [ ] **Step 2: Push the branch**

```bash
git push -u origin feat/schwabish
```

- [ ] **Step 3: Confirm with the user before merging to main**

CLAUDE.md hard constraint — never push to main directly without confirmation. Ask the user to review the branch and decide between a merge commit, fast-forward merge, or PR.

---

## Appendix — Total Test Footprint Added

| Sub-phase | New test files |
|---|---|
| SB1 | `test_title_value_class.py`, `test_chart_title_accepts_title_class.py`, `test_subtitle_renders.py`, `test_auc_label.py`, `test_ap_label.py`, `test_brier_label.py`, `test_outlier_label.py`, `test_annotate_arrow.py` |
| SB2 | none (docs + scaffolding) |
| SB3 | `test_roc_default_annotates_auc.py`, `test_roc_annotate_auc_false.py`, `test_roc_chart_active_title.py`, `test_pr_default_annotates_ap.py`, `test_pr_baseline_hline.py`, `test_calibration_default_annotates_brier.py`, `test_confusion_matrix_cell_counts_render.py`, `test_residuals_default_annotates_metrics.py`, `test_importance_default_shows_values.py`, `test_direct_label_helper.py`, `test_learning_curve_default_direct_labels.py`, `test_validation_curve_default_direct_labels.py` |
| SB4 | `test_schwabish_judge_dispatch.py` |
| SB5 | `test_schwabish_eligibility_list.py`, `test_schwabish_from_audit_idempotent.py` |

**Total: 23 new test files.** Plus 3 inline Rust unit tests in `crates/ferrum-core/src/spec/title.rs`.
