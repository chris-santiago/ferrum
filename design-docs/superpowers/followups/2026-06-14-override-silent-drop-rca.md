# RCA — `Chart.override()` is a silent no-op

- **Date:** 2026-06-14
- **Status:** Open — root cause identified, remediation not yet started
- **Severity:** S2 (documented public API silently does nothing; misleads users)
- **Discovered:** while fixing rotated x-axis label anchoring (`fix/rotated-axis-label-anchor`). `.override(x_axis_label_angle=-45)` failed to rotate labels; investigation showed the whole method is dead.
- **Subsystem:** `src/ferrum/chart.py` (`Chart.override`), `src/ferrum/_render.py` (render pipeline), docs `concepts/override.md`

---

## 1. Symptom

```python
fm.Chart(df).mark_bar().encode(x="cat:N", y="v:Q").override(x_axis_label_angle=-45).to_svg()
# → x labels are NOT rotated (flat, text-anchor="middle")

fm.Chart(df).mark_bar().encode(x="g:N", y="v:Q").override(width=900).to_svg()
# → SVG width stays 640 (default), NOT 900
```

`.override(...)` has **no effect on rendered output for any key**. The typed equivalent works:
`.configure_axis(label_angle=-45)` rotates; `.properties(width=900)` resizes.

This is not "ignored when labels fit" (my first, narrower framing). It is ignored unconditionally, for every key.

## 2. Root cause

`Chart.override()` stores its kwargs in a per-chart dict `self._overrides` and **nothing ever reads that dict in the render path.** It is a write-only field.

```python
# src/ferrum/chart.py:2495-2518
def override(self, **kwargs: Any) -> "Chart":
    """Store low-level spec-path overrides to be applied at render time. ..."""
    new = self._clone()
    new._overrides = {**new._overrides, **kwargs}   # only ever WRITTEN
    return new
```

Every reference to the instance dict `self._overrides` in `src/ferrum/`:

| Location | Role |
|---|---|
| `chart.py:783` | `__slots__` declaration |
| `chart.py:831` | `self._overrides = {}` (init) |
| `chart.py:2166-2167` | merge on chart combination (`__or__`/`+`) — still only propagates |
| `chart.py:2517` | the `override()` write |

`grep -n _overrides src/ferrum/_render.py` → **no matches.** `_resolve_chart_config()` / `_render_inputs()` (the methods that build the `chart_config` dict handed to Rust) never consult `self._overrides`. The value is dropped on the floor at render time.

> Note: the module `src/ferrum/_overrides.py` (`_apply_overrides`, `register_layer_names`) is a **different, working** mechanism — it forwards `mark=`/`encode=`/`properties=`/`layers=` kwargs inside the figure functions (`plots/*`). It has nothing to do with `Chart.override()`'s `self._overrides` dict. Do not conflate them.

### Contrast: why `.configure_axis()` works

`.configure_axis(label_angle=-45)` → `AxisConfig(label_angle=-45)` → appended to `self._configure` (`_configure_mixin.py`) → `_render.py:_resolve_chart_config()` iterates `self._configure`, merges each `.to_dict()` into `{"axis": {"label_angle": -45}}` → passed to Rust `chart_config` → `binding.rs` → `apply_axis_config_to_axis_input` sets `axis.label_angle_override` → consumed in `layout/axis.rs:layout_x_axis` (`if let Some(override_angle) = input.label_angle_override`).

The override path has **no analogue of step "`_render.py` iterates and serializes."** That step was never written.

## 3. Blast radius

- **Every** `.override()` key is dead: `x_axis_*`, `y_axis_*`, `legend_*`, `mark_*`, `width`, `height` — all of it.
- The documented six-level cascade claims override is **level 1, "wins everything"** (`override.md:109-124`). That level does not exist; override loses to (is absent from) every other level.
- Anyone following the docs to rotate labels / set tick counts / move legends via override gets a silently wrong chart.

## 4. Documentation-vs-implementation gap (the severe part)

`docs/site/guide/concepts/override.md` and `docs/site/guide/customizing-charts.md` document a full feature set, **none of which is implemented:**

| Documented (override.md) | Reality |
|---|---|
| Spec-path injection into the chart spec at render time (§intro) | dict is never read |
| `FerrumOverrideError` on unknown paths, with "did you mean" suggestions (§Validation) | `FerrumOverrideError` **does not exist** anywhere in `src/` or `crates/` |
| `DeprecationWarning` for paths with typed equivalents (§Deprecation warnings) | not implemented |
| Multi-call merge, later wins (§Multiple override calls) | merge happens in the dict, but is moot — never applied |
| Override beats per-channel/configure/theme/defaults (§cascade) | applies to nothing |

Tell: the worked example admits the gap in a parenthetical —
`override.md:144`: `.override(x_axis_label_angle=-30)  # uses typed method in practice; shown for illustration`.
The screenshot (`override_example.png`) was produced with `configure`, not override.

## 5. Why it was never caught

`tests/test_override.py` tests **storage only**, never rendered output:

```python
def test_stores_kwarg(self, base_chart):
    c = base_chart.override(x_axis_label_angle=-45)
    assert c._overrides == {"x_axis_label_angle": -45}   # dict populated — but never applied
```

All six `Chart.override()` tests assert dict mechanics (`_overrides == {...}`, merge, immutability, returns-new). None calls `.to_svg()` and checks the result. The suite green-lights a no-op. This is the "test the implementation, not the symptom" anti-pattern: the tests lock in the dead behavior instead of the user-visible contract.

## 6. Where a fix goes

To make `.override()` honor its documented contract:

1. **Consumer (the missing step).** In `src/ferrum/_render.py`, where `_resolve_chart_config()` builds the `chart_config` dict, also fold in `self._overrides`: parse each snake_case spec-path (`x_axis_label_angle` → `axis_x.label_angle`, `y_axis_tick_count` → `axis_y.tick_count`, `legend_orient` → `legend.orient`, `mark_opacity` → mark style, top-level `width`/`height` → properties), and merge **last** so override wins the cascade (level 1).
2. **Path registry + validation.** A canonical map of valid override paths → spec locations. Unknown path → raise `FerrumOverrideError` (new exception) at render with a closest-match suggestion, as documented.
3. **Deprecation routing.** Paths with a typed equivalent emit `DeprecationWarning` pointing at the `.configure_*()` method, but still apply.
4. **End-to-end tests.** Replace/extend `tests/test_override.py` with render-level assertions: `.override(x_axis_label_angle=-45)` produces `rotate(` + `text-anchor="end"` in the SVG; `.override(width=900)` yields a 900-wide viewBox; unknown path raises `FerrumOverrideError`; override beats a conflicting `.configure_axis(...)`.

## 7. Remediation options

- **(A) Implement the documented feature fully** — preferred, and required by the project's "do the work now, do it the right way; NotImplementedErrors are not acceptable" rule. The spec/docs already define the contract; this closes a real capability gap (override is the escape hatch for things the typed surface can't yet express).
- **(B) Remove the API + docs** — only if every documented path now has a typed equivalent and the escape-hatch value is judged nil. Loses the stated purpose ("for the rare case where the typed surface hasn't caught up"). Not recommended without product sign-off.
- A do-nothing / `NotImplementedError` stub is explicitly **not** acceptable per project rules.

## 8. Links

- Override method: `src/ferrum/chart.py:2495`
- Render pipeline (missing consumer): `src/ferrum/_render.py` `_resolve_chart_config` / `_render_inputs`
- Storage-only tests: `tests/test_override.py`
- Docs: `docs/site/guide/concepts/override.md`, `docs/site/guide/customizing-charts.md:417`
- Working sibling (do not confuse): `src/ferrum/_overrides.py`
- Tracker entry: `design-docs/superpowers/followups/2026-05-15-code-archaeology.md`
