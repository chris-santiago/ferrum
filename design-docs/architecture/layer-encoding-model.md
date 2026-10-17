# Layer encoding and statistics model

Status: proposal for user review, written 2026-10-16 after Task 16 halted at decision c7 (ledger, task 16: "stop per-finding decisions ... write one holistic layer encoding/statistics model"). It replaces per-finding reasoning with one model: a handful of principles that generate every cell of the channel × statistic × door space, a measured rule table, a gap list and the decisions only the user can make. Earlier rulings (Q19, M1, Ruling A, the promotion rulings, decisions c1–c7) are treated as inputs, not constraints. Every place the model contradicts one is named, and the contradictions that need a ruling are in §5.

**Trees.** "CUR" is the working tree on `fix/batch-i-regressions` with Task 16's uncommitted work through c7 (extension rebuilt 2026-10-16 06:55, after the last Rust edit). "SM1" is the stage baseline before Task 16 (`/tmp/bir-base-SM1`).

**Probes.** All under `/Users/chrissantiago/.claude/jobs/a41e4f71/tmp/model/`. `run.sh <module>` runs a case module on both trees and `cmp.py` prints them side by side. Each case renders static SVG (mark census inside the plot clip groups, chrome text, warnings) and `.interactive().to_html()`. Where a layered chart has an oracle, the oracle is the sum of each part rendered standalone: a layered chart should draw what its layers draw alone. Modules: `c_pos` (positional, 58 cases), `c_val` (11 value channels × 3 statistics × 11 doors, 275 cases), `c_sib` (sibling bindings and grain), `c_lc` (LayerChart routes), `c_comp` (composite marks and user transforms, 58), `c_grp` (grouping channels and composite grouping, 80), `c_title`, `c_misc` (polar, error, facet, SecondaryY), `c_cnt`, `c_bin2`, `c_dom`, `c_dbg4`. Raw results are in `out/*.json` and `out/*.txt`. About 600 cases in all.

**Door abbreviations.** F flat chart · L `Chart.layer(fm.Layer(...))` (and a figure's `layers=`, which goes through it) · LD `Layer(data=...)` · +s / +d `a + b` over the same / a different frame · +r the statistic sits on the right operand · SY `+ fm.SecondaryY(...)` · LCs / LCm / LCi / LCx `fm.LayerChart` static, `_build_merged()`, `resolve={"y": "independent"}`, `.interactive()` · Fc faceted.

---

## 1. Principles

Eight rules. Every cell in §2 is derived from them. A cell no rule produces is flagged as a design gap.

### P1. Rows: a layer reads exactly one row set, and never another layer's statistics

A layer with its own data (a `+` operand, a `LayerChart` member, `Layer(data=)`) reads its own frame through its own operand's `.transform()` chain. An attached layer (`Chart.layer`, `SecondaryY`, a figure's `layers=`) reads its host's rows after the host's `.transform()` chain. A composition's own rows are its left operand's, so a step added after the merge runs over them (Task 6). Facets partition the source before anything runs. A composite mark's statistics (bins, density grid, fit, box stats) and every encoding aggregate or bin belong to the layer that computes them. They are never a row set another layer reads; to share them, compute them in `.transform()`.

- *Coherence:* this is M1 (adopted 2026-10-15) plus c3's per-layer statistics. One row rule serves every door. `Branch`-scoped publication (T4) already gives each operand its own rows.
- *Users:* what a layer draws never depends on how a sibling composite names its internal steps. Adding a layer never changes another layer's rows.
- *Libraries:* Vega-Lite children inherit the parent's data and transforms, and a child with its own `data` runs its own pipeline. ggplot2 gives each layer the plot data or its own `data=`, and stats are private (`after_stat`).

### P2. Bindings: a layer inherits from exactly one parent, never from a sibling

A layer's effective encoding is its own bindings merged over its parent's, and its own bindings win channel by channel. An attached layer's parent is its host's written encoding. A composition's parent is what the caller wrote on the composition itself (`(a + b).encode(...)`, `LayerChart` has none). An inherited channel arrives whole: field, type, aggregate, bin, scale, title. A `+` operand's own bindings are scoped to that operand's layers, the scope `ferrum-spec.md` already gives conditionals ("an operand of `chart + chart` ... keep[s] their conditionals on their own layers"). A parent field names the same-named column in each operand's own frame (Ruling A's by-name reading, extended from x/y to every channel). The one cross-sibling flow is positional and comes from P5.

- *Coherence:* one inheritance rule replaces three that disagree today: left encoding as the composition's, right-to-chart promotion with statistics stripped (c7), and statistic copying (c3). It is the conditional scope rule applied to bindings. Rust already has the primitive (`Encoding::inherit_from` / `inherit_non_positional`).
- *Users:* adding a layer never recolors, resizes, splits or regroups another layer. `point + sized_line` and `sized_line + point` draw the same chart (the order independence promotion was built for), because neither leaks. To share a binding, write it where sharing is visible: on the host or on the composition.
- *Libraries:* Vega-Lite layer children inherit only from the parent and never from siblings. ggplot2 shares only plot-level `aes()`, and a geom's own `aes()` is local. Neither has promotion.
- *Drawn channels (user ruling 2026-10-17):* a layer takes an inherited positional binding (its parent's, or layer 0's x/y/x2/y2 fallback) only on the channels its mark draws, whatever its data holds. A reference rule, function line or ROC diagonal draws as it does alone. A drawn channel whose column the layer's own rows lack is refused with the typed missing-column message. The same rule already decides geometry (`normalize_rule_tick_own_span`), so the statistic grain and the gate now read the same encoding the mark builder draws. ggplot2 gives `geom_hline`/`geom_vline`/`geom_abline` `inherit.aes = FALSE` and errors on a drawn aesthetic whose column is missing; Vega-Lite drops `x2` on a line. Accepted cost: `P(x=g) + rule(y=count())` draws one rule at the total, not one per `g`.
- *Contradicts:* `ferrum-spec.md` "Chart-level encodings are shared" as implemented (the left operand's own encoding is the composition's), "Layer promotion" (rulings 2026-10-01 and 2026-10-12), c7 ruling 3 (a promoted field reaches the other operand) and the c3 case "a `+` right layer inheriting the left's color ... groups by c too". These go to the user as D1.

### P3. Statistics are per layer, at the layer's own grain

A layer's grain is decided by its own effective encoding (P2) and nothing else. If any channel carries an aggregate, the layer aggregates, grouped by every non-aggregated field it draws plus the facet fields. A positional bin with a count or aggregate on the opposing channel bins and aggregates per bin. A positional bin without an aggregate bins each row (no aggregation). A grouped bin shares one set of edges, pooled (c5). A composite mark is a layer with a built-in statistic, and it groups by the same set of grouping fields (color, detail, shape, stroke_dash, facet). Since P2 lets nothing into the effective encoding from a sibling, a sibling can never change a layer's grain.

- *Coherence:* one groupby helper (`_chart_agg_groupby` over the effective encoding) serves flat, per-layer and composite statistics. A flat chart is one layer (c4).
- *Users:* labels on bars show the bars' values, a colored layer groups by its color, and the count of marks a layer draws equals what it draws alone.
- *Libraries:* Vega-Lite aggregates each unit view by all its non-aggregated fields, and a `bin` without an aggregate is a per-row transform. ggplot2 groups every stat by the interaction of its discrete aesthetics (`geom_smooth(aes(color = g))` fits per group). seaborn's `histplot(hue=, multiple="stack")` uses common bins.

### P4. One channel vocabulary on every door

A channel means the same thing in `encode()`, `Layer(encoding=)`, `SecondaryY`, figure kwargs, conditionals, promotion copies and renames. Shorthand is parsed everywhere, and a string means what the equivalent channel object means. Each channel's honored kwargs are one table. `aggregate=` is honored on every channel that draws a data value: x, y, x2, y2, theta, radius, theta2, radius2, and the value channels. It is refused, by name and with a remedy, on channels that draw no value (detail, key, href, description, url, facet, shape, stroke_dash). `bin=` is honored where a binned meaning exists (positional channels, and color per D3) and refused elsewhere. An internal rename or copy never adds a kwarg the channel refuses.

- *Coherence:* `_honored_kwargs` and `_refusal_message` are already the one gate (c7). This principle finishes it for `bin=`, for the span and polar channels, and for `Layer(encoding=)`.
- *Users:* the same words do the same thing on every door. Nothing is warned "not yet honored" while it changes the output.
- *Libraries:* Altair parses the same shorthand in every encoding position. Vega-Lite allows `aggregate` on `x2`/`y2` (floating bars) and `theta` (pies), and `bin` on color (binned legends).

### P5. One scale per channel per panel, resolved over every layer that draws it

On one panel each channel has one scale, except an independent-y layer, which owns its y scale (c2). Layer 0 establishes the scale's kind (wave-6 "layer 0 decides"). The domain is the union of what each admitted layer draws, read from the batch that layer draws (c6). A layer whose column conflicts with the established kind is excluded with `LayerScaleExcluded` (c7). The scale exists because some layer draws the channel, not because the chart-level encoding names it. Legends are drawn from the scale, one per channel per panel. The positional axis binding is layer 0's written x/y, and a layer that binds no x/y takes it (Ruling A, 2026-10-13/15). This is the one flow between siblings: the axes are the panel's, and layer 0 establishes them.

- *Coherence:* one rule for positional and auxiliary scales (`seed_reads` / `aux_reads` already read per layer). It removes the reason promotion exists (Rust built auxiliary scales from `spec.encoding` only, per the `_promote_layer_channels` docstring).
- *Users:* one legend per field, listing every value drawn. A channel bound only on a later layer still gets its scale and legend.
- *Every mark (user ruling 2026-10-16):* a channel's scale maps its values the same way on every mark that draws it, so the legend states what each mark draws. The opacity family (`opacity`, `fill_opacity`, `stroke_opacity`) maps through its scale on line, area, bar, tick, rule and segment marks as it does on point, rect, arc and polygon. Before the ruling those six drew the raw value clamped to 0–1, which the opacity legend would misstate. Vega-Lite and ggplot2 map opacity/alpha through one scale on every mark.
- *Libraries:* Vega-Lite layers share scales and merge legends by default, with `resolve` to opt out. ggplot2 trains each scale on every layer. The divergence is the x/y fallback (Ruling A): Vega-Lite gives a sibling nothing. It is justified because one panel has one coordinate space, and the wave-6 rule already makes layer 0 its owner.

### P6. Titles come from the binding the caller wrote

An axis or legend title is the written `title=`, else a name derived from the written binding: the field (`v`), the statistic output for an aggregate (`mean_y`, `mean_z`, per the 2026-10-16 legend-title ruling), or the source field for a bin (`v`, per the 2026-10-16 c8 ruling). The title is read from the establishing layer (layer 0 for axes, the first layer that draws the channel for a legend). It is the same on every door, and never an internal column (`bin_start`, `agg_value`, `z__rhs_0001`).

- *Coherence:* one title derivation instead of three (flat channel rebind, promotion copy, rename stamp).
- *Users:* the same chart is titled the same however it was assembled.
- *Libraries:* Vega-Lite titles from the field definition ("Mean of y", "v (binned)"). ferrum's `mean_y` convention diverges in wording only, and is user-ruled.

### P7. Refuse malformed, warn degraded, at the earliest door that knows

A field absent from the rows a layer reads, a reference to another layer's statistic, or a kwarg a channel cannot draw is refused with a typed error naming that door's remedy. The refusal comes at construction when the schema is knowable in Python (`_known_columns`), else at render by Rust's gate, which is the single owner of the message text. Output that draws but loses something (an excluded layer, an ignored detail, a facet on a right operand) is warned. Nothing silently draws a null group, an empty layer or a multiplied mark count.

- *Coherence:* CLAUDE.md "Refuse malformed, warn degraded", with the c2–c4 knowable-schema and published-formatter machinery.
- *Users:* every failure names its cause.
- *Libraries:* Vega-Lite warns on unknown fields and draws nothing. ferrum is stricter by policy.

### P8. Every route draws the same chart

A door is a spelling, not a semantics. Static and interactive output, `+` and `LayerChart` (static, merged, independent-y, interactive), a flat chart and the same chart built with `Layer`, and either operand order (P2 and P5 make order irrelevant except for layer 0's axis ownership) all produce the same marks, scales, legends and titles.

- *Coherence:* the composite overlay tree and the merged path must agree, because interactive output must use the merged path (CLAUDE.md, Composite rendering).
- *Users:* `.interactive()` shows the static chart.
- *Libraries:* Vega-Lite has one semantics per spec, whatever the renderer.

---

## 2. Rule table

Each row is one cell group. Doors are collapsed where P1–P8 make them identical, and one representative of each collapsed group was probed (the probe module is named). "Principled" is what P1–P8 produce. Status is one of: **consistent**, **gap** (no rule satisfied, or a capability missing), **contradiction** (two routes or two rules disagree), or **regression** (SM1 drew it right, or refused it safely, and CUR draws it wrong or refuses a valid chart). Gap ids point to §4.

### 2.1 Positional x / y

| Statistic | Doors | Principled | CUR | SM1 | Vega-Lite / Altair | Status |
|---|---|---|---|---|---|---|
| none | F, L, LD, +s, +d (both orders, right binds none), SY, LCs/m/i/x, Fc | each layer at its own x/y, else its parent's, else layer 0's written x/y | every case matches its oracle (`c_pos none/*`) | same, except LD: the host layer drew the host's 8 rows plus df2's 6 (14 circles) | same, but a sibling never supplies x/y | consistent |
| enc aggregate | F; L inheriting host `y=mean(y)`; L with its own `sum(y)`, raw `y` or finer `color` | each layer aggregates its own rows by its own effective groupby | renders, oracle match (2 bars + 2 points; finer color gives 4) | refuses `stat_aggregate: column 'y' not found` | child inherits the aggregate and aggregates its own data | consistent |
| enc aggregate | L / SY naming the output (`text="mean_y"`, `SY("mean_y")`) | refuse with the statistic remedy | refuses (Rust gate statistic remedy; SY Python statistic message) | transform error / source-schema message | unknown field: nothing drawn | consistent |
| enc aggregate | +s / +d, right binds none (Ruling A); +r either frame; post-merge `(a+b).encode(y="mean(y)")` | right computes its own mean; post-merge binding reaches only layers binding none | right draws 2 (same frame) / 3 (df2) means; post-merge: left keeps raw 8, right 3 | +s drew 8 raw points; post-merge dropped the left (3 marks) | sibling gives nothing; parent `.encode` reaches children without the channel | consistent (fixed vs SM1); x/y fallback is the P5 divergence |
| enc aggregate | LCs/m/i/x, layered LC member, Fc flat and layered | per layer, per panel | all render, oracle match | layered member and faceted layered refused | same | consistent |
| enc aggregate | `Layer(encoding={"y": "sum(y)"})` (shorthand) | parsed as `encode()` parses it | refused: `unknown column 'sum(y)'`; also `"r:Q"` → `unknown column 'r:Q'` | same failure | Altair parses shorthand everywhere | gap G7 |
| enc bin, no aggregate | F `x=X("v", bin=True), y="y"` | per-row bin: 8 points at their bins | **28 circles**: the Bin is grouped by `y` and emits a count row per (bin, y) | refused `unknown column 'y'` | per-row bin, 8 points | **regression R4** |
| bin + count | F, grouped by color, L inheriting, L label (`Text(aggregate="count")`), +s/+d/+r, LCs/m, Fc | per-layer Bin; grouped → pooled shared edges | all match oracles; grouped draws 8 rects; labels 1/3/2/2 | grouped refused; every layered variant refused | bin once, count per (bin, color) | consistent |
| bin + value aggregate | `tooltip=Tooltip("y", aggregate="mean")` / `text=mean` beside a bin | fold per bin | renders, 4 rects | refused | per-bin aggregate | consistent |
| bin + two distinct aggregates | `y=mean(y), color=mean(z)` beside a bin | both per bin | refused: "a Bin computes one aggregate column per bin" | transform error | computes both | gap G12 |
| composite stat | H, D, B, V, S, S(ci), errorbar across F, L (rug, points), LD, +s/+d/+rev, SY, LCs/m/i, Fc, filter→H | stat per layer; attached layers read host rows; SY reads host rows | all match oracles (`c_comp`) | H rug drew 4 ticks at bin starts; density rug 512 ticks; smooth points 200; `H.layer(L).transform(f)` drew nothing; SY over H/D/S refused | composite marks are macros; siblings read parent data | consistent (fixed vs SM1) |
| composite stat | attached layer or SY naming the stat output (`H.layer(y="count")`, `H + SY("count")`) | refuse with the statistic remedy | refuses | rendered the tail / source message | field not found | consistent (M1 reversal of Q19) |
| user transform | filter / calculate / aggregate / Swarm across F, L, LD, +d, post-merge, SY, SY on a composition | attached layers read the host chain's output; own-data layers do not; post-merge runs on the left | all match oracles | LD ran the host chain over df2 (7 circles) | parent transforms reach inheriting children | consistent |
| inherited binding, undrawn channel | a reference rule, `mark_function`, `mark_roc` diagonal or count rule under a host or layer 0 that writes x/y/x2/y2 its mark does not draw, over a frame without those columns | the layer takes only the channels its mark draws (user ruling 2026-10-17); a drawn channel with no column refuses with the typed message | Task 20 cycle 1: refuses or fails on the inherited column (e.g. `stat_aggregate: groupby column 'w' not found`) | rendered through null-padded columns | x2 dropped on a line; siblings give nothing | **gap**, fixed in plan Task 20 |
| user transform | SY on a different-frame composition naming a column only the right frame has: `(P(df) + P(df2, y="w")) + SY("w")` | refuse: `w` is not in the chart's data (the left's rows) | **draws nothing, no warning** (14 circles, no line) | same | field missing: nothing drawn, warning | gap G6 |

### 2.2 x2 / y2 and other span ends

| Statistic | Doors | Principled | CUR | SM1 | Vega-Lite | Status |
|---|---|---|---|---|---|---|
| none | F; L inheriting `y2`; +d right binds none | span read per layer; x2/y2 follow x/y (Ruling A covers them) | renders (8 rects; 8 ticks; 8 rules) | same | same | consistent |
| enc aggregate | F ranged bar `y="g", x="min(v)", x2="max(v)"`; rule `y="min(y)", y2="max(y)"`; L variants | renders: one Aggregate with both ops (user ruling 2026-10-16, c8) | refused at construction: `X2(aggregate='max') is not supported: X2 draws no aggregated value` (c7) | `stat_aggregate: column 'v' not found` | floating bar from min/max | **gap (open item 1, §3)**, G11 |
| enc bin | every door | binned channel gets `x2=bin_end` | renders | renders | `x2` implied by bin | consistent |
| user transform | `transform_aggregate(min, max)` then `x="lo", x2="hi"` | renders | renders (2 rects) | renders | same | consistent |

### 2.3 Value channels: color / fill / stroke, size, opacity / fill_opacity / stroke_opacity, stroke_width, angle, text, tooltip

| Statistic | Doors | Principled | CUR | SM1 | Vega-Lite | Status |
|---|---|---|---|---|---|---|
| none | F, L (own and inherited), +s/+d (left binds, right binds), LCs/m/i, SY, Fc | P2: a sibling's binding never reaches another layer; one shared scale and legend (P5) | every oracle count matches, but bindings cross siblings: `P(color="c") + line` splits the line into 2 by `c` (left → right); the right's color, size, shape, opacity family and stroke_dash are promoted to the chart (right → left); **LCs keeps the line whole (1 polyline) while LCm / LCi / LCx split it (2)** | same | children never inherit from siblings; scales and legends shared | **contradiction K1** (routes differ), D1 |
| enc aggregate | F; L inheriting the host's aggregated channel | per layer | renders (one Aggregate per layer, c4) | refused (`column 'z' not found`) | same | consistent |
| enc aggregate | L binds its own aggregated channel on an aggregating host: `bar(x=g, y=mean(y)).layer(Layer(point, color=mean(z)))` | host 2 bars, layer 2 points | **host draws 7 bars**: the layer's color is promoted as the field `z` alone (c7), the host inherits it, and its own `mean(y)` groups by `(g, z)` | refused | host untouched | **regression R1** |
| enc aggregate | +s, left binds, right aggregates: `point(g, mean(y), color=mean(z)) + tick(g, mean(y))` (every value channel, including stroke_width and text) | right 2 ticks | **right 7 ticks**: the composition keeps the left's color as the field `z` alone, and the right groups by it | right 2 ticks | right uncolored, 2 ticks | **regression R1** |
| enc aggregate | +s, right binds (promoted: color, size, opacity family), left aggregates | left 2 | **left 7** | left 2 | left untouched | **regression R1** |
| enc aggregate | +d, right binds, left aggregates: `tick(df, g, mean(y)) + point(df2, g, mean(y), color=mean(z))` | left 2 ticks, right 3 points | color/size: **the left draws nothing** (groups by `z__rhs_0001`, null on its rows); opacity family: **BUILD-ERR** `Opacity(title='z') is not honored` | left 2, right 3 | left untouched | **regression R2, R6** |
| enc aggregate | +s, right binds, left raw: `P(x=g) + point(g, mean(y), color=mean(z))` (the c7 case) | P2: left uncolored, 8 raw points; one color scale and legend from the right | left keeps 8 raw points, drawn by raw `z` through the union scale; legend title `mean_z` (c7 ruling 3 plus the title ruling) | left 8 colored by `z`; legend `z`, domain 10..45 | left uncolored | consistent with c7, contradicts P2 → D1 |
| enc aggregate | LCs vs LCm / LCi / LCx (right binds, different frame) | identical | LCs renders 2 + 3; LCm / LCi / LCx lose the left (color, size) or BUILD-ERR (opacity family, including `.interactive()`) | LCs refused; LCm rendered | one semantics | **regression R2/R6**, contradiction K1 |
| enc aggregate | SY over a base with an aggregated value channel; Fc `(tick + point(color=mean(z))).facet("c")` | SY: own y; Fc: per panel, grain unchanged | SY renders; Fc: **ticks 8 instead of 4** | SY refused for size/opacity (`unknown column 'mean_z'`); Fc 4/4 | same | SY consistent; Fc **regression R1** |
| enc aggregate | fill / stroke aliases: `bar_mean().layer(Layer(point, fill=Fill("z", aggregate="mean")))` | the layer's `fill` is its color binding: 2 points | the alias does not shadow the inherited chart color `z`, so the layer groups by `(g, z)`: 7 points, host 7 bars; over different frames whole layers vanish (`plus-both-diff`: 0 circles) | refused | `fill` is its own channel | **regression R5** |
| enc bin | `Color/Size/Opacity("z", bin=True)` on any door | D3 | **32 marks from 8 rows** (a grouped count Bin) while warning "color(bin=...) is accepted but not yet honored"; promoted bin legend titled `bin_start` | refused (`unknown column 'v'` / `Opacity(title=)`) | binned legend, rows unchanged | **regression R3**, D3 |
| composite stat | `H/D/B(color=g)`; `S(color=g)`; `S(color=g).layer(tick)` | the stat groups by color (P3) | H/D/B group; `S(color=g)` refused (`unknown column 'c'`); `S(color=g).layer(tick)` **renders an ungrouped fit, color dropped silently** | H/D/B group; S refused flat and layered | per-group fit | gap G8 |
| user transform | aggregate in `.transform()`, bind the output (`SY("mean_r")`, `layer(text="my")`) | renders | renders | partly refused | same | consistent |
| scale domain | color/size/opacity union over layers (c6); type conflict excluded with `LayerScaleExcluded` (c7) | union; one legend | union on `+` (numeric 5..45, categorical a, b, c); **LCs draws two `g` legends (a, b and a, b, c)** while `+` draws one | `+` used the first color layer's domain (a, b) | shared scale, merged legend | `+` consistent; LCs **contradiction K1** |
| mark kind | `opacity` / `fill_opacity` / `stroke_opacity` bound to a field on line, area, bar, tick, rule, segment vs point, rect, arc, polygon | one scale maps the value on every mark; the legend shows the alphas the marks draw (user ruling 2026-10-16) | point, rect, arc, polygon map through the scale; the other six draw the raw value clamped to 0–1 | same | one opacity scale on every mark | **gap**, fixed with the opacity legend (plan Task 31) |
| legend title | promoted statistic, same vs different frame; promoted bin | written-binding title: `mean_z`; bin → `z` | same frame `mean_z`, different frame `z`; bin `bin_start` | `z` | "Mean of z", "z (binned)" | **contradiction K3 (open item 2, §3)** |
| `Layer(encoding=)` strings | `P.layer(Layer(point, encoding={"y": "r", "color": "c"}))` | draws colored by `c` with a legend | **not drawn**: a string is not promoted and Rust builds no scale for a later layer's own channel; a `Color("c")` object is promoted and drawn | same | Altair: string ≡ object | gap G7 |

### 2.4 Grouping channels: detail, shape, stroke_dash, key, href (and raw text / tooltip fields)

| Statistic | Doors | Principled | CUR | SM1 | Vega-Lite | Status |
|---|---|---|---|---|---|---|
| none | `line(detail=c) + line` vs `line + line(detail=c)`; shape / stroke_dash on the right | P2: never across siblings; order-independent | detail: 4 polylines vs 3 (left → right, never promoted, so order-dependent); shape and stroke_dash promoted (symmetric); LCs isolates them | same | sibling-local | **contradiction K2**, K1 |
| enc aggregate on the channel | `Detail/Shape/StrokeDash/Key/Href("z", aggregate="mean")` | refused at construction (P4) | refused by name with the `.transform(aggregate)` remedy (c7) | warned, then the sentinel ran | allowed, but with no drawing meaning | consistent (divergence justified: there is nothing to draw) |
| raw channel inside an aggregating layer, own or parent-inherited | F `point(g, mean(y), shape=c)`; host binds, L inherits | group key (P3) | flat 4 groups; L inherits and groups (4 or 8 ticks) | refused | groups by every field | consistent |
| raw channel bound by a sibling of an aggregating layer | `line(g, y, detail/shape/stroke_dash/key/href/text/tooltip=…) + tick(g, mean(y))` and the reverse | the aggregating layer's grain is unchanged (2) | **grain changes**: 4 ticks (detail, shape, stroke_dash, tooltip, text), 8 ticks (key, href); promoted shape / stroke_dash also regroup the left | 2 | 2 | **regression R1** |
| enc bin on the channel | `Detail/Shape/StrokeDash/Key/Href/Tooltip/Text("z", bin=True)` | refused (P4) | **32 marks** plus a "not yet honored" warning | refused (title error or unknown column) | `bin` on detail allowed as a per-row group | **regression R3** |
| the layer's own channel on an attached layer | `P.layer(Layer(point, encoding={"shape": "c"}))`, `stroke_dash` likewise | drawn, with a legend | not drawn (all circles, line unsplit) | same | drawn | gap G7 |
| composite stat × grouping | H / D with shape or stroke_dash; D / S with detail; B with detail / shape / stroke_dash | the stat groups by them (P3) | H / D refuse `unknown column 'c'`; D and S ignore detail (1 path, 1 fit); B ignores all three silently | same | ggplot2 groups by the interaction of discrete aesthetics | gap G8 |

### 2.5 Facet

| Statistic | Doors | Principled | CUR | SM1 | Vega-Lite | Status |
|---|---|---|---|---|---|---|
| all | F, L, LD on a faceted host (both orders), +d faceted (both orders), SY, layered statistics | partition the source first; every statistic groups by the facet fields | renders per panel (c3 coalescing) | faceted LD and faceted +d dropped the right rows | facet wraps the whole layered spec | consistent |
| facet on a right operand only | `P + P(df2).facet("g")` | ignored with a warning (degraded) | warns "secondary layer's ... facet is ignored" | same | n/a | consistent |
| faceted LD with an inherited aggregate | `bar_mean().facet("c").layer(Layer(data=df2, mark="point"))` | renders 2 bars and 3 points in every panel (df2 has no `c`, so it repeats); no false warning | renders 4 + 3, but warns "every row has a null in its groupby key (g__rhs_0001, c)": the overlap no-chain route reads the combined rows | refused | n/a | gap G10 |
| layer frame lacks the facet column | LD, +d (both orders), LCs/m/i over a frame without the facet field | the layer repeats in every panel, its statistic over its own rows (user ruling 2026-10-17) | rows in no panel, drawn nowhere, no warning (task-16-decision-c3 pinned this) | same | ggplot2 repeats the layer in every panel; Vega-Lite puts the rows in a `null` cell | **gap**, fixed in plan Task 20 |
| null facet values | a frame that has the facet column with some null values | dropped with a warning naming the field and the row count (user ruling 2026-10-17) | dropped silently | same | ggplot2 draws an `NA` panel, Vega-Lite a `null` cell; seaborn and plotly drop | **gap**, fixed in plan Task 20 |
| aggregate on a facet channel | `Facet(..., aggregate=)` | refused | refused (c7) | warned | n/a | consistent |

### 2.6 Polar and error channels

| Statistic | Doors | Principled | CUR | SM1 | Vega-Lite | Status |
|---|---|---|---|---|---|---|
| none | F arc, `theta=z` | renders | 8 arcs | same | same | consistent |
| enc aggregate | `Theta("z", aggregate="sum")`, `theta="sum(z)"`, `radius="mean(z)"` | renders (P4: theta and radius draw a value) | refused at construction (c7) | **rendered the pie** (2 arcs) despite a "not yet honored" warning | the canonical pie spelling | **regression R7**, D2 |
| user transform | aggregate then `theta="sz"` | renders | renders | renders | same | consistent |
| layered polar | `arc + text` (both CoordPolar), `arc.layer(Layer(text))` | renders | both fail: `encoding 'y' expected EncodingSpec, got None` | same | layered arc and text is the canonical labelled pie | gap G9 |
| error channels | `x_error` / `y_error` on encode | warned and omitted (spec warn bucket) | same | same | `xError` feeds errorbar | consistent (documented divergence) |
| errorbar composite | F, + point, L point, SY | per layer | renders | renders | same | consistent |

Interactive parity: every rendering case above also rendered `.interactive().to_html()`, except the opacity-family `+d` and `LayerChart` cases (R6). No other route-specific interactive defect appeared in about 600 cases.

---

## 3. Known open items placed in the table

- **X2/Y2 aggregate refused by c7** (§2.2 row 2). P4 says span ends draw a value, so they honor `aggregate=`, and the user already ruled this on 2026-10-16 (c8, halted before it was written). The flat ranged bar `mark_bar().encode(y="g", x="min(v)", x2="max(v)")` refuses at construction today and failed in the transform stage on SM1, so it is a gap, not a regression. The same principle covers `theta`/`radius`/`theta2`/`radius2`, which **did** render on SM1 (R7, D2). Owner: `src/ferrum/encoding/base.py` `_refusal_message` (l. 552) and the honored sets in `src/ferrum/encoding/_honored.py`. The lowering (`_resolve_pending_aggregates`, and the per-layer `_resolve_layer_aggregates` at `_layer_transforms.py:274`) must give x2/y2 a slot in the layer's one Aggregate, grouped like x/y.
- **A promoted binned channel titled `bin_start`** (§2.3 "legend title" row). Under P6 a bin titles by its source field, as the flat chart does (`Axis titles keep the user's field names`), and the user ruled the same on 2026-10-16. The source is `_promoted_statistic_field` (`src/ferrum/composition.py`, after l. 4580), which stamps the Bin's output column as the title. The same row shows its sibling defect: a promoted aggregate is titled `mean_z` over the same frame and `z` over a different frame, because `_rename_encoding_fields` (`src/ferrum/chart.py:147`) stamps the original field first. Under D1-A (no promotion) both stamps are deleted and P6 titles from the establishing layer's written binding.

---

## 4. Gap list

Ordered by severity: silent wrong output first, then refusal of a valid chart, then inconsistent titles and messages. Each entry gives the principle it violates, a minimal repro (`df` has `v, y, z, r, g, c, k`; `df2` has `v, y, z, g, w`, with `g` overlapping), and the owning code.

**Totals: 20 findings in 19 entries (entry 11 holds two). 7 regressions vs SM1 (R1–R7), 9 gaps (G6–G13 and G-spec), 4 contradictions (K1–K4).**

1. **R1. A sibling's binding changes another layer's grain** (P2, P3; regression). `P(x="g", color="c") + Chart(df).mark_bar().encode(x="g", y="mean(y)")` draws 4 bars (SM1 drew 2). `Chart(df).mark_bar().encode(x="g", y="mean(y)") + P(x="g", color="c")` draws 4 (SM1 2). `point(g, mean(y), color=mean(z)) + tick(g, mean(y))` draws 7 ticks (SM1 2). The same happens for every value and grouping channel, on L (`bar_mean().layer(Layer(point, color=mean(z)))` → host 7 bars), on LCm/LCi/LCx and faceted. Root: c3's effective groupby counts inherited fields as group keys, and c7 strips a promoted or left-operand statistic to its bare field, which then becomes a group key in every other aggregating layer. Owner, Python: `_effective_encoding` (`src/ferrum/_layer_transforms.py:445`) feeding `_chart_agg_groupby` (`src/ferrum/_encode_lowering.py:52`); `_promote_layer_channels` / `_promoted_statistic_field` (`src/ferrum/composition.py:4518`); `Chart._merge`'s `_take_chart_statistics` calls (`src/ferrum/chart.py:2956-2957`). Closed by P2 (D1-A) or by the grain clause of D1-B.
2. **R2. A renamed sibling field nulls out a whole layer** (P1, P7; regression). `Chart(df).mark_bar().encode(x="g", y="mean(y)") + P(df2, x="g", color="g")` draws no bars, warning only "every row has a null in its groupby key (g__rhs_0001, g)" (SM1 drew 2 bars). `tick(df) + point(df2, color=mean(z))` loses the ticks, and so do LCm/LCi/LCx. Root: the promoted binding carries the right frame's renamed column (`z__rhs_0001`), which is null on the left's rows, and the left aggregates by it. Owner, Python: `_rename_encoding_fields` (`chart.py:147`) applied before promotion (`composition.py:4518`).
3. **R3. `bin=` on a non-positional channel multiplies marks while warning it is ignored** (P4, P7; regression). `P(color=fm.Color("z", bin=True))` draws 32 circles from 8 rows and warns "color(bin=...) is accepted but not yet honored". The same holds for size, opacity, detail, shape, stroke_dash, key, href, tooltip and text. SM1 refused each. Root: `ChannelBase.to_implicit_transforms` emits a `_PendingBin` for every channel. The c4 grouped Bin groups by x and y and emits a count row per (bin, group), so the honored-set warning lies about the output (the c7 pattern, for `bin`). Owner, Python: `src/ferrum/encoding/base.py` (`to_implicit_transforms`, `_refusal_message`), `_bin_groupby` (`_encode_lowering.py:67`). D3 decides what color bin means.
4. **R4. A positional bin without an aggregate becomes a count grain** (P3; regression). `Chart(df).mark_point().encode(x=fm.X("v", bin=True), y="y")` draws 28 circles. The wire is `Bin(groupby=["y"], shared_extent=true)`, a count per (bin, y). SM1 refused it. Root: every encoding bin lowers to the Rust `Bin`, which always aggregates, and c4's groupby adds every unaggregated field. Owner: Python `_lower_bin_encoding` / `_bin_groupby` (`_encode_lowering.py:244`, `:67`); Rust `crates/ferrum-core/src/transform/bin.rs` has no per-row mode. D4.
5. **R5. The fill/stroke aliases do not shadow an inherited color** (P2, P4; regression). `Chart(df).mark_bar().encode(x="g", y="mean(y)").layer(fm.Layer(mark="point", encoding={"fill": fm.Fill("z", aggregate="mean")}))` draws 7 bars and 7 points. Over different frames whole layers vanish (`fill/agg/plus-both-diff`: circles 0). Root: `_effective_encoding` merges by key, so a layer's `fill` does not override the chart's (promoted) `color`. Owner, Python: `_layer_transforms.py:445`, `resolve_channel_alias` usage.
6. **G6. A SecondaryY on a different-frame composition naming a right-only column draws nothing** (P1, P7; gap, both trees). `(P(df) + P(df2, y="w")) + fm.SecondaryY("w")` renders no line and no warning. Under M1 the secondary reads the left's rows, where `w` is null padding. Owner, Python: `_refuse_secondary_field` (`chart.py:690`) and `_known_chart_data_columns` (`_layer_transforms.py:542`). The left's frame is knowable, so this should refuse. Rust's gate also passes null-padded columns.
7. **G7. `Layer(encoding=)` is a second vocabulary** (P4, P5; gap, both trees). Shorthand is not parsed (`{"y": "sum(y)"}` → `unknown column 'sum(y)'`; `"r:Q"` likewise), and a plain string on a non-positional channel is not drawn: `P.layer(Layer(point, encoding={"y": "r", "color": "c"}))` draws one color with no legend, and `shape` / `stroke_dash` likewise. A `fm.Color("c")` object is drawn, because only `ChannelBase` values are promoted. Owner: Python `Chart.layer` (`chart.py:2651`) should build channels through `encode()`'s resolver; Rust `build_auxiliary_scales` (`crates/ferrum-core/src/render/scale_resolve/mod.rs:1566`) builds a scale only when `spec.encoding` (layer 0 merged) names the channel, and shape / stroke_dash take no per-layer reads.
8. **G8. Composite statistics ignore grouping channels** (P3; gap, both trees, with one new silent drop). `mark_smooth().encode(color="c")` refuses `unknown column 'c'`, while `mark_smooth().encode(color="c").layer(fm.Layer(mark="tick"))` renders an ungrouped fit and drops the color silently (new in CUR). `mark_density(detail="c")` draws 1 path, `mark_boxplot(detail/shape/stroke_dash="c")` ignores them silently, and `mark_histogram(shape="c")` refuses. Owner: Python composite builders (`src/ferrum/marks/statistical.py`, the `_chart_methods_statistical` mixins); Rust smooth/kde groupby.
9. **K1. LayerChart routes draw different charts** (P8; contradiction, both trees, widened by R1/R2). `fm.LayerChart(P(color="c"), line)`: LCs 1 polyline, LCm / LCi / LCx 2. `LayerChart(P(x="g", color="c"), bar_mean())`: LCs 2 bars, LCm 4. LCs draws two `g` legends where `+` draws one. The spec documents the static/interactive split ("the static overlay renders each member alone"). Root: the static overlay tree isolates members (P2 semantics), while the merged path lowers through `+` (sibling sharing). Owner: Python `LayerChart._build_merged` (`src/ferrum/composition.py`) vs the overlay lowering; Rust composite legend dedup. Closed by D1-A with no route change.
10. **R6. Opacity-family promotion over different frames raises at build** (P4; regression). `tick(df) + point(df2, g, mean(y), opacity=fm.Opacity("z", aggregate="mean"))` raises `Opacity(title='z') is not honored`, and so do `fill_opacity`, `stroke_opacity`, LCm and LCs `.interactive()`. SM1 rendered it. Root: `_rename_encoding_fields` stamps `title=` regardless of the honored set (its docstring assumes serialization drops it), and `_promoted_statistic_field` reconstructs the channel with it. Owner, Python: `chart.py:147`, `composition.py` `_promoted_statistic_field`.
11. **R7 / G11. Span and polar aggregates refused** (P4). `theta="sum(z)"` (a pie) rendered on SM1 and refuses on CUR (regression, c7). `x2="max(v)"` refuses on CUR and failed on SM1 (gap, open item 1). Owner: `encoding/base.py:552`, `_honored.py`, the aggregate lowering and the polar remap (`_spec_build._remap_polar_channels`, l. 371). D2 for theta/radius.
12. **G9. Layered polar charts fail** (P8; gap, both trees). `arc(theta="z", color="g").coord(CoordPolar()) + text(theta="z", text="g").coord(CoordPolar())`, and `arc.layer(Layer(text))`, raise `encoding 'y' expected EncodingSpec, got None`. Root not traced beyond the serialization of a polar layer with no radius. Owner: Python `_spec_build` polar layer path.
13. **G12. A bin holds one aggregate** (P3; gap). `bar(x=bin(v), y=mean(y), color=mean(z))` refuses. Owner: Rust `transform/bin.rs` (single `agg_field`/`agg`) and the Python fold in `_lower_bin_encoding`.
14. **G13. A composition-level binding misses a different-frame operand's same-named column** (P2 by-name; gap, both trees). `(P(df) + Chart(df2).mark_line().encode(x="v", y="y")).encode(color="g")` draws the line uncolored, with a legend of a, b only. Without the name overlap (`gg`), or as `LayerChart`, it draws 3 lines with a, b, c. Root: x/y are re-read by name in the right frame (`positional_renames`, T6), other channels are not. Owner, Python: `_point_operand_layers` / `_inherit_chart_positional` (`chart.py:415` area). On the engine-defect list since T6 (b).
15. **K2. Order dependence for channels outside the promotion list** (P2, P8; contradiction, both trees). `line(detail="c") + line` draws 4 polylines and `line + line(detail="c")` draws 3: detail leaks left → right but is never promoted right → left. Owner: `composition.py:4518` (promotion set) vs the left-encoding-as-composition rule (`Chart._merge`, `chart.py:2931`). Closed by D1-A.
16. **K3. Titles differ by door** (P6; contradiction). A promoted bin is titled `bin_start` (open item 2). A promoted aggregate is titled `mean_z` over the same frame and `z` over a different frame. The axis title of an aggregated `y` is `mean_y` flat, which is consistent. Owner: `composition.py` `_promoted_statistic_field`, `chart.py:147`.
17. **G10. False "every row has a null in its groupby key" warning** (P7; gap). A faceted host with `Layer(data=df2)` inheriting `y=mean(y)` renders correctly, but the overlap no-chain route makes the right layer aggregate the combined rows, and an empty panel warns. Owner, Python: the overlap no-chain branch of `Chart._merge` (`chart.py:2931ff`), the T6 engine item (c): route every operand through its own-rows branch.
18. **G-spec. Spec text drift** (P8 for the contract). `ferrum-spec.md` "Color domains" says `+` uses the first color layer's domain (CUR unions, per c6). "Encode-level binning" says more than one aggregating channel refuses (c5 counts distinct (field, op)). "Layer promotion" says `Chart.layer` promotes (only channel objects are). "Chart-level encodings are shared" and the `LayerChart` interactive note describe sibling sharing that D1-A would remove. Owner: `ferrum-spec.md` §3.1 / §3.2 / §3.6, Task 17.
19. **K4. Ruling-level contradiction.** c3 named "a `+` right layer inheriting the left's color ... groups by c too, colored per (g, c)" as an intended change. c7 ruling 3 states the user's expectation that "adding a layer never changes how many marks another operand draws". R1's repros satisfy the first and violate the second. Only D1 resolves it.

---

## 5. Decisions that need the user

### D1. Do bindings cross between siblings? (P2; reverses spec "Chart-level encodings are shared" as implemented, "Layer promotion" (2026-10-01, 2026-10-12), c7 ruling 3, and c3's named change)

- **D1-A. Sibling isolation (Vega-Lite / ggplot2).** A `+` operand's or a `Layer`'s bindings are its own layers'. The composition's parent is only what is written on the composition. Every auxiliary scale is resolved from every layer that draws it, which makes promotion unnecessary. Ruling A stays as P5's axis fallback (layer 0's written x/y).
  - *Coherence:* one inheritance rule. It reuses the conditional scope rule, and deletes promotion, statistic stripping, the `_take_chart_statistics` merge copy and the renamed-field leak. R1, R2, R5, R6, K1, K2, K3 and K4 close by construction, and `LayerChart` static and merged agree with no route change.
  - *Users:* no layer's marks change when another is added; a shared binding is written where it is visibly shared (`(a + b).encode(color=...)` or the host).
  - *Libraries:* matches Vega-Lite and ggplot2 exactly.
  - *Cost:* `P(color="c") + line` stops splitting the line (today 2 polylines, Altair 1). Figure functions that compose `+` and relied on the left's color reaching the right must bind it explicitly; their goldens need measuring.
- **D1-B. Keep sharing, make it grain-preserving.** Keep the left encoding as the parent and keep promotion, with one added rule: a binding that arrives from a sibling never changes a layer's grain. A raw layer draws its raw field. An aggregating layer computes an arriving aggregate as one more op, and ignores an arriving raw field as a group key (it shares only the scale). The field is read by name in each operand's frame.
  - *Coherence:* needs a sibling-versus-parent provenance mark on every inherited channel, the shape that took four cycles in Task 6. `LayerChart` static still differs from merged unless static also adopts sharing.
  - *Users:* keeps today's colored overlays. "Inherited but not grouped" is subtle: a promoted `color="c"` reaching a mean bar would be dropped there.
  - *Libraries:* no incumbent does this.
- **D1-C. Status quo.** Leaves R1, R2, R5, K1, K2 and K4 open.

**Recommendation: D1-A.** It is the only option where the three criteria agree, and it removes a mechanism instead of adding one.

### D2. Do `theta` / `radius` honor `aggregate=`? (P4; reverses c7 choice 1 for the polar channels)

- **D2-A. Honor them, as c8 ruled for x2/y2.** `theta="sum(z)"` is the Vega-Lite pie spelling and rendered on SM1. Coherent with P4 ("draws a value"), it meets user expectations, and it matches Vega-Lite. Cost: the polar remap must run before aggregate lowering.
- **D2-B. Keep the refusal with the `.transform(aggregate)` remedy.** Coherent with today's polar desugar, but it fails the user-expectation test and diverges from Vega-Lite.

**Recommendation: D2-A.**

### D3. What does `bin=` mean on a non-positional channel? (P3, P4; c5 and c7 left this open)

- **D3-A. Color, fill and stroke bins become a discretizing color scale** (Vega-Lite's binned legend). Rows are unchanged and the edges are nice bins of the field, reusing ferrum's existing discretizing color scales. Every other non-positional channel refuses `bin=` at construction in c7's voice. Coherent (it reuses scale machinery and adds no row statistic), it is what users expect from Vega-Lite, and it aligns with that library.
- **D3-B. Refuse `bin=` on every non-positional channel**, with the remedy `.transform(fm.transform_bin(...))`. Simplest, and coherent with c7. It refuses a Vega-Lite idiom.

**Recommendation: D3-A for color/fill/stroke, D3-B for every other channel.**

### D4. What does a positional bin without an aggregate draw? (P3)

- **D4-A. A per-row bin**: each row placed at its bin, with `x2 = bin_end`, as Vega-Lite does. Needs a non-aggregating mode on the Rust `Bin` (or a `Calculate` floor). Coherent with P3, it is what users expect, and it aligns with Vega-Lite.
- **D4-B. An implicit count**, as ggplot2's `stat_bin` does. It would surprise anyone who bound `y` to a real column.
- **D4-C. Refuse**, naming `y="count()"` or `transform_bin`. Safe, but it refuses a valid Vega-Lite chart.

**Recommendation: D4-A.**

No other cell needs the user: X2/Y2 aggregates and the source-field bin title are already ruled (c8, 2026-10-16), and every remaining gap follows from P1–P8 with a single answer.

### Rulings recorded after adoption

- **Opacity maps on every mark (2026-10-16).** Raised while adding the opacity legend: six marks drew `opacity=<field>` raw while four mapped it. Ruled: every mark maps the opacity family through its one scale (P4, P5), so marks and legend agree. See the P5 *Every mark* line and the §2.3 "mark kind" row.
- **Facet over a frame without the facet column (2026-10-17).** Raised in Task 20, when own-rows routing left a different-frame layer's rows in no panel. Ruled: a layer whose frame lacks a facet field repeats in every panel, with its statistic over its own rows; rows whose facet value is null, in a frame that has the field, are dropped with a warning that names the count. P1 holds (each layer still reads only its own rows) and P7 holds (nothing drops silently). Reverses task-16-decision-c3's "rows in no panel" pin. See the §2.5 rows.
- **Inherited positional bindings only on drawn channels (2026-10-17).** Raised in Task 20, when per-operand column projection made reference rules, function lines and count rules fail on an inherited x/y/x2/y2 column their rows lack. Ruled: a layer takes an inherited positional binding only on channels its mark draws; a drawn channel with no column refuses. See the P2 *Drawn channels* line and the §2.1 "inherited binding, undrawn channel" row.

---

## 6. Architecture: reuse vs rewrite

The principles need eight subsystems. For each: the verdict, the evidence (the gaps today's structure causes) and the library precedent.

| Subsystem | Today | Verdict | Evidence | Precedent |
|---|---|---|---|---|
| **A1. Row scoping** (which rows a layer reads) | `ChartRows` record, own-rows `Branch`es, `Branch`-scoped publication (T4), the `__lhs_row` / `__rhs_row` markers, diagonal concat with `__rhs_N` renames, plus an "overlap no-chain" route that reads the combined rows | **Reshape.** Keep the record and branches. Route every different-frame operand through its own-rows branch, so a renamed column is restored inside its operand and never escapes. | G10 (false null-group warning), R2 (`z__rhs_0001` null groups), G13 (a by-name binding misses the right frame) | Vega-Lite gives a layer with its own `data` a separate dataset; ggplot2 a separate layer `data` |
| **A2. Binding inheritance** | Five overlapping mechanisms: the left encoding as the composition's, `_promote_layer_channels` (right → chart), `_promoted_statistic_field` (stripped statistics), `_take_chart_statistics` / `_inherit_chart_statistics` (statistic copies), plus `_written_positional`, `inherited_positional`, `positional_renames`, alias handling and Rust `inherit_from` | **Rewrite** (under D1-A) into one parent model. A layer's effective encoding is its own bindings merged over its parent's, computed once at `to_spec` by one Python function that is alias-aware and reads each field by name in the layer's frame. Rust `inherit_from` receives that parent, not layer 0's merged encoding. Keep `_written_positional` for the P5 axis fallback. Delete promotion and statistic stripping. Under D1-B it would be a reshape with a provenance flag instead. | R1, R2, R5, R6, K1, K2, K3, K4: eight entries from one structure. Each of c3, c6 and c7 patched one face of it | Vega-Lite parent/child encoding; ggplot2 `inherit.aes` |
| **A3. Statistic planning** (grain, groupby, ops, bins) | Flat `_resolve_pending_aggregates` / `_resolve_pending_bins`, per-layer `_resolve_layer_aggregates` / `_resolve_layer_bins`, `_lower_bin_encoding`, `_bin_groupby`, and the composite builders each choose their own grouping | **Reshape.** One plan per layer from its effective encoding (grain, groupby including the facet fields, the aggregate ops, the bin plan), used by flat, per-layer and composite statistics alike. Rust gains a per-row bin mode (D4-A) and multi-aggregate bins (G12). Smooth and KDE gain a groupby. | R3, R4, G8, G12; composites group by color only, smooth by nothing | Vega-Lite: one aggregate per view, bin separate from aggregate; ggplot2 `group` = interaction of the discrete aesthetics |
| **A4. Scale and legend resolution** (Rust) | `build_auxiliary_scales` decides whether a scale exists from `spec.encoding` (layer 0 merged); c6/c7 `aux_reads` unions color/size/opacity domains; shape and stroke_dash read the chart encoding only; static `LayerChart` resolves legends per member | **Reshape.** A channel's scale exists, takes its kind (layer 0 decides) and takes its domain from every layer that draws it, for every auxiliary channel including shape and stroke_dash. The composite overlay dedups legends by channel and field. | G7 (a layer-only channel is not drawn), K1 (two `g` legends), the reason promotion exists (its docstring) | Vega-Lite shared scales and merged legends by default; ggplot2 trains scales over every layer |
| **A5. Channel vocabulary** | `_honored_kwargs` / `_refusal_message` (c7) for `aggregate`, warnings for `bin`, a `Layer(encoding=)` that bypasses `encode()`'s parser, rename and promotion copies that stamp `title=` | **Reuse and extend.** Keep the table and the gate. Add the `bin` refusal (R3), the x2/y2/theta/radius aggregate honor (G11, R7), `Layer(encoding=)` through `encode()`'s resolver (G7), and no kwarg stamping that bypasses the gate (R6). | R3, R6, R7, G7, G11 | Vega-Lite's per-channel schema; Altair's one shorthand parser |
| **A6. Composition routes** | `LayerChart` static overlay tree vs `_build_merged` through `+` | **Reuse as-is under D1-A.** Isolation makes the overlay tree and the merged path compute the same chart. Add a route-parity test over the rule table. Under D1-B the static route would have to adopt sharing (a reshape). | K1 | Vega-Lite has one semantics per spec |
| **A7. Refusals** | `_known_columns`, `_refuse_missing_statistic_inputs`, `_refuse_secondary_field`, Rust `LayerUnknownColumn` with the published formatter | **Reuse.** Extend the known schema to "the left operand's frame" for `SecondaryY` (G6), and Rust's gate to null-padded and non-positional columns (engine item (a) from c2). | G6 | Vega-Lite warns on unknown fields; ferrum refuses by policy |
| **A8. Titles** | Flat channel rebind, `_promoted_statistic_field` title stamp, `_rename_encoding_fields` title stamp | **Reshape.** One title derivation from the establishing layer's written binding (P6). The two stamp paths are deleted with promotion and renaming under A1/A2. | K3, R6 | Vega-Lite titles from the field definition |

Summary: one rewrite (A2, binding inheritance), four reshapes (A1, A3, A4, A8) and three reuses (A5, A6, A7). The rewrite is justified because eight of the twenty entries trace to A2's five overlapping mechanisms, and decisions c3, c6 and c7 each closed one face of it while opening another. That is the pattern this document was asked to end.
