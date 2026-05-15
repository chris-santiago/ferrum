# Interactivity is a renderer, not a rewrite

Static and interactive charts should not be different species of object.

Most Python visualization stacks force a hard boundary between the two. A scatter plot you authored in seaborn for an EDA notebook is not the same kind of artifact as a Plotly scatter plot in a dashboard, even though both are "the same chart" in any sense that matters to the user. The conventional way out of that fragmentation is to author the chart twice: once for static output, once for interactive. The cost is paid every time the question changes.

Ferrum's position is that interactivity is rendering, not authorship. If a chart becomes interactive, your conceptual model should stay intact. The selections, zoom, pan, and linked-view declarations live inside the chart spec; the difference between "static" and "interactive" is which renderer consumes the spec, not what kind of object you wrote.

## The design contract

The contract has three parts, and they only make sense together:

**One chart object.** A static scatter plot and its interactive counterpart are the same `Chart`. You do not construct an "interactive scatter" — you construct a scatter, and then a renderer makes it interactive. The chart spec carries the selection and behavior declarations; the renderer carries the runtime that resolves them.

**Selections are declared in the spec.** A selection — "click points to highlight," "drag to brush," "hover for details" — is a property of the chart, not a side-effect of how the renderer is wired. Selections live in the spec the same way encodings and marks do. That means a chart with selections is still a value that can be themed, concatenated, faceted, and saved.

**Linked views fall out of composition.** Because composition operators (`hconcat`, `vconcat`, `JointChart`, `RepeatChart`) take charts and return compound views, and selections live in charts, linked-view behavior comes from declaring a selection in one chart and referencing it in another. There is no second API for "make these two plots linked."

The principle is the same as for stats and for diagnostics: the structural decision — *where does interactivity live?* — is fixed by the library so the user-facing grammar can stay invariant.

## Why not "an interactive plotting library"

Existing tools take the opposite stance: interactivity is a different object model. Plotly, Bokeh, and Altair-with-VegaLite-runtime each have their own chart types, their own theming systems, their own composition rules, and their own static-export quirks. The cost is felt by users who want both static and interactive output: you maintain two specs for the same plot, drift them out of sync, and discover the inconsistency when a slide deck looks different from the dashboard.

Ferrum is built around a different bet. The single hard problem in interactivity is the runtime — making selections fast, keeping linked views in sync, handling user input without a perceptible lag. That problem is in the renderer, not in the chart grammar. Solving it once, at the renderer layer, lets the same chart grammar feed both static and interactive output.

This is also why the interactive renderer is GPU-backed (and WASM-targeted). The same scale problem that makes large static plots painful — millions of marks — makes interactive plots impossible without GPU-level throughput. Putting the renderer in Rust + GPU + WASM is what makes [SHAP and ICE at full sample size](performance-scale.md#shap-and-ice-at-full-sample-size) viable as interactive views, not just as static rasters.

## What "interactive" covers

The interactive renderer resolves four classes of behavior, all declared in the chart spec:

- **Selections** — point, brush, interval, and value-binding selections. The chart says "this selection exists"; the renderer makes it respond to clicks and drags. See [`selection_point`][ferrum.selection_point] and [`selection_interval`][ferrum.selection_interval].
- **Zoom and pan** — declared per-coordinate or per-encoding. The chart says which axes are zoomable; the renderer wires the gestures.
- **Linked views** — a selection in one chart can drive an encoding in another when the two charts are composed. Compound views carry the linking; no separate "link API" is needed.
- **Tooltips and hover** — the `Tooltip` encoding and `TooltipField` declarations are honored by both static and interactive renderers. In static output they become accessibility metadata; in interactive output they become the hover layer.

What the interactive renderer is **not** trying to do: animation as a first-class encoding, real-time streaming data sources, or general-purpose dashboard layout.

## What changes for the user, and what doesn't

A chart you wrote with the static renderer in mind keeps working when interactivity lands. You don't rewrite encodings, you don't swap your composition operators, and you don't reach for a different library. You call a different render path — `.interactive()` — and the same chart spec becomes interactive output.

What does change is what you can express. Once interactivity is part of the renderer contract, you have a vocabulary for selections and linked views inside the chart spec itself. The shape of the spec gets richer; the shape of your code does not get more complicated.

## Current status

| Capability | Status |
|---|---|
| Chart spec accepts encodings, marks, scales, composition | ✓ Shipping |
| `Chart.interactive()` returns interactive WASM view | ✓ Shipping |
| Static SVG renderer | ✓ Shipping |
| Static CPU raster renderer | ✓ Shipping |
| Selection declarations in chart spec | ✓ Shipping |
| WASM/GPU interactive renderer | ✓ Shipping |
| Linked views via composition operators | ✓ Shipping |
| Zoom / pan / brush / hover gestures | ✓ Shipping |

For worked examples and API details, see the [Interactive rendering](../interactive.md) guide.

## Where to go next

- [One chart model](one-chart-model.md) for the grammar that interactive output is built to preserve.
- [Performance & scale](performance-scale.md) for the architecture that lets the same chart spec render at very different data sizes — the same architecture that makes interactive views viable.
- [Model outputs are data](model-outputs-as-data.md) for why diagnostic plots, including the interactive ones, are charts in the same sense as everything else.
