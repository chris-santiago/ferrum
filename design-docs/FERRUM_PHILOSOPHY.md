# Ferrum Philosophy

Ferrum exists because Python visualization still fragments one activity into too many mental models.

Statistical graphics, interactive graphics, convenience plotting, and machine learning diagnostics are often treated as separate domains with separate abstractions, even though they are all just different ways of turning structured data into visual form.

The result is unnecessary cognitive switching.

A user may start with a layered charting library, move to a convenience library for a distribution plot, switch again for interactivity, and then adopt a completely separate package for ROC curves, calibration, or SHAP summaries.

Each switch introduces a new object model, a new set of defaults, and a new set of limitations.

Ferrum is a rejection of that fragmentation.

It is built on the belief that the same conceptual system should cover exploratory plotting, statistical transformation, interactive analysis, and model diagnostics.

## Why Ferrum exists

Ferrum is not trying to be another plotting wrapper.

It is trying to make a stronger claim: that a scatter plot, a faceted histogram, a confusion matrix, a partial dependence curve, and a SHAP beeswarm are all charts and should therefore live inside one coherent grammar.

That claim has consequences.

It means convenience functions cannot become separate worlds, interactivity cannot require a different chart object, and model diagnostics cannot be treated as bespoke artifacts that stop being composable the moment they appear.

This is why Ferrum is grammar-first.

Not because low-level APIs are inherently virtuous, but because a common grammar is the only stable foundation for a broad library that wants to span statistics, interactivity, and diagnostics without collapsing into inconsistency.

## Core beliefs

### Grammar first, convenience second

Every visualization in Ferrum is a composition of the same primitives: data, encodings, marks, scales, coordinate systems, transformations, and views.

High-level helpers such as `displot`, `lmplot`, `rocchart`, and similar figure-level functions exist because speed matters, but they are sugar over the grammar rather than parallel APIs with different rules.

This principle prevents feature silos.

A convenience plot should not trap users in an object that behaves differently from the rest of the library, and a chart created through a helper should remain themeable, composable, and extensible in the same way as a chart written from first principles.

### Model artifacts are data

A confusion matrix is a table.

A ROC curve is derived tabular data from predicted scores.

A SHAP explanation is a set of columns.

There is no philosophical reason these objects should require a separate visualization universe.

Ferrum therefore treats model outputs as data sources that feed the same chart system as any other dataset.

This is the rationale behind `ModelSource`, diagnostic marks, figure-level model helper functions, and sklearn-protocol visualizers that return charts rather than foreign backend objects.

The point is not only conceptual cleanliness.

It also means a ROC curve and a confusion matrix can be concatenated, themed, saved, and embedded using the same composition rules as any other Ferrum chart.

### Statistics belong in the rendering pipeline

Users should declare intent, not manually stage every transformation outside the chart.

Computing a KDE, bootstrapping a confidence interval, fitting a LOESS curve, binning a field, or deriving a calibration curve should be first-class operations in the visualization system.

Ferrum therefore places statistical transforms inside the engine rather than forcing users to precompute every intermediate table in Python.

This keeps plotting code shorter, makes statistical assumptions visible in the chart specification, and helps ensure that statistical layers compose naturally with facets, themes, interactivity, and export.

Ferrum is not merely a renderer with nicer defaults.

It is intended to be statistically literate software.

### Interactivity is a renderer, not a rewrite

Static and interactive charts should not be different species of object.

If a chart becomes interactive, the user’s conceptual model should stay intact.

Ferrum takes the position that interactivity belongs to rendering, not authorship.

Selections, linked views, zoom, pan, and conditional encodings are declared in the chart specification, while `.interactive()` changes the rendering path rather than requiring a new library or a new grammar.

### Performance is part of the API

A plotting library cannot claim a coherent user model if that model collapses under real data volume.

Performance is not only an engineering concern; it shapes what the public API can honestly promise.

Ferrum therefore treats scale as a first-class design problem.

Python serves as the declaration layer, Rust as the computation layer, and data crosses the boundary once through the Arrow C Data Interface to avoid unnecessary row-level copying.

The rendering system includes SVG, CPU raster, and GPU/WASM backends, plus explicit raster marks and configurable auto-raster policies for large mark counts.

The philosophical commitment here is continuity.

Users should not have to change libraries, rewrite their plots, or adopt a second API just because their dataset stopped being toy-sized.

### Dataframe pluralism is part of the contract

Ferrum is designed for the Python data ecosystem as it exists, not as it would look if everyone standardized on one dataframe implementation.

Users bring pandas, Polars, Arrow, modin, cuDF, dask, ibis, NumPy arrays, and model-derived tables into the same plotting workflow, so the visualization layer should meet them where they are rather than force a conversion-first worldview.

This is why Ferrum treats interoperability as part of its public contract.

Narwhals broadens compatibility across dataframe APIs, Polars can cross the Python-Rust boundary through direct Arrow CDI handoff, and native Arrow inputs preserve the columnar execution model that Ferrum’s stat and rendering pipeline is built around.

The point is not “support many dataframe libraries” as a checklist item.

The point is that one chart grammar should remain stable across the messy reality of Python data work, just as it should remain stable across small and large datasets, static and interactive output, and ordinary plots versus model diagnostics.

### Defaults should be correct, not merely attractive

Default choices are not cosmetic.

They teach users what the library considers normal, safe, and statistically honest.

Ferrum’s philosophy is that defaults should begin from epistemic correctness: perceptually uniform and colorblind-safe palettes, accessible typography, sensible binning floors, and descriptive errors when inference is ambiguous or likely to mislead.

A plotting library should not quietly guess its way into a wrong chart when it could instead explain the ambiguity and ask the user to be explicit.

## Design consequences

These beliefs lead directly to Ferrum’s public shape.

The library has one primary chart object, typed encoding channels, composition operators, chart- and layer-level transforms, statistical marks, figure-level helpers, and themes as value objects rather than mutable module state.

They also explain Ferrum’s architecture.

Python constructs the specification, Rust executes layout and transform work, the SceneGraph acts as a renderer-agnostic intermediate form, and different backends serve static vector output, static raster output, and interactive WASM output without changing the chart’s conceptual identity.

They explain the diagnostics layer as well.

`ModelSource` exposes derived tables for predictions, probabilities, ROC and PR curves, confusion matrices, calibration curves, lift and gain curves, SHAP values, partial dependence, learning curves, validation curves, and related outputs so that model evaluation remains a part of charting rather than a detour away from it.

They also explain why Ferrum cares about operational simplicity.

A visualization library that requires fragile system dependencies is harder to use in the places where real work happens, so Ferrum favors a pure-Rust rendering stack, wheel-based installation, and headless execution paths that work cleanly in CI, containers, servers, and remote shells.

Finally, they explain Ferrum’s stance on explainability at scale.

If SHAP, ICE, and related model-understanding plots matter most when datasets are large, then they must remain first-class citizens even when that requires rasterization, GPU-backed interactivity, or other backend-level adaptation.

## Tradeoffs

Ferrum’s philosophy is intentionally opinionated.

It rejects a matplotlib fallback, rejects mutable global styling in the style of `rcParams`, and rejects silent inference that fails without explanation.

These choices trade familiarity for consistency and explicitness.

The library also accepts a broad API surface in order to avoid conceptual fragmentation.

That makes Ferrum larger than a minimalist plotting package, but the bet is that one large coherent system is better than several smaller incompatible ones stitched together at the application layer.

There are also scope limits.

The 1.0 target aims to cover the chart types a user would reasonably reach for when moving from Altair, Seaborn, or Yellowbrick, while explicitly leaving features such as graph layout, Gantt charts, geographic tile layers, 3D coordinate systems, animation frame encoding, real-time streaming sources, and non-Python bindings outside the 1.0 boundary.

## Relationship to prior art

Ferrum is not built in opposition to prior libraries so much as in response to the seams between them.

It inherits grammar-of-graphics layering, explicit scales, and faceting ideas from plotnine and ggplot2; typed encodings, selections, and composition ideas from Altair; statistical vocabulary and figure-level helpers from Seaborn; interactive output ideas from Plotly; and diagnostic vocabulary from Yellowbrick and scikit-plot.

What Ferrum rejects is not the value of those libraries, but the fractures between their strengths.

Ferrum does not want users to choose between a grammar library, an interactive library, and a diagnostics library depending on the day’s task.

It wants those tasks to feel like one practice carried by one system.

## What success looks like

Ferrum succeeds if users can stay inside one mental model from first exploration to final diagnostic review.

A simple distribution plot, a polished publication graphic, a linked interactive view, and a threshold-tuning chart for a classifier should feel like variations of the same language rather than migrations across tools.

That is the reason Ferrum exists.

The goal is not maximal novelty, but a plotting system that is broad, fast, statistically honest, operationally simple, and conceptually unified enough that users stop paying the tax of switching abstractions every time their question changes.
