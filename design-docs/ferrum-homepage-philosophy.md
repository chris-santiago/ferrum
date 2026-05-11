# Ferrum Philosophy for the Docs Home Page

Ferrum is a statistical visualization library for Python built around one idea: every chart should follow the same mental model.

A scatter plot, a faceted histogram, a ROC curve, and a SHAP beeswarm are all charts, so Ferrum builds them from the same grammar of data, encodings, marks, scales, coordinates, and statistical transforms.

Ferrum is grammar-first, but not grammar-only.

Figure-level helpers such as `displot`, `lmplot`, `rocchart`, and other convenience functions exist for speed, yet they are sugar over the same chart system rather than parallel APIs with different rules.

Ferrum is interactive without becoming a different library.

Static and interactive charts share one spec and one chart object; calling `.interactive()` changes the renderer, not the user’s mental model.

Ferrum is built for real datasets and real Python data stacks.

Python is the declaration layer, Rust is the computation layer, Arrow CDI avoids unnecessary copies, Narwhals broadens dataframe interoperability across the wider ecosystem, and high-mark plots can switch to rasterized representations when scale demands it.

The goal is not just speed in isolation, but continuity: the same chart model should work at 100 rows and at production-scale data sizes without forcing users to swap libraries or rewrite their plots.

Ferrum also aims to be operationally simple.

It ships without a matplotlib dependency, avoids system-level rendering stacks such as Cairo and X11, and is designed to render cleanly in notebooks, CI, containers, SSH sessions, and other headless environments.

Ferrum treats model outputs as data.

ROC curves, confusion matrices, calibration plots, lift charts, learning curves, and SHAP summaries are first-class chart types because they compile into the same grammar and remain composable, themeable, and interactive.

That means diagnostics are not a separate universe: a ROC curve can be concatenated, themed, and saved with the same tools as any other chart.

Ferrum’s defaults aim to be statistically and visually correct.

Default palettes are perceptually sound, typography is accessibility-minded, summary statistics live in the rendering pipeline, and inference should fail descriptively rather than silently drifting into the wrong chart.

## What Ferrum is trying to unify

Most Python visualization workflows still fracture across multiple mental models.

One library handles layered statistical graphics, another handles interactivity, another handles convenience plots, and another handles machine learning diagnostics, which means users keep switching abstractions as soon as the task changes.

Ferrum is designed to remove that boundary.

The same chart system should support a simple scatter plot, a faceted distribution analysis, a linked interactive view, and a model diagnostic suite without forcing users into separate object types or unrelated APIs.

## Principles

### One chart model

Every visualization in Ferrum starts from the same building blocks: data, encodings, marks, scales, coordinate systems, transforms, and composition.

That is why high-level helpers are convenience layers rather than special cases, and why model diagnostics are charts instead of bespoke visualization objects.

### Statistics belong in the chart

KDEs, binned summaries, smoothing lines, confidence intervals, regression fits, and diagnostic transforms should be declarative plotting operations rather than manual pre-processing steps in userspace.

Ferrum treats these as first-class statistical transforms executed by the engine before rendering.

### Interactivity is a render mode

Interactivity should not require a second library or a second object model.

Selections, zoom, pan, and linked views are declared in the chart spec and resolved by the renderer, so interactive charts preserve the same structure as static ones.

### Scale is part of the API

Large-data behavior should not be an afterthought.

Ferrum includes explicit raster marks, auto-raster policies, and multiple rendering backends so charts can remain usable when mark counts grow beyond what vector rendering can comfortably support.

### The data ecosystem is plural

Ferrum is built for the Python data ecosystem as it actually exists.

Users bring pandas, Polars, Arrow tables, NumPy arrays, and other dataframe-like inputs into the same workflow, so the plotting layer should meet them where they are instead of requiring one blessed table type.

### Model outputs are data

Model diagnostics should not live in a separate conceptual world.

Ferrum’s `ModelSource`, model-diagnostic marks, figure-level diagnostic helpers, and sklearn-protocol visualizers all exist to keep evaluation plots composable with the rest of the grammar.

## What Ferrum takes from prior art

Ferrum inherits grammar-of-graphics layering, explicit scales, and faceting ideas from plotnine and ggplot2; typed encodings, selections, and composition ideas from Altair; statistical vocabulary and figure-level helpers from Seaborn; and interactive output ideas from Plotly.

It also adopts diagnostic vocabulary from Yellowbrick and scikit-plot, while deliberately avoiding the split mental models, backend coupling, and non-composable helper APIs that those ecosystems often impose.

## The product promise

The goal for Ferrum 1.0 is straightforward: a practitioner moving from Altair, Seaborn, or Yellowbrick should be able to reach for the charts they need without falling back to matplotlib.

Ferrum’s public surface is therefore broad by design, but it is broad around one system rather than a pile of disconnected subsystems.

That promise includes full-sample explainability workflows.

Plots such as SHAP summaries and ICE or partial-dependence views should remain part of the same chart language even when scale requires rasterization or GPU-backed interaction.
