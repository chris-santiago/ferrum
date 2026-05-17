# Ferrum Marketing Copy

## Positioning lines

- Ferrum is a new statistical visualization library for Python built around one idea: every chart should follow the same mental model.
- Ferrum keeps layered grammar, interactivity, statistical transforms, and model diagnostics in one chart system instead of splitting them across separate tools.
- Ferrum uses Python as the declaration layer and Rust as the computation layer, with Arrow CDI handoff and multiple rendering backends for scale.
- Ferrum treats model outputs as data, so diagnostics stay composable, themeable, and interactive instead of becoming one-off visualization objects.
- Ferrum is built for the real Python data ecosystem, with broad dataframe interoperability through Narwhals and direct zero-copy paths where possible.

## Short posts

- Ferrum is a new statistical visualization library for Python. One chart model for scatter plots, faceted distributions, ROC curves, SHAP beeswarms, and interactive analysis.

- Ferrum brings together grammar-of-graphics composition, renderer-driven interactivity, Rust-backed performance, first-class ML diagnostics, and dataframe interoperability through Narwhals in one library.

- Plotting shouldn’t require switching mental models every time the question changes. Ferrum is built so statistical plots, interactive charts, and model diagnostics all feel like the same language.

- Ferrum is grammar-first, but not grammar-only: high-level helpers are just sugar over the same chart system, not a separate API surface.

- Ferrum is built for real data and real Python workflows: SVG when it makes sense, raster when scale demands it, WASM when you want interaction, and Narwhals when your dataframe stack is not just one library.

- Ferrum is designed for the messy reality of Python data work: pandas, Polars, Arrow, and broader dataframe interoperability without changing the chart model.

## Longer posts

### Launch-style

Ferrum is a statistical visualization library for Python designed around a simple idea: every chart should follow the same mental model.

That means scatter plots, faceted distributions, calibration curves, lift charts, SHAP summaries, and interactive views all live inside one grammar instead of separate toolchains.

Under the hood, Ferrum uses Python for declaration, Rust for computation, Arrow CDI for data handoff, Narwhals for broad dataframe interoperability, and multiple rendering backends for static, raster, and interactive output.

### Why-now style

Python visualization still fragments one workflow into too many abstractions: one tool for layered charts, another for interactivity, another for convenience plots, and another for ML diagnostics.

Ferrum is an attempt to unify those worlds so users can stay inside one coherent chart system from exploration to explanation to model evaluation.

It is also built for the Python ecosystem as it actually exists, where teams move between pandas, Polars, Arrow, and other dataframe APIs rather than standardizing on a single table type.

### Technical-audience style

Ferrum is built around grammar-first charting, first-class statistical transforms, renderer-level interactivity, and model artifacts as data.

It includes typed encodings, layers, themes, faceting, compound views, figure-level helpers, model-diagnostic marks, and sklearn-style visualizers without falling back to a separate plotting backend or a disconnected diagnostics API.

It also meets users where their data already lives, combining Narwhals-based interoperability with direct columnar execution paths for high-performance rendering and transforms.

## Hooks and taglines

- One chart model for statistical graphics and ML diagnostics.
- Grammar-first plotting for real Python workflows.
- Interactive when needed, composable by default.
- From scatter plots to SHAP, one mental model.
- Statistical visualization without fractured APIs.
- Built for layered charts, large data, and model evaluation.
- Declarative plotting with Rust-backed execution.
- Diagnostics are charts, not special cases.
- One system for plotting, statistics, interaction, and dataframe interoperability.
- Fast. Composable. Statistically honest.
- Built for pandas, Polars, Arrow, and beyond.

## Channel-specific angles

### X / Threads

Lead with the problem-solution contrast: too many plotting mental models versus one chart system.

A second strong angle is that Ferrum works across the broader dataframe ecosystem instead of assuming one blessed table type.

### LinkedIn

Emphasize the architecture story — Python declaration, Rust execution, Arrow CDI, Narwhals interoperability, and the unification of diagnostics with charting.

### GitHub / README / Show HN

Emphasize the design thesis, prior-art synthesis, breadth of coverage from Altair/Seaborn-style plotting through Yellowbrick-style diagnostics, and support for real-world dataframe stacks.

### Product Hunt

Lead with user value: composable charts, interactive views, strong defaults, ML diagnostics, and compatibility with the dataframe tools people already use.

## Internal Slack prompts

- Tired of bouncing between plotting libraries every time the task changes or the data gets bigger? Check out Ferrum — one chart system for statistical plots, model diagnostics, and interactive analysis.
- Tired of switching plotting libraries just to go from EDA to diagnostics to large-data rendering? Check out Ferrum.
- Bouncing between Seaborn, Altair, Plotly, and diagnostics tools depending on the job? Check out Ferrum — one coherent plotting system built for real data science workflows.
- Frustrated that your plotting stack changes every time your question changes? Check out Ferrum.
