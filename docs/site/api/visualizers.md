# Visualizers

Sklearn-protocol diagnostic visualizers. Each visualizer follows the
`Viz(model).fit(X, y).show()` pattern and is composable inside sklearn
pipelines, `GridSearchCV`, and external evaluators.

For the equivalent one-liner functional API, see [ferrum.plots](plots.md).

## Base class

::: ferrum._diagnostics.visualizers.base.FerrumVisualizer
    options:
      members_order: source
      show_root_heading: true
      show_root_toc_entry: true
      filters: ["!^_"]
      heading_level: 3

## Classification

::: ferrum._diagnostics.visualizers.classification
    options:
      members_order: source
      show_root_heading: false
      show_root_toc_entry: false
      filters: ["!^_"]
      heading_level: 3

::: ferrum._diagnostics.visualizers.classification_extra
    options:
      members_order: source
      show_root_heading: false
      show_root_toc_entry: false
      filters: ["!^_"]
      heading_level: 3

## Regression

::: ferrum._diagnostics.visualizers.regression
    options:
      members_order: source
      show_root_heading: false
      show_root_toc_entry: false
      filters: ["!^_"]
      heading_level: 3

## Clustering

::: ferrum._diagnostics.visualizers.clustering
    options:
      members_order: source
      show_root_heading: false
      show_root_toc_entry: false
      filters: ["!^_"]
      heading_level: 3

## Explanation (SHAP)

::: ferrum._diagnostics.visualizers.explanation
    options:
      members_order: source
      show_root_heading: false
      show_root_toc_entry: false
      filters: ["!^_"]
      heading_level: 3

## Model selection

::: ferrum._diagnostics.visualizers.selection
    options:
      members_order: source
      show_root_heading: false
      show_root_toc_entry: false
      filters: ["!^_"]
      heading_level: 3

## Ranking

::: ferrum._diagnostics.visualizers.ranking
    options:
      members_order: source
      show_root_heading: false
      show_root_toc_entry: false
      filters: ["!^_"]
      heading_level: 3

## Data sources

::: ferrum._diagnostics.sources
    options:
      members_order: source
      show_root_heading: false
      show_root_toc_entry: false
      filters: ["!^_"]
      heading_level: 3
