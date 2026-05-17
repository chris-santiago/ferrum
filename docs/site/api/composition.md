# ferrum.composition

Compound view classes for combining multiple charts into a single output.

This module provides:

- **HConcatChart** / **VConcatChart** -- horizontal and vertical concatenation (`|` and `&` operators)
- **LayerChart** -- overlay multiple charts on shared axes (class-based `+` alternative)
- **ConcatChart** -- wrapping grid layout with configurable column count
- **JointChart** -- central plot with marginal distributions
- **RepeatChart** -- template chart repeated across field combinations
- **ClusterMapChart** -- clustered heatmap with dendrograms

::: ferrum.composition
    options:
      members_order: source
      show_root_heading: false
      show_root_toc_entry: false
      filters: ["!^_"]
