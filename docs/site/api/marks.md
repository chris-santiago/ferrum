# Marks

Marks are the visual primitives of a ferrum chart. Each mark method is
called on a `Chart` instance and returns the chart for fluent chaining.

All mark methods accept keyword arguments that control visual properties
(color, size, opacity, stroke, etc.) as constants. To map these properties
from data columns, use [encoding channels](encoding.md) instead.

This holds for statistical and composite marks too (`mark_density`, `mark_smooth`,
`mark_boxplot`, …): their transform kwargs (`bandwidth`, `ci`, `bin_count`, …) and
the constant mark-style kwargs are independent, and style applies to every layer the
mark emits. See [Marks & Encodings](../guide/marks-encodings.md#friendly-kwarg-aliases).

## Usage

```python
import ferrum as fm

chart = (
    fm.Chart(df)
    .mark_point(size=60, opacity=0.8)
    .encode(x="weight", y="horsepower", color="origin")
)
```

## Mark reference

::: ferrum.chart.Chart
    options:
      members_order: source
      show_root_heading: false
      show_root_toc_entry: false
      filters: ["^mark_"]
      show_labels: false
      group_by_category: false
      heading_level: 3
