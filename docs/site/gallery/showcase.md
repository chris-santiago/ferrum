# Showcase: What You Can Build

Well-known chart designs from the matplotlib, seaborn, Altair, and D3 traditions — built in Ferrum's Rust engine. Every image below was produced entirely in Python using Ferrum's grammar-of-graphics API; no matplotlib, no browser, no external renderer. These designs lean on Ferrum features like categorical color ranges, temporal axis auto-inference, size and shape legends, stroke routing on line marks, sort specs, and annotation date coordinates.

Each card shows the concise API call alongside the rendered output.

---

## Four-encoding charts

Encode x, y, size, and color simultaneously for maximum information density.

<div class="grid cards" markdown>

-   **Gapminder Bubble**

    ---

    ![showcase_gapminder_bubble](showcase_img/showcase_gapminder_bubble.png)

    `fm.Chart(df).mark_point().encode(x="gdp_per_capita", y="life_expectancy", size=fm.Size("population"), color="region")`

    Four simultaneous encodings — position, size, and color — in one readable chart. The size legend uses Ferrum's new multi-legend stacking; both legends render side by side.

</div>

---

## Comparison and annotation

Charts built for the "story in the data" — where one series, one event, or one gap is the point.

<div class="grid cards" markdown>

-   **Highlight Line**

    ---

    ![showcase_highlight_line](showcase_img/showcase_highlight_line.png)

    `ctx = fm.Chart(ctx_df).mark_line(stroke="#cccccc", opacity=0.55).encode(x="x", y="y", detail="series")` + `fm.Chart(hi_df).mark_line(stroke="#e4572e")`

    Nine gray context series recede; one accent series commands attention. `detail=` groups the context lines without adding a color legend.

-   **Dumbbell Chart**

    ---

    ![showcase_dumbbell](showcase_img/showcase_dumbbell.png)

    `rule = fm.Chart(df).mark_rule(stroke="#cccccc").encode(x="score_before", x2=fm.X2("score_after"), y="category")` + `fm.Chart(df_pts).mark_point().encode(x="score", color="period")`

    Before-and-after per category. Connecting rules use `x` + `x2` encoding; a categorical color range pins "Before" and "After" to consistent hues.

-   **Time Series with Events**

    ---

    ![showcase_time_series](showcase_img/showcase_time_series.png)

    `fm.annotate_rect(x1="2023-06-01", x2="2023-09-01", y1=70, y2=125, fill="#fbbf24")` + `fm.Chart(df).mark_line().encode(x=fm.X("date", axis=fm.Axis(label_format="%b")))` + `fm.annotate_vline(x="2023-09-01")`

    Temporal columns auto-infer their scale. Annotation coordinates accept ISO date strings. The yellow band is `annotate_rect`; the rule is `annotate_vline`.

</div>

---

## Ranked and sorted

Sorted layouts for ranked comparisons — sorted along the axis, not alphabetically.

<div class="grid cards" markdown>

-   **Cleveland Dot Plot**

    ---

    ![showcase_cleveland_dot](showcase_img/showcase_cleveland_dot.png)

    `stem = fm.Chart(df).mark_rule(stroke="#d1d5db").encode(x=fm.X("score_zero", axis=fm.Axis(title="score")), x2=fm.X2("score"), y=fm.Y("skill", scale=ordinal))` + `fm.Chart(df).mark_point(fill="#3b82f6")`

    Lollipop composition: a `mark_rule` stem anchored to zero plus a `mark_point` dot. Category order is explicit via `OrdinalScale(domain=sorted_domain)`.

-   **Stacked Bar**

    ---

    ![showcase_stacked_bar](showcase_img/showcase_stacked_bar.png)

    `fm.Chart(df).mark_bar(position=fm.Stack()).encode(x="quarter", y="share", color=fm.Color("segment", scale=fm.OrdinalScale(domain=segments, range=colors)))`

    A four-color categorical range maps segment names to a monochromatic blue progression. `position=fm.Stack()` handles the cumulative offset.

</div>

---

## Matrix and field

Dense data structures — correlations, bivariate density — rendered as color-mapped fields.

<div class="grid cards" markdown>

-   **Correlation Heatmap**

    ---

    ![showcase_correlation_heatmap](showcase_img/showcase_correlation_heatmap.png)

    `fm.heatmap(df_wide, cmap="rdbu", center=0, annot=True)`

    The `fm.heatmap()` helper accepts wide-format DataFrames, unpivots them internally, and applies a diverging color scheme. `center=0` anchors the midpoint; values are annotated in each cell.

-   **Bivariate Density (Hexbin)**

    ---

    ![showcase_hexbin](showcase_img/showcase_hexbin.png)

    `fm.Chart(df).mark_hex().encode(x="x", y="y")`

    800 bivariate observations collapsed into hex bins. The count-based sequential colormap reveals the correlation structure without overplotting.

</div>

---

## Distributions

Shape-first views of one or more distributions.

<div class="grid cards" markdown>

-   **Grouped KDE Density**

    ---

    ![showcase_ridgeline](showcase_img/showcase_ridgeline.png)

    `fm.Chart(df).mark_density(groupby="group").encode(x="value", color=fm.Color("group"))`

    `groupby=` on `mark_density` computes a per-group KDE. Four conditions with different means and spreads are directly comparable as overlapping filled areas.

-   **Faceted Small Multiples**

    ---

    ![showcase_faceted_scatter](showcase_img/showcase_faceted_scatter.png)

    `fm.Chart(df).mark_point(opacity=0.65).encode(x="petal_length", y="sepal_width", color=fm.Color("species", scale=fm.OrdinalScale(domain=species_order, range=colors))).facet(col="species")`

    Three scatter panels sharing a common axis range. An explicit `OrdinalScale(domain=..., range=...)` keeps each species consistently colored across facet panels.

</div>
