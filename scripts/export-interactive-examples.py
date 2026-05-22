"""Export each interactive example from docs/site/guide/interactive.md as HTML."""

import ferrum as fm
import polars as pl
from pathlib import Path

OUT = Path("interactive-exports")
OUT.mkdir(exist_ok=True)


# 1. Basic interactive — point chart with zoom/pan
df1 = pl.DataFrame({"x": [1, 2, 3, 4, 5], "y": [2, 4, 1, 5, 3]})
chart1 = fm.Chart(df1).mark_point().encode(x="x", y="y").properties(title="Basic Interactive")
chart1.interactive().save(str(OUT / "01_basic_interactive.html"))
print("1/8 basic interactive")

# 2. Point selection — click to select by group
df2 = pl.DataFrame({
    "x": [1, 2, 3, 4, 5],
    "y": [2, 4, 1, 5, 3],
    "group": ["a", "b", "a", "b", "a"],
})
sel2 = fm.selection_point()
chart2 = (
    fm.Chart(df2)
    .mark_point(size=100)
    .encode(x="x", y="y", color="group:N")
    .add_selection(sel2)
    .conditional(sel2.when(fm.Color("group")).otherwise(fm.value("#cccccc")))
    .properties(title="Single Point Selection")
    .interactive()
)
chart2.save(str(OUT / "02_point_selection.html"))
print("2/8 point selection")

# 3. Interval selection — brush to select
df3 = pl.DataFrame({
    "x": [1, 2, 3, 4, 5, 6, 7, 8],
    "y": [2, 4, 1, 5, 3, 6, 2, 4],
})
brush3 = fm.selection_interval()
chart3 = (
    fm.Chart(df3)
    .mark_point(size=60)
    .encode(x="x", y="y")
    .add_selection(brush3)
    .conditional(brush3.when(fm.value("#2563eb")).otherwise(fm.value("#cccccc")))
    .properties(title="Brush Selection")
    .interactive()
)
chart3.save(str(OUT / "03_interval_selection.html"))
print("3/8 interval selection")

# 4. Conditional encoding — color by species on click
df4 = pl.DataFrame({
    "x": [1, 2, 3, 4, 5],
    "y": [2, 4, 1, 5, 3],
    "species": ["setosa", "versicolor", "setosa", "versicolor", "setosa"],
})
sel4 = fm.selection_point(fields=["species"])
chart4 = (
    fm.Chart(df4)
    .mark_point(size=100)
    .encode(x="x", y="y", color="species:N")
    .add_selection(sel4)
    .conditional(sel4.when(fm.Color("species")).otherwise(fm.value("#cccccc")))
    .properties(title="Conditional Color")
    .interactive()
)
chart4.save(str(OUT / "04_conditional_color.html"))
print("4/8 conditional color")

# 5. Conditional encoding — opacity by group on click
df5 = pl.DataFrame({"x": [1, 2, 3], "y": [3, 1, 2], "g": ["a", "b", "a"]})
sel5 = fm.selection_point(fields=["g"])
chart5 = (
    fm.Chart(df5)
    .mark_point(size=100)
    .encode(x="x", y="y")
    .add_selection(sel5)
    .conditional(sel5.when(fm.Opacity("g")).otherwise(fm.value(0.2)))
    .interactive()
)
chart5.save(str(OUT / "05_conditional_opacity.html"))
print("5/8 conditional opacity")

# 6. Linked views — click-selection with field linking across panels
df6 = pl.DataFrame({
    "x": [1, 2, 3, 4, 5, 6, 7, 8],
    "y": [2, 4, 1, 5, 3, 6, 2, 4],
    "category": ["a", "b", "a", "b", "a", "b", "a", "b"],
})
sel6 = fm.selection_point(fields=["category"])
scatter6 = (
    fm.Chart(df6)
    .mark_point(size=80)
    .encode(x="x", y="y", color="category:N")
    .add_selection(sel6)
    .conditional(sel6.when(fm.Color("category")).otherwise(fm.value("#cccccc")))
    .properties(title="Scatter (click to select)")
)
bars6 = (
    fm.Chart(df6)
    .mark_bar()
    .transform(fm.transform_aggregate(
        {"field": "category", "fn": "count", "as": "n"}, groupby=["category"]
    ))
    .encode(x="category:N", y="n:Q", color="category:N")
    .add_selection(sel6)
    .conditional(sel6.when(fm.Color("category")).otherwise(fm.value("#cccccc")))
    .properties(title="Bar (linked)")
)
linked6 = scatter6 | bars6
linked6.interactive().save(str(OUT / "06_linked_views.html"))
print("6/8 linked views (HConcat)")

# 7. Brush selection with conditional color
df7 = pl.DataFrame({
    "x": list(range(20)),
    "y": [i * 0.5 + (i % 3) for i in range(20)],
    "cat": ["A", "B", "C", "D"] * 5,
})
brush7 = fm.selection_interval()
chart7 = (
    fm.Chart(df7)
    .mark_point(size=80)
    .encode(x="x:Q", y="y:Q", color="cat:N")
    .add_selection(brush7)
    .conditional(brush7.when(fm.Color("cat")).otherwise(fm.value("#cccccc")))
    .interactive()
)
chart7.save(str(OUT / "07_brush_conditional.html"))
print("7/8 brush with conditional color")

# 8. VConcat composition
df8 = pl.DataFrame({"x": [1, 2, 3, 4, 5], "y": [5, 3, 4, 2, 1], "g": ["a", "b", "a", "b", "a"]})
top8 = fm.Chart(df8).mark_point().encode(x="x:Q", y="y:Q", color="g:N").properties(title="Scatter")
bottom8 = fm.Chart(df8).mark_bar().encode(x="g:N", y="y:Q", color="g:N").properties(title="Bars")
vconcat8 = top8 & bottom8
vconcat8.interactive().save(str(OUT / "08_vconcat.html"))
print("8/8 VConcat composition")

# 9. Large scatter — 50k points to exercise R-tree, grid snap, zoom clip
import numpy as np
rng = np.random.default_rng(42)
n = 50_000
df9 = pl.DataFrame({"x": rng.normal(size=n), "y": rng.normal(size=n)})
brush9 = fm.selection_interval()
chart9 = (
    fm.Chart(df9)
    .mark_point(size=4, opacity=0.3)
    .encode(x="x:Q", y="y:Q")
    .add_selection(brush9)
    .conditional(brush9.when(fm.value("#2563eb")).otherwise(fm.value("#cccccc")))
    .properties(title="50k Point Cloud (zoom clip + grid snap test)")
    .interactive()
)
chart9.save(str(OUT / "09_large_scatter.html"))
print("9/9 large scatter (50k points)")

print(f"\nAll 9 HTML files saved to {OUT.resolve()}")
