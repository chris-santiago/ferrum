# Gallery

A visual tour of Ferrum's chart surface. Every image below was rendered by Ferrum's Rust engine — no matplotlib, no browser, no external renderer. Each card includes the key API call so you can reproduce it with your own data.

## Primitive marks

The building blocks. Each mark maps directly to a geometric shape.

<div class="grid cards" markdown>

-   **Point**

    ---

    ![mark_point](img/mark_point.png)

    `fm.Chart(df).mark_point().encode(x="x", y="y", color="class")`

-   **Line**

    ---

    ![mark_line](img/mark_line.png)

    `fm.Chart(df).mark_line().encode(x="month", y="value", color="group")`

-   **Bar**

    ---

    ![mark_bar](img/mark_bar.png)

    `fm.Chart(df).mark_bar(position=fm.Dodge()).encode(x="quarter", y="sales", color="region")`

-   **Area**

    ---

    ![mark_area](img/mark_area.png)

    `fm.Chart(df).mark_area(position=fm.Stack()).encode(x="x", y="value", color="series")`

-   **Rule**

    ---

    ![mark_rule](img/mark_rule.png)

    `fm.Chart(df).mark_point() + fm.Chart(rule_df).mark_rule().encode(y="ref_y")`

-   **Tick**

    ---

    ![mark_tick](img/mark_tick.png)

    `fm.Chart(df).mark_tick().encode(x="value", y="group")`

-   **Rect**

    ---

    ![mark_rect](img/mark_rect.png)

    `fm.Chart(df).mark_rect().encode(x="x", y="y", color="value")`

-   **Text**

    ---

    ![mark_text](img/mark_text.png)

    `fm.Chart(df).mark_text().encode(x="x", y="y", text="label")`

</div>

## Statistical marks

These marks compute a transform on your data before rendering — KDE, binning, smoothing, quantiles.

<div class="grid cards" markdown>

-   **Histogram**

    ---

    ![mark_histogram](img/mark_histogram.png)

    `fm.Chart(df).mark_histogram(groupby="group").encode(x="value", y="count", color="group")`

-   **Density (KDE)**

    ---

    ![mark_density](img/mark_density.png)

    `fm.Chart(df).mark_density(groupby="group").encode(x="value", color="group")`

-   **Smooth (linear)**

    ---

    ![mark_smooth_lm](img/mark_smooth_lm.png)

    `fm.Chart(df).mark_smooth(method="lm").encode(x="x", y="y")`

-   **Smooth (LOESS + CI)**

    ---

    ![mark_smooth_ci](img/mark_smooth_ci.png)

    `fm.Chart(df).mark_smooth(ci=0.95).encode(x="x", y="y")`

-   **Contour**

    ---

    ![mark_contour](img/mark_contour.png)

    `fm.Chart(df).mark_contour().encode(x="x", y="y")`

-   **QQ plot**

    ---

    ![mark_qq](img/mark_qq.png)

    `fm.Chart(df).mark_qq().encode(x="value")`

-   **Hex binning**

    ---

    ![mark_hex](img/mark_hex.png)

    `fm.Chart(df).mark_hex().encode(x="x", y="y")`

-   **Raster**

    ---

    ![mark_raster](img/mark_raster.png)

    `fm.Chart(df).mark_raster().encode(x="x", y="y")`

</div>

## Distribution marks

Summarize distributions across categories.

<div class="grid cards" markdown>

-   **Boxplot**

    ---

    ![mark_boxplot](img/mark_boxplot.png)

    `fm.Chart(df).mark_boxplot().encode(x="group", y="value")`

-   **Boxen (letter-value)**

    ---

    ![mark_boxen](img/mark_boxen.png)

    `fm.Chart(df).mark_boxen().encode(x="group", y="value")`

-   **Violin**

    ---

    ![mark_violin](img/mark_violin.png)

    `fm.Chart(df).mark_violin().encode(x="group", y="value")`

-   **Swarm**

    ---

    ![mark_swarm](img/mark_swarm.png)

    `fm.Chart(df).mark_swarm().encode(x="group", y="value")`

</div>

## Uncertainty marks

Show the spread around an estimate.

<div class="grid cards" markdown>

-   **Error bar**

    ---

    ![mark_errorbar](img/mark_errorbar.png)

    `fm.Chart(df).mark_errorbar(extent="stdev").encode(x="group", y="value")`

-   **Ribbon**

    ---

    ![mark_ribbon](img/mark_ribbon.png)

    `fm.Chart(df).mark_ribbon().encode(x="x", y="lower", y2="upper")`

</div>

## Composition

Multiple marks and charts combined into compound views.

<div class="grid cards" markdown>

-   **Scatter + smooth + CI**

    ---

    ![comp_scatter_smooth_ci](img/comp_scatter_smooth_ci.png)

    `fm.Chart(df).mark_point(opacity=0.3).mark_smooth(ci=0.95).encode(x="x", y="y")`

-   **Histogram + density overlay**

    ---

    ![comp_histogram_density](img/comp_histogram_density.png)

    `fm.Chart(df).mark_histogram() + fm.Chart(df).mark_density()`

-   **Line + ribbon**

    ---

    ![comp_line_ribbon](img/comp_line_ribbon.png)

    `fm.Chart(df).mark_ribbon().encode(x="x", y="lower", y2="upper") + fm.Chart(df).mark_line()`

-   **Boxplot + swarm**

    ---

    ![comp_boxplot_swarm](img/comp_boxplot_swarm.png)

    `fm.Chart(df).mark_boxplot() + fm.Chart(df).mark_swarm()`

-   **Dodged bar + error bar**

    ---

    ![comp_dodge_bar_errorbar](img/comp_dodge_bar_errorbar.png)

    `fm.Chart(df).mark_bar(position=fm.Dodge()) + fm.Chart(df).mark_errorbar()`

-   **Grouped smooth**

    ---

    ![comp_grouped_smooth](img/comp_grouped_smooth.png)

    `fm.Chart(df).mark_smooth(method="lm", groupby="group").encode(x="x", y="y", color="group")`

-   **Faceted smooth**

    ---

    ![comp_faceted_smooth](img/comp_faceted_smooth.png)

    `fm.Chart(df).mark_point().mark_smooth(groupby="dose").encode(...).facet(col="dose")`

-   **Horizontal concat**

    ---

    ![comp_hconcat](img/comp_hconcat.png)

    `fm.hconcat(chart_hex, chart_violin)`

</div>

## Figure-level helpers

One-line entry points for common chart patterns.

<div class="grid cards" markdown>

-   **displot (histogram)**

    ---

    ![displot_hist](img/displot_hist.png)

    `fm.displot(df, x="value", hue="group", kind="hist")`

-   **displot (KDE)**

    ---

    ![displot_kde](img/displot_kde.png)

    `fm.displot(df, x="value", hue="group", kind="kde")`

-   **displot (ECDF)**

    ---

    ![displot_ecdf](img/displot_ecdf.png)

    `fm.displot(df, x="value", kind="ecdf")`

-   **catplot (box)**

    ---

    ![catplot_box](img/catplot_box.png)

    `fm.catplot(df, x="group", y="value", kind="box")`

-   **catplot (violin)**

    ---

    ![catplot_violin](img/catplot_violin.png)

    `fm.catplot(df, x="group", y="value", kind="violin")`

-   **catplot (strip)**

    ---

    ![catplot_strip](img/catplot_strip.png)

    `fm.catplot(df, x="group", y="value", kind="strip")`

-   **lmplot**

    ---

    ![lmplot](img/lmplot.png)

    `fm.lmplot(df, x="x", y="y", hue="group")`

-   **residplot**

    ---

    ![residplot](img/residplot.png)

    `fm.residplot(df, x="x", y="y")`

-   **pairplot**

    ---

    ![pairplot](img/pairplot.png)

    `fm.pairplot(df, vars=[...], hue="species")`

-   **jointplot**

    ---

    ![jointplot](img/jointplot.png)

    `fm.jointplot(df, x="sepal_length", y="sepal_width")`

-   **heatmap**

    ---

    ![heatmap](img/heatmap.png)

    `fm.heatmap(df_corr, center=0, annot=True)`

-   **clustermap**

    ---

    ![clustermap](img/clustermap.png)

    `fm.clustermap(df_cluster)`

</div>

## Model diagnostics

Classification, regression, feature explanation, model selection, and clustering — all as charts.

<div class="grid cards" markdown>

-   **ROC curve**

    ---

    ![roc_chart](img/roc_chart.png)

    `fm.roc_chart(model, X_test, y_test)`

-   **Precision-recall**

    ---

    ![pr_chart](img/pr_chart.png)

    `fm.pr_chart(model, X_test, y_test)`

-   **Confusion matrix**

    ---

    ![confusion_matrix_chart](img/confusion_matrix_chart.png)

    `fm.confusion_matrix_chart(model, X_test, y_test)`

-   **Class prediction error**

    ---

    ![class_prediction_error](img/class_prediction_error.png)

    `fm.class_prediction_error_chart(model, X_test, y_test)`

-   **Discrimination threshold**

    ---

    ![discrimination_threshold](img/discrimination_threshold.png)

    `fm.discrimination_threshold_chart(model, X_test, y_test)`

-   **Gain chart**

    ---

    ![gain_chart](img/gain_chart.png)

    `fm.gain_chart(model, X_test, y_test)`

-   **Lift chart**

    ---

    ![lift_chart](img/lift_chart.png)

    `fm.lift_chart(model, X_test, y_test)`

-   **Feature importance**

    ---

    ![importance_chart](img/importance_chart.png)

    `fm.importance_chart(model, X_test, y_test)`

-   **SHAP beeswarm**

    ---

    ![shap_beeswarm_chart](img/shap_beeswarm_chart.png)

    `fm.shap_beeswarm_chart(model, X, y)`

-   **SHAP bar**

    ---

    ![shap_bar_chart](img/shap_bar_chart.png)

    `fm.shap_bar_chart(model, X, y)`

-   **Partial dependence**

    ---

    ![pdp_chart](img/pdp_chart.png)

    `fm.pdp_chart(model, X, y, features=[0, 1])`

-   **Residuals**

    ---

    ![residuals_chart](img/residuals_chart.png)

    `fm.residuals_chart(model, X, y)`

-   **Prediction error**

    ---

    ![prediction_error_chart](img/prediction_error_chart.png)

    `fm.Chart(source.predictions()).mark_prediction_error()`

-   **Learning curve**

    ---

    ![learning_curve_chart](img/learning_curve_chart.png)

    `fm.learning_curve_chart(model, X, y, cv=3)`

-   **Validation curve**

    ---

    ![validation_curve_chart](img/validation_curve_chart.png)

    `fm.validation_curve_chart(Ridge(), X, y, param="alpha", values=[...])`

-   **CV scores**

    ---

    ![cv_scores_chart](img/cv_scores_chart.png)

    `fm.cv_scores_chart(model, X, y, cv=5)`

-   **Calibration**

    ---

    ![calibration_chart](img/calibration_chart.png)

    `fm.calibration_chart(model, X, y)`

-   **Alpha selection**

    ---

    ![alpha_selection_chart](img/alpha_selection_chart.png)

    `fm.alpha_selection_chart(Ridge(), X, y, alphas=[...])`

-   **Decision boundary**

    ---

    ![decision_boundary_chart](img/decision_boundary_chart.png)

    `fm.decision_boundary_chart(model, X, y, features=(0, 1))`

-   **Silhouette / cluster diagnostics**

    ---

    ![cluster_diagnostics](img/cluster_diagnostics.png)

    `fm.cluster_diagnostics(X, ks=range(2, 7))`

-   **Intercluster distance**

    ---

    ![intercluster_distance_chart](img/intercluster_distance_chart.png)

    `fm.intercluster_distance_chart(km, X)`

-   **PCA scree**

    ---

    ![pca_scree_chart](img/pca_scree_chart.png)

    `fm.pca_scree_chart(pca, X)`

</div>

## Ferrum theme identities

Three original theme identities ship with Ferrum. Each pairs a background, typography, mark palette, and color schemes into a cohesive look.

### Paper Ink (default)

Warm cream background, blue lead marks, perceptually balanced categorical cycle.

<div class="grid cards" markdown>

-   ![scatter_paper_ink](img/scatter_paper_ink.png)

-   ![bar_paper_ink](img/bar_paper_ink.png)

-   ![heatmap_paper_ink](img/heatmap_paper_ink.png)

</div>

### Slate Citrus

Dark navy background, vibrant neon accents, lime/cyan categorical cycle.

<div class="grid cards" markdown>

-   ![scatter_slate_citrus](img/scatter_slate_citrus.png)

-   ![bar_slate_citrus](img/bar_slate_citrus.png)

-   ![heatmap_slate_citrus](img/heatmap_slate_citrus.png)

</div>

### Arctic Signal

Cool white background, sky blue lead mark, precise signal palette.

<div class="grid cards" markdown>

-   ![scatter_arctic_signal](img/scatter_arctic_signal.png)

-   ![bar_arctic_signal](img/bar_arctic_signal.png)

-   ![heatmap_arctic_signal](img/heatmap_arctic_signal.png)

</div>

See the full [Themes guide](../guide/themes.md) for the complete list of twelve built-in themes and how to build your own.
