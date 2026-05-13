# Gallery

A visual tour of Ferrum's chart surface. Every image below was rendered by Ferrum's Rust engine — no matplotlib, no browser, no external renderer.

## Primitive marks

The building blocks. Each mark maps directly to a geometric shape.

<div class="grid cards" markdown>

-   **Point**

    ---

    ![mark_point](img/mark_point.png)

-   **Line**

    ---

    ![mark_line](img/mark_line.png)

-   **Bar**

    ---

    ![mark_bar](img/mark_bar.png)

-   **Area**

    ---

    ![mark_area](img/mark_area.png)

-   **Rule**

    ---

    ![mark_rule](img/mark_rule.png)

-   **Tick**

    ---

    ![mark_tick](img/mark_tick.png)

-   **Rect**

    ---

    ![mark_rect](img/mark_rect.png)

-   **Text**

    ---

    ![mark_text](img/mark_text.png)

</div>

## Statistical marks

These marks compute a transform on your data before rendering — KDE, binning, smoothing, quantiles.

<div class="grid cards" markdown>

-   **Histogram**

    ---

    ![mark_histogram](img/mark_histogram.png)

-   **Density (KDE)**

    ---

    ![mark_density](img/mark_density.png)

-   **Smooth (linear)**

    ---

    ![mark_smooth_lm](img/mark_smooth_lm.png)

-   **Smooth (LOESS + CI)**

    ---

    ![mark_smooth_ci](img/mark_smooth_ci.png)

-   **Contour**

    ---

    ![mark_contour](img/mark_contour.png)

-   **QQ plot**

    ---

    ![mark_qq](img/mark_qq.png)

-   **Hex binning**

    ---

    ![mark_hex](img/mark_hex.png)

-   **Raster**

    ---

    ![mark_raster](img/mark_raster.png)

</div>

## Distribution marks

Summarize distributions across categories.

<div class="grid cards" markdown>

-   **Boxplot**

    ---

    ![mark_boxplot](img/mark_boxplot.png)

-   **Boxen (letter-value)**

    ---

    ![mark_boxen](img/mark_boxen.png)

-   **Violin**

    ---

    ![mark_violin](img/mark_violin.png)

-   **Swarm**

    ---

    ![mark_swarm](img/mark_swarm.png)

</div>

## Uncertainty marks

Show the spread around an estimate.

<div class="grid cards" markdown>

-   **Error bar**

    ---

    ![mark_errorbar](img/mark_errorbar.png)

-   **Ribbon**

    ---

    ![mark_ribbon](img/mark_ribbon.png)

</div>

## Composition

Multiple marks and charts combined into compound views.

<div class="grid cards" markdown>

-   **Scatter + smooth + CI**

    ---

    ![comp_scatter_smooth_ci](img/comp_scatter_smooth_ci.png)

-   **Histogram + density overlay**

    ---

    ![comp_histogram_density](img/comp_histogram_density.png)

-   **Line + ribbon**

    ---

    ![comp_line_ribbon](img/comp_line_ribbon.png)

-   **Boxplot + swarm**

    ---

    ![comp_boxplot_swarm](img/comp_boxplot_swarm.png)

-   **Dodged bar + error bar**

    ---

    ![comp_dodge_bar_errorbar](img/comp_dodge_bar_errorbar.png)

-   **Grouped smooth**

    ---

    ![comp_grouped_smooth](img/comp_grouped_smooth.png)

-   **Faceted smooth**

    ---

    ![comp_faceted_smooth](img/comp_faceted_smooth.png)

-   **Horizontal concat**

    ---

    ![comp_hconcat](img/comp_hconcat.png)

</div>

## Figure-level helpers

One-line entry points for common chart patterns.

<div class="grid cards" markdown>

-   **displot (histogram)**

    ---

    ![displot_hist](img/displot_hist.png)

-   **displot (KDE)**

    ---

    ![displot_kde](img/displot_kde.png)

-   **displot (ECDF)**

    ---

    ![displot_ecdf](img/displot_ecdf.png)

-   **catplot (box)**

    ---

    ![catplot_box](img/catplot_box.png)

-   **catplot (violin)**

    ---

    ![catplot_violin](img/catplot_violin.png)

-   **catplot (strip)**

    ---

    ![catplot_strip](img/catplot_strip.png)

-   **lmplot**

    ---

    ![lmplot](img/lmplot.png)

-   **residplot**

    ---

    ![residplot](img/residplot.png)

-   **pairplot**

    ---

    ![pairplot](img/pairplot.png)

-   **jointplot**

    ---

    ![jointplot](img/jointplot.png)

-   **heatmap**

    ---

    ![heatmap](img/heatmap.png)

-   **clustermap**

    ---

    ![clustermap](img/clustermap.png)

</div>

## Model diagnostics

Classification, regression, feature explanation, model selection, and clustering — all as charts.

<div class="grid cards" markdown>

-   **ROC curve**

    ---

    ![roc_chart](img/roc_chart.png)

-   **Precision-recall**

    ---

    ![pr_chart](img/pr_chart.png)

-   **Confusion matrix**

    ---

    ![confusion_matrix_chart](img/confusion_matrix_chart.png)

-   **Class prediction error**

    ---

    ![class_prediction_error](img/class_prediction_error.png)

-   **Discrimination threshold**

    ---

    ![discrimination_threshold](img/discrimination_threshold.png)

-   **Gain chart**

    ---

    ![gain_chart](img/gain_chart.png)

-   **Lift chart**

    ---

    ![lift_chart](img/lift_chart.png)

-   **Feature importance**

    ---

    ![importance_chart](img/importance_chart.png)

-   **SHAP beeswarm**

    ---

    ![shap_beeswarm_chart](img/shap_beeswarm_chart.png)

-   **SHAP bar**

    ---

    ![shap_bar_chart](img/shap_bar_chart.png)

-   **Partial dependence**

    ---

    ![pdp_chart](img/pdp_chart.png)

-   **Residuals**

    ---

    ![residuals_chart](img/residuals_chart.png)

-   **Prediction error**

    ---

    ![prediction_error_chart](img/prediction_error_chart.png)

-   **Learning curve**

    ---

    ![learning_curve_chart](img/learning_curve_chart.png)

-   **Validation curve**

    ---

    ![validation_curve_chart](img/validation_curve_chart.png)

-   **CV scores**

    ---

    ![cv_scores_chart](img/cv_scores_chart.png)

-   **Calibration**

    ---

    ![calibration_chart](img/calibration_chart.png)

-   **Alpha selection**

    ---

    ![alpha_selection_chart](img/alpha_selection_chart.png)

-   **Decision boundary**

    ---

    ![decision_boundary_chart](img/decision_boundary_chart.png)

-   **Silhouette / cluster diagnostics**

    ---

    ![cluster_diagnostics](img/cluster_diagnostics.png)

-   **Intercluster distance**

    ---

    ![intercluster_distance_chart](img/intercluster_distance_chart.png)

-   **PCA scree**

    ---

    ![pca_scree_chart](img/pca_scree_chart.png)

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
