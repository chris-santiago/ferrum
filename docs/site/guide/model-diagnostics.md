# Model diagnostics

This is where Ferrum's headline claim — *model outputs are data* — becomes specific code.

The diagnostic surface consists of three coordinated layers. You can mix and match them freely; they all return regular `Chart` objects (or compound views) that compose with the rest of the grammar.

| Layer | What it is | When to reach for it |
|---|---|---|
| **Figure-level helpers** | `roc_chart`, `calibration_chart`, `confusion_matrix_chart`, `shap_chart`, etc. | One-line entry points. Takes a fitted model + test data, returns a `Chart`. |
| **`ModelSource`** | The data interface | When you want to compute derived diagnostic data once and reuse it across multiple charts. |
| **sklearn-protocol visualizers** | `ROCVisualizer`, `CalibrationVisualizer`, `ConfusionMatrixVisualizer`, etc. | When you want lifecycle control (`.fit()` / `.score()` / `.show()`) or are following a yellowbrick-style pattern. |

The design rationale is on the [Model outputs are data](concepts/model-outputs-as-data.md) Concepts page; this page is the practical reference.

## Figure-level helpers

The fast path: pass a fitted model and held-out data, get a `Chart` back. The helpers cover the standard model-evaluation surface and dispatch the underlying transforms in Rust.

```python
import ferrum as fm
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, random_state=0)
model = RandomForestClassifier(n_estimators=20, random_state=0).fit(X_train, y_train)

roc = fm.roc_chart(model, X_test, y_test)
assert roc.show_svg().startswith("<svg")
```

The same pattern produces a confusion matrix:

```python
import ferrum as fm
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, random_state=0)
model = RandomForestClassifier(n_estimators=20, random_state=0).fit(X_train, y_train)

cm = fm.confusion_matrix_chart(model, X_test, y_test, normalize="true")
assert cm.show_svg().startswith("<svg")
```

Or feature importances, with the helper handling whichever importance method the estimator exposes (`feature_importances_`, permutation importance, coefficients):

```python
import ferrum as fm
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, random_state=0)
model = RandomForestClassifier(n_estimators=20, random_state=0).fit(X_train, y_train)

importances = fm.importance_chart(model, X_test, y_test)
assert importances.show_svg().startswith("<svg")
```

## The full helper menu

Every helper follows the same signature shape: `helper(model_or_source, X=None, y=None, **kwargs) -> Chart`. Pass a fitted model + held-out data, or pass a pre-constructed `ModelSource` as the first argument (next section). All helpers accept a `theme=` keyword.

| Family | Helpers |
|---|---|
| Classification | `roc_chart`, `pr_chart`, `calibration_chart`, `confusion_matrix_chart`, `class_prediction_error_chart`, `discrimination_threshold_chart`, `gain_chart`, `lift_chart` |
| Regression | `residuals_chart`, `prediction_error_chart` (via the regression visualizers), `cooks_distance_chart` |
| Feature explanation | `importance_chart`, `shap_chart`, `pdp_chart` |
| Model selection | `learning_curve_chart`, `validation_curve_chart`, `cv_scores_chart`, `alpha_selection_chart` |
| Clustering / manifold | `silhouette_chart`, `pca_scree_chart`, `intercluster_distance_chart`, `cluster_diagnostics`, `decision_boundary_chart` |

The full API surface is on the [API Reference / ferrum](../api/ferrum.md) page.

## `ModelSource`: derived diagnostic data

`ModelSource` wraps a fitted estimator and held-out data, then exposes derived diagnostic tables (predicted probabilities, ROC curve points, calibration bins, confusion counts, residuals, SHAP values, partial dependence grids, ...) as polars DataFrames.

When you call `roc_chart(model, X, y)`, the helper builds a `ModelSource` internally, asks it for the ROC curve points, and feeds those points to a chart spec. If you're computing multiple diagnostics on the same model + dataset, it's more efficient — and cleaner — to build the `ModelSource` once and pass it to each helper:

```python
import ferrum as fm
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, random_state=0)
model = RandomForestClassifier(n_estimators=20, random_state=0).fit(X_train, y_train)

source = fm.ModelSource(model, X_test, y_test)
roc = fm.roc_chart(source)
cm = fm.confusion_matrix_chart(source)
importances = fm.importance_chart(source)
report = (roc | cm) & importances
assert report.show_svg().startswith("<svg")
```

The `report` value is a regular composed chart — `(roc | cm) & importances` lays the three diagnostics into a 2 × 2 grid (with the importance chart spanning the bottom row), and you can save it, theme it, or further compose it as one artifact. The composition operators are the same `|` and `&` you use for any other charts (see [Composition](composition.md)).

### Why `ModelSource` matters

The boundary `ModelSource` enforces is the load-bearing one: it computes the derived diagnostic tables *once*, then every chart consumes the result. Without `ModelSource`, computing a ROC curve and a calibration curve on the same model would re-predict probabilities twice, and you'd have to thread that data plumbing through your own code.

`ModelSource` also lazy-imports sklearn, shap, and umap as needed: `import ferrum` does not pull those packages into your process. They load only when you actually compute a diagnostic that requires them.

## sklearn-protocol visualizers

For lifecycle control or yellowbrick-style ergonomics, every diagnostic also has a visualizer class. The visualizer takes the model at construction time, runs through `.fit()` / `.score()`, and exposes `.show()` which returns a `Chart`:

```python
import ferrum as fm
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, random_state=0)
model = RandomForestClassifier(n_estimators=20, random_state=0).fit(X_train, y_train)

visualizer = fm.ROCVisualizer(model)
visualizer.fit(X_train, y_train).score(X_test, y_test)
chart = visualizer.show()
assert chart.show_svg().startswith("<svg")
```

The full visualizer menu mirrors the helpers:

| Family | Visualizers |
|---|---|
| Classification | `ROCVisualizer`, `PRVisualizer`, `CalibrationVisualizer`, `ConfusionMatrixVisualizer`, `ClassificationReportVisualizer`, `ClassPredictionErrorVisualizer`, `ClassBalanceVisualizer`, `DiscriminationThresholdVisualizer` |
| Regression | `ResidualsVisualizer`, `PredictionErrorVisualizer`, `CooksDistanceVisualizer` |
| Explanation | `FeatureImportancesVisualizer`, `SHAPVisualizer` |
| Model selection | `LearningCurveVisualizer`, `ValidationCurveVisualizer`, `CVScoresVisualizer`, `AlphaSelectionVisualizer` |
| Clustering / manifold | `SilhouetteVisualizer`, `ElbowVisualizer`, `ManifoldVisualizer`, `InterclusterDistanceVisualizer`, `PCAVarianceVisualizer` |

Pick the helper when you want the diagnostic with minimal ceremony. Pick the visualizer when you want CV-fold lifecycle, custom training/scoring splits, or compatibility with code patterns from yellowbrick.

## Diagnostics compose with everything else

The most important property of these helpers is structural: their output is a regular Ferrum chart. That means a ROC curve participates in the rest of the grammar identically to a scatter plot:

- **Theme** it with `.theme(fm.themes.publication)` or set a process default with `set_default_theme`.
- **Save** it with `.save("roc.svg")`.
- **Concatenate** it with `|` or `&` (as shown above).
- **Pass** it through anywhere a `Chart` is expected.

A four-panel model report is `(roc | cm) & (residuals | importances)` — same composition operators as any other compound view. There is no separate API for "make these diagnostic charts work together."

## Caveats and limitations

A few sharp edges worth knowing:

- **`calibration_chart` rendering**: at the time of writing, `calibration_chart` builds the right `Chart` value but has a layering wiring gap that prevents `.show_svg()` from succeeding in the standard configuration. The chart object is well-formed; rendering will work once Phase 8a layer-data resolution is wired through for this helper. Other diagnostics on the same `ModelSource` are unaffected.
- **SHAP, UMAP, and shap-style helpers**: require their respective packages installed (`shap`, `umap-learn`). They lazy-import on first call; install the optional `ferrum[shap]` / `ferrum[umap]` extras to pull them in.
- **Per-class breakdowns**: classifier diagnostics default to a per-class view when the model has more than two classes. Pass `per_class=False` to collapse to a macro / micro / weighted average.
- **Compare multiple models**: most classification helpers accept a `compare=` keyword (or a `ComparedModelSource` data source) for side-by-side comparison. See the API reference for the per-helper signatures.

## Where to go next

- [Model outputs are data](concepts/model-outputs-as-data.md) for the design rationale behind treating diagnostics as charts.
- [Figure-level helpers](figure-helpers.md) for the broader family of one-line chart helpers (most diagnostic helpers follow the same pattern).
- [Composition](composition.md) for the operators (`+`, `|`, `&`) used to compose multiple diagnostics into a single model report.
- [Themes](themes.md) for applying consistent styling to a multi-chart diagnostic view.
- The [API Reference / ferrum](../api/ferrum.md) for the full signatures of every `*_chart` helper and every `*Visualizer` class.
