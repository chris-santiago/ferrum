"""No public call reaches an internal rebuild invariant in ``_compose.py``.

A composite mark, and a figure built on one, can record their statistics
before the grouping that should shape them is known (``.encode()`` applied
after the chart was built, a ``+`` composition's or outer composition's
grouping). ``ferrum._compose`` rebuilds them under the grouping it finds,
and guards the shapes that rebuild must hold (same operand count, same
mark, same layer names) with ``RuntimeError`` -- the project's
internal-invariant type, never a typed user refusal, because no public
input names a remedy for it.

This module is the population proof that those guards stay unreachable.
Every ``ferrum.plots`` figure that takes ``encode=``, and every ``mark_*``
method that routes through ``Chart._set_composite_mark`` (the statistical
mixin and the six diagnostic domain mixins under
``ferrum.marks._chart_mixins`` alike -- ``mark_function`` does not: it
builds via plain ``self._clone()``/``self + fn_chart`` and never records a
rebuild source, so it is not in this population and needs no case), is
built once with no grouping and then regrouped through every door that can
reach the rebuild machinery: its own ``.encode(color=)`` and
``.encode(detail=)``, ``+`` with an unrelated point chart in both orders,
and a further ``+`` that nests the first composition inside a second one
before encoding it. A handful of figures (``jointplot``, ``pairplot``,
``clustermap``) return a grid composite with no ``.encode()`` or ``+`` of
its own (``_ChartOperatorMixin`` defines neither), so only their own
``encode=`` door applies -- passed as a constructor keyword, which fans
into each child panel the same way a chained ``.encode()`` would on a
plain ``Chart``.

A build alone does not reach the rebuild: a chart's pending mark resolves
once with no grouping, records that build, and returns -- it never calls
``regrouped()`` on that same pass. Only a *second* resolution -- after
``.encode()`` adds a grouping the first resolution never had -- rebuilds
against the recorded source and runs the guarded functions. So this sweep
has two kinds of own-encode door, both real and both kept: "own-color"/
"own-detail" is the user's literal raw chain, ``build().encode(color=)``
straight onto a chart whose composite mark may still be pending -- for a
chart still pending at that point, that chain never reaches the rebuild
at all (a single resolution bakes the grouping in from the start), and
this door pins that the raw chain still renders regardless.
"layer-color"/"layer-detail" reach the second-resolution shape the
rebuild guards are actually checked against through a *public* route:
``Chart.layer()`` (``_attach_host``) resolves a chart with no layers of
its own exactly the way attaching a real layer would, so
``build().layer().encode(color=...)`` reaches ``regrouped()`` the same
way a caller's ``.layer(fm.Layer(...)).encode(color=...)`` does, with no
private call. ``+`` forces its own two-sided resolution the same way (its
``__add__`` resolves both operands before merging), so "plus-left"/
"plus-right"/"nested" reach it too, composing a fresh, unresolved
``build()`` each time -- ``.layer()`` is used beside them only to read
each chart's own established fields when picking the point partner's
axis and the ``+`` grouping field, the same shape a caller could read by
rendering the chart once.

Both populations are derived from the package, never listed, with a
completeness check (one shared helper, driven by both the real test and
its control) that a new figure or mark without a build case fails by
name. A composite mark whose kwargs change its layer topology (filled
density, filled contour, a confidence-interval smooth) gets an additional
build *beside* its default one, never replacing it, both still swept and
both still counted under the same mark name for completeness. Every case
either renders (a non-empty SVG) or raises one of the rebuild-invariant
``RuntimeError``\\ s this module pins unreachable -- there is no other
expected refusal once the fields below are chosen to fit each case's own
data, so no case is listed against one.
"""

from __future__ import annotations

import ast
import functools
import importlib
import inspect
import pkgutil
from typing import Any, Callable

import numpy as np
import polars as pl
import pyarrow as pa
import pytest
from sklearn.cluster import KMeans
from sklearn.datasets import (
    load_diabetes,
    load_iris,
    load_wine,
    make_blobs,
    make_classification,
)
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge

import ferrum as fm
import ferrum._compose
import ferrum.plots
from ferrum.marks._chart_mixins import (
    ClassificationMarksMixin,
    ClusteringMarksMixin,
    ExplanationMarksMixin,
    RankingMarksMixin,
    RegressionMarksMixin,
    SelectionMarksMixin,
)
from ferrum.marks._chart_methods_statistical import StatisticalMarksMixin

_DOORS = (
    "own-color",
    "own-detail",
    "layer-color",
    "layer-detail",
    "plus-left",
    "plus-right",
    "nested",
)

# ── Lazy, cached fixtures ────────────────────────────────────────────────
#
# Every data/model fixture below builds on first use (inside a test body),
# never at import or collection: the whole point of a population sweep is
# that a single broken figure fails by its own name, not by turning
# collection of the entire module into one undifferentiated error.


@functools.lru_cache(maxsize=None)
def _df() -> pl.DataFrame:
    rng = np.random.default_rng(0)
    n = 150
    return pl.DataFrame(
        {
            "v": rng.normal(size=n),
            "w": rng.normal(size=n),
            "g": ["a", "b", "c"] * (n // 3),
            "h": ["p", "q"] * (n // 2),
        }
    )


@functools.lru_cache(maxsize=None)
def _multiclass() -> tuple[Any, Any, Any]:
    X, y = make_classification(
        n_samples=200, n_features=8, n_classes=3, n_informative=5, random_state=0
    )
    return LogisticRegression(max_iter=500).fit(X, y), X, y


@functools.lru_cache(maxsize=None)
def _binary() -> tuple[Any, Any, Any]:
    X, y = make_classification(n_samples=200, n_features=8, n_classes=2, random_state=0)
    return LogisticRegression(max_iter=500).fit(X, y), X, y


@functools.lru_cache(maxsize=None)
def _random_forest() -> tuple[Any, Any, Any]:
    clf, X, y = _multiclass()
    return RandomForestClassifier(n_estimators=20, random_state=0).fit(X, y), X, y


@functools.lru_cache(maxsize=None)
def _regression() -> tuple[Any, Any, Any]:
    X, y = load_diabetes(return_X_y=True)
    return LinearRegression().fit(X, y), X, y


@functools.lru_cache(maxsize=None)
def _blobs() -> Any:
    X, _ = make_blobs(n_samples=200, centers=4, random_state=0)
    return X


@functools.lru_cache(maxsize=None)
def _iris() -> tuple[Any, Any]:
    return load_iris(return_X_y=True)


@functools.lru_cache(maxsize=None)
def _wine() -> Any:
    X, _ = load_wine(return_X_y=True)
    return X


@functools.lru_cache(maxsize=None)
def _iris_wide() -> pl.DataFrame:
    frame = load_iris(as_frame=True).frame
    return pl.from_pandas(frame).rename({c: c.replace(" ", "_") for c in frame.columns})


@functools.lru_cache(maxsize=None)
def _iris_species() -> pl.DataFrame:
    _, y = _iris()
    return _iris_wide().with_columns(pl.Series("species", [str(v) for v in y]))


@functools.lru_cache(maxsize=None)
def _figure_data(name: str) -> Any:
    """The resolved chart's own data for figure *name* -- the schema a
    diagnostic ``mark_*`` built on the same domain expects.

    ``.layer()`` (no layers) is the public resolve-only route (see the
    module docstring); this is plain fixture setup, not a door being
    swept, so it reads through the same public call the sweep itself
    uses rather than inventing a second way to resolve a chart.
    ``chart_frames(...)[0]`` is the host operand's own frame -- a
    composed chart's ``._data`` is the composition's internal, diagonally
    concatenated combination of every operand's rows (kept for shared-scale
    resolution, never read by a real render), so reading it here would
    hand a fresh chart rows no caller's own data ever produces.
    """
    from ferrum._compose import chart_frames

    return chart_frames(_FIGURE_BUILDS[name]().layer())[0]


# ── Population discovery (mirrors tests/test_public_entry_voice.py) ────────


def _plots_functions() -> list[tuple[str, Callable]]:
    """Every public function defined in a ``ferrum.plots`` submodule."""
    found = []
    for info in pkgutil.iter_modules(ferrum.plots.__path__):
        if info.name.startswith("_"):
            continue
        module = importlib.import_module(f"ferrum.plots.{info.name}")
        for name, obj in vars(module).items():
            if name.startswith("_") or not inspect.isfunction(obj):
                continue
            if obj.__module__ == module.__name__:
                found.append((f"{info.name}.{name}", obj))
    return sorted(found)


def _figure_population() -> set[str]:
    """Every public figure function that takes ``encode=``."""
    return {
        fn.__name__ for _, fn in _plots_functions() if "encode" in inspect.signature(fn).parameters
    }


#: The statistical mixin plus the six per-domain diagnostic mixins
#: (``ferrum.marks._chart_methods_diagnostic.DiagnosticMarksMixin``'s own
#: bases) -- together, every mixin a ``mark_*`` method reaching
#: ``Chart._set_composite_mark`` is declared on.
_COMPOSITE_MARK_MIXINS = (
    StatisticalMarksMixin,
    ClassificationMarksMixin,
    ClusteringMarksMixin,
    ExplanationMarksMixin,
    RankingMarksMixin,
    RegressionMarksMixin,
    SelectionMarksMixin,
)


def _composite_mark_population() -> set[str]:
    """Every ``mark_*`` method that resolves through ``Chart._set_composite_mark``.

    Reads each mixin method's own source rather than trusting its name:
    ``mark_function`` lives on ``StatisticalMarksMixin`` but builds via
    ``self._clone()``/``self + fn_chart``, never the scaffold, so it is
    correctly excluded.
    """
    return {
        name
        for mixin in _COMPOSITE_MARK_MIXINS
        for name, method in vars(mixin).items()
        if name.startswith("mark_") and "_set_composite_mark(" in inspect.getsource(method)
    }


def _missing_build_cases(population: set[str], covered: set[str]) -> list[str]:
    """Sorted *population* names absent from *covered*.

    The one completeness check both a real population test and its
    control run, so the control actually exercises the logic the real
    test depends on rather than re-implementing it over literals.
    """
    return sorted(population - covered)


# ── Column and axis-field helpers ───────────────────────────────────────────


def _as_polars(data: Any) -> pl.DataFrame:
    return pl.from_arrow(data) if isinstance(data, pa.Table) else data


def _cols_for(chart: Any) -> tuple[str, str, str]:
    """A numeric ``x``/``y`` pair and a grouping field, read off *chart*'s own data.

    Prefers numeric columns for the point overlay's position (so its axis
    type never clashes with the chart's own), and a string column for
    grouping when one exists. Used only as a fallback when the chart's own
    established ``x``/``y`` field is not literally present in its data (a
    composite's statistic output, e.g. a histogram's ``bin_start``).
    """
    data = _as_polars(chart._data)
    visible = [c for c in data.columns if not c.startswith("_")]
    numeric = [c for c in visible if data.schema[c].is_numeric()]
    strings = [c for c in visible if data.schema[c] == pl.Utf8]
    pool = numeric or visible
    x = pool[0]
    y = pool[1] if len(pool) > 1 else x
    group = strings[0] if strings else pool[-1]
    return x, y, group


def _established_field(chart: Any, channel: str) -> str | None:
    """The literal data column *channel* is bound to: the chart-level
    encoding, else layer 0's own, when that field is actually a column of
    *chart*'s data; ``None`` when neither is (a statistic's own output,
    e.g. a histogram's ``count`` or a heatmap's unpivoted ``column``)."""
    columns = set(_as_polars(chart._data).columns)
    candidates = [chart._encoding.get(channel)]
    if chart._layers:
        candidates.append(chart._layers[0].encoding.get(channel))
    for value in candidates:
        if value is None:
            continue
        field = value.field if hasattr(value, "field") else value
        if isinstance(field, str) and field in columns:
            return field
    return None


def _axes_for(chart: Any) -> tuple[str, str]:
    """The ``x``/``y`` fields the point-chart partner should share with *chart*.

    Reuses *chart*'s own established fields so the partner's axis type
    (categorical or continuous) can never clash with *chart*'s; falls back
    to a numeric pair off its data when *chart* establishes no literal
    column for one (a 1-D statistic mark binds only ``x``).
    """
    x = _established_field(chart, "x")
    y = _established_field(chart, "y")
    if x is None or y is None:
        gx, gy, _ = _cols_for(chart)
        x = x or gx
        y = y or gy
    return x, y


#: Per-entry override of the field ``.encode(color=)``/``.encode(detail=)``
#: binds (first) and the field the ``+`` doors bind on the composed chart
#: (second), when the generic string-column choice off the chart's own
#: data would reach a sub-layer that does not carry it (an operand or a
#: named secondary output with a disjoint schema from the chart's main
#: rows). ``None`` for an entry not listed here: the own-encode and
#: ``+``-door fields are the same, generically derived one.
_GROUP_OVERRIDES: dict[str, tuple[str, str]] = {
    # hex colors its cells by its own aggregate output ("value"), never a
    # literal input column (it refuses one -- "color=<col> is drawn
    # nowhere"); the `+` doors still need a real column both operands share.
    "mark_hex": ("value", "h"),
    # calibration's annotation layers (reference line, Brier label) don't
    # carry 'count'/'brier'; the established x ('mean_predicted') reaches
    # every layer, including those.
    "calibration_chart": ("mean_predicted", "mean_predicted"),
    "mark_calibration": ("mean_predicted", "mean_predicted"),
}

#: Per-entry declared type for the point partner's ``x``/``y``, when the
#: established axis is categorical (ordinal row/column labels) but every
#: column available for the partner's position in the chart's own data is
#: numeric (heatmap's row/column axes are its Unpivot transform's output,
#: never literal columns of its own wide input data).
_AXIS_TYPE_OVERRIDES: dict[str, str] = {
    "heatmap": "N",
}


def _render_door(build: Callable[[], Any], door: str, name: str) -> str:
    """Render a fresh ``build()`` chart through *door* (see the module
    docstring for what each door reaches and why)."""
    enc_group, plus_group = _GROUP_OVERRIDES.get(name, (None, None))

    if door in ("own-color", "own-detail"):
        # The user's literal raw chain: no resolution before .encode(), so
        # a composite mark still pending bakes the grouping into its one
        # and only resolution rather than reaching regrouped().
        chart = build()
        group = enc_group if enc_group is not None else _cols_for(chart)[2]
        if door == "own-color":
            return chart.encode(color=fm.Color(group)).to_svg()
        return chart.encode(detail=fm.Detail(group)).to_svg()

    # Every remaining door needs the resolved-with-no-grouping shape the
    # rebuild guards are checked against. `.layer()` with no layers is the
    # public route to it (`_attach_host`), the same mechanism `Chart.layer()`
    # and `+` use on their own operands.
    resolved = build().layer()
    if enc_group is None:
        _, _, enc_group = _cols_for(resolved)
    if plus_group is None:
        plus_group = enc_group

    if door in ("layer-color", "layer-detail"):
        if door == "layer-color":
            return resolved.encode(color=fm.Color(enc_group)).to_svg()
        return resolved.encode(detail=fm.Detail(enc_group)).to_svg()

    x, y = _axes_for(resolved)
    declared_type = _AXIS_TYPE_OVERRIDES.get(name)
    x_channel = fm.X(x, type=declared_type) if declared_type else fm.X(x)
    y_channel = fm.Y(y, type=declared_type) if declared_type else fm.Y(y)
    # `chart_frames(resolved)[0]` is the host operand's own frame, never
    # the composition's internal, diagonally concatenated combination of
    # every operand's rows (see `_figure_data`'s own docstring above).
    from ferrum._compose import chart_frames

    pt = fm.Chart(chart_frames(resolved)[0]).mark_point().encode(x=x_channel, y=y_channel)
    # The `+` doors compose a fresh, unresolved build(): `+` resolves both
    # of its operands itself (chart.py's __add__), so this is the public
    # route end to end, not a second pre-resolved chart feeding `+`.
    chart = build()
    if door == "plus-left":
        return (chart + pt).encode(color=fm.Color(plus_group)).to_svg()
    if door == "plus-right":
        return (pt + chart).encode(color=fm.Color(plus_group)).to_svg()
    if door == "nested":
        return ((chart + pt) + pt).encode(color=fm.Color(plus_group)).to_svg()
    raise AssertionError(f"unknown door {door!r}")


def _assert_renders(build: Callable[[], Any], door: str, name: str) -> None:
    svg = _render_door(build, door, name)
    assert isinstance(svg, str) and "<svg" in svg


# ── Figure build cases ───────────────────────────────────────────────────────
#
# One build per figure in the population, with no grouping of its own, so
# every door above adds the first grouping binding.

_FIGURE_BUILDS: dict[str, Callable[[], Any]] = {
    "roc_chart": lambda: fm.roc_chart(*_multiclass()),
    "pr_chart": lambda: fm.pr_chart(*_multiclass()),
    "calibration_chart": lambda: fm.calibration_chart(*_binary()),
    "gain_chart": lambda: fm.gain_chart(*_multiclass()),
    "lift_chart": lambda: fm.lift_chart(*_multiclass()),
    "discrimination_threshold_chart": lambda: fm.discrimination_threshold_chart(*_binary()),
    "confusion_matrix_chart": lambda: fm.confusion_matrix_chart(*_multiclass()),
    "class_prediction_error_chart": lambda: fm.class_prediction_error_chart(*_multiclass()),
    "classification_report_chart": lambda: fm.classification_report_chart(*_multiclass()),
    "class_balance_chart": lambda: fm.class_balance_chart(_multiclass()[2]),
    "cluster_diagnostics": lambda: fm.cluster_diagnostics(
        _blobs(), ks=list(range(2, 6)), scoring="elbow"
    ),
    "elbow_chart": lambda: fm.elbow_chart(KMeans, _blobs(), ks=list(range(2, 6))),
    "intercluster_distance_chart": lambda: fm.intercluster_distance_chart(
        KMeans(n_clusters=4, n_init=10, random_state=0).fit(_blobs()), _blobs()
    ),
    "manifold_chart": lambda: fm.manifold_chart(
        KMeans(n_clusters=3, n_init=10, random_state=0).fit(_iris()[0]), _iris()[0], method="pca"
    ),
    "pca_scree_chart": lambda: fm.pca_scree_chart(
        PCA(n_components=4, random_state=0).fit(_wine()), _wine()
    ),
    "silhouette_chart": lambda: fm.silhouette_chart(
        KMeans(n_clusters=4, n_init=10, random_state=0).fit(_blobs()), _blobs()
    ),
    "importance_chart": lambda: fm.importance_chart(*_random_forest()),
    "pdp_chart": lambda: fm.pdp_chart(*_random_forest(), features=[0, 1]),
    "shap_bar_chart": lambda: fm.shap_bar_chart(_random_forest()[0], _random_forest()[1]),
    "shap_beeswarm_chart": lambda: fm.shap_beeswarm_chart(_random_forest()[0], _random_forest()[1]),
    "shap_chart": lambda: fm.shap_chart(_random_forest()[0], _random_forest()[1]),
    "shap_waterfall_chart": lambda: fm.shap_waterfall_chart(
        _random_forest()[0], _random_forest()[1], sample_idx=0
    ),
    "heatmap": lambda: fm.heatmap(_iris_wide().drop("target")),
    "alpha_selection_chart": lambda: fm.alpha_selection_chart(
        Ridge(), _regression()[1], _regression()[2], alphas=[0.1, 1.0, 10.0]
    ),
    "cv_scores_chart": lambda: fm.cv_scores_chart(Ridge(), _regression()[1], _regression()[2]),
    "learning_curve_chart": lambda: fm.learning_curve_chart(
        Ridge(), _regression()[1], _regression()[2]
    ),
    "validation_curve_chart": lambda: fm.validation_curve_chart(
        Ridge(), _regression()[1], _regression()[2], param="alpha", values=[0.1, 1.0, 10.0]
    ),
    "decision_boundary_chart": lambda: fm.decision_boundary_chart(
        LogisticRegression(max_iter=500).fit(_iris()[0][:, :2], _iris()[1]),
        _iris()[0][:, :2],
        _iris()[1],
    ),
    "parallel_coordinates_chart": lambda: fm.parallel_coordinates_chart(
        _iris_species(), hue="species"
    ),
    "rank1d_chart": lambda: fm.rank1d_chart(_regression()[1], _regression()[1], _regression()[2]),
    "rank2d_chart": lambda: fm.rank2d_chart(_regression()[1], _regression()[1], _regression()[2]),
    "rank_chart": lambda: fm.rank_chart(_regression()[1], _regression()[1], _regression()[2]),
    "cooks_distance_chart": lambda: fm.cooks_distance_chart(*_regression()),
    "lmplot": lambda: fm.lmplot(_df(), x="v", y="w"),
    "prediction_error_chart": lambda: fm.prediction_error_chart(*_regression()),
    "regplot": lambda: fm.regplot(_df(), x="v", y="w"),
    "residplot": lambda: fm.residplot(_df(), x="v", y="w"),
    "residuals_chart": lambda: fm.residuals_chart(*_regression(), panels="single"),
    "catplot": lambda: fm.catplot(_df(), x="g", y="v"),
    "displot": lambda: fm.displot(_df(), x="v"),
    "relplot": lambda: fm.relplot(_df(), x="v", y="w"),
}

# ``jointplot``, ``pairplot`` and ``clustermap`` return a grid composite
# (``JointChart`` / ``RepeatChart`` / ``ClusterMapChart``) with no
# ``.encode()`` or ``+`` of its own, so only the figure's own ``encode=``
# door applies, passed at construction (fanning into each child panel).
_COMPOSITE_ONLY_FIGURE_BUILDS: dict[str, Callable[[dict], Any]] = {
    "jointplot": lambda encode: fm.jointplot(_df(), x="v", y="w", encode=encode),
    "pairplot": lambda encode: fm.pairplot(
        _iris_species(), vars=["sepal_length_(cm)", "sepal_width_(cm)"], encode=encode
    ),
    "clustermap": lambda encode: fm.clustermap(_iris_species().drop("target"), encode=encode),
}
#: (color field, detail field) each composite-only figure's own data carries.
_COMPOSITE_ONLY_GROUPING_FIELDS: dict[str, tuple[str, str]] = {
    "jointplot": ("g", "h"),
    "pairplot": ("species", "species"),
    # clustermap picks its row-label column (id_col) as the one non-numeric
    # column of its input (`_matrix_columns`); giving it the user's own
    # "species" label column (instead of letting every column be numeric,
    # which falls back to a synthetic row index) means the center heatmap
    # panel's own row axis is literally "species", a column the caller
    # wrote. The dendrogram panels' own segments don't carry it, so it
    # warns-and-drops there rather than refusing.
    "clustermap": ("species", "species"),
}


def _c() -> Any:
    return fm.Chart(_df())


# ── Composite mark build cases (statistical + diagnostic) ──────────────────
#
# One or more builds per mark in _composite_mark_population(), keyed by
# mark name (completeness is checked against that outer key, never the
# variant). A mark whose kwargs change its layer topology gets a "default"
# build plus a named variant *beside* it -- never replacing it -- so both
# shapes stay swept. The diagnostic marks reuse the resolved data of the
# figure built on the same domain (the exact schema ``ModelSource``/the
# figure's own transform produces), rather than re-deriving it, per the
# figure -> mark pairing read out of each figure's own source (e.g.
# ``ferrum.Chart(df).mark_roc(...)`` in
# ``src/ferrum/plots/classification.py``).

_MARK_BUILD_VARIANTS: dict[str, dict[str, Callable[[], Any]]] = {
    "mark_density": {
        "default": lambda: _c().mark_density().encode(x="v"),
        # Both x and y bound routes through desugar_contour(fill=True),
        # the mark's own documented bivariate path; exercises _paired's
        # added/unmatched-pairing branch (its one layer renames under
        # regroup, "polygon" -> "segment").
        "filled_2d": lambda: _c().mark_density(fill=True).encode(x="v", y="w"),
    },
    "mark_histogram": {"default": lambda: _c().mark_histogram().encode(x="v")},
    "mark_smooth": {
        "default": lambda: _c().mark_smooth().encode(x="v", y="w"),
        # A confidence band gives _paired/_redrawn_layers a two-layer
        # ("ribbon", "line") matched-name case instead of the flat
        # single-mark shape.
        "ci": lambda: _c().mark_smooth(ci=0.95).encode(x="v", y="w"),
    },
    "mark_boxplot": {"default": lambda: _c().mark_boxplot().encode(x="g", y="v")},
    "mark_boxen": {"default": lambda: _c().mark_boxen().encode(x="g", y="v")},
    "mark_errorbar": {"default": lambda: _c().mark_errorbar().encode(x="g", y="v")},
    "mark_errorband": {"default": lambda: _c().mark_errorband().encode(x="g", y="v")},
    "mark_ribbon": {"default": lambda: _c().mark_ribbon().encode(x="v", y="w", y2="w")},
    "mark_contour": {
        "default": lambda: _c().mark_contour().encode(x="v", y="w"),
        # Same added/unmatched-pairing branch as mark_density's filled_2d.
        "filled": lambda: _c().mark_contour(fill=True).encode(x="v", y="w"),
    },
    "mark_violin": {"default": lambda: _c().mark_violin().encode(x="g", y="v")},
    "mark_qq": {"default": lambda: _c().mark_qq().encode(x="v")},
    "mark_raster": {"default": lambda: _c().mark_raster().encode(x="v", y="w")},
    "mark_hex": {"default": lambda: _c().mark_hex().encode(x="v", y="w")},
    "mark_swarm": {"default": lambda: _c().mark_swarm().encode(x="g", y="v")},
}

_DIAGNOSTIC_MARK_BUILDS: dict[str, Callable[[], Any]] = {
    "mark_roc": lambda: fm.Chart(_figure_data("roc_chart")).mark_roc(),
    "mark_pr": lambda: fm.Chart(_figure_data("pr_chart")).mark_pr(),
    "mark_calibration": lambda: fm.Chart(_figure_data("calibration_chart")).mark_calibration(),
    "mark_gain": lambda: fm.Chart(_figure_data("gain_chart")).mark_gain(),
    "mark_lift": lambda: fm.Chart(_figure_data("lift_chart")).mark_lift(),
    "mark_discrimination_threshold": lambda: fm.Chart(
        _figure_data("discrimination_threshold_chart")
    ).mark_discrimination_threshold(),
    "mark_confusion": lambda: fm.Chart(_figure_data("confusion_matrix_chart")).mark_confusion(),
    "mark_class_prediction_error": lambda: fm.Chart(
        _figure_data("class_prediction_error_chart")
    ).mark_class_prediction_error(),
    "mark_silhouette": lambda: fm.Chart(_figure_data("silhouette_chart")).mark_silhouette(),
    "mark_pca_scree": lambda: fm.Chart(_figure_data("pca_scree_chart")).mark_pca_scree(),
    "mark_intercluster_distance": lambda: fm.Chart(
        _figure_data("intercluster_distance_chart")
    ).mark_intercluster_distance(),
    "mark_decision_boundary": lambda: fm.Chart(
        _figure_data("decision_boundary_chart")
    ).mark_decision_boundary(),
    "mark_residuals": lambda: fm.Chart(_figure_data("residuals_chart")).mark_residuals(),
    "mark_prediction_error": lambda: fm.Chart(
        _figure_data("prediction_error_chart")
    ).mark_prediction_error(),
    "mark_learning_curve": lambda: fm.Chart(
        _figure_data("learning_curve_chart")
    ).mark_learning_curve(),
    "mark_validation_curve": lambda: fm.Chart(
        _figure_data("validation_curve_chart")
    ).mark_validation_curve(),
    "mark_cv_scores": lambda: fm.Chart(_figure_data("cv_scores_chart")).mark_cv_scores(),
    "mark_alpha_selection": lambda: fm.Chart(
        _figure_data("alpha_selection_chart")
    ).mark_alpha_selection(),
    "mark_importance": lambda: fm.Chart(_figure_data("importance_chart")).mark_importance(),
    "mark_shap_beeswarm": lambda: fm.Chart(
        _figure_data("shap_beeswarm_chart")
    ).mark_shap_beeswarm(),
    "mark_shap_bar": lambda: fm.Chart(_figure_data("shap_bar_chart")).mark_shap_bar(),
    "mark_shap_waterfall": lambda: fm.Chart(
        _figure_data("shap_waterfall_chart")
    ).mark_shap_waterfall(sample_idx=0),
    "mark_pdp": lambda: fm.Chart(_figure_data("pdp_chart")).mark_pdp(),
    "mark_rank1d": lambda: fm.Chart(_figure_data("rank1d_chart")).mark_rank1d(),
    "mark_rank2d": lambda: fm.Chart(_figure_data("rank2d_chart")).mark_rank2d(),
    "mark_parallel_coordinates": lambda: fm.Chart(
        _figure_data("parallel_coordinates_chart")
    ).mark_parallel_coordinates(),
}

# Each diagnostic mark has exactly one shape (its own domain's fixture), so
# it joins _MARK_BUILD_VARIANTS as a single "default" variant, the same
# shape every mark without a topology-changing kwarg variant has above.
for _name, _build in _DIAGNOSTIC_MARK_BUILDS.items():
    _MARK_BUILD_VARIANTS[_name] = {"default": _build}
del _name, _build

#: Every (mark name, build) pair the sweep below runs, keyed by a case
#: label that names the variant when there is more than one -- the
#: completeness check still reads the outer ``_MARK_BUILD_VARIANTS`` key
#: (the mark name), never this label.
_MARK_CASES: dict[str, tuple[str, Callable[[], Any]]] = {
    (mark_name if variant_label == "default" else f"{mark_name}[{variant_label}]"): (
        mark_name,
        build,
    )
    for mark_name, variants in _MARK_BUILD_VARIANTS.items()
    for variant_label, build in variants.items()
}


# ── Population completeness ──────────────────────────────────────────────────


def test_every_figure_taking_encode_has_a_build_case() -> None:
    missing = _missing_build_cases(
        _figure_population(), set(_FIGURE_BUILDS) | set(_COMPOSITE_ONLY_FIGURE_BUILDS)
    )
    assert not missing, f"no build case for figure(s): {missing}"
    assert len(_figure_population()) >= 40


def test_the_population_check_catches_a_figure_without_a_build_case() -> None:
    """Control: the real figure population, plus one name with no build
    case, fails ``_missing_build_cases`` by that name."""
    population = _figure_population() | {"not_a_real_figure"}
    covered = set(_FIGURE_BUILDS) | set(_COMPOSITE_ONLY_FIGURE_BUILDS)
    assert _missing_build_cases(population, covered) == ["not_a_real_figure"]


def test_every_composite_mark_has_a_build_case() -> None:
    """Completeness is keyed by mark name (``_MARK_BUILD_VARIANTS``'s outer
    key), never by a variant label: a mark with several builds is still
    one population entry."""
    missing = _missing_build_cases(_composite_mark_population(), set(_MARK_BUILD_VARIANTS))
    assert not missing, f"no build case for mark(s): {missing}"
    assert len(_composite_mark_population()) >= 39
    assert "mark_function" not in _composite_mark_population()


def test_the_population_check_catches_a_mark_without_a_build_case() -> None:
    """Control: the real mark population, plus one name with no build
    case, fails ``_missing_build_cases`` by that name."""
    population = _composite_mark_population() | {"mark_not_a_real_mark"}
    covered = set(_MARK_BUILD_VARIANTS)
    assert _missing_build_cases(population, covered) == ["mark_not_a_real_mark"]


# ── The regroup sweep ─────────────────────────────────────────────────────
#
# Parametrized on (door, name) string labels only -- no chart, model, or
# fixture is built until a specific test case actually runs.


@pytest.mark.parametrize("name", sorted(_FIGURE_BUILDS))
@pytest.mark.parametrize("door", _DOORS)
def test_every_figure_regroups_through_every_door(door: str, name: str) -> None:
    _assert_renders(_FIGURE_BUILDS[name], door, name)


@pytest.mark.parametrize("name", sorted(_COMPOSITE_ONLY_FIGURE_BUILDS))
@pytest.mark.parametrize("door", ["own-color", "own-detail"])
def test_every_grid_figure_regroups_through_its_own_encode(door: str, name: str) -> None:
    color_field, detail_field = _COMPOSITE_ONLY_GROUPING_FIELDS[name]
    build = _COMPOSITE_ONLY_FIGURE_BUILDS[name]
    encode = (
        {"color": fm.Color(color_field)}
        if door == "own-color"
        else {"detail": fm.Detail(detail_field)}
    )
    svg = build(encode).to_svg()
    assert isinstance(svg, str) and "<svg" in svg


@pytest.mark.parametrize("case", sorted(_MARK_CASES))
@pytest.mark.parametrize("door", _DOORS)
def test_every_composite_mark_regroups_through_every_door(door: str, case: str) -> None:
    mark_name, build = _MARK_CASES[case]
    _assert_renders(build, door, mark_name)


# ── The guards stay internal-invariant RuntimeErrors ────────────────────────


def _rebuild_runtime_error_messages() -> list[str]:
    """Every ``raise RuntimeError("...")`` message in ``_compose.py`` that
    names a rebuild it refuses (contains "rebuilt") -- not every
    ``RuntimeError`` the module might ever raise for an unrelated internal
    invariant."""
    tree = ast.parse(inspect.getsource(ferrum._compose))
    messages = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Raise)
            and isinstance(node.exc, ast.Call)
            and isinstance(node.exc.func, ast.Name)
            and node.exc.func.id == "RuntimeError"
        ):
            (arg,) = node.exc.args
            assert isinstance(arg, ast.Constant) and isinstance(arg.value, str)
            if "rebuilt" in arg.value:
                messages.append(arg.value)
    return messages


#: Every raise site in ``_compose.py`` that guards a rebuild shape invariant,
#: named for the refusal its message states:
_REBUILD_GUARD_SITES = (
    # _figure_source: a figure changed between one chart and a composition.
    "_figure_source (chart-vs-composition shape change)",
    # _figure_source: a figure drew fewer operands than before.
    "_figure_source (fewer operands drawn)",
    # _regrouped_alone: a single-operand chart drew a different mark.
    "_regrouped_alone (different mark drawn)",
    # _with_run_replaced: the statistics run no longer holds the exact
    # steps its composite mark lowered.
    "_with_run_replaced (lowered steps not found)",
    # _redrawn_layers: the chart lost layers its statistics drew before.
    "_redrawn_layers (layers lost)",
    # _paired: a composite mark drew a different number of layers.
    "_paired (different layer count)",
    # _paired: a chart drew a layer its statistics never drew before.
    "_paired (layer never drawn before)",
)


def test_the_rebuild_invariants_stay_internal_runtime_errors() -> None:
    """Every structural-invariant guard in ``_compose.py`` that refuses an
    impossible rebuild shape raises ``RuntimeError`` -- the project's
    internal-invariant type -- naming what could not be rebuilt. The sweep
    above proves no public call reaches any of them."""
    messages = _rebuild_runtime_error_messages()
    assert len(messages) == len(_REBUILD_GUARD_SITES)
    assert all(message.endswith("rebuilt") for message in messages)
