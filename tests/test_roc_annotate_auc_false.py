"""``annotate_auc=False`` means no AUC anywhere on the chart (#219 / D4).

The flag suppressed the endpoint overlay but never the title, which printed
``"ROC Curve — AUC 0.984"`` regardless, so opting out of the annotation still
left the number on the figure. The title now keeps its plain name and the
overlay is the only metric surface, which is what makes the opt-out real.
"""

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

import ferrum as fm
from tests._svg_extents import text_runs


def _iris_model():
    X, y = load_iris(return_X_y=True)
    return LogisticRegression(max_iter=200).fit(X, y), X, y


def test_roc_annotate_auc_false_omits_every_auc_string():
    model, X, y = _iris_model()
    svg = fm.roc_chart(model, X, y, annotate_auc=False).to_svg()
    assert [t for t in text_runs(svg) if "AUC" in t] == []
    # The chart is still titled, and the classes are still in the legend.
    assert ">ROC Curve<" in svg


def test_roc_annotate_auc_true_emits_exactly_one_auc_per_curve():
    model, X, y = _iris_model()
    chart = fm.roc_chart(model, X, y, annotate_auc=True)
    n_curves = chart._data["class"].n_unique()
    assert n_curves == 3  # iris: one ROC curve per class
    auc_texts = [t for t in text_runs(chart.to_svg()) if "AUC" in t]
    assert len(auc_texts) == n_curves
    assert all(t.startswith("AUC = ") for t in auc_texts)


def test_the_class_legend_is_titled_whether_or_not_the_curves_are_annotated():
    """The legend is titled ``Class`` with and without the metric overlay.

    The per-class hue is a layer-level binding of the ROC/PR composite, and
    the overlay adds no color of its own, so the title comes from that
    binding alone: turning the annotation off must not untitle the legend.
    """
    model, X, y = _iris_model()
    for build, flag in ((fm.roc_chart, "annotate_auc"), (fm.pr_chart, "annotate_ap")):
        for annotate in (False, True):
            runs = text_runs(build(model, X, y, **{flag: annotate}).to_svg())
            assert runs.count("Class") == 1, (build.__name__, annotate, runs)


def test_the_layer_level_class_legend_is_titled_on_every_composition_route():
    """The flat chart, an overlay leaf and a shared figure-legend band all title it ``Class``."""
    model, X, y = _iris_model()

    def roc():
        return fm.roc_chart(model, X, y, annotate_auc=False)

    other = fm.Chart({"fpr": [0.0, 1.0], "tpr": [0.0, 1.0]}).mark_point().encode(x="fpr", y="tpr")
    shared = fm.Resolve(scale={"color": "shared"}, legend={"color": "shared"})
    routes = {
        "flat": roc(),
        "overlay": fm.LayerChart(roc(), other),
        "band": fm.hconcat(roc(), roc(), resolve=shared),
    }
    for route, chart in routes.items():
        assert text_runs(chart.to_svg()).count("Class") == 1, route
