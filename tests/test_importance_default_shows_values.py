import re

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

import ferrum as fm


def test_importance_default_shows_numeric_labels_on_bars():
    X, y = load_iris(return_X_y=True)
    model = RandomForestClassifier(random_state=0).fit(X, y)
    chart = fm.importance_chart(model, X, y)
    svg = chart.show_svg()
    # Each bar should carry a numeric importance label (formatted `.3f`).
    assert re.search(r"<text[^>]*>\s*0\.\d{2,}\s*<", svg)


def test_importance_show_values_false_omits():
    X, y = load_iris(return_X_y=True)
    model = RandomForestClassifier(random_state=0).fit(X, y)
    chart = fm.importance_chart(model, X, y, show_values=False)
    svg = chart.show_svg()
    # No bar-end labels — only axis tick labels remain, which are <text>
    # nodes carrying ticks. Heuristic upper bound: ≤ 12 numeric labels.
    matches = re.findall(r"<text[^>]*>\s*0\.\d", svg)
    assert len(matches) <= 12
