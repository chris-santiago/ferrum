import re

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

import ferrum as fm


def test_pr_baseline_hline_drawn_for_binary():
    """Binary PR chart shows a horizontal baseline at positive-class prevalence."""
    X, y = make_classification(
        n_samples=400,
        n_classes=2,
        weights=[0.7, 0.3],
        n_informative=4,
        random_state=0,
    )
    model = LogisticRegression(max_iter=300).fit(X, y)
    chart = fm.pr_chart(model, X, y, per_class=False)
    svg = chart.to_svg()
    # Baseline prevalence ≈ 0.3; the mark_rule overlay emits an <line>
    # spanning the plot at that y. Heuristic: SVG must contain at least one
    # horizontal line element (y1 == y2) since the rule is horizontal.
    horizontal = re.findall(
        r'<line[^>]*y1="([\d.]+)"[^>]*y2="([\d.]+)"',
        svg,
    )
    assert any(abs(float(a) - float(b)) < 0.5 for a, b in horizontal), (
        "expected at least one horizontal <line> element from the baseline rule"
    )
