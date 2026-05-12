"""Ferrum lift-curve panel."""
from __future__ import annotations
import os, sys
from pathlib import Path

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
import ferrum
from ferrum._direct_label import _direct_label_endpoint

SEED = int(os.environ.get("FERRUM_AUDIT_SEED", "0"))
W = int(os.environ.get("FERRUM_AUDIT_W", "600"))
H = int(os.environ.get("FERRUM_AUDIT_H", "500"))


def main(out_dir: Path) -> None:
    """Render ferrum's lift chart for the synthetic-binary fixture."""
    X, y = make_classification(n_samples=500, n_features=10, random_state=SEED)
    model = LogisticRegression(max_iter=500, random_state=SEED).fit(X, y)
    chart = ferrum.lift_chart(model, X, y, random_state=SEED)
    # Schwabish T2: ≤4 short-label series → swap right-gutter legend for
    # endpoint-anchored direct labels.
    chart = chart.encode(color=ferrum.Color("class", legend=None))
    chart = _direct_label_endpoint(
        chart, label_field="class",
        x_col="percent_population", y_col="lift",
    )
    chart = chart.properties(width=W, height=H)
    (out_dir / "ferrum.svg").write_text(chart.show_svg(), encoding="utf-8")


if __name__ == "__main__":
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True); main(out)
