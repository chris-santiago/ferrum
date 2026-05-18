"""Ferrum SHAP summary panel."""
from __future__ import annotations
import os, sys
from pathlib import Path

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
import ferrum

SEED = int(os.environ.get("FERRUM_AUDIT_SEED", "0"))
W = int(os.environ.get("FERRUM_AUDIT_W", "700"))
H = int(os.environ.get("FERRUM_AUDIT_H", "500"))


def main(out_dir: Path) -> None:
    X, y = make_classification(n_samples=200, n_features=8, random_state=SEED)
    model = RandomForestClassifier(n_estimators=20, random_state=SEED).fit(X, y)
    chart = ferrum.shap_chart(model, X, random_state=SEED).properties(width=W, height=H)
    (out_dir / "ferrum.svg").write_text(chart.show_svg(), encoding="utf-8")
    (out_dir / "ferrum.png").write_bytes(chart.show_png())


if __name__ == "__main__":
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True); main(out)
