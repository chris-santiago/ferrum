"""Ferrum learning-curve panel."""
from __future__ import annotations
import os, sys
from pathlib import Path

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
import ferrum

SEED = int(os.environ.get("FERRUM_AUDIT_SEED", "0"))
W = int(os.environ.get("FERRUM_AUDIT_W", "720"))
H = int(os.environ.get("FERRUM_AUDIT_H", "480"))


def main(out_dir: Path) -> None:
    X, y = load_breast_cancer(return_X_y=True)
    model = LogisticRegression(max_iter=2000, random_state=SEED)
    chart = ferrum.learning_curve_chart(model, X, y).properties(width=W, height=H)
    (out_dir / "ferrum.svg").write_text(chart.show_svg(), encoding="utf-8")


if __name__ == "__main__":
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True); main(out)
