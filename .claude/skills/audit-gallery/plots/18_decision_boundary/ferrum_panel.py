"""Ferrum decision-boundary panel."""
from __future__ import annotations
import os, sys
from pathlib import Path

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import ferrum

SEED = int(os.environ.get("FERRUM_AUDIT_SEED", "0"))
W = int(os.environ.get("FERRUM_AUDIT_W", "600"))
H = int(os.environ.get("FERRUM_AUDIT_H", "500"))


def main(out_dir: Path) -> None:
    X, y = load_iris(return_X_y=True)
    X2 = X[:, :2]
    model = LogisticRegression(max_iter=500, random_state=SEED).fit(X2, y)
    chart = ferrum.decision_boundary_chart(
        model, X2, y, features=(0, 1), scatter=True, random_state=SEED
    ).properties(width=W, height=H)
    (out_dir / "ferrum.svg").write_text(chart.show_svg(), encoding="utf-8")
    (out_dir / "ferrum.png").write_bytes(chart.show_png())


if __name__ == "__main__":
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True); main(out)
