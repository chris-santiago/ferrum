"""Ferrum calibration panel."""
from __future__ import annotations
import os, sys
from pathlib import Path

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import ferrum

SEED = int(os.environ.get("FERRUM_AUDIT_SEED", "0"))
W = int(os.environ.get("FERRUM_AUDIT_W", "640"))
H = int(os.environ.get("FERRUM_AUDIT_H", "480"))


def main(out_dir: Path) -> None:
    X, y = load_breast_cancer(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=SEED)
    model = LogisticRegression(max_iter=2000, random_state=SEED).fit(Xtr, ytr)
    chart = ferrum.calibration_chart(model, Xte, yte).properties(width=W, height=H)
    (out_dir / "ferrum.svg").write_text(chart.show_svg(), encoding="utf-8")


if __name__ == "__main__":
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True); main(out)
