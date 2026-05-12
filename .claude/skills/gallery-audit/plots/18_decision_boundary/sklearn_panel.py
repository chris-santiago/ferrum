#!/usr/bin/env -S uv run --no-project --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["scikit-learn>=1.4", "matplotlib>=3.7", "numpy>=1.24"]
# ///
"""sklearn decision-boundary panel."""
from __future__ import annotations
import os, sys
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.linear_model import LogisticRegression

SEED = int(os.environ.get("FERRUM_AUDIT_SEED", "0"))
W = int(os.environ.get("FERRUM_AUDIT_W", "600"))
H = int(os.environ.get("FERRUM_AUDIT_H", "500"))
DPI = int(os.environ.get("FERRUM_AUDIT_DPI", "100"))
plt.rcParams["font.family"] = os.environ.get("FERRUM_AUDIT_FONT", "DejaVu Sans")
plt.rcParams["figure.autolayout"] = False


def main(out_dir: Path) -> None:
    X, y = load_iris(return_X_y=True)
    X2 = X[:, :2]
    model = LogisticRegression(max_iter=2000, random_state=SEED).fit(X2, y)
    fig, ax = plt.subplots(figsize=(W/DPI, H/DPI), dpi=DPI)
    DecisionBoundaryDisplay.from_estimator(model, X2, ax=ax)
    ax.scatter(X2[:, 0], X2[:, 1], c=y, edgecolor="k")
    fig.savefig(out_dir / "sklearn.png", dpi=DPI); plt.close(fig)


if __name__ == "__main__":
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True); main(out)
