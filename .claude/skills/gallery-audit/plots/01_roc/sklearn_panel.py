#!/usr/bin/env -S uv run --no-project --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "scikit-learn>=1.4",
#   "matplotlib>=3.7",
#   "numpy>=1.24",
# ]
# ///
"""sklearn ROC panel — isolated PEP 723 env."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import train_test_split

SEED = int(os.environ.get("FERRUM_AUDIT_SEED", "0"))
W = int(os.environ.get("FERRUM_AUDIT_W", "640"))
H = int(os.environ.get("FERRUM_AUDIT_H", "480"))
DPI = int(os.environ.get("FERRUM_AUDIT_DPI", "100"))
FONT = os.environ.get("FERRUM_AUDIT_FONT", "DejaVu Sans")

plt.rcParams["font.family"] = FONT
plt.rcParams["figure.autolayout"] = False


def main(out_dir: Path) -> None:
    X, y = load_breast_cancer(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=SEED)
    model = LogisticRegression(max_iter=2000, random_state=SEED).fit(Xtr, ytr)

    fig, ax = plt.subplots(figsize=(W / DPI, H / DPI), dpi=DPI)
    # Defaults only — no name=, no color=, no plot_chance_level= overrides.
    # (sklearn's default for plot_chance_level changed across versions; whatever the
    # installed pin gives is what "default" means for this audit.)
    RocCurveDisplay.from_estimator(model, Xte, yte, ax=ax)
    fig.savefig(out_dir / "sklearn.png", dpi=DPI)
    plt.close(fig)


if __name__ == "__main__":
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    main(out)
