#!/usr/bin/env -S uv run --no-project --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["scikit-plot>=0.3", "scikit-learn>=1.3,<1.5", "scipy>=1.10,<1.12", "matplotlib>=3.7", "numpy>=1.24,<2"]
# ///
"""scikit-plot PR panel."""
from __future__ import annotations
import os, sys
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scikitplot as skplt
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

SEED = int(os.environ.get("FERRUM_AUDIT_SEED", "0"))
W = int(os.environ.get("FERRUM_AUDIT_W", "640"))
H = int(os.environ.get("FERRUM_AUDIT_H", "480"))
DPI = int(os.environ.get("FERRUM_AUDIT_DPI", "100"))
plt.rcParams["font.family"] = os.environ.get("FERRUM_AUDIT_FONT", "DejaVu Sans")
plt.rcParams["figure.autolayout"] = False


def main(out_dir: Path) -> None:
    X, y = load_breast_cancer(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=SEED)
    model = LogisticRegression(max_iter=2000, random_state=SEED).fit(Xtr, ytr)
    probas = model.predict_proba(Xte)
    fig, ax = plt.subplots(figsize=(W/DPI, H/DPI), dpi=DPI)
    skplt.metrics.plot_precision_recall(yte, probas, ax=ax)
    fig.savefig(out_dir / "skp.png", dpi=DPI); plt.close(fig)


if __name__ == "__main__":
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True); main(out)
