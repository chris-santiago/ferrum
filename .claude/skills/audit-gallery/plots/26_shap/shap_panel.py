#!/usr/bin/env -S uv run --no-project --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["shap>=0.44", "scikit-learn>=1.3,<1.5", "matplotlib>=3.7", "numpy>=1.24,<2"]
# ///
"""shap summary panel (beeswarm)."""
from __future__ import annotations
import os, sys
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

SEED = int(os.environ.get("FERRUM_AUDIT_SEED", "0"))
W = int(os.environ.get("FERRUM_AUDIT_W", "700"))
H = int(os.environ.get("FERRUM_AUDIT_H", "500"))
DPI = int(os.environ.get("FERRUM_AUDIT_DPI", "100"))
plt.rcParams["font.family"] = os.environ.get("FERRUM_AUDIT_FONT", "DejaVu Sans")
plt.rcParams["figure.autolayout"] = False


def main(out_dir: Path) -> None:
    X, y = make_classification(n_samples=200, n_features=8, random_state=SEED)
    model = RandomForestClassifier(n_estimators=20, random_state=SEED).fit(X, y)
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)
    # binary classifier: shap returns either list[2] or 3-D array; reduce to class-1 contributions
    if isinstance(sv, list):
        values = sv[1]
    elif hasattr(sv, "ndim") and sv.ndim == 3:
        values = sv[..., 1]
    else:
        values = sv
    fig = plt.figure(figsize=(W/DPI, H/DPI), dpi=DPI)
    shap.summary_plot(values, X, show=False, plot_size=None)
    fig.savefig(out_dir / "shap.png", dpi=DPI); plt.close("all")


if __name__ == "__main__":
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True); main(out)
