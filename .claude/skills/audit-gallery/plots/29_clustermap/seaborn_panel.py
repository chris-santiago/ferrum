#!/usr/bin/env -S uv run --no-project --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["seaborn>=0.13", "pandas>=2", "matplotlib>=3.7", "numpy>=1.24", "scikit-learn>=1.3", "scipy>=1.10"]
# ///
"""seaborn clustermap panel."""
from __future__ import annotations
import os, sys
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

W = int(os.environ.get("FERRUM_AUDIT_W", "700"))
H = int(os.environ.get("FERRUM_AUDIT_H", "700"))
DPI = int(os.environ.get("FERRUM_AUDIT_DPI", "100"))
plt.rcParams["font.family"] = os.environ.get("FERRUM_AUDIT_FONT", "DejaVu Sans")


def main(out_dir: Path) -> None:
    raw = load_iris(as_frame=True)
    df = raw.frame.drop(columns=["target"])
    g = sns.clustermap(df, figsize=(W/DPI, H/DPI))
    g.figure.savefig(out_dir / "seaborn.png", dpi=DPI); plt.close("all")


if __name__ == "__main__":
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True); main(out)
