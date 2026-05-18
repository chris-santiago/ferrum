#!/usr/bin/env -S uv run --no-project --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["seaborn>=0.13", "matplotlib>=3.7", "numpy>=1.24", "pandas>=2.0"]
# ///
"""seaborn catplot panel (default kind='strip')."""
from __future__ import annotations
import os, sys
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

SEED = int(os.environ.get("FERRUM_AUDIT_SEED", "0"))
W = int(os.environ.get("FERRUM_AUDIT_W", "640"))
H = int(os.environ.get("FERRUM_AUDIT_H", "480"))
DPI = int(os.environ.get("FERRUM_AUDIT_DPI", "100"))
plt.rcParams["font.family"] = os.environ.get("FERRUM_AUDIT_FONT", "DejaVu Sans")
plt.rcParams["figure.autolayout"] = False


def main(out_dir: Path) -> None:
    tips = sns.load_dataset("tips")
    g = sns.catplot(data=tips, x="day", y="total_bill")
    g.figure.set_size_inches(W/DPI, H/DPI)
    g.figure.savefig(out_dir / "seaborn.png", dpi=DPI); plt.close(g.figure)


if __name__ == "__main__":
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True); main(out)
