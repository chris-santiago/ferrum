"""Ferrum manifold-embedding panel (t-SNE)."""
from __future__ import annotations
import os, sys
from pathlib import Path

from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
import ferrum

SEED = int(os.environ.get("FERRUM_AUDIT_SEED", "0"))
W = int(os.environ.get("FERRUM_AUDIT_W", "640"))
H = int(os.environ.get("FERRUM_AUDIT_H", "480"))


def main(out_dir: Path) -> None:
    X, _ = load_iris(return_X_y=True)
    model = KMeans(n_clusters=3, n_init=10, random_state=SEED).fit(X)
    viz = ferrum.ManifoldVisualizer(model, method="tsne", random_state=SEED).fit(X)
    chart = viz.show().properties(width=W, height=H)
    (out_dir / "ferrum.svg").write_text(chart.show_svg(), encoding="utf-8")


if __name__ == "__main__":
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True); main(out)
