"""Ferrum cluster-diagnostics panel."""
from __future__ import annotations
import os, sys
from pathlib import Path

from sklearn.datasets import make_blobs
import ferrum

SEED = int(os.environ.get("FERRUM_AUDIT_SEED", "0"))
W = int(os.environ.get("FERRUM_AUDIT_W", "700"))
H = int(os.environ.get("FERRUM_AUDIT_H", "500"))


def main(out_dir: Path) -> None:
    X, _ = make_blobs(n_samples=300, centers=5, random_state=SEED)
    chart = ferrum.cluster_diagnostics(X, ks=list(range(2, 11)), random_state=SEED)
    (out_dir / "ferrum.svg").write_text(chart.show_svg(), encoding="utf-8")


if __name__ == "__main__":
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True); main(out)
