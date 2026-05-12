"""Ferrum CV-scores panel."""
from __future__ import annotations
import os, sys
from pathlib import Path

from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
import ferrum

SEED = int(os.environ.get("FERRUM_AUDIT_SEED", "0"))
W = int(os.environ.get("FERRUM_AUDIT_W", "640"))
H = int(os.environ.get("FERRUM_AUDIT_H", "480"))


def main(out_dir: Path) -> None:
    X, y = load_diabetes(return_X_y=True)
    model = Ridge(random_state=SEED)
    chart = ferrum.cv_scores_chart(model, X, y).properties(width=W, height=H)
    (out_dir / "ferrum.svg").write_text(chart.show_svg(), encoding="utf-8")
    (out_dir / "ferrum.png").write_bytes(chart.show_png())


if __name__ == "__main__":
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True); main(out)
