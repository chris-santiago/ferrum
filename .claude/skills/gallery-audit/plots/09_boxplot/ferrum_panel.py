"""Ferrum boxplot panel."""
from __future__ import annotations
import os, sys
from pathlib import Path

import polars as pl
import ferrum

W = int(os.environ.get("FERRUM_AUDIT_W", "640"))
H = int(os.environ.get("FERRUM_AUDIT_H", "480"))
FIXTURES = Path(__file__).parent.parent / "_fixtures"


def main(out_dir: Path) -> None:
    df = pl.read_csv(FIXTURES / "tips.csv")
    chart = ferrum.catplot(df, x="day", y="total_bill", kind="box").properties(width=W, height=H)
    (out_dir / "ferrum.svg").write_text(chart.show_svg(), encoding="utf-8")


if __name__ == "__main__":
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True); main(out)
