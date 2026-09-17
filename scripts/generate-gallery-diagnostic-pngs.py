"""Regenerate the three classification-diagnostic cards on the Gallery page.

    unset CONDA_PREFIX && uv run --no-sync python scripts/generate-gallery-diagnostic-pngs.py

Writes ``docs/site/gallery/img/{roc_chart,pr_chart,calibration_chart}.png``,
the images behind the "ROC curve", "Precision-recall" and "Calibration curve"
cards in ``docs/site/gallery/gallery.md``.

Why this script exists (#219, 2026-09-13)
-----------------------------------------
The 94 gallery images were generated ad hoc in May 2026 (``9aba30b3``) with no
committed recipe, so a change to what a diagnostic chart *says* left them
stale with no way to refresh them. That bit twice: the calibration card still
showed the bin-level ``Brier = 0.007`` that F-L11-04 replaced with the true
per-sample score in September, and #219 retired the legend's ``(AUC = x)``
suffix that all three cards display. The recipe below is a **reconstruction**,
not the original one -- the original data and split were not recoverable from
the committed PNGs -- so the metric values on these three cards moved when it
first ran. Everything else (canvas size, theme, card titles) matches the
images it replaces.

Only these three cards are covered. The other 91 images are unaffected by
#219 and are deliberately left alone; a card that needs regenerating should
gain its own entry here rather than being re-rendered by hand.
"""

from __future__ import annotations

from pathlib import Path

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import ferrum as fm

OUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "site" / "gallery" / "img"

#: Chart size of every gallery image: the committed PNGs measure 1160x800,
#: which is this canvas through ``to_png``'s default 2x scale.
WIDTH, HEIGHT = 580, 400

#: The site's own theme, so a card looks like the docs page it sits on.
THEME = fm.themes.paper_ink

#: One well-separated binary problem, shared by all three cards so the ROC,
#: PR and reliability views describe the same model. Seeded end to end.
SEED = 0


def _fit():
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=6,
        n_redundant=2,
        random_state=SEED,
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=SEED)
    model = LogisticRegression(max_iter=2000, random_state=SEED).fit(X_train, y_train)
    return model, X_test, y_test


def _save(chart: "fm.Chart", name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{name}.png"
    path.write_bytes(chart.properties(width=WIDTH, height=HEIGHT).theme(THEME).to_png())
    print(f"  [OK]  {path} ({path.stat().st_size} bytes)")


def main() -> None:
    """Render and write all three cards."""
    model, X_test, y_test = _fit()
    # Card titles are the docs' own labels, not the figure functions' -- they
    # name the call the card is demonstrating. Kept verbatim from the images
    # these replace.
    _save(
        fm.roc_chart(model, X_test, y_test).properties(title="roc_chart() — ROC Curve"),
        "roc_chart",
    )
    _save(
        fm.pr_chart(model, X_test, y_test).properties(title="pr_chart() — Precision-Recall Curve"),
        "pr_chart",
    )
    _save(
        fm.calibration_chart(model, X_test, y_test).properties(title="calibration_chart()"),
        "calibration_chart",
    )


if __name__ == "__main__":
    main()
