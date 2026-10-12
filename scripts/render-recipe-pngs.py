"""Render the customization recipe scripts that the docs show to PNG.

Imports each recipe module, finds the top-level ``chart`` variable, calls
``to_png(scale=2.0)``, and writes the result to ``docs/site/assets/recipes/``.

A recipe in ``KEPT_UNSHOWN`` is kept as source but rendered by nothing: no
docs page shows it, so rendering it would only produce an unreferenced image.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
RECIPE_DIR = REPO_ROOT / "docs/site/recipes/customization"
OUTPUT_DIR = REPO_ROOT / "docs/site/assets/recipes"

#: Recipe sources kept in the repo but shown on no docs page, each with the
#: reason it is kept. Their images are not rendered.
KEPT_UNSHOWN: dict[str, str] = {
    "dashboard_inset": "kept for provenance; no docs page shows a dashboard-inset recipe",
    "highlight_region": "kept for provenance; no docs page shows a highlight-region recipe",
}


def recipes_to_render() -> list[Path]:
    """Return the recipe scripts whose images the docs show, in name order."""
    return [p for p in sorted(RECIPE_DIR.glob("*.py")) if p.stem not in KEPT_UNSHOWN]


def run_recipe(script_path: Path) -> None:
    """Execute a recipe script and save its ``chart`` variable as PNG."""
    spec = importlib.util.spec_from_file_location(script_path.stem, script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    chart = getattr(mod, "chart", None)
    if chart is None:
        print(f"  SKIP {script_path.name}: no 'chart' variable")
        return

    png_bytes = chart.to_png(scale=2.0)
    out_path = OUTPUT_DIR / f"{script_path.stem}.png"
    out_path.write_bytes(png_bytes)
    print(f"  OK   {out_path.relative_to(REPO_ROOT)} ({len(png_bytes):,} bytes)")


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    recipes = recipes_to_render()
    for stem, reason in KEPT_UNSHOWN.items():
        print(f"  KEEP {stem}.py (not rendered): {reason}")
    print(f"Rendering {len(recipes)} recipes to {OUTPUT_DIR.relative_to(REPO_ROOT)}/")
    errors: list[tuple[str, Exception]] = []
    for r in recipes:
        try:
            run_recipe(r)
        except Exception as exc:
            print(f"  FAIL {r.name}: {exc}")
            errors.append((r.name, exc))

    print()
    if errors:
        print(f"{len(errors)} recipe(s) failed:")
        for name, exc in errors:
            print(f"  {name}: {exc}")
        sys.exit(1)
    else:
        print(f"All {len(recipes)} recipes rendered successfully.")
