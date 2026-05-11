#!/usr/bin/env -S uv run --no-project --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "anthropic>=0.40",
#   "rich>=13",
#   "pillow>=10",
#   "resvg-py>=0.3",
#   "tomli>=2; python_version < '3.11'",
# ]
# ///
"""
Gallery audit orchestrator.

Stages
------
generate  : Run all panel scripts for selected rows → write panel images.
judge     : Read panel images, call Claude with cached rubric → write verdicts.
report    : Aggregate verdicts into output/REPORT.md.
all       : generate → judge → report.

Determinism env vars (passed to every panel subprocess so panels can't drift):
  FERRUM_AUDIT_SEED      = 0
  FERRUM_AUDIT_W         = <row width px>
  FERRUM_AUDIT_H         = <row height px>
  FERRUM_AUDIT_DPI       = 100
  FERRUM_AUDIT_FONT      = DejaVu Sans
  MPLBACKEND             = Agg

audit.py itself is a PEP 723 script (inline deps: anthropic, rich, pillow,
resvg-py) — run via `uv run --no-project --script audit.py …`. Comparator
panels run the same way in their own isolated PEP 723 envs. The ferrum panel
is dispatched via `uv run --no-sync python <ferrum_panel.py>` so it picks up
the project's `.venv` (where maturin-built `ferrum._core` lives) without
re-resolving the lockfile.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    import tomllib  # py311+
except ModuleNotFoundError:  # py310
    import tomli as tomllib  # type: ignore

from rich.console import Console
from rich.table import Table

SKILL_DIR = Path(__file__).resolve().parent
PLOTS_DIR = SKILL_DIR / "plots"
OUTPUT_DIR = SKILL_DIR / "output"
RUBRIC_PATH = SKILL_DIR / "rubric.md"
JUDGE_PROMPT_PATH = SKILL_DIR / "judge_prompt.md"

REPO_ROOT = SKILL_DIR.parent.parent.parent  # .claude/skills/gallery-audit/ → repo root

console = Console()


# --------------------------------------------------------------------------- #
# Row config
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class RowConfig:
    row_id: str           # "01_roc"
    plot_type: str        # human-readable
    width: int
    height: int
    dataset: str
    panels: list[str]     # ["ferrum", "sklearn", "yellowbrick", "skp"]
    ferrum_status: str    # "READY" | "PARTIAL" | "BLOCKED"
    ferrum_notes: str

    @classmethod
    def load(cls, row_dir: Path) -> RowConfig:
        cfg_path = row_dir / "config.toml"
        with cfg_path.open("rb") as f:
            data = tomllib.load(f)
        return cls(
            row_id=row_dir.name,
            plot_type=data["plot_type"],
            width=data.get("width", 640),
            height=data.get("height", 480),
            dataset=data.get("dataset", ""),
            panels=data.get("panels", []),
            ferrum_status=data.get("ferrum_status", "READY"),
            ferrum_notes=data.get("ferrum_notes", ""),
        )


def discover_rows(filter_ids: list[str] | None = None) -> list[RowConfig]:
    rows: list[RowConfig] = []
    for d in sorted(PLOTS_DIR.iterdir()):
        if not d.is_dir() or not (d / "config.toml").exists():
            continue
        cfg = RowConfig.load(d)
        if filter_ids and not any(
            cfg.row_id == fid or cfg.row_id.startswith(f"{fid}_") or
            cfg.row_id.split("_", 1)[0].lstrip("0") == fid.lstrip("0")
            for fid in filter_ids
        ):
            continue
        rows.append(cfg)
    return rows


# --------------------------------------------------------------------------- #
# Determinism env
# --------------------------------------------------------------------------- #

def panel_env(cfg: RowConfig) -> dict[str, str]:
    env = os.environ.copy()
    env.update({
        "FERRUM_AUDIT_SEED": "0",
        "FERRUM_AUDIT_W": str(cfg.width),
        "FERRUM_AUDIT_H": str(cfg.height),
        "FERRUM_AUDIT_DPI": "100",
        "FERRUM_AUDIT_FONT": "DejaVu Sans",
        "MPLBACKEND": "Agg",
        "PYTHONHASHSEED": "0",
    })
    return env


# --------------------------------------------------------------------------- #
# Generate stage
# --------------------------------------------------------------------------- #

def run_panel(cfg: RowConfig, panel: str, out_dir: Path) -> dict:
    """Run one panel script. Returns {ok, path, status, log}."""
    row_dir = PLOTS_DIR / cfg.row_id
    script = row_dir / f"{panel}_panel.py"
    if not script.exists():
        return {"ok": False, "path": None, "status": "MISSING_SCRIPT",
                "log": f"{script} does not exist"}

    out_dir.mkdir(parents=True, exist_ok=True)

    if panel == "ferrum":
        # ferrum panel must run in the project .venv (needs maturin-built ferrum._core).
        # `--no-sync` reuses the existing venv without resolving deps.
        cmd = ["uv", "run", "--no-sync", "python", str(script), str(out_dir)]
    else:
        # Comparator panels: isolated PEP 723 env
        cmd = ["uv", "run", "--no-project", "--script", str(script), str(out_dir)]

    proc = subprocess.run(
        cmd, capture_output=True, text=True,
        env=panel_env(cfg), cwd=REPO_ROOT,
    )
    if proc.returncode != 0:
        return {"ok": False, "path": None,
                "status": "NOT_IMPLEMENTED" if panel == "ferrum" else "RENDER_ERROR",
                "log": (proc.stdout + "\n" + proc.stderr).strip()}

    # ferrum panel emits SVG, rasterize to PNG; comparators emit PNG directly.
    if panel == "ferrum":
        svg_path = out_dir / "ferrum.svg"
        png_path = out_dir / "ferrum.png"
        if not svg_path.exists():
            return {"ok": False, "path": None, "status": "RENDER_ERROR",
                    "log": f"ferrum_panel.py did not write {svg_path}"}
        rasterize_svg(svg_path, png_path, cfg.width, cfg.height)
        return {"ok": True, "path": png_path, "status": "OK", "log": ""}

    png_path = out_dir / f"{panel}.png"
    if not png_path.exists():
        return {"ok": False, "path": None, "status": "RENDER_ERROR",
                "log": f"{panel}_panel.py did not write {png_path}"}
    return {"ok": True, "path": png_path, "status": "OK", "log": ""}


def rasterize_svg(svg_path: Path, png_path: Path, w: int, h: int) -> None:
    """Rasterize an SVG to PNG. Mirrors `tests/_snapshots.py::rasterize_svg`.

    The output PNG dimensions come from the SVG's intrinsic width/height —
    `ferrum.Chart.properties(width=W, height=H)` is expected to emit
    `<svg width="W" height="H">` so the PNG comes out at W×H.
    """
    import resvg_py
    svg = svg_path.read_text(encoding="utf-8")
    png_bytes = bytes(resvg_py.svg_to_bytes(svg_string=svg, dpi=96))
    png_path.write_bytes(png_bytes)
    _ = (w, h)  # reserved for future size-verification logging


def stage_generate(rows: list[RowConfig]) -> dict[str, dict[str, dict]]:
    """For each row × panel, run the script and record the result."""
    results: dict[str, dict[str, dict]] = {}
    for cfg in rows:
        console.rule(f"[bold]Row {cfg.row_id} — {cfg.plot_type}")
        if not cfg.panels:
            console.print(f"  [yellow]No panels wired — see plots/{cfg.row_id}/TODO.md[/yellow]")
            continue
        out_dir = OUTPUT_DIR / cfg.row_id
        # clean previous panels for this row to avoid stale artifacts
        if out_dir.exists():
            for f in out_dir.glob("*.png"):
                f.unlink()
            for f in out_dir.glob("*.svg"):
                f.unlink()
        results[cfg.row_id] = {}
        for panel in cfg.panels:
            console.print(f"  • {panel}_panel.py ... ", end="")
            res = run_panel(cfg, panel, out_dir)
            results[cfg.row_id][panel] = res
            if res["ok"]:
                console.print("[green]ok[/green]")
            else:
                console.print(f"[red]{res['status']}[/red]")
                if res["log"]:
                    console.print(f"      [dim]{res['log'].splitlines()[-1][:120]}[/dim]")
    return results


# --------------------------------------------------------------------------- #
# Judge stage
# --------------------------------------------------------------------------- #

def stage_judge(rows: list[RowConfig]) -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        console.print("[red]ANTHROPIC_API_KEY not set; cannot run judge stage.[/red]")
        sys.exit(2)

    import anthropic
    import base64
    client = anthropic.Anthropic(api_key=api_key)

    rubric = RUBRIC_PATH.read_text(encoding="utf-8")
    judge_prompt = JUDGE_PROMPT_PATH.read_text(encoding="utf-8")

    # Cached prefix: rubric + judge prompt instructions. Invariant across rows.
    system_blocks = [
        {"type": "text", "text": "You are an expert plot reviewer. Cached rubric follows."},
        {"type": "text", "text": rubric,
         "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": judge_prompt,
         "cache_control": {"type": "ephemeral"}},
    ]

    for cfg in rows:
        if not cfg.panels:
            console.print(f"[yellow]Skipping {cfg.row_id}[/yellow] — no panels wired yet (see plots/{cfg.row_id}/TODO.md)")
            continue
        out_dir = OUTPUT_DIR / cfg.row_id
        verdict_path = out_dir / "verdict.md"
        console.print(f"[bold]Judging {cfg.row_id}[/bold] — {cfg.plot_type}")

        # Gather available panel PNGs.
        content: list[dict] = []
        content.append({"type": "text",
                        "text": f"# Row metadata\n\nrow_id: {cfg.row_id}\n"
                                f"plot_type: {cfg.plot_type}\n"
                                f"dataset: {cfg.dataset}\n"
                                f"width × height: {cfg.width}×{cfg.height}\n"})
        ferrum_status = "OK"
        for panel in cfg.panels:
            png_path = out_dir / f"{panel}.png"
            if not png_path.exists():
                if panel == "ferrum":
                    ferrum_status = "NOT_IMPLEMENTED"
                content.append({"type": "text",
                                "text": f"\n## Panel `{panel}`\n\n(no PNG produced — render failed or script missing)\n"})
                continue
            b64 = base64.standard_b64encode(png_path.read_bytes()).decode("ascii")
            content.append({"type": "text", "text": f"\n## Panel `{panel}`\n"})
            content.append({"type": "image",
                            "source": {"type": "base64", "media_type": "image/png", "data": b64}})

        content.append({"type": "text",
                        "text": f"\nferrum_status_hint: {ferrum_status}\n\n"
                                "Produce the verdict now in the format specified in the cached judge prompt."})

        resp = client.messages.create(
            model="claude-opus-4-7",
            max_tokens=2000,
            system=system_blocks,  # type: ignore[arg-type]
            messages=[{"role": "user", "content": content}],  # type: ignore[arg-type]
        )
        text = "".join(b.text for b in resp.content if b.type == "text")
        verdict_path.write_text(text, encoding="utf-8")
        console.print(f"  → {verdict_path.relative_to(REPO_ROOT)}")


# --------------------------------------------------------------------------- #
# Report stage
# --------------------------------------------------------------------------- #

FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n(.*)$", re.DOTALL)


def parse_verdict(path: Path) -> dict | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    m = FRONTMATTER_RE.match(text)
    if not m:
        return None
    fm_raw, body = m.group(1), m.group(2)
    fm: dict = {}
    current_list_key: str | None = None
    for line in fm_raw.splitlines():
        if not line.strip():
            continue
        if line.startswith("  - "):
            if current_list_key is None:
                continue
            fm.setdefault(current_list_key, []).append(line[4:].strip())
        elif ":" in line:
            key, _, val = line.partition(":")
            val = val.strip()
            current_list_key = None
            if val == "":
                fm[key.strip()] = []
                current_list_key = key.strip()
            else:
                fm[key.strip()] = val
    return {"frontmatter": fm, "body": body}


SEVERITY_RANK = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "NONE": 3}


def stage_report() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    verdicts = []
    for row_dir in sorted(OUTPUT_DIR.iterdir()):
        if not row_dir.is_dir():
            continue
        v = parse_verdict(row_dir / "verdict.md")
        if v is None:
            continue
        v["row_id"] = row_dir.name
        verdicts.append(v)

    verdicts.sort(key=lambda v: SEVERITY_RANK.get(v["frontmatter"].get("severity", "NONE"), 9))

    report = REPORT_HEADER + "\n"
    if not verdicts:
        report += "_No verdicts found. Run `audit.py judge` first._\n"
    else:
        # Summary table
        report += "## Summary\n\n"
        report += "| Row | Plot | Severity | Ferrum status | Top gap |\n"
        report += "|---|---|---|---|---|\n"
        for v in verdicts:
            fm = v["frontmatter"]
            top = (fm.get("ferrum_missing") or ["—"])[0]
            report += (f"| `{v['row_id']}` | — | {fm.get('severity', '?')} | "
                       f"{fm.get('ferrum_status', '?')} | {top} |\n")
        report += "\n---\n\n"
        # Per-row detail
        for v in verdicts:
            report += f"## `{v['row_id']}`\n\n"
            report += v["body"].strip() + "\n\n"

    out = OUTPUT_DIR / "REPORT.md"
    out.write_text(report, encoding="utf-8")
    console.print(f"[green]Wrote {out.relative_to(REPO_ROOT)}[/green]")

    # Quick top-line table to console
    tbl = Table(title="Gallery audit summary")
    tbl.add_column("Row"); tbl.add_column("Severity"); tbl.add_column("Ferrum")
    for v in verdicts:
        fm = v["frontmatter"]
        tbl.add_row(v["row_id"], str(fm.get("severity", "?")),
                    str(fm.get("ferrum_status", "?")))
    console.print(tbl)


REPORT_HEADER = """# Gallery audit report

Generated by `.claude/skills/gallery-audit/audit.py`.

Rows are ranked by severity. HIGH-severity findings are missing domain-expected
annotations (B-category rubric items); MEDIUM are reference lines or
uncertainty bands; LOW are cosmetic. Use this as a punchlist for the
`gallery-fixer` agent.
"""


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_rows_arg(s: str | None) -> list[str] | None:
    if s is None:
        return None
    return [x.strip() for x in s.split(",") if x.strip()]


def main() -> int:
    p = argparse.ArgumentParser(prog="audit.py")
    p.add_argument("stage", choices=["generate", "judge", "report", "all"])
    p.add_argument("--rows", help="Comma-separated row IDs or prefixes (e.g. '1,3' or '01_roc')")
    args = p.parse_args()

    filter_ids = parse_rows_arg(args.rows)
    rows = discover_rows(filter_ids)
    if not rows and args.stage != "report":
        console.print("[red]No rows match filter.[/red]")
        return 1

    if not shutil.which("uv"):
        console.print("[yellow]warning: 'uv' not found on PATH; comparator panels will fail.[/yellow]")

    if args.stage in ("generate", "all"):
        stage_generate(rows)
    if args.stage in ("judge", "all"):
        stage_judge(rows)
    if args.stage in ("report", "all"):
        stage_report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
