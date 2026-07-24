#!/usr/bin/env python
"""Regenerate tests/_fixtures/scale_wire_baseline.json from a tagged ref.

Usage:
    python scripts/regen-scale-wire-baseline.py               # regen from the latest v* tag
    python scripts/regen-scale-wire-baseline.py v0.20.2        # regen from a specific ref
    python scripts/regen-scale-wire-baseline.py --check        # print provenance, write nothing

Why a worktree, not the working tree
-------------------------------------
``tests/test_scale_spec_parity.py``'s ``TestByteIdentity`` compares today's
``Chart.to_json()`` output against this fixture to catch accidental drift in
the ``*Scale`` wire format. If the fixture were regenerated from the CURRENT
working tree, the guard becomes a tautology -- whatever the working tree
currently emits always matches "the baseline," even if the wire format
silently changed underneath a refactor. This script instead builds and
captures from an isolated ``git worktree`` checkout of a known-good tagged
ref, so "baseline" means "what a specific frozen commit produced," never
"whatever HEAD happens to produce right now."

Build approach
---------------
Each worktree gets its OWN ``uv sync`` + ``maturin develop`` build, entirely
separate from the main checkout's ``.venv``. Reusing the main checkout's venv
would overwrite its installed ``ferrum._core`` with the ref's build -- exactly
the kind of environment cross-contamination this script exists to avoid, and
a real hazard if a regen is run while other work is in flight against the
main venv. ferrum's project convention otherwise avoids worktrees for
day-to-day coding (see CLAUDE.md) because a full venv setup costs more than
the isolation buys for routine edits; that tradeoff inverts here, since this
script runs rarely (only when the baseline legitimately needs to move
forward) and byte-for-byte provenance is the entire point.

The capture step runs the WORKTREE's own interpreter (``worktree/.venv/bin/
python``) as a subprocess rather than importing the worktree's ``ferrum``
into this process: two different compiled ``ferrum._core`` extensions cannot
coexist in one interpreter, and this script's own process may already have
the main tree's ``ferrum`` on ``sys.path``. That subprocess loads
``_build_baseline_charts`` directly from the WORKTREE's copy of
``tests/test_scale_spec_parity.py`` (not the main tree's), so the generator
and the byte-identity guard always agree on which chart constructions
produced the baseline at that ref.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_FIXTURE_PATH = _REPO_ROOT / "tests" / "_fixtures" / "scale_wire_baseline.json"

# Run inside the worktree's own interpreter (see module docstring) via a temp
# file -- passed the worktree's tests/test_scale_spec_parity.py path so it
# loads that ref's chart constructions, not this script's.
_CAPTURE_SCRIPT = """
import importlib.util
import json
import sys
from pathlib import Path

module_path = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location("_scale_wire_capture", module_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

charts = module._build_baseline_charts()
payload = {name: chart.to_json() for name, chart in charts.items()}
print(json.dumps(payload))
"""


def _run(cmd: list[str], *, cwd: Path | None = None, env: dict | None = None) -> str:
    result = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def _latest_tag() -> str:
    return _run(["git", "describe", "--tags", "--abbrev=0"], cwd=_REPO_ROOT)


def _resolve_sha(ref: str) -> str:
    try:
        return _run(["git", "rev-parse", ref], cwd=_REPO_ROOT)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"unknown git ref {ref!r}: {exc.stderr.strip()}") from exc


def _add_worktree(ref: str) -> Path:
    """Create a fresh temp-dir git worktree checked out (detached) at *ref*.

    If ``git worktree add`` itself fails, the temp dir it was created in is
    removed before the error propagates -- otherwise a bad ref would leak an
    empty ``ferrum-scale-wire-regen-*`` directory in the system temp dir.
    """
    tmp_dir = Path(tempfile.mkdtemp(prefix="ferrum-scale-wire-regen-"))
    worktree = tmp_dir / "worktree"
    try:
        subprocess.run(
            ["git", "worktree", "add", "--detach", str(worktree), ref], cwd=_REPO_ROOT, check=True
        )
    except subprocess.CalledProcessError:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    return worktree


def _remove_worktree(worktree: Path) -> None:
    subprocess.run(
        ["git", "worktree", "remove", "--force", str(worktree)], cwd=_REPO_ROOT, check=True
    )
    parent = worktree.parent
    if parent.exists() and not any(parent.iterdir()):
        parent.rmdir()


def _build_worktree_extension(worktree: Path) -> None:
    """Sync an isolated venv and compile the Rust extension for *worktree*."""
    subprocess.run(["uv", "sync"], cwd=worktree, check=True)
    env = os.environ.copy()
    env.pop("CONDA_PREFIX", None)  # see CLAUDE.md's maturin develop note
    subprocess.run(
        ["uv", "run", "--no-sync", "maturin", "develop"], cwd=worktree, env=env, check=True
    )


def _capture_baseline(worktree: Path) -> dict[str, str]:
    """Run the worktree's own interpreter to build and serialize every baseline chart."""
    python = worktree / ".venv" / "bin" / "python"
    module_path = worktree / "tests" / "test_scale_spec_parity.py"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(_CAPTURE_SCRIPT)
        capture_script = Path(f.name)
    try:
        output = _run([str(python), str(capture_script), str(module_path)], cwd=worktree)
    finally:
        capture_script.unlink(missing_ok=True)
    return json.loads(output)


def regenerate(ref: str) -> None:
    """Build the extension at *ref* in an isolated worktree and rewrite the fixture."""
    sha = _resolve_sha(ref)
    rel_fixture = _FIXTURE_PATH.relative_to(_REPO_ROOT)
    print(f"regenerating {rel_fixture} from {ref} ({sha})", file=sys.stderr)

    worktree = _add_worktree(ref)
    try:
        _build_worktree_extension(worktree)
        baseline = _capture_baseline(worktree)
    finally:
        # Cleanup failure here must never mask a real exception from the try
        # block above (e.g. a maturin build failure) -- swallow it (with a
        # warning) rather than letting it propagate and replace the original.
        try:
            _remove_worktree(worktree)
        except subprocess.CalledProcessError as exc:
            print(f"warning: failed to remove worktree {worktree}: {exc}", file=sys.stderr)

    baseline["_provenance"] = {
        "ref": ref,
        "captured": sha,
        "script": "scripts/regen-scale-wire-baseline.py",
    }
    _FIXTURE_PATH.write_text(json.dumps(baseline, indent=2, sort_keys=True) + "\n")
    print(f"wrote {rel_fixture}", file=sys.stderr)


def check() -> int:
    """Print the existing fixture's provenance and return 0, or 1 if missing/stale."""
    if not _FIXTURE_PATH.exists():
        print(f"missing: {_FIXTURE_PATH}", file=sys.stderr)
        return 1
    payload = json.loads(_FIXTURE_PATH.read_text())
    provenance = payload.get("_provenance")
    if provenance is None:
        print(
            f"{_FIXTURE_PATH}: no _provenance key -- regenerate with "
            "`python scripts/regen-scale-wire-baseline.py`",
            file=sys.stderr,
        )
        return 1
    print(f"{_FIXTURE_PATH}: ref={provenance['ref']} captured={provenance['captured']}")
    return 0


def main() -> int:
    """CLI entry point: regenerate the fixture from *ref*, or ``--check`` its provenance."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "ref", nargs="?", default=None, help="Git ref to regenerate from (default: latest v* tag)."
    )
    parser.add_argument(
        "--check", action="store_true", help="Verify the fixture's provenance, write nothing."
    )
    args = parser.parse_args()

    if args.check:
        # --check only inspects the fixture already on disk -- it never
        # rebuilds anything, so a ref argument (which only makes sense for a
        # regeneration run) has nothing to do. Reject the combination rather
        # than silently ignoring the ref, which would look like --check
        # verified that specific ref when it did not.
        if args.ref is not None:
            parser.error("--check takes no ref argument (it only inspects the existing fixture)")
        return check()

    ref = args.ref or _latest_tag()
    regenerate(ref)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
