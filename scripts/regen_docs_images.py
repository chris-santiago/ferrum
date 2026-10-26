"""Regenerate every docs image through its existing generator and rewrite the image manifest.

    unset CONDA_PREFIX && uv run --no-sync python scripts/regen_docs_images.py
    unset CONDA_PREFIX && uv run --no-sync python scripts/regen_docs_images.py --check

The driver contains no chart code. It runs each generator in ``GENERATORS`` in
its own process, from a scratch working directory so that no generator can
write files relative to the caller's directory. It records which docs images
each generator wrote, copies the design-docs demo outputs that the gallery
shows into ``docs/site/gallery/img/``, and then checks the result against every
image reference in the docs pages, the page templates, ``zensical.toml`` and
``README.md``:

- an image a page references that no generator produced is an error;
- an image a generator produced that no page references is an error;
- an image on disk under ``docs/site/`` that is neither is an error;
- an image two generators both produced is an error;
- a demo output that this run did not rewrite (its entry failed) is an error;
- a generator that exits non-zero is an error.

Running the driver rewrites tracked files outside ``docs/``: the three
design-docs demos regenerate their HTML galleries and every PNG beside them
(``design-docs/demos/demo_grammar_marks.html``, ``demo_grammar_pngs/``,
``demo_diagnostics.html``, ``demo_diag_pngs/``,
``design-docs/themes/demo_themes.html``, ``demo_pngs/``), including the
demo-only charts that no docs page shows. These are declared per generator in
``Generator.side_outputs``, and the run summary lists every one it rewrote.

The brand and social assets in ``EXEMPT`` are outside these rules: they are
drawn by hand, or by ``scripts/social-card.py``, which needs the network and
stays outside the driver.

An image in ``KNOWN_FAILURES`` belongs to a generator entry that is known to
fail, tracked by an open issue. Its card stays on its page, its stale image
stays on disk, every run prints it, and its manifest record carries a non-ok
verdict citing the issue, so release acceptance still blocks on it. The entry
expires by itself: once the image renders again, the run fails until the
entry is removed.

When every check passes the driver rewrites ``docs/images-manifest.json``,
one record per docs image (see ``ManifestRecord``). An image whose bytes did
not change keeps its ``inspected`` and ``verdict``; a changed or new image gets
them cleared, because an inspection only vouches for the render it looked at.

``--check`` renders nothing: it checks the committed manifest against the
current references and the images on disk.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypedDict

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "docs" / "site"
ZENSICAL_TOML = ROOT / "zensical.toml"
README = ROOT / "README.md"
MANIFEST = ROOT / "docs" / "images-manifest.json"

IMAGE_SUFFIXES = frozenset({".png", ".svg", ".jpg", ".jpeg", ".gif", ".webp"})

#: Brand and social assets: no generator in the driver produces them, and
#: they need no page reference (the theme config and the page template use
#: some of them).
EXEMPT = frozenset(
    f"docs/site/assets/images/{name}"
    for name in (
        "favicon-b-180.png",
        "favicon-b-32.png",
        "favicon-b-64.png",
        "favicon.svg",
        "hero.svg",
        "hero-strip.png",
        "lockup-a-dark.svg",
        "lockup-a-light.svg",
        "lockup-b-light.svg",
        "mark-a.svg",
        "mark-b.svg",
        "social-card.png",
    )
)

_GALLERY_IMG = "docs/site/gallery/img"


@dataclass(frozen=True)
class Placement:
    """Copy a generator's out-of-site PNGs into a docs image directory.

    Every ``*.png`` in ``source`` is copied to ``target`` except the stems in
    ``exclude``.
    """

    source: str
    target: str
    exclude: frozenset[str] = frozenset()


@dataclass(frozen=True)
class Generator:
    """One existing image script, where its out-of-site outputs go, and what else it rewrites.

    ``side_outputs`` are the repo-relative files and directories outside
    ``docs/`` that the script rewrites as it runs.
    """

    script: str
    placements: tuple[Placement, ...] = field(default=())
    side_outputs: tuple[str, ...] = field(default=())


@dataclass(frozen=True)
class KnownFailure:
    """A referenced docs image whose generator entry fails, tracked by an open issue.

    ``source`` is the generator output the image is placed from; ``generator``
    and ``verdict`` become the image's manifest record. ``verdict`` must cite
    ``issue`` and is never ``ok``.
    """

    image: str
    source: str
    generator: str
    issue: str
    verdict: str


GENERATORS: tuple[Generator, ...] = (
    Generator(
        "design-docs/demos/demo_grammar_marks.py",
        (
            Placement(
                "design-docs/demos/demo_grammar_pngs",
                _GALLERY_IMG,
                # Rendered for the design-docs HTML demo only; no gallery card shows them.
                exclude=frozenset({"mark_errorband", "mark_function"}),
            ),
        ),
        side_outputs=(
            "design-docs/demos/demo_grammar_marks.html",
            "design-docs/demos/demo_grammar_pngs",
        ),
    ),
    Generator(
        "design-docs/demos/demo_diagnostics.py",
        (
            Placement(
                "design-docs/demos/demo_diag_pngs",
                _GALLERY_IMG,
                exclude=frozenset(
                    {
                        # Rendered for the design-docs HTML demo only; no gallery card shows them.
                        "class_balance_chart",
                        "classification_report_chart",
                        "cooks_distance_chart",
                        "elbow_chart",
                        "manifold_chart",
                        "silhouette_chart",
                        # The gallery cards come from generate-gallery-diagnostic-pngs.py.
                        "roc_chart",
                        "pr_chart",
                        "calibration_chart",
                    }
                ),
            ),
        ),
        side_outputs=(
            "design-docs/demos/demo_diagnostics.html",
            "design-docs/demos/demo_diag_pngs",
        ),
    ),
    Generator(
        "design-docs/themes/demo_themes.py",
        (Placement("design-docs/themes/demo_pngs", _GALLERY_IMG),),
        side_outputs=("design-docs/themes/demo_themes.html", "design-docs/themes/demo_pngs"),
    ),
    Generator("scripts/generate-gallery-diagnostic-pngs.py"),
    Generator("scripts/generate-guide-pngs.py"),
    Generator("scripts/gen_concept_pngs.py"),
    Generator("scripts/render-recipe-pngs.py"),
    Generator("scripts/generate-showcase-pngs.py"),
    Generator("scripts/generate-theme-grid.py"),
)

KNOWN_FAILURES: tuple[KnownFailure, ...] = ()


class ManifestRecord(TypedDict):
    """One docs image in ``docs/images-manifest.json``.

    ``image`` is the repo-relative image path; ``pages`` the repo-relative docs
    pages that reference it (non-empty); ``generator`` the producing script and
    entry as ``<script>:<entry>``, where the entry is the image's stem (the
    registry entry, demo slug, recipe, or guide block name); ``inspected`` the
    date of the PNG inspection of the current render, or empty; ``verdict``
    ``ok`` or a one-line defect description, or empty before inspection.
    """

    image: str
    pages: list[str]
    generator: str
    inspected: str
    verdict: str


_RECORD_FIELDS = frozenset(ManifestRecord.__annotations__)


# ── references ────────────────────────────────────────────────────────────────

_PAGE_REF_RE = re.compile(r'(?:\]\(|src=")([^)"\s]+)')
_TOML_REF_RE = re.compile(r'^\s*(?:logo|favicon)\s*=\s*"([^"]+)"', re.MULTILINE)


def _rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def _is_image(target: str) -> bool:
    return Path(target).suffix.lower() in IMAGE_SUFFIXES


def _site_images() -> list[Path]:
    """Return every image file under ``docs/site/``."""
    return [p for p in SITE.rglob("*") if p.is_file() and _is_image(p.name)]


def collect_references() -> dict[str, list[str]]:
    """Map every image referenced by a docs page, template, the site config or the README to its referrers.

    Markdown and HTML references resolve relative to their file; the
    ``logo``/``favicon`` entries of ``zensical.toml`` resolve relative to the
    docs directory. External URLs and template expressions are ignored.
    """
    refs: dict[str, set[str]] = {}

    def add(target: str, base: Path, referrer: str) -> None:
        if "://" in target or "{{" in target or not _is_image(target):
            return
        refs.setdefault(_rel(base / target), set()).add(referrer)

    pages = sorted([*SITE.rglob("*.md"), *SITE.rglob("*.html")])
    if README.exists():
        pages.append(README)
    for page in pages:
        for match in _PAGE_REF_RE.finditer(page.read_text(encoding="utf-8")):
            add(match.group(1), page.parent, _rel(page))
    for match in _TOML_REF_RE.finditer(ZENSICAL_TOML.read_text(encoding="utf-8")):
        add(match.group(1), SITE, _rel(ZENSICAL_TOML))
    return {image: sorted(referrers) for image, referrers in sorted(refs.items())}


def images_on_disk() -> set[str]:
    """Return every image file under ``docs/site/``, repo-relative."""
    return {_rel(p) for p in _site_images()}


# ── checks ────────────────────────────────────────────────────────────────────


def _known_failures() -> dict[str, KnownFailure]:
    return {failure.image: failure for failure in KNOWN_FAILURES}


def check_run(
    references: Mapping[str, Iterable[str]],
    produced: Mapping[str, str],
    on_disk: set[str],
) -> list[str]:
    """Return one error per broken image after a run; empty when the docs images are consistent.

    ``produced`` maps each image a generator wrote to its generator field.
    """
    known = _known_failures()
    errors = []
    for image, pages in references.items():
        if image not in EXEMPT and image not in produced and image not in known:
            errors.append(f"{image}: referenced by {', '.join(pages)} but no generator produced it")
    for image, generator in produced.items():
        if image in known:
            errors.append(
                f"{image}: renders again, so known failure {known[image].issue} is fixed;"
                " remove its KNOWN_FAILURES entry"
            )
        elif image not in references and image not in EXEMPT:
            errors.append(f"{image}: produced by {generator} but no page references it")
    for image in sorted(set(known) - set(references)):
        errors.append(f"{image}: known failure {known[image].issue} but no page references it")
    for image in sorted(on_disk - set(references) - set(produced) - EXEMPT):
        errors.append(f"{image}: on disk, but no page references it and no generator produces it")
    return errors


def manifest_shape_errors(raw: Any) -> list[str]:
    """Return one error per malformed part of a decoded manifest.

    ``raw`` is the decoded JSON as read from disk: well formed, it is a list of
    objects with exactly the ``ManifestRecord`` fields.
    """
    if not isinstance(raw, list):
        return [f"manifest: expected a JSON list of records, got {type(raw).__name__}"]
    errors = []
    for index, record in enumerate(raw):
        if not isinstance(record, dict):
            errors.append(
                f"manifest record {index}: expected an object, got {type(record).__name__}"
            )
        elif set(record) != _RECORD_FIELDS:
            errors.append(
                f"manifest record {index} ({record.get('image')}): fields are {sorted(record)}"
            )
    return errors


def check_manifest(
    raw: Any,
    references: Mapping[str, Iterable[str]],
    on_disk: set[str],
) -> list[str]:
    """Return one error per way the manifest disagrees with the references and the disk.

    ``raw`` is the decoded manifest JSON; malformed records are reported and
    left out of the remaining checks.
    """
    errors = manifest_shape_errors(raw)
    if not isinstance(raw, list):
        return errors
    known = _known_failures()
    by_image: dict[str, ManifestRecord] = {}
    for record in raw:
        if not isinstance(record, dict) or set(record) != _RECORD_FIELDS:
            continue
        if record["image"] in by_image:
            errors.append(f"{record['image']}: more than one record")
        by_image[record["image"]] = record
    for image, record in by_image.items():
        if image not in on_disk:
            errors.append(f"{image}: has a record but the file does not exist")
        pages = sorted(references.get(image, ()))
        if not pages:
            errors.append(f"{image}: has a record but no page references it")
        elif record["pages"] != pages:
            errors.append(f"{image}: record pages {record['pages']} != referenced by {pages}")
        script = record["generator"].rsplit(":", 1)[0]
        if not (ROOT / script).is_file():
            errors.append(f"{image}: generator script {script} does not exist")
        failure = known.get(image)
        if failure and (record["verdict"] == "ok" or failure.issue not in record["verdict"]):
            errors.append(
                f"{image}: known failure {failure.issue} needs a non-ok verdict citing it,"
                f" got {record['verdict']!r}"
            )
    for image, pages in references.items():
        if image not in EXEMPT and image not in by_image:
            errors.append(f"{image}: referenced by {', '.join(pages)} but has no manifest record")
    for image in sorted(on_disk - set(references) - EXEMPT):
        errors.append(f"{image}: on disk, but no page references it")
    return errors


def load_manifest() -> Any:
    """Return the committed manifest's decoded JSON, or an empty list when it does not exist yet."""
    if not MANIFEST.exists():
        return []
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def build_records(
    references: Mapping[str, Iterable[str]],
    produced: Mapping[str, str],
    previous: Iterable[ManifestRecord],
    unchanged: set[str],
) -> list[ManifestRecord]:
    """Build the manifest of a clean run.

    An image in ``unchanged`` (bytes identical to before the run) keeps the
    ``inspected`` and ``verdict`` of its previous record; any other image
    starts uninspected. A referenced known failure gets its fixed generator
    and verdict.
    """
    kept = {r["image"]: r for r in previous if r["image"] in unchanged}
    records: list[ManifestRecord] = []
    for image in sorted(produced):
        if image not in references:
            continue
        old = kept.get(image)
        records.append(
            ManifestRecord(
                image=image,
                pages=sorted(references[image]),
                generator=produced[image],
                inspected=old["inspected"] if old else "",
                verdict=old["verdict"] if old else "",
            )
        )
    for image, failure in _known_failures().items():
        if image in references and image not in produced:
            records.append(
                ManifestRecord(
                    image=image,
                    pages=sorted(references[image]),
                    generator=failure.generator,
                    inspected="",
                    verdict=failure.verdict,
                )
            )
    return sorted(records, key=lambda record: record["image"])


# ── generation ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class GeneratorRun:
    """What one generator run produced.

    ``outputs`` pairs each docs image written with its generator field;
    ``side_written`` lists the declared out-of-site files it rewrote.
    """

    outputs: list[tuple[str, str]]
    errors: list[str]
    side_written: list[str]


def _stamps(paths: Iterable[Path]) -> dict[str, int]:
    return {_rel(p): p.stat().st_mtime_ns for p in paths}


def _side_files(generator: Generator) -> list[Path]:
    files = []
    for entry in generator.side_outputs:
        path = ROOT / entry
        if path.is_dir():
            files += [p for p in path.rglob("*") if p.is_file()]
        elif path.is_file():
            files.append(path)
    return files


def _digests() -> dict[str, str]:
    return {_rel(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in _site_images()}


def written_between(before: Mapping[str, int], after: Mapping[str, int]) -> list[str]:
    """Return the files created or rewritten between two ``mtime`` snapshots."""
    return sorted(path for path, stamp in after.items() if before.get(path) != stamp)


def _run_generator(generator: Generator, scratch: Path) -> GeneratorRun:
    """Run one generator and collect what it wrote, its placements and its errors."""
    known_sources = {failure.source: failure for failure in KNOWN_FAILURES}
    before = _stamps(_site_images())
    side_before = _stamps(_side_files(generator))
    started = time.time_ns()
    result = subprocess.run([sys.executable, str(ROOT / generator.script)], cwd=scratch)
    outputs = [
        (image, f"{generator.script}:{Path(image).stem}")
        for image in written_between(before, _stamps(_site_images()))
    ]
    errors = (
        [f"{generator.script} exited with status {result.returncode}"] if result.returncode else []
    )
    for placement in generator.placements:
        target = ROOT / placement.target
        target.mkdir(parents=True, exist_ok=True)
        for png in sorted((ROOT / placement.source).glob("*.png")):
            if png.stem in placement.exclude:
                continue
            if png.stat().st_mtime_ns < started:
                failure = known_sources.get(_rel(png))
                if failure is None:
                    # A left-over output of an earlier run: the entry failed or no longer exists.
                    errors.append(f"{_rel(png)}: not rewritten by this run of {generator.script}")
                else:
                    print(f"  known failure {failure.issue}: {_rel(png)} not rewritten")
                continue
            shutil.copyfile(png, target / png.name)
            outputs.append((_rel(target / png.name), f"{generator.script}:{png.stem}"))
    side_written = written_between(side_before, _stamps(_side_files(generator)))
    return GeneratorRun(outputs, errors, side_written)


def record_outputs(produced: dict[str, str], outputs: Iterable[tuple[str, str]]) -> list[str]:
    """Add a generator's outputs to ``produced``; return an error per image another generator made."""
    errors = []
    for image, source in outputs:
        if image in produced and produced[image] != source:
            errors.append(f"{image}: produced by both {produced[image]} and {source}")
        produced[image] = source
    return errors


def _announce_known_failures() -> None:
    for failure in KNOWN_FAILURES:
        print(f"KNOWN FAILURE {failure.issue}: {failure.image}: {failure.verdict}")


def regenerate() -> int:
    """Run every generator, check the docs images, and rewrite the manifest when clean."""
    _announce_known_failures()
    previous = load_manifest()
    shape_errors = manifest_shape_errors(previous)
    if shape_errors:
        # Rewriting a malformed manifest would silently drop its inspections.
        _report(shape_errors)
        return 1
    digests_before = _digests()
    produced: dict[str, str] = {}
    side_written: list[str] = []
    errors: list[str] = []
    with tempfile.TemporaryDirectory(prefix="regen-docs-images-") as scratch:
        for generator in GENERATORS:
            print(f"\n=== {generator.script}", flush=True)
            run = _run_generator(generator, Path(scratch))
            errors += run.errors
            errors += record_outputs(produced, run.outputs)
            side_written += run.side_written

    references = collect_references()
    errors += check_run(references, produced, images_on_disk())
    print(f"\n{len(side_written)} files rewritten outside docs/:")
    for path in side_written:
        print(f"  rewritten: {path}")
    if errors:
        _report(errors)
        return 1
    digests_after = _digests()
    unchanged = {img for img, digest in digests_after.items() if digests_before.get(img) == digest}
    records = build_records(references, produced, previous, unchanged)
    MANIFEST.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    changed = sorted(r["image"] for r in records if r["image"] not in unchanged)
    print(f"\n{len(records)} images; {len(changed)} changed bytes; manifest: {_rel(MANIFEST)}")
    for image in changed:
        print(f"  changed: {image}")
    return 0


def check() -> int:
    """Check the committed manifest against the references and the disk without rendering."""
    _announce_known_failures()
    errors = check_manifest(load_manifest(), collect_references(), images_on_disk())
    if errors:
        _report(errors)
        return 1
    print(f"OK: {_rel(MANIFEST)} matches every docs image reference.")
    return 0


def _report(errors: list[str]) -> None:
    print(f"\n{len(errors)} docs-image error(s):", file=sys.stderr)
    for error in errors:
        print(f"  {error}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and run the regeneration or the check."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--check", action="store_true", help="check the manifest without rendering")
    args = parser.parse_args(argv)
    return check() if args.check else regenerate()


if __name__ == "__main__":
    sys.exit(main())
