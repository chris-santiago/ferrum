"""The docs-image manifest agrees with the docs pages, without rendering anything.

``scripts/regen_docs_images.py`` regenerates every docs image through its
existing generator and rewrites ``docs/images-manifest.json``. These tests pin
what the driver promises: every image a page references has a record and a
generator, every image on disk is referenced (brand assets aside), guide block
images are named by their page, and each generator writes inside the repo
whatever the working directory.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from types import ModuleType

import pytest

from tests._script_loader import load_script

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"


@pytest.fixture(scope="module")
def driver() -> ModuleType:
    return load_script("regen_docs_images", SCRIPTS / "regen_docs_images.py")


@pytest.fixture(scope="module")
def guide() -> ModuleType:
    return load_script("generate_guide_pngs", SCRIPTS / "generate-guide-pngs.py")


@pytest.fixture(scope="module")
def references(driver: ModuleType) -> dict[str, list[str]]:
    return driver.collect_references()


# ── the committed manifest ────────────────────────────────────────────────────


def test_manifest_matches_page_references_and_disk(driver: ModuleType, references) -> None:
    errors = driver.check_manifest(driver.load_manifest(), references, driver.images_on_disk())
    assert errors == []


def test_check_mode_exits_nonzero_when_a_referenced_image_has_no_record(
    driver: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    records = driver.load_manifest()
    dropped = records.pop()
    manifest = tmp_path / "images-manifest.json"
    manifest.write_text(json.dumps(records))
    monkeypatch.setattr(driver, "MANIFEST", manifest)
    assert driver.main(["--check"]) == 1
    assert dropped["image"] in capsys.readouterr().err


# ── run checks ────────────────────────────────────────────────────────────────


def _all_produced(driver: ModuleType, references) -> dict[str, str]:
    return {
        image: f"scripts/example.py:{Path(image).stem}"
        for image in references
        if image not in driver.EXEMPT and image != _HIST_DENSITY
    }


def test_clean_run_has_no_errors(driver: ModuleType, references) -> None:
    produced = _all_produced(driver, references)
    assert driver.check_run(references, produced, set(produced)) == []


def test_referenced_image_no_generator_produced_is_an_error_naming_it(
    driver: ModuleType, references
) -> None:
    produced = _all_produced(driver, references)
    missing = "docs/site/gallery/img/mark_point.png"
    del produced[missing]
    errors = driver.check_run(references, produced, set(produced) | {missing})
    assert len(errors) == 1
    assert errors[0].startswith(missing)
    assert "no generator produced it" in errors[0]


def test_produced_image_no_page_references_is_an_error(driver: ModuleType, references) -> None:
    produced = _all_produced(driver, references)
    orphan = "docs/site/gallery/img/not_on_any_page.png"
    produced[orphan] = "scripts/example.py:not_on_any_page"
    errors = driver.check_run(references, produced, set(produced))
    assert errors == [
        f"{orphan}: produced by scripts/example.py:not_on_any_page but no page references it"
    ]


def test_unreferenced_unproduced_image_on_disk_is_an_error(driver: ModuleType, references) -> None:
    produced = _all_produced(driver, references)
    stray = "docs/site/guide/img/composition_09.png"
    errors = driver.check_run(references, produced, set(produced) | {stray})
    assert errors == [f"{stray}: on disk, but no page references it and no generator produces it"]


def test_brand_and_social_assets_are_exempt(driver: ModuleType, references) -> None:
    produced = _all_produced(driver, references)
    on_disk = set(produced) | set(driver.EXEMPT)
    assert driver.check_run(references, produced, on_disk) == []
    assert "docs/site/assets/images/social-card.png" in driver.EXEMPT


def test_record_shape_and_inspection_carry_over(driver: ModuleType, references) -> None:
    image = "docs/site/gallery/img/mark_point.png"
    other = "docs/site/gallery/img/mark_line.png"
    produced = {
        image: "design-docs/demos/demo_grammar_marks.py:mark_point",
        other: "design-docs/demos/demo_grammar_marks.py:mark_line",
    }
    previous = [
        {"image": image, "pages": [], "generator": "", "inspected": "2026-10-12", "verdict": "ok"},
        {"image": other, "pages": [], "generator": "", "inspected": "2026-10-12", "verdict": "ok"},
    ]
    records = driver.build_records(references, produced, previous, unchanged={image})
    assert [r for r in records if r["image"] in produced] == [
        {
            "image": other,
            "pages": ["docs/site/gallery/gallery.md"],
            "generator": "design-docs/demos/demo_grammar_marks.py:mark_line",
            "inspected": "",
            "verdict": "",
        },
        {
            "image": image,
            "pages": ["docs/site/gallery/gallery.md"],
            "generator": "design-docs/demos/demo_grammar_marks.py:mark_point",
            "inspected": "2026-10-12",
            "verdict": "ok",
        },
    ]


def test_written_between_reports_new_and_rewritten_files(driver: ModuleType) -> None:
    before = {"a.png": 1, "b.png": 2}
    after = {"a.png": 1, "b.png": 3, "c.png": 4}
    assert driver.written_between(before, after) == ["b.png", "c.png"]


# ── guide block names ─────────────────────────────────────────────────────────

_PAGE = """\
```python
a = 1
first
```

Some prose.

![First](img/first_chart.png)

```python
helper = 2
```

```python
second
```

![Second](img/second_chart.png)
"""


def _names(guide: ModuleType, text: str) -> list[str | None]:
    return [block.image for block in guide.extract_blocks(text)]


def test_block_is_named_by_the_image_its_page_references_after_it(guide: ModuleType) -> None:
    blocks = guide.extract_blocks(_PAGE)
    assert [(b.image, b.display) for b in blocks] == [
        ("first_chart.png", "first"),
        (None, None),
        ("second_chart.png", "second"),
    ]


def test_editing_another_block_does_not_rename_a_block(guide: ModuleType) -> None:
    edited = _PAGE.replace("helper = 2\n", "helper = 3\nmore = 4\n")
    edited = "```python\nnew_first_block = 0\nnew_first_block\n```\n\n" + edited
    assert _names(guide, edited) == [None, "first_chart.png", None, "second_chart.png"]


def test_every_guide_page_image_is_named_by_a_displaying_block(guide: ModuleType) -> None:
    """Each ``img/`` image on a guide page is rendered by exactly one block that displays a chart."""
    # Drawn by scripts/generate-theme-grid.py, above the page's first block.
    not_from_blocks = {"theme-grid.png"}
    for page in guide.PAGES:
        text = page.read_text(encoding="utf-8")
        referenced = set(re.findall(r"\]\(img/([^)\s]+\.png)\)", text)) - not_from_blocks
        named = [b for b in guide.extract_blocks(text) if b.image]
        assert sorted(b.image for b in named) == sorted(referenced), page.name
        assert all(b.display for b in named), page.name


# ── generator fixes ───────────────────────────────────────────────────────────


def test_theme_variants_are_shown_on_the_themes_guide(references) -> None:
    gallery = ROOT / "docs" / "site" / "gallery" / "img"
    variants = sorted(
        f"docs/site/gallery/img/{p.name}"
        for p in gallery.glob("*.png")
        if re.search(r"_(paper_ink|slate_citrus|arctic_signal)\.png$", p.name)
    )
    assert len(variants) == 21
    for image in variants:
        assert "docs/site/guide/themes.md" in references.get(image, []), image


def test_every_showcase_entry_is_on_the_showcase_page(references) -> None:
    showcase = load_script("generate_showcase_pngs", SCRIPTS / "generate-showcase-pngs.py")
    for name, _ in showcase.CHARTS:
        image = f"docs/site/gallery/showcase_img/{name}.png"
        assert references.get(image) == ["docs/site/gallery/showcase.md"], image


def test_concept_generator_writes_inside_this_repo() -> None:
    concepts = load_script("gen_concept_pngs", SCRIPTS / "gen_concept_pngs.py")
    assert concepts.BASE == ROOT / "docs" / "site" / "assets" / "concepts"


def test_no_generator_hardcodes_an_absolute_or_working_directory_path(driver: ModuleType) -> None:
    for generator in driver.GENERATORS:
        text = (ROOT / generator.script).read_text(encoding="utf-8")
        assert not re.search(r"""["'](/Users/|/home/)""", text), generator.script
        assert not re.search(r"""Path\(["']docs/""", text), generator.script


# ── known failure ─────────────────────────────────────────────────────────────

_HIST_DENSITY = "docs/site/gallery/img/comp_histogram_density.png"


def test_known_failure_is_tolerated_only_while_it_fails(
    driver: ModuleType, references, monkeypatch: pytest.MonkeyPatch
) -> None:
    produced = _all_produced(driver, references)
    assert _HIST_DENSITY in references and _HIST_DENSITY not in produced
    assert driver.check_run(references, produced, set(produced) | {_HIST_DENSITY}) == []

    produced[_HIST_DENSITY] = "design-docs/demos/demo_grammar_marks.py:comp_histogram_density"
    (error,) = driver.check_run(references, produced, set(produced))
    assert error.startswith(_HIST_DENSITY) and "renders again" in error and "#511" in error

    monkeypatch.setattr(driver, "KNOWN_FAILURES", ())
    del produced[_HIST_DENSITY]
    (error,) = driver.check_run(references, produced, set(produced) | {_HIST_DENSITY})
    assert error.startswith(_HIST_DENSITY) and "no generator produced it" in error


def test_known_failure_record_blocks_with_a_verdict_citing_its_issue(
    driver: ModuleType, references
) -> None:
    (record,) = [r for r in driver.load_manifest() if r["image"] == _HIST_DENSITY]
    assert record["verdict"] != "ok" and "#511" in record["verdict"]

    built = driver.build_records(references, {}, [], unchanged=set())
    assert [r for r in built if r["image"] == _HIST_DENSITY] == [record]

    records = [
        dict(r, verdict="ok") if r["image"] == _HIST_DENSITY else r for r in driver.load_manifest()
    ]
    errors = driver.check_manifest(records, references, driver.images_on_disk())
    assert len(errors) == 1 and "needs a non-ok verdict citing it" in errors[0]


def test_check_mode_prints_the_known_failure(driver: ModuleType, capsys) -> None:
    driver.main(["--check"])
    assert "KNOWN FAILURE #511: docs/site/gallery/img/comp_histogram_density.png" in (
        capsys.readouterr().out
    )


# ── kept recipe sources ───────────────────────────────────────────────────────


def test_kept_recipe_sources_render_nothing_and_every_rendered_recipe_is_shown(references) -> None:
    recipes = load_script("render_recipe_pngs", SCRIPTS / "render-recipe-pngs.py")
    rendered = {p.stem for p in recipes.recipes_to_render()}
    all_sources = {p.stem for p in recipes.RECIPE_DIR.glob("*.py")}
    assert set(recipes.KEPT_UNSHOWN) == {"dashboard_inset", "highlight_region"}
    for stem, reason in recipes.KEPT_UNSHOWN.items():
        assert stem in all_sources and stem not in rendered and reason
        assert f"docs/site/assets/recipes/{stem}.png" not in references
    assert rendered == all_sources - set(recipes.KEPT_UNSHOWN)
    for stem in rendered:
        assert f"docs/site/assets/recipes/{stem}.png" in references, stem


def test_readme_is_scanned_for_image_references(references) -> None:
    assert "README.md" in references["docs/site/assets/images/hero-strip.png"]


# ── check_manifest branches ───────────────────────────────────────────────────


def _record(image: str, **overrides: object) -> dict:
    record = {
        "image": image,
        "pages": ["docs/site/gallery/gallery.md"],
        "generator": "design-docs/demos/demo_grammar_marks.py:mark_point",
        "inspected": "",
        "verdict": "",
    }
    return {**record, **overrides}


_POINT = "docs/site/gallery/img/mark_point.png"


@pytest.mark.parametrize(
    ("records", "on_disk", "expected"),
    [
        ([_record(_POINT)], set(), f"{_POINT}: has a record but the file does not exist"),
        (
            [_record(_POINT, pages=["docs/site/index.md"])],
            {_POINT},
            f"{_POINT}: record pages ['docs/site/index.md'] != referenced by",
        ),
        ([_record(_POINT), _record(_POINT)], {_POINT}, f"{_POINT}: more than one record"),
        (
            [_record(_POINT, generator="scripts/nope.py:mark_point")],
            {_POINT},
            f"{_POINT}: generator script scripts/nope.py does not exist",
        ),
        (
            [_record(_POINT, extra=1)],
            {_POINT},
            f"manifest record 0 ({_POINT}): fields are",
        ),
        (["not a record"], set(), "manifest record 0: expected an object, got str"),
        (
            [_record("docs/site/gallery/img/nowhere.png")],
            {"docs/site/gallery/img/nowhere.png"},
            "docs/site/gallery/img/nowhere.png: has a record but no page references it",
        ),
    ],
)
def test_check_manifest_branches(driver: ModuleType, records, on_disk, expected: str) -> None:
    references = {_POINT: ["docs/site/gallery/gallery.md"]}
    errors = driver.check_manifest(records, references, on_disk)
    assert any(error.startswith(expected) for error in errors), errors


def test_check_manifest_reports_an_unreferenced_image_on_disk(driver: ModuleType) -> None:
    references = {_POINT: ["docs/site/gallery/gallery.md"]}
    stray = "docs/site/gallery/img/stray.png"
    errors = driver.check_manifest([_record(_POINT)], references, {_POINT, stray})
    assert errors == [f"{stray}: on disk, but no page references it"]


def test_check_manifest_refuses_a_non_list_manifest(driver: ModuleType) -> None:
    assert driver.check_manifest({"image": _POINT}, {}, set()) == [
        "manifest: expected a JSON list of records, got dict"
    ]


# ── generator runs, in a scratch repo with stub generators ────────────────────


@pytest.fixture
def sandbox(driver: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the driver at an empty scratch repo with no generators and no known failures."""
    root = tmp_path.resolve()
    (root / "docs" / "site").mkdir(parents=True)
    (root / "zensical.toml").write_text("")
    monkeypatch.setattr(driver, "ROOT", root)
    monkeypatch.setattr(driver, "SITE", root / "docs" / "site")
    monkeypatch.setattr(driver, "ZENSICAL_TOML", root / "zensical.toml")
    monkeypatch.setattr(driver, "README", root / "README.md")
    monkeypatch.setattr(driver, "MANIFEST", root / "docs" / "images-manifest.json")
    monkeypatch.setattr(driver, "GENERATORS", ())
    monkeypatch.setattr(driver, "KNOWN_FAILURES", ())
    return root


def _stub(root: Path, name: str, body: str) -> str:
    """Write a stub generator script under ``root/gen`` and return its repo-relative path."""
    script = root / "gen" / name
    script.parent.mkdir(exist_ok=True)
    script.write_text(
        "import sys\nfrom pathlib import Path\nROOT = Path(__file__).resolve().parent.parent\n"
        + body
    )
    return f"gen/{name}"


def _write(path: str) -> str:
    return f"(ROOT / {path!r}).parent.mkdir(parents=True, exist_ok=True)\n(ROOT / {path!r}).write_bytes(b'png')\n"


def test_run_generator_attributes_its_site_writes_to_it(
    driver: ModuleType, sandbox: Path, tmp_path
) -> None:
    script = _stub(sandbox, "draw.py", _write("docs/site/img/a.png"))
    run = driver._run_generator(driver.Generator(script), tmp_path)
    assert run.outputs == [("docs/site/img/a.png", "gen/draw.py:a")]
    assert run.errors == [] and run.side_written == []


def test_run_generator_reports_a_nonzero_exit(driver: ModuleType, sandbox: Path, tmp_path) -> None:
    script = _stub(sandbox, "fail.py", "sys.exit(3)\n")
    run = driver._run_generator(driver.Generator(script), tmp_path)
    assert run.errors == ["gen/fail.py exited with status 3"]


def _placed_generator(driver: ModuleType, sandbox: Path, body: str):
    source = sandbox / "demo" / "out"
    source.mkdir(parents=True)
    (source / "stale.png").write_bytes(b"old")
    os.utime(source / "stale.png", ns=(1, 1))
    script = _stub(sandbox, "demo.py", body)
    return driver.Generator(
        script,
        (driver.Placement("demo/out", "docs/site/gallery/img"),),
        side_outputs=("demo/out",),
    )


def test_run_generator_copies_fresh_placements_and_flags_stale_ones(
    driver: ModuleType, sandbox: Path, tmp_path
) -> None:
    generator = _placed_generator(driver, sandbox, _write("demo/out/fresh.png"))
    run = driver._run_generator(generator, tmp_path)
    assert run.outputs == [("docs/site/gallery/img/fresh.png", "gen/demo.py:fresh")]
    assert (sandbox / "docs/site/gallery/img/fresh.png").read_bytes() == b"png"
    assert not (sandbox / "docs/site/gallery/img/stale.png").exists()
    assert run.errors == ["demo/out/stale.png: not rewritten by this run of gen/demo.py"]
    assert run.side_written == ["demo/out/fresh.png"]


def test_run_generator_lets_a_known_failure_source_stay_stale(
    driver: ModuleType, sandbox: Path, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    generator = _placed_generator(driver, sandbox, "")
    failure = driver.KnownFailure(
        image="docs/site/gallery/img/stale.png",
        source="demo/out/stale.png",
        generator="gen/demo.py:stale",
        issue="#0",
        verdict="regression #0",
    )
    monkeypatch.setattr(driver, "KNOWN_FAILURES", (failure,))
    run = driver._run_generator(generator, tmp_path)
    assert run.errors == [] and run.outputs == []


def test_record_outputs_flags_an_image_two_generators_produced(driver: ModuleType) -> None:
    produced = {"docs/site/img/a.png": "gen/one.py:a"}
    errors = driver.record_outputs(produced, [("docs/site/img/a.png", "gen/two.py:a")])
    assert errors == ["docs/site/img/a.png: produced by both gen/one.py:a and gen/two.py:a"]


def test_regenerate_writes_the_manifest_of_a_clean_run(
    driver: ModuleType, sandbox: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (sandbox / "docs/site/page.md").write_text("![A](img/a.png)\n")
    script = _stub(sandbox, "draw.py", _write("docs/site/img/a.png"))
    monkeypatch.setattr(driver, "GENERATORS", (driver.Generator(script),))
    assert driver.regenerate() == 0
    assert json.loads((sandbox / "docs/images-manifest.json").read_text()) == [
        {
            "image": "docs/site/img/a.png",
            "pages": ["docs/site/page.md"],
            "generator": "gen/draw.py:a",
            "inspected": "",
            "verdict": "",
        }
    ]
    assert driver.check() == 0


def test_regenerate_fails_on_two_producers_and_writes_no_manifest(
    driver: ModuleType, sandbox: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    (sandbox / "docs/site/page.md").write_text("![A](img/a.png)\n")
    one = _stub(sandbox, "one.py", _write("docs/site/img/a.png"))
    two = _stub(sandbox, "two.py", _write("docs/site/img/a.png"))
    monkeypatch.setattr(driver, "GENERATORS", (driver.Generator(one), driver.Generator(two)))
    assert driver.regenerate() == 1
    assert "produced by both gen/one.py:a and gen/two.py:a" in capsys.readouterr().err
    assert not (sandbox / "docs/images-manifest.json").exists()


def test_regenerate_refuses_a_malformed_manifest(driver: ModuleType, sandbox: Path, capsys) -> None:
    (sandbox / "docs/images-manifest.json").write_text('{"image": "x"}')
    assert driver.regenerate() == 1
    assert "expected a JSON list of records, got dict" in capsys.readouterr().err
