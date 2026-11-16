"""An interactive chart renders once: every HTML and JSON output reuses the scene built at construction.

Renders are counted at the Rust entry points (``render_interactive``, ``render_svg``) by
wrapping them, so a second render anywhere on the path is seen whatever Python layer made it.
"""

from __future__ import annotations

import contextlib
import json
import re
import warnings

import pandas as pd
import polars as pl
import pytest

import ferrum as fm
from ferrum import _core
from ferrum.display import html_string


@pytest.fixture
def counts(monkeypatch):
    """Per-entry render counters over the Rust render functions."""
    seen = {"interactive": 0, "svg": 0}
    real_interactive = _core.render_interactive
    real_svg = _core.render_svg

    def counting_interactive(*args, **kwargs):
        seen["interactive"] += 1
        return real_interactive(*args, **kwargs)

    def counting_svg(*args, **kwargs):
        seen["svg"] += 1
        return real_svg(*args, **kwargs)

    monkeypatch.setattr(_core, "render_interactive", counting_interactive)
    monkeypatch.setattr(_core, "render_svg", counting_svg)
    return seen


def _chart() -> fm.Chart:
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [2.0, 1.0, 3.0]})
    return fm.Chart(df).mark_point().encode(x="x", y="y")


def test_interactive_to_html_renders_once(counts):
    _chart().interactive().to_html()
    assert counts == {"interactive": 1, "svg": 0}


def test_repeated_to_html_renders_nothing_after_construction(counts):
    ic = _chart().interactive()
    after_construction = dict(counts)
    first = ic.to_html()
    second = ic.to_html()
    assert counts == after_construction
    assert first == second


@pytest.mark.parametrize("suffix", ["html", "json"])
def test_html_and_json_saves_render_nothing_after_construction(counts, tmp_path, suffix):
    ic = _chart().interactive()
    after_construction = dict(counts)
    ic.save(str(tmp_path / f"out.{suffix}"))
    assert counts == after_construction


@pytest.mark.parametrize("suffix", ["svg", "png"])
def test_svg_and_png_saves_render_their_own_kind_once(counts, tmp_path, suffix):
    ic = _chart().interactive()
    after_construction = dict(counts)
    ic.save(str(tmp_path / f"out.{suffix}"))
    assert counts["interactive"] == after_construction["interactive"]
    assert counts["svg"] == after_construction["svg"] + 1


def test_saved_html_and_json_equal_the_chart_routes(tmp_path):
    chart = _chart()
    ic = chart.interactive()
    ic.save(str(tmp_path / "ic.html"))
    ic.save(str(tmp_path / "ic.json"))
    chart.save(str(tmp_path / "plain.html"))
    chart.save(str(tmp_path / "plain.json"))
    assert (tmp_path / "ic.html").read_bytes() == (tmp_path / "plain.html").read_bytes()
    assert (tmp_path / "ic.json").read_bytes() == (tmp_path / "plain.json").read_bytes()


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"embed_wasm": False},
        {"csp_nonce": "abc123"},
    ],
)
def test_to_html_bytes_equal_html_string(options):
    chart = _chart()
    assert chart.interactive().to_html(**options) == html_string(chart, **options)


def test_the_toolbar_flag_reaches_the_html():
    chart = _chart()
    assert chart.interactive(toolbar=False).to_html() == html_string(chart, toolbar=False)
    assert chart.interactive(toolbar=False).to_html() != chart.interactive().to_html()


def test_a_titled_chart_keeps_its_title_in_the_html():
    chart = _chart().properties(title="Renders once")
    html = chart.interactive().to_html()
    assert html == html_string(chart)
    assert "<title>Renders once</title>" in html


@pytest.mark.filterwarnings("ignore:.*explicit domain")
def test_a_zoom_change_is_a_fresh_render(counts):
    ic = _chart().interactive()
    before = counts["interactive"]
    ic._on_zoom_change({"new": '{"0": {"x_domain": [1.5, 2.5], "y_domain": [1.0, 3.0]}}'})
    assert counts["interactive"] == before + 1


def test_plain_chart_routes_render_once_each(counts):
    chart = _chart()
    chart.to_html()
    assert counts == {"interactive": 1, "svg": 0}
    chart.to_svg()
    assert counts == {"interactive": 1, "svg": 1}
    ic = chart.interactive()
    assert counts == {"interactive": 2, "svg": 1}
    assert ic.scene_json
    assert counts == {"interactive": 2, "svg": 1}


def test_a_composition_interactive_to_html_renders_nothing_after_construction(monkeypatch):
    chart = _chart()
    comp = fm.hconcat(chart, chart)
    ic = comp.interactive()
    rendered = []
    from ferrum import _scene

    real = _scene._render_scene
    monkeypatch.setattr(_scene, "_render_scene", lambda c: rendered.append(c) or real(c))
    ic.to_html()
    assert rendered == []


def test_a_render_warning_is_raised_once_through_interactive_to_html():
    df = pl.DataFrame({"x": [1.0, None, 3.0], "y": [2.0, 1.0, 3.0], "c": ["a", None, "b"]})
    chart = fm.Chart(df).mark_point().encode(x="x", y="y", color="c")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        chart.interactive().to_html()
    with warnings.catch_warnings(record=True) as single:
        warnings.simplefilter("always")
        chart.to_html()
    assert single
    assert [str(w.message) for w in caught] == [str(w.message) for w in single]


# ---------------------------------------------------------------------------
# One object, one snapshot: every output reads the defaults in force at construction
# ---------------------------------------------------------------------------

_DARK_BACKGROUND = 'fill="#1a1a2e"'
_LIGHT_BACKGROUND = 'fill="#faf7f2"'


def _first_fill(svg: str) -> str:
    match = re.search(r'fill="#[0-9a-f]{6}"', svg)
    assert match is not None
    return match.group(0)


def _subject(kind: str):
    chart = _chart()
    return chart if kind == "chart" else fm.hconcat(chart, chart)


@pytest.mark.parametrize("kind", ["chart", "composition"])
def test_every_save_draws_the_default_theme_of_construction_svg(tmp_path, kind):
    subject = _subject(kind)
    with fm.set_default_theme(fm.themes.dark):
        ic = subject.interactive()
        expected = subject.to_svg()
    path = tmp_path / "a.svg"
    ic.save(str(path))
    saved = path.read_text()
    assert saved == expected
    assert _first_fill(saved) == _DARK_BACKGROUND
    assert _first_fill(subject.to_svg()) == _LIGHT_BACKGROUND


def test_every_save_draws_the_default_theme_of_construction_png_and_pdf(tmp_path):
    chart = _chart()
    with fm.set_default_theme(fm.themes.dark):
        ic = chart.interactive()
        expected_png = chart.to_png()
        chart.save(str(tmp_path / "expected.pdf"))
    ic.save(str(tmp_path / "a.png"))
    ic.save(str(tmp_path / "a.pdf"))
    assert (tmp_path / "a.png").read_bytes() == expected_png
    assert (tmp_path / "a.pdf").read_bytes() == (tmp_path / "expected.pdf").read_bytes()
    assert (tmp_path / "a.png").read_bytes() != chart.to_png()


def test_a_save_inside_another_theme_scope_keeps_the_construction_theme(tmp_path):
    chart = _chart()
    ic = chart.interactive()
    with fm.set_default_theme(fm.themes.dark):
        ic.save(str(tmp_path / "a.svg"))
    saved = (tmp_path / "a.svg").read_text()
    assert saved == chart.to_svg()
    assert _first_fill(saved) == _LIGHT_BACKGROUND


def test_config_defaults_of_construction_reach_every_save(tmp_path):
    chart = _chart()
    with fm.config.defaults(width=300, height=200):
        ic = chart.interactive()
    ic.save(str(tmp_path / "a.svg"))
    assert 'width="300" height="200"' in (tmp_path / "a.svg").read_text()
    assert 'width="640"' in chart.to_svg()


class _FakeWidget:
    """What ``_on_zoom_change`` writes a rebuilt scene to."""

    scene_json = ""
    packed_data = b""
    interaction_config = ""

    @contextlib.contextmanager
    def hold_sync(self):
        yield


@pytest.mark.filterwarnings("ignore:.*explicit domain")
def test_a_zoom_rebuild_keeps_the_construction_theme():
    chart = _chart()
    with fm.set_default_theme(fm.themes.dark):
        ic = chart.interactive()
    ic._widget = _FakeWidget()
    ic._on_zoom_change({"new": '{"0": {"x_domain": [1.5, 2.5], "y_domain": [1.0, 3.0]}}'})
    background = json.loads(ic._widget.scene_json)["background"]
    assert background == {"r": 26, "g": 26, "b": 46, "a": 255}


def test_an_svg_save_equals_the_chart_svg_when_no_scope_differs(tmp_path):
    chart = _chart()
    chart.interactive().save(str(tmp_path / "a.svg"))
    assert (tmp_path / "a.svg").read_text() == chart.to_svg()


def test_an_in_place_frame_edit_after_construction_reaches_only_the_static_saves(tmp_path):
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [2.0, 1.0, 3.0]})
    chart = fm.Chart(frame).mark_point().encode(x="x", y="y")
    ic = chart.interactive()
    frame.loc[0, "y"] = 30.0
    ic.save(str(tmp_path / "a.svg"))
    ic.save(str(tmp_path / "a.json"))
    assert (tmp_path / "a.svg").read_text() == chart.to_svg()
    assert (tmp_path / "a.json").read_text() == ic.scene_json
    assert json.loads(ic.scene_json) != json.loads(chart.interactive().scene_json)
