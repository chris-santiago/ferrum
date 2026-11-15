"""An interactive chart renders once: every HTML and JSON output reuses the scene built at construction.

Renders are counted at the Rust entry points (``render_interactive``, ``render_svg``) by
wrapping them, so a second render anywhere on the path is seen whatever Python layer made it.
"""

from __future__ import annotations

import warnings

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
