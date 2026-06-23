"""Regression tests for T0.5 — unified save / HTML / title dispatch.

Closes the cohesion-audit findings REND-01 (three parallel save() dispatchers
with a silently-lossy ``InteractiveChart.save``), REND-02/03/07/09/10 (the
HTML-assembly and title-resolution drift across Chart / composite / interactive
paths).

The assertions here are written to FAIL on the pre-fix code:

- Pre-fix, ``chart.interactive().save("foo.png")`` ignored the extension and
  wrote HTML bytes into a ``.png`` file with no error.
- Pre-fix, ``composite.save("foo.html", scale=2)`` forwarded ``scale`` into
  ``InteractiveChart.save`` which silently dropped it; ``format=``/``scale=``
  on the interactive path were swallowed entirely.
- Pre-fix, ``Chart.to_html`` and ``save_chart``'s HTML branch resolved the
  ``<title>`` via the leaf ``_extract_title_text`` rather than the canonical
  ``figure_title_text`` dispatcher.
"""

from __future__ import annotations

import polars as pl
import pytest

import ferrum as fm
from ferrum._interactive import InteractiveChart


@pytest.fixture
def df():
    return pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})


@pytest.fixture
def chart(df):
    return fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q").properties(width=200, height=200)


@pytest.fixture
def composite(chart):
    return chart | chart


# ── REND-01: InteractiveChart.save honors the path extension ──────────────────


def test_interactive_save_png_writes_png_not_html(chart, tmp_path):
    """interactive().save("foo.png") must write a real PNG, not HTML bytes
    into a .png file.  Pre-fix this produced an HTML document named .png."""
    out = tmp_path / "foo.png"
    chart.interactive().save(str(out))
    data = out.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n", (
        "interactive().save('foo.png') must produce PNG magic bytes, not HTML"
    )
    assert b"<!DOCTYPE html>" not in data, "PNG save must not contain HTML markup"


def test_interactive_save_svg_writes_svg(chart, tmp_path):
    """interactive().save("foo.svg") must write SVG, honoring the extension."""
    out = tmp_path / "foo.svg"
    chart.interactive().save(str(out))
    text = out.read_text()
    assert text.startswith("<svg") or text.startswith("<?xml"), (
        "interactive().save('foo.svg') must produce SVG markup"
    )


def test_interactive_save_json_writes_scene_json(chart, tmp_path):
    """interactive().save("foo.json") must write scene JSON, honoring extension."""
    import json

    out = tmp_path / "foo.json"
    chart.interactive().save(str(out))
    scene = json.loads(out.read_text())
    assert "panels" in scene and "width" in scene


def test_interactive_save_html_still_works(chart, tmp_path):
    """interactive().save("foo.html") must still write valid HTML (non-breaking)."""
    out = tmp_path / "foo.html"
    chart.interactive().save(str(out))
    content = out.read_text()
    assert "<!DOCTYPE html>" in content
    assert "_render" in content


def test_interactive_save_unknown_extension_raises(chart, tmp_path):
    """interactive().save("foo.weird") must raise a clear error rather than
    silently writing HTML into the wrong file type."""
    out = tmp_path / "foo.weird"
    with pytest.raises(ValueError, match="extension"):
        chart.interactive().save(str(out))


def test_interactive_save_toolbar_flag_carried_into_html(chart, tmp_path):
    """The interactive toolbar flag must be carried into HTML output via the
    unified save path."""
    import json

    out = tmp_path / "no_toolbar.html"
    chart.interactive(toolbar=False).save(str(out))
    content = out.read_text()
    # The interaction config embedded in the HTML must reflect toolbar=False.
    assert '"toolbar":false' in content.replace(" ", "")


# ── REND-01: composite.save no longer silently drops scale=/format= ───────────


def test_composite_save_html_with_scale_does_not_raise(composite, tmp_path):
    """composite.save("foo.html", scale=2) must be accepted (scale is a no-op
    for HTML) rather than threaded into an opaque **kwargs that swallows it.

    Pre-fix, _CompositeBase.save passed scale into InteractiveChart.save via
    **kwargs, which popped only embed_wasm/csp_nonce and silently dropped the
    rest.  This test locks that scale is an explicit, documented parameter."""
    out = tmp_path / "comp.html"
    composite.save(str(out), scale=2.0)
    content = out.read_text()
    assert "<!DOCTYPE html>" in content


def test_composite_save_png_honors_scale(composite, tmp_path):
    """composite.save with PNG must honor scale (larger scale → larger file)."""
    p1 = tmp_path / "comp_1x.png"
    p2 = tmp_path / "comp_2x.png"
    composite.save(str(p1), scale=1.0)
    composite.save(str(p2), scale=2.0)
    d1 = p1.read_bytes()
    d2 = p2.read_bytes()
    assert d1[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(d1) < len(d2), "composite.save must honor PNG scale"


def test_composite_save_json_supported(composite, tmp_path):
    """composite.save now supports JSON (was missing from the composite-only
    dispatcher pre-fix) because it routes through the single save_chart router."""
    import json

    out = tmp_path / "comp.json"
    composite.save(str(out), format="json")
    scene = json.loads(out.read_text())
    assert "panels" in scene


def test_composite_save_unknown_format_raises(composite, tmp_path):
    """composite.save with an unknown format raises ValueError via the unified
    router (no longer the composite-specific message)."""
    with pytest.raises(ValueError):
        composite.save(str(tmp_path / "comp.weird"))


# ── REND-02/03/09: one HTML helper, no widget for file save ───────────────────


def test_html_string_helper_exists_and_returns_document(chart):
    """The single html_string() helper must exist and return a complete HTML
    document for any chart-like."""
    from ferrum.display import html_string

    html = html_string(chart)
    assert html.lstrip().startswith("<!")
    assert "_render" in html


def test_composite_to_html_does_not_construct_widget(composite, monkeypatch):
    """_CompositeBase.to_html must NOT construct an InteractiveChart widget
    (REND-09).  Pre-fix it called self.interactive() and reached into the
    widget's private ic._scene_json / ic._packed_data."""
    import ferrum._interactive as interactive_mod

    constructed = []
    real_init = interactive_mod.InteractiveChart.__init__

    def _tracking_init(self, *args, **kwargs):
        constructed.append(self)
        return real_init(self, *args, **kwargs)

    monkeypatch.setattr(interactive_mod.InteractiveChart, "__init__", _tracking_init)

    html = composite.to_html()
    assert html.lstrip().startswith("<!")
    assert not constructed, "_CompositeBase.to_html must not construct an InteractiveChart widget"


def test_save_chart_html_branch_does_not_construct_widget(chart, tmp_path, monkeypatch):
    """save_chart's HTML branch must not construct an InteractiveChart widget."""
    import ferrum._interactive as interactive_mod
    from ferrum.display import save_chart

    constructed = []
    real_init = interactive_mod.InteractiveChart.__init__

    def _tracking_init(self, *args, **kwargs):
        constructed.append(self)
        return real_init(self, *args, **kwargs)

    monkeypatch.setattr(interactive_mod.InteractiveChart, "__init__", _tracking_init)

    save_chart(chart, str(tmp_path / "out.html"))
    assert not constructed, "save_chart HTML branch must not construct a widget"


# ── REND-02/10: all HTML paths resolve the title via figure_title_text ────────


def test_all_html_paths_resolve_title_via_figure_title_text(df, tmp_path):
    """Chart.to_html, save_chart, and InteractiveChart.save must all resolve
    the browser-tab title through the single figure_title_text() dispatcher,
    producing an identical <title> for the same chart."""
    from ferrum.display import figure_title_text, save_chart

    chart = (
        fm.Chart(df)
        .mark_point()
        .encode(x="x:Q", y="y:Q")
        .properties(title="Unified Title", width=200, height=200)
    )

    expected = f"<title>{figure_title_text(chart)}</title>"
    assert expected == "<title>Unified Title</title>"

    # Path 1: Chart.to_html
    html_to_html = chart.to_html()

    # Path 2: display.save_chart
    out = tmp_path / "via_save_chart.html"
    save_chart(chart, str(out))
    html_save_chart = out.read_text()

    # Path 3: InteractiveChart.save
    out2 = tmp_path / "via_interactive.html"
    InteractiveChart(chart).save(str(out2))
    html_interactive = out2.read_text()

    for label, html in (
        ("Chart.to_html", html_to_html),
        ("save_chart", html_save_chart),
        ("InteractiveChart.save", html_interactive),
    ):
        assert expected in html, f"{label} did not resolve title via figure_title_text"
        assert "Title(" not in html, f"{label} emitted a Title dataclass repr"


def test_chart_to_html_routes_through_figure_title_text(df, monkeypatch):
    """Chart.to_html must call figure_title_text (the dispatcher), not the
    lower-level _extract_title_text leaf directly.  Pre-fix it bypassed the
    dispatcher."""
    import ferrum.display as display_mod

    calls = []
    real = display_mod.figure_title_text

    def _spy(chart_like):
        calls.append(chart_like)
        return real(chart_like)

    monkeypatch.setattr(display_mod, "figure_title_text", _spy)

    chart = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q").properties(width=200, height=200)
    chart.to_html()
    assert calls, "Chart.to_html must route title resolution through figure_title_text"


def test_composite_html_title_resolves_figure_title(df, tmp_path):
    """A composite's figure-level title must reach the HTML <title> through the
    unified path."""
    chart = fm.Chart(df).mark_point().encode(x="x:Q", y="y:Q").properties(width=200, height=200)
    composite = (chart | chart).properties(title="Composite Figure Title")
    out = tmp_path / "comp_title.html"
    composite.save(str(out))
    content = out.read_text()
    assert "<title>Composite Figure Title</title>" in content


# ── REND-07: to_html/save signature parity (csp_nonce on both) ─────────────────


def test_chart_to_html_accepts_csp_nonce(chart):
    """Chart.to_html must accept csp_nonce for parity with the composite path."""
    html = chart.to_html(csp_nonce="abc123")
    assert 'nonce="abc123"' in html


def test_composite_to_html_accepts_csp_nonce(composite):
    """_CompositeBase.to_html must accept csp_nonce for sibling parity."""
    html = composite.to_html(csp_nonce="def456")
    assert 'nonce="def456"' in html


def test_composite_save_html_accepts_csp_nonce(composite, tmp_path):
    """composite.save must accept csp_nonce as an explicit kwarg (not a magic
    **kwargs key)."""
    out = tmp_path / "comp_nonce.html"
    composite.save(str(out), csp_nonce="ghi789")
    assert 'nonce="ghi789"' in out.read_text()
