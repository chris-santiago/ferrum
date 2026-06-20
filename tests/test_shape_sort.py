"""KG-8: shape encoding honors sort.

Tests that ``shape=fm.Shape(field, sort=...)`` reorders the glyph-to-category
assignment and the legend order, mirroring the categorical color sort path.

Discriminating assertions parse the rendered SVG legend to confirm that:
- absent sort keeps first-appearance order (byte-identical baseline)
- sort="ascending" reorders alphabetically
- sort="descending" reorders in reverse alphabetical order
- sort=[...] (explicit array) reorders to match the array
- sorted and unsorted SVGs differ (the rendered output actually changes)
"""

from __future__ import annotations

import xml.etree.ElementTree as ET  # SVGs from our own renderer; no XXE risk

import polars as pl

import ferrum as fm


SVG_NS = "http://www.w3.org/2000/svg"

# Categories used in test data; keep them in a known first-appearance order so
# assertions are stable regardless of polars internal sort behaviour.
_CATS = ("A", "B", "C")


# ── helpers ───────────────────────────────────────────────────────────────────

def _legend_labels(svg: str) -> list[str]:
    """Return the ordered list of legend labels for the shape channel.

    Parses the SVG, iterates every <text> element in document order, and
    collects those whose text content is exactly one of our test category
    labels.  Document order matches the legend's top-to-bottom entry order,
    which reflects the domain order the renderer used.
    """
    root = ET.fromstring(svg)
    labels = []
    for elem in root.iter(f"{{{SVG_NS}}}text"):
        text = (elem.text or "").strip()
        if text in _CATS:
            labels.append(text)
    return labels


# ── data ─────────────────────────────────────────────────────────────────────

def _make_df() -> pl.DataFrame:
    """3-category data; A appears first, then B, then C."""
    return pl.DataFrame({
        "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "y": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        "g": ["A", "B", "C", "A", "B", "C"],
    })


def _chart(df: pl.DataFrame, **shape_kwargs) -> fm.Chart:
    return (
        fm.Chart(df)
        .mark_point()
        .encode(x="x:Q", y="y:Q", shape=fm.Shape("g", **shape_kwargs))
    )


# ── tests ─────────────────────────────────────────────────────────────────────

def test_shape_absent_sort_first_appearance_order():
    """Without sort the legend lists categories in first-appearance order: A, B, C."""
    svg = _chart(_make_df()).to_svg()
    labels = _legend_labels(svg)
    assert labels == ["A", "B", "C"], (
        f"absent sort must keep first-appearance order A→B→C, got: {labels}"
    )


def test_shape_sort_ascending_reorders_legend():
    """sort='ascending' reorders the legend alphabetically: A, B, C."""
    # Data arrives C, A, B so first-appearance order differs from alphabetical.
    df = pl.DataFrame({
        "x": [1.0, 2.0, 3.0],
        "y": [10.0, 20.0, 30.0],
        "g": ["C", "A", "B"],
    })
    svg = _chart(df, sort="ascending").to_svg()
    labels = _legend_labels(svg)
    assert labels == ["A", "B", "C"], (
        f"sort='ascending' must produce alphabetical legend order A→B→C, got: {labels}"
    )


def test_shape_sort_ascending_differs_from_no_sort():
    """When first-appearance order != alphabetical, ascending sort changes the SVG."""
    df = pl.DataFrame({
        "x": [1.0, 2.0, 3.0],
        "y": [10.0, 20.0, 30.0],
        "g": ["C", "A", "B"],          # first-appearance: C, A, B
    })
    svg_nosort = _chart(df).to_svg()
    svg_asc    = _chart(df, sort="ascending").to_svg()
    assert svg_nosort != svg_asc, (
        "sort='ascending' must produce a different SVG when first-appearance order "
        "differs from alphabetical order"
    )


def test_shape_sort_descending_reorders_legend():
    """sort='descending' reorders the legend in reverse alphabetical order: C, B, A."""
    svg = _chart(_make_df(), sort="descending").to_svg()
    labels = _legend_labels(svg)
    assert labels == ["C", "B", "A"], (
        f"sort='descending' must produce reverse alphabetical legend C→B→A, got: {labels}"
    )


def test_shape_sort_descending_differs_from_no_sort():
    """sort='descending' produces a different SVG than absent sort."""
    df = _make_df()
    svg_nosort = _chart(df).to_svg()
    svg_desc   = _chart(df, sort="descending").to_svg()
    assert svg_nosort != svg_desc, (
        "sort='descending' must change the rendered SVG vs absent sort"
    )


def test_shape_sort_explicit_array_reorders_legend():
    """sort=['B', 'C', 'A'] produces exactly that legend order."""
    svg = _chart(_make_df(), sort=["B", "C", "A"]).to_svg()
    labels = _legend_labels(svg)
    assert labels == ["B", "C", "A"], (
        f"sort=['B','C','A'] must reorder legend to B→C→A, got: {labels}"
    )


def test_shape_sort_explicit_array_differs_from_no_sort():
    """Explicit-array sort SVG differs from no-sort SVG (real discriminator)."""
    df = _make_df()
    svg_nosort  = _chart(df).to_svg()
    svg_explicit = _chart(df, sort=["C", "A", "B"]).to_svg()
    assert svg_nosort != svg_explicit, (
        "explicit-array sort must produce a different SVG than absent sort"
    )


def test_shape_sort_no_spurious_warning():
    """fm.Shape(field, sort=...) must NOT emit a 'not yet honored' warning (KG-8 fix)."""
    import warnings
    from ferrum._warn import reset_warnings

    reset_warnings()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _ = fm.Shape("g", sort="ascending")

    sort_warns = [w for w in caught if "sort" in str(w.message)]
    assert not sort_warns, (
        f"fm.Shape(sort=...) must not emit a spurious warning after KG-8; "
        f"got: {[str(w.message) for w in sort_warns]}"
    )
