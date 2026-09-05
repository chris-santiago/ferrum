"""Tests for `transform_top_k(groupby=)` (F-L05-01).

`transform_top_k` used to group by the aggregate field's own distinct
values, making top-k-per-category impossible: `transform_top_k(3,
field="revenue", op="sum")` ranked by each row's raw `revenue` value
instead of by `product`. After the fix, `groupby=` ranks the groupby-column
value tuples (aggregating `op` over `field` within each group) and keeps
every row belonging to one of the top-`n` groups — no rows or columns are
synthesized. Omitting `groupby` preserves the legacy, byte-identical
behavior (documented historical quirk, not the intended use case).

The `row_{n}` synthesized-key fallback — the ungrouped path's old silent
handling of a `field` column whose dtype is neither Utf8 nor numeric (e.g.
Boolean), which made every non-null row its own singleton group — is now a
typed `ValueError` instead.

New-feature file, not an extension of `tests/test_transform_groupby.py`
--------------------------------------------------------------------------
Per the project's test-file convention (`tests/test_boxen_palette.py`'s
module docstring): a findings-ID-named file stays scoped to pinning the
specific regression it describes, while net-new feature coverage for a
transform gets its own `test_<feature>.py` module even when it lands
alongside a findings fix. `test_transform_groupby.py` already owns two
grouped-transform shapes — regression/loess's partition-fit-concat (T6) and
impute's partition-scatter with key/keyvals (T7). `top_k`'s grouped path is
a third, distinct shape (row-preserving selection with no synthesized rows
or columns, per the Rust implementation's own doc comment on
`apply_grouped`), so it gets its own sibling file here rather than being
folded into the existing one.
"""

from __future__ import annotations

import hashlib
import re

import polars as pl
import pytest

import ferrum as fm

from tests._svg_extents import ordered_text_entries

# ---------------------------------------------------------------------------
# The finding's own product/revenue repro (F-L05-01), verbatim.
# ---------------------------------------------------------------------------
# Per-product revenue sums: A=105, B=80, C=170, D=82, E=85 -> top 3 by sum
# = {A, C, E} (the finding's own stated set). A fix that silently falls
# through to the legacy "group by field's own raw value" path would instead
# rank by each row's near-unique raw revenue value and keep {C(170),
# D(82), B(80)} -- a verifiably DIFFERENT set of products, which is what
# makes this repro discriminating rather than merely a smoke test.
_PRODUCT_REVENUE = pl.DataFrame(
    {
        "product": ["A", "A", "B", "C", "D", "E", "E"],
        "revenue": [50.0, 55.0, 80.0, 170.0, 82.0, 40.0, 45.0],
    }
)
_KNOWN_PRODUCTS = {"A", "B", "C", "D", "E"}

# ---------------------------------------------------------------------------
# Byte-identity pin (ungrouped path must be untouched by the groupby fix).
# ---------------------------------------------------------------------------
# Captured against this branch's rebuilt extension (the Rust groupby fix is
# already in the tree — there is no earlier commit on this branch to diff
# against without reverting tracked files, which is out of bounds here).
# The pin is still meaningful as a forward-looking regression guard, and the
# claim that it also matches the PRE-fix byte output is provable by
# inspection rather than by a literal before/after run: `apply_ungrouped`'s
# only changes are (1) an upfront dtype check that a Float64 `field` column
# already satisfied silently before this fix (Utf8-or-numeric was always
# accepted), and (2) turning the dead `row_{n}` fallback branch into
# `unreachable!()`, which a Float64 column never reached either way. Neither
# changes the code path this fixture exercises.
_TOPK_UNGROUPED_FIXTURE = pl.DataFrame({"value": [10.0, 20.0, 1.0, 2.0, 100.0]})
_TOPK_UNGROUPED_MD5 = "5dc2224d03acec397389746d7e0f6ef6"


def _rendered_products(svg: str) -> set[str]:
    """Distinct `product` text-label values rendered by `mark_text(text="product")`.

    Filters `ordered_text_entries` (the shared `<text x y>value</text>`
    parser in `tests/_svg_extents.py`) down to the fixture's known category
    labels, since the same chart also renders numeric axis tick labels.
    """
    return {text for _, _, text in ordered_text_entries(svg) if text in _KNOWN_PRODUCTS}


class TestTopKGroupbyRepro:
    """The finding's own product/revenue repro, through `to_svg`."""

    def test_keeps_top_3_products_by_revenue_sum(self):
        t = fm.transform_top_k(3, field="revenue", op="sum", groupby=["product"])
        chart = (
            fm.Chart(_PRODUCT_REVENUE, width=400, height=300)
            .transform(t)
            .mark_text()
            .encode(x="revenue:Q", y="revenue:Q", text="product")
        )
        svg = chart.to_svg()
        kept = _rendered_products(svg)
        assert kept == {"A", "C", "E"}, (
            f"top-3 by per-product revenue sum must be {{A, C, E}} (sums 105/170/85), got {kept}"
        )
        # Discriminating: a fix that silently falls back to the legacy
        # "group by field's own raw value" path keeps {C, D, B} (170/82/80)
        # instead -- a different set entirely -- so this fails loudly
        # against the pre-groupby (or a fake-groupby) implementation.
        assert kept != {"C", "D", "B"}, (
            "kept set matches the OLD global-ranking-by-raw-value behavior "
            "({C, D, B}), not per-groupby-key aggregation"
        )

    def test_groupby_n5_keeps_every_group_row_preserving(self):
        # n=5 keeps all 5 distinct products -> all 7 input rows survive,
        # confirming the grouped path selects existing rows rather than
        # synthesizing or dropping any.
        t = fm.transform_top_k(5, field="revenue", op="sum", groupby=["product"])
        chart = (
            fm.Chart(_PRODUCT_REVENUE, width=400, height=300)
            .transform(t)
            .mark_point()
            .encode(x="product:N", y="revenue:Q")
        )
        svg = chart.to_svg()
        n_points = len(re.findall(r"<circle[^>]*cx=", svg))
        assert n_points == 7, f"expected all 7 input rows preserved, got {n_points} points"


class TestTopKUngroupedByteIdentical:
    """The groupby fix must not move a single byte of ungrouped output."""

    def test_ungrouped_byte_identical(self):
        t = fm.transform_top_k(2, field="value", op="sum")
        chart = (
            fm.Chart(_TOPK_UNGROUPED_FIXTURE, width=400, height=300)
            .transform(t)
            .mark_point()
            .encode(x="value:Q", y="value:Q")
        )
        svg = chart.to_svg()
        assert hashlib.md5(svg.encode()).hexdigest() == _TOPK_UNGROUPED_MD5, (
            "ungrouped transform_top_k output moved -- the groupby fix must "
            "be byte-identical when groupby is unset"
        )

    def test_ungrouped_dict_has_no_groupby_key(self):
        # groupby=None must not add a "groupby" key to the wire dict at all
        # (serde default / skip_serializing_if on the Rust side mirrors
        # this): a present-but-null key is a different wire shape than an
        # absent key, even though both deserialize the same today.
        t = fm.transform_top_k(2, field="value", op="sum")
        assert "groupby" not in t


class TestTopKGroupbyEmptyListRoutesUngrouped:
    """`groupby=[]` takes the same legacy ungrouped path as omitting it.

    Per the wrapper's docstring, an empty sequence is not "no groups": the
    wire dict still carries `"groupby": []` (not omitted), and Rust's
    `Some(cols) if !cols.is_empty()` dispatch (`top_k.rs`) routes an empty
    `Vec` the same as `None` to `apply_ungrouped`. Pinned here against the
    finding's own product/revenue fixture -- `groupby=[]` must keep the SAME
    set as omitting `groupby` entirely, not the per-groupby-key set.
    """

    def test_empty_list_groupby_matches_omitted_groupby(self):
        omitted = fm.transform_top_k(3, field="revenue", op="sum")
        empty = fm.transform_top_k(3, field="revenue", op="sum", groupby=[])

        def _rendered(transform: dict) -> set[str]:
            chart = (
                fm.Chart(_PRODUCT_REVENUE, width=400, height=300)
                .transform(transform)
                .mark_text()
                .encode(x="revenue:Q", y="revenue:Q", text="product")
            )
            return _rendered_products(chart.to_svg())

        kept_omitted = _rendered(omitted)
        kept_empty = _rendered(empty)
        assert kept_empty == kept_omitted, (
            "groupby=[] must route to the same legacy ungrouped path as "
            f"omitting groupby entirely; omitted kept {kept_omitted}, "
            f"groupby=[] kept {kept_empty}"
        )
        # Discriminating: the per-groupby-key path keeps a different,
        # verifiably distinct set ({A, C, E}), so a fix that treats [] as a
        # real (empty) groupby rather than routing to the legacy ungrouped
        # path would fail this assertion.
        assert kept_empty != {"A", "C", "E"}


class TestTopKGroupbyBareStringRejected:
    """A bare string groupby must refuse, not silently char-split.

    `list("product")` would silently become `["p", "r", "o", ...]` --
    three-plus bogus single-character "columns" sent to Rust -- rather than
    the one intended column. Mirrors `transform_impute(keyvals=...)`'s bare-
    string refusal (`src/ferrum/transforms.py`), and `_validate_as_pair`'s
    refusal for `as_` in the same module.
    """

    def test_bare_string_groupby_raises_before_render(self):
        with pytest.raises(ValueError, match="transform_top_k.*groupby.*bare string"):
            fm.transform_top_k(3, field="revenue", op="sum", groupby="product")

    def test_guard_fires_at_construction_not_render_time(self):
        # No Chart/to_svg needed -- the guard raises directly from the
        # wrapper call, before any dict ever reaches the Rust engine.
        with pytest.raises(ValueError):
            fm.transform_top_k(1, field="x", groupby="x")


class TestTopKTypedErrorFromRust:
    """The `row_{n}` synthesized-key fallback is now a typed refusal.

    Before this fix, a `field` column whose dtype is neither Utf8 nor
    numeric (e.g. Boolean) silently succeeded: every non-null row became
    its own singleton group under a synthesized `row_{n}` key, defeating
    grouping entirely without any error. Verifies the Rust-side
    `PyValueError` crosses the PyO3 boundary as a catchable Python
    `ValueError`, end to end through `to_svg`.
    """

    def test_unsupported_field_dtype_without_groupby_raises_named_error(self):
        df = pl.DataFrame({"flag": [True, False, True]})
        t = fm.transform_top_k(1, field="flag", op="count")
        chart = (
            fm.Chart(df, width=200, height=200)
            .transform(t)
            .mark_point()
            .encode(x="flag:N", y="flag:N")
        )
        with pytest.raises(ValueError, match="data_top_k.*flag.*Boolean"):
            chart.to_svg()

    def test_groupby_path_accepts_any_field_dtype_unaffected(self):
        # With groupby= set, `field` is purely the aggregate target, not a
        # group key -- a Boolean field with op="count" is a legitimate ask
        # there and must NOT raise (the refusal above is scoped to the
        # ungrouped path only).
        df = pl.DataFrame(
            {
                "product": ["A", "A", "B", "B"],
                "flag": [True, False, True, True],
            }
        )
        t = fm.transform_top_k(1, field="flag", op="count", groupby=["product"])
        chart = (
            fm.Chart(df, width=200, height=200)
            .transform(t)
            .mark_point()
            .encode(x="product:N", y="flag:N")
        )
        svg = chart.to_svg()
        assert svg.startswith("<svg")
