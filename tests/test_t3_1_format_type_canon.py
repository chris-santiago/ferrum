"""Regression tests for T3.1 — format_type canonical, formatType alias (ENC-03, D-FMT-1).

Covers:
- TEXT channel: format_type= (canonical snake_case) now honored (was silently dropped).
- TEXT channel: formatType= (Vega-compat alias) still honored.
- POSITIONAL channel: formatType= (alias) now honored (was silently dropped).
- POSITIONAL channel: format_type= (canonical) still honored.
- Both spellings serialize to the wire key format_type via _emit_format_type.
- No golden changes — wire output is byte-identical for pre-existing uses of
  formatType on text channels and format_type on positional channels.

The text-format_type case and the positional-formatType case MUST FAIL on the
pre-fix code (where TEXT_FORMATTED lacked format_type and PRIMARY_POSITIONAL
lacked formatType), confirming the regression tests catch the real asymmetry.
"""

from __future__ import annotations

import warnings

import pytest

import ferrum as fm
from ferrum._warn import reset_warnings
from ferrum.encoding._honored import TEXT_FORMATTED, PRIMARY_POSITIONAL


# ---------------------------------------------------------------------------
# Vocabulary membership (the _honored_kwargs contract)
# ---------------------------------------------------------------------------


def test_text_formatted_honors_format_type_canonical():
    """TEXT_FORMATTED must include the canonical snake_case spelling."""
    assert "format_type" in TEXT_FORMATTED


def test_text_formatted_honors_format_type_alias():
    """TEXT_FORMATTED must include the Vega-compat camelCase alias."""
    assert "formatType" in TEXT_FORMATTED


def test_primary_positional_honors_format_type_canonical():
    """PRIMARY_POSITIONAL must include the canonical snake_case spelling."""
    assert "format_type" in PRIMARY_POSITIONAL


def test_primary_positional_honors_format_type_alias():
    """PRIMARY_POSITIONAL must include the Vega-compat camelCase alias."""
    assert "formatType" in PRIMARY_POSITIONAL


# ---------------------------------------------------------------------------
# Text channel — format_type= (canonical) honored without warning
# ---------------------------------------------------------------------------


def test_text_format_type_canonical_no_warn():
    """Text(format_type='number') must not trigger a warn_once (ENC-03 root fix).

    On the pre-fix code, TEXT_FORMATTED lacked 'format_type', so the warn guard
    in ChannelBase.__init__ fired and the kwarg was dropped.  After the fix,
    'format_type' is in TEXT_FORMATTED and no warning is emitted.
    """
    reset_warnings()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ch = fm.Text("label", format_type="number")
    assert len(w) == 0, f"Unexpected warning: {[str(x.message) for x in w]}"


def test_text_format_type_canonical_reaches_wire():
    """Text(format_type='number') serializes format_type onto the wire dict.

    On the pre-fix code, the kwarg was dropped by the warn guard so
    to_encoding_spec_dict() did not include format_type.
    """
    reset_warnings()
    ch = fm.Text("label", format_type="number")
    d = ch.to_encoding_spec_dict()
    assert "format_type" in d
    assert d["format_type"] == "number"


# ---------------------------------------------------------------------------
# Text channel — formatType= (alias) still honored
# ---------------------------------------------------------------------------


def test_text_format_type_alias_no_warn():
    """Text(formatType='time') must not trigger a warn_once."""
    reset_warnings()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ch = fm.Text("label", formatType="time")
    assert len(w) == 0, f"Unexpected warning: {[str(x.message) for x in w]}"


def test_text_format_type_alias_reaches_wire():
    """Text(formatType='time') serializes as format_type on the wire dict."""
    reset_warnings()
    ch = fm.Text("label", formatType="time")
    d = ch.to_encoding_spec_dict()
    assert "format_type" in d
    assert d["format_type"] == "time"


# ---------------------------------------------------------------------------
# Positional channel — format_type= (canonical) still honored
# ---------------------------------------------------------------------------


def test_positional_format_type_canonical_no_warn():
    """X(format_type='number') must not trigger a warn_once."""
    reset_warnings()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ch = fm.X("value", format_type="number")
    assert len(w) == 0, f"Unexpected warning: {[str(x.message) for x in w]}"


def test_positional_format_type_canonical_reaches_wire():
    """X(format_type='number') serializes format_type onto the wire dict."""
    reset_warnings()
    ch = fm.X("value", format_type="number")
    d = ch.to_encoding_spec_dict()
    assert "format_type" in d
    assert d["format_type"] == "number"


# ---------------------------------------------------------------------------
# Positional channel — formatType= (alias) now honored
# ---------------------------------------------------------------------------


def test_positional_format_type_alias_no_warn():
    """X(formatType='time') must not trigger a warn_once (ENC-03 positional side).

    On the pre-fix code, PRIMARY_POSITIONAL lacked 'formatType', so the warn
    guard fired and the kwarg was dropped.
    """
    reset_warnings()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ch = fm.X("date", formatType="time")
    assert len(w) == 0, f"Unexpected warning: {[str(x.message) for x in w]}"


def test_positional_format_type_alias_reaches_wire():
    """X(formatType='time') serializes as format_type on the wire dict.

    On the pre-fix code, the kwarg was dropped, so format_type was absent.
    """
    reset_warnings()
    ch = fm.X("date", formatType="time")
    d = ch.to_encoding_spec_dict()
    assert "format_type" in d
    assert d["format_type"] == "time"


# ---------------------------------------------------------------------------
# Wire output is byte-stable for pre-existing uses
# ---------------------------------------------------------------------------


def test_text_format_type_wire_key_is_snake_case():
    """Both spellings serialize to wire key 'format_type', never 'formatType'."""
    reset_warnings()
    via_snake = fm.Text("label", format_type="number").to_encoding_spec_dict()
    via_camel = fm.Text("label", formatType="number").to_encoding_spec_dict()
    assert "format_type" in via_snake
    assert "format_type" in via_camel
    assert "formatType" not in via_snake
    assert "formatType" not in via_camel
    assert via_snake["format_type"] == via_camel["format_type"]


def test_positional_format_type_wire_key_is_snake_case():
    """Both spellings serialize to wire key 'format_type', never 'formatType'."""
    reset_warnings()
    via_snake = fm.X("v", format_type="number").to_encoding_spec_dict()
    via_camel = fm.X("v", formatType="number").to_encoding_spec_dict()
    assert "format_type" in via_snake
    assert "format_type" in via_camel
    assert "formatType" not in via_snake
    assert "formatType" not in via_camel
    assert via_snake["format_type"] == via_camel["format_type"]


# ---------------------------------------------------------------------------
# Y channel — same positional role, confirm symmetry
# ---------------------------------------------------------------------------


def test_y_format_type_alias_no_warn():
    """Y(formatType='time') also honors the alias (same PRIMARY_POSITIONAL role)."""
    reset_warnings()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        fm.Y("ts", formatType="time")
    assert len(w) == 0


def test_y_format_type_alias_reaches_wire():
    """Y(formatType='time') serializes as format_type on the wire dict."""
    reset_warnings()
    ch = fm.Y("ts", formatType="time")
    d = ch.to_encoding_spec_dict()
    assert d.get("format_type") == "time"
