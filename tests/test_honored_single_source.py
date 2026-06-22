"""Regression tests for T2.3 — honored-kwargs as the single source of truth.

Covers decision D-HONORED-1 (ENC-01, ENC-04, ENC-08, ENC-09, ENC-10, XSIB-03,
XSIB-05):

- ``to_encoding_spec_dict`` is driven off the channel's own ``_honored_kwargs``;
  a representative channel of each role serializes the same key set as before
  (the serialization contract is pinned here so iteration cannot drop a key).
- ENC-08 short/long ``type=`` spellings are normalized to the short code so they
  are equal, hash-equal, and serialize identically.
- ENC-10 alias ordering and the drop-with-warn-once-when-target-present policy.
- ENC-01/XSIB-05: one honored-kwarg vocabulary module; no stray
  ``_RENDERED_HONORED`` name survives in the channel modules.
"""

from __future__ import annotations

import warnings

import pytest

import ferrum as fm
from ferrum._warn import reset_warnings
from ferrum.encoding import _apply_channel_aliases
from ferrum.encoding._aliases import apply_channel_aliases
from ferrum.encoding._honored import (
    APPEARANCE_BASE,
    APPEARANCE_COLOR,
    APPEARANCE_CONDITION,
    APPEARANCE_FULL,
    APPEARANCE_SCHEME,
    APPEARANCE_SORT,
    BARE,
    FACET,
    POLAR_PRIMARY,
    PRIMARY_POSITIONAL,
    SECONDARY_EXTENT,
    TEXT_FORMATTED,
    TEXT_FORMATTED_TITLED,
)


# ---------------------------------------------------------------------------
# ENC-04 — serializer iterates _honored_kwargs (per-role contract pin)
# ---------------------------------------------------------------------------

# Every kwarg the old hard-coded serializer ever emitted, with a representative
# value.  Each channel below is built with the subset it honors; the expected
# emitted key set is pinned so iterating _honored_kwargs cannot silently drop or
# add a wire key.
_REPRESENTATIVE_KWARGS = {
    "type": "Q",
    "scale": fm.LinearScale(),
    "title": "T",
    "axis": {"grid": False},
    "legend": {"orient": "top"},
    "sort": ["a", "b"],
    "stack": "zero",
    "impute": {"value": 0},
    "scheme": "set1",
    "format": ".2f",
    "format_type": "number",
    "formatType": "time",
    "condition": {"test": "x", "value": "red"},
    "bin": True,
    "aggregate": "mean",
}

# One representative channel per role → the exact key set its spec dict must emit.
# ``field`` is always present; ``bin``/``aggregate`` are honored but routed
# through to_implicit_transforms, so they never appear in the spec dict.
_ROLE_CONTRACT = [
    # (channel class, role frozenset, expected emitted spec-dict keys)
    (
        fm.X,
        PRIMARY_POSITIONAL,
        {
            "field",
            "type_",
            "scale",
            "title",
            "axis",
            "legend",
            "sort",
            "stack",
            "impute",
            "format",
            "format_type",
        },
    ),
    (
        fm.Y,
        PRIMARY_POSITIONAL,
        {
            "field",
            "type_",
            "scale",
            "title",
            "axis",
            "legend",
            "sort",
            "stack",
            "impute",
            "format",
            "format_type",
        },
    ),
    (fm.X2, SECONDARY_EXTENT, {"field", "type_"}),
    (fm.Theta, POLAR_PRIMARY, {"field", "type_", "stack"}),
    (
        fm.Color,
        APPEARANCE_COLOR,
        {
            "field",
            "type_",
            "scale",
            "title",
            "legend",
            "condition",
            "sort",
            "scheme",
        },
    ),
    (
        fm.Fill,
        APPEARANCE_SCHEME,
        {
            "field",
            "type_",
            "scale",
            "title",
            "legend",
            "condition",
            "scheme",
        },
    ),
    (
        fm.Shape,
        APPEARANCE_SORT,
        {
            "field",
            "type_",
            "scale",
            "title",
            "legend",
            "condition",
            "sort",
        },
    ),
    (
        fm.Size,
        APPEARANCE_CONDITION,
        {
            "field",
            "type_",
            "scale",
            "title",
            "legend",
            "condition",
        },
    ),
    (fm.FillOpacity, APPEARANCE_FULL, {"field", "type_", "scale", "title", "legend"}),
    (fm.Angle, APPEARANCE_BASE, {"field", "type_", "legend"}),
    (fm.Text, TEXT_FORMATTED, {"field", "type_", "format", "format_type"}),
    (
        fm.TooltipField,
        TEXT_FORMATTED_TITLED,
        {
            "field",
            "type_",
            "title",
            "format",
            "format_type",
        },
    ),
    (fm.Detail, BARE, {"field", "type_"}),
    (fm.Facet, FACET, {"field", "type_", "title"}),
]


@pytest.mark.parametrize(
    "channel_cls,role,expected_keys",
    _ROLE_CONTRACT,
    ids=[c.__name__ for c, _, _ in _ROLE_CONTRACT],
)
def test_spec_dict_emits_pinned_keys_per_role(channel_cls, role, expected_keys):
    """A representative channel of each role serializes the pinned key set.

    Pins the serialization contract so iterating _honored_kwargs cannot drift
    from what the old hard-coded serializer emitted.
    """
    reset_warnings()
    kw = {k: v for k, v in _REPRESENTATIVE_KWARGS.items() if k in role}
    kw.setdefault("type", "Q")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ch = channel_cls("f", **kw)
    assert set(ch.to_encoding_spec_dict().keys()) == expected_keys


def test_spec_dict_drops_bin_and_aggregate():
    """bin/aggregate are honored but consumed via to_implicit_transforms only."""
    reset_warnings()
    x = fm.X("f", type="Q", bin=True, aggregate="mean")
    d = x.to_encoding_spec_dict()
    assert "bin" not in d and "aggregate" not in d
    assert {"field", "type_"} == set(d.keys())


def test_spec_dict_only_serializes_honored_kwargs():
    """A kwarg the channel does NOT honor is not serialized (ENC-04 core).

    The old serializer forwarded a hard-coded tuple regardless of the honored
    set; the new one iterates the honored set, so a non-honored ``format`` on a
    Color (which warns) does not reach the wire.
    """
    reset_warnings()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = fm.Color("f", type="Q", format=".2f")  # Color does not honor 'format'
    assert "format" not in c.to_encoding_spec_dict()


# ---------------------------------------------------------------------------
# ENC-08 — short/long type spellings normalize to the short code
# ---------------------------------------------------------------------------


def test_type_long_spelling_equals_short():
    """X('a', type='quantitative') equals/hashes-equal X('a', type='Q')."""
    reset_warnings()
    long = fm.X("a", type_="quantitative")
    short = fm.X("a", type_="Q")
    assert long == short
    assert hash(long) == hash(short)


def test_type_long_spelling_serializes_to_short():
    """The long spelling serializes to the canonical single-letter code."""
    reset_warnings()
    assert fm.X("a", type_="quantitative").to_encoding_spec_dict()["type_"] == "Q"


@pytest.mark.parametrize(
    "long,short",
    [("quantitative", "Q"), ("nominal", "N"), ("ordinal", "O"), ("temporal", "T")],
)
def test_all_long_type_spellings_normalize(long, short):
    """Every long type spelling normalizes to its short code on the wire."""
    reset_warnings()
    assert fm.Color("a", type=long).to_encoding_spec_dict()["type_"] == short


def test_invalid_type_still_rejected():
    """An unrecognized type spelling still raises ValueError."""
    reset_warnings()
    with pytest.raises(ValueError, match="expected one of"):
        fm.X("a", type_="bogus")


# ---------------------------------------------------------------------------
# ENC-10 — alias ordering + drop-with-warn-once policy
# ---------------------------------------------------------------------------


def test_apply_channel_aliases_reexport_is_same_object():
    """The package re-export points at the relocated _aliases implementation."""
    assert _apply_channel_aliases is apply_channel_aliases


def test_fill_aliases_to_color_when_color_absent():
    reset_warnings()
    fill = fm.Fill("origin")
    enc, _mk = apply_channel_aliases({"fill": fill}, {})
    assert enc["color"] is fill


def test_fill_takes_priority_over_stroke_for_color_target():
    """Ordering: fill is resolved before stroke, so fill wins the color target."""
    reset_warnings()
    fill = fm.Fill("a")
    stroke = fm.Stroke("b")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        enc, _mk = apply_channel_aliases({"fill": fill, "stroke": stroke}, {})
    assert enc["color"] is fill  # earlier alias wins


def test_stroke_aliases_to_color_when_color_absent():
    reset_warnings()
    stroke = fm.Stroke("origin")
    enc, _mk = apply_channel_aliases({"stroke": stroke}, {})
    assert enc["color"] is stroke


def test_stroke_dropped_with_warn_once_when_color_present():
    """stroke -> color conflict warns exactly once and leaves color untouched."""
    reset_warnings()
    color = fm.Color("c")
    stroke = fm.Stroke("d")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        enc, _mk = apply_channel_aliases({"color": color, "stroke": stroke}, {})
    assert enc["color"] is color  # existing target preserved
    assert len(w) == 1
    assert "stroke" in str(w[0].message)


def test_fill_kept_silently_when_color_present():
    """fill -> color conflict drops fill silently (keep_target, no warning)."""
    reset_warnings()
    color = fm.Color("c")
    fill = fm.Fill("d")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        enc, _mk = apply_channel_aliases({"color": color, "fill": fill}, {})
    assert enc["color"] is color
    assert len(w) == 0


def test_detail_routes_into_mark_kwargs():
    reset_warnings()
    detail = fm.Detail("series")
    _enc, mk = apply_channel_aliases({"detail": detail}, {})
    assert mk["detail"] == "series"


# ---------------------------------------------------------------------------
# ENC-01 / XSIB-05 — one vocabulary module; no stray _RENDERED_HONORED name
# ---------------------------------------------------------------------------


def test_rendered_honored_name_removed_from_channel_modules():
    """The dead _RENDERED_HONORED alias is gone from both channel modules."""
    from ferrum.encoding import appearance, positional

    assert not hasattr(positional, "_RENDERED_HONORED")
    assert not hasattr(appearance, "_RENDERED_HONORED")


def test_x_and_y_share_the_primary_positional_role():
    assert fm.X._honored_kwargs is PRIMARY_POSITIONAL
    assert fm.Y._honored_kwargs is PRIMARY_POSITIONAL
