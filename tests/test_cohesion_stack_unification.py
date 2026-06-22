"""Regression tests for T2.4 / D-STACK-1 — one STACK_OFFSETS + validator.

ENC-02: valid stack offsets were hard-coded three times with divergent
memberships and three error formats. They are now unified onto one canonical
``STACK_OFFSETS`` frozenset + ``_validate_stack_offset(value, *, where)`` in
``position.py``. The three entry points (encoding ``stack=``, ``transform_stack``,
``position.Stack``) all route through it. The encoding path layers bool/falsy
normalization on top; transform/position accept only the three real offsets.

ENC-07: X/Y/Theta/Radius no longer each override ``_validate`` with an
identical body; the normalization is hoisted into ``ChannelBase`` behind a
``_stack_kwarg`` flag.

CHART-08: ``_STACKABLE_MARKS`` (stack-base consumers) and ``_STACK_ELIGIBLE``
(``position=Stack`` acceptors) are co-located in ``position.py`` but remain two
distinct sets answering two distinct questions.
"""

from __future__ import annotations

import pytest

import ferrum as fm
from ferrum import position
from ferrum.position import STACK_OFFSETS, _STACKABLE_MARKS, _STACK_ELIGIBLE


# ---------------------------------------------------------------------------
# ENC-02: one canonical offset set + one validator, three entry points
# ---------------------------------------------------------------------------


def test_canonical_offset_set_is_the_three_real_offsets():
    """``STACK_OFFSETS`` is exactly the three real stack strategies."""
    assert STACK_OFFSETS == frozenset({"zero", "normalize", "center"})


def _canonical_core(value: str) -> str:
    """The message body all three sites share (sans the per-site prefix)."""
    return f"stack offset {value!r} is not valid; expected one of {sorted(STACK_OFFSETS)}"


def test_all_three_entry_points_share_one_canonical_message():
    """encode / transform_stack / Stack all fail with the same canonical body."""
    bad = "streamgraph"

    with pytest.raises(ValueError) as enc_exc:
        fm.Y("val", stack=bad)
    with pytest.raises(ValueError) as xform_exc:
        fm.transform_stack("val", groupby=["cat"], offset=bad)
    with pytest.raises(ValueError) as pos_exc:
        fm.Stack(offset=bad)

    core = _canonical_core(bad)
    enc_msg = str(enc_exc.value)
    xform_msg = str(xform_exc.value)
    pos_msg = str(pos_exc.value)

    # The shared canonical body is byte-identical across all three sites.
    assert core in enc_msg
    assert core in xform_msg
    assert core in pos_msg

    # Each site prefixes the body with its own label, so the full messages
    # differ only by that ``where`` prefix.
    assert enc_msg == f"Y: {core}"
    assert xform_msg == f"transform_stack: {core}"
    assert pos_msg == f"Stack: {core}"


def test_validator_message_format_is_stable():
    """``_validate_stack_offset`` produces the documented template."""
    with pytest.raises(ValueError) as exc:
        position._validate_stack_offset("nope", where="Frob")
    assert str(exc.value) == f"Frob: {_canonical_core('nope')}"


def test_validator_accepts_real_offsets():
    """All three real offsets pass the canonical validator silently."""
    for off in ("zero", "normalize", "center"):
        position._validate_stack_offset(off, where="anywhere")  # no raise


# ---------------------------------------------------------------------------
# Per-path accept/reject semantics preserved
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("falsy", ["false", "null", "none", "FALSE", "None"])
def test_encode_path_accepts_falsy_spellings(falsy):
    """The encoding ``stack=`` path still accepts the falsy spellings."""
    y = fm.Y("val", stack=falsy)
    # Falsy strings pass through verbatim (Rust's Option<String> → None).
    assert y._kwargs["stack"] == falsy


def test_encode_path_accepts_bool_true_false():
    """bool ``stack=`` normalizes to 'zero'/'false' on the encoding path."""
    assert fm.Y("val", stack=True)._kwargs["stack"] == "zero"
    assert fm.Y("val", stack=False)._kwargs["stack"] == "false"


@pytest.mark.parametrize("offset", ["zero", "normalize", "center"])
def test_encode_path_accepts_real_offsets(offset):
    """The encoding path accepts the three real offsets verbatim."""
    assert fm.Y("val", stack=offset)._kwargs["stack"] == offset


@pytest.mark.parametrize("falsy", ["false", "null", "none"])
def test_transform_stack_rejects_falsy_spellings(falsy):
    """``transform_stack`` accepts ONLY the real offsets, never falsy spellings."""
    with pytest.raises(ValueError):
        fm.transform_stack("val", groupby=["cat"], offset=falsy)


@pytest.mark.parametrize("falsy", ["false", "null", "none"])
def test_position_stack_rejects_falsy_spellings(falsy):
    """``position.Stack`` accepts ONLY the real offsets, never falsy spellings."""
    with pytest.raises(ValueError):
        fm.Stack(offset=falsy)


@pytest.mark.parametrize("offset", ["zero", "normalize", "center"])
def test_transform_and_position_accept_real_offsets(offset):
    """Both non-encoding sites accept all three real offsets."""
    assert fm.transform_stack("v", groupby=["c"], offset=offset)["offset"] == offset
    assert fm.Stack(offset=offset).offset == offset


# ---------------------------------------------------------------------------
# ENC-07: ChannelBase hoist — non-bool/non-str raises TypeError naming channel
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("channel_cls", [fm.X, fm.Y, fm.Theta, fm.Radius])
def test_stacking_channels_route_through_base_normalization(channel_cls):
    """X/Y/Theta/Radius all normalize via the hoisted ChannelBase behavior."""
    assert channel_cls._stack_kwarg is True
    # bool True normalizes identically across the four channels.
    assert channel_cls("v", stack=True)._kwargs["stack"] == "zero"


@pytest.mark.parametrize("channel_cls", [fm.X, fm.Y, fm.Theta, fm.Radius])
def test_stacking_channels_bad_offset_names_channel(channel_cls):
    """A bad offset raises ValueError prefixed with the channel name."""
    with pytest.raises(ValueError) as exc:
        channel_cls("v", stack="bogus")
    assert str(exc.value) == f"{channel_cls.__name__}: {_canonical_core('bogus')}"


@pytest.mark.parametrize("channel_cls", [fm.X, fm.Y, fm.Theta, fm.Radius])
def test_stacking_channels_non_str_raises_type_error(channel_cls):
    """Non-bool, non-str stack= raises TypeError naming the channel (not assert)."""
    with pytest.raises(TypeError, match=channel_cls.__name__):
        channel_cls("v", stack=1)


def test_non_stacking_channel_does_not_normalize_stack():
    """A channel with _stack_kwarg=False does not touch a stray stack kwarg."""
    # X2 is positional but not a stacking channel; stack= is not honored and
    # must not be normalized/validated by the base.
    assert fm.X2._stack_kwarg is False


# ---------------------------------------------------------------------------
# CHART-08: two co-located but distinct capability sets
# ---------------------------------------------------------------------------


def test_stackable_marks_membership_preserved():
    """``_STACKABLE_MARKS`` = the renderers that consume __stack_y_base__."""
    assert _STACKABLE_MARKS == frozenset({"bar", "area"})


def test_stack_eligible_membership_preserved():
    """``_STACK_ELIGIBLE`` = the marks that accept position=Stack()."""
    assert _STACK_ELIGIBLE == frozenset(
        {"bar", "area", "ribbon", "text", "point", "rule", "tick", "histogram", "density"}
    )


def test_two_capability_sets_are_distinct():
    """The two sets are genuinely different concepts, not one split in two."""
    assert _STACKABLE_MARKS != _STACK_ELIGIBLE
    assert _STACKABLE_MARKS < _STACK_ELIGIBLE  # base-consumers are a strict subset


def test_chart_module_reexports_stackable_marks():
    """``chart.py`` still exposes ``_STACKABLE_MARKS`` (now imported from position)."""
    from ferrum.chart import _STACKABLE_MARKS as chart_stackable

    assert chart_stackable is _STACKABLE_MARKS
