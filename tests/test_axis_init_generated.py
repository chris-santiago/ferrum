"""``Axis``/``AxisConfig`` construct through the generated dataclass ``__init__`` (#149).

Both classes used to be ``@dataclass(frozen=True, init=False)`` with a
73-line hand-written ``__init__`` that mirrored every field three times
(field declaration, ``__init__`` parameter, ``object.__setattr__`` line) —
see the retired leg 3 of ``test_config_manifest.py`` for the guard that used
to police that mirror. They now declare ``min_extent``/``max_extent`` as
``dataclasses.InitVar`` fields, resolved into the canonical
``min_band``/``max_band`` fields inside ``__post_init__``, so the generated
``__init__`` owns construction and an unassigned field is impossible by
construction rather than merely detected after the fact.

This module pins:

1. Baseline properties (repr, eq, hash, ``fields()``, ``replace``, pickle,
   deepcopy, positional construction, the unknown-kwarg ``TypeError`` text,
   ``to_dict()``) are unchanged by the refactor.
2. ``min_extent=``/``max_extent=`` are excluded from ``dataclasses.fields()``
   (the InitVar mechanism, not merely a documented convention).
3. LB-149-b: the deprecated-alias ``DeprecationWarning`` is attributed to
   the caller's line, not to ``axis.py``/``configure.py`` themselves. This
   is a live behavior change (spec §9): under the old hand-written
   ``__init__`` the warning misattributed to ferrum's own source
   (``stacklevel=2`` was one short); under the new ``__post_init__`` shape
   it is attributed correctly at ``stacklevel=4``.
4. The same defect class as LB-149-b, fixed in the same cycle:
   ``.configure_axis(x=False)``'s ``x=``/``y=`` deprecation warning used to
   attribute to ``_configure_mixin.py``'s own internal ``AxisConfig(...)``
   call site rather than the caller (both before and after the ``#149``
   refactor — no single ``AxisConfig`` constructor stacklevel can serve
   both the direct-construction depth and the mixin depth).
   ``configure_axis`` now warns itself before constructing ``AxisConfig``
   and never forwards ``x=``/``y=`` at all: they are vestigial no-op flags
   (nothing reads ``.x``/``.y`` off a ``configure_axis``-produced config,
   and ``to_dict()`` always pops both), so a config built by
   ``configure_axis(x=False)`` carries the internal default ``x=True``
   (wire-identical either way) rather than round-tripping the deprecated
   input through a second, differently-depthed copy of the same check. An
   earlier, since-reverted version of this fix forwarded ``x=``/``y=`` and
   used ``warnings.catch_warnings()`` to suppress the resulting nested
   re-warning; that touched process-global warnings state (pinned as a
   regression by the behavioral dedup-registry probe and structural guard
   below) and was replaced.
"""

from __future__ import annotations

import copy
import inspect
import pickle
import warnings
from dataclasses import fields, replace

import polars as pl
import pytest

import ferrum as fm
from ferrum._configure_mixin import ConfigureMixin
from ferrum.axis import Axis
from ferrum.configure import AxisConfig


# ---------------------------------------------------------------------------
# Baseline properties — unchanged by the generated-__init__ refactor.
# ---------------------------------------------------------------------------


class TestAxisBaselineProperties:
    def test_repr_round_trips_readable_values(self) -> None:
        ax = Axis(title="Speed", orient="left")
        assert repr(ax) == (
            "Axis(title='Speed', orient='left', ticks=_UNSET, tick_count=None, "
            "tick_extra=_UNSET, tick_min_step=None, grid=_UNSET, grid_dash=None, "
            "grid_width=None, grid_color=None, grid_opacity=None, labels=_UNSET, "
            "label_angle=None, label_flush=_UNSET, label_overlap=_UNSET, "
            "label_format=None, label_format_type=None, label_font_size=None, "
            "label_color=None, domain=_UNSET, domain_width=None, domain_color=None, "
            "offset=None, translate=None, min_band=None, max_band=None, "
            "title_orient=None, title_font_size=None, title_color=None, "
            "title_padding=None, values=None, zindex=None, label_map=None)"
        )

    def test_eq(self) -> None:
        assert Axis(title="A") == Axis(title="A")
        assert Axis(title="A") != Axis(title="B")

    def test_hash_is_stable_and_usable_as_dict_key(self) -> None:
        ax = Axis(title="A", orient="left")
        assert hash(ax) == hash(Axis(title="A", orient="left"))
        assert {ax: 1}[Axis(title="A", orient="left")] == 1

    def test_fields_excludes_min_extent_and_max_extent(self) -> None:
        """InitVars are excluded from ``fields()`` — the mechanism, not a docstring claim."""
        names = {f.name for f in fields(Axis)}
        assert "min_band" in names
        assert "max_band" in names
        assert "min_extent" not in names
        assert "max_extent" not in names

    def test_replace(self) -> None:
        ax = Axis(title="A", orient="left")
        replaced = replace(ax, orient="right")
        assert replaced.orient == "right"
        assert replaced.title == "A"
        assert ax.orient == "left"

    def test_pickle_round_trip(self) -> None:
        # Trusted, locally-created data — same-process round trip, not
        # deserializing anything from an external/untrusted source.
        ax = Axis(title="A", orient="left", min_band=30.0)
        assert pickle.loads(pickle.dumps(ax)) == ax

    def test_deepcopy_round_trip(self) -> None:
        ax = Axis(title="A", values=[1, 2, 3])
        cloned = copy.deepcopy(ax)
        assert cloned == ax
        assert cloned.values is not ax.values

    def test_positional_construction(self) -> None:
        ax = Axis("Speed", "left")
        assert ax.title == "Speed"
        assert ax.orient == "left"

    def test_unknown_kwarg_type_error_text(self) -> None:
        with pytest.raises(TypeError) as exc_info:
            Axis(bogus=1)  # type: ignore[call-arg]
        assert str(exc_info.value) == "Axis.__init__() got an unexpected keyword argument 'bogus'"

    def test_to_dict(self) -> None:
        assert Axis(title="Speed", label_angle=-45).to_dict() == {
            "title": "Speed",
            "label_angle": -45,
        }


class TestAxisConfigBaselineProperties:
    def test_repr_round_trips_readable_values(self) -> None:
        cfg = AxisConfig(label_angle=-45.0)
        assert repr(cfg) == (
            "AxisConfig(x=True, y=True, label_angle=-45.0, label_font_size=None, "
            "label_color=None, label_format=None, label_format_raw=None, "
            "label_overlap=None, tick_count=None, tick_size=None, tick_values=None, "
            "title_font_size=None, title_color=None, title_padding=None, domain=None, "
            "domain_color=None, domain_width=None, grid=None, grid_color=None, "
            "grid_dash=None, grid_width=None, domain_min=None, domain_max=None, "
            "nice=None, zero=None, label_padding=None, grid_opacity=None, orient=None, "
            "translate=None, min_band=None, max_band=None, tick_extra=None, "
            "tick_min_step=None, title_orient=None, zindex=None, labels=None, "
            "ticks=None, title=_UNSET, offset=None, label_flush=None)"
        )

    def test_eq(self) -> None:
        assert AxisConfig(label_angle=5.0) == AxisConfig(label_angle=5.0)
        assert AxisConfig(label_angle=5.0) != AxisConfig(label_angle=6.0)

    def test_hash_is_stable_and_usable_as_dict_key(self) -> None:
        cfg = AxisConfig(label_angle=5.0)
        assert hash(cfg) == hash(AxisConfig(label_angle=5.0))
        assert {cfg: 1}[AxisConfig(label_angle=5.0)] == 1

    def test_fields_excludes_min_extent_and_max_extent(self) -> None:
        names = {f.name for f in fields(AxisConfig)}
        assert "min_band" in names
        assert "max_band" in names
        assert "min_extent" not in names
        assert "max_extent" not in names

    def test_replace(self) -> None:
        cfg = AxisConfig(label_angle=5.0)
        replaced = replace(cfg, label_angle=10.0)
        assert replaced.label_angle == 10.0
        assert cfg.label_angle == 5.0

    def test_pickle_round_trip(self) -> None:
        # Trusted, locally-created data — same-process round trip, not
        # deserializing anything from an external/untrusted source.
        cfg = AxisConfig(label_angle=5.0, min_band=30.0)
        assert pickle.loads(pickle.dumps(cfg)) == cfg

    def test_deepcopy_round_trip(self) -> None:
        cfg = AxisConfig(tick_values=[1, 2, 3])
        cloned = copy.deepcopy(cfg)
        assert cloned == cfg
        assert cloned.tick_values is not cfg.tick_values

    def test_positional_construction(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            cfg = AxisConfig(False, True)
        assert (cfg.x, cfg.y) == (False, True)

    def test_unknown_kwarg_type_error_text(self) -> None:
        with pytest.raises(TypeError) as exc_info:
            AxisConfig(bogus=1)  # type: ignore[call-arg]
        assert (
            str(exc_info.value)
            == "AxisConfig.__init__() got an unexpected keyword argument 'bogus'"
        )

    def test_to_dict(self) -> None:
        assert AxisConfig(label_angle=5.0).to_dict() == {"label_angle": 5.0}


# ---------------------------------------------------------------------------
# LB-149-b — the deprecated band-alias warning is attributed to the caller's
# line, not to ferrum's own axis.py / configure.py.
# ---------------------------------------------------------------------------


def _sole_deprecation_warning(caught: list[warnings.WarningMessage]) -> warnings.WarningMessage:
    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(deprecations) == 1, f"expected exactly one DeprecationWarning, got {deprecations}"
    return deprecations[0]


class TestBandAliasWarningLocation:
    def test_axis_min_extent_warning_attributes_to_caller_line(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            expected_line = inspect.currentframe().f_lineno + 1  # type: ignore[union-attr]
            Axis(min_extent=30)
        warning = _sole_deprecation_warning(caught)
        assert warning.filename == __file__
        assert warning.lineno == expected_line

    def test_axis_max_extent_warning_attributes_to_caller_line(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            expected_line = inspect.currentframe().f_lineno + 1  # type: ignore[union-attr]
            Axis(max_extent=30)
        warning = _sole_deprecation_warning(caught)
        assert warning.filename == __file__
        assert warning.lineno == expected_line

    def test_axisconfig_min_extent_warning_attributes_to_caller_line(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            expected_line = inspect.currentframe().f_lineno + 1  # type: ignore[union-attr]
            AxisConfig(min_extent=30)
        warning = _sole_deprecation_warning(caught)
        assert warning.filename == __file__
        assert warning.lineno == expected_line

    def test_axisconfig_max_extent_warning_attributes_to_caller_line(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            expected_line = inspect.currentframe().f_lineno + 1  # type: ignore[union-attr]
            AxisConfig(max_extent=30)
        warning = _sole_deprecation_warning(caught)
        assert warning.filename == __file__
        assert warning.lineno == expected_line

    def test_axisconfig_x_false_warning_still_attributes_to_caller_line(self) -> None:
        """Regression control: the unrelated x=/y= deprecation warning (resolved
        directly in __post_init__, not through the InitVar alias machinery) must
        keep attributing correctly — the extra __post_init__ frame this refactor
        introduces must not silently regress it (its stacklevel is bumped from
        2 to 3 to compensate, alongside the band-alias fix)."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            expected_line = inspect.currentframe().f_lineno + 1  # type: ignore[union-attr]
            AxisConfig(x=False)
        warning = _sole_deprecation_warning(caught)
        assert warning.filename == __file__
        assert warning.lineno == expected_line

    def test_configure_axis_x_false_warning_attributes_to_caller_line(self) -> None:
        """The same defect class as LB-149-b, fixed in the same cycle: through
        ``.configure_axis(x=False)`` this warning used to attribute to
        ``_configure_mixin.py``'s own internal ``AxisConfig(...)`` call site
        (not the caller), for both the pre-#149 hand-written ``__init__`` and
        the post-#149 ``__post_init__`` shape — no single ``AxisConfig``
        constructor stacklevel can serve both the direct-construction call
        depth and the mixin call depth. ``configure_axis`` now warns itself,
        correctly attributed, and never forwards ``x=``/``y=`` to
        ``AxisConfig`` at all (they are vestigial no-op flags nothing reads
        back), so ``AxisConfig``'s own identical check never fires — no
        ``warnings.catch_warnings()`` suppression, nothing to suppress, and
        no ``object.__setattr__`` round-trip either. Exactly one warning
        reaches the caller, not zero and not two."""
        df = pl.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
        chart = fm.Chart(df).mark_point().encode(x="x", y="y")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            expected_line = inspect.currentframe().f_lineno + 1  # type: ignore[union-attr]
            chart.configure_axis(x=False)
        warning = _sole_deprecation_warning(caught)
        assert warning.filename == __file__
        assert warning.lineno == expected_line

    def test_configure_axis_does_not_reset_the_warnings_dedup_registry(self) -> None:
        """Behavioral probe replacing the S3-flagged negative control this fix
        replaces (quality-reviewer cycle 4): the removed
        ``warnings.filters``-identity check would have passed even under the
        removed ``warnings.catch_warnings()`` mechanism (``catch_warnings()``
        restores ``warnings.filters`` to its prior value on ``__exit__``, so a
        transient swap self-heals before that check ever ran) — a
        non-discriminating assertion the binding constraint forbids. This
        replaces it with the actual observable symptom: ``catch_warnings()``
        also bumps a global filters-version counter on exit that invalidates
        every module's ``__warningregistry__`` dedup cache, so an unrelated
        warning that should show once (Python's ``"default"`` action
        deduplicates same message/category/module/lineno) shows again after a
        ``configure_axis`` call that used ``catch_warnings()`` internally.
        Scoped entirely inside this test's own ``catch_warnings(record=True)``
        — test code, not library state, so this does not reintroduce the
        global-mutation problem it is checking for."""
        df = pl.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
        chart = fm.Chart(df).mark_point().encode(x="x", y="y")

        def _emit_control_warning() -> None:
            warnings.warn("axis-init-generated dedup-registry control", UserWarning)

        with warnings.catch_warnings(record=True) as caught:
            warnings.resetwarnings()
            warnings.simplefilter("default")
            _emit_control_warning()
            chart.configure_axis(labels=False)
            _emit_control_warning()

        control = [w for w in caught if issubclass(w.category, UserWarning)]
        assert len(control) == 1, (
            f"expected the control warning to show exactly once (default-action "
            f"dedup across two identical calls), got {len(control)} — a "
            "configure_axis call must not reset the warnings dedup registry"
        )

    def test_configure_axis_never_touches_process_global_warnings_state(self) -> None:
        """Structural guard matching the decided mechanism verbatim: 'no
        ``warnings.catch_warnings``/``filterwarnings`` in library code at
        all' (it mutates process-global filter state and resets every
        module's ``__warningregistry__`` on exit — see the behavioral probe
        above for the observable symptom this mechanism produces). Pins the
        mechanism inside ``configure_axis``'s own body; the behavioral probe
        above carries the actual regression coverage (this guard would miss a
        violation hidden behind a helper call)."""
        source = inspect.getsource(ConfigureMixin.configure_axis)
        assert "catch_warnings(" not in source
        assert "filterwarnings(" not in source
