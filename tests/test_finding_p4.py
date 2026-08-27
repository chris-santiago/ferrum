"""Regression tests for finding P4 (2026-08-27 batch): coercion consolidation.

Task 4 collapsed three "coerce chart data to polars" authorities
(``chart.py::_to_polars``, ``plots/_helpers.py::_to_polars`` — a byte-identical
twin — and ``plots/_helpers.py::_coerce_to_polars``) into one canonical
``ferrum._coerce.to_polars``, called by every former call site. The former
``_coerce_to_polars`` (now a private helper in ``plots/ranking.py``, the only
consumer) had a pandas duck-test branch that called ``pl.from_pandas`` directly,
skipping ``to_arrow_table``'s datetime-unit normalization — a live bug on the
``parallel_coordinates`` path. A spec-review sweep found the same defect shape
survived in ``plots/clustering.py::pca_scree_chart``'s raw-DataFrame path,
also fixed here.

These tests pin: (a) the parallel_coordinates bug fix (pandas datetime64[ns]
now normalizes to ``ms``), (b) the 2D-ndarray / list-of-lists fallback is
unchanged, (c) no duplicate ``_to_polars`` definition and no stray
``pl.from_pandas`` call survives outside ``_coerce``, (d) ``_coerce.py``'s
leaf-module import invariant, (e) ``pca_scree_chart``'s raw-pandas path
routes through the canonical ``to_polars`` instead of ``pl.from_pandas``,
and (f) a named frame whose conversion fails raises loudly rather than
silently degrading to ``col_0``/``col_1`` axis labels (quality-review round:
``ranking._coerce_to_polars``'s fallback used to key on "``to_polars`` raised
``TypeError``", but ``pyarrow.lib.ArrowTypeError`` is a ``TypeError``
subclass, so a genuinely named frame that fails conversion silently lost its
column names -- the fallback is now gated on input *type*
(``list``/``tuple``) instead).
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

import ferrum._coerce as _coerce_mod
import ferrum.plots.clustering as _clustering_mod
from ferrum._coerce import to_polars
from ferrum.plots.ranking import _coerce_to_polars, parallel_coordinates_chart

_COERCE_PATH = Path(_coerce_mod.__file__).resolve()


def _src_python_files_excluding_coerce():
    """Yield every ``src/ferrum/**/*.py`` file except ``_coerce.py`` itself.

    Shared by the two structural (AST-grep) tests below.
    """
    src_root = Path(__file__).resolve().parent.parent / "src" / "ferrum"
    for path in src_root.rglob("*.py"):
        if path.resolve() == _COERCE_PATH:
            continue
        yield path


def test_parallel_coordinates_normalizes_pandas_datetime_to_ms():
    """Regression: a pandas datetime64[ns] frame through parallel_coordinates
    must be ms-normalized, matching every other coercion call site.

    Pre-fix, ``_coerce_to_polars``'s pandas duck-test branch called
    ``pl.from_pandas(data)`` directly, bypassing ``to_arrow_table``'s
    timestamp[ns]->timestamp[ms] normalization, so the resulting "value"
    column stayed at nanosecond precision instead of the renderer's
    canonical millisecond unit.
    """
    pdf = pd.DataFrame(
        {
            "dt": pd.to_datetime(["2024-01-01T00:00:00.123", "2024-01-02T00:00:00.987"]),
        }
    )
    chart = parallel_coordinates_chart(pdf, features=["dt"], rescale=None)
    value_dtype = chart._data.schema["value"]
    assert value_dtype == pl.Datetime("ms"), (
        f"expected ms-normalized temporal value column, got {value_dtype!r}"
    )


def test_coerce_to_polars_2d_ndarray_unchanged():
    """Regression: a bare 2D numpy array still auto-names col_0, col_1, ...

    ``to_arrow_table`` already accepts 2D numpy arrays directly (same naming
    convention), so this exercises the canonical path, not the ranking-local
    fallback.
    """
    arr = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    df = _coerce_to_polars(arr)
    assert df.columns == ["col_0", "col_1"]
    assert df["col_0"].to_list() == [1.0, 3.0, 5.0]
    assert df["col_1"].to_list() == [2.0, 4.0, 6.0]


def test_coerce_to_polars_list_of_lists_unchanged():
    """Regression: list-of-lists input still auto-names col_0, col_1, ....

    ``to_arrow_table`` rejects a plain (non-dict) list with a ``TypeError``,
    so this exercises the ranking-local fallback branch that ``to_polars``
    cannot itself serve.
    """
    lol = [[1.0, 2.0], [3.0, 4.0]]
    df = _coerce_to_polars(lol)
    assert df.columns == ["col_0", "col_1"]
    assert df["col_0"].to_list() == [1.0, 3.0]
    assert df["col_1"].to_list() == [2.0, 4.0]


def test_coerce_to_polars_named_frame_conversion_failure_raises():
    """Regression: a named DataFrame whose conversion fails must raise, not
    silently degrade to ``col_0``/``col_1`` axis labels.

    Quality-review finding: gating the ranking-local fallback on "``to_polars``
    raised a ``TypeError``" is unsound because ``pyarrow.lib.ArrowTypeError``
    (and narwhals' own conversion-failure surface) are both ``TypeError``
    subclasses. A pandas frame with a sparse-dtype column reproduces exactly
    this: ``to_arrow_table`` can't convert it and raises ``TypeError``, but
    the frame is numpy-coercible via ``np.asarray(..., dtype=float64)`` -- so
    the old exception-typed gate would silently rename its real columns
    (``a``, ``b``) to ``col_0``/``col_1`` instead of propagating the failure.
    The fallback is now gated on input *type* (``list``/``tuple`` only), so a
    ``pd.DataFrame`` never reaches it and this raises loudly.
    """
    pdf = pd.DataFrame(
        {
            "a": pd.arrays.SparseArray([1.0, 0.0, 2.0]),
            "b": [1.0, 2.0, 3.0],
        }
    )
    with pytest.raises(TypeError):
        _coerce_to_polars(pdf)


def test_coerce_to_polars_list_and_tuple_gate_is_type_based():
    """Regression: the ranking-local fallback is reachable for ``tuple``
    inputs too, not just ``list`` (the gate is ``isinstance(data, (list,
    tuple))``), and a tuple-of-tuples produces the same ``col_0, col_1, ...``
    frame as the list-of-lists case.
    """
    tot = ((1.0, 2.0), (3.0, 4.0))
    df = _coerce_to_polars(tot)
    assert df.columns == ["col_0", "col_1"]
    assert df["col_0"].to_list() == [1.0, 3.0]
    assert df["col_1"].to_list() == [2.0, 4.0]


def test_no_duplicate_to_polars_definition():
    """Grep-proof: no ``def _to_polars`` survives outside ``ferrum._coerce``.

    Structural completeness check for the P4 consolidation -- the finding was
    that three authorities existed; this pins that only one ``to_polars``
    definition remains in the package.
    """
    offenders = []
    for path in _src_python_files_excluding_coerce():
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name in ("_to_polars", "to_polars"):
                offenders.append(str(path))
    assert not offenders, f"found duplicate to_polars definition(s): {offenders}"


def test_no_stray_pl_from_pandas_call():
    """Grep-proof: no ``pl.from_pandas(...)`` call survives outside
    ``ferrum._coerce``.

    A spec-review sweep on this task found a surviving normalization-bypassing
    ``pl.from_pandas(model)`` call in ``plots/clustering.py::pca_scree_chart``
    -- the exact defect shape P4 removes elsewhere (skips ``to_arrow_table``'s
    datetime/categorical normalization). This is the completeness check the
    decision record's P4 section named explicitly.

    Scoped to the ``pl`` alias specifically (the only alias ``polars`` is
    ever imported under in ``src/ferrum``) rather than every attribute named
    ``from_pandas`` package-wide, so an unrelated ``nw.from_pandas`` or
    ``pa.Table.from_pandas`` in a non-chart-data context does not false-fire.
    """
    offenders = []
    for path in _src_python_files_excluding_coerce():
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "from_pandas"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "pl"
            ):
                offenders.append(f"{path}:{node.lineno}")
    assert not offenders, f"found stray pl.from_pandas() call(s) outside _coerce: {offenders}"


def test_coerce_module_is_a_leaf():
    """Regression: ``ferrum._coerce`` must import nothing from ``ferrum``.

    The leaf invariant (spec §7) is what lets ``to_arrow_table``/``to_polars``
    be adopted anywhere in the package without inverting the dependency graph.
    """
    tree = ast.parse(Path(_coerce_mod.__file__).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        for name in names:
            assert not name.startswith("ferrum"), f"leaf module imports {name!r}"


def test_to_polars_passthrough_and_arrow_backed():
    """Sanity: ``to_polars`` passes through an existing DataFrame and
    otherwise routes through ``to_arrow_table``.
    """
    df = pl.DataFrame({"a": [1, 2], "b": [3.0, 4.0]})
    assert to_polars(df) is df

    from_dict = to_polars({"a": [1, 2], "b": [3.0, 4.0]})
    assert isinstance(from_dict, pl.DataFrame)
    assert from_dict["a"].to_list() == [1, 2]


def test_coerce_to_polars_raises_for_unsupported_1d_array():
    """Regression: a 1D numpy array still raises the ``to_arrow_table``
    "needs column names" TypeError rather than silently succeeding via the
    ranking-local fallback (which only accepts 2D shapes).
    """
    with pytest.raises(TypeError, match="1D numpy arrays need column names"):
        _coerce_to_polars(np.array([1.0, 2.0, 3.0]))


def test_pca_scree_chart_raw_pandas_uses_canonical_to_polars(monkeypatch):
    """Regression: ``pca_scree_chart``'s raw-DataFrame path must route a
    pandas input through the canonical ``to_polars``, not a direct
    ``pl.from_pandas`` call.

    Pre-fix, the raw-data branch called ``pl.from_pandas(model)`` for any
    non-polars, non-ndarray input -- the same normalization-bypassing shape
    P4 removed from ``_coerce_to_polars``. Patches ``pl.from_pandas`` to
    fail loudly (proving it is never reached) and spies on ``to_polars`` to
    prove it is the coercion actually used.
    """
    calls = []
    original_to_polars = _clustering_mod.to_polars

    def _spy(data):
        calls.append(data)
        return original_to_polars(data)

    monkeypatch.setattr(_clustering_mod, "to_polars", _spy)
    monkeypatch.setattr(
        _clustering_mod.pl,
        "from_pandas",
        lambda *a, **k: pytest.fail("pca_scree_chart must not call pl.from_pandas directly"),
    )

    pdf = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "b": [5.0, 3.0, 4.0, 1.0, 2.0],
            "c": [2.0, 2.0, 3.0, 5.0, 1.0],
        }
    )
    chart = _clustering_mod.pca_scree_chart(pdf, n_components=2)
    assert len(calls) == 1
    assert calls[0] is pdf
    assert chart is not None
