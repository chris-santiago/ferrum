"""Batch-D T12: Python adopts `ferrum._core.transform_registry` as the
single source of truth for mark-level closed vocabularies.

`_VALID_KERNELS` (`ferrum/marks/statistical.py`), `_VALID_QQ_DISTRIBUTIONS`
and the local `_VALID_AGGREGATES` (`ferrum/marks/heavy_stat.py`) were
hand-mirrored copies of Rust vocabularies that could silently drift (the
class of bug batch-D T11 fixed for kernels/distributions). This round
deletes all three literals; the desugar functions call
`transform_registry(kind)["methods"]` live instead. `mark_smooth` gains a
NEW Python-side method gate (previously validated only by the Rust `Smooth`
constructor), deriving from `transform_registry("smooth")["methods"]`.

Each test below is a behavioral proof against the real gate -- not a
self-referential "derived value equals itself" check, which would be
vacuous now that there is no second literal to compare against.
"""

from __future__ import annotations

import polars as pl
import pytest

import ferrum as fm
from ferrum import _core


def test_mark_hex_accepts_every_registry_aggregate():
    df = pl.DataFrame({"x": [0.0, 1.0, 2.0, 3.0, 4.0], "y": [0.0, 1.0, 2.0, 1.0, 0.0]})
    aggregates = _core.transform_registry("hex")["methods"]
    assert set(aggregates) == {"count", "mean", "sum", "min", "max", "median", "std", "var"}
    for aggregate in aggregates:
        field = None if aggregate == "count" else "y"
        svg = fm.Chart(df).mark_hex(aggregate=aggregate, field=field).encode(x="x", y="y").to_svg()
        assert svg.startswith("<svg")


def test_mark_hex_unknown_aggregate_still_refused_by_python_gate():
    df = pl.DataFrame({"x": [0.0, 1.0, 2.0], "y": [0.0, 1.0, 2.0]})
    with pytest.raises(ValueError, match="aggregate"):
        fm.Chart(df).mark_hex(aggregate="not-an-aggregate").encode(x="x", y="y").to_svg()


def test_mark_smooth_accepts_every_registry_method():
    """NEW gate: `mark_smooth`'s `method=` is now validated in Python
    against `transform_registry("smooth")["methods"]` before the `Smooth`
    transform is constructed -- previously there was no Python-side check
    at all (an unknown spelling only failed once it reached Rust)."""
    df = pl.DataFrame(
        {"x": [float(i) for i in range(1, 21)], "y": [float(i) + (-1) ** i for i in range(20)]}
    )
    methods = _core.transform_registry("smooth")["methods"]
    assert set(methods) == {
        "lm",
        "linear",
        "loess",
        "quadratic",
        "cubic",
        "log",
        "sqrt",
        "glm",
        "gam",
    }
    for method in methods:
        svg = fm.Chart(df).mark_smooth(method=method).encode(x="x", y="y").to_svg()
        assert svg.startswith("<svg")


def test_mark_smooth_unknown_method_refused_by_python_gate_before_rust():
    """The new Python-side gate raises before any `Smooth` transform is
    ever constructed -- proven by matching `validate_choice`'s own message
    shape (`mark_smooth: method must be one of [...]`), not the Rust
    engine's separate 'unknown method' wording."""
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="mark_smooth: method must be one of"):
        fm.Chart(df).mark_smooth(method="not-a-method").encode(x="x", y="y").to_svg()


def test_mark_density_kernel_and_mark_qq_distribution_gates_derive_live():
    """Companion pin to `test_density_kernels.py`'s behavioral checks --
    confirms all three gates read straight from the registry rather than a
    module constant that happens to have the same values today. Behaviour
    alone cannot tell those two apart, which is why this reads the source.

    That the deleted constants are *gone* is a separate claim, owned by
    `test_transform_registry.py::test_deleted_mirror_literal_does_not_reappear_anywhere_in_src`,
    which greps all of `src/ferrum/` rather than the three modules named here.
    """
    import inspect

    from ferrum.marks import heavy_stat, statistical

    assert 'transform_registry("kde")' in inspect.getsource(statistical.desugar_density)
    assert 'transform_registry("qq")' in inspect.getsource(heavy_stat.desugar_qq)
    assert 'transform_registry("hex")' in inspect.getsource(heavy_stat.desugar_hex)
    assert 'transform_registry("smooth")' in inspect.getsource(statistical.desugar_smooth)
