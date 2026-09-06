"""Package-level access to the Rust transform registry.

``ferrum._core.transform_registry(kind)`` is the single source of truth for
what a transform emits: its primary output columns (``outputs``), its
additional named batches (``secondary_outputs``), its method vocabulary
(``methods``) and its accepted parameters (``params``). This module holds the
Python-side readers of the two output halves, so a consumer that reads a
column out of a transform's batch names it through a registry-checked lookup
rather than a bare string literal (spec §4.6 / decision M6a).

Homed here rather than in ``marks/_desugar_helpers.py`` (its batch-D
birthplace): consumers live in ``marks/``, ``plots/`` and
``_layer_transforms.py`` alike, and nothing about registry access is
mark- or desugar-specific. ``plots/`` depending on ``marks/`` for a
``ferrum._core`` accessor inverted the intended layering.
"""

from __future__ import annotations

from collections.abc import Sequence


def registry_output_columns(kind: str, names: Sequence[str]) -> dict[str, str]:
    """Validate a desugar/figure consumer's output-column names against
    ``transform_registry(kind)["outputs"]`` and return them as a lookup.

    *names* are the column names the consumer reads out of *kind*'s PRIMARY
    output batch. The return is ``{name: name}`` so a call site indexes it by
    the same name it would otherwise have written as a bare literal
    (``_smooth_cols["ci_lower"]``). What this converts is the *shape* of the
    reference: a bare string literal embedded directly in an
    ``encoding=``/``mark_kwargs=``/``remap=`` dict becomes a lookup checked
    against the registry at desugar time, so a Rust-side rename or removal of
    a declared output column raises here -- naming the stale column --
    instead of silently reading a column the transform no longer emits
    (spec §4.6 / decision M6a).

    Deliberately NOT a positional ``outputs[i][0]`` lookup: ``outputs``
    carries no positional contract (the registry only promises *membership*,
    not order), so indexing into it would encode an ordering dependency the
    registry never guaranteed.

    For a column of one of *kind*'s additional named batches, use
    :func:`registry_secondary_columns` -- validating a secondary column
    against the primary schema would pin a schema the call site never reads.

    Parameters
    ----------
    kind : str
        The transform kind name passed to ``transform_registry``.
    names : sequence of str
        The output-column names this consumer reads.

    Returns
    -------
    dict of str to str
        ``{name: name}`` for every name in *names*, in the order given, so a
        call site that needs the roles as a sequence can iterate the lookup
        instead of re-listing them alongside it.

    Raises
    ------
    ValueError
        If any name is not declared in
        ``transform_registry(kind)["outputs"]``.
    """
    from ferrum._core import transform_registry

    declared = {name for name, _dtype in transform_registry(kind)["outputs"]}
    missing = sorted({name for name in names if name not in declared})
    if missing:
        raise ValueError(
            f"transform_registry({kind!r})['outputs'] does not declare "
            f"{missing}; known output columns: {sorted(declared)}"
        )
    return {name: name for name in names}


def registry_secondary_columns(kind: str, key: str, names: Sequence[str]) -> dict[str, str]:
    """The :func:`registry_output_columns` discipline for a SECONDARY batch.

    Three kinds publish additional named batches beside their primary output
    -- ``qq`` (the reference line), ``letter_value`` (the outlier rows plus
    one band batch per depth) and ``linkage`` (the linkage table plus the
    dendrogram order/coords/segments). Their column names are the
    least-guessable in the crate (``qq_line_x_start``, ``lv_depth_3``'s
    ``lower``/``upper``, the dendrogram ``x2``/``y2``), and until the Rust
    registry declared them (2026-09-06) every Python consumer read them as a
    bare literal with a dispositioned "not registry-declared" comment beside
    it. They are declared now, so they check like any other column.

    *key* is the batch's key under DEFAULT construction (``name=None``) --
    ``"qq_line"``, ``"depth_3"``, ``"segments"`` -- and is itself validated,
    so a Rust-side rename of the key raises here too.

    The wire name a layer's ``data_source=`` must use is NOT uniformly
    derivable from *key*: ``letter_value`` and ``linkage`` prefix a
    caller-supplied ``name=`` onto every key (``"lv_depth_3"``,
    ``"row_link_segments"``), while ``qq`` publishes ``"qq_line"``
    unprefixed whatever its spec is named. Rather than encode a rule that
    holds for two of the three kinds, this helper stays scoped to columns
    and each call site composes its own ``data_source`` from the key it
    validated here.

    Parameters
    ----------
    kind : str
        The transform kind name passed to ``transform_registry``.
    key : str
        The secondary batch's declared key, under default construction.
    names : sequence of str
        The column names this consumer reads out of that batch.

    Returns
    -------
    dict of str to str
        ``{name: name}`` for every name in *names*, in the order given.

    Raises
    ------
    ValueError
        If *kind* declares no secondary batch under *key*, or if any name is
        not declared in that batch's schema.
    """
    from ferrum._core import transform_registry

    declared_batches = dict(transform_registry(kind)["secondary_outputs"])
    if key not in declared_batches:
        raise ValueError(
            f"transform_registry({kind!r})['secondary_outputs'] declares no "
            f"{key!r} batch; known keys: {sorted(declared_batches)}"
        )
    declared = {name for name, _dtype in declared_batches[key]}
    missing = sorted({name for name in names if name not in declared})
    if missing:
        raise ValueError(
            f"transform_registry({kind!r})['secondary_outputs'][{key!r}] does "
            f"not declare {missing}; known columns: {sorted(declared)}"
        )
    return {name: name for name in names}
