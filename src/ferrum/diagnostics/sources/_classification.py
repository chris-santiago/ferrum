"""Phase 10b — classification curves (ROC, PR, calibration, gain, lift, discrimination threshold, confusion matrix)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from ferrum._validate import validate_choice
from .._internal import _curve_frames
from .._internal.deps import require_sklearn

if TYPE_CHECKING:
    from ._protocols import _SourceState as _MixinBase
else:
    _MixinBase = object


def _resolve_proba(method_name: str, y, proba_df: pl.DataFrame, real_classes: Any = None):
    """Resolve proba columns and y_true from a probabilities DataFrame.

    Returns ``(y_true_np, score_matrix, labels, class_values)`` where:
    - ``y_true_np`` is the ground-truth as a numpy array
    - ``score_matrix`` is ``(n_samples, n_classes)`` float64
    - ``labels`` is a list of class label strings (stripped ``"proba_"`` prefix),
      used for *display* only (the chart's ``class`` discriminator column)
    - ``class_values`` is a list of values in ``y_true``'s domain, used to
      binarize ``y_true == class_values[i]`` for each class ``i``

    Parameters
    ----------
    real_classes : array-like, optional
        The wrapped model's ``classes_``, positionally aligned with
        ``proba_df``'s ``proba_<label>`` columns regardless of what
        display label each column was given (see
        :func:`_coerce_real_classes`). When given and its length matches
        ``labels``, it is coerced into ``y_true``'s domain and used for
        ``class_values`` instead of parsing the display label. ``None``
        (the default) falls back to parsing the label itself, the only
        option left when the model exposes no ``classes_`` at all.

    Raises ``ValueError`` when ``y`` is ``None``.
    """
    if y is None:
        raise ValueError(f"ModelSource.{method_name}() requires y to be provided.")
    proba_cols = [c for c in proba_df.columns if c.startswith("proba_")]
    y_true = np.asarray(y)
    labels = [c[len("proba_") :] for c in proba_cols]
    class_values = _resolve_class_values(real_classes, y, labels)
    score_matrix = proba_df[proba_cols].to_numpy()
    return y_true, score_matrix, labels, class_values


def _resolve_class_values(real_classes: Any, y, labels: list[str]) -> list:
    """Per-class comparison values in ``y``'s domain, in score-column order.

    Every classification-curve method binarizes ``y_true == class_values[i]``,
    so these values must live in ``y``'s domain. Three tiers, in order:

    1. **The model's own ``classes_``**, coerced into ``y``'s dtype
       (:func:`_coerce_real_classes`). Positionally aligned with the
       ``proba_`` columns by ``predict_proba``'s contract.
    2. **``y``'s own distinct values**, ascending, when there are exactly as
       many as there are score columns (:func:`_curve_frames
       .binary_class_values` supplies the binary pair; the multiclass case
       is the same rule over ``np.unique``). This is the sibling rule the
       precomputed path already uses, and it assumes the same ascending
       class order every score matrix here assumes.
    3. **The display labels**, parsed back into ``y``'s dtype -- the last
       resort, only when neither of the above can produce the right count.

    Tier 2 exists because tier 3 is *unsound as a general fallback*, which
    the mandatory binarization turned from latent into live (2026-09-10
    close review, cycle 3). ``probabilities()`` names its columns from
    ``class_names``, which is either a user-supplied display alias or a
    synthetic ``class_{i}`` when the model exposes no ``classes_``. Neither
    is a value ``y`` contains, so binarizing against it zeroes the indicator
    and collapses every binary curve -- the batch's own
    display-label-as-comparison-value defect, inside the resolution helper
    written to prevent it. Tier 3 now only runs when ``y``'s cardinality
    disagrees with the score-column count, where no sound answer exists and
    a parsed label is at least a guess made from the caller's own spelling.

    All three tiers are reachable, and measured (2026-09-11 close mutation
    gate): tier 1 on any scikit-learn estimator, tier 2 on a ``classes_``-less
    model whose ``y`` carries exactly as many distinct values as there are
    score columns (verified to match sklearn's one-vs-rest AUCs), tier 3 on
    a ``classes_``-less model whose ``y`` cardinality disagrees with that
    count. Tier 3's two routes differ in how the unsound values surface:
    a **binary** curve refuses by name, because
    :func:`_curve_frames.binary_indicator`'s total-mismatch guard catches
    comparison values that match no row. A **multiclass** curve has no such
    guard -- one-vs-rest legitimately tolerates a class no row matches --
    so it returns all-NaN curves silently instead. That silence is a known
    gap, tracked with the target-hygiene follow-up (the same "what does
    ferrum owe a caller whose target and model disagree" question), not a
    property this function is asserting is correct.
    """
    if real_classes is not None and len(real_classes) == len(labels):
        return _coerce_real_classes(real_classes, y.dtype)

    y_true = np.asarray(y)
    if len(labels) == 2:
        values, _ = _curve_frames.binary_class_values(y_true)
        if len(np.unique(y_true)) == 2:
            return values
    else:
        observed = np.unique(y_true).tolist()
        if len(observed) == len(labels):
            return observed

    return [_coerce_class_label(label, y_true.dtype) for label in labels]


def _positive_class_value(model: Any, y, proba_cols: list[str]) -> object:
    """Resolve the value in ``y``'s domain that means "positive".

    The binary slice of :func:`_resolve_class_values`: the positive class is
    the second of the two, matching ``predict_proba``'s column order.

    Shared by the two binary-only diagnostics that binarize ``y`` against a
    single class -- ``calibration_curve`` and ``discrimination_threshold``
    -- so they cannot resolve "positive" two different ways, and shared with
    ``roc_curve``/``pr_curve``/``cumulative_gain``/``lift_curve`` through the
    common resolver, so neither can the two families.
    """
    labels = [c[len("proba_") :] for c in proba_cols]
    return _resolve_class_values(getattr(model, "classes_", None), y, labels)[1]


def _predictions_in_y_domain(
    real_classes: Any, encode_values: list, y_pred: np.ndarray
) -> np.ndarray:
    """Translate predicted class values from ``classes_``'s domain into ``y``'s.

    ``model.predict`` returns values drawn from ``model.classes_``, which
    carries the *model's* dtype, while ``confusion_frame`` encodes against
    comparison values living in ``y``'s dtype (see
    :func:`_coerce_real_classes`). When the two dtypes differ -- an
    int-coded ``classes_`` against a ``Utf8``/``Categorical``/``Enum`` ``y``
    is the common shape -- every predicted value misses the encode map,
    the kernel drops the pair as out-of-vocabulary, and the rendered
    confusion matrix is **entirely zeros** (found by the batch G close's
    own dtype matrix, 2026-09-10).

    ``classes_`` and ``encode_values`` are the same classes in the two
    domains, positionally aligned by construction, so the translation is
    that correspondence applied per prediction -- not a second, separately
    derived cast that could disagree with the encoding. A value not in
    ``classes_`` is passed through untouched and drops as before.
    """
    translation = dict(zip(np.asarray(real_classes).tolist(), encode_values))
    return np.array([translation.get(value, value) for value in y_pred.tolist()], dtype=object)


class ClassificationCurvesMixin(_MixinBase):
    """Phase 10b — classification curves (ROC, PR, calibration, gain, lift, discrimination threshold, confusion matrix)."""

    # --- 10b: classification curves --------------------------------------

    def roc_curve(
        self,
        *,
        average: str | None = None,
        drop_intermediate: bool = True,
    ) -> pl.DataFrame:
        """ROC curve(s). One row per (class, threshold). ``auc`` repeats per class.

        For binary classifiers, returns a single curve on the positive
        (second) class. For multiclass, returns one-vs-rest curves per class.

        Parameters
        ----------
        average : {None, "micro", "macro", "weighted"}, default None
            Multiclass averaging. ``None`` returns per-class one-vs-rest
            curves (or the single positive-class curve for binary
            classifiers); ``"micro"`` / ``"macro"`` / ``"weighted"``
            additionally include a summary curve under ``class="<average>"``.
        drop_intermediate : bool, default True
            Drop collinear ROC points that do not change the curve's shape
            (sklearn's ``drop_intermediate``), yielding a lighter frame.
        """
        key = self._cache_key(
            "roc_curve",
            average=average,
            drop_intermediate=drop_intermediate,
        )
        if key in self._cache:
            return self._cache[key]
        require_sklearn("roc_curve")

        proba_df = self.probabilities()
        y_true, score_matrix, labels, class_values = _resolve_proba(
            "roc_curve", self._y, proba_df, real_classes=getattr(self._model, "classes_", None)
        )
        df = _curve_frames.roc_frame(
            y_true,
            score_matrix,
            class_values,
            labels,
            average=average,
            drop_intermediate=drop_intermediate,
        )
        self._cache[key] = df
        return df

    def pr_curve(self, *, average: str | None = None) -> pl.DataFrame:
        """Precision-recall curve(s). One row per (class, threshold).

        For binary classifiers, returns a single curve on the positive
        (second) class. For multiclass, ``average=None`` returns
        one-vs-rest curves per class, while ``average`` in
        {"micro", "macro", "weighted"} returns a single summary curve with
        ``class="<average>"`` and no per-class rows. Macro / weighted
        variants interpolate per-class precision over a shared recall grid
        (100 points); micro ravels the binarized labels into one curve.

        ``threshold`` is NaN at the final (recall=0) point of every
        per-class curve per sklearn's convention. For macro / weighted
        summaries it is NaN on every row (recall-grid interpolation
        discards thresholds); micro follows sklearn's padding convention.

        Parameters
        ----------
        average : {None, "micro", "macro", "weighted"}, default None
            Multiclass averaging strategy. Accepted but inert for binary
            classifiers, which have only one curve to draw.
        """
        if average is not None:
            validate_choice(
                "ModelSource.pr_curve", "average", average, ("micro", "macro", "weighted")
            )
        key = self._cache_key("pr_curve", average=average)
        if key in self._cache:
            return self._cache[key]
        require_sklearn("pr_curve")

        proba_df = self.probabilities()
        y_true, score_matrix, labels, class_values = _resolve_proba(
            "pr_curve", self._y, proba_df, real_classes=getattr(self._model, "classes_", None)
        )
        df = _curve_frames.pr_frame(
            y_true,
            score_matrix,
            class_values,
            labels,
            average=average,
        )
        self._cache[key] = df
        return df

    def calibration_curve(
        self,
        *,
        n_bins: int = 10,
        strategy: str = "uniform",
    ) -> pl.DataFrame:
        """Calibration (reliability) curve for binary classifiers.

        Returns one row per non-empty bin with ``mean_predicted``,
        ``fraction_positive``, and ``count``, plus a broadcast ``brier``
        column carrying the per-sample Brier score of the predictions the
        bins were built from. Delegates to the ``calibration_kernel`` Rust
        kernel.

        ``y`` is binarized against the positive class
        (:func:`_positive_class_value`) before binning, so every binary
        label convention -- ``{0, 1}``, ``{-1, 1}``, ``{1, 2}``, string
        labels -- yields the same reliability curve and the same Brier
        score for the same probabilities.

        Parameters
        ----------
        n_bins : int, default 10
            Number of bins used to group predicted probabilities.
        strategy : {"uniform", "quantile"}, default "uniform"
            Bin-edge strategy (matches ``sklearn.calibration``).
            ``"uniform"`` uses equal-width bins; ``"quantile"`` uses bins
            holding an equal number of samples.
        """
        key = self._cache_key(
            "calibration_curve",
            n_bins=n_bins,
            strategy=strategy,
        )
        if key in self._cache:
            return self._cache[key]
        require_sklearn("calibration_curve")

        if self._y is None:
            raise ValueError("ModelSource.calibration_curve() requires y to be provided.")
        proba_df = self.probabilities()
        proba_cols = [c for c in proba_df.columns if c.startswith("proba_")]
        if len(proba_cols) != 2:
            raise ValueError(
                f"calibration_curve() is binary-classifier only; got {len(proba_cols)} classes."
            )
        y_true = np.asarray(self._y)
        y_score = proba_df[proba_cols[1]].to_numpy().astype(np.float64)
        positive_class = _positive_class_value(self._model, self._y, proba_cols)

        df = _curve_frames.calibration_frame(y_true, y_score, positive_class, n_bins, strategy)
        self._cache[key] = df
        return df

    def cumulative_gain(self) -> pl.DataFrame:
        """Cumulative-gain curve per class. Appends a 2-row ``class='baseline'``
        diagonal for plotting reference.
        """
        key = self._cache_key("cumulative_gain")
        if key in self._cache:
            return self._cache[key]

        proba_df = self.probabilities()
        y_true, score_matrix, labels, class_values = _resolve_proba(
            "cumulative_gain",
            self._y,
            proba_df,
            real_classes=getattr(self._model, "classes_", None),
        )
        df = _curve_frames.gain_frame(y_true, score_matrix, class_values, labels)
        self._cache[key] = df
        return df

    def lift_curve(self) -> pl.DataFrame:
        """Lift curve per class. Appends a 2-row ``class='baseline'`` line at
        lift=1.0.
        """
        key = self._cache_key("lift_curve")
        if key in self._cache:
            return self._cache[key]

        proba_df = self.probabilities()
        y_true, score_matrix, labels, class_values = _resolve_proba(
            "lift_curve", self._y, proba_df, real_classes=getattr(self._model, "classes_", None)
        )
        df = _curve_frames.lift_frame(y_true, score_matrix, class_values, labels)
        self._cache[key] = df
        return df

    def discrimination_threshold(
        self,
        *,
        n_thresholds: int = 50,
        cv: Any = None,
    ) -> pl.DataFrame:
        """Discrimination threshold sweep — binary classifiers only.

        Sweeps ``n_thresholds`` evenly-spaced thresholds in [0, 1] and
        reports precision, recall, F1, and queue_rate at each. ``queue_rate``
        is the hand-computed fraction ``(y_score >= t).mean()``.

        Parameters
        ----------
        n_thresholds : int, default 50
            Number of evenly-spaced thresholds swept across [0, 1].
        cv : int or cross-validation splitter, optional
            When an int, runs the same sweep on each fold's held-out scores
            from a freshly-cloned + re-fit estimator and averages
            per-threshold metrics across folds. Pass a splitter object with
            a ``.split()`` method to override the fold generator. ``None``
            (default) sweeps the in-sample scores directly.
        """
        key = self._cache_key(
            "discrimination_threshold",
            n_thresholds=n_thresholds,
            cv=cv,
        )
        if key in self._cache:
            return self._cache[key]
        require_sklearn("discrimination_threshold")

        if self._y is None:
            raise ValueError("ModelSource.discrimination_threshold() requires y to be provided.")
        proba_df = self.probabilities()
        proba_cols = [c for c in proba_df.columns if c.startswith("proba_")]
        if len(proba_cols) != 2:
            raise ValueError(
                "discrimination_threshold() is binary-classifier only; "
                f"got {len(proba_cols)} classes."
            )
        y_true = np.asarray(self._y)
        # Positionally aligned with proba_cols[1] by the same
        # probabilities()/predict_proba guarantee _resolve_proba's
        # real_classes parameter relies on.
        positive_class = _positive_class_value(self._model, self._y, proba_cols)
        thresholds = np.linspace(0.0, 1.0, n_thresholds)

        if cv is None:
            y_score = np.asarray(proba_df[proba_cols[1]])
            df = _curve_frames.threshold_sweep_frame(y_true, y_score, positive_class, thresholds)
        else:
            df = self._discrimination_threshold_cv(
                cv,
                y_true,
                thresholds,
                positive_class,
            )

        self._cache[key] = df
        return df

    def _discrimination_threshold_cv(
        self,
        cv: Any,
        y_true: np.ndarray,
        thresholds: np.ndarray,
        positive_class: object,
    ) -> pl.DataFrame:
        """Cross-validated threshold sweep — average per-fold metrics.

        The positive class is resolved and validated **once, here, against
        the whole target** (2026-09-11 close review, cycle 4). Folds then
        receive rows of that already-validated indicator rather than a raw
        slice to re-resolve.

        That split matters because the two failures look identical to a
        frame builder and are not the same thing. "``y`` and the model's
        classes never met" is a bad request and refuses. "this ``KFold``
        slice happens to hold no positives" is a property of the split of a
        perfectly good target -- routine on the imbalanced data this
        diagnostic exists for (measured: a 10% positive rate with ``cv=5``
        produced a zero-positive fold in 4 of 20 seeds), and doubly so
        because the default generator here is plain ``KFold(shuffle=True)``
        rather than a stratified one. Such a fold contributes the kernel's
        honest ``precision = recall = f1 = 0`` with its real ``queue_rate``,
        which is what the fold average has always incorporated and what this
        call returned before the whole-target guard existed.
        """
        from sklearn.base import clone
        from sklearn.model_selection import KFold

        # Request-level: refuses a target that genuinely never matches the
        # positive class, before any fold exists to be blamed for it.
        y_true_bin = _curve_frames.binary_indicator(
            y_true, positive_class, diagnostic="discrimination_threshold"
        )

        # numpy required: CV split uses integer-array row indexing (X_np[tr], X_np[te])
        # which polars does not support.
        X_np = self._require_X("discrimination_threshold").to_numpy()
        splitter = (
            cv
            if hasattr(cv, "split")
            else KFold(
                n_splits=int(cv),
                shuffle=True,
                random_state=self._random_state or 0,
            )
        )
        fold_dfs: list[pl.DataFrame] = []
        for tr, te in splitter.split(X_np):
            m = clone(self._model).fit(X_np[tr], y_true[tr])
            s = _score_fold(m, X_np[te])
            fold_dfs.append(
                _curve_frames.threshold_sweep_from_indicator(y_true_bin[te], s, thresholds)
            )
        return (
            pl.concat(fold_dfs, how="vertical")
            .group_by("threshold")
            .agg(
                [
                    pl.col("precision").mean(),
                    pl.col("recall").mean(),
                    pl.col("f1").mean(),
                    pl.col("queue_rate").mean(),
                ]
            )
            .sort("threshold")
        )

    def confusion_matrix(self, *, normalize: str | None = None) -> pl.DataFrame:
        """Confusion matrix in long form: one row per (actual, predicted) cell.

        ``value`` is the (possibly normalized) count; ``value_fmt`` is a
        stringified label suitable for ``mark_text`` overlay (integer counts
        when unnormalized, two-decimal fractions when normalized).

        Parameters
        ----------
        normalize : {None, "true", "pred", "all"}, default None
            Normalization mode. ``None`` reports raw counts; ``"true"`` /
            ``"pred"`` / ``"all"`` apply sklearn-style normalization over
            rows / columns / the whole matrix.
        """
        key = self._cache_key("confusion_matrix", normalize=normalize)
        if key in self._cache:
            return self._cache[key]
        require_sklearn("confusion_matrix")

        if self._y is None:
            raise ValueError("ModelSource.confusion_matrix() requires y to be provided.")
        X = self._require_X("confusion_matrix")
        y_true = np.asarray(self._y)
        y_pred = np.asarray(self._model.predict(X))

        # `encode_values` is the comparison-value list confusion_frame uses
        # to build its row/col encoding (must live in y's domain); display
        # aliasing (class_names=) is applied separately below, after
        # encoding, so a display alias never corrupts the encode/decode
        # round-trip. This is unconditional regardless of whether the
        # model exposes `classes_`: when it doesn't, `class_names=` is
        # still display-only, never the encode key.
        real_classes = getattr(self._model, "classes_", None)
        if real_classes is not None:
            encode_values = _coerce_real_classes(real_classes, self._y.dtype)
            y_pred = _predictions_in_y_domain(real_classes, encode_values, y_pred)
        else:
            encode_values = sorted(set(y_true.tolist()) | set(y_pred.tolist()))

        df = _curve_frames.confusion_frame(y_true, y_pred, encode_values, normalize=normalize)

        if self._class_names is not None and len(self._class_names) == len(encode_values):
            alias_map = {str(v): alias for v, alias in zip(encode_values, self._class_names)}
            df = df.with_columns(
                pl.col("actual").replace(alias_map),
                pl.col("predicted").replace(alias_map),
            )

        self._cache[key] = df
        return df


# ---------------------------------------------------------------------------
# Model-path helpers (sklearn-bound: label coercion + per-fold scoring)
# ---------------------------------------------------------------------------


def _coerce_class_label(label_str: str, target_dtype) -> object:
    """Coerce a stringified class label back to y's dtype for equality
    comparison. Falls back to the original string if conversion fails.
    """
    if np.issubdtype(target_dtype, np.integer):
        try:
            return int(label_str)
        except ValueError:
            return label_str
    if np.issubdtype(target_dtype, np.floating):
        try:
            return float(label_str)
        except ValueError:
            return label_str
    return label_str


def _coerce_real_classes(real_classes: Any, y_dtype: pl.DataType) -> list:
    """Coerce a model's real class values into ``y``'s own dtype domain.

    Every classification-curve method on this class binarizes
    ``y_true == comparison_value`` for each class; the comparison value
    must live in ``y``'s domain regardless of which side of the
    resolution (a display label, or ``real_classes`` -- typically
    ``model.classes_``) supplied the raw value. Builds ``real_classes``
    as a polars Series and casts it to ``y_dtype`` via polars' own cast
    machinery -- real type conversion, not a hand-rolled per-dtype
    guess, so it is correct for any dtype ``y`` can legally carry
    rather than only the ones this function's author thought to
    enumerate.

    String-backed dtypes (``Categorical``, ``Enum``) are a special
    case: casting a non-string source directly into an
    already-materialized dictionary-encoded dtype reinterprets the
    source's values as physical dictionary *codes* rather than values
    to encode (a documented polars characteristic, not a bug here) --
    so any dtype with that property always routes through ``Utf8``
    first, the same representation any dictionary-encoded column is
    built from. This check is on the *property* (string-backed,
    dictionary-encoded), not an enumeration of dtype names, precisely
    because a name-based list silently excludes the next dtype that
    shares the property.

    When a value genuinely cannot cast into ``y_dtype`` -- a true
    value-domain mismatch, e.g. ``classes_=["c0"]`` against a numeric
    ``y`` -- the cast raises, and this function falls back to
    ``real_classes`` unconverted: comparing a mismatched value against
    ``y`` already returns ``False`` for every row today (the
    documented, out-of-scope behavior for a class set that was never
    in ``y``'s domain to begin with), so this preserves that instead of
    raising somewhere the caller doesn't expect it. A malformed
    ``real_classes`` (e.g. mixed-type values) can raise ``TypeError``
    while building the intermediate Series, before any cast is
    attempted; that also falls back rather than propagating.

    Shared by ``_resolve_proba`` (feeding ``roc_curve``, ``pr_curve``,
    ``cumulative_gain``, ``lift_curve``), ``discrimination_threshold``,
    and ``confusion_matrix`` -- the three call sites that need this
    coercion.
    """
    # np.asarray(...).tolist() (not list(real_classes)) so numpy scalar
    # elements (np.bool_, np.int64, ...) become native Python objects --
    # pl.Series() constructed straight from numpy scalars triggers an
    # internal polars/numpy deprecation warning ("np.bool scalars ...
    # interpreted as an index") and native values are what every dtype
    # arm below expects to compare/serialize anyway.
    values = np.asarray(real_classes).tolist()
    try:
        source = pl.Series(values=values)
        if y_dtype.base_type() in (pl.Categorical, pl.Enum):
            source = source.cast(pl.Utf8)
        return source.cast(y_dtype, strict=True).to_list()
    except (pl.exceptions.InvalidOperationError, TypeError):
        return values


def _score_fold(model, X_te: np.ndarray) -> np.ndarray:
    """Return positive-class scores for one CV fold's held-out X."""
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X_te), dtype=np.float64)[:, 1]
    if hasattr(model, "decision_function"):
        raw = np.asarray(model.decision_function(X_te), dtype=np.float64)
        return 1.0 / (1.0 + np.exp(-raw))
    raise AttributeError(
        "discrimination_threshold(cv=...) requires the wrapped "
        "model to implement 'predict_proba' or 'decision_function'."
    )
