"""Importable preprocessing pipeline for the ASIA-FLOOD one-day-ahead task.

Task: **will station s flood on day t+1?** (binary classification, 2.166 % positives,
45.2 : 1 imbalance). Every statistic fitted here is fitted on the **training window only**
(2000-2016); the validation (2017-2020) and test (2021-2024) frames are only ever
transformed.  No model is trained in this module or in this deliverable.

The seam
--------
    from data import train_test_frames
    from pipeline import fit_preprocessor, transform, feature_names

    X_train, X_test, y_train, y_test = train_test_frames()
    preprocess = fit_preprocessor(X_train)      # fit on TRAIN ONLY
    X_train_t = transform(preprocess, X_train)  # (310500, 21) dense float64
    X_test_t = transform(preprocess, X_test)    # ( 73050, 21) dense float64
    names = feature_names(preprocess)           # 21 names, num__ / cat__ prefixed

Imported by ``t5_pipeline.py`` (stage 4A/4C evidence), by ``t6_reduction.py`` (stage 4B)
and by the later Classification deliverable, so it stays free of plotting and of I/O
beyond the ``__main__`` self-report.

Strategies, each tied to the finding that motivates it
------------------------------------------------------
1. **Missing values** - ``SimpleImputer(strategy="median")`` on the numeric branch and
   ``SimpleImputer(strategy="most_frequent")`` on the categorical branch.  This data has
   **nothing to impute**: 0 missing cells in 456,600 x 21 (0.00 %).  The imputers are a
   robustness measure for a partial future day, not a fix for observed missingness.  The
   categorical one is not cosmetic either: measured on this data, an all-missing float
   ``basin`` column handed straight to the encoder raises a ``TypeError`` (the encoder ends up
   comparing a missing float against the string categories), while a missing value in the
   ``str``/object dtype is silently encoded as all-zeros like any unknown level.  Imputing
   first removes both the crash and the ambiguity (``t5_pipeline.py`` prints all three cases).
2. **Outliers** - never drop, never winsorise.  100.000 % of the 9,274 rows above the
   ``rainfall_mm`` 1.5 x IQR fence (56.30 mm) are floods, against 0.137 % below it: an
   IQR filter would delete the entire positive class.  Instead ``np.log1p`` compresses the
   right tail of ``rainfall_mm`` and ``river_level_m`` (skew 10.51 / 9.53) so it cannot
   dominate a neural network's gradients while keeping every extreme day present and
   *ordered*, and ``is_extreme_rainfall`` gives the magnitude back as an explicit flag.
3. **Categorical encoding** - ``OneHotEncoder(handle_unknown="ignore", sparse_output=False)``
   on ``basin`` (10 levels) and ``season`` (4).  A level absent from the training window -
   the concrete "new category in the test set" case - maps to all-zeros instead of raising
   at transform time.  ``sparse_output=False`` is deliberate: the output is one dense array
   a neural network can consume directly.
4. **Scaling** - ``log1p`` first, then ``StandardScaler`` on all numerics.  The four raw
   measurements span 0.39-1775.72 across three units, and inside the monsoon rainfall's
   spread is 16.6x its outside value (std 60.014 vs 3.611), so the standardiser must see
   the compressed tail, not the raw one.  ``soil_moisture_percent`` is censored at exactly
   100.00; log1p and z-scoring are monotone affine maps, so the tie at the ceiling is kept.

Constants that are *declared*, not fitted
-----------------------------------------
``EXTREME_RAINFALL_MM`` is the stage-2A 1.5 x IQR fence of ``rainfall_mm``, pinned as a
module constant and never re-estimated inside ``fit``.  It is therefore identical for
train, validation and test - the leakage-free direction.  ``t5_pipeline.py`` reports the
train-only fence and the number of rows that would change flag if the two disagreed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data import CATEGORICAL, NUMERIC, SEED, configure_env

configure_env()

__all__ = [
    "NUMERIC_RAW",
    "CATEGORICAL_RAW",
    "RAW_FEATURES",
    "LOG1P_COLS",
    "EXTREME_FLAG_COL",
    "EXTREME_RAINFALL_MM",
    "NUMERIC_BRANCH_IN",
    "RANDOM_STATE",
    "ExtremeRainfallFlag",
    "SkewCompressor",
    "build_preprocessor",
    "fit_preprocessor",
    "transform",
    "feature_names",
]

# --------------------------------------------------------------------------------------
# Raw feature contract - single source of truth is data.py
# --------------------------------------------------------------------------------------
NUMERIC_RAW: list[str] = list(NUMERIC)
CATEGORICAL_RAW: list[str] = list(CATEGORICAL)
RAW_FEATURES: list[str] = NUMERIC_RAW + CATEGORICAL_RAW

#: measurements carrying a heavy right tail (skew 10.51 / 9.53) -> log1p first
LOG1P_COLS: tuple[str, ...] = ("rainfall_mm", "river_level_m")

#: explicit extreme-day indicator, appended by :class:`ExtremeRainfallFlag`
EXTREME_FLAG_COL: str = "is_extreme_rainfall"

#: 1.5 x IQR fence of rainfall_mm from stage 2A (mm/day). Declared, never fitted here.
EXTREME_RAINFALL_MM: float = 56.30

#: column order produced by :class:`ExtremeRainfallFlag` and consumed by :class:`SkewCompressor`
NUMERIC_BRANCH_IN: tuple[str, ...] = tuple(NUMERIC_RAW) + (EXTREME_FLAG_COL,)

#: declared for the deliverable's seed contract. No step below is stochastic - the imputer,
#: the encoder and the scaler are deterministic - so ``random_state`` never enters here;
#: SEED=42 governs the chronological split and the stage-4B projections.
RANDOM_STATE: int = SEED


# --------------------------------------------------------------------------------------
# Derived feature - a stateless transformer, so it is re-applied by ``transform``
# --------------------------------------------------------------------------------------
class ExtremeRainfallFlag(TransformerMixin, BaseEstimator):
    """Append ``is_extreme_rainfall`` = 1.0 where the **raw** rainfall exceeds the fence.

    Stateless and deterministic: the threshold is the module constant
    :data:`EXTREME_RAINFALL_MM`, not a statistic estimated from the data, so the flag is
    identical on train, validation and test and cannot leak.  Living inside the pipeline
    (rather than being computed on the raw frame) is what keeps :func:`transform` correct
    on new data - a flag added outside would silently be missing at inference time.

    Order of columns out = order in + ``[is_extreme_rainfall]``.
    A missing rainfall value is not flagged (``NaN > 56.30`` is ``False``); the median
    imputer downstream then fills it with the train median of ``log1p(rainfall_mm)``,
    which sits far below the fence, so the two steps agree.
    """

    def __init__(
        self,
        column: str = "rainfall_mm",
        threshold: float = EXTREME_RAINFALL_MM,
        flag_name: str = EXTREME_FLAG_COL,
    ) -> None:
        self.column = column
        self.threshold = threshold
        self.flag_name = flag_name

    def fit(self, X, y=None) -> "ExtremeRainfallFlag":
        self.n_features_in_ = self._as_array(X).shape[1]
        self._column_index(X)
        return self

    def transform(self, X) -> np.ndarray:
        arr = self._as_array(X)
        raw = arr[:, self._column_index(X)]
        flag = (raw > self.threshold).astype(float).reshape(-1, 1)
        return np.hstack([arr, flag])

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        names = list(input_features) if input_features is not None else list(NUMERIC_RAW)
        return np.asarray(names + [self.flag_name], dtype=object)

    # -- helpers ------------------------------------------------------------------
    @staticmethod
    def _as_array(X) -> np.ndarray:
        arr = X.to_numpy(dtype=float) if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=float)
        if arr.ndim != 2:
            raise ValueError(f"expected a 2-D feature matrix, got shape {arr.shape}")
        return arr

    def _column_index(self, X) -> int:
        if isinstance(X, pd.DataFrame):
            if self.column not in X.columns:
                raise ValueError(f"column {self.column!r} not in the input frame; got {list(X.columns)}")
            return list(X.columns).index(self.column)
        if self.column not in NUMERIC_RAW:
            raise ValueError(f"column {self.column!r} is not part of NUMERIC_RAW={NUMERIC_RAW}")
        return NUMERIC_RAW.index(self.column)


# --------------------------------------------------------------------------------------
# Tail compression
# --------------------------------------------------------------------------------------
class SkewCompressor(TransformerMixin, BaseEstimator):
    """``np.log1p`` the heavy-tailed columns, pass every other column through untouched.

    Stateless.  Input columns are addressed positionally through ``input_columns``,
    the exact order produced by :class:`ExtremeRainfallFlag`
    (``NUMERIC_RAW + [is_extreme_rainfall]``), so the appended flag is *not* logged -
    ``log1p(1.0) = 0.693`` would turn a binary indicator into a non-binary column.

    ``log1p`` is strictly monotone, so the rank order of the extreme days is preserved
    exactly (Spearman rho = 1.0 against the raw column) and a missing value stays missing
    for the imputer.  It is not applied to ``temperature_celsius``, which is centred on
    zero and slightly negative in the minimum (-2.31 degC), where ``log1p`` is undefined.
    """

    def __init__(
        self,
        columns: tuple[str, ...] = LOG1P_COLS,
        input_columns: tuple[str, ...] = NUMERIC_BRANCH_IN,
    ) -> None:
        self.columns = columns
        self.input_columns = input_columns

    def fit(self, X, y=None) -> "SkewCompressor":
        arr, positions = self._checked(X)
        self.n_features_in_ = arr.shape[1]
        self.positions_ = positions
        return self

    def transform(self, X) -> np.ndarray:
        arr, positions = self._checked(X)
        out = arr.copy()
        out[:, positions] = np.log1p(out[:, positions])
        return out

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        names = list(input_features) if input_features is not None else list(self.input_columns)
        return np.asarray(names, dtype=object)

    def _checked(self, X) -> tuple[np.ndarray, list[int]]:
        """Validate the width, then map column names to positions in ``input_columns``."""
        arr = np.asarray(X, dtype=float)
        ref = list(self.input_columns)
        if arr.ndim != 2:
            raise ValueError(f"expected a 2-D feature matrix, got shape {arr.shape}")
        if arr.shape[1] != len(ref):
            raise ValueError(f"expected {len(ref)} columns {ref}, got {arr.shape[1]}")
        missing = [c for c in self.columns if c not in ref]
        if missing:
            raise ValueError(f"{missing} not in input_columns={ref}")
        return arr, [ref.index(c) for c in self.columns]


# --------------------------------------------------------------------------------------
# The ColumnTransformer
# --------------------------------------------------------------------------------------
def build_preprocessor() -> ColumnTransformer:
    """Return the **unfitted** preprocessing ``ColumnTransformer``.

    6 raw numerics -> 7 (``is_extreme_rainfall`` appended) and 2 raw categoricals -> 14
    one-hot columns: **21 features out of 8 in**.  Nothing is fitted here.
    """
    numeric_branch = Pipeline(
        [
            ("extreme_flag", ExtremeRainfallFlag()),
            ("log1p", SkewCompressor()),
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_branch = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric_branch, NUMERIC_RAW),
            ("cat", categorical_branch, CATEGORICAL_RAW),
        ],
        remainder="drop",
        sparse_threshold=0.0,  # force one dense array, never a scipy sparse matrix
    )


def fit_preprocessor(X_train) -> ColumnTransformer:
    """Fit the pipeline on the **training frame only** and return it fitted.

    ``X_train`` must hold :data:`RAW_FEATURES`; it is never concatenated with validation
    or test rows here.  The only fitted statistics in the whole pipeline are the imputer
    medians/modes and the ``StandardScaler`` means and variances.
    """
    missing = [c for c in RAW_FEATURES if c not in getattr(X_train, "columns", [])]
    if missing:
        raise ValueError(f"X_train is missing raw feature columns: {missing}")
    preprocess = build_preprocessor()
    preprocess.fit(X_train)
    return preprocess


def transform(preprocess: ColumnTransformer, X) -> np.ndarray:
    """Transform ``X`` (train, validation, test or new data) with a **fitted** pipeline.

    Returns a dense ``float64`` array; raises if the result contains a non-finite value,
    so "no NaN" is a guarantee of the seam rather than a hope.
    """
    out = np.asarray(preprocess.transform(X), dtype=float)
    if not np.isfinite(out).all():
        bad = int((~np.isfinite(out)).sum())
        raise ValueError(f"pipeline produced {bad} non-finite value(s)")
    return out


def feature_names(preprocess: ColumnTransformer) -> list[str]:
    """The transformed feature names, in column order, as ``num__...`` / ``cat__...``."""
    return [str(n) for n in preprocess.get_feature_names_out()]


if __name__ == "__main__":
    from data import TARGET_NEXT, add_next_day_target, load_raw, split_frames, train_test_frames

    raw = add_next_day_target(load_raw())
    X_train, X_test, y_train, y_test = train_test_frames(raw)

    print(f"raw panel                  : {raw.shape[0]} rows x {raw.shape[1] - 1} cols (+1 next-day target)")
    print(f"raw feature matrix         : {len(RAW_FEATURES)} cols = {len(NUMERIC_RAW)} numeric + {len(CATEGORICAL_RAW)} categorical")
    sizes = split_frames(raw).groupby("split")[TARGET_NEXT].agg(["size", "sum"])
    for name in ("train", "validation", "test"):
        print(f"usable rows {name:<11}: {int(sizes.loc[name, 'size']):>7} rows, {int(sizes.loc[name, 'sum']):>5} positives")
    print("  (the 50 rows dropped by the t+1 shift are the 2024-12-31 last days, inside the test window)")
    print(f"X_train (2000-2016)        : {X_train.shape}")
    print(f"X_test  (2021-2024)        : {X_test.shape}")

    preprocess = fit_preprocessor(X_train)          # fit on train only
    X_train_t = transform(preprocess, X_train)
    X_test_t = transform(preprocess, X_test)

    print(f"X_train_t                  : {X_train_t.shape}  {X_train_t.dtype}")
    print(f"X_test_t                   : {X_test_t.shape}  {X_test_t.dtype}")
    print(f"np.isnan(X_train_t).sum()  : {int(np.isnan(X_train_t).sum())}")
    print(f"np.isnan(X_test_t).sum()   : {int(np.isnan(X_test_t).sum())}")
    names = feature_names(preprocess)
    print(f"n features                 : {len(names)}  (was {len(RAW_FEATURES)})")
    for i, name in enumerate(names, start=1):
        print(f"  {i:>2}. {name}")