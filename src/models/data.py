"""Loading the processed train/test sets for modelling.

The prep pipeline's ``metadata.json`` is the contract: it names which columns are
features, which are targets, and which are passthrough. This module reads that
contract rather than re-deriving it, so a feature added in ``features.py`` flows
through without touching the training scripts -- and, more importantly, a column
excluded there can never silently become a feature here.

Categorical levels are fitted on **train only**. A category that appears solely
in the test set becomes NA, which XGBoost routes down its missing-value branch.
Fitting them on the union would leak test-set vocabulary into training.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports"

# MLflow 3.x retired the ``./mlruns`` file store (it raises unless
# MLFLOW_ALLOW_FILE_STORE is set). SQLite is the supported local backend and is
# what `mlflow ui --backend-store-uri ...` expects, so the project uses a
# database at the root with artifacts alongside it. Both are gitignored.
DEFAULT_TRACKING_DB = PROJECT_ROOT / "mlflow.db"
DEFAULT_MLARTIFACT_DIR = PROJECT_ROOT / "mlartifacts"


def default_tracking_uri() -> str:
    """SQLite URI for the local tracking store, POSIX-slashed for SQLAlchemy."""
    return "sqlite:///" + DEFAULT_TRACKING_DB.as_posix()


class DataError(RuntimeError):
    """Raised when the processed dataset is missing or unusable."""


@dataclass
class ModellingData:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    feature_names: list[str]
    categorical_features: list[str]
    categories: dict[str, list]
    target: str
    metadata: dict
    train_rows: int = 0
    test_rows: int = 0
    dropped_missing_target: dict[str, int] = field(default_factory=dict)
    # The split-year value per row, kept whether or not it is a feature, so the
    # validation carve-out stays chronological even when the column is excluded.
    split_year_train: pd.Series | None = None
    split_year_test: pd.Series | None = None
    excluded_features: list[str] = field(default_factory=list)

    @property
    def split_description(self) -> str:
        sp = self.metadata.get("split", {})
        tr, te = sp.get("train_years", []), sp.get("test_years", [])
        if not tr or not te:
            return "unknown split"
        return (f"split on {sp.get('split_on', '?')} FY | "
                f"train {tr[0]}-{tr[-1]} | test {te[0]}-{te[-1]}")


def _find_pair(data_dir: Path) -> tuple[Path, Path, str]:
    """Locate train/test files, preferring parquet when both formats exist."""
    for fmt in ("parquet", "csv"):
        train, test = data_dir / f"train.{fmt}", data_dir / f"test.{fmt}"
        if train.exists() and test.exists():
            return train, test, fmt
    raise DataError(
        f"no train/test pair found in {data_dir}.\n"
        "Build the processed dataset first:\n"
        "    python -m src.data_prep.build_dataset"
    )


def _read(path: Path, fmt: str) -> pd.DataFrame:
    return pd.read_parquet(path) if fmt == "parquet" else pd.read_csv(path, low_memory=False)


def _coerce_features(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """Numeric stays numeric, booleans become 0/1 floats, the rest become strings.

    Nullable booleans are converted through float so ``pd.NA`` survives as NaN
    rather than collapsing to False -- the difference between "the field said no"
    and "the field was blank" is real and XGBoost can use it.
    """
    out = df[feature_names].copy()
    for col in out.columns:
        s = out[col]
        if pd.api.types.is_bool_dtype(s) or str(s.dtype) == "boolean":
            out[col] = s.astype("Float64").astype("float64")
        elif pd.api.types.is_numeric_dtype(s):
            out[col] = pd.to_numeric(s, errors="coerce").astype("float64")
        else:
            out[col] = s.astype("string")
    return out


def _apply_categories(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, list]]:
    """Convert string columns to pandas categoricals with train-fitted levels."""
    cat_cols = [c for c in train.columns if str(train[c].dtype) == "string"]
    categories: dict[str, list] = {}
    for col in cat_cols:
        levels = sorted(train[col].dropna().unique().tolist())
        categories[col] = levels
        dtype = pd.CategoricalDtype(categories=levels, ordered=False)
        train[col] = train[col].astype(dtype)
        # Unseen test levels fall out as NaN here, by design.
        test[col] = test[col].astype(dtype)
    return train, test, cat_cols, categories


def load_full_dataset(data_dir: Path = DEFAULT_DATA_DIR) -> tuple[pd.DataFrame, dict]:
    """Load train and test back into one frame, with the prep metadata.

    The temporal-validation harness needs to re-split the data itself across
    many folds, so it wants the whole dataset rather than one fixed split.
    """
    train_path, test_path, fmt = _find_pair(data_dir)
    meta_path = data_dir / "metadata.json"
    if not meta_path.exists():
        raise DataError(f"{meta_path} is missing; re-run src.data_prep.build_dataset")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    combined = pd.concat([_read(train_path, fmt), _read(test_path, fmt)],
                         ignore_index=True)
    return combined, metadata


def prepare_fold(train_df: pd.DataFrame, test_df: pd.DataFrame,
                 feature_names: list[str]
                 ) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, list]]:
    """Build model-ready matrices for one fold.

    Category levels are re-fitted on **this fold's** training rows every time.
    Fitting them once over the whole dataset would leak later years' vocabulary
    into earlier folds, which is exactly what a temporal harness exists to avoid.
    """
    X_train = _coerce_features(train_df, feature_names)
    X_test = _coerce_features(test_df, feature_names)
    return _apply_categories(X_train, X_test)


def load_modelling_data(
    target: str,
    data_dir: Path = DEFAULT_DATA_DIR,
    drop_features: tuple[str, ...] = (),
    drop_split_year: bool = True,
) -> ModellingData:
    """Load train/test and return model-ready matrices for ``target``.

    ``drop_split_year`` removes the column the temporal split was made on --
    ``filing_fiscal_year`` or ``decision_fiscal_year`` -- from the feature set.
    This is on by default and it matters: the test set's year value never occurs
    in training, and a tree cannot extrapolate past the range it was fitted on.
    Left in, the year becomes the highest-gain feature during fitting and then
    silently degrades to a constant at prediction time, which is how a model ends
    up scoring *worse* than predicting the training median. Seasonality is still
    available through ``filing_fiscal_quarter`` and ``filing_month``, which
    repeat across years and therefore do generalise.
    """
    train_path, test_path, fmt = _find_pair(data_dir)
    meta_path = data_dir / "metadata.json"
    if not meta_path.exists():
        raise DataError(
            f"{meta_path} is missing; re-run `python -m src.data_prep.build_dataset` "
            "so the feature/target contract is available."
        )
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))

    train_df = _read(train_path, fmt)
    test_df = _read(test_path, fmt)

    year_col = metadata.get("split", {}).get("year_column", "filing_fiscal_year")
    excluded = list(drop_features)
    if drop_split_year and year_col in metadata["feature_columns"]:
        excluded.append(year_col)

    feature_names = [
        c for c in metadata["feature_columns"]
        if c in train_df.columns and c not in excluded
    ]
    if not feature_names:
        raise DataError("metadata.json lists no usable feature columns")
    if target not in train_df.columns:
        raise DataError(
            f"target {target!r} not found in {train_path.name}; "
            f"available targets: {metadata.get('target_columns')}"
        )

    # Guard the contract: no target may ever appear among the features.
    leaked = set(metadata.get("target_columns", [])) & set(feature_names)
    if leaked:
        raise DataError(
            f"target column(s) {sorted(leaked)} appear in the feature list -- "
            "this would leak the answer. Fix FEATURE_COLUMNS in src/data_prep/features.py."
        )

    dropped = {}
    for name, frame in (("train", train_df), ("test", test_df)):
        missing = frame[target].isna()
        dropped[name] = int(missing.sum())
    train_df = train_df[train_df[target].notna()]
    test_df = test_df[test_df[target].notna()]

    X_train = _coerce_features(train_df, feature_names)
    X_test = _coerce_features(test_df, feature_names)
    X_train, X_test, cat_cols, categories = _apply_categories(X_train, X_test)

    split_year_train = (
        train_df[year_col].reset_index(drop=True) if year_col in train_df.columns else None
    )
    split_year_test = (
        test_df[year_col].reset_index(drop=True) if year_col in test_df.columns else None
    )

    return ModellingData(
        X_train=X_train, y_train=train_df[target].reset_index(drop=True),
        X_test=X_test, y_test=test_df[target].reset_index(drop=True),
        feature_names=feature_names, categorical_features=cat_cols,
        categories=categories, target=target, metadata=metadata,
        train_rows=len(train_df), test_rows=len(test_df),
        dropped_missing_target=dropped,
        split_year_train=split_year_train, split_year_test=split_year_test,
        excluded_features=excluded,
    )


def temporal_validation_split(
    data: ModellingData, val_years: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Carve a validation set off the *end* of the training period.

    Early stopping needs held-out data, and using the test set for it would leak.
    The newest training cohort(s) stand in instead, which keeps the validation
    chronological and consistent with how the train/test split itself was made.
    Returns boolean masks over the training rows: (fit_mask, val_mask).

    Reads the split year from ``data.split_year_train`` rather than from the
    feature matrix, so the carve-out stays chronological even when the year
    column has been excluded from the features (which it is, by default).
    """
    years = data.split_year_train
    if years is None:
        # No year information at all: fall back to a positional tail slice and
        # accept that it is not chronological.
        n = len(data.X_train)
        cut = int(n * 0.85)
        fit = np.zeros(n, dtype=bool)
        fit[:cut] = True
        return fit, ~fit

    unique = sorted(pd.unique(years.dropna()))
    if len(unique) <= val_years:
        n = len(data.X_train)
        cut = int(n * 0.85)
        fit = np.zeros(n, dtype=bool)
        fit[:cut] = True
        return fit, ~fit

    val_set = set(unique[-val_years:])
    val_mask = years.isin(val_set).to_numpy()
    return ~val_mask, val_mask


def random_validation_split(
    data: ModellingData, val_fraction: float = 0.15, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Random hold-out drawn from within the training period only.

    No test-period row is involved, so this leaks nothing into the reported
    score. See ``make_validation_split`` for why it is the default.
    """
    n = len(data.X_train)
    rng = np.random.default_rng(seed)
    val_mask = np.zeros(n, dtype=bool)
    val_mask[rng.choice(n, size=max(1, int(n * val_fraction)), replace=False)] = True
    return ~val_mask, val_mask


def make_validation_split(
    data: ModellingData, strategy: str = "random", val_years: int = 1,
    val_fraction: float = 0.15, seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Choose the early-stopping hold-out.

    Early stopping answers one narrow question: how many boosting rounds before
    the model starts memorising noise? A *temporal* hold-out cannot answer it
    here, because processing time is dominated by the fiscal-year backlog regime
    and the last training cohort sits in a different regime from the rest. Every
    additional tree looks worse on it, so early stopping selects round 0 and
    ships a one-tree model -- a regime shift misread as overfitting.

    ``random`` therefore draws the hold-out from inside the training period,
    where the regime question does not arise. This leaks nothing: the temporal
    test set is untouched and remains the honest measure of whether the model
    generalises across years. ``temporal`` is kept for comparison and to make
    the degenerate behaviour reproducible.
    """
    if strategy == "temporal":
        return temporal_validation_split(data, val_years=val_years)
    if strategy == "random":
        return random_validation_split(data, val_fraction=val_fraction, seed=seed)
    raise DataError(f"unknown validation strategy {strategy!r}; use 'random' or 'temporal'")
