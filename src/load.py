"""Load and clean the Ames housing dataset (Kaggle House Prices, OpenML id 42165)."""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "ames.csv"
REPORTS_DIR = ROOT / "reports"

TARGET = "SalePrice"
ID_COL = "Id"

# Drop entirely: near-constant or leaking/duplicate of another column.
DROP_COLS = ["Id", "Utilities", "Street", "Condition2", "RoofMatl"]


def is_text(series: pd.Series) -> bool:
    """True for a non-numeric (categorical/string) column.

    Deliberately not `dtype == object`: pandas 3.0 reads strings as the
    dedicated "str" dtype, which would silently break that comparison and
    route category columns into the numeric scaler.
    """
    return not pd.api.types.is_numeric_dtype(series)


def text_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if is_text(df[c])]


def numeric_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def load_raw(path: Path = DATA_FILE) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Regenerate with:\n"
            "  python -c \"from sklearn.datasets import fetch_openml as f; "
            "f(data_id=42165, as_frame=True).frame.to_csv('data/ames.csv', index=False)\""
        )
    return pd.read_csv(path)


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Domain features that repeatedly show up as strong in this dataset."""
    out = df.copy()

    # Age at sale, and whether it has been remodelled.
    out["HouseAge"] = out["YrSold"] - out["YearBuilt"]
    out["RemodAge"] = out["YrSold"] - out["YearRemodAdd"]
    out["IsRemodeled"] = (out["YearRemodAdd"] != out["YearBuilt"]).astype(int)

    # Total usable area vs. total lot — often stronger than either alone.
    out["TotalSF"] = out["TotalBsmtSF"] + out["1stFlrSF"] + out["2ndFlrSF"]
    out["TotalBath"] = (out["FullBath"] + 0.5 * out["HalfBath"]
                        + out["BsmtFullBath"] + 0.5 * out["BsmtHalfBath"])

    # How much of the living area is above ground.
    out["FirstFlrShare"] = out["1stFlrSF"] / out["GrLivArea"].replace(0, np.nan)

    # Quality x area: a 10/10 on 100 sq ft is not a 10/10 on 3000 sq ft.
    out["QualXArea"] = out["OverallQual"] * out["GrLivArea"]

    return out


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values with rules, not blanket dropping.

    In Ames, NA often means "absent feature" (no garage, no basement) rather
    than "unknown", so those become explicit categories / zeros instead of
    being dropped.
    """
    out = df.copy()

    # "No basement/garage/pool" style columns: fill NA with the neutral value.
    none_cols = [
        c for c in out.columns
        if is_text(out[c]) and out[c].isna().any()
        and any(k in c.lower() for k in ("bsmt", "garage", "fireplace", "pool", "fence", "alley", "misc"))
    ]
    for c in none_cols:
        out[c] = out[c].fillna("None")

    # Remaining string columns -> mode (rare, usually one or two rows).
    # Note: pandas 3.0 types these as "str", not "object".
    for c in out.select_dtypes(include="string").columns:
        if out[c].isna().any():
            out[c] = out[c].fillna(out[c].mode().iloc[0])

    # LotFrontage is missing for real lots -> group median by neighbourhood.
    if out["LotFrontage"].isna().any():
        out["LotFrontage"] = out.groupby("Neighborhood")["LotFrontage"].transform(
            lambda s: s.fillna(s.median())
        )

    # Numeric NA -> 0 where it means "none", median otherwise.
    numeric_na = out.select_dtypes(include=[np.number]).columns
    zeroes = [c for c in numeric_na if out[c].isna().any()
              and any(k in c.lower() for k in ("bsmt", "garage", "masvnr", "pool"))]
    for c in zeroes:
        out[c] = out[c].fillna(0)
    for c in numeric_na:
        if out[c].isna().any():
            out[c] = out[c].fillna(out[c].median())

    return out


def drop_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Remove the documented GrLivArea outliers (large area, low price).

    These four points are famously noted in the dataset documentation as
    partial-sale artefacts, not real observations.
    """
    mask = ~((df["GrLivArea"] > 4000) & (df[TARGET] < 300000))
    removed = int((~mask).sum())
    if removed:
        print(f"  dropped {removed} documented outliers (GrLivArea > 4000 & price < 300k)")
    return df[mask].reset_index(drop=True)


def load(path: Path = DATA_FILE, log_target: bool = True) -> tuple:
    """Return (features, target) with cleaning applied.

    With log_target=True the target is log1p-transformed: house prices are
    strongly right-skewed, and a linear model fits the log much better.
    """
    df = clean(engineer(load_raw(path)))
    df = drop_outliers(df)

    y = np.log1p(df[TARGET]) if log_target else df[TARGET].copy()
    X = df.drop(columns=[TARGET])
    return X, y
