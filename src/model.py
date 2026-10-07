"""Compare OLS against regularised linear models with cross-validation.

Project 1 stopped at plain OLS. This one asks the natural follow-up:
does regularisation actually help? Ridge (L2), Lasso (L1) and ElasticNet
(L1+L2) are compared against the OLS baseline under the same CV folds,
with a log-transformed target to handle the skewed price distribution.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge  # noqa: E402
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # noqa: E402
from sklearn.model_selection import KFold, cross_validate  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402

from .load import REPORTS_DIR, load, numeric_columns, text_columns  # noqa: E402

sns.set_theme(style="whitegrid", palette="viridis")

CV_FOLDS = 5
RANDOM_STATE = 42


def build_pipeline(model) -> Pipeline:
    """One-hot encode categories, scale numerics, then fit the model."""
    numeric = X_COLUMNS_NUMERIC
    categorical = X_COLUMNS_CATEGORICAL
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=10), categorical),
        ]
    )
    return Pipeline([("pre", pre), ("model", model)])


# Filled in by run() once the data is loaded — keeps build_pipeline simple.
X_COLUMNS_NUMERIC: list = []
X_COLUMNS_CATEGORICAL: list = []


def candidate_models() -> dict:
    """The four models being compared."""
    return {
        # alpha=0 behaves like OLS, kept explicitly as the baseline.
        "OLS (baseline)": LinearRegression(),
        "Ridge (L2)": Ridge(alpha=10.0),
        # Lasso shrinks some coefficients fully to zero -> implicit selection.
        "Lasso (L1)": Lasso(alpha=0.001, max_iter=100000),
        "ElasticNet (L1+L2)": ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=100000),
    }


def evaluate_cv(X, y) -> pd.DataFrame:
    """5-fold CV for every candidate; same folds so results are comparable."""
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for name, est in candidate_models().items():
        pipe = build_pipeline(est)
        res = cross_validate(
            pipe, X, y, cv=cv,
            scoring={"r2": "r2",
                     "mae": "neg_mean_absolute_error",
                     "rmse": "neg_root_mean_squared_error"},
            return_train_score=True,
        )
        rows.append({
            "model": name,
            "cv_r2_mean": float(res["test_r2"].mean()),
            "cv_r2_std": float(res["test_r2"].std()),
            "cv_mae_log_mean": float(-res["test_mae"].mean()),
            "cv_rmse_log_mean": float(-res["test_rmse"].mean()),
            "train_r2_mean": float(res["train_r2"].mean()),
            "overfit_gap": float(res["train_r2"].mean() - res["test_r2"].mean()),
        })
    return pd.DataFrame(rows)


def holdout_report(X, y, model_name: str) -> dict:
    """Fit one chosen model on a holdout split and report real-dollar error.

    `model_name` must be a key of candidate_models() so the holdout always
    reports the same model we actually selected from CV.
    """
    from sklearn.model_selection import train_test_split

    candidates = candidate_models()
    if model_name not in candidates:
        raise KeyError(f"unknown model {model_name!r}; expected one of {list(candidates)}")

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    pipe = build_pipeline(candidates[model_name])
    pipe.fit(X_tr, y_tr)
    pred_log = pipe.predict(X_te)

    # Convert back to dollars so the numbers mean something to a reader.
    actual = np.expm1(y_te.to_numpy())
    predicted = np.expm1(pred_log)

    return {
        "model": f"{model_name}, log1p target",
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "r2": float(r2_score(y_te, pred_log)),
        "mae_usd": float(mean_absolute_error(actual, predicted)),
        "rmse_usd": float(np.sqrt(mean_squared_error(actual, predicted))),
        "mape_pct": float(np.mean(np.abs((actual - predicted) / actual)) * 100),
        "median_abs_error_usd": float(np.median(np.abs(actual - predicted))),
        "actual": actual,
        "predicted": predicted,
    }


def plot_cv(results: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    order = results.sort_values("cv_r2_mean")
    ax.barh(order["model"], order["cv_r2_mean"],
            xerr=order["cv_r2_std"], color="#d9a441", alpha=0.9, capsize=4)
    ax.set_xlabel("5-fold CV R² (log target)")
    ax.set_title("Regularisation comparison — 5-fold cross-validation\n"
                 "error bars = std across folds")
    ax.set_xlim(0, 1)
    REPORTS_DIR.mkdir(exist_ok=True)
    fig.savefig(REPORTS_DIR / "01_cv_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  saved 01_cv_comparison.png")

    fig, ax = plt.subplots(figsize=(9, 5))
    order = results.sort_values("overfit_gap")
    ax.barh(order["model"], order["overfit_gap"], color="#563d7c")
    ax.set_xlabel("Train R² − Test R²  (smaller = less overfitting)")
    ax.set_title("Generalisation gap per model")
    fig.savefig(REPORTS_DIR / "02_overfit_gap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  saved 02_overfit_gap.png")


def plot_holdout(rep: dict) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(rep["actual"], rep["predicted"], alpha=0.45, s=16)
    lo = min(rep["actual"].min(), rep["predicted"].min())
    hi = max(rep["actual"].max(), rep["predicted"].max())
    ax.plot([lo, hi], [lo, hi], "--", color="#d9a441")
    ax.set_xlabel("Actual sale price (USD)")
    ax.set_ylabel("Predicted sale price (USD)")
    ax.set_title(f"{rep['model'].split(',')[0]} on holdout — R² = {rep['r2']:.3f}, "
                 f"MAE = ${rep['mae_usd']:,.0f}")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v/1000:,.0f}k"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v/1000:,.0f}k"))
    fig.savefig(REPORTS_DIR / "03_holdout_predictions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  saved 03_holdout_predictions.png")


def plot_error_distribution(rep: dict) -> None:
    errors = rep["predicted"] - rep["actual"]
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(errors, bins=50, kde=True, ax=ax, color="#563d7c")
    ax.axvline(0, color="#d9a441", linestyle="--")
    ax.set_xlabel("Prediction error (predicted − actual, USD)")
    ax.set_title("Error distribution on the holdout set")
    fig.savefig(REPORTS_DIR / "04_error_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  saved 04_error_distribution.png")


def run() -> dict:
    global X_COLUMNS_NUMERIC, X_COLUMNS_CATEGORICAL

    print("Loading Ames housing data...")
    X, y = load(log_target=True)
    # pandas 3.0 types strings as "str", not "object", so detect by
    # numeric-ness rather than a specific dtype sentinel.
    X_COLUMNS_NUMERIC = numeric_columns(X)
    X_COLUMNS_CATEGORICAL = text_columns(X)
    print(f"  rows={len(X):,}  numeric={len(X_COLUMNS_NUMERIC)}  categorical={len(X_COLUMNS_CATEGORICAL)}")

    REPORTS_DIR.mkdir(exist_ok=True)

    results = evaluate_cv(X, y)
    print("\n5-fold cross-validation (target = log1p price):")
    print(f"  {'model':<24}{'CV R2':>9}{'±std':>8}{'train R2':>11}{'gap':>8}")
    print("  " + "-" * 60)
    for _, r in results.iterrows():
        print(f"  {r['model']:<24}{r['cv_r2_mean']:>9.3f}{r['cv_r2_std']:>8.3f}"
              f"{r['train_r2_mean']:>11.3f}{r['overfit_gap']:>8.3f}")

    results.to_csv(REPORTS_DIR / "cv_results.csv", index=False)
    plot_cv(results)

    # Pick the winner from CV first, then report its holdout error — otherwise
    # the printed "best model" and the fitted model could disagree.
    best = results.loc[results["cv_r2_mean"].idxmax(), "model"]

    rep = holdout_report(X, y, best)
    actual, predicted = rep.pop("actual"), rep.pop("predicted")

    print("\nHoldout (20%) with the CV-best model — reported in dollars:")
    print(f"  model            {rep['model']}")
    print(f"  R2 (log target)  {rep['r2']:.3f}")
    print(f"  MAE              ${rep['mae_usd']:,.0f}")
    print(f"  RMSE             ${rep['rmse_usd']:,.0f}")
    print(f"  median |error|   ${rep['median_abs_error_usd']:,.0f}")
    print(f"  MAPE             {rep['mape_pct']:.1f}%")

    with open(REPORTS_DIR / "holdout_metrics.json", "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=2)

    rep["actual"], rep["predicted"] = actual, predicted
    plot_holdout(rep)
    plot_error_distribution(rep)

    print(f"\nBest by CV R²: {best}")
    return {"cv": results.to_dict("records"), "holdout": rep}


if __name__ == "__main__":
    run()
