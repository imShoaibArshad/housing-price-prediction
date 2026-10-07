# Housing Price Prediction — Regularisation & Cross-Validation

Predicting sale prices for **1,458 homes in Ames, Iowa** (the Kaggle *House Prices*
dataset) and answering one question properly:

> **Does regularisation actually beat plain linear regression here?**

Project 1 stopped at a single OLS baseline. This is the natural follow-up: a
log-transformed target, four models compared under **identical CV folds**, and an
explicit look at generalisation gap rather than just headline accuracy.

## Headline results

**5-fold cross-validation** (target = `log1p(SalePrice)`, same folds for every model):

| Model | CV R² | ± std | Train R² | Gap |
| --- | ---: | ---: | ---: | ---: |
| OLS (baseline) | 0.904 | 0.013 | 0.948 | 0.044 |
| Ridge (L2) | 0.916 | 0.009 | 0.943 | 0.027 |
| Lasso (L1) | 0.917 | 0.010 | 0.933 | 0.017 |
| **ElasticNet (L1+L2)** | **0.918** | 0.009 | 0.939 | 0.021 |

**Holdout (20%), best model by CV — reported in real dollars:**

| Metric | Value |
| --- | ---: |
| R² (log target) | 0.918 |
| MAE | **$14,170** |
| RMSE | $19,525 |
| Median \|error\| | $9,299 |
| MAPE | **8.7%** |

### What this shows

- Every regularised model **beats OLS**: 0.904 → 0.916–0.918 CV R².
- **The train/test gap shrinks by ~39%** (0.044 → 0.017 for Lasso). The win here
  is not raw accuracy so much as *less overfitting* — OLS fits noise in the
  87-feature training set, regularisation does not.
- The four models land within 0.002 R² of each other. Reporting that honestly is
  more useful than pretending ElasticNet is a landslide winner; with 1,458 rows
  the differences are close to fold noise (±0.009–0.013).
- **The holdout model is selected from CV, not hardcoded** — `run()` picks the
  CV winner and feeds it to the holdout step, so the two can't disagree.

## What's in here

```
housing-price-prediction/
├── run.py                 # entry point
├── data/ames.csv          # Ames housing (OpenML id 42165), 1,460 rows × 81 cols
├── src/
│   ├── load.py            # fetch-ready loader, NA rules, feature engineering
│   └── model.py           # CV comparison, holdout metrics, plots
└── reports/               # 4 charts + cv_results.csv + holdout_metrics.json
```

### Feature engineering (8 domain features)

| Feature | Why |
| --- | --- |
| `TotalSF` | basement + 1st + 2nd floor, stronger than any single area |
| `QualXArea` | `OverallQual × GrLivArea` — quality means more with size |
| `TotalBath` | half-baths counted as 0.5, matching how buyers value them |
| `HouseAge` / `RemodAge` | `YrSold − YearBuilt` / `YearRemodAdd` |
| `IsRemodeled` | remodelled or not |
| `FirstFlrShare` | how much living area is above ground |

### Preprocessing pipeline

- **`ColumnTransformer`**: `StandardScaler` on 44 numeric, `OneHotEncoder` on 43
  categorical (`handle_unknown="ignore"`, `min_frequency=10` to collapse rare levels).
- **Missing values handled by meaning, not blanket drop**: NA in `BsmtQual` /
  `GarageType` means *"no basement / no garage"*, so those become explicit
  `"None"` categories or `0`. `LotFrontage` missing means *"unknown for this lot"*
  → filled with the **neighbourhood median**.
- **2 documented outliers removed** (`GrLivArea > 4000` & price < $300k) — noted
  in the dataset docs as partial-sale artefacts.
- **`log1p` target**: sale prices are strongly right-skewed; a linear model fits
  the log far better. All dollar figures are converted **back** with `expm1` before
  reporting, so the metrics mean something.

## Data

[Ames House Prices](https://www.openml.org/d/42165) (OpenML id 42165) — De Cock
(2011), *Journal of Statistics Education*. The Kaggle *House Prices* competition
set: 1,460 rows × 81 columns.

Regenerate `data/ames.csv` if missing:

```bash
python -c "from sklearn.datasets import fetch_openml as f; f(data_id=42165, as_frame=True).frame.to_csv('data/ames.csv', index=False)"
```

## Limitations

- **No hyperparameter search** — `alpha` values are fixed (Ridge 10, Lasso/ElasticNet 0.001).
  A `GridSearchCV` pass is the obvious next step and would likely lift all three.
- **Linear models only** — no tree ensembles (XGBoost/LightGBM win this
  competition; that's a separate project, not something bolted on here).
- **5 folds, one split** — the ±0.009 std means the 0.002 gap between Lasso and
  ElasticNet is not meaningful. Treat "best" as "statistically tied with two others".
- CV R² is on the **log** target, so it is not directly comparable to the dollar MAE.
- Trained on all of Ames, Iowa only — no geographic transfer.

## Ideas for next steps

- `GridSearchCV` over `alpha` and `l1_ratio`
- Target encoding instead of one-hot for high-cardinality `Neighborhood`
- Gradient boosting as the non-linear baseline
- Residual analysis on the biggest errors (are they specific neighbourhoods?)

## Run it

```bash
pip install -r requirements.txt
python run.py
```

## Tech

Python · pandas · NumPy · scikit-learn · Matplotlib · Seaborn
