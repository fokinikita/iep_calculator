"""Train the price / hedonic-index CatBoost model.

A single model is trained with a `year_month` time feature added on top of the
regular flat features. It serves two purposes in the API:

* estimation      — predict with `year_month` fixed to the latest month;
* hedonic index   — hold the flat's features fixed and vary only `year_month`.

Because both use the same model, the hedonic curve's last month equals the
point estimate by construction.

Run:  poetry run python train_hedonic_model.py
"""

import json
from pathlib import Path

import polars as pl
from catboost import CatBoostRegressor, Pool

# --- data (top_flip feature store) ---
TRAIN_PATH = r"C:\Users\pc\Dropbox\ПК\Desktop\top_flip\top_flip\data\features\train_features.parquet"
VALID_PATH = r"C:\Users\pc\Dropbox\ПК\Desktop\top_flip\top_flip\data\features\valid_features.parquet"
TEST_PATH = r"C:\Users\pc\Dropbox\ПК\Desktop\top_flip\top_flip\data\features\test_features.parquet"

BASE_DIR = Path(__file__).resolve().parents[1]
IEP_META_PATH = BASE_DIR / "data/features_iep/features_meta.json"
MODEL_OUT_PATH = BASE_DIR / "data/models_iep/model_hedonic.cbm"

TARGET_NAME = "log_price_sq"
TIME_FEATURE = "year_month"
# First month shown on the per-flat hedonic charts (base = 100).
HEDONIC_START = "2025_06"
# Feature not present in the training store — dropped from the served set.
DROP_FEATURES = {"neighbors_median_total_rooms"}

CATBOOST_PARAMS = {
    "iterations": 1450,
    "learning_rate": 0.142,
    "depth": 8,
    "l2_leaf_reg": 0.005,
    "random_seed": 228,
    "verbose": 100,
    "thread_count": 4,
    "loss_function": "MAE",
}


def add_year_month(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.col("max_start_status3").dt.strftime("%Y_%m").alias(TIME_FEATURE)
    )


def main() -> None:
    train = add_year_month(pl.read_parquet(TRAIN_PATH))
    valid = add_year_month(pl.read_parquet(VALID_PATH))
    test = add_year_month(pl.read_parquet(TEST_PATH))

    iep_meta = json.loads(IEP_META_PATH.read_text(encoding="utf-8"))
    continuous_features = [
        f for f in iep_meta["continuous_features"] if f not in DROP_FEATURES
    ]
    categorical_features = [
        f for f in iep_meta["categorical_features"] if f not in DROP_FEATURES
    ]
    # Time feature is categorical (one effect per month = classic hedonic index).
    categorical_features = categorical_features + [TIME_FEATURE]
    all_features = continuous_features + categorical_features

    def prep(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
            pl.col(categorical_features).cast(pl.String).fill_null("missing")
        )

    train, valid, test = prep(train), prep(valid), prep(test)

    train_valid = pl.concat([train, valid])

    train_pool = Pool(
        data=train_valid.select(all_features).to_pandas(),
        label=train_valid.select(TARGET_NAME).to_series().to_numpy(),
        cat_features=categorical_features,
    )
    test_pool = Pool(
        data=test.select(all_features).to_pandas(),
        label=test.select(TARGET_NAME).to_series().to_numpy(),
        cat_features=categorical_features,
    )

    model = CatBoostRegressor(**CATBOOST_PARAMS)
    model.fit(train_pool, eval_set=test_pool)

    # --- test metrics (WAPE on price/m² + mean APE), mirrors the notebook ---
    pred = (
        test.with_columns(pl.Series(TARGET_NAME + "_pred", model.predict(test_pool)))
        .with_columns(
            pl.col(TARGET_NAME).exp().alias("price_sq"),
            pl.col(TARGET_NAME + "_pred").exp().alias("price_sq_pred"),
        )
        .with_columns(
            (pl.col("price_sq") - pl.col("price_sq_pred")).alias("e"),
            ((pl.col("price_sq") - pl.col("price_sq_pred")) / pl.col("price_sq"))
            .abs()
            .alias("ape"),
        )
    )
    wape = pred["e"].abs().sum() / pred["price_sq"].sum()
    mape = pred["ape"].mean()
    print(f"\nTest WAPE={wape:.4f}  MAPE={mape:.4f}  n={pred.height}")

    # --- sanity: does the latest month equal the flat's point estimate? ---
    latest_month = train_valid[TIME_FEATURE].max()
    print(f"Latest month (estimation anchor): {latest_month}")

    model.save_model(str(MODEL_OUT_PATH))
    print(f"Saved model -> {MODEL_OUT_PATH}")

    # Months shown on the hedonic charts: from HEDONIC_START to the latest month.
    hedonic_months = [
        m
        for m in train_valid[TIME_FEATURE].unique().sort().to_list()
        if m >= HEDONIC_START
    ]

    # --- persist the served feature set (pipeline features only, no time) ---
    served_meta = {
        "continuous_features": continuous_features,
        "categorical_features": [f for f in categorical_features if f != TIME_FEATURE],
        "time_feature": TIME_FEATURE,
        "latest_month": latest_month,
        "hedonic_months": hedonic_months,
    }
    out_meta = BASE_DIR / "data/features_iep/features_meta_hedonic.json"
    out_meta.write_text(
        json.dumps(served_meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved meta  -> {out_meta}")

    # --- month range actually available (for the API's hedonic curve) ---
    months = train_valid[TIME_FEATURE].unique().sort().to_list()
    print(f"Available months: {months[0]} .. {months[-1]} ({len(months)} total)")


if __name__ == "__main__":
    main()
