"""
Exercise 3 - Preparing Real-World Data for a Neural Network (Spaceship Titanic)

Standalone script version of the preprocessing pipeline: load the Kaggle
"Spaceship Titanic" training table, describe it, split it (leakage-free),
impute / engineer / encode / transform / scale it for a tanh-activated
network, and save Figure 6 (FoodCourt before vs. after log1p).

Run from this file's own directory:
    python ex3_spaceship_titanic.py
Figure is written to ../figures/fig6.png (created if missing).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

SCRIPT_DIR = Path(__file__).resolve().parent
FIG_PATH = SCRIPT_DIR.parent / "figures" / "fig6.png"

# Global assignment rule: one fixed, shared generator for the whole pipeline.
# sklearn's train_test_split only accepts an int/RandomState for random_state
# (a Generator raises ValueError - verified), so the same seed used to build
# rng is passed straight through, keeping a single source of truth for
# reproducibility instead of a second, disconnected magic number.
SEED = 42
rng = np.random.default_rng(SEED)

NUMERIC_COLS = ["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
CAT_COLS = ["HomePlanet", "CryoSleep", "Destination", "VIP"]
SPEND_COLS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
DROP_COLS = ["PassengerId", "Cabin", "Name"]

BEFORE_COLOR = "#2a78d6"
AFTER_COLOR = "#eb6834"


def load_data() -> pd.DataFrame:
    """Primary path: kagglehub competition download. Fallback: local CSV."""
    try:
        import kagglehub

        comp_dir = Path(kagglehub.competition_download("spaceship-titanic"))
        train_path = comp_dir / "train.csv"
        df = pd.read_csv(train_path)
        print(f"[data] loaded via kagglehub: {train_path}")
        return df
    except Exception as exc:  # no credentials / no network / package missing
        fallback = SCRIPT_DIR / "data" / "train.csv"
        print(f"[data] kagglehub unavailable ({exc}); falling back to {fallback}")
        return pd.read_csv(fallback)


def part_a(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("PART A - Get to know the data")
    print("=" * 70)
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")

    balance = df["Transported"].value_counts()
    balance_pct = df["Transported"].value_counts(normalize=True) * 100
    print("\nClass balance (Transported):")
    for label in balance.index:
        print(f"  {label}: {balance[label]} ({balance_pct[label]:.2f}%)")

    print(f"\nNumerical features: {NUMERIC_COLS}")
    other_cat = [c for c in df.columns if c not in NUMERIC_COLS + ["PassengerId", "Name", "Transported"]]
    print(f"Categorical / free-text features: {other_cat}")

    missing = df.isna().sum().to_frame("missing_count")
    missing["missing_pct"] = (missing["missing_count"] / len(df) * 100).round(2)
    missing = missing[missing["missing_count"] > 0].sort_values("missing_count", ascending=False)
    print("\nMissing values per column:")
    print(missing.to_string())

    print("\nSpending columns - mean / median / max (full dataset):")
    spend_stats = df[SPEND_COLS].agg(["mean", "median", "max"]).T
    print(spend_stats.to_string())
    for col in SPEND_COLS:
        m, med = spend_stats.loc[col, "mean"], spend_stats.loc[col, "median"]
        print(f"  {col}: mean={m:.2f} >> median={med:.2f} -> heavy right skew (most passengers spend 0)")


def part_b(df: pd.DataFrame):
    print("\n" + "=" * 70)
    print("PART B - Split before you transform")
    print("=" * 70)
    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df["Transported"], random_state=SEED
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    print(f"train_df: {train_df.shape}, test_df: {test_df.shape}")

    pos_train = train_df["Transported"].mean()
    pos_test = test_df["Transported"].mean()
    print(f"Positive class share (Transported=True) - train: {pos_train:.4f}, test: {pos_test:.4f}")
    return train_df, test_df


def part_c(train_df: pd.DataFrame, test_df: pd.DataFrame):
    print("\n" + "=" * 70)
    print("PART C - Preprocess")
    print("=" * 70)

    train_df = train_df.drop(columns=DROP_COLS)
    test_df = test_df.drop(columns=DROP_COLS)
    y_train = train_df.pop("Transported")
    y_test = test_df.pop("Transported")

    # --- C1: impute (numeric -> train median, categorical -> train mode) ---
    num_imputer = SimpleImputer(strategy="median")
    train_df[NUMERIC_COLS] = num_imputer.fit_transform(train_df[NUMERIC_COLS])
    test_df[NUMERIC_COLS] = num_imputer.transform(test_df[NUMERIC_COLS])

    cat_imputer = SimpleImputer(strategy="most_frequent")
    train_df[CAT_COLS] = cat_imputer.fit_transform(train_df[CAT_COLS])
    test_df[CAT_COLS] = cat_imputer.transform(test_df[CAT_COLS])
    print(
        f"NaNs after imputation - train: {train_df[NUMERIC_COLS + CAT_COLS].isna().sum().sum()}, "
        f"test: {test_df[NUMERIC_COLS + CAT_COLS].isna().sum().sum()}"
    )

    # --- C3: feature engineering (after imputation, so no NaN in the sum) ---
    train_df["TotalSpend"] = train_df[SPEND_COLS].sum(axis=1)
    test_df["TotalSpend"] = test_df[SPEND_COLS].sum(axis=1)

    foodcourt_before = train_df["FoodCourt"].copy()
    fc_mean, fc_median = foodcourt_before.mean(), foodcourt_before.median()
    print(f"FoodCourt (train, post-impute, pre-log1p): mean={fc_mean:.2f}, median={fc_median:.2f}")

    # --- C2: one-hot encode categoricals. handle_unknown="ignore" means any
    # category value seen only in test (never in train) is encoded as an
    # all-zero row instead of raising, so the transform never fails on an
    # unseen category. ---
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    train_cat = pd.DataFrame(
        encoder.fit_transform(train_df[CAT_COLS]),
        columns=encoder.get_feature_names_out(CAT_COLS),
        index=train_df.index,
    )
    test_cat = pd.DataFrame(
        encoder.transform(test_df[CAT_COLS]),
        columns=encoder.get_feature_names_out(CAT_COLS),
        index=test_df.index,
    )
    print(f"One-hot columns ({len(train_cat.columns)}): {list(train_cat.columns)}")

    # --- C4: log1p the heavy-tailed spending columns (+ engineered total).
    # Stat-free, so identical on both splits with no leakage risk. ---
    log_cols = SPEND_COLS + ["TotalSpend"]
    train_df[log_cols] = np.log1p(train_df[log_cols])
    test_df[log_cols] = np.log1p(test_df[log_cols])
    foodcourt_after = train_df["FoodCourt"].copy()

    # --- C5: scale numeric columns to [-1, 1], fit on train only ---
    numeric_all = NUMERIC_COLS + ["TotalSpend"]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    train_df[numeric_all] = scaler.fit_transform(train_df[numeric_all])
    test_df[numeric_all] = scaler.transform(test_df[numeric_all])

    train_min, train_max = train_df[numeric_all].to_numpy().min(), train_df[numeric_all].to_numpy().max()
    test_min, test_max = test_df[numeric_all].to_numpy().min(), test_df[numeric_all].to_numpy().max()
    print(f"Post-scaling range - train: [{train_min:.2f}, {train_max:.2f}], test: [{test_min:.2f}, {test_max:.2f}]")

    X_train = pd.concat([train_df[numeric_all].reset_index(drop=True), train_cat.reset_index(drop=True)], axis=1)
    X_test = pd.concat([test_df[numeric_all].reset_index(drop=True), test_cat.reset_index(drop=True)], axis=1)

    return X_train, y_train, X_test, y_test, foodcourt_before, foodcourt_after


def part_d(X_train, y_train, X_test, y_test, foodcourt_before, foodcourt_after):
    print("\n" + "=" * 70)
    print("PART D - Verify and visualize")
    print("=" * 70)

    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)

    ax[0].hist(foodcourt_before, bins=40, histtype="step", linewidth=1.6, color=BEFORE_COLOR, label="Before log1p")
    ax[0].set_title("FoodCourt (train) - before log1p")
    ax[0].set_xlabel("FoodCourt spend (credits)")
    ax[0].set_ylabel("Count")
    ax[0].legend()
    ax[0].grid(axis="y", alpha=0.25, linewidth=0.6)

    ax[1].hist(foodcourt_after, bins=40, histtype="step", linewidth=1.6, color=AFTER_COLOR, label="After log1p")
    ax[1].set_title("FoodCourt (train) - after log1p")
    ax[1].set_xlabel("log1p(FoodCourt spend)")
    ax[1].legend()
    ax[1].grid(axis="y", alpha=0.25, linewidth=0.6)

    for a in ax:
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)

    fig.suptitle("Figure 6 - Effect of log1p on the heavy-tailed FoodCourt column")
    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=150)
    plt.close(fig)
    print(f"Figure 6 saved to {FIG_PATH}")

    nan_train = X_train.isna().sum().sum()
    nan_test = X_test.isna().sum().sum()
    print(f"Final NaN check - train: {nan_train}, test: {nan_test}")
    print(f"Final train feature matrix shape: {X_train.shape}")
    print(f"Final test feature matrix shape: {X_test.shape}")

    lo, hi = X_train.to_numpy().min(), X_train.to_numpy().max()
    print(f"Final train value range: [{lo:.2f}, {hi:.2f}] -> within/near tanh's [-1, 1] output range")


def main():
    df = load_data()
    part_a(df)
    train_df, test_df = part_b(df)
    X_train, y_train, X_test, y_test, fc_before, fc_after = part_c(train_df, test_df)
    part_d(X_train, y_train, X_test, y_test, fc_before, fc_after)
    print("\nDone.")


if __name__ == "__main__":
    main()
