import os
from typing import Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.logger import logger
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '../configs/.env'))

def load_data(file_path: str) -> pd.DataFrame:
    """Load dataset from a CSV file."""
    try:
        df = pd.read_csv(file_path, low_memory=False, encoding_errors="ignore", delimiter=";")
        logger.info("Dataset loaded successfully.")
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise

def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleanup:
    - strip column names
    - parse common columns (Dt_Customer, Year_Birth)
    - filter implausible Year_Birth
    """
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    if "Dt_Customer" in df.columns:
        df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], errors="coerce")

    if "Year_Birth" in df.columns:
        df["Year_Birth"] = pd.to_numeric(df["Year_Birth"], errors="coerce").astype("Int64")
        before = len(df)
        df = df[(df["Year_Birth"].notna()) & (df["Year_Birth"] > 1940) & (df["Year_Birth"] < 2005)]
        logger.info(f"Filtered Year_Birth outliers/NaN: {before - len(df)} rows removed")

    dup_count = df.duplicated().sum()
    if dup_count > 0:
        df = df.drop_duplicates().reset_index(drop=True)
        logger.info(f"Dropped duplicates: {dup_count} | new shape={df.shape}")

    return df


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Proper imputation by dtype:
    - numeric -> mean (or median if skewed)
    - categorical/object -> mode or 'Unknown'
    """
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # numeric: use mean (fallback to median if all-NaN handling needed)
    for col in num_cols:
        if df[col].isna().any():
            mean_val = df[col].mean()
            if pd.isna(mean_val):
                median_val = df[col].median()
                df.loc[:, col] = df[col].fillna(median_val)
                logger.warning(f"Imputed numeric column '{col}' with MEDIAN (mean was NaN).")
            else:
                df.loc[:, col] = df[col].fillna(mean_val)
                logger.warning(f"Imputed numeric column '{col}' with MEAN.")

    # categorical: mode or 'Unknown'
    for col in cat_cols:
        if df[col].isna().any():
            mode_series = df[col].mode(dropna=True)
            fill_val = mode_series.iloc[0] if not mode_series.empty else "Unknown"
            df.loc[:, col] = df[col].fillna(fill_val)
            logger.warning(f"Imputed categorical column '{col}' with '{fill_val}'.")

    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create useful features:
    - Age, TotalPurchases, TotalSpent
    - Optional log transforms for skewed distributions
    """
    df = df.copy()

    # Age
    if "Year_Birth" in df.columns:
        current_year = pd.Timestamp.now().year
        df["Age"] = current_year - df["Year_Birth"].astype("float64")

    # TotalPurchases
    purchase_cols = [c for c in ["NumDealsPurchases", "NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases"] if c in df.columns]
    if purchase_cols:
        df["TotalPurchases"] = df[purchase_cols].sum(axis=1)

    # TotalSpent
    amount_cols = [c for c in ["MntWines", "MntFruits", "MntGoldProds", "MntMeatProducts", "MntSweetProducts", "MntFishProducts"] if c in df.columns]
    if amount_cols:
        df["TotalSpent"] = df[amount_cols].sum(axis=1)

    # Optional: log transforms for heavy skew
    for col in ["Income", "TotalSpent"]:
        if col in df.columns and (df[col] > 0).any():
            df[f"Log_{col}"] = np.log1p(df[col].clip(lower=0))

    return df


def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """Compute missing values report."""
    missing_cnt = df.isna().sum().sort_values(ascending=False)
    missing_pct = (missing_cnt / len(df) * 100).round(2)
    rep = pd.DataFrame({"missing_count": missing_cnt, "missing_pct": missing_pct})
    return rep


def plot_missing(df: pd.DataFrame, save_dir: str | None = None) -> None:
    """Plot missing values; optionally save to PNG."""
    mis = df.isna().sum().sort_values(ascending=False)
    if mis.max() == 0:
        logger.info("No missing values to plot.")
        return
    ax = mis[mis > 0].plot(kind="bar")
    ax.set_title("Missing values per column")
    ax.set_xlabel("Column")
    ax.set_ylabel("Count")
    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        out = os.path.join(save_dir, "missing_values.png")
        plt.savefig(out)
        logger.info(f"Missingness plot saved to {out}")
        plt.close()
    else:
        plt.show()


def save_processed(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(f"Processed dataset saved to: {path}")


def analyze_data(file_path: str, processed_out: str = "data/processed/marketing_campaign_clean.csv", plots_dir: str | None = "docs/eda") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Main EDA pipeline:
    - load, basic cleaning, imputation, feature engineering
    - report & plot missing
    - save processed CSV
    Returns: (df_clean, missing_report_df)
    """
    logger.info(f"Loading data from: {file_path}")  # Log the file path
    df = load_data(file_path)  # consider sep=";" if you want to force it
    df = basic_cleaning(df)
    df = impute_missing(df)
    df = feature_engineering(df)

    rep = missing_report(df)
    logger.info("Top 10 missingness columns:")
    logger.info(f"\n{rep.head(10)}")

    plot_missing(df, save_dir=plots_dir)
    save_processed(df, processed_out)
    return df, rep

if __name__ == "__main__":
    FILE_PATH = os.getenv("DATA_RAW")
    logger.info(f"FILE_PATH is set to: {FILE_PATH}")  # Log the file path
    analyze_data(FILE_PATH)