import os
from typing import Tuple, List, Optional, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy import stats
import plotly.express as px
from datetime import datetime
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '../configs/.env'))

# Adjust import to your path if needed (e.g., from src.utils.logger import logger)
from utils.logger import logger


# -----------------------------
# Loading & basic cleaning
# -----------------------------
def load_data(file_path: str, sep: str | None = None) -> pd.DataFrame:
    """
    Load dataset from CSV with robust delimiter handling.
    - If sep is None, pandas will try to infer it with engine="python".
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(file_path)

    if sep is None:
        df = pd.read_csv(file_path, encoding_errors="ignore", sep=None, engine="python")  # Removed low_memory
    else:
        df = pd.read_csv(file_path, encoding_errors="ignore", sep=sep)

    logger.info(f"Dataset loaded successfully from: {file_path} | shape={df.shape}")
    return df


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleanup:
    - strip column names
    - parse common columns (Dt_Customer, Year_Birth)
    - filter implausible Year_Birth
    - drop duplicates
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


# -----------------------------
# Imputation by dtype
# -----------------------------
def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values:
    - numeric -> mean (fallback to median if mean is NaN)
    - categorical/object -> mode or 'Unknown'
    """
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

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

    for col in cat_cols:
        if df[col].isna().any():
            mode_series = df[col].mode(dropna=True)
            fill_val = mode_series.iloc[0] if not mode_series.empty else "Unknown"
            df.loc[:, col] = df[col].fillna(fill_val)
            logger.warning(f"Imputed categorical column '{col}' with '{fill_val}'.")

    return df


# -----------------------------
# Feature engineering
# -----------------------------
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create useful features:
    - Age, TotalPurchases, TotalSpent
    - Optional log transforms for skewed distributions
    - AgeGroup, RecencyGroup bins (if columns exist)
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

    # Binning
    if "Age" in df.columns:
        df["AgeGroup"] = pd.cut(df["Age"], bins=[18, 30, 45, 60, 120], labels=["Young", "Adult", "Mature", "Senior"], include_lowest=True)
    if "Recency" in df.columns:
        df["RecencyGroup"] = pd.cut(df["Recency"], bins=[-1, 30, 60, 120, 10000], labels=["Recent", "Mid", "Old", "Dormant"])

    # Optional log transforms for heavy skew
    for col in ["Income", "TotalSpent"]:
        if col in df.columns and (df[col] > 0).any():
            df[f"Log_{col}"] = np.log1p(df[col].clip(lower=0))

    return df


# -----------------------------
# Column profiling & recommendations
# -----------------------------

def suggest_treatment(col: str, dtype: str, n_unique: int, is_numeric: bool, high_card: bool, leak: bool, skew: Optional[float], target: Optional[str]) -> str:
    """
    Rule-based recommendations for ML preprocessing & modeling.
    """
    # Build guidance
    parts = []
    if leak:
        parts.append("⚠️ possible target leakage — review semantics")

    # ID-like columns
    # If a column is nearly unique per row, it's likely an identifier (drop from features)
    # We'll approximate this if n_unique is large relative to dataset in the calling code if needed.

    if is_numeric:
        parts.append("impute: mean/median")
        if skew is not None and abs(skew) > 1.0:
            parts.append("consider log/robust scaling (skewed)")
        else:
            parts.append("scale: standard/minmax (depending on model)")
    else:
        parts.append("impute: mode/'Unknown'")
        if high_card:
            parts.append("encoding: target/WOE or hashing; consider dimensionality reduction")
        else:
            parts.append("encoding: one-hot")

    return "; ".join(parts)


# -----------------------------
# Visualization helpers
# -----------------------------
def plot_histograms(df: pd.DataFrame, cols: List[str], outdir: Optional[str] = None) -> None:
    for col in cols:
        if col not in df.columns:
            continue
        try:
            df[col].plot(kind="hist", bins=30, alpha=0.85, title=f"Histogram: {col}")
            plt.xlabel(col); plt.ylabel("Frequency")
            plt.tight_layout()
            if outdir:
                os.makedirs(outdir, exist_ok=True)
                plt.savefig(os.path.join(outdir, f"hist_{col}.png"))
                plt.close()
            else:
                plt.show()
        except Exception as e:
            logger.error(f"Failed histogram for {col}: {e}")


def plot_boxplots(df: pd.DataFrame, cols: List[str], outdir: Optional[str] = None) -> None:
    for col in cols:
        if col not in df.columns:
            continue
        try:
            plt.boxplot(df[col].dropna(), vert=True, labels=[col])
            plt.title(f"Boxplot: {col}")
            plt.tight_layout()
            if outdir:
                os.makedirs(outdir, exist_ok=True)
                plt.savefig(os.path.join(outdir, f"box_{col}.png"))
                plt.close()
            else:
                plt.show()
        except Exception as e:
            logger.error(f"Failed boxplot for {col}: {e}")


def correlation_analysis(df: pd.DataFrame, outdir: Optional[str] = None, threshold: float = 0.9) -> pd.DataFrame:
    """
    Plot correlation matrix for numeric features and return highly-correlated pairs.
    """
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] < 2:
        logger.info("Not enough numeric columns for correlation matrix.")
        return pd.DataFrame(columns=["feature_1", "feature_2", "corr"])

    corr = num_df.corr(numeric_only=True)
    # Heatmap (matplotlib only, no custom colors)
    fig, ax = plt.subplots()
    cax = ax.imshow(corr, interpolation="nearest")
    ax.set_title("Correlation matrix (numeric features)")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticklabels(corr.columns)
    fig.colorbar(cax)
    plt.tight_layout()
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        plt.savefig(os.path.join(outdir, "correlation_matrix.png"))
        plt.close()
    else:
        plt.show()

    # Extract pairs above threshold (upper triangle only)
    high_pairs = []
    cols = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = corr.iloc[i, j]
            if abs(val) >= threshold and not np.isnan(val):
                high_pairs.append({"feature_1": cols[i], "feature_2": cols[j], "corr": float(val)})

    high_df = pd.DataFrame(high_pairs).sort_values("corr", key=lambda s: s.abs(), ascending=False)
    return high_df


def target_analysis(df: pd.DataFrame, target: str = "Response", outdir: Optional[str] = None) -> Dict[str, object]:
    """
    Analyze target distribution and basic relationships with key features.
    - If target is object with yes/no, map to 1/0 as Response_numeric.
    - Returns dict with 'distribution', 'imbalance_ratio'.
    """
    result = {}
    if target not in df.columns:
        logger.warning(f"Target '{target}' not found; skipping target analysis.")
        return result

    y = df[target]
    if y.dtype == "object":
        mapped = y.astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})
        if mapped.notna().any():
            df["Response_numeric"] = mapped
            y = df["Response_numeric"]

    dist = y.value_counts(dropna=False)
    dist_pct = (dist / len(df) * 100).round(2)
    result["distribution"] = dist.to_dict()
    result["distribution_pct"] = dist_pct.to_dict()

    if len(dist) == 2 and 0 in dist.index and 1 in dist.index:
        minority = min(dist[0], dist[1])
        majority = max(dist[0], dist[1])
        result["imbalance_ratio"] = round(majority / minority, 2) if minority > 0 else np.inf
    else:
        result["imbalance_ratio"] = None

    # Simple bar plot
    try:
        dist.plot(kind="bar", title=f"Target distribution: {target}")
        plt.xlabel(target); plt.ylabel("Count")
        plt.tight_layout()
        if outdir:
            os.makedirs(outdir, exist_ok=True)
            plt.savefig(os.path.join(outdir, f"target_{target}_distribution.png"))
            plt.close()
        else:
            plt.show()
    except Exception as e:
        logger.error(f"Failed target distribution plot: {e}")

    return result

# -----------------------------
# Advanced Analysis
# -----------------------------
def perform_rfm_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    RFM (Recency, Frequency, Monetary) Analysis
    """
    # Ensure required columns exist
    if not all(col in df.columns for col in ['Dt_Customer', 'TotalPurchases', 'TotalSpent']):
        raise ValueError("Required columns missing for RFM analysis")
    
    current_date = pd.Timestamp.now()
    
    rfm = pd.DataFrame()
    rfm['Recency'] = (current_date - pd.to_datetime(df['Dt_Customer'])).dt.days
    rfm['Frequency'] = df['TotalPurchases']
    rfm['Monetary'] = df['TotalSpent']
    
    # Calculate RFM scores (1-5)
    r_labels = range(5, 0, -1)
    r_quartiles = pd.qcut(rfm['Recency'], q=5, labels=r_labels)
    f_labels = range(1, 6)
    f_quartiles = pd.qcut(rfm['Frequency'], q=5, labels=f_labels)
    m_quartiles = pd.qcut(rfm['Monetary'], q=5, labels=f_labels)
    
    rfm['R'] = r_quartiles
    rfm['F'] = f_quartiles
    rfm['M'] = m_quartiles
    
    # Calculate RFM Score
    rfm['RFM_Score'] = rfm['R'].astype(str) + rfm['F'].astype(str) + rfm['M'].astype(str)
    
    # Customer Segmentation
    def segment_customers(row):
        if row['R'] >= 4 and row['F'] >= 4 and row['M'] >= 4:
            return 'Best Customers'
        elif row['R'] >= 3 and row['F'] >= 3 and row['M'] >= 3:
            return 'Loyal Customers'
        elif row['R'] >= 3 and row['F'] >= 1 and row['M'] >= 2:
            return 'Lost Customers'
        else:
            return 'Lost Cheap Customers'
    
    rfm['Customer_Segment'] = rfm.apply(segment_customers, axis=1)
    
    return rfm

def customer_segmentation(df: pd.DataFrame, n_clusters: int = 4) -> Dict:
    """
    Perform customer segmentation using K-means clustering
    """
    # Select features for clustering
    features = ['Age', 'Income', 'TotalSpent', 'TotalPurchases']
    X = df[features].copy()
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Add cluster labels to original dataframe
    df_clustered = df.copy()
    df_clustered['Cluster'] = clusters
    
    # Calculate cluster profiles
    cluster_profiles = df_clustered.groupby('Cluster')[features].mean()
    
    # Visualize clusters
    fig = plt.figure(figsize=(12, 8))
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap='viridis')
    plt.title('Customer Segments')
    plt.xlabel('Age (scaled)')
    plt.ylabel('Income (scaled)')
    
    return {
        'clustered_data': df_clustered,
        'cluster_profiles': cluster_profiles,
        'kmeans_model': kmeans,
        'scaler': scaler
    }


def temporal_analysis(df: pd.DataFrame) -> Dict:
    """
    Analyze temporal patterns in purchases
    """
    if 'Dt_Customer' not in df.columns:
        raise ValueError("Dt_Customer column required for temporal analysis")
    
    # Convert to datetime if needed
    df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'])
    
    # Extract time components
    df['Year'] = df['Dt_Customer'].dt.year
    df['Month'] = df['Dt_Customer'].dt.month
    df['DayOfWeek'] = df['Dt_Customer'].dt.dayofweek
    
    # Analyze patterns
    yearly_trends = df.groupby('Year')['TotalSpent'].agg(['mean', 'count'])
    monthly_trends = df.groupby('Month')['TotalSpent'].agg(['mean', 'count'])
    dow_trends = df.groupby('DayOfWeek')['TotalSpent'].agg(['mean', 'count'])
    
    return {
        'yearly_trends': yearly_trends,
        'monthly_trends': monthly_trends,
        'dow_trends': dow_trends
    }

def run_advanced_analysis(df: pd.DataFrame, output_dir: str) -> Dict:
    """
    Run all advanced analyses
    """
    results = {}
    
    # RFM Analysis
    results['rfm'] = perform_rfm_analysis(df)
    
    # Customer Segmentation
    results['segmentation'] = customer_segmentation(df)
    
    # Temporal Analysis
    results['temporal'] = temporal_analysis(df)
    
    return results

# -----------------------------
# Missingness report & plotting
# -----------------------------
def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    missing_cnt = df.isna().sum().sort_values(ascending=False)
    missing_pct = (missing_cnt / len(df) * 100).round(2)
    rep = pd.DataFrame({"missing_count": missing_cnt, "missing_pct": missing_pct})
    return rep


def plot_missing(df: pd.DataFrame, save_dir: str | None = None) -> None:
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


# -----------------------------
# Save utilities
# -----------------------------
def save_processed(df: pd.DataFrame, path: str) -> None:
    if not path:
        logger.error("No path provided for saving the processed dataset.")
        return  # Exit the function if path is empty

    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(f"Processed dataset saved to: {path}")


def save_report_csv(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(f"Report saved to: {path}")


# -----------------------------
# Main EDA runner
# -----------------------------
def analyze_data(
    file_path: str,
    processed_out: str = "data/processed/marketing_campaign_clean.csv",
    reports_dir: str | None = "docs/eda",
    advanced_dir: str | None = "docs/advanced_analysis",
    target: str = "Response",
    corr_threshold: float = 0.9,
    sep: str | None = None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Full EDA pipeline with basic and advanced analysis
    """
    # Basic EDA
    df = load_data(file_path, sep=sep)
    df = basic_cleaning(df)
    df = impute_missing(df)
    df = feature_engineering(df)

    # Basic reports and visualizations
    os.makedirs(reports_dir, exist_ok=True)  # Ensure reports directory exists
    missing_rep = missing_report(df)
    save_report_csv(missing_rep, os.path.join(reports_dir, "missing_report.csv"))

    # Basic visuals
    plot_missing(df, save_dir=reports_dir)
    num_cols = [c for c in ["Income", "TotalSpent", "TotalPurchases", "Recency", "Age"] if c in df.columns]
    plot_histograms(df, num_cols, outdir=reports_dir)
    plot_boxplots(df, num_cols, outdir=reports_dir)

    # Correlation analysis
    high_corr = correlation_analysis(df, outdir=reports_dir, threshold=corr_threshold)
    if not high_corr.empty:
        save_report_csv(high_corr, os.path.join(reports_dir, "high_correlation_pairs.csv"))

    # Target analysis
    tgt_summary = target_analysis(df, target=target, outdir=reports_dir)

    os.makedirs(advanced_dir, exist_ok=True)  # Ensure reports directory exists

    # Advanced analysis
    if advanced_dir:
        advanced_results = run_advanced_analysis(df, advanced_dir)
        for key, value in advanced_results.items():
            if isinstance(value, pd.DataFrame):
                value.to_csv(os.path.join(advanced_dir, f"{key}_results.csv"))
    else:
        advanced_results = {}

    # Save processed dataset
    save_processed(df, processed_out)

    results = {
        "column_profile_path": os.path.join(reports_dir, "column_profile.csv"),
        "missing_report_path": os.path.join(reports_dir, "missing_report.csv"),
        "high_corr_pairs_path": os.path.join(reports_dir, "high_correlation_pairs.csv") if not high_corr.empty else None,
        "target_summary": tgt_summary,
        "processed_out": processed_out,
        "advanced_analysis": advanced_results
    }
    
    logger.info(f"EDA completed. Results: {results}")
    return df, results

if __name__ == "__main__":
    FILE_PATH = os.getenv("DATA_RAW")
    logger.info(f"FILE_PATH is set to: {FILE_PATH}")
    
    analyze_data(
        FILE_PATH, 
        sep=';',
        processed_out="data/processed/marketing_campaign_clean.csv",
        reports_dir="docs/eda",
        advanced_dir="docs/advanced_analysis"
    )