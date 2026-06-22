"""
data_cleaning.py
-----------------
Loads the raw Default.xlsx dataset and applies cleaning steps:
- drops redundant index column
- checks/handles missing values and duplicates
- fixes data types
- encodes categorical Yes/No columns for modeling

Dataset: Credit Card Customer Default dataset (10,000 rows)
Columns:
    Unnamed: 0  -> redundant row index (dropped)
    default     -> target variable, "Yes"/"No" (did the customer default?)
    student     -> "Yes"/"No" (is the customer a student?)
    balance     -> average credit card balance ($)
    income      -> annual income ($)
"""

import pandas as pd
import warnings

# Suppress openpyxl warning about workbook styling
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")


def load_data(path: str) -> pd.DataFrame:
    """Load the raw Excel file into a DataFrame."""
    df = pd.read_excel(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all cleaning steps and return an analysis-ready DataFrame.

    Steps:
        1. Drop the unnamed index column (it's just a row counter, no
           predictive value).
        2. Check for missing values (this dataset has none, but we check
           anyway -- good practice for any real-world project).
        3. Check for duplicate rows.
        4. Confirm/cast data types (categorical columns as 'category',
           numeric columns as float).
        5. Create encoded numeric versions of the Yes/No columns, since
           ML models need numbers, not text.
    """
    df = df.copy()

    # 1. Drop redundant index column if present
    unnamed_cols = [c for c in df.columns if c.lower().startswith("unnamed")]
    df = df.drop(columns=unnamed_cols, errors="ignore")

    # 2. Handle missing values
    missing_before = df.isnull().sum().sum()
    if missing_before > 0:
        # Numeric columns -> fill with median (robust to outliers)
        for col in df.select_dtypes(include="number").columns:
            df[col] = df[col].fillna(df[col].median())
        # Categorical columns -> fill with mode
        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].fillna(df[col].mode()[0])

    # 3. Drop exact duplicate rows
    df = df.drop_duplicates()

    # 4. Fix data types
    df["default"] = df["default"].astype("category")
    df["student"] = df["student"].astype("category")
    df["balance"] = df["balance"].astype(float)
    df["income"] = df["income"].astype(float)

    # 5. Encode categorical Yes/No columns as 0/1 for modeling
    df["default_flag"] = (df["default"] == "Yes").astype(int)
    df["student_flag"] = (df["student"] == "Yes").astype(int)

    return df


def data_summary(df_raw: pd.DataFrame, df_clean: pd.DataFrame) -> str:
    """Return a short human-readable cleaning report (for README / console)."""
    lines = [
        f"Raw shape:      {df_raw.shape}",
        f"Cleaned shape:  {df_clean.shape}",
        f"Missing values (raw):   {df_raw.isnull().sum().sum()}",
        f"Missing values (clean): {df_clean.isnull().sum().sum()}",
        f"Duplicate rows removed: {df_raw.shape[0] - df_clean.drop_duplicates().shape[0] if df_raw.shape[0]==df_clean.shape[0] else 'see drop step'}",
        f"Default rate: {df_clean['default_flag'].mean():.2%}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raw = load_data("data/Default.xlsx")
    clean = clean_data(raw)
    print(data_summary(raw, clean))
    print(clean.head())
