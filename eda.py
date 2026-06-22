"""
eda.py
------
Exploratory Data Analysis helpers: summary statistics, correlation
analysis, and automatically generated text insights describing the
key financial patterns in the dataset.
"""

import pandas as pd


def summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Return descriptive statistics for the numeric columns."""
    return df[["balance", "income"]].describe().round(2)


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Return the correlation matrix for the (encoded) numeric features."""
    cols = ["default_flag", "student_flag", "balance", "income"]
    return df[cols].corr().round(3)


def class_balance(df: pd.DataFrame) -> pd.Series:
    """Return the count and % share of defaulters vs non-defaulters."""
    counts = df["default"].value_counts()
    pct = df["default"].value_counts(normalize=True).mul(100).round(2)
    return pd.DataFrame({"count": counts, "percent": pct})


def group_means(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Return mean balance/income split by a grouping column (e.g. default, student)."""
    return df.groupby(group_col)[["balance", "income"]].mean().round(2)


def generate_insights(df: pd.DataFrame) -> list:
    """
    Generate a list of plain-English insight strings from the data.
    This mirrors the manual analysis a data scientist would write up,
    but computed directly from the current dataframe so it stays
    accurate if the data changes.
    """
    insights = []

    default_rate = df["default_flag"].mean()
    insights.append(
        f"Only {default_rate:.2%} of customers defaulted -- this is a highly "
        f"imbalanced classification problem, so accuracy alone will be a "
        f"misleading metric (a model that always predicts 'No' would already "
        f"be ~{1-default_rate:.1%} 'accurate')."
    )

    bal_by_default = df.groupby("default")["balance"].mean()
    insights.append(
        f"Defaulters carry a much higher average balance (${bal_by_default['Yes']:.0f}) "
        f"than non-defaulters (${bal_by_default['No']:.0f}) -- balance is the "
        f"single strongest behavioral signal of default risk in this dataset."
    )

    inc_by_default = df.groupby("default")["income"].mean()
    insights.append(
        f"Average income is almost identical between defaulters (${inc_by_default['Yes']:.0f}) "
        f"and non-defaulters (${inc_by_default['No']:.0f}) -- income alone is a "
        f"weak predictor of default risk."
    )

    default_rate_by_student = df.groupby("student")["default_flag"].mean()
    insights.append(
        f"Students default at a higher raw rate ({default_rate_by_student['Yes']:.2%}) than "
        f"non-students ({default_rate_by_student['No']:.2%})."
    )

    bal_by_student = df.groupby("student")["balance"].mean()
    inc_by_student = df.groupby("student")["income"].mean()
    insights.append(
        f"However, this is a confounding effect: students carry higher balances "
        f"(${bal_by_student['Yes']:.0f} vs ${bal_by_student['No']:.0f}) but much "
        f"lower income (${inc_by_student['Yes']:.0f} vs ${inc_by_student['No']:.0f}). "
        f"So 'being a student' isn't directly risky -- it's a proxy for carrying "
        f"a higher balance relative to income."
    )

    zero_balance_pct = (df["balance"] == 0).mean()
    insights.append(
        f"About {zero_balance_pct:.1%} of customers carry a $0 balance (i.e. they "
        f"don't revolve credit) -- these customers almost never default, and could "
        f"be treated as a distinct 'low-risk / inactive' segment for credit policy."
    )

    return insights


if __name__ == "__main__":
    from data_cleaning import load_data, clean_data

    df = clean_data(load_data("data/Default.xlsx"))
    print(summary_statistics(df))
    print()
    print(correlation_matrix(df))
    print()
    for i, insight in enumerate(generate_insights(df), 1):
        print(f"{i}. {insight}\n")
