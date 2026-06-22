"""
main.py
-------
Single-command entry point that runs the full end-to-end pipeline:

    1. Load + clean the data
    2. Run EDA (summary stats, correlation, insights)
    3. Generate all visualizations
    4. Train + evaluate the Logistic Regression model
    5. Print a final business-facing summary

Run with:
    python main.py
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from data_cleaning import load_data, clean_data, data_summary
from eda import summary_statistics, correlation_matrix, generate_insights
from visualization import generate_all_plots
from model import run_full_pipeline


DATA_PATH = "data/Default.xlsx"


def line(char="-", length=60):
    print(char * length)


def main():
    print("\n" + "=" * 60)
    print(" CREDIT CARD DEFAULT RISK -- END-TO-END PIPELINE")
    print("=" * 60)

    # 1. Load + clean
    print("\n[1/4] Loading and cleaning data...")
    raw_df = load_data(DATA_PATH)
    df = clean_data(raw_df)
    print(data_summary(raw_df, df))

    # 2. EDA
    print("\n[2/4] Running exploratory data analysis...")
    line()
    print("Summary statistics:\n")
    print(summary_statistics(df))
    line()
    print("\nCorrelation matrix:\n")
    print(correlation_matrix(df))
    line()
    print("\nKey insights:\n")
    for i, insight in enumerate(generate_insights(df), 1):
        print(f"{i}. {insight}\n")

    # 3. Visualizations
    print("[3/4] Generating visualizations -> outputs/figures/ ...")
    plot_paths = generate_all_plots(df)
    for p in plot_paths:
        print("  saved:", p)

    # 4. Model
    print("\n[4/4] Training and evaluating Logistic Regression model...")
    results, y_test = run_full_pipeline(df)
    lr_results = results["Logistic Regression"]
    m = lr_results["metrics"]

    line()
    print("\nMODEL PERFORMANCE (on held-out 20% test set):\n")
    print(f"  Accuracy : {m['accuracy']:.2%}")
    print(f"  Precision: {m['precision']:.2%}  (of predicted defaulters, % who actually defaulted)")
    print(f"  Recall   : {m['recall']:.2%}  (of actual defaulters, % the model caught)")
    print(f"  F1 Score : {m['f1_score']:.2%}")
    print(f"  ROC-AUC  : {m['roc_auc']:.4f}")
    print(f"\n  Confusion matrix saved -> {lr_results['cm_path']}")
    print(f"  ROC curve saved        -> {lr_results['roc_path']}")

    print("\nFeature importance (standardized coefficients):\n")
    print(lr_results["importance"].to_string(index=False))

    print("\n" + "=" * 60)
    print(" PIPELINE COMPLETE -- see outputs/figures and outputs/model")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
