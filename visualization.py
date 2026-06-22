"""
visualization.py
-----------------
Generates the key EDA visualizations (dark theme, portfolio-ready):
    1. Class balance bar chart (default Yes vs No)
    2. Balance distribution by default status
    3. Income distribution by default status
    4. Correlation heatmap
    5. Balance vs Income scatter, colored by default
    6. Default rate by student status (the confounding-variable chart)

All figures are saved to outputs/figures/ as PNG files.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns

# ---- Dark theme styling (consistent with prior portfolio projects) ----
plt.style.use("dark_background")
PALETTE = {"No": "#3DDC97", "Yes": "#FF5C5C"}  # green = no default, red = default
sns.set_palette([PALETTE["No"], PALETTE["Yes"]])
FIGSIZE = (8, 5)
DPI = 150


def _save(fig, name, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def plot_class_balance(df, out_dir="outputs/figures"):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    counts = df["default"].value_counts()
    ax.bar(counts.index, counts.values, color=[PALETTE[c] for c in counts.index])
    ax.set_title("Class Balance: Default vs No Default", fontsize=14, fontweight="bold")
    ax.set_ylabel("Number of Customers")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 100, f"{v} ({v/len(df):.1%})", ha="center")
    return _save(fig, "01_class_balance.png", out_dir)


def plot_balance_distribution(df, out_dir="outputs/figures"):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for status, color in PALETTE.items():
        sns.kdeplot(df.loc[df["default"] == status, "balance"], fill=True,
                    color=color, label=status, ax=ax, alpha=0.4)
    ax.set_title("Balance Distribution by Default Status", fontsize=14, fontweight="bold")
    ax.set_xlabel("Credit Card Balance ($)")
    ax.legend(title="Defaulted?")
    return _save(fig, "02_balance_distribution.png", out_dir)


def plot_income_distribution(df, out_dir="outputs/figures"):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for status, color in PALETTE.items():
        sns.kdeplot(df.loc[df["default"] == status, "income"], fill=True,
                    color=color, label=status, ax=ax, alpha=0.4)
    ax.set_title("Income Distribution by Default Status", fontsize=14, fontweight="bold")
    ax.set_xlabel("Annual Income ($)")
    ax.legend(title="Defaulted?")
    return _save(fig, "03_income_distribution.png", out_dir)


def plot_correlation_heatmap(df, out_dir="outputs/figures"):
    fig, ax = plt.subplots(figsize=(6, 5))
    cols = ["default_flag", "student_flag", "balance", "income"]
    corr = df[cols].corr()
    sns.heatmap(corr, annot=True, cmap="mako", fmt=".2f", ax=ax,
                cbar_kws={"label": "Correlation"})
    ax.set_title("Correlation Heatmap", fontsize=14, fontweight="bold")
    return _save(fig, "04_correlation_heatmap.png", out_dir)


def plot_balance_vs_income(df, out_dir="outputs/figures"):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for status, color in PALETTE.items():
        sub = df[df["default"] == status]
        ax.scatter(sub["income"], sub["balance"], s=12, alpha=0.5,
                    color=color, label=status)
    ax.set_title("Balance vs Income, colored by Default", fontsize=14, fontweight="bold")
    ax.set_xlabel("Annual Income ($)")
    ax.set_ylabel("Credit Card Balance ($)")
    ax.legend(title="Defaulted?")
    return _save(fig, "05_balance_vs_income.png", out_dir)


def plot_default_rate_by_student(df, out_dir="outputs/figures"):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    rate = df.groupby("student")["default_flag"].mean() * 100
    ax.bar(rate.index, rate.values, color=["#5DA9E9", "#F2B134"])
    ax.set_title("Default Rate: Student vs Non-Student", fontsize=14, fontweight="bold")
    ax.set_ylabel("Default Rate (%)")
    for i, v in enumerate(rate.values):
        ax.text(i, v + 0.05, f"{v:.2f}%", ha="center")
    return _save(fig, "06_default_rate_by_student.png", out_dir)


def generate_all_plots(df, out_dir="outputs/figures"):
    """Run every plotting function and return the list of saved file paths."""
    funcs = [
        plot_class_balance,
        plot_balance_distribution,
        plot_income_distribution,
        plot_correlation_heatmap,
        plot_balance_vs_income,
        plot_default_rate_by_student,
    ]
    return [f(df, out_dir) for f in funcs]


if __name__ == "__main__":
    from data_cleaning import load_data, clean_data

    df = clean_data(load_data("data/Default.xlsx"))
    paths = generate_all_plots(df)
    print("Saved plots:")
    for p in paths:
        print(" -", p)
