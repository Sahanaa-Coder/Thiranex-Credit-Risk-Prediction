"""
plotly_charts.py
-----------------
Interactive (Plotly) versions of the EDA / model charts, built specifically
for the Streamlit dashboard. Kept separate from visualization.py (which
generates the static matplotlib PNGs used in the notebook / README) so the
two can evolve independently.

Color scheme matches the rest of the portfolio (dark theme):
    No default  -> green  (#3DDC97)
    Default     -> red    (#FF5C5C)
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

COLOR_NO = "#3DDC97"
COLOR_YES = "#FF5C5C"
COLOR_MAP = {"No": COLOR_NO, "Yes": COLOR_YES}

TEMPLATE = "plotly_dark"
PAPER_BG = "rgba(0,0,0,0)"
PLOT_BG = "rgba(0,0,0,0)"


def _style(fig, height=420):
    fig.update_layout(
        template=TEMPLATE,
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        height=height,
        margin=dict(l=40, r=20, t=50, b=40),
        font=dict(family="Inter, Helvetica, Arial, sans-serif", size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def class_balance_chart(df: pd.DataFrame):
    counts = df["default"].value_counts().reindex(["No", "Yes"])
    fig = go.Figure(go.Bar(
        x=counts.index, y=counts.values,
        marker_color=[COLOR_NO, COLOR_YES],
        text=[f"{v:,} ({v/len(df):.1%})" for v in counts.values],
        textposition="outside",
    ))
    fig.update_layout(title="Class Balance: Default vs No Default",
                       xaxis_title="", yaxis_title="Number of Customers")
    return _style(fig)


def distribution_chart(df: pd.DataFrame, column: str, title: str, xaxis_title: str):
    fig = go.Figure()
    for status, color in COLOR_MAP.items():
        sub = df.loc[df["default"] == status, column]
        fig.add_trace(go.Violin(
            x=sub, line_color=color, fillcolor=color, opacity=0.5,
            name=status, side="positive", orientation="h", points=False,
        ))
    fig.update_layout(title=title, xaxis_title=xaxis_title, yaxis_title="")
    return _style(fig)


def correlation_heatmap(df: pd.DataFrame):
    cols = ["default_flag", "student_flag", "balance", "income"]
    corr = df[cols].corr().round(3)
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.columns,
        colorscale="Tealgrn", zmid=0,
        text=corr.values, texttemplate="%{text}",
    ))
    fig.update_layout(title="Correlation Heatmap")
    return _style(fig, height=400)


def balance_vs_income_scatter(df: pd.DataFrame, sample_n: int = 4000):
    plot_df = df if len(df) <= sample_n else df.sample(sample_n, random_state=42)
    fig = px.scatter(
        plot_df, x="income", y="balance", color="default",
        color_discrete_map=COLOR_MAP, opacity=0.55,
        labels={"income": "Annual Income ($)", "balance": "Credit Card Balance ($)",
                "default": "Defaulted?"},
        title="Balance vs Income, colored by Default",
    )
    fig.update_traces(marker=dict(size=6))
    return _style(fig, height=460)


def default_rate_by_student_chart(df: pd.DataFrame):
    rate = df.groupby("student")["default_flag"].mean() * 100
    fig = go.Figure(go.Bar(
        x=rate.index, y=rate.values,
        marker_color=["#5DA9E9", "#F2B134"],
        text=[f"{v:.2f}%" for v in rate.values],
        textposition="outside",
    ))
    fig.update_layout(title="Default Rate: Student vs Non-Student",
                       xaxis_title="", yaxis_title="Default Rate (%)")
    return _style(fig)


def confusion_matrix_chart(cm: np.ndarray):
    labels = ["No Default", "Default"]
    fig = go.Figure(go.Heatmap(
        z=cm, x=labels, y=labels,
        colorscale="Tealgrn",
        text=cm, texttemplate="%{text}",
        textfont=dict(size=18),
    ))
    fig.update_layout(title="Confusion Matrix", xaxis_title="Predicted", yaxis_title="Actual")
    return _style(fig, height=400)


def roc_curve_chart(fpr, tpr, roc_auc):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                              line=dict(color=COLOR_NO, width=3),
                              name=f"ROC curve (AUC = {roc_auc:.3f})"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                              line=dict(color="gray", dash="dash"),
                              name="Random guess"))
    fig.update_layout(title="ROC Curve", xaxis_title="False Positive Rate",
                       yaxis_title="True Positive Rate")
    return _style(fig, height=420)


def feature_importance_chart(importance_df: pd.DataFrame):
    df_sorted = importance_df.sort_values("coefficient")
    colors = [COLOR_YES if c > 0 else COLOR_NO for c in df_sorted["coefficient"]]
    fig = go.Figure(go.Bar(
        x=df_sorted["coefficient"], y=df_sorted["feature"],
        orientation="h", marker_color=colors,
    ))
    fig.update_layout(title="Feature Importance (Standardized Coefficients)",
                       xaxis_title="Coefficient", yaxis_title="")
    return _style(fig, height=320)


def risk_gauge_chart(probability: float):
    """Gauge chart for the live predictor tab — shows a single customer's risk score."""
    if probability < 0.3:
        bar_color = COLOR_NO
    elif probability < 0.6:
        bar_color = "#F2B134"
    else:
        bar_color = COLOR_YES

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        number={"suffix": "%", "font": {"size": 40}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": bar_color},
            "steps": [
                {"range": [0, 30], "color": "rgba(61,220,151,0.2)"},
                {"range": [30, 60], "color": "rgba(242,177,52,0.2)"},
                {"range": [60, 100], "color": "rgba(255,92,92,0.2)"},
            ],
        },
        title={"text": "Predicted Default Risk"},
    ))
    fig.update_layout(template=TEMPLATE, paper_bgcolor=PAPER_BG, height=320,
                       margin=dict(l=20, r=20, t=60, b=20))
    return fig


def model_comparison_bar_chart(comparison_df: pd.DataFrame):
    """Grouped bar chart comparing metrics across multiple models."""
    melted = comparison_df.reset_index().rename(columns={"index": "Model"}).melt(
        id_vars="Model", var_name="Metric", value_name="Score"
    )
    fig = px.bar(
        melted, x="Metric", y="Score", color="Model", barmode="group",
        color_discrete_sequence=["#3DDC97", "#5DA9E9", "#F2B134"],
        labels={"Score": "Score (0.0 to 1.0)", "Metric": ""},
        title="Model Performance Metrics Comparison",
    )
    fig.update_layout(yaxis_range=[0, 1.05])
    return _style(fig, height=400)


def multi_model_roc_curve(roc_curves_dict: dict):
    """Plot ROC curves for multiple models on a single chart."""
    fig = go.Figure()
    colors = {
        "Logistic Regression": "#3DDC97",
        "Random Forest": "#5DA9E9",
        "Gradient Boosting": "#F2B134"
    }
    for name, (fpr, tpr, roc_auc) in roc_curves_dict.items():
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines",
            line=dict(color=colors.get(name, "#FFFFFF"), width=2.5),
            name=f"{name} (AUC = {roc_auc:.3f})"
        ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(color="gray", dash="dash"),
        name="Random guess"
    ))
    fig.update_layout(
        title="Receiver Operating Characteristic (ROC) Comparison",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate"
    )
    return _style(fig, height=420)


def cost_curve_chart(thresholds, costs, optimal_t, min_cost):
    """Plot total business cost vs decision threshold."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=thresholds, y=costs, mode="lines",
        line=dict(color="#FF5C5C", width=3),
        name="Business Cost"
    ))
    
    fig.add_vline(
        x=optimal_t, line_width=2, line_dash="dash", line_color="#3DDC97",
        annotation_text=f"Optimal: {optimal_t:.2f} (${min_cost:,.0f})",
        annotation_position="bottom right"
    )
    
    fig.update_layout(
        title="Total Business Cost vs. Decision Threshold",
        xaxis_title="Classification Threshold",
        yaxis_title="Total Financial Loss ($)",
    )
    return _style(fig, height=380)
