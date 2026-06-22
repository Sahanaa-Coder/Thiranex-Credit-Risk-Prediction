"""
app.py
------
Streamlit dashboard for the Credit Card Default Risk project.

Run with:
    streamlit run app.py

Reuses the existing src/ pipeline (data_cleaning, model) so the dashboard
always stays in sync with the core analysis code.
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve

from data_cleaning import load_data, clean_data
from eda import summary_statistics, correlation_matrix, generate_insights, class_balance
from model import prepare_features, train_model, evaluate_model, feature_importance
from sklearn.model_selection import train_test_split
import plotly_charts as pc


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Credit Default Risk Dashboard",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM DARK THEME CSS
# ============================================================
st.markdown("""
<style>
    .stApp {
        background-color: #0E1117;
    }
    .metric-card {
        background: linear-gradient(135deg, #1A1F2B 0%, #161B26 100%);
        border: 1px solid #2A3040;
        border-radius: 12px;
        padding: 18px 20px;
        text-align: center;
    }
    .metric-label {
        font-size: 13px;
        color: #9BA3B4;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #F5F7FA;
    }
    .insight-box {
        background: #161B26;
        border-left: 4px solid #3DDC97;
        border-radius: 6px;
        padding: 12px 16px;
        margin-bottom: 10px;
        color: #D7DCE5;
        font-size: 14.5px;
    }
    h1, h2, h3 { color: #F5F7FA; }
    section[data-testid="stSidebar"] {
        background-color: #11151D;
        border-right: 1px solid #232938;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# DATA + MODEL (cached so they don't re-run on every interaction)
# ============================================================
@st.cache_data
def get_data():
    raw = load_data("data/Default.xlsx")
    return clean_data(raw)


@st.cache_resource
def get_trained_model(_df):
    X, y = prepare_features(_df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    scaler, model = train_model(X_train, y_train)
    metrics, y_pred, y_proba = evaluate_model(model, scaler, X_test, y_test)
    importance = feature_importance(model, scaler)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    return {
        "scaler": scaler, "model": model, "metrics": metrics,
        "importance": importance, "fpr": fpr, "tpr": tpr,
    }


df = get_data()
trained = get_trained_model(df)


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## 💳 Default Risk Dashboard")
    st.caption("CodeAlpha Data Science Internship · Finance")
    st.markdown("---")
    st.markdown("### About this dataset")
    st.write(f"**Customers:** {len(df):,}")
    st.write(f"**Default rate:** {df['default_flag'].mean():.2%}")
    st.write("**Features:** balance, income, student status")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["📊 EDA", "🤖 Model Performance", "🔮 Live Risk Predictor"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("Built with Streamlit · scikit-learn · Plotly")


# ============================================================
# PAGE 1 — EDA
# ============================================================
if page == "📊 EDA":
    st.title("📊 Exploratory Data Analysis")
    st.caption("Understanding what separates defaulters from non-defaulters.")

    # Top metric cards
    c1, c2, c3, c4 = st.columns(4)
    cards = [
        ("Total Customers", f"{len(df):,}"),
        ("Default Rate", f"{df['default_flag'].mean():.2%}"),
        ("Avg Balance (Default)", f"${df.loc[df['default']=='Yes','balance'].mean():,.0f}"),
        ("Avg Balance (No Default)", f"${df.loc[df['default']=='No','balance'].mean():,.0f}"),
    ]
    for col, (label, value) in zip([c1, c2, c3, c4], cards):
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("###")

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(pc.class_balance_chart(df), use_container_width=True)
    with col2:
        st.plotly_chart(pc.default_rate_by_student_chart(df), use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(pc.distribution_chart(df, "balance", "Balance Distribution by Default Status",
                                               "Credit Card Balance ($)"), use_container_width=True)
    with col4:
        st.plotly_chart(pc.distribution_chart(df, "income", "Income Distribution by Default Status",
                                               "Annual Income ($)"), use_container_width=True)

    col5, col6 = st.columns([1.3, 1])
    with col5:
        st.plotly_chart(pc.balance_vs_income_scatter(df), use_container_width=True)
    with col6:
        st.plotly_chart(pc.correlation_heatmap(df), use_container_width=True)

    st.markdown("### 🔍 Summary Statistics")
    st.dataframe(summary_statistics(df), use_container_width=True)

    st.markdown("### 💡 Key Insights")
    for insight in generate_insights(df):
        st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)


# ============================================================
# PAGE 2 — MODEL PERFORMANCE
# ============================================================
elif page == "🤖 Model Performance":
    st.title("🤖 Model Performance")
    st.caption("Logistic Regression, trained on balance / income / student status.")

    m = trained["metrics"]
    c1, c2, c3, c4, c5 = st.columns(5)
    metric_cards = [
        ("Accuracy", f"{m['accuracy']:.1%}"),
        ("Precision", f"{m['precision']:.1%}"),
        ("Recall", f"{m['recall']:.1%}"),
        ("F1 Score", f"{m['f1_score']:.1%}"),
        ("ROC-AUC", f"{m['roc_auc']:.3f}"),
    ]
    for col, (label, value) in zip([c1, c2, c3, c4, c5], metric_cards):
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("###")

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(pc.confusion_matrix_chart(m["confusion_matrix"]), use_container_width=True)
    with col2:
        st.plotly_chart(pc.roc_curve_chart(trained["fpr"], trained["tpr"], m["roc_auc"]),
                         use_container_width=True)

    st.plotly_chart(pc.feature_importance_chart(trained["importance"]), use_container_width=True)

    with st.expander("📋 Full classification report"):
        st.text(m["classification_report"])

    st.markdown("""
    <div class="insight-box">
    <b>Why recall over precision?</b> In credit risk, missing an actual defaulter is usually
    far more costly than a false alarm. This model is tuned (via balanced class weights) to
    catch as many real defaulters as possible, accepting more false positives as the trade-off.
    Treat the model's output as a <i>risk score for review</i>, not an automatic rejection.
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# PAGE 3 — LIVE RISK PREDICTOR
# ============================================================
else:
    st.title("🔮 Live Risk Predictor")
    st.caption("Enter a hypothetical customer's details to see their predicted default risk.")

    col_input, col_result = st.columns([1, 1.4])

    with col_input:
        st.markdown("#### Customer details")
        balance = st.slider("Credit card balance ($)", 0, 3000, 800, step=10)
        income = st.slider("Annual income ($)", 0, 80000, 35000, step=500)
        student = st.selectbox("Student?", ["No", "Yes"])
        student_flag = 1 if student == "Yes" else 0

        X_input = pd.DataFrame([{
            "balance": balance, "income": income, "student_flag": student_flag
        }])
        X_scaled = trained["scaler"].transform(X_input)
        proba = trained["model"].predict_proba(X_scaled)[0, 1]
        prediction = "Default" if proba >= 0.5 else "No Default"

        st.markdown("####")
        if prediction == "Default":
            st.error(f"⚠️ Predicted: **{prediction}**")
        else:
            st.success(f"✅ Predicted: **{prediction}**")

    with col_result:
        st.plotly_chart(pc.risk_gauge_chart(proba), use_container_width=True)

    st.markdown("---")
    st.markdown("#### How this customer compares to the dataset")
    comp = pd.DataFrame({
        "Metric": ["Balance ($)", "Income ($)"],
        "This customer": [balance, income],
        "Dataset average": [round(df["balance"].mean()), round(df["income"].mean())],
        "Average for defaulters": [
            round(df.loc[df["default"] == "Yes", "balance"].mean()),
            round(df.loc[df["default"] == "Yes", "income"].mean()),
        ],
    })
    st.dataframe(comp, use_container_width=True, hide_index=True)

    st.markdown("""
    <div class="insight-box">
    💡 Try pushing balance above ~$1,800 with average income — notice how sharply the risk
    score climbs. Balance is by far the strongest driver of predicted default risk in this model.
    </div>
    """, unsafe_allow_html=True)
