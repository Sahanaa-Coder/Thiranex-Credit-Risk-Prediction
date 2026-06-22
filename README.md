# 💳 Credit Card Default Risk — Prediction & Dashboard

A compact end-to-end credit risk project that cleans the dataset, performs exploratory
analysis, trains a Logistic Regression model, and presents results in a Streamlit dashboard.

> Built as part of a CodeAlpha Data Science Internship project.

---

## 📊 Project Overview

| | |
|---|---|
| **Dataset** | 10,000 credit card customers |
| **Target** | `default` — did the customer default? (Yes/No) |
| **Features** | `balance`, `income`, `student` status |
| **Problem type** | Binary classification (imbalanced — 3.33% default rate) |
| **Model** | Logistic Regression (`class_weight='balanced'`) |
| **Dashboard** | `app.py` — interactive Streamlit dashboard |

---

## 🗂️ Project Structure

```
Default-Risk-Prediction/
├── data/
│   └── Default.xlsx              # raw dataset
├── outputs/
│   ├── figures/                  # generated chart images
│   └── model/                    # saved model + scaler (.joblib)
├── notebooks/
│   └── Default_Risk_Analysis.ipynb
├── src/
│   ├── data_cleaning.py          # load + clean data
│   ├── eda.py                    # summary statistics and insights
│   ├── model.py                  # model training and evaluation
│   ├── plotly_charts.py          # Streamlit chart helpers
│   └── visualization.py          # static chart generation
├── app.py                        # Streamlit dashboard
├── main.py                       # pipeline runner
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

```bash
cd Default-Risk-Prediction
pip install -r requirements.txt

# Run the full pipeline
python main.py

# Or launch the dashboard
streamlit run app.py
```

Then open the dashboard at `http://localhost:8501`.

---

## 🔎 What the project does

- Loads and cleans `data/Default.xlsx`
- Computes summary statistics, correlations, and automatic insights
- Trains a balanced Logistic Regression model on `balance`, `income`, and `student`
- Saves charts to `outputs/figures/` and model artifacts to `outputs/model/`
- Provides an interactive Streamlit dashboard for EDA, model performance, and live risk prediction

---

## 📈 Model Performance Summary

- Accuracy: ~85.5%
- Precision: ~17.3%
- Recall: ~88.1%
- F1 Score: ~28.9%
- ROC-AUC: ~0.95

This model uses balanced class weights to handle the low default rate in the data.

---

## 🌐 Dashboard Pages

- **📊 EDA** — interactive charts and dataset metrics
- **🤖 Model Performance** — evaluation metrics, confusion matrix, ROC curve, feature importance
- **🔮 Live Risk Predictor** — user inputs a sample customer and sees predicted default risk

---

## 🛠️ Tech Stack

Python · pandas · NumPy · scikit-learn · Streamlit · Plotly · matplotlib · seaborn

---

## 📌 Notes

- `main.py` runs the full data + model pipeline and generates static charts
- `app.py` runs the Streamlit dashboard using the same `src/` modules
- The project root no longer contains duplicate files or nested duplicate folders

---

## 📬 Built by

Sahanaa — CodeAlpha Data Science Internship portfolio
- Deploy as an interactive Streamlit dashboard (see below)

---

## 🌐 Interactive Streamlit Dashboard

`app.py` is a fully working, dark-themed Streamlit dashboard built on top of the `src/` modules
— run it with `streamlit run app.py`. It has three pages:

- **📊 EDA** — live metric cards, interactive Plotly versions of the class balance, balance/income
  distributions, correlation heatmap, and balance-vs-income scatter, plus the auto-generated
  text insights.
- **🤖 Model Performance** — accuracy/precision/recall/F1/ROC-AUC metric cards, interactive
  confusion matrix, ROC curve, and feature importance chart.
- **🔮 Live Risk Predictor** — sliders for balance, income, and a student toggle. Instantly
  scores a hypothetical customer with the trained Logistic Regression model and shows their
  risk as a gauge chart, plus a comparison table against dataset/defaulter averages.

The model is trained once on app startup (cached with `@st.cache_resource`) so the dashboard
stays fast even though predictions are computed live as you move the sliders.

---

## 🎤 Internship Presentation Outline

1. **Title slide** — Project name, your name, CodeAlpha internship, date
2. **Problem statement** — Why predicting credit default matters for a business
3. **Dataset overview** — 10,000 customers, 3 features, target variable
4. **Data cleaning** — what was checked (missing values, duplicates, types)
5. **EDA highlights** — class imbalance chart, balance-vs-income scatter, correlation heatmap
6. **The confounding-variable story** — students vs. balance vs. income (great talking point!)
7. **Model choice & why** — Logistic Regression, balanced class weights
8. **Results** — ROC-AUC, confusion matrix, recall-over-precision rationale
9. **Business insights** — the 7 insights above, framed as recommendations
10. **Next steps** — improvements + Streamlit dashboard demo
11. **Thank you / Q&A**

---

## 📬 Connect

Built by **Sahanaa** as part of a CodeAlpha Data Science & AI/ML internship portfolio.
