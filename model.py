"""
model.py
--------
Trains and evaluates a simple, interpretable classification model to
predict credit default.

Why Logistic Regression?
    - The target (default: Yes/No) is binary -> classification problem.
    - Logistic Regression is easy to explain to a non-technical audience
      (it gives each feature a clear positive/negative weight).
    - It's the standard beginner-friendly baseline for credit risk problems.

Why "class_weight='balanced'"?
    - Only ~3.3% of customers default. A model trained naively will just
      learn to predict "No" every time and still score >96% accuracy --
      while being completely useless. Balancing the class weights forces
      the model to actually pay attention to the rare "Yes" class.
"""

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
import matplotlib.pyplot as plt


FEATURES = ["balance", "income", "student_flag"]
TARGET = "default_flag"


def prepare_features(df: pd.DataFrame):
    """Split into X (features) and y (target)."""
    X = df[FEATURES]
    y = df[TARGET]
    return X, y


def train_model(X_train, y_train):
    """
    Scale features and train a balanced Logistic Regression model.
    Returns the fitted scaler and model (both needed at prediction time).
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(class_weight="balanced", random_state=42)
    model.fit(X_train_scaled, y_train)

    return scaler, model


def train_models(X_train, y_train):
    """
    Scale features and train three models: Logistic Regression, Random Forest,
    and Gradient Boosting. Returns the fitted scaler and a dictionary of fitted models.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    models = {
        "Logistic Regression": LogisticRegression(class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(class_weight="balanced", random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42)
    }

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)

    return scaler, models


def evaluate_model(model, scaler, X_test, y_test) -> dict:
    """Compute the standard classification metrics on the held-out test set."""
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred),
    }
    return metrics, y_pred, y_proba


def plot_confusion_matrix(cm, out_dir="outputs/figures", suffix=""):
    import seaborn as sns
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="mako",
                xticklabels=["No Default", "Default"],
                yticklabels=["No Default", "Default"], ax=ax)
    ax.set_title("Confusion Matrix", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    os.makedirs(out_dir, exist_ok=True)
    
    if suffix == "logistic_regression":
        filename = "07_confusion_matrix.png"
    elif suffix:
        filename = f"07_{suffix}_confusion_matrix.png"
    else:
        filename = "07_confusion_matrix.png"
        
    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def plot_roc_curve(y_test, y_proba, roc_auc, out_dir="outputs/figures", suffix=""):
    plt.style.use("dark_background")
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#3DDC97", linewidth=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
    ax.set_title("ROC Curve", fontsize=13, fontweight="bold")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    os.makedirs(out_dir, exist_ok=True)
    
    if suffix == "logistic_regression":
        filename = "08_roc_curve.png"
    elif suffix:
        filename = f"08_{suffix}_roc_curve.png"
    else:
        filename = "08_roc_curve.png"
        
    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def feature_importance(model, scaler) -> pd.DataFrame:
    """Return feature importances/coefficients depending on model type."""
    if hasattr(model, "coef_"):
        importance = model.coef_[0]
        col_name = "coefficient"
    elif hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
        col_name = "coefficient"
    else:
        importance = [0.0] * len(FEATURES)
        col_name = "coefficient"

    df_imp = pd.DataFrame({
        "feature": FEATURES,
        col_name: importance
    })

    if hasattr(model, "coef_"):
        df_imp = df_imp.sort_values(col_name, key=abs, ascending=False)
    else:
        df_imp = df_imp.sort_values(col_name, ascending=False)
        
    return df_imp


def save_model(model, scaler, out_dir="outputs/model"):
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(model, os.path.join(out_dir, "logistic_regression.joblib"))
    joblib.dump(scaler, os.path.join(out_dir, "scaler.joblib"))


def save_models(models, scaler, out_dir="outputs/model"):
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(out_dir, "scaler.joblib"))
    for name, model in models.items():
        filename = name.lower().replace(" ", "_") + ".joblib"
        joblib.dump(model, os.path.join(out_dir, filename))


def run_full_pipeline(df: pd.DataFrame):
    """Split, train all models, evaluate, plot, save -- in one call."""
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler, models = train_models(X_train, y_train)
    save_models(models, scaler)

    results = {}
    for name, model in models.items():
        metrics, y_pred, y_proba = evaluate_model(model, scaler, X_test, y_test)
        importance = feature_importance(model, scaler)

        prefix = name.lower().replace(" ", "_")
        cm_path = plot_confusion_matrix(metrics["confusion_matrix"], suffix=prefix)
        roc_path = plot_roc_curve(y_test, y_proba, metrics["roc_auc"], suffix=prefix)

        results[name] = {
            "metrics": metrics,
            "importance": importance,
            "y_pred": y_pred,
            "y_proba": y_proba,
            "cm_path": cm_path,
            "roc_path": roc_path,
        }

    return results, y_test


if __name__ == "__main__":
    from data_cleaning import load_data, clean_data

    df = clean_data(load_data("data/Default.xlsx"))
    results, y_test = run_full_pipeline(df)

    m = results["metrics"]
    print(f"Accuracy : {m['accuracy']:.4f}")
    print(f"Precision: {m['precision']:.4f}")
    print(f"Recall   : {m['recall']:.4f}")
    print(f"F1 Score : {m['f1_score']:.4f}")
    print(f"ROC-AUC  : {m['roc_auc']:.4f}")
    print()
    print(m["classification_report"])
    print(results["importance"])
