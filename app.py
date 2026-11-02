"""
Streamlit web app for BITS WILP ML Assignment 2.
Loads pre-trained classifiers and evaluates an uploaded test CSV
without retraining.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}


@st.cache_resource
def load_artifacts():
    label_encoder = joblib.load(MODEL_DIR / "label_encoder.joblib")
    feature_columns = joblib.load(MODEL_DIR / "feature_columns.joblib")
    models = {
        name: joblib.load(MODEL_DIR / filename)
        for name, filename in MODEL_FILES.items()
    }
    return models, label_encoder, feature_columns


def evaluate_model(model, X, y_true_encoded, label_encoder):
    y_pred = model.predict(X)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X)
    else:
        n_classes = len(label_encoder.classes_)
        y_proba = np.zeros((len(y_true_encoded), n_classes))
        y_proba[np.arange(len(y_true_encoded)), y_pred] = 1.0

    metrics = {
        "Accuracy": accuracy_score(y_true_encoded, y_pred),
        "AUC": roc_auc_score(
            y_true_encoded,
            y_proba,
            multi_class="ovr",
            average="weighted",
            labels=list(range(len(label_encoder.classes_))),
        ),
        "Precision": precision_score(
            y_true_encoded, y_pred, average="weighted", zero_division=0
        ),
        "Recall": recall_score(
            y_true_encoded, y_pred, average="weighted", zero_division=0
        ),
        "F1 Score": f1_score(
            y_true_encoded, y_pred, average="weighted", zero_division=0
        ),
        "MCC": matthews_corrcoef(y_true_encoded, y_pred),
    }

    report = classification_report(
        y_true_encoded,
        y_pred,
        target_names=list(label_encoder.classes_),
        zero_division=0,
    )
    cm = confusion_matrix(y_true_encoded, y_pred)
    return metrics, report, cm, y_pred


def plot_confusion_matrix(cm, class_names, model_name):
    fig, ax = plt.subplots(figsize=(7.5, 5.8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="YlGnBu",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix — {model_name}")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    fig.tight_layout()
    return fig


def main():
    st.set_page_config(
        page_title="Dry Bean Classifier | ML Assignment 2",
        page_icon="🫘",
        layout="wide",
    )

    st.title("Dry Bean Variety Classifier")
    st.markdown(
        """
        **BITS WILP M.Tech (AIML) — Machine Learning Assignment 2**

        Upload a test CSV that contains the 16 geometric features plus the
        `Class` target column. The app loads the **already trained** models
        and reports Accuracy, AUC, Precision, Recall, F1, and MCC without
        retraining.
        """
    )

    models, label_encoder, feature_columns = load_artifacts()

    with st.sidebar:
        st.header("Controls")
        selected_model_name = st.selectbox(
            "Select classification model",
            options=list(MODEL_FILES.keys()),
            index=0,
        )
        st.markdown("---")
        st.subheader("Expected columns")
        st.caption(", ".join(feature_columns + ["Class"]))
        st.markdown("---")
        st.info(
            "Tip: use the bundled `test_data.csv` from the GitHub repository "
            "for a quick demo."
        )

    uploaded = st.file_uploader(
        "Upload test data CSV (features + Class target)",
        type=["csv"],
        help="Streamlit free tier has limited capacity — upload only test data.",
    )

    if uploaded is None:
        st.warning("Upload a CSV file to begin evaluation.")
        st.markdown("### Sample of required schema")
        sample = pd.DataFrame(
            columns=feature_columns + ["Class"],
            data=[[0.0] * len(feature_columns) + ["SEKER"]],
        )
        st.dataframe(sample.head(1), use_container_width=True)
        return

    try:
        test_df = pd.read_csv(uploaded)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not read CSV: {exc}")
        return

    missing = [c for c in feature_columns + ["Class"] if c not in test_df.columns]
    if missing:
        st.error(
            "Uploaded CSV is missing required columns: "
            + ", ".join(missing)
        )
        return

    X_test = test_df[feature_columns]
    y_true_labels = test_df["Class"].astype(str).str.strip()

    unknown = sorted(set(y_true_labels) - set(label_encoder.classes_))
    if unknown:
        st.error(
            "Uploaded Class labels contain values unseen during training: "
            + ", ".join(unknown)
        )
        return

    y_true = label_encoder.transform(y_true_labels)
    model = models[selected_model_name]

    with st.spinner(f"Evaluating {selected_model_name} on uploaded test data..."):
        metrics, report, cm, y_pred = evaluate_model(
            model, X_test, y_true, label_encoder
        )

    st.success(
        f"Evaluated **{selected_model_name}** on **{len(test_df)}** test rows "
        "(no retraining performed)."
    )

    st.subheader("Evaluation Metrics")
    metric_cols = st.columns(6)
    for col, (name, value) in zip(metric_cols, metrics.items()):
        col.metric(name, f"{value:.4f}")

    left, right = st.columns(2)
    with left:
        st.subheader("Confusion Matrix")
        fig = plot_confusion_matrix(cm, list(label_encoder.classes_), selected_model_name)
        st.pyplot(fig)
        plt.close(fig)

    with right:
        st.subheader("Classification Report")
        st.code(report, language="text")

    pred_labels = label_encoder.inverse_transform(y_pred)
    preview = test_df.copy()
    preview["Predicted_Class"] = pred_labels
    st.subheader("Prediction Preview")
    st.dataframe(preview.head(25), use_container_width=True)

    st.markdown("---")
    st.caption(
        "Models: Logistic Regression · Decision Tree · KNN · Gaussian Naive Bayes · SVM (RBF). "
        "Dataset: UCI Dry Bean (16 features, 7 classes)."
    )


if __name__ == "__main__":
    main()
