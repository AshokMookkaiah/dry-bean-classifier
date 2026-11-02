"""
BITS WILP M.Tech (AIML) - Machine Learning Assignment 2
Train five classifiers on the UCI Dry Bean dataset, evaluate metrics,
persist models/artifacts, and export a held-out test CSV for Streamlit.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "dry_bean_full.csv"
MODEL_DIR = ROOT / "model"
ARTIFACT_DIR = ROOT / "artifacts"
TEST_DATA_PATH = ROOT / "test_data.csv"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"
REPORT_PATH = ARTIFACT_DIR / "classification_reports.json"
CM_DIR = ARTIFACT_DIR / "confusion_matrices"

RANDOM_STATE = 42
TEST_SIZE = 0.20


def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Run download_dataset.py first."
        )
    df = pd.read_csv(DATA_PATH)
    # Normalize target column name if needed
    if "Class" not in df.columns:
        df = df.rename(columns={df.columns[-1]: "Class"})
    df["Class"] = df["Class"].astype(str).str.strip()
    return df


def build_models() -> dict:
    # Scale-sensitive models use a Pipeline with StandardScaler.
    return {
        "Logistic Regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        solver="lbfgs",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Decision Tree": DecisionTreeClassifier(
            criterion="gini",
            max_depth=18,
            min_samples_leaf=4,
            random_state=RANDOM_STATE,
        ),
        "KNN": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("clf", KNeighborsClassifier(n_neighbors=7, weights="distance")),
            ]
        ),
        "Naive Bayes": GaussianNB(),
        "Support Vector Machine": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    SVC(
                        kernel="rbf",
                        C=2.0,
                        gamma="scale",
                        probability=True,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def compute_metrics(y_true, y_pred, y_proba, label_encoder: LabelEncoder) -> dict:
    classes = label_encoder.classes_
    metrics = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "Recall": float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "F1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
    }

    # Multi-class AUC with one-vs-rest using probability estimates
    try:
        metrics["AUC"] = float(
            roc_auc_score(
                y_true,
                y_proba,
                multi_class="ovr",
                average="weighted",
                labels=list(range(len(classes))),
            )
        )
    except ValueError:
        metrics["AUC"] = float("nan")

    return metrics


def save_confusion_matrix(y_true, y_pred, labels, model_name: str) -> None:
    CM_DIR.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix - {model_name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    safe_name = model_name.lower().replace(" ", "_")
    fig.savefig(CM_DIR / f"{safe_name}_cm.png", dpi=140)
    plt.close(fig)


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    CM_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset()
    feature_cols = [c for c in df.columns if c != "Class"]
    X = df[feature_cols]
    y_raw = df["Class"]

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Persist held-out test CSV with original class labels for Streamlit upload
    test_export = X_test.copy()
    test_export["Class"] = label_encoder.inverse_transform(y_test)
    test_export.to_csv(TEST_DATA_PATH, index=False)

    models = build_models()
    all_metrics: dict[str, dict] = {}
    all_reports: dict[str, dict] = {}

    print("=" * 72)
    print("Dry Bean Classification - Training & Evaluation")
    print(f"Instances: {len(df)} | Features: {len(feature_cols)} | Classes: {len(label_encoder.classes_)}")
    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
    print("=" * 72)

    for name, model in models.items():
        print(f"\nTraining: {name}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)
        else:
            # Should not happen for configured models; keep a safe fallback
            n_classes = len(label_encoder.classes_)
            y_proba = np.zeros((len(y_test), n_classes))
            y_proba[np.arange(len(y_test)), y_pred] = 1.0

        metrics = compute_metrics(y_test, y_pred, y_proba, label_encoder)
        all_metrics[name] = metrics

        report = classification_report(
            y_test,
            y_pred,
            target_names=list(label_encoder.classes_),
            output_dict=True,
            zero_division=0,
        )
        all_reports[name] = report

        save_confusion_matrix(
            y_test, y_pred, list(label_encoder.classes_), name
        )

        model_path = MODEL_DIR / f"{name.lower().replace(' ', '_')}.joblib"
        joblib.dump(model, model_path)
        print(f"  Saved model -> {model_path.name}")
        print(
            "  Metrics | "
            + " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
        )

    # Shared preprocessing artifacts for the Streamlit app
    joblib.dump(label_encoder, MODEL_DIR / "label_encoder.joblib")
    joblib.dump(feature_cols, MODEL_DIR / "feature_columns.joblib")

    with METRICS_PATH.open("w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    with REPORT_PATH.open("w", encoding="utf-8") as f:
        json.dump(all_reports, f, indent=2)

    # Pretty comparison table for console / README drafting
    table = pd.DataFrame(all_metrics).T[
        ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    ]
    print("\n" + "=" * 72)
    print("Comparison Table")
    print(table.round(4).to_string())
    print("=" * 72)
    print(f"\nWrote test_data.csv ({len(test_export)} rows)")
    print(f"Wrote metrics -> {METRICS_PATH}")
    print("Training complete.")


if __name__ == "__main__":
    main()
