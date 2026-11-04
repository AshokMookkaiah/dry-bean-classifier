# Dry Bean Variety Classification — ML Assignment 2

BITS Pilani WILP | M.Tech (AIML) | Machine Learning Assignment - 2

## a. Problem statement

Dry bean seed varieties look similar to the human eye, but commercial sorting needs consistent automatic classification. This project builds five supervised machine learning classifiers that map **16 geometric / shape features** extracted from bean images to one of **seven registered dry bean varieties**.

The end-to-end workflow covers:

1. Dataset selection and preprocessing
2. Training five classical classifiers on the same data split
3. Computing Accuracy, AUC, Precision, Recall, F1, and Matthews Correlation Coefficient (MCC)
4. Saving trained models (`.joblib`)
5. Serving an interactive Streamlit app that evaluates uploaded **test** CSV data **without retraining**
6. Deploying the app on Streamlit Community Cloud

## b. Dataset description

| Item | Details |
|---|---|
| Dataset name | **Dry Bean Dataset** |
| Source | [UCI Machine Learning Repository — Dry Bean (id=602)](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset) |
| Task type | Multi-class classification |
| Instances | **13,611** (meets minimum ≥ 500) |
| Features | **16** numeric geometric / shape descriptors (meets minimum ≥ 12) |
| Target column | `Class` |
| Classes (7) | BARBUNYA, BOMBAY, CALI, DERMASON, HOROZ, SEKER, SIRA |

**Feature list:** Area, Perimeter, MajorAxisLength, MinorAxisLength, AspectRatio, Eccentricity, ConvexArea, EquivDiameter, Extent, Solidity, Roundness, Compactness, ShapeFactor1, ShapeFactor2, ShapeFactor3, ShapeFactor4.

**Train / test split used in experiments:** stratified 80 / 20 split (`random_state=42`) → 10,888 train rows and **2,723 test rows**. The held-out test set is saved as `test_data.csv` for Streamlit upload and reproducibility.

**Class distribution (full dataset):**

| Class | Count |
|---|---|
| DERMASON | 3546 |
| SIRA | 2636 |
| SEKER | 2027 |
| HOROZ | 1928 |
| CALI | 1630 |
| BARBUNYA | 1322 |
| BOMBAY | 522 |

## c. Github Repository Link

> **Replace this line with your public GitHub repository URL after you push the code.**  
> Example format: `https://github.com/<your-username>/dry-bean-classifier`

Live Streamlit app (after deployment):

> **Replace this line with your Streamlit Community Cloud URL.**  
> Example format: `https://<your-app-name>.streamlit.app`

## d. Models used

All five models were trained on the **same** Dry Bean dataset and evaluated on the **same held-out test set**. Scale-sensitive models (Logistic Regression, KNN, SVM) use a `StandardScaler` inside a scikit-learn `Pipeline`. AUC is computed with the **one-vs-rest, weighted** strategy suitable for multi-class problems.

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9207 | 0.9934 | 0.9215 | 0.9207 | 0.9209 | 0.9042 |
| Decision Tree | 0.9045 | 0.9603 | 0.9044 | 0.9045 | 0.9041 | 0.8846 |
| KNN | 0.9166 | 0.9839 | 0.9173 | 0.9166 | 0.9168 | 0.8992 |
| Naive Bayes | 0.7639 | 0.9644 | 0.7654 | 0.7639 | 0.7615 | 0.7154 |
| Support Vector Machine | 0.9254 | 0.9936 | 0.9255 | 0.9254 | 0.9255 | 0.9099 |

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Strong linear baseline after feature scaling. Accuracy ≈ 92.1% with very high AUC (0.9934). Precision/Recall/F1 stay tightly aligned, showing balanced errors across the seven classes. |
| Decision Tree | Interpretable non-linear model with Accuracy ≈ 90.5%. Slightly lower AUC than the scaled linear / kernel models, which is expected because a single tree can overfit local partitions and has weaker probability calibration. |
| KNN | Distance-based classifier (k=7, distance weighting) reaches Accuracy ≈ 91.7% after scaling. Competitive with Logistic Regression; sensitive to feature scale, which the pipeline handles. |
| Naive Bayes | Gaussian Naive Bayes is the weakest performer (Accuracy ≈ 76.4%). The strong feature correlations among geometric descriptors violate the conditional-independence assumption, so the generative model underfits relative to discriminative methods. AUC remains decent, but thresholded decisions are poorer (lowest MCC). |
| Support Vector Machine | Best overall model on this dataset. RBF-kernel SVM achieves the highest Accuracy (0.9254), F1 (0.9255), MCC (0.9099), and AUC (0.9936). The kernel boundary captures non-linear class separation that a pure linear model only approximates. |
| **Overall winner for this dataset?** | **Support Vector Machine (RBF)** — highest Accuracy, F1, MCC, and AUC on the held-out test set. |

---


