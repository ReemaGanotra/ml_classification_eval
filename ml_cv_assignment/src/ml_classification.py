"""
Task 1 – Adult Income Binary Classification
============================================
Dataset : UCI Adult Income (census data, ~48 842 rows)
Target  : income >50K  →  1 (high income) | <=50K  →  0 (low income)
Models  : Logistic Regression | Random Forest | Support Vector Machine
Strategy: Train all three, compare on held-out test set, persist the winner.
"""

import sys
import time
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                       # non-interactive backend (Windows-safe)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.svm             import SVC
from sklearn.pipeline        import Pipeline
from sklearn.preprocessing   import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics         import (
    classification_report, confusion_matrix,
    precision_recall_fscore_support, roc_auc_score, roc_curve,
)
from sklearn.inspection import permutation_importance

from utils import (
    set_seed, setup_logging, save_metrics,
    save_plot, set_style, DATA_DIR, MODELS_DIR,
)


SEED = 42

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 – Data Loading
# ══════════════════════════════════════════════════════════════════════════════

COLUMN_NAMES = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country",
    "income",
]

def load_data() -> pd.DataFrame:
    """
    Download and return the UCI Adult Income dataset.

    Dataset facts
    -------------
    Rows    : 48 842  (train 32 561 + test 16 281)
    Columns : 14 features + income label
    Target  : binarised  →  income >50K = 1 (high),  <=50K = 0 (low)
    Balance : ~76 % low income, ~24 % high income  →  mild class imbalance

    Citation
    --------
    Dua, D. & Graff, C. (2019). UCI Machine Learning Repository.
    https://archive.ics.uci.edu/ml/datasets/adult

    Returns
    -------
    pd.DataFrame with cleaned data and binary 'income_binary' column
    """
    train_path = DATA_DIR / "adult_train.csv"
    test_path  = DATA_DIR / "adult_test.csv"

    if not train_path.exists():
        import requests
        base = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult"
        for url, path in [
            (f"{base}/adult.data", train_path),
            (f"{base}/adult.test", test_path),
        ]:
            logging.info("Downloading %s …", url)
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            path.write_bytes(r.content)
            logging.info("Saved → %s", path)

    # UCI adult.data has no header; adult.test has a leading comment line
    train_df = pd.read_csv(train_path, header=None, names=COLUMN_NAMES,
                           skipinitialspace=True)
    test_df  = pd.read_csv(test_path,  header=None, names=COLUMN_NAMES,
                           skipinitialspace=True, skiprows=1)

    df = pd.concat([train_df, test_df], ignore_index=True)

    # Strip trailing dots from the test split's income column  ("<=50K." → "<=50K")
    df["income"] = df["income"].str.strip().str.rstrip(".")

    # Replace " ?" with NaN then drop
    df.replace("?", np.nan, inplace=True)
    missing_before = df.isnull().sum().sum()
    df.dropna(inplace=True)
    logging.info("Dropped %d rows with missing values (%.1f %%)",
                 missing_before // 3, 100 * missing_before / (df.shape[0] * df.shape[1]))

    # Binary target
    df["income_binary"] = (df["income"] == ">50K").astype(int)

    logging.info("Dataset ready  –  shape: %s", df.shape)
    logging.info("Class balance  –  >50K: %.1f %%", 100 * df["income_binary"].mean())
    return df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 – EDA
# ══════════════════════════════════════════════════════════════════════════════

def run_eda(df: pd.DataFrame) -> None:
    """
    Exploratory Data Analysis.

    Artefacts written to artifacts/
    --------------------------------
    correlation_heatmap.png     – Pearson correlation of numeric features
    feature_distributions.png  – histogram overlay (>50K vs <=50K)
    class_balance.png           – bar chart of target counts
    income_by_education.png     – income rate grouped by education level
    """
    logging.info("Running EDA …")

    numeric_cols = ["age", "fnlwgt", "education_num",
                    "capital_gain", "capital_loss", "hours_per_week"]

    logging.info("\n%s", df[numeric_cols + ["income_binary"]].describe().to_string())
    logging.info("Missing values: 0  (dropped during load)")

    counts = df["income_binary"].value_counts()
    logging.info("Class counts  –  <=50K: %d | >50K: %d", counts[0], counts[1])

    # ── 1. Correlation heatmap (numeric features only) ───────────────────────
    corr = df[numeric_cols + ["income_binary"]].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="coolwarm", center=0, square=True,
                linewidths=0.4, ax=ax)
    ax.set_title("Pearson Correlation – Numeric Features", fontsize=14, fontweight="bold")
    save_plot(fig, "correlation_heatmap.png")

    # ── 2. Distributions for key numeric features ────────────────────────────
    palette = {0: "steelblue", 1: "coral"}
    labels  = {0: "<=50K", 1: ">50K"}
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for ax, col in zip(axes.ravel(), numeric_cols):
        for cls, colour in palette.items():
            subset = df.loc[df["income_binary"] == cls, col]
            ax.hist(subset, bins=30, alpha=0.55, color=colour,
                    density=True, label=labels[cls])
        ax.set_xlabel(col.replace("_", " ").title(), fontsize=10)
        ax.set_ylabel("Density")
        ax.set_title(col.replace("_", " ").title())
        ax.legend(fontsize=9)
    plt.suptitle("Feature Distributions  –  >50K vs <=50K",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    save_plot(fig, "feature_distributions.png")

    # ── 3. Class balance bar ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(["<=50K  (0)", ">50K  (1)"],
                  [counts[0], counts[1]],
                  color=["steelblue", "coral"], edgecolor="white", width=0.5)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 80,
                f"{int(bar.get_height()):,}",
                ha="center", fontsize=11, fontweight="bold")
    ax.set_ylabel("Count")
    ax.set_title("Class Balance  –  income_binary")
    plt.tight_layout()
    save_plot(fig, "class_balance.png")

    # ── 4. Income rate by education level ────────────────────────────────────
    edu_order = (
        df.groupby("education")["income_binary"]
          .mean()
          .sort_values()
          .index.tolist()
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    income_by_edu = (df.groupby("education")["income_binary"]
                       .mean()
                       .loc[edu_order] * 100)
    income_by_edu.plot(kind="barh", ax=ax, color="steelblue", edgecolor="white")
    ax.set_xlabel("Income >50K Rate (%)")
    ax.set_title("Income Rate by Education Level")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    save_plot(fig, "income_by_education.png")

    logging.info("EDA complete – 4 plots saved to artifacts/")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 – Feature Engineering
# ══════════════════════════════════════════════════════════════════════════════

def engineer_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Encode and engineer features for modelling.

    Encoding
    --------
    Categorical columns  →  LabelEncoder (ordinal integers)
    All features         →  StandardScaler (inside Pipeline per model)

    Engineered features (4)
    -----------------------
    capital_net         = capital_gain – capital_loss
        Net capital movement; more informative than two separate sparse columns.

    hours_per_week_norm = hours_per_week / 40
        Normalises to a "full-time ratio"; part-time (<1) and overtime (>1)
        become easily comparable.

    age_edu_interaction = age × education_num
        Older individuals with more education tend to have higher incomes;
        the product captures this joint effect directly.

    is_married          = 1 if marital_status in {Married-civ-spouse,
                                                   Married-AF-spouse}
        Marriage strongly correlates with higher reported income
        (dual-income households, filing status).

    Returns
    -------
    X              : ndarray  (n, n_features)
    y              : ndarray  (n,)
    feature_names  : list[str]
    """
    d = df.copy()

    # Engineered features
    d["capital_net"]          = d["capital_gain"] - d["capital_loss"]
    d["hours_per_week_norm"]  = d["hours_per_week"] / 40.0
    d["age_edu_interaction"]  = d["age"] * d["education_num"]
    d["is_married"]           = d["marital_status"].isin(
                                    ["Married-civ-spouse", "Married-AF-spouse"]
                                ).astype(int)

    # Drop original columns that were combined or are redundant
    d.drop(columns=["capital_gain", "capital_loss",
                     "income", "fnlwgt"], inplace=True)

    # Label-encode all remaining categorical columns
    cat_cols = d.select_dtypes(include="object").columns.tolist()
    for col in cat_cols:
        d[col] = LabelEncoder().fit_transform(d[col].astype(str))

    feature_cols = [c for c in d.columns if c != "income_binary"]
    X = d[feature_cols].values.astype(np.float32)
    y = d["income_binary"].values

    logging.info("Feature matrix: %d samples × %d features", *X.shape)
    logging.info("Engineered: capital_net | hours_per_week_norm | "
                 "age_edu_interaction | is_married")
    return X, y, feature_cols


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 – Model Comparison
# ══════════════════════════════════════════════════════════════════════════════

def build_candidates() -> dict:
    """
    Return three sklearn Pipelines:
      StandardScaler  →  classifier

    Each Pipeline is self-contained so the same X (un-scaled) can be passed
    to all three without any data-leakage risk.

    Hyper-parameter rationale
    -------------------------
    LogisticRegression  C=1.0, max_iter=1000
        Default regularisation; enough iterations for convergence on ~40 k rows.

    RandomForestClassifier  n_estimators=200, max_depth=15
        Larger forest than default; depth cap prevents overfitting.

    SVC  C=1.0, kernel='rbf', probability=True
        RBF kernel handles non-linear boundaries; probability=True enables
        ROC-AUC computation. Scaled data is critical for SVM convergence.
    """
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(
                C=1.0, max_iter=1000,
                class_weight="balanced",
                random_state=SEED, n_jobs=-1,
            )),
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    RandomForestClassifier(
                n_estimators=200, max_depth=15,
                min_samples_leaf=4,
                class_weight="balanced",
                random_state=SEED, n_jobs=-1,
            )),
        ]),
        "Support Vector Machine": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    SVC(
                C=1.0, kernel="rbf",
                class_weight="balanced",
                probability=True,
                random_state=SEED,
            )),
        ]),
    }


def compare_models(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test:  np.ndarray, y_test:  np.ndarray,
) -> Tuple[dict, str]:
    """
    Train every candidate, evaluate on the test set, log a comparison table,
    produce a bar-chart artefact, and return the results + the winner name.

    Evaluation metric used for ranking: F1-Score (binary)
    Why F1? The dataset has mild class imbalance (~24 % positive);
    F1 balances precision and recall better than raw accuracy.

    Returns
    -------
    results  : dict  {model_name: {precision, recall, f1, roc_auc, fit_time_s}}
    best_name: str   name of the model with the highest test F1
    """
    candidates = build_candidates()
    results    = {}
    cv         = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    for name, pipeline in candidates.items():
        logging.info("─" * 50)
        logging.info("Training: %s", name)

        # CV on train set
        cv_f1 = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1")
        logging.info("  5-fold CV F1 : %.4f  (±%.4f)", cv_f1.mean(), cv_f1.std())

        # Full fit on train, evaluate on test
        t0 = time.perf_counter()
        pipeline.fit(X_train, y_train)
        fit_time = time.perf_counter() - t0

        y_pred  = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary")
        auc         = roc_auc_score(y_test, y_proba)

        results[name] = dict(
            precision=float(p), recall=float(r),
            f1=float(f1), roc_auc=float(auc),
            fit_time_s=round(fit_time, 2),
        )
        logging.info("  Precision=%0.4f  Recall=%0.4f  F1=%0.4f  AUC=%0.4f  "
                     "Time=%.1fs", p, r, f1, auc, fit_time)

    # ── comparison table in log ───────────────────────────────────────────────
    logging.info("\n%s", "=" * 65)
    logging.info("  %-28s  %7s  %7s  %7s  %7s",
                 "Model", "Prec", "Recall", "F1", "AUC")
    logging.info("  %s", "-" * 60)
    for name, m in results.items():
        logging.info("  %-28s  %7.4f  %7.4f  %7.4f  %7.4f",
                     name, m["precision"], m["recall"], m["f1"], m["roc_auc"])
    logging.info("%s", "=" * 65)

    save_metrics(results, "model_comparison.json")

    # ── bar chart: F1 per model ───────────────────────────────────────────────
    names  = list(results.keys())
    f1s    = [results[n]["f1"]     for n in names]
    aucs   = [results[n]["roc_auc"] for n in names]
    x      = np.arange(len(names))
    width  = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, f1s,  width, label="F1-Score",  color="steelblue")
    bars2 = ax.bar(x + width / 2, aucs, width, label="ROC-AUC",   color="coral")
    for bar in bars1 + bars2:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{bar.get_height():.3f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison  –  F1 & ROC-AUC on Test Set")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    save_plot(fig, "model_comparison.png")

    best_name = max(results, key=lambda n: results[n]["f1"])
    logging.info("Best model by F1: '%s'  (F1 = %.4f)",
                 best_name, results[best_name]["f1"])
    return results, best_name


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 – Final Model Evaluation
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_final_model(
    pipeline,
    best_name: str,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list,
) -> dict:
    """
    Deep evaluation of the winning model.

    Artefacts written to artifacts/
    --------------------------------
    confusion_matrix.png
    roc_curve.png
    feature_importance.png          (permutation importance)
    ml_classification_metrics.json  (final metrics for deployment.py to read)

    Returns
    -------
    dict  {precision, recall, f1_score, roc_auc, test_accuracy}
    """
    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary"
    )
    roc_auc  = roc_auc_score(y_test, y_proba)
    accuracy = (y_pred == y_test).mean()

    metrics = dict(
        model=best_name,
        precision=float(precision), recall=float(recall),
        f1_score=float(f1), roc_auc=float(roc_auc),
        test_accuracy=float(accuracy),
    )

    logging.info("\n%s", "═" * 50)
    logging.info("  FINAL MODEL : %s", best_name)
    logging.info("  Precision   : %.4f", precision)
    logging.info("  Recall      : %.4f", recall)
    logging.info("  F1-Score    : %.4f", f1)
    logging.info("  ROC-AUC     : %.4f", roc_auc)
    logging.info("  Accuracy    : %.4f", accuracy)
    logging.info("%s", "═" * 50)
    logging.info("\n%s",
        classification_report(y_test, y_pred, target_names=["<=50K", ">50K"]))

    save_metrics(metrics, "ml_classification_metrics.json")

    # ── confusion matrix ─────────────────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["<=50K", ">50K"],
                yticklabels=["<=50K", ">50K"])
    for label, (r, c) in zip(["TN", "FP", "FN", "TP"],
                              [(0,0),(0,1),(1,0),(1,1)]):
        ax.text(c + 0.5, r + 0.75, label,
                ha="center", va="center", fontsize=9, color="dimgray")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix  –  {best_name}")
    save_plot(fig, "confusion_matrix.png")

    # ── ROC curve ────────────────────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, lw=2, color="coral",
            label=f"{best_name}  (AUC = {roc_auc:.3f})")
    ax.fill_between(fpr, tpr, alpha=0.08, color="coral")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    save_plot(fig, "roc_curve.png")

    # ── permutation importance (uses the scaler-wrapped pipeline) ────────────
    logging.info("Computing permutation importance …")

    # Extract scaled test data for permutation importance
    X_test_scaled = pipeline.named_steps["scaler"].transform(X_test)
    clf           = pipeline.named_steps["clf"]

    perm = permutation_importance(
        clf, X_test_scaled, y_test,
        n_repeats=10, random_state=SEED, n_jobs=-1, scoring="f1",
    )
    top_n   = min(15, len(feature_names))
    indices = np.argsort(perm.importances_mean)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(range(top_n), perm.importances_mean[indices],
            xerr=perm.importances_std[indices],
            align="center", color="steelblue", ecolor="grey")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.invert_yaxis()
    ax.set_xlabel("Mean decrease in F1  (permutation importance)")
    ax.set_title(f"Feature Importance – {best_name}  (Top {top_n})")
    plt.tight_layout()
    save_plot(fig, "feature_importance.png")

    logging.info("Top 5 features:")
    for rank, idx in enumerate(indices[:5], 1):
        logging.info("  %d. %-28s %.4f",
                     rank, feature_names[idx], perm.importances_mean[idx])

    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 – Persistence
# ══════════════════════════════════════════════════════════════════════════════

def save_best_model(pipeline, feature_names: list) -> None:
    """
    Persist the winning Pipeline (scaler + classifier together).

    Files written
    -------------
    models/best_model.pkl          – the full sklearn Pipeline
    models/feature_names.json      – ordered feature list for deployment.py

    Why save the whole Pipeline?
    The Pipeline bundles StandardScaler + classifier so deployment.py can call
    pipeline.predict(raw_X) without re-applying the scaler manually, eliminating
    a common source of train/serve skew.
    """
    joblib.dump(pipeline, MODELS_DIR / "best_model.pkl")
    import json
    (MODELS_DIR / "feature_names.json").write_text(
        json.dumps(feature_names, indent=2)
    )
    logging.info("Pipeline saved  →  models/best_model.pkl")
    logging.info("Feature list saved  →  models/feature_names.json")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline() -> dict:
    """
    Execute all stages in order:
        load → EDA → engineer → split → compare → evaluate winner → save
    """
    set_seed(SEED)
    setup_logging("ml_classification.log")
    set_style()

    logging.info("=" * 60)
    logging.info("TASK 1  –  ADULT INCOME CLASSIFICATION")
    logging.info("=" * 60)

    # 1. Load
    df = load_data()

    # 2. EDA
    run_eda(df)

    # 3. Feature engineering
    X, y, feature_names = engineer_features(df)

    # 4. Stratified train / test split (80 / 20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )
    logging.info("Split  –  Train: %s  |  Test: %s",
                 X_train.shape, X_test.shape)

    # 5. Compare all three models
    candidates  = build_candidates()
    results, best_name = compare_models(X_train, y_train, X_test, y_test)

    # 6. Re-fit the winner on full training data (best practice before deploy)
    best_pipeline = candidates[best_name]           # fresh, unfitted instance
    logging.info("Re-fitting '%s' on full training data …", best_name)
    best_pipeline.fit(X_train, y_train)

    # 7. Deep evaluation of the final model
    metrics = evaluate_final_model(
        best_pipeline, best_name, X_test, y_test, feature_names
    )

    # 8. Save
    save_best_model(best_pipeline, feature_names)

    logging.info("=" * 60)
    logging.info("TASK 1 COMPLETE  –  Best model: %s", best_name)
    logging.info("=" * 60)
    return metrics


if __name__ == "__main__":
    metrics = run_pipeline()

    print("\n" + "=" * 55)
    print("ADULT INCOME  –  FINAL RESULTS")
    print("=" * 55)
    print(f"  Model       : {metrics['model']}")
    print(f"  Precision   : {metrics['precision']:.4f}")
    print(f"  Recall      : {metrics['recall']:.4f}")
    print(f"  F1-Score    : {metrics['f1_score']:.4f}")
    print(f"  ROC-AUC     : {metrics['roc_auc']:.4f}")
    print(f"  Accuracy    : {metrics['test_accuracy']:.4f}")
    print("=" * 55)
