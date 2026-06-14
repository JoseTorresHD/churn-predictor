import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from app.features import FEATURE_COLUMNS


FINAL_FEATURES = [
    "days_since_last_activity",
    "repos_per_year",
    "follower_ratio",
    "followers_per_repo",
    "engagement_score",
]


FINAL_FEATURES_NO_RECENCY = [
    "repos_per_year",
    "follower_ratio",
    "followers_per_repo",
    "following_per_repo",
    "engagement_score",
    "is_organization",
    "has_company",
    "is_hireable",
]


def run_feature_selection(df):
    X = df[FEATURE_COLUMNS]
    y = df["churned"]

    results = pd.DataFrame({"feature": FEATURE_COLUMNS})

    filter_model = SelectKBest(score_func=f_classif, k="all")
    filter_model.fit(X, y)
    results["filter_score"] = filter_model.scores_
    results["filter_score"] = results["filter_score"].replace([float("inf"), float("-inf")], 0).fillna(0)
    results["filter_rank"] = results["filter_score"].rank(ascending=False, method="dense").astype(int)

    lr = LogisticRegression(max_iter=2000, class_weight="balanced")
    rfe = RFE(estimator=lr, n_features_to_select=5)
    rfe.fit(X, y)
    results["rfe_selected"] = rfe.support_
    results["rfe_rank"] = rfe.ranking_

    dt = DecisionTreeClassifier(random_state=42, class_weight="balanced")
    dt.fit(X, y)
    results["decision_tree_importance"] = dt.feature_importances_
    results["decision_tree_rank"] = results["decision_tree_importance"].rank(ascending=False, method="dense").astype(int)

    rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
    rf.fit(X, y)
    results["random_forest_importance"] = rf.feature_importances_
    results["random_forest_rank"] = results["random_forest_importance"].rank(ascending=False, method="dense").astype(int)

    votes = []
    for _, row in results.iterrows():
        vote_count = 0

        if row["filter_rank"] <= 5:
            vote_count += 1
        if row["rfe_selected"]:
            vote_count += 1
        if row["decision_tree_rank"] <= 5:
            vote_count += 1
        if row["random_forest_rank"] <= 5:
            vote_count += 1

        if vote_count >= 3:
            votes.append("Keep")
        elif vote_count == 2:
            votes.append("Optional")
        else:
            votes.append("Drop")

    results["decision"] = votes

    decision_order = {"Keep": 0, "Optional": 1, "Drop": 2}
    results["decision_order"] = results["decision"].map(decision_order)
    results = results.sort_values(["decision_order", "filter_rank"]).drop(columns=["decision_order"])

    return results


def evaluate_model(df, features, model_path=None):
    X = df[features]
    y = df["churned"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
    }

    report = classification_report(y_test, predictions, zero_division=0)

    if model_path:
        joblib.dump(model, model_path)

    return model, metrics, report


def train_and_save_model(df, model_path="app/model.pkl"):
    model, metrics, report = evaluate_model(df, FINAL_FEATURES, model_path=model_path)

    print("\nClassification report:")
    print(report)

    print("\nMetrics:")
    print(metrics)

    print(f"\nModel saved to {model_path}")
    return model, metrics


def compare_models(df, output_path="data/raw/model_comparison.csv"):
    _, metrics_main, report_main = evaluate_model(df, FINAL_FEATURES)
    _, metrics_no_recency, report_no_recency = evaluate_model(df, FINAL_FEATURES_NO_RECENCY)

    comparison = pd.DataFrame([
        {
            "model": "with_recency",
            "features": ", ".join(FINAL_FEATURES),
            **metrics_main
        },
        {
            "model": "without_days_since_last_activity",
            "features": ", ".join(FINAL_FEATURES_NO_RECENCY),
            **metrics_no_recency
        }
    ])

    comparison.to_csv(output_path, index=False)

    print("\nModel comparison:")
    print(comparison)

    print("\nReport - with recency:")
    print(report_main)

    print("\nReport - without days_since_last_activity:")
    print(report_no_recency)

    return comparison
