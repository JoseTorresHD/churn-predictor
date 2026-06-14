import os
import pandas as pd

from app.scraper import fetch_github_users
from app.features import generate_features
from app.model import train_and_save_model, run_feature_selection, compare_models


def main():
    raw_path = "data/raw/github_users.csv"

    if os.path.exists(raw_path):
        print(f"Using existing data from {raw_path}")
        raw_df = pd.read_csv(raw_path)
    else:
        print("Fetching GitHub data...")
        raw_df = fetch_github_users(output_path=raw_path, target_users=250)

    print("\nGenerating features...")
    df = generate_features(raw_df)
    df.to_csv("data/raw/github_users_features.csv", index=False)

    print("\nClass balance:")
    print(df["churned"].value_counts())

    if df["churned"].nunique() < 2:
        raise ValueError("Dataset has only one class. Adjust churn threshold or collect more users.")

    print("\nRunning feature selection...")
    selection = run_feature_selection(df)
    selection.to_csv("data/raw/feature_selection_comparison.csv", index=False)
    print("\nFeature selection comparison:")
    print(selection)

    print("\nTraining final model...")
    model, metrics = train_and_save_model(df)
    pd.DataFrame([metrics]).to_csv("data/raw/model_metrics.csv", index=False)

    print("\nComparing model variants...")
    compare_models(df, output_path="data/raw/model_comparison.csv")


if __name__ == "__main__":
    main()
