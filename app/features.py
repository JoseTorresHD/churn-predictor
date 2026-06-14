from datetime import datetime, timezone
import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "account_age_days",
    "days_since_last_activity",
    "repos_per_year",
    "gists_per_year",
    "follower_ratio",
    "followers_per_repo",
    "following_per_repo",
    "has_no_repos",
    "has_company",
    "has_blog",
    "has_location",
    "is_hireable",
    "is_organization",
    "engagement_score",
]


def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    now = datetime.now(timezone.utc)

    df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
    df["updated_at"] = pd.to_datetime(df["updated_at"], utc=True, errors="coerce")

    df["account_age_days"] = (now - df["created_at"]).dt.days
    df["days_since_last_activity"] = (now - df["updated_at"]).dt.days
    df["account_age_years"] = df["account_age_days"] / 365.25

    # Aggregation features
    df["repos_per_year"] = df["public_repos"] / (df["account_age_years"] + 1)
    df["gists_per_year"] = df["public_gists"] / (df["account_age_years"] + 1)

    # Ratio features
    df["follower_ratio"] = df["followers"] / (df["following"] + 1)
    df["followers_per_repo"] = df["followers"] / (df["public_repos"] + 1)
    df["following_per_repo"] = df["following"] / (df["public_repos"] + 1)

    # Binary features
    df["has_no_repos"] = (df["public_repos"] == 0).astype(int)
    df["has_company"] = df["company"].notna().astype(int)
    df["has_blog"] = df["blog"].fillna("").astype(str).str.len().gt(0).astype(int)
    df["has_location"] = df["location"].notna().astype(int)
    df["is_hireable"] = df["hireable"].fillna(False).astype(bool).astype(int)
    df["is_organization"] = (df["type"] == "Organization").astype(int)

    # Combined engagement feature
    df["engagement_score"] = (
        df["public_repos"] * 2
        + df["public_gists"]
        + df["followers"] * 0.5
        + df["following"] * 0.2
    )

    # Churn label: inactive for more than 90 days
    df["churned"] = (df["days_since_last_activity"] > 90).astype(int)

    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].replace([np.inf, -np.inf], 0)
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].fillna(0)

    return df
