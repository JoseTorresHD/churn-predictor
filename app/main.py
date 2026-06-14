import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel


MODEL_PATH = "model.pkl"

app = FastAPI(title="GitHub Churn Predictor API")
model = joblib.load(MODEL_PATH)


FINAL_FEATURES = [
    "days_since_last_activity",
    "repos_per_year",
    "follower_ratio",
    "followers_per_repo",
    "engagement_score",
]


class GitHubUserFeatures(BaseModel):
    days_since_last_activity: float
    repos_per_year: float
    follower_ratio: float
    followers_per_repo: float
    engagement_score: float


@app.get("/")
def root():
    return {"message": "GitHub Churn Predictor API"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/features")
def features():
    return {"features": FINAL_FEATURES}


@app.post("/predict")
def predict_churn(user: GitHubUserFeatures):
    X = np.array([[
        user.days_since_last_activity,
        user.repos_per_year,
        user.follower_ratio,
        user.followers_per_repo,
        user.engagement_score,
    ]])

    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0][1]

    return {
        "churned": bool(prediction),
        "churn_probability": round(float(probability), 3)
    }
