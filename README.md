# GitHub Customer Churn Predictor

Dockerized FastAPI application that predicts whether a GitHub user is likely to churn based on public activity and engagement signals.

The project includes data collection from the GitHub REST API, feature engineering, exploratory data analysis, feature selection, model training, model comparison, and API deployment using Docker.

## Project Structure

- app/: FastAPI source code, feature logic, model logic, scraper, and trained model
- data/raw/: raw dataset, generated features, feature selection results, metrics, and model comparison
- notebooks/: EDA and feature selection notebook
- Dockerfile: container image definition
- docker-compose.yml: Docker Compose execution file
- train.py: training pipeline
- report.md: short project report

## Data Source

The project uses the public GitHub REST API. The scraper collects public user and organization profile data discovered from popular repositories and seed queries.

The collected dataset contains 250 GitHub profiles and is saved in:

data/raw/github_users.csv

A GitHub Personal Access Token can be used through the GITHUB_TOKEN environment variable to avoid unauthenticated API rate limits. The token is not stored in the source code.

Example:

export GITHUB_TOKEN="YOUR_TOKEN_HERE"

## Churn Definition

A user is labeled as churned when:

days_since_last_activity > 90

This means the user has not shown account activity for more than 90 days.

Current generated class balance:

- Active users: 157
- Churned users: 93

## Feature Engineering

The project creates 14 features from the raw GitHub data.

Time-based features:
- account_age_days
- days_since_last_activity

Ratio features:
- follower_ratio
- followers_per_repo
- following_per_repo

Aggregation features:
- repos_per_year
- gists_per_year
- engagement_score

Binary features:
- has_no_repos
- has_company
- has_blog
- has_location
- is_hireable
- is_organization

Generated features are saved in:

data/raw/github_users_features.csv

## Exploratory Data Analysis

The EDA and feature selection notebook is located at:

notebooks/eda_and_selection.ipynb

It includes raw dataset loading, feature generation, churn label distribution, descriptive statistics, variance threshold analysis, correlation matrix, feature selection comparison, model comparison, and final conclusions.

## Feature Selection

Four feature selection approaches are applied:

1. Filter method using ANOVA F-test
2. Wrapper method using Recursive Feature Elimination
3. Decision Tree feature importance
4. Random Forest feature importance

The notebook also includes additional filter analysis through variance threshold and correlation matrix.

The comparison table is saved in:

data/raw/feature_selection_comparison.csv

Important selected features include:

- days_since_last_activity
- is_organization
- following_per_repo
- is_hireable

## Model Training

The final production model is a Random Forest Classifier saved in:

app/model.pkl

The API uses the following operational features:

- days_since_last_activity
- repos_per_year
- follower_ratio
- followers_per_repo
- engagement_score

Training metrics are saved in:

data/raw/model_metrics.csv

## Model Comparison

Because days_since_last_activity is directly related to the churn definition, a second model was trained without that feature. This avoids relying only on the same variable used to define churn and provides a more realistic comparison.

The comparison is saved in:

data/raw/model_comparison.csv

Results:

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| With recency | 1.00 | 1.00 | 1.00 | 1.00 |
| Without days_since_last_activity | 0.60 | 0.46 | 0.57 | 0.51 |

The model with recency performs perfectly because the recency feature is strongly tied to the churn label. The model without recency performs lower but more realistically, showing that indirect engagement and profile signals still provide predictive value.

## API Endpoints

Health check:

curl http://127.0.0.1:8000/health

Feature list:

curl http://127.0.0.1:8000/features

Prediction:

curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{"days_since_last_activity":120,"repos_per_year":1.5,"follower_ratio":0.4,"followers_per_repo":1.0,"engagement_score":30}'

Example response:

{"churned":true,"churn_probability":0.92}

## Run with Docker

Build and start the API:

sudo docker compose up -d --build

Test the API:

curl http://127.0.0.1:8000/health

curl http://127.0.0.1:8000/features

curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{"days_since_last_activity":120,"repos_per_year":1.5,"follower_ratio":0.4,"followers_per_repo":1.0,"engagement_score":30}'

Stop the API:

sudo docker compose down

The API runs at:

http://127.0.0.1:8000

The compose file uses network_mode: host to avoid local Docker proxy issues in this development environment.

## Run Locally

Create and activate a virtual environment:

python3 -m venv .venv
source .venv/bin/activate

Install dependencies:

pip install -r requirements.txt

Train the model:

python train.py

Run the API locally:

uvicorn app.main:app --reload

## Report

A short project report is included in:

report.md

It explains the data source, churn definition, feature engineering, feature selection results, model comparison, retention strategy, ethical considerations, and project limitations.

## Security Notes

Do not commit GitHub tokens or .env files.

Before submitting or pushing to GitHub, check:

grep -R --exclude-dir=.venv "ghp_" .

This command should return no results.
