import os
import time
import requests
import pandas as pd


SEED_QUERIES = [
    "language:python stars:>5000",
    "language:javascript stars:>5000",
    "language:typescript stars:>5000",
    "language:go stars:>5000",
    "language:rust stars:>3000",
    "language:java stars:>5000",
    "language:php stars:>3000",
    "language:ruby stars:>3000",
    "topic:machine-learning stars:>3000",
    "topic:web-framework stars:>3000",
]


FALLBACK_USERS = [
    "torvalds", "gaearon", "addyosmani", "yyx990803", "sindresorhus",
    "tj", "kennethreitz", "mojombo", "defunkt", "dhh",
    "matz", "antirez", "mitsuhiko", "tiangolo", "fastapi",
    "microsoft", "google", "facebook", "vercel", "nodejs",
    "rust-lang", "golang", "kubernetes", "docker", "elastic",
    "grafana", "prometheus", "ansible", "hashicorp", "laravel",
    "symfony", "rails", "spring-projects", "nestjs", "electron",
    "webpack", "babel", "vitejs", "openai", "redis",
    "mozilla", "canonical", "jetbrains", "cloudflare", "stripe",
    "octocat", "basecamp", "getsentry", "homebrew", "bitcoin",
    "ethereum", "metabase", "supabase", "denoland", "nuxt",
    "tailwindlabs", "alpinejs", "prettier", "eslint", "typicode",
    "nlohmann", "opencv", "godotengine", "bevyengine", "pytorch",
    "tensorflow", "huggingface", "mlflow"
]


def get_headers():
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "github-churn-predictor-project"
    }

    token = os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    return headers


def discover_users_from_repos(target_users=250):
    headers = get_headers()
    usernames = []

    for query in SEED_QUERIES:
        if len(usernames) >= target_users:
            break

        print(f"Searching repos: {query}")

        for page in range(1, 4):
            if len(usernames) >= target_users:
                break

            url = "https://api.github.com/search/repositories"
            params = {
                "q": query,
                "sort": "stars",
                "order": "desc",
                "per_page": 50,
                "page": page
            }

            response = requests.get(url, headers=headers, params=params, timeout=20)

            if response.status_code != 200:
                print(f"Repo search skipped: HTTP {response.status_code} - {response.text[:120]}")
                break

            data = response.json()

            for item in data.get("items", []):
                owner = item.get("owner", {}).get("login")
                if owner and owner not in usernames:
                    usernames.append(owner)

                if len(usernames) >= target_users:
                    break

            time.sleep(1)

    for user in FALLBACK_USERS:
        if len(usernames) >= target_users:
            break
        if user not in usernames:
            usernames.append(user)

    return usernames[:target_users]


def fetch_github_users(usernames=None, output_path="data/raw/github_users.csv", target_users=250):
    headers = get_headers()

    if usernames is None:
        usernames = discover_users_from_repos(target_users=target_users)

    records = []
    seen = set()

    for idx, username in enumerate(usernames, start=1):
        if username in seen:
            continue

        seen.add(username)
        url = f"https://api.github.com/users/{username}"

        try:
            response = requests.get(url, headers=headers, timeout=20)

            if response.status_code != 200:
                print(f"Skipping {username}: HTTP {response.status_code}")
                continue

            data = response.json()

            records.append({
                "username": username,
                "public_repos": data.get("public_repos", 0),
                "public_gists": data.get("public_gists", 0),
                "followers": data.get("followers", 0),
                "following": data.get("following", 0),
                "created_at": data.get("created_at"),
                "updated_at": data.get("updated_at"),
                "hireable": data.get("hireable"),
                "company": data.get("company"),
                "blog": data.get("blog"),
                "location": data.get("location"),
                "type": data.get("type"),
                "site_admin": data.get("site_admin", False),
            })

            print(f"Fetched {idx}/{len(usernames)}: {username}")
            time.sleep(0.2)

        except Exception as e:
            print(f"Error fetching {username}: {e}")

    df = pd.DataFrame(records)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Saved {len(df)} users to {output_path}")
    return df


if __name__ == "__main__":
    df = fetch_github_users(target_users=250)
    print(df.head())
    print(f"Rows: {len(df)}")
