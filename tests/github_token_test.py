"""Standalone sanity check: confirms GITHUB_TOKEN in .env is valid. Not pytest -- just run directly."""
from pathlib import Path

from dotenv import load_dotenv
import os
import requests

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

token = os.getenv("GITHUB_TOKEN")

if not token:
    raise SystemExit("GITHUB_TOKEN is missing.")

response = requests.get(
    "https://api.github.com/rate_limit",
    headers={"Authorization": f"token {token}"},
    timeout=30,
)

print("Status:", response.status_code)

if response.ok:
    limit = response.json()["resources"]["core"]
    print(f"GitHub token is valid. Rate limit: {limit['remaining']}/{limit['limit']} remaining.")
else:
    print("GitHub token validation failed.")
    print(response.text[:500])
