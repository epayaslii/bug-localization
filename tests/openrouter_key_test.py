"""Standalone sanity check: confirms OPENROUTER_API_KEY in .env is valid. Not pytest -- just run directly."""
from pathlib import Path

from dotenv import load_dotenv
import os
import requests

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

key = os.getenv("OPENROUTER_API_KEY")

if not key:
    raise SystemExit("OPENROUTER_API_KEY is missing.")

response = requests.get(
    "https://openrouter.ai/api/v1/key",
    headers={"Authorization": f"Bearer {key}"},
    timeout=30,
)

print("Status:", response.status_code)

if response.ok:
    print("OpenRouter key is valid.")
else:
    print("OpenRouter validation failed.")
    print(response.text[:500])
