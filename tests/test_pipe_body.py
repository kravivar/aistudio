import json

body = {
  "model": "gpt-4",
  "seed": 42,
  "options": {"seed": 99}
}

seed = body.get("seed") or body.get("options", {}).get("seed")
print(f"Seed is {seed}")
