import json
from pathlib import Path
import pandas as pd


MODELS = [
    "gpt-4o-mini",
    "mistral-nemo-12b",
    "mistral-7b",
    "llama-32-3b",
    "llama-32-1b",
]

BASE_DIR = Path("outputs")


def collect_scores(path, score_key):
    scores = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)

            for response in item["scored_responses"]:
                score = response.get(score_key)

                if score is not None:
                    scores.append(score)

    return scores


rows = []

for model in MODELS:

    bart_path = (
        BASE_DIR
        / model
        / "scores"
        / "bartscore"
        / f"{model}_scored.jsonl"
    )

    armorm_path = (
        BASE_DIR
        / model
        / "scores"
        / "armorm"
        / f"{model}_0_scored2.jsonl"
    )

    bart_scores = collect_scores(bart_path, "bart_score")
    armorm_scores = collect_scores(armorm_path, "armorm_score")

    rows.append({
        "model": model,
        "avg_bartscore": sum(bart_scores)/len(bart_scores),
        "avg_armorm": sum(armorm_scores)/len(armorm_scores),
        "num_bart_scores": len(bart_scores),
        "num_armorm_scores": len(armorm_scores),
    })


df = pd.DataFrame(rows)

print("\n=== OVERALL MODEL RANKING ===\n")

print(df.sort_values("avg_armorm", ascending=False))

print("\n=== SORTED BY BARTSCORE ===\n")

print(df.sort_values("avg_bartscore", ascending=False))