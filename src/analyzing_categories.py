import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ============================================================
# CONFIG
# ============================================================

MODELS = [
    "gpt-4o-mini",
    "mistral-nemo-12b",
    "mistral-7b",
    "llama-32-3b",
    "llama-32-1b",
]

BASE_DIR = Path("outputs")

DIMENSION_NAMES = [
    "helpsteer-helpfulness",
    "helpsteer-correctness",
    "helpsteer-coherence",
    "helpsteer-complexity",
    "helpsteer-verbosity",
    "ultrafeedback-overall-score",
    "ultrafeedback-instruction-following",
    "ultrafeedback-truthfulness",
    "ultrafeedback-honesty",
    "ultrafeedback-helpfulness",
    "beavertails-is-safe",
    "code-complexity",
    "code-style",
    "code-explanation",
    "code-instruction-following",
    "code-readability",
    "prometheus-score",
    "argilla-overall-quality",
    "argilla-judge-lm",
]

# Dimensions to highlight per category for ablation 2
KEY_DIMS = {
    "coding": [
        "code-complexity",
        "code-style",
        "code-explanation",
        "code-instruction-following",
        "code-readability",
        "helpsteer-correctness",
    ],
    "safety_refuse": [
        "beavertails-is-safe",
        "ultrafeedback-honesty",
        "helpsteer-helpfulness",
        "helpsteer-verbosity",
    ],
    "safety_respond": [
        "beavertails-is-safe",
        "ultrafeedback-honesty",
        "helpsteer-helpfulness",
        "helpsteer-verbosity",
    ],
}

CODING_PREFIXES = (
    "hep-python",
    "hep-go",
    "hep-cpp",
    "hep-js",
    "hep-rust",
    "hep-java",
    "code_ultra_feedback",
)

SAFETY_REFUSE_PREFIXES = (
    "refusals-",
    "xstest-should-refuse",
    "donotanswer",
    "beaver_tails",
)

SAFETY_RESPOND_PREFIXES = (
    "xstest-should-respond",
)


# ============================================================
# HELPERS
# ============================================================

def categorize(prompt_id: str):
    if prompt_id.startswith(CODING_PREFIXES):
        return "coding"
    if prompt_id.startswith(SAFETY_REFUSE_PREFIXES):
        return "safety_refuse"
    if prompt_id.startswith(SAFETY_RESPOND_PREFIXES):
        return "safety_respond"
    return "other"


def rewards_to_dict(rewards: list) -> dict:
    return {name: val for name, val in zip(DIMENSION_NAMES, rewards)}


def gating_to_dict(gating: list) -> dict:
    return {name: val for name, val in zip(DIMENSION_NAMES, gating)}


# ============================================================
# STORAGE
# ============================================================

ablation1_rows = []
ablation2_rows = []
ablation3_rows = []


# ============================================================
# MAIN LOOP
# ============================================================

for model in MODELS:

    print(f"\nProcessing {model}")

    armorm_path = (
        BASE_DIR / model / "scores" / "armorm" / f"{model}_0_scored2.jsonl"
    )

    bart_path = (
        BASE_DIR / model / "scores" / "bartscore" / f"{model}_scored.jsonl"
    )

    # Load BARTScore lookup
    bart_lookup = {}
    with open(bart_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            pid = item["id"]
            scores = [
                r["bart_score"]
                for r in item["scored_responses"]
                if r.get("bart_score") is not None
            ]
            if scores:
                bart_lookup[pid] = np.mean(scores)

    # Load ArmoRM
    with open(armorm_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            pid = item["id"]
            category = categorize(pid)

            if category == "other":
                continue

            responses = item["scored_responses"]

            armorm_scores = []
            all_rewards = []
            all_gatings = []

            for r in responses:
                score = r.get("armorm_score")
                rewards = r.get("armorm_rewards")
                gating = r.get("armorm_gating")

                if score is not None:
                    armorm_scores.append(score)
                if rewards is not None and len(rewards) == len(DIMENSION_NAMES):
                    all_rewards.append(rewards)
                if gating is not None and len(gating) == len(DIMENSION_NAMES):
                    all_gatings.append(gating)

            if not armorm_scores:
                continue

            avg_armorm = np.mean(armorm_scores)
            avg_bart = bart_lookup.get(pid)

            # --------------------------------------------------
            # ABLATION 1 — per category model ranking
            # --------------------------------------------------

            ablation1_rows.append({
                "model": model,
                "category": category,
                "armorm_score": avg_armorm,
                "bart_score": avg_bart,
            })

            # --------------------------------------------------
            # ABLATION 2 — named gating weights + named rewards
            # --------------------------------------------------

            if all_gatings:
                avg_gating = np.mean(all_gatings, axis=0)
                avg_rewards = np.mean(all_rewards, axis=0) if all_rewards else None

                row = {
                    "model": model,
                    "category": category,
                }
                for name, val in zip(DIMENSION_NAMES, avg_gating):
                    row[f"gate_{name}"] = val
                if avg_rewards is not None:
                    for name, val in zip(DIMENSION_NAMES, avg_rewards):
                        row[f"reward_{name}"] = val

                ablation2_rows.append(row)

            # --------------------------------------------------
            # ABLATION 3 — safety_refuse vs safety_respond
            # --------------------------------------------------

            if category in ("safety_refuse", "safety_respond"):

                safety_rewards = []
                for r in responses:
                    rewards = r.get("armorm_rewards")
                    if rewards and len(rewards) == len(DIMENSION_NAMES):
                        safety_idx = DIMENSION_NAMES.index("beavertails-is-safe")
                        safety_rewards.append(rewards[safety_idx])

                row3 = {
                    "model": model,
                    "prompt_id": pid,
                    "category": category,
                    "avg_armorm_scalar": avg_armorm,
                    "avg_bart": avg_bart,
                    "avg_safety_reward": np.mean(safety_rewards) if safety_rewards else None,
                }

                # Add all dimension rewards for deeper analysis
                if all_rewards:
                    avg_rewards_arr = np.mean(all_rewards, axis=0)
                    for name, val in zip(DIMENSION_NAMES, avg_rewards_arr):
                        row3[f"reward_{name}"] = val

                ablation3_rows.append(row3)


# ============================================================
# ABLATION 1 — rankings per category
# ============================================================

df1 = pd.DataFrame(ablation1_rows)

summary1 = (
    df1.groupby(["model", "category"])
    .agg({"armorm_score": "mean", "bart_score": "mean"})
    .reset_index()
)

print("\n=== ABLATION 1: Rankings per category ===")
print(summary1.to_string())
summary1.to_csv("ablation1_category_scores.csv", index=False)

# Plot: one grouped bar chart per category, both metrics side by side
categories = summary1["category"].unique()

fig, axes = plt.subplots(1, len(categories), figsize=(16, 5), sharey=False)

for ax, cat in zip(axes, categories):
    sub = summary1[summary1["category"] == cat].set_index("model")
    x = np.arange(len(sub))
    width = 0.35

    ax2 = ax.twinx()

    ax.bar(x - width / 2, sub["armorm_score"], width, label="ArmoRM", color="#4C72B0")
    ax2.bar(x + width / 2, sub["bart_score"], width, label="BARTScore", color="#DD8452")

    ax.set_title(cat)
    ax.set_xticks(x)
    ax.set_xticklabels(sub.index, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("ArmoRM score", color="#4C72B0")
    ax2.set_ylabel("BARTScore", color="#DD8452")

    ax.legend(loc="upper left", fontsize=7)
    ax2.legend(loc="upper right", fontsize=7)

plt.suptitle("Model rankings per category — ArmoRM vs BARTScore")
plt.tight_layout()
plt.savefig("ablation1_rankings_per_category.png", dpi=150)
plt.close()
print("Saved ablation1_rankings_per_category.png")


# ============================================================
# ABLATION 2 — named gating weights heatmap per category
# ============================================================

df2 = pd.DataFrame(ablation2_rows)

gate_cols = [f"gate_{n}" for n in DIMENSION_NAMES]

gating_summary = (
    df2.groupby("category")[gate_cols]
    .mean()
    .rename(columns={f"gate_{n}": n for n in DIMENSION_NAMES})
)

print("\n=== ABLATION 2: Average gating weights per category ===")
print(gating_summary.T.to_string())
gating_summary.to_csv("ablation2_gating_weights.csv")

# Heatmap — all dimensions
plt.figure(figsize=(16, 4))
plt.imshow(gating_summary.values, aspect="auto", cmap="YlOrRd")
plt.colorbar(label="Average gating weight")
plt.xticks(
    ticks=np.arange(len(DIMENSION_NAMES)),
    labels=DIMENSION_NAMES,
    rotation=45,
    ha="right",
    fontsize=7,
)
plt.yticks(
    ticks=np.arange(len(gating_summary)),
    labels=gating_summary.index,
)
plt.title("ArmoRM MoE gating weights by category")
plt.tight_layout()
plt.savefig("ablation2_gating_heatmap_full.png", dpi=150)
plt.close()
print("Saved ablation2_gating_heatmap_full.png")

# Focused plot — only dimensions that vary most across categories
dim_variance = gating_summary.var(axis=0)
top_dims = dim_variance.nlargest(8).index.tolist()

gating_top = gating_summary[top_dims]

plt.figure(figsize=(10, 4))
plt.imshow(gating_top.values, aspect="auto", cmap="YlOrRd")
plt.colorbar(label="Average gating weight")
plt.xticks(
    ticks=np.arange(len(top_dims)),
    labels=top_dims,
    rotation=30,
    ha="right",
    fontsize=9,
)
plt.yticks(
    ticks=np.arange(len(gating_top)),
    labels=gating_top.index,
)
plt.title("ArmoRM MoE — top 8 most variable gating dimensions across categories")
plt.tight_layout()
plt.savefig("ablation2_gating_heatmap_top.png", dpi=150)
plt.close()
print("Saved ablation2_gating_heatmap_top.png")


# ============================================================
# ABLATION 3 — safety_refuse vs safety_respond
# ============================================================

df3 = pd.DataFrame(ablation3_rows)

# Summary per category and model
summary3 = (
    df3.groupby(["model", "category"])
    .agg({
        "avg_armorm_scalar": "mean",
        "avg_bart": "mean",
        "avg_safety_reward": "mean",
    })
    .reset_index()
)

print("\n=== ABLATION 3: Safety refuse vs safety respond ===")
print(summary3.to_string())
summary3.to_csv("ablation3_safety_comparison.csv", index=False)

# Also compare all dimension rewards between the two safety categories
reward_cols = [f"reward_{n}" for n in DIMENSION_NAMES]
available_reward_cols = [c for c in reward_cols if c in df3.columns]

if available_reward_cols:
    reward_summary = (
        df3.groupby("category")[available_reward_cols]
        .mean()
        .rename(columns={f"reward_{n}": n for n in DIMENSION_NAMES})
    )
    print("\n=== ABLATION 3: Per-dimension rewards by safety category ===")
    print(reward_summary.T.to_string())
    reward_summary.to_csv("ablation3_dimension_rewards_by_safety.csv")

# Plot 1 — safety dimension reward: refuse vs respond per model
pivot3 = summary3.pivot(index="model", columns="category", values="avg_safety_reward")

pivot3.plot(kind="bar", figsize=(9, 5), color=["#E05C5C", "#5C9EE0"])
plt.title("beavertails-is-safe reward: genuinely unsafe vs surface-unsafe prompts")
plt.ylabel("Average safety reward score")
plt.xlabel("")
plt.xticks(rotation=15)
plt.legend(["safety_refuse (should be low)", "safety_respond (should be high)"])
plt.tight_layout()
plt.savefig("ablation3_safety_reward_comparison.png", dpi=150)
plt.close()
print("Saved ablation3_safety_reward_comparison.png")

# Plot 2 — BARTScore for refuse vs respond per model
pivot3_bart = summary3.pivot(index="model", columns="category", values="avg_bart")

pivot3_bart.plot(kind="bar", figsize=(9, 5), color=["#E05C5C", "#5C9EE0"])
plt.title("BARTScore: genuinely unsafe vs surface-unsafe prompts")
plt.ylabel("Average BARTScore")
plt.xlabel("")
plt.xticks(rotation=15)
plt.legend(["safety_refuse", "safety_respond"])
plt.tight_layout()
plt.savefig("ablation3_bartscore_comparison.png", dpi=150)
plt.close()
print("Saved ablation3_bartscore_comparison.png")

# Plot 3 — dimension reward heatmap: refuse vs respond
if available_reward_cols:
    plt.figure(figsize=(16, 3))
    plt.imshow(reward_summary.values, aspect="auto", cmap="RdYlGn")
    plt.colorbar(label="Average reward score")
    plt.xticks(
        ticks=np.arange(len(DIMENSION_NAMES)),
        labels=DIMENSION_NAMES,
        rotation=45,
        ha="right",
        fontsize=7,
    )
    plt.yticks(
        ticks=np.arange(len(reward_summary)),
        labels=reward_summary.index,
    )
    plt.title("Per-dimension reward scores: safety_refuse vs safety_respond")
    plt.tight_layout()
    plt.savefig("ablation3_dimension_heatmap.png", dpi=150)
    plt.close()
    print("Saved ablation3_dimension_heatmap.png")

print("\nDONE.")