"""
armorm_scoring_pipeline.py

Stage-2 pipeline:
- Load ArmoRM reward model locally (RLHFlow/ArmoRM-Llama3-8B-v0.1)
- Read responses JSONL (output of Stage-1 generate)
- Score every response for each prompt using ArmoRM
- Write scored results to output JSONL
"""

import json
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from jsonl_io import append_json_object_to_jsonl, load_already_processed_prompt_ids

ARMORM_MODEL_NAME = "RLHFlow/ArmoRM-Llama3-8B-v0.1"


def load_armorm_model(model_name: str = ARMORM_MODEL_NAME):
    """
    Load ArmoRM tokenizer and model onto available device.

    Returns
    -------
    (tokenizer, model)
    """
    print(f"Loading ArmoRM model: {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print(f"ArmoRM loaded on device: {model.device}")
    return tokenizer, model


def score_single_response(
    prompt,
    response,
    tokenizer = AutoTokenizer.from_pretrained("RLHFlow/ArmoRM-Llama3-8B-v0.1", use_fast=True),
    model = AutoModelForSequenceClassification.from_pretrained(
        "RLHFlow/ArmoRM-Llama3-8B-v0.1",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
) -> dict:
    """
    Score one (prompt, response) pair with ArmoRM.

    Returns
    -------
    dict with keys:
        armorm_score   – scalar float (main reward score)
        armorm_rewards – per-attribute reward list
        armorm_gating  – gating weights list
    """
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages, return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        output = model(input_ids)

    return {
        "armorm_score": float(output.score.squeeze().detach().cpu().item()),
        "armorm_rewards": output.rewards.squeeze().detach().cpu().tolist(),
        "armorm_gating": output.gating_output.squeeze().detach().cpu().tolist(),
    }


def _read_response_items_from_jsonl(input_jsonl_path: str) -> list[dict]:
    """Read all items from a responses JSONL file."""
    items = []
    path = Path(input_jsonl_path)
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "id" not in obj or "prompt" not in obj:
                raise ValueError(
                    f"{path}:{line_number} must contain keys 'id' and 'prompt'."
                )
            items.append(obj)
    return items


def _extract_responses(item: dict) -> list[dict]:
    """
    Extract the list of response dicts from a JSONL item.

    Handles two formats produced in this project:
      Standard (sampling_pipeline output):
        {"id": ..., "prompt": ..., "model": ..., "responses": [...]}
      Legacy (model-keyed):
        {"id": ..., "prompt": ..., "<model-name>": {"responses": [...]}}
    """
    if "responses" in item:
        return item["responses"]

    for key, val in item.items():
        if key in ("id", "prompt"):
            continue
        if isinstance(val, dict) and "responses" in val:
            return val["responses"]

    return []


def score_and_save_responses(
    *,
    input_responses_jsonl_path: str,
    output_scored_jsonl_path: str,
    model_name: str = ARMORM_MODEL_NAME,
    max_number_of_prompts: Optional[int] = None,
    resume_if_output_exists: bool = True,
) -> None:
    """
    End-to-end Stage-2:
    - Load ArmoRM model
    - Read responses JSONL
    - Score every response for each prompt
    - Write scored JSONL output

    Output JSONL line format
    ------------------------
    {
        "id": "...",
        "prompt": "...",
        "model": "...",
        "scored_responses": [
            {
                "response_text": "...",
                "metadata": {...},
                "token_logprobs": null,
                "armorm_score": 0.123,
                "armorm_rewards": [...],
                "armorm_gating": [...]
            },
            ...
        ]
    }
    """
    tokenizer, model = load_armorm_model(model_name)

    items = _read_response_items_from_jsonl(input_responses_jsonl_path)
    if max_number_of_prompts is not None:
        items = items[:max_number_of_prompts]

    processed_ids = (
        load_already_processed_prompt_ids(output_scored_jsonl_path)
        if resume_if_output_exists
        else set()
    )

    for i, item in enumerate(items):
        prompt_id = str(item["id"])
        if prompt_id in processed_ids:
            continue

        prompt = item["prompt"]
        responses = _extract_responses(item)

        scored_responses = []
        for resp in responses:
            # Responses can be plain strings (legacy format) or dicts (standard format)
            if isinstance(resp, str):
                response_text = resp
                resp_base = {"response_text": resp}
            else:
                response_text = resp.get("response_text", "")
                resp_base = resp
            try:
                scores = score_single_response(tokenizer, model, prompt, response_text)
            except Exception as exc:
                print(f"  [warning] scoring failed for id={prompt_id}: {exc}")
                scores = {"armorm_score": None, "armorm_rewards": None, "armorm_gating": None}
            scored_responses.append({**resp_base, **scores})

        output_obj = {
            "id": prompt_id,
            "prompt": prompt,
            "model": item.get("model", ""),
            "scored_responses": scored_responses,
        }

        append_json_object_to_jsonl(output_scored_jsonl_path, output_obj)

        if (i + 1) % 10 == 0:
            print(f"Scored: {i + 1} / {len(items)}")