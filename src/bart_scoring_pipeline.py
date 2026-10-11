"""
bart_scoring_pipeline.py

Stage-2 alternative scoring pipeline using BARTScore.
- Load BART seq2seq model locally
- Read responses JSONL (output of Stage-1 generate)
- Score every response for each prompt using BARTScore
- Write scored results to output JSONL
"""

import json
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from jsonl_io import append_json_object_to_jsonl, load_already_processed_prompt_ids


BART_MODEL_NAME = "facebook/bart-large"


def load_bart_model(model_name: str = BART_MODEL_NAME):
    """
    Load BART tokenizer and seq2seq model onto available device.
    """
    print(f"Loading BART model: {model_name} ...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    if not torch.cuda.is_available():
        model = model.to("cpu")

    model.eval()
    print(f"BART loaded on device: {next(model.parameters()).device}")
    return tokenizer, model

def score_batch_responses(
    tokenizer,
    model,
    prompt: str,
    responses: list[str],
    batch_size: int = 8,
) -> list[dict]:

    device = next(model.parameters()).device
    results = []

    for i in range(0, len(responses), batch_size):
        batch_responses = responses[i:i + batch_size]

        inputs = tokenizer(
            [prompt] * len(batch_responses),
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )

        targets = tokenizer(
            batch_responses,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = targets["input_ids"].to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=labels,
            )

            # IMPORTANT FIX: per-token loss
            logits = outputs.logits  # [B, T, V]

            shift_logits = logits[:, :-1, :]
            shift_labels = labels[:, 1:]

            loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=tokenizer.pad_token_id)

            vocab_size = shift_logits.size(-1)

            loss = loss_fct(
                shift_logits.reshape(-1, vocab_size),
                shift_labels.reshape(-1),
            )

            loss = loss.view(len(batch_responses), -1)

            seq_logprob = -loss.sum(dim=1) / (shift_labels != tokenizer.pad_token_id).sum(dim=1).clamp(min=1)

        for j, resp in enumerate(batch_responses):
            results.append({
                "response_text": resp,
                "bart_score": float(seq_logprob[j].cpu().item()),
            })

    return results


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
      Standard:
        {"id": ..., "prompt": ..., "model": ..., "responses": [...]}
      Legacy:
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
    model_name: str = BART_MODEL_NAME,
    max_number_of_prompts: Optional[int] = None,
    resume_if_output_exists: bool = True,
) -> None:
    """
    End-to-end Stage-2:
    - Load BART model
    - Read responses JSONL
    - Score every response for each prompt
    - Write scored JSONL output

    Output JSONL line format:
    {
        "id": "...",
        "prompt": "...",
        "model": "...",
        "scored_responses": [
            {
                "response_text": "...",
                "metadata": {...},
                "token_logprobs": null,
                "bart_score": -12.34
            },
            ...
        ]
    }
    """
    tokenizer, model = load_bart_model(model_name)

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

        response_texts = [r["response_text"] if isinstance(r, dict) else r for r in responses]

        try:
            scored_batch = score_batch_responses(
                tokenizer,
                model,
                prompt,
                response_texts,
                batch_size=8,   # tweak 4–16 for T4
            )
        except Exception as exc:
            print(f"[warning] batch scoring failed for id={prompt_id}: {exc}")
            scored_batch = [{"response_text": r, "bart_score": None} for r in response_texts]

        scored_responses = scored_batch

        output_obj = {
            "id": prompt_id,
            "prompt": prompt,
            "model": item.get("model", ""),
            "scored_responses": scored_responses,
        }

        append_json_object_to_jsonl(output_scored_jsonl_path, output_obj)

        if (i + 1) % 10 == 0:
            print(f"Scored: {i + 1} / {len(items)}")