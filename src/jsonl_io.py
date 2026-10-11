"""
jsonl_io.py

JSONL ("JSON Lines") format:
- Each line is one JSON object.
- Great for streaming read/write.

This module provides:
1) read prompts from input JSONL
2) append results to output JSONL
3) resume support: skip prompts already written to output
"""

import json
from pathlib import Path

from schema import PromptItem


def read_prompt_items_from_jsonl(input_jsonl_path: str) -> list[PromptItem]:
    """
    Read prompts from a JSONL file.

    Expected line format:
        {"id": "...", "prompt": "..."}

    Parameters
    ----------
    input_jsonl_path:
        Path to input JSONL file.

    Returns
    -------
    list[PromptItem]
    """
    input_path = Path(input_jsonl_path)
    prompt_items: list[PromptItem] = []

    with input_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue

            json_obj = json.loads(line)
            if "id" not in json_obj or "prompt" not in json_obj:
                raise ValueError(
                    f"{input_path}:{line_number} must contain keys: 'id' and 'prompt'."
                )

            prompt_items.append(
                PromptItem(
                    prompt_id=str(json_obj["id"]),
                    prompt_text=str(json_obj["prompt"]),
                )
            )

    return prompt_items


def append_json_object_to_jsonl(output_jsonl_path: str, json_obj: dict) -> None:
    """
    Append a JSON object as one line to output JSONL file.

    Parameters
    ----------
    output_jsonl_path:
        Output path.
    json_obj:
        JSON-serializable dict.
    """
    output_path = Path(output_jsonl_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(json_obj, ensure_ascii=False) + "\n")


def load_already_processed_prompt_ids(output_jsonl_path: str) -> set[str]:
    """
    Read existing output JSONL and collect already-written prompt ids.
    Used for --resume behavior.

    Parameters
    ----------
    output_jsonl_path:
        Output path.

    Returns
    -------
    set[str]
    """
    output_path = Path(output_jsonl_path)
    if not output_path.exists():
        return set()

    processed_ids: set[str] = set()
    with output_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            json_obj = json.loads(line)
            if "id" in json_obj:
                processed_ids.add(str(json_obj["id"]))

    return processed_ids