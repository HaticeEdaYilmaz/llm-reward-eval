"""
sampling_pipeline.py

Stage-1 pipeline:
- Read prompt dataset from JSONL
- Generate N responses per prompt using a chosen model (provider auto-selected)
- Write results to output JSONL
"""

from dataclasses import asdict
from typing import Optional

from jsonl_io import (
    append_json_object_to_jsonl,
    load_already_processed_prompt_ids,
    read_prompt_items_from_jsonl,
)
from model_registry import get_model_spec
from providers.base_provider import LLMResponseProvider
from providers.hf_api_provider import HuggingFaceInferenceAPIProvider
from providers.mistral_provider import MistralChatProvider
from providers.openai_provider import OpenAIChatCompletionsProvider


def create_provider_for_model(model_key: str) -> tuple[LLMResponseProvider, str]:
    """
    Create the correct provider instance for a given model_key using the model registry.

    Returns
    -------
    (provider_instance, model_name_for_output)
    """
    model_spec = get_model_spec(model_key)

    if model_spec.provider == "openai":
        provider = OpenAIChatCompletionsProvider(requested_model_name=model_spec.provider_model_id)
        return provider, model_spec.model_key

    if model_spec.provider == "hf_api":
        provider = HuggingFaceInferenceAPIProvider(hf_model_id=model_spec.provider_model_id)
        return provider, model_spec.model_key

    if model_spec.provider == "mistral":
        provider = MistralChatProvider(mistral_model_name=model_spec.provider_model_id)
        return provider, model_spec.model_key

    raise ValueError(f"Unsupported provider type: {model_spec.provider}")

def generate_responses_for_prompt(
    *,
    model_key: str,
    prompt_text: str,
    number_of_samples: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,):
    """
    Generate responses for a single prompt without saving to disk.
    """

    provider, model_name_for_output = create_provider_for_model(model_key)

    generated_responses = provider.generate_multiple_responses(
        prompt_text=prompt_text,
        number_of_samples=number_of_samples,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        return_token_logprobs=False,
        random_seed=0,
    )

    return {"model": model_name_for_output, "responses": [asdict(r) for r in generated_responses]}

def generate_and_save_responses_for_dataset(
    *,
    model_key: str,
    input_prompts_jsonl_path: str,
    output_responses_jsonl_path: str,
    number_of_samples_per_prompt: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    max_number_of_prompts: Optional[int],
    start_index: int = 0,
    resume_if_output_exists: bool,
) -> None:
    """
    End-to-end Stage-1:
    - create provider
    - read prompts
    - generate N responses per prompt
    - write JSONL output
    """
    provider, model_name_for_output = create_provider_for_model(model_key)

    prompt_items = read_prompt_items_from_jsonl(input_prompts_jsonl_path)

    if max_number_of_prompts is not None:
        prompt_items = prompt_items[start_index:start_index + max_number_of_prompts]
    else:
        prompt_items = prompt_items[start_index:]

    processed_ids = (
        load_already_processed_prompt_ids(output_responses_jsonl_path)
        if resume_if_output_exists
        else set()
    )

    for prompt_item in prompt_items:
        if prompt_item.prompt_id in processed_ids:
            continue

        generated_responses = provider.generate_multiple_responses(
            prompt_text=prompt_item.prompt_text,
            number_of_samples=number_of_samples_per_prompt,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            return_token_logprobs=False,
            random_seed=0,
        )

        output_json_obj = {
            "id": prompt_item.prompt_id,
            "prompt": prompt_item.prompt_text,
            "model": model_name_for_output,
            "responses": [asdict(r) for r in generated_responses],
        }

        append_json_object_to_jsonl(output_responses_jsonl_path, output_json_obj)