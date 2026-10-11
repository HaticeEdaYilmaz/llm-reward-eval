"""
model_registry.py

This is the "single source of truth" for supported models.

Students should only need to:
- choose a model_key from SUPPORTED_MODEL_KEYS
- run the CLI with --model-name <model_key>

We automatically pick the correct provider (OpenAI / HF API / Mistral).
"""

from dataclasses import dataclass
from typing import Literal

ProviderName = Literal["openai", "hf_api", "mistral"]


@dataclass(frozen=True)
class ModelSpec:
    """
    Model specification used by the factory to create the correct provider.

    Attributes
    ----------
    model_key:
        The friendly name used in CLI (e.g., 'llama-31-8b').
    provider:
        Which provider is responsible for this model.
    provider_model_id:
        The real model id used by that provider API.
        - For OpenAI: the 'model' string sent to OpenAI API
        - For HF API: the Hugging Face repo id (or endpoint model id)
        - For Mistral: the model string for Mistral API (e.g., 'codestral-latest')
    """
    model_key: str
    provider: ProviderName
    provider_model_id: str


MODEL_SPECS: dict[str, ModelSpec] = {
    # OpenAI 
    "gpt-35-turbo": ModelSpec("gpt-35-turbo", "openai", "gpt-35-turbo"),
    "gpt-4o-mini": ModelSpec("gpt-4o-mini", "openai", "gpt-4o-mini"),
    "gpt-4o": ModelSpec("gpt-4o", "openai", "gpt-4o"),

    # Mistral API (based on your /models list)
    "codestral-latest": ModelSpec("codestral-latest", "mistral", "codestral-latest"),
    "mistral-small-latest": ModelSpec("mistral-small-latest", "mistral", "mistral-small-latest"),
    "mistral-large-latest": ModelSpec("mistral-large-latest", "mistral", "mistral-large-latest"),

    # Hugging Face Inference API (stabil)
    "phi-3-mini": ModelSpec("phi-3-mini", "hf_api", "microsoft/Phi-3-mini-4k-instruct"),
    "phi-3-medium": ModelSpec("phi-3-medium", "hf_api", "microsoft/Phi-3-medium-4k-instruct"),
}

def list_supported_model_keys() -> list[str]:
    """Return a sorted list of model keys supported by this repo."""
    return sorted(MODEL_SPECS.keys())


def get_model_spec(model_key: str) -> ModelSpec:
    """
    Get a ModelSpec for a given model_key.

    If the user passes an unknown model_key, we raise a clear error.
    """
    if model_key not in MODEL_SPECS:
        supported = ", ".join(list_supported_model_keys())
        raise ValueError(
            f"Unknown model_name '{model_key}'. Supported model names: {supported}"
        )
    return MODEL_SPECS[model_key]