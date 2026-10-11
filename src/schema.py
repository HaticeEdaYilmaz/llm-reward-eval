"""
schema.py

Very small data containers used across the project.
Keeping them here makes the code easier to understand and refactor.
"""

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class PromptItem:
    """
    One prompt (query) from the dataset.

    Attributes
    ----------
    prompt_id:
        Unique identifier for the prompt.
    prompt_text:
        The user prompt string to send to the model.
    """
    prompt_id: str
    prompt_text: str


@dataclass(frozen=True)
class GeneratedResponse:
    """
    One generated response sample from a model.

    Attributes
    ----------
    response_text:
        The generated text.
    metadata:
        Helpful information about this generation (sample_index, temperature, etc.).
    token_logprobs:
        Placeholder for future (token probabilities / logprobs). Stage-1 sets this to None.
    """
    response_text: str
    metadata: dict[str, Any]
    token_logprobs: Optional[Any] = None