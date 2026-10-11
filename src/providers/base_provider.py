"""
base_provider.py

A Provider is a small class that knows how to talk to a model backend:
- OpenAI API
- Hugging Face Inference API
- Mistral API

The rest of the codebase does NOT care which backend is used.
"""

from abc import ABC, abstractmethod

from schema import GeneratedResponse


class LLMResponseProvider(ABC):
    """
    Interface for any provider that can generate multiple response samples.
    """

    @abstractmethod
    def generate_multiple_responses(
        self,
        *,
        prompt_text: str,
        number_of_samples: int,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
        return_token_logprobs: bool,
        random_seed: int,
    ) -> list[GeneratedResponse]:
        """
        Generate multiple responses for one prompt.

        Stage-1 expectation:
        - return_token_logprobs is usually False
        - token_logprobs in GeneratedResponse can be None
        """
        raise NotImplementedError