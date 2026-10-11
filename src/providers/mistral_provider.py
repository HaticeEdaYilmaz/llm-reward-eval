"""
mistral_provider.py

Mistral API provider for models like 'codestral-latest'.

Environment variable needed:
- MISTRAL_API_KEY
"""

import os
from typing import Optional

from mistralai.client import Mistral

from schema import GeneratedResponse
from .base_provider import LLMResponseProvider


class MistralChatProvider(LLMResponseProvider):
    """
    Provider that generates responses using Mistral API.
    """

    def __init__(self, *, mistral_model_name: str, api_key: Optional[str] = None) -> None:
        resolved_key = api_key or os.getenv("MISTRAL_API_KEY")
        if not resolved_key:
            raise RuntimeError("MISTRAL_API_KEY is missing. Please set it in your environment.")

        self._client = Mistral(api_key=resolved_key)
        self._mistral_model_name = mistral_model_name

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
        Generate N responses for one prompt.

        Stage-1: token logprobs not collected.
        """
        generated: list[GeneratedResponse] = []

        for sample_index in range(number_of_samples):
            result = self._client.chat.complete(
                model=self._mistral_model_name,
                messages=[{"role": "user", "content": prompt_text}],
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_new_tokens,
            )
            response_text = result.choices[0].message.content or ""

            generated.append(
                GeneratedResponse(
                    response_text=response_text,
                    metadata={
                        "sample_index": sample_index,
                        "temperature": temperature,
                        "top_p": top_p,
                        "max_new_tokens": max_new_tokens,
                        "random_seed": random_seed,
                        "provider": "mistral_api",
                        "mistral_model": self._mistral_model_name,
                    },
                    token_logprobs=None,
                )
            )

        return generated