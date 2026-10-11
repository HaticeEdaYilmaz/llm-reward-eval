"""
openai_provider.py

OpenAI provider using Chat Completions API.

Important:
- 'gpt-35-turbo' may NOT exist on OpenAI standard API (often Azure name).
  To be robust, we try the requested model first, then fall back to a known alias if needed.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from typing import Optional

from openai import OpenAI

from schema import GeneratedResponse
from .base_provider import LLMResponseProvider


# If the user asks for older/alternative names, we can fallback.
# Students can add more aliases here if needed.
OPENAI_MODEL_FALLBACKS: dict[str, str] = {
    "gpt-35-turbo": "gpt-3.5-turbo",
}


class OpenAIChatCompletionsProvider(LLMResponseProvider):
    """
    Simple sampling provider for OpenAI models.
    Generates N samples by calling the API N times.
    """

    def __init__(self, *, requested_model_name: str, api_key: Optional[str] = None) -> None:
        resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_api_key:
            raise RuntimeError("OPENAI_API_KEY is missing. Please set it in your environment.")

        self._client = OpenAI(api_key=resolved_api_key)
        self._requested_model_name = requested_model_name

    def _call_openai_once(
        self,
        *,
        model_name: str,
        prompt_text: str,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
    ) -> str:
        """One OpenAI call returning only the response text."""
        api_result = self._client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt_text}],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
        )
        return api_result.choices[0].message.content or ""

    from concurrent.futures import ThreadPoolExecutor, as_completed

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

        primary_model = self._requested_model_name
        fallback_model = OPENAI_MODEL_FALLBACKS.get(primary_model)

        # decide working model first (once)
        try:
            self._call_openai_once(
                model_name=primary_model,
                prompt_text=prompt_text,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
            )
            working_model = primary_model
        except Exception:
            if not fallback_model:
                raise
            working_model = fallback_model

        def task(i: int):
            text = self._call_openai_once(
                model_name=working_model,
                prompt_text=prompt_text,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
            )
            return GeneratedResponse(
                response_text=text,
                metadata={
                    "sample_index": i,
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_new_tokens": max_new_tokens,
                    "random_seed": random_seed,
                    "provider": "openai_chat_completions",
                    "openai_model_used": working_model,
                },
                token_logprobs=None,
            )

        generated: list[GeneratedResponse] = []

        with ThreadPoolExecutor(max_workers=number_of_samples) as executor:
            futures = [executor.submit(task, i) for i in range(number_of_samples)]

            for f in as_completed(futures):
                generated.append(f.result())

        return generated