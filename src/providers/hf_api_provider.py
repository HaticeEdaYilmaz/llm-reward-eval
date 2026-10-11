"""
hf_api_provider.py

Hugging Face Inference API provider (stable mode).

We FORCE the provider to "hf-inference" to avoid auto-routing to 3rd-party providers
(e.g. together) which can cause inconsistent task support and StopIteration errors.

Env required:
- HF_TOKEN
"""

import os
from typing import Optional

from huggingface_hub import InferenceClient

from schema import GeneratedResponse
from .base_provider import LLMResponseProvider


class HuggingFaceInferenceAPIProvider(LLMResponseProvider):
    """
    Generate responses using Hugging Face serverless inference ("hf-inference").
    """

    def __init__(self, *, hf_model_id: str, hf_token: Optional[str] = None) -> None:
        resolved_token = hf_token or os.getenv("HF_TOKEN")
        if not resolved_token:
            raise RuntimeError("HF_TOKEN is missing. Please set it in your environment.")

        # Force provider to avoid 'together' auto-routing issues
        self._client = InferenceClient(
            model=hf_model_id,
            token=resolved_token,
            provider="hf-inference",
        )
        self._hf_model_id = hf_model_id

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
        Generate N responses via text_generation.

        If HF serverless doesn't host the model, we raise a clear error so students can
        switch to another model.
        """
        generated: list[GeneratedResponse] = []

        for sample_index in range(number_of_samples):
            try:
                response_text = self._client.text_generation(
                    prompt_text,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                )
            except Exception as e:
                raise RuntimeError(
                    f"HF inference failed for model '{self._hf_model_id}'. "
                    f"This usually means the model is not available on HF serverless right now "
                    f"(or requires a different backend). Original error: {e}"
                ) from e

            generated.append(
                GeneratedResponse(
                    response_text=response_text,
                    metadata={
                        "sample_index": sample_index,
                        "provider": "hf_inference_api",
                        "hf_model_id": self._hf_model_id,
                    },
                    token_logprobs=None,
                )
            )

        return generated