"""Send prompts to an LLM.

Any provider with an OpenAI-compatible API works (Groq, Google Gemini,
OpenAI and others). Settings come from environment variables or a .env file:

    LLM_API_KEY   your API key (required unless using the mock provider)
    LLM_BASE_URL  the provider's OpenAI-compatible URL (default: Groq)
    LLM_MODEL     the model name (default: a Groq-hosted Llama model)

The MockLLM needs no key or internet. It is used by the unit tests and CI so
the pipeline can be checked without spending API calls.
"""

from __future__ import annotations

import os
import time

from dotenv import load_dotenv

load_dotenv()

DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "llama-3.3-70b-versatile"


class LLMClient:
    """Thin wrapper around the OpenAI Python client."""

    def __init__(self, api_key: str | None = None, base_url: str | None = None, model: str | None = None):
        from openai import OpenAI  # imported here so the mock path works without the package

        api_key = api_key or os.getenv("LLM_API_KEY")
        if not api_key:
            raise RuntimeError(
                "No API key found. Set LLM_API_KEY in a .env file or as a Codespaces secret "
                "(see the README), or run with --provider mock."
            )
        self.model = model or os.getenv("LLM_MODEL", DEFAULT_MODEL)
        self.client = OpenAI(api_key=api_key, base_url=base_url or os.getenv("LLM_BASE_URL", DEFAULT_BASE_URL))

    def complete(self, system: str, user: str, retries: int = 3) -> str:
        """Return the model's reply. Retries with a growing wait on errors such as rate limits."""
        for attempt in range(retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    temperature=0,  # make answers as repeatable as possible for evaluation
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                )
                return (response.choices[0].message.content or "").strip()
            except Exception as error:  # noqa: BLE001 - show any provider error to the user
                if attempt == retries - 1:
                    raise
                wait = 2 ** (attempt + 1)
                print(f"  LLM call failed ({error.__class__.__name__}); retrying in {wait}s...")
                time.sleep(wait)
        return ""


class MockLLM:
    """Offline stand-in: answers with the first chunk it was given.

    It cannot really answer questions; it only lets you check that the
    pipeline, evaluation script and tests run end to end.
    """

    model = "mock"

    def complete(self, system: str, user: str, retries: int = 3) -> str:
        marker = "Context:\n"
        if marker not in user:
            return "I don't know."
        context = user.split(marker, 1)[1].split("\n\nQuestion:", 1)[0]
        first_chunk = context.strip().split("\n\n", 1)[0]
        return first_chunk or "I don't know."


def make_llm(provider: str = "llm"):
    """Create the client for a provider name: "llm" (real API) or "mock"."""
    if provider == "mock":
        return MockLLM()
    if provider == "llm":
        return LLMClient()
    raise ValueError(f"Unknown provider {provider!r}; use 'llm' or 'mock'")
