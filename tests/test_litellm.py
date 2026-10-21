import asyncio
import os
import time

# Configure LiteLLM settings prior to importing
os.environ["LITELLM_TELEMETRY"] = "False"
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"

import litellm
from litellm import ModelResponse
import pytest

# Suppress verbose debug logs
litellm.suppress_debug_info = True

PRIMARY_MODEL = os.getenv("PRIMARY_LLM_MODEL", "gemini/gemini-3.5-flash")
FALLBACK_MODEL = os.getenv("FALLBACK_LLM_MODEL", "ollama/gemma3:1b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL") or os.getenv("OLLAMA_HOST") or "http://127.0.0.1:11434"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")


async def async_call_with_fallback(
    prompt: str = "Explain in one sentence what an abstract syntax tree (AST) is.",
    primary_model: str = PRIMARY_MODEL,
    fallback_model: str = FALLBACK_MODEL,
    ollama_base_url: str = OLLAMA_BASE_URL,
    gemini_api_key: str | None = GEMINI_API_KEY,
):
    """
    Executes an asynchronous LiteLLM chat completion with Gemini as the primary
    model and Ollama as the fallback model.
    """
    messages = [
        {"role": "system", "content": "You are a concise technical assistant."},
        {"role": "user", "content": prompt},
    ]

    # Primary kwargs (e.g. Gemini API key if available)
    primary_kwargs = {}
    if gemini_api_key:
        primary_kwargs["api_key"] = gemini_api_key

    # Fallback configuration targeting local Ollama
    fallback_dict = {
        "model": fallback_model,
        "api_base": ollama_base_url,
    }

    print(f"\n--- Async LiteLLM Call ---")
    print(f"Primary model : {primary_model}")
    print(f"Fallback model: {fallback_model} (base: {ollama_base_url})")
    print(f"User prompt   : {prompt}")

    start_time = time.perf_counter()
    response = await litellm.acompletion(
        model=primary_model,
        messages=messages,
        temperature=0.2,
        max_tokens=150,
        fallbacks=[fallback_dict],
        **primary_kwargs,
    )
    elapsed = time.perf_counter() - start_time

    model_used = getattr(response, "model", "unknown")
    content = ""
    if isinstance(response, ModelResponse) and response.choices:
        content = response.choices[0].message.content or ""

    print(f"Response time : {elapsed:.2f}s")
    print(f"Model answered: {model_used}")
    print(f"Response      : {content.strip()}")
    print("--------------------------\n")

    return response


def test_async_gemini_with_ollama_fallback():
    """Test async litellm call with Gemini as primary and Ollama as fallback."""
    response = asyncio.run(async_call_with_fallback())
    assert isinstance(response, ModelResponse)
    assert response.choices and len(response.choices) > 0
    assert response.choices[0].message.content is not None
    assert len(response.choices[0].message.content.strip()) > 0


if __name__ == "__main__":
    asyncio.run(async_call_with_fallback())
