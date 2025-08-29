"""Static configuration and helpers for selectors."""
from __future__ import annotations
import os

from typing import Dict

# Hugging Face API Setup
HF_API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"


def get_hf_headers() -> Dict[str, str]:
    """Build authorization headers for HuggingFace requests.

    The router validates required environment variables on startup, so here we
    simply read the key from the environment without additional checks. If the
    key is missing, an empty header is returned and requests will fail upstream,
    but importing selectors will not raise errors.
    """

    api_key = os.getenv("HF_API_KEY", "")
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


# Router configs
CANDIDATE_LABELS = ["simple", "general", "complex", "code", "math"]

# Mapping from topic to model names for each provider.
#
# These values should correspond to model identifiers accepted by the
# respective provider implementations (e.g., via LiteLLM).
TOPIC_TO_MODEL = {
    "simple": {
        "anthropic": "claude-3-haiku-20240307",
        "openai": "gpt-3.5-turbo",
        "google": "gemini-1.5-flash",
    },
    "general": {
        "anthropic": "claude-3-sonnet-20240229",
        "openai": "gpt-4o-mini",
        "google": "gemini-1.5-pro",
    },
    "complex": {
        "anthropic": "claude-3-opus-20240229",
        "openai": "gpt-4o",
        "google": "gemini-1.5-pro",
    },
    "code": {
        "anthropic": "claude-3-sonnet-20240229",
        "openai": "gpt-4o-mini",
        "google": "gemini-1.5-pro",
    },
    "math": {
        "anthropic": "claude-3-haiku-20240307",
        "openai": "gpt-4o-mini",
        "google": "gemini-1.5-pro",
    },
}
