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

TOPIC_TO_MODEL = {
    "simple": "ollama/gemma2:2b",
    "general": "ollama/phi3:latest",
    "complex": "ollama/mistral:7b",
    "code": "ollama/qwen2.5-coder:latest",
    "math": "ollama/qwen2-math:latest",
}
