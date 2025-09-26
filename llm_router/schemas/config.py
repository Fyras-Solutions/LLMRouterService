"""Static configuration and helpers for selectors."""
from __future__ import annotations
import os

from typing import Dict

# # Hugging Face API Setup
# HF_API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"


# def get_hf_headers() -> Dict[str, str]:
#     """Build authorization headers for HuggingFace requests.

#     The router validates required environment variables on startup, so here we
#     simply read the key from the environment without additional checks. If the
#     key is missing, an empty header is returned and requests will fail upstream,
#     but importing selectors will not raise errors.
#     """

#     api_key = os.getenv("HF_API_KEY", "")
#     return {"Authorization": f"Bearer {api_key}"} if api_key else {}


# Router configs
CANDIDATE_LABELS = ["SIMPLE","COMPLEX","FINANCE","PROGRAMMING","TECHNOLOGY","ENTERTAINMENT","HEALTH"]


# Mapping from topic to model names for each provider.
#
# These values should correspond to model identifiers accepted by the
# respective provider implementations (e.g., via LiteLLM).
TOPIC_TO_MODEL = {
    "SIMPLE": {
        "anthropic": "claude-3-haiku-20240307",
        "openai": "gpt-3.5-turbo",
        "google": "gemini-2.5-flash-lite" 
    },
    "FINANCE": {
        "anthropic": "claude-3-5-haiku-20241022", 
        "openai": "gpt-4o-mini",
        "google": "gemini-2.5-flash" 
    },
    "COMPLEX": {
        "anthropic": "claude-opus-4-20250514", 
        "openai": "gpt-4o",
        "google": "gemini-2.5-pro" 
    },
    "PROGRAMMING": {
        "anthropic": "claude-sonnet-4-20250514", 
        "openai": "gpt-4.1-mini", 
        "google": "gemini-2.5-pro", 
    },
    "TECHNOLOGY": {
        "anthropic": "claude-3-haiku-20240307",
        "openai": "gpt-4o-mini",
        "google": "gemini-2.5-flash-lite" 
    },
    "ENTERTAINMENT":{
        "anthropic": "claude-3-haiku-20240307",
        "openai": "gpt-3.5-turbo",
        "google": "gemini-2.5-flash-lite"
    },
    "HEALTH":{
        "anthropic": "claude-opus-4-20250514", 
        "openai": "gpt-4o",
        "google": "gemini-2.5-pro"
    },
    "GENERAL":{
        "anthropic": "claude-3-5-haiku-20241022", 
        "openai": "gpt-4o-mini",
        "google": "gemini-2.5-flash"  
    }
}

