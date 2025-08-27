from llm_router.schemas.env_validator import get_env_var, validate_env_vars

# Validate environment variables at import time
validate_env_vars()

# Hugging Face API Setup
HF_API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
HF_HEADERS = {"Authorization": f"Bearer {get_env_var('HF_API_KEY')}"}

# Router configs
CANDIDATE_LABELS = ["simple", "general", "complex", "code", "math"]

TOPIC_TO_MODEL = {
    "simple": "ollama/gemma2:2b",
    "general": "ollama/phi3:latest",
    "complex": "ollama/mistral:7b",
    "code": "ollama/qwen2.5-coder:latest",
    "math": "ollama/qwen2-math:latest",
}
