import os
from dotenv import load_dotenv
load_dotenv()

# Hugging Face API Setup
HF_API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
HF_HEADERS = {"Authorization": f"Bearer {os.getenv('HF_API_KEY')}"}

# Router configs
CANDIDATE_LABELS = ["simple", "general", "complex", "code", "math"]

TOPIC_TO_MODEL = {
    "simple": "ollama/gemma2:2b",
    "general": "ollama/phi3:latest",
    "complex": "ollama/mistral:7b",
    "code": "ollama/qwen2.5-coder:latest",
    "math": "ollama/qwen2-math:latest",
}