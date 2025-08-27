"""Environment variable validation for LLM Router Service."""
import os
from typing import List, Dict
from pathlib import Path
from dotenv import load_dotenv
from llm_router.exceptions.exceptions import LLMRouterError

class EnvVarError(LLMRouterError):
    """Raised when required environment variables are missing."""
    pass

REQUIRED_ENV_VARS = {
    "HF_API_KEY": "Your HuggingFace API key for zero-shot classification",
    "PROMPTLAYER_API_KEY": "Your PromptLayer API key for observability",
}

def validate_env_vars() -> None:
    """
    Validates that all required environment variables are set.
    Raises EnvVarError with detailed instructions if any are missing.
    """
    load_dotenv()  # Try to load from .env file if it exists

    missing_vars: Dict[str, str] = {}

    for var, description in REQUIRED_ENV_VARS.items():
        if not os.getenv(var):
            missing_vars[var] = description

    if missing_vars:
        error_msg = ["Missing required environment variables:"]
        error_msg.extend([f"\n- {var}: {desc}" for var, desc in missing_vars.items()])
        error_msg.append("\n\nTo fix this:")
        error_msg.append("\n1. Create a .env file in your project root")
        error_msg.append("\n2. Add the following lines to your .env file:")
        for var in missing_vars:
            error_msg.append(f"\n   {var}=your_{var.lower()}_here")
        error_msg.append("\n\nMake sure to replace the placeholder values with your actual API keys.")
        raise EnvVarError("".join(error_msg))

def get_env_var(var_name: str) -> str:
    """
    Safely retrieves an environment variable, with validation.

    Args:
        var_name: Name of the environment variable to retrieve

    Returns:
        str: The value of the environment variable

    Raises:
        EnvVarError: If the variable is not set
    """
    value = os.getenv(var_name)
    if not value:
        if var_name in REQUIRED_ENV_VARS:
            validate_env_vars()  # This will raise a more detailed error
        raise EnvVarError(f"Environment variable {var_name} is not set")
    return value
