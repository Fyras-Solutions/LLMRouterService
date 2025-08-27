"""Environment variable validation for LLM Router Service."""
import os
from typing import Dict, Optional
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

def validate_env_vars(env_path: Optional[Path] = None) -> None:
    """Validate required environment variables.

    Args:
        env_path: Optional path to a ``.env`` file to load.

    Raises:
        EnvVarError: If loading fails or variables are missing.
    """
    if env_path:
        if not load_dotenv(env_path):
            raise EnvVarError(
                f"Unable to load environment variables from {env_path}. "
                "Ensure the file exists and pass a valid Path object."
            )
    else:
        load_dotenv()  # Try default .env location

    missing_vars: Dict[str, str] = {
        var: desc for var, desc in REQUIRED_ENV_VARS.items() if not os.getenv(var)
    }

    if missing_vars:
        error_msg = ["Missing required environment variables:"]
        error_msg.extend(f"\n- {var}: {desc}" for var, desc in missing_vars.items())
        error_msg.append(
            "\n\nCreate a .env file with the above variables and pass its Path "
            "using the `env_path` parameter."
        )
        raise EnvVarError("".join(error_msg))


def get_env_var(var_name: str, env_path: Optional[Path] = None) -> str:
    """Safely retrieve an environment variable, with validation.

    Args:
        var_name: Name of the environment variable to retrieve.
        env_path: Optional path to a ``.env`` file to load if needed.

    Returns:
        str: The value of the environment variable.

    Raises:
        EnvVarError: If the variable is not set.
    """
    value = os.getenv(var_name)
    if not value:
        if var_name in REQUIRED_ENV_VARS:
            validate_env_vars(env_path)  # This will raise a more detailed error
        raise EnvVarError(f"Environment variable {var_name} is not set")
    return value
