import logging
from pathlib import Path
from llm_router.councils.parallel import ParallelCouncil
from llm_router.selectors.heuristics import HeuristicsSelector
from llm_router.selectors.classifier import HFZeroShotSelector
from llm_router.selectors.slm import SLMSelector
from llm_router.routers.router import LLMRouterService
from llm_router.schemas.council_schemas import LLMRouterResponse
from llm_router.providers import OpenAIProvider

# Get the project root directory (2 levels up from this file)
ROOT_DIR = Path(__file__).parent.parent.parent
ENV_PATH = ROOT_DIR / '.env'


def main() -> None:
    selectors = [
        HeuristicsSelector(provider_name="openai"),
        HFZeroShotSelector(provider_name="openai"),
        SLMSelector(provider_name="openai"),
    ]
    council = ParallelCouncil(selectors=selectors, provider_name="openai")

    # Check if Path is valid
    if not Path(ENV_PATH).exists():
        raise FileNotFoundError(f".env file not found at {ENV_PATH}")

    provider = OpenAIProvider(env_path=ENV_PATH)
    router_service = LLMRouterService(council=council, env_path=ENV_PATH, provider=provider)

    prompt = "What is the capital of France?"
    response: LLMRouterResponse = router_service.invoke(prompt)

    print("Model:", response.model)
    print("Prompt:", response.prompt)
    print("Response:", response.response)
    print("Cost:", response.cost)
    print("Latency:", response.latency)
    if response.metadata:
        print("Metadata:")
        print("  Votes:", response.metadata.votes)
        print("  Weighted Results:", response.metadata.weighted_results)
        print("  Tags:", response.metadata.tags)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
