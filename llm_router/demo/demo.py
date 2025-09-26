import logging
from pathlib import Path
from llm_router.selectors.heuristics import HeuristicsSelector
from llm_router.selectors.classifier import HFZeroShotSelector
from llm_router.routers.router import LLMRouterService
from llm_router.schemas.council_schemas import LLMRouterResponse
from llm_router.providers import OpenAIProvider, AnthropicProvider,GoogleProvider
from llm_router.councils import ParallelCouncil,RandomCouncil,CascadeCouncil,WeightedMajorityVoteCouncil,UnanimousCouncil

import time
# Get the project root directory (2 levels up from this file)
ROOT_DIR = Path(__file__).parent.parent.parent
ENV_PATH = ROOT_DIR / '.env'


def main() -> None:
    
    tenant_providers=["openai","google"]
    default_provider = None #if tenant have any default provider
    
    # Check default Provider    
    if default_provider is None:
        user_provider=input("Enter your provider(openai/google/anthropic):")
    else:
        user_provider = default_provider
    
    # Check Providers in Tenant DB
    if user_provider not in tenant_providers:
        print("You don't have access for this provider:", user_provider)
        user_provider = "google" # defalut (if tenant Didn't have that provider it assign google)
        print("Continue with default provider",user_provider) 
         
    selectors = [
        HFZeroShotSelector(provider_name=user_provider),
        HeuristicsSelector(provider_name=user_provider)
    ]
    
    council = RandomCouncil(selectors=selectors)
    
    # Check if Path is valid
    if not Path(ENV_PATH).exists():
        raise FileNotFoundError(f".env file not found at {ENV_PATH}")
    
    match user_provider:
        case "openai":
            provider = OpenAIProvider(env_path=ENV_PATH)
        case "google":
            provider = GoogleProvider(env_path=ENV_PATH)
        case "anthropic":
            provider = AnthropicProvider(env_path=ENV_PATH)
    
    router_service = LLMRouterService(council=council, env_path=ENV_PATH, provider=provider)
    
    while True:
        prompt = input("Enter your Prompt:")
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
