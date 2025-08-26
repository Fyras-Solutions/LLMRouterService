import logging

from llm_router.councils.parallel import ParallelCouncil
from llm_router.selectors.heuristics import HeuristicsSelector
from llm_router.selectors.classifier import HFZeroShotSelector
from llm_router.selectors.slm import SLMSelector
from llm_router.selectors.length_selector import PromptLengthSelector
from llm_router.routers.router import LLMRouterService
from llm_router.schemas.council_schemas import LLMRouterResponse


def main() -> None:
    selectors = [
        HeuristicsSelector(),
        PromptLengthSelector(),
        HFZeroShotSelector(),
        SLMSelector(),
    ]
    council = ParallelCouncil(selectors=selectors)
    router_service = LLMRouterService(council=council)

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

