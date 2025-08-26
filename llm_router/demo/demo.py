from llm_router.councils.random import RandomCouncil
from llm_router.selectors.heuristics import HeuristicsSelector
from llm_router.routers.router import LLMRouterService
from llm_router.schemas.council_schemas import LLMRouterResponse

# Instantiate a selector and council
selector = HeuristicsSelector()
council = RandomCouncil(selectors=[selector])

# Create the router service (API key can be set via .env)
router_service = LLMRouterService(council=council)

# Send a prompt
prompt = "What is the capital of France?"
response: LLMRouterResponse = router_service.invoke(prompt)

# Print the structured response
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

