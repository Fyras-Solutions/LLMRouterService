import time
import json, os
import promptlayer
from litellm import completion, cost_per_token
from llm_router.schemas.council_schemas import LLMRouterResponse, RouterMetadata, CouncilDecision
from llm_router.schemas.abstractions import Council
from dotenv import load_dotenv


class LLMRouterService:
    def __init__(self, council: Council, api_key: str = None):
        self.council = council
        load_dotenv()

        if not api_key:
            api_key = os.getenv("PROMPTLAYER_API_KEY")

        self.pl_client = promptlayer.PromptLayer(api_key=api_key)

    def _execute(self, decision: CouncilDecision, prompt: str) -> LLMRouterResponse:
        """Execute call through LiteLLM + log with PromptLayer"""
        model = decision.final_model

        start = time.time()
        resp = completion(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        end = time.time()

        # Cost tracking (safe fallback for Ollama)
        try:
            cost = cost_per_token(
                model=model,
                prompt_tokens=resp.usage.prompt_tokens,
                completion_tokens=resp.usage.completion_tokens
            )
            if isinstance(cost, tuple) and cost:
                cost = float(cost[0])
        except Exception:
            cost = 0.0

        response_text = resp["choices"][0]["message"]["content"]

        # Build structured I/O for PromptLayer
        prompt_struct = {
            "content": [{"type": "text", "text": prompt}],
            "input_variables": [],
            "template_format": "f-string",
            "type": "completion"
        }
        response_struct = {
            "content": [{"type": "text", "text": response_text}],
            "input_variables": [],
            "template_format": "f-string",
            "type": "completion"
        }

        self.pl_client.log_request(
            provider="ollama",
            model=model,
            input=prompt_struct,
            output=response_struct,
            request_start_time=start,
            request_end_time=end,
            parameters={},
            tags=["council-router", "local-llm"],
            metadata={
                "votes": json.dumps([v.model_dump() for v in decision.votes]),
                "weighted_results": json.dumps(decision.weighted_results),
            },
            function_name="LLMRouterService._execute",
        )

        router_metadata = RouterMetadata(
            votes=[v.model_dump() for v in decision.votes],
            weighted_results=decision.weighted_results,
            tags=["council-router", "local-llm"]
        )

        return LLMRouterResponse(
            model=model,
            prompt=prompt,
            response=response_text,
            cost=cost,
            latency=end - start,
            metadata=router_metadata
        )

    def invoke(self, prompt: str) -> LLMRouterResponse:
        """Main entry point: ask council to decide, then execute"""
        decision = self.council.decide(prompt)
        return self._execute(decision, prompt)