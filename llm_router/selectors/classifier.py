from llm_router.schemas.council_schemas import SelectorVote
from llm_router.schemas.config import CANDIDATE_LABELS, HF_HEADERS, HF_API_URL, TOPIC_TO_MODEL
import requests
import json


class HFZeroShotSelector:
    def select_model(self, prompt: str) -> SelectorVote:
        payload = {"inputs": prompt, "parameters": {"candidate_labels": CANDIDATE_LABELS}}
        try:
            resp = requests.post(HF_API_URL, headers=HF_HEADERS, json=payload, timeout=30)
            if resp.status_code != 200:
                return SelectorVote(
                    selector_name="HFZeroShotSelector",
                    model="ollama/phi3:latest",
                    rationale=f"HF API error: status code {resp.status_code}. Proposing default model."
                )
            try:
                result = resp.json()
            except (ValueError, json.JSONDecodeError):
                return SelectorVote(
                    selector_name="HFZeroShotSelector",
                    model="ollama/phi3:latest",
                    rationale="HF API error: invalid JSON response. Proposing default model."
                )
        except requests.RequestException as e:
            return SelectorVote(
                selector_name="HFZeroShotSelector",
                model="ollama/phi3:latest",
                rationale=f"HF API request failed: {str(e)}. Proposing default model."
            )
        result = resp.json()

        if "labels" in result:
            top_label = result["labels"][0]
            mapped_model = TOPIC_TO_MODEL.get(top_label, "ollama/phi3:latest")
            return SelectorVote(
                selector_name="HFZeroShotSelector",
                model=mapped_model,
                rationale=f"Zero-shot classified as {top_label}"
            )

        return SelectorVote(
            selector_name="HFZeroShotSelector",
            model="ollama/phi3:latest",
            rationale="Fallback default model"
        )
