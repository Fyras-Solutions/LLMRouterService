import tiktoken
from textstat import textstat
from llm_router.schemas.council_schemas import SelectorVote


class HeuristicsSelector:
    def select_model(self, prompt: str) -> SelectorVote:
        enc = tiktoken.get_encoding("cl100k_base")
        num_tokens = len(enc.encode(prompt))
        readability = textstat.flesch_kincaid_grade(prompt)

        if any(x in prompt.lower() for x in ["code", "python", "function", "class"]):
            return SelectorVote(
                selector_name="HeuristicsSelector",
                model="ollama/qwen2.5-coder:latest",
                rationale="Keyword match: code-related"
            )
        if any(x in prompt.lower() for x in ["solve", "integral", "equation", "math"]):
            return SelectorVote(
                selector_name="HeuristicsSelector",
                model="ollama/qwen2-math:latest",
                rationale="Keyword match: math-related"
            )
        if num_tokens < 15 and readability < 6:
            return SelectorVote(
                selector_name="HeuristicsSelector",
                model="ollama/gemma2:2b",
                rationale="Short/simple prompt"
            )
        if num_tokens < 80:
            return SelectorVote(
                selector_name="HeuristicsSelector",
                model="ollama/phi3:latest",
                rationale="Medium complexity"
            )
        return SelectorVote(
            selector_name="HeuristicsSelector",
            model="ollama/mistral:7b",
            rationale="Long/complex prompt"
        )
