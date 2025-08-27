# LLMRouterService

A modular, extensible service for orchestrating and routing requests to multiple Large Language Models (LLMs) using council-based decision logic, selectors, and robust metadata tracking.

## Architecture Overview

```mermaid
graph TD
    A[Client Request] -->|Prompt| B[LLMRouterService]
    B --> C[Council]
    C --> D[Selectors]
    C --> E[CouncilDecision]
    B --> F[LLM Execution]
    F --> G[LLMResponse]
    G --> H[RouterMetadata]
    H --> I[LLMRouterResponse]
    I --> J[Client Response]
```

## Components

### Councils (`llm_router/councils/`)
- **Purpose:** Aggregate votes from selectors and make a final model selection.
- **Files:**
  - `random.py`: Randomized council decision logic.
  - `iterative.py`: Iterative voting and selection.
  - `weighted.py`: Weighted and unanimous voting.
  - `parallel.py`: Queries selectors and chooses the majority model.

### Selectors (`llm_router/selectors/`)
- **Purpose:** Implement model selection strategies (heuristics, classifiers, SLMs).
- **Files:**
  - `classifier.py`: HuggingFace zero-shot classifier.
  - `heuristics.py`: Heuristic-based selection.
  - `slm.py`: Small language model selector.

### Routers (`llm_router/routers/`)
- **Purpose:** Main service interface for routing requests, executing LLM calls, and logging.
- **Files:**
  - `router.py`: Router service with PromptLayer logging.

### Schemas (`llm_router/schemas/`)
- **Purpose:** Define data contracts for council decisions, LLM responses, and metadata.
- **Files:**
  - `council_schemas.py`: Main schemas for responses and decisions.
  - `abstractions.py`, `config.py`: Abstract base classes and configuration schemas.

### Exceptions (`llm_router/exceptions/`)
- **Purpose:** Custom exception handling for router and council logic.
- **Files:**
  - `exceptions.py`: Exception definitions and hierarchy.

## API Contracts

### Schemas

#### `LLMResponse`
```python
class LLMResponse(BaseModel):
    model: str
    prompt: str
    response: str
    cost: float
    latency: float
```

#### `RouterMetadata`
```python
class RouterMetadata(BaseModel):
    votes: Optional[List[Dict[str, Any]]] = None
    weighted_results: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None
```

#### `LLMRouterResponse`
```python
class LLMRouterResponse(LLMResponse):
    metadata: Optional[RouterMetadata] = None
```

#### `CouncilDecision`
```python
class CouncilDecision(BaseModel):
    final_model: str
    votes: List[SelectorVote]
    weighted_results: Dict[str, float]
    metadata: Dict[str, str] = {}
```

### Main Service: `LLMRouterService`

#### Methods
- `invoke(prompt: str) -> LLMRouterResponse`
  - **Input:** Prompt string
  - **Output:** Structured LLMRouterResponse with metadata

#### Example Request
```json
{
  "prompt": "What is the capital of France?"
}
```

#### Example Response
```json
{
  "model": "gpt-3.5-turbo",
  "prompt": "What is the capital of France?",
  "response": "The capital of France is Paris.",
  "cost": 0.002,
  "latency": 0.45,
  "metadata": {
    "votes": [
      {"selector_name": "HeuristicsSelector", "model": "gpt-3.5-turbo", "weight": 1.0, "rationale": "Best for general knowledge."}
    ],
    "weighted_results": {"gpt-3.5-turbo": 1.0},
    "tags": ["council-router", "local-llm"]
  }
}
```

## Usage

1. **Install dependencies:**
   ```bash
   pip install -e .
   ```
2. **Configure environment:**
   - Set your LLM API keys in `.env`.
3. **Run the service:**
   - Call `LLMRouterService.invoke` with your prompt to route and execute a request.
   - Progress bars are displayed via `tqdm` and logging is emitted with Python's `logging` module.

## Extending
- Add new selectors or councils by implementing the appropriate base classes in `schemas/abstractions.py`.
- Customize routing logic in `routers/router.py`.

## Testing
- Unit tests are located in the `tests/` directory.
- See `COVERAGE.md` for last written coverage reports.

## License
MIT

