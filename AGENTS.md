# AGENTS.md - LangChain Use Cases Repository

## Project Overview

This is a LangChain learning repository containing various use case examples across multiple domains (finance, healthcare, legal, customer service). The repository collects and documents AI application examples built with LangChain, LangGraph, and related tools.

---

## Development Commands

### Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies (when pyproject.toml exists)
pip install -e .
```

### Running Code

```bash
# Run a Python script
python examples/<example-name>/main.py

# Run with specific environment variables
OPENAI_API_KEY=sk-xxx python examples/<example-name>/main.py
```

### Testing

```bash
# Run all tests (when pytest is configured)
pytest

# Run a single test file
pytest tests/test_example.py

# Run a single test function
pytest tests/test_example.py::test_function_name

# Run tests with verbose output
pytest -v

# Run tests matching a pattern
pytest -k "test_pattern"
```

### Linting & Formatting

```bash
# Run ruff linter
ruff check .

# Run ruff with auto-fix
ruff check --fix .

# Format code with ruff
ruff format .

# Run mypy type checking
mypy .

# Run all checks
ruff check . && ruff format --check . && mypy .
```

---

## Code Style Guidelines

### General Principles

- Write clean, readable code with clear variable and function names
- Keep functions focused on a single responsibility
- Add type hints to all function signatures
- Use docstrings for classes and complex functions
- Avoid hardcoding API keys; use environment variables

### Imports

```python
# Standard library first
import os
import sys
from typing import List, Optional

# Third-party packages
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Local imports
from .utils import helper_function
from ..models import DataModel
```

### Naming Conventions

- **Variables/Functions**: `snake_case` (e.g., `get_response`, `user_input`)
- **Classes**: `PascalCase` (e.g., `FinancialAnalyzer`, `AgentRouter`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `MAX_TOKENS`, `DEFAULT_TEMPERATURE`)
- **Private methods/attributes**: Prefix with `_` (e.g., `_internal_method`)
- **Files**: `snake_case.py` (e.g., `rag_pipeline.py`, `agent_tools.py`)

### Type Hints

Always use type hints:

```python
from typing import List, Dict, Optional, Any

def process_query(
    query: str,
    max_tokens: Optional[int] = None,
    temperature: float = 0.7
) -> str:
    """Process user query and return response."""
    ...
    
def analyze_documents(docs: List[str]) -> Dict[str, float]:
    ...
```

### Error Handling

Use specific exceptions and meaningful error messages:

```python
try:
    result = llm.invoke(prompt)
except ValueError as e:
    logger.error(f"Invalid prompt format: {e}")
    raise
except Exception as e:
    logger.error(f"Unexpected error during LLM invocation: {e}")
    raise
```

### LangChain Best Practices

1. **Prompt Templates**: Use `ChatPromptTemplate` for structured prompts
2. **Chain Composition**: Use LangChain Expression Language (LCEL)
3. **Tool Definition**: Define tools with proper descriptions for LLM understanding
4. **Streaming**: Implement streaming for better UX when applicable
5. **Token Tracking**: Monitor token usage for cost control
6. **Async Support**: Use `async`/`await` for I/O-bound operations

### Configuration

Store configuration in environment variables or config files:

```python
from pydantic_settings import BaseSettings

class Config(BaseSettings):
    openai_api_key: str
    model_name: str = "gpt-4"
    temperature: float = 0.7
    
    class Config:
        env_file = ".env"
```

### Testing Guidelines

- Place tests in `tests/` directory mirroring source structure
- Use descriptive test names: `test_<function>_returns_expected_format`
- Mock external API calls (LLMs, tools)
- Test error handling paths
- Use fixtures for common setup

```python
import pytest
from unittest.mock import Mock, patch

@pytest.fixture
def mock_llm():
    llm = Mock()
    llm.invoke.return_value = "Mocked response"
    return llm

def test_rag_pipeline_returns_context(mock_llm):
    result = rag_pipeline.invoke("test query")
    assert "context" in result
```

### Logging

Use structured logging:

```python
import logging

logger = logging.getLogger(__name__)

logger.info(f"Processing request: {request_id}")
logger.error(f"Failed to process request: {error}")
```

---

## Project Structure

```
langchain-usecase/
├── examples/           # Example implementations
│   └── <example-name>/
│       ├── main.py
│       ├── utils/
│       └── tests/
├── AGENTS.md          # This file
├── README.md
└── .env.example       # Environment template
```

---

## Environment Variables

Required environment variables (create `.env` file):

```bash
# OpenAI
OPENAI_API_KEY=sk-...

# Optional: Anthropic
ANTHROPIC_API_KEY=sk-...

# Optional: Other providers
OLLAMA_BASE_URL=http://localhost:11434
```

---

## Additional Notes

- Always review API costs before running examples
- Use rate limiting for production deployments
- Keep API keys out of version control
- Test with smaller models before scaling up
- Use LangSmith for debugging and monitoring chains

---

*Last updated: 2026-03-03*
