# Build a multi-source knowledge base with routing

## Overview

The **router pattern** is a [multi-agent](/oss/python/langchain/multi-agent) architecture where a routing step classifies input and directs it to specialized agents, with results synthesized into a combined response.

In this tutorial, you'll build a multi-source knowledge base router with three specialists:
* A **GitHub agent** for code, issues, and PRs
* A **Notion agent** for documentation
* A **Slack agent** for discussions

When a user asks "How do I authenticate API requests?", the router:
1. Decomposes the query into source-specific sub-questions
2. Routes them to relevant agents in parallel
3. Synthesizes results into a coherent answer

### Architecture

```
Query → Classify → [GitHub agent] ─┐
                [Notion agent]  ──┼→ Synthesize → Combined Answer
                [Slack agent]  ──┘
```

### Why use a router?

* **Parallel execution**: Query multiple sources simultaneously
* **Specialized agents**: Each vertical has focused tools
* **Selective routing**: Intelligently select relevant verticals
* **Targeted sub-questions**: Better result quality
* **Clean synthesis**: Combined coherent response

## Setup

```bash
pip install langchain langgraph
```

## 1. Define state

```python
from typing import Annotated, Literal, TypedDict
import operator

class AgentInput(TypedDict):
    """Simple input state for each subagent."""
    query: str

class AgentOutput(TypedDict):
    """Output from each subagent."""
    source: str
    result: str

class Classification(TypedDict):
    """A single routing decision."""
    source: Literal["github", "notion", "slack"]
    query: str

class RouterState(TypedDict):
    query: str
    classifications: list[Classification]
    results: Annotated[list[AgentOutput], operator.add]
    final_answer: str
```

## 2. Define tools for each vertical

```python
from langchain.tools import tool

@tool
def search_code(query: str, repo: str = "main") -> str:
    """Search code in GitHub repositories."""
    return f"Found code matching '{query}' in {repo}"

@tool
def search_issues(query: str) -> str:
    """Search GitHub issues and pull requests."""
    return f"Found issues matching '{query}'"

@tool
def search_prs(query: str) -> str:
    """Search pull requests."""
    return f"Found PRs matching '{query}'"

@tool
def search_notion(query: str) -> str:
    """Search Notion workspace."""
    return f"Found documentation: '{query}'"

@tool
def get_page(page_id: str) -> str:
    """Get a specific Notion page."""
    return f"Page content: {page_id}"

@tool
def search_slack(query: str) -> str:
    """Search Slack messages."""
    return f"Found discussion matching '{query}'"

@tool
def get_thread(thread_id: str) -> str:
    """Get a specific Slack thread."""
    return f"Thread: {thread_id}"
```

## 3. Create specialized agents

```python
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

model = init_chat_model("openai:gpt-4.1")

github_agent = create_agent(
    model,
    tools=[search_code, search_issues, search_prs],
    system_prompt=(
        "You are a GitHub expert. Answer questions about code, "
        "API references, and implementation details."
    ),
)

notion_agent = create_agent(
    model,
    tools=[search_notion, get_page],
    system_prompt=(
        "You are a Notion expert. Answer questions about internal "
        "documentation and processes."
    ),
)

slack_agent = create_agent(
    model,
    tools=[search_slack, get_thread],
    system_prompt=(
        "You are a Slack expert. Answer questions by searching "
        "relevant discussions."
    ),
)
```

## 4. Build the router workflow

### Classify query

```python
from pydantic import BaseModel, Field

class ClassificationResult(BaseModel):
    """Result of classifying a user query."""
    classifications: list[Classification] = Field(
        description="List of agents to invoke"
    )

router_llm = init_chat_model("openai:gpt-4.1-mini")

def classify_query(state: RouterState) -> dict:
    """Classify query and determine which agents to invoke."""
    structured_llm = router_llm.with_structured_output(ClassificationResult)

    result = structured_llm.invoke([
        {
            "role": "system",
            "content": """Analyze this query and determine which knowledge bases to consult.
Available sources:
- github: Code, API references, implementation details
- notion: Internal documentation, processes
- slack: Team discussions
Return ONLY relevant sources."""
        },
        {"role": "user", "content": state["query"]}
    ])

    return {"classifications": result.classifications}
```

### Route to agents

```python
from langgraph.types import Send

def route_to_agents(state: RouterState) -> list[Send]:
    """Fan out to agents based on classifications."""
    return [
        Send(c["source"], {"query": c["query"]})
        for c in state["classifications"]
    ]
```

### Query agents

```python
def query_github(state: AgentInput) -> dict:
    result = github_agent.invoke({
        "messages": [{"role": "user", "content": state["query"]}]
    })
    return {"results": [{"source": "github", "result": result["messages"][-1].content}]}

def query_notion(state: AgentInput) -> dict:
    result = notion_agent.invoke({
        "messages": [{"role": "user", "content": state["query"]}]
    })
    return {"results": [{"source": "notion", "result": result["messages"][-1].content}]}

def query_slack(state: AgentInput) -> dict:
    result = slack_agent.invoke({
        "messages": [{"role": "user", "content": state["query"]}]
    })
    return {"results": [{"source": "slack", "result": result["messages"][-1].content}]}
```

### Synthesize results

```python
def synthesize_results(state: RouterState) -> dict:
    """Combine results from all agents."""
    if not state["results"]:
        return {"final_answer": "No results found."}

    formatted = [
        f"**{r['source'].title()}:**\n{r['result']}"
        for r in state["results"]
    ]

    synthesis_response = router_llm.invoke([
        {
            "role": "system",
            "content": f"""Synthesize these results to answer: "{state['query']}"
Combine info, highlight relevant details, keep concise."""
        },
        {"role": "user", "content": "\n\n".join(formatted)}
    ])

    return {"final_answer": synthesis_response.content}
```

## 5. Compile the workflow

```python
from langgraph.graph import StateGraph, START, END

workflow = (
    StateGraph(RouterState)
    .add_node("classify", classify_query)
    .add_node("github", query_github)
    .add_node("notion", query_notion)
    .add_node("slack", query_slack)
    .add_node("synthesize", synthesize_results)
    .add_edge(START, "classify")
    .add_conditional_edges("classify", route_to_agents, ["github", "notion", "slack"])
    .add_edge("github", "synthesize")
    .add_edge("notion", "synthesize")
    .add_edge("slack", "synthesize")
    .add_edge("synthesize", END)
    .compile()
)
```

## 6. Use the router

```python
result = workflow.invoke({
    "query": "How do I authenticate API requests?"
})

print("Original query:", result["query"])
print("\nClassifications:")
for c in result["classifications"]:
    print(f"  {c['source']}: {c['query']}")
print("\nFinal Answer:")
print(result["final_answer"])
```

Output:
```
Original query: How do I authenticate API requests?

Classifications:
  github: What authentication code exists?
  notion: What authentication documentation exists?

============================================================

Final Answer:
To authenticate API requests, you have several options:
1. JWT Tokens: Implementation in src/auth.py (PR #156)
2. OAuth2: Documented in Notion's API Auth Guide
3. API Keys: Use Bearer tokens
```

## 7. Understanding the architecture

### Classification phase
The `classify_query` function uses structured output to analyze the query and return relevant sources.

### Parallel execution with Send
```python
# Classifications become Send objects
[Send("github", {"query": "..."}), Send("notion", {"query": "..."})]
# Both agents execute simultaneously
```

### Result collection with reducers
```python
{"results": [{"source": "github", "result": "..."}]}
# Reducer (operator.add) concatenates lists
```

## 8. Advanced: Stateful routers

### Tool wrapper approach

Wrap the stateless router as a tool:

```python
from langgraph.checkpoint.memory import InMemorySaver

@tool
def search_knowledge_base(query: str) -> str:
    """Search across multiple knowledge sources."""
    result = workflow.invoke({"query": query})
    return result["final_answer"]

conversational_agent = create_agent(
    model,
    tools=[search_knowledge_base],
    system_prompt="Use search_knowledge_base to find information.",
    checkpointer=InMemorySaver(),
)
```

This keeps the router stateless while the conversational agent handles memory.

## Key takeaways

**Router pattern excels when:**
* Distinct verticals (separate knowledge domains)
* Parallel query needs
* Synthesis requirements

**Three phases:**
1. **Decompose**: Analyze query, generate targeted sub-questions
2. **Route**: Execute queries in parallel
3. **Synthesize**: Combine results

**When to use:**
- Multiple independent knowledge sources
- Need low-latency parallel queries
- Want explicit control over routing

**When NOT to use:**
- Dynamic tool selection → Use subagents pattern
- Agents need to converse with users → Use handoffs pattern

## Next steps

* Learn about [handoffs](/oss/python/langchain/multi-agent/handoffs)
* Explore [subagents pattern](/oss/python/langchain/multi-agent/subagents-personal-assistant)
* Read the [multi-agent overview](/oss/python/langchain/multi-agent)
