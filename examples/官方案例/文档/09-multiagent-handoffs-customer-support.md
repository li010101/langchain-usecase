# Build customer support with handoffs

## Overview

The **state machine pattern** describes workflows where an agent's behavior changes as it moves through different states. This tutorial shows how to implement a state machine by using tool calls to dynamically change a single agent's configuration.

In this tutorial, you'll build a customer support agent that:

* Collects warranty information before proceeding
* Classifies issues as hardware or software
* Provides solutions or escalates to human support
* Maintains conversation state across multiple turns

The **state machine pattern** uses a single agent whose configuration changes based on workflow progress.

### Workflow

```
Customer reports issue → Warranty Check → Issue Classification → Resolution → Close
```

## Setup

```bash
pip install langchain
```

## 1. Define custom state

Define a custom state schema:

```python
from langchain.agents import AgentState
from typing_extensions import NotRequired
from typing import Literal

SupportStep = Literal["warranty_collector", "issue_classifier", "resolution_specialist"]

class SupportState(AgentState):
    """State for customer support workflow."""
    current_step: NotRequired[SupportStep]
    warranty_status: NotRequired[Literal["in_warranty", "out_of_warranty"]]
    issue_type: NotRequired[Literal["hardware", "software"]]
```

## 2. Create tools that manage workflow state

Create tools that update the workflow state:

```python
from langchain.tools import tool, ToolRuntime
from langchain.messages import ToolMessage
from langgraph.types import Command

@tool
def record_warranty_status(
    status: Literal["in_warranty", "out_of_warranty"],
    runtime: ToolRuntime[None, SupportState],
) -> Command:
    """Record the customer's warranty status and transition to issue classification."""
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=f"Warranty status recorded as: {status}",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
            "warranty_status": status,
            "current_step": "issue_classifier",
        }
    )

@tool
def record_issue_type(
    issue_type: Literal["hardware", "software"],
    runtime: ToolRuntime[None, SupportState],
) -> Command:
    """Record the type of issue and transition to resolution specialist."""
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=f"Issue type recorded as: {issue_type}",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
            "issue_type": issue_type,
            "current_step": "resolution_specialist",
        }
    )

@tool
def escalate_to_human(reason: str) -> str:
    """Escalate the case to a human support specialist."""
    return f"Escalating to human support. Reason: {reason}"

@tool
def provide_solution(solution: str) -> str:
    """Provide a solution to the customer's issue."""
    return f"Solution provided: {solution}"
```

## 3. Define step configurations

Define prompts and tools for each step:

```python
WARRANTY_COLLECTOR_PROMPT = """You are a customer support agent.

CURRENT STAGE: Warranty verification

At this step:
1. Greet the customer warmly
2. Ask if their device is under warranty
3. Use record_warranty_status to record their response"""

ISSUE_CLASSIFIER_PROMPT = """You are a customer support agent.

CURRENT STAGE: Issue classification
CUSTOMER INFO: Warranty status is {warranty_status}

At this step:
1. Ask the customer to describe their issue
2. Determine if it's hardware or software
3. Use record_issue_type to record the classification"""

RESOLUTION_SPECIALIST_PROMPT = """You are a customer support agent.

CURRENT STAGE: Resolution
CUSTOMER INFO: Warranty status is {warranty_status}, issue type is {issue_type}

At this step:
1. For SOFTWARE issues: provide troubleshooting steps
2. For HARDWARE issues:
   - If IN WARRANTY: explain warranty repair process
   - If OUT OF WARRANTY: escalate_to_human"""

STEP_CONFIG = {
    "warranty_collector": {
        "prompt": WARRANTY_COLLECTOR_PROMPT,
        "tools": [record_warranty_status],
        "requires": [],
    },
    "issue_classifier": {
        "prompt": ISSUE_CLASSIFIER_PROMPT,
        "tools": [record_issue_type],
        "requires": ["warranty_status"],
    },
    "resolution_specialist": {
        "prompt": RESOLUTION_SPECIALIST_PROMPT,
        "tools": [provide_solution, escalate_to_human],
        "requires": ["warranty_status", "issue_type"],
    },
}
```

## 4. Create step-based middleware

Create middleware that applies the appropriate configuration:

```python
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from typing import Callable

@wrap_model_call
def apply_step_config(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """Configure agent behavior based on the current step."""
    current_step = request.state.get("current_step", "warranty_collector")
    stage_config = STEP_CONFIG[current_step]

    # Validate required state exists
    for key in stage_config["requires"]:
        if request.state.get(key) is None:
            raise ValueError(f"{key} must be set before reaching {current_step}")

    # Format prompt with state values
    system_prompt = stage_config["prompt"].format(**request.state)

    # Inject system prompt and step-specific tools
    request = request.override(
        system_prompt=system_prompt,
        tools=stage_config["tools"],
    )

    return handler(request)
```

## 5. Create the agent

```python
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

all_tools = [
    record_warranty_status,
    record_issue_type,
    provide_solution,
    escalate_to_human,
]

agent = create_agent(
    model,
    tools=all_tools,
    state_schema=SupportState,
    middleware=[apply_step_config],
    checkpointer=InMemorySaver(),
)
```

## 6. Test the workflow

```python
import uuid
from langchain.messages import HumanMessage

thread_id = str(uuid.uuid4())
config = {"configurable": {"thread_id": thread_id}}

# Turn 1
result = agent.invoke(
    {"messages": [HumanMessage("Hi, my phone screen is cracked")]},
    config
)

# Turn 2
result = agent.invoke(
    {"messages": [HumanMessage("Yes, it's still under warranty")]},
    config
)

# Turn 3
result = agent.invoke(
    {"messages": [HumanMessage("The screen is physically cracked from dropping it")]},
    config
)

# Turn 4
result = agent.invoke(
    {"messages": [HumanMessage("What should I do?")]},
    config
)
```

Flow:
1. **Warranty verification**: Ask about warranty status
2. **Issue classification**: Determine hardware or software issue
3. **Resolution**: Provide warranty repair instructions

## 7. State transitions

Tools drive workflow by updating `current_step`:

```python
# After recording warranty
Command(update={
    "warranty_status": "in_warranty",
    "current_step": "issue_classifier"  # State transition!
})

# After recording issue type
Command(update={
    "issue_type": "hardware",
    "current_step": "resolution_specialist"  # State transition!
})
```

## 8. Add flexibility: Go back

Allow users to return to previous steps:

```python
@tool
def go_back_to_warranty() -> Command:
    """Go back to warranty verification step."""
    return Command(update={"current_step": "warranty_collector"})

@tool
def go_back_to_classification() -> Command:
    """Go back to issue classification step."""
    return Command(update={"current_step": "issue_classifier"})

STEP_CONFIG["resolution_specialist"]["tools"].extend([
    go_back_to_warranty,
    go_back_to_classification
])
```

## 9. Manage message history

Use summarization to compress long conversations:

```python
from langchain.agents.middleware import SummarizationMiddleware

agent = create_agent(
    model,
    tools=all_tools,
    state_schema=SupportState,
    middleware=[
        apply_step_config,
        SummarizationMiddleware(
            model="gpt-4.1-mini",
            trigger=("tokens", 4000),
            keep=("messages", 10)
        )
    ],
    checkpointer=InMemorySaver(),
)
```

## Key takeaways

* **State machine pattern**: Single agent with dynamic configuration
* **Tools drive workflow**: State transitions via tool calls
* **Middleware responds**: Applies appropriate configuration on each turn

**When to use:**
- Sequential information collection workflows
- Multi-step processes with clear stages
- Need for step-specific prompts and tools

## Next steps

* Learn about the [subagents pattern](/oss/python/langchain/multi-agent/subagents-personal-assistant)
* Explore [middleware](/oss/python/langchain/middleware)
* Read the [multi-agent overview](/oss/python/langchain/multi-agent)
