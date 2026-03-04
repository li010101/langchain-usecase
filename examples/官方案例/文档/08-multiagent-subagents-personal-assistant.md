# Build a personal assistant with subagents

## Overview

The **supervisor pattern** is a [multi-agent](/oss/python/langchain/multi-agent) architecture where a central supervisor agent coordinates specialized worker agents.

In this tutorial, you'll build a personal assistant that coordinates two specialists:
* A **calendar agent** for scheduling, availability checking, and event management
* An **email agent** for communication, drafting messages, and sending notifications

We'll also incorporate human-in-the-loop review for approval of sensitive actions.

### Why use a supervisor?

Multi-agent architectures allow you to partition tools across workers, each with their own prompts. Consider an agent with all calendar and email APIs - it must choose from many similar tools. Separating related tools into logical groups can improve performance.

### Concepts

We will cover:
* [Multi-agent systems](/oss/python/langchain/multi-agent)
* [Human-in-the-loop review](/oss/python/langchain/human-in-the-loop)

## Setup

```bash
pip install langchain
```

## 1. Define tools

Define tools with structured inputs:

```python
from langchain.tools import tool

@tool
def create_calendar_event(
    title: str,
    start_time: str,       # ISO format: "2024-01-15T14:00:00"
    end_time: str,         # ISO format: "2024-01-15T15:00:00"
    attendees: list[str],  # email addresses
    location: str = ""
) -> str:
    """Create a calendar event."""
    return f"Event created: {title} from {start_time} to {end_time}"

@tool
def send_email(
    to: list[str],
    subject: str,
    body: str,
    cc: list[str] = []
) -> str:
    """Send an email."""
    return f"Email sent to {', '.join(to)} - Subject: {subject}"

@tool
def get_available_time_slots(
    attendees: list[str],
    date: str,
    duration_minutes: int
) -> list[str]:
    """Check calendar availability."""
    return ["09:00", "14:00", "16:00"]
```

## 2. Create specialized sub-agents

### Create a calendar agent

```python
from langchain.agents import create_agent

CALENDAR_AGENT_PROMPT = (
    "You are a calendar scheduling assistant. "
    "Parse natural language scheduling requests into proper ISO datetime formats. "
    "Use get_available_time_slots to check availability. "
    "Use create_calendar_event to schedule events. "
    "Always confirm what was scheduled in your final response."
)

calendar_agent = create_agent(
    model,
    tools=[create_calendar_event, get_available_time_slots],
    system_prompt=CALENDAR_AGENT_PROMPT,
)
```

### Create an email agent

```python
EMAIL_AGENT_PROMPT = (
    "You are an email assistant. "
    "Compose professional emails based on natural language requests. "
    "Extract recipient information and craft appropriate subject lines and body text. "
    "Use send_email to send the message."
)

email_agent = create_agent(
    model,
    tools=[send_email],
    system_prompt=EMAIL_AGENT_PROMPT,
)
```

## 3. Wrap sub-agents as tools

Now wrap each sub-agent as a tool:

```python
@tool
def schedule_event(request: str) -> str:
    """Schedule calendar events using natural language."""
    result = calendar_agent.invoke({
        "messages": [{"role": "user", "content": request}]
    })
    return result["messages"][-1].text

@tool
def manage_email(request: str) -> str:
    """Send emails using natural language."""
    result = email_agent.invoke({
        "messages": [{"role": "user", "content": request}]
    })
    return result["messages"][-1].text
```

## 4. Create the supervisor agent

```python
SUPERVISOR_PROMPT = (
    "You are a helpful personal assistant. "
    "You can schedule calendar events and send emails. "
    "Break down user requests into appropriate tool calls and coordinate the results."
)

supervisor_agent = create_agent(
    model,
    tools=[schedule_event, manage_email],
    system_prompt=SUPERVISOR_PROMPT,
)
```

## 5. Use the supervisor

### Example 1: Simple request

```python
query = "Schedule a team standup for tomorrow at 9am"

for step in supervisor_agent.stream(
    {"messages": [{"role": "user", "content": query}]}
):
    for update in step.values():
        for message in update.get("messages", []):
            message.pretty_print()
```

### Example 2: Multi-domain request

```python
query = (
    "Schedule a meeting with the design team next Tuesday at 2pm for 1 hour, "
    "and send them an email reminder about reviewing the new mockups."
)

for step in supervisor_agent.stream(
    {"messages": [{"role": "user", "content": query}]}
):
    # Supervisor coordinates both calendar and email agents
    pass
```

## 6. Add human-in-the-loop review

Add middleware to pause for review:

```python
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver

calendar_agent = create_agent(
    model,
    tools=[create_calendar_event, get_available_time_slots],
    system_prompt=CALENDAR_AGENT_PROMPT,
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={"create_calendar_event": True},
            description_prefix="Calendar event pending approval",
        ),
    ],
)

email_agent = create_agent(
    model,
    tools=[send_email],
    system_prompt=EMAIL_AGENT_PROMPT,
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={"send_email": True},
            description_prefix="Outbound email pending approval",
        ),
    ],
)

supervisor_agent = create_agent(
    model,
    tools=[schedule_event, manage_email],
    system_prompt=SUPERVISOR_PROMPT,
    checkpointer=InMemorySaver(),
)
```

Resume with approval:

```python
from langgraph.types import Command

# Resume with decisions
Command(resume={"decisions": [{"type": "approve"}]})
```

Or edit before approving:

```python
# Edit the action before executing
Command(resume={
    "decisions": [{
        "type": "edit",
        "edited_action": edited_action
    }]
})
```

## 7. Advanced: Control information flow

### Pass additional context to sub-agents

```python
from langchain.tools import tool, ToolRuntime

@tool
def schedule_event(request: str, runtime: ToolRuntime) -> str:
    """Schedule calendar events using natural language."""
    original_user_message = next(
        message for message in runtime.state["messages"]
        if message.type == "human"
    )
    prompt = (
        "You are assisting with the following user inquiry:\n\n"
        f"{original_user_message.text}\n\n"
        "You are tasked with:\n\n"
        f"{request}"
    )
    result = calendar_agent.invoke({
        "messages": [{"role": "user", "content": prompt}],
    })
    return result["messages"][-1].text
```

## 8. Key takeaways

The supervisor pattern creates layers of abstraction:
- **Bottom layer**: Rigid API tools requiring exact formats
- **Middle layer**: Sub-agents accepting natural language
- **Top layer**: Supervisor routing to high-level capabilities

**When to use the supervisor pattern:**
- Multiple distinct domains (calendar, email, CRM)
- Each domain has multiple tools or complex logic
- Want centralized workflow control
- Sub-agents don't need to converse directly with users

## Next steps

* Learn about [handoffs](/oss/python/langchain/multi-agent/handoffs) for agent-to-agent conversations
* Explore [context engineering](/oss/python/langchain/context-engineering)
* Read the [multi-agent overview](/oss/python/langchain/multi-agent)
