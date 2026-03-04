# Build a data analysis agent

> Build an agent that analyzes data files, generates visualizations, and shares results

## Overview

This guide demonstrates how to build a data analysis agent using a [deep agent](/oss/python/deepagents). Data analysis tasks typically require planning, code execution, and working with artifacts such as scripts, reports, and plots—capabilities that deep agents are designed to handle.

The agent we'll build will:

1. Accept a CSV file for analysis
2. Perform exploratory data analysis and generate visualizations
3. Share results to a Slack channel

<Tip>
  The Slack integration is optional. The agent can be modified to save artifacts locally or share results through other channels.
</Tip>

### Key concepts

This tutorial covers:

* [Backends](/oss/python/deepagents/backends) for sandboxed code execution
* Custom [tools](/oss/python/langchain/tools) for external integrations

## Setup

### Installation

Install the core dependencies:

```bash
pip install deepagents
```

### Optional dependencies

For this tutorial, we'll use:

* [Slack Python SDK](https://docs.slack.dev/tools/python-slack-sdk/) for sharing results ([token setup](https://docs.slack.dev/authentication/tokens/))
* A [sandbox](/oss/python/deepagents/sandboxes) environment for code execution. See [available providers](/oss/python/deepagents/sandboxes#available-providers) for setup details

```bash
pip install slack-sdk
```

<Note>
  These services are optional, though a sandboxed environment is highly recommended for any production use. You can alternatively use the local shell backend (with important [security considerations](/oss/python/deepagents/backends#local-shell)) or download artifacts directly from the backend.
</Note>

### LangSmith

Many of the applications you build with LangChain will contain multiple steps with multiple invocations of LLM calls. As these applications get more complex, it becomes crucial to be able to inspect what exactly is going on inside your chain or agent. The best way to do this is with [LangSmith](https://smith.langchain.com).

After you sign up at the link above, make sure to set your environment variables to start logging traces:

```shell
export LANGSMITH_TRACING="true"
export LANGSMITH_API_KEY="..."
```

Or, set them in Python:

```python
import getpass
import os

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = getpass.getpass()
```

## Set up the backend

Deep agents use [backends](/oss/python/deepagents/backends) to execute code in sandboxed environments.

See [available providers](/oss/python/deepagents/sandboxes#available-providers) for setup details.

<Tabs>
  <Tab title="Daytona">
    <CodeGroup>
      ```bash
      pip install langchain-daytona
      ```

      ```bash
      uv add langchain-daytona
      ```
    </CodeGroup>

    ```python
    from daytona import Daytona

    from langchain_daytona import DaytonaSandbox

    sandbox = Daytona().create()
    backend = DaytonaSandbox(sandbox=sandbox)
    ```

    Verify the sandbox is ready:

    ```python
    result = backend.execute("echo ready")
    print(result)
    # ExecuteResponse(output='ready', exit_code=0, ...)
    ```
  </Tab>

  <Tab title="Modal">
    ```python
    import modal

    from langchain_modal import ModalSandbox

    app = modal.App.lookup("your-app")
    modal_sandbox = modal.Sandbox.create(app=app)
    backend = ModalSandbox(sandbox=modal_sandbox)
    ```
  </Tab>

  <Tab title="Runloop">
    <CodeGroup>
      ```bash
      pip install langchain-runloop
      ```

      ```bash
      uv add langchain-runloop
      ```
    </CodeGroup>

    ```python
    from runloop_api_client import RunloopSDK

    from langchain_runloop import RunloopSandbox

    api_key = "..."
    client = RunloopSDK(bearer_token=api_key)

    devbox = client.devbox.create()
    backend = RunloopSandbox(devbox=devbox)
    ```
  </Tab>

  <Tab title="Local shell">
    <Warning>
      This backend provides unrestricted filesystem and shell access. Use only in controlled environments for development and testing. See the [security considerations](/oss/python/deepagents/backends#local-shell) for more details.
    </Warning>

    ```python
    from deepagents.backends import LocalShellBackend

    backend = LocalShellBackend(root_dir=".", env={"PATH": "/usr/bin:/bin"})
    ```
  </Tab>
</Tabs>

### Upload sample data

Create and upload sample sales data to the backend:

```python
import csv
import io

# Create sample sales data
data = [
    ["Date", "Product", "Units Sold", "Revenue"],
    ["2025-08-01", "Widget A", 10, 250],
    ["2025-08-02", "Widget B", 5, 125],
    ["2025-08-03", "Widget A", 7, 175],
    ["2025-08-04", "Widget C", 3, 90],
    ["2025-08-05", "Widget B", 8, 200],
]

# Convert to CSV bytes
text_buf = io.StringIO()
writer = csv.writer(text_buf)
writer.writerows(data)
csv_bytes = text_buf.getvalue().encode("utf-8")
text_buf.close()

# Upload to backend
backend.upload_files([("/home/daytona/data/sales_data.csv", csv_bytes)])
```

## Implement custom tools

Data analysis tasks might produce artefacts, like reports or plots.
The following simple [tool](/oss/python/langchain/tools) downloads them with `backend.download_files` and then uploads them using the Slack SDK.
We could also ask our agent to list the relevant file paths instead of uploading them, so interested parties can obtain them separately as needed.

```python
from langchain.tools import tool
from slack_sdk import WebClient


slack_token = os.environ["SLACK_USER_TOKEN"]
slack_client = WebClient(token=slack_token)


@tool(parse_docstring=True)
def slack_send_message(text: str, file_path: str | None = None) -> str:
    """Send message, optionally including attachments such as images.

    Args:
        text: (str) text content of the message
        file_path: (str) file path of attachment in the filesystem.
    """
    if not file_path:
        slack_client.chat_postMessage(channel=channel, text=text)
    else:
        fp = backend.download_files([file_path])
        slack_client.files_upload_v2(
            channel="C0123456ABC",  # specify your own channel here
            content=fp[0].content,
            initial_comment=text,
        )

    return "Message sent."
```

<Note>
  It is generally good practice to avoid adding credentials and other secrets to the sandbox. Here we manage the Slack token outside the sandbox in a tool.
</Note>

## Run the agent

Let's instantiate an agent:

```python
import uuid

from langgraph.checkpoint.memory import InMemorySaver
from deepagents import create_deep_agent


checkpointer = InMemorySaver()

agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-5",
    tools=[slack_send_message],
    backend=backend,
    checkpointer=checkpointer,
)

thread_id = str(uuid.uuid4())
config={"configurable": {"thread_id": thread_id}}
```

We include:

* A choice of [model](/oss/python/deepagents/customization#model)
* Our custom [tool](/oss/python/deepagents/customization#tools)
* The [backend](/oss/python/deepagents/backends)
* A [checkpointer](/oss/python/langchain/short-term-memory) to support multi-turn conversations

Let's now invoke our agent.

```python
input_message = {
    "role": "user",
    "content": (
        "Analyze ./data/sales_data.csv in the current dir and generate a beautiful plot. "
        "When finished, send your analysis and the plot to Slack using the tool."
    ),
}
for step in agent.stream(
    {"messages": [input_message]},
    config,
    stream_mode="updates",
):
    for _, update in step.items():
        if update and (messages := update.get("messages")) and isinstance(messages, list):
            for message in messages:
                message.pretty_print()
```

## Results

The agent successfully analyzes the data and shares a comprehensive report with visualizations to Slack.

## Next steps

Now that you've built a data analysis agent, explore these resources to extend its capabilities:

* [Backends](/oss/python/deepagents/backends): Learn about the Deep Agents backend system
* [Sandboxes](/oss/python/deepagents/sandboxes): Review backends for sandboxed code execution, including security considerations and advanced configurations
* [Customization](/oss/python/deepagents/customization): Discover how to customize your agent with different models, tools, prompts, and planning strategies
* [CLI](/oss/python/deepagents/cli/overview): Try the Deep Agents CLI as a terminal coding agent to assist with data analysis and other agentic tasks locally
* [Skills](/oss/python/deepagents/skills): Equip your agent with reusable skills for common workflows
* [Human-in-the-loop](/oss/python/deepagents/human-in-the-loop): Add interactive approval steps for critical operations in your data analysis workflow
