# Build a custom SQL agent with LangGraph

## Overview

In this tutorial we will build a custom agent that can answer questions about a SQL database using LangGraph.

LangChain offers built-in [agent](/oss/python/langchain/agents) implementations. If deeper customization is required, agents can be implemented directly in LangGraph.

<Warning>
  Building Q&A systems of SQL databases requires executing model-generated SQL queries. There are inherent risks in doing this. Make sure your database connection permissions are always scoped as narrowly as possible.
</Warning>

### Concepts

We will cover:
* [Tools](/oss/python/langchain/tools) for reading from SQL databases
* The LangGraph [Graph API](/oss/python/langgraph/graph-api)
* [Human-in-the-loop](/oss/python/langgraph/interrupts) processes

## Setup

```bash
pip install langchain langgraph langchain-community
```

## 1. Select an LLM

```python
import os
from langchain.chat_models import init_chat_model

os.environ["OPENAI_API_KEY"] = "sk-..."

model = init_chat_model("gpt-4o")
```

## 2. Configure the database

Use the Chinook SQLite database:

```python
import requests, pathlib

url = "https://storage.googleapis.com/benchmarks-artifacts/chinook/Chinook.db"
local_path = pathlib.Path("Chinook.db")

if local_path.exists():
    print(f"{local_path} already exists")
else:
    response = requests.get(url)
    if response.status_code == 200:
        local_path.write_bytes(response.content)
```

Connect:

```python
from langchain_community.utilities import SQLDatabase

db = SQLDatabase.from_uri("sqlite:///Chinook.db")

print(f"Dialect: {db.dialect}")
print(f"Available tables: {db.get_usable_table_names()}")
```

## 3. Add tools for database interactions

```python
from langchain_community.agent_toolkits import SQLDatabaseToolkit

toolkit = SQLDatabaseToolkit(db=db, llm=model)
tools = toolkit.get_tools()
```

Tools:
- `sql_db_query`: Execute SQL queries
- `sql_db_schema`: Get table schemas
- `sql_db_list_tables`: List tables
- `sql_db_query_checker`: Check queries

## 4. Define application steps

Create dedicated nodes:

```python
from typing import Literal
from langchain.messages import AIMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

get_schema_tool = next(tool for tool in tools if tool.name == "sql_db_schema")
get_schema_node = ToolNode([get_schema_tool], name="get_schema")

run_query_tool = next(tool for tool in tools if tool.name == "sql_db_query")
run_query_node = ToolNode([run_query_tool], name="run_query")

# List tables node
def list_tables(state: MessagesState):
    tool_call = {
        "name": "sql_db_list_tables",
        "args": {},
        "id": "abc123",
        "type": "tool_call",
    }
    tool_call_message = AIMessage(content="", tool_calls=[tool_call])
    list_tables_tool = next(tool for tool in tools if tool.name == "sql_db_list_tables")
    tool_message = list_tables_tool.invoke(tool_call)
    response = AIMessage(f"Available tables: {tool_message.content}")
    return {"messages": [tool_call_message, tool_message, response]}

# Call get schema node
def call_get_schema(state: MessagesState):
    llm_with_tools = model.bind_tools([get_schema_tool], tool_choice="any")
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

# Generate query
generate_query_system_prompt = """
You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {dialect} query to run.
""".format(dialect=db.dialect)

def generate_query(state: MessagesState):
    system_message = {"role": "system", "content": generate_query_system_prompt}
    llm_with_tools = model.bind_tools([run_query_tool])
    response = llm_with_tools.invoke([system_message] + state["messages"])
    return {"messages": [response]}

# Check query
check_query_system_prompt = """
You are a SQL expert with a strong attention to detail.
Double check the {dialect} query for common mistakes.
""".format(dialect=db.dialect)

def check_query(state: MessagesState):
    system_message = {"role": "system", "content": check_query_system_prompt}
    tool_call = state["messages"][-1].tool_calls[0]
    user_message = {"role": "user", "content": tool_call["args"]["query"]}
    llm_with_tools = model.bind_tools([run_query_tool], tool_choice="any")
    response = llm_with_tools.invoke([system_message, user_message])
    response.id = state["messages"][-1].id
    return {"messages": [response]}
```

## 5. Implement the agent

Assemble the workflow:

```python
def should_continue(state: MessagesState) -> Literal[END, "check_query"]:
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return END
    else:
        return "check_query"

builder = StateGraph(MessagesState)
builder.add_node(list_tables)
builder.add_node(call_get_schema)
builder.add_node(get_schema_node, "get_schema")
builder.add_node(generate_query)
builder.add_node(check_query)
builder.add_node(run_query_node, "run_query")

builder.add_edge(START, "list_tables")
builder.add_edge("list_tables", "call_get_schema")
builder.add_edge("call_get_schema", "get_schema")
builder.add_edge("get_schema", "generate_query")
builder.add_conditional_edges("generate_query", should_continue)
builder.add_edge("check_query", "run_query")
builder.add_edge("run_query", "generate_query")

agent = builder.compile()
```

## 6. Run the agent

```python
question = "Which genre on average has the longest tracks?"

for step in agent.stream(
    {"messages": [{"role": "user", "content": question}]},
    stream_mode="values",
):
    step["messages"][-1].pretty_print()
```

The agent workflow:
1. List available tables
2. Get schema for relevant tables
3. Generate a query
4. Check the query
5. Run the query
6. Generate final answer

## 7. Implement human-in-the-loop review

Add interrupt for human review:

```python
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt

@tool(run_query_tool.name, description=run_query_tool.description)
def run_query_tool_with_interrupt(config: RunnableConfig, **tool_input):
    request = {
        "action": run_query_tool.name,
        "args": tool_input,
        "description": "Please review the tool call"
    }
    response = interrupt([request])
    
    if response["type"] == "accept":
        tool_response = run_query_tool.invoke(tool_input, config)
    elif response["type"] == "edit":
        tool_input = response["args"]["args"]
        tool_response = run_query_tool.invoke(tool_input, config)
    elif response["type"] == "response":
        tool_response = response["args"]
    else:
        raise ValueError(f"Unsupported response type: {response['type']}")

    return tool_response

# Replace the tool node
run_query_node = ToolNode([run_query_tool_with_interrupt], name="run_query")
```

Add checkpointer:

```python
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()
agent = builder.compile(checkpointer=checkpointer)
```

Run with interrupt:

```python
config = {"configurable": {"thread_id": "1"}}

for step in agent.stream(
    {"messages": [{"role": "user", "content": question}]},
    config,
    stream_mode="values",
):
    if "__interrupt__" in step:
        print("INTERRUPTED:")
        print(step["__interrupt__"])
```

Resume:

```python
from langgraph.types import Command

for step in agent.stream(
    Command(resume={"type": "accept"}),
    config,
    stream_mode="values",
):
    step["messages"][-1].pretty_print()
```

## Next steps

Check out the [Evaluate a graph](/langsmith/evaluate-graph) guide for evaluating LangGraph applications.
