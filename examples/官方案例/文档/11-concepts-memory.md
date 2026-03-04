# Memory Overview

[Memory](/oss/python/langgraph/add-memory) is a system that remembers information about previous interactions. For AI agents, memory is crucial for remembering previous interactions, learning from feedback, and adapting to user preferences.

## Two Types of Memory

### Short-term Memory

Short-term memory, also called thread-scoped memory, tracks the ongoing conversation by maintaining message history within a session. LangGraph manages short-term memory as part of the agent's [state](/oss/python/langgraph/graph-api#state).

* Managed as part of the graph's state
* Persisted using a [checkpointer](/oss/python/langgraph/persistence#checkpoints)
* Thread can be resumed at any time
* Updated when the graph is invoked or a step completes

### Long-term Memory

Long-term memory stores user-specific or application-level data across sessions and is shared across conversational threads.

* Scoped to custom namespaces
* Can be recalled at any time and in any thread
* Uses [stores](/oss/python/langgraph/persistence#memory-store)

## Managing Short-term Memory

Conversation history is the most common form of short-term memory. Long conversations pose challenges:

1. May not fit inside LLM's context window
2. LLMs perform poorly over long contexts
3. Slower response times and higher costs

### Common Techniques

- **Filter**: Remove stale information
- **Summarize**: Compress message history
- **Select**: Keep only relevant messages

## Long-term Memory Types

### Semantic Memory

Stores facts and concepts. For AI agents, often used to personalize applications by remembering facts from past interactions.

**Management approaches:**

1. **Profile**: Single JSON document with key-value pairs
   - Continuously updated
   - May benefit from splitting into multiple documents

2. **Collection**: Continuous collection of documents
   - Each memory narrowly scoped
   - Higher recall downstream
   - Shifts complexity to search

### Episodic Memory

Recalls past events or actions. For AI agents, helps remember how to accomplish a task.

**Implementation**: Few-shot example prompting

```python
# Store past examples as few-shot prompts
# Retrieve relevant examples based on user input
```

### Procedural Memory

Remembers rules for performing tasks. For AI agents, combination of model weights, agent code, and prompts.

**Implementation**: Reflection or meta-prompting
- Prompt agent with current instructions
- Refine based on recent conversations or feedback

## Writing Memories

### In the hot path

Creating memories during runtime:

**Pros:**
- Real-time updates
- New memories immediately available
- Transparency with users

**Cons:**
- May increase complexity
- Can impact latency
- Agent must multitask

### In the background

Creating memories as a background task:

**Pros:**
- No latency in primary application
- Separate concerns
- Flexible timing

**Cons:**
- Determining frequency is crucial
- Deciding when to trigger

## Memory Storage

LangGraph stores long-term memories as JSON in a [store](/oss/python/langgraph/persistence#memory-store):

```python
from langgraph.store.memory import InMemoryStore

# Create store
store = InMemoryStore(index={"embed": embed, "dims": 2})

# Save memory
namespace = ("user_id", "context")
store.put(
    namespace,
    "memory_key",
    {"key": "value"},
)

# Retrieve memory
item = store.get(namespace, "memory_key")

# Search memories
items = store.search(
    namespace,
    filter={"key": "value"},
    query="search terms"
)
```

## Learn more

- [Context conceptual overview](/oss/python/concepts/context)
- [Short-term memory in LangChain](/oss/python/langchain/short-term-memory)
- [Memory in LangGraph](/oss/python/langgraph/add-memory)
