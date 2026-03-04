# Build a RAG agent with LangChain

## Overview

One of the most powerful applications enabled by LLMs is sophisticated question-answering (Q&A) chatbots. These are applications that can answer questions about specific source information. These applications use a technique known as Retrieval Augmented Generation, or [RAG](/oss/python/langchain/retrieval/).

This tutorial will show how to build a simple Q&A application over an unstructured text data source. We will demonstrate:

1. A RAG [agent](#rag-agents) that executes searches with a simple tool. This is a good general-purpose implementation.
2. A two-step RAG [chain](#rag-chains) that uses just a single LLM call per query. This is a fast and effective method for simple queries.

### Concepts

We will cover the following concepts:

* **Indexing**: a pipeline for ingesting data from a source and indexing it. *This usually happens in a separate process.*
* **Retrieval and generation**: the actual RAG process, which takes the user query at run time and retrieves the relevant data from the index, then passes that to the model.

### Preview

In this guide we'll build an app that answers questions about the website's content. The specific website we will use is the [LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/) blog post by Lilian Weng.

## Setup

### Installation

This tutorial requires these langchain dependencies:

```bash
pip install langchain langchain-text-splitters langchain-community bs4
```

### LangSmith

Set up [LangSmith](https://smith.langchain.com) to inspect what's happening inside your chain or agent:

```shell
export LANGSMITH_TRACING="true"
export LANGSMITH_API_KEY="..."
```

### Components

We will need to select three components from LangChain's suite of integrations:

1. **Chat Model**: OpenAI, Anthropic, Azure, Google Gemini, AWS Bedrock, HuggingFace
2. **Embeddings Model**: OpenAI, Azure, Google, AWS, HuggingFace, Ollama, Cohere, etc.
3. **Vector Store**: In-memory, Chroma, FAISS, Milvus, MongoDB, PGVector, Pinecone, Qdrant, etc.

## 1. Indexing

<Note>
  **This section is an abbreviated version of the content in the [semantic search tutorial](/oss/python/langchain/knowledge-base).**
</Note>

Indexing commonly works as follows:

1. **Load**: First we need to load our data using [Document Loaders](/oss/python/langchain/retrieval#document_loaders).
2. **Split**: [Text splitters](/oss/python/langchain/retrieval#text_splitters) break large `Documents` into smaller chunks.
3. **Store**: We need somewhere to store and index our splits using a [VectorStore](/oss/python/langchain/retrieval#vectorstores) and [Embeddings](/oss/python/langchain/retrieval#embedding_models) model.

### Loading documents

```python
import bs4
from langchain_community.document_loaders import WebBaseLoader

# Only keep post title, headers, and content from the full HTML.
bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs={"parse_only": bs4_strainer},
)
docs = loader.load()

print(f"Total characters: {len(docs[0].page_content)}")
```

### Splitting documents

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True,
)
all_splits = text_splitter.split_documents(docs)

print(f"Split blog post into {len(all_splits)} sub-documents.")
```

### Storing documents

```python
# Embed and store all document splits
document_ids = vector_store.add_documents(documents=all_splits)

print(document_ids[:3])
```

## 2. Retrieval and generation

RAG applications commonly work as follows:

1. **Retrieve**: Given a user input, relevant splits are retrieved from storage using a [Retriever](/oss/python/langchain/retrieval#retrievers).
2. **Generate**: A [model](/oss/python/langchain/models) produces an answer using a prompt that includes both the question with the retrieved data.

### RAG agents

One formulation of a RAG application is as a simple [agent](/oss/python/langchain/agents) with a tool that retrieves information:

```python
from langchain.tools import tool

@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

tools = [retrieve_context]

# Create agent
prompt = (
    "You have access to a tool that retrieves context from a blog post. "
    "Use the tool to help answer user queries."
)
agent = create_agent(model, tools, system_prompt=prompt)
```

### Running the agent

```python
query = "What is task decomposition?"

for step in agent.stream(
    {"messages": [{"role": "user", "content": query}]},
    stream_mode="values",
):
    step["messages"][-1].pretty_print()
```

The agent will:
1. Generate a query to search for task decomposition
2. Receive the answer and generate a second query to search for common extensions
3. Having received all necessary context, answer the question

### RAG chains

Another common approach is a two-step chain, in which we always run a search and incorporate the result as context for a single LLM query:

```python
from langchain.agents.middleware import dynamic_prompt, ModelRequest

@dynamic_prompt
def prompt_with_context(request: ModelRequest) -> str:
    """Inject context into state messages."""
    last_query = request.state["messages"][-1].text
    retrieved_docs = vector_store.similarity_search(last_query)

    docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

    system_message = (
        "You are a helpful assistant. Use the following context in your response:"
        f"\n\n{docs_content}"
    )

    return system_message


agent = create_agent(model, tools=[], middleware=[prompt_with_context])
```

This is a fast and effective method for simple queries.

## Benefits and Drawbacks

| ✅ Benefits | ⚠️ Drawbacks |
|------------|--------------|
| Search only when needed | Two inference calls |
| Contextual search queries | Reduced control |
| Multiple searches allowed | |

## Next steps

Now that we've implemented a simple RAG application, we can easily incorporate new features:

* [Stream](/oss/python/langchain/streaming) tokens and other information
* Add [conversational memory](/oss/python/langchain/short-term-memory) for multi-turn interactions
* Add [long-term memory](/oss/python/langchain/long-term-memory) for memory across threads
* Add [structured responses](/oss/python/langchain/structured-output)
* Deploy with [LangSmith Deployment](/langsmith/deployments)
