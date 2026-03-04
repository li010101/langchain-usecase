# Build a semantic search engine with LangChain

## Overview

This tutorial will familiarize you with LangChain's [document loader](/oss/python/langchain/retrieval#document-loaders), [embedding](/oss/python/langchain/retrieval#embedding-models), and [vector store](/oss/python/langchain/retrieval#vector-store) abstractions. These abstractions are designed to support retrieval of data--  from (vector) databases and other sources -- for integration with LLM workflows. They are important for applications that fetch data to be reasoned over as part of model inference, as in the case of retrieval-augmented generation, or [RAG](/oss/python/langchain/retrieval).

Here we will build a search engine over a PDF document. This will allow us to retrieve passages in the PDF that are similar to an input query. The guide also includes a minimal RAG implementation on top of the search engine.

### Concepts

This guide focuses on retrieval of text data. We will cover the following concepts:

* [Documents and document loaders](/oss/python/integrations/document_loaders);
* [Text splitters](/oss/python/integrations/splitters);
* [Embeddings](/oss/python/integrations/text_embedding);
* [Vector stores](/oss/python/integrations/vectorstores) and [retrievers](/oss/python/integrations/retrievers).

## Setup

### Installation

This tutorial requires the `langchain-community` and `pypdf` packages:

```bash
pip install langchain-community pypdf
```

Or with conda:

```bash
conda install langchain-community pypdf -c conda-forge
```

Or with uv:

```bash
uv add langchain-community pypdf
```

### LangSmith

Many of the applications you build with LangChain will contain multiple steps with multiple invocations of LLM calls.
As these applications get more and more complex, it becomes crucial to be able to inspect what exactly is going on inside your chain or agent.
The best way to do this is with [LangSmith](https://smith.langchain.com).

After you sign up at the link above, make sure to set your environment variables to start logging traces:

```shell
export LANGSMITH_TRACING="true"
export LANGSMITH_API_KEY="..."
```

## 1. Documents and document loaders

LangChain implements a [Document](https://reference.langchain.com/python/langchain-core/documents/base/Document) abstraction, which is intended to represent a unit of text and associated metadata. It has three attributes:

* `page_content`: a string representing the content;
* `metadata`: a dict containing arbitrary metadata;
* `id`: (optional) a string identifier for the document.

The `metadata` attribute can capture information about the source of the document, its relationship to other documents, and other information. Note that an individual [`Document`](https://reference.langchain.com/python/langchain-core/documents/base/Document) object often represents a chunk of a larger document.

### Loading documents

Let's load a PDF into a sequence of [`Document`](https://reference.langchain.com/python/langchain-core/documents/base/Document) objects. [Here is a sample PDF](https://github.com/langchain-ai/langchain/blob/v0.3/docs/docs/example_data/nke-10k-2023.pdf) -- a 10-k filing for Nike from 2023.

```python
from langchain_community.document_loaders import PyPDFLoader

file_path = "../example_data/nke-10k-2023.pdf"
loader = PyPDFLoader(file_path)

docs = loader.load()

print(len(docs))
```

`PyPDFLoader` loads one [`Document`](https://reference.langchain.com/python/langchain-core/documents/base/Document) object per PDF page.

### Splitting

For both information retrieval and downstream question-answering purposes, a page may be too coarse a representation. Our goal in the end will be to retrieve [`Document`](https://reference.langchain.com/python/langchain-core/documents/base/Document) objects that answer an input query, and further splitting our PDF will help ensure that the meanings of relevant portions of the document are not "washed out" by surrounding text.

We can use [text splitters](/oss/python/langchain/retrieval#text_splitters) for this purpose. Here we will use a simple text splitter that partitions based on characters. We will split our documents into chunks of 1000 characters with 200 characters of overlap between chunks.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

print(len(all_splits))
```

## 2. Embeddings

Vector search is a common way to store and search over unstructured data (such as unstructured text). The idea is to store numeric vectors that are associated with the text. Given a query, we can [embed](/oss/python/langchain/retrieval#embedding_models) it as a vector of the same dimension and use vector similarity metrics (such as cosine similarity) to identify related text.

LangChain supports embeddings from dozens of providers. Let's select a model:

### OpenAI

```bash
pip install -U "langchain-openai"
```

```python
import getpass
import os

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
```

### Other Providers

LangChain supports many embedding providers:
- Azure OpenAI
- Google Gemini
- Google Vertex
- AWS Bedrock
- HuggingFace
- Ollama
- Cohere
- MistralAI
- Nomic
- NVIDIA
- Voyage AI
- IBM watsonx

```python
vector_1 = embeddings.embed_query(all_splits[0].page_content)
vector_2 = embeddings.embed_query(all_splits[1].page_content)

assert len(vector_1) == len(vector_2)
print(f"Generated vectors of length {len(vector_1)}")
```

## 3. Vector stores

LangChain [VectorStore](https://reference.langchain.com/python/langchain-core/vectorstores/base/VectorStore) objects contain methods for adding text and [`Document`](https://reference.langchain.com/python/langchain-core/documents/base/Document) objects to the store, and querying them using various similarity metrics.

LangChain includes a suite of integrations with different vector store technologies:

### In-memory

```python
from langchain_core.vectorstores import InMemoryVectorStore

vector_store = InMemoryVectorStore(embeddings)
```

### Chroma

```bash
pip install -qU langchain-chroma
```

```python
from langchain_chroma import Chroma

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",
)
```

### FAISS

```bash
pip install -qU langchain-community faiss-cpu
```

```python
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS

embedding_dim = len(embeddings.embed_query("hello world"))
index = faiss.IndexFlatL2(embedding_dim)

vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)
```

### Other Vector Stores

LangChain supports many vector stores:
- Amazon OpenSearch
- AstraDB
- Milvus
- MongoDB Atlas
- PGVector
- Pinecone
- Qdrant

### Adding documents and querying

```python
# Add documents to the vector store
ids = vector_store.add_documents(documents=all_splits)

# Search with similarity
results = vector_store.similarity_search(
    "How many distribution centers does Nike have in the US?"
)

print(results[0])

# Search with scores
results = vector_store.similarity_search_with_score("What was Nike's revenue in 2023?")
doc, score = results[0]
print(f"Score: {score}")
print(doc)
```

## 4. Retrievers

LangChain [`VectorStore`](https://reference.langchain.com/python/langchain-core/vectorstores/base/VectorStore) objects do not subclass [Runnable](https://reference.langchain.com/python/langchain-core/runnables/base/Runnable). LangChain [Retrievers](https://reference.langchain.com/python/langchain-core/retrievers/BaseRetriever) are Runnables, so they implement a standard set of methods.

```python
# Create a retriever
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1},
)

# Batch retrieve
retriever.batch(
    [
        "How many distribution centers does Nike have in the US?",
        "When was Nike incorporated?",
    ],
)
```

`VectorStoreRetriever` supports search types of `"similarity"` (default), `"mmr"` (maximum marginal relevance), and `"similarity_score_threshold"`.

## Next steps

You've now seen how to build a semantic search engine over a PDF document.

For more on document loaders:
* [Overview](/oss/python/langchain/retrieval#document_loaders)
* [Available integrations](/oss/python/integrations/document_loaders/)

For more on embeddings:
* [Overview](/oss/python/langchain/retrieval#embedding_models/)
* [Available integrations](/oss/python/integrations/text_embedding/)

For more on vector stores:
* [Overview](/oss/python/langchain/retrieval#vectorstores/)
* [Available integrations](/oss/python/integrations/vectorstores/)

For more on RAG, see:
* [Build a Retrieval Augmented Generation (RAG) App](/oss/python/langchain/rag/)
