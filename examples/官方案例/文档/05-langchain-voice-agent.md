# Build a voice agent with LangChain

## Overview

Chat interfaces have dominated how we interact with AI, but recent breakthroughs in multimodal AI are opening up exciting new possibilities. High-quality generative models and expressive text-to-speech (TTS) systems now make it possible to build agents that feel less like tools and more like conversational partners.

Voice agents are one example of this. Instead of relying on a keyboard and mouse to type inputs into an agent, you can use spoken words to interact with it.

### What are voice agents?

Voice agents are [agents](/oss/python/langchain/agents) that can engage in natural spoken conversations with users. These agents combine speech recognition, natural language processing, generative AI, and text-to-speech technologies.

They're suited for:
* Customer support
* Personal assistants
* Hands-free interfaces
* Coaching and training

### How do voice agents work?

At a high level, every voice agent needs to handle three tasks:

1. **Listen** - capture audio and transcribe it
2. **Think** - interpret intent, reason, plan
3. **Speak** - generate audio and stream it back to the user

There are two main architectures:

#### 1. STT > Agent > TTS architecture (The "Sandwich")

```
User Audio → Speech-to-Text → LangChain Agent → Text-to-Speech → Audio Output
```

**Pros:**
* Full control over each component
* Access to latest capabilities from modern text-modality models
* Transparent behavior with clear boundaries

**Cons:**
* Requires orchestrating multiple services
* Additional complexity
* Conversion from speech to text loses information

#### 2. Speech-to-Speech architecture (S2S)

```
User Audio → Multimodal Model → Audio Output
```

**Pros:**
* Simpler architecture
* Typically lower latency
* Direct audio processing

**Cons:**
* Limited model options
* Features may lag behind text-modality models
* Less transparency

This guide demonstrates the **sandwich architecture** to balance performance, controllability, and access to modern model capabilities.

### Demo Application overview

We'll walk through building a voice-based agent using the sandwich architecture. The agent will manage orders for a sandwich shop.

The demo uses WebSockets for real-time bidirectional communication between the browser and server.

### Architecture

The demo implements a streaming pipeline:

**Client (Browser)**
* Captures microphone audio and encodes it as PCM
* Establishes WebSocket connection
* Streams audio chunks in real-time
* Receives and plays back synthesized speech audio

**Server (Python)**
* Accepts WebSocket connections
* Orchestrates the three-step pipeline:
  1. **Speech-to-text (STT)**: Forward audio to STT provider (e.g., AssemblyAI)
  2. **Agent**: Process transcripts with LangChain agent
  3. **Text-to-speech (TTS)**: Send to TTS provider (e.g., Cartesia)

## Setup

For detailed installation instructions, see the [repository README](https://github.com/langchain-ai/voice-sandwich-demo#readme).

## 1. Speech-to-text

The STT stage transforms an incoming audio stream into text transcripts.

### Key concepts

**Producer-Consumer Pattern**: Audio chunks are sent to the STT service concurrently with receiving transcript events.

**Event Types**:
* `stt_chunk`: Partial transcripts
* `stt_output`: Final, formatted transcripts

### Implementation

```python
from typing import AsyncIterator
import asyncio
from assemblyai_stt import AssemblyAISTT
from events import VoiceAgentEvent

async def stt_stream(
    audio_stream: AsyncIterator[bytes],
) -> AsyncIterator[VoiceAgentEvent]:
    """
    Transform stream: Audio (Bytes) → Voice Events (VoiceAgentEvent)
    """
    stt = AssemblyAISTT(sample_rate=16000)

    async def send_audio():
        """Background task that pumps audio chunks to AssemblyAI."""
        try:
            async for audio_chunk in audio_stream:
                await stt.send_audio(audio_chunk)
        finally:
            await stt.close()

    send_task = asyncio.create_task(send_audio())

    try:
        async for event in stt.receive_events():
            yield event
    finally:
        with contextlib.suppress(asyncio.CancelledError):
            send_task.cancel()
            await send_task
        await stt.close()
```

## 2. LangChain agent

The agent stage processes text transcripts through a LangChain [agent](/oss/python/langchain/agents) and streams the response tokens.

### Key concepts

**Streaming Responses**: The agent uses `stream_mode="messages"` to emit response tokens as they're generated.

**Conversation Memory**: A checkpointer maintains conversation state across turns using a unique thread ID.

### Implementation

```python
from uuid import uuid4
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

# Define agent tools
def add_to_order(item: str, quantity: int) -> str:
    """Add an item to the customer's sandwich order."""
    return f"Added {quantity} x {item} to the order."

def confirm_order(order_summary: str) -> str:
    """Confirm the final order with the customer."""
    return f"Order confirmed: {order_summary}. Sending to kitchen."

# Create agent with tools and memory
agent = create_agent(
    model="anthropic:claude-haiku-4-5",
    tools=[add_to_order, confirm_order],
    system_prompt="""You are a helpful sandwich shop assistant.
    Your goal is to take the user's order. Be concise and friendly.
    Do NOT use emojis, special characters, or markdown.
    Your responses will be read by a text-to-speech engine.""",
    checkpointer=InMemorySaver(),
)

async def agent_stream(
    event_stream: AsyncIterator[VoiceAgentEvent],
) -> AsyncIterator[VoiceAgentEvent]:
    """Transform stream: Voice Events → Voice Events (with Agent Responses)"""
    thread_id = str(uuid4())

    async for event in event_stream:
        yield event

        if event.type == "stt_output":
            stream = agent.astream(
                {"messages": [HumanMessage(content=event.transcript)]},
                {"configurable": {"thread_id": thread_id}},
                stream_mode="messages",
            )

            async for message, _ in stream:
                if message.text:
                    yield AgentChunkEvent.create(message.text)
```

## 3. Text-to-speech

The TTS stage synthesizes agent response text into audio and streams it back to the client.

### Key concepts

**Concurrent Processing**: Merges two async streams:
* **Upstream processing**: Pass through events and send text to TTS provider
* **Audio reception**: Receive synthesized audio chunks

**Streaming TTS**: Begin synthesizing as soon as text arrives

### Implementation

```python
from cartesia_tts import CartesiaTTS
from utils import merge_async_iters

async def tts_stream(
    event_stream: AsyncIterator[VoiceAgentEvent],
) -> AsyncIterator[VoiceAgentEvent]:
    """Transform stream: Voice Events → Voice Events (with Audio)"""
    tts = CartesiaTTS()

    async def process_upstream() -> AsyncIterator[VoiceAgentEvent]:
        async for event in event_stream:
            yield event
            if event.type == "agent_chunk":
                await tts.send_text(event.text)

    try:
        async for event in merge_async_iters(
            process_upstream(),
            tts.receive_events()
        ):
            yield event
    finally:
        await tts.close()
```

## Putting it all together

The complete pipeline chains the three stages together:

```python
from langchain_core.runnables import RunnableGenerator

pipeline = (
    RunnableGenerator(stt_stream)      # Audio → STT events
    | RunnableGenerator(agent_stream)  # STT events → Agent events
    | RunnableGenerator(tts_stream)    # Agent events → TTS audio
)

# Use in WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    async def websocket_audio_stream():
        while True:
            data = await websocket.receive_bytes()
            yield data

    output_stream = pipeline.atransform(websocket_audio_stream())

    async for event in output_stream:
        if event.type == "tts_chunk":
            await websocket.send_bytes(event.audio)
```

Each stage processes events independently and concurrently, enabling sub-700ms latency for natural conversation.

## Next steps

For more on building agents with LangChain, see the [Agents guide](/oss/python/langchain/agents).
