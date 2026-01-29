import asyncio
import queue
from collections.abc import AsyncGenerator, Generator
from .message import AssistantMessage, ToolMessage, MessageChunk

# --- --- --- --- --- ---

GenerateTextResponse = list[AssistantMessage | ToolMessage]
FullMessageQueueSync = queue.Queue[AssistantMessage | ToolMessage | None]
FullMessageQueueAsync = asyncio.Queue[AssistantMessage | ToolMessage | None]
StreamTextResponseSync = tuple[Generator[MessageChunk], FullMessageQueueSync]
StreamTextResponseAsync = tuple[AsyncGenerator[MessageChunk], FullMessageQueueAsync]

__all__ = [
    "GenerateTextResponse",
    "StreamTextResponseSync",
    "StreamTextResponseAsync",
    "FullMessageQueueSync",
    "FullMessageQueueAsync",
]
