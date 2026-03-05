import asyncio
import queue
from collections.abc import AsyncGenerator, Generator
from .message import AssistantMessage, ToolMessage
from .event import StreamMessageEvent

# --- --- --- --- --- ---

# type GenerateTextResponse = list[AssistantMessage | ToolMessage]
# type FullMessageQueueSync = queue.Queue[AssistantMessage | ToolMessage | None]
# type FullMessageQueueAsync = asyncio.Queue[AssistantMessage | ToolMessage | None]
# type StreamTextResponseSync = tuple[Generator[MessageChunk], FullMessageQueueSync]
# type StreamTextResponseAsync = tuple[AsyncGenerator[MessageChunk], FullMessageQueueAsync]

# __all__ = [
#     "GenerateTextResponse",
#     "StreamTextResponseSync",
#     "StreamTextResponseAsync",
#     "FullMessageQueueSync",
#     "FullMessageQueueAsync",
# ]
