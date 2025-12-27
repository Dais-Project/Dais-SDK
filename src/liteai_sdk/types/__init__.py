import asyncio
import dataclasses
import queue
from typing import Any, Generator, Literal
from collections.abc import AsyncGenerator, Generator
from ..tool import ToolLike
from .message import ChatMessage, AssistantMessage, ToolMessage, MessageChunk

@dataclasses.dataclass
class LlmRequestParams:
    model: str
    messages: list[ChatMessage]
    tools: list[ToolLike] | None = None
    tool_choice: Literal["auto", "required", "none"] = "auto"
    execute_tools: bool = False

    timeout_sec: float | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    headers: dict[str, str] | None = None

    extra_args: dict[str, Any] | None = None

# --- --- --- --- --- ---

GenerateTextResponse = list[AssistantMessage | ToolMessage]
StreamTextResponseSync = tuple[
    Generator[MessageChunk],
    queue.Queue[AssistantMessage | ToolMessage | None]]
StreamTextResponseAsync = tuple[
    AsyncGenerator[MessageChunk],
    asyncio.Queue[AssistantMessage | ToolMessage | None]]
