from dataclasses import dataclass
from typing import AsyncGenerator
from .message import AssistantMessage

@dataclass
class TextChunkEvent:
    content: str

@dataclass
class UsageChunkEvent:
    input_tokens: int
    output_tokens: int
    total_tokens: int

@dataclass
class ToolCallChunkEvent:
    id: str | None
    name: str | None
    arguments: str | None
    index: int

@dataclass
class AssistantMessageEvent:
    """
    This event is sent when the assistant message is complete.
    """
    message: AssistantMessage

type StreamMessageEvent = TextChunkEvent | UsageChunkEvent | ToolCallChunkEvent | AssistantMessageEvent
type StreamMessageGenerator = AsyncGenerator[StreamMessageEvent, None]
