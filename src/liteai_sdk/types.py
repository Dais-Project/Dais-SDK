from dataclasses import dataclass
from typing import Any, Literal
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import Message
from .tool import ToolFn, RawToolDefinition

ChatMessage = AllMessageValues
ModelResponse = Message

@dataclass
class LlmRequestParams:
    model: str
    messages: list[ChatMessage]
    tools: list[ToolFn | RawToolDefinition] | None = None
    tool_choice: Literal["auto", "required", "none"] = "auto"

    timeout_sec: float | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    headers: dict[str, str] | None = None

    extra_args: dict[str, Any] | None = None
