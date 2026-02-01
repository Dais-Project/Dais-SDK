import json
import dataclasses
import uuid
from abc import ABC, abstractmethod
from typing import Any, Literal, NamedTuple, cast
from pydantic import BaseModel, ConfigDict, Field, field_validator
from litellm.types.utils import (
    Message as LiteLlmMessage,
    ModelResponse as LiteLlmModelResponse,
    ModelResponseStream as LiteLlmModelResponseStream,
    Choices as LiteLlmModelResponseChoices,
    ChatCompletionAudioResponse,
    ChatCompletionMessageToolCall,
    ChatCompletionDeltaToolCall,
    Usage as LiteLlmUsage
)
from litellm.types.llms.openai import (
    AllMessageValues,
    OpenAIMessageContent,
    ChatCompletionAssistantToolCall,
    ImageURLListItem as ChatCompletionImageURL,

    ChatCompletionUserMessage,
    ChatCompletionAssistantMessage,
    ChatCompletionToolMessage,
    ChatCompletionSystemMessage,
)
from ..logger import logger

class ChatMessage(BaseModel, ABC):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
    )
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    @abstractmethod
    def to_litellm_message(self) -> AllMessageValues: ...

class UserMessage(ChatMessage):
    content: OpenAIMessageContent
    role: Literal["user"] = "user"

    def to_litellm_message(self) -> ChatCompletionUserMessage:
        return ChatCompletionUserMessage(role=self.role, content=self.content)

class ToolMessage(ChatMessage):
    tool_call_id: str
    name: str
    arguments: str
    result: str | None = None
    error: str | None = None
    role: Literal["tool"] = "tool"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("result", mode="before")
    def validate_result(cls, v: Any) -> Any:
        if v is None: return v
        if isinstance(v, str): return v
        return json.dumps(v, ensure_ascii=False)

    def with_result(self, result: str | None, error: str | None) -> "ToolMessage":
        return ToolMessage(
            tool_call_id=self.tool_call_id,
            name=self.name,
            arguments=self.arguments,
            result=result,
            error=error)

    def to_litellm_message(self) -> ChatCompletionToolMessage:
        if self.result is None and self.error is None:
            raise ValueError(f"ToolMessage({self.id}, {self.name}) is incomplete, "
                              "result and error cannot be both None")

        if self.error is not None:
            content = json.dumps({"error": self.error}, ensure_ascii=False)
        else:
            assert self.result is not None
            content = self.result

        return ChatCompletionToolMessage(
            role=self.role,
            content=content,
            tool_call_id=self.tool_call_id)

class AssistantMessage(ChatMessage):
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[ChatCompletionAssistantToolCall] | None = None
    audio: ChatCompletionAudioResponse | None = None
    images: list[ChatCompletionImageURL] | None = None
    usage: LiteLlmUsage | None = None
    role: Literal["assistant"] = "assistant"

    @classmethod
    def from_litellm_message(cls, response: LiteLlmModelResponse) -> "AssistantMessage":
        choices = cast(list[LiteLlmModelResponseChoices], response.choices)
        message = choices[0].message

        tool_calls: list[ChatCompletionAssistantToolCall] | None = None
        if (message_tool_calls := message.get("tool_calls")) is not None:
            tool_calls = [ChatCompletionAssistantToolCall(
                id=tool_call.id,
                function={
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                },
                type="function",
            ) for tool_call in cast(list[ChatCompletionMessageToolCall], message_tool_calls)]

        return cls.model_construct(
            content=message.get("content"),
            reasoning_content=message.get("reasoning_content"),
            tool_calls=tool_calls,
            audio=message.get("audio"),
            images=message.get("images"),
            usage=response.get("usage"),
        )

    def to_litellm_message(self) -> ChatCompletionAssistantMessage:
        return ChatCompletionAssistantMessage(role=self.role,
                                              content=self.content or "",
                                              reasoning_content=self.reasoning_content,
                                              tool_calls=self.tool_calls)

    def get_incomplete_tool_messages(self) -> list[ToolMessage] | None:
        """
        Get a incomplete tool message from the assistant message.
        The returned tool message is incomplete,
        which means it only contains the tool call id, name and arguments.
        Returns None if there is no tool call in the assistant message.
        """
        if self.tool_calls is None: return None
        results: list[ToolMessage] = []
        for tool_call in self.tool_calls:
            id = tool_call.get("id")
            function = tool_call.get("function") # this can not be None
            function_name = function.get("name")
            function_arguments = function.get("arguments", "")
            if (id is None or
                function is None or
                function_name is None):
                logger.warning(f"Broken tool call: {tool_call}")
                continue # broken tool call
            results.append(ToolMessage(
                tool_call_id=id,
                name=function_name,
                arguments=function_arguments,
                result=None,
                error=None))
        return results

class SystemMessage(ChatMessage):
    content: str
    role: Literal["system"] = "system"

    def to_litellm_message(self) -> ChatCompletionSystemMessage:
        return ChatCompletionSystemMessage(role=self.role, content=self.content)

@dataclasses.dataclass
class TextChunk:
    content: str

@dataclasses.dataclass
class UsageChunk:
    input_tokens: int
    output_tokens: int
    total_tokens: int

@dataclasses.dataclass
class ReasoningChunk:
    content: str

@dataclasses.dataclass
class AudioChunk:
    data: ChatCompletionAudioResponse

@dataclasses.dataclass
class ImageChunk:
    data: list[ChatCompletionImageURL]

@dataclasses.dataclass
class ToolCallChunk:
    id: str | None
    name: str | None
    arguments: str
    index: int

MessageChunk = TextChunk | UsageChunk | ReasoningChunk | AudioChunk | ImageChunk | ToolCallChunk

def openai_chunk_normalizer(
        chunk: LiteLlmModelResponseStream
        ) -> list[MessageChunk]:
    if len(chunk.choices) == 0: return []

    result = []
    delta = chunk.choices[0].delta
    if delta.get("content"):
        result.append(TextChunk(cast(str, delta.content)))
    if delta.get("reasoning_content"):
        result.append(ReasoningChunk(cast(str, delta.reasoning_content)))
    if delta.get("audio"):
        result.append(AudioChunk(cast(ChatCompletionAudioResponse, delta.audio)))
    if delta.get("images"):
        result.append(ImageChunk(cast(list[ChatCompletionImageURL], delta.images)))
    if delta.get("tool_calls"):
        for tool_call in cast(list[ChatCompletionDeltaToolCall], delta.tool_calls):
            result.append(ToolCallChunk(
                tool_call.id,
                tool_call.function.name,
                tool_call.function.arguments,
                tool_call.index))
    if (usage := getattr(chunk, "usage", None)) is not None:
        usage = cast(LiteLlmUsage, usage)
        result.append(UsageChunk(
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens))
    return result
