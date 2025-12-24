import json
import dataclasses
from abc import ABC, abstractmethod
from typing import Any, Literal, cast
from pydantic import BaseModel, ConfigDict, field_serializer
from litellm.types.utils import Message as LiteLlmMessage,\
                                ModelResponseStream as LiteLlmModelResponseStream,\
                                ChatCompletionAudioResponse,\
                                ChatCompletionMessageToolCall,\
                                ChatCompletionDeltaToolCall
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

class ChatMessage(BaseModel, ABC):    
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
    )
    
    @abstractmethod
    def to_litellm_message(self) -> AllMessageValues: ...

class UserMessage(ChatMessage):
    content: OpenAIMessageContent
    role: Literal["user"] = "user"

    def to_litellm_message(self) -> ChatCompletionUserMessage:
        return ChatCompletionUserMessage(role=self.role, content=self.content)

class AssistantMessage(ChatMessage):
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[ChatCompletionAssistantToolCall] | None = None
    audio: ChatCompletionAudioResponse | None = None
    images: list[ChatCompletionImageURL] | None = None
    role: Literal["assistant"] = "assistant"

    @classmethod
    def from_litellm_message(cls, message: LiteLlmMessage) -> "AssistantMessage":
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
            images=message.get("images")
        )

    def to_litellm_message(self) -> ChatCompletionAssistantMessage:
        return ChatCompletionAssistantMessage(role=self.role,
                                              content=self.content,
                                              reasoning_content=self.reasoning_content,
                                              tool_calls=self.tool_calls)

class ToolMessage(ChatMessage):
    id: str
    name: str
    arguments: dict
    result: Any
    role: Literal["tool"] = "tool"

    @field_serializer("result", when_used="json")
    def serialize_result(self, result: Any) -> str:
        return json.dumps(result, ensure_ascii=False)

    def to_litellm_message(self) -> ChatCompletionToolMessage:
        return ChatCompletionToolMessage(
            role=self.role,
            content=json.dumps(self.result, ensure_ascii=False),
            tool_call_id=self.id)

class SystemMessage(ChatMessage):
    content: str
    role: Literal["system"] = "system"

    def to_litellm_message(self) -> ChatCompletionSystemMessage:
        return ChatCompletionSystemMessage(role=self.role, content=self.content)

@dataclasses.dataclass
class TextChunk:
    content: str

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

MessageChunk = TextChunk | ReasoningChunk | AudioChunk | ImageChunk | ToolCallChunk

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
    return result
