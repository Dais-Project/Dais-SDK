import json
import dataclasses
from abc import ABC
from typing import Any, Literal
from litellm.types.utils import Message as LiteLlmMessage,\
                                ModelResponseStream as LiteLlmModelResponseStream,\
                                ChatCompletionAudioResponse
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

@dataclasses.dataclass
class ChatMessage(ABC):
    def to_litellm_message(self) -> AllMessageValues: ...

@dataclasses.dataclass
class UserMessage(ChatMessage):
    content: OpenAIMessageContent
    role: Literal["user"] = "user"

    def to_litellm_message(self) -> ChatCompletionUserMessage:
        return ChatCompletionUserMessage(role=self.role, content=self.content)

@dataclasses.dataclass
class AssistantMessage(ChatMessage):
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[ChatCompletionAssistantToolCall] | None = None
    audio: ChatCompletionAudioResponse | None = None
    images: list[ChatCompletionImageURL] | None = None
    role: Literal["assistant"] = "assistant"

    @staticmethod
    def from_litellm_message(message: LiteLlmMessage) -> "AssistantMessage":
        tool_calls: list[ChatCompletionAssistantToolCall] | None = None
        if message.get("tool_calls"):
            assert message.tool_calls is not None
            tool_calls = [{
                "id": tool_call.id,
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                },
                "type": "function",
            } for tool_call in message.tool_calls]

        result = AssistantMessage(
            content=message.get("content"),
            reasoning_content=message.get("reasoning_content"),
            tool_calls=tool_calls)

        if message.get("audio"):
            result.audio = message.audio
        if message.get("images"):
            result.images = message.images

        return result

    def to_litellm_message(self) -> ChatCompletionAssistantMessage:
        return ChatCompletionAssistantMessage(role=self.role,
                                              content=self.content,
                                              reasoning_content=self.reasoning_content,
                                              tool_calls=self.tool_calls)

@dataclasses.dataclass
class ToolMessage(ChatMessage):
    id: str
    name: str
    arguments: dict
    result: Any
    role: Literal["tool"] = "tool"

    def to_litellm_message(self) -> ChatCompletionToolMessage:
        return ChatCompletionToolMessage(
            role=self.role,
            content=json.dumps(self.result),
            tool_call_id=self.id)

@dataclasses.dataclass
class SystemMessage(ChatMessage):
    content: str
    role: Literal["system"] = "system"

    def to_litellm_message(self) -> ChatCompletionSystemMessage:
        return ChatCompletionSystemMessage(role=self.role, content=self.content)

@dataclasses.dataclass
class AssistantMessageChunk:
    content: str | None = None
    reasoning_content: str | None = None
    audio: ChatCompletionAudioResponse | None = None
    images: list[ChatCompletionImageURL] | None = None

    @staticmethod
    def from_litellm_chunk(chunk: LiteLlmModelResponseStream) -> "AssistantMessageChunk":
        delta = chunk.choices[0].delta
        temp_chunk = AssistantMessageChunk()
        if delta.get("content"):
            temp_chunk.content = delta.content
        if delta.get("reasoning_content"):
            temp_chunk.reasoning_content = delta.reasoning_content
        if delta.get("audio"):
            temp_chunk.audio = delta.audio
        if delta.get("images"):
            temp_chunk.images = delta.images
        return temp_chunk
