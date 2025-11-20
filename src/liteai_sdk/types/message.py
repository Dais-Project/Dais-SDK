from abc import ABC
from typing import Literal
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

class ChatMessage(ABC):
    def to_litellm_message(self) -> AllMessageValues: ...

class UserMessage(ChatMessage):
    role: Literal["user"] = "user"
    def __init__(self, content: OpenAIMessageContent):
        self.content = content

    def to_litellm_message(self) -> ChatCompletionUserMessage:
        return ChatCompletionUserMessage(role=self.role, content=self.content)

class AssistantMessage(ChatMessage):
    role: Literal["assistant"] = "assistant"
    def __init__(self,
                 content: str | None,
                 reasoning_content: str | None = None,
                 tool_calls: list[ChatCompletionAssistantToolCall] | None = None):
        self.content = content
        self.reasoning_content = reasoning_content
        self.tool_calls = tool_calls
        self.audio: ChatCompletionAudioResponse | None = None
        self.images: list[ChatCompletionImageURL] | None = None

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

class ToolMessage(ChatMessage):
    role: Literal["tool"] = "tool"
    def __init__(self, tool_call_id: str, content: str):
        self.tool_call_id = tool_call_id
        self.content = content

    def to_litellm_message(self) -> ChatCompletionToolMessage:
        return ChatCompletionToolMessage(
            role=self.role,
            content=self.content,
            tool_call_id=self.tool_call_id)

class SystemMessage(ChatMessage):
    role: Literal["system"] = "system"
    def __init__(self, content: str):
        self.content = content

    def to_litellm_message(self) -> ChatCompletionSystemMessage:
        return ChatCompletionSystemMessage(role=self.role, content=self.content)

class AssistantMessageChunk:
    def __init__(self,
                 content: str | None = None,
                 reasoning_content: str | None = None,
                 audio: ChatCompletionAudioResponse | None = None,
                 images: list[ChatCompletionImageURL] | None = None):
        self.content = content
        self.reasoning_content = reasoning_content
        self.audio = audio
        self.images = images

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
