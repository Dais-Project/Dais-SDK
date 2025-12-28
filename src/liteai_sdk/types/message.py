from __future__ import annotations

import json
import dataclasses
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Literal, cast
from pydantic import BaseModel, ConfigDict, PrivateAttr, field_validator
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
from ..tool import ToolLike
from ..tool.utils import find_tool_by_name
from ..logger import logger

if TYPE_CHECKING:
    from . import LlmRequestParams

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

class ToolMessage(ChatMessage):
    """
    The `tool_def` field is ref to the target tool of the tool call, and 
    it will only be None when the target tool is not found
    """
    id: str
    name: str
    arguments: str
    result: str | None = None
    error: str | None = None
    role: Literal["tool"] = "tool"

    _tool_def: ToolLike | None = PrivateAttr(default=None)

    @field_validator("result", mode="before")
    def validate_result(cls, v: Any) -> Any:
        if v is None: return v
        if isinstance(v, str): return v
        return json.dumps(v, ensure_ascii=False)

    @property
    def tool_def(self) -> ToolLike | None:
        return self._tool_def

    def with_tool_def(self, tool_def: ToolLike) -> "ToolMessage":
        self._tool_def = tool_def
        return self

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
            tool_call_id=self.id)

ToolCallTuple = tuple[str, str, str]
class AssistantMessage(ChatMessage):
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[ChatCompletionAssistantToolCall] | None = None
    audio: ChatCompletionAudioResponse | None = None
    images: list[ChatCompletionImageURL] | None = None
    role: Literal["assistant"] = "assistant"

    _request_params_ref: LlmRequestParams | None = PrivateAttr(default=None)

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
            images=message.get("images"),
        )

    def with_request_params(self, request_params: LlmRequestParams) -> "AssistantMessage":
        self._request_params_ref = request_params
        return self

    def to_litellm_message(self) -> ChatCompletionAssistantMessage:
        return ChatCompletionAssistantMessage(role=self.role,
                                              content=self.content or "",
                                              reasoning_content=self.reasoning_content,
                                              tool_calls=self.tool_calls)

    def parse_tool_calls(self) -> list[ToolCallTuple] | None:
        if self.tool_calls is None: return None
        results = []
        for tool_call in self.tool_calls:
            id = tool_call.get("id")
            function = tool_call.get("function") # this can not be None
            function_name = function.get("name")
            function_arguments = function.get("arguments")
            if id is None or\
               function is None or\
               function_name is None or\
               function_arguments is None:
                return None
            results.append((id, function_name, function_arguments))
        return results

    def get_partial_tool_messages(self) -> list[ToolMessage] | None:
        """
        Get a partial tool message from the assistant message.
        The returned tool message is not complete,
        it only contains the tool call id, name and arguments.
        Returns None if there is no tool call in the assistant message.
        """
        has_tool_def = self._request_params_ref is not None and\
                       self._request_params_ref.tools is not None
        if not has_tool_def:
            logger.warning("AssistantMessage.get_partial_tool_messages() called without request params. "
                           "Call with_request_params() first to enable auto tool_def attachment feature.")

        parsed_tool_calls = self.parse_tool_calls()
        if parsed_tool_calls is None: return None

        results = []
        for tool_call in parsed_tool_calls:
            id, name, arguments = tool_call

            tool_message = ToolMessage(
                id=id,
                name=name,
                arguments=arguments,
                result=None,
                error=None)

            if has_tool_def:
                assert self._request_params_ref and self._request_params_ref.tools
                target_tool = find_tool_by_name(self._request_params_ref.tools, name)
                if target_tool:
                    tool_message = tool_message.with_tool_def(target_tool)
                else:
                    logger.warning(f"Tool {name} not found in request params, "
                                    "tool_def will not be attached to the tool message")

            results.append(tool_message)
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
