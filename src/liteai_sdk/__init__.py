import asyncio
import queue
from typing import cast
from collections.abc import AsyncGenerator, Generator
from litellm import ChatCompletionAssistantToolCall, CustomStreamWrapper, completion, acompletion
from litellm.exceptions import (  
    AuthenticationError,
    PermissionDeniedError,
    RateLimitError,
    ContextWindowExceededError,
    BadRequestError,
    InvalidRequestError,
    InternalServerError,
    ServiceUnavailableError,
    ContentPolicyViolationError,
    APIError,
    Timeout,
)
from litellm.utils import get_valid_models
from litellm.types.utils import LlmProviders,\
                                ModelResponse as LiteLlmModelResponse,\
                                ModelResponseStream as LiteLlmModelResponseStream,\
                                Choices as LiteLlmModelResponseChoices
from .debug import enable_debugging
from .stream import AssistantMessageCollector
from .tool import ToolFn, ToolDef, RawToolDef, prepare_tools
from .tool.execute import execute_tool_sync, execute_tool, parse_arguments
from .tool.utils import filter_executable_tools, find_tool_by_name
from .types import LlmRequestParams, GenerateTextResponse, StreamTextResponseSync, StreamTextResponseAsync
from .types.exceptions import *
from .types.message import ChatMessage, AssistantMessageChunk, UserMessage, SystemMessage, AssistantMessage, ToolMessage

class LLM:
    """
    Possible exceptions raises for `generate_text` and `stream_text`:
        - AuthenticationError
        - PermissionDeniedError
        - RateLimitError
        - ContextWindowExceededError
        - BadRequestError
        - InvalidRequestError
        - InternalServerError
        - ServiceUnavailableError
        - ContentPolicyViolationError

        - APIError
        - Timeout
    """

    def __init__(self,
                 provider: LlmProviders,
                 base_url: str,
                 api_key: str):
        self.provider = provider
        self.base_url = base_url
        self.api_key = api_key

    def _parse_params_nonstream(self, params: LlmRequestParams):
        tools = params.tools and prepare_tools(params.tools)
        return {
            "model": f"{self.provider.value}/{params.model}",
            "messages": [message.to_litellm_message() for message in params.messages],
            "base_url": self.base_url,
            "api_key": self.api_key,
            "tools": tools,
            "tool_choice": params.tool_choice,
            "stream": False,
            "timeout": params.timeout_sec,
            "extra_headers": params.headers,
            **(params.extra_args or {})
        }

    def _parse_params_stream(self, params: LlmRequestParams):
        tools = params.tools and prepare_tools(params.tools)
        return {
            "model": f"{self.provider.value}/{params.model}",
            "messages": [message.to_litellm_message() for message in params.messages],
            "base_url": self.base_url,
            "api_key": self.api_key,
            "tools": tools,
            "tool_choice": params.tool_choice,
            "stream": True,
            "timeout": params.timeout_sec,
            "extra_headers": params.headers,
            **(params.extra_args or {})
        }

    @staticmethod
    def _should_resolve_tool_calls(
            params: LlmRequestParams,
            message: AssistantMessage,
            ) -> tuple[list[ToolFn | ToolDef | RawToolDef],
                       list[ChatCompletionAssistantToolCall]] | None:
        message.tool_calls
        condition = params.execute_tools and\
                    params.tools is not None and\
                    message.tool_calls is not None
        if condition:
            assert params.tools is not None
            assert message.tool_calls is not None
            return params.tools, message.tool_calls
        return None
    
    @staticmethod
    def _parse_tool_call(tool_call: ChatCompletionAssistantToolCall) -> tuple[str, str, str] | None:
        id = tool_call.get("id")
        function = tool_call.get("function")
        function_name = function.get("name")
        function_arguments = function.get("arguments")
        if id is None or\
           function is None or\
           function_name is None or\
           function_arguments is None: return None
        return id, function_name, function_arguments

    @staticmethod
    async def _execute_tool_calls(
        tools: list[ToolFn | ToolDef | RawToolDef],
        tool_calls: list[ChatCompletionAssistantToolCall]
        ) -> list[ToolMessage]:
        executable_tools = filter_executable_tools(tools)
        result = []
        for tool_call in tool_calls:
            if (tool_call_data := LLM._parse_tool_call(tool_call)) is None: continue
            id, function_name, function_arguments = tool_call_data
            if (target_tool := find_tool_by_name(cast(list, executable_tools), function_name)) is None: continue
            parsed_arguments = parse_arguments(function_arguments)
            ret = await execute_tool(target_tool, parsed_arguments)
            result.append(ToolMessage(
                id=id,
                name=function_name,
                arguments=parsed_arguments,
                result=ret))
        return result

    @staticmethod
    def _execute_tool_calls_sync(
        tools: list[ToolFn | ToolDef | RawToolDef],
        tool_calls: list[ChatCompletionAssistantToolCall]
        ) -> list[ToolMessage]:
        executable_tools = filter_executable_tools(tools)
        result = []
        for tool_call in tool_calls:
            if (tool_call_data := LLM._parse_tool_call(tool_call)) is None: continue
            id, function_name, function_arguments = tool_call_data
            if (target_tool := find_tool_by_name(cast(list, executable_tools), function_name)) is None: continue
            parsed_arguments = parse_arguments(function_arguments)
            ret = execute_tool_sync(target_tool, parsed_arguments)
            result.append(ToolMessage(
                id=id,
                name=function_name,
                arguments=parsed_arguments,
                result=ret))
        return result

    def list_models(self) -> list[str]:
        return get_valid_models(
            custom_llm_provider=self.provider.value,
            check_provider_endpoint=True,
            api_base=self.base_url,
            api_key=self.api_key)

    def generate_text_sync(self, params: LlmRequestParams):
        response = completion(**self._parse_params_nonstream(params))
        response = cast(LiteLlmModelResponse, response)
        choices = cast(list[LiteLlmModelResponseChoices], response.choices)
        message = choices[0].message
        assistant_message = AssistantMessage.from_litellm_message(message)
        result: GenerateTextResponse = [assistant_message]
        if (tools_and_tool_calls := self._should_resolve_tool_calls(params, assistant_message)):
            tools, tool_calls = tools_and_tool_calls
            result += self._execute_tool_calls_sync(tools, tool_calls)
        return result

    async def generate_text(self, params: LlmRequestParams) -> GenerateTextResponse:
        response = await acompletion(**self._parse_params_nonstream(params))
        response = cast(LiteLlmModelResponse, response)
        choices = cast(list[LiteLlmModelResponseChoices], response.choices)
        message = choices[0].message
        assistant_message = AssistantMessage.from_litellm_message(message)
        result: GenerateTextResponse = [assistant_message]
        if (tools_and_tool_calls := self._should_resolve_tool_calls(params, assistant_message)):
            tools, tool_calls = tools_and_tool_calls
            result += await self._execute_tool_calls(tools, tool_calls)
        return result

    def stream_text_sync(self, params: LlmRequestParams) -> StreamTextResponseSync:
        def stream(response: CustomStreamWrapper) -> Generator[AssistantMessageChunk]:
            nonlocal message_collector
            for chunk in response:
                chunk = cast(LiteLlmModelResponseStream, chunk)
                yield AssistantMessageChunk.from_litellm_chunk(chunk)
                message_collector.collect(chunk)

            message = message_collector.get_message()
            full_message_queue.put(message)
            if (tools_and_tool_calls := self._should_resolve_tool_calls(params, message)):
                tools, tool_calls = tools_and_tool_calls
                tool_messages = self._execute_tool_calls_sync(tools, tool_calls)
                for tool_message in tool_messages:
                    full_message_queue.put(tool_message)
            full_message_queue.put(None)

        response = completion(**self._parse_params_stream(params))
        message_collector = AssistantMessageCollector()
        returned_stream = stream(cast(CustomStreamWrapper, response))
        full_message_queue = queue.Queue[AssistantMessage | ToolMessage | None]()
        return returned_stream, full_message_queue

    async def stream_text(self, params: LlmRequestParams) -> StreamTextResponseAsync:
        async def stream(response: CustomStreamWrapper) -> AsyncGenerator[AssistantMessageChunk]:
            nonlocal message_collector
            async for chunk in response:
                chunk = cast(LiteLlmModelResponseStream, chunk)
                yield AssistantMessageChunk.from_litellm_chunk(chunk)
                message_collector.collect(chunk)

            message = message_collector.get_message()
            await full_message_queue.put(message)
            if (tools_and_tool_calls := self._should_resolve_tool_calls(params, message)):
                tools, tool_calls = tools_and_tool_calls
                tool_messages = await self._execute_tool_calls(tools, tool_calls)
                for tool_message in tool_messages:
                    await full_message_queue.put(tool_message)
            await full_message_queue.put(None)

        response = await acompletion(**self._parse_params_stream(params))
        message_collector = AssistantMessageCollector()
        returned_stream = stream(cast(CustomStreamWrapper, response))
        full_message_queue = asyncio.Queue[AssistantMessage | ToolMessage | None]()
        return returned_stream, full_message_queue

__all__ = [
    # Exceptions
    "AuthenticationError",
    "PermissionDeniedError",
    "RateLimitError",
    "ContextWindowExceededError",
    "BadRequestError",
    "InvalidRequestError",
    "InternalServerError",
    "ServiceUnavailableError",
    "ContentPolicyViolationError",
    "APIError",
    "Timeout",

    "enable_debugging",

    "LLM",
    "LlmRequestParams",
    "ToolFn",
    "ToolDef",
    "RawToolDef",

    "ChatMessage",
    "UserMessage",
    "SystemMessage",
    "AssistantMessage",
    "ToolMessage",
    "AssistantMessageChunk",

    "GenerateTextResponse",
    "StreamTextResponseSync",
    "StreamTextResponseAsync"
]
