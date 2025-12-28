import asyncio
import queue
from typing import cast
from collections.abc import AsyncGenerator, Generator
from litellm import CustomStreamWrapper, completion, acompletion
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
from .param_parser import ParamParser
from .stream import AssistantMessageCollector
from .tool import ToolFn, ToolDef, RawToolDef, ToolLike
from .tool.execute import execute_tool_sync, execute_tool
from .tool.utils import find_tool_by_name
from .types import LlmRequestParams, GenerateTextResponse, StreamTextResponseSync, StreamTextResponseAsync
from .types.exceptions import *
from .types.message import ChatMessage, UserMessage, SystemMessage, AssistantMessage, ToolMessage,\
                           MessageChunk, TextChunk, ReasoningChunk, AudioChunk, ImageChunk, ToolCallChunk,\
                           ToolCallTuple, openai_chunk_normalizer
from .logger import logger, enable_logging

class LLM:
    """
    The `generate_text` and `stream_text` API will return ToolMessage in the returned sequence
    only if `params.execute_tools` is True.

    - - -

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
        self._param_parser = ParamParser(self.provider, self.base_url, self.api_key)

    @staticmethod
    def _should_resolve_tool_calls(
            params: LlmRequestParams,
            message: AssistantMessage,
            ) -> tuple[list[ToolLike],
                       list[ToolCallTuple]] | None:
        parsed_tool_calls = message.parse_tool_calls()
        condition = params.execute_tools and\
                    params.tools is not None and\
                    parsed_tool_calls is not None
        if condition:
            assert params.tools is not None
            assert parsed_tool_calls is not None
            return params.tools, parsed_tool_calls
        return None

    @staticmethod
    async def _execute_tool_calls(
        tools: list[ToolLike],
        tool_call_tuples: list[ToolCallTuple]
        ) -> list[ToolMessage]:
        results = []
        for tool_call_tuple in tool_call_tuples:
            id, function_name, function_arguments = tool_call_tuple
            if (target_tool := find_tool_by_name(tools, function_name)) is None:
                logger.warning(f"Tool \"{function_name}\" not found, skipping execution.")
                continue
            if isinstance(target_tool, dict):
                logger.warning(f"Tool \"{function_name}\" is a raw tool, skipping execution.")
                continue

            result, error = None, None
            try:
                result = await execute_tool(target_tool, function_arguments)
            except Exception as e:
                error = f"{type(e).__name__}: {str(e)}"
            results.append(ToolMessage(
                id=id,
                name=function_name,
                arguments=function_arguments,
                result=result,
                error=error).with_tool_def(target_tool))
        return results

    @staticmethod
    def _execute_tool_calls_sync(
        tools: list[ToolLike],
        tool_call_tuples: list[ToolCallTuple]
        ) -> list[ToolMessage]:
        results = []
        for tool_call_tuple in tool_call_tuples:
            id, function_name, function_arguments = tool_call_tuple
            if (target_tool := find_tool_by_name(tools, function_name)) is None:
                logger.warning(f"Tool \"{function_name}\" not found, skipping execution.")
                continue
            if isinstance(target_tool, dict):
                logger.warning(f"Tool \"{function_name}\" is a raw tool, skipping execution.")
                continue

            result, error = None, None
            try:
                result = execute_tool_sync(target_tool, function_arguments)
            except Exception as e:
                error = f"{type(e).__name__}: {str(e)}"
            results.append(ToolMessage(
                id=id,
                name=function_name,
                arguments=function_arguments,
                result=result,
                error=error).with_tool_def(target_tool))
        return results

    def list_models(self) -> list[str]:
        return get_valid_models(
            custom_llm_provider=self.provider.value,
            check_provider_endpoint=True,
            api_base=self.base_url,
            api_key=self.api_key)

    def generate_text_sync(self, params: LlmRequestParams) -> GenerateTextResponse:
        response = completion(**self._param_parser.parse_nonstream(params))
        response = cast(LiteLlmModelResponse, response)
        choices = cast(list[LiteLlmModelResponseChoices], response.choices)
        message = choices[0].message
        assistant_message = AssistantMessage\
                            .from_litellm_message(message)\
                            .with_request_params(params)
        result: GenerateTextResponse = [assistant_message]
        if (tools_and_tool_calls := self._should_resolve_tool_calls(params, assistant_message)):
            tools, tool_calls = tools_and_tool_calls
            result += self._execute_tool_calls_sync(tools, tool_calls)
        return result

    async def generate_text(self, params: LlmRequestParams) -> GenerateTextResponse:
        response = await acompletion(**self._param_parser.parse_nonstream(params))
        response = cast(LiteLlmModelResponse, response)
        choices = cast(list[LiteLlmModelResponseChoices], response.choices)
        message = choices[0].message
        assistant_message = AssistantMessage\
                            .from_litellm_message(message)\
                            .with_request_params(params)
        result: GenerateTextResponse = [assistant_message]
        if (tools_and_tool_calls := self._should_resolve_tool_calls(params, assistant_message)):
            tools, tool_calls = tools_and_tool_calls
            result += await self._execute_tool_calls(tools, tool_calls)
        return result

    def stream_text_sync(self, params: LlmRequestParams) -> StreamTextResponseSync:
        def stream(response: CustomStreamWrapper) -> Generator[MessageChunk]:
            nonlocal message_collector
            for chunk in response:
                chunk = cast(LiteLlmModelResponseStream, chunk)
                yield from openai_chunk_normalizer(chunk)
                message_collector.collect(chunk)

            message = message_collector.get_message().with_request_params(params)
            full_message_queue.put(message)
            if (tools_and_tool_calls := self._should_resolve_tool_calls(params, message)):
                tools, tool_calls = tools_and_tool_calls
                tool_messages = self._execute_tool_calls_sync(tools, tool_calls)
                for tool_message in tool_messages:
                    full_message_queue.put(tool_message)
            full_message_queue.put(None)

        response = completion(**self._param_parser.parse_stream(params))
        message_collector = AssistantMessageCollector()
        returned_stream = stream(cast(CustomStreamWrapper, response))
        full_message_queue = queue.Queue[AssistantMessage | ToolMessage | None]()
        return returned_stream, full_message_queue

    async def stream_text(self, params: LlmRequestParams) -> StreamTextResponseAsync:
        async def stream(response: CustomStreamWrapper) -> AsyncGenerator[TextChunk | ReasoningChunk | AudioChunk | ImageChunk | ToolCallChunk]:
            nonlocal message_collector
            async for chunk in response:
                chunk = cast(LiteLlmModelResponseStream, chunk)
                for normalized_chunk in openai_chunk_normalizer(chunk):
                    yield normalized_chunk
                message_collector.collect(chunk)

            message = message_collector.get_message().with_request_params(params)
            await full_message_queue.put(message)
            if (tools_and_tool_calls := self._should_resolve_tool_calls(params, message)):
                tools, tool_calls = tools_and_tool_calls
                tool_messages = await self._execute_tool_calls(tools, tool_calls)
                for tool_message in tool_messages:
                    await full_message_queue.put(tool_message)
            await full_message_queue.put(None)

        response = await acompletion(**self._param_parser.parse_stream(params))
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

    "LLM",
    "LlmRequestParams",

    "ToolFn",
    "ToolDef",
    "RawToolDef",
    "ToolLike",
    "execute_tool",
    "execute_tool_sync",

    "ChatMessage",
    "UserMessage",
    "SystemMessage",
    "AssistantMessage",
    "ToolMessage",

    "MessageChunk",
    "TextChunk",
    "ReasoningChunk",
    "AudioChunk",
    "ImageChunk",
    "ToolCallChunk",

    "GenerateTextResponse",
    "StreamTextResponseSync",
    "StreamTextResponseAsync",

    "enable_debugging",
    "enable_logging",
]
