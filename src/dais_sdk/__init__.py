import asyncio
import queue
from typing import cast
from collections.abc import AsyncGenerator, Generator
from litellm import CustomStreamWrapper, completion, acompletion
from litellm.utils import ProviderConfigManager
from litellm.types.utils import (
    LlmProviders,
    ModelResponse as LiteLlmModelResponse,
    ModelResponseStream as LiteLlmModelResponseStream
)
from .debug import enable_debugging
from .param_parser import ParamParser
from .stream import AssistantMessageCollector
from .tool.execute import execute_tool_sync, execute_tool
from .tool.toolset import (
    Toolset,
    python_tool,
    PythonToolset,
    McpToolset,
    LocalMcpToolset,
    RemoteMcpToolset,
)
from .tool.utils import find_tool_by_name
from .mcp_client import (
    McpClient,
    McpTool,
    McpToolResult,
    LocalMcpClient,
    RemoteMcpClient,
    LocalServerParams,
    RemoteServerParams,
    OAuthParams,
)
from .types import (
    GenerateTextResponse,
    StreamTextResponseSync, StreamTextResponseAsync,
    FullMessageQueueSync, FullMessageQueueAsync,
)
from .types.request_params import LlmRequestParams
from .types.tool import ToolFn, ToolDef, RawToolDef, ToolLike
from .types.exceptions import (
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
from .types.message import (
    ChatMessage, UserMessage, SystemMessage, AssistantMessage, ToolMessage,
    MessageChunk, TextChunk, UsageChunk, ReasoningChunk, AudioChunk, ImageChunk, ToolCallChunk,
    openai_chunk_normalizer
)
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
    async def execute_tool_call(
            params: LlmRequestParams,
            incomplete_tool_message: ToolMessage
            ) -> tuple[str | None, str | None]:
        """
        Receive incomplete tool messages, execute the tool calls and
        return the result and error tuple.
        """
        name, arguments = incomplete_tool_message.name, incomplete_tool_message.arguments
        tool_def = params.find_tool(incomplete_tool_message.name)
        if tool_def is None:
            raise LlmRequestParams.ToolDoesNotExistError(name)

        result, error = None, None
        try:
            result = await execute_tool(tool_def, arguments)
        except Exception as e:
            error = f"{type(e).__name__}: {str(e)}"
        return result, error

    @staticmethod
    def execute_tool_call_sync(
            params: LlmRequestParams,
            incomplete_tool_message: ToolMessage
            ) -> tuple[str | None, str | None]:
        """
        Synchronous version of `execute_tool_call`.
        """
        name, arguments = incomplete_tool_message.name, incomplete_tool_message.arguments
        tool_def = params.find_tool(incomplete_tool_message.name)
        if tool_def is None:
            raise LlmRequestParams.ToolDoesNotExistError(name)

        result, error = None, None
        try:
            result = execute_tool_sync(tool_def, arguments)
        except Exception as e:
            error = f"{type(e).__name__}: {str(e)}"
        return result, error

    def _resolve_tool_calls_sync(self, params: LlmRequestParams, assistant_message: AssistantMessage) -> Generator[ToolMessage]:
        if not params.execute_tools: return
        if (incomplete_tool_messages
            := assistant_message.get_incomplete_tool_messages()) is None:
            return
        for incomplete_tool_message in incomplete_tool_messages:
            try:
                result, error = LLM.execute_tool_call_sync(params, incomplete_tool_message)
            except LlmRequestParams.ToolDoesNotExistError as e:
                logger.warning(f"{e.message} Skipping this tool call.")
                continue
            yield ToolMessage(
                tool_call_id=incomplete_tool_message.tool_call_id,
                name=incomplete_tool_message.name,
                arguments=incomplete_tool_message.arguments,
                result=result,
                error=error)

    async def _resolve_tool_calls(self, params: LlmRequestParams, assistant_message: AssistantMessage) -> AsyncGenerator[ToolMessage]:
        if not params.execute_tools: return
        if (incomplete_tool_messages :=
            assistant_message.get_incomplete_tool_messages()) is None:
            return
        for incomplete_tool_message in incomplete_tool_messages:
            try:
                result, error = await LLM.execute_tool_call(params, incomplete_tool_message)
            except LlmRequestParams.ToolDoesNotExistError as e:
                logger.warning(f"{e.message} Skipping this tool call.")
                continue
            yield ToolMessage(
                tool_call_id=incomplete_tool_message.tool_call_id,
                name=incomplete_tool_message.name,
                arguments=incomplete_tool_message.arguments,
                result=result,
                error=error)

    def list_models(self) -> list[str]:
        provider_config = ProviderConfigManager.get_provider_model_info(
            model=None,
            provider=self.provider,
        )

        if provider_config is None:
            raise ValueError(f"The '{self.provider}' provider is not supported to list models.")

        try:
            models = provider_config.get_models(
                api_key=self.api_key,
                api_base=self.base_url
            )
        except Exception as e:
            raise e
        return models

    def generate_text_sync(self, params: LlmRequestParams) -> GenerateTextResponse:
        response = completion(**self._param_parser.parse_nonstream(params))
        response = cast(LiteLlmModelResponse, response)
        assistant_message = AssistantMessage.from_litellm_message(response)
        result: GenerateTextResponse = [assistant_message]
        for tool_message in self._resolve_tool_calls_sync(params, assistant_message):
            result.append(tool_message)
        return result

    async def generate_text(self, params: LlmRequestParams) -> GenerateTextResponse:
        response = await acompletion(**self._param_parser.parse_nonstream(params))
        response = cast(LiteLlmModelResponse, response)
        assistant_message = AssistantMessage.from_litellm_message(response)
        result: GenerateTextResponse = [assistant_message]
        async for tool_message in self._resolve_tool_calls(params, assistant_message):
            result.append(tool_message)
        return result

    def stream_text_sync(self, params: LlmRequestParams) -> StreamTextResponseSync:
        """
        Returns:
            - stream: Generator yielding `MessageChunk` objects
            - full_message_queue: Queue containing complete `AssistantMessage`, `ToolMessage` (or `None` when done)
        """
        def stream(response: CustomStreamWrapper) -> Generator[MessageChunk]:
            nonlocal message_collector
            for chunk in response:
                chunk = cast(LiteLlmModelResponseStream, chunk)
                yield from openai_chunk_normalizer(chunk)
                message_collector.collect(chunk)

            message = message_collector.get_message()
            full_message_queue.put(message)

            for tool_message in self._resolve_tool_calls_sync(params, message):
                full_message_queue.put(tool_message)
            full_message_queue.put(None)

        response = completion(**self._param_parser.parse_stream(params))
        message_collector = AssistantMessageCollector()
        returned_stream = stream(cast(CustomStreamWrapper, response))
        full_message_queue = FullMessageQueueSync()
        return returned_stream, full_message_queue

    async def stream_text(self, params: LlmRequestParams) -> StreamTextResponseAsync:
        """
        Returns:
            - stream: Generator yielding `MessageChunk` objects
            - full_message_queue: Queue containing complete `AssistantMessage`, `ToolMessage` (or `None` when done)
        """
        async def stream(response: CustomStreamWrapper) -> AsyncGenerator[MessageChunk]:
            nonlocal message_collector
            async for chunk in response:
                chunk = cast(LiteLlmModelResponseStream, chunk)
                for normalized_chunk in openai_chunk_normalizer(chunk):
                    yield normalized_chunk
                message_collector.collect(chunk)

            message = message_collector.get_message()
            await full_message_queue.put(message)
            async for tool_message in self._resolve_tool_calls(params, message):
                await full_message_queue.put(tool_message)
            await full_message_queue.put(None)

        response = await acompletion(**self._param_parser.parse_stream(params))
        message_collector = AssistantMessageCollector()
        returned_stream = stream(cast(CustomStreamWrapper, response))
        full_message_queue = FullMessageQueueAsync()
        return returned_stream, full_message_queue

__all__ = [
    "LLM",
    "LlmProviders",
    "LlmRequestParams",

    "Toolset",
    "python_tool",
    "PythonToolset",
    "McpToolset",
    "LocalMcpToolset",
    "RemoteMcpToolset",

    "McpClient",
    "McpTool",
    "McpToolResult",
    "LocalMcpClient",
    "RemoteMcpClient",
    "LocalServerParams",
    "RemoteServerParams",
    "OAuthParams",

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
    "UsageChunk",
    "ReasoningChunk",
    "AudioChunk",
    "ImageChunk",
    "ToolCallChunk",

    "GenerateTextResponse",
    "StreamTextResponseSync",
    "StreamTextResponseAsync",
    "FullMessageQueueSync",
    "FullMessageQueueAsync",

    "enable_debugging",
    "enable_logging",

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
]
