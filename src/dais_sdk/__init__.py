import asyncio
import json
from collections.abc import Generator
from .tool.execute import ToolExceptionHandlerManager, execute_tool
from .tool.toolset import (
    Toolset,
    python_tool,
    PythonToolset,
    McpToolset,
    LocalMcpToolset,
    RemoteMcpToolset,
)
from .tool.utils import get_tool_name
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
from .providers import LlmProviders, BaseProvider, OpenAIProvider
from .types.event import StreamMessageEvent, StreamMessageGenerator
from .types.request_params import LlmRequestParams
from .types.tool import ToolFn, ToolDef, RawToolDef, ToolLike, ToolSchema
from .types.exceptions import (
    LlmToolException,
    ToolDoesNotExistError,
    ToolArgumentDecodeError,
    ToolExecutionError,
)
from .types.message import (
    ChatMessage, UserMessage, SystemMessage, AssistantMessage, ToolMessage,
)
from .logger import enable_logging

class LLM:
    def __init__(self, provider: BaseProvider):
        self._provider = provider
        self._tool_exception_handler_manager = ToolExceptionHandlerManager()

    @property
    def tool_exception_handler_manager(self) -> ToolExceptionHandlerManager:
        return self._tool_exception_handler_manager

    async def execute_tool_call(self,
                                tool: ToolLike,
                                arguments: str | dict) -> tuple[str | None, str | None]:
        """
        Returns:
            A tuple of (result, error)
        """
        result, error = None, None
        try:
            result = await execute_tool(tool, arguments)
        except json.JSONDecodeError as e:
            assert type(arguments) is str
            _error = ToolArgumentDecodeError(get_tool_name(tool), arguments, e)
            error = self._tool_exception_handler_manager.handle(_error)
        except Exception as e:
            _error = ToolExecutionError(tool, arguments, e)
            error = self._tool_exception_handler_manager.handle(_error)
        return result, error

    def execute_tool_call_sync(self,
                               tool: ToolLike,
                               arguments: str | dict
                               ) -> tuple[str | None, str | None]:
        """
        Synchronous wrapper of `execute_tool_call`.
        """
        return asyncio.run(self.execute_tool_call(tool, arguments))


    async def list_models(self) -> list[str]:
        return await self._provider.list_models()

    def list_models_sync(self) -> list[str]:
        return asyncio.run(self.list_models())

    async def generate_text(self, params: LlmRequestParams) -> AssistantMessage:
        return await self._provider.request_nonstream(params)

    def generate_text_sync(self, params: LlmRequestParams) -> AssistantMessage:
        return asyncio.run(self.generate_text(params))

    async def stream_text(self, params: LlmRequestParams) -> StreamMessageGenerator:
        async for chunk in self._provider.request_stream(params):
            yield chunk

    def stream_text_sync(self, params: LlmRequestParams) -> Generator[StreamMessageEvent, None, None]:
        with asyncio.Runner() as runner:
            while True:
                gen = self.stream_text(params)
                try:
                    chunk = runner.run(gen.__anext__())
                    yield chunk
                except StopAsyncIteration:
                    break

__all__ = [
    "LLM",
    "LlmRequestParams",

    "LlmProviders",
    "BaseProvider",
    "OpenAIProvider",

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
    "ToolSchema",
    "execute_tool",

    "ChatMessage",
    "UserMessage",
    "SystemMessage",
    "AssistantMessage",
    "ToolMessage",

    "enable_logging",

    # Exceptions
    "LlmToolException",
    "ToolDoesNotExistError",
    "ToolArgumentDecodeError",
    "ToolExecutionError",
]
