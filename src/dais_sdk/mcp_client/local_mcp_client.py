from contextlib import AsyncExitStack
from typing import Any, override
from mcp import ClientSession, StdioServerParameters as StdioServerParams
from mcp.client.stdio import stdio_client
from .base_mcp_client import McpClient, Tool, ToolResult, McpSessionNotEstablishedError

class LocalServerParams(StdioServerParams): ...

class LocalMcpClient(McpClient):
    def __init__(self, name: str, params: LocalServerParams):
        self._name: str = name
        self._params: LocalServerParams = params
        self._session: ClientSession | None = None
        self._exit_stack: AsyncExitStack | None = None

    @property
    @override
    def name(self) -> str:
        return self._name

    @override
    async def connect(self):
        self._exit_stack = AsyncExitStack()

        try:
            read_stream, write_stream = await self._exit_stack.enter_async_context(
                stdio_client(self._params)
            )
            self._session = await self._exit_stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
            await self._session.initialize()
        except Exception:
            await self.disconnect()
            raise

    @override
    async def list_tools(self) -> list[Tool]:
        if not self._session:
            raise McpSessionNotEstablishedError()

        result = await self._session.list_tools()
        return result.tools

    @override
    async def call_tool(
        self, tool_name: str, arguments: dict[str, Any] | None = None
    ) -> ToolResult:
        if not self._session:
            raise McpSessionNotEstablishedError()

        response = await self._session.call_tool(tool_name, arguments)
        return ToolResult(response.isError, response.content)

    @override
    async def disconnect(self) -> None:
        if self._exit_stack:
            await self._exit_stack.aclose()
            self._session = None
            self._exit_stack = None
