import asyncio
from typing import Any, override
from mcp import Client, StdioServerParameters as StdioServerParams
from .base_mcp_client import McpClient, Tool, ToolResult, McpClientNotEstablishedError


class LocalServerParams(StdioServerParams): ...

class LocalMcpClient(McpClient):
    def __init__(self, name: str, params: LocalServerParams):
        self._name: str = name
        self._description: str | None = None
        self._params: LocalServerParams = params
        self._client: Client | None = None
        self._run_task: asyncio.Task | None = None

        self._connect_error: BaseException | None = None
        self._ready_event = asyncio.Event()
        self._disconnect_event = asyncio.Event()

    async def _run(self):
        try:
            async with Client(self._params, mode="auto") as client:
                self._client = client
                self._description = client.instructions
                self._ready_event.set()
                await self._disconnect_event.wait()
        except BaseException as e:
            self._connect_error = e
            self._ready_event.set()
        finally:
            self._client = None
            self._description = None

    @property
    @override
    def name(self) -> str:
        return self._name

    @property
    @override
    def description(self) -> str | None:
        return self._description

    @override
    async def connect(self):
        self._run_task = asyncio.create_task(self._run())
        await self._ready_event.wait()
        if self._connect_error:
            raise self._connect_error

    @override
    async def list_tools(self) -> list[Tool]:
        if not self._client:
            raise McpClientNotEstablishedError()

        result = await self._client.list_tools()
        return result.tools

    @override
    async def call_tool(
        self, tool_name: str, arguments: dict[str, Any] | None = None
    ) -> ToolResult:
        if not self._client:
            raise McpClientNotEstablishedError()

        response = await self._client.call_tool(tool_name, arguments)
        return ToolResult(response.is_error, response.content)

    @override
    async def disconnect(self) -> None:
        try:
            if self._disconnect_event:
                self._disconnect_event.set()

            if self._run_task and not self._run_task.done():
                try:
                    await self._run_task
                except Exception: pass
        finally:
            self._ready_event.clear()
            self._disconnect_event.clear()
            self._connect_error = None
            self._run_task = None
