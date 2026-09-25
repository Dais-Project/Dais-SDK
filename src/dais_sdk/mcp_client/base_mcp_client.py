from abc import ABC, abstractmethod
from typing import Any, NamedTuple
from mcp import Tool as McpTool
from mcp.types import ContentBlock as ToolResultBlock


Tool = McpTool

class ToolResult(NamedTuple):
    is_error: bool
    content: list[ToolResultBlock]

class McpClient(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    def description(self) -> str | None:
        return None

    @abstractmethod
    async def connect(self): ...
    @abstractmethod
    async def disconnect(self): ...
    @abstractmethod
    async def list_tools(self) -> list[Tool]:
        """
        Raises:
            McpClientNotEstablishedError: If the client is not established.
        """
    @abstractmethod
    async def call_tool(
        self, tool_name: str, arguments: dict[str, Any] | None = None
    ) -> ToolResult:
        """
        Raises:
            McpClientNotEstablishedError: If the client is not established.
        """

class McpClientNotEstablishedError(RuntimeError):
    def __init__(self):
        super().__init__("MCP Session not established, please call connect() first")
