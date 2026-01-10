from abc import ABC
from mcp.types import TextContent, ImageContent, AudioContent, ResourceLink, EmbeddedResource, TextResourceContents, BlobResourceContents
from ..._mcp_client.mcp_client import McpClient, Tool, ToolResult
from ..._mcp_client.local_mcp_client import LocalMcpClient, LocalServerParams
from ..._mcp_client.remote_mcp_client import RemoteMcpClient, RemoteServerParams, OAuthParams
from .toolset import Toolset
from ...types.tool import ToolDef, ToolLike

class McpToolset(ABC, Toolset):
    def __init__(self, client: McpClient):
        self._client = client
        self._tools_cache: list[ToolLike] | None = None

    def _mcp_tool_to_tool_def(self, mcp_tool: Tool) -> ToolDef:
        async def wrapper(**kwargs) -> str:
            result = await self._client.call_tool(mcp_tool.name, kwargs)
            return self._format_tool_result(result)

        tool_def = ToolDef(
            name=mcp_tool.name,
            description=mcp_tool.description or f"MCP tool: {mcp_tool.name}",
            parameters=mcp_tool.inputSchema,
            execute=wrapper
        )
        return tool_def

    def _format_tool_result(self, result: ToolResult) -> str:
        is_error, content = result.is_error, result.content
        content_parts = []

        if is_error:
            content_parts.append("Error executing tool:")

        for block in content:
            match block:
                case TextContent():
                    content_parts.append(block.text)
                case ImageContent():
                    content_parts.append(f"[Generated Image: {block.mimeType}]")
                case AudioContent():
                    content_parts.append(f"[Generated Audio: {block.mimeType}]")
                case ResourceLink():
                    details = [f"Resource Reference: {block.uri}"]
                    if block.mimeType: details.append(f"Type: {block.mimeType}")
                    if block.size: details.append(f"Size: {block.size} bytes")
                    if block.description: details.append(f"Description: {block.description}")
                    content_parts.append("\n".join(details))
                case EmbeddedResource():
                    resource = block.resource
                    header = f"Resource ({resource.uri}):"
                    if isinstance(resource, TextResourceContents):
                        content_parts.append(f"{header}\n{resource.text}")
                    elif isinstance(resource, BlobResourceContents):
                        content_parts.append(f"{header}\n[Binary data: {resource.mimeType}]")
                case _:
                    logger.warning(f"Unknown tool result block type: {type(block)}")

        return "\n\n".join(content_parts)

    async def connect(self) -> None:
        await self._client.connect()
        await self.refresh_tools()

    async def disconnect(self) -> None:
        await self._client.disconnect()
        self._tools_cache = None

    async def refresh_tools(self) -> None:
        mcp_tools = await self._client.list_tools()
        self._tools_cache = [self._mcp_tool_to_tool_def(tool) for tool in mcp_tools]

    def get_tools(self) -> list[ToolLike]:
        if self._tools_cache is None:
            raise RuntimeError(f"Not connected to MCP server. Call await {self.__class__.__name__}(...).connect() first")
        return self._tools_cache


class LocalMcpToolset(McpToolset):    
    def __init__(self, name: str, params: LocalServerParams):
        client = LocalMcpClient(name, params)
        super().__init__(client)

class RemoteMcpToolset(McpToolset):    
    def __init__(self, name: str, params: RemoteServerParams):
        client = RemoteMcpClient(name, params)
        super().__init__(client)
