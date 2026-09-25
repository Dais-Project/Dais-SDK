import httpx2
import pytest
from mcp import MCPError
from mcp.client.auth import AuthorizationCodeResult
from mcp.types import AudioContent, CallToolResult, ImageContent, TextContent, Tool

from dais_sdk.mcp_client.base_mcp_client import ToolResult
from dais_sdk.mcp_client.local_mcp_client import LocalMcpClient, LocalServerParams
from dais_sdk.mcp_client.oauth_server import LocalOAuthServer
from dais_sdk.mcp_client.remote_mcp_client import OAuthParams, RemoteMcpClient, RemoteServerParams
from dais_sdk.tool.exceptions import McpConnectionErrorCode
from dais_sdk.tool.toolset.mcp_toolset import McpToolset
from dais_sdk.types import AudioBlock, ImageBlock


class Session:
    async def call_tool(self, name, arguments):
        return CallToolResult(content=[TextContent(type="text", text="done")], is_error=False)


@pytest.mark.asyncio
@pytest.mark.parametrize("remote", [False, True])
async def test_client_maps_v2_call_result(remote):
    if remote:
        client = RemoteMcpClient("test", RemoteServerParams(url="http://localhost/mcp", bearer_token=None, oauth_params=None, http_headers=None))
    else:
        client = LocalMcpClient("test", LocalServerParams(command="python"))
    client._session = Session()
    assert (await client.call_tool("test")).is_error is False


def test_toolset_reads_v2_schema_and_mime_types():
    client = LocalMcpClient("test", LocalServerParams(command="python"))
    toolset = McpToolset(client)
    schema = {"type": "object", "properties": {}}
    tool = Tool(name="example", input_schema=schema)
    assert toolset._mcp_tool_to_tool_def(tool).parameters == schema
    result = toolset._format_tool_result(
        ToolResult(False, [ImageContent(type="image", data="aGVsbG8=", mime_type="image/png"),
                 AudioContent(type="audio", data="aGVsbG8=", mime_type="audio/wav")])
    )
    assert isinstance(result[0], ImageBlock) and result[0].source.mime_type == "image/png"
    assert isinstance(result[1], AudioBlock) and result[1].source.mime_type == "audio/wav"


def test_v2_error_classification():
    assert McpConnectionErrorCode.from_exception(MCPError(-32600, "invalid")) == McpConnectionErrorCode.PROTOCOL_ERROR
    assert McpConnectionErrorCode.from_exception(httpx2.ConnectTimeout("timeout")) == McpConnectionErrorCode.CONNECTION_TIMEOUT
    assert McpConnectionErrorCode.from_exception(httpx2.ConnectError("failed")) == McpConnectionErrorCode.CONNECTION_FAILED


@pytest.mark.asyncio
async def test_oauth_callback_returns_issuer():
    server = LocalOAuthServer(timeout=1)
    from starlette.requests import Request
    request = Request({"type": "http", "method": "GET", "path": "/callback", "query_string": b"code=abc&state=xyz&iss=https%3A%2F%2Fissuer.example", "headers": []})
    await server._handle_callback(request)
    assert await server.wait_for_code() == AuthorizationCodeResult(code="abc", state="xyz", iss="https://issuer.example")


@pytest.mark.asyncio
async def test_remote_oauth_uses_httpx2():
    client = RemoteMcpClient("test", RemoteServerParams(url="https://example.com/mcp", bearer_token=None, http_headers=None, oauth_params=OAuthParams(oauth_scopes=None)))
    try:
        assert isinstance(client._oauth_context.client, httpx2.AsyncClient)
    finally:
        await client._oauth_context.client.aclose()
