import httpx
import webbrowser
from dataclasses import dataclass
from contextlib import AsyncExitStack
from typing import Any, NamedTuple, override
from mcp import ClientSession
from mcp.client.auth import OAuthClientProvider
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.auth import OAuthClientMetadata
from pydantic import AnyUrl, BaseModel, Field, ConfigDict, SkipValidation
from .oauth_server import LocalOAuthServer, OAuthCode, TokenStorage, InMemoryTokenStorage
from .base_mcp_client import McpClient, Tool, ToolResult, McpSessionNotEstablishedError
from ..logger import logger

@dataclass
class OAuthParams:
    oauth_scopes: list[str] | None = None
    oauth_timeout: int = 120
    oauth_token_storage: SkipValidation[TokenStorage] = Field(
        default_factory=InMemoryTokenStorage,
        exclude=True,
    )

class RemoteServerParams(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    url: str
    bearer_token: str | None = None
    oauth_params: OAuthParams | None = None
    http_headers: dict[str, str] | None = None

# --- --- --- --- --- ---

class OAuthContext(NamedTuple):
    client: httpx.AsyncClient
    server: LocalOAuthServer

class RemoteMcpClient(McpClient):
    def __init__(self,
                 name: str,
                 params: RemoteServerParams,
                 storage: TokenStorage | None = None):
        self._name = name
        self._params = params
        self._session: ClientSession | None = None
        self._exit_stack: AsyncExitStack | None = None
        self._oauth_context: OAuthContext | None = self._init_oauth()
        if self._params.oauth_params is not None and storage is not None:
            self._params.oauth_params.oauth_token_storage = storage

    @property
    @override
    def name(self) -> str:
        return self._name

    def _init_http_headers(self) -> dict[str, str] | None:
        if self._params.http_headers is None and self._params.bearer_token is None:
            return None
        headers = {}
        if self._params.http_headers is not None:
            headers.update(self._params.http_headers)
        if self._params.bearer_token is not None:
            headers["Authorization"] = f"Bearer {self._params.bearer_token}"
        return headers

    def _init_oauth(self) -> OAuthContext | None:
        if self._params.oauth_params is None:
            return None

        oauth_params = self._params.oauth_params

        server = LocalOAuthServer(timeout=oauth_params.oauth_timeout)
        scopes = None
        if oauth_params.oauth_scopes is not None:
            scopes = " ".join(oauth_params.oauth_scopes)

        client_provider = OAuthClientProvider(
            server_url=self._params.url,
            client_metadata=OAuthClientMetadata(
                client_name=self._name,
                redirect_uris=[AnyUrl(server.callback_url)],
                grant_types=["authorization_code", "refresh_token"],
                response_types=["code"],
                scope=scopes,
                token_endpoint_auth_method="none",
            ),
            storage=oauth_params.oauth_token_storage,
            redirect_handler=self._handle_redirect,
            callback_handler=self._handle_oauth_callback,
        )
        client = httpx.AsyncClient(auth=client_provider,
                                   headers=self._init_http_headers(),
                                   follow_redirects=True)
        return OAuthContext(client, server)

    async def _handle_redirect(self, url: str) -> None:
        logger.info("[OAuth] Authentication required, opening browser...")
        logger.info(f"[OAuth] If browser does not open automatically, copy and open the following link: \n{url}\n")
        try:
            webbrowser.open(url)
        except Exception as e:
            logger.error(f"[OAuth] Not able to open browser", exc_info=e)

    async def _handle_oauth_callback(self) -> OAuthCode:
        if self._oauth_context is None:
            raise ValueError("OAuth context not initialized")
        return await self._oauth_context.server.wait_for_code()

    @override
    async def connect(self):
        self._exit_stack = AsyncExitStack()
        if self._oauth_context:
            http_client = self._oauth_context.client
            await self._oauth_context.server.start()
        else:
            http_client = await self._exit_stack.enter_async_context(
                httpx.AsyncClient(headers=self._init_http_headers(), follow_redirects=True))

        try:
            read_stream, write_stream, _ = await self._exit_stack.enter_async_context(
                streamable_http_client(self._params.url, http_client=http_client)
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
    async def disconnect(self):
        try:
            if self._exit_stack:
                await self._exit_stack.aclose()
        finally:
            self._session = None
            self._exit_stack = None

            if self._oauth_context:
                try: await self._oauth_context.client.aclose()
                except Exception: pass
                try: await self._oauth_context.server.stop()
                except Exception: pass
