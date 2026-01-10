import asyncio
import socket
import uvicorn
from starlette.applications import Starlette
from starlette.responses import HTMLResponse
from starlette.routing import Route
from starlette.requests import Request
from mcp.client.auth import TokenStorage
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken

CALLBACK_PAGE = """
<html>
    <body style="font-family: sans-serif; text-align: center; padding-top: 50px;">
        <h2 style="color: green;">Login Successful!</h2>
        <p>You can close this window now.</p>
        <script>window.close();</script>
    </body>
</html>
"""

OAuthCode = tuple[str, str | None]
OAuthCallbackFuture = asyncio.Future[OAuthCode]

class InMemoryTokenStorage(TokenStorage):
    def __init__(self):
        self.tokens: OAuthToken | None = None
        self.client_info: OAuthClientInformationFull | None = None

    async def get_tokens(self) -> OAuthToken | None:
        return self.tokens

    async def set_tokens(self, tokens: OAuthToken) -> None:
        self.tokens = tokens

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        return self.client_info

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        self.client_info = client_info

def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("0.0.0.0", 0))
        port = s.getsockname()[1]
        return port

class LocalOAuthServer:
    def __init__(self, timeout: int):
        self._port = _find_free_port()
        self._timeout = timeout
        self._future = OAuthCallbackFuture()
        self._server: uvicorn.Server | None = None
        self._server_task: asyncio.Task | None = None

    @property
    def callback_url(self) -> str:
        return f"http://localhost:{self._port}/callback"

    async def _handle_callback(self, request: Request):
        params = request.query_params

        # check if error
        if "error" in params:
            error = params.get("error")
            desc = params.get("error_description", "Unknown error")
            if not self._future.done():
                self._future.set_exception(RuntimeError(f"OAuth Error: {error} - {desc}"))
            return HTMLResponse(f"<h3>Auth failed: {error}</h3>", status_code=400)

        code = params.get("code")
        state = params.get("state")

        if not code:
            return HTMLResponse("<h3>Missing 'code' parameter</h3>", status_code=400)

        if not self._future.done():
            self._future.set_result((code, state))

        return HTMLResponse(CALLBACK_PAGE)

    async def wait_for_code(self) -> OAuthCode:
        auth_code =  await asyncio.wait_for(self._future, timeout=self._timeout)
        self._future = OAuthCallbackFuture()
        return auth_code

    async def start(self):
        routes = [
            Route("/callback", self._handle_callback, methods=["GET"])
        ]
        app = Starlette(routes=routes)

        config = uvicorn.Config(app=app, host="127.0.0.1", port=self._port, log_level="error")
        self._server = uvicorn.Server(config)
        self._server_task = asyncio.create_task(self._server.serve())

    async def stop(self):
        if self._server:
            self._server.should_exit = True
        if self._server_task:
            await self._server_task
