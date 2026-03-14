import asyncio
from typing import TYPE_CHECKING
from collections.abc import Generator
from .logger import enable_logging

if TYPE_CHECKING:
    from .providers import BaseProvider
    from .types import (
        LlmRequestParams, StreamMessageGenerator,
        StreamMessageEvent, AssistantMessage,
    )

class LLM:
    def __init__(self, provider: BaseProvider):
        self._provider = provider

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
    "enable_logging",
]
