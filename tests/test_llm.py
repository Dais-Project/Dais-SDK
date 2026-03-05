import pytest

from dais_sdk import LLM
from dais_sdk.providers.base_provider import BaseProvider
from dais_sdk.types.event import AssistantMessageEvent, TextChunkEvent
from dais_sdk.types.message import AssistantMessage, UserMessage
from dais_sdk.types.request_params import LlmRequestParams


class StubProvider(BaseProvider):
    def __init__(self, base_url: str = "", api_key: str = ""):
        self.base_url = base_url
        self.api_key = api_key
        self.calls: list[tuple[str, LlmRequestParams | None]] = []
        self.models = ["mock-model-1", "mock-model-2"]
        self.nonstream_response = AssistantMessage()
        self.nonstream_response.content = "nonstream response"
        final_message = AssistantMessage()
        final_message.content = "final response"
        self.stream_events = [
            TextChunkEvent(content="chunk-1"),
            TextChunkEvent(content="chunk-2"),
            AssistantMessageEvent(message=final_message),
        ]

    async def list_models(self) -> list[str]:
        self.calls.append(("list_models", None))
        return self.models

    async def request_nonstream(self, params: LlmRequestParams) -> AssistantMessage:
        self.calls.append(("request_nonstream", params))
        return self.nonstream_response

    async def request_stream(self, params: LlmRequestParams):
        self.calls.append(("request_stream", params))
        for event in self.stream_events:
            yield event


@pytest.fixture
def sample_params() -> LlmRequestParams:
    return LlmRequestParams(model="mock-model", messages=[UserMessage(content="hello")])


@pytest.mark.asyncio
async def test_list_models_delegates_to_provider() -> None:
    provider = StubProvider()
    llm = LLM(provider)

    models = await llm.list_models()

    assert models == ["mock-model-1", "mock-model-2"]
    assert provider.calls == [("list_models", None)]


def test_list_models_sync_delegates_to_provider() -> None:
    provider = StubProvider()
    llm = LLM(provider)

    models = llm.list_models_sync()

    assert models == ["mock-model-1", "mock-model-2"]
    assert provider.calls == [("list_models", None)]


@pytest.mark.asyncio
async def test_generate_text_delegates_to_provider(sample_params: LlmRequestParams) -> None:
    provider = StubProvider()
    llm = LLM(provider)

    message = await llm.generate_text(sample_params)

    assert message.content == "nonstream response"
    assert provider.calls == [("request_nonstream", sample_params)]


def test_generate_text_sync_delegates_to_provider(sample_params: LlmRequestParams) -> None:
    provider = StubProvider()
    llm = LLM(provider)

    message = llm.generate_text_sync(sample_params)

    assert message.content == "nonstream response"
    assert provider.calls == [("request_nonstream", sample_params)]


@pytest.mark.asyncio
async def test_stream_text_passthrough_events(sample_params: LlmRequestParams) -> None:
    provider = StubProvider()
    llm = LLM(provider)

    events = [event async for event in llm.stream_text(sample_params)]

    assert events == provider.stream_events
    assert provider.calls == [("request_stream", sample_params)]


def test_stream_text_sync_passthrough_first_event(sample_params: LlmRequestParams) -> None:
    provider = StubProvider()
    llm = LLM(provider)

    generator = llm.stream_text_sync(sample_params)
    first_event = next(generator)
    generator.close()

    assert first_event == provider.stream_events[0]
    assert provider.calls == [("request_stream", sample_params)]
