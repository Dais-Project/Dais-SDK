from typing import cast
from litellm import CustomStreamWrapper, completion, acompletion
from litellm.utils import get_valid_models
from litellm.types.utils import LlmProviders,\
                                ModelResponse as LiteLlmModelResponse,\
                                ModelResponseStream as LiteLlmModelResponseStream,\
                                Choices as LiteLlmModelResponseChoices
from .types import ChatMessage, LlmRequestParams, ModelResponse
from .tool import ToolFn, ToolDef, prepare_tools

class LLM:
    def __init__(self,
                 provider: LlmProviders,
                 base_url: str,
                 api_key: str):
        self.provider = provider
        self.base_url = base_url
        self.api_key = api_key
    
    def _parse_params_nonstream(self, params: LlmRequestParams):
        tools = params.tools and prepare_tools(params.tools)
        return {
            "model": f"{self.provider.value}/{params.model}",
            "messages": params.messages,
            "base_url": self.base_url,
            "api_key": self.api_key,
            "tools": tools,
            "stream": False,
            "extra_headers": params.headers,
            **(params.extra_args or {})
        }
    
    def _parse_params_stream(self, params: LlmRequestParams):
        tools = params.tools and prepare_tools(params.tools)
        return {
            "model": f"{self.provider.value}/{params.model}",
            "messages": params.messages,
            "base_url": self.base_url,
            "api_key": self.api_key,
            "tools": tools,
            "stream": True,
            "extra_headers": params.headers,
            **(params.extra_args or {})
        }
    
    def list_models(self) -> list[str]:
        return get_valid_models(
            custom_llm_provider=self.provider.value,
            check_provider_endpoint=True,
            api_base=self.base_url,
            api_key=self.api_key)

    def generate_text_sync(self, params: LlmRequestParams) -> ModelResponse:
        response = completion(**self._parse_params_nonstream(params))
        response = cast(LiteLlmModelResponse, response)
        choices = cast(list[LiteLlmModelResponseChoices], response.choices)
        return choices[0].message

    async def generate_text(self, params: LlmRequestParams) -> ModelResponse:
        response = await acompletion(**self._parse_params_nonstream(params))
        response = cast(LiteLlmModelResponse, response)
        choices = cast(list[LiteLlmModelResponseChoices], response.choices)
        return choices[0].message
    
    def stream_text_sync(self, params: LlmRequestParams):
        response = completion(**self._parse_params_stream(params))
        for chunk in response:
            chunk = cast(LiteLlmModelResponseStream, chunk)
            yield chunk.choices[0]

    async def stream_text(self, params: LlmRequestParams):
        response = await acompletion(**self._parse_params_stream(params))
        response = cast(CustomStreamWrapper, response)
        async for chunk in response:
            chunk = cast(LiteLlmModelResponseStream, chunk)
            yield chunk.choices[0]

__all__ = [
    "LLM",
    "LlmRequestParams",
    "ToolFn",
    "ToolDef",
    "ChatMessage",
    "ModelResponse"
]
