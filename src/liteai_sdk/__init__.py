from typing import cast
from litellm import completion, acompletion
from litellm.types.utils import LlmProviders, ModelResponse as LiteLlmModelResponse,\
                                Choices as LiteLlmModelResponseChoices
from .types import ChatMessage, GenerateTextRequest, ModelResponse
from .tool import ToolFn, RawToolDefinition, prepare_tools

class LLM:
    def __init__(self,
                 provider: LlmProviders,
                 base_url: str,
                 api_key: str):
        self.provider = provider
        self.base_url = base_url
        self.api_key = api_key
    
    def _generate_text_params(self, params: GenerateTextRequest):
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

    def generate_text_sync(self, params: GenerateTextRequest) -> ModelResponse:
        response = completion(**self._generate_text_params(params))
        response = cast(LiteLlmModelResponse, response)
        choices = cast(list[LiteLlmModelResponseChoices], response.choices)
        return choices[0].message

    async def generate_text(self, params: GenerateTextRequest) -> ModelResponse:
        response = await acompletion(**self._generate_text_params(params))
        response = cast(LiteLlmModelResponse, response)
        choices = cast(list[LiteLlmModelResponseChoices], response.choices)
        return choices[0].message

__all__ = [
    "LLM",
    "GenerateTextRequest",
    "ToolFn",
    "RawToolDefinition",
    "ChatMessage",
    "ModelResponse"
]
