from typing import Any
from litellm.types.utils import LlmProviders
from .tool import prepare_tools
from .types import LlmRequestParams, ToolMessage

ParsedParams = dict[str, Any]

class ParamParser:
    def __init__(self,
                 provider: LlmProviders,
                 base_url: str,
                 api_key: str):
        self._provider = provider
        self._base_url = base_url
        self._api_key = api_key

    def _parse(self, params: LlmRequestParams) -> ParsedParams:
        tools = params.tools and prepare_tools(params.tools)
        transformed_messages = []
        for message in params.messages:
            if type(message) is ToolMessage and\
               message.result is None and\
               message.error is None:
                # skip ToolMessage that is not resolved
                continue
            transformed_messages.append(message.to_litellm_message())

        return {
            "model": f"{self._provider.value}/{params.model}",
            "messages": transformed_messages,
            "base_url": self._base_url,
            "api_key": self._api_key,
            "tools": tools,
            "tool_choice": params.tool_choice,
            "timeout": params.timeout_sec,
            "extra_headers": params.headers,
            **(params.extra_args or {})
        }

    def parse_nonstream(self, params: LlmRequestParams) -> ParsedParams:
        parsed = self._parse(params)
        parsed["stream"] = False
        return parsed

    def parse_stream(self, params: LlmRequestParams) -> ParsedParams:
        parsed = self._parse(params)
        parsed["stream"] = True
        return parsed
