from litellm import completion, acompletion
from litellm.types.utils import LlmProviders
from .types import ChatMessages
from .tool import ToolFn, RawToolDefinition, prepare_tools

class LLM:
    def __init__(self,
                 provider: LlmProviders,
                 base_url: str,
                 api_key: str):
        self.provider = provider
        self.base_url = base_url
        self.api_key = api_key

    def generate_text_sync(self,
                   model: str,
                   messages: ChatMessages,
                   tools: list[ToolFn | RawToolDefinition] | None = None,
                   **kwargs):
        response = completion(model=f"{self.provider.name}/{model}",
                              messages=messages,
                              base_url=self.base_url,
                              api_key=self.api_key,
                              tools=tools and prepare_tools(tools),
                              **kwargs)

    async def generate_text(self,
                         model: str,
                         messages: ChatMessages,
                         tools: list[ToolFn | RawToolDefinition] | None = None,
                         **kwargs):
        response = await acompletion(model=f"{self.provider.name}/{model}",
                                    messages=messages,
                                    base_url=self.base_url,
                                    api_key=self.api_key,
                                    tools=tools and prepare_tools(tools),
                                    **kwargs)

__all__ = [
    "LLM",
    "ToolFn",
    "RawToolDefinition",
    "ChatMessages"
]
