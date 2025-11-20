import asyncio
import json
import queue
from typing import cast
from collections.abc import AsyncGenerator, Generator
from litellm import ChatCompletionAssistantToolCall, CustomStreamWrapper, completion, acompletion
from litellm.utils import get_valid_models
from litellm.types.utils import LlmProviders,\
                                ModelResponse as LiteLlmModelResponse,\
                                ModelResponseStream as LiteLlmModelResponseStream,\
                                Choices as LiteLlmModelResponseChoices
from .stream import AssistantMessageCollector
from .tool import ToolFn, ToolDef, prepare_tools
from .types import LlmRequestParams, GenerateTextResponse, StreamTextResponseSync, StreamTextResponseAsync
from .types.message import AssistantMessageChunk, UserMessage, SystemMessage, AssistantMessage, ToolMessage

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
            "messages": [message.to_litellm_message() for message in params.messages],
            "base_url": self.base_url,
            "api_key": self.api_key,
            "tools": tools,
            "stream": False,
            "timeout": params.timeout_sec,
            "extra_headers": params.headers,
            **(params.extra_args or {})
        }

    def _parse_params_stream(self, params: LlmRequestParams):
        tools = params.tools and prepare_tools(params.tools)
        return {
            "model": f"{self.provider.value}/{params.model}",
            "messages": [message.to_litellm_message() for message in params.messages],
            "base_url": self.base_url,
            "api_key": self.api_key,
            "tools": tools,
            "stream": True,
            "timeout": params.timeout_sec,
            "extra_headers": params.headers,
            **(params.extra_args or {})
        }

    @staticmethod
    def _should_resolve_tool_calls(
            params: LlmRequestParams,
            message: AssistantMessage,
            ) -> tuple[list[ToolFn | ToolDef],
                       list[ChatCompletionAssistantToolCall]] | None:
        message.tool_calls
        condition = params.execute_tools and\
                    params.tools is not None and\
                    message.tool_calls is not None
        if condition:
            assert params.tools is not None
            assert message.tool_calls is not None
            return params.tools, message.tool_calls
        return None
    
    @staticmethod
    def _execute_tool_calls(
        tools: list[ToolFn | ToolDef],
        tool_calls: list[ChatCompletionAssistantToolCall]
        ) -> list[ToolMessage]:
        def find_target_tool(tool_fns: list[ToolFn], tool_name: str) -> ToolFn | None:
            target_tools = [tool for tool in tool_fns if tool.__name__ == tool_name]
            if len(target_tools) > 0: return target_tools[0]
            return None

        tool_fns = list(filter(lambda tool: callable(tool), tools))
        result = []
        for tool_call in tool_calls:
            id = tool_call.get("id")
            function = tool_call.get("function")
            function_name = function.get("name")
            function_arguments = function.get("arguments")
            if id is None or\
               function is None or\
               function_name is None or\
               function_arguments is None: continue

            if (target_tool := find_target_tool(tool_fns, function_name)) is None: continue
            tool_call_args = cast(dict, json.loads(function_arguments))
            tool_call_ret = target_tool(**tool_call_args)
            result.append(ToolMessage(tool_call_id=id, content=json.dumps(tool_call_ret)))
        return result

    def list_models(self) -> list[str]:
        return get_valid_models(
            custom_llm_provider=self.provider.value,
            check_provider_endpoint=True,
            api_base=self.base_url,
            api_key=self.api_key)

    def generate_text_sync(self, params: LlmRequestParams):
        response = completion(**self._parse_params_nonstream(params))
        response = cast(LiteLlmModelResponse, response)
        choices = cast(list[LiteLlmModelResponseChoices], response.choices)
        message = choices[0].message
        assistant_message = AssistantMessage.from_litellm_message(message)
        result: GenerateTextResponse = [assistant_message]
        if (tools_and_tool_calls := self._should_resolve_tool_calls(params, assistant_message)):
            tools, tool_calls = tools_and_tool_calls
            result += self._execute_tool_calls(tools, tool_calls)
        return result

    async def generate_text(self, params: LlmRequestParams) -> GenerateTextResponse:
        response = await acompletion(**self._parse_params_nonstream(params))
        response = cast(LiteLlmModelResponse, response)
        choices = cast(list[LiteLlmModelResponseChoices], response.choices)
        message = choices[0].message
        assistant_message = AssistantMessage.from_litellm_message(message)
        result: GenerateTextResponse = [assistant_message]
        if (tools_and_tool_calls := self._should_resolve_tool_calls(params, assistant_message)):
            tools, tool_calls = tools_and_tool_calls
            result += self._execute_tool_calls(tools, tool_calls)
        return result

    def stream_text_sync(self, params: LlmRequestParams) -> StreamTextResponseSync:
        def stream(response: CustomStreamWrapper) -> Generator[AssistantMessageChunk]:
            nonlocal message_collector
            for chunk in response:
                chunk = cast(LiteLlmModelResponseStream, chunk)
                yield AssistantMessageChunk.from_litellm_chunk(chunk)
                message_collector.collect(chunk)

            message = message_collector.get_message()
            full_message_queue.put(message)
            if (tools_and_tool_calls := self._should_resolve_tool_calls(params, message)):
                tools, tool_calls = tools_and_tool_calls
                tool_messages = self._execute_tool_calls(tools, tool_calls)
                for tool_message in tool_messages:
                    full_message_queue.put(tool_message)
            full_message_queue.put(None)

        response = completion(**self._parse_params_stream(params))
        message_collector = AssistantMessageCollector()
        returned_stream = stream(cast(CustomStreamWrapper, response))
        full_message_queue = queue.Queue[AssistantMessage | ToolMessage | None]()
        return returned_stream, full_message_queue

    async def stream_text(self, params: LlmRequestParams) -> StreamTextResponseAsync:
        async def stream(response: CustomStreamWrapper) -> AsyncGenerator[AssistantMessageChunk]:
            nonlocal message_collector
            async for chunk in response:
                chunk = cast(LiteLlmModelResponseStream, chunk)
                yield AssistantMessageChunk.from_litellm_chunk(chunk)
                message_collector.collect(chunk)

            message = message_collector.get_message()
            await full_message_queue.put(message)
            if (tools_and_tool_calls := self._should_resolve_tool_calls(params, message)):
                tools, tool_calls = tools_and_tool_calls
                tool_messages = self._execute_tool_calls(tools, tool_calls)
                for tool_message in tool_messages:
                    await full_message_queue.put(tool_message)
            await full_message_queue.put(None)

        response = await acompletion(**self._parse_params_stream(params))
        message_collector = AssistantMessageCollector()
        returned_stream = stream(cast(CustomStreamWrapper, response))
        full_message_queue = asyncio.Queue[AssistantMessage | ToolMessage | None]()
        return returned_stream, full_message_queue

__all__ = [
    "LLM",
    "LlmRequestParams",
    "ToolFn",
    "ToolDef",

    "UserMessage",
    "SystemMessage",
    "AssistantMessage",
    "ToolMessage",
    "AssistantMessageChunk",

    "GenerateTextResponse",
    "StreamTextResponseSync",
    "StreamTextResponseAsync"
]
