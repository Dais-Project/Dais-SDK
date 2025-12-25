import dataclasses
from litellm import ChatCompletionAssistantToolCall
from litellm.types.utils import ChatCompletionDeltaToolCall,\
                                ModelResponseStream as LiteLlmModelResponseStream
from .types.message import AssistantMessage

@dataclasses.dataclass
class ToolCallTemp:
    id: str | None = None
    name: str = ""
    arguments: str = ""

class ToolCallCollector:
    def __init__(self):
        self.tool_call_map: dict[int, ToolCallTemp] = {}

    def collect(self, tool_call_chunk: ChatCompletionDeltaToolCall):
        if tool_call_chunk.index not in self.tool_call_map:
            self.tool_call_map[tool_call_chunk.index] = ToolCallTemp()

        temp_tool_call = self.tool_call_map[tool_call_chunk.index]
        if tool_call_chunk.get("id"):
            temp_tool_call.id = tool_call_chunk.id
        if tool_call_chunk.function.get("name"):
            assert tool_call_chunk.function.name is not None
            temp_tool_call.name += tool_call_chunk.function.name
        if tool_call_chunk.function.get("arguments"):
            assert tool_call_chunk.function.arguments is not None
            temp_tool_call.arguments += tool_call_chunk.function.arguments

    def get_tool_calls(self) -> list[ChatCompletionAssistantToolCall]:
        return [{
            "id": tool_call.id,
            "function": {
                "name": tool_call.name,
                "arguments": tool_call.arguments,
            },
            "type": "function"
        } for tool_call in self.tool_call_map.values()]

class AssistantMessageCollector:
    def __init__(self):
        self.message_buf = AssistantMessage(content=None)
        self.tool_call_collector = ToolCallCollector()

    def collect(self, chunk: LiteLlmModelResponseStream):
        delta = chunk.choices[0].delta
        if delta.get("content"):
            assert delta.content is not None
            if self.message_buf.content is None:
                self.message_buf.content = ""
            self.message_buf.content += delta.content

        if delta.get("reasoning_content"):
            assert delta.reasoning_content is not None
            if self.message_buf.reasoning_content is None:
                self.message_buf.reasoning_content = ""
            self.message_buf.reasoning_content += delta.reasoning_content

        if delta.get("tool_calls"):
            assert delta.tool_calls is not None
            for tool_call_chunk in delta.tool_calls:
                self.tool_call_collector.collect(tool_call_chunk)

        if delta.get("images"):
            assert delta.images is not None
            if self.message_buf.images is None:
                self.message_buf.images = []
            self.message_buf.images = delta.images
        
        if delta.get("audio"):
            assert delta.audio is not None
            self.message_buf.audio = delta.audio

    def get_message(self) -> AssistantMessage:
        self.message_buf.tool_calls = self.tool_call_collector.get_tool_calls()
        return self.message_buf

    def clear(self):
        """
        This class will be created for each stream, so we don't need to clear it.
        """
