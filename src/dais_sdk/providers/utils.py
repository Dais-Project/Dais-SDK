import dataclasses
import json
from ..types.event import TextChunkEvent, ToolCallChunkEvent, UsageChunkEvent
from ..types.message import AssistantMessage


class ToolCallCollector:
    @dataclasses.dataclass
    class ToolCallTemp:
        id: str = ""
        name: str = ""
        arguments: str = ""

    def __init__(self):
        self.tool_call_map: dict[int, ToolCallCollector.ToolCallTemp] = {}

    def collect(self, tool_call_chunk: ToolCallChunkEvent):
        if tool_call_chunk.index not in self.tool_call_map:
            self.tool_call_map[tool_call_chunk.index] = ToolCallCollector.ToolCallTemp()

        temp_tool_call = self.tool_call_map[tool_call_chunk.index]
        if tool_call_chunk.id:
            temp_tool_call.id = tool_call_chunk.id
        if tool_call_chunk.name:
            temp_tool_call.name += tool_call_chunk.name
        if tool_call_chunk.arguments:
            temp_tool_call.arguments += tool_call_chunk.arguments

    def get_tool_calls(self) -> list[AssistantMessage.ToolCall]:
        return [AssistantMessage.ToolCall(
            id=tool_call.id,
            name=tool_call.name,
            arguments=json.loads(tool_call.arguments),
        ) for tool_call in self.tool_call_map.values()]

class StreamMessageCollector:
    def __init__(self):
        self._message_buf = AssistantMessage(content=None)
        self._tool_call_collector = ToolCallCollector()

    def collect(self, chunk: TextChunkEvent | ToolCallChunkEvent | UsageChunkEvent):
        match chunk:
            case ToolCallChunkEvent():
                self._tool_call_collector.collect(chunk)
            case TextChunkEvent():
                if self._message_buf.content is None:
                    self._message_buf.content = ""
                self._message_buf.content += chunk.content
            case UsageChunkEvent():
                if self._message_buf.usage is None:
                    self._message_buf.usage = AssistantMessage.Usage.default()
                # since the usage chunk is the last chunk, we can directly assign the values here
                self._message_buf.usage.input_tokens = chunk.input_tokens
                self._message_buf.usage.output_tokens = chunk.output_tokens
                self._message_buf.usage.total_tokens = chunk.total_tokens

    def get_message(self) -> AssistantMessage:
        self._message_buf.tool_calls = self._tool_call_collector.get_tool_calls()
        # if self._message_buf.content is not None and self._message_buf.reasoning_content is None:
        #     self._message_buf.content, self._message_buf.reasoning_content =\
        #         AssistantMessage.extract_thinking_content(self._message_buf.content)
        result = self._message_buf

        # reset the message buffer
        self._message_buf = AssistantMessage(content=None)
        self._tool_call_collector = ToolCallCollector()
        return result
