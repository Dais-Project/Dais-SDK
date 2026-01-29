from dataclasses import dataclass, field
from typing import Any, Literal
from .tool import ToolLike
from .message import ChatMessage
from ..tool.toolset import Toolset
from ..tool.utils import find_tool_by_name

@dataclass
class LlmRequestParams:
    model: str
    messages: list[ChatMessage]
    tools: list[ToolLike] | None = None
    toolsets: list[Toolset] | None = None
    tool_choice: Literal["auto", "required", "none"] = "auto"
    execute_tools: bool = False

    timeout_sec: float | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    headers: dict[str, str] | None = None

    extra_args: dict[str, Any] | None = None

    _extract_tools_cache: list[ToolLike] | None = field(default=None, init=False, repr=False)

    def extract_tools(self) -> list[ToolLike] | None:
        if self._extract_tools_cache is not None:
            return self._extract_tools_cache

        if self.tools is None and self.toolsets is None:
            return None
        tools = []
        if self.tools:
            tools = self.tools
        if self.toolsets:
            for toolset in self.toolsets:
                tools.extend(toolset.get_tools())

        self._extract_tools_cache = tools
        return tools

    def find_tool(self, tool_name: str) -> ToolLike | None:
        has_tool_def = ((self.tools is not None and len(self.tools) > 0) or
                        (self.toolsets is not None and len(self.toolsets) > 0))
        if not has_tool_def: return None

        if (tools := self.extract_tools()) is None:
            return None
        return find_tool_by_name(tools, tool_name)

    class ToolDoesNotExistError(Exception):
        def __init__(self, tool_name: str):
            self.tool_name = tool_name
            self.message = f"Tool \"{tool_name}\" does not exist in the request params."
            super().__init__(self.message)
