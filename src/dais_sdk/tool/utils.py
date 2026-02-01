from typing import assert_never
from ..types.tool import ToolDef, ToolLike

def get_tool_name(tool: ToolLike) -> str:
    if callable(tool):
        return tool.__name__
    elif isinstance(tool, ToolDef):
        return tool.name
    elif isinstance(tool, dict):
        return tool.get("name", "")
    else:
        assert_never(tool)

def find_tool_by_name(tools: list[ToolLike], name: str) -> ToolLike | None:
    for tool in tools:
        if callable(tool) and tool.__name__ == name:
            return tool
        elif isinstance(tool, ToolDef) and tool.name == name:
            return tool
        elif isinstance(tool, dict) and tool.get("name") == name:
            return tool
    return None
