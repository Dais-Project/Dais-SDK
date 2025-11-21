from typing import Any
from . import ToolFn, ToolDef, RawToolDef

def filter_executable_tools(tools: list[ToolFn | ToolDef | RawToolDef]) -> list[ToolFn | ToolDef]:
    """
    Since when we are going to execute the tools,
    we do not care the raw tool definitions, they are usually the built-in tools from the provider.
    """
    return [tool for tool in tools if callable(tool) or isinstance(tool, ToolDef)]

def find_tool_by_name(tools: list[ToolFn | ToolDef | RawToolDef], name: str) -> ToolFn | ToolDef | RawToolDef | None:
    for tool in tools:
        if callable(tool) and tool.__name__ == name:
            return tool
        elif isinstance(tool, ToolDef) and tool.name == name:
            return tool
        elif isinstance(tool, dict) and tool.get("function", {}).get("name") == name:
            return tool
    return None
