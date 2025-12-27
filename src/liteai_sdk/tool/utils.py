from . import ToolFn, ToolDef, ToolLike

def find_tool_by_name(tools: list[ToolLike], name: str) -> ToolLike | None:
    for tool in tools:
        if callable(tool) and tool.__name__ == name:
            return tool
        elif isinstance(tool, ToolDef) and tool.name == name:
            return tool
        elif isinstance(tool, dict) and tool.get("name") == name:
            return tool
    return None
