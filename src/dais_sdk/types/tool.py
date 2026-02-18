import dataclasses
from collections.abc import Callable
from typing import Any, Awaitable, Literal, TypedDict
from ..logger import logger

class _ToolFunctionParameterSchema(TypedDict):
    type: Literal["object"]
    properties: dict[str, Any]
    required: list[str]
class _ToolFunctionSchema(TypedDict):
    name: str
    description: str
    parameters: _ToolFunctionParameterSchema
class ToolSchema(TypedDict):
    type: Literal["function"]
    function: _ToolFunctionSchema

type ToolFn = Callable[..., Any] | Callable[..., Awaitable[Any]]

"""
RawToolDef example:
{
    "name": "get_current_weather",
    "description": "Get the current weather in a given location",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The city and state, e.g. San Francisco, CA",
            },
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        },
        "required": ["location"],
    }
}
"""
type RawToolDef = _ToolFunctionSchema

@dataclasses.dataclass
class ToolDef:
    name: str
    description: str
    execute: ToolFn
    parameters: _ToolFunctionParameterSchema | None = None
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)

    @staticmethod
    def from_tool_fn(tool_fn: ToolFn) -> "ToolDef":
        if tool_fn.__doc__ is None:
            logger.warning(f"Tool function {tool_fn.__name__} has no docstring, "
                            "which is recommended to be used as the tool description")
        return ToolDef(
            name=tool_fn.__name__,
            description=tool_fn.__doc__ or "",
            execute=tool_fn,
        )

type ToolLike = ToolDef | RawToolDef | ToolFn
