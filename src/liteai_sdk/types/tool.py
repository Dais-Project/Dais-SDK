import dataclasses
from collections.abc import Callable
from typing import Any, Awaitable

ToolFn = Callable[..., Any] | Callable[..., Awaitable[Any]]

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
RawToolDef = dict[str, Any]

@dataclasses.dataclass
class ToolDef:
    name: str
    description: str
    execute: ToolFn

ToolLike = ToolDef | RawToolDef | ToolFn
