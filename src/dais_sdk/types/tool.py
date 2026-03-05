import dataclasses
from collections.abc import Callable, Mapping
from types import MethodType
from typing import Any, Awaitable, Literal, TypedDict
from ..logger import logger


class _ToolFunctionParameterSchema(TypedDict):
    type: Literal["object"]
    properties: dict[str, Any]
    required: list[str]

class ToolSchema(TypedDict):
    name: str
    description: str
    parameters: _ToolFunctionParameterSchema


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
type RawToolDef = ToolSchema

@dataclasses.dataclass
class ToolDef:
    name: str
    description: str
    execute: ToolFn
    parameters: _ToolFunctionParameterSchema | None = None
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)
    defaults: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def executes(self, fn: ToolFn) -> bool:
        """
        Check if this tool's execute function is the same as the given function.

        Handles the case where execute is a bound method by comparing the underlying
        function, so both ``instance.method`` and ``Class.method`` are considered equal.

        Examples:
            ```python
            def some_tool():
                pass

            tool = ToolDef(...) # ToolDef with some_tool as execute
            tool.executes(some_tool) # True
            ```
            ```python
            class BrowserToolset(PythonToolset):
                @python_tool
                def screenshot(self):
                    ...

            tool = BrowserToolset().get_tools()[0] # ToolDef with BrowserToolset.screenshot as execute
            tool.executes(BrowserToolset.screenshot) # True
            tool.executes(BrowserToolset().screenshot) # True
            ```
        """
        def normalize(f: ToolFn) -> ToolFn:
            if isinstance(f, MethodType):
                return f.__func__
            while hasattr(f, "__wrapped__"):
                f = getattr(f, "__wrapped__")
            return f
        return normalize(self.execute) is normalize(fn)

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

__all__ = [
    "ToolFn",
    "ToolDef",
    "RawToolDef",
    "ToolLike",
    "ToolSchema",
]
