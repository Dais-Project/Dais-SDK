import inspect
from typing import Any, Callable, TypeVar
from ..types.tool import ToolFn

F = TypeVar("F", bound=Callable[..., Any])
TOOL_FLAG = "__is_tool__"

def tool(func: F) -> F:
    setattr(func, TOOL_FLAG, True)
    return func

class Toolset:
    def get_tool_methods(self) -> list[ToolFn]:
        return [
            method
            for _, method in inspect.getmembers(self, predicate=inspect.ismethod)
            if getattr(method, TOOL_FLAG, False)
        ]
