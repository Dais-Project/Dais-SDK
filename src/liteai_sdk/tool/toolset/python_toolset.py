import inspect
from typing import Any, Callable, TypeVar
from .toolset import Toolset
from ...types.tool import ToolLike

F = TypeVar("F", bound=Callable[..., Any])
TOOL_FLAG = "__is_tool__"

def python_tool(func: F) -> F:
    setattr(func, TOOL_FLAG, True)
    return func

class PythonToolset(Toolset):
    def get_tools(self) -> list[ToolLike]:
        return [
            method
            for _, method in inspect.getmembers(self, predicate=inspect.ismethod)
            if getattr(method, TOOL_FLAG, False)
        ]
