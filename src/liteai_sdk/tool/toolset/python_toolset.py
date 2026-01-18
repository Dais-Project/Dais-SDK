import inspect
from typing import Any, Callable, TypeVar
from .toolset import Toolset
from ...types.tool import ToolDef

F = TypeVar("F", bound=Callable[..., Any])
TOOL_FLAG = "__is_tool__"

def python_tool(func: F) -> F:
    setattr(func, TOOL_FLAG, True)
    return func

class PythonToolset(Toolset):
    def get_toolset_name(self) -> str:
        return self.__class__.__name__

    def get_tools(self) -> list[ToolDef]:
        toolset_name = self.get_toolset_name()
        result = []
        for _, method in inspect.getmembers(self, predicate=inspect.ismethod):
            if not getattr(method, TOOL_FLAG, False): continue
            tool_def = ToolDef.from_tool_fn(method)
            tool_def.name = f"{toolset_name}__{tool_def.name}"
            result.append(tool_def)
        return result
