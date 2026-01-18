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
    @property
    def name(self) -> str:
        """
        Since the usage of PythonToolset is to inherit and define methods as tools,
        the name of the toolset is the name of the subclass.
        """
        return self.__class__.__name__

    def get_tools(self) -> list[ToolDef]:
        result = []
        for _, method in inspect.getmembers(self, predicate=inspect.ismethod):
            if not getattr(method, TOOL_FLAG, False): continue
            tool_def = ToolDef.from_tool_fn(method)
            tool_def.name = self.format_tool_name(tool_def.name)
            result.append(tool_def)
        return result
