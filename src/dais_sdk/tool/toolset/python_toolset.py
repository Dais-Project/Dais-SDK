import inspect
from typing import Any, Callable, override
from .toolset import Toolset
from ...types.tool import ToolDef

TOOL_FLAG = "__is_tool__"

def python_tool[F: Callable[..., Any]](func: F) -> F:
    setattr(func, TOOL_FLAG, True)
    return func

class PythonToolset(Toolset):
    @property
    @override
    def name(self) -> str:
        """
        Since the usage of PythonToolset is to inherit and define methods as tools,
        the name of the toolset is the name of the subclass.
        """
        return self.__class__.__name__

    @override
    def get_tools(self, namespaced_tool_name: bool = True) -> list[ToolDef]:
        result = []
        for _, method in inspect.getmembers(self, predicate=inspect.ismethod):
            if not getattr(method, TOOL_FLAG, False): continue
            tool_def = ToolDef.from_tool_fn(method)
            tool_def.name = (self.format_tool_name(tool_def.name)
                             if namespaced_tool_name
                             else tool_def.name)
            result.append(tool_def)
        return result
