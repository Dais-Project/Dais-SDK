from abc import ABC, abstractmethod
from ...types.tool import ToolDef

class Toolset(ABC):
    def format_tool_name(self, tool_name: str) -> str:
        return f"{self.name}__{tool_name}"

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def get_tools(self, namespaced_tool_name: bool = True) -> list[ToolDef]: ...
