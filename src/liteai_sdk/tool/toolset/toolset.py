from typing import Protocol
from ...types.tool import ToolDef

class Toolset(Protocol):
    def get_tools(self) -> list[ToolDef]: ...
