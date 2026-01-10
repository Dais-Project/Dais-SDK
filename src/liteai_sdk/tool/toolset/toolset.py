from typing import Protocol
from ...types.tool import ToolLike

class Toolset(Protocol):
    def get_tools(self) -> list[ToolLike]: ...
