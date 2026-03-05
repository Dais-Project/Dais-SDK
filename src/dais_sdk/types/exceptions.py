import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .tool import ToolLike

class LlmToolException(Exception): pass

class ToolDoesNotExistError(LlmToolException):
    def __init__(self, tool_name: str):
        self.tool_name = tool_name

class ToolArgumentDecodeError(LlmToolException):
    def __init__(self, tool_name: str, arguments: str, raw_error: json.JSONDecodeError):
        self.tool_name = tool_name
        self.arguments = arguments
        self.raw_error = raw_error

class ToolExecutionError(LlmToolException):
    def __init__(self, tool: ToolLike, arguments: str | dict, raw_error: Exception):
        self.tool = tool
        self.arguments = arguments
        self.raw_error = raw_error

__all__ = [
    "LlmToolException",
    "ToolDoesNotExistError",
    "ToolArgumentDecodeError",
    "ToolExecutionError",
]
