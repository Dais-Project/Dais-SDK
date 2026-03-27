import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .tool import ToolLike

class LlmToolException(Exception): ...

class ToolDoesNotExistError(LlmToolException):
    def __init__(self, tool_name: str):
        super().__init__(f"Tool does not exist: ", tool_name)
        self.tool_name = tool_name

class ToolArgumentDecodeError(LlmToolException):
    def __init__(self, tool_name: str, arguments: str, raw_error: json.JSONDecodeError):
        super().__init__(f"Tool argument decode error: ", raw_error)
        self.tool_name = tool_name
        self.arguments = arguments
        self.raw_error = raw_error

class ToolExecutionError(LlmToolException):
    def __init__(self, tool: ToolLike, arguments: str | dict, raw_error: Exception):
        super().__init__(f"Tool execution error: ", raw_error)
        self.tool = tool
        self.arguments = arguments
        self.raw_error = raw_error

class LlmProviderException(Exception): ...

class AttachmentTypeNotSupportedError(LlmProviderException):
    def __init__(self, attachment_type: str):
        super().__init__(f"Attachment type not supported: ", attachment_type)
        self.attachment_type = attachment_type

class SkillException(Exception): ...

class InvalidSkillArchiveError(SkillException):
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

__all__ = [
    "LlmToolException",
    "ToolDoesNotExistError",
    "ToolArgumentDecodeError",
    "ToolExecutionError",

    "LlmProviderException",
    "AttachmentTypeNotSupportedError",
    
    "SkillException",
    "InvalidSkillArchiveError",
]
