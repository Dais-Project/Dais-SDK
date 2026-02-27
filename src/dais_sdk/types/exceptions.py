import json
from typing import TYPE_CHECKING
from types import SimpleNamespace
from litellm.exceptions import (
    AuthenticationError,
    NotFoundError,
    BadRequestError,
    UnprocessableEntityError,
    UnsupportedParamsError,
    Timeout,
    PermissionDeniedError,
    RateLimitError,
    ContextWindowExceededError,
    RejectedRequestError,
    ContentPolicyViolationError,
    InternalServerError,
    ServiceUnavailableError,
    BadGatewayError,
    APIError,
    APIConnectionError,
    APIResponseValidationError,
    OpenAIError,
    JSONSchemaValidationError,
)

if TYPE_CHECKING:
    from .tool import ToolLike

LiteLlmExceptions = SimpleNamespace(
    AuthenticationError=AuthenticationError,
    NotFoundError=NotFoundError,
    BadRequestError=BadRequestError,
    UnprocessableEntityError=UnprocessableEntityError,
    UnsupportedParamsError=UnsupportedParamsError,
    Timeout=Timeout,
    PermissionDeniedError=PermissionDeniedError,
    RateLimitError=RateLimitError,
    ContextWindowExceededError=ContextWindowExceededError,
    RejectedRequestError=RejectedRequestError,
    ContentPolicyViolationError=ContentPolicyViolationError,
    InternalServerError=InternalServerError,
    ServiceUnavailableError=ServiceUnavailableError,
    BadGatewayError=BadGatewayError,
    APIError=APIError,
    APIConnectionError=APIConnectionError,
    APIResponseValidationError=APIResponseValidationError,
    OpenAIError=OpenAIError,
    JSONSchemaValidationError=JSONSchemaValidationError,
)


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
    "LiteLlmExceptions",

    "LlmToolException",
    "ToolDoesNotExistError",
    "ToolArgumentDecodeError",
    "ToolExecutionError",
]
