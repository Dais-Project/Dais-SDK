import asyncio
import json
import inspect
from functools import singledispatch
from typing import Any, TypeAlias, assert_never, Callable, cast, TypeVar
from types import FunctionType, MethodType
from ..types.tool import ToolDef, ToolLike
from ..types.exceptions import LlmToolException
from ..logger import logger

E = TypeVar("E", bound=LlmToolException)
ExceptionHandler: TypeAlias = Callable[[E], str]

class ToolExceptionHandlerManager:
    def __init__(self):
        self._handlers: dict[type[LlmToolException], Callable[[Any], str]] = {}

    def register(self, exception_type: type[E]):
        def decorator(handler: ExceptionHandler) -> ExceptionHandler:
            self.set_handler(exception_type, handler)
            return handler
        return decorator

    def set_handler(self, exception_type: type[E], handler: ExceptionHandler):
        self._handlers[exception_type] = handler

    def get_handler(self, exception_type: type[E]) -> ExceptionHandler | None:
        return self._handlers.get(exception_type)

    def handle(self, e: LlmToolException) -> str:
        handler = self.get_handler(type(e))
        if handler is None:
            logger.warning(f"Unhandled tool exception: {type(e).__name__}", exc_info=e)
            return f"Unhandled tool exception | {type(e).__name__}: {e}"
        return handler(e)

# --- --- --- --- --- ---

def _arguments_normalizer(arguments: str | dict) -> dict:
    if isinstance(arguments, str):
        parsed = json.loads(arguments)
        return cast(dict, parsed)
    elif isinstance(arguments, dict):
        return arguments
    else:
        assert_never(arguments)

def _result_normalizer(result: Any) -> str:
    if isinstance(result, str):
        return result
    return json.dumps(result, ensure_ascii=False)

@singledispatch
def execute_tool_sync(tool: ToolLike, arguments: str | dict) -> str:
    """
    Raises:
        ValueError: If the tool type is not supported.
        JSONDecodeError: If the arguments is a string but not valid JSON.
    """
    raise ValueError(f"Invalid tool type: {type(tool)}")

@execute_tool_sync.register(FunctionType)
@execute_tool_sync.register(MethodType)
def _(toolfn: Callable, arguments: str | dict) -> str:
    arguments = _arguments_normalizer(arguments)
    result = (asyncio.run(toolfn(**arguments))
              if inspect.iscoroutinefunction(toolfn)
              else toolfn(**arguments))
    return _result_normalizer(result)

@execute_tool_sync.register(ToolDef)
def _(tooldef: ToolDef, arguments: str | dict) -> str:
    arguments = _arguments_normalizer(arguments)
    result = (asyncio.run(tooldef.execute(**arguments))
              if inspect.iscoroutinefunction(tooldef.execute)
              else tooldef.execute(**arguments))
    return _result_normalizer(result)

@singledispatch
async def execute_tool(tool: ToolLike, arguments: str | dict) -> str:
    """
    Raises:
        ValueError: If the tool type is not supported.
        JSONDecodeError: If the arguments is a string but not valid JSON.
    """
    raise ValueError(f"Invalid tool type: {type(tool)}")

@execute_tool.register(FunctionType)
@execute_tool.register(MethodType)
async def _(toolfn: Callable, arguments: str | dict) -> str:
    arguments = _arguments_normalizer(arguments)
    result = (await toolfn(**arguments)
             if inspect.iscoroutinefunction(toolfn)
             else toolfn(**arguments))
    return _result_normalizer(result)

@execute_tool.register(ToolDef)
async def _(tooldef: ToolDef, arguments: str | dict) -> str:
    arguments = _arguments_normalizer(arguments)
    result = (await tooldef.execute(**arguments)
             if inspect.iscoroutinefunction(tooldef.execute)
             else tooldef.execute(**arguments))
    return _result_normalizer(result)
