import asyncio
import json
from functools import singledispatch
from typing import Any, Awaitable, Callable, cast
from types import FunctionType, CoroutineType
from . import ToolDef

async def _coroutine_wrapper(awaitable: Awaitable[Any]) -> CoroutineType:
    return await awaitable

def _arguments_normalizer(arguments: str | dict) -> dict:
    if isinstance(arguments, str):
        parsed = json.loads(arguments)
        return cast(dict, parsed)
    elif isinstance(arguments, dict):
        return arguments
    else:
        raise ValueError(f"Invalid arguments type: {type(arguments)}")

@singledispatch
def execute_tool_sync(tool, arguments: str | dict) -> Any: pass

@execute_tool_sync.register(FunctionType)
def _(toolfn: Callable, arguments: str | dict) -> Any:
    arguments = _arguments_normalizer(arguments)
    if asyncio.iscoroutinefunction(toolfn):
        return asyncio.run(
            _coroutine_wrapper(
                toolfn(**arguments)))
    return toolfn(**arguments)

@execute_tool_sync.register(ToolDef)
def _(tooldef: ToolDef, arguments: str | dict) -> Any:
    arguments = _arguments_normalizer(arguments)
    if asyncio.iscoroutinefunction(tooldef.execute):
        return asyncio.run(
            _coroutine_wrapper(
                tooldef.execute(**arguments)))
    return tooldef.execute(**arguments)

@singledispatch
async def execute_tool(tool, arguments: str | dict) -> Any: pass

@execute_tool.register(FunctionType)
async def _(toolfn: Callable, arguments: str | dict) -> Any:
    arguments = _arguments_normalizer(arguments)
    if asyncio.iscoroutinefunction(toolfn):
        return await toolfn(**arguments)
    return toolfn(**arguments)

@execute_tool.register(ToolDef)
async def _(tooldef: ToolDef, arguments: str | dict) -> Any:
    arguments = _arguments_normalizer(arguments)
    if asyncio.iscoroutinefunction(tooldef.execute):
        return await tooldef.execute(**arguments)
    return tooldef.execute(**arguments)
