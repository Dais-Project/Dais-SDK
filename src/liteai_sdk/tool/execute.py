import asyncio
import json
from functools import singledispatch
from typing import Any, Awaitable, Callable, cast
from types import FunctionType, MethodType, CoroutineType
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

def _result_normalizer(result: Any) -> str:
    if isinstance(result, str):
        return result
    return json.dumps(result, ensure_ascii=False)

@singledispatch
def execute_tool_sync(tool, arguments: str | dict) -> str:
    raise ValueError(f"Invalid tool type: {type(tool)}")

@execute_tool_sync.register(FunctionType)
@execute_tool_sync.register(MethodType)
def _(toolfn: Callable, arguments: str | dict) -> str:
    arguments = _arguments_normalizer(arguments)
    result = asyncio.run(_coroutine_wrapper(toolfn(**arguments)))\
             if asyncio.iscoroutinefunction(toolfn)\
             else toolfn(**arguments)
    return _result_normalizer(result)

@execute_tool_sync.register(ToolDef)
def _(tooldef: ToolDef, arguments: str | dict) -> str:
    arguments = _arguments_normalizer(arguments)
    result = asyncio.run(_coroutine_wrapper(tooldef.execute(**arguments)))\
             if asyncio.iscoroutinefunction(tooldef.execute)\
             else tooldef.execute(**arguments)
    return _result_normalizer(result)

@singledispatch
async def execute_tool(tool, arguments: str | dict) -> str:
    raise ValueError(f"Invalid tool type: {type(tool)}")

@execute_tool.register(FunctionType)
@execute_tool.register(MethodType)
async def _(toolfn: Callable, arguments: str | dict) -> str:
    arguments = _arguments_normalizer(arguments)
    result = await toolfn(**arguments)\
             if asyncio.iscoroutinefunction(toolfn)\
             else toolfn(**arguments)
    return _result_normalizer(result)

@execute_tool.register(ToolDef)
async def _(tooldef: ToolDef, arguments: str | dict) -> str:
    arguments = _arguments_normalizer(arguments)
    result = await tooldef.execute(**arguments)\
             if asyncio.iscoroutinefunction(tooldef.execute)\
             else tooldef.execute(**arguments)
    return _result_normalizer(result)
