import asyncio
import json
from functools import singledispatch
from typing import Any, Awaitable, Callable, cast
from types import FunctionType, CoroutineType
from . import ToolFn, ToolDef

def _parse_arguments(arguments: str) -> dict:
    args = json.loads(arguments)
    return cast(dict, args)

async def _coroutine_wrapper(awaitable: Awaitable[Any]) -> CoroutineType:
    return await awaitable

@singledispatch
def execute_tool_sync(tool, arguments: str) -> Any: pass

@execute_tool_sync.register(FunctionType)
def _(tool: Callable, arguments: str) -> Any:
    if asyncio.iscoroutinefunction(tool):
        return asyncio.run(
            _coroutine_wrapper(
                tool(**_parse_arguments(arguments))))
    return tool(**_parse_arguments(arguments))

@execute_tool_sync.register(ToolDef)
def _(tool: ToolDef, arguments: str):
    if asyncio.iscoroutinefunction(tool.execute):
        return asyncio.run(
            _coroutine_wrapper(
                tool.execute(**_parse_arguments(arguments))))
    return tool.execute(**_parse_arguments(arguments))

@singledispatch
async def execute_tool(tool, arguments: str) -> Any: pass

@execute_tool.register(FunctionType)
async def _(toolfn: Callable, arguments: str) -> Any:
    if asyncio.iscoroutinefunction(toolfn):
        return await toolfn(**_parse_arguments(arguments))
    return toolfn(**_parse_arguments(arguments))

@execute_tool.register(ToolDef)
async def _(tooldef: ToolDef, arguments: str):
    if asyncio.iscoroutinefunction(tooldef.execute):
        return await tooldef.execute(**_parse_arguments(arguments))
    return tooldef.execute(**_parse_arguments(arguments))
