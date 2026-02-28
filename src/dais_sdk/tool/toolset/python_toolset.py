import inspect
from typing import Any, Callable, override, overload
from pydantic import validate_call, ConfigDict
from .toolset import Toolset
from ...types.tool import ToolDef

TOOL_FLAG = "__is_tool__"
TOOL_DEFAULTS = "__tool_defaults__"

def is_tool(func: Callable) -> bool:
    return getattr(func, TOOL_FLAG, False)

def get_tool_defaults(func: Callable) -> dict[str, Any]:
    return getattr(func, TOOL_DEFAULTS, {})

@overload
def python_tool[F: Callable[..., Any]](func: F) -> F: ...

@overload
def python_tool[F: Callable[..., Any]](func: None = None,
                                       *,
                                       validate: bool = False,
                                       validate_config: ConfigDict | None = None,
                                       defaults: dict[str, Any] | None = None,
                                       ) -> Callable[[F], F]: ...

def python_tool[F: Callable[..., Any]](func: F | None = None,
                                       *,
                                       validate: bool = False,
                                       validate_config: ConfigDict | None = None,
                                       defaults: dict[str, Any] | None = None,
                                       ) -> F | Callable[[F], F]:
    """
    Mark a callable as a tool and optionally enable runtime argument validation.

    This decorator supports both forms:
    - @python_tool
    - @python_tool(validate=True, validate_config=...)

    When validation is enabled, the target callable is wrapped by `pydantic.validate_call`,
    so arguments are validated (and may be coerced) according to type annotations before
    execution. The internal tool flag is attached to the final callable so it can be
    recognized by ``PythonToolset.get_tools``.

    Args:
        func: The target callable when used as ``@python_tool``. ``None`` when used as
            a configurable decorator factory.
        validate: Whether to enable Pydantic runtime argument validation.
            Defaults to ``False``.
        validate_config: Optional Pydantic ``ConfigDict`` passed to
            ``validate_call(config=...)``.
        defaults: Optional static configs for tool.

    Returns:
        The decorated callable (when ``func`` is provided), or a decorator that accepts
        a callable and returns the decorated callable.

    Examples:
        @python_tool
        def add(x: int, y: int) -> int:
            return x + y

        @python_tool(validate=True)
        def add_checked(x: int, y: int) -> int:
            return x + y

        @python_tool(defaults={"auto_approve": True})
        def read_file(path: str) -> str
            ...
    """

    def decorator(f: F) -> F:
        if validate:
            f = validate_call(config=validate_config)(f)
        setattr(f, TOOL_FLAG, True)
        setattr(f, TOOL_DEFAULTS, defaults or {})
        return f

    if func is not None:
        return decorator(func)
    return decorator

# --- --- --- --- --- ---

class PythonToolset(Toolset):
    @property
    @override
    def name(self) -> str:
        """
        Since the usage of PythonToolset is to inherit and define methods as tools,
        the name of the toolset is the name of the subclass.
        """
        return self.__class__.__name__

    @override
    def get_tools(self, namespaced_tool_name: bool = True) -> list[ToolDef]:
        result = []
        for _, method in inspect.getmembers(self, predicate=inspect.ismethod):
            if not is_tool(method): continue
            tool_def = ToolDef.from_tool_fn(method)
            tool_def.name = (self.format_tool_name(tool_def.name)
                             if namespaced_tool_name
                             else tool_def.name)
            tool_def.defaults = get_tool_defaults(method)
            result.append(tool_def)
        return result
