import dataclasses
import enum
from collections.abc import Mapping, Sequence
from datetime import date, datetime, time
from typing import Annotated, Any, Literal, Optional, Union, get_args, get_origin

import pytest
from pydantic import BaseModel as PydanticBaseModel

from liteai_sdk.tool import (
    ToolDef,
    _python_type_to_json_schema,
    generate_tool_definition_from_callable,
    generate_tool_definition_from_raw_tool_def,
    generate_tool_definition_from_tool_def,
    prepare_tools,
)


# ============================================================================
# Test Suite 1: _python_type_to_json_schema
# ============================================================================


class TestPythonTypeToJsonSchema:
    # ------------------------------------------------------------------------
    # 1.1 primitive types
    # ------------------------------------------------------------------------

    def test_primitive_types(self):
        assert _python_type_to_json_schema(str) == {"type": "string"}
        assert _python_type_to_json_schema(int) == {"type": "integer"}
        assert _python_type_to_json_schema(float) == {"type": "number"}
        assert _python_type_to_json_schema(bool) == {"type": "boolean"}
        assert _python_type_to_json_schema(Any) == {"type": "string"}

    # ------------------------------------------------------------------------
    # 1.2 bytes and datetime types
    # ------------------------------------------------------------------------

    def test_bytes_type(self):
        assert _python_type_to_json_schema(bytes) == {"type": "string", "contentEncoding": "base64"}

    def test_datetime_types(self):
        assert _python_type_to_json_schema(datetime) == {"type": "string", "format": "date-time"}
        assert _python_type_to_json_schema(date) == {"type": "string", "format": "date"}
        assert _python_type_to_json_schema(time) == {"type": "string", "format": "time"}

    # ------------------------------------------------------------------------
    # 1.3 container types
    # ------------------------------------------------------------------------

    def test_list_without_type_args(self):
        assert _python_type_to_json_schema(list) == {"type": "array", "items": {"type": "string"}}

    def test_list_with_type(self):
        assert _python_type_to_json_schema(list[int]) == {"type": "array", "items": {"type": "integer"}}
        assert _python_type_to_json_schema(list[str]) == {"type": "array", "items": {"type": "string"}}

    def test_sequence_type(self):
        assert _python_type_to_json_schema(Sequence[str]) == {"type": "array", "items": {"type": "string"}}
        assert _python_type_to_json_schema(Sequence[int]) == {"type": "array", "items": {"type": "integer"}}

    def test_set_types(self):
        assert _python_type_to_json_schema(set[str]) == {
            "type": "array",
            "items": {"type": "string"},
            "uniqueItems": True,
        }
        assert _python_type_to_json_schema(frozenset[int]) == {
            "type": "array",
            "items": {"type": "integer"},
            "uniqueItems": True,
        }

    def test_tuple_variable_length(self):
        assert _python_type_to_json_schema(tuple[int, ...]) == {"type": "array", "items": {"type": "integer"}}

    def test_tuple_fixed_length(self):
        result = _python_type_to_json_schema(tuple[str, int, bool])
        assert result == {
            "type": "array",
            "prefixItems": [{"type": "string"}, {"type": "integer"}, {"type": "boolean"}],
            "minItems": 3,
            "maxItems": 3,
        }

    def test_dict_without_type_args(self):
        assert _python_type_to_json_schema(dict) == {"type": "object", "additionalProperties": {"type": "string"}}

    def test_dict_with_types(self):
        assert _python_type_to_json_schema(dict[str, int]) == {
            "type": "object",
            "additionalProperties": {"type": "integer"},
        }
        assert _python_type_to_json_schema(dict[str, str]) == {
            "type": "object",
            "additionalProperties": {"type": "string"},
        }

    def test_mapping_type(self):
        assert _python_type_to_json_schema(Mapping[str, float]) == {
            "type": "object",
            "additionalProperties": {"type": "number"},
        }

    # ------------------------------------------------------------------------
    # 1.6 Union types
    # ------------------------------------------------------------------------

    def test_union_two_types(self):
        result = _python_type_to_json_schema(Union[str, int])
        assert result == {"oneOf": [{"type": "string"}, {"type": "integer"}]}

    def test_union_pipe_syntax(self):
        result = _python_type_to_json_schema(str | int)
        assert result == {"oneOf": [{"type": "string"}, {"type": "integer"}]}

    def test_optional_type(self):
        assert _python_type_to_json_schema(Optional[str]) == {"type": "string"}
        assert _python_type_to_json_schema(Optional[int]) == {"type": "integer"}

    def test_union_with_none_only(self):
        result = _python_type_to_json_schema(type(None))
        assert result == {"type": "string"}

    def test_union_multiple_types(self):
        result = _python_type_to_json_schema(Union[str, int, float])
        assert result == {"oneOf": [{"type": "string"}, {"type": "integer"}, {"type": "number"}]}

    # ------------------------------------------------------------------------
    # 1.7 Literal types
    # ------------------------------------------------------------------------

    def test_literal_bool(self):
        assert _python_type_to_json_schema(Literal[True, False]) == {"type": "boolean", "enum": [True, False]}

    def test_literal_string(self):
        assert _python_type_to_json_schema(Literal["a", "b", "c"]) == {"type": "string", "enum": ["a", "b", "c"]}

    def test_literal_integer(self):
        assert _python_type_to_json_schema(Literal[1, 2, 3]) == {"type": "integer", "enum": [1, 2, 3]}

    def test_literal_mixed_number(self):
        assert _python_type_to_json_schema(Literal[1, 2.5, 3]) == {"type": "number", "enum": [1, 2.5, 3]}

    # ------------------------------------------------------------------------
    # 1.8 Enum types
    # ------------------------------------------------------------------------

    def test_string_enum(self):
        class Color(enum.Enum):
            RED = "red"
            GREEN = "green"
            BLUE = "blue"

        assert _python_type_to_json_schema(Color) == {"type": "string", "enum": ["red", "green", "blue"]}

    def test_integer_enum(self):
        class Status(enum.Enum):
            PENDING = 0
            ACTIVE = 1
            COMPLETED = 2

        assert _python_type_to_json_schema(Status) == {"type": "integer", "enum": [0, 1, 2]}

    def test_mixed_number_enum(self):
        class Priority(enum.Enum):
            LOW = 1
            MEDIUM = 2.5
            HIGH = 5

        assert _python_type_to_json_schema(Priority) == {"type": "number", "enum": [1, 2.5, 5]}

    def test_boolean_enum(self):
        class Toggle(enum.Enum):
            ON = True
            OFF = False

        assert _python_type_to_json_schema(Toggle) == {"type": "boolean", "enum": [True, False]}

    # ------------------------------------------------------------------------
    # 1.9 TypedDict tests
    # ------------------------------------------------------------------------

    def test_typeddict_all_required(self):
        from typing import TypedDict

        class Person(TypedDict):
            name: str
            age: int

        result = _python_type_to_json_schema(Person)
        assert result == {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
            "required": ["name", "age"],
        }

    def test_typeddict_partial_required(self):
        from typing import TypedDict

        class User(TypedDict, total=False):
            username: str
            email: str

        result = _python_type_to_json_schema(User)
        assert result["type"] == "object"
        assert "username" in result["properties"]
        assert "email" in result["properties"]
        assert "required" not in result or result.get("required") == []

    # ------------------------------------------------------------------------
    # 1.10 dataclass tests
    # ------------------------------------------------------------------------

    def test_dataclass_with_defaults(self):
        @dataclasses.dataclass
        class Config:
            host: str
            port: int = 8080
            debug: bool = False

        result = _python_type_to_json_schema(Config)
        assert result == {
            "type": "object",
            "properties": {
                "host": {"type": "string"},
                "port": {"type": "integer"},
                "debug": {"type": "boolean"},
            },
            "required": ["host"],
        }

    def test_dataclass_with_factory(self):
        @dataclasses.dataclass
        class Container:
            name: str
            items: list[str] = dataclasses.field(default_factory=list)

        result = _python_type_to_json_schema(Container)
        assert "name" in result["required"]
        assert "items" not in result.get("required", [])
        assert result["properties"]["items"] == {"type": "array", "items": {"type": "string"}}

    # ------------------------------------------------------------------------
    # 1.11 Pydantic model tests
    # ------------------------------------------------------------------------

    def test_pydantic_model(self):
        class User(PydanticBaseModel):
            username: str
            email: str
            age: int = 18

        result = _python_type_to_json_schema(User)
        assert result["type"] == "object"
        assert "username" in result["properties"]
        assert "email" in result["properties"]
        assert "age" in result["properties"]
        assert "username" in result["required"]
        assert "email" in result["required"]
        assert "age" not in result["required"]

    # ------------------------------------------------------------------------
    # 1.12 Annotated type tests
    # ------------------------------------------------------------------------

    def test_annotated_type(self):
        assert _python_type_to_json_schema(Annotated[str, "metadata"]) == {"type": "string"}
        assert _python_type_to_json_schema(Annotated[int, "positive"]) == {"type": "integer"}
        assert _python_type_to_json_schema(Annotated[list[str], "non-empty"]) == {
            "type": "array",
            "items": {"type": "string"},
        }

    # ------------------------------------------------------------------------
    # 1.13 Nested types
    # ------------------------------------------------------------------------

    def test_nested_list_dict(self):
        result = _python_type_to_json_schema(list[dict[str, int]])
        assert result == {"type": "array", "items": {"type": "object", "additionalProperties": {"type": "integer"}}}

    def test_complex_nested_structure(self):
        result = _python_type_to_json_schema(dict[str, list[tuple[int, str]]])
        assert result["type"] == "object"
        assert result["additionalProperties"]["type"] == "array"
        assert result["additionalProperties"]["items"]["type"] == "array"
        assert result["additionalProperties"]["items"]["prefixItems"] == [
            {"type": "integer"},
            {"type": "string"},
        ]

    def test_nested_optional(self):
        result = _python_type_to_json_schema(list[Optional[str]])
        assert result == {"type": "array", "items": {"type": "string"}}


# ============================================================================
# Test Suite 2: generate_tool_definition
# ============================================================================


class TestGenerateToolDefinition:
    # ------------------------------------------------------------------------
    # 2.1 basic functionality
    # ------------------------------------------------------------------------

    def test_simple_function(self):

        def greet(name: str) -> str:
            """Greet a person by name"""
            return f"Hello, {name}!"

        result = generate_tool_definition_from_callable(greet)
        assert result["type"] == "function"
        assert result["function"]["name"] == "greet"
        assert result["function"]["description"] == "Greet a person by name"
        assert result["function"]["parameters"]["type"] == "object"
        assert "name" in result["function"]["parameters"]["properties"]
        assert result["function"]["parameters"]["properties"]["name"]["type"] == "string"
        assert result["function"]["parameters"]["required"] == ["name"]

    def test_function_with_default_params(self):

        def connect(host: str, port: int = 8080, timeout: float = 30.0) -> bool:
            """Connect to a server"""
            return True

        result = generate_tool_definition_from_callable(connect)
        params = result["function"]["parameters"]
        assert params["required"] == ["host"]
        assert "port" in params["properties"]
        assert "timeout" in params["properties"]
        assert params["properties"]["port"]["type"] == "integer"
        assert params["properties"]["timeout"]["type"] == "number"

    # ------------------------------------------------------------------------
    # 2.2 missing docstring
    # ------------------------------------------------------------------------

    def test_function_without_docstring(self):

        def no_doc_function():
            pass

        with pytest.raises(ValueError) as exc_info:
            generate_tool_definition_from_callable(no_doc_function)
        assert "must have a docstring" in str(exc_info.value)

    # ------------------------------------------------------------------------
    # 2.3 complex type parameters
    # ------------------------------------------------------------------------

    def test_function_with_complex_types(self):

        def process_data(items: list[str], config: dict[str, Any], options: Optional[dict] = None) -> dict:
            """Process data with configuration"""
            return {}

        result = generate_tool_definition_from_callable(process_data)
        props = result["function"]["parameters"]["properties"]

        assert props["items"]["type"] == "array"
        assert props["items"]["items"]["type"] == "string"
        assert props["config"]["type"] == "object"
        assert "options" in props
        assert result["function"]["parameters"]["required"] == ["items", "config"]

    # ------------------------------------------------------------------------
    # 2.4 skip special parameters
    # ------------------------------------------------------------------------

    def test_function_with_var_args(self):

        def flexible_func(required: str, *args, **kwargs) -> None:
            """A flexible function"""
            pass

        result = generate_tool_definition_from_callable(flexible_func)
        params = result["function"]["parameters"]

        assert list(params["properties"].keys()) == ["required"]
        assert params["required"] == ["required"]

    # ------------------------------------------------------------------------
    # 2.5 no type hints
    # ------------------------------------------------------------------------

    def test_function_without_type_hints(self):

        def legacy_func(param1, param2="default"):
            """Legacy function without type hints"""
            return param1

        result = generate_tool_definition_from_callable(legacy_func)
        props = result["function"]["parameters"]["properties"]

        assert props["param1"]["type"] == "string"
        assert props["param2"]["type"] == "string"
        assert result["function"]["parameters"]["required"] == ["param1"]

    # ------------------------------------------------------------------------
    # 2.6 multiline docstring
    # ------------------------------------------------------------------------

    def test_function_with_multiline_docstring(self):

        def documented_func(x: int) -> int:
            """
            This is a longer docstring.
            It spans multiple lines.
            """
            return x * 2

        result = generate_tool_definition_from_callable(documented_func)
        assert result["function"]["description"] == "This is a longer docstring.\n            It spans multiple lines."

class TestGenerateToolDefinitionFromToolDef:
    # ------------------------------------------------------------------------
    # 2.7 basic ToolDef conversion
    # ------------------------------------------------------------------------

    def test_tool_def_with_simple_function(self):
        def greet(name: str) -> str:
            """Greet a person"""
            return f"Hello, {name}!"

        tool_def = ToolDef(
            name="greet_user",
            description="Greet a user by name",
            execute=greet,
        )

        result = generate_tool_definition_from_tool_def(tool_def)

        assert result["type"] == "function"
        assert result["function"]["name"] == "greet_user"
        assert result["function"]["description"] == "Greet a user by name"
        assert "name" in result["function"]["parameters"]["properties"]
        assert result["function"]["parameters"]["properties"]["name"]["type"] == "string"
        assert result["function"]["parameters"]["required"] == ["name"]

    # ------------------------------------------------------------------------
    # 2.8 ToolDef with default parameters
    # ------------------------------------------------------------------------

    def test_tool_def_with_default_params(self):
        def connect(host: str, port: int = 8080, timeout: float = 30.0) -> bool:
            """Connect to a server"""
            return True

        tool_def = ToolDef(
            name="connect_to_server",
            description="Connect to a remote server",
            execute=connect,
        )

        result = generate_tool_definition_from_tool_def(tool_def)
        params = result["function"]["parameters"]

        assert result["function"]["name"] == "connect_to_server"
        assert result["function"]["description"] == "Connect to a remote server"
        assert params["required"] == ["host"]
        assert "port" in params["properties"]
        assert "timeout" in params["properties"]
        assert params["properties"]["port"]["type"] == "integer"
        assert params["properties"]["timeout"]["type"] == "number"

    # ------------------------------------------------------------------------
    # 2.9 ToolDef with complex types
    # ------------------------------------------------------------------------

    def test_tool_def_with_complex_types(self):
        def process_data(items: list[str], config: dict[str, Any], options: Optional[dict] = None) -> dict:
            """Process data with configuration"""
            return {}

        tool_def = ToolDef(
            name="data_processor",
            description="Process data items with custom configuration",
            execute=process_data,
        )

        result = generate_tool_definition_from_tool_def(tool_def)
        props = result["function"]["parameters"]["properties"]

        assert result["function"]["name"] == "data_processor"
        assert props["items"]["type"] == "array"
        assert props["items"]["items"]["type"] == "string"
        assert props["config"]["type"] == "object"
        assert "options" in props
        assert result["function"]["parameters"]["required"] == ["items", "config"]

    # ------------------------------------------------------------------------
    # 2.10 ToolDef name and description override
    # ------------------------------------------------------------------------

    def test_tool_def_overrides_function_metadata(self):
        def original_func(x: int) -> int:
            """Original function description"""
            return x * 2

        tool_def = ToolDef(
            name="custom_name",
            description="Custom description",
            execute=original_func,
        )

        result = generate_tool_definition_from_tool_def(tool_def)

        # Should use ToolDef name and description, not function's
        assert result["function"]["name"] == "custom_name"
        assert result["function"]["description"] == "Custom description"
        # But parameters should come from function signature
        assert "x" in result["function"]["parameters"]["properties"]

class TestGenerateToolDefinitionFromRawToolDef:
    # ------------------------------------------------------------------------
    # 2.11 basic raw tool definition
    # ------------------------------------------------------------------------

    def test_raw_tool_def_simple(self):
        raw_tool_def = {
            "name": "get_weather",
            "description": "Get weather information",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"},
                },
                "required": ["location"],
            },
        }

        result = generate_tool_definition_from_raw_tool_def(raw_tool_def)

        assert result["type"] == "function"
        assert result["function"] == raw_tool_def
        assert result["function"]["name"] == "get_weather"
        assert result["function"]["description"] == "Get weather information"

    # ------------------------------------------------------------------------
    # 2.12 raw tool def with complex parameters
    # ------------------------------------------------------------------------

    def test_raw_tool_def_complex(self):
        raw_tool_def = {
            "name": "search_database",
            "description": "Search database with filters",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "filters": {
                        "type": "object",
                        "additionalProperties": {"type": "string"},
                        "description": "Filter conditions",
                    },
                    "limit": {"type": "integer", "description": "Result limit"},
                },
                "required": ["query"],
            },
        }

        result = generate_tool_definition_from_raw_tool_def(raw_tool_def)

        assert result["type"] == "function"
        assert result["function"]["name"] == "search_database"
        assert "query" in result["function"]["parameters"]["properties"]
        assert "filters" in result["function"]["parameters"]["properties"]
        assert "limit" in result["function"]["parameters"]["properties"]
        assert result["function"]["parameters"]["required"] == ["query"]

    # ------------------------------------------------------------------------
    # 2.13 raw tool def minimal structure
    # ------------------------------------------------------------------------

    def test_raw_tool_def_minimal(self):
        raw_tool_def = {
            "name": "simple_tool",
            "description": "A simple tool",
            "parameters": {"type": "object", "properties": {}},
        }

        result = generate_tool_definition_from_raw_tool_def(raw_tool_def)

        assert result["type"] == "function"
        assert result["function"]["name"] == "simple_tool"
        assert result["function"]["parameters"]["properties"] == {}

# ============================================================================
# Test Suite 3: prepare_tools
# ============================================================================

class TestPrepareTools:
    # ------------------------------------------------------------------------
    # 3.1 callable list
    # ------------------------------------------------------------------------

    def test_prepare_tools_with_callables(self):

        def tool1(x: int) -> int:
            """Tool 1"""
            return x * 2

        def tool2(y: str) -> str:
            """Tool 2"""
            return y.upper()

        result = prepare_tools([tool1, tool2])

        assert len(result) == 2
        assert result[0]["function"]["name"] == "tool1"
        assert result[1]["function"]["name"] == "tool2"

    # ------------------------------------------------------------------------
    # 3.2 raw definition list
    # ------------------------------------------------------------------------

    def test_prepare_tools_with_raw_definitions(self):
        raw_tool = {
            "name": "custom_tool",
            "description": "A custom tool",
            "parameters": {"type": "object", "properties": {}},
        }

        result = prepare_tools([raw_tool])

        assert len(result) == 1
        assert result[0] == {
            "type": "function",
            "function": raw_tool
        }

    # ------------------------------------------------------------------------
    # 3.3 mixed input
    # ------------------------------------------------------------------------

    def test_prepare_tools_mixed(self):

        def func_tool(a: str) -> str:
            """Function tool"""
            return a

        tool_def = ToolDef(
            name="tool_def",
            description="Tool definition",
            execute=lambda x: x,
        )

        dict_tool = {
            "name": "dict_tool",
            "description": "Dict tool",
            "parameters": {"type": "object", "properties": {}},
        }

        result = prepare_tools([func_tool, tool_def, dict_tool])

        assert len(result) == 3
        assert result[0]["function"]["name"] == "func_tool"
        assert result[0]["function"]["description"] == "Function tool"
        assert result[1]["function"]["name"] == "tool_def"
        assert result[2]["function"]["name"] == "dict_tool"

    # ------------------------------------------------------------------------
    # 3.4 empty list
    # ------------------------------------------------------------------------

    def test_prepare_tools_empty_list(self):
        result = prepare_tools([])
        assert result == []

# ============================================================================
# Test Suite 4: Tool Execution
# ============================================================================

import asyncio
import json


class TestToolExecution:
    # ------------------------------------------------------------------------
    # 4.1 _parse_arguments tests
    # ------------------------------------------------------------------------

    def test_parse_arguments_simple(self):
        from liteai_sdk.tool.execute import _parse_arguments

        arguments = '{"name": "Alice", "age": 30}'
        result = _parse_arguments(arguments)
        assert result == {"name": "Alice", "age": 30}

    def test_parse_arguments_nested(self):
        from liteai_sdk.tool.execute import _parse_arguments

        arguments = '{"user": {"name": "Bob", "roles": ["admin", "user"]}, "count": 5}'
        result = _parse_arguments(arguments)
        assert result == {"user": {"name": "Bob", "roles": ["admin", "user"]}, "count": 5}

    def test_parse_arguments_empty(self):
        from liteai_sdk.tool.execute import _parse_arguments

        arguments = '{}'
        result = _parse_arguments(arguments)
        assert result == {}

    # ------------------------------------------------------------------------
    # 4.2 execute_tool_sync with regular functions
    # ------------------------------------------------------------------------

    def test_execute_tool_sync_simple_function(self):
        from liteai_sdk.tool.execute import execute_tool_sync

        def add(a: int, b: int) -> int:
            """Add two numbers"""
            return a + b

        result = execute_tool_sync(add, '{"a": 5, "b": 3}')
        assert result == 8

    def test_execute_tool_sync_function_with_string(self):
        from liteai_sdk.tool.execute import execute_tool_sync

        def greet(name: str) -> str:
            """Greet a person"""
            return f"Hello, {name}!"

        result = execute_tool_sync(greet, '{"name": "Alice"}')
        assert result == "Hello, Alice!"

    def test_execute_tool_sync_function_with_dict(self):
        from liteai_sdk.tool.execute import execute_tool_sync

        def process(data: dict) -> int:
            """Process data"""
            return len(data)

        result = execute_tool_sync(process, '{"data": {"a": 1, "b": 2, "c": 3}}')
        assert result == 3

    # ------------------------------------------------------------------------
    # 4.3 execute_tool_sync with async functions
    # ------------------------------------------------------------------------

    def test_execute_tool_sync_async_function(self):
        from liteai_sdk.tool.execute import execute_tool_sync

        async def async_multiply(x: int, y: int) -> int:
            """Async multiply"""
            await asyncio.sleep(0.01)
            return x * y

        result = execute_tool_sync(async_multiply, '{"x": 4, "y": 7}')
        assert result == 28

    def test_execute_tool_sync_async_function_with_string(self):
        from liteai_sdk.tool.execute import execute_tool_sync

        async def async_upper(text: str) -> str:
            """Async uppercase"""
            await asyncio.sleep(0.01)
            return text.upper()

        result = execute_tool_sync(async_upper, '{"text": "hello"}')
        assert result == "HELLO"

    # ------------------------------------------------------------------------
    # 4.4 execute_tool_sync with ToolDef
    # ------------------------------------------------------------------------

    def test_execute_tool_sync_tooldef_sync(self):
        from liteai_sdk.tool.execute import execute_tool_sync

        def concat(a: str, b: str) -> str:
            """Concatenate strings"""
            return a + b

        tool_def = ToolDef(
            name="concat_tool",
            description="Concatenate two strings",
            execute=concat,
        )

        result = execute_tool_sync(tool_def, '{"a": "Hello", "b": " World"}')
        assert result == "Hello World"

    def test_execute_tool_sync_tooldef_async(self):
        from liteai_sdk.tool.execute import execute_tool_sync

        async def async_subtract(x: int, y: int) -> int:
            """Async subtract"""
            await asyncio.sleep(0.01)
            return x - y

        tool_def = ToolDef(
            name="subtract_tool",
            description="Subtract two numbers",
            execute=async_subtract,
        )

        result = execute_tool_sync(tool_def, '{"x": 10, "y": 3}')
        assert result == 7

    # ------------------------------------------------------------------------
    # 4.5 execute_tool (async) with regular functions
    # ------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_execute_tool_async_simple_function(self):
        from liteai_sdk.tool.execute import execute_tool

        def divide(a: int, b: int) -> float:
            """Divide two numbers"""
            return a / b

        result = await execute_tool(divide, '{"a": 10, "b": 2}')
        assert result == 5.0

    @pytest.mark.asyncio
    async def test_execute_tool_async_function_with_list(self):
        from liteai_sdk.tool.execute import execute_tool

        def sum_list(numbers: list) -> int:
            """Sum a list of numbers"""
            return sum(numbers)

        result = await execute_tool(sum_list, '{"numbers": [1, 2, 3, 4, 5]}')
        assert result == 15

    # ------------------------------------------------------------------------
    # 4.6 execute_tool (async) with async functions
    # ------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_execute_tool_async_async_function(self):
        from liteai_sdk.tool.execute import execute_tool

        async def async_power(base: int, exp: int) -> int:
            """Async power calculation"""
            await asyncio.sleep(0.01)
            return base ** exp

        result = await execute_tool(async_power, '{"base": 2, "exp": 10}')
        assert result == 1024

    @pytest.mark.asyncio
    async def test_execute_tool_async_async_function_complex(self):
        from liteai_sdk.tool.execute import execute_tool

        async def async_process(items: list[str], prefix: str) -> list[str]:
            """Async process list"""
            await asyncio.sleep(0.01)
            return [prefix + item for item in items]

        result = await execute_tool(async_process, '{"items": ["a", "b", "c"], "prefix": "x_"}')
        assert result == ["x_a", "x_b", "x_c"]

    # ------------------------------------------------------------------------
    # 4.7 execute_tool (async) with ToolDef
    # ------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_execute_tool_async_tooldef_sync(self):
        from liteai_sdk.tool.execute import execute_tool

        def multiply_by_two(n: int) -> int:
            """Multiply by two"""
            return n * 2

        tool_def = ToolDef(
            name="multiply_tool",
            description="Multiply number by two",
            execute=multiply_by_two,
        )

        result = await execute_tool(tool_def, '{"n": 21}')
        assert result == 42

    @pytest.mark.asyncio
    async def test_execute_tool_async_tooldef_async(self):
        from liteai_sdk.tool.execute import execute_tool

        async def async_reverse(text: str) -> str:
            """Async reverse string"""
            await asyncio.sleep(0.01)
            return text[::-1]

        tool_def = ToolDef(
            name="reverse_tool",
            description="Reverse a string",
            execute=async_reverse,
        )

        result = await execute_tool(tool_def, '{"text": "hello"}')
        assert result == "olleh"

    # ------------------------------------------------------------------------
    # 4.8 error handling
    # ------------------------------------------------------------------------

    def test_execute_tool_sync_invalid_json(self):
        from liteai_sdk.tool.execute import execute_tool_sync

        def dummy(x: int) -> int:
            """Dummy function"""
            return x

        with pytest.raises(json.JSONDecodeError):
            execute_tool_sync(dummy, 'invalid json')

    def test_execute_tool_sync_missing_argument(self):
        from liteai_sdk.tool.execute import execute_tool_sync

        def requires_arg(required: str) -> str:
            """Requires argument"""
            return required

        with pytest.raises(TypeError):
            execute_tool_sync(requires_arg, '{}')

    @pytest.mark.asyncio
    async def test_execute_tool_async_invalid_json(self):
        from liteai_sdk.tool.execute import execute_tool

        def dummy(x: int) -> int:
            """Dummy function"""
            return x

        with pytest.raises(json.JSONDecodeError):
            await execute_tool(dummy, 'not valid json')

# ============================================================================
# Fixtures and Helper Functions
# ============================================================================

@pytest.fixture
def sample_functions():

    def simple_func(x: int) -> int:
        """Simple function"""
        return x

    def complex_func(a: str, b: list[int], c: Optional[dict[str, Any]] = None) -> dict:
        """Complex function"""
        return {}

    return {"simple": simple_func, "complex": complex_func}


@pytest.fixture
def sample_types():
    from typing import TypedDict

    class UserDict(TypedDict):
        name: str
        age: int

    @dataclasses.dataclass
    class UserData:
        username: str
        email: str = "default@example.com"

    return {"typed_dict": UserDict, "dataclass": UserData}
