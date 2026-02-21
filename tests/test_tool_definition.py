from typing import Annotated, Any, Optional

import pytest

from dais_sdk.types.tool import ToolDef
from dais_sdk.tool.prepare import (
    generate_tool_definition_from_callable,
    generate_tool_definition_from_raw_tool_def,
    generate_tool_definition_from_tool_def,
)


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
        assert result["function"]["description"] == "This is a longer docstring.\nIt spans multiple lines."

    # ------------------------------------------------------------------------
    # 2.7 Annotated parameter description parsing
    # ------------------------------------------------------------------------

    def test_function_parameter_description_from_annotated_string(self):

        def search(query: Annotated[str, "User search query"]) -> str:
            """Search content by query"""
            return query

        result = generate_tool_definition_from_callable(search)
        query_schema = result["function"]["parameters"]["properties"]["query"]

        assert query_schema["type"] == "string"
        assert query_schema["description"] == "User search query"

    def test_function_parameter_description_fallback_when_annotated_metadata_invalid_or_missing(self):

        def summarize(
            text: Annotated[str, 123],
            language: str,
        ) -> str:
            """Summarize text"""
            return text

        result = generate_tool_definition_from_callable(summarize)
        props = result["function"]["parameters"]["properties"]

        assert props["text"]["description"] == "Parameter text of type str"
        assert props["language"]["description"] == "Parameter language of type str"

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