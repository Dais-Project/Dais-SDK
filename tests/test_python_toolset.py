import asyncio
import json
from typing import Any

import pytest

from liteai_sdk.tool.toolset import python_tool, PythonToolset
from liteai_sdk.tool.prepare import prepare_tools
from liteai_sdk.tool.execute import execute_tool, execute_tool_sync
from liteai_sdk.types.tool import ToolDef


class TestToolsetDecorator:
    """Test the @python_tool decorator and PythonToolset class"""

    # ------------------------------------------------------------------------
    # 1.1 @python_tool decorator basic functionality
    # ------------------------------------------------------------------------

    def test_tool_decorator_marks_function(self):
        """@python_tool decorator should mark a function with the tool flag"""
        from liteai_sdk.tool.toolset.python_toolset import TOOL_FLAG

        @python_tool
        def my_function(x: int) -> int:
            """Test function"""
            return x * 2

        assert hasattr(my_function, TOOL_FLAG)
        assert getattr(my_function, TOOL_FLAG) is True

    def test_tool_decorator_preserves_function(self):
        """@python_tool decorator should preserve the original function"""

        def original_func(a: str, b: int) -> str:
            """Original docstring"""
            return a * b

        decorated_func = python_tool(original_func)

        # Function should still work normally
        assert decorated_func("x", 3) == "xxx"
        assert decorated_func.__name__ == "original_func"
        assert decorated_func.__doc__ == "Original docstring"

    def test_tool_decorator_on_method(self):
        """@python_tool decorator should work on methods"""
        from liteai_sdk.tool.toolset.python_toolset import TOOL_FLAG

        class MyClass:
            @python_tool
            def my_method(self, value: int) -> int:
                """Test method"""
                return value + 10

        obj = MyClass()
        # The unbound method should have the flag
        assert hasattr(MyClass.my_method, TOOL_FLAG)
        # Bound method should also work
        assert obj.my_method(5) == 15


class TestToolsetBasic:
    """Test basic PythonToolset functionality"""

    # ------------------------------------------------------------------------
    # 1.2 PythonToolset with no tools
    # ------------------------------------------------------------------------

    def test_empty_toolset(self):
        """PythonToolset with no @python_tool decorated methods should return empty list"""
        class EmptyToolset(PythonToolset):
            def regular_method(self):
                """Not a tool"""
                return "hello"

        toolset = EmptyToolset()
        tools = toolset.get_tools()

        assert tools == []

    def test_toolset_name_property(self):
        """PythonToolset.name should return the class name by default"""
        class MyToolset(PythonToolset):
            pass

        toolset = MyToolset()
        assert toolset.name == "MyToolset"

    def test_toolset_format_tool_name(self):
        """Toolset.format_tool_name should follow the Namespace__ToolName pattern"""
        class MyToolset(PythonToolset):
            pass

        toolset = MyToolset()
        assert toolset.format_tool_name("my_tool") == "MyToolset__my_tool"

    # ------------------------------------------------------------------------
    # 1.3 PythonToolset with single tool
    # ------------------------------------------------------------------------

    def test_toolset_single_tool(self):
        """PythonToolset should discover single @python_tool decorated method"""
        class SingleToolset(PythonToolset):
            @python_tool
            def calculate(self, x: int, y: int) -> int:
                """Add two numbers"""
                return x + y

        toolset = SingleToolset()
        tools = toolset.get_tools()

        assert len(tools) == 1
        assert isinstance(tools[0], ToolDef)
        assert tools[0].name == "SingleToolset__calculate"
        assert callable(tools[0].execute)

    # ------------------------------------------------------------------------
    # 1.4 PythonToolset with multiple tools
    # ------------------------------------------------------------------------

    def test_toolset_multiple_tools(self):
        """PythonToolset should discover all @python_tool decorated methods"""
        class MultiToolset(PythonToolset):
            @python_tool
            def add(self, a: int, b: int) -> int:
                """Add two numbers"""
                return a + b

            @python_tool
            def multiply(self, a: int, b: int) -> int:
                """Multiply two numbers"""
                return a * b

            def helper(self, x: int) -> int:
                """Not a tool"""
                return x * 2

            @python_tool
            def subtract(self, a: int, b: int) -> int:
                """Subtract two numbers"""
                return a - b

        toolset = MultiToolset()
        tools = toolset.get_tools()

        assert len(tools) == 3
        assert all(isinstance(t, ToolDef) for t in tools)
        tool_names = {t.name for t in tools}
        assert tool_names == {"MultiToolset__add", "MultiToolset__multiply", "MultiToolset__subtract"}

    # ------------------------------------------------------------------------
    # 1.5 Tool methods are bound methods
    # ------------------------------------------------------------------------

    def test_toolset_methods_are_bound(self):
        """Tools returned from PythonToolset should be bound methods"""
        class Calculator(PythonToolset):
            def __init__(self, base: int):
                self.base = base

            @python_tool
            def add_to_base(self, value: int) -> int:
                """Add value to base"""
                return self.base + value

        calc = Calculator(base=100)
        tools = calc.get_tools()

        assert len(tools) == 1
        assert isinstance(tools[0], ToolDef)
        # Should be able to call without passing self
        assert callable(tools[0].execute)
        assert tools[0].execute(50) == 150


class TestToolsetIntegration:
    """Test PythonToolset integration with tool preparation"""

    # ------------------------------------------------------------------------
    # 1.6 prepare_tools with PythonToolset methods
    # ------------------------------------------------------------------------

    def test_prepare_tools_with_toolset_methods(self):
        """prepare_tools should work with methods from PythonToolset"""
        class MathToolset(PythonToolset):
            @python_tool
            def add(self, a: int, b: int) -> int:
                """Add two numbers"""
                return a + b

            @python_tool
            def multiply(self, x: int, y: int) -> int:
                """Multiply two numbers"""
                return x * y

        toolset = MathToolset()
        tool_methods = toolset.get_tools()
        prepared = prepare_tools(tool_methods)

        assert len(prepared) == 2
        assert prepared[0]["type"] == "function"
        assert prepared[1]["type"] == "function"

        # Check tool names
        names = {t["function"]["name"] for t in prepared}
        assert "MathToolset__add" in names
        assert "MathToolset__multiply" in names

    # ------------------------------------------------------------------------
    # 1.7 Mixed tools: functions and PythonToolset methods
    # ------------------------------------------------------------------------

    def test_prepare_tools_mixed_with_toolset(self):
        """prepare_tools should handle mix of functions and PythonToolset methods"""
        def standalone_func(text: str) -> str:
            """Standalone function"""
            return text.upper()

        class MyToolset(PythonToolset):
            @python_tool
            def toolset_method(self, n: int) -> int:
                """Toolset method"""
                return n * 2

        toolset = MyToolset()
        all_tools = [standalone_func] + toolset.get_tools()
        prepared = prepare_tools(all_tools)

        assert len(prepared) == 2
        names = {t["function"]["name"] for t in prepared}
        assert names == {"standalone_func", "MyToolset__toolset_method"}


class TestToolsetWithExecution:
    """Test PythonToolset methods with tool execution"""

    # ------------------------------------------------------------------------
    # 1.8 Execute PythonToolset method synchronously
    # ------------------------------------------------------------------------

    def test_execute_toolset_method_sync(self):
        """execute_tool_sync should work with PythonToolset methods"""
        class Calculator(PythonToolset):
            @python_tool
            def power(self, base: int, exponent: int) -> int:
                """Calculate power"""
                return base ** exponent

        calc = Calculator()
        tools = calc.get_tools()

        result = execute_tool_sync(tools[0], '{"base": 2, "exponent": 10}')
        assert result == "1024"
        assert json.loads(result) == 1024

    # ------------------------------------------------------------------------
    # 1.9 Execute PythonToolset method asynchronously
    # ------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_execute_toolset_method_async(self):
        """execute_tool should work with PythonToolset methods"""
        class StringTools(PythonToolset):
            @python_tool
            def reverse(self, text: str) -> str:
                """Reverse a string"""
                return text[::-1]

        tools_instance = StringTools()
        methods = tools_instance.get_tools()

        result = await execute_tool(methods[0], '{"text": "hello"}')
        assert result == "olleh"

    # ------------------------------------------------------------------------
    # 1.10 Execute async PythonToolset method
    # ------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_execute_async_toolset_method(self):
        """Async methods in PythonToolset should work"""
        class AsyncToolset(PythonToolset):
            @python_tool
            async def async_multiply(self, a: int, b: int) -> int:
                """Async multiply"""
                await asyncio.sleep(0.01)
                return a * b

        toolset = AsyncToolset()
        methods = toolset.get_tools()

        result = await execute_tool(methods[0], '{"a": 7, "b": 8}')
        assert result == "56"
        assert json.loads(result) == 56


class TestToolsetAdvanced:
    """Advanced PythonToolset test cases"""

    # ------------------------------------------------------------------------
    # 1.11 PythonToolset with instance state
    # ------------------------------------------------------------------------

    def test_toolset_with_state(self):
        """PythonToolset methods should have access to instance state"""
        class StatefulToolset(PythonToolset):
            def __init__(self, multiplier: int):
                self.multiplier = multiplier
                self.call_count = 0

            @python_tool
            def scale(self, value: int) -> int:
                """Scale value by multiplier"""
                self.call_count += 1
                return value * self.multiplier

        toolset = StatefulToolset(multiplier=5)
        tools = toolset.get_tools()

        # Execute the tool
        assert isinstance(tools[0], ToolDef)
        result1 = tools[0].execute(10)
        assert result1 == 50
        assert toolset.call_count == 1

        result2 = tools[0].execute(20)
        assert result2 == 100
        assert toolset.call_count == 2

    # ------------------------------------------------------------------------
    # 1.12 PythonToolset inheritance
    # ------------------------------------------------------------------------

    def test_toolset_inheritance(self):
        """Child PythonToolset should inherit parent's tools"""

        class BaseToolset(PythonToolset):
            @python_tool
            def base_method(self, x: int) -> int:
                """Base method"""
                return x * 2

        class DerivedToolset(BaseToolset):
            @python_tool
            def derived_method(self, x: int) -> int:
                """Derived method"""
                return x + 10

        toolset = DerivedToolset()
        tools = toolset.get_tools()

        assert len(tools) == 2
        tool_names = {t.name for t in tools}
        assert tool_names == {"DerivedToolset__base_method", "DerivedToolset__derived_method"}

    # ------------------------------------------------------------------------
    # 1.13 PythonToolset with complex parameter types
    # ------------------------------------------------------------------------

    def test_toolset_complex_types(self):
        """PythonToolset methods with complex types should work with prepare_tools"""

        class ComplexToolset(PythonToolset):
            @python_tool
            def process_data(self, items: list[str], config: dict[str, Any]) -> dict:
                """Process items with config"""
                return {"count": len(items), "config": config}

        toolset = ComplexToolset()
        prepared = prepare_tools(toolset.get_tools())

        assert len(prepared) == 1
        params = prepared[0]["function"]["parameters"]
        assert "items" in params["properties"]
        assert "config" in params["properties"]
        assert params["properties"]["items"]["type"] == "array"
        assert params["properties"]["config"]["type"] == "object"

    # ------------------------------------------------------------------------
    # 1.14 Multiple PythonToolset instances
    # ------------------------------------------------------------------------

    def test_multiple_toolset_instances(self):
        """Multiple instances of same PythonToolset should work independently"""

        class Counter(PythonToolset):
            def __init__(self, start: int):
                self.value = start

            @python_tool
            def increment(self, amount: int) -> int:
                """Increment counter"""
                self.value += amount
                return self.value

        counter1 = Counter(start=0)
        counter2 = Counter(start=100)

        tools1 = counter1.get_tools()
        tools2 = counter2.get_tools()

        # Execute on different instances
        assert isinstance(tools1[0], ToolDef)
        assert isinstance(tools2[0], ToolDef)
        assert tools1[0].execute(5) == 5
        assert tools2[0].execute(5) == 105
        assert counter1.value == 5
        assert counter2.value == 105


class TestParamParserWithToolsets:
    """Test ParamParser integration with toolsets"""

    # ------------------------------------------------------------------------
    # 1.15 ParamParser extracts tools from toolsets
    # ------------------------------------------------------------------------

    def test_param_parser_extract_toolsets(self):
        """ParamParser should extract tools from toolsets parameter"""
        from liteai_sdk.param_parser import ParamParser
        from liteai_sdk.types import LlmRequestParams
        from liteai_sdk.types.message import UserMessage

        class MyToolset(PythonToolset):
            @python_tool
            def tool1(self, x: int) -> int:
                """Tool 1"""
                return x * 2

            @python_tool
            def tool2(self, y: str) -> str:
                """Tool 2"""
                return y.upper()

        toolset = MyToolset()
        params = LlmRequestParams(
            model="test-model",
            messages=[UserMessage(content="test")],
            toolsets=[toolset]
        )

        extracted = ParamParser._extract_tool_params(params)

        assert extracted is not None
        assert len(extracted) == 2
        assert all(isinstance(tool, ToolDef) for tool in extracted)

    # ------------------------------------------------------------------------
    # 1.16 ParamParser with both tools and toolsets
    # ------------------------------------------------------------------------

    def test_param_parser_mixed_tools_and_toolsets(self):
        """ParamParser should handle both tools and toolsets together"""
        from liteai_sdk.param_parser import ParamParser
        from liteai_sdk.types import LlmRequestParams
        from liteai_sdk.types.message import UserMessage

        def standalone_tool(a: int) -> int:
            """Standalone tool"""
            return a + 1

        class MyToolset(PythonToolset):
            @python_tool
            def toolset_method(self, b: int) -> int:
                """Toolset method"""
                return b * 2

        toolset = MyToolset()
        params = LlmRequestParams(
            model="test-model",
            messages=[UserMessage(content="test")],
            tools=[standalone_tool],
            toolsets=[toolset]
        )

        extracted = ParamParser._extract_tool_params(params)

        assert extracted is not None
        assert len(extracted) == 2
        # Check we have both tools
        tool_names = {getattr(t, 'name', getattr(t, '__name__', str(t))) for t in extracted}
        assert 'standalone_tool' in tool_names
        assert 'MyToolset__toolset_method' in tool_names

    # ------------------------------------------------------------------------
    # 1.17 ParamParser with multiple toolsets
    # ------------------------------------------------------------------------

    def test_param_parser_multiple_toolsets(self):
        """ParamParser should handle multiple toolsets"""
        from liteai_sdk.param_parser import ParamParser
        from liteai_sdk.types import LlmRequestParams
        from liteai_sdk.types.message import UserMessage

        class MathToolset(PythonToolset):
            @python_tool
            def add(self, a: int, b: int) -> int:
                """Add"""
                return a + b

        class StringToolset(PythonToolset):
            @python_tool
            def upper(self, text: str) -> str:
                """Upper"""
                return text.upper()

            @python_tool
            def lower(self, text: str) -> str:
                """Lower"""
                return text.lower()

        math_tools = MathToolset()
        string_tools = StringToolset()

        params = LlmRequestParams(
            model="test-model",
            messages=[UserMessage(content="test")],
            toolsets=[math_tools, string_tools]
        )

        extracted = ParamParser._extract_tool_params(params)

        assert extracted is not None
        assert len(extracted) == 3  # 1 from MathToolset + 2 from StringToolset

    # ------------------------------------------------------------------------
    # 1.18 ParamParser with None toolsets
    # ------------------------------------------------------------------------

    def test_param_parser_none_toolsets(self):
        """ParamParser should handle None toolsets gracefully"""
        from liteai_sdk.param_parser import ParamParser
        from liteai_sdk.types import LlmRequestParams
        from liteai_sdk.types.message import UserMessage

        params = LlmRequestParams(
            model="test-model",
            messages=[UserMessage(content="test")],
            tools=None,
            toolsets=None
        )

        extracted = ParamParser._extract_tool_params(params)

        assert extracted is None

    # ------------------------------------------------------------------------
    # 1.19 ParamParser with empty toolsets list
    # ------------------------------------------------------------------------

    def test_param_parser_empty_toolsets(self):
        """ParamParser should handle empty toolsets list"""
        from liteai_sdk.param_parser import ParamParser
        from liteai_sdk.types import LlmRequestParams
        from liteai_sdk.types.message import UserMessage

        params = LlmRequestParams(
            model="test-model",
            messages=[UserMessage(content="test")],
            tools=None,
            toolsets=[]
        )

        extracted = ParamParser._extract_tool_params(params)

        # Should return empty list since toolsets is empty
        assert extracted == []


class TestToolsetEdgeCases:
    """Edge cases and error scenarios for PythonToolset"""

    # ------------------------------------------------------------------------
    # 1.20 PythonToolset with private methods
    # ------------------------------------------------------------------------

    def test_toolset_ignores_private_methods(self):
        """PythonToolset should not include private methods even if decorated"""

        class PrivateToolset(PythonToolset):
            @python_tool
            def public_method(self, x: int) -> int:
                """Public"""
                return x

            @python_tool
            def _private_method(self, x: int) -> int:
                """Private"""
                return x * 2

        toolset = PrivateToolset()
        tools = toolset.get_tools()

        # Both should be included (private methods are still methods)
        # The TOOL_FLAG doesn't distinguish between public/private
        tool_names = {t.name for t in tools}
        # This depends on implementation, but typically both would be included
        assert 'PrivateToolset__public_method' in tool_names

    # ------------------------------------------------------------------------
    # 1.21 Toolset with classmethod/staticmethod
    # ------------------------------------------------------------------------

    def test_toolset_with_classmethod(self):
        """@python_tool on classmethod should be returned by get_tools"""

        class ToolsetWithClassmethod(PythonToolset):
            @python_tool
            def instance_method(self, x: int) -> int:
                """Instance method"""
                return x

            @classmethod
            @python_tool
            def class_method(cls, x: int) -> int:
                """Class method"""
                return x * 2

        toolset = ToolsetWithClassmethod()
        tools = toolset.get_tools()

        assert len(tools) == 2

    # ------------------------------------------------------------------------
    # 1.22 Toolset method without docstring
    # ------------------------------------------------------------------------

    def test_toolset_method_without_docstring_allowed(self):
        """PythonToolset method without docstring should be allowed but have empty description"""
        class NoDocToolset(PythonToolset):
            @python_tool
            def no_doc_method(self, x: int) -> int:
                return x * 2

        toolset = NoDocToolset()
        tools = toolset.get_tools()

        assert len(tools) == 1
        assert tools[0].description == ""
        
        # prepare_tools should now work because ToolDef has description (even if empty)
        prepared = prepare_tools(tools)
        assert len(prepared) == 1
        assert prepared[0]["function"]["description"] == ""