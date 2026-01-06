import asyncio
import json

import pytest

from liteai_sdk.types.tool import ToolDef
from liteai_sdk.tool.execute import execute_tool, execute_tool_sync


class TestToolExecution:
    # ------------------------------------------------------------------------
    # 4.1 _arguments_normalizer tests (via execute_tool_sync)
    # ------------------------------------------------------------------------
    # Note: _arguments_normalizer is a private function, so we test it indirectly
    # through execute_tool_sync which uses it internally

    # ------------------------------------------------------------------------
    # 4.2 execute_tool_sync with regular functions
    # ------------------------------------------------------------------------

    def test_execute_tool_sync_simple_function(self):
        def add(a: int, b: int) -> int:
            """Add two numbers"""
            return a + b

        result = execute_tool_sync(add, '{"a": 5, "b": 3}')
        assert result == "8"
        assert json.loads(result) == 8

    def test_execute_tool_sync_function_with_string(self):
        def greet(name: str) -> str:
            """Greet a person"""
            return f"Hello, {name}!"

        result = execute_tool_sync(greet, '{"name": "Alice"}')
        # String results should pass through unchanged
        assert result == "Hello, Alice!"

    def test_execute_tool_sync_function_with_dict(self):
        def process(data: dict) -> int:
            """Process data"""
            return len(data)

        result = execute_tool_sync(process, '{"data": {"a": 1, "b": 2, "c": 3}}')
        assert result == "3"
        assert json.loads(result) == 3

    # ------------------------------------------------------------------------
    # 4.3 execute_tool_sync with async functions
    # ------------------------------------------------------------------------

    def test_execute_tool_sync_async_function(self):
        async def async_multiply(x: int, y: int) -> int:
            """Async multiply"""
            await asyncio.sleep(0.01)
            return x * y

        result = execute_tool_sync(async_multiply, '{"x": 4, "y": 7}')
        assert result == "28"
        assert json.loads(result) == 28

    def test_execute_tool_sync_async_function_with_string(self):
        async def async_upper(text: str) -> str:
            """Async uppercase"""
            await asyncio.sleep(0.01)
            return text.upper()

        result = execute_tool_sync(async_upper, '{"text": "hello"}')
        # String results should pass through unchanged
        assert result == "HELLO"

    # ------------------------------------------------------------------------
    # 4.4 execute_tool_sync with ToolDef
    # ------------------------------------------------------------------------

    def test_execute_tool_sync_tooldef_sync(self):
        def concat(a: str, b: str) -> str:
            """Concatenate strings"""
            return a + b

        tool_def = ToolDef(
            name="concat_tool",
            description="Concatenate two strings",
            execute=concat,
        )

        result = execute_tool_sync(tool_def, '{"a": "Hello", "b": " World"}')
        # String results should pass through unchanged
        assert result == "Hello World"

    def test_execute_tool_sync_tooldef_async(self):
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
        assert result == "7"
        assert json.loads(result) == 7

    # ------------------------------------------------------------------------
    # 4.5 execute_tool (async) with regular functions
    # ------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_execute_tool_async_simple_function(self):
        def divide(a: int, b: int) -> float:
            """Divide two numbers"""
            return a / b

        result = await execute_tool(divide, '{"a": 10, "b": 2}')
        assert result == "5.0"
        assert json.loads(result) == 5.0

    @pytest.mark.asyncio
    async def test_execute_tool_async_function_with_list(self):
        def sum_list(numbers: list) -> int:
            """Sum a list of numbers"""
            return sum(numbers)

        result = await execute_tool(sum_list, '{"numbers": [1, 2, 3, 4, 5]}')
        assert result == "15"
        assert json.loads(result) == 15

    # ------------------------------------------------------------------------
    # 4.6 execute_tool (async) with async functions
    # ------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_execute_tool_async_async_function(self):
        async def async_power(base: int, exp: int) -> int:
            """Async power calculation"""
            await asyncio.sleep(0.01)
            return base ** exp

        result = await execute_tool(async_power, '{"base": 2, "exp": 10}')
        assert result == "1024"
        assert json.loads(result) == 1024

    @pytest.mark.asyncio
    async def test_execute_tool_async_async_function_complex(self):
        async def async_process(items: list[str], prefix: str) -> list[str]:
            """Async process list"""
            await asyncio.sleep(0.01)
            return [prefix + item for item in items]

        result = await execute_tool(async_process, '{"items": ["a", "b", "c"], "prefix": "x_"}')
        assert isinstance(result, str)
        assert json.loads(result) == ["x_a", "x_b", "x_c"]

    # ------------------------------------------------------------------------
    # 4.7 execute_tool (async) with ToolDef
    # ------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_execute_tool_async_tooldef_sync(self):
        def multiply_by_two(n: int) -> int:
            """Multiply by two"""
            return n * 2

        tool_def = ToolDef(
            name="multiply_tool",
            description="Multiply number by two",
            execute=multiply_by_two,
        )

        result = await execute_tool(tool_def, '{"n": 21}')
        assert result == "42"
        assert json.loads(result) == 42

    @pytest.mark.asyncio
    async def test_execute_tool_async_tooldef_async(self):
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
        # String results should pass through unchanged
        assert result == "olleh"

    # ------------------------------------------------------------------------
    # 4.8 error handling
    # ------------------------------------------------------------------------

    def test_execute_tool_sync_invalid_json(self):
        def dummy(x: int) -> int:
            """Dummy function"""
            return x

        with pytest.raises(json.JSONDecodeError):
            execute_tool_sync(dummy, 'invalid json')

    def test_execute_tool_sync_missing_argument(self):
        def requires_arg(required: str) -> str:
            """Requires argument"""
            return required

        with pytest.raises(TypeError):
            execute_tool_sync(requires_arg, '{}')

    @pytest.mark.asyncio
    async def test_execute_tool_async_invalid_json(self):
        def dummy(x: int) -> int:
            """Dummy function"""
            return x

        with pytest.raises(json.JSONDecodeError):
            await execute_tool(dummy, 'not valid json')

    # ------------------------------------------------------------------------
    # 4.9 class method tests
    # ------------------------------------------------------------------------

    def test_execute_tool_sync_instance_method(self):
        class Calculator:
            def multiply(self, x: int, y: int) -> int:
                """Multiply two numbers"""
                return x * y

        calc = Calculator()
        result = execute_tool_sync(calc.multiply, '{"x": 6, "y": 7}')
        assert result == "42"
        assert json.loads(result) == 42

    def test_execute_tool_sync_classmethod(self):
        class MathUtils:
            @classmethod
            def power(cls, base: int, exp: int) -> int:
                """Calculate power"""
                return base ** exp

        result = execute_tool_sync(MathUtils.power, '{"base": 2, "exp": 8}')
        assert result == "256"
        assert json.loads(result) == 256

    def test_execute_tool_sync_staticmethod(self):
        class StringUtils:
            @staticmethod
            def reverse(text: str) -> str:
                """Reverse a string"""
                return text[::-1]

        result = execute_tool_sync(StringUtils.reverse, '{"text": "hello"}')
        assert result == "olleh"

    def test_execute_tool_sync_async_instance_method(self):
        class AsyncCalculator:
            async def add(self, a: int, b: int) -> int:
                """Async add"""
                await asyncio.sleep(0.01)
                return a + b

        calc = AsyncCalculator()
        result = execute_tool_sync(calc.add, '{"a": 10, "b": 15}')
        assert result == "25"
        assert json.loads(result) == 25

    @pytest.mark.asyncio
    async def test_execute_tool_async_instance_method(self):
        class Calculator:
            def subtract(self, x: int, y: int) -> int:
                """Subtract two numbers"""
                return x - y

        calc = Calculator()
        result = await execute_tool(calc.subtract, '{"x": 20, "y": 8}')
        assert result == "12"
        assert json.loads(result) == 12

    @pytest.mark.asyncio
    async def test_execute_tool_async_classmethod(self):
        class MathUtils:
            @classmethod
            def multiply(cls, a: int, b: int) -> int:
                """Multiply numbers"""
                return a * b

        result = await execute_tool(MathUtils.multiply, '{"a": 9, "b": 6}')
        assert result == "54"
        assert json.loads(result) == 54

    @pytest.mark.asyncio
    async def test_execute_tool_async_staticmethod(self):
        class StringUtils:
            @staticmethod
            def uppercase(text: str) -> str:
                """Convert to uppercase"""
                return text.upper()

        result = await execute_tool(StringUtils.uppercase, '{"text": "world"}')
        assert result == "WORLD"

    @pytest.mark.asyncio
    async def test_execute_tool_async_async_instance_method(self):
        class AsyncCalculator:
            async def divide(self, x: float, y: float) -> float:
                """Async divide"""
                await asyncio.sleep(0.01)
                return x / y

        calc = AsyncCalculator()
        result = await execute_tool(calc.divide, '{"x": 100, "y": 4}')
        assert result == "25.0"
        assert json.loads(result) == 25.0

    # ------------------------------------------------------------------------
    # 4.10 invalid tool type error tests
    # ------------------------------------------------------------------------

    def test_execute_tool_sync_invalid_type_int(self):
        with pytest.raises(ValueError, match="Invalid tool type"):
            execute_tool_sync(123, '{"x": 1}')  # type: ignore

    def test_execute_tool_sync_invalid_type_string(self):
        with pytest.raises(ValueError, match="Invalid tool type"):
            execute_tool_sync("not a function", '{"x": 1}')  # type: ignore

    def test_execute_tool_sync_invalid_type_list(self):
        with pytest.raises(ValueError, match="Invalid tool type"):
            execute_tool_sync([1, 2, 3], '{"x": 1}')  # type: ignore

    def test_execute_tool_sync_invalid_type_dict(self):
        with pytest.raises(ValueError, match="Invalid tool type"):
            execute_tool_sync({"key": "value"}, '{"x": 1}')  # type: ignore

    def test_execute_tool_sync_invalid_type_none(self):
        with pytest.raises(ValueError, match="Invalid tool type"):
            execute_tool_sync(None, '{"x": 1}')  # type: ignore

    @pytest.mark.asyncio
    async def test_execute_tool_async_invalid_type_int(self):
        with pytest.raises(ValueError, match="Invalid tool type"):
            await execute_tool(456, '{"x": 1}')  # type: ignore

    @pytest.mark.asyncio
    async def test_execute_tool_async_invalid_type_string(self):
        with pytest.raises(ValueError, match="Invalid tool type"):
            await execute_tool("invalid", '{"x": 1}')  # type: ignore

    @pytest.mark.asyncio
    async def test_execute_tool_async_invalid_type_list(self):
        with pytest.raises(ValueError, match="Invalid tool type"):
            await execute_tool([4, 5, 6], '{"x": 1}')  # type: ignore

    @pytest.mark.asyncio
    async def test_execute_tool_async_invalid_type_dict(self):
        with pytest.raises(ValueError, match="Invalid tool type"):
            await execute_tool({"foo": "bar"}, '{"x": 1}')  # type: ignore

    @pytest.mark.asyncio
    async def test_execute_tool_async_invalid_type_none(self):
        with pytest.raises(ValueError, match="Invalid tool type"):
            await execute_tool(None, '{"x": 1}')  # type: ignore