"""
Test suite for tool utility functions in liteai_sdk.tool.utils
"""

import pytest
from liteai_sdk.tool import ToolDef
from liteai_sdk.tool.utils import find_tool_by_name

# ============================================================================
# Test Suite 1: find_tool_by_name
# ============================================================================


class TestFindToolByName:
    """Test find_tool_by_name function"""

    # ------------------------------------------------------------------------
    # 1.1 Test finding from callable functions list
    # ------------------------------------------------------------------------

    def test_find_callable_by_name(self):
        """Test finding tool by name from callable functions list"""

        def get_weather(location: str) -> str:
            """Get weather"""
            return f"Weather in {location}"

        def get_time() -> str:
            """Get time"""
            return "12:00"

        tools = [get_weather, get_time]

        # Find existing tools
        result = find_tool_by_name(tools, "get_weather")
        assert result is get_weather

        result = find_tool_by_name(tools, "get_time")
        assert result is get_time

    # ------------------------------------------------------------------------
    # 1.2 Test finding from ToolDef list
    # ------------------------------------------------------------------------

    def test_find_tool_def_by_name(self):
        """Test finding tool by name from ToolDef list"""

        def execute1(x: int) -> int:
            return x

        def execute2(y: str) -> str:
            return y

        tool_def1 = ToolDef(name="calculator", description="Calculate", execute=execute1)
        tool_def2 = ToolDef(name="formatter", description="Format", execute=execute2)

        tools = [tool_def1, tool_def2]

        # Find existing tools
        result = find_tool_by_name(tools, "calculator")
        assert result is tool_def1

        result = find_tool_by_name(tools, "formatter")
        assert result is tool_def2

    # ------------------------------------------------------------------------
    # 1.3 Test finding from raw tool definition (dict) list
    # ------------------------------------------------------------------------

    def test_find_raw_tool_def_by_name(self):
        """Test finding tool by name from raw tool definition (dict) list"""
        raw_tool1 = {
            "name": "search",
            "description": "Search data",
            "parameters": {"type": "object", "properties": {}},
        }
        raw_tool2 = {
            "name": "filter",
            "description": "Filter data",
            "parameters": {"type": "object", "properties": {}},
        }

        tools = [raw_tool1, raw_tool2]

        # Find existing tools
        result = find_tool_by_name(tools, "search")
        assert result is raw_tool1

        result = find_tool_by_name(tools, "filter")
        assert result is raw_tool2

    # ------------------------------------------------------------------------
    # 1.4 Test finding from mixed type list
    # ------------------------------------------------------------------------

    def test_find_from_mixed_tools(self):
        """Test finding tool from mixed type tool list"""

        def func_tool(x: int) -> int:
            """Function tool"""
            return x

        def execute_fn(y: str) -> str:
            return y

        tool_def = ToolDef(name="my_tool_def", description="Tool def", execute=execute_fn)

        raw_tool = {
            "name": "my_raw_tool",
            "description": "Raw tool",
            "parameters": {"type": "object", "properties": {}},
        }

        tools = [func_tool, tool_def, raw_tool]

        # Find each type of tool
        result = find_tool_by_name(tools, "func_tool")
        assert result is func_tool

        result = find_tool_by_name(tools, "my_tool_def")
        assert result is tool_def

        result = find_tool_by_name(tools, "my_raw_tool")
        assert result is raw_tool

    # ------------------------------------------------------------------------
    # 1.5 Test finding non-existent tool
    # ------------------------------------------------------------------------

    def test_find_nonexistent_tool(self):
        """Test finding non-existent tool should return None"""

        def tool1(x: int) -> int:
            """Tool 1"""
            return x

        tools = [tool1]

        result = find_tool_by_name(tools, "nonexistent")
        assert result is None

    # ------------------------------------------------------------------------
    # 1.6 Test finding in empty list
    # ------------------------------------------------------------------------

    def test_find_in_empty_list(self):
        """Test finding in empty list should return None"""
        tools = []
        result = find_tool_by_name(tools, "anything")
        assert result is None

    # ------------------------------------------------------------------------
    # 1.7 Test exact name matching only
    # ------------------------------------------------------------------------

    def test_find_exact_match_only(self):
        """Test only exact name matches will be returned"""

        def get_user(id: int) -> dict:
            """Get user"""
            return {}

        def get_users() -> list:
            """Get users"""
            return []

        tools = [get_user, get_users]

        # Should find exact match
        result = find_tool_by_name(tools, "get_user")
        assert result is get_user

        result = find_tool_by_name(tools, "get_users")
        assert result is get_users

        # Partial matches should return None
        result = find_tool_by_name(tools, "get")
        assert result is None

        result = find_tool_by_name(tools, "user")
        assert result is None

    # ------------------------------------------------------------------------
    # 1.8 Test case-sensitive name matching
    # ------------------------------------------------------------------------

    def test_find_case_sensitive(self):
        """Test name matching is case-sensitive"""

        def MyTool(x: int) -> int:
            """My tool"""
            return x

        tools = [MyTool]

        # Exact match should be found
        result = find_tool_by_name(tools, "MyTool")
        assert result is MyTool

        # Case mismatch should return None
        result = find_tool_by_name(tools, "mytool")
        assert result is None

        result = find_tool_by_name(tools, "MYTOOL")
        assert result is None

    # ------------------------------------------------------------------------
    # 1.9 Test returning first match
    # ------------------------------------------------------------------------

    def test_find_returns_first_match(self):
        """Test when multiple tools have same name, return first match"""

        def tool(x: int) -> int:
            """First tool"""
            return x

        def another_execute(y: str) -> str:
            return y

        # Create two ToolDefs with same name
        tool_def1 = ToolDef(name="duplicate", description="First", execute=another_execute)
        tool_def2 = ToolDef(name="duplicate", description="Second", execute=another_execute)

        tools = [tool_def1, tool_def2]

        result = find_tool_by_name(tools, "duplicate")
        assert result is tool_def1  # Should return the first one

    # ------------------------------------------------------------------------
    # 1.10 Test lambda function
    # ------------------------------------------------------------------------

    def test_find_lambda_function(self):
        """Test finding lambda function (lambda's __name__ is '<lambda>')"""
        lambda_tool = lambda x: x * 2
        normal_tool = lambda y: y + 1

        tools = [lambda_tool, normal_tool]

        # Lambda functions all have __name__ as '<lambda>'
        result = find_tool_by_name(tools, "<lambda>")
        assert result is lambda_tool  # Return first lambda

    # ------------------------------------------------------------------------
    # 1.11 Test complex mixed scenario
    # ------------------------------------------------------------------------

    def test_find_complex_mixed_scenario(self):
        """Test complex mixed scenario"""

        def add(a: int, b: int) -> int:
            """Add numbers"""
            return a + b

        def multiply_execute(x: int, y: int) -> int:
            return x * y

        tool_def = ToolDef(name="multiply", description="Multiply", execute=multiply_execute)

        raw_tool = {
            "name": "divide",
            "description": "Divide numbers",
            "parameters": {"type": "object", "properties": {}},
        }

        def subtract(a: int, b: int) -> int:
            """Subtract numbers"""
            return a - b

        tools = [add, tool_def, raw_tool, subtract]

        # Test finding each tool
        assert find_tool_by_name(tools, "add") is add
        assert find_tool_by_name(tools, "multiply") is tool_def
        assert find_tool_by_name(tools, "divide") is raw_tool
        assert find_tool_by_name(tools, "subtract") is subtract

        # Test finding non-existent tool
        assert find_tool_by_name(tools, "power") is None

    # ------------------------------------------------------------------------
    # 1.12 Test dict without name field
    # ------------------------------------------------------------------------

    def test_find_dict_without_name(self):
        """Test handling dict tool without name field"""
        incomplete_dict = {
            "description": "Some tool",
            "parameters": {"type": "object", "properties": {}},
        }

        tools = [incomplete_dict]

        # Dict without name field should not be matched
        result = find_tool_by_name(tools, "anything")
        assert result is None

    # ------------------------------------------------------------------------
    # 1.13 Test special character names
    # ------------------------------------------------------------------------

    def test_find_special_character_names(self):
        """Test tool names containing special characters"""

        def tool_with_underscore(x: int) -> int:
            """Tool with underscore"""
            return x

        def execute_dash(y: str) -> str:
            return y

        # Note: function names cannot contain '-', but ToolDef's name can
        tool_def = ToolDef(name="tool-with-dash", description="Tool", execute=execute_dash)

        raw_tool = {
            "name": "tool.with.dot",
            "description": "Tool with dot",
            "parameters": {},
        }

        tools = [tool_with_underscore, tool_def, raw_tool]

        assert find_tool_by_name(tools, "tool_with_underscore") is tool_with_underscore
        assert find_tool_by_name(tools, "tool-with-dash") is tool_def
        assert find_tool_by_name(tools, "tool.with.dot") is raw_tool
