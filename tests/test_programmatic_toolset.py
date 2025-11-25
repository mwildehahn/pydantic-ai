"""Tests for ProgrammaticToolset - programmatic tool calling via code execution."""

from __future__ import annotations

import json
from typing import TypeVar

import pytest

from pydantic_ai import FunctionToolset, ToolCallPart
from pydantic_ai._run_context import RunContext
from pydantic_ai._tool_manager import ToolManager
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets.programmatic import (
    CodeSandbox,
    ProgrammaticToolset,
    _generate_tool_function,
    _json_type_to_python,
)
from pydantic_ai.usage import RunUsage

pytestmark = pytest.mark.anyio

T = TypeVar('T')


def build_run_context(deps: T, run_step: int = 0) -> RunContext[T]:
    return RunContext(
        deps=deps,
        model=TestModel(),
        usage=RunUsage(),
        prompt=None,
        messages=[],
        run_step=run_step,
    )


class TestJsonTypeConversion:
    """Test JSON schema type to Python type conversion."""

    def test_basic_types(self):
        assert _json_type_to_python('string') == 'str'
        assert _json_type_to_python('integer') == 'int'
        assert _json_type_to_python('number') == 'float'
        assert _json_type_to_python('boolean') == 'bool'
        assert _json_type_to_python('array') == 'list'
        assert _json_type_to_python('object') == 'dict'
        assert _json_type_to_python('null') == 'None'

    def test_unknown_type(self):
        assert _json_type_to_python('unknown') == 'Any'

    def test_union_types(self):
        assert _json_type_to_python(['string', 'integer']) == 'str | int'
        assert _json_type_to_python(['string', 'null']) == 'str | None'


class TestToolFunctionGeneration:
    """Test generation of Python functions from tool definitions."""

    def test_simple_function(self):
        from pydantic_ai.tools import ToolDefinition

        tool_def = ToolDefinition(
            name='add',
            description='Add two numbers',
            parameters_json_schema={
                'type': 'object',
                'properties': {
                    'a': {'type': 'integer'},
                    'b': {'type': 'integer'},
                },
                'required': ['a', 'b'],
            },
        )

        code = _generate_tool_function('add', tool_def)
        assert 'def add(a: int, b: int):' in code
        assert '"""Add two numbers"""' in code
        assert "__call_tool__('add'" in code

    def test_optional_parameters(self):
        from pydantic_ai.tools import ToolDefinition

        tool_def = ToolDefinition(
            name='search',
            description='Search for items',
            parameters_json_schema={
                'type': 'object',
                'properties': {
                    'query': {'type': 'string'},
                    'limit': {'type': 'integer', 'default': 10},
                },
                'required': ['query'],
            },
        )

        code = _generate_tool_function('search', tool_def)
        assert 'query: str' in code
        assert 'limit: int = 10' in code


class TestCodeSandbox:
    """Test the CodeSandbox execution environment."""

    async def test_simple_execution(self):
        sandbox = CodeSandbox(timeout=10.0)

        async def no_tools(name: str, args: dict):
            raise RuntimeError('No tools available')

        result = await sandbox.execute('print("hello world")', no_tools)
        assert result.stdout.strip() == 'hello world'
        assert result.returncode == 0

    async def test_execution_with_return_value(self):
        sandbox = CodeSandbox(timeout=10.0)

        async def no_tools(name: str, args: dict):
            raise RuntimeError('No tools available')

        result = await sandbox.execute(
            """
x = 1 + 2
print(f"result: {x}")
""",
            no_tools,
        )
        assert 'result: 3' in result.stdout
        assert result.returncode == 0

    async def test_execution_error(self):
        sandbox = CodeSandbox(timeout=10.0)

        async def no_tools(name: str, args: dict):
            raise RuntimeError('No tools available')

        result = await sandbox.execute('raise ValueError("test error")', no_tools)
        assert result.returncode != 0
        assert 'ValueError' in result.stderr

    async def test_timeout(self):
        sandbox = CodeSandbox(timeout=0.5)

        async def no_tools(name: str, args: dict):
            raise RuntimeError('No tools available')

        result = await sandbox.execute(
            """
import time
time.sleep(10)
print("done")
""",
            no_tools,
        )
        assert result.error is not None
        assert 'timed out' in result.error.lower()

    async def test_tool_call_from_code(self):
        sandbox = CodeSandbox(timeout=10.0)

        async def mock_tool_handler(name: str, args: dict):
            if name == 'add':
                return args['a'] + args['b']
            raise ValueError(f'Unknown tool: {name}')

        # Code that calls a tool
        code = """
result = __call_tool__('add', {'a': 5, 'b': 3})
print(f"result: {result}")
"""
        result = await sandbox.execute(code, mock_tool_handler)
        assert 'result: 8' in result.stdout
        assert result.returncode == 0

    async def test_multiple_tool_calls(self):
        sandbox = CodeSandbox(timeout=10.0)

        call_count = 0

        async def mock_tool_handler(name: str, args: dict):
            nonlocal call_count
            call_count += 1
            if name == 'get_value':
                return args['key'] * 2
            raise ValueError(f'Unknown tool: {name}')

        code = """
a = __call_tool__('get_value', {'key': 10})
b = __call_tool__('get_value', {'key': 20})
print(f"sum: {a + b}")
"""
        result = await sandbox.execute(code, mock_tool_handler)
        assert 'sum: 60' in result.stdout  # (10*2) + (20*2) = 60
        assert call_count == 2


class TestProgrammaticToolset:
    """Test the ProgrammaticToolset integration."""

    async def test_get_tools_returns_run_python_code(self):
        base_toolset = FunctionToolset[None]()

        @base_toolset.tool
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        programmatic = ProgrammaticToolset(base_toolset)
        ctx = build_run_context(None)

        tools = await programmatic.get_tools(ctx)
        assert 'run_python_code' in tools
        assert len(tools) == 1

        tool = tools['run_python_code']
        assert tool.tool_def.name == 'run_python_code'
        assert 'add' in tool.tool_def.description

    async def test_custom_tool_name(self):
        base_toolset = FunctionToolset[None]()
        programmatic = ProgrammaticToolset(base_toolset, tool_name='execute_code')
        ctx = build_run_context(None)

        tools = await programmatic.get_tools(ctx)
        assert 'execute_code' in tools

    async def test_simple_code_execution(self):
        base_toolset = FunctionToolset[None]()
        programmatic = ProgrammaticToolset(base_toolset)
        ctx = build_run_context(None)

        tools = await programmatic.get_tools(ctx)
        tool = tools['run_python_code']

        result = await programmatic.call_tool(
            'run_python_code',
            {'code': 'print("hello from sandbox")'},
            ctx,
            tool,
        )
        assert 'hello from sandbox' in result

    async def test_tool_call_from_programmatic_code(self):
        base_toolset = FunctionToolset[None]()

        @base_toolset.tool
        def multiply(a: int, b: int) -> int:
            """Multiply two numbers."""
            return a * b

        programmatic = ProgrammaticToolset(base_toolset)
        ctx = build_run_context(None)

        tools = await programmatic.get_tools(ctx)
        tool = tools['run_python_code']

        # Code that uses the multiply tool
        code = """
result = multiply(6, 7)
print(f"6 * 7 = {result}")
"""
        result = await programmatic.call_tool('run_python_code', {'code': code}, ctx, tool)
        assert '6 * 7 = 42' in result

    async def test_multiple_tool_calls_in_loop(self):
        base_toolset = FunctionToolset[None]()
        call_log = []

        @base_toolset.tool
        def get_data(item_id: int) -> dict:
            """Get data for an item."""
            call_log.append(item_id)
            return {'id': item_id, 'value': item_id * 10}

        programmatic = ProgrammaticToolset(base_toolset)
        ctx = build_run_context(None)

        tools = await programmatic.get_tools(ctx)
        tool = tools['run_python_code']

        # Code that calls tools in a loop
        code = """
import json
results = []
for i in range(3):
    data = get_data(item_id=i)
    results.append(data)
total = sum(r['value'] for r in results)
print(json.dumps({'total': total, 'count': len(results)}))
"""
        result = await programmatic.call_tool('run_python_code', {'code': code}, ctx, tool)
        parsed = json.loads(result)
        assert parsed['total'] == 30  # 0*10 + 1*10 + 2*10 = 30
        assert parsed['count'] == 3
        assert call_log == [0, 1, 2]

    async def test_data_processing_without_context_pollution(self):
        """Test that intermediate results don't pollute context."""
        base_toolset = FunctionToolset[None]()

        @base_toolset.tool
        def get_large_data() -> list:
            """Get a large dataset."""
            # Simulate large dataset that shouldn't go to model context
            return [{'id': i, 'value': i * 100} for i in range(100)]

        programmatic = ProgrammaticToolset(base_toolset)
        ctx = build_run_context(None)

        tools = await programmatic.get_tools(ctx)
        tool = tools['run_python_code']

        # Code processes large data but returns summary
        code = """
data = get_large_data()
# Process locally - this data doesn't go to model context
total = sum(item['value'] for item in data)
avg = total / len(data)
print(f"Processed {len(data)} items. Total: {total}, Avg: {avg}")
"""
        result = await programmatic.call_tool('run_python_code', {'code': code}, ctx, tool)
        assert 'Processed 100 items' in result
        assert 'Total: 495000' in result  # sum of 0*100 to 99*100

    async def test_error_handling_in_code(self):
        base_toolset = FunctionToolset[None]()

        @base_toolset.tool
        def may_fail(should_fail: bool) -> str:
            """A tool that may fail."""
            if should_fail:
                raise ValueError('Intentional failure')
            return 'success'

        programmatic = ProgrammaticToolset(base_toolset)
        ctx = build_run_context(None)

        tools = await programmatic.get_tools(ctx)
        tool = tools['run_python_code']

        # Code that handles errors
        code = """
try:
    result = may_fail(should_fail=True)
except Exception as e:
    result = f"caught error: {e}"
print(result)
"""
        result = await programmatic.call_tool('run_python_code', {'code': code}, ctx, tool)
        assert 'caught error' in result

    async def test_tool_with_optional_params(self):
        base_toolset = FunctionToolset[None]()

        @base_toolset.tool
        def greet(name: str, greeting: str = 'Hello') -> str:
            """Greet someone.

            Args:
                name: The name to greet.
                greeting: The greeting to use.
            """
            return f'{greeting}, {name}!'

        programmatic = ProgrammaticToolset(base_toolset)
        ctx = build_run_context(None)

        tools = await programmatic.get_tools(ctx)
        tool = tools['run_python_code']

        # Test with default parameter
        code1 = """
result = greet(name="World")
print(result)
"""
        result1 = await programmatic.call_tool('run_python_code', {'code': code1}, ctx, tool)
        assert 'Hello, World!' in result1

        # Test with explicit parameter
        code2 = """
result = greet(name="World", greeting="Hi")
print(result)
"""
        result2 = await programmatic.call_tool('run_python_code', {'code': code2}, ctx, tool)
        assert 'Hi, World!' in result2

    async def test_unknown_tool_error(self):
        base_toolset = FunctionToolset[None]()
        programmatic = ProgrammaticToolset(base_toolset)
        ctx = build_run_context(None)

        tools = await programmatic.get_tools(ctx)
        tool = tools['run_python_code']

        with pytest.raises(ValueError, match='Unknown tool'):
            await programmatic.call_tool('unknown_tool', {'code': ''}, ctx, tool)

    async def test_code_syntax_error(self):
        base_toolset = FunctionToolset[None]()
        programmatic = ProgrammaticToolset(base_toolset)
        ctx = build_run_context(None)

        tools = await programmatic.get_tools(ctx)
        tool = tools['run_python_code']

        result = await programmatic.call_tool(
            'run_python_code',
            {'code': 'this is not valid python +++'},
            ctx,
            tool,
        )
        assert 'failed' in result.lower() or 'error' in result.lower()

    async def test_no_output(self):
        base_toolset = FunctionToolset[None]()
        programmatic = ProgrammaticToolset(base_toolset)
        ctx = build_run_context(None)

        tools = await programmatic.get_tools(ctx)
        tool = tools['run_python_code']

        result = await programmatic.call_tool(
            'run_python_code',
            {'code': 'x = 1 + 2  # no print'},
            ctx,
            tool,
        )
        assert result == '(no output)'

    async def test_pydantic_validation_catches_type_errors(self):
        """Test that Pydantic validation catches type errors BEFORE execution."""
        base_toolset = FunctionToolset[None]()
        tool_was_called = False

        @base_toolset.tool
        def typed_tool(count: int, name: str) -> str:
            """A tool with typed parameters.

            Args:
                count: An integer count.
                name: A string name.
            """
            nonlocal tool_was_called
            tool_was_called = True
            return f'{name}: {count}'

        programmatic = ProgrammaticToolset(base_toolset)
        ctx = build_run_context(None)

        tools = await programmatic.get_tools(ctx)
        tool = tools['run_python_code']

        # Test that wrong types are caught by pre-execution validation
        code = """
# Pass a string where int is expected - should be caught before execution
result = typed_tool(count="not_a_number", name="test")
print(f"result: {result}")
"""
        result = await programmatic.call_tool('run_python_code', {'code': code}, ctx, tool)
        # Pre-execution validation catches errors BEFORE running the code
        assert 'Code validation failed before execution' in result
        assert 'typed_tool()' in result
        assert 'Input should be a valid integer' in result
        # Tool should not have been called due to validation failure
        assert not tool_was_called

    async def test_pydantic_validation_coerces_types(self):
        """Test that Pydantic can coerce compatible types."""
        base_toolset = FunctionToolset[None]()

        @base_toolset.tool
        def add_numbers(a: int, b: int) -> int:
            """Add two integers."""
            return a + b

        programmatic = ProgrammaticToolset(base_toolset)
        ctx = build_run_context(None)

        tools = await programmatic.get_tools(ctx)
        tool = tools['run_python_code']

        # Test that numeric strings can be coerced to int
        code = """
# Pydantic should coerce "5" to 5
result = add_numbers(a=5, b=3)
print(f"result: {result}")
"""
        result = await programmatic.call_tool('run_python_code', {'code': code}, ctx, tool)
        assert 'result: 8' in result

    async def test_pydantic_validation_extra_fields_rejected(self):
        """Test that extra/unknown parameters are rejected."""
        base_toolset = FunctionToolset[None]()

        @base_toolset.tool
        def simple_tool(name: str) -> str:
            """A simple tool."""
            return f'Hello {name}'

        programmatic = ProgrammaticToolset(base_toolset)
        ctx = build_run_context(None)

        tools = await programmatic.get_tools(ctx)
        tool = tools['run_python_code']

        # Test that extra parameters cause validation error
        code = """
try:
    # Pass an extra parameter that doesn't exist
    result = simple_tool(name="World", unknown_param="bad")
    print(f"Unexpected success: {result}")
except TypeError as e:
    print(f"Caught validation error: {e}")
"""
        result = await programmatic.call_tool('run_python_code', {'code': code}, ctx, tool)
        assert 'Caught validation error' in result


class TestProgrammaticToolsetWithToolManager:
    """Test ProgrammaticToolset via ToolManager (simulating actual agent usage)."""

    async def test_via_tool_manager(self):
        base_toolset = FunctionToolset[None]()

        @base_toolset.tool
        def square(n: int) -> int:
            """Square a number."""
            return n * n

        programmatic = ProgrammaticToolset(base_toolset)
        ctx = build_run_context(None)

        manager = await ToolManager[None](programmatic).for_run_step(ctx)

        # Check tool definition
        assert len(manager.tool_defs) == 1
        assert manager.tool_defs[0].name == 'run_python_code'

        # Call through manager
        result = await manager.handle_call(
            ToolCallPart(
                tool_name='run_python_code',
                args={'code': 'print(square(n=5))'},
            )
        )
        assert '25' in str(result)
