"""Programmatic Tool Calling - execute tools from Python code in a sandbox.

This module provides a provider-agnostic implementation of Programmatic Tool Calling,
allowing any model to write Python code that orchestrates multiple tool calls.
Tool results stay in the sandbox and don't pollute the model's context - only the
final output is returned.

Example:
    ```python
    from pydantic_ai import Agent
    from pydantic_ai.toolsets import FunctionToolset, ProgrammaticToolset

    # Create toolset with multiple tools
    toolset = FunctionToolset()

    @toolset.tool
    async def get_expenses(user_id: str, quarter: str) -> list[dict]:
        ...

    @toolset.tool
    async def get_team_members(department: str) -> list[dict]:
        ...

    # Wrap for programmatic calling
    programmatic = ProgrammaticToolset(toolset)

    agent = Agent('openai:gpt-4o', toolsets=[programmatic])

    # The model can now write Python code that calls tools
    result = await agent.run(
        "Which team members exceeded their Q3 travel budget?"
    )
    ```
"""

from __future__ import annotations

import ast
import asyncio
import json
import sys
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError, create_model
from typing_extensions import TypedDict

from .._run_context import AgentDepsT, RunContext
from ..tools import ToolDefinition
from .abstract import AbstractToolset, ToolsetTool

__all__ = ('ProgrammaticToolset', 'CodeSandbox', 'SandboxResult')


# Protocol marker for tool calls from sandbox
_TOOL_CALL_MARKER = '__PYDANTIC_AI_TOOL_CALL__:'
_TOOL_RESULT_MARKER = '__PYDANTIC_AI_TOOL_RESULT__:'


@dataclass
class SandboxResult:
    """Result from executing code in a sandbox."""

    stdout: str
    """Standard output from the code execution."""

    stderr: str
    """Standard error from the code execution."""

    returncode: int
    """Return code from the subprocess."""

    error: str | None = None
    """Error message if execution failed."""


@dataclass
class CodeSandbox:
    """Executes Python code in an isolated subprocess with tool call support.

    The sandbox runs Python code in a subprocess and intercepts tool calls
    made via a special protocol. Tool calls are executed by the parent process
    and results are sent back to the sandbox.
    """

    timeout: float = 300.0
    """Maximum execution time in seconds."""

    python_executable: str | None = None
    """Path to Python executable. Uses sys.executable if not specified."""

    async def execute(
        self,
        code: str,
        tool_handler: Callable[[str, dict[str, Any]], Awaitable[Any]],
    ) -> SandboxResult:
        """Execute Python code in an isolated subprocess.

        Args:
            code: The Python code to execute.
            tool_handler: Async function to handle tool calls from the sandbox.
                Takes tool name and arguments, returns the tool result.

        Returns:
            SandboxResult with stdout, stderr, and return code.
        """
        python_exe = self.python_executable or sys.executable

        # Wrap code in async runner
        wrapped_code = self._wrap_code(code)

        try:
            proc = await asyncio.create_subprocess_exec(
                python_exe,
                '-c',
                wrapped_code,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout_parts: list[str] = []
            stderr_parts: list[str] = []

            async def read_stderr():
                """Read stderr and handle tool calls."""
                assert proc.stderr is not None
                assert proc.stdin is not None

                while True:
                    line = await proc.stderr.readline()
                    if not line:
                        break

                    line_str = line.decode('utf-8', errors='replace')

                    if line_str.startswith(_TOOL_CALL_MARKER):
                        # Parse and handle tool call
                        try:
                            request_json = line_str[len(_TOOL_CALL_MARKER) :].strip()
                            request = json.loads(request_json)
                            tool_name = request['name']
                            tool_args = request['args']

                            # Execute the tool
                            try:
                                result = await tool_handler(tool_name, tool_args)
                                response = {'value': result}
                            except Exception as e:
                                response = {'error': str(e)}

                            # Send result back to sandbox
                            response_line = json.dumps(response) + '\n'
                            proc.stdin.write(response_line.encode('utf-8'))
                            await proc.stdin.drain()

                        except Exception as e:
                            # Send error back to sandbox
                            response = {'error': f'Tool call failed: {e}'}
                            response_line = json.dumps(response) + '\n'
                            proc.stdin.write(response_line.encode('utf-8'))
                            await proc.stdin.drain()
                    else:
                        stderr_parts.append(line_str)

            async def read_stdout():
                """Read stdout."""
                assert proc.stdout is not None
                while True:
                    line = await proc.stdout.readline()
                    if not line:
                        break
                    stdout_parts.append(line.decode('utf-8', errors='replace'))

            # Run readers concurrently with timeout
            try:
                await asyncio.wait_for(
                    asyncio.gather(read_stderr(), read_stdout()),
                    timeout=self.timeout,
                )
            except TimeoutError:
                proc.kill()
                return SandboxResult(
                    stdout=''.join(stdout_parts),
                    stderr=''.join(stderr_parts),
                    returncode=-1,
                    error=f'Execution timed out after {self.timeout} seconds',
                )

            await proc.wait()

            return SandboxResult(
                stdout=''.join(stdout_parts),
                stderr=''.join(stderr_parts),
                returncode=proc.returncode or 0,
            )

        except Exception as e:
            return SandboxResult(
                stdout='',
                stderr='',
                returncode=-1,
                error=f'Failed to execute code: {e}',
            )

    def _wrap_code(self, code: str) -> str:
        """Wrap user code with async runner."""
        return f'''
import asyncio
import json
import sys

# Tool call protocol
_TOOL_CALL_MARKER = {_TOOL_CALL_MARKER!r}

def __call_tool__(name: str, args: dict):
    """Call a tool and wait for result."""
    request = json.dumps({{"name": name, "args": args}})
    print(f"{{_TOOL_CALL_MARKER}}{{request}}", file=sys.stderr, flush=True)

    # Read result from parent
    result_line = sys.stdin.readline()
    if not result_line:
        raise RuntimeError("Lost connection to tool executor")

    result = json.loads(result_line)
    if "error" in result:
        raise RuntimeError(result["error"])
    return result["value"]

# User code
{code}
'''


def _json_schema_to_pydantic_field(param_name: str, param_schema: dict[str, Any], is_required: bool) -> str:
    """Generate a Pydantic field definition from JSON schema."""
    json_type = param_schema.get('type', 'any')
    py_type = _json_type_to_python(json_type)

    # Handle defaults
    if is_required:
        return f'{param_name}: {py_type}'
    else:
        default = param_schema.get('default')
        if default is None:
            return f'{param_name}: {py_type} | None = None'
        elif isinstance(default, str):
            return f'{param_name}: {py_type} = {default!r}'
        else:
            return f'{param_name}: {py_type} = {default!r}'


def _generate_tool_function(name: str, tool_def: ToolDefinition) -> str:
    """Generate a Python function definition for a tool with Pydantic validation."""
    # Extract parameters from JSON schema
    schema = tool_def.parameters_json_schema
    properties = schema.get('properties', {})
    required = set(schema.get('required', []))

    # Generate Pydantic model for validation
    model_name = f'__{name}_Args__'
    model_fields = []
    for param_name, param_schema in properties.items():
        is_required = param_name in required
        field_def = _json_schema_to_pydantic_field(param_name, param_schema, is_required)
        model_fields.append(f'    {field_def}')

    model_fields_str = '\n'.join(model_fields) if model_fields else '    pass'

    # Build parameter list for the function
    params: list[str] = []
    for param_name, param_schema in properties.items():
        param_type = _json_type_to_python(param_schema.get('type', 'any'))
        if param_name in required:
            params.append(f'{param_name}: {param_type}')
        else:
            default = param_schema.get('default')
            if default is None:
                params.append(f'{param_name}: {param_type} | None = None')
            elif isinstance(default, str):
                params.append(f'{param_name}: {param_type} = {default!r}')
            else:
                params.append(f'{param_name}: {param_type} = {default!r}')

    params_str = ', '.join(params)

    # Build docstring
    description = tool_def.description or f'Call the {name} tool.'
    # Escape triple quotes in description
    safe_description = description.replace('"""', '\\"\\"\\"')
    docstring = f'"""{safe_description}"""'

    # Build args dict - collect all parameters
    args_items = [f"'{p}': {p}" for p in properties.keys()]
    args_dict = '{' + ', '.join(args_items) + '}'

    return f"""
class {model_name}(BaseModel):
    model_config = ConfigDict(extra='forbid')
{model_fields_str}

def {name}({params_str}):
    {docstring}
    args = {args_dict}
    # Filter out None values for optional parameters
    args = {{k: v for k, v in args.items() if v is not None}}
    # Validate arguments using Pydantic
    try:
        validated = {model_name}(**args)
        args = validated.model_dump(exclude_none=True)
    except ValidationError as e:
        raise TypeError(f"Invalid arguments for {name!r}: {{e}}")
    return __call_tool__({name!r}, args)
"""


def _json_type_to_python(json_type: str | list[str]) -> str:
    """Convert JSON schema type to Python type hint."""
    if isinstance(json_type, list):
        # Union type
        types = [_json_type_to_python(t) for t in json_type if t != 'null']
        if 'null' in json_type:
            types.append('None')
        return ' | '.join(types) if types else 'Any'

    type_map = {
        'string': 'str',
        'integer': 'int',
        'number': 'float',
        'boolean': 'bool',
        'array': 'list',
        'object': 'dict',
        'null': 'None',
    }
    return type_map.get(json_type, 'Any')


def _generate_sdk(tools: dict[str, ToolsetTool[Any]]) -> str:
    """Generate the complete SDK code for available tools with Pydantic validation."""
    lines = [
        '# Auto-generated tool SDK with Pydantic validation',
        'from pydantic import BaseModel, ConfigDict, ValidationError',
        '',
        '# Available tools:',
    ]

    for name, tool in tools.items():
        desc = (tool.tool_def.description or 'No description').split('\n')[0]
        lines.append(f'#   - {name}: {desc}')

    lines.append('')

    for name, tool in tools.items():
        lines.append(_generate_tool_function(name, tool.tool_def))

    return '\n'.join(lines)


@dataclass
class CodeValidationError:
    """Error from pre-execution code validation."""

    line: int
    """Line number in the code (1-indexed)."""

    column: int
    """Column number in the code (0-indexed)."""

    tool_name: str
    """Name of the tool that has invalid arguments."""

    message: str
    """Description of the validation error."""

    def __str__(self) -> str:
        return f'Line {self.line}, col {self.column}: {self.tool_name}() - {self.message}'


def _json_type_to_pydantic_type(json_type: str | list[str]) -> type:
    """Convert JSON schema type to Python type for Pydantic model creation."""
    if isinstance(json_type, list):
        # For union types, use the first non-null type or Any
        for t in json_type:
            if t != 'null':
                return _json_type_to_pydantic_type(t)
        return type(None)

    type_map: dict[str, type] = {
        'string': str,
        'integer': int,
        'number': float,
        'boolean': bool,
        'array': list,
        'object': dict,
        'null': type(None),
    }
    return type_map.get(json_type, object)


def _create_validator_model(tool_name: str, tool_def: ToolDefinition) -> type[BaseModel]:
    """Create a Pydantic model for validating tool arguments."""
    schema = tool_def.parameters_json_schema
    properties = schema.get('properties', {})
    required = set(schema.get('required', []))

    # Build field definitions for create_model
    field_definitions: dict[str, Any] = {}
    for param_name, param_schema in properties.items():
        param_type = _json_type_to_pydantic_type(param_schema.get('type', 'object'))
        if param_name in required:
            field_definitions[param_name] = (param_type, ...)
        else:
            default = param_schema.get('default')
            field_definitions[param_name] = (param_type | None, default)

    # Create model with extra='forbid' to catch unknown arguments
    return create_model(
        f'__{tool_name}_Validator__',
        __config__=ConfigDict(extra='forbid'),
        **field_definitions,
    )


def _ast_value_to_python(node: ast.expr) -> Any:
    """Try to convert an AST node to a Python value for validation.

    Returns the value if it can be statically determined, otherwise returns a
    sentinel object indicating the value is dynamic.
    """

    class _DynamicValue:
        """Sentinel for values that can't be determined statically."""

        pass

    DYNAMIC = _DynamicValue()

    if isinstance(node, ast.Constant):
        return node.value
    elif isinstance(node, ast.List):
        items = [_ast_value_to_python(elt) for elt in node.elts]
        if any(isinstance(item, _DynamicValue) for item in items):
            return DYNAMIC
        return items
    elif isinstance(node, ast.Dict):
        keys = []
        values = []
        for k, v in zip(node.keys, node.values):
            if k is None:  # **kwargs spread
                return DYNAMIC
            key_val = _ast_value_to_python(k)
            val_val = _ast_value_to_python(v)
            if isinstance(key_val, _DynamicValue) or isinstance(val_val, _DynamicValue):
                return DYNAMIC
            keys.append(key_val)
            values.append(val_val)
        return dict(zip(keys, values))
    elif isinstance(node, ast.Tuple):
        items = [_ast_value_to_python(elt) for elt in node.elts]
        if any(isinstance(item, _DynamicValue) for item in items):
            return DYNAMIC
        return tuple(items)
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        # Handle negative numbers like -1
        operand = _ast_value_to_python(node.operand)
        if isinstance(operand, _DynamicValue):
            return DYNAMIC
        return -operand
    else:
        # Variable reference, function call, etc. - can't determine statically
        return DYNAMIC

    return DYNAMIC


class _DynamicValue:
    """Sentinel for values that can't be determined statically."""

    pass


def validate_code_tool_calls(
    code: str,
    tools: dict[str, ToolsetTool[Any]],
) -> list[CodeValidationError]:
    """Validate tool calls in code before execution.

    Parses the code using AST and validates any calls to known tools
    against their schemas. Returns a list of validation errors.

    This allows catching type errors, missing arguments, and unknown
    arguments BEFORE running the code.

    Args:
        code: The Python code to validate.
        tools: Dictionary of available tools.

    Returns:
        List of validation errors. Empty list if code is valid.
    """
    errors: list[CodeValidationError] = []

    # Parse the code
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [
            CodeValidationError(
                line=e.lineno or 1,
                column=e.offset or 0,
                tool_name='<syntax>',
                message=f'Syntax error: {e.msg}',
            )
        ]

    # Build validators for each tool
    validators: dict[str, type[BaseModel]] = {}
    for name, tool in tools.items():
        validators[name] = _create_validator_model(name, tool.tool_def)

    # Walk the AST looking for function calls
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            # Check if this is a call to one of our tools
            func_name = None
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                # Handle method calls like obj.method() - skip these
                continue

            if func_name not in validators:
                # Not a tool call, skip
                continue

            validator = validators[func_name]
            tool = tools[func_name]
            schema = tool.tool_def.parameters_json_schema
            properties = schema.get('properties', {})
            required = set(schema.get('required', []))
            param_names = list(properties.keys())

            # Build arguments dict from the call
            args_dict: dict[str, Any] = {}
            has_dynamic_args = False

            # Handle positional arguments
            for i, arg in enumerate(node.args):
                if i < len(param_names):
                    value = _ast_value_to_python(arg)
                    if isinstance(value, _DynamicValue):
                        has_dynamic_args = True
                    else:
                        args_dict[param_names[i]] = value
                else:
                    # Too many positional arguments
                    errors.append(
                        CodeValidationError(
                            line=node.lineno,
                            column=node.col_offset,
                            tool_name=func_name,
                            message=f'Too many positional arguments. Expected at most {len(param_names)}, got {len(node.args)}',
                        )
                    )
                    break

            # Handle keyword arguments
            for kw in node.keywords:
                if kw.arg is None:
                    # **kwargs - can't validate statically
                    has_dynamic_args = True
                    continue

                value = _ast_value_to_python(kw.value)
                if isinstance(value, _DynamicValue):
                    has_dynamic_args = True
                else:
                    if kw.arg in args_dict:
                        errors.append(
                            CodeValidationError(
                                line=node.lineno,
                                column=node.col_offset,
                                tool_name=func_name,
                                message=f"Duplicate argument: '{kw.arg}'",
                            )
                        )
                    args_dict[kw.arg] = value

            # Skip validation if we have dynamic arguments we can't check
            if has_dynamic_args:
                # Still check for required arguments that we DO have
                for req_arg in required:
                    if req_arg not in args_dict:
                        # Can't tell if it's provided dynamically, skip
                        pass
                continue

            # Validate using Pydantic
            try:
                validator(**args_dict)
            except ValidationError as e:
                for error in e.errors():
                    loc = '.'.join(str(x) for x in error['loc'])
                    msg = error['msg']
                    errors.append(
                        CodeValidationError(
                            line=node.lineno,
                            column=node.col_offset,
                            tool_name=func_name,
                            message=f"Invalid argument '{loc}': {msg}",
                        )
                    )

    return errors


class _CodeArgs(TypedDict):
    """Arguments for the run_python_code tool."""

    code: str


# Validator for the run_python_code tool arguments (using public Pydantic API)
_code_validator = TypeAdapter(_CodeArgs).validator


@dataclass
class ProgrammaticToolset(AbstractToolset[AgentDepsT]):
    """A toolset that enables programmatic tool calling via Python code execution.

    This toolset wraps other tools and exposes a single `run_python_code` tool
    to the model. When the model calls this tool with Python code, the code is
    executed in an isolated sandbox with access to all wrapped tools as Python
    functions.

    The key benefits:
    - **Reduced context usage**: Intermediate tool results stay in the sandbox
      and don't pollute the model's context window
    - **Efficient orchestration**: Multiple tool calls can be made in a single
      model turn, with loops, conditionals, and data transformations
    - **Provider agnostic**: Works with any model provider (OpenAI, Anthropic,
      Google, etc.)

    Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_ai.toolsets import FunctionToolset, ProgrammaticToolset

        toolset = FunctionToolset()

        @toolset.tool
        async def get_user(user_id: int) -> dict:
            return {"id": user_id, "name": f"User {user_id}"}

        @toolset.tool
        async def get_orders(user_id: int) -> list[dict]:
            return [{"id": 1, "amount": 100}, {"id": 2, "amount": 200}]

        # Wrap for programmatic calling
        programmatic = ProgrammaticToolset(toolset)

        agent = Agent('openai:gpt-4o', toolsets=[programmatic])

        # Model can write code like:
        # user = get_user(123)
        # orders = get_orders(123)
        # total = sum(o['amount'] for o in orders)
        # print(f"User {user['name']} has {len(orders)} orders totaling ${total}")
        ```

    See [programmatic tool calling docs](../programmatic-tools.md) for more information.
    """

    wrapped: AbstractToolset[AgentDepsT]
    """The toolset containing tools to make available in the sandbox."""

    tool_name: str = 'run_python_code'
    """Name of the tool exposed to the model."""

    max_retries: int = 1
    """Maximum retries for the run_python_code tool."""

    sandbox_timeout: float = 300.0
    """Maximum execution time for code in seconds."""

    validate_before_execution: bool = True
    """Whether to validate tool calls before execution using AST analysis.

    When enabled (default), the code is parsed and tool calls are validated
    against their schemas BEFORE execution. This catches errors like:
    - Wrong argument types
    - Missing required arguments
    - Unknown arguments
    - Syntax errors

    This allows returning detailed error messages to the LLM without
    wasting time on execution. Disable for slightly faster execution
    if you only want runtime validation.
    """

    _id: str | None = None
    """Optional unique ID for the toolset."""

    @property
    def id(self) -> str | None:
        return self._id

    @property
    def label(self) -> str:
        return f'ProgrammaticToolset({self.wrapped.label})'

    async def __aenter__(self):
        await self.wrapped.__aenter__()
        return self

    async def __aexit__(self, *args: Any) -> bool | None:
        return await self.wrapped.__aexit__(*args)

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        """Return the run_python_code tool with documentation of available functions."""
        # Get wrapped tools to generate documentation
        wrapped_tools = await self.wrapped.get_tools(ctx)

        # Build description with available functions
        tool_docs = []
        for name, tool in wrapped_tools.items():
            desc = tool.tool_def.description or 'No description'
            # Extract first line of description
            first_line = desc.split('\n')[0].strip()

            # Get parameter info
            schema = tool.tool_def.parameters_json_schema
            properties = schema.get('properties', {})
            required = set(schema.get('required', []))

            params = []
            for param_name, param_schema in properties.items():
                param_type = _json_type_to_python(param_schema.get('type', 'any'))
                is_required = param_name in required
                param_desc = param_schema.get('description', '')
                req_marker = '' if is_required else ' (optional)'
                params.append(f'    - {param_name}: {param_type}{req_marker} - {param_desc}')

            params_str = '\n'.join(params) if params else '    (no parameters)'
            tool_docs.append(f'- {name}({", ".join(properties.keys())}): {first_line}\n{params_str}')

        tools_description = '\n\n'.join(tool_docs)

        description = f"""Execute Python code with access to tools as functions.

The code runs in an isolated environment. Use print() to output results.
Only the final printed output is returned to you - intermediate results
from tool calls stay in the execution environment.

## Available Functions

{tools_description}

## Example Usage

```python
# Get data from multiple sources
users = get_users(department="engineering")
budgets = get_budgets(year=2024)

# Process locally (doesn't use your context)
over_budget = []
for user in users:
    expenses = get_expenses(user_id=user["id"])
    total = sum(e["amount"] for e in expenses)
    if total > budgets[user["level"]]:
        over_budget.append({{"name": user["name"], "spent": total}})

# Only this output is returned to you
import json
print(json.dumps(over_budget, indent=2))
```

Important:
- Use print() to output your final result
- All function calls are synchronous (no await needed)
- Handle errors with try/except if needed
"""

        tool_def = ToolDefinition(
            name=self.tool_name,
            description=description,
            parameters_json_schema={
                'type': 'object',
                'properties': {
                    'code': {
                        'type': 'string',
                        'description': 'Python code to execute. Use print() for output.',
                    }
                },
                'required': ['code'],
            },
        )

        return {
            self.tool_name: ToolsetTool(
                toolset=self,
                tool_def=tool_def,
                max_retries=self.max_retries,
                args_validator=_code_validator,
            )
        }

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[AgentDepsT],
        tool: ToolsetTool[AgentDepsT],
    ) -> Any:
        """Execute Python code in a sandbox with tool access."""
        if name != self.tool_name:
            raise ValueError(f'Unknown tool: {name}')

        code = tool_args['code']

        # Get wrapped tools
        wrapped_tools = await self.wrapped.get_tools(ctx)

        # Pre-execution validation: Check tool calls before running
        if self.validate_before_execution:
            validation_errors = validate_code_tool_calls(code, wrapped_tools)
            if validation_errors:
                error_messages = ['Code validation failed before execution:']
                for err in validation_errors:
                    error_messages.append(f'  - {err}')
                error_messages.append('')
                error_messages.append('Please fix these errors and try again.')
                return '\n'.join(error_messages)

        # Generate SDK
        sdk_code = _generate_sdk(wrapped_tools)

        # Full code to execute
        full_code = sdk_code + '\n\n# User code\n' + code

        # Create sandbox and execute
        sandbox = CodeSandbox(timeout=self.sandbox_timeout)

        async def tool_handler(tool_name: str, args: dict[str, Any]) -> Any:
            """Handle tool calls from the sandbox."""
            if tool_name not in wrapped_tools:
                raise ValueError(f'Unknown tool: {tool_name}')
            wrapped_tool = wrapped_tools[tool_name]
            return await self.wrapped.call_tool(tool_name, args, ctx, wrapped_tool)

        result = await sandbox.execute(full_code, tool_handler)

        if result.error:
            return f'Error: {result.error}'

        if result.returncode != 0:
            # Include stderr for debugging
            error_output = result.stderr.strip() if result.stderr else 'Unknown error'
            return f'Code execution failed (exit code {result.returncode}):\n{error_output}'

        return result.stdout.strip() if result.stdout else '(no output)'

    def apply(self, visitor: Callable[[AbstractToolset[AgentDepsT]], None]) -> None:
        self.wrapped.apply(visitor)

    def visit_and_replace(
        self, visitor: Callable[[AbstractToolset[AgentDepsT]], AbstractToolset[AgentDepsT]]
    ) -> AbstractToolset[AgentDepsT]:
        from dataclasses import replace

        return replace(self, wrapped=self.wrapped.visit_and_replace(visitor))
