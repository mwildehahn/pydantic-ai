# Native Provider Tool Search — Implementation Plan

## Goal

Add native Anthropic and OpenAI tool search support to pydantic-ai so that `defer_loading=True` tools use the provider's built-in tool search rather than the client-side `search_tools` synthetic tool. This gives us server-side caching, better relevance ranking, and lower latency for tool discovery.

The client-side `ToolSearchToolset` / `search_tools` remains as a portable fallback for providers without native support.

## Provider API Reference

### Anthropic

**SDK types** (requires `anthropic >= 0.86.0`, current fork pins `0.80.0`):

- **Tool search tool types** (added to the `tools` array alongside regular tools):
  - `ToolSearchToolBm25_20251119Param`: BM25 search over tool names/descriptions
    ```python
    {"type": "tool_search_tool_bm25_20251119", "name": "tool_search_tool_bm25"}
    ```
  - `ToolSearchToolRegex20251119Param`: Regex search over tool names/descriptions
    ```python
    {"type": "tool_search_tool_regex_20251119", "name": "tool_search_tool_regex"}
    ```

- **`defer_loading` on regular tools**: Anthropic's `ToolParam` already has a `defer_loading: bool` field. When `True`, the tool is excluded from the initial system prompt and only loaded when returned via `tool_reference` from tool search. This hides **everything** — name, description, and schema.

- **Response blocks**:
  - `ServerToolUseBlock` with `name` in `["tool_search_tool_regex", "tool_search_tool_bm25"]` — the model searched tools
  - `ToolSearchToolResultBlock` containing `ToolSearchToolSearchResultBlock` with `tool_references: list[ToolReferenceBlock]` — each has `tool_name: str`

- **Message history**: Uses the existing `server_tool_use` / result pattern (same as web_search, code_execution)

### OpenAI

**SDK types** (requires `openai >= 2.25.0`):

- **Tool search tool** (added to the `tools` array):
  ```python
  {"type": "tool_search"}
  ```
  Optional fields: `execution` (`"server"` | `"client"`), `description`, `parameters`

- **`defer_loading` on regular tools**: OpenAI's approach is different — tool search only defers the **schema**. Tool names and descriptions are always visible to the model. The `FunctionToolParam` does NOT have a `defer_loading` field; instead, the presence of `{"type": "tool_search"}` in tools triggers deferred schema loading.

  **Update**: Looking at the SDK, `FunctionToolParam` doesn't have `defer_loading`. The OpenAI tool search works by: (1) adding `{"type": "tool_search"}` to tools, (2) all function tools with schemas are candidates for deferred loading — the model sees names/descriptions but schemas are loaded on demand by the server.

- **Response items**:
  - `ResponseToolSearchCall`: `{"type": "tool_search_call", "execution": "server"|"client", "status": "..."}`
  - `ResponseToolSearchOutputItem`: `{"type": "tool_search_output", "tools": [Tool, ...]}` — returns full tool definitions

## Current Architecture

### How tools flow today

```
Tool(defer_loading=True)
  → ToolDefinition(defer_loading=True)
    → ToolSearchToolset.get_tools()
      → Separates deferred vs visible
      → Injects synthetic `search_tools` tool
      → Model calls `search_tools` → discovers tools → they become visible
    → Model._map_tool_definition(td)
      → Anthropic: BetaToolParam (ignores defer_loading)
      → OpenAI: ChatCompletionToolParam (ignores defer_loading)
```

**Key point**: `defer_loading` is currently consumed only by `ToolSearchToolset` (client-side). The model adapters ignore it completely.

### Existing patterns to follow

**Builtin tools** (web_search, code_execution, web_fetch):
1. Defined as `AbstractBuiltinTool` subclasses in `builtin_tools.py`
2. Added in `_add_builtin_tools()` per model adapter
3. Response blocks mapped to `BuiltinToolCallPart` / `BuiltinToolReturnPart`
4. History replay: `BuiltinToolCallPart` with matching `provider_name` → reconstructed as `BetaServerToolUseBlockParam`; mismatched provider → silently skipped

**`ServerToolUseBlock` handling** already exists for web_search/code_execution/web_fetch — tool_search is the same pattern.

## Tasks

### Anthropic
- [x] 1. Add `ToolSearchTool` builtin to `builtin_tools.py`
- [x] 2. Bump anthropic SDK to `>= 0.86.0` in `pyproject.toml`
- [x] 3. `_map_tool_definition`: pass `defer_loading` through to `BetaToolParam`
- [x] 4. `_add_builtin_tools`: handle `ToolSearchTool` → add tool_search_tool_bm25/regex
- [x] 5. `_map_server_tool_use_block`: handle `tool_search_tool_*` names
- [x] 6. `_process_response`: handle `ToolSearchToolResultBlock`
- [x] 7. Streaming: handle tool_search blocks in `_get_event_iterator`
- [x] 8. Message history replay: reconstruct tool_search blocks from `BuiltinToolCallPart`/`BuiltinToolReturnPart`
- [x] 9. `supported_builtin_tools`: add `ToolSearchTool`

### OpenAI (follow-up)
- [ ] 10. `_get_builtin_tools`: add `tool_search` when `ToolSearchTool` present
- [ ] 11. Response parsing: `ResponseToolSearchCall` / `ResponseToolSearchOutputItem`
- [ ] 12. Message history replay
- [ ] 13. Streaming

### Cross-cutting
- [ ] 14. Model profiles: `supports_native_tool_search` capability flag
- [ ] 15. `ToolSearchToolset`: native mode pass-through

---

## Implementation Details

### Step 1: Add `ToolSearchTool` builtin

**File**: `pydantic_ai_slim/pydantic_ai/builtin_tools.py`

```python
class ToolSearchTool(AbstractBuiltinTool):
    """Native provider tool search for deferred tools."""
    kind: ClassVar[Literal['tool_search']] = 'tool_search'

    search_type: Literal['bm25', 'regex'] = 'bm25'
    """Anthropic search algorithm. Ignored by OpenAI."""
```

### Step 2: Anthropic adapter changes

**File**: `pydantic_ai_slim/pydantic_ai/models/anthropic.py`

#### 2a. `_map_tool_definition` — pass through `defer_loading`

```python
def _map_tool_definition(self, f: ToolDefinition, model_settings: AnthropicModelSettings) -> BetaToolParam:
    tool_param: BetaToolParam = {
        'name': f.name,
        'description': f.description or '',
        'input_schema': f.parameters_json_schema,
    }
    if f.defer_loading:
        tool_param['defer_loading'] = True
    if f.strict and self.profile.supports_json_schema_output:
        tool_param['strict'] = f.strict
    if model_settings.get('anthropic_eager_input_streaming'):
        tool_param['eager_input_streaming'] = True
    return tool_param
```

#### 2b. `_add_builtin_tools` — add tool search tool when deferred tools exist

```python
# In _add_builtin_tools, after existing builtin tool handling:
elif isinstance(tool, ToolSearchTool):
    if tool.search_type == 'regex':
        tools.append({
            'type': 'tool_search_tool_regex_20251119',
            'name': 'tool_search_tool_regex',
        })
    else:
        tools.append({
            'type': 'tool_search_tool_bm25_20251119',
            'name': 'tool_search_tool_bm25',
        })
```

#### 2c. `_process_response` / streaming — handle tool_search result blocks

In `_map_server_tool_use_block`:
```python
elif item.name in ('tool_search_tool_regex', 'tool_search_tool_bm25'):
    return BuiltinToolCallPart(
        provider_name=provider_name,
        tool_name=ToolSearchTool.kind,
        args=tool_args,
        tool_call_id=item.id,
    )
```

Add new handler for `ToolSearchToolResultBlock` (same pattern as `_map_web_search_tool_result_block`):
```python
elif isinstance(item, ToolSearchToolResultBlock):
    items.append(_map_tool_search_result_block(item, self.system))
```

```python
def _map_tool_search_result_block(item: ToolSearchToolResultBlock, provider_name: str) -> BuiltinToolReturnPart:
    return BuiltinToolReturnPart(
        provider_name=provider_name,
        tool_name=ToolSearchTool.kind,
        content=item.model_dump(mode='json'),
        tool_call_id=item.tool_use_id,
    )
```

#### 2d. Message history replay — reconstruct tool_search blocks

In the `BuiltinToolCallPart` branch of `_map_message`:
```python
elif response_part.tool_name == ToolSearchTool.kind:
    server_tool_use_block_param = BetaServerToolUseBlockParam(
        id=tool_use_id,
        type='server_tool_use',
        name='tool_search_tool_bm25',  # default; could store original in provider_details
        input=response_part.args_as_dict(),
    )
    assistant_content_params.append(server_tool_use_block_param)
```

In the `BuiltinToolReturnPart` branch, add `ToolSearchTool.kind` to the set of known tool names that map to Anthropic result blocks.

#### 2e. SDK bump

In `pyproject.toml`, bump: `anthropic >= 0.86.0` (currently `0.80.0`)

### Step 3: OpenAI Responses adapter changes

**File**: `pydantic_ai_slim/pydantic_ai/models/openai.py`

#### 3a. `_get_builtin_tools` — add tool_search

```python
elif isinstance(tool, ToolSearchTool):
    tools.append(responses.ToolSearchToolParam(type='tool_search'))
```

Note: OpenAI doesn't use `defer_loading` on individual function tools. When `tool_search` is present, the server handles deferred schema loading automatically. We still pass all tool definitions (with names/descriptions) — the server decides what to defer.

#### 3b. Response handling — parse `ResponseToolSearchCall` and `ResponseToolSearchOutputItem`

In `_process_response` (for Responses API):
```python
elif isinstance(item, ResponseToolSearchCall):
    parts.append(BuiltinToolCallPart(
        provider_name=self._provider.name,
        tool_name=ToolSearchTool.kind,
        args={'arguments': item.arguments},
        tool_call_id=item.call_id or item.id,
    ))
elif isinstance(item, ResponseToolSearchOutputItem):
    parts.append(BuiltinToolReturnPart(
        provider_name=self._provider.name,
        tool_name=ToolSearchTool.kind,
        content={'tools': [t.model_dump(mode='json') for t in item.tools]},
        tool_call_id=item.call_id or item.id,
    ))
```

Same for streaming.

#### 3c. Message history replay

In `_map_response_to_input_items`, add handling for `BuiltinToolCallPart`/`BuiltinToolReturnPart` with `tool_name == ToolSearchTool.kind` — reconstruct as `ResponseToolSearchCall` / `ResponseToolSearchOutputItem` input params.

### Step 4: Auto-inject `ToolSearchTool` when deferred tools exist

**Option A**: In the agent graph, detect deferred tools and auto-add `ToolSearchTool` to `builtin_tools`.

**Option B** (simpler): User explicitly adds `ToolSearchTool()` to `builtin_tools`, like they do with `WebSearchTool()`.

**Recommendation**: Option B for the initial implementation. Explicit is better — the user opts into native tool search per-agent. The `ToolSearchToolset` (client-side) continues to work automatically as fallback.

### Step 5: Model profiles — capability flag

```python
# In profiles:
supports_native_tool_search: bool = False
```

Set to `True` for Anthropic Sonnet 4.0+, Opus 4.0+, and OpenAI gpt-5.4+.

When the profile doesn't support native tool search but `ToolSearchTool` is in `builtin_tools`, fall back to the client-side `ToolSearchToolset`.

### Step 6: `ToolSearchToolset` interaction

When native tool search is active AND `ToolSearchToolset` is wrapping the toolset:

- `ToolSearchToolset` still separates deferred tools from visible ones
- But instead of injecting `search_tools` synthetic tool, it passes deferred tools through with `defer_loading=True` on the `ToolDefinition`
- The model adapter picks up `defer_loading` and handles it natively
- `search_tools` synthetic tool is NOT added (native search replaces it)

**Detection**: `ToolSearchToolset.get_tools()` checks if `ToolSearchTool` is in the active builtin tools. If yes, it returns all tools (deferred ones keep `defer_loading=True`) and skips the synthetic `search_tools` injection.

This requires `ToolSearchToolset` to have visibility into `model_request_parameters.builtin_tools`. Currently it doesn't — it only sees `RunContext`. We may need to pass this through or check the model profile.

**Simpler alternative**: Add a flag to `ToolSearchToolset` constructor: `native_mode=False`. When `True`, deferred tools pass through with `defer_loading=True` instead of being hidden and replaced by `search_tools`.

## File Change Summary

| File | Changes |
|------|---------|
| `builtin_tools.py` | Add `ToolSearchTool` class |
| `models/anthropic.py` | `_map_tool_definition`: pass `defer_loading`; `_add_builtin_tools`: handle `ToolSearchTool`; response parsing: `ToolSearchToolResultBlock`; message history: reconstruct tool_search blocks |
| `models/openai.py` | `_get_builtin_tools`: add `tool_search`; response parsing: `ResponseToolSearchCall`/`ResponseToolSearchOutputItem`; message history replay |
| `profiles/__init__.py` | Add `supports_native_tool_search` flag |
| `profiles/anthropic.py` | Set flag for Sonnet 4.0+, Opus 4.0+ |
| `profiles/openai.py` | Set flag for gpt-5.4+ |
| `toolsets/_tool_search.py` | Handle native mode (skip synthetic `search_tools` when native is active) |
| `pyproject.toml` | Bump `anthropic >= 0.86.0` |

## Verification

1. **Unit tests**: Mock Anthropic/OpenAI responses with tool_search blocks, verify correct parsing to `BuiltinToolCallPart`/`BuiltinToolReturnPart`
2. **Integration test**: Agent with `defer_loading=True` tools + `ToolSearchTool()` builtin, verify:
   - Anthropic: tools sent with `defer_loading: true`, tool_search_tool_bm25 in tools array
   - OpenAI: `tool_search` in tools array
   - Response parsing produces correct message parts
   - History replay reconstructs correct provider blocks
3. **Fallback test**: Same agent on a provider without native support → falls back to `search_tools` synthetic tool
4. **Provider switch test**: Start on Anthropic (native), switch to OpenAI — verify `BuiltinToolCallPart` with `provider_name='anthropic'` is skipped, OpenAI can rediscover tools via its own search
