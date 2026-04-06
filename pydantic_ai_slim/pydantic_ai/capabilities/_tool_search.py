"""Internal ToolSearch capability that wraps tools with ToolSearchToolset."""

from __future__ import annotations

from dataclasses import dataclass

from .._run_context import AgentDepsT
from ..toolsets import AbstractToolset
from ..toolsets._tool_search import ToolSearchToolset
from .abstract import AbstractCapability, CapabilityOrdering


def has_native_tool_search_builtin(capabilities: list[AbstractCapability[AgentDepsT]]) -> bool:
    """Detect whether the capability list already contains a `ToolSearchTool` builtin.

    Used by auto-injection to flip ``ToolSearch(native_tool_search=True)`` on
    automatically when the user has registered the native ``ToolSearchTool``
    builtin alongside their tools.
    """
    from ..builtin_tools import ToolSearchTool
    from .builtin_tool import BuiltinTool as BuiltinToolCap

    return any(isinstance(c, BuiltinToolCap) and isinstance(c.tool, ToolSearchTool) for c in capabilities)


@dataclass
class ToolSearch(AbstractCapability[AgentDepsT]):
    """Internal capability that wraps tools with ToolSearchToolset for deferred tool discovery.

    Auto-injected when not explicitly provided by the user. Short-circuits
    when no deferred tools exist, so there is zero overhead for agents
    without deferred loading.

    Set ``native_tool_search=True`` to skip the synthetic ``search_tools`` tool
    and let the model provider's native tool search handle discovery. This is
    intended to be paired with ``builtin_tools=[ToolSearchTool()]`` so the
    provider can serve deferred tool definitions on demand without polluting
    the cached prompt prefix.

    Internal for now — will be exported publicly once we add
    user-facing configuration options.
    """

    native_tool_search: bool = False
    """When True, deferred tools pass through with ``defer_loading=True`` preserved
    instead of being hidden behind the synthetic ``search_tools`` tool. The
    provider's native tool search handles discovery."""

    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(position='outermost')

    @classmethod
    def get_serialization_name(cls) -> str | None:
        return None  # not spec-constructible (internal)

    def get_wrapper_toolset(self, toolset: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
        return ToolSearchToolset(wrapped=toolset, native_tool_search=self.native_tool_search)
