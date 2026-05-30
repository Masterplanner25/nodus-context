"""nodus-context — LLM context window management.

Token budget tracking, message lifecycle, and compaction strategies for AI systems
that need to stay within finite LLM context windows.

Core types:
    ContextMessage       — one message (role, content, token_count)
    ContextBudget        — token budget (max, usage, utilization, overflow check)
    ContextWindow        — message list with budget enforcement + compaction

Token counting:
    estimate_tokens      — word-count estimate (~4 chars per token, no dep)
    count_tokens         — accurate via tiktoken if installed, else estimate

Compaction strategies:
    CompactionStrategy   — protocol any strategy must satisfy
    DropToolInternalsStrategy — remove tool_use/tool_result except final pairs
    SummarizeStrategy    — keep head + tail, replace middle with summary

Role constants:
    ROLE_USER, ROLE_ASSISTANT, ROLE_SYSTEM,
    ROLE_TOOL_USE, ROLE_TOOL_RESULT, ROLE_THINKING
"""
from .budget import ContextBudget, count_tokens, estimate_tokens
from .message import (
    ROLE_ASSISTANT,
    ROLE_SYSTEM,
    ROLE_THINKING,
    ROLE_TOOL_RESULT,
    ROLE_TOOL_USE,
    ROLE_USER,
    ContextMessage,
)
from .strategies import CompactionStrategy, DropToolInternalsStrategy, SummarizeStrategy
from .window import ContextWindow

__all__ = [
    # Core types
    "ContextMessage",
    "ContextBudget",
    "ContextWindow",
    # Token counting
    "estimate_tokens",
    "count_tokens",
    # Strategies
    "CompactionStrategy",
    "DropToolInternalsStrategy",
    "SummarizeStrategy",
    # Role constants
    "ROLE_USER",
    "ROLE_ASSISTANT",
    "ROLE_SYSTEM",
    "ROLE_TOOL_USE",
    "ROLE_TOOL_RESULT",
    "ROLE_THINKING",
]
