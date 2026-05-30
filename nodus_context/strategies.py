"""Compaction strategies for reducing context window token usage."""
from __future__ import annotations

from typing import Callable, Optional
from .message import (
    ROLE_ASSISTANT,
    ROLE_SYSTEM,
    ROLE_THINKING,
    ROLE_TOOL_RESULT,
    ROLE_TOOL_USE,
    ROLE_USER,
    ContextMessage,
)

try:
    from typing import Protocol, runtime_checkable
except ImportError:
    from typing_extensions import Protocol, runtime_checkable  # type: ignore[assignment]


@runtime_checkable
class CompactionStrategy(Protocol):
    """Protocol for context compaction strategies."""

    def compact(self, messages: list[ContextMessage]) -> list[ContextMessage]:
        """Return a reduced list of messages preserving semantic meaning."""
        ...


class DropToolInternalsStrategy:
    """Remove ``tool_use`` and ``tool_result`` blocks except the final pairs.

    Preserves all ``user`` and ``assistant`` turns as well as ``system``
    and ``thinking`` blocks.  Does NOT call any LLM.

    Args:
        keep_last_pairs: Number of final tool_use/tool_result pairs to keep.
    """

    def __init__(self, keep_last_pairs: int = 1) -> None:
        self.keep_last_pairs = max(0, keep_last_pairs)

    def compact(self, messages: list[ContextMessage]) -> list[ContextMessage]:
        tool_result_indices = [
            i for i, m in enumerate(messages) if m.role == ROLE_TOOL_RESULT
        ]
        if len(tool_result_indices) <= self.keep_last_pairs:
            return messages

        keep_from = tool_result_indices[-self.keep_last_pairs] if self.keep_last_pairs else None

        # IDs of tool_use blocks linked to kept tool_results
        kept_ids: set[str] = set()
        if keep_from is not None:
            for m in messages[keep_from:]:
                if m.role == ROLE_TOOL_RESULT and m.message_id:
                    kept_ids.add(m.message_id)

        result: list[ContextMessage] = []
        for i, m in enumerate(messages):
            if m.role == ROLE_TOOL_RESULT:
                if keep_from is not None and i >= keep_from:
                    result.append(m)
                # else: drop
            elif m.role == ROLE_TOOL_USE:
                if m.message_id and m.message_id in kept_ids:
                    result.append(m)
                elif keep_from is not None and i >= keep_from:
                    result.append(m)
                # else: drop
            else:
                result.append(m)
        return result


class SummarizeStrategy:
    """Keep the first *keep_first* and last *keep_last* messages; summarize the middle.

    The middle messages are replaced with a single ``assistant`` block containing
    the text returned by ``summary_fn``.  When ``summary_fn`` is None the middle
    is replaced with a placeholder.

    Args:
        keep_first:  Number of messages to keep at the start (e.g. system prompt).
        keep_last:   Number of messages to keep at the end (recent context).
        summary_fn:  Callable that receives the dropped messages and returns a
                     summary string.  When None a placeholder is used.
    """

    def __init__(
        self,
        keep_first: int = 2,
        keep_last: int = 10,
        summary_fn: Optional[Callable[[list[ContextMessage]], str]] = None,
    ) -> None:
        self.keep_first = max(0, keep_first)
        self.keep_last = max(0, keep_last)
        self.summary_fn = summary_fn

    def compact(self, messages: list[ContextMessage]) -> list[ContextMessage]:
        total = len(messages)
        if total <= self.keep_first + self.keep_last:
            return messages

        head = messages[: self.keep_first]
        tail = messages[total - self.keep_last :]
        middle = messages[self.keep_first : total - self.keep_last]

        if not middle:
            return messages

        if self.summary_fn is not None:
            summary_text = self.summary_fn(middle)
        else:
            summary_text = (
                f"[{len(middle)} earlier messages summarized — "
                f"approximately {sum((m.token_count or 0) for m in middle)} tokens]"
            )

        summary_msg = ContextMessage(role=ROLE_ASSISTANT, content=summary_text)
        return head + [summary_msg] + tail
