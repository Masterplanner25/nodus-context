"""ContextWindow — manages a list of ContextMessages within a ContextBudget."""
from __future__ import annotations

from typing import TYPE_CHECKING

from .budget import ContextBudget, estimate_tokens
from .message import ROLE_TOOL_RESULT, ROLE_TOOL_USE, ContextMessage

if TYPE_CHECKING:
    from .strategies import CompactionStrategy


class ContextWindow:
    """Ordered list of ``ContextMessage`` objects with budget enforcement.

    Usage::

        budget = ContextBudget(max_tokens=128_000)
        window = ContextWindow(budget)
        window.add_force(ContextMessage(role="system", content="You are helpful."))
        added = window.add(ContextMessage(role="user", content="Hello"))
        if not added:
            # window is full — compact before adding
            window.compact(DropToolInternalsStrategy())
    """

    def __init__(self, budget: ContextBudget) -> None:
        self._budget = budget
        self._messages: list[ContextMessage] = []

    # ── Adding messages ────────────────────────────────────────────────────────

    def add(self, message: ContextMessage) -> bool:
        """Add *message* only if it fits within the budget.

        Returns False and does NOT add the message if adding it would overflow.
        Token count is estimated lazily if ``message.token_count`` is None.
        """
        tokens = self._resolve_tokens(message)
        if self._budget.would_overflow(tokens):
            return False
        message = _with_token_count(message, tokens)
        self._messages.append(message)
        self._budget.charge(tokens)
        return True

    def add_force(self, message: ContextMessage) -> None:
        """Add *message* regardless of budget (for system prompts, required context).

        Charges the tokens but does NOT block on overflow.
        """
        tokens = self._resolve_tokens(message)
        message = _with_token_count(message, tokens)
        self._messages.append(message)
        self._budget.charge(tokens)

    # ── Reading ────────────────────────────────────────────────────────────────

    def messages(self) -> list[ContextMessage]:
        """Return the current message list (live reference — do not mutate)."""
        return self._messages

    def snapshot(self) -> list[ContextMessage]:
        """Return a shallow copy of the message list."""
        return list(self._messages)

    def budget(self) -> ContextBudget:
        return self._budget

    def __len__(self) -> int:
        return len(self._messages)

    # ── Compaction ─────────────────────────────────────────────────────────────

    def compact(self, strategy: "CompactionStrategy") -> int:
        """Run *strategy* and return the number of tokens freed.

        The strategy receives the full message list and returns a reduced list.
        ``current_usage`` is recomputed from the surviving messages.
        """
        before = self._budget.current_usage
        self._messages = strategy.compact(list(self._messages))
        new_usage = sum(
            self._resolve_tokens(m) for m in self._messages
        )
        self._budget.current_usage = new_usage
        freed = before - new_usage
        return max(0, freed)

    def guard_tool_results(self, keep_last: int = 3) -> int:
        """Drop ``tool_result`` messages beyond the most recent *keep_last*.

        Preserves all ``user`` and ``assistant`` turns.  Drops the matching
        ``tool_use`` block for each dropped ``tool_result`` when possible.

        Returns the number of tokens freed.
        """
        # Collect tool_result indices in order
        result_indices = [
            i for i, m in enumerate(self._messages) if m.role == ROLE_TOOL_RESULT
        ]
        if len(result_indices) <= keep_last:
            return 0

        drop_indices = set(result_indices[: -keep_last])

        # Also drop the matching tool_use for each dropped tool_result
        ids_to_drop = {
            self._messages[i].message_id
            for i in drop_indices
            if self._messages[i].message_id
        }
        for i, m in enumerate(self._messages):
            if m.role == ROLE_TOOL_USE and m.message_id in ids_to_drop:
                drop_indices.add(i)

        freed = sum(
            self._resolve_tokens(self._messages[i]) for i in drop_indices
        )
        self._messages = [
            m for i, m in enumerate(self._messages) if i not in drop_indices
        ]
        self._budget.release(freed)
        return freed

    # ── Private ────────────────────────────────────────────────────────────────

    def _resolve_tokens(self, message: ContextMessage) -> int:
        if message.token_count is not None:
            return message.token_count
        return estimate_tokens(message.content)


def _with_token_count(message: ContextMessage, count: int) -> ContextMessage:
    """Return message with token_count filled in (no-op if already set)."""
    if message.token_count is not None:
        return message
    from dataclasses import replace
    return replace(message, token_count=count)
