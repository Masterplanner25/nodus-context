"""ContextMessage — a single message in an LLM context window."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Valid role values (not enforced — kept as documentation)
ROLE_USER       = "user"
ROLE_ASSISTANT  = "assistant"
ROLE_SYSTEM     = "system"
ROLE_TOOL_USE   = "tool_use"
ROLE_TOOL_RESULT = "tool_result"
ROLE_THINKING   = "thinking"


@dataclass
class ContextMessage:
    """One message in an LLM context window.

    Attributes
    ----------
    role:         Sender role — see ROLE_* constants.
    content:      Message content (str or list of content blocks).
    token_count:  Estimated or measured token count.  None = unknown; the
                  window computes an estimate lazily when needed.
    message_id:   Optional ID linking a tool_result to its tool_use block.
    """

    role: str
    content: Any
    token_count: int | None = None
    message_id: str | None = None
