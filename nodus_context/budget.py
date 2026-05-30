"""ContextBudget — track token usage against a finite context window."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ContextBudget:
    """Token budget for one LLM context window.

    Attributes
    ----------
    max_tokens:         Hard limit for the context window.
    warning_threshold:  Fraction of capacity (0–1) at which ``is_warning``
                        becomes True.  Default: 0.80 (80 % full).
    current_usage:      Running total of tokens charged so far.
    """

    max_tokens: int
    warning_threshold: float = 0.80
    current_usage: int = 0

    @property
    def remaining(self) -> int:
        return max(0, self.max_tokens - self.current_usage)

    @property
    def utilization(self) -> float:
        """Fraction of max_tokens used (0.0–1.0+)."""
        if self.max_tokens <= 0:
            return 1.0
        return self.current_usage / self.max_tokens

    @property
    def is_warning(self) -> bool:
        return self.utilization >= self.warning_threshold

    @property
    def is_full(self) -> bool:
        return self.current_usage >= self.max_tokens

    def would_overflow(self, tokens: int) -> bool:
        """True if adding *tokens* would exceed max_tokens."""
        return (self.current_usage + tokens) > self.max_tokens

    def charge(self, tokens: int) -> None:
        """Increase current_usage by *tokens*."""
        self.current_usage += tokens

    def release(self, tokens: int) -> None:
        """Decrease current_usage by *tokens* (clamped to 0)."""
        self.current_usage = max(0, self.current_usage - tokens)

    def reset(self) -> None:
        self.current_usage = 0


# ── Token counting helpers ─────────────────────────────────────────────────────

def estimate_tokens(content: Any) -> int:
    """Word-count estimate: ~4 characters per token.  No external dependencies."""
    if content is None:
        return 0
    if isinstance(content, str):
        chars = len(content)
        return max(1, chars // 4) if chars > 0 else 0
    if isinstance(content, (list, tuple)):
        return sum(estimate_tokens(item) for item in content)
    if isinstance(content, dict):
        return sum(
            estimate_tokens(k) + estimate_tokens(v) for k, v in content.items()
        )
    return max(1, len(str(content)) // 4)


def count_tokens(content: Any, *, model: str | None = None) -> int:
    """Return accurate token count via tiktoken when installed; else estimate.

    Args:
        content: The text or content object to count.
        model:   Optional model name for tiktoken encoding selection.
    """
    try:
        import tiktoken  # noqa: PLC0415

        enc_name = "cl100k_base"
        if model:
            try:
                enc = tiktoken.encoding_for_model(model)
                return len(enc.encode(str(content)))
            except KeyError:
                pass
        enc = tiktoken.get_encoding(enc_name)
        return len(enc.encode(str(content)))
    except ImportError:
        return estimate_tokens(content)
