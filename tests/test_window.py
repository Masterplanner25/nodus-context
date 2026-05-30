import pytest
from nodus_context import ContextBudget, ContextMessage, ContextWindow, DropToolInternalsStrategy
from nodus_context.message import ROLE_TOOL_RESULT, ROLE_TOOL_USE


def _msg(role="user", content="hello", token_count=10, message_id=None):
    return ContextMessage(role=role, content=content, token_count=token_count, message_id=message_id)


def _window(max_tokens=1000):
    return ContextWindow(ContextBudget(max_tokens=max_tokens))


# ── add / add_force ───────────────────────────────────────────────────────────

def test_add_returns_true_when_space():
    w = _window(1000)
    assert w.add(_msg(token_count=10)) is True
    assert len(w) == 1


def test_add_returns_false_when_full():
    w = _window(max_tokens=5)
    assert w.add(_msg(token_count=10)) is False
    assert len(w) == 0


def test_add_charges_budget():
    w = _window(1000)
    w.add(_msg(token_count=50))
    assert w.budget().current_usage == 50


def test_add_force_ignores_overflow():
    w = _window(max_tokens=5)
    w.add_force(_msg(token_count=100))
    assert len(w) == 1
    assert w.budget().current_usage == 100


def test_add_estimates_tokens_when_missing():
    w = _window(1000)
    msg = ContextMessage(role="user", content="hello world test")
    added = w.add(msg)
    assert added is True
    assert w.budget().current_usage > 0


# ── snapshot ──────────────────────────────────────────────────────────────────

def test_snapshot_is_copy():
    w = _window(1000)
    w.add(_msg())
    snap = w.snapshot()
    snap.clear()
    assert len(w) == 1  # original unchanged


# ── compact ───────────────────────────────────────────────────────────────────

def test_compact_returns_tokens_freed():
    w = _window(1000)
    for _ in range(5):
        w.add(_msg(role=ROLE_TOOL_USE, content="call", token_count=20, message_id="id1"))
        w.add(_msg(role=ROLE_TOOL_RESULT, content="result", token_count=20, message_id="id1"))
    freed = w.compact(DropToolInternalsStrategy(keep_last_pairs=1))
    assert freed > 0


def test_compact_reduces_message_count():
    w = _window(10000)
    w.add(_msg(role="user", token_count=10))
    for _ in range(4):
        w.add(_msg(role=ROLE_TOOL_USE, content="c", token_count=10, message_id=f"id{_}"))
        w.add(_msg(role=ROLE_TOOL_RESULT, content="r", token_count=10, message_id=f"id{_}"))
    before = len(w)
    w.compact(DropToolInternalsStrategy(keep_last_pairs=1))
    assert len(w) < before


# ── guard_tool_results ────────────────────────────────────────────────────────

def test_guard_tool_results_keeps_last():
    w = _window(10000)
    for i in range(5):
        w.add(_msg(role=ROLE_TOOL_USE, token_count=10, message_id=f"id{i}"))
        w.add(_msg(role=ROLE_TOOL_RESULT, token_count=10, message_id=f"id{i}"))
    freed = w.guard_tool_results(keep_last=2)
    assert freed > 0
    result_msgs = [m for m in w.messages() if m.role == ROLE_TOOL_RESULT]
    assert len(result_msgs) == 2


def test_guard_noop_when_within_limit():
    w = _window(10000)
    w.add(_msg(role=ROLE_TOOL_RESULT, token_count=10, message_id="id1"))
    freed = w.guard_tool_results(keep_last=3)
    assert freed == 0
