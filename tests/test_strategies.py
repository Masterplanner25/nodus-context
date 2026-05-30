from nodus_context import (
    ContextMessage,
    DropToolInternalsStrategy,
    SummarizeStrategy,
)
from nodus_context.message import (
    ROLE_ASSISTANT,
    ROLE_TOOL_RESULT,
    ROLE_TOOL_USE,
    ROLE_USER,
)


def _msg(role, content="x", token_count=10, message_id=None):
    return ContextMessage(role=role, content=content,
                          token_count=token_count, message_id=message_id)


# ── DropToolInternalsStrategy ─────────────────────────────────────────────────

def test_drop_tool_internals_keeps_last_pair():
    msgs = [
        _msg(ROLE_USER),
        _msg(ROLE_TOOL_USE, message_id="a"),
        _msg(ROLE_TOOL_RESULT, message_id="a"),
        _msg(ROLE_TOOL_USE, message_id="b"),
        _msg(ROLE_TOOL_RESULT, message_id="b"),
        _msg(ROLE_ASSISTANT),
    ]
    result = DropToolInternalsStrategy(keep_last_pairs=1).compact(msgs)
    result_roles = [m.role for m in result]
    # Should keep user, last tool pair, assistant
    assert ROLE_USER in result_roles
    assert ROLE_ASSISTANT in result_roles
    tool_results = [m for m in result if m.role == ROLE_TOOL_RESULT]
    assert len(tool_results) == 1
    assert tool_results[0].message_id == "b"


def test_drop_tool_internals_noop_when_within_limit():
    msgs = [_msg(ROLE_TOOL_RESULT, message_id="a")]
    result = DropToolInternalsStrategy(keep_last_pairs=2).compact(msgs)
    assert result == msgs


def test_drop_tool_internals_zero_keep():
    msgs = [
        _msg(ROLE_USER),
        _msg(ROLE_TOOL_USE, message_id="a"),
        _msg(ROLE_TOOL_RESULT, message_id="a"),
    ]
    result = DropToolInternalsStrategy(keep_last_pairs=0).compact(msgs)
    assert all(m.role not in (ROLE_TOOL_USE, ROLE_TOOL_RESULT) for m in result)


def test_preserves_user_and_assistant():
    msgs = [
        _msg(ROLE_USER, content="q"),
        _msg(ROLE_TOOL_USE, message_id="x"),
        _msg(ROLE_TOOL_RESULT, message_id="x"),
        _msg(ROLE_ASSISTANT, content="a"),
    ]
    result = DropToolInternalsStrategy(keep_last_pairs=0).compact(msgs)
    roles = [m.role for m in result]
    assert ROLE_USER in roles
    assert ROLE_ASSISTANT in roles


# ── SummarizeStrategy ─────────────────────────────────────────────────────────

def test_summarize_replaces_middle():
    msgs = [_msg(ROLE_USER, content=f"msg{i}") for i in range(15)]
    result = SummarizeStrategy(keep_first=2, keep_last=3).compact(msgs)
    # head (2) + summary (1) + tail (3) = 6
    assert len(result) == 6
    # summary is in position 2
    assert result[2].role == ROLE_ASSISTANT
    assert "summarized" in result[2].content.lower()


def test_summarize_noop_when_small():
    msgs = [_msg(ROLE_USER) for _ in range(4)]
    result = SummarizeStrategy(keep_first=2, keep_last=3).compact(msgs)
    assert result == msgs


def test_summarize_custom_fn():
    msgs = [_msg(ROLE_USER, content=f"m{i}", token_count=5) for i in range(10)]
    fn_called = []

    def my_fn(dropped):
        fn_called.extend(dropped)
        return "custom summary"

    result = SummarizeStrategy(keep_first=1, keep_last=2, summary_fn=my_fn).compact(msgs)
    assert len(fn_called) > 0
    assert result[1].content == "custom summary"
